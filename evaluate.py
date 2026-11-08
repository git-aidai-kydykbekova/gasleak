"""Evaluate SNN vs CNN baselines over all trained seeds (runs/seeds/<seed>/{snn,cnn}.pt).

Detectors
  snn        event-driven SNN on encoder spikes
  cnn        frame-based CNN on dense band levels, always on
  cnn_gated  the same CNN, but only awake while the event encoder produces spikes
             (wake-up baseline: E.HOLD_MS after the last event, asleep = "no leak")
Operating points: for each detector the threshold is chosen on VALIDATION so that the false-alarm
rate over leak-free segments is <= 1 % or <= 5 %; test metrics are then compared at equal budget.
Uncertainty: mean +- std over seeds; 95 % CIs by bootstrap over seeds and test clips.

Outputs (runs/): results.json, per_snr.csv, det.csv, energy.csv, thresholds.json, figs/*.png
"""
import glob, json, os
import numpy as np
import pandas as pd
import torch
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from train import load
from gasleak.models import LeakSNN, CausalCNN
from gasleak import metrics as M, energy as E

TARGETS = (0.01, 0.05)                       # false-alarm budgets
DETECTORS = ("snn", "cnn", "cnn_gated")
COLORS = {"snn": "#2a7ab9", "cnn": "#d4762c", "cnn_gated": "#5a9e4b"}
ASLEEP = -50.0                               # gated CNN output (logit) while asleep
N_BOOT = 2000
OUT = "runs"


@torch.no_grad()
def infer(kind, ckpt, X, idx, bs=150):
    """-> model, logits [N, T], (spikes per step of hidden layers 1 and 2) for the SNN."""
    model = LeakSNN() if kind == "snn" else CausalCNN()
    model.load_state_dict(torch.load(ckpt)); model.eval()
    z, s1s, s2s = [], [], []
    for i in range(0, len(idx), bs):
        x = torch.from_numpy(X[idx[i:i + bs]])
        if kind == "snn":
            lg, s1, s2 = model(x); s1s.append(s1.sum(-1).numpy()); s2s.append(s2.sum(-1).numpy())
        else:
            lg = model(x)
        z.append(lg.numpy())
    spk = (np.concatenate(s1s), np.concatenate(s2s)) if kind == "snn" else None
    return model, np.concatenate(z), spk


def boot_ci(v, rng):
    """v: [S seeds, n clips] per-clip outcomes -> 95 % CI of the seed-mean, resampling seeds and clips."""
    S, n = v.shape
    si = rng.integers(0, S, (N_BOOT, S)); ci = rng.integers(0, n, (N_BOOT, n))
    return np.percentile(v[si[:, :, None], ci[:, None, :]].mean((1, 2)), [2.5, 97.5])


def eval_seed(sd, meta, ev, dense, va, te):
    mva = meta.iloc[va].reset_index(drop=True); mte = meta.iloc[te].reset_index(drop=True)
    snn, zv_s, _ = infer("snn", f"{sd}/snn.pt", ev, va)
    _, zt_s, (s1, s2) = infer("snn", f"{sd}/snn.pt", ev, te)
    cnn, zv_c, _ = infer("cnn", f"{sd}/cnn.pt", dense, va)
    _, zt_c, _ = infer("cnn", f"{sd}/cnn.pt", dense, te)
    ev_v, ev_t = ev[va].sum(-1), ev[te].sum(-1)
    logits = {"snn": (zv_s, zt_s), "cnn": (zv_c, zt_c),
              "cnn_gated": (np.where(E.wake_mask(ev_v), zv_c, ASLEEP), np.where(E.wake_mask(ev_t), zt_c, ASLEEP))}
    out = dict(summary={}, clips={}, det={}, thresholds={})
    for name, (zv, zt) in logits.items():
        qv, qt = M.window_min(zv), M.window_min(zt)
        fa, miss = M.det_curve(qt, mte)
        out["det"][name] = (fa, miss)
        for target in TARGETS:
            tau = M.pick_threshold(qv, mva, target)
            r = M.score(qt, mte, tau)
            out["thresholds"][(name, target)] = tau
            out["summary"][(name, target)] = dict(threshold_logit=tau, **M.summarize(r, mte, zt.shape[1]))
            out["clips"][(name, target)] = r

    # compute power per condition (test clips)
    T = ev_t.shape[1]
    t = np.arange(T)[None]
    on = M.onsets_ms(mte, T)[:, None]
    conds = {"no-leak, quiet": ((mte.label == 0) & (mte.scene == "quiet")).values[:, None] & (t >= M.WARMUP),
             "no-leak, factory": ((mte.label == 0) & (mte.scene == "factory")).values[:, None] & (t >= M.WARMUP),
             "leak active": (mte.label == 1).values[:, None] & (t >= on)}
    energy = []
    for cond, sel in conds.items():
        power, extras = E.pipeline_power(snn, cnn, ev_t, s1, s2, sel)
        for pipe, p in power.items():
            energy.append(dict(condition=cond, pipeline=pipe, **p, **extras))
    out["energy"] = pd.DataFrame(energy)
    out["example"] = (zt_s, s1, s2)
    return out


def mean_std(rows):
    df = pd.DataFrame(rows)
    return {c: dict(mean=float(df[c].mean()), std=float(df[c].std(ddof=0)), per_seed=df[c].tolist()) for c in df}


if __name__ == "__main__":
    seed_dirs = sorted(d for d in glob.glob(f"{OUT}/seeds/*") if os.path.exists(f"{d}/snn.pt") and os.path.exists(f"{d}/cnn.pt"))
    assert seed_dirs, "no trained seeds in runs/seeds/ - run ./run_seeds.sh first"
    seeds = [os.path.basename(d) for d in seed_dirs]
    print("seeds:", seeds)
    meta, ev, dense, y, mask = load()
    va = np.where(meta.split == "val")[0]; te = np.where(meta.split == "test")[0]
    mte = meta.iloc[te].reset_index(drop=True)
    runs = [eval_seed(d, meta, ev, dense, va, te) for d in seed_dirs]
    rng = np.random.default_rng(0)

    # ---------------- detection quality at matched false-alarm budgets ----------------
    results = {"seeds": seeds, "targets": list(TARGETS), "detectors": {}}
    per_snr = []
    for name in DETECTORS:
        results["detectors"][name] = {}
        for target in TARGETS:
            key = f"fa_{target:.0%}"
            res = mean_std([r["summary"][(name, target)] for r in runs])
            tp = np.stack([r["clips"][(name, target)]["tp"] for r in runs])
            na = np.stack([r["clips"][(name, target)]["neg_alarm"] for r in runs])
            leak = (mte.label == 1).values
            res["recall"]["ci95"] = boot_ci(tp[:, leak].astype(float), rng).tolist()
            res["false_alarm_rate"]["ci95"] = boot_ci(na.astype(float), rng).tolist()
            results["detectors"][name][key] = res
            delay = np.stack([r["clips"][(name, target)]["delay"] for r in runs])
            for snr in sorted(mte.snr_db.dropna().unique()):
                sel = (mte.snr_db == snr).values
                lo, hi = boot_ci(tp[:, sel].astype(float), rng)
                d = delay[:, sel]
                per_snr.append(dict(detector=name, fa_target=target, snr_db=snr, n_clips=int(sel.sum()),
                                    recall=tp[:, sel].mean(), recall_std_seeds=tp[:, sel].mean(1).std(),
                                    recall_ci_lo=lo, recall_ci_hi=hi,
                                    median_delay_ms=float(np.nanmedian(d)) if np.isfinite(d).any() else np.nan))
    per_snr = pd.DataFrame(per_snr)

    # ---------------- compute power ----------------
    en = pd.concat([r["energy"].assign(seed=s) for r, s in zip(runs, seeds)])
    cols = ["frontend_uW", "encoder_uW", "network_uW", "total_uW", "input_events_per_s", "gated_cnn_duty"]
    energy = en.groupby(["condition", "pipeline"], sort=False)[cols].mean().reset_index()
    energy["total_uW_std"] = en.groupby(["condition", "pipeline"], sort=False).total_uW.std(ddof=0).values
    ref = energy[energy.pipeline == "snn (clock-driven)"].set_index("condition")
    energy["total_over_snn"] = energy.total_uW / energy.condition.map(ref.total_uW)
    energy["network_over_snn"] = energy.network_uW / energy.condition.map(ref.network_uW)

    # RQ: power reduction during quiescent (no-leak) periods, whole pipeline vs network only
    rq = {}
    for cond in ["no-leak, quiet", "no-leak, factory"]:
        e = energy[energy.condition == cond].set_index("pipeline")
        rq[cond] = {base: dict(total_ratio=float(e.total_uW[base] / e.total_uW["snn (clock-driven)"]),
                               network_ratio=float(e.network_uW[base] / e.network_uW["snn (clock-driven)"]))
                    for base in ["cnn (always on)", "cnn (event-gated)", "dense twin"]}
    results["rq_quiescent_power_ratio"] = rq
    results["energy_model"] = dict(frontend_ops_per_frame=dict(zip(("mul", "add"), E.frontend_ops())),
                                   frontend_uW=E.frontend_power_uW(), E_mul_pJ=E.E_MUL * 1e12,
                                   E_add_pJ=E.E_ADD * 1e12, hold_ms=E.HOLD_MS)

    thresholds = {s: {name: {str(t): r["thresholds"][(name, t)] for t in TARGETS} for name in DETECTORS}
                  for r, s in zip(runs, seeds)}
    det = pd.concat([pd.DataFrame(dict(detector=name, seed=s, false_alarm_rate=r["det"][name][0], miss_rate=r["det"][name][1]))
                     for r, s in zip(runs, seeds) for name in DETECTORS])

    os.makedirs(f"{OUT}/figs", exist_ok=True)
    json.dump(results, open(f"{OUT}/results.json", "w"), indent=2, default=float)
    json.dump(thresholds, open(f"{OUT}/thresholds.json", "w"), indent=2)
    per_snr.to_csv(f"{OUT}/per_snr.csv", index=False)
    energy.to_csv(f"{OUT}/energy.csv", index=False)
    det.to_csv(f"{OUT}/det.csv", index=False)

    table = pd.DataFrame({(n, f"{t:.0%}"): {k: f"{v['mean']:.3f} ± {v['std']:.3f}" for k, v in results["detectors"][n][f"fa_{t:.0%}"].items()
                                            if k in ("recall", "precision", "false_alarm_rate", "pre_onset_alarm_rate", "median_delay_ms", "p90_delay_ms")}
                          for n in DETECTORS for t in TARGETS})
    print(table.to_string())
    print(per_snr.round(3).to_string())
    print(energy.round(3).to_string())
    print(json.dumps(rq, indent=2))

    # ---------------- figures ----------------
    fig, ax = plt.subplots(1, 2, figsize=(11, 3.8))
    for name in DETECTORS:
        for target, ls in [(0.05, "-"), (0.01, "--")]:
            d = per_snr[(per_snr.detector == name) & (per_snr.fa_target == target)]
            ax[0].plot(d.snr_db, d.recall, "o" + ls, c=COLORS[name], label=f"{name} @ FA {target:.0%}", ms=4)
            if target == 0.05:
                ax[0].fill_between(d.snr_db, d.recall_ci_lo, d.recall_ci_hi, color=COLORS[name], alpha=.15)
                ax[1].plot(d.snr_db, d.median_delay_ms, "o-", c=COLORS[name], label=name, ms=4)
    ax[0].set(xlabel="SNR in ultrasonic band (dB)", ylabel="Detection rate (recall)", ylim=(0, 1.05),
              title=f"SNR robustness (band = 95 % CI, {len(seeds)} seeds)")
    ax[1].set(xlabel="SNR (dB)", ylabel="Median detection delay (ms)", title="Detection delay @ FA 5 %")
    for a in ax: a.grid(alpha=.3); a.legend(fontsize=7)
    plt.tight_layout(); plt.savefig(f"{OUT}/figs/snr_robustness.png", dpi=150); plt.close()

    fig, ax = plt.subplots(figsize=(5.5, 4.5))
    floor = 1 / (2 * len(mte))
    for name in DETECTORS:
        for i, s in enumerate(seeds):
            d = det[(det.detector == name) & (det.seed == s)]
            ax.plot(np.maximum(d.false_alarm_rate, floor), np.maximum(d.miss_rate, floor), c=COLORS[name],
                    lw=1, alpha=.7, label=name if i == 0 else None)
    for t in TARGETS: ax.axvline(t, c="k", ls=":", lw=.8)
    ax.set(xscale="log", yscale="log", xlabel="False-alarm rate (leak-free segments)", ylabel="Miss rate",
           title="Detection error trade-off (test, every seed)")
    ax.grid(alpha=.3, which="both"); ax.legend(); plt.tight_layout(); plt.savefig(f"{OUT}/figs/det.png", dpi=150); plt.close()

    conds = list(energy.condition.unique())
    fig, ax = plt.subplots(1, len(conds), figsize=(13, 4), sharey=True)
    for a, cond in zip(ax, conds):
        e = energy[energy.condition == cond]
        xs = np.arange(len(e))
        a.bar(xs, e.frontend_uW + e.encoder_uW, color="#bbb", label="front-end + encoder")
        a.bar(xs, e.network_uW, bottom=e.frontend_uW + e.encoder_uW, color="#2a7ab9", label="network")
        for x, v in zip(xs, e.total_uW): a.text(x, v, f"{v:.1f}", ha="center", va="bottom", fontsize=7)
        a.set_xticks(xs, [p.replace(" (", "\n(") for p in e.pipeline], fontsize=7); a.set_title(cond)
        a.grid(alpha=.3, axis="y")
    ax[0].set_ylabel("Compute power (µW, estimate)"); ax[0].legend(fontsize=8)
    plt.tight_layout(); plt.savefig(f"{OUT}/figs/energy.png", dpi=150); plt.close()

    # raster examples (first seed, SNN, FA 5 % threshold)
    zt_s, _, _ = runs[0]["example"]
    tau = 1 / (1 + np.exp(-runs[0]["thresholds"][("snn", 0.05)]))
    pt = 1 / (1 + np.exp(-zt_s))
    ev_te = ev[te]; on = M.onsets_ms(mte, ev_te.shape[1])
    ex = [np.where((mte.label == 0) & (mte.scene == "quiet"))[0][0], np.where((mte.label == 0) & (mte.scene == "factory"))[0][0],
          np.where((mte.label == 1) & (mte.snr_db == 0))[0][0]]
    fig, ax = plt.subplots(2, 3, figsize=(12, 5), sharex=True, gridspec_kw=dict(height_ratios=[3, 1]))
    for j, i in enumerate(ex):
        tt, ch = np.nonzero(ev_te[i]); ax[0, j].scatter(tt, ch, s=1, c="k")
        ttl = "no leak, quiet" if j == 0 else "no leak, factory" if j == 1 else f"leak (SNR 0 dB), onset {on[i]} ms"
        ax[0, j].set_title(f"{ttl}\n{len(tt)} events"); ax[0, j].set_ylabel("input channel")
        ax[1, j].plot(pt[i], c="#2a7ab9"); ax[1, j].axhline(tau, ls="--", c="r", lw=.8)
        ax[1, j].set(ylim=(0, 1), xlabel="time (ms)", ylabel="P(leak)")
        if mte.label[i] == 1: ax[1, j].axvline(on[i], c="g", lw=.8)
    plt.tight_layout(); plt.savefig(f"{OUT}/figs/event_raster.png", dpi=150); plt.close()
