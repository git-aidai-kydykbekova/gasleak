"""Evaluate SNN vs CNN: detection quality, detection delay, SNR robustness, compute energy."""
import json, os
import numpy as np
import pandas as pd
import torch
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from train import load
from gasleak.models import LeakSNN, CausalCNN

WARMUP = 100        # ms used by the encoder to learn the noise floor
K = 5               # consecutive ms above threshold to declare a detection
FA_MAX = 0.05       # max fraction of no-leak val clips allowed to raise an alarm
E_MAC, E_AC = 4.6e-12, 0.9e-12   # J/op, 32-bit float, 45 nm (Horowitz, ISSCC 2014)
OUT = "runs"


@torch.no_grad()
def infer(kind, X, idx):
    model = LeakSNN() if kind == "snn" else CausalCNN()
    model.load_state_dict(torch.load(f"{OUT}/{kind}.pt")); model.eval()
    probs, s1s, s2s = [], [], []
    for i in range(0, len(idx), 150):
        x = torch.from_numpy(X[idx[i:i + 150]])
        if kind == "snn":
            lg, s1, s2 = model(x); s1s.append(s1.sum(-1).numpy()); s2s.append(s2.sum(-1).numpy())
        else:
            lg = model(x)
        probs.append(torch.sigmoid(lg).numpy())
    p = np.concatenate(probs)
    spk = (np.concatenate(s1s), np.concatenate(s2s)) if kind == "snn" else None
    return p, spk, model


def first_detection(p, tau):
    """index of first time step where p>tau for K consecutive steps (after warm-up); -1 if none"""
    above = p[:, WARMUP:] > tau
    run = np.zeros(above.shape[0], int); det = np.full(above.shape[0], -1)
    for t in range(above.shape[1]):
        run = np.where(above[:, t], run + 1, 0)
        hit = (run >= K) & (det < 0)
        det[hit] = t + WARMUP
    return det


def score(p, meta, tau):
    det = first_detection(p, tau)
    lab = meta.label.values; on = (meta.onset_s.fillna(0).values * 1000).round().astype(int)
    # a leak clip is a true detection if the alarm comes at/after the onset (before = false alarm)
    pre_alarm = np.zeros(len(det), bool)
    for i in np.where(lab == 1)[0]:
        pre_alarm[i] = (first_detection(p[i:i + 1, :on[i]], tau)[0] >= 0) if on[i] > WARMUP + K else False
    det_after = np.array([first_detection(p[i:i + 1], tau)[0] if lab[i] == 0 else
                          _det_after(p[i], on[i], tau) for i in range(len(det))])
    tp = (lab == 1) & (det_after >= 0)
    fa = (lab == 0) & (det >= 0)
    delay = np.where(tp, det_after - on, np.nan)
    return dict(tp=tp, fa=fa, pre_alarm=pre_alarm, delay=delay)


def _det_after(pi, on, tau):
    above = pi[on:] > tau; run = 0
    for t, a in enumerate(above):
        run = run + 1 if a else 0
        if run >= K:
            return on + t
    return -1


def summarize(r, meta):
    lab = meta.label.values
    tp, fa = r["tp"].sum(), r["fa"].sum()
    fn = (lab == 1).sum() - tp
    prec = tp / max(tp + fa, 1); rec = tp / max((lab == 1).sum(), 1)
    return dict(recall=rec, precision=prec, f1=2 * prec * rec / max(prec + rec, 1e-9),
                false_alarm_rate=fa / max((lab == 0).sum(), 1),
                accuracy=(tp + ((lab == 0).sum() - fa)) / len(lab),
                median_delay_ms=float(np.nanmedian(r["delay"])), mean_delay_ms=float(np.nanmean(r["delay"])),
                p90_delay_ms=float(np.nanpercentile(r["delay"], 90)))


def pick_tau(p, meta):
    best = None
    for tau in np.linspace(0.3, 0.98, 35):
        s = summarize(score(p, meta, tau), meta)
        if s["false_alarm_rate"] <= FA_MAX and (best is None or s["f1"] > best[1]):
            best = (tau, s["f1"])
    return best[0] if best else 0.98


if __name__ == "__main__":
    meta, ev, dense, y, mask = load()
    va = np.where(meta.split == "val")[0]; te = np.where(meta.split == "test")[0]
    mte = meta.iloc[te].reset_index(drop=True)
    res, curves, per_snr = {}, {}, []
    for kind, X in [("snn", ev), ("cnn", dense)]:
        pv, _, _ = infer(kind, X, va)
        tau = pick_tau(pv, meta.iloc[va].reset_index(drop=True))
        pt, spk, model = infer(kind, X, te)
        r = score(pt, mte, tau)
        res[kind] = dict(threshold=float(tau), **summarize(r, mte))
        curves[kind] = (pt, spk, model, r)
        for snr in sorted(mte.snr_db.dropna().unique()):
            sel = (mte.snr_db == snr).values
            per_snr.append(dict(model=kind, snr_db=snr, recall=r["tp"][sel].mean(),
                                median_delay_ms=float(np.nanmedian(r["delay"][sel])) if r["tp"][sel].any() else np.nan))
        for sc in ["quiet", "factory"]:
            sel = ((mte.label == 0) & (mte.scene == sc)).values
            res[kind][f"false_alarm_rate_{sc}"] = float(r["fa"][sel].mean())

    # ---------------- compute / energy (per second of operation) ----------------
    _, (s1, s2), snn, _ = curves["snn"]
    cnn = curves["cnn"][2]
    ev_te = ev[te]
    n_hid = snn.fc1.out_features
    synops = ev_te.sum(-1) * n_hid + s1 * n_hid + s2 * 1          # [N, T] accumulate ops per ms
    on = (mte.onset_s.fillna(0).values * 1000).astype(int)
    def rate(sel, after_onset=False):
        vals = []
        for i in np.where(sel)[0]:
            seg = synops[i, on[i]:] if after_onset else synops[i, WARMUP:]
            ein = ev_te[i, on[i]:] if after_onset else ev_te[i, WARMUP:]
            vals.append((seg.sum() * 1000 / len(seg), ein.sum() * 1000 / len(ein)))
        return np.mean(vals, 0)
    cnn_macs = cnn.macs_per_step() * 1000
    twin_macs = (snn.fc1.in_features * n_hid + n_hid * n_hid + n_hid) * 1000
    energy = []
    for name, sel, ao in [("no-leak, quiet", (mte.label == 0) & (mte.scene == "quiet"), False),
                          ("no-leak, factory", (mte.label == 0) & (mte.scene == "factory"), False),
                          ("leak active", mte.label == 1, True)]:
        so, evr = rate(sel.values, ao)
        energy.append(dict(condition=name, input_events_per_s=evr, snn_synops_per_s=so,
                           snn_power_uW=so * E_AC * 1e6, cnn_macs_per_s=cnn_macs,
                           cnn_power_uW=cnn_macs * E_MAC * 1e6, dense_twin_power_uW=twin_macs * E_MAC * 1e6,
                           ratio_cnn_over_snn=cnn_macs * E_MAC / max(so * E_AC, 1e-18),
                           ratio_twin_over_snn=twin_macs * E_MAC / max(so * E_AC, 1e-18)))
    energy = pd.DataFrame(energy)
    per_snr = pd.DataFrame(per_snr)
    os.makedirs(f"{OUT}/figs", exist_ok=True)
    energy.to_csv(f"{OUT}/energy.csv", index=False); per_snr.to_csv(f"{OUT}/per_snr.csv", index=False)
    json.dump(res, open(f"{OUT}/results.json", "w"), indent=2, default=float)
    print(pd.DataFrame(res).T.round(3).to_string())
    print(per_snr.round(3).to_string())
    print(energy.round(2).T.to_string())

    # ---------------- figures ----------------
    fig, ax = plt.subplots(1, 2, figsize=(10, 3.6))
    for kind, c in [("snn", "#2a7ab9"), ("cnn", "#d4762c")]:
        d = per_snr[per_snr.model == kind]
        ax[0].plot(d.snr_db, d.recall, "o-", c=c, label=kind.upper())
        ax[1].plot(d.snr_db, d.median_delay_ms, "o-", c=c, label=kind.upper())
    ax[0].set(xlabel="SNR in ultrasonic band (dB)", ylabel="Detection rate (recall)", ylim=(0, 1.05), title="SNR robustness")
    ax[1].set(xlabel="SNR (dB)", ylabel="Median detection delay (ms)", title="Detection delay")
    for a in ax: a.grid(alpha=.3); a.legend()
    plt.tight_layout(); plt.savefig(f"{OUT}/figs/snr_robustness.png", dpi=150); plt.close()

    fig, ax = plt.subplots(figsize=(7, 3.6))
    xs = np.arange(len(energy)); w = 0.27
    ax.bar(xs - w, energy.cnn_power_uW, w, label="CNN (always on)", color="#d4762c")
    ax.bar(xs, energy.dense_twin_power_uW, w, label="Same net, dense/frame-based", color="#999")
    ax.bar(xs + w, energy.snn_power_uW, w, label="SNN (event-driven)", color="#2a7ab9")
    ax.set_yscale("log"); ax.set_xticks(xs, energy.condition); ax.set_ylabel("Compute power (µW, est.)")
    ax.legend(fontsize=8); ax.grid(alpha=.3, axis="y"); plt.tight_layout()
    plt.savefig(f"{OUT}/figs/energy.png", dpi=150); plt.close()

    # raster examples
    pt = curves["snn"][0]
    ex = [np.where((mte.label == 0) & (mte.scene == "quiet"))[0][0], np.where((mte.label == 0) & (mte.scene == "factory"))[0][0],
          np.where((mte.label == 1) & (mte.snr_db == 0))[0][0]]
    fig, ax = plt.subplots(2, 3, figsize=(12, 5), sharex=True, gridspec_kw=dict(height_ratios=[3, 1]))
    for j, i in enumerate(ex):
        tt, ch = np.nonzero(ev_te[i]); ax[0, j].scatter(tt, ch, s=1, c="k")
        ttl = "no leak, quiet" if j == 0 else "no leak, factory" if j == 1 else f"leak (SNR 0 dB), onset {on[i]} ms"
        ax[0, j].set_title(f"{ttl}\n{len(tt)} events"); ax[0, j].set_ylabel("input channel")
        ax[1, j].plot(pt[i], c="#2a7ab9"); ax[1, j].axhline(res['snn']['threshold'], ls="--", c="r", lw=.8)
        ax[1, j].set(ylim=(0, 1), xlabel="time (ms)", ylabel="P(leak)")
        if mte.label[i] == 1: ax[1, j].axvline(on[i], c="g", lw=.8)
    plt.tight_layout(); plt.savefig(f"{OUT}/figs/event_raster.png", dpi=150); plt.close()
