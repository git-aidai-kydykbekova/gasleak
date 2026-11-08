"""Quiescent operation: long continuous no-leak recordings (the condition the RQ is about).

For each condition, REPS recordings of DUR_S seconds are simulated and streamed through every
detector without resets.  Reported per condition: input events/s, false alarms per hour at the
validation thresholds of evaluate.py (FA 1 % / 5 %), and compute power of every pipeline.

Conditions: quiet room, factory at fixed ultrasonic-band levels, and a factory recording whose
background steps up by +10 dB half-way (a machine switching on) - a hard case for the encoder's
noise-floor tracker, which only adapts while the signal is near the floor.

Needs runs/seeds/<seed>/*.pt and runs/thresholds.json (run evaluate.py first).
Outputs: data/quiescent/*.npz (cache), runs/quiescent.csv, runs/figs/quiescent.png
"""
import argparse, glob, json, os
import numpy as np
import pandas as pd
import torch
from multiprocessing import Pool
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from train import dense_input
from evaluate import infer, ASLEEP, TARGETS
from gasleak.synth import make_clip, FS
from gasleak.features import band_levels_db, encode_events
from gasleak import metrics as M, energy as E

DUR_S = 60
REPS = 5
# name, scene, ultrasonic background level (dB, None = scene default), level step half-way (dB)
CONDITIONS = [("quiet", "quiet", None, 0),
              ("factory -28 dB", "factory", -28.0, 0),
              ("factory -20 dB", "factory", -20.0, 0),
              ("factory -12 dB", "factory", -12.0, 0),
              ("factory -20 dB, +10 dB step", "factory", -20.0, 10)]
OUT = "runs"


def _one(args):
    ci, rep, seed, dur = args
    _, scene, level, step = CONDITIONS[ci]
    rng = np.random.default_rng(seed)
    x = make_clip(rng, label=0, scene=scene, n=dur * FS, level_db=level).astype(np.float64)
    if step:
        x[len(x) // 2:] *= 10 ** (step / 20)
    return ci, rep, band_levels_db(x)


def recordings(dur, reps):
    """-> levels [C, reps, T, B], events [C, reps, T, 2B] (cached)."""
    cache = f"data/quiescent/recordings_{dur}s_x{reps}.npz"
    if os.path.exists(cache):
        d = np.load(cache)
        return d["levels"].astype(np.float32), d["events"]
    pairs = [(ci, r) for ci in range(len(CONDITIONS)) for r in range(reps)]
    jobs = [(ci, r, s, dur) for (ci, r), s in zip(pairs, np.random.SeedSequence(7).generate_state(len(pairs)))]
    with Pool(min(os.cpu_count(), len(jobs))) as p:
        res = p.map(_one, jobs)
    T, B = res[0][2].shape
    L = np.zeros((len(CONDITIONS), reps, T, B), np.float32)
    for ci, r, l in res:
        L[ci, r] = l
    ev = np.stack([encode_events(L[ci]) for ci in range(len(CONDITIONS))])
    os.makedirs(os.path.dirname(cache), exist_ok=True)
    np.savez_compressed(cache, levels=L.astype(np.float16), events=ev)
    return L, ev


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--dur", type=int, default=DUR_S, help="seconds per recording")
    ap.add_argument("--reps", type=int, default=REPS, help="recordings per condition")
    a = ap.parse_args()
    thresholds = json.load(open(f"{OUT}/thresholds.json"))
    seed_dirs = [d for d in sorted(glob.glob(f"{OUT}/seeds/*")) if os.path.basename(d) in thresholds]
    assert seed_dirs, "no evaluated seeds - run evaluate.py first"
    torch.set_num_threads(os.cpu_count())
    L, EV = recordings(a.dur, a.reps)
    hours = a.reps * (L.shape[2] - M.WARMUP) / 3.6e6
    rows = []
    for ci, (cname, *_rest) in enumerate(CONDITIONS):
        ev = EV[ci].astype(np.float32)
        dense = dense_input(L[ci])
        idx = np.arange(a.reps)
        ev_count = ev.sum(-1)
        sel = np.zeros(ev_count.shape, bool); sel[:, M.WARMUP:] = True
        for sd in seed_dirs:
            seed = os.path.basename(sd)
            snn, z_s, (s1, s2) = infer("snn", f"{sd}/snn.pt", ev, idx, bs=a.reps)
            cnn, z_c, _ = infer("cnn", f"{sd}/cnn.pt", dense, idx, bs=a.reps)
            logits = {"snn": z_s, "cnn": z_c, "cnn_gated": np.where(E.wake_mask(ev_count), z_c, ASLEEP)}
            power, extras = E.pipeline_power(snn, cnn, ev_count, s1, s2, sel)
            row = dict(condition=cname, seed=seed, **extras)
            for name, z in logits.items():
                q = M.window_min(z)
                for t in TARGETS:
                    tau = thresholds[seed][name][str(t)]
                    row[f"{name}_false_alarms_per_h@{t:.0%}"] = M.alarm_events(q, tau).sum() / hours
                    row[f"{name}_time_in_alarm@{t:.0%}"] = float((q[:, M.WARMUP:] > tau).mean())
            for pipe, p in power.items():
                row[f"{pipe} total_uW"] = p["total_uW"]
                row[f"{pipe} network_uW"] = p["network_uW"]
            rows.append(row)
            print(f"[{cname}] seed {seed}: events/s {extras['input_events_per_s']:.0f}, "
                  f"SNN FA/h@5% {row['snn_false_alarms_per_h@5%']:.0f}", flush=True)
    df = pd.DataFrame(rows)
    df.to_csv(f"{OUT}/quiescent_per_seed.csv", index=False)
    summary = df.drop(columns="seed").groupby("condition", sort=False).mean()
    summary.to_csv(f"{OUT}/quiescent.csv")
    print(summary.T.round(3).to_string())

    fig, ax = plt.subplots(1, 2, figsize=(12, 4))
    xs = np.arange(len(summary)); w = 0.16
    for i, pipe in enumerate(E.PIPELINES):
        ax[0].bar(xs + (i - 2) * w, summary[f"{pipe} total_uW"], w, label=pipe)
    ax[0].set(ylabel="Compute power incl. front-end (µW, est.)", title=f"Quiescent power ({a.dur} s x {a.reps} recordings)")
    for name, c in [("snn", "#2a7ab9"), ("cnn", "#d4762c"), ("cnn_gated", "#5a9e4b")]:
        ax[1].bar(xs + {"snn": -w, "cnn": 0, "cnn_gated": w}[name], summary[f"{name}_false_alarms_per_h@5%"], w, label=name, color=c)
    ax[1].set(ylabel="False alarms per hour @ FA 5 %", title="False alarms in long no-leak streams")
    for a in ax:
        a.set_xticks(xs, [c.replace(", ", "\n") for c in summary.index], fontsize=7); a.grid(alpha=.3, axis="y"); a.legend(fontsize=7)
    plt.tight_layout(); plt.savefig(f"{OUT}/figs/quiescent.png", dpi=150); plt.close()
