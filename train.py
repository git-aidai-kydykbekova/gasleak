"""Train the SNN (on events) and the CNN baseline (on dense band levels).

Per-time-step target: 0 before the leak onset, 1 after it.  The first GRACE ms after
onset are not scored (the detector needs a little evidence).  No-leak clips: 0 everywhere.
"""
import argparse, json, os, time
import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F
from gasleak.models import LeakSNN, CausalCNN

GRACE = 30   # ms


def load(path="data/synth"):
    d = np.load(f"{path}/dataset.npz")
    meta = pd.read_csv(f"{path}/meta.csv")
    L = d["levels"].astype(np.float32)
    floor = np.median(L[:, :100], axis=1, keepdims=True)
    dense = np.concatenate([L - floor, (L + 60) / 20], -1)          # CNN input [N,T,32]
    ev = d["events"].astype(np.float32)                                # SNN input [N,T,32]
    T = L.shape[1]
    t = np.arange(T)[None]
    on = np.where(meta.label.values == 1, (meta.onset_s.fillna(9).values * 1000).round(), 1e9)[:, None]
    y = (t >= on).astype(np.float32)
    mask = ~((t >= on) & (t < on + GRACE))
    return meta, ev, dense, y, mask.astype(np.float32)


def step_loss(logits, y, m):
    return (F.binary_cross_entropy_with_logits(logits, y, reduction="none") * m).sum() / m.sum()


def run_epoch(model, X, Y, M, idx, bs, opt=None, kind="snn"):
    train = opt is not None
    model.train(train)
    tot, n = 0.0, 0
    order = np.random.permutation(idx) if train else idx
    for i in range(0, len(order), bs):
        b = order[i:i + bs]
        x, y, m = (torch.from_numpy(a[b]) for a in (X, Y, M))
        with torch.set_grad_enabled(train):
            if kind == "snn":
                logits, s1, s2 = model(x)
                loss = step_loss(logits, y, m) + 1e-3 * (s1.mean() + s2.mean())  # mild sparsity reg.
            else:
                loss = step_loss(model(x), y, m)
        if train:
            opt.zero_grad(); loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0); opt.step()
        tot += loss.item() * len(b); n += len(b)
    return tot / n


def fit(kind, meta, X, Y, M, epochs, bs, lr, out):
    torch.manual_seed(0); np.random.seed(0)
    model = LeakSNN() if kind == "snn" else CausalCNN()
    opt = torch.optim.Adam(model.parameters(), lr=lr)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, epochs)
    tr = np.where(meta.split == "train")[0]; va = np.where(meta.split == "val")[0]
    best, hist = 1e9, []
    for ep in range(epochs):
        t0 = time.time()
        ltr = run_epoch(model, X, Y, M, tr, bs, opt, kind)
        lva = run_epoch(model, X, Y, M, va, 128, None, kind)
        sched.step()
        hist.append(dict(epoch=ep + 1, train_loss=ltr, val_loss=lva))
        print(f"[{kind}] epoch {ep+1:02d}  train {ltr:.4f}  val {lva:.4f}  ({time.time()-t0:.0f}s)", flush=True)
        if lva < best:
            best = lva; torch.save(model.state_dict(), f"{out}/{kind}.pt")
    pd.DataFrame(hist).to_csv(f"{out}/{kind}_history.csv", index=False)


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", choices=["snn", "cnn"], required=True)
    ap.add_argument("--epochs", type=int, default=25)
    ap.add_argument("--bs", type=int, default=32)
    ap.add_argument("--lr", type=float, default=2e-3)
    ap.add_argument("--out", default="runs")
    a = ap.parse_args()
    os.makedirs(a.out, exist_ok=True)
    torch.set_num_threads(os.cpu_count())
    meta, ev, dense, y, mask = load()
    fit(a.model, meta, ev if a.model == "snn" else dense, y, mask, a.epochs, a.bs, a.lr, a.out)
