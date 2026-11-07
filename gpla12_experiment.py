"""Real-data check on GPLA-12 (gas pipeline leakage acoustics, 12 classes, 684 clips).

Classes = 3 pressures x {clean, strong noise} x 2 microphones (all clips contain a leak).
We test whether the same event-driven SNN recipe works on real leak recordings:
  level-crossing encoder (8 levels x ON/OFF = 16 spike channels) -> LIF -> LIF -> readout(12)
vs a 1-D CNN on the raw signal.  Split: stratified 70/15/15.
"""
import json, os, time
import numpy as np
import torch, torch.nn as nn, torch.nn.functional as F
import snntorch as snn
from snntorch import surrogate
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, f1_score, confusion_matrix

DATA = "data/gpla12"   # folder with data.csv + label.csv
OUT = "runs/gpla12"; os.makedirs(OUT, exist_ok=True)
torch.manual_seed(0); np.random.seed(0); torch.set_num_threads(os.cpu_count())

X = np.loadtxt(f"{DATA}/data.csv", delimiter=",").astype(np.float32)
y = np.loadtxt(f"{DATA}/label.csv", delimiter=",").astype(int) - 1
idx = np.arange(len(y))
tr, tmp = train_test_split(idx, test_size=0.30, stratify=y, random_state=0)
va, te = train_test_split(tmp, test_size=0.50, stratify=y[tmp], random_state=0)
np.savez(f"{OUT}/split.npz", train=tr, val=va, test=te)

# ---- level-crossing encoder (like an asynchronous level-crossing ADC) ----
xn = X / X[tr].max()
levels = np.linspace(0.05, 0.95, 8)
above = (xn[:, :, None] > levels[None, None]).astype(np.int8)          # [N,T,8]
d = np.diff(above, axis=1, prepend=above[:, :1])
EV = np.concatenate([(d > 0), (d < 0)], -1).astype(np.float32)           # [N,T,16]
print("events per sample (mean):", EV.sum((1, 2)).mean().round(1), "of", EV.shape[1], "time steps")


class SNN(nn.Module):
    def __init__(s, n_in=16, h=64, n_out=12):
        super().__init__(); sg = surrogate.fast_sigmoid(25)
        s.fc1, s.l1 = nn.Linear(n_in, h), snn.Leaky(0.95, learn_beta=True, spike_grad=sg)
        s.fc2, s.l2 = nn.Linear(h, h), snn.Leaky(0.95, learn_beta=True, spike_grad=sg)
        s.fc3, s.o = nn.Linear(h, n_out), snn.Leaky(0.99, learn_beta=True, reset_mechanism="none")
    def forward(s, x):
        m1, m2, m3 = s.l1.reset_mem(), s.l2.reset_mem(), s.o.reset_mem(); acc, n1, n2 = 0, 0, 0
        for t in range(x.shape[1]):
            s1, m1 = s.l1(s.fc1(x[:, t]), m1); s2, m2 = s.l2(s.fc2(s1), m2); _, m3 = s.o(s.fc3(s2), m3)
            acc = acc + m3; n1 = n1 + s1.sum(1); n2 = n2 + s2.sum(1)
        return acc / x.shape[1], n1, n2


class CNN(nn.Module):
    def __init__(s, n_out=12):
        super().__init__()
        s.c = nn.Sequential(nn.Conv1d(1, 16, 9, 2), nn.ReLU(), nn.Conv1d(16, 32, 9, 2), nn.ReLU(),
                            nn.Conv1d(32, 32, 9, 2), nn.ReLU(), nn.AdaptiveAvgPool1d(1))
        s.fc = nn.Linear(32, n_out)
    def forward(s, x): return s.fc(s.c(x.unsqueeze(1)).squeeze(-1))
    def macs(s, T=1460):
        m, L = 0, T
        for l in s.c:
            if isinstance(l, nn.Conv1d):
                L = (L - l.kernel_size[0]) // l.stride[0] + 1; m += L * l.in_channels * l.out_channels * l.kernel_size[0]
        return m + 32 * 12


def train(kind, inp, epochs, bs=32, lr=2e-3):
    model = SNN() if kind == "snn" else CNN()
    opt = torch.optim.Adam(model.parameters(), lr=lr); best, state = -1, None
    T = torch.from_numpy(inp); Y = torch.from_numpy(y)
    def fwd(b):
        o = model(T[b]); return o[0] if kind == "snn" else o
    for ep in range(epochs):
        model.train(); t0 = time.time()
        for b in np.array_split(np.random.permutation(tr), len(tr) // bs):
            loss = F.cross_entropy(fwd(b), Y[b]); opt.zero_grad(); loss.backward(); opt.step()
        model.eval()
        with torch.no_grad(): acc = (fwd(va).argmax(1) == Y[va]).float().mean().item()
        if acc >= best: best, state = acc, {k: v.clone() for k, v in model.state_dict().items()}
        print(f"[gpla12 {kind}] ep {ep+1:02d} val_acc {acc:.3f} ({time.time()-t0:.0f}s)", flush=True)
    model.load_state_dict(state); model.eval()
    with torch.no_grad():
        o = model(T[te]); pred = (o[0] if kind == "snn" else o).argmax(1).numpy()
    r = dict(val_acc=best, test_acc=accuracy_score(y[te], pred), test_macro_f1=f1_score(y[te], pred, average="macro"))
    if kind == "snn":
        n_in = EV[te].sum((1, 2)).mean(); n1, n2 = o[1].mean().item(), o[2].mean().item()
        r["synops_per_clip"] = float(n_in * 64 + n1 * 64 + n2 * 12)
        r["dense_twin_macs_per_clip"] = (16 * 64 + 64 * 64 + 64 * 12) * EV.shape[1]
    else:
        r["macs_per_clip"] = model.macs()
    np.save(f"{OUT}/{kind}_confusion.npy", confusion_matrix(y[te], pred))
    return r


res = {"snn": train("snn", EV, 30), "cnn": train("cnn", xn, 60)}
json.dump(res, open(f"{OUT}/results.json", "w"), indent=2, default=float)
print(json.dumps(res, indent=2, default=float))
