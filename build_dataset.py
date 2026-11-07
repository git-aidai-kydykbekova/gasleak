"""Step 1: build the synthetic ultrasonic leak dataset and split it 70/15/15 (stratified)."""
import argparse, json, os
import numpy as np
import pandas as pd
from multiprocessing import Pool
from sklearn.model_selection import train_test_split
from gasleak.synth import make_clip, sample_meta
from gasleak.features import band_levels_db, encode_events


def _one(i_seed):
    i, seed = i_seed
    rng = np.random.default_rng(seed)
    label = int(i % 2 == 0)                    # balanced leak / no-leak
    m = sample_meta(rng, label)
    x = make_clip(rng, **m)
    return m, band_levels_db(x)


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--n", type=int, default=3000)
    ap.add_argument("--out", default="data/synth")
    ap.add_argument("--seed", type=int, default=42)
    a = ap.parse_args()
    os.makedirs(a.out, exist_ok=True)
    seeds = np.random.SeedSequence(a.seed).generate_state(a.n)
    with Pool(os.cpu_count()) as p:
        res = p.map(_one, list(enumerate(seeds)), chunksize=20)
    meta = pd.DataFrame([r[0] for r in res])
    L = np.stack([r[1] for r in res])
    E = encode_events(L)
    meta["events_per_s"] = E.sum((1, 2)) / (E.shape[1] / 1000)

    strat = meta.label.astype(str) + "_" + meta.scene + "_" + meta.snr_db.fillna(-99).astype(int).astype(str)
    idx = np.arange(a.n)
    tr, tmp = train_test_split(idx, test_size=0.30, stratify=strat, random_state=a.seed)
    va, te = train_test_split(tmp, test_size=0.50, stratify=strat.iloc[tmp], random_state=a.seed)
    meta["split"] = ""
    meta.loc[tr, "split"], meta.loc[va, "split"], meta.loc[te, "split"] = "train", "val", "test"

    np.savez_compressed(f"{a.out}/dataset.npz", levels=L.astype(np.float16), events=E)
    meta.to_csv(f"{a.out}/meta.csv", index_label="clip_id")
    summary = meta.groupby(["split", "label"]).size().unstack().to_dict()
    print(meta.groupby(["split", "label", "scene"]).size().unstack([1, 2]))
    print(meta.groupby(["label", "scene"]).events_per_s.median().round(0))
    json.dump({"n": a.n, "shape_levels": L.shape, "shape_events": E.shape}, open(f"{a.out}/info.json", "w"))
