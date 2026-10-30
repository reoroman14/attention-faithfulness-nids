# Stage 8a2: near-duplicate shares with the Stage 7c direct exact-distance method, both datasets.
# For every TEST row (all classes) of LYCOS-IDS2017 and LycoS-Unicas-IDS2018: the nearest TRAIN row (all classes) in
# that dataset's standardised space (its own preprocess.json), found by direct squared differences on the GPU
# (src/stage7c_explain_2018.exact_nn, never the ||a||^2 + ||b||^2 - 2ab expansion), and the distance to it recomputed in
# float64. Share below 1e-3 per class and over all test rows.
# Read-only on results/stage1, stage2 and stage7; writes results/stage8/ (parts/ per dataset and class chunk for resume,
# neardup_exact_2017_2018.csv summary, neardup_rows_{2017,2018}.parquet per-row distances).
#
# Usage: .venv\Scripts\python.exe -m src.stage8_neardup
import json
import time

import numpy as np
import pandas as pd
import torch

from src.preprocess import Preprocessor
from src.stage7c_explain_2018 import exact_nn
from src.utils import ROOT

OUT = ROOT / "results" / "stage8"
THR = 1e-3
CHUNK = 20000
COMMON = ["dos_slowloris", "bot", "ddos", "dos_goldeneye", "dos_hulk", "ssh_patator", "webattack"]
SETS = {"2017": {"feats": ROOT / "results/stage1/features.json", "labels": ROOT / "results/stage1/label_map.json",
                 "pp": ROOT / "results/stage2/preprocess.json", "train": ROOT / "results/stage1/train.parquet",
                 "test": ROOT / "results/stage1/test.parquet"},
        "2018": {"feats": ROOT / "results/stage7/data/features.json", "labels": ROOT / "results/stage7/data/label_map.json",
                 "pp": ROOT / "results/stage7/preprocess.json", "train": ROOT / "results/stage7/data/train.parquet",
                 "test": ROOT / "results/stage7/data/test.parquet"}}


def main():
    dev = torch.device("cuda")
    (OUT / "parts").mkdir(parents=True, exist_ok=True)
    t_all = time.time()
    summary = []
    for ds, P in SETS.items():
        feats = json.loads(P["feats"].read_text(encoding="utf-8"))
        lm = json.loads(P["labels"].read_text(encoding="utf-8"))
        classes = {v: k for k, v in lm.items()}
        pp = Preprocessor.load(P["pp"])
        Xtr = pp.transform(pd.read_parquet(P["train"], columns=feats))
        te = pd.read_parquet(P["test"], columns=feats + ["label_id"])
        Xte, yte = pp.transform(te), te["label_id"].to_numpy()
        del te
        cls_names = [classes[k] for k in sorted(classes)]
        order = COMMON + [c for c in cls_names if c not in COMMON]       # slowloris first, then common, then rest
        rows = []
        for cn in [c for c in order if c in cls_names]:
            idx = np.flatnonzero(yte == lm[cn])
            for i in range(0, len(idx), CHUNK):
                f = OUT / "parts" / f"{ds}_{cn}_{i // CHUNK:03d}.npz"
                part = idx[i:i + CHUNK]
                if f.exists() and np.array_equal(np.load(f)["test_row"], part):
                    z = np.load(f)
                else:
                    t0 = time.time()
                    nn_i, nn_d = exact_nn(Xte[part], Xtr, dev, 16, 65536)
                    np.savez(f, test_row=part, nn_train_row=nn_i, nn_dist_exact=nn_d)
                    z = np.load(f)
                    print(f"{ds} {cn} rows {i}..{i + len(part)}: {time.time() - t0:.0f} s "
                          f"(share < {THR:g}: {(nn_d < THR).mean():.4f})", flush=True)
                rows.append(pd.DataFrame({"dataset": ds, "class": cn, "test_row": z["test_row"],
                                          "nn_train_row": z["nn_train_row"], "nn_dist_exact": z["nn_dist_exact"]}))
        R = pd.concat(rows, ignore_index=True)
        R.to_parquet(OUT / f"neardup_rows_{ds}.parquet", index=False)
        for cn, g in list(R.groupby("class")) + [("ALL test rows (any class)", R)]:
            d = g.nn_dist_exact.to_numpy()
            summary.append({"dataset": ds, "class": cn, "common_class": cn in COMMON, "n_test_rows": len(d),
                            "n_lt_1e-3": int((d < THR).sum()), "share_lt_1e-3": float((d < THR).mean()),
                            "n_exact_0": int((d == 0).sum()), "median_dist": float(np.median(d)),
                            "p5_dist": float(np.percentile(d, 5)), "n_train_rows": len(Xtr),
                            "method": "direct exact distance to the nearest train row (Stage 7c exact_nn), "
                                      "standardised space of the dataset's own preprocessing"})
        del Xtr
        torch.cuda.empty_cache()
    S = pd.DataFrame(summary)
    S.to_csv(OUT / "neardup_exact_2017_2018.csv", index=False)
    print(S[["dataset", "class", "n_test_rows", "n_lt_1e-3", "share_lt_1e-3", "n_exact_0", "median_dist"]].to_string(index=False))
    print(f"total {time.time() - t_all:.0f} s")


if __name__ == "__main__":
    main()
