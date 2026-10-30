# Stage 4c: clean-data control for the static-attention finding.
# Does class-averaging an explanation help every method, or only attention? For each FT-T seed, train-set
# class means (2,000 sampled train rows per class, same sample as stage4b_followup) of:
#   static_rollout     attention rollout
#   static_attn_mean   CLS attention averaged over layers and heads
#   static_ig          |IG zero| (50 steps, train-mean baseline, target = the train row's label)
#   static_occlusion   |LOFO occlusion| (train mean, target = label)
# are used as rankings (the vector of the row's true class, identical for every row of the class) in the
# Stage 3 faithfulness metrics and masking baselines on the clean Stage 3 explanation set. Dynamic versions are
# the Stage 3 results (same code path; stage4b_followup reproduced them exactly for attention).
# The static vectors are saved to results/stage4/static_vectors_seed{s}.npz for reuse by the full run.
#
# Usage:
#   .venv\Scripts\python.exe -m src.stage4c_static_control
import copy
import json
import time

import numpy as np
import pandas as pd
import torch

from src import config as cfgmod
from src.config import load_config
from src.preprocess import Preprocessor
from src.stage3_explain import (attention_variants, class_rows, curve_metrics, integrated_gradients, load_ftt,
                                occlusion, perturbation_curves, ranking)
from src.stage4_attack import STAGE4
from src.utils import finish_run, resolve, start_run

STAGE4C = {"seeds": [0, 1, 2], "train_rows_per_class": 2000, "stage3_topk": [1, 3, 5, 10, 20, 36]}
STATIC = ["static_rollout", "static_attn_mean", "static_ig", "static_occlusion"]
DYNAMIC_OF = {"static_rollout": "attn_rollout", "static_attn_mean": "attn_mean", "static_ig": "ig_zero",
              "static_occlusion": "occlusion"}


def static_vectors(model, Ztr, ytr_cls, c, dev):
    # Class means over the sampled train rows of each class: {name: (C, F)}.
    out = {k: [] for k in STATIC}
    for ci in sorted(Ztr):
        Z = Ztr[ci]
        tgt = torch.full((Z.shape[0],), ci, dtype=torch.long, device=dev)
        att, _ = attention_variants(model, Z, c["attn_batch"])
        ig, _, _ = integrated_gradients(model, Z, tgt, torch.zeros(1, Z.shape[1], device=dev), c)
        occ = occlusion(model, Z, tgt, torch.zeros_like(Z), c)
        out["static_rollout"].append(att["attn_rollout"].mean(0))
        out["static_attn_mean"].append(att["attn_mean"].mean(0))
        out["static_ig"].append(np.abs(ig).mean(0))
        out["static_occlusion"].append(np.abs(occ).mean(0))
    return {k: np.stack(v) for k, v in out.items()}


def main():
    cfgmod.DEFAULTS.setdefault("stage4c", {**copy.deepcopy(STAGE4), **copy.deepcopy(STAGE4C)})
    cfg, overrides = load_config("Stage 4c: static explanation control on clean data.")
    c = cfg["stage4c"]
    t0 = time.time()
    run_dir, log = start_run("stage4c_static_control", cfg, overrides)
    dev = torch.device("cuda")
    s1, s2, s3 = resolve("results/stage1"), resolve("results/stage2"), resolve("results/stage3")
    out = resolve("results/stage4")
    feats = json.loads((s1 / "features.json").read_text(encoding="utf-8"))
    lm = json.loads((s1 / "label_map.json").read_text(encoding="utf-8"))
    classes = [k for k, _ in sorted(lm.items(), key=lambda kv: kv[1])]
    pp = Preprocessor.load(s2 / "preprocess.json")
    sel = pd.read_parquet(s3 / "selection.parquet")
    y = sel["label_id"].to_numpy().astype(np.int64)
    test = pd.read_parquet(s1 / "test.parquet", columns=feats)
    Xs = torch.from_numpy(pp.transform(test.iloc[sel["test_row"].to_numpy()])).to(dev)
    del test
    train = pd.read_parquet(s1 / "train.parquet", columns=feats + ["label_id"])
    ytr = train["label_id"].to_numpy()
    # Same train sample as stage4b_followup (rng_seed + 500).
    rng = np.random.default_rng(c["rng_seed"] + 500)
    per = c["train_rows_per_class"]
    tr_idx = {ci: rng.choice(np.flatnonzero(ytr == ci), size=min(per, int((ytr == ci).sum())), replace=False)
              for ci in range(len(classes))}
    mask_rows = np.load(s3 / "baseline_rows.npz")["mask_benign_train_rows"]
    bases = {"train_mean": torch.zeros_like(Xs),
             "benign_sample": torch.from_numpy(pp.transform(train.iloc[mask_rows])).to(dev)}
    Ztr = {ci: torch.from_numpy(pp.transform(train.iloc[ix])).to(dev) for ci, ix in tr_idx.items()}
    del train
    target = torch.from_numpy(y).to(dev)
    faith = []
    for s in c["seeds"]:
        model, _ = load_ftt(s2 / f"ftt_seed{s}.pt", dev)
        ts = time.time()
        sv = static_vectors(model, Ztr, None, c, dev)
        np.savez(out / f"static_vectors_seed{s}.npz", classes=np.array(classes),
                 train_rows={ci: ix for ci, ix in tr_idx.items()}.__repr__(), **sv)
        log.info(f"seed {s}: static vectors in {time.time() - ts:.0f} s")
        rng_s = np.random.default_rng(c["rng_seed"] + 700 + s)
        for mname in STATIC:
            order = torch.from_numpy(ranking(sv[mname][y], rng_s)).to(dev)
            for bname, base in bases.items():
                cur = perturbation_curves(model, Xs, base, order, target, c)
                for metric, vals in curve_metrics(cur, c["stage3_topk"], c["aopc_steps"], c["norm_min_gap"]).items():
                    faith += class_rows(vals, y, classes, normalised="_norm" in metric, seed=s, mask_baseline=bname,
                                        method=mname, metric=metric)
        log.info(f"seed {s} done ({time.time() - ts:.0f} s)")
        del model
        torch.cuda.empty_cache()
    fdf = pd.DataFrame(faith)
    fdf.to_csv(out / "pilot_followup" / "s4c_static_control_faithfulness.csv", index=False)
    s3f = pd.read_csv(resolve("results/stage3_faithfulness.csv"))
    keep = ["del_auc", "ins_auc", "aopc_gap", "comp@5", "suff@20", "del_auc_margin", "ins_auc_margin"]
    H = "macro_attack_classes"
    st = fdf[(fdf["class"] == H) & fdf.metric.isin(keep)].groupby(["mask_baseline", "method", "metric"])["mean"].mean()
    dy = s3f[(s3f["class"] == H) & s3f.metric.isin(keep)].groupby(["mask_baseline", "method", "metric"])["mean"].mean()
    rows = []
    for b in ("train_mean", "benign_sample"):
        for sm in STATIC:
            dm = DYNAMIC_OF[sm]
            for m in keep:
                vs, vd = st.get((b, sm, m), np.nan), dy.get((b, dm, m), np.nan)
                rows.append({"mask_baseline": b, "method": dm, "metric": m, "dynamic": vd, "static": vs,
                             "static_minus_dynamic": vs - vd})
    cmp = pd.DataFrame(rows)
    cmp.to_csv(out / "pilot_followup" / "s4c_static_vs_dynamic.csv", index=False)
    pd.set_option("display.width", 250)
    log.info("Static minus dynamic (macro attack classes, mean over seeds):\n"
             + cmp.pivot_table(index=["mask_baseline", "method"], columns="metric",
                               values=["dynamic", "static"]).round(4).to_string())
    finish_run(run_dir, log, {"seeds": c["seeds"]}, t0)


if __name__ == "__main__":
    main()
