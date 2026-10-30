# Read-only reproduction check for the Stage 2 artifacts (writes nothing).
# Loads results/stage2/ftt_seed{0,1,2}.pt, xgb.json and preprocess.json, re-predicts val and test from
# results/stage1/{val,test}.parquet, and compares every per-class / macro / weighted F1 and accuracy with
# results/stage2_per_class_f1.csv, and the macro-F1 means with results/stage2_summary.csv.
# Values are derived from integer confusion counts, so identical predictions give identical numbers;
# the tolerance below only absorbs float round-off from the CSV round trip.
# Usage: .venv\Scripts\python.exe -m src.check_stage2
import json
import sys

import numpy as np
import pandas as pd
import torch
import xgboost as xgb

from src.metrics import per_class_report
from src.models.ft_transformer import FTTransformer
from src.preprocess import Preprocessor
from src.stage2_train import predict_torch
from src.utils import RESULTS

TOL = 1e-9
S1, S2 = RESULTS / "stage1", RESULTS / "stage2"


def compare(model, seed, split, rep, ref_all):
    ref = ref_all[(ref_all["model"] == model) & (ref_all["seed"] == str(seed)) & (ref_all["split"] == split)]
    m = rep.merge(ref, on="class", suffixes=("", "_ref"), validate="one_to_one")
    assert len(m) == len(rep) == len(ref), (model, seed, split, len(rep), len(ref))
    diff = {c: float(np.nanmax(np.abs(m[c] - m[f"{c}_ref"]))) for c in ("precision", "recall", "f1")}
    sup_ok = bool((m["support"] == m["support_ref"]).all())
    ok = max(diff.values()) < TOL and sup_ok
    macro = float(rep.loc[rep["class"] == "macro_avg", "f1"].iloc[0])
    macro_ref = float(m.loc[m["class"] == "macro_avg", "f1_ref"].iloc[0])
    print(f"  {model:4s} seed={seed} {split:4s} macro-F1 {macro:.6f} vs {macro_ref:.6f}  "
          f"max|diff| P/R/F1 over {len(rep)} rows = {max(diff.values()):.1e}  support {'ok' if sup_ok else 'DIFF'}"
          f"  -> {'MATCH' if ok else 'MISMATCH'}")
    return ok, macro


def main():
    feats = json.loads((S1 / "features.json").read_text(encoding="utf-8"))
    label_map = json.loads((S1 / "label_map.json").read_text(encoding="utf-8"))
    classes = [c for c, _ in sorted(label_map.items(), key=lambda kv: kv[1])]
    pp = Preprocessor.load(S2 / "preprocess.json")
    assert pp.features == feats, "preprocess.json features differ from stage1 features.json"
    ref_all = pd.read_csv(RESULTS / "stage2_per_class_f1.csv", dtype={"seed": str})
    summary = pd.read_csv(RESULTS / "stage2_summary.csv")

    data = {}
    for split in ("val", "test"):
        df = pd.read_parquet(S1 / f"{split}.parquet", columns=feats + ["label_id"])
        data[split] = (pp.transform(df), df["label_id"].to_numpy().astype(np.int64))
        print(f"{split}: {len(df)} rows")

    ok_all, macros = True, {}
    dev = torch.device("cuda")
    print("FT-Transformer:")
    for seed in (0, 1, 2):
        ckpt = torch.load(S2 / f"ftt_seed{seed}.pt", map_location=dev, weights_only=False)
        assert ckpt["features"] == feats and ckpt["classes"] == classes, f"seed {seed}: feature/class mismatch"
        model = FTTransformer(**ckpt["model_kwargs"]).to(dev)
        model.load_state_dict(ckpt["state_dict"])
        bs = ckpt["config"]["stage2"]["ftt"]["eval_batch_size"]
        for split, (X, y) in data.items():
            pred = predict_torch(model, torch.from_numpy(X).to(dev), bs).argmax(1).cpu().numpy()
            ok, macro = compare("ftt", seed, split, per_class_report(y, pred, classes), ref_all)
            ok_all &= ok
            macros.setdefault(("ftt", split), []).append(macro)
        del model
        torch.cuda.empty_cache()

    print("XGBoost:")
    bst = xgb.Booster()
    bst.load_model(S2 / "xgb.json")
    # Slicing drops the best_iteration attribute, so read it from the untruncated model.
    full = xgb.Booster()
    full.load_model(S2 / "xgb_all_rounds.json")
    best = int(full.attributes()["best_iteration"])
    ok = bst.num_boosted_rounds() == best + 1
    ok_all &= ok
    print(f"  xgb.json: {bst.num_boosted_rounds()} rounds; xgb_all_rounds.json: {full.num_boosted_rounds()} rounds, "
          f"best_iteration={best} -> {'consistent' if ok else 'INCONSISTENT'}")
    xgb_seed = ref_all.loc[ref_all["model"] == "xgb", "seed"].iloc[0]
    for split, (X, y) in data.items():
        pred = bst.predict(xgb.DMatrix(X)).argmax(1)
        ok, macro = compare("xgb", xgb_seed, split, per_class_report(y, pred, classes), ref_all)
        ok_all &= ok
        macros[("xgb", split)] = [macro]

    print("Summary (stage2_summary.csv, macro_f1):")
    for (model, split), vals in macros.items():
        row = summary[(summary["model"] == model) & (summary["split"] == split) & (summary["metric"] == "macro_f1")].iloc[0]
        per_seed = ";".join(f"{v:.4f}" for v in vals)
        ok = abs(np.mean(vals) - row["mean"]) < TOL and per_seed == row["per_seed"]
        ok_all &= ok
        print(f"  {model:4s} {split:4s} mean {np.mean(vals):.6f} vs {row['mean']:.6f}, per-seed {per_seed} "
              f"vs {row['per_seed']} -> {'MATCH' if ok else 'MISMATCH'}")

    print("ALL MATCH" if ok_all else "SOME MISMATCHES")
    sys.exit(0 if ok_all else 1)


if __name__ == "__main__":
    main()
