# One-off fix (2026-10-04): the first full Stage 2 run saved the XGBoost booster with all 70 boosted
# rounds, but metrics were computed with the best validation iteration (19 -> 20 trees). A loaded booster
# predicts with all trees by default, so this script:
#   1. keeps the original file as results/stage2/xgb_all_rounds.json,
#   2. writes results/stage2/xgb.json truncated to best_iteration + 1 trees (no retraining),
#   3. checks that the truncated model reproduces the reported val/test macro-F1 exactly.
# Usage: .venv\Scripts\python.exe -m src.fix_xgb_checkpoint
import json
import shutil
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import xgboost as xgb

from src.metrics import macro_f1
from src.preprocess import Preprocessor
from src.utils import RESULTS, ROOT

s2 = ROOT / "results" / "stage2"
src_path, full_path, out_path = s2 / "xgb.json", s2 / "xgb_all_rounds.json", s2 / "xgb.json"

bst = xgb.Booster()
bst.load_model(src_path)
best = int(bst.attributes()["best_iteration"])
print(f"loaded {src_path.name}: {bst.num_boosted_rounds()} rounds, best_iteration={best}")
if bst.num_boosted_rounds() == best + 1:
    print("already truncated; nothing to do")
    sys.exit(0)
if not full_path.exists():
    shutil.copy2(src_path, full_path)
    print(f"kept original as {full_path.name}")
trunc = bst[: best + 1]
trunc.save_model(out_path)

check = xgb.Booster()
check.load_model(out_path)
pp = Preprocessor.load(s2 / "preprocess.json")
lm = json.loads((ROOT / "results" / "stage1" / "label_map.json").read_text())
rep = pd.read_csv(RESULTS / "stage2_per_class_f1.csv")
ok = True
for split in ("val", "test"):
    df = pd.read_parquet(ROOT / "results" / "stage1" / f"{split}.parquet")
    y = df["label_id"].to_numpy()
    p = check.predict(xgb.DMatrix(pp.transform(df))).argmax(1)
    got = macro_f1(y, p, len(lm))
    ref = float(rep[(rep.model == "xgb") & (rep.split == split) & (rep["class"] == "macro_avg")]["f1"].iloc[0])
    match = abs(got - ref) < 1e-9
    ok &= match
    print(f"{split}: truncated model macro-F1 {got:.6f} vs reported {ref:.6f} -> {'MATCH' if match else 'MISMATCH'}")
print(f"saved {out_path.name} with {check.num_boosted_rounds()} rounds")
sys.exit(0 if ok else 1)
