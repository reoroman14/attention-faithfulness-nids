# Stage 8c: checks before the 2017 A8 seed 1 rows (results/stage6/strong/seed1/) are used.
#   1. A8 stand-in quality (R^2, Spearman of each differentiable stand-in vs the true consistency feature) on 2017
#      VALIDATION rows at eps 1.0 (the only eps of Stage 6e), for seeds 0, 1 and 2 with the Stage 6e procedure (rng
#      offsets + 9100 + s and + 9200 + s, so seed 0 is exactly the Stage 6e check and must reproduce
#      results/stage6/strong/standin_quality.csv).
#   2. The seed 1 A8 source rows are the Stage 6 FINAL seed 1 rows (same pos and class, same order).
#   3. The Stage 6e run log reports the independent validity check for every seed 1 variant (the run stops on any
#      invalid row; Stage 6e does not save the raw adversarial rows, so the check cannot be repeated offline).
# Writes results/stage8/stage8c_standin_quality_2017.csv and results/stage8/stage8c_checks.csv.
#
# Usage: .venv\Scripts\python.exe -m src.stage8c_verify_a8_seed1
import copy
import logging

import numpy as np
import pandas as pd
import torch

from src.stage3_explain import load_ftt
from src.stage4_attack import STAGE4
from src.stage6_defense import STAGE6
from src.stage6b_defense import STAGE6B
from src.stage6c_defense import STAGE6C
from src.stage6d_final import STAGE6D, load_common
from src.stage6e_strong import STAGE6E
from src.stage7e_defense_2018 import standin_quality
from src.utils import ROOT

RES = ROOT / "results"
OUT = RES / "stage8"


def main():
    c = {**copy.deepcopy(STAGE4), **copy.deepcopy(STAGE6), **copy.deepcopy(STAGE6B), **copy.deepcopy(STAGE6C),
         **copy.deepcopy(STAGE6D), **copy.deepcopy(STAGE6E)}
    dev = torch.device("cuda")
    log = logging.getLogger("stage8c")
    logging.basicConfig(level=logging.INFO)
    checks = []
    G = load_common(c, dev)
    parts = []
    for s in (0, 1, 2):
        m, _ = load_ftt(G.s2 / f"ftt_seed{s}.pt", dev)
        for p in m.parameters():
            p.requires_grad_(False)
        parts.append(standin_quality(m, G, c, dev, s, log).assign(eps=1.0))
        del m
        torch.cuda.empty_cache()
    S = pd.concat(parts, ignore_index=True)
    S.to_csv(OUT / "stage8c_standin_quality_2017.csv", index=False)
    old = pd.read_csv(RES / "stage6" / "strong" / "standin_quality.csv").set_index("feature")
    s0 = S[S.seed == 0].set_index("feature")
    d = float(max((s0.r2 - old.loc[s0.index, "r2"]).abs().max(), (s0.spearman - old.loc[s0.index, "spearman"]).abs().max()))
    checks.append({"check": "seed 0 stand-in quality reproduces results/stage6/strong/standin_quality.csv (max abs diff)",
                   "value": d, "pass": d < 1e-6})
    for s, g in S.groupby("seed"):
        checks.append({"check": f"seed {s} stand-in median R^2 at eps 1.0 (n rows {int(g.n_rows.iloc[0])})",
                       "value": float(g.r2.median()), "pass": True})
    # Source rows: Stage 6e seed 1 rows vs Stage 6 FINAL seed 1 rows.
    fin = pd.read_parquet(RES / "stage6" / "final" / "seed1" / "rows_eps1_A6_lam1.parquet", columns=["pos", "class"])
    for n in ("A8_lam1", "A8_lam100", "A8norm_w0.5"):
        r = pd.read_parquet(RES / "stage6" / "strong" / "seed1" / f"rows_eps1_{n}.parquet", columns=["pos", "class", "valid"])
        same = len(r) == len(fin) and (r.pos.to_numpy() == fin.pos.to_numpy()).all() and \
            (r["class"].to_numpy() == fin["class"].to_numpy()).all()
        checks.append({"check": f"seed 1 {n}: same pos / class / order as Stage 6 FINAL seed 1 ({len(r)} rows)",
                       "value": bool(same), "pass": bool(same)})
        checks.append({"check": f"seed 1 {n}: rows flagged valid", "value": int(r.valid.sum()), "pass": bool(r.valid.all())})
    con = (RES / "stage6" / "strong_s8c_staging" / "console.txt").read_text(encoding="utf-8", errors="replace")
    ok = all(f"seed 1 {n}: success" in con for n in ("A8_lam1", "A8_lam100", "A8norm_w0.5")) and "STOP" not in con
    checks.append({"check": "Stage 6e seed 1 run: every variant passed check_valid_or_stop (no STOP in the log)",
                   "value": ok, "pass": ok})
    C = pd.DataFrame(checks)
    C.to_csv(OUT / "stage8c_checks.csv", index=False)
    print(S.groupby("seed").r2.median().round(3).to_string())
    print(C.to_string(index=False))
    if not C["pass"].all():
        raise SystemExit(2)


if __name__ == "__main__":
    main()
