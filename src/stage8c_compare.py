# Stage 8c: compare the refreshed Stage 8 sensitivity outputs (2017 A8 now on seeds 0, 1, 2) with the snapshot taken
# before the refresh (2017 A8 on seeds 0 and 2). Every seed 0 / seed 2 value and every 2018 value must be unchanged;
# bootstrap CIs of the 2017 union are reported separately because their draws come from one sequential stream over
# the seeds (adding seed 1 shifts the stream for seed 2 and pooled).
# Writes results/stage8/stage8c_seed02_unchanged.csv.
#
# Usage: .venv\Scripts\python.exe -m src.stage8c_compare <snapshot dir with the old sensitivity_*.csv>
import sys
from pathlib import Path

import numpy as np
import pandas as pd

from src.utils import ROOT

NEW = ROOT / "results" / "stage8"


def diff(a, b, keys, cols):
    m = a.merge(b, on=keys, suffixes=("_new", "_old"), how="inner")
    if not len(m):
        return np.nan, 0
    return float(max((m[f"{c}_new"] - m[f"{c}_old"]).abs().max() for c in cols)), len(m)


def main(snap):
    snap = Path(snap)
    out = []
    o = lambda n: pd.read_csv(snap / f"sensitivity_{n}.csv")
    n = lambda n: pd.read_csv(NEW / f"sensitivity_{n}.csv")
    # Defender rates per seed (seed_only rows): 2017 seeds 0 and 2, and every 2018 row.
    a, b = n("defenders"), o("defenders")
    cols = ["n_rows", "success", "undet_D", "undet_Dplus", "tpr_D", "tpr_Dplus"]
    for lab, sel in (("2017 seed 0 and seed 2 defender rows (per seed)",
                      lambda d: d[(d.dataset == 2017) & d.seed_only.isin([0, 2])]),
                     ("2018 defender rows (pooled and per seed)", lambda d: d[d.dataset == 2018])):
        x, k = diff(sel(a).fillna({"seed_only": -1}), sel(b).fillna({"seed_only": -1}),
                    ["dataset", "variant", "attack", "seed_only"], cols)
        out.append({"comparison": lab, "rows_compared": k, "max_abs_diff": x})
    # Equal-effort union: point estimates per seed, CIs separately.
    a, b = n("union"), o("union")
    pts = ["n_source_rows", "succ_undet_union_D", "succ_undet_union_Dplus", "Dplus_minus_D"]
    sel = lambda d: d[(d.dataset == 2017) & d.seeds.isin(["seed 0", "seed 2"])]
    x, k = diff(sel(a), sel(b), ["dataset", "variant", "eps", "seeds"], pts)
    out.append({"comparison": "2017 union, seed 0 and seed 2 point estimates", "rows_compared": k, "max_abs_diff": x})
    for s in ("seed 0", "seed 2"):
        sel = lambda d, s=s: d[(d.dataset == 2017) & (d.seeds == s)]
        x, k = diff(sel(a), sel(b), ["dataset", "variant", "eps", "seeds"], ["ci_lo", "ci_hi"])
        out.append({"comparison": f"2017 union, {s} bootstrap CI (sequential stream; seed 1 now drawn before seed 2)",
                    "rows_compared": k, "max_abs_diff": x})
    sel = lambda d: d[d.dataset == 2018]
    x, k = diff(sel(a), sel(b), ["dataset", "variant", "eps", "seeds"], pts + ["ci_lo", "ci_hi"])
    out.append({"comparison": "2018 union (all rows incl. CI)", "rows_compared": k, "max_abs_diff": x})
    # Attack-stage quantities and 2018 paired / per-class / negatives: no A8 involved, must be identical.
    a, b = n("stage5"), o("stage5")
    x, k = diff(a, b, ["dataset", "variant", "seed", "quantity", "mode", "attack", "eps", "method"], ["value"])
    out.append({"comparison": "Stage 5 quantities, both datasets", "rows_compared": k, "max_abs_diff": x})
    a, b = n("paired"), o("paired")
    x, k = diff(a[a.dataset == 2018], b[b.dataset == 2018], ["dataset", "variant", "attack"],
                ["n_pos", "tpr_diff", "tpr_diff_lo", "tpr_diff_hi"])
    out.append({"comparison": "2018 paired TPR (incl. CI)", "rows_compared": k, "max_abs_diff": x})
    a, b = n("negatives"), o("negatives")
    x, k = diff(a, b, ["dataset", "seed", "eps"], ["n_neardup", "fpr_D_neardup", "fpr_Dplus_neardup",
                                                   "fpr_D_non_neardup", "fpr_Dplus_non_neardup"])
    out.append({"comparison": "defender negatives, both datasets", "rows_compared": k, "max_abs_diff": x})
    R = pd.DataFrame(out)
    R["unchanged"] = R.max_abs_diff.fillna(0) < 1e-12
    R.to_csv(NEW / "stage8c_seed02_unchanged.csv", index=False)
    print(R.to_string(index=False))


if __name__ == "__main__":
    main(sys.argv[1])
