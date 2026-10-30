# Stage 8d: margin AOPC gap change (change vs clean, minus the A0 change over all A0 rows; stage5_fullrun.tables,
# "faith_excess_vs_A0") at eps 1.0, split by row status, for both datasets and the row sets V0 / V1 / V2:
#   A1 rows that flipped to benign ("flipped"), A1 rows that did not flip ("not_flipped"), A0 rows that flipped to
#   benign by chance ("to_benign"). From the saved Stage 5 (2017) and Stage 7d (2018) per-row files; no attack is run.
# Writes results/stage8/flip_split.csv (per seed: value and row count) and flip_split_counts.csv.
#
# Usage: set CUDA_VISIBLE_DEVICES=-1 and run .venv\Scripts\python.exe -m src.stage8d_flip_split
import pandas as pd

from src.stage5_fullrun import load_seed, tables
from src.stage8_sensitivity import C5, P, SEEDS, flags, keep_pos
from src.utils import ROOT

OUT = ROOT / "results" / "stage8"
SPLITS = [("A1", "flipped"), ("A1", "not_flipped"), ("A0", "to_benign")]
METHODS = ["ig_zero", "occlusion", "attn_rollout", "attn_mean", "attn_last", "static_rollout", "static_ig", "abs_x",
           "random"]


def main():
    F = {yr: flags(yr) for yr in ("2017", "2018")}
    out, cnt = [], []
    for yr in ("2017", "2018"):
        for s in SEEDS:
            rows, fa, clean = load_seed(P[yr]["s5"] / f"seed{s}")
            for v in ("V0", "V1", "V2"):
                kp = keep_pos(F, yr, v)
                r = rows[rows.pos.isin(kp)]
                T = tables(r.assign(seed=s), fa[fa.pos.isin(kp)].assign(seed=s),
                           clean[clean.pos.isin(kp)].assign(seed=s), None, C5)
                ex = T["faith_excess_vs_A0"]
                ex = ex[(ex["mode"] == "constrained") & (ex.eps == 1.0)]
                for atk, sub in SPLITS:
                    x = ex[(ex.attack == atk) & (ex.subset == sub) & ex.method.isin(METHODS)]
                    for _, y in x.iterrows():
                        out.append({"dataset": yr, "variant": v, "seed": s, "attack": atk, "subset": sub,
                                    "method": y.method, "aopc_gap_margin_minus_A0": y.aopc_gap_margin,
                                    "n_rows": int(y.n_row_obs)})
                    rr = r[(r["mode"] == "constrained") & (r.attack == atk) & (r.eps == 1.0)]
                    cnt.append({"dataset": yr, "variant": v, "seed": s, "attack": atk, "subset": sub,
                                "n_rows": int((rr.subset == sub).sum()), "n_source_rows": len(rr)})
    pd.DataFrame(out).to_csv(OUT / "flip_split.csv", index=False)
    C = pd.DataFrame(cnt)
    C.to_csv(OUT / "flip_split_counts.csv", index=False)
    print(C.groupby(["dataset", "variant", "attack", "subset"]).n_rows.sum().to_string())


if __name__ == "__main__":
    main()
