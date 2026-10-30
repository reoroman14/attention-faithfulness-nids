# Stage 8b: clean-data agreement and faithfulness (table T3 quantities) for the variants V0 / V1 / V2, from the saved
# per-row Stage 3 (2017) and Stage 7c (2018) outputs. No model is run.
#   per-row deletion / insertion AUC: trapezoid of the stored target-probability curves (train-mean masking),
#     dx = 1 / 72, exactly as stage3_explain.curve_metrics
#   per-row agreement: Spearman of the stored attention vs |attribution| (stage3_explain.rowwise_spearman); the stored
#     attributions are float16 (Stage 3 computed agreement on float32 in memory), so V0 reproduces T3 to about 1e-3
#   class means per seed, unweighted macro over the 7 common classes (V2: 5 classes), mean / min / max over seeds
#   V0 all explanation-set rows; V1 rows whose test row has a train row within 1e-3 (results/stage8/neardup_rows_*)
#   removed; V2 dos_slowloris and webattack removed
# Also the Spearman correlation between the 2017 and 2018 orderings of the 8 methods (from the 7-class means).
# Writes results/stage8/clean_variants.csv, clean_variants_ordering.csv, clean_variants_checks.csv.
#
# Usage: .venv\Scripts\python.exe -m src.stage8b_clean_v1
import numpy as np
import pandas as pd
from scipy.stats import spearmanr

from src.stage3_explain import rowwise_spearman
from src.utils import ROOT

RES = ROOT / "results"
OUT = RES / "stage8"
COMMON = ["bot", "ddos", "dos_goldeneye", "dos_hulk", "dos_slowloris", "ssh_patator", "webattack"]
V2_DROP = ["dos_slowloris", "webattack"]
METHODS = ["attn_rollout", "attn_mean", "attn_last", "ig_zero", "gshap", "occlusion", "abs_x", "random"]
ATTN = ["attn_rollout", "attn_mean", "attn_last"]
REFS = ["ig_zero", "gshap", "occlusion", "treeshap_xgb"]
SRC = {"2017": (RES / "stage3", RES / "stage3" / "selection.parquet"),
       "2018": (RES / "stage7" / "stage7c" / "raw", RES / "stage7" / "stage7c" / "raw" / "selection.parquet")}


def main():
    rows, checks = [], []
    for yr, (d, selp) in SRC.items():
        sel = pd.read_parquet(selp)
        nd = pd.read_parquet(OUT / f"neardup_rows_{yr}.parquet").set_index("test_row")["nn_dist_exact"]
        flag = nd.reindex(sel["test_row"]).to_numpy() < 1e-3
        cls = sel["class"].to_numpy()
        masks = {"V0": np.isin(cls, COMMON), "V1": np.isin(cls, COMMON) & ~flag,
                 "V2": np.isin(cls, COMMON) & ~np.isin(cls, V2_DROP)}
        ts = np.abs(np.load(d / "treeshap_xgb.npz")["treeshap"].astype(np.float32))
        for s in (0, 1, 2):
            cur = np.load(d / f"curves_seed{s}.npz")
            att = np.load(d / f"attr_seed{s}.npz")
            per_row = {}
            for m in METHODS:
                for metric, key in (("del_auc", "del_morf"), ("ins_auc", "ins_morf")):
                    c = cur[f"train_mean/{m}/{key}"].astype(np.float64)
                    per_row[(metric, m, "")] = np.trapezoid(c, dx=1 / (c.shape[1] - 1), axis=1)
            sc = {k: att[k].astype(np.float32) for k in ATTN}
            sc.update({k: np.abs(att[k].astype(np.float32)) for k in ("ig_zero", "gshap", "occlusion")})
            sc["treeshap_xgb"] = ts
            for a in ATTN:
                for r in REFS:
                    per_row[("spearman", a, r)] = rowwise_spearman(sc[a], sc[r])
            for v, mk in masks.items():
                for (metric, a, b), vals in per_row.items():
                    cm = [np.nanmean(vals[mk & (cls == c)]) for c in COMMON if (mk & (cls == c)).any()]
                    rows.append({"dataset": yr, "variant": v, "seed": s, "metric": metric, "method": a, "ref": b,
                                 "value": float(np.mean(cm)), "n_classes": len(cm), "n_rows": int(mk.sum())})
    R = pd.DataFrame(rows)
    R.to_csv(OUT / "clean_variants.csv", index=False)
    # Ordering of the 8 methods (7-class means over seeds), Spearman between datasets, per variant and metric.
    mean = R[R.metric != "spearman"].groupby(["dataset", "variant", "metric", "method"]).value.mean().reset_index()
    ordr = []
    for (v, metric), g in mean.groupby(["variant", "metric"]):
        p = g.pivot(index="method", columns="dataset", values="value").loc[METHODS]
        asc = metric == "del_auc"
        r17 = p["2017"].rank(ascending=asc, method="min").astype(int)
        r18 = p["2018"].rank(ascending=asc, method="min").astype(int)
        ordr.append({"variant": v, "metric": metric, "spearman_orderings": spearmanr(p["2017"], p["2018"]).statistic,
                     **{f"rank2017_{m}": r17[m] for m in METHODS}, **{f"rank2018_{m}": r18[m] for m in METHODS}})
    pd.DataFrame(ordr).to_csv(OUT / "clean_variants_ordering.csv", index=False)
    # V0 against the T3 sources (stage7c cmp2017 files, macro over the 7 common classes).
    a = pd.read_csv(RES / "stage7" / "stage7c" / "stage7c_cmp2017_agreement.csv")
    f = pd.read_csv(RES / "stage7" / "stage7c" / "stage7c_cmp2017_faithfulness.csv")
    v0 = R[R.variant == "V0"].groupby(["dataset", "metric", "method", "ref"]).value.mean()
    for yr in ("2017", "2018"):
        dmax_f = max(abs(v0[(yr, mt, m, "")] - f[(f["class"] == "macro_common7") & (f.mask_baseline == "train_mean")
                                                 & (f.metric == mt) & (f.method == m)][f"mean_{yr}"].iloc[0])
                     for mt in ("del_auc", "ins_auc") for m in METHODS)
        dmax_a = max(abs(v0[(yr, "spearman", x, r)] - a[(a["class"] == "macro_common7") & (a.metric == "spearman")
                                                       & (a.method_a == x) & (a.method_b == r)][f"mean_{yr}"].iloc[0])
                     for x in ATTN for r in REFS)
        checks.append({"check": f"{yr} V0 deletion / insertion AUC vs T3 source (max abs diff; float16 curves)",
                       "value": dmax_f, "pass": dmax_f < 5e-3})
        checks.append({"check": f"{yr} V0 Spearman agreement vs T3 source (max abs diff; float16 attributions)",
                       "value": dmax_a, "pass": dmax_a < 5e-3})
    C = pd.DataFrame(checks)
    C.to_csv(OUT / "clean_variants_checks.csv", index=False)
    print(C.to_string(index=False))
    print(pd.DataFrame(ordr)[["variant", "metric", "spearman_orderings"]].to_string(index=False))
    if not C["pass"].all():
        raise SystemExit(2)


if __name__ == "__main__":
    main()
