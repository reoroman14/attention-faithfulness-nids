# Shared readers for the Stage 8a tables, figures and NUMBERS.md (paper/make_tables_v2.py, paper/make_figures_v2.py).
# Read-only on results/. Every number is read from a result file or recomputed from saved per-row result files with
# the project's own code (src/stage7e_defense_2018.py); nothing is typed in.
#
# 2017 vs 2018 comparisons use the 7 attack classes common to both datasets. The 2017 class-restricted values come
# from results/stage7/stage7c/stage7c_cmp2017_*.csv (clean data), results/stage7/stage7d/cmp2017_seed{s}_*.csv
# (attacks) and, for the defenders, from the saved 2017 per-row files results/stage6/final/seed{s}/rows_*.parquet
# and results/stage6/strong/seed{s}/rows_*.parquet restricted to the 7 classes (the same computation that produced
# results/stage7/stage7e/stage7e_cmp2017_headline.csv).
import glob
import json
import re
import sys
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
RES = ROOT / "results"
S7 = RES / "stage7"
sys.path.insert(0, str(ROOT))

SEEDS = [0, 1, 2]
COMMON = ["bot", "ddos", "dos_goldeneye", "dos_hulk", "dos_slowloris", "ssh_patator", "webattack"]
M7 = "macro_common7"
CLASS_NAMES = {
    "benign": "Benign", "bot": "Bot", "ddos": "DDoS", "dos_goldeneye": "DoS GoldenEye", "dos_hulk": "DoS Hulk",
    "dos_slowhttptest": "DoS Slowhttptest", "dos_slowloris": "DoS Slowloris", "ftp_patator": "FTP-Patator",
    "portscan": "PortScan", "ssh_patator": "SSH-Patator", "webattack": "Web attack", "heartbleed": "Heartbleed",
    "dos_slowloris_excl_neardup": "DoS Slowloris without near duplicates", "every class": "every class",
}
A6N = ["A6_lam1", "A6_lam100", "A6norm_w0.5"]
A8N = ["A8_lam1", "A8_lam100", "A8norm_w0.5"]
ATTACK_NAMES = {"A1": "A1", "A6_lam1": r"A6 $\lambda{=}1$", "A6_lam100": r"A6 $\lambda{=}100$",
                "A6norm_w0.5": "A6 norm.", "A8_lam1": r"A8 $\lambda{=}1$", "A8_lam100": r"A8 $\lambda{=}100$",
                "A8norm_w0.5": "A8 norm.", "A3_lam100": r"A3 $\lambda{=}100$", "A3_lam1000": r"A3 $\lambda{=}1000$",
                "A3norm_w0.5": "A3 norm.", "A2": "A2", "A0": "A0", "A1b": "A1b"}


def rel(p):
    return Path(p).resolve().relative_to(ROOT).as_posix()


# ---------------------------------------------------------------- T1 datasets

def datasets():
    # Rows per class and split after cleaning, for both datasets, plus the removed classes.
    c17 = pd.read_csv(RES / "stage1_class_counts.csv")
    d17 = c17[(c17.label_id >= 0)][["class", "train", "val", "test"]].assign(dataset="2017")
    steps = pd.read_csv(S7 / "stage7b_counts_by_step.csv")
    sp = steps[steps.step.str.startswith("6 split")].assign(split=lambda d: d.step.str.replace("6 split ", ""))
    d18 = sp.pivot_table(index="class", columns="split", values="rows", aggfunc="sum").reset_index()
    d18 = d18[["class", "train", "val", "test"]].assign(dataset="2018")
    rep = pd.read_csv(RES / "stage1_cleaning_report.csv")
    sql = sum(json.loads(x).get("webattack_sql_injection", 0) for x in rep.dropped_labels)
    hb = int(c17.loc[c17["class"] == "heartbleed", "raw_labelled"].iloc[0])
    raw18 = steps[steps.step == "1 raw"].groupby("raw_label")["rows"].sum()
    nf = n_features()
    removed = [
        {"dataset": "2017", "label": "Heartbleed", "rows": hb, "reason": "too few rows for a train / val / test split",
         "source": rel(RES / "stage1_class_counts.csv")},
        {"dataset": "2017", "label": "Web Attack SQL Injection", "rows": int(sql),
         "reason": "all rows fall after the temporal cut (test only)", "source": rel(RES / "stage1_cleaning_report.csv")},
        {"dataset": "2018", "label": "FTP-Patator", "rows": int(raw18["FTP-Patator"]),
         "reason": f"indistinguishable from DoS Slowhttptest in the {nf} features",
         "source": rel(S7 / "stage7b_counts_by_step.csv")},
        {"dataset": "2018", "label": "DoS Slowhttptest", "rows": int(raw18["DoS Slowhttptest"]),
         "reason": f"indistinguishable from FTP-Patator in the {nf} features",
         "source": rel(S7 / "stage7b_counts_by_step.csv")},
    ]
    return pd.concat([d17, d18], ignore_index=True), pd.DataFrame(removed)


# ---------------------------------------------------------------- T2 performance

def performance():
    p = pd.read_csv(RES / "stage2_per_class_f1.csv", dtype={"seed": str})
    p = p[(p["class"] == "macro_avg") & (p.split == "test")]
    q = pd.read_csv(S7 / "stage7b_per_class.csv")
    q = q[(q["class"] == "macro_avg") & (q.split == "test")]
    rows = []
    for yr, d, src in (("2017", p, RES / "stage2_per_class_f1.csv"), ("2018", q, S7 / "stage7b_per_class.csv")):
        ftt = {int(s): float(d[(d.model == "ftt") & (d.seed.astype(str) == str(s))]["f1"].iloc[0]) for s in SEEDS}
        xg = float(d[d.model == "xgb"]["f1"].iloc[0])
        rows.append({"dataset": yr, **{f"ftt_seed{s}": v for s, v in ftt.items()}, "ftt_mean": np.mean(list(ftt.values())),
                     "ftt_min": min(ftt.values()), "ftt_max": max(ftt.values()), "xgb": xg, "source": rel(src)})
    return pd.DataFrame(rows)


# ---------------------------------------------------------------- T3 clean faithfulness and agreement

def clean_cmp():
    a = pd.read_csv(S7 / "stage7c" / "stage7c_cmp2017_agreement.csv")
    f = pd.read_csv(S7 / "stage7c" / "stage7c_cmp2017_faithfulness.csv")
    return a[a["class"] == M7], f[f["class"] == M7]


# ---------------------------------------------------------------- T5 / T6 attacks (per-seed Stage 5 summary tables)

def stage5_tables(year, name):
    # Per-seed Stage 5 summary tables on the 7 common classes (2018: all 7 attack classes).
    if year == "2018":
        files = [S7 / "stage7d" / f"seed{s}" / f"summary_{name}.csv" for s in SEEDS]
    else:
        files = [S7 / "stage7d" / f"cmp2017_seed{s}_{name}.csv" for s in SEEDS]
    return [pd.read_csv(f).assign(seed=s) for s, f in zip(SEEDS, files)], [rel(f) for f in files]


def success_long(year):
    T, src = stage5_tables(year, "success")
    return pd.concat(T, ignore_index=True), src


def excess_long(year):
    T, src = stage5_tables(year, "faith_excess_vs_A0")
    return pd.concat(T, ignore_index=True), src


def agg_seeds(df, keys, col):
    g = df.groupby(keys)[col]
    return pd.DataFrame({"mean": g.mean(), "lo": g.min(), "hi": g.max(), "n_seeds": g.count()}).reset_index()


# ---------------------------------------------------------------- defenders (per-row files)

def defender_rows(year):
    if year == "2018":
        files = sorted(glob.glob(str(S7 / "stage7e" / "seed*" / "rows_eps*_*.parquet")))
        R = pd.concat([pd.read_parquet(f) for f in files], ignore_index=True)
        src = "results/stage7/stage7e/seed{0,1,2}/rows_eps*_*.parquet"
    else:
        files = sorted(glob.glob(str(RES / "stage6" / "final" / "seed*" / "rows_eps*_*.parquet"))) + \
            sorted(glob.glob(str(RES / "stage6" / "strong" / "seed*" / "rows_eps*_*.parquet")))
        R = pd.concat([pd.read_parquet(f) for f in files], ignore_index=True)
        R = R[R["class"].isin(COMMON)]
        sd = lambda p: ",".join(sorted(Path(x).name[4:] for x in glob.glob(str(p / "seed*")) if Path(x).is_dir()))
        src = (f"results/stage6/final/seed{{{sd(RES / 'stage6' / 'final')}}}/rows_eps*_*.parquet and "
               f"results/stage6/strong/seed{{{sd(RES / 'stage6' / 'strong')}}}/rows_eps1_A8*.parquet (7 common classes)")
    return R, src


def undet(r, det):
    return float((r.flipped & (r[f"score_{det}"] < r[f"thr_{det}"])).mean())


def tpr(r, det):
    f = r.flipped.to_numpy()
    return float((r[f"score_{det}"] >= r[f"thr_{det}"]).to_numpy()[f].mean()) if f.any() else np.nan


def defender_table(R, eps=1.0, attacks=("A1",) + tuple(A6N) + tuple(A8N)):
    # Success, success AND undetected and TPR per attack, pooled and per seed (min / max over seeds).
    out = []
    for a in attacks:
        r = R[(R.eps == eps) & (R.attack == a)]
        if not len(r):
            continue
        per = {s: g for s, g in r.groupby("seed")}
        rec = {"attack": a, "seeds": ",".join(str(int(s)) for s in per), "n_rows": len(r)}
        for k, fn in (("success", lambda x: float(x.flipped.mean())), ("undet_D", lambda x: undet(x, "D")),
                      ("undet_Dplus", lambda x: undet(x, "Dplus")), ("tpr_D", lambda x: tpr(x, "D")),
                      ("tpr_Dplus", lambda x: tpr(x, "Dplus"))):
            v = [fn(g) for g in per.values()]
            rec[k], rec[f"{k}_lo"], rec[f"{k}_hi"] = fn(r), np.nanmin(v), np.nanmax(v)
        out.append(rec)
    return pd.DataFrame(out)


def defender_meta(year):
    # Per seed and eps: validation AUROC of D and D_plus, training positives / negatives, held-out negatives (for the
    # threshold), test negatives and the realised test FPR, all read from the saved Stage 6 FINAL / 7e files.
    import pickle
    base = RES / "stage6" / "final" if year == "2017" else S7 / "stage7e"
    out = []
    for s in SEEDS:
        sd = base / f"seed{s}"
        dfn = pd.read_csv(sd / "summary_defenders.csv")
        det = pd.read_csv(sd / "summary_detection.csv")
        NEG = pickle.loads((sd / "negatives.pkl").read_bytes())
        for _, r in dfn.iterrows():
            fpr = det[det.eps == r.eps].groupby("detector").test_fpr.agg(["min", "max"])
            out.append({"dataset": year, "seed": s, "eps": r.eps, "val_auroc_D": r.val_auroc_D,
                        "val_auroc_Dplus": r.val_auroc_Dplus, "n_train_pos": int(r.n_val_pos),
                        "n_train_neg": len(NEG["val_train"]), "n_heldout_neg": len(NEG["val_held"]),
                        "n_test_neg": len(NEG["test"]), "test_fpr_D": fpr.loc["D", "min"],
                        "test_fpr_D_max": fpr.loc["D", "max"], "test_fpr_Dplus": fpr.loc["D_plus", "min"],
                        "test_fpr_Dplus_max": fpr.loc["D_plus", "max"],
                        "source": f"{rel(sd / 'summary_defenders.csv')}; {rel(sd / 'summary_detection.csv')}; "
                                  f"{rel(sd / 'negatives.pkl')}"})
    return pd.DataFrame(out)


def equal_effort_table(year, eps=1.0):
    # PRIMARY comparison: per seed and pooled. 2018 from the Stage 7e summary; 2017 recomputed with the same function
    # (src/stage7e_defense_2018.equal_effort, rng seed 4026 + 992 as in Stage 7e) on the seeds where A8 exists.
    if year == "2018":
        f = S7 / "stage7e" / "summary_pooled_union_equal_effort.csv"
        d = pd.read_csv(f)
        return d[d.eps == eps].reset_index(drop=True), rel(f), None
    from src.stage7e_defense_2018 import equal_effort
    R, src = defender_rows("2017")
    R = R[R.eps == eps]
    seeds = sorted(R[R.attack.isin(A8N)].seed.unique())
    R = R[R.seed.isin(seeds)]
    rng = np.random.default_rng(4026 + 992)
    rows, units = [], []
    for lab, sd in [(f"seed {int(s)}", [s]) for s in seeds] + [("pooled", seeds)]:
        u, d = equal_effort(R[R.seed.isin(sd)], 1000, rng)
        rows.append({"eps": eps, "seeds": lab, **d})
        if lab == "pooled":
            units = u
    return pd.DataFrame(rows), src, units


def per_class_attacks(year):
    if year == "2018":
        f = S7 / "stage7e" / "summary_pooled_per_class_attacks.csv"
        return pd.read_csv(f), rel(f)
    from src.stage7e_defense_2018 import a8_per_class
    R, src = defender_rows("2017")
    R = R[R.eps == 1.0]
    parts = [a8_per_class(R[R.attack.isin(["A1", "A6_lam1"])]), a8_per_class(R[R.attack == "A8_lam1"])]
    return pd.concat(parts, ignore_index=True), src


def per_class_union(year):
    if year == "2018":
        f = S7 / "stage7e" / "summary_pooled_per_class_union.csv"
        d = pd.read_csv(f)
        return d[d.eps == 1.0], rel(f)
    _, src, u = equal_effort_table("2017")
    g = u.groupby("class")
    d = pd.DataFrame({"n": g.size() / u.seed.nunique(), "union_undet_D": g.union_undet_D.mean(),
                      "union_undet_Dplus": g.union_undet_Dplus.mean()}).reset_index()
    return d, src


# ---------------------------------------------------------------- T10 limitations

def seeds_by_stage():
    def dirs(p):
        return sorted(int(Path(x).name[4:]) for x in glob.glob(str(p / "seed*")) if Path(x).is_dir())

    def col(f, c="seed"):
        return sorted(int(x) for x in pd.read_csv(f)[c].dropna().unique() if str(x).isdigit())

    p2 = pd.read_csv(RES / "stage2_per_class_f1.csv", dtype={"seed": str})
    # Labels name the pipeline step by its function (LaTeX, not escaped); the scripts are stage2 to stage7e in src/.
    rows = [
        ("2017", "FT-T training", sorted(int(s) for s in p2.seed.unique() if s.isdigit()), RES / "stage2_per_class_f1.csv"),
        ("2017", "Clean explanations", col(RES / "stage3_faithfulness.csv"), RES / "stage3_faithfulness.csv"),
        ("2017", "Explanation diagnostics", col(RES / "stage3c_faithfulness.csv"), RES / "stage3c_faithfulness.csv"),
        ("2017", "Parameter randomisation", col(RES / "stage3_randomisation.csv"), RES / "stage3_randomisation.csv"),
        ("2017", "Constrained attacks", dirs(RES / "stage5"), RES / "stage5"),
        ("2017", r"Defenders D and D$_+$", dirs(RES / "stage6" / "final"), RES / "stage6" / "final"),
        ("2017", "Adaptive attack A8", dirs(RES / "stage6" / "strong"), RES / "stage6" / "strong"),
        ("2017", r"Controls for D$_+$", col(RES / "stage9" / "ablation" / "2017" / "stageA_defenders.csv"),
         RES / "stage9" / "ablation" / "2017" / "stageA_defenders.csv"),
        ("2018", "FT-T training", dirs(S7), S7),
        ("2018", "Clean explanations", col(S7 / "stage7c" / "stage7c_faithfulness.csv"), S7 / "stage7c" / "stage7c_faithfulness.csv"),
        ("2018", "Explanation diagnostics", col(S7 / "stage7c" / "stage7c_3c_faithfulness.csv"),
         S7 / "stage7c" / "stage7c_3c_faithfulness.csv"),
        ("2018", "Parameter randomisation", col(S7 / "stage7c" / "stage7c_randomisation.csv"),
         S7 / "stage7c" / "stage7c_randomisation.csv"),
        ("2018", "Constrained attacks", dirs(S7 / "stage7d"), S7 / "stage7d"),
        ("2018", r"Defenders D and D$_+$; attacks A6, A8", dirs(S7 / "stage7e"), S7 / "stage7e"),
        ("2018", r"Controls for D$_+$", col(RES / "stage9" / "ablation" / "2018" / "stageA_defenders.csv"),
         RES / "stage9" / "ablation" / "2018" / "stageA_defenders.csv"),
    ]
    return pd.DataFrame([{"dataset": d, "stage": s, "seeds": ", ".join(map(str, sd)), "source": rel(src)}
                         for d, s, sd, src in rows])


def small_n():
    # Every per-class row count below 100 in the explanation, attack and defender sets.
    out = []

    def add(ds, what, d, cls_col, n_col, src):
        for _, r in d.iterrows():
            if r[cls_col] != "benign" and pd.notna(r[n_col]) and r[n_col] < 100:
                out.append({"dataset": ds, "set": what, "class": r[cls_col], "n": int(r[n_col]), "source": rel(src)})

    for ds, f in (("2017", RES / "stage3_selection_counts.csv"), ("2018", S7 / "stage7c" / "stage7c_selection_counts.csv")):
        add(ds, "clean explanation set (test)", pd.read_csv(f), "class", "selected", f)
    for ds, f in (("2017", RES / "stage3c_val_selection_counts.csv"),
                  ("2018", S7 / "stage7c" / "stage7c_3c_val_selection_counts.csv")):
        add(ds, "VAL explanation set (head choice, defender positives)", pd.read_csv(f), "class", "selected", f)
    f = S7 / "stage7d" / "stage7d_source_counts.csv"
    add("2018", "attack and defender source rows", pd.read_csv(f), "class", "used", f)
    for ds, f in (("2017", RES / "stage3_randomisation.csv"), ("2018", S7 / "stage7c" / "stage7c_randomisation.csv")):
        r = pd.read_csv(f)
        r = r[(r.seed == 0) & (r.level == "head_only") & (r.method == "ig_zero") & (r.metric == "spearman_vs_original")
              & ~r["class"].str.startswith("macro_") & (r["class"] != "benign")]
        reg = r[~r["class"].str.endswith("_excl_neardup")]
        out.append({"dataset": ds, "set": "randomisation subset (by design)", "class": "every class",
                    "n": int(reg.n.max()), "source": rel(f)})
        add(ds, "randomisation subset", r[r["class"].str.endswith("_excl_neardup")], "class", "n", f)
    return pd.DataFrame(out)


def randomisation():
    # Parameter randomisation check (Stage 3 / 7c): Spearman and top-k Jaccard between the explanation of the original
    # and of the partly re-initialised model, per row of the randomisation subset (all its rows). Per seed: macro over
    # the 7 common classes; then mean and range over seeds.
    out = []
    for ds, f in (("2017", RES / "stage3_randomisation.csv"), ("2018", S7 / "stage7c" / "stage7c_randomisation.csv")):
        r = pd.read_csv(f)
        r = r[r["class"].isin(COMMON)]
        for (lv, m, met), g in r.groupby(["level", "method", "metric"]):
            s = g.groupby("seed")["mean"].agg(["mean", "count"])
            s = s.loc[s["count"] == len(COMMON), "mean"]
            out.append({"dataset": ds, "level": lv, "method": m, "metric": met, "mean": s.mean(), "lo": s.min(),
                        "hi": s.max(), "n_seeds": len(s), "n_min": int(g.n.min()), "n_max": int(g.n.max()),
                        "source": rel(f)})
    return pd.DataFrame(out)


ABL = RES / "stage9" / "ablation"
ABL_VARIANTS = ["a1", "a2", "b", "c0", "c1", "c2"]        # a1s / a2s: the same controls with a second shuffle seed


def dplus_ablation():
    # Controls for D+ (Stage 9, src/stage9_dplus_ablation.py): detection summaries (pooled over seeds; seed range),
    # paired TPR differences (paired bootstrap over flipped rows pooled over seeds, threshold-centred per seed) and the
    # validation AUROC per detector. Read-only.
    files = {k: ABL / f"summary_{k}.csv" for k in ("detection", "paired", "defenders")}
    out = {k: pd.read_csv(f) for k, f in files.items()}
    for d in out.values():
        d["year"] = d["year"].astype(str)
    out["sources"] = [rel(f) for f in files.values()]
    return out


def neardup():
    # results/stage8/neardup_exact_2017_2018.csv: direct exact distance, every test row of both datasets.
    f = RES / "stage8" / "neardup_exact_2017_2018.csv"
    d = pd.read_csv(f)
    d = d[d.common_class | d["class"].str.startswith("ALL")]
    cur = pd.DataFrame({"dataset": d.dataset.astype(str), "class": d["class"], "n": d.n_test_rows,
                        "n_neardup": d["n_lt_1e-3"], "share_neardup": d["share_lt_1e-3"], "source": rel(f)})
    order = {c: i for i, c in enumerate(["dos_slowloris"] + [c for c in COMMON if c != "dos_slowloris"])}
    return cur.assign(_o=cur["class"].map(order).fillna(99)).sort_values(["dataset", "_o"]).drop(columns="_o") \
        .reset_index(drop=True)


SENS = "results/stage8/"


def sens(name):
    # Stage 8a3 sensitivity outputs (src/stage8_sensitivity.py).
    return pd.read_csv(RES / "stage8" / f"sensitivity_{name}.csv")


def source_flags(year):
    # pos (selection position) -> near-duplicate flag (exact nearest-train distance < 1e-3), Stage 8a2 per-row file.
    sel = pd.read_parquet(RES / "stage3" / "selection.parquet" if year == "2017"
                          else S7 / "stage7c" / "raw" / "selection.parquet")
    nd = pd.read_parquet(RES / "stage8" / f"neardup_rows_{year}.parquet").set_index("test_row")["nn_dist_exact"]
    return pd.Series(nd.reindex(sel["test_row"]).to_numpy() < 1e-3, index=np.arange(len(sel)))


def neardup_split():
    # TPR on flipped A6 lambda 1 / A8 lambda 1 rows and FPR on the benign test negatives, flagged vs other rows, for D and
    # D_plus, both datasets, eps 0.5 and 1.0 (7 common classes, all source rows; the split itself is by the flag).
    out = []
    ng = sens("negatives")
    for yr in ("2017", "2018"):
        R, _ = defender_rows(yr)
        fl = source_flags(yr)
        R = R.assign(nd=fl.reindex(R.pos).to_numpy())
        for eps in (0.5, 1.0):
            for atk, lab in (("A6_lam1", "TPR, A6 lambda 1 rows"), ("A8_lam1", "TPR, A8 lambda 1 rows")):
                r = R[(R.eps == eps) & (R.attack == atk) & R.flipped]
                rec = {"dataset": yr, "eps": eps, "quantity": lab}
                for sub, x in (("flagged", r[r.nd]), ("other", r[~r.nd])):
                    rec[f"n_{sub}"] = len(x)
                    for det in ("D", "Dplus"):
                        rec[f"{det}_{sub}"] = float((x[f"score_{det}"] >= x[f"thr_{det}"]).mean()) if len(x) else np.nan
                out.append(rec)
            x = ng[(ng.dataset == int(yr)) & (ng.seed.astype(str) == "pooled") & (ng.eps == eps)].iloc[0]
            out.append({"dataset": yr, "eps": eps, "quantity": "FPR, benign test negatives",
                        "n_flagged": int(x.n_neardup), "n_other": int(x.n_non_neardup),
                        "D_flagged": x.fpr_D_neardup, "Dplus_flagged": x.fpr_Dplus_neardup,
                        "D_other": x.fpr_D_non_neardup, "Dplus_other": x.fpr_Dplus_non_neardup})
    return pd.DataFrame(out)


REPORTED_ATTACKS = ["A0", "A1", "A1b", "A2", "A3_lam100", "A3_lam1000", "A3norm_w0.5"] + A6N + A8N


def reported_attack(path):
    # Attack name of a saved per-row file: rows_constrained_{attack}_eps{e}.parquet or rows_eps{e}_{attack}.parquet.
    name = Path(path).stem
    m = re.fullmatch(r"rows_constrained_(.+)_eps[0-9.]+", name) or re.fullmatch(r"rows_eps[0-9.]+_(.+)", name)
    return m is not None and m.group(1) in REPORTED_ATTACKS


def invalid_constrained_rows():
    # Invalid constrained adversarial rows over the saved files of the attacks reported in the paper (validity from the
    # checker). A7 and the single-score A6 variants (A6_msp_only, A6_knn5_only) are not reported and not counted.
    n_bad, n_all = 0, 0
    for f in glob.glob(str(RES / "stage5" / "seed*" / "rows_constrained_*.parquet")) + \
            glob.glob(str(S7 / "stage7d" / "seed*" / "rows_constrained_*.parquet")) + \
            glob.glob(str(RES / "stage6" / "final" / "seed*" / "rows_eps*_*.parquet")) + \
            glob.glob(str(RES / "stage6" / "strong" / "seed*" / "rows_eps*_*.parquet")) + \
            glob.glob(str(S7 / "stage7e" / "seed*" / "rows_eps*_*.parquet")):
        if not reported_attack(f):
            continue
        v = pd.read_parquet(f, columns=["valid"])["valid"]
        n_bad += int((~v.astype(bool)).sum())
        n_all += len(v)
    return n_bad, n_all


def n_features():
    return len(json.loads((S7 / "data" / "features.json").read_text(encoding="utf-8")))


def constraint_summary():
    f = RES / "stage4_constraints.csv"
    sp = pd.read_csv(f, keep_default_na=False)
    roles = sp.role.value_counts().to_dict()
    fams = {}
    for _, r in sp.iterrows():
        for fam in [x for x in str(r.frozen_for_class_families).split(";") if x]:
            fams.setdefault(fam, []).append(r.feature)
    fz18 = pd.read_csv(S7 / "stage7d" / "stage7d_frozen_share.csv")
    fz17 = pd.read_csv(S7 / "stage7d" / "stage7d_frozen_share_2017.csv")
    return sp, roles, fams, fz17, fz18, rel(f)
