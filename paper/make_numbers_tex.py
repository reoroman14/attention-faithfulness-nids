# Writes paper/generated/numbers.tex: one \newcommand per number used in the prose of paper/sections/*.tex, each with
# its rounding and a comment line naming its source file. Read-only on results/ (through paper/data_v2.py and the
# Stage 8 result files). Method parameters are read from the project configuration modules (src/*.py) or result files.
#
# Usage (from the project root): .venv\Scripts\python.exe paper\make_numbers_tex.py
import json
import sys
from datetime import date
from pathlib import Path

import numpy as np
import pandas as pd

import data_v2 as D

sys.path.insert(0, str(D.ROOT))
from src.stage3_explain import STAGE3  # noqa: E402
from src.stage4_attack import STAGE4  # noqa: E402
from src.stage5_fullrun import STAGE5  # noqa: E402
from src.stage6_defense import STAGE6  # noqa: E402
from src.stage6b_defense import STAGE6B  # noqa: E402
from src.stage6c_defense import STAGE6C  # noqa: E402
from src.stage6d_final import STAGE6D  # noqa: E402
from src.stage6e_strong import STAGE6E  # noqa: E402

OUT = Path(__file__).resolve().parent / "generated"
RES, S7, S8 = D.RES, D.S7, D.RES / "stage8"
MACROS = []                 # (name, body, source)
YR = {2017: "Seventeen", 2018: "Eighteen", "2017": "Seventeen", "2018": "Eighteen"}
VN = {"V0": "Vzero", "V1": "Vone", "V2": "Vtwo"}


def fmt(v, d=3):
    t = f"{v:.{d}f}"
    if t.startswith("-") and float(t) == 0:
        t = t[1:]
    return t


def num(v, d=3):
    return r"\ensuremath{" + fmt(v, d).replace("-", "-") + "}"


def rng(lo, hi, d=3):
    return r"\ensuremath{" + fmt(lo, d) + r"}~to~\ensuremath{" + fmt(hi, d) + "}"


def fmt_bound(v, d=3):
    # Interval bound: keeps the sign of a slightly negative bound that rounds to zero (-0.000), so that the interval
    # does not look as if it excluded zero (Stage 9m).
    return f"{v:.{d}f}"


def ci(lo, hi, d=3):
    return r"\ensuremath{[" + fmt_bound(lo, d) + ", " + fmt_bound(hi, d) + "]}"


def intc(v):
    return f"{int(v):,}".replace(",", "{,}")


def add(name, body, source):
    assert name.isalpha(), name
    assert name not in [m[0] for m in MACROS], name
    MACROS.append((name, body, source))


def mean_range(name, vals, source, d=3):
    v = np.asarray(vals, float)
    add(name, num(v.mean(), d), source)
    add(name + "Range", rng(v.min(), v.max(), d), source)


# ---------------------------------------------------------------- parameters and setup

def params():
    add("nFeatures", str(D.n_features()), "results/stage7/data/features.json")
    sp = pd.read_csv(RES / "stage4_constraints.csv", keep_default_na=False)
    for role, nm in (("control", "nControls"), ("derived", "nDerived"), ("immutable", "nImmutable")):
        add(nm, str(int((sp.role == role).sum())), "results/stage4_constraints.csv")
    add("nSeeds", str(len(STAGE5["seeds"])), "src/stage5_fullrun.py STAGE5['seeds']")
    add("paramTopK", str(STAGE4["a2_k"]), "src/stage4_attack.py STAGE4['a2_k']")
    add("paramSteps", str(STAGE4["steps"]), "src/stage4_attack.py STAGE4['steps']")
    add("paramFPR", f"{STAGE6['fpr'] * 100:g}\\%", "src/stage6_defense.py STAGE6['fpr']")
    add("paramBoot", intc(STAGE6B["n_boot"]), "src/stage6b_defense.py STAGE6B['n_boot']")
    add("paramNegTest", intc(STAGE6B["n_test_benign"]), "src/stage6b_defense.py STAGE6B['n_test_benign']")
    add("paramNegTrain", intc(STAGE6C["d_train_neg"]), "src/stage6c_defense.py STAGE6C['d_train_neg']")
    add("paramNegHeld", intc(STAGE6D["n_heldout"]), "src/stage6d_final.py STAGE6D['n_heldout']")
    add("paramPerClass", str(STAGE5["n_per_class"]), "src/stage5_fullrun.py STAGE5['n_per_class']")
    add("paramExplPerClass", str(STAGE3["n_per_class"]), "src/stage3_explain.py STAGE3['n_per_class']")
    add("paramIGLoss", str(STAGE6E["ig_steps_loss"]), "src/stage6e_strong.py STAGE6E['ig_steps_loss']")
    add("paramIGDet", str(STAGE6["ig_steps_det"]), "src/stage6_defense.py STAGE6['ig_steps_det']")
    add("paramKnn", str(STAGE6["knn_k"]), "src/stage6_defense.py STAGE6['knn_k']")
    add("paramKappa", f"{STAGE6C['kappa_a6']:g}", "src/stage6c_defense.py STAGE6C['kappa_a6']")
    add("paramNormW", f"{STAGE6D['norm_w']:g}", "src/stage6d_final.py STAGE6D['norm_w']")
    add("paramNDThr", r"\ensuremath{10^{-3}}", "src/stage8_neardup.py THR")
    add("nCommonClasses", str(len(D.COMMON)), "paper/data_v2.py COMMON")
    add("nVtwoClasses", str(len(D.COMMON) - 2), "paper/data_v2.py COMMON minus dos_slowloris, webattack")
    o = pd.read_csv(RES / "stage8" / "clean_variants_ordering.csv")
    add("nMethods", str(sum(c.startswith("rank2017_") for c in o.columns)), "results/stage8/clean_variants_ordering.csv")
    s = pd.read_csv(RES / "stage7" / "stage7e" / "standin_quality.csv")
    add("nConsistency", str(s[s.seed == s.seed.iloc[0]].feature.nunique()), "results/stage7/stage7e/standin_quality.csv")
    add("paramCILevel", "95\\%", "src/stage6d_final.py paired_boot and src/stage6e_strong.py paired_union (2.5 / 97.5 percentiles)")
    add("paramEpsHead", f"{STAGE5['headline_max_eps']:g}", "src/stage5_fullrun.py STAGE5['headline_max_eps']")


def datasets():
    d, rem = D.datasets()
    for yr in ("2017", "2018"):
        x = d[d.dataset == yr]
        src = "results/stage1_class_counts.csv" if yr == "2017" else "results/stage7/stage7b_counts_by_step.csv"
        for k in ("train", "val", "test"):
            add(f"nRows{k.capitalize()}{YR[yr]}", intc(x[k].sum()), src)
        add(f"nClasses{YR[yr]}", str(len(x)), src)
    x = d[(d.dataset == "2018") & (d["class"] == "webattack")]
    add("nWebTestEighteen", intc(x.test.iloc[0]), "results/stage7/stage7b_counts_by_step.csv")
    vs = pd.read_csv(S7 / "stage7c" / "stage7c_3c_val_selection_counts.csv")
    add("nWebValEighteen", str(int(vs[vs["class"] == "webattack"].selected.iloc[0])),
        "results/stage7/stage7c/stage7c_3c_val_selection_counts.csv")
    for _, r in rem.iterrows():
        nm = {"Heartbleed": "nHeartbleed", "Web Attack SQL Injection": "nSqlInjection", "FTP-Patator": "nFtpRemoved",
              "DoS Slowhttptest": "nSlowhttpRemoved"}[r.label]
        add(nm, intc(r.rows), r.source)
    p = D.performance()
    for _, r in p.iterrows():
        add(f"fOneFTT{YR[r.dataset]}", num(r.ftt_mean, 4), r.source)
        add(f"fOneFTT{YR[r.dataset]}Range", rng(r.ftt_min, r.ftt_max, 4), r.source)
        add(f"fOneXGB{YR[r.dataset]}", num(r.xgb, 4), r.source)
    for yr in ("2017", "2018"):
        m = D.defender_meta(yr)
        m = m[m.eps == 1.0]
        add(f"valAurocD{YR[yr]}Range", rng(m.val_auroc_D.min(), m.val_auroc_D.max()), m.source.iloc[0])
        add(f"valAurocDplus{YR[yr]}Range", rng(m.val_auroc_Dplus.min(), m.val_auroc_Dplus.max()), m.source.iloc[0])
        add(f"testFprD{YR[yr]}Range", rng(m.test_fpr_D.min(), m.test_fpr_D.max()), m.source.iloc[0])
        add(f"testFprDplus{YR[yr]}Range", rng(m.test_fpr_Dplus.min(), m.test_fpr_Dplus.max()), m.source.iloc[0])
        add(f"nPosTrain{YR[yr]}Range", f"{int(m.n_train_pos.min())}~to~{int(m.n_train_pos.max())}", m.source.iloc[0])


# ---------------------------------------------------------------- near duplicates

def neardup():
    f = S8 / "neardup_exact_2017_2018.csv"
    d = pd.read_csv(f)
    for yr in (2017, 2018):
        for c, nm in (("dos_slowloris", "Slowloris"), ("webattack", "Web"), ("ALL test rows (any class)", "All")):
            x = d[(d.dataset == yr) & (d["class"] == c)].iloc[0]
            add(f"nd{nm}{YR[yr]}", num(x["share_lt_1e-3"]), "results/stage8/neardup_exact_2017_2018.csv")
    x = d[(d.dataset == 2017) & (d["class"] == "dos_slowhttptest")].iloc[0]
    add("ndSlowhttpSeventeen", num(x["share_lt_1e-3"]), "results/stage8/neardup_exact_2017_2018.csv")
    fl = D.sens("flags")
    s5 = fl[(fl.dataset == 2017) & (fl.row_set == "Stage 5 rows")].set_index("class")
    add("ndSrcAllSeventeen", f"{int(s5.loc['ALL', 'n_neardup'])}~of~{intc(s5.loc['ALL', 'n_source_rows'])}",
        "results/stage8/sensitivity_flags.csv")
    for c, nm in (("dos_slowloris", "Slowloris"), ("webattack", "Web")):
        add(f"ndSrc{nm}Seventeen", f"{int(s5.loc[c, 'n_neardup'])}~of~{int(s5.loc[c, 'n_source_rows'])}",
            "results/stage8/sensitivity_flags.csv")
    s8 = fl[(fl.dataset == 2018) & (fl.row_set == "Stage 7d rows")].set_index("class")
    add("ndSrcAllEighteen", f"{int(s8.loc['ALL', 'n_neardup'])}~of~{intc(s8.loc['ALL', 'n_source_rows'])}",
        "results/stage8/sensitivity_flags.csv")
    st = D.sens("stage5")
    for yr in (2017, 2018):
        for v in VN:
            n = st[(st.dataset == yr) & (st.variant == v) & (st.quantity == "n_source_rows")].value.iloc[0]
            add(f"nSrc{VN[v]}{YR[yr]}", intc(n), "results/stage8/sensitivity_stage5.csv")
    ng = D.sens("negatives")
    for yr in (2017, 2018):
        x = ng[(ng.dataset == yr) & (ng.seed.astype(str) == "pooled") & (ng.eps == 1.0)].iloc[0]
        src = "results/stage8/sensitivity_negatives.csv"
        add(f"ndNeg{YR[yr]}", num(x.share_neardup), src)
        add(f"fprDNd{YR[yr]}", num(x.fpr_D_neardup), src)
        add(f"fprDOther{YR[yr]}", num(x.fpr_D_non_neardup), src)
        add(f"fprDplusNd{YR[yr]}", num(x.fpr_Dplus_neardup), src)
        add(f"fprDplusOther{YR[yr]}", num(x.fpr_Dplus_non_neardup), src)
    tp = D.sens("tpr_by_flag")
    x = tp[(tp.dataset == 2017) & (tp.attack == "A6_lam1")].set_index("source_rows")
    src = "results/stage8/sensitivity_tpr_by_flag.csv"
    add("tprDNdSeventeen", num(x.loc["neardup", "tpr_D"]), src)
    add("tprDOtherSeventeen", num(x.loc["non_neardup", "tpr_D"]), src)
    add("tprDplusNdSeventeen", num(x.loc["neardup", "tpr_Dplus"]), src)
    add("tprDplusOtherSeventeen", num(x.loc["non_neardup", "tpr_Dplus"]), src)


# ---------------------------------------------------------------- RQ1 clean data

def clean():
    src = "results/stage8/clean_variants.csv"
    R = pd.read_csv(S8 / "clean_variants.csv")
    nm = {"attn_rollout": "Rollout", "attn_mean": "Mean", "attn_last": "Last", "ig_zero": "IG", "gshap": "GShap",
          "occlusion": "Occl", "treeshap_xgb": "TreeShap", "abs_x": "AbsX", "random": "Random"}
    for yr in (2017, 2018):
        for v in VN:
            g = R[(R.dataset == yr) & (R.variant == v)]
            for a in ("attn_rollout", "attn_mean", "attn_last"):
                for r in ("ig_zero", "gshap", "occlusion", "treeshap_xgb"):
                    x = g[(g.metric == "spearman") & (g.method == a) & (g.ref == r)].value
                    if v == "V1" or (a == "attn_rollout" and r == "ig_zero"):
                        mean_range(f"sp{nm[a]}{nm[r]}{VN[v]}{YR[yr]}", x, src)
            for metric, mn in (("del_auc", "Del"), ("ins_auc", "Ins")):
                for m in ("attn_rollout", "attn_mean", "attn_last", "ig_zero", "gshap", "occlusion", "abs_x", "random"):
                    x = g[(g.metric == metric) & (g.method == m)].value
                    if v == "V1":
                        mean_range(f"auc{mn}{nm[m]}{VN[v]}{YR[yr]}", x, src)
    o = pd.read_csv(S8 / "clean_variants_ordering.csv")
    srco = "results/stage8/clean_variants_ordering.csv"
    for _, r in o.iterrows():
        mn = "Del" if r.metric == "del_auc" else "Ins"
        add(f"ordSp{mn}{VN[r.variant]}", num(r.spearman_orderings), srco)
        if r.variant == "V1":
            for m in ("attn_rollout", "ig_zero", "occlusion", "abs_x", "random", "attn_last"):
                for yr in (2017, 2018):
                    add(f"rank{mn}{nm[m]}{YR[yr]}", str(int(r[f"rank{yr}_{m}"])), srco)
    # Findings that do not replicate (all explanation-set rows; per-row values not stored for these).
    f = pd.read_csv(S7 / "stage7c" / "stage7c_cmp2017_findings.csv").set_index("finding")
    srcf = "results/stage7/stage7c/stage7c_cmp2017_findings.csv"
    k = [i for i in f.index if i.startswith("F6 static minus dynamic attn_rollout, aopc_gap, train_mean")][0]
    add("staticGapSeventeen", num(f.loc[k, "mean_2017"]), srcf)
    add("staticGapEighteen", num(f.loc[k, "mean_2018"]), srcf)
    k = [i for i in f.index if i.startswith("F11")][0]
    add("headMinusRolloutSeventeen", num(f.loc[k, "mean_2017"]), srcf)
    add("headMinusRolloutEighteen", num(f.loc[k, "mean_2018"]), srcf)


# ---------------------------------------------------------------- RQ2 / RQ3 attacks

def attacks():
    st = D.sens("stage5")
    src = "results/stage8/sensitivity_stage5.csv"

    def vals(yr, v, q, **f):
        x = st[(st.dataset == yr) & (st.variant == v) & (st.quantity == q)]
        for a, b in f.items():
            x = x[x[a] == b]
        return x.value

    for yr in (2017, 2018):
        for v in VN:
            mean_range(f"massAtwo{VN[v]}{YR[yr]}", vals(yr, v, "mass_ratio_rollout", mode="constrained", attack="A2", eps=1.0), src)
            for m, mn in (("ig_zero", "IG"), ("attn_rollout", "Rollout")):
                if v != "V1":
                    mean_range(f"flip{mn}{VN[v]}{YR[yr]}",
                               vals(yr, v, "aopc_gap_margin_flipped_minus_A0", eps=1.0, method=m), src, d=2)
            if v == "V1":
                mean_range(f"massAthree{VN[v]}{YR[yr]}",
                           vals(yr, v, "mass_ratio_rollout", mode="constrained", attack="A3_lam1000", eps=1.0), src)
                mean_range(f"massUnc{VN[v]}{YR[yr]}", vals(yr, v, "mass_ratio_rollout", mode="unconstrained", attack="A2", eps=0.5), src)
                mean_range(f"succAone{VN[v]}{YR[yr]}", vals(yr, v, "success", mode="constrained", attack="A1", eps=1.0), src)
                mean_range(f"succAzero{VN[v]}{YR[yr]}", vals(yr, v, "success", mode="constrained", attack="A0", eps=1.0), src)
                # Attention mass under the random control A0 (reference for A2, Stage 9m).
                mean_range(f"massAzero{VN[v]}{YR[yr]}", vals(yr, v, "mass_ratio_rollout", mode="constrained", attack="A0", eps=1.0), src)
                mean_range(f"aoneMinusAzero{VN[v]}{YR[yr]}", vals(yr, v, "A1_minus_A0_success", eps=1.0), src)
                for m, mn in (("ig_zero", "IG"), ("occlusion", "Occl"), ("attn_rollout", "Rollout"),
                              ("attn_mean", "Mean"), ("attn_last", "Last"), ("static_rollout", "StaticRollout"),
                              ("static_ig", "StaticIG"), ("abs_x", "AbsX"), ("random", "Random")):
                    mean_range(f"flip{mn}{VN[v]}{YR[yr]}",
                               vals(yr, v, "aopc_gap_margin_flipped_minus_A0", eps=1.0, method=m), src, d=2)
            if v == "V0":
                # A0 success (any change of prediction) as in Table t_evasion (V0), Stage 9m.
                add(f"succAzero{VN[v]}{YR[yr]}", num(vals(yr, v, "success", mode="constrained", attack="A0", eps=1.0).mean()), src)


# ---------------------------------------------------------------- RQ4 / RQ5 defenders

def defenders():
    pa, un, de, pc = D.sens("paired"), D.sens("union"), D.sens("defenders"), D.sens("per_class")
    de = de[de.seed_only.isna()]
    for yr in (2017, 2018):
        for v in VN:
            src = "results/stage8/sensitivity_paired.csv"
            for atk, an in (("A6_lam1", "AsixOne"), ("A6_lam100", "AsixHundred"), ("A6norm_w0.5", "AsixNorm"),
                            ("A8_lam1", "AeightOne"), ("A8_lam100", "AeightHundred"), ("A8norm_w0.5", "AeightNorm")):
                x = pa[(pa.dataset == yr) & (pa.variant == v) & (pa.attack == atk)].iloc[0]
                if v == "V1" or atk.startswith("A8"):
                    add(f"tpr{an}{VN[v]}{YR[yr]}", num(x.tpr_diff), src)
                    add(f"tpr{an}{VN[v]}{YR[yr]}CI", ci(x.tpr_diff_lo, x.tpr_diff_hi), src)
            src = "results/stage8/sensitivity_union.csv"
            u = un[(un.dataset == yr) & (un.variant == v) & (un.eps == 1.0)]
            p = u[u.seeds == "pooled"].iloc[0]
            add(f"union{VN[v]}{YR[yr]}", num(p.Dplus_minus_D), src)
            add(f"union{VN[v]}{YR[yr]}CI", ci(p.ci_lo, p.ci_hi), src)
            # Magnitude of the pooled difference, for prose that states the direction in words.
            add(f"union{VN[v]}{YR[yr]}Abs", num(abs(p.Dplus_minus_D)), src + " (absolute value of Dplus_minus_D)")
            add(f"unionD{VN[v]}{YR[yr]}", num(p.succ_undet_union_D), src)
            add(f"unionDplus{VN[v]}{YR[yr]}", num(p.succ_undet_union_Dplus), src)
            for _, r in u[u.seeds != "pooled"].iterrows():
                sn = {"seed 0": "SeedZero", "seed 1": "SeedOne", "seed 2": "SeedTwo"}[r.seeds]
                add(f"union{VN[v]}{YR[yr]}{sn}", num(r.Dplus_minus_D), src)
                add(f"union{VN[v]}{YR[yr]}{sn}CI", ci(r.ci_lo, r.ci_hi), src)
            src = "results/stage8/sensitivity_defenders.csv"
            for atk, an, det in (("A6_lam1", "AsixOne", "D"), ("A6_lam1", "AsixOne", "Dplus"),
                                 ("A8_lam1", "AeightOne", "D"), ("A8_lam1", "AeightOne", "Dplus"),
                                 ("A8_lam100", "AeightHundred", "Dplus"), ("A1", "Aone", "D")):
                x = de[(de.dataset == yr) & (de.variant == v) & (de.attack == atk)].iloc[0]
                if v == "V1" or (atk == "A6_lam1" and det == "D"):
                    add(f"undet{det}{an}{VN[v]}{YR[yr]}", num(x[f"undet_{det}"]), src)
                    add(f"undet{det}{an}{VN[v]}{YR[yr]}Range", rng(x[f"undet_{det}_lo"], x[f"undet_{det}_hi"]), src)
    # Per class (V1, eps 1, A6 lambda 1 against D).
    src = "results/stage8/sensitivity_per_class.csv"
    x = pc[(pc.variant == "V1") & (pc.eps == 1.0) & (pc.attack == "A6_lam1")]
    for yr in (2017, 2018):
        g = x[x.dataset == yr].set_index("class")
        top = g.succ_undet_D.idxmax()
        add(f"pcTopClass{YR[yr]}", D.CLASS_NAMES[top], src)
        add(f"pcTop{YR[yr]}", num(g.loc[top, "succ_undet_D"]), src)
        add(f"pcRestMax{YR[yr]}", num(g.drop(index=top).succ_undet_D.max()), src)
    # Stand-in quality (median R^2 over the 12 stand-ins, per seed; lowest single stand-in).
    s17 = pd.read_csv(S8 / "stage8c_standin_quality_2017.csv")
    s18 = pd.read_csv(S7 / "stage7e" / "standin_quality.csv")
    for yr, s, src in ((2017, s17, "results/stage8/stage8c_standin_quality_2017.csv"),
                       (2018, s18, "results/stage7/stage7e/standin_quality.csv")):
        med = s.groupby("seed").r2.median()
        add(f"standinMed{YR[yr]}Range", rng(med.min(), med.max()), src)
        add(f"standinMin{YR[yr]}", num(s.r2.min()), src)
        add(f"standinMinName{YR[yr]}", s.loc[s.r2.idxmin(), "feature"].replace("_", r"\_"), src)


# ---------------------------------------------------------------- RQ3 split by row status (Stage 8d)

def flip_split():
    src = "results/stage8/flip_split.csv"
    d = pd.read_csv(S8 / "flip_split.csv")
    c = pd.read_csv(S8 / "flip_split_counts.csv")
    nm = {"ig_zero": "IG", "occlusion": "Occl", "attn_rollout": "Rollout", "attn_mean": "Mean", "attn_last": "Last",
          "static_rollout": "StaticRollout", "static_ig": "StaticIG", "abs_x": "AbsX", "random": "Random"}
    for yr in (2017, 2018):
        for (atk, sub), pre in ((("A1", "not_flipped"), "notflip"), (("A0", "to_benign"), "azeroflip")):
            x = d[(d.dataset == yr) & (d.variant == "V1") & (d.attack == atk) & (d.subset == sub)]
            for m, mn in nm.items():
                mean_range(f"{pre}{mn}Vone{YR[yr]}", x[x.method == m].aopc_gap_margin_minus_A0, src, d=2)
        for (atk, sub), nmc in ((("A1", "flipped"), "nAoneFlip"), (("A1", "not_flipped"), "nAoneNotFlip"),
                                (("A0", "to_benign"), "nAzeroFlip")):
            n = c[(c.dataset == yr) & (c.variant == "V1") & (c.attack == atk) & (c.subset == sub)].n_rows.sum()
            add(f"{nmc}Vone{YR[yr]}", intc(n), "results/stage8/flip_split_counts.csv")


# ---------------------------------------------------------------- appendix: configuration and training outcomes

def sci(v):
    m, e = f"{v:.0e}".split("e")
    return r"\ensuremath{" + f"{int(float(m))} \\times 10^{{{int(e)}}}" + "}"


def appendix():
    from fractions import Fraction
    import re
    from src.config import DEFAULTS
    from src.stage7e_defense_2018 import STAGE7E
    f, x, pp = DEFAULTS["stage2"]["ftt"], DEFAULTS["stage2"]["xgb"], DEFAULTS["stage2"]["preprocess"]
    cfg = "src/config.py DEFAULTS['stage2']"
    add("apDModel", str(f["d_model"]), cfg)
    add("apBlocks", str(f["n_blocks"]), cfg)
    add("apHeads", str(f["n_heads"]), cfg)
    fr = Fraction(f["ffn_factor"]).limit_denominator(10)
    add("apFfnFactor", rf"\ensuremath{{{fr.numerator}/{fr.denominator}}}", cfg)
    add("apAttnDropout", f"{f['attn_dropout']:g}", cfg)
    add("apFfnDropout", f"{f['ffn_dropout']:g}", cfg)
    add("apResDropout", f"{f['residual_dropout']:g}", cfg)
    add("apBatch", intc(f["batch_size"]), cfg)
    add("apLR", sci(f["lr"]), cfg)
    add("apWeightDecay", sci(f["weight_decay"]), cfg)
    add("apWarmup", str(f["warmup_steps"]), cfg)
    add("apMaxEpochs", str(f["max_epochs"]), cfg)
    add("apPatience", str(f["patience"]), cfg)
    add("apGradClip", f"{f['grad_clip']:g}", cfg)
    add("apSkewThr", f"{pp['log1p_skew_threshold']:g}", cfg)
    add("apClip", f"{pp['clip']:g}", cfg)
    add("apClassWeight", DEFAULTS["stage2"]["class_weight"].replace("_", r"\_"), cfg)
    xc = "src/config.py DEFAULTS['stage2']['xgb']"
    add("apXgbDepth", str(x["max_depth"]), xc)
    add("apXgbEta", f"{x['eta']:g}", xc)
    add("apXgbSubsample", f"{x['subsample']:g}", xc)
    add("apXgbColsample", f"{x['colsample_bytree']:g}", xc)
    add("apXgbMinChild", f"{x['min_child_weight']:g}", xc)
    add("apXgbRounds", intc(x["num_boost_round"]), xc)
    add("apXgbEarly", str(x["early_stopping_rounds"]), xc)
    cur = pd.read_csv(RES / "stage2_ftt_training_curves.csv")
    best17 = [int(g.loc[g.val_macro_f1.idxmax(), "epoch"]) for _, g in cur.groupby("seed")]
    run17 = [int(g.epoch.max()) for _, g in cur.groupby("seed")]
    add("apBestEpochSeventeen", ", ".join(map(str, best17)), "results/stage2_ftt_training_curves.csv")
    add("apEpochsRunSeventeen", ", ".join(map(str, run17)), "results/stage2_ftt_training_curves.csv")
    inf = [json.loads((S7 / f"seed{s}" / "info.json").read_text(encoding="utf-8")) for s in (0, 1, 2)]
    add("apBestEpochEighteen", ", ".join(str(i["best_epoch"]) for i in inf), "results/stage7/seed{0,1,2}/info.json")
    add("apEpochsRunEighteen", ", ".join(str(i["epochs_run"]) for i in inf), "results/stage7/seed{0,1,2}/info.json")
    add("apParamsEighteen", intc(inf[0]["n_params"]), "results/stage7/seed0/info.json")
    log = (RES / "stage2_full_run_console.txt").read_text(encoding="utf-8", errors="replace")
    add("apXgbBestSeventeen", re.search(r"\[xgb\] best iteration (\d+)", log).group(1),
        "results/stage2_full_run_console.txt")
    add("apXgbBestEighteen", str(json.loads((S7 / "xgb_info.json").read_text(encoding="utf-8"))["best_iteration"]),
        "results/stage7/xgb_info.json")
    a4 = "src/stage4_attack.py STAGE4"
    add("apAlpha", f"{STAGE4['alpha0']:g}", a4)
    add("apRestarts", str(STAGE4["restarts"]), a4)
    add("apAtwoTau", f"{STAGE4['a2_tau']:g}", a4)
    add("apAtwoPenalty", f"{STAGE4['a2_penalty']:g}", a4)
    add("apAthreeKappa", f"{STAGE4['a3_kappa']:g}", a4)
    add("apIGSteps", str(STAGE3["ig_steps"]), "src/stage3_explain.py STAGE3['ig_steps']")
    add("apIGBenignK", str(STAGE3["ig_benign_k"]), "src/stage3_explain.py STAGE3['ig_benign_k']")
    add("apGsBackground", str(STAGE3["gs_background"]), "src/stage3_explain.py STAGE3['gs_background']")
    add("apGsSamples", str(STAGE3["gs_n_samples"]), "src/stage3_explain.py STAGE3['gs_n_samples']")
    st = {a: e for m, a, e in STAGE5["settings"] if m == "constrained"}
    add("apEpsList", r"\ensuremath{\{" + ", ".join(f"{e:g}" for e in st["A1"]) + r"\}}", "src/stage5_fullrun.py STAGE5['settings']")
    add("apEpsAthree", r"\ensuremath{\{" + ", ".join(f"{e:g}" for e in st["A3_lam1000"]) + r"\}}", "src/stage5_fullrun.py STAGE5['settings']")
    un = {a: e for m, a, e in STAGE5["settings"] if m == "unconstrained"}
    add("apEpsUnc", r"\ensuremath{\{" + ", ".join(f"{e:g}" for e in un["A2"]) + r"\}}", "src/stage5_fullrun.py STAGE5['settings']")
    s6 = "src/stage6_defense.py STAGE6"
    add("apKnnRef", intc(STAGE6["knn_train_per_class"]), s6)
    add("apMahaRows", intc(STAGE6["maha_train_rows"]), s6)
    add("apAsixSafety", f"{STAGE6C['a6_safety']:g}", "src/stage6c_defense.py STAGE6C['a6_safety']")
    add("apAsixSingleLambda", f"{STAGE6C['a6_single_lambda']:g}", "src/stage6c_defense.py STAGE6C['a6_single_lambda']")
    add("apCplus", f"{STAGE6D['C_plus']:g}", "src/stage6d_final.py STAGE6D['C_plus']")
    add("apTau", f"{STAGE6E['tau']:g}", "src/stage6e_strong.py STAGE6E['tau']")
    add("apChunkAsix", str(STAGE7E["a6_chunk"]), "src/stage7e_defense_2018.py STAGE7E['a6_chunk'] (= STAGE6D['attack_chunk'])")
    add("apChunkAeight", str(STAGE7E["a8_chunk"]), "src/stage7e_defense_2018.py STAGE7E['a8_chunk']")
    est = (RES / "stage6" / "strong" / "runtime_estimate.txt").read_text(encoding="utf-8")
    add("apChunkAeightSeventeen", re.search(r"chunk (\d+)", est).group(1), "results/stage6/strong/runtime_estimate.txt")


# ---------------------------------------------------------------- Stage 8e audit fixes

def audit():
    # 1. flagged source rows on the 7 common classes (2017), and the dos_slowhttptest share of the 10-class count.
    fl = D.sens("flags")
    s5 = fl[(fl.dataset == 2017) & (fl.row_set == "Stage 5 rows")].set_index("class")
    com = s5.loc[[c for c in D.COMMON if c in s5.index]]
    src = "results/stage8/sensitivity_flags.csv"
    add("ndSrcCommonSeventeen", f"{int(com.n_neardup.sum())}~of~{intc(com.n_source_rows.sum())}", src)
    add("ndSrcSlowhttpSeventeen", f"{int(s5.loc['dos_slowhttptest', 'n_neardup'])}~of~"
        f"{int(s5.loc['dos_slowhttptest', 'n_source_rows'])}", src)
    # 2. A1 success as a share of source rows (pooled over rows, V1).
    de = D.sens("defenders")
    de = de[de.seed_only.isna() & (de.variant == "V1") & (de.attack == "A1")]
    for yr in (2017, 2018):
        add(f"succAoneRowsVone{YR[yr]}", num(de[de.dataset == yr].success.iloc[0]), "results/stage8/sensitivity_defenders.csv")
    # 3. relative reduction of the top-10 rollout mass (1 - mean ratio), constrained A2 and unconstrained A2.
    st = D.sens("stage5")
    srcs = "results/stage8/sensitivity_stage5.csv"

    def mean_of(yr, v, mode, atk, eps, q="mass_ratio_rollout"):
        x = st[(st.dataset == yr) & (st.variant == v) & (st.quantity == q) & (st["mode"] == mode) & (st.attack == atk)
               & (st.eps == eps)]
        return float(x.value.mean())
    for yr in (2017, 2018):
        add(f"massRedAtwoVone{YR[yr]}", num(1 - mean_of(yr, "V1", "constrained", "A2", 1.0)), srcs)
        add(f"massRedUncVone{YR[yr]}", num(1 - mean_of(yr, "V1", "unconstrained", "A2", 0.5)), srcs)
    # 5. A0 rows flipped to benign by chance, per seed (V1, eps 1).
    c = pd.read_csv(S8 / "flip_split_counts.csv")
    for yr in (2017, 2018):
        x = c[(c.dataset == yr) & (c.variant == "V1") & (c.attack == "A0") & (c.subset == "to_benign")].sort_values("seed")
        add(f"nAzeroFlipVone{YR[yr]}Seeds", ", ".join(str(int(v)) for v in x.n_rows), "results/stage8/flip_split_counts.csv")
        add(f"nAzeroFlipVone{YR[yr]}Min", str(int(x.n_rows.min())), "results/stage8/flip_split_counts.csv")
    # 6. seed ranges of the non-reproduced findings (all explanation-set rows).
    f = pd.read_csv(S7 / "stage7c" / "stage7c_cmp2017_findings.csv").set_index("finding")
    srcf = "results/stage7/stage7c/stage7c_cmp2017_findings.csv"
    for key, nmx in (("F6 static minus dynamic attn_rollout, aopc_gap, train_mean", "staticGap"), ("F11", "headMinusRollout")):
        k = [i for i in f.index if i.startswith(key)][0]
        for yr in ("2017", "2018"):
            lo, hi = (float(v) for v in str(f.loc[k, f"seed_range_{yr}"]).split(" to "))
            add(f"{nmx}{YR[yr]}Range", rng(lo, hi), srcf)
    # 11. FPR split at eps 0.5 (pooled over seeds).
    ng = D.sens("negatives")
    for yr in (2017, 2018):
        x = ng[(ng.dataset == yr) & (ng.seed.astype(str) == "pooled") & (ng.eps == 0.5)].iloc[0]
        src = "results/stage8/sensitivity_negatives.csv"
        add(f"fprDNd{YR[yr]}Half", num(x.fpr_D_neardup), src)
        add(f"fprDOther{YR[yr]}Half", num(x.fpr_D_non_neardup), src)
        add(f"fprDplusNd{YR[yr]}Half", num(x.fpr_Dplus_neardup), src)
        add(f"fprDplusOther{YR[yr]}Half", num(x.fpr_Dplus_non_neardup), src)
    # 14. largest absolute change from V0 to V1 / V2 (means over seeds), both datasets, eps 1: A2 and A3 (lambda 1000)
    # mass ratio, and the flipped-row margin AOPC gap minus A0 for IG and rollout.
    dm, df_ = [], []
    for yr in (2017, 2018):
        for atk in ("A2", "A3_lam1000"):
            v0 = mean_of(yr, "V0", "constrained", atk, 1.0)
            dm += [abs(mean_of(yr, v, "constrained", atk, 1.0) - v0) for v in ("V1", "V2")]
        for m in ("ig_zero", "attn_rollout"):
            g = st[(st.dataset == yr) & (st.quantity == "aopc_gap_margin_flipped_minus_A0") & (st.eps == 1.0) & (st.method == m)]
            mv = g.groupby("variant").value.mean()
            df_ += [abs(mv[v] - mv["V0"]) for v in ("V1", "V2")]
    add("sensMaxMassChange", num(max(dm)), srcs)
    add("sensMaxFlipChange", num(max(df_), 2), srcs)


def review():
    # Stage 9m: facts added after the check of an external review (handoff/15_review_check.md).
    from src.config import DEFAULTS
    k = STAGE4["a2_k"]
    # 1. Immutable features among the clean top-k rollout and IG features: explanation rows of the 7 common classes,
    #    count per row, macro over classes, mean over seeds. Roles from the constraint specification of each dataset.
    for yr, adir, spec, feats in ((2017, RES / "stage3", RES / "stage4_constraints.csv", RES / "stage1" / "features.json"),
                                  (2018, S7 / "stage7c" / "raw", S7 / "stage7d" / "stage7d_constraints_2018_floors.csv",
                                   S7 / "data" / "features.json")):
        fl = json.loads(feats.read_text(encoding="utf-8"))
        role = pd.read_csv(spec).set_index("feature").role.reindex(fl).to_numpy()
        assert not pd.isna(role).any()
        sel = pd.read_parquet(adir / "selection.parquet")
        keep = sel["class"].isin(D.COMMON).to_numpy()
        src = f"{D.rel(adir)}/attr_seed{{0,1,2}}.npz; {D.rel(adir)}/selection.parquet; {D.rel(spec)}"
        for m, mn in (("attn_rollout", "Rollout"), ("ig_zero", "IG")):
            per_seed = []
            for s in D.SEEDS:
                a = np.abs(np.load(adir / f"attr_seed{s}.npz")[m])[keep]
                imm = (role[np.argsort(-a, axis=1)[:, :k]] == "immutable").sum(1)
                per_seed.append(pd.Series(imm).groupby(sel["class"].to_numpy()[keep]).mean().mean())
            add(f"topImm{mn}{YR[yr]}", num(np.mean(per_seed), 2), src)
    # 2. Share of row observations that A0 flipped to benign at eps 1 (V0 for 2017; 2018: V1 equals V0).
    fc = pd.read_csv(S8 / "flip_split_counts.csv")
    for yr in (2017, 2018):
        x = fc[(fc.dataset == yr) & (fc.variant == "V0") & (fc.attack == "A0") & (fc.subset == "to_benign")]
        add(f"azeroBenignShareVzero{YR[yr]}", num(x.n_rows.sum() / x.n_source_rows.sum()),
            "results/stage8/flip_split_counts.csv")
    # 3. Clean-explanation timing on 2017 (GPU, per row, range over seeds): IG and the attention variants.
    t = pd.read_csv(RES / "stage3_timing.csv")
    t = t[t.seed.astype(str).isin([str(s) for s in D.SEEDS])]
    for m, mn, d in (("ig_zero", "IG", 2), ("attention (all variants)", "Attn", 3)):
        v = t[t.method == m].ms_per_row
        assert len(v) == len(D.SEEDS), m
        add(f"msPerRow{mn}Range", rng(v.min(), v.max(), d), "results/stage3_timing.csv")
    # 4. 2018 training: validation macro-F1 over the last `patience` epochs of every seed (all ran to max_epochs).
    f = DEFAULTS["stage2"]["ftt"]
    vf = []
    for s in D.SEEDS:
        c = pd.read_csv(S7 / f"seed{s}" / "curves.csv")
        assert c.epoch.max() == f["max_epochs"]
        vf += c[c.epoch > c.epoch.max() - f["patience"]].val_macro_f1.tolist()
    add("valFlatEighteenRange", rng(min(vf), max(vf), 4), "results/stage7/seed{0,1,2}/curves.csv")


def randomisation():
    # Stage 9n: parameter randomisation check (appendix subsection), macro over the 7 common classes, mean over seeds.
    from src.stage3_explain import RAND_LEVELS
    R = D.randomisation()
    add("nRandLevels", str(len(RAND_LEVELS)), "src/stage3_explain.py RAND_LEVELS")
    n = int(R.n_min.min())
    assert n == int(R.n_max.max()) == STAGE3["rand_n_per_class"]
    add("randRowsPerClass", str(n), "src/stage3_explain.py STAGE3['rand_n_per_class']; results/stage3_randomisation.csv")
    for yr in (2017, 2018):
        for lv, ln, m, mn in (("cascade_top1", "TopOne", "attn_rollout", "Rollout"),
                              ("cascade_top1", "TopOne", "attn_last", "Last"),
                              ("cascade_top3_tokenizer", "All", "attn_rollout", "Rollout"),
                              ("cascade_top3_tokenizer", "All", "ig_zero", "IGAbs"),
                              ("cascade_top3_tokenizer", "All", "ig_zero_signed", "IGSigned")):
            x = R[(R.dataset == str(yr)) & (R.level == lv) & (R.method == m) & (R.metric == "spearman_vs_original")]
            x = x.iloc[0]
            add(f"rand{mn}{ln}{YR[yr]}", num(x["mean"]), x.source)


def ablation():
    # Stage 9 (Task 20): controls for D+ (results/stage9/ablation/summary_*.csv), V1, eps 1, A6 lambda 1, paired TPR
    # differences pooled over seeds. D+ minus D is the published macro tprAsixOneVone*; it is checked here, not redefined.
    A = D.dplus_ablation()
    P, T = A["paired"], A["detection"]
    src_p, src_t = D.rel(D.ABL / "summary_paired.csv"), D.rel(D.ABL / "summary_detection.csv")
    names = {"a1": "Aone", "a2": "Atwo", "b": "B", "c0": "Czero", "c1": "Cone", "c2": "Ctwo"}
    sel = lambda yr, det, comp: P[(P.year == str(yr)) & (P.rowset == "V1") & (P.eps == 1.0) & (P.attack == "A6_lam1")
                                  & (P.detector == det) & (P.comparison == comp)].iloc[0]
    pub = {n: b for n, b, _ in MACROS}
    shuffle = []
    for yr in (2017, 2018):
        x = sel(yr, "Dplus", "variant_minus_D")
        assert pub[f"tprAsixOneVone{YR[yr]}"] == num(x.tpr_diff), (yr, x.tpr_diff)
        for v, vn in names.items():
            add(f"abl{vn}MinusD{YR[yr]}", num(sel(yr, v, "variant_minus_D").tpr_diff), src_p)
        for v in ("a1", "a2"):
            shuffle.append(abs(sel(yr, v, "variant_minus_D").tpr_diff - sel(yr, v + "s", "variant_minus_D").tpr_diff))
        # Pairing of a row's own attention with its attribution: D+ minus a1, with both shuffle seeds.
        x = sel(yr, "a1", "Dplus_minus_variant")
        add(f"ablPair{YR[yr]}", num(x.tpr_diff), src_p)
        add(f"ablPair{YR[yr]}CI", ci(x.tpr_diff_lo, x.tpr_diff_hi), src_p)
        add(f"ablPairAlt{YR[yr]}", num(sel(yr, "a1s", "Dplus_minus_variant").tpr_diff), src_p)
    add("ablShuffleMaxDiff", num(max(shuffle)), src_p)
    t = T[(T.year == "2017") & (T.rowset == "V1") & (T.eps == 1.0)].set_index("detector")
    for det, dn in (("a1", "Aone"), ("Dplus", "Dplus")):
        add(f"ablFpr{dn}Seventeen", num(t.loc[det, "test_fpr"]), src_t)


def main():
    for fn in (params, datasets, neardup, clean, attacks, defenders, flip_split, appendix, audit, review,
               randomisation, ablation):
        fn()
    lines = [f"% Generated by paper/make_numbers_tex.py on {date.today().isoformat()}. Do not edit; regenerate.",
             "% One macro per number used in the prose; each line above a macro names its source."]
    for n, b, s in MACROS:
        lines.append(f"% source: {s}")
        lines.append(f"\\newcommand{{\\{n}}}{{{b}}}")
    (OUT / "numbers.tex").write_text("\n".join(lines) + "\n", encoding="utf-8")
    (OUT / "numbers_index.json").write_text(json.dumps({n: {"value": b, "source": s} for n, b, s in MACROS}, indent=1),
                                            encoding="utf-8")
    print(f"wrote generated/numbers.tex ({len(MACROS)} macros)")


if __name__ == "__main__":
    main()
