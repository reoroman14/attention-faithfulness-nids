# Writes paper_v3/generated/numbers.tex: every macro of paper/make_numbers_tex.py (run unchanged, imported as a
# module, output redirected) followed by the new Task 42 macros for Stage B, A8x and A6x. Each macro line is preceded
# by a comment naming its source file. Read-only on results/ and paper/.
#
# Also writes paper_v3/generated/numbers_index.json and paper_v3/generated/old_numbers_check.txt (every old macro
# compared with paper/generated/numbers.tex).
#
# Usage (from the project root): .venv\Scripts\python.exe paper_v3\gen\make_numbers_v3.py
import glob
import json
import re
from datetime import date

import numpy as np
import pandas as pd

from common import (A6X, A8X, CONTROLS, CTLN, GEN, OLD, ROOT, SB, SEEDN, SETN, VN, YR, bounds, fmt, fmts, rel)
import make_numbers_tex as mn                       # paper/make_numbers_tex.py, imported, not run as a script

add = mn.add


def D_ABL():
    return mn.D.ABL_VARIANTS


def sv(v, d=3):
    return r"\ensuremath{" + fmts(v, d) + "}"


def uv(v, d=3):
    return r"\ensuremath{" + fmt(v, d) + "}"


def sci(lo, hi, d=3):
    return r"\ensuremath{[" + bounds(lo, d) + ", " + bounds(hi, d) + "]}"


def intc(v):
    return f"{int(v):,}".replace(",", "{,}")


def expo(v, digits=1):
    # 2.4e-04 -> 2.4 x 10^{-4}; 0 -> 0
    if v == 0:
        return r"\ensuremath{0}"
    m, e = f"{v:.{digits}e}".split("e")
    return r"\ensuremath{" + f"{m} \\times 10^{{{int(e)}}}" + "}"


def seeds_line(x, col):
    # "+0.114 / -0.037 / +0.149" in seed order
    x = x[x.seeds != "pooled"].sort_values("seeds")
    return r"\ensuremath{" + " / ".join(fmts(v) for v in x[col]) + "}"


def const(path, name):
    # A numeric constant of a src/ script, read from its source text (the scripts import torch, so they are not
    # imported here). Handles "NAME = v" and "A, B = v1, v2".
    t = (ROOT / path).read_text(encoding="utf-8")
    for line in t.splitlines():
        m = re.match(r"^([A-Z0-9_, ]+?)\s*=\s*([^#]+)", line)
        if not m:
            continue
        names = [n.strip() for n in m.group(1).split(",")]
        if name in names:
            vals = [v.strip() for v in m.group(2).split(",")]
            return float(vals[names.index(name)])
    raise KeyError(f"{name} not in {path}")


def gfmt(v):
    return f"{v:g}"


# ---------------------------------------------------------------- A8x equal-effort union

UNION_A8X = {"headline": "Head", "Stage A": "Standin", "sensitivity": "Sens", "per setting lam1": "LamOne",
             "per setting lam100": "LamHundred", "per setting norm_w0.5": "Norm"}


def union_a8x():
    f = A8X / "summary_union.csv"
    src = rel(f)
    U = pd.read_csv(f)
    for lab, nm in UNION_A8X.items():
        g = U[U.union.str.split(":").str[0].str.replace(r" \(.*\)$", "", regex=True) == lab]
        assert g.union.nunique() == 1, lab
        for (yr, rv), x in g.groupby(["year", "rowset"]):
            p = x[x.seeds == "pooled"].iloc[0]
            k = f"ax{nm}{VN[rv]}{YR[yr]}"
            add(k, sv(p.Dplus_minus_D), src)
            add(k + "CI", sci(p.lo, p.hi), src)
            add(k + "Abs", uv(abs(p.Dplus_minus_D)), src + " (absolute value of Dplus_minus_D)")
            add(k + "Seeds", seeds_line(x, "Dplus_minus_D"), src)
            add(k + "D", uv(p.undet_D), src)
            add(k + "Dplus", uv(p.undet_Dplus), src)
            add(k + "N", intc(p.n_rows / 3), src + " (n_rows / 3 seeds)")
            for _, r in x[x.seeds != "pooled"].iterrows():
                add(k + SEEDN[r.seeds], sv(r.Dplus_minus_D), src)
                if nm == "Head":
                    add(k + SEEDN[r.seeds] + "D", uv(r.undet_D), src)
                    add(k + SEEDN[r.seeds] + "Dplus", uv(r.undet_Dplus), src)
    # Seeds with the opposite sign of the pooled headline, per dataset (V1).
    opp = {}
    for yr in (2017, 2018):
        x = U[U.union.str.startswith("headline: ") & (U.year == yr) & (U.rowset == "V1")]
        p = x[x.seeds == "pooled"].Dplus_minus_D.iloc[0]
        n_opp = int((np.sign(x[x.seeds != "pooled"].Dplus_minus_D) != np.sign(p)).sum())
        add(f"axHeadOppSeeds{YR[yr]}", str(n_opp), src + " (seeds whose sign differs from the pooled V1 headline)")
        opp[yr] = n_opp
    # The same two counts as words, with singular / plural, for the abstract, introduction and conclusion.
    word = {0: "no", 1: "one", 2: "two", 3: "three"}
    noun = lambda n: "seed" if n == 1 else "seeds"
    a, b = opp[2017], opp[2018]
    if a == b:
        phrase = "no seed on either dataset" if a == 0 else f"{word[a]} {noun(a)} on each dataset"
    else:
        phrase = f"{word[a]} {noun(a)} on 2017 and {word[b]} on 2018"
    add("axHeadOppSeedsPhrase", phrase, src + " (axHeadOppSeedsSeventeen and axHeadOppSeedsEighteen in words)")


# ---------------------------------------------------------------- TPR / AUROC on A8x and A8 rows

ATKN = {"A8x_lam1": "AxOne", "A8x_lam100": "AxHundred", "A8x_norm_w0.5": "AxNorm", "A8_lam1": "AeightOne",
        "A8_lam100": "AeightHundred", "A8norm_w0.5": "AeightNorm"}


def tpr_a8x():
    f = A8X / "summary_tpr.csv"
    src = rel(f)
    T = pd.read_csv(f)
    for _, r in T.iterrows():
        k = f"{ATKN[r.attack]}{VN[r.rowset]}{YR[r.year]}"
        add("xtpr" + k, sv(r.tpr_diff), src)
        add("xtpr" + k + "CI", sci(r.tpr_diff_lo, r.tpr_diff_hi), src)
        add("xauroc" + k, sv(r.auroc_diff), src)
        add("xauroc" + k + "CI", sci(r.auroc_diff_lo, r.auroc_diff_hi), src)
        add("xsucc" + k, uv(r.success), src)
        add("xundet" + k, uv(r.undet_Dplus), src)
    # Range of the A8x TPR differences over settings and row sets (2017), for prose.
    for yr in (2017, 2018):
        x = T[(T.year == yr) & T.attack.str.startswith("A8x")]
        add(f"xtprAxMax{YR[yr]}", sv(x.tpr_diff.max()), src + " (largest A8x TPR difference over settings and row sets)")
        v = x[x.rowset == "V1"].tpr_diff
        add(f"xtprAxVoneRange{YR[yr]}", r"\ensuremath{" + fmts(v.min()) + r"}~to~\ensuremath{" + fmts(v.max()) + "}",
            src + " (V1, three settings)")


# ---------------------------------------------------------------- A8x step 2 analyses (summary_extra.csv)

def extra_a8x():
    f = A8X / "summary_extra.csv"
    src = rel(f)
    E = pd.read_csv(f)
    a = E[(E.part == "a") & (E.seeds == "pooled")]
    for _, r in a.iterrows():
        if r.rowset != "V1":
            continue
        k = f"{SETN[r.setting]}{YR[r.year]}"
        add("evadeDAsix" + k, uv(r.A6_undetD_share_of_source), src + " (A6 rows undetected by D, share of source rows)")
        add("evadeDAx" + k, uv(r.A8x_undetD_share_of_source), src + " (A8x rows undetected by D, share of source rows)")
    b = E[E.part == "b"]
    for (yr, rv, c), x in b.groupby(["year", "rowset", "control"]):
        p = x[x.seeds == "pooled"].iloc[0]
        k = f"cmx{CTLN[c]}{VN[rv]}{YR[yr]}"
        add(k, sv(p.control_minus_Dplus), src + " (control minus D+ under A8x)")
        add(k + "CI", sci(p.lo, p.hi), src)
        add(k + "Seeds", seeds_line(x, "control_minus_Dplus"), src)


# ---------------------------------------------------------------- Stage B: controls under their own adaptive attackers

def stage_b():
    f = SB / "summary_union.csv"
    src = rel(f)
    U = pd.read_csv(f)
    for (yr, rv, c), x in U.groupby(["year", "rowset", "control"]):
        p = x[x.seeds == "pooled"].iloc[0]
        k = f"{CTLN[c]}{VN[rv]}{YR[yr]}"
        add("cmd" + k, sv(p.control_minus_D), src + " (control minus D, D under A1 + A6 x 3)")
        add("cmd" + k + "CI", sci(p.cmD_lo, p.cmD_hi), src)
        add("cmd" + k + "Seeds", seeds_line(x, "control_minus_D"), src)
        add("cundet" + k, uv(p.undet_control), src)
        add("cma" + k, sv(p.control_minus_Dplus), src + " (control minus D+ under the stand-in attacker A8)")
        add("cma" + k + "CI", sci(p.cmDp_lo, p.cmDp_hi), src)
    for yr in (2017, 2018):
        f2 = SB / str(yr) / "standin_quality.csv"
        s = pd.read_csv(f2)
        for c in ("a1", "a2"):
            x = s[s.control == c].r2
            add(f"rtwoMed{CTLN[c]}{YR[yr]}", uv(x.median()), rel(f2) + " (median over features and seeds)")
            add(f"rtwoMin{CTLN[c]}{YR[yr]}", uv(x.min()), rel(f2) + " (minimum over features and seeds)")
        x = s[s.control.isin(["c0", "c1", "c2"])].r2
        add(f"rtwoMinExact{YR[yr]}", uv(x.min()), rel(f2) + " (minimum over the features of c0, c1, c2)")
    q = pd.read_csv(ROOT / "results" / "stage7" / "stage7e" / "standin_quality.csv")
    fs = q[q.seed == q.seed.iloc[0]].feature
    add("nRankScores", str(int(fs.str.contains("spearman|jacc").sum())),
        "results/stage7/stage7e/standin_quality.csv (Spearman and top-k Jaccard consistency scores)")
    add("nCosScores", str(int(fs.str.endswith("_cos").sum())), "results/stage7/stage7e/standin_quality.csv (cosine scores)")
    add("nControlsAll", str(len(D_ABL())), "paper/data_v2.py ABL_VARIANTS")
    add("nControlsAttacked", str(U.control.nunique()), src + " (controls with adaptive-attack rows)")
    files = sorted(glob.glob(str(SB / "*" / "seed*" / "rows_B_*.parquet")))
    v = pd.concat([pd.read_parquet(x, columns=["valid"]) for x in files]).valid
    add("sbRows", intc(len(v)), "results/stage9/adaptive/{2017,2018}/seed*/rows_B_*.parquet (row count)")
    add("sbInvalid", intc(int((~v.astype(bool)).sum())), "results/stage9/adaptive/{2017,2018}/seed*/rows_B_*.parquet (valid)")
    add("sbRuns", str(len(files)), "results/stage9/adaptive/{2017,2018}/seed*/rows_B_*.parquet (file count)")


# ---------------------------------------------------------------- A8x runs: rows, time, memory, sanity, amendment

def run_files(base, prefix):
    files = sorted(glob.glob(str(base / "*" / "seed*" / f"rows_{prefix}_*.parquet")))
    R = [pd.read_parquet(x) for x in files]
    return files, R


def a8x_runs():
    files, R = run_files(A8X, "A8x")
    srcr = "results/stage9/a8x/{2017,2018}/seed*/rows_A8x_*.parquet"
    allr = pd.concat(R, ignore_index=True)
    add("axRuns", str(len(files)), srcr + " (file count)")
    add("axRows", intc(len(allr)), srcr + " (row count)")
    add("axInvalid", intc(int((~allr.valid.astype(bool)).sum())), srcr + " (valid)")
    for yr in ("2017", "2018"):
        sec = [r.seconds_total.iloc[0] for f, r in zip(files, R) if f"{yr}" in f.split("a8x")[1][:6]]
        add(f"axMin{YR[yr]}Range", f"{round(min(sec) / 60)}~to~{round(max(sec) / 60)}", srcr + " (seconds_total / 60, per run)")
    m = pd.read_csv(A8X / "memlog.csv")
    add("axSharedMax", uv(m.shared_gb.max()), "results/stage9/a8x/memlog.csv (max shared_gb)")
    add("axReservedMax", r"\ensuremath{" + f"{m.reserved_gb.max():.2f}" + "}", "results/stage9/a8x/memlog.csv (max reserved_gb)")
    # Amendment 1: in-loop count of flipped and undetected rows with and without the 1e-3 margin, and the evaluation.
    mg = const("src/stage9c_a8x.py", "SEL_MARGIN")
    for yr in ("2017", "2018"):
        x = pd.concat([r for f, r in zip(files, R) if f"{yr}" in f.split("a8x")[1][:6]], ignore_index=True)
        f_ = x.flipped.astype(bool)
        add(f"axFlipped{YR[yr]}", intc(f_.sum()), srcr)
        add(f"axInloopNoMargin{YR[yr]}", intc((f_ & (x.score_Dplus_sel < x.thr_Dplus)).sum()), srcr + " (score_Dplus_sel < thr)")
        add(f"axInloopMargin{YR[yr]}", intc((f_ & (x.score_Dplus_sel < x.thr_Dplus - mg)).sum()),
            srcr + " (score_Dplus_sel < thr - SEL_MARGIN)")
        add(f"axEvalUndet{YR[yr]}", intc((f_ & (x.score_Dplus < x.thr_Dplus)).sum()), srcr + " (score_Dplus < thr)")
        add(f"axInloopEvalMax{YR[yr]}", expo(float((x.score_Dplus_sel - x.score_Dplus)[f_].abs().max())),
            srcr + " (max |score_Dplus_sel - score_Dplus| on flipped rows)")
    # Sanity checks (run 2) and the failed run 1, parsed from the logs.
    for log, nm in ((A8X / "log_sanity.txt", ""), (A8X / "log_sanity_failed_run1.txt", "Runone")):
        t = log.read_text(encoding="utf-8")
        for yr in ("2017", "2018"):
            def mx(pat):
                v = [float(m.group(1)) for m in re.finditer(rf"\] {yr} sanity {pat}: ([0-9.e+-]+)", t)]
                return max(v) if v else None
            maha = mx(r"\(c\) mahalanobis_benign max relative \|inside - saved\|")
            mabs = mx(r"\(c, info\) mahalanobis_benign max \|inside - saved\| absolute") if nm == "" else                 mx(r"\(c\) mahalanobis_benign max \|inside - saved\|")
            dec = mx(r"\(c\) D_plus decision max \|inside - saved\|")
            if maha is not None:
                add(f"axSan{nm}MahaRel{YR[yr]}", expo(maha), rel(log))
            if mabs is not None:
                add(f"axSan{nm}MahaAbs{YR[yr]}", expo(mabs), rel(log))
            if dec is not None:
                add(f"axSan{nm}Dec{YR[yr]}", expo(dec), rel(log))
            if nm == "":
                oth = [float(m.group(1)) for m in re.finditer(
                    rf"\] {yr} sanity \([abc]\) (?:stand-in|loss|IG-20|msp|margin|knn5|cons_)[^:]*: ([0-9.e+-]+)", t)]
                add(f"axSanOtherMax{YR[yr]}", expo(max(oth)), rel(log) + " (max over checks (a), (b) and the other (c) checks)")
                add(f"axSanN{YR[yr]}", str(len(re.findall(rf"\] {yr} sanity \([abc]\)[^\n]*PASS", t))), rel(log) + " (PASS lines)")
                assert not re.search(rf"\] {yr} sanity [^\n]*FAIL", t)
    sc = "src/stage9c_a8x.py"
    add("axChunk", gfmt(const(sc, "CHUNK")), sc + " CHUNK")
    add("axIGSteps", gfmt(const(sc, "IG_STEPS")), sc + " IG_STEPS")
    add("axTauStart", gfmt(const(sc, "TAU0")), sc + " TAU0")
    add("axTauEnd", gfmt(const(sc, "TAU1")), sc + " TAU1")
    add("axMargin", expo(mg, 0).replace("1 \\times ", ""), sc + " SEL_MARGIN")
    add("axTol", expo(const(sc, "TOL"), 0).replace("1 \\times ", ""), sc + " TOL")


# ---------------------------------------------------------------- A6x

UNION_A6X = {"headline": "Head", "selection effect on D": "Sel", "sensitivity": "Sens", "per setting lam1": "LamOne",
             "per setting lam100": "LamHundred", "per setting norm_w0.5": "Norm"}


def a6x():
    f = A6X / "summary_union.csv"
    src = rel(f)
    U = pd.read_csv(f)
    for lab, nm in UNION_A6X.items():
        g = U[U.union.str.split(":").str[0].str.replace(r" \(.*\)$", "", regex=True) == lab]
        assert g.union.nunique() == 1, lab
        for (yr, rv), x in g.groupby(["year", "rowset"]):
            p = x[x.seeds == "pooled"].iloc[0]
            k = f"sx{nm}{VN[rv]}{YR[yr]}"
            add(k, sv(p.second_minus_first), src)
            add(k + "CI", sci(p.lo, p.hi), src)
            add(k + "Lo", sv(p.lo), src + " (lower bound)")
            add(k + "Seeds", seeds_line(x, "second_minus_first"), src)
            if nm == "Head":
                add(k + "D", uv(p.undet_first), src)
    sel = U[U.union.str.startswith("selection effect on D:")]
    add("sxSelMaxAbs", uv(sel.second_minus_first.abs().max()), src + " (max |A6x minus A6| over years, row sets, seeds)")
    f = A6X / "rowlevel_A6x_vs_A6.csv"
    r = pd.read_csv(f)
    src = rel(f)
    add("sxRowsAll", intc(r.n.sum()), src)
    add("sxRowsDiffer", intc(r.rows_X_differ.sum()), src)
    add("sxFlipChanged", intc(r.flip_diff.abs().sum()), src)
    add("sxUndetChanged", intc(r.undet_diff.abs().sum()), src)
    add("sxMarginInloop", intc(r.margin_inloop_differ.sum()), src)
    add("sxInloopEval", intc(r.inloop_vs_eval_differ.sum()), src)
    add("sxMaxSelEval", expo(r.max_sel_eval.max()), src)
    files, R = run_files(A6X, "A6x")
    srcr = "results/stage9/a6x/{2017,2018}/seed*/rows_A6x_*.parquet"
    allr = pd.concat(R, ignore_index=True)
    add("sxRuns", str(len(files)), srcr + " (file count)")
    add("sxRowsWritten", intc(len(allr)), srcr + " (row count)")
    add("sxInvalid", intc(int((~allr.valid.astype(bool)).sum())), srcr + " (valid)")
    for yr in ("2017", "2018"):
        sec = [x.seconds_total.iloc[0] for fn, x in zip(files, R) if f"{yr}" in fn.split("a6x")[1][:6]]
        add(f"sxSec{YR[yr]}Range", f"{int(min(sec))}~to~{int(max(sec))}", srcr + " (seconds_total, per run)")
    m = pd.read_csv(A6X / "memlog.csv")
    add("sxSharedMax", uv(m.shared_gb.max()), "results/stage9/a6x/memlog.csv (max shared_gb)")
    add("sxReservedMax", r"\ensuremath{" + f"{m.reserved_gb.max():.2f}" + "}", "results/stage9/a6x/memlog.csv (max reserved_gb)")
    t = (A6X / "log_sanity.txt").read_text(encoding="utf-8")
    assert "FAIL" not in t
    sd = "src/stage9d_a6x.py"
    add("sxChunk", gfmt(const(sd, "CHUNK")), sd + " CHUNK")
    tf = (A6X / "log_sanity_failed_chunk50.txt").read_text(encoding="utf-8")
    m1 = re.search(r"2017 sanity \(s1\) selection off, chunk \d+: max [^:]*: ([0-9.e+-]+)", tf)
    m2 = re.search(r"2017 sanity \(s1, info\) rows with any difference > 1e-5: (\d+)", tf)
    add("sxRunoneMax", expo(float(m1.group(1))), "results/stage9/a6x/log_sanity_failed_chunk50.txt")
    add("sxRunoneRows", m2.group(1), "results/stage9/a6x/log_sanity_failed_chunk50.txt")
    for yr in ("2017", "2018"):
        m3 = re.search(rf"{yr} sanity \(s1\) selection off, chunk \d+: max [^:]*: ([0-9.e+-]+)", t)
        add(f"sxSanSone{YR[yr]}", expo(float(m3.group(1))), "results/stage9/a6x/log_sanity.txt")


# ---------------------------------------------------------------- main

NEW = (union_a8x, tpr_a8x, extra_a8x, stage_b, a8x_runs, a6x)
OLD_FNS = ("params", "datasets", "neardup", "clean", "attacks", "defenders", "flip_split", "appendix", "audit",
           "review", "randomisation", "ablation")


def check_old(n_old):
    # Every old macro: name and body equal to paper/generated/numbers.tex.
    ref = {}
    for m in re.finditer(r"\\newcommand\{\\([A-Za-z]+)\}\{(.*)\}\s*$", (OLD / "generated" / "numbers.tex").read_text(
            encoding="utf-8"), re.M):
        ref[m.group(1)] = m.group(2)
    old = mn.MACROS[:n_old]
    same = [n for n, b, _ in old if ref.get(n) == b]
    diff = [(n, ref.get(n), b) for n, b, _ in old if ref.get(n) != b]
    missing = sorted(set(ref) - {n for n, _, _ in old})
    lines = [f"Old macros regenerated: {len(old)}; in paper/generated/numbers.tex: {len(ref)}",
             f"identical name and body: {len(same)}", f"different: {len(diff)}", f"missing from regeneration: {len(missing)}"]
    lines += [f"DIFF {n}: old {a!r} new {b!r}" for n, a, b in diff] + [f"MISSING {n}" for n in missing]
    (GEN / "old_numbers_check.txt").write_text("\n".join(lines) + "\n", encoding="utf-8")
    print("\n".join(lines[:4]))
    return not diff and not missing


def main():
    GEN.mkdir(parents=True, exist_ok=True)
    for fn in OLD_FNS:
        getattr(mn, fn)()
    n_old = len(mn.MACROS)
    for fn in NEW:
        fn()
    lines = [f"% Generated by paper_v3/gen/make_numbers_v3.py on {date.today().isoformat()}. Do not edit; regenerate.",
             "% Old macros (paper/make_numbers_tex.py) first, then the Task 42 macros."]
    for n, b, s in mn.MACROS:
        lines.append(f"% source: {s}")
        lines.append(f"\\newcommand{{\\{n}}}{{{b}}}")
    (GEN / "numbers.tex").write_text("\n".join(lines) + "\n", encoding="utf-8")
    (GEN / "numbers_index.json").write_text(json.dumps(
        {n: {"value": b, "source": s, "new": i >= n_old} for i, (n, b, s) in enumerate(mn.MACROS)}, indent=1),
        encoding="utf-8")
    ok = check_old(n_old)
    print(f"wrote generated/numbers.tex ({n_old} old + {len(mn.MACROS) - n_old} new macros); old unchanged: {ok}")


if __name__ == "__main__":
    main()
