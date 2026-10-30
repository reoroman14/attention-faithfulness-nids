# Stage 7c: attention faithfulness on LycoS-Unicas-IDS2018. Replicates Stage 3 (src/stage3_explain.py), Stage 3c
# (src/stage3c_diagnostics.py) and the clean-data static class-mean control (src/stage4c_static_control.py) on the
# Stage 7b artifacts with NO methodological change and NO tuning. No attacks, no defenders.
#
# Only the data, checkpoints, preprocess.json and class list change. Every explainer, metric, fraction, baseline,
# sample cap, sampling seed and rng offset is imported from (or copied verbatim out of) the 2017 modules.
# The single forced adaptation: 2018 has no day column, so the Stage 3 / 3c selection runs with one day group
# (every row in day "all"), i.e. a plain random draw of up to n_per_class rows per class with the same seed.
#
# Additional reporting for 2018:
#   1. dos_slowloris near duplicates: the EXACT (direct difference, no matrix-product formula) distance of every
#      dos_slowloris test row to its nearest train row in standardised space. Every class_rows-based table (all
#      faithfulness and agreement tables, stability, randomisation, partial agreement, heads, static control) gets
#      an extra class "dos_slowloris_excl_neardup": the slowloris rows whose nearest-train distance is >= 1e-3.
#      The plain "dos_slowloris" row is on all selected rows. n is reported for both. Macros are unchanged.
#   2. n per class and per seed is the "n" column of every table; classes with n < 100 are marked in "flag".
#   3. 2017 vs 2018 side-by-side tables on the classes common to both, from the saved 2017 CSVs.
#   4. runtime per seed and total, GPU memory peak.
# Thin copies (with the near-duplicate split added): class_rows (wrapper), agreement_rows, partial_rows.
#
# Outputs: results/stage7/stage7c/ (raw float16 tensors in raw/, per-seed resume files in parts/, tables
# stage7c_*.csv, a progress entry stage7c_progress_entry.md). Nothing outside that folder is written except the
# run dir under results/runs/ and the run_log.csv line written by start_run.
# Resumable per seed: a seed whose parts/seed{s}.pkl exists is loaded instead of recomputed.
#
# Usage:
#   .venv\Scripts\python.exe -m src.stage7c_explain_2018
#   smoke test: add --set stage3.n_per_class=10 --set stage3.rand_n_per_class=5 --set stage3c.val_n_per_class=10
#               --set stage4c.train_rows_per_class=50 --set stage7c.min_rows=5 --set stage7c.out_dir=<scratch dir>
import copy
import json
import pickle
import sys
import time
from datetime import datetime
from itertools import combinations

import numpy as np
import pandas as pd
import torch
import xgboost as xgb

from src import config as cfgmod
from src.config import load_config
from src.preprocess import Preprocessor
from src.stage2_train import predict_torch
from src.stage3_explain import (ATTN, BENIGN, HEAD_NOTE, RAND_LEVELS, SIGNED, STAGE3, all_attributions,
                                attention_variants, centred_ranks, chance_jaccard, curve_metrics,
                                importance_scores, integrated_gradients, jaccard, load_ftt, perturbation_curves,
                                randomised, ranking, robustness_rows, rowwise_spearman, select_rows,
                                spearman_from_ranks, target_out, timed)
from src.stage3_explain import class_rows as class_rows3
from src.stage3c_diagnostics import GRAD, REFS, STAGE3C, deletion_probs, grad_attention, macro_attack
from src.stage4_attack import STAGE4
from src.stage4c_static_control import DYNAMIC_OF, STAGE4C, STATIC, static_vectors
from src.stage5_fullrun import md
from src.utils import RESULTS, ROOT, finish_run, resolve, start_run

STAGE7C = {
    "out_dir": "results/stage7/stage7c",
    "neardup_threshold": 1e-3,
    # Stop condition per class: fewer selected rows than this -> class excluded, the others continue.
    "min_rows": 30,
    "max_hours": 2.5,
    "nn_query_batch": 16,
    "nn_train_chunk": 65536,
}
D7 = ROOT / "results" / "stage7"
SLOW, SLOW_X = "dos_slowloris", "dos_slowloris_excl_neardup"
COMMON = ["bot", "ddos", "dos_goldeneye", "dos_hulk", "dos_slowloris", "ssh_patator", "webattack"]
SMALL_N = 100
H = "macro_attack_classes"
M7 = "macro_common7"


class StopRun(Exception):
    pass


# ---------------------------------------------------------------- thin copies with the near-duplicate split

def class_rows(vals, y, classes, normalised=False, nd=None, **key):
    # Stage 3 class_rows, plus (if nd is given) an extra dos_slowloris row restricted to rows with nd False, and a
    # "n < 100" mark in the flag column. Macro rows are exactly the Stage 3 ones (the extra row is not in them).
    out = class_rows3(vals, y, classes, normalised=normalised, **key)
    if nd is not None and SLOW in classes:
        keep = ~np.asarray(nd, bool)
        extra = [r for r in class_rows3(vals[keep], y[keep], classes, normalised=normalised, **key)
                 if r["class"] == SLOW][0]
        extra["class"] = SLOW_X
        out.insert(next(i for i, r in enumerate(out) if r["class"] == SLOW) + 1, extra)
    for r in out:
        if r["class"].startswith("macro_") or (normalised and r["class"] == BENIGN):
            continue
        if r["n"] < SMALL_N:
            r["flag"] = (r["flag"] + "; " if r["flag"] else "") + f"n < {SMALL_N}"
    return out


def agreement_rows(scores, y, classes, rng, topk, nd=None, **key):
    # Copy of stage3_explain.agreement_rows (same rng use), routed through the class_rows wrapper.
    F = next(iter(scores.values())).shape[1]
    orders = {k: ranking(v, rng) for k, v in scores.items()}
    ranks = {k: centred_ranks(v) for k, v in scores.items()}
    rows = []
    for a, b in combinations(scores, 2):
        rows += class_rows(spearman_from_ranks(ranks[a], ranks[b]), y, classes, nd=nd, **key, method_a=a,
                           method_b=b, metric="spearman", chance=0.0)
        for k in topk:
            rows += class_rows(jaccard(orders[a], orders[b], k), y, classes, nd=nd, **key, method_a=a, method_b=b,
                               metric=f"jaccard@{k}", chance=chance_jaccard(F, k))
    return rows


def partial_rows(scores, z, y, classes, seed, pairs, nd=None):
    # Copy of stage3c_diagnostics.partial_rows, routed through the class_rows wrapper.
    rk = {k: centred_ranks(v) for k, v in scores.items()}
    rz = centred_ranks(z)
    rows = []
    for a, b in pairs:
        r_ab = spearman_from_ranks(rk[a], rk[b])
        rows += class_rows(r_ab, y, classes, nd=nd, seed=seed, method_a=a, method_b=b, metric="spearman")
        if a == "abs_x" or b == "abs_x":
            continue
        r_az, r_bz = spearman_from_ranks(rk[a], rz), spearman_from_ranks(rk[b], rz)
        den = np.sqrt((1 - r_az ** 2) * (1 - r_bz ** 2))
        part = np.divide(r_ab - r_az * r_bz, den, out=np.full(len(r_ab), np.nan), where=den > 1e-12)
        rows += class_rows(part, y, classes, nd=nd, seed=seed, method_a=a, method_b=b,
                           metric="partial_spearman_given_abs_x")
    return rows


# ---------------------------------------------------------------- exact nearest-train distance

@torch.no_grad()
def exact_nn(Q, T, dev, qb, tb):
    # Nearest train row by direct squared differences (sum over features of (q - t)^2, never the
    # ||q||^2 + ||t||^2 - 2 q.t expansion), then the distance to that row recomputed in float64 on the CPU.
    Tt = torch.from_numpy(T).to(dev)
    idx = np.empty(len(Q), np.int64)
    for i in range(0, len(Q), qb):
        q = torch.from_numpy(Q[i:i + qb]).to(dev)
        best = torch.full((len(q),), float("inf"), device=dev)
        arg = torch.zeros(len(q), dtype=torch.long, device=dev)
        for j in range(0, len(Tt), tb):
            d = ((q[:, None, :] - Tt[None, j:j + tb, :]) ** 2).sum(-1)
            m, a = d.min(1)
            upd = m < best
            best = torch.where(upd, m, best)
            arg = torch.where(upd, a + j, arg)
        idx[i:i + qb] = arg.cpu().numpy()
    del Tt
    torch.cuda.empty_cache()
    dist = np.sqrt(((Q.astype(np.float64) - T[idx].astype(np.float64)) ** 2).sum(1))
    return idx, dist


# ---------------------------------------------------------------- 2017 vs 2018 helpers

def seed_summary(df, keys, cls_list):
    # Per class: mean over seeds, per-seed min / max, n range. Plus the unweighted macro over the 7 common classes,
    # computed per seed (only when all 7 class means exist), then mean / min / max over seeds.
    d = df[df["class"].isin(cls_list)].copy()
    d["seed"] = d["seed"].astype(str)
    pc = d.groupby(keys + ["class"], sort=False).agg(mean=("mean", "mean"), lo=("mean", "min"), hi=("mean", "max"),
                                                      n_lo=("n", "min"), n_hi=("n", "max")).reset_index()
    m = d[d["class"].isin(COMMON)].groupby(keys + ["seed"])["mean"].agg(["mean", "count"]).reset_index()
    m = m[m["count"] == len(COMMON)]
    mac = m.groupby(keys)["mean"].agg(mean="mean", lo="min", hi="max").reset_index()
    mac["class"], mac["n_lo"], mac["n_hi"] = M7, len(COMMON), len(COMMON)
    return pd.concat([pc, mac], ignore_index=True)


def side_by_side(d17, d18, keys):
    cls_order = COMMON + [SLOW_X, M7]
    a = seed_summary(d17, keys, COMMON)
    b = seed_summary(d18, keys, COMMON + [SLOW_X])
    m = a.merge(b, on=keys + ["class"], how="outer", suffixes=("_2017", "_2018"))
    out = m[keys + ["class"]].copy()
    for yr in ("2017", "2018"):
        out[f"mean_{yr}"] = m[f"mean_{yr}"]
        out[f"seed_range_{yr}"] = [("" if pd.isna(lo) else f"{lo:.3f} to {hi:.3f}")
                                   for lo, hi in zip(m[f"lo_{yr}"], m[f"hi_{yr}"])]
        out[f"n_{yr}"] = [("" if pd.isna(lo) else (f"{int(lo)}" if lo == hi else f"{int(lo)} to {int(hi)}"))
                          for lo, hi in zip(m[f"n_lo_{yr}"], m[f"n_hi_{yr}"])]
    out["flag_2018"] = [(f"n < {SMALL_N}" if (c != M7 and not pd.isna(n) and n < SMALL_N) else "")
                        for c, n in zip(m["class"], m["n_lo_2018"])]
    out["diff_2018_minus_2017"] = out["mean_2018"] - out["mean_2017"]
    out["class"] = pd.Categorical(out["class"], cls_order, ordered=True)
    return out.sort_values(keys + ["class"]).reset_index(drop=True).assign(**{"class": lambda x: x["class"].astype(str)})


def m7(df, **f):
    # Per-seed macro over the 7 common classes for one filtered slice; Series indexed by seed (str).
    d = df
    for k, v in f.items():
        d = d[d[k] == v]
    d = d[d["class"].isin(COMMON)].assign(seed=lambda x: x["seed"].astype(str))
    g = d.groupby("seed")["mean"].agg(["mean", "count"])
    return g.loc[g["count"] == len(COMMON), "mean"]


def static_minus_dynamic(st, dy):
    st = st.copy()
    st["method"] = st["method"].map(DYNAMIC_OF)
    m = st.merge(dy, on=["seed", "mask_baseline", "method", "metric", "class"], suffixes=("_st", "_dy"))
    return m.assign(mean=m["mean_st"] - m["mean_dy"], n=m["n_st"])[
        ["seed", "mask_baseline", "method", "metric", "class", "mean", "n"]]


def findings(D):
    # Key 2017 clean-data quantities as per-seed macros over the 7 common classes (labels state the reading).
    fa, ag, ra, sb, pa, f3c, st = (D[k] for k in ("faith", "agree", "rand", "stab", "part", "f3c", "static"))
    f = lambda m, met, b="train_mean": m7(fa, method=m, metric=met, mask_baseline=b)
    a = lambda x, y_, met="spearman": m7(ag, method_a=x, method_b=y_, metric=met)
    r = lambda m, lv: m7(ra, method=m, level=lv, metric="spearman_vs_original")
    s_ = lambda sm, met, b="train_mean": m7(st, method=sm, metric=met, mask_baseline=b)
    out = [
        ("F1 attn_last AOPC gap, train mean (2017: about 0, like random)", f("attn_last", "aopc_gap")),
        ("F1 random AOPC gap, train mean (floor, about 0)", f("random", "aopc_gap")),
        ("F1 attn_last minus random, del AUC, train mean", f("attn_last", "del_auc") - f("random", "del_auc")),
        ("F1 attn_last vs ig_zero, Spearman", a("attn_last", "ig_zero")),
        ("F2 attn_rollout vs ig_zero, Spearman", a("attn_rollout", "ig_zero")),
        ("F2 attn_rollout vs gshap, Spearman", a("attn_rollout", "gshap")),
        ("F2 attn_rollout vs occlusion, Spearman", a("attn_rollout", "occlusion")),
        ("F2 attn_rollout vs treeshap_xgb, Spearman", a("attn_rollout", "treeshap_xgb")),
        ("F2 attn_mean vs ig_zero, Spearman", a("attn_mean", "ig_zero")),
        ("F2 attn_rollout vs ig_zero, Jaccard@5", a("attn_rollout", "ig_zero", "jaccard@5")),
        ("F3 random minus attn_rollout, del AUC, train mean (> 0: rollout beats random)",
         f("random", "del_auc") - f("attn_rollout", "del_auc")),
        ("F3 attn_rollout minus ig_zero, del AUC, train mean (> 0: IG more faithful)",
         f("attn_rollout", "del_auc") - f("ig_zero", "del_auc")),
        ("F4 attn_rollout minus abs_x, del AUC, train mean (> 0: |x| more faithful)",
         f("attn_rollout", "del_auc") - f("abs_x", "del_auc")),
        ("F4 attn_rollout minus abs_x, del AUC, benign sample (> 0: |x| more faithful)",
         f("attn_rollout", "del_auc", "benign_sample") - f("abs_x", "del_auc", "benign_sample")),
        ("F4 abs_x minus attn_rollout, AOPC gap, train mean (> 0: |x| more faithful)",
         f("abs_x", "aopc_gap") - f("attn_rollout", "aopc_gap")),
        ("F5 occlusion minus ig_zero, del AUC, train mean", f("occlusion", "del_auc") - f("ig_zero", "del_auc")),
        ("F5 gshap minus ig_zero, del AUC, train mean", f("gshap", "del_auc") - f("ig_zero", "del_auc")),
    ]
    for sm, dm in DYNAMIC_OF.items():
        for met, b in (("aopc_gap", "train_mean"), ("aopc_gap", "benign_sample"), ("del_auc", "train_mean"),
                       ("comp@5", "train_mean")):
            out.append((f"F6 static minus dynamic {dm}, {met}, {b}", s_(sm, met, b) - f(dm, met, b)))
    out += [
        ("F7 randomisation cascade_top1: attn_rollout Spearman vs original", r("attn_rollout", "cascade_top1")),
        ("F7 randomisation cascade_top1: attn_mean", r("attn_mean", "cascade_top1")),
        ("F7 randomisation cascade_top1: attn_last", r("attn_last", "cascade_top1")),
        ("F7 randomisation cascade_top3_tokenizer: attn_rollout", r("attn_rollout", "cascade_top3_tokenizer")),
        ("F7 randomisation cascade_top3_tokenizer: ig_zero (abs)", r("ig_zero", "cascade_top3_tokenizer")),
        ("F7 randomisation cascade_top3_tokenizer: ig_zero_signed", r("ig_zero_signed", "cascade_top3_tokenizer")),
        ("F7 randomisation indep_block1: attn_rollout", r("attn_rollout", "indep_block1")),
        ("F8 attn_rollout vs ig_zero: raw minus partial Spearman given |x|",
         m7(pa, method_a="attn_rollout", method_b="ig_zero", metric="spearman")
         - m7(pa, method_a="attn_rollout", method_b="ig_zero", metric="partial_spearman_given_abs_x")),
        ("F9 cross-seed Spearman attn_last", m7(sb, method="attn_last", metric="spearman")),
        ("F9 cross-seed Spearman attn_rollout", m7(sb, method="attn_rollout", metric="spearman")),
        ("F9 cross-seed Spearman ig_zero", m7(sb, method="ig_zero", metric="spearman")),
        ("F10 gradattn_rollout minus attn_rollout, del AUC, train mean",
         m7(f3c, method="gradattn_rollout", metric="del_auc", mask_baseline="train_mean") - f("attn_rollout", "del_auc")),
        ("F11 head_val_selected minus attn_rollout, del AUC, train mean",
         m7(f3c, method="head_val_selected", metric="del_auc", mask_baseline="train_mean") - f("attn_rollout", "del_auc")),
    ]
    return out


# ---------------------------------------------------------------- main

def main():
    cfgmod.DEFAULTS.setdefault("stage3", copy.deepcopy(STAGE3))
    cfgmod.DEFAULTS.setdefault("stage3c", copy.deepcopy(STAGE3C))
    cfgmod.DEFAULTS.setdefault("stage4c", {**copy.deepcopy(STAGE4), **copy.deepcopy(STAGE4C)})
    cfgmod.DEFAULTS.setdefault("stage7c", copy.deepcopy(STAGE7C))
    cfg, overrides = load_config("Stage 7c: Stage 3 / 3c / static control replicated on LycoS-Unicas-IDS2018.")
    c, cc, c4, c7 = (cfg[k] for k in ("stage3", "stage3c", "stage4c", "stage7c"))
    out = resolve(c7["out_dir"])
    raw, parts = out / "raw", out / "parts"
    for p in (out, raw, parts):
        p.mkdir(parents=True, exist_ok=True)
    t_start = time.time()
    run_dir, log = start_run("stage7c_explain_2018", cfg, overrides)
    dev = torch.device("cuda")
    seeds, topk, n_per_class, rand_n = c["seeds"], c["topk"], c["n_per_class"], c["rand_n_per_class"]
    G_sanity, G_timing = [], []

    def make_sane(store):
        def sane(name, seed, value, expected, passed):
            store.append({"check": name, "seed": seed, "value": value, "expected": expected, "pass": bool(passed)})
            log.info(f"[sanity] seed={seed} {name}: {value} (expected {expected}) -> {'PASS' if passed else 'FAIL'}")
        return sane

    gsane = make_sane(G_sanity)

    def write_stop(msg):
        log.error(f"STOPPED: {msg}")
        (out / "stage7c_STOPPED.md").write_text(
            f"## {datetime.now():%Y-%m-%d %H:%M}: Stage 7c stopped\n\nStop condition: {msg}\n", encoding="utf-8")

    try:
        # ---- data (read-only)
        feats = json.loads((D7 / "data" / "features.json").read_text(encoding="utf-8"))
        label_map = json.loads((D7 / "data" / "label_map.json").read_text(encoding="utf-8"))
        classes = [k for k, _ in sorted(label_map.items(), key=lambda kv: kv[1])]
        F, C = len(feats), len(classes)
        attack = [k for k in classes if k != BENIGN]
        pp = Preprocessor.load(D7 / "preprocess.json")
        assert pp.features == feats
        test = pd.read_parquet(D7 / "data" / "test.parquet", columns=feats + ["label_id"])
        Xte = pp.transform(test)
        y_te = test["label_id"].to_numpy().astype(np.int64)
        d_te = np.full(len(y_te), "all", dtype=object)          # 2018 has no day column: one day group
        del test

        # ---- selection: every trained seed must be correct
        all_seeds = cfg["stage2"]["ftt"]["seeds"]
        logits_full, preds_full = {}, {}
        Xte_t = torch.from_numpy(Xte).to(dev)
        for s in all_seeds:
            m, ck = load_ftt(D7 / f"seed{s}" / "ftt.pt", dev)
            assert ck["features"] == feats and ck["classes"] == classes
            logits_full[s] = predict_torch(m, Xte_t, ck["config"]["stage2"]["ftt"]["eval_batch_size"]).cpu().numpy()
            preds_full[s] = logits_full[s].argmax(1)
            del m
        del Xte_t
        torch.cuda.empty_cache()
        sel, rank_in_class, counts = select_rows(y_te, d_te, preds_full, classes, n_per_class, c["min_per_class"],
                                                 c["sel_seed"])
        counts["selected_full_run"] = counts["selected"].to_numpy()
        small = counts.loc[counts["selected"] < c7["min_rows"], "class"].tolist()
        if small:
            log.error(f"STOP CONDITION (class level): fewer than {c7['min_rows']} selected rows for {small}; "
                      f"these classes are excluded, the others continue")
            keep_sel = ~np.isin(y_te[sel], [classes.index(k) for k in small])
            sel, rank_in_class = sel[keep_sel], rank_in_class[keep_sel]
        counts["excluded_below_min_rows"] = counts["class"].isin(small)
        N = len(sel)
        y = y_te[sel]
        Xs_np = Xte[sel]
        Xs = torch.from_numpy(Xs_np).to(dev)
        target = torch.from_numpy(y).to(dev)
        absx = np.abs(Xs_np)

        # ---- exact nearest-train distance for every dos_slowloris test row
        train = pd.read_parquet(D7 / "data" / "train.parquet", columns=feats + ["label_id"])
        ytr = train["label_id"].to_numpy()
        Xtr = pp.transform(train)
        del train
        slow_rows = np.flatnonzero(y_te == classes.index(SLOW))
        nnf = raw / "slowloris_nn.npz"
        if nnf.exists() and np.array_equal(np.load(nnf)["test_row"], slow_rows):
            z = np.load(nnf)
            nn_idx, nn_d = z["nn_train_row"], z["nn_dist_exact"]
        else:
            with timed(G_timing, "exact nearest-train distance, dos_slowloris test rows", "pre", len(slow_rows)):
                nn_idx, nn_d = exact_nn(Xte[slow_rows], Xtr, dev, c7["nn_query_batch"], c7["nn_train_chunk"])
            np.savez(nnf, test_row=slow_rows, nn_train_row=nn_idx, nn_dist_exact=nn_d)
        thr = c7["neardup_threshold"]
        nd_test = np.zeros(len(y_te), bool)
        nd_test[slow_rows] = nn_d < thr
        nd_main = nd_test[sel]
        share_all = float((nn_d < thr).mean())
        gsane(f"dos_slowloris test rows with exact nearest-train distance < {thr:g} (share of all test rows)",
              "pre", f"{share_all:.4f} ({int((nn_d < thr).sum())} of {len(nn_d)})",
              "informative; Stage 7b reported 0.2250, but it measured the exact distance to the cdist argmin, which "
              "is often not the nearest row at tiny distances (checked against a float64 CPU brute force)", True)
        sel_slow = y == classes.index(SLOW)
        nd_summary = pd.DataFrame([
            {"set": "all dos_slowloris test rows", "n": len(nn_d), "n_neardup": int((nn_d < thr).sum()),
             "share_neardup": share_all, "n_exact_0": int((nn_d == 0).sum()), "median_dist": float(np.median(nn_d)),
             "p5_dist": float(np.percentile(nn_d, 5))},
            {"set": "selected dos_slowloris rows (explanation set)", "n": int(sel_slow.sum()),
             "n_neardup": int(nd_main.sum()), "share_neardup": float(nd_main[sel_slow].mean()) if sel_slow.any() else np.nan,
             "n_exact_0": int((nn_d[np.isin(slow_rows, sel[sel_slow])] == 0).sum()),
             "median_dist": float(np.median(nn_d[np.isin(slow_rows, sel[sel_slow])])) if sel_slow.any() else np.nan,
             "p5_dist": float(np.percentile(nn_d[np.isin(slow_rows, sel[sel_slow])], 5)) if sel_slow.any() else np.nan}])
        nd_summary["kept_after_exclusion"] = nd_summary["n"] - nd_summary["n_neardup"]
        log.info(f"Near duplicates:\n{nd_summary.to_string(index=False)}")
        counts["flag"] = np.where(counts["selected"] < SMALL_N, f"n < {SMALL_N}", "")
        log.info(f"Selected {N} rows:\n{counts.to_string(index=False)}")

        # ---- XGBoost predictions and TreeSHAP (cross-model reference)
        bst = xgb.Booster()
        bst.load_model(D7 / "xgb.json")
        bst.set_param({"device": "cuda"})

        def treeshap_for(X_np, yy):
            contrib = bst.predict(xgb.DMatrix(X_np), pred_contribs=True)
            return contrib[np.arange(len(yy)), yy, :F]

        xgb_pred = bst.predict(xgb.DMatrix(Xs_np)).argmax(1)
        counts["xgb_correct_selected"] = [int(((y == ci) & (xgb_pred == y)).sum()) for ci in range(C)]
        with timed(G_timing, "treeshap_xgb", "xgb", N):
            treeshap = treeshap_for(Xs_np, y)
        np.savez_compressed(raw / "treeshap_xgb.npz", treeshap=treeshap.astype(np.float16))
        # Stage 3c reads the float16-stored TreeSHAP; reproduced here.
        treeshap16 = np.abs(treeshap.astype(np.float16).astype(np.float32))
        sel_df = pd.DataFrame({"test_row": sel, "label_id": y, "class": [classes[i] for i in y], "day": d_te[sel],
                               "rank_in_class": rank_in_class, "neardup_lt_1e-3": nd_main,
                               **{f"pred_seed{s}": preds_full[s][sel] for s in all_seeds}, "xgb_pred": xgb_pred})
        sel_df.to_parquet(raw / "selection.parquet", index=False)
        np.save(raw / "selection_logits.npy", np.stack([logits_full[s][sel] for s in all_seeds], 1))
        del logits_full

        # ---- baselines from the train split (Stage 3 rng logic)
        rng = np.random.default_rng(c["sel_seed"] + 1)
        ben = np.flatnonzero(ytr == classes.index(BENIGN))
        mask_rows = rng.choice(ben, size=N, replace=False)
        ig_rows = rng.choice(ben, size=c["ig_benign_k"], replace=False)
        per, extra = divmod(c["gs_background"], C)
        bg_rows = np.concatenate([rng.choice(np.flatnonzero(ytr == ci), size=per + (extra if ci == 0 else 0),
                                             replace=False) for ci in range(C)])
        np.savez(raw / "baseline_rows.npz", mask_benign_train_rows=mask_rows, ig_benign_train_rows=ig_rows,
                 gshap_background_train_rows=bg_rows)
        to_t = lambda idx: torch.from_numpy(Xtr[idx]).to(dev)       # transform is row-wise: same as Stage 3
        mask_bases = {"train_mean": torch.zeros_like(Xs), "benign_sample": to_t(mask_rows)}
        ig_ben_base, gs_bg = to_t(ig_rows), to_t(bg_rows)
        # Static control train sample (stage4c: rng_seed + 500, up to train_rows_per_class per class).
        rng4 = np.random.default_rng(c4["rng_seed"] + 500)
        per4 = c4["train_rows_per_class"]
        tr_idx = {ci: rng4.choice(np.flatnonzero(ytr == ci), size=min(per4, int((ytr == ci).sum())), replace=False)
                  for ci in range(C)}
        Ztr = {ci: to_t(ix) for ci, ix in tr_idx.items()}
        n_static_rows = int(sum(len(v) for v in tr_idx.values()))
        del Xtr

        # ---- VAL explanation set (Stage 3c procedure)
        val = pd.read_parquet(D7 / "data" / "val.parquet", columns=feats + ["label_id"])
        Xva = pp.transform(val)
        y_va = val["label_id"].to_numpy().astype(np.int64)
        d_va = np.full(len(y_va), "all", dtype=object)
        del val
        preds_va = {}
        Xva_t = torch.from_numpy(Xva).to(dev)
        for s in seeds:
            m, ck = load_ftt(D7 / f"seed{s}" / "ftt.pt", dev)
            preds_va[s] = predict_torch(m, Xva_t, ck["config"]["stage2"]["ftt"]["eval_batch_size"]).argmax(1).cpu().numpy()
            del m
        del Xva_t
        torch.cuda.empty_cache()
        vsel, _, vcounts = select_rows(y_va, d_va, preds_va, classes, cc["val_n_per_class"], c["min_per_class"],
                                       cc["val_sel_seed"])
        vcounts["flag"] = np.where(vcounts["selected"] < SMALL_N, f"n < {SMALL_N}", "")
        pd.DataFrame({"val_row": vsel, "label_id": y_va[vsel], "day": d_va[vsel]}).to_parquet(
            raw / "val_selection.parquet", index=False)
        yv = y_va[vsel]
        Xv, tv = torch.from_numpy(Xva[vsel]).to(dev), torch.from_numpy(yv).to(dev)
        del Xva
        log.info(f"VAL explanation set: {len(vsel)} rows\n"
                 f"{vcounts[['class', 'correct_all_seeds', 'selected']].to_string(index=False)}")
        rand_sub = np.flatnonzero(rank_in_class < rand_n)
        pre_s = time.time() - t_start
        log.info(f"Pre-phase done in {pre_s:.0f} s")

        # ---------------------------------------------------------------- one seed (Stage 3, 3c, static)
        def run_seed(s):
            R = {k: [] for k in ("faith", "agree", "robust", "rand", "part", "faith3c", "gagree", "hagree", "val",
                                 "faith_static", "sanity", "timing")}
            sane = make_sane(R["sanity"])
            timing = R["timing"]
            t0 = time.time()
            torch.cuda.reset_peak_memory_stats()
            model, _ = load_ftt(D7 / f"seed{s}" / "ftt.pt", dev)

            def ig_sanity(diag, tag=""):
                for name in ("ig_zero", "ig_benign"):
                    r, d = diag[f"{name}_rel_delta"], diag[f"{name}_abs_delta"]
                    sane(f"{tag}{name} completeness |delta| / |f(x)-f(b)| (median, p95, max)", s,
                         f"{np.median(r):.2e}, {np.quantile(r, 0.95):.2e}, {r.max():.2e}; max |delta| {d.max():.2e}",
                         "reported as is (50 steps)", True)

            # ======== Stage 3
            lg, pr = target_out(model, Xs, target, c["eval_rows"])
            with torch.no_grad():
                pred = torch.cat([model(Xs[i:i + c["eval_rows"]]).argmax(1) for i in range(0, N, c["eval_rows"])])
            sane("selected rows predicted as label", s, int((pred == target).sum()), N, bool((pred == target).all()))
            attr, diag, max_err = all_attributions(model, Xs, target, s, c, ig_ben_base, gs_bg, c["sel_seed"] + s,
                                                   timing)
            sane("attention max |row sum - 1|", s, f"{max_err:.2e}", "< 1e-5", max_err < 1e-5)
            ig_sanity(diag)
            sane("gshap mean |convergence delta| (50-sample estimate, informative only)", s,
                 f"{diag['gshap_abs_delta'].mean():.3e}", "informative", True)
            np.savez_compressed(raw / f"attr_seed{s}.npz", **{k: v.astype(np.float16) for k, v in attr.items()},
                                **diag, target_logit=lg.cpu().numpy(), target_prob=pr.cpu().numpy())
            rng_s = np.random.default_rng(c["sel_seed"] + 100 + s)
            sc = importance_scores(attr, Xs_np, rng_s)
            rank_scores = dict(sc)
            rank_scores.update({f"{k}_signed": attr[k] for k in SIGNED})
            orders = {k: ranking(v, rng_s) for k, v in rank_scores.items()}
            curves_store = {}
            for bname, base in mask_bases.items():
                with timed(timing, f"curves x{len(orders)} rankings ({bname})", s, N):
                    for mname, order in orders.items():
                        cur = perturbation_curves(model, Xs, base, torch.from_numpy(order).to(dev), target, c)
                        end_err = max(np.abs(cur["del_morf"][:, 0] - cur["ins_morf"][:, -1]).max(),
                                      np.abs(cur["del_morf"][:, -1] - cur["ins_morf"][:, 0]).max(),
                                      np.abs(cur["del_morf"][:, 0] - pr.cpu().numpy()).max())
                        if end_err > 1e-5:
                            sane(f"curve endpoints {mname}/{bname}", s, f"{end_err:.2e}", "< 1e-5", False)
                        for k, v in cur.items():
                            curves_store[f"{bname}/{mname}/{k}"] = v.astype(np.float16)
                        for metric, vals in curve_metrics(cur, topk, c["aopc_steps"], c["norm_min_gap"]).items():
                            R["faith"] += class_rows(vals, y, classes, normalised="_norm" in metric, nd=nd_main,
                                                     seed=s, mask_baseline=bname, method=mname, metric=metric)
            np.savez_compressed(raw / f"curves_seed{s}.npz", **curves_store)
            del curves_store
            ag = dict(sc)
            ag["treeshap_xgb"] = np.abs(treeshap)
            rows = agreement_rows(ag, y, classes, rng_s, topk, nd=nd_main, seed=s)
            R["agree"] += rows
            R["robust"] += [{**r, "subset": "all_seeds_correct"} for r in rows]
            # ---- robustness set (rows THIS seed classifies correctly)
            rng_r = np.random.default_rng(c["sel_seed"] + 400 + s)
            keep_pos, extra_rows, rc = robustness_rows(y_te, preds_full, s, sel, rank_in_class, classes,
                                                       n_per_class, rng_r)
            if small:
                extra_rows = extra_rows[~np.isin(y_te[extra_rows], [classes.index(k) for k in small])]
            R["robust_counts"] = rc
            ye = y_te[extra_rows]
            Xe_np = Xte[extra_rows]
            sub_scores = {k: v[keep_pos] for k, v in ag.items() if k != "random"}
            if len(extra_rows):
                Xe, te = torch.from_numpy(Xe_np).to(dev), torch.from_numpy(ye).to(dev)
                with torch.no_grad():
                    pe = torch.cat([model(Xe[i:i + c["eval_rows"]]).argmax(1)
                                    for i in range(0, len(ye), c["eval_rows"])])
                sane("robustness extra rows predicted as label", s, int((pe == te).sum()), len(ye),
                     bool((pe == te).all()))
                ea, ediag, _ = all_attributions(model, Xe, te, s, c, ig_ben_base, gs_bg, c["sel_seed"] + 50 + s,
                                                timing, tag="robustness extra: ")
                ig_sanity(ediag, tag="robustness extra: ")
                es = importance_scores(ea, Xe_np, rng_r, treeshap_for(Xe_np, ye))
                np.savez_compressed(raw / f"robustness_extra_seed{s}.npz", test_row=extra_rows,
                                    **{k: v.astype(np.float16) for k, v in ea.items()}, **ediag)
                sub_scores = {k: np.concatenate([v, es[k]]) for k, v in sub_scores.items()}
            y_r = np.concatenate([y[keep_pos], ye])
            nd_r = np.concatenate([nd_main[keep_pos], nd_test[extra_rows]])
            sub_scores["random"] = rng_r.random((len(y_r), F))
            np.savez(raw / f"robustness_rows_seed{s}.npz", main_positions=keep_pos, extra_test_rows=extra_rows)
            R["robust"] += [{**r, "subset": "seed_correct"} for r in
                            agreement_rows(sub_scores, y_r, classes, rng_r, topk, nd=nd_r, seed=s)]
            # ---- parameter randomisation sanity check (attention and IG)
            sub = rand_sub
            Xr, tr = Xs[sub], target[sub]
            with timed(timing, f"randomisation ({len(RAND_LEVELS)} levels)", s, len(sub)):
                rstore = {}
                for li, (level, parts_) in enumerate(RAND_LEVELS):
                    seed_l = c["sel_seed"] + 200 + s + (0 if level.startswith("cascade") else 10 * (li + 1))
                    rm = randomised(model, parts_, seed_l)
                    ra, _ = attention_variants(rm, Xr, c["attn_batch"])
                    rig, _, _ = integrated_gradients(rm, Xr, tr, torch.zeros(1, F, device=dev), c)
                    pairs = [(k, ra[k], attr[k][sub]) for k in ATTN] + \
                            [("ig_zero", np.abs(rig), np.abs(attr["ig_zero"][sub])),
                             ("ig_zero_signed", rig, attr["ig_zero"][sub])]
                    for mname, new, orig in pairs:
                        note = HEAD_NOTE if level == "head_only" and mname in ATTN else ""
                        R["rand"] += class_rows(rowwise_spearman(orig, new), y[sub], classes, nd=nd_main[sub], seed=s,
                                                level=level, randomised=";".join(parts_), method=mname,
                                                metric="spearman_vs_original", note=note)
                        for k in topk:
                            R["rand"] += class_rows(jaccard(ranking(orig, rng_s), ranking(new, rng_s), k), y[sub],
                                                    classes, nd=nd_main[sub], seed=s, level=level,
                                                    randomised=";".join(parts_), method=mname,
                                                    metric=f"jaccard@{k}_vs_original", note=note)
                    rstore.update({f"{level}/{k}": ra[k].astype(np.float16) for k in ATTN})
                    rstore[f"{level}/ig_zero"] = rig.astype(np.float16)
                    del rm
                np.savez_compressed(raw / f"randomisation_seed{s}.npz", rows=sub, **rstore)
            t3 = time.time() - t0
            pk3 = torch.cuda.max_memory_allocated() / 2**30

            # ======== Stage 3c (reference attributions from the float16 store, as in 2017)
            t1 = time.time()
            st = dict(np.load(raw / f"attr_seed{s}.npz"))
            _, pr2 = target_out(model, Xs, target, c["eval_rows"])
            d = float(np.abs(pr2.cpu().numpy() - st["target_prob"]).max())
            sane("target prob matches Stage 3", s, f"{d:.1e}", "< 1e-5", d < 1e-5)
            att, _ = attention_variants(model, Xs, c["attn_batch"])
            d = max(float(np.abs(att[k] - st[k].astype(np.float32)).max()) for k in ATTN + ["attn_cls_all"])
            sane("fp32 attention vs stored float16 (max abs diff)", s, f"{d:.1e}", "< 1e-3 (float16 rounding)",
                 d < 1e-3)
            rng = np.random.default_rng(c["sel_seed"] + 600 + s)
            scores = {k: att[k] for k in ATTN}
            scores.update({k: np.abs(st[k].astype(np.float32)) for k in REFS})
            with timed(timing, "gradattn (both variants)", s, N):
                ga, zero_rows = grad_attention(model, Xs, target, cc["grad_attn_batch"])
            sane("gradattn: (row, layer) pairs with an all-zero CLS row in A_bar", s, zero_rows, "small", True)
            np.savez_compressed(raw / f"gradattn_seed{s}.npz", **{k: v.astype(np.float16) for k, v in ga.items()})
            scores.update(ga)
            scores["abs_x"] = absx
            scores["random"] = rng.random((N, F))
            pairs = [(a, r) for a in ATTN + GRAD + ["abs_x", "random"] for r in REFS] + \
                    [(a, b) for i, a in enumerate(REFS) for b in REFS[i + 1:]]
            R["part"] += partial_rows(scores, absx, y, classes, s, pairs, nd=nd_main)
            with timed(timing, "gradattn curves (2 variants x 2 baselines)", s, N):
                for k in GRAD:
                    order = torch.from_numpy(ranking(ga[k], rng)).to(dev)
                    for bname, base in mask_bases.items():
                        cur = perturbation_curves(model, Xs, base, order, target, c)
                        for metric, vals in curve_metrics(cur, topk, c["aopc_steps"], c["norm_min_gap"]).items():
                            R["faith3c"] += class_rows(vals, y, classes, normalised="_norm" in metric, nd=nd_main,
                                                       seed=s, mask_baseline=bname, method=k, metric=metric)
            ag3 = {k: scores[k] for k in GRAD + ATTN + REFS + ["abs_x", "random"]}
            ag3["treeshap_xgb"] = treeshap16
            R["gagree"] += [r for r in agreement_rows(ag3, y, classes, rng, topk, nd=nd_main, seed=s)
                            if r["method_a"] in GRAD]
            n_layers, n_heads = att["attn_cls_all"].shape[1:3]
            heads = [(l_, h) for l_ in range(n_layers) for h in range(n_heads)]
            hname = lambda lh: f"head_L{lh[0] + 1}H{lh[1] + 1}"
            with timed(timing, f"head curves ({len(heads)} heads, train mean)", s, N):
                for lh in heads:
                    hs = att["attn_cls_all"][:, lh[0], lh[1], 1:]
                    cur = perturbation_curves(model, Xs, mask_bases["train_mean"],
                                              torch.from_numpy(ranking(hs, rng)).to(dev), target, c)
                    for metric, vals in curve_metrics(cur, topk, c["aopc_steps"], c["norm_min_gap"]).items():
                        R["faith3c"] += class_rows(vals, y, classes, normalised="_norm" in metric, nd=nd_main, seed=s,
                                                   mask_baseline="train_mean", method=hname(lh), metric=metric)
                    for ref in ("ig_zero", "occlusion"):
                        R["hagree"] += class_rows(rowwise_spearman(hs, scores[ref]), y, classes, nd=nd_main, seed=s,
                                                  method_a=hname(lh), method_b=ref, metric="spearman")
            with timed(timing, f"val: attention + ig_zero + {len(heads)}-head deletion", s, len(yv)):
                vatt, _ = attention_variants(model, Xv, c["attn_batch"])
                vig, _, vrel = integrated_gradients(model, Xv, tv, torch.zeros(1, F, device=dev), c)
                for lh in heads:
                    hs = vatt["attn_cls_all"][:, lh[0], lh[1], 1:]
                    dp = deletion_probs(model, Xv, torch.zeros_like(Xv), torch.from_numpy(ranking(hs, rng)).to(dev),
                                        tv, c["eval_rows"])
                    R["val"].append({"seed": s, "head": hname(lh), "layer": lh[0] + 1, "head_idx": lh[1] + 1,
                                     "val_del_auc": macro_attack(np.trapezoid(dp, dx=1 / F, axis=1), yv, classes),
                                     "val_spearman_ig_zero": macro_attack(rowwise_spearman(hs, np.abs(vig)), yv,
                                                                          classes)})
            sane("val ig_zero completeness |delta|/|f(x)-f(b)| p95", s, f"{np.quantile(vrel, 0.95):.2e}",
                 "reported as is (50 steps)", True)
            t3c = time.time() - t1

            # ======== static class-mean control (stage4c)
            t2 = time.time()
            with timed(timing, "static vectors (train class means)", s, n_static_rows):
                sv = static_vectors(model, Ztr, None, c4, dev)
            np.savez(raw / f"static_vectors_seed{s}.npz", classes=np.array(classes),
                     train_rows=repr({ci: ix for ci, ix in tr_idx.items()}), **sv)
            rng_st = np.random.default_rng(c4["rng_seed"] + 700 + s)
            with timed(timing, f"static curves ({len(STATIC)} rankings x 2 baselines)", s, N):
                for mname in STATIC:
                    order = torch.from_numpy(ranking(sv[mname][y], rng_st)).to(dev)
                    for bname, base in mask_bases.items():
                        cur = perturbation_curves(model, Xs, base, order, target, c4)
                        for metric, vals in curve_metrics(cur, c4["stage3_topk"], c4["aopc_steps"],
                                                          c4["norm_min_gap"]).items():
                            R["faith_static"] += class_rows(vals, y, classes, normalised="_norm" in metric,
                                                            nd=nd_main, seed=s, mask_baseline=bname, method=mname,
                                                            metric=metric)
            tst = time.time() - t2
            del model
            torch.cuda.empty_cache()
            peak = max([pk3, torch.cuda.max_memory_allocated() / 2**30] + [r["peak_gpu_mem_gb"] for r in timing])
            R["runtime"] = {"seed": s, "stage3_s": round(t3, 1), "stage3c_s": round(t3c, 1), "static_s": round(tst, 1),
                            "seed_total_s": round(time.time() - t0, 1), "peak_gpu_mem_gb": round(peak, 3)}
            np.savez(parts / f"scores_seed{s}.npz", **{f"s3/{k}": v for k, v in sc.items()},
                     **{f"ga/{k}": v for k, v in ga.items()})
            return R

        results = {}
        for si, s in enumerate(seeds):
            pf = parts / f"seed{s}.pkl"
            if pf.exists():
                with open(pf, "rb") as fh:
                    R = pickle.load(fh)
                log.info(f"seed {s}: loaded from {pf.relative_to(ROOT) if pf.is_relative_to(ROOT) else pf} (resume)")
                R["runtime"]["resumed"] = True
            else:
                R = run_seed(s)
                with open(pf, "wb") as fh:
                    pickle.dump(R, fh)
                log.info(f"seed {s} done: {R['runtime']}")
            results[s] = R
            fails = [r for r in R["sanity"] if not r["pass"]]
            if fails:
                raise StopRun(f"seed {s}: sanity check(s) failed: " + "; ".join(f"{r['check']} = {r['value']}"
                                                                                for r in fails))
            if si == 0:
                est_h = (pre_s + len(seeds) * R["runtime"]["seed_total_s"] + 120) / 3600
                log.info(f"RUNTIME ESTIMATE from seed {s}: {est_h:.2f} h for {len(seeds)} seeds")
                gsane("runtime estimate after the first seed (h)", "all", f"{est_h:.2f}",
                      f"<= {c7['max_hours']}", est_h <= c7["max_hours"])
                if est_h > c7["max_hours"]:
                    raise StopRun(f"runtime estimate {est_h:.2f} h > {c7['max_hours']} h")

        # ---------------------------------------------------------------- aggregation
        t_agg = time.time()
        cat = lambda k: [r for s in seeds for r in results[s][k]]
        faith_df, agree_df, rand_df = pd.DataFrame(cat("faith")), pd.DataFrame(cat("agree")), pd.DataFrame(cat("rand"))
        robust_df = pd.DataFrame(cat("robust"))
        robust_counts = pd.concat([results[s]["robust_counts"] for s in seeds], ignore_index=True)
        sanity = list(G_sanity) + cat("sanity")
        timing = list(G_timing) + cat("timing")
        S = {s: dict(np.load(parts / f"scores_seed{s}.npz")) for s in seeds}

        # ---- Stage 3 cross-seed stability (abs_x is identical across seeds and left out)
        stab = []
        for sa, sb in combinations(seeds, 2):
            rng_p = np.random.default_rng(c["sel_seed"] + 300)
            for mname in [k[3:] for k in S[sa] if k.startswith("s3/")]:
                if mname == "abs_x":
                    continue
                a, b = S[sa][f"s3/{mname}"], S[sb][f"s3/{mname}"]
                stab += class_rows(rowwise_spearman(a, b), y, classes, nd=nd_main, method=mname, seed_a=sa, seed_b=sb,
                                   metric="spearman")
                for k in topk:
                    stab += class_rows(jaccard(ranking(a, rng_p), ranking(b, rng_p), k), y, classes, nd=nd_main,
                                       method=mname, seed_a=sa, seed_b=sb, metric=f"jaccard@{k}")
        stab_df = pd.DataFrame(stab)

        # ---- Stage 3 random-baseline sanity, plus random-floor and randomisation checks
        gs2 = make_sane(sanity)
        macro = agree_df[agree_df["class"] == "macro_all_classes"]
        for s in seeds:
            r = macro[(macro.seed == s) & ((macro.method_a == "random") | (macro.method_b == "random"))]
            sp = r[r.metric == "spearman"]["mean"]
            gs2("random vs every method: macro Spearman (max |.|)", s, f"{sp.abs().max():.3f}", "about 0",
                sp.abs().max() < 0.05)
            j5 = r[r.metric == "jaccard@5"]["mean"]
            gs2("random vs every method: macro Jaccard@5 (min, max)", s, f"{j5.min():.3f}, {j5.max():.3f}",
                f"about {chance_jaccard(F, 5):.3f}", (j5 - chance_jaccard(F, 5)).abs().max() < 0.03)
            g = faith_df[(faith_df.seed == s) & (faith_df.method == "random") & (faith_df.metric == "aopc_gap")
                         & (faith_df["class"] == "macro_all_classes")]
            gs2("random ranking: macro AOPC MoRF-LeRF gap per baseline", s,
                "; ".join(f"{b} {v:+.4f}" for b, v in zip(g.mask_baseline, g["mean"])), "about 0",
                g["mean"].abs().max() < 0.02)
            fm = faith_df[(faith_df.seed == s) & (faith_df["class"] == H)]
            for b in ("train_mean", "benign_sample"):
                v = fm[fm.mask_baseline == b].pivot_table(index="method", columns="metric", values="mean")
                ok = all(v.loc[m_, "del_auc"] < v.loc["random", "del_auc"]
                         and v.loc[m_, "aopc_gap"] > v.loc["random", "aopc_gap"] for m_ in ("ig_zero", "occlusion", "gshap"))
                gs2(f"random floor ({b}): ig_zero / occlusion / gshap beat random on macro del AUC and AOPC gap", s,
                    "; ".join(f"{m_} {v.loc[m_, 'del_auc']:.3f}/{v.loc[m_, 'aopc_gap']:.3f}"
                              for m_ in ("ig_zero", "occlusion", "gshap", "random")),
                    "all below random del AUC and above random AOPC gap", ok)
            rr = rand_df[(rand_df.seed == s) & (rand_df["class"] == H) & (rand_df.metric == "spearman_vs_original")]
            ho = rr[(rr.level == "head_only") & rr.method.isin(ATTN)]["mean"]
            gs2("randomisation head_only: attention unchanged (trivial), macro Spearman min", s, f"{ho.min():.4f}",
                ">= 0.999", ho.min() >= 0.999)
            full = rr[rr.level == "cascade_top3_tokenizer"].set_index("method")["mean"]
            gs2("randomisation cascade_top3_tokenizer: explanation changed, macro Spearman per method", s,
                "; ".join(f"{k} {v:+.3f}" for k, v in full.items()),
                "every method < 0.7 (2017: max 0.36, |IG|)", (full < 0.7).all())

        # ---- Stage 3c aggregation
        part_df = pd.DataFrame(cat("part"))
        f3c_df = pd.DataFrame(cat("faith3c"))
        gagree_df, hagree_df, val_df = pd.DataFrame(cat("gagree")), pd.DataFrame(cat("hagree")), pd.DataFrame(cat("val"))
        gstab = []
        for i, sa in enumerate(seeds):
            for sb in seeds[i + 1:]:
                for k in GRAD:
                    gstab += class_rows(rowwise_spearman(S[sa][f"ga/{k}"], S[sb][f"ga/{k}"]), y, classes, nd=nd_main,
                                        method=k, seed_a=sa, seed_b=sb, metric="spearman")
        hm = f3c_df[(f3c_df["class"] == H) & f3c_df.method.str.startswith("head_")]
        ha = hagree_df[hagree_df["class"] == H]
        choice = []
        for s in seeds:
            v = val_df[val_df.seed == s]
            test_del = hm[(hm.seed == s) & (hm.metric == "del_auc")].set_index("method")["mean"]
            for crit, best in (("val_del_auc (min)", v.loc[v.val_del_auc.idxmin()]),
                               ("val_spearman_ig_zero (max)", v.loc[v.val_spearman_ig_zero.idxmax()])):
                hn = best["head"]
                g = lambda metric: float(hm[(hm.seed == s) & (hm.method == hn) & (hm.metric == metric)]["mean"].iloc[0])
                choice.append({"seed": s, "criterion": crit, "chosen_head": hn, "layer": int(best["layer"]),
                               "val_del_auc": best["val_del_auc"], "val_spearman_ig_zero": best["val_spearman_ig_zero"],
                               "test_del_auc": g("del_auc"), "test_aopc_gap": g("aopc_gap"), "test_ins_auc": g("ins_auc"),
                               "test_spearman_ig_zero": float(ha[(ha.seed == s) & (ha.method_a == hn) & (ha.method_b == "ig_zero")]["mean"].iloc[0]),
                               "test_spearman_occlusion": float(ha[(ha.seed == s) & (ha.method_a == hn) & (ha.method_b == "occlusion")]["mean"].iloc[0]),
                               "test_rank_by_del_auc_of_12": int((test_del < g("del_auc")).sum() + 1),
                               "test_oracle_head": test_del.idxmin(), "test_oracle_del_auc": float(test_del.min())})
        choice_df = pd.DataFrame(choice)
        for crit, g in choice_df.groupby("criterion"):
            same_head, same_layer = g.chosen_head.nunique() == 1, g.layer.nunique() == 1
            log.info(f"Head choice by {crit}: {dict(zip(g.seed, g.chosen_head))}; same head: {same_head}; "
                     f"same layer: {same_layer}")
            choice_df.loc[g.index, "same_head_all_seeds"] = same_head
            choice_df.loc[g.index, "same_layer_all_seeds"] = same_layer
        sel_rows = []
        for _, r in choice_df[choice_df.criterion == "val_del_auc (min)"].iterrows():
            sel_rows.append(f3c_df[(f3c_df.seed == r.seed) & (f3c_df.method == r.chosen_head)].assign(method="head_val_selected"))
        f3c_df = pd.concat([f3c_df] + sel_rows, ignore_index=True)
        # floor / ceiling (random = 0, occlusion = 1), attack-class macro
        allf = pd.concat([faith_df, f3c_df], ignore_index=True)
        allf = allf[(allf["class"] == H) & allf.metric.isin(["del_auc", "aopc_gap"])]
        fc = []
        for (s, b), g in allf.groupby(["seed", "mask_baseline"]):
            piv = g.pivot_table(index="method", columns="metric", values="mean")
            for mth, row in piv.iterrows():
                rec = {"seed": s, "mask_baseline": b, "method": mth}
                for metric in ("del_auc", "aopc_gap"):
                    lo, hi = piv.loc["random", metric], piv.loc["occlusion", metric]
                    rec[f"{metric}_raw"] = row[metric]
                    rec[f"{metric}_scaled"] = (row[metric] - lo) / (hi - lo)
                fc.append(rec)
        fc_df = pd.DataFrame(fc)
        fc_mean = fc_df.groupby(["mask_baseline", "method"], as_index=False)[
            ["del_auc_raw", "del_auc_scaled", "aopc_gap_raw", "aopc_gap_scaled"]].mean().assign(seed="mean")
        fc_df = pd.concat([fc_df.assign(seed=fc_df.seed.astype(str)), fc_mean], ignore_index=True)
        # per-class table (attack classes + slowloris without near duplicates); n and flag appended
        meths = ["attn_rollout", "ig_zero", "occlusion", "abs_x", "random"]
        pcls = attack + [SLOW_X, H]
        pc = faith_df[faith_df.method.isin(meths) & faith_df.metric.isin(["del_auc", "aopc_gap"])
                      & faith_df["class"].isin(pcls)][["seed", "mask_baseline", "method", "metric", "class", "mean", "n"]]
        sp = agree_df[(agree_df.metric == "spearman") & (((agree_df.method_a == "ig_zero") & agree_df.method_b.isin(meths))
                                                         | ((agree_df.method_b == "ig_zero") & agree_df.method_a.isin(meths)))
                      & agree_df["class"].isin(pcls)].copy()
        sp["method"] = np.where(sp.method_a == "ig_zero", sp.method_b, sp.method_a)
        sp = sp.assign(mask_baseline="n/a", metric="spearman_with_ig_zero")[pc.columns]
        pc_df = pd.concat([pc, sp], ignore_index=True)
        pc_mean = pc_df.groupby(["mask_baseline", "method", "metric", "class"], as_index=False)[["mean", "n"]].mean().assign(seed="mean")
        pc_df = pd.concat([pc_df.assign(seed=pc_df.seed.astype(str)), pc_mean], ignore_index=True)
        pc_df["flag"] = np.where((pc_df["class"] != H) & (pc_df["n"] < SMALL_N), f"n < {SMALL_N}", "")

        # ---- static control vs dynamic (stage4c layout, attack macro)
        st_df = pd.DataFrame(cat("faith_static"))
        keep_m = ["del_auc", "ins_auc", "aopc_gap", "comp@5", "suff@20", "del_auc_margin", "ins_auc_margin"]
        stm = st_df[(st_df["class"] == H) & st_df.metric.isin(keep_m)].groupby(["mask_baseline", "method", "metric"])["mean"].mean()
        dym = faith_df[(faith_df["class"] == H) & faith_df.metric.isin(keep_m)].groupby(["mask_baseline", "method", "metric"])["mean"].mean()
        cmp_rows = []
        for b in ("train_mean", "benign_sample"):
            for sm in STATIC:
                dm = DYNAMIC_OF[sm]
                for m_ in keep_m:
                    vs, vd = stm.get((b, sm, m_), np.nan), dym.get((b, dm, m_), np.nan)
                    cmp_rows.append({"mask_baseline": b, "method": dm, "metric": m_, "dynamic": vd, "static": vs,
                                     "static_minus_dynamic": vs - vd})
        st_cmp = pd.DataFrame(cmp_rows)

        # ---- dos_slowloris: all selected rows vs near duplicates excluded (key metrics, per seed)
        sl_rows = []
        kf = faith_df[faith_df["class"].isin([SLOW, SLOW_X]) & faith_df.metric.isin(["del_auc", "ins_auc", "aopc_gap"])
                      & faith_df.method.isin(["attn_rollout", "attn_mean", "attn_last", "ig_zero", "gshap", "occlusion",
                                              "abs_x", "random"])]
        for (s, b, mth, met), g in kf.groupby(["seed", "mask_baseline", "method", "metric"]):
            gg = g.set_index("class")
            sl_rows.append({"table": "faithfulness", "seed": s, "mask_baseline": b, "method": mth, "metric": met,
                            "all_rows": gg.loc[SLOW, "mean"], "n_all": gg.loc[SLOW, "n"],
                            "excl_neardup": gg.loc[SLOW_X, "mean"], "n_excl": gg.loc[SLOW_X, "n"]})
        ka = agree_df[agree_df["class"].isin([SLOW, SLOW_X]) & agree_df.method_a.isin(ATTN)
                      & agree_df.method_b.isin(["ig_zero", "gshap", "occlusion", "treeshap_xgb", "abs_x"])
                      & agree_df.metric.isin(["spearman", "jaccard@5"])]
        for (s, ma, mb, met), g in ka.groupby(["seed", "method_a", "method_b", "metric"]):
            gg = g.set_index("class")
            sl_rows.append({"table": "agreement", "seed": s, "mask_baseline": "n/a", "method": f"{ma} vs {mb}",
                            "metric": met, "all_rows": gg.loc[SLOW, "mean"], "n_all": gg.loc[SLOW, "n"],
                            "excl_neardup": gg.loc[SLOW_X, "mean"], "n_excl": gg.loc[SLOW_X, "n"]})
        sl_df = pd.DataFrame(sl_rows)
        sl_df["excl_minus_all"] = sl_df["excl_neardup"] - sl_df["all_rows"]

        # ---- 2017 vs 2018 (classes common to both)
        D18 = {"faith": faith_df, "agree": agree_df, "rand": rand_df,
               "stab": stab_df.assign(seed=stab_df.seed_a.astype(str) + "-" + stab_df.seed_b.astype(str)),
               "part": part_df, "f3c": f3c_df, "static": st_df}
        s17 = pd.read_csv(RESULTS / "stage3_stability.csv")
        D17 = {"faith": pd.read_csv(RESULTS / "stage3_faithfulness.csv"),
               "agree": pd.read_csv(RESULTS / "stage3_agreement.csv"),
               "rand": pd.read_csv(RESULTS / "stage3_randomisation.csv"),
               "stab": s17.assign(seed=s17.seed_a.astype(str) + "-" + s17.seed_b.astype(str)),
               "part": pd.read_csv(RESULTS / "stage3c_partial_agreement.csv"),
               "f3c": pd.read_csv(RESULTS / "stage3c_faithfulness.csv"),
               "static": pd.read_csv(RESULTS / "stage4" / "pilot_followup" / "s4c_static_control_faithfulness.csv")}
        REF_B = ["ig_zero", "gshap", "occlusion", "treeshap_xgb"]
        sel_agree = lambda D: D["agree"][D["agree"].method_a.isin(ATTN) & D["agree"].method_b.isin(REF_B)
                                         & D["agree"].metric.isin(["spearman", "jaccard@5"])]
        cmp_agree = side_by_side(sel_agree(D17), sel_agree(D18), ["method_a", "method_b", "metric"])
        FM = ["attn_rollout", "attn_mean", "attn_last", "ig_zero", "gshap", "occlusion", "abs_x", "random"]
        sel_faith = lambda D: D["faith"][D["faith"].method.isin(FM) & D["faith"].metric.isin(["del_auc", "ins_auc", "aopc_gap"])]
        cmp_faith = side_by_side(sel_faith(D17), sel_faith(D18), ["mask_baseline", "metric", "method"])
        sel_st = lambda D: static_minus_dynamic(D["static"][D["static"].metric.isin(["del_auc", "aopc_gap", "comp@5", "ins_auc"])],
                                                D["faith"])
        cmp_static = side_by_side(sel_st(D17), sel_st(D18), ["mask_baseline", "metric", "method"])
        sel_rand = lambda D: D["rand"][(D["rand"].metric == "spearman_vs_original")]
        cmp_rand = side_by_side(sel_rand(D17), sel_rand(D18), ["method", "level"])
        fl = []
        F17, F18 = findings(D17), findings(D18)
        for (lab, v17), (_, v18) in zip(F17, F18):
            a, b = float(v17.mean()), float(v18.mean())
            fl.append({"finding": lab, "mean_2017": a,
                       "seed_range_2017": f"{v17.min():.3f} to {v17.max():.3f}" if len(v17) else "",
                       "mean_2018": b, "seed_range_2018": f"{v18.min():.3f} to {v18.max():.3f}" if len(v18) else "",
                       "same_sign": bool(np.sign(a) == np.sign(b)), "ratio_2018_over_2017": b / a if a else np.nan})
        find_df = pd.DataFrame(fl)
        agg_s = time.time() - t_agg

        # ---- runtime
        rt = pd.DataFrame([results[s]["runtime"] for s in seeds])
        total_s = time.time() - t_start
        rt_all = pd.concat([pd.DataFrame([{"seed": "pre (selection, NN distances, val set)", "seed_total_s": round(pre_s, 1)}]),
                            rt.assign(seed=rt.seed.astype(str)),
                            pd.DataFrame([{"seed": "aggregation and tables", "seed_total_s": round(agg_s, 1)},
                                          {"seed": "total (this invocation)", "seed_total_s": round(total_s, 1),
                                           "peak_gpu_mem_gb": max([r["peak_gpu_mem_gb"] for r in timing] + rt.peak_gpu_mem_gb.tolist())}])],
                           ignore_index=True)

        outputs = {"selection_counts": counts, "faithfulness": faith_df, "agreement": agree_df,
                   "agreement_robustness": robust_df, "robustness_counts": robust_counts, "stability": stab_df,
                   "randomisation": rand_df, "sanity": pd.DataFrame(sanity), "timing": pd.DataFrame(timing),
                   "3c_partial_agreement": part_df, "3c_faithfulness": f3c_df, "3c_gradattn_agreement": gagree_df,
                   "3c_gradattn_stability": pd.DataFrame(gstab), "3c_head_agreement": hagree_df, "3c_head_val": val_df,
                   "3c_head_selection": choice_df, "3c_floor_ceiling": fc_df, "3c_per_class": pc_df,
                   "3c_val_selection_counts": vcounts, "static_control_faithfulness": st_df,
                   "static_vs_dynamic": st_cmp, "slowloris_neardup_summary": nd_summary,
                   "slowloris_all_vs_excl": sl_df, "cmp2017_agreement": cmp_agree, "cmp2017_faithfulness": cmp_faith,
                   "cmp2017_static_minus_dynamic": cmp_static, "cmp2017_randomisation": cmp_rand,
                   "cmp2017_findings": find_df, "runtime": rt_all}
        for name, df in outputs.items():
            df.to_csv(out / f"stage7c_{name}.csv", index=False)
        entry = build_entry(outputs, classes, N, len(vsel), seeds, c7, out)
        (out / "stage7c_progress_entry.md").write_text(entry, encoding="utf-8")
        log.info(f"Wrote {len(outputs)} tables and the progress entry to {out}")
        fails = [r for r in sanity if not r["pass"]]
        if fails:
            raise StopRun("sanity check(s) failed after aggregation: "
                          + "; ".join(f"seed {r['seed']} {r['check']} = {r['value']}" for r in fails))
    except StopRun as e:
        write_stop(str(e))
        finish_run(run_dir, log, {"stopped": str(e)}, t_start)
        sys.exit(2)
    finish_run(run_dir, log, {"n_rows": N, "seeds": seeds, "sanity_all_pass": True}, t_start)


# ---------------------------------------------------------------- progress entry

def build_entry(T, classes, N, n_val, seeds, c7, out):
    def mean_over_seeds(df, idx, col, cls=H, **f):
        d = df[df["class"] == cls]
        for k, v in f.items():
            d = d[d[k].isin(v)] if isinstance(v, list) else d[d[k] == v]
        return d.pivot_table(index=idx, columns=col, values="mean", aggfunc="mean")

    fa, ag, st, ra = T["faithfulness"], T["agreement"], T["stability"], T["randomisation"]
    keep = ["del_auc", "ins_auc", "aopc_gap", "comp@5", "suff@20", "del_auc_margin", "ins_auc_margin"]
    order = ["ig_zero", "occlusion", "gshap", "abs_x", "attn_rollout", "attn_mean", "attn_last", "random"]
    ftab = {b: mean_over_seeds(fa, "method", "metric", mask_baseline=b, metric=keep).reindex(order)[keep].reset_index()
            for b in ("train_mean", "benign_sample")}
    atab = mean_over_seeds(ag, "method_a", "method_b", metric="spearman", method_a=ATTN,
                           method_b=["ig_zero", "gshap", "occlusion", "abs_x", "treeshap_xgb"]).reset_index()
    stab = mean_over_seeds(st, "method", "metric", metric=["spearman", "jaccard@5"]).reset_index()
    rtab = mean_over_seeds(ra, "method", "level", metric="spearman_vs_original")[[lv for lv, _ in RAND_LEVELS]].reset_index()
    pa = T["3c_partial_agreement"]
    ptab = mean_over_seeds(pa, ["method_a", "method_b"], "metric", method_a=ATTN + GRAD,
                           method_b=["ig_zero", "occlusion", "gshap"]).reset_index()
    fc = T["3c_floor_ceiling"]
    fct = fc[fc.seed == "mean"].pivot_table(index="method", columns="mask_baseline",
                                            values=["del_auc_scaled", "aopc_gap_scaled"])
    fct.columns = [f"{a} {b}" for a, b in fct.columns]
    fct = fct.reindex([m for m in ["ig_zero", "ig_benign", "gshap", "abs_x", "head_val_selected", "attn_rollout",
                                   "gradattn_rollout", "gradattn_chefer", "attn_mean", "attn_last"] if m in fct.index]).reset_index()
    hs = T["3c_head_selection"]
    sanity = T["sanity"]
    sel = T["selection_counts"][["class", "test_support", "correct_seed0", "correct_seed1", "correct_seed2",
                                 "correct_all_seeds", "selected", "xgb_correct_selected", "excluded_below_min_rows", "flag"]]
    # n per class and per seed for the main tables (n of del_auc rows, train mean; spearman rows; randomisation)
    ncls = fa[(fa.metric == "del_auc") & (fa.mask_baseline == "train_mean") & (fa.method == "random")
              & ~fa["class"].str.startswith("macro_")].pivot_table(index="class", columns="seed", values="n").reset_index()
    nrand = ra[(ra.metric == "spearman_vs_original") & (ra.level == "head_only") & (ra.method == "ig_zero")
               & ~ra["class"].str.startswith("macro_")].pivot_table(index="class", columns="seed", values="n").reset_index()
    sl = T["slowloris_all_vs_excl"]
    slm = sl.groupby(["table", "mask_baseline", "method", "metric"], as_index=False).agg(
        all_rows=("all_rows", "mean"), all_lo=("all_rows", "min"), all_hi=("all_rows", "max"), n_all=("n_all", "min"),
        excl=("excl_neardup", "mean"), excl_lo=("excl_neardup", "min"), excl_hi=("excl_neardup", "max"),
        n_excl=("n_excl", "min"), diff=("excl_minus_all", "mean"))
    slm = slm[(slm.mask_baseline.isin(["train_mean", "n/a"])) & slm.metric.isin(["del_auc", "aopc_gap", "spearman"])]
    ca, cf, cs, cr = (T[k] for k in ("cmp2017_agreement", "cmp2017_faithfulness", "cmp2017_static_minus_dynamic",
                                     "cmp2017_randomisation"))
    ca_s = ca[ca.metric == "spearman"]
    cf_m = cf[cf.metric.isin(["del_auc", "ins_auc", "aopc_gap"])]
    cs_m = cs[cs.metric.isin(["del_auc", "aopc_gap", "comp@5"])]
    cr_m = cr[cr["class"] == M7]
    rt = T["runtime"]
    L = [f"\n---\n\n## {datetime.now():%Y-%m-%d %H:%M}: Stage 7c, attention faithfulness on LycoS-Unicas-IDS2018 "
         f"(Stage 3, 3c and static-control equivalents)\n"]
    L.append(f"""**Files (all new):** `src/stage7c_explain_2018.py`; outputs in `{out.relative_to(ROOT).as_posix() if out.is_relative_to(ROOT) else out}/`
- `stage7c_*.csv` tables. Stage 3 layout: selection_counts, faithfulness, agreement, agreement_robustness, robustness_counts, stability, randomisation, sanity, timing. Stage 3c layout: `stage7c_3c_*`. Static control: static_control_faithfulness, static_vs_dynamic.
- New tables: slowloris_neardup_summary, slowloris_all_vs_excl, cmp2017_*, runtime.
- `raw/`: float16 attributions, curves, randomisation, gradattn, static vectors, selection, exact NN distances.
- `parts/`: per-seed resume files.

**Setup:** Stage 3, Stage 3c and the clean static class-mean control (stage4c) were rerun with no change of method, parameters or seeds. Only the data, checkpoints, preprocess.json and class list are new: 8 classes, 7 of them attack classes.
- The one forced adaptation: 2018 has no day column, so selection runs with a single day group. This is a plain random draw per class with the same sampling seed.
- Explanation set: {N:,} test rows. VAL set: {n_val:,} rows.
- In this entry, macro = `{H}`, the unweighted mean over the 7 attack classes. On 2018 these are exactly the 7 classes shared with 2017.
- `{SLOW_X}`: dos_slowloris rows whose exact nearest-train distance (standardised space) is >= {c7['neardup_threshold']:g}. It is reported as an extra class in every faithfulness and agreement table and is not part of any macro.
- In every table, `n` is per class and per seed, and classes with n < {SMALL_N} carry `n < {SMALL_N}` in `flag`.
""")
    L.append(f"**Selection counts**\n\n{md(sel)}\n\nn per class and seed in the faithfulness tables:\n\n{md(ncls)}\n\n"
             f"n per class and seed in the randomisation tables (a fixed per-class subset by design, as in 2017, so "
             f"every class is marked n < 100 there):\n\n{md(nrand)}\n")
    L.append(f"**dos_slowloris near duplicates** (exact direct-difference distance to the nearest of all 1,095,031 train rows)\n\n"
             f"{md(T['slowloris_neardup_summary'], '{:.4g}')}\n")
    L.append(f"**Sanity checks**\n\n{md(sanity[['check', 'seed', 'value', 'expected', 'pass']])}\n")
    L.append(f"**2018 faithfulness, train-mean masking** (macro, mean over seeds)\n\n{md(ftab['train_mean'], '{:.4f}')}\n\n"
             f"Benign-sample masking:\n\n{md(ftab['benign_sample'], '{:.4f}')}\n")
    L.append(f"**2018 agreement, Spearman** (macro, mean over seeds)\n\n{md(atab)}\n\n"
             f"**2018 cross-seed stability** (macro, mean over seed pairs)\n\n{md(stab)}\n\n"
             f"**2018 parameter randomisation, Spearman vs original** (macro, mean over seeds; head_only is trivially 1 "
             f"for attention)\n\n{md(rtab)}\n")
    L.append(f"**2018 Stage 3c**\n\nPartial agreement given |x| (macro, mean over seeds):\n\n{md(ptab)}\n\n"
             f"Head selection on VAL, evaluated on test:\n\n{md(hs[['seed', 'criterion', 'chosen_head', 'layer', 'test_del_auc', 'test_aopc_gap', 'test_rank_by_del_auc_of_12', 'test_oracle_head']], '{:.4f}')}\n\n"
             f"Floor/ceiling (random = 0, occlusion = 1), mean over seeds:\n\n{md(fct)}\n")
    L.append(f"**2018 static class-mean vs dynamic** (macro, mean over seeds)\n\n"
             f"{md(T['static_vs_dynamic'][T['static_vs_dynamic'].metric.isin(['del_auc', 'aopc_gap', 'comp@5', 'ins_auc', 'suff@20'])], '{:.4f}')}\n")
    L.append(f"**dos_slowloris, all selected rows vs near duplicates excluded** (mean over seeds, per-seed min to max; "
             f"faithfulness train-mean masking)\n\n{md(slm, '{:.4f}')}\n")
    L.append(f"**2017 vs 2018, common classes.** Every value is the mean over seeds, with the per-seed range; "
             f"`{M7}` is the macro over the 7 common classes. The 2017 values come from the saved Stage 3 / 3c / 4c CSVs.\n\n"
             f"Agreement, Spearman (attention vs references):\n\n{md(ca_s)}\n\n"
             f"Faithfulness:\n\n{md(cf_m)}\n\n"
             f"Static minus dynamic, deletion side (negative del AUC or positive AOPC gap / comp@5 means the class mean "
             f"is more faithful):\n\n{md(cs_m)}\n\n"
             f"Randomisation, Spearman vs original ({M7}):\n\n{md(cr_m)}\n\n"
             f"Key quantities ({M7}):\n\n{md(T['cmp2017_findings'])}\n")
    L.append(f"**Runtime**\n\n{md(rt, '{:.2f}')}\n")
    return "\n".join(L)


if __name__ == "__main__":
    main()
