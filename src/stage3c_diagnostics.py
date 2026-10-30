# Stage 3c: post-hoc diagnostics on the Stage 3 explanation set (clean test data, no attack).
# Read-only on results/stage1, results/stage2 and results/stage3. Writes results/stage3c/ (raw float16
# tensors, val selection), results/stage3c_*.csv and a run dir under results/runs/.
#
#   1. Partial Spearman of each attention variant vs each reference attribution, controlling for the |x|
#      ranking: per row, ranks are linearly residualised on the |x| rank and the residuals correlated, i.e.
#      (r_ar - r_az r_rz) / sqrt((1 - r_az^2) (1 - r_rz^2)) on Spearman correlations.
#   2. Head-level analysis: each of the 3 x 4 CLS rows ranked on its own (test set), and a head choice made on
#      a VAL explanation set (rows correct for all 3 seeds, up to 500 per class) by deletion AUC, then
#      evaluated on the existing test set.
#   3. Gradient-weighted attention (Chefer et al. 2021): per layer A_bar = mean over heads of
#      (dlogit_target / dA * A)^+, then
#        gradattn_rollout  same rollout as attn_rollout (0.5 A_bar + 0.5 I, row-normalised, multiplied)
#        gradattn_chefer   the original update R <- R + A_bar R (no normalisation), as a secondary variant
#      CLS row, CLS self-weight excluded.
#   4. Floor / ceiling: deletion AUC and AOPC gap rescaled so random = 0 and occlusion = 1.
#   5. Per-class table for the 10 attack classes.
#
# Attention is recomputed in fp32 from the checkpoints; reference attributions (IG, GradientSHAP, occlusion)
# are the stored float16 tensors from results/stage3/attr_seed*.npz.
#
# Usage: .venv\Scripts\python.exe -m src.stage3c_diagnostics
import copy
import json
import time

import numpy as np
import pandas as pd
import torch

from src import config as cfgmod
from src.config import load_config
from src.stage2_train import predict_torch
from src.preprocess import Preprocessor
from src.stage3_explain import (ATTN, BENIGN, STAGE3, agreement_rows, attention_variants, centred_ranks,
                                class_rows, curve_metrics, integrated_gradients, load_ftt, perturbation_curves,
                                ranking, rowwise_spearman, select_rows, spearman_from_ranks, target_out, timed)
from src.utils import RESULTS, finish_run, resolve, start_run

STAGE3C = {
    "val_n_per_class": 500,
    "val_sel_seed": 2027,
    "grad_attn_batch": 128,
    # Smoke test only: if > 0, use the first n rows per class (by rank_in_class) of the test set.
    "debug_rows_per_class": 0,
}
REFS = ["ig_zero", "ig_benign", "gshap", "occlusion"]
GRAD = ["gradattn_rollout", "gradattn_chefer"]
H = "macro_attack_classes"


# ---------------------------------------------------------------- helpers

def grad_attention(model, X, target, bs):
    # Chefer et al. 2021 gradient-weighted attention, two rollout variants (see header).
    out = {k: [] for k in GRAD}
    zero_rows = 0
    for i in range(0, X.shape[0], bs):
        xb, t = X[i:i + bs], target[i:i + bs]
        with torch.enable_grad():
            logits, attns = model(xb, return_attn=True)
            grads = torch.autograd.grad(logits.gather(1, t[:, None]).sum(), attns)
        B, T = xb.shape[0], attns[0].shape[-1]
        eye = torch.eye(T, device=X.device)
        R1 = eye.expand(B, T, T)
        R2 = eye.expand(B, T, T).clone()
        for A, G in zip(attns, grads):
            Abar = (G * A).clamp(min=0).mean(1).detach()      # (B, T, T), mean over heads
            zero_rows += int((Abar[:, 0, 1:].sum(-1) == 0).sum())
            Al = 0.5 * Abar + 0.5 * eye
            R1 = (Al / Al.sum(-1, keepdim=True)) @ R1
            R2 = R2 + Abar @ R2
        out["gradattn_rollout"].append(R1[:, 0, 1:].cpu())
        out["gradattn_chefer"].append(R2[:, 0, 1:].cpu())
    return {k: torch.cat(v).numpy() for k, v in out.items()}, zero_rows


@torch.no_grad()
def deletion_probs(model, X, base, order, target, rows):
    # MoRF deletion only: target-class probability with the first k ranked features masked, k = 0..F.
    N, F = X.shape
    pos = torch.empty_like(order)
    pos.scatter_(1, order, torch.arange(F, device=X.device).expand(N, F))
    ks = torch.arange(F + 1, device=X.device).view(1, F + 1, 1)
    step = max(1, rows // (F + 1))
    out = torch.empty(N, F + 1)
    for i in range(0, N, step):
        xb, bb, p, t = X[i:i + step, None, :], base[i:i + step, None, :], pos[i:i + step, None, :], target[i:i + step]
        B = xb.shape[0]
        pr = torch.softmax(model(torch.where(p < ks, bb, xb).reshape(-1, F)), -1).view(B, F + 1, -1)
        out[i:i + B] = pr.gather(2, t.view(B, 1, 1).expand(B, F + 1, 1)).squeeze(2).cpu()
    return out.numpy().astype(np.float64)


def macro_attack(vals, y, classes):
    # Unweighted mean over the attack classes of the per-class means.
    return float(np.mean([np.nanmean(vals[y == ci]) for ci, n in enumerate(classes) if n != BENIGN]))


def partial_rows(scores, z, y, classes, seed, pairs):
    # Raw and partial (controlling for the |x| rank) Spearman for each (a, b) pair.
    rk = {k: centred_ranks(v) for k, v in scores.items()}
    rz = centred_ranks(z)
    rows = []
    for a, b in pairs:
        r_ab = spearman_from_ranks(rk[a], rk[b])
        rows += class_rows(r_ab, y, classes, seed=seed, method_a=a, method_b=b, metric="spearman")
        if a == "abs_x" or b == "abs_x":
            continue                       # partial on itself is undefined
        r_az, r_bz = spearman_from_ranks(rk[a], rz), spearman_from_ranks(rk[b], rz)
        den = np.sqrt((1 - r_az ** 2) * (1 - r_bz ** 2))
        part = np.divide(r_ab - r_az * r_bz, den, out=np.full(len(r_ab), np.nan), where=den > 1e-12)
        rows += class_rows(part, y, classes, seed=seed, method_a=a, method_b=b, metric="partial_spearman_given_abs_x")
    return rows


# ---------------------------------------------------------------- main

def main():
    cfgmod.DEFAULTS.setdefault("stage3", copy.deepcopy(STAGE3))
    cfgmod.DEFAULTS.setdefault("stage3c", copy.deepcopy(STAGE3C))
    cfgmod.DEFAULTS["paths"].setdefault("stage3_dir", "results/stage3")
    cfg, overrides = load_config("Stage 3c: post-hoc diagnostics on the Stage 3 explanation set.")
    c, cc = cfg["stage3"], cfg["stage3c"]
    t_start = time.time()
    run_dir, log = start_run("stage3c_diagnostics", cfg, overrides)
    dev = torch.device("cuda")
    s1, s2, s3 = (resolve(cfg["paths"][k]) for k in ("stage1_dir", "stage2_dir", "stage3_dir"))
    out_dir = RESULTS / "stage3c"
    out_dir.mkdir(parents=True, exist_ok=True)
    timing, sanity, part, faith, gagree, hagree, val_rows, choice, stab = [], [], [], [], [], [], [], [], []

    def sane(name, seed, value, expected, passed):
        sanity.append({"check": name, "seed": seed, "value": value, "expected": expected, "pass": bool(passed)})
        log.info(f"[sanity] seed={seed} {name}: {value} (expected {expected}) -> {'PASS' if passed else 'FAIL'}")

    # ---- Stage 3 explanation set, rebuilt from the saved selection (read-only)
    feats = json.loads((s1 / "features.json").read_text(encoding="utf-8"))
    label_map = json.loads((s1 / "label_map.json").read_text(encoding="utf-8"))
    classes = [k for k, _ in sorted(label_map.items(), key=lambda kv: kv[1])]
    F = len(feats)
    pp = Preprocessor.load(s2 / "preprocess.json")
    sel = pd.read_parquet(s3 / "selection.parquet")
    keep = np.arange(len(sel))
    if cc["debug_rows_per_class"] > 0:
        keep = np.flatnonzero(sel["rank_in_class"].to_numpy() < cc["debug_rows_per_class"])
        sel = sel.iloc[keep].reset_index(drop=True)
    test = pd.read_parquet(s1 / "test.parquet", columns=feats)
    Xs_np = pp.transform(test.iloc[sel["test_row"].to_numpy()])
    del test
    y = sel["label_id"].to_numpy().astype(np.int64)
    N = len(y)
    Xs, target = torch.from_numpy(Xs_np).to(dev), torch.from_numpy(y).to(dev)
    absx = np.abs(Xs_np)
    rows_bl = np.load(s3 / "baseline_rows.npz")
    train = pd.read_parquet(s1 / "train.parquet", columns=feats)
    mask_bases = {"train_mean": torch.zeros_like(Xs),
                  "benign_sample": torch.from_numpy(pp.transform(train.iloc[rows_bl["mask_benign_train_rows"][keep]])).to(dev)}
    del train
    treeshap = np.abs(np.load(s3 / "treeshap_xgb.npz")["treeshap"].astype(np.float32))[keep]

    # ---- VAL explanation set: rows correct for all 3 seeds, up to 500 per class (same procedure as test)
    val = pd.read_parquet(s1 / "val.parquet", columns=feats + ["label_id", "day"])
    Xva = pp.transform(val)
    y_va, d_va = val["label_id"].to_numpy().astype(np.int64), val["day"].to_numpy()
    del val
    seeds = c["seeds"]
    preds_va = {}
    Xva_t = torch.from_numpy(Xva).to(dev)
    for s in seeds:
        m, ck = load_ftt(s2 / f"ftt_seed{s}.pt", dev)
        preds_va[s] = predict_torch(m, Xva_t, ck["config"]["stage2"]["ftt"]["eval_batch_size"]).argmax(1).cpu().numpy()
        del m
    del Xva_t
    torch.cuda.empty_cache()
    vsel, _, vcounts = select_rows(y_va, d_va, preds_va, classes, cc["val_n_per_class"], c["min_per_class"],
                                   cc["val_sel_seed"])
    vcounts.to_csv(RESULTS / "stage3c_val_selection_counts.csv", index=False)
    pd.DataFrame({"val_row": vsel, "label_id": y_va[vsel], "day": d_va[vsel]}).to_parquet(
        out_dir / "val_selection.parquet", index=False)
    yv = y_va[vsel]
    Xv, tv = torch.from_numpy(Xva[vsel]).to(dev), torch.from_numpy(yv).to(dev)
    del Xva
    log.info(f"VAL explanation set: {len(vsel)} rows\n{vcounts[['class', 'correct_all_seeds', 'selected']].to_string(index=False)}")

    heads = [(l, h) for l in range(3) for h in range(4)]
    hname = lambda lh: f"head_L{lh[0] + 1}H{lh[1] + 1}"
    grad_scores = {}
    for s in seeds:
        model, _ = load_ftt(s2 / f"ftt_seed{s}.pt", dev)
        st = {k: v[keep] for k, v in np.load(s3 / f"attr_seed{s}.npz").items()}
        _, pr = target_out(model, Xs, target, c["eval_rows"])
        d = float(np.abs(pr.cpu().numpy() - st["target_prob"]).max())
        sane("target prob matches Stage 3", s, f"{d:.1e}", "< 1e-5", d < 1e-5)
        att, _ = attention_variants(model, Xs, c["attn_batch"])
        d = max(float(np.abs(att[k] - st[k].astype(np.float32)).max()) for k in ATTN + ["attn_cls_all"])
        sane("fp32 attention vs stored float16 (max abs diff)", s, f"{d:.1e}", "< 1e-3 (float16 rounding)", d < 1e-3)

        rng = np.random.default_rng(c["sel_seed"] + 600 + s)
        scores = {k: att[k] for k in ATTN}
        scores.update({k: np.abs(st[k].astype(np.float32)) for k in REFS})

        # ---- 3. gradient-weighted attention
        with timed(timing, "gradattn (both variants)", s, N):
            ga, zero_rows = grad_attention(model, Xs, target, cc["grad_attn_batch"])
        sane("gradattn: (row, layer) pairs with an all-zero CLS row in A_bar", s, zero_rows, "small", True)
        np.savez_compressed(out_dir / f"gradattn_seed{s}.npz", **{k: v.astype(np.float16) for k, v in ga.items()})
        scores.update(ga)
        grad_scores[s] = ga
        scores["abs_x"] = absx
        scores["random"] = rng.random((N, F))

        # ---- 1. partial agreement controlling for |x|
        pairs = [(a, r) for a in ATTN + GRAD + ["abs_x", "random"] for r in REFS] + \
                [(a, b) for i, a in enumerate(REFS) for b in REFS[i + 1:]]
        part += partial_rows(scores, absx, y, classes, s, pairs)

        # ---- 3. gradattn faithfulness and agreement (same metrics as Stage 3)
        with timed(timing, "gradattn curves (2 variants x 2 baselines)", s, N):
            for k in GRAD:
                order = torch.from_numpy(ranking(ga[k], rng)).to(dev)
                for bname, base in mask_bases.items():
                    cur = perturbation_curves(model, Xs, base, order, target, c)
                    for metric, vals in curve_metrics(cur, c["topk"], c["aopc_steps"], c["norm_min_gap"]).items():
                        faith += class_rows(vals, y, classes, normalised="_norm" in metric, seed=s,
                                            mask_baseline=bname, method=k, metric=metric)
        ag = {k: scores[k] for k in GRAD + ATTN + REFS + ["abs_x", "random"]}
        ag["treeshap_xgb"] = treeshap
        gagree += [r for r in agreement_rows(ag, y, classes, rng, c["topk"], seed=s) if r["method_a"] in GRAD]

        # ---- 2a. every head on the test set (train-mean masking)
        with timed(timing, "head curves (12 heads, train mean)", s, N):
            for lh in heads:
                hs = att["attn_cls_all"][:, lh[0], lh[1], 1:]
                cur = perturbation_curves(model, Xs, mask_bases["train_mean"],
                                          torch.from_numpy(ranking(hs, rng)).to(dev), target, c)
                for metric, vals in curve_metrics(cur, c["topk"], c["aopc_steps"], c["norm_min_gap"]).items():
                    faith += class_rows(vals, y, classes, normalised="_norm" in metric, seed=s,
                                        mask_baseline="train_mean", method=hname(lh), metric=metric)
                for ref in ("ig_zero", "occlusion"):
                    hagree += class_rows(rowwise_spearman(hs, scores[ref]), y, classes, seed=s, method_a=hname(lh),
                                         method_b=ref, metric="spearman")

        # ---- 2b. head choice on VAL (attention, IG zero and deletion only)
        with timed(timing, "val: attention + ig_zero + 12-head deletion", s, len(yv)):
            vatt, _ = attention_variants(model, Xv, c["attn_batch"])
            vig, _, vrel = integrated_gradients(model, Xv, tv, torch.zeros(1, F, device=dev), c)
            for lh in heads:
                hs = vatt["attn_cls_all"][:, lh[0], lh[1], 1:]
                dp = deletion_probs(model, Xv, torch.zeros_like(Xv), torch.from_numpy(ranking(hs, rng)).to(dev), tv,
                                    c["eval_rows"])
                val_rows.append({"seed": s, "head": hname(lh), "layer": lh[0] + 1, "head_idx": lh[1] + 1,
                                 "val_del_auc": macro_attack(np.trapezoid(dp, dx=1 / F, axis=1), yv, classes),
                                 "val_spearman_ig_zero": macro_attack(rowwise_spearman(hs, np.abs(vig)), yv, classes)})
        sane("val ig_zero completeness |delta|/|f(x)-f(b)| p95", s, f"{np.quantile(vrel, 0.95):.2e}",
             "reported as is (50 steps)", True)
        del model
        torch.cuda.empty_cache()

    faith_df, gagree_df, hagree_df, val_df = pd.DataFrame(faith), pd.DataFrame(gagree), pd.DataFrame(hagree), \
        pd.DataFrame(val_rows)
    part_df = pd.DataFrame(part)

    # ---- gradattn cross-seed stability
    for i, sa in enumerate(seeds):
        for sb in seeds[i + 1:]:
            for k in GRAD:
                stab += class_rows(rowwise_spearman(grad_scores[sa][k], grad_scores[sb][k]), y, classes, method=k,
                                   seed_a=sa, seed_b=sb, metric="spearman")

    # ---- 2c. head selection summary
    hm = faith_df[(faith_df["class"] == H) & faith_df.method.str.startswith("head_")]
    ha = hagree_df[hagree_df["class"] == H]
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
        log.info(f"Head choice by {crit}: {dict(zip(g.seed, g.chosen_head))}; same head across seeds: {same_head}; "
                 f"same layer: {same_layer}")
        choice_df.loc[g.index, "same_head_all_seeds"] = same_head
        choice_df.loc[g.index, "same_layer_all_seeds"] = same_layer
    # Add the val-selected head as a method for the floor/ceiling table.
    sel_rows = []
    for _, r in choice_df[choice_df.criterion == "val_del_auc (min)"].iterrows():
        sel_rows.append(faith_df[(faith_df.seed == r.seed) & (faith_df.method == r.chosen_head)].assign(method="head_val_selected"))
    faith_df = pd.concat([faith_df] + sel_rows, ignore_index=True)

    # ---- 4. floor / ceiling (random = 0, occlusion = 1), attack-class macro
    f3 = pd.read_csv(RESULTS / "stage3_faithfulness.csv")
    allf = pd.concat([f3, faith_df], ignore_index=True)
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

    # ---- 5. per-class table (attack classes): deletion AUC, AOPC gap, Spearman with ig_zero
    a3 = pd.read_csv(RESULTS / "stage3_agreement.csv")
    meths = ["attn_rollout", "ig_zero", "occlusion", "abs_x", "random"]
    attack = [k for k in classes if k != BENIGN]
    pc = f3[f3.method.isin(meths) & f3.metric.isin(["del_auc", "aopc_gap"]) & f3["class"].isin(attack + [H])]
    pc = pc[["seed", "mask_baseline", "method", "metric", "class", "mean"]]
    sp = a3[(a3.metric == "spearman") & (((a3.method_a == "ig_zero") & a3.method_b.isin(meths))
                                         | ((a3.method_b == "ig_zero") & a3.method_a.isin(meths)))
            & a3["class"].isin(attack + [H])].copy()
    sp["method"] = np.where(sp.method_a == "ig_zero", sp.method_b, sp.method_a)
    sp = sp.assign(mask_baseline="n/a", metric="spearman_with_ig_zero")[pc.columns]
    pc_df = pd.concat([pc, sp], ignore_index=True)
    pc_mean = pc_df.groupby(["mask_baseline", "method", "metric", "class"], as_index=False)["mean"].mean().assign(seed="mean")
    pc_df = pd.concat([pc_df.assign(seed=pc_df.seed.astype(str)), pc_mean], ignore_index=True)

    outputs = {"partial_agreement": part_df, "faithfulness": faith_df, "gradattn_agreement": gagree_df,
               "gradattn_stability": pd.DataFrame(stab), "head_agreement": hagree_df, "head_val": val_df,
               "head_selection": choice_df, "floor_ceiling": fc_df, "per_class": pc_df,
               "sanity": pd.DataFrame(sanity), "timing": pd.DataFrame(timing)}
    for name, df in outputs.items():
        df.to_csv(RESULTS / f"stage3c_{name}.csv", index=False)

    # ---- console summary (macro over attack classes)
    pd.set_option("display.width", 250)
    pd.set_option("display.max_columns", 40)
    pm = part_df[part_df["class"] == H]
    log.info("1. Spearman and partial Spearman given |x| (macro attack classes, mean over seeds):\n"
             + pm.groupby(["method_a", "method_b", "metric"])["mean"].mean().unstack().round(3).to_string())
    log.info("2. Heads on test (train-mean masking, macro attack classes, mean over seeds):\n"
             + pd.concat([hm.groupby(["method", "metric"])["mean"].mean().unstack()[["del_auc", "aopc_gap"]],
                          ha.groupby(["method_a", "method_b"])["mean"].mean().unstack()], axis=1).round(3).to_string())
    log.info("2. Head choice on VAL, evaluated on test:\n" + choice_df.round(4).to_string(index=False))
    gm = faith_df[(faith_df["class"] == H) & faith_df.method.isin(GRAD)
                  & faith_df.metric.isin(["del_auc", "ins_auc", "aopc_gap", "comp@5", "comp@20", "suff@20"])]
    log.info("3. Gradient-weighted attention faithfulness (mean over seeds):\n"
             + gm.groupby(["mask_baseline", "method", "metric"])["mean"].mean().unstack().round(4).to_string())
    gg = gagree_df[(gagree_df["class"] == H) & gagree_df.metric.isin(["spearman", "jaccard@5"])]
    log.info("3. Gradient-weighted attention agreement (mean over seeds):\n"
             + gg.groupby(["method_a", "method_b", "metric"])["mean"].mean().unstack().round(3).to_string())
    if stab:
        sd = pd.DataFrame(stab)
        log.info("3. Gradient-weighted attention cross-seed Spearman (macro attack, mean over pairs):\n"
                 + sd[sd["class"] == H].groupby("method")["mean"].mean().round(3).to_string())
    log.info("4. Floor/ceiling (random = 0, occlusion = 1), mean over seeds:\n"
             + fc_mean.set_index(["mask_baseline", "method"]).round(3).to_string())
    log.info("5. Per-class (attack classes, mean over seeds; del/aopc with train-mean masking):\n"
             + pc_mean[pc_mean.mask_baseline.isin(["train_mean", "n/a"])]
             .pivot_table(index="class", columns=["metric", "method"], values="mean").round(3).to_string())
    log.info("Timing:\n" + pd.DataFrame(timing).to_string(index=False))
    finish_run(run_dir, log, {"n_test_rows": N, "n_val_rows": int(len(vsel)), "seeds": seeds,
                              "sanity_all_pass": all(r["pass"] for r in sanity)}, t_start)


if __name__ == "__main__":
    main()
