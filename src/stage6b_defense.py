# Stage 6 PILOT v2: corrected evaluation protocol for the attention-attribution consistency detector.
#
# v1 problem: every positive (successful evasion) is predicted benign while every negative was an attack-class row,
# so a detector could separate "predicted benign" from "predicted attack". v2 (primary protocol):
#   positives  adversarial rows predicted benign (A1 from Stage 5, adaptive A4 / A5), Stage 4 pilot rows, seed 0
#   negatives  GENUINE benign TEST rows predicted benign (1,000 sampled)
#   thresholds 95th percentile of each score on ALL validation benign rows predicted benign (5% FPR); the realised
#              test FPR is reported, as is the FPR under thresholds from a 1,000-row val sample
# v1 (secondary column): negatives = clean attack-class test rows + non-flipped A0 rows, thresholds from val
# attack-class rows + their non-flipped A0 rows (as in src/stage6_defense.py).
# Class-confound check: on CLEAN rows only, AUROC of each score for predicted-benign (test benign) vs
# predicted-attack (clean attack-class pilot rows).
# Detectors (higher = more suspicious): 12 consistency scores ({rollout, attn_mean} x {grad x input, IG-20} x
# {Spearman, top-10 Jaccard with a Spearman tie-breaker, cosine}), -max softmax, -logit margin, Mahalanobis and kNN-5
# to the predicted class, and Mahalanobis and kNN-5 to BENIGN train rows.
# Adaptive attackers (constrained, same projection and independent checker):
#   A4  keep the consistency surrogate (cosine rollout vs |grad x input| of the benign logit) above its v2 threshold
#   A5  keep the differentiable kNN-5 distance to benign train rows below its v2 threshold minus a margin
#   each with a hinge (lambda in {1, 10, 100}) and a normalised-gradient variant (weight 0.5).
#
# Usage:
#   .venv\Scripts\python.exe -m src.stage6b_defense
import copy
import json
import math
import time

import numpy as np
import pandas as pd
import torch

from src import config as cfgmod
from src.config import load_config
from src.preprocess import Preprocessor
from src.stage3_explain import integrated_gradients, jaccard, load_ftt, ranking, rowwise_spearman
from src.stage4_attack import STAGE4, rollout_cls
from src.stage4_constraints import TorchPreprocessor, check_validity, derive, load_spec_csv, make_context, project
from src.stage6_defense import (BENIGN, STAGE6, Baselines, a0_rows, a4_attack, attention_and_logits, cv_gain,
                                grad_x_input, make_rows, norm_abs)
from src.utils import finish_run, resolve, start_run

STAGE6B = {
    "n_test_benign": 1000,
    "n_val_benign_sample": 1000,
    "a5_lambdas": [1.0, 10.0, 100.0],
    "a5_norm_w": 0.5,
    "a5_knn_margin_frac": 0.02,
    "n_boot": 1000,
    # Cap on val benign rows used for calibration (null = all; set only for smoke tests).
    "val_benign_cap": None,
    "score_batch": 4096,
    # Larger IG batches for scoring the ~247k val benign rows (same 20-step IG).
    "grad_batch": 128,
    "ig_internal_batch": 2560,
}
PRIMARY = "cons_rollout_gxi_spearman"
BASE_V2 = ["msp", "margin", "mahalanobis_benign", "knn5_benign"]


# ---------------------------------------------------------------- scores

def detector_scores(model, Z, base, c):
    # All detector scores for rows Z (higher = more suspicious), computed in chunks.
    parts, preds = [], []
    for i in range(0, Z.shape[0], c["score_batch"]):
        zb = Z[i:i + c["score_batch"]]
        lg, ro, me = attention_and_logits(model, zb)
        pred = lg.argmax(1)
        p = torch.softmax(lg, -1)
        top2 = lg.topk(2, dim=1).values
        gxi = norm_abs(grad_x_input(model, zb, pred))
        ig, _, _ = integrated_gradients(model, zb, pred, torch.zeros(1, zb.shape[1], device=zb.device),
                                        {**c, "ig_steps": c["ig_steps_det"]})
        ig = norm_abs(ig)
        maha, knn = base.scores(zb, pred)
        ben = torch.zeros_like(pred)
        maha_b, knn_b = base.scores(zb, ben)
        s = {"msp": -p.max(1).values.cpu().numpy(), "margin": -(top2[:, 0] - top2[:, 1]).cpu().numpy(),
             "mahalanobis": maha, "knn5": knn, "mahalanobis_benign": maha_b, "knn5_benign": knn_b}
        rng = np.random.default_rng(0)
        for an, A in (("rollout", norm_abs(ro)), ("attnmean", norm_abs(me))):
            for bn, B in (("gxi", gxi), ("ig20", ig)):
                sp = rowwise_spearman(A, B)
                s[f"cons_{an}_{bn}_spearman"] = -sp
                # Jaccard takes few discrete values; a tiny Spearman term (|.| <= 1e-3, below one Jaccard step
                # of 1/19) breaks ties without changing the order between different Jaccard values.
                s[f"cons_{an}_{bn}_jacc10"] = -(jaccard(ranking(A, rng), ranking(B, rng), 10) + 1e-3 * sp)
                s[f"cons_{an}_{bn}_cos"] = -(A * B).sum(1) / np.clip(np.linalg.norm(A, axis=1) * np.linalg.norm(B, axis=1),
                                                                     1e-30, None)
        parts.append(pd.DataFrame(s))
        preds.append(pred.cpu().numpy())
    return pd.concat(parts, ignore_index=True), np.concatenate(preds)


# ---------------------------------------------------------------- AUROC with fast bootstrap

def auroc(neg, pos):
    # Mann-Whitney AUROC (ties count one half).
    if len(neg) == 0 or len(pos) == 0:
        return np.nan
    allv = np.r_[neg, pos]
    r = pd.Series(allv).rank().to_numpy()
    return (r[len(neg):].sum() - len(pos) * (len(pos) + 1) / 2) / (len(pos) * len(neg))


def boot(neg, pos, thr, n, rng, chunk=50):
    # Bootstrap CIs of AUROC and TPR (negatives and positives resampled separately), vectorised on the GPU.
    if len(pos) == 0 or n == 0:
        return (np.nan, np.nan), (np.nan, np.nan)
    N = torch.tensor(neg, device="cuda", dtype=torch.float32)
    P = torch.tensor(pos, device="cuda", dtype=torch.float32)
    au, tp = [], []
    for i in range(0, n, chunk):
        b = min(chunk, n - i)
        ni = torch.from_numpy(rng.integers(0, len(neg), (b, len(neg)))).cuda()
        pi = torch.from_numpy(rng.integers(0, len(pos), (b, len(pos)))).cuda()
        nb, pb = N[ni], P[pi]
        cmp = (pb[:, :, None] > nb[:, None, :]).float() + 0.5 * (pb[:, :, None] == nb[:, None, :]).float()
        au.append(cmp.mean((1, 2)).cpu().numpy())
        tp.append((pb >= thr).float().mean(1).cpu().numpy())
    au, tp = np.concatenate(au), np.concatenate(tp)
    return np.percentile(au, [2.5, 97.5]), np.percentile(tp, [2.5, 97.5])


def evaluate(neg, pos, thr, cls_pos, classes, dets, n_boot, rng, **key):
    rows = []
    for det in dets:
        n, p = neg[det].to_numpy(), pos[det].to_numpy()
        (alo, ahi), (tlo, thi) = boot(n, p, thr[det], n_boot, rng)
        rows.append({**key, "detector": det, "source_class": "all", "n_neg": len(n), "n_pos": len(p),
                     "auroc": auroc(n, p), "auroc_lo": alo, "auroc_hi": ahi,
                     "tpr_at_5fpr": float((p >= thr[det]).mean()) if len(p) else np.nan, "tpr_lo": tlo, "tpr_hi": thi,
                     "test_fpr": float((n >= thr[det]).mean())})
        for k in sorted(set(cls_pos)):
            pk = p[cls_pos == k]
            (alo, ahi), (tlo, thi) = boot(n, pk, thr[det], n_boot, rng)
            rows.append({**key, "detector": det, "source_class": classes[k], "n_neg": len(n), "n_pos": len(pk),
                         "auroc": auroc(n, pk), "auroc_lo": alo, "auroc_hi": ahi,
                         "tpr_at_5fpr": float((pk >= thr[det]).mean()), "tpr_lo": tlo, "tpr_hi": thi,
                         "test_fpr": float((n >= thr[det]).mean())})
    return rows


# ---------------------------------------------------------------- adaptive attacker A5 (kNN to benign)

def a5_attack(model, S, tp, eps, c, lam, norm_w, knn_thr, base, dev, seed):
    # Constrained PGD: maximise min(margin_benign, kappa) - lam * relu(knn - target) (hinge), or mix unit-norm
    # gradients of min(margin, kappa) and -knn (normalised variant). knn = mean distance of the preprocessed row
    # to its 5 nearest benign train rows (differentiable through cdist and topk). target = threshold x (1 - margin).
    ctx = make_context(S.R, S.feats, S.floors, eps, S.frozen, dev)
    gen = torch.Generator(device=dev).manual_seed(seed)
    kappa, ref, k = c["a3_kappa"], base.ref[BENIGN], base.k
    target = knn_thr * (1 - c["a5_knn_margin_frac"])

    def terms(xv):
        z = tp(derive(xv, ctx)).float()
        lg = model(z)
        knn = torch.cdist(z, ref).topk(k, largest=False).values.mean(1)
        return lg[:, BENIGN] - lg[:, 1:].amax(1), knn

    best_key = torch.full((S.R.shape[0],), float("-inf"), dtype=torch.float64, device=dev)
    best_x = None
    for run in range(1 + c["restarts"]):
        delta = torch.zeros_like(ctx.lo) if run == 0 else \
            ctx.lo + (ctx.hi - ctx.lo) * torch.rand(ctx.lo.shape, generator=gen, device=dev, dtype=torch.float64)
        for step in range(c["steps"] + 1):
            x, _ = project(delta, ctx)
            xv = x.detach().requires_grad_(True)
            e, knn = terms(xv)
            obj = torch.clamp(e, max=kappa) - lam * torch.relu(knn - target)
            succ = e > 0
            key = (1e7 * (succ & (knn < knn_thr)) + 1e6 * succ).double() + obj.detach().double()
            b = key.detach() > best_key
            best_key = torch.where(b, key.detach(), best_key)
            best_x = x.detach().clone() if best_x is None else torch.where(b[:, None], x.detach(), best_x)
            if step == c["steps"]:
                break
            if norm_w is None:
                g, = torch.autograd.grad(obj.sum(), xv)
                g = g.double() * ctx.budget
            else:
                g1, = torch.autograd.grad(torch.clamp(e, max=kappa).sum(), xv, retain_graph=True)
                g2, = torch.autograd.grad((-knn).sum(), xv)
                g1, g2 = g1.double() * ctx.budget, g2.double() * ctx.budget
                g = (1 - norm_w) * g1 / g1.norm(dim=1, keepdim=True).clamp_min(1e-30) + \
                    norm_w * g2 / g2.norm(dim=1, keepdim=True).clamp_min(1e-30)
            a_t = c["alpha0"] * 0.5 * (1 + math.cos(math.pi * step / c["steps"]))
            delta = torch.minimum(torch.maximum(delta + a_t * torch.sign(g), ctx.lo), ctx.hi)
    with torch.no_grad():
        return best_x.cpu().numpy(), tp(best_x).float()


# ---------------------------------------------------------------- main

def main():
    cfgmod.DEFAULTS.setdefault("stage6b", {**copy.deepcopy(STAGE4), **copy.deepcopy(STAGE6), **copy.deepcopy(STAGE6B)})
    cfg, overrides = load_config("Stage 6 pilot v2: corrected protocol.")
    c = cfg["stage6b"]
    out = resolve("results/stage6/pilot_v2")
    out.mkdir(parents=True, exist_ok=True)
    t0 = time.time()
    run_dir, log = start_run("stage6b_pilot_v2", cfg, overrides)
    dev = torch.device("cuda")
    s1, s2, s3 = resolve("results/stage1"), resolve("results/stage2"), resolve("results/stage3")
    feats = json.loads((s1 / "features.json").read_text(encoding="utf-8"))
    lm = json.loads((s1 / "label_map.json").read_text(encoding="utf-8"))
    classes = [k for k, _ in sorted(lm.items(), key=lambda kv: kv[1])]
    pp = Preprocessor.load(s2 / "preprocess.json")
    tp = TorchPreprocessor(pp, dev)
    spec = load_spec_csv()
    floors = spec.set_index("feature")["zero_floor_scale"].astype(float).to_dict()
    rng = np.random.default_rng(c["rng_seed"] + 6000)       # same stream as v1 for the v1 val sample and baselines
    sel = pd.read_parquet(s3 / "selection.parquet")
    pos = np.flatnonzero((sel["class"] != "benign").to_numpy() & (sel["rank_in_class"] < c["n_per_class"]).to_numpy())
    pos = pos[np.lexsort((sel["rank_in_class"].to_numpy()[pos], sel["label_id"].to_numpy()[pos]))]
    test = pd.read_parquet(s1 / "test.parquet", columns=feats + ["label_id"])
    Rte = test.iloc[sel["test_row"].to_numpy()[pos]][feats].to_numpy(np.float64).copy()
    yte = sel["label_id"].to_numpy()[pos].astype(np.int64)
    ben_te_idx = np.flatnonzero(test["label_id"].to_numpy() == BENIGN)
    Zben_te_all = test.iloc[ben_te_idx][feats]
    vs = pd.read_parquet(resolve("results/stage3c") / "val_selection.parquet")
    vs = vs[vs.label_id != BENIGN]
    vpick = np.concatenate([rng.choice(g.index.to_numpy(), size=min(c["val_per_class"], len(g)), replace=False)
                            for _, g in vs.groupby("label_id")])
    val = pd.read_parquet(s1 / "val.parquet", columns=feats + ["label_id"])
    Rva = val.iloc[vs.loc[vpick, "val_row"].to_numpy()][feats].to_numpy(np.float64).copy()
    yva = vs.loc[vpick, "label_id"].to_numpy().astype(np.int64)
    Zben_va = torch.from_numpy(pp.transform(val[val["label_id"] == BENIGN])).to(dev)
    del val
    train = pd.read_parquet(s1 / "train.parquet", columns=feats + ["label_id"])
    Ztr, ytr = pp.transform(train), train["label_id"].to_numpy()
    del train
    base = Baselines(Ztr, ytr, len(classes), c, dev, rng)
    del Ztr
    Ste = make_rows(Rte, yte, classes, feats, spec, floors, tp, dev)
    Sva = make_rows(Rva, yva, classes, feats, spec, floors, tp, dev)
    s = c["seeds"][0]
    model, _ = load_ftt(s2 / f"ftt_seed{s}.pt", dev)
    for p in model.parameters():
        p.requires_grad_(False)
    # ---- v2 negatives: genuine benign rows predicted benign
    with torch.no_grad():
        pv = torch.cat([model(Zben_va[i:i + 8192]).argmax(1) for i in range(0, len(Zben_va), 8192)])
    Zben_va = Zben_va[pv == BENIGN]
    if c["val_benign_cap"]:
        Zben_va = Zben_va[:c["val_benign_cap"]]
    ts = time.time()
    sc_vb, _ = detector_scores(model, Zben_va, base, c)
    log.info(f"val benign predicted benign: {len(Zben_va)} rows scored in {time.time() - ts:.0f} s")
    Zte_b = torch.from_numpy(pp.transform(Zben_te_all)).to(dev)
    with torch.no_grad():
        pt = torch.cat([model(Zte_b[i:i + 8192]).argmax(1) for i in range(0, len(Zte_b), 8192)]).cpu().numpy()
    ok = np.flatnonzero(pt == BENIGN)
    pick = rng.choice(ok, size=c["n_test_benign"], replace=False)
    sc_tb, _ = detector_scores(model, Zte_b[pick], base, c)
    del Zte_b
    dets = list(sc_vb.columns)
    thr2 = {d: float(np.quantile(sc_vb[d], 0.95)) for d in dets}
    sub = np.random.default_rng(c["rng_seed"] + 77).choice(len(sc_vb), size=c["n_val_benign_sample"], replace=False)
    thr2_small = {d: float(np.quantile(sc_vb[d].to_numpy()[sub], 0.95)) for d in dets}
    thr_rows = [{"detector": d, "thr_v2_all_val_benign": thr2[d], "thr_v2_val_sample_1000": thr2_small[d],
                 "test_fpr_v2_all": float((sc_tb[d] >= thr2[d]).mean()),
                 "test_fpr_v2_sample_1000": float((sc_tb[d] >= thr2_small[d]).mean())} for d in dets]
    # ---- class-confound check on clean rows: predicted benign (test benign) vs predicted attack (clean pilot rows)
    sc_te_clean, p_te_clean = detector_scores(model, Ste.z0, base, c)
    conf = [{"detector": d, "auroc_benign_vs_attack_clean": auroc(sc_te_clean[d].to_numpy(), sc_tb[d].to_numpy())}
            for d in dets]
    # ---- v1 calibration (secondary): val attack rows + non-flipped A0
    sc_val_clean, pvv = detector_scores(model, Sva.z0, base, c)
    det_rows, att_rows, cv_rows = [], [], []
    for eps in c["eps"]:
        _, zva0 = a0_rows(Sva, eps, tp, dev, c["rng_seed"] + 61 + int(eps * 100))      # same seed as v1
        sva0, pva0 = detector_scores(model, zva0, base, c)
        neg_v1val = pd.concat([sc_val_clean[pvv == yva], sva0[pva0 == yva]], ignore_index=True)
        thr1 = {d: float(np.quantile(neg_v1val[d], 0.95)) for d in dets}
        s5 = resolve("results/stage5") / f"seed{s}"
        a0 = np.load(s5 / f"adv_raw_constrained_A0_eps{eps:g}.npz")
        a1 = np.load(s5 / f"adv_raw_constrained_A1_eps{eps:g}.npz")
        i0 = np.array([np.flatnonzero(a0["pos"] == q)[0] for q in pos])
        i1 = np.array([np.flatnonzero(a1["pos"] == q)[0] for q in pos])
        with torch.no_grad():
            z_a0 = tp(torch.from_numpy(a0["x_adv"][i0]).to(dev)).float()
            z_a1 = tp(torch.from_numpy(a1["x_adv"][i1]).to(dev)).float()
        s_a0, p_a0 = detector_scores(model, z_a0, base, c)
        neg_v1 = pd.concat([sc_te_clean[p_te_clean == yte], s_a0[p_a0 == yte]], ignore_index=True)
        cos_thr = -thr2["cons_rollout_gxi_cos"] + c["a4_cos_margin"]
        knn_thr = thr2["knn5_benign"]
        attacks = [("A1", None)]
        attacks += [(f"A4_lam{l:g}", ("A4", l, None)) for l in c["a4_lambdas"]]
        attacks += [(f"A4norm_w{c['a4_norm_w']:g}", ("A4", 0.0, c["a4_norm_w"]))]
        attacks += [(f"A5_lam{l:g}", ("A5", l, None)) for l in c["a5_lambdas"]]
        attacks += [(f"A5norm_w{c['a5_norm_w']:g}", ("A5", 0.0, c["a5_norm_w"]))]
        for name, spec_a in attacks:
            ta = time.time()
            seed_a = c["rng_seed"] + 900 + int(eps * 100) + len(att_rows)
            if spec_a is None:
                X, z = a1["x_adv"][i1], z_a1
            elif spec_a[0] == "A4":
                X, z = a4_attack(model, Ste, tp, eps, c, spec_a[1], spec_a[2], cos_thr, dev, seed_a)
            else:
                X, z = a5_attack(model, Ste, tp, eps, c, spec_a[1], spec_a[2], knn_thr, base, dev, seed_a)
            sa, pa = detector_scores(model, z, base, c)
            valid, viol, _ = check_validity(X, Rte, feats, spec, eps, Ste.cls_rows)
            flip = pa == BENIGN
            pos_sc = sa[flip].reset_index(drop=True)
            key = dict(seed=s, eps=eps, attack=name)
            r2 = evaluate(sc_tb, pos_sc, thr2, yte[flip], classes, dets, c["n_boot"],
                          np.random.default_rng(c["rng_seed"] + 66), protocol="v2_benign_negatives", **key)
            r1 = evaluate(neg_v1, pos_sc, thr1, yte[flip], classes, dets, 0, np.random.default_rng(0),
                          protocol="v1_attack_class_negatives", **key)
            det_rows += r2 + [r for r in r1 if r["source_class"] == "all"]
            und = {d: float((flip & (sa[d].to_numpy() < thr2[d])).mean()) for d in (PRIMARY, "knn5_benign",
                                                                                   "mahalanobis_benign", "msp")}
            att_rows.append({**key, "evasion_success": float(flip.mean()), "validity": float(valid.mean()),
                             "invalid_rows": int((~valid).sum()),
                             "violations": ";".join(k for k, v in viol.items() if v.any()),
                             **{f"success_and_undetected_{d}": v for d, v in und.items()},
                             "mean_cos_flipped": float(-sa.loc[flip, "cons_rollout_gxi_cos"].mean()) if flip.any() else np.nan,
                             "cos_threshold": -thr2["cons_rollout_gxi_cos"],
                             "mean_knn_benign_flipped": float(sa.loc[flip, "knn5_benign"].mean()) if flip.any() else np.nan,
                             "knn_threshold": knn_thr, "seconds": round(time.time() - ta, 1)})
            cons_cols = [k for k in dets if k.startswith("cons_")]
            g = cv_gain(sc_tb, pos_sc, BASE_V2, cons_cols, c, c["rng_seed"])
            if g:
                cv_rows.append({**key, **g})
            log.info(f"eps {eps} {name}: success {flip.mean():.3f} valid {valid.mean():.3f} "
                     f"TPR v2 primary {[r['tpr_at_5fpr'] for r in r2 if r['detector'] == PRIMARY and r['source_class'] == 'all']} "
                     f"knn5_benign {[r['tpr_at_5fpr'] for r in r2 if r['detector'] == 'knn5_benign' and r['source_class'] == 'all']}")
    pd.DataFrame(det_rows).to_csv(out / "s6b_detection.csv", index=False)
    pd.DataFrame(att_rows).to_csv(out / "s6b_attacks.csv", index=False)
    pd.DataFrame(cv_rows).to_csv(out / "s6b_cv_gain.csv", index=False)
    pd.DataFrame(thr_rows).to_csv(out / "s6b_thresholds.csv", index=False)
    pd.DataFrame(conf).to_csv(out / "s6b_class_confound.csv", index=False)
    pd.set_option("display.width", 250)
    log.info("Attacks:\n" + pd.DataFrame(att_rows).round(3).to_string(index=False))
    log.info("Class confound:\n" + pd.DataFrame(conf).round(3).to_string(index=False))
    finish_run(run_dir, log, {"seed": s, "n_val_benign": len(sc_vb)}, t0)


if __name__ == "__main__":
    main()
