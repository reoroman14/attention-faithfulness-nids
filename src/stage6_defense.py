# Stage 6 PILOT: attention-attribution consistency as an adversarial-evasion detector, its baselines, and an
# adaptive attacker that knows the detector.
#
# Detector (per row, predicted class c): consistency between an attention explanation (rollout; attn_mean as a
# second variant) and an attention-free attribution of logit c (primary: gradient x input in the preprocessed
# space, i.e. relative to the train mean; secondary: IG with 20 steps). Both are |.| and normalised to sum 1.
# Consistency = Spearman and top-10 Jaccard. Detector score = -consistency (higher = more suspicious).
# Baselines: -max softmax probability, -logit margin (top1 - top2), Mahalanobis distance to the predicted class
# mean (train class means, shared covariance, preprocessed space), mean distance to the 5 nearest train rows of the
# predicted class.
# Thresholds: calibrated at 5% FPR on VALIDATION negatives only (Stage 3c val selection: clean rows of the attack
# classes plus their A0 perturbations at the same eps that did not flip). Evaluation on TEST rows (Stage 4 pilot
# rows): negatives = clean rows + A0 rows that did not flip; positives = A1 / A4 rows that flipped to benign.
# Adaptive attacker A4 (white-box, constrained, same projection and checker): maximise the evasion margin while
# keeping the differentiable surrogate (cosine between normalised rollout and normalised |gradient x input| of the
# benign logit, double backprop) above its own val-calibrated threshold: lambda in {1, 10, 100} with a hinge,
# plus a normalised-gradient variant (unit-norm margin and cosine gradients, weight 0.5).
# Item 5: cross-validated logistic regression on the baselines vs baselines + consistency scores.
#
# Usage:
#   .venv\Scripts\python.exe -m src.stage6_defense --set stage6.pilot=true
import copy
import json
import math
import time

import numpy as np
import pandas as pd
import torch
from scipy.stats import rankdata
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import RepeatedStratifiedKFold
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

from src import config as cfgmod
from src.config import load_config
from src.preprocess import Preprocessor
from src.stage3_explain import integrated_gradients, jaccard, load_ftt, ranking, rowwise_spearman
from src.stage4_attack import STAGE4, rollout_cls
from src.stage4_constraints import (TorchPreprocessor, check_validity, derive, freeze_mask, load_spec_csv,
                                    make_context, project)
from src.stage4b_followup import attn_vec
from src.utils import ROOT, finish_run, resolve, start_run

STAGE6 = {
    "pilot": True,
    "seeds": [0],
    "n_per_class": 20,
    "val_per_class": 100,
    "eps": [0.5, 1.0],
    "fpr": 0.05,
    "ig_steps_det": 20,
    "knn_k": 5,
    "knn_train_per_class": 20000,
    "maha_train_rows": 200000,
    "a4_lambdas": [1.0, 10.0, 100.0],
    "a4_norm_w": 0.5,
    "a4_cos_margin": 0.02,
    "n_boot": 1000,
    "cv_splits": 5,
    "cv_repeats": 5,
}
BENIGN = 0
DET_PRIMARY = "cons_rollout_gxi_spearman"


# ---------------------------------------------------------------- detector scores

def norm_abs(a):
    a = np.abs(a)
    return a / np.clip(a.sum(1, keepdims=True), 1e-30, None)


def grad_x_input(model, Z, cls, bs=512):
    # |z * d logit_c / d z| (relative to the train mean, which is 0 in the preprocessed space).
    out = []
    for i in range(0, Z.shape[0], bs):
        z = Z[i:i + bs].clone().requires_grad_(True)
        lg = model(z)
        g, = torch.autograd.grad(lg.gather(1, cls[i:i + bs, None]).sum(), z)
        out.append((z * g).detach())
    return torch.cat(out).cpu().numpy()


@torch.no_grad()
def attention_and_logits(model, Z, bs=512):
    lg, ro, me = [], [], []
    for i in range(0, Z.shape[0], bs):
        z, at = model(Z[i:i + bs], return_attn=True)
        lg.append(z), ro.append(rollout_cls(at)), me.append(attn_vec(at, "mean"))
    return torch.cat(lg), torch.cat(ro).cpu().numpy(), torch.cat(me).cpu().numpy()


class Baselines:
    # Mahalanobis (shared covariance, train class means) and kNN distance, both in the preprocessed space.
    def __init__(self, Ztr, ytr, n_classes, c, dev, rng):
        sub = rng.choice(len(Ztr), size=min(c["maha_train_rows"], len(Ztr)), replace=False)
        Zs, ys = Ztr[sub].astype(np.float64), ytr[sub]
        self.mu = np.stack([Ztr[ytr == k].astype(np.float64).mean(0) for k in range(n_classes)])
        cov = np.cov((Zs - self.mu[ys]).T)
        self.P = torch.from_numpy(np.linalg.inv(cov + 1e-6 * np.eye(cov.shape[0]))).float().to(dev)
        self.mu_t = torch.from_numpy(self.mu).float().to(dev)
        self.ref = {}
        for k in range(n_classes):
            idx = np.flatnonzero(ytr == k)
            idx = rng.choice(idx, size=min(c["knn_train_per_class"], len(idx)), replace=False)
            self.ref[k] = torch.from_numpy(Ztr[idx]).to(dev)
        self.k = c["knn_k"]

    @torch.no_grad()
    def scores(self, Z, pred):
        d = Z - self.mu_t[pred]
        maha = ((d @ self.P) * d).sum(1).clamp_min(0).sqrt()
        knn = torch.empty(Z.shape[0], device=Z.device)
        for k in pred.unique().tolist():
            m = pred == k
            D = torch.cdist(Z[m], self.ref[k])
            knn[m] = D.topk(self.k, largest=False).values.mean(1)
        return maha.cpu().numpy(), knn.cpu().numpy()


def detector_scores(model, Z, base, c):
    # Every detector score for rows Z (higher = more suspicious) and the predicted class.
    lg, ro, me = attention_and_logits(model, Z)
    pred = lg.argmax(1)
    p = torch.softmax(lg, -1)
    top2 = lg.topk(2, dim=1).values
    gxi = norm_abs(grad_x_input(model, Z, pred))
    ig, _, _ = integrated_gradients(model, Z, pred, torch.zeros(1, Z.shape[1], device=Z.device),
                                    {**c, "ig_steps": c["ig_steps_det"]})
    ig = norm_abs(ig)
    maha, knn = base.scores(Z, pred)
    s = {"msp": -p.max(1).values.cpu().numpy(), "margin": -(top2[:, 0] - top2[:, 1]).cpu().numpy(),
         "mahalanobis": maha, "knn5": knn}
    rng = np.random.default_rng(0)
    for an, A in (("rollout", norm_abs(ro)), ("attnmean", norm_abs(me))):
        for bn, B in (("gxi", gxi), ("ig20", ig)):
            s[f"cons_{an}_{bn}_spearman"] = -rowwise_spearman(A, B)
            s[f"cons_{an}_{bn}_jacc10"] = -jaccard(ranking(A, rng), ranking(B, rng), 10)
            s[f"cos_{an}_{bn}"] = -(A * B).sum(1) / np.clip(np.linalg.norm(A, axis=1) * np.linalg.norm(B, axis=1),
                                                             1e-30, None)
    return pd.DataFrame(s), pred.cpu().numpy()


# ---------------------------------------------------------------- data helpers

class Rows:
    # Raw rows plus everything the constrained attacks need.
    pass


def make_rows(R, y, classes, feats, spec, floors, tp, dev):
    S = Rows()
    S.R, S.y_np = R, y
    S.cls_rows = [classes[k] for k in y]
    _, _, cf = check_validity(R, R, feats, spec, 0.0, S.cls_rows)
    S.frozen = freeze_mask(R, feats, S.cls_rows, cf)
    S.y = torch.from_numpy(y).to(dev)
    with torch.no_grad():
        S.z0 = tp(torch.from_numpy(R).to(dev)).float()
    S.feats, S.floors, S.spec = feats, floors, spec
    return S


def a0_rows(S, eps, tp, dev, seed):
    ctx = make_context(S.R, S.feats, S.floors, eps, S.frozen, dev)
    g = torch.Generator(device=dev).manual_seed(seed)
    u = ctx.lo + (ctx.hi - ctx.lo) * torch.rand(ctx.lo.shape, generator=g, device=dev, dtype=torch.float64)
    x, _ = project(u, ctx)
    with torch.no_grad():
        return x.cpu().numpy(), tp(x).float()


# ---------------------------------------------------------------- adaptive attacker A4

def a4_attack(model, S, tp, eps, c, lam, norm_w, cos_thr, dev, seed):
    # Constrained PGD; objective clamp(margin_benign, kappa) - lam * relu(cos_thr - cos) (hinge) or, for the
    # normalised variant, unit-norm gradients of clamp(margin) and cos mixed with weights (1 - w, w).
    # cos = cosine(normalised rollout, normalised |z * d logit_benign / d z|), computed with double backprop.
    ctx = make_context(S.R, S.feats, S.floors, eps, S.frozen, dev)
    gen = torch.Generator(device=dev).manual_seed(seed)
    kappa = c["a3_kappa"]

    def terms(xv):
        z = tp(derive(xv, ctx)).float()
        lg, attns = model(z, return_attn=True)
        roll = rollout_cls(attns)
        gb, = torch.autograd.grad(lg[:, BENIGN].sum(), z, create_graph=True)
        gxi = (z * gb).abs()
        A = roll / roll.sum(1, keepdim=True)
        B = gxi / gxi.sum(1, keepdim=True).clamp_min(1e-12)
        cos = (A * B).sum(1) / (A.norm(dim=1) * B.norm(dim=1)).clamp_min(1e-12)
        e = lg[:, BENIGN] - lg[:, 1:].amax(1)
        return e, cos

    best_key = torch.full((S.R.shape[0],), float("-inf"), dtype=torch.float64, device=dev)
    best_x = None
    for run in range(1 + c["restarts"]):
        delta = torch.zeros_like(ctx.lo) if run == 0 else \
            ctx.lo + (ctx.hi - ctx.lo) * torch.rand(ctx.lo.shape, generator=gen, device=dev, dtype=torch.float64)
        for step in range(c["steps"] + 1):
            x, _ = project(delta, ctx)
            xv = x.detach().requires_grad_(True)
            e, cos = terms(xv)
            hinge = torch.relu(cos_thr - cos)
            obj = torch.clamp(e, max=kappa) - lam * hinge
            succ = e > 0
            key = (1e7 * (succ & (cos >= cos_thr)) + 1e6 * succ).double() + obj.detach().double()
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
                g2, = torch.autograd.grad(cos.sum(), xv)
                g1, g2 = g1.double() * ctx.budget, g2.double() * ctx.budget
                g = (1 - norm_w) * g1 / g1.norm(dim=1, keepdim=True).clamp_min(1e-30) + \
                    norm_w * g2 / g2.norm(dim=1, keepdim=True).clamp_min(1e-30)
            a_t = c["alpha0"] * 0.5 * (1 + math.cos(math.pi * step / c["steps"]))
            delta = torch.minimum(torch.maximum(delta + a_t * torch.sign(g), ctx.lo), ctx.hi)
    with torch.no_grad():
        return best_x.cpu().numpy(), tp(best_x).float()


# ---------------------------------------------------------------- metrics

def tpr_at(thr, pos):
    return float((pos >= thr).mean()) if len(pos) else np.nan


def boot_ci(neg, pos, thr, n, rng):
    au, tp = [], []
    for _ in range(n):
        nb, pb = rng.choice(neg, len(neg)), rng.choice(pos, len(pos))
        au.append(roc_auc_score(np.r_[np.zeros(len(nb)), np.ones(len(pb))], np.r_[nb, pb]))
        tp.append((pb >= thr).mean())
    return np.percentile(au, [2.5, 97.5]), np.percentile(tp, [2.5, 97.5])


def evaluate_detectors(neg, pos, thr, cls_neg, cls_pos, classes, n_boot, rng, **key):
    # AUROC and TPR at the val-calibrated threshold, pooled (with bootstrap CI) and per class (macro).
    rows = []
    for det in thr:
        n, p = neg[det].to_numpy(), pos[det].to_numpy()
        if len(p) == 0 or len(n) == 0:
            continue
        au = roc_auc_score(np.r_[np.zeros(len(n)), np.ones(len(p))], np.r_[n, p])
        (alo, ahi), (tlo, thi) = boot_ci(n, p, thr[det], n_boot, rng)
        per = []
        for k in sorted(set(cls_pos)):
            nk, pk = n[cls_neg == k], p[cls_pos == k]
            if len(nk) and len(pk):
                per.append((classes[k], roc_auc_score(np.r_[np.zeros(len(nk)), np.ones(len(pk))], np.r_[nk, pk]),
                            tpr_at(thr[det], pk)))
        rows.append({**key, "detector": det, "n_neg": len(n), "n_pos": len(p), "auroc": au, "auroc_lo": alo,
                     "auroc_hi": ahi, "tpr_at_5fpr": tpr_at(thr[det], p), "tpr_lo": tlo, "tpr_hi": thi,
                     "test_fpr": float((n >= thr[det]).mean()),
                     "auroc_macro_class": np.mean([a for _, a, _ in per]) if per else np.nan,
                     "tpr_macro_class": np.mean([t for _, _, t in per]) if per else np.nan,
                     "n_classes": len(per), "per_class": "; ".join(f"{cn} {a:.2f}/{t:.2f}" for cn, a, t in per)})
    return rows


def cv_gain(neg, pos, base_cols, cons_cols, c, seed):
    # Cross-validated AUROC of logistic regression on baselines vs baselines + consistency scores.
    X = pd.concat([neg, pos], ignore_index=True)
    yv = np.r_[np.zeros(len(neg)), np.ones(len(pos))]
    if len(pos) < c["cv_splits"] or len(neg) < c["cv_splits"]:
        return None
    cv = RepeatedStratifiedKFold(n_splits=c["cv_splits"], n_repeats=c["cv_repeats"], random_state=seed)
    res = {"base": [], "base_plus_cons": [], "cons_only": []}
    for tr, te in cv.split(X, yv):
        for name, cols in (("base", base_cols), ("base_plus_cons", base_cols + cons_cols), ("cons_only", cons_cols)):
            m = make_pipeline(StandardScaler(), LogisticRegression(max_iter=2000))
            m.fit(X.iloc[tr][cols], yv[tr])
            res[name].append(roc_auc_score(yv[te], m.predict_proba(X.iloc[te][cols])[:, 1]))
    d = np.array(res["base_plus_cons"]) - np.array(res["base"])
    return {**{f"auroc_{k}": float(np.mean(v)) for k, v in res.items()},
            **{f"auroc_{k}_sd": float(np.std(v)) for k, v in res.items()},
            "gain_mean": float(d.mean()), "gain_sd": float(d.std()), "gain_share_folds_positive": float((d > 0).mean())}


# ---------------------------------------------------------------- main

def main():
    cfgmod.DEFAULTS.setdefault("stage6", {**copy.deepcopy(STAGE4), **copy.deepcopy(STAGE6)})
    cfg, overrides = load_config("Stage 6 pilot: consistency detector and adaptive attacker.")
    c = cfg["stage6"]
    out = resolve("results/stage6/pilot")
    out.mkdir(parents=True, exist_ok=True)
    t0 = time.time()
    run_dir, log = start_run("stage6_pilot", cfg, overrides)
    dev = torch.device("cuda")
    s1, s2, s3 = resolve("results/stage1"), resolve("results/stage2"), resolve("results/stage3")
    feats = json.loads((s1 / "features.json").read_text(encoding="utf-8"))
    lm = json.loads((s1 / "label_map.json").read_text(encoding="utf-8"))
    classes = [k for k, _ in sorted(lm.items(), key=lambda kv: kv[1])]
    pp = Preprocessor.load(s2 / "preprocess.json")
    tp = TorchPreprocessor(pp, dev)
    spec = load_spec_csv()
    floors = spec.set_index("feature")["zero_floor_scale"].astype(float).to_dict()
    rng = np.random.default_rng(c["rng_seed"] + 6000)
    # Test rows: the Stage 4 pilot rows (attack classes, rank_in_class < n_per_class).
    sel = pd.read_parquet(s3 / "selection.parquet")
    pos = np.flatnonzero((sel["class"] != "benign").to_numpy() & (sel["rank_in_class"] < c["n_per_class"]).to_numpy())
    pos = pos[np.lexsort((sel["rank_in_class"].to_numpy()[pos], sel["label_id"].to_numpy()[pos]))]
    test = pd.read_parquet(s1 / "test.parquet", columns=feats)
    Rte = test.iloc[sel["test_row"].to_numpy()[pos]].to_numpy(np.float64).copy()
    yte = sel["label_id"].to_numpy()[pos].astype(np.int64)
    del test
    # Validation rows: Stage 3c val selection, attack classes, up to val_per_class each (fixed rng).
    vs = pd.read_parquet(resolve("results/stage3c") / "val_selection.parquet")
    vs = vs[vs.label_id != BENIGN]
    vpick = np.concatenate([rng.choice(g.index.to_numpy(), size=min(c["val_per_class"], len(g)), replace=False)
                            for _, g in vs.groupby("label_id")])
    val = pd.read_parquet(s1 / "val.parquet", columns=feats)
    Rva = val.iloc[vs.loc[vpick, "val_row"].to_numpy()].to_numpy(np.float64).copy()
    yva = vs.loc[vpick, "label_id"].to_numpy().astype(np.int64)
    del val
    train = pd.read_parquet(s1 / "train.parquet", columns=feats + ["label_id"])
    Ztr, ytr = pp.transform(train), train["label_id"].to_numpy()
    del train
    base = Baselines(Ztr, ytr, len(classes), c, dev, rng)
    del Ztr
    Ste = make_rows(Rte, yte, classes, feats, spec, floors, tp, dev)
    Sva = make_rows(Rva, yva, classes, feats, spec, floors, tp, dev)
    log.info(f"test rows {len(yte)}, val rows {len(yva)}")
    all_rows, cv_rows, a4_rows, cal_rows = [], [], [], []
    for s in c["seeds"]:
        model, _ = load_ftt(s2 / f"ftt_seed{s}.pt", dev)
        for p in model.parameters():
            p.requires_grad_(False)
        with torch.no_grad():
            okv = (model(Sva.z0).argmax(1) == Sva.y).float().mean().item()
        log.info(f"seed {s}: val rows predicted as label {okv:.3f}")
        sc_val_clean, pv = detector_scores(model, Sva.z0, base, c)
        sc_te_clean, pt = detector_scores(model, Ste.z0, base, c)
        for eps in c["eps"]:
            # Calibration on val negatives: clean + non-flipped A0 at this eps.
            _, zva0 = a0_rows(Sva, eps, tp, dev, c["rng_seed"] + 61 + int(eps * 100))
            sva0, pva0 = detector_scores(model, zva0, base, c)
            neg_val = pd.concat([sc_val_clean[pv == yva], sva0[pva0 == yva]], ignore_index=True)
            thr = {d: float(np.quantile(neg_val[d], 1 - c["fpr"])) for d in neg_val.columns}
            cal_rows += [{"seed": s, "eps": eps, "detector": d, "threshold": v, "n_val_neg": len(neg_val)}
                         for d, v in thr.items()]
            # Test negatives: clean + non-flipped A0 (Stage 5 rows for these positions, same eps).
            s5 = resolve("results/stage5") / f"seed{s}"
            a0 = np.load(s5 / f"adv_raw_constrained_A0_eps{eps:g}.npz")
            a1 = np.load(s5 / f"adv_raw_constrained_A1_eps{eps:g}.npz")
            idx0 = np.searchsorted(a0["pos"], pos) if np.all(np.diff(a0["pos"]) > 0) else \
                np.array([np.flatnonzero(a0["pos"] == q)[0] for q in pos])
            idx1 = np.array([np.flatnonzero(a1["pos"] == q)[0] for q in pos])
            assert np.array_equal(a0["pos"][idx0], pos) and np.array_equal(a1["pos"][idx1], pos)
            with torch.no_grad():
                z_a0 = tp(torch.from_numpy(a0["x_adv"][idx0]).to(dev)).float()
                z_a1 = tp(torch.from_numpy(a1["x_adv"][idx1]).to(dev)).float()
            s_a0, p_a0 = detector_scores(model, z_a0, base, c)
            s_a1, p_a1 = detector_scores(model, z_a1, base, c)
            negm0, negma = pt == yte, p_a0 == yte
            neg = pd.concat([sc_te_clean[negm0], s_a0[negma]], ignore_index=True)
            cls_neg = np.r_[yte[negm0], yte[negma]]
            # Validity of the reused rows (independent checker).
            v1, _, _ = check_validity(a1["x_adv"][idx1], Rte, feats, spec, eps, Ste.cls_rows)
            flip1 = p_a1 == BENIGN
            log.info(f"seed {s} eps {eps}: A1 flipped {flip1.mean():.3f}, valid {v1.mean():.3f}")
            # Cosine-surrogate threshold for A4 (val calibrated, same 5% FPR rule on cos_rollout_gxi).
            cos_thr = -thr["cos_rollout_gxi"] + c["a4_cos_margin"]
            attacks = [("A1", None, None, a1["x_adv"][idx1], z_a1, s_a1, p_a1)]
            for lam in c["a4_lambdas"]:
                attacks.append((f"A4_lam{lam:g}", lam, None))
            attacks.append((f"A4norm_w{c['a4_norm_w']:g}", 0.0, c["a4_norm_w"]))
            for at in attacks:
                name = at[0]
                ta = time.time()
                if name == "A1":
                    X, z, sa, pa = at[3], at[4], at[5], at[6]
                else:
                    X, z = a4_attack(model, Ste, tp, eps, c, at[1], at[2], cos_thr, dev,
                                     c["rng_seed"] + 600 + int(eps * 100) + len(a4_rows))
                    sa, pa = detector_scores(model, z, base, c)
                valid, viol, _ = check_validity(X, Rte, feats, spec, eps, Ste.cls_rows)
                flip = pa == BENIGN
                pos_sc = sa[flip].reset_index(drop=True)
                ev = evaluate_detectors(neg, pos_sc, thr, cls_neg, yte[flip], classes, c["n_boot"],
                                        np.random.default_rng(c["rng_seed"] + 66), seed=s, eps=eps, attack=name)
                all_rows += ev
                undetected = (pos_sc[DET_PRIMARY].to_numpy() < thr[DET_PRIMARY]).mean() if flip.any() else np.nan
                a4_rows.append({"seed": s, "eps": eps, "attack": name, "evasion_success": flip.mean(),
                                "validity": valid.mean(), "invalid_rows": int((~valid).sum()),
                                "violations": ";".join(k for k, v in viol.items() if v.any()),
                                "success_and_undetected_primary": flip.mean() * undetected if flip.any() else 0.0,
                                "mean_cos_flipped": float(-sa.loc[flip, "cos_rollout_gxi"].mean()) if flip.any() else np.nan,
                                "cos_threshold": cos_thr - c["a4_cos_margin"], "seconds": round(time.time() - ta, 1)})
                base_cols = ["msp", "margin", "mahalanobis", "knn5"]
                cons_cols = [k for k in sa.columns if k.startswith("cons_")]
                g = cv_gain(neg, pos_sc, base_cols, cons_cols, c, c["rng_seed"])
                if g:
                    cv_rows.append({"seed": s, "eps": eps, "attack": name, **g})
                log.info(f"seed {s} eps {eps} {name}: success {flip.mean():.3f} valid {valid.mean():.3f} "
                         f"TPR primary {[r['tpr_at_5fpr'] for r in ev if r['detector'] == DET_PRIMARY]}")
        del model
        torch.cuda.empty_cache()
    det = pd.DataFrame(all_rows)
    det.to_csv(out / "s6_detection.csv", index=False)
    pd.DataFrame(a4_rows).to_csv(out / "s6_attacks.csv", index=False)
    pd.DataFrame(cv_rows).to_csv(out / "s6_cv_gain.csv", index=False)
    pd.DataFrame(cal_rows).to_csv(out / "s6_thresholds.csv", index=False)
    pd.set_option("display.width", 250)
    pd.set_option("display.max_columns", 30)
    show = det[det.detector.isin(["msp", "margin", "mahalanobis", "knn5", DET_PRIMARY, "cons_rollout_gxi_jacc10",
                                  "cons_attnmean_gxi_spearman", "cons_rollout_ig20_spearman"])]
    log.info("Detection:\n" + show[["eps", "attack", "detector", "n_neg", "n_pos", "auroc", "auroc_lo", "auroc_hi",
                                    "tpr_at_5fpr", "tpr_lo", "tpr_hi", "test_fpr"]].round(3).to_string(index=False))
    log.info("Attacks:\n" + pd.DataFrame(a4_rows).round(3).to_string(index=False))
    log.info("CV gain:\n" + pd.DataFrame(cv_rows).round(3).to_string(index=False))
    finish_run(run_dir, log, {"seeds": c["seeds"]}, t0)


if __name__ == "__main__":
    main()
