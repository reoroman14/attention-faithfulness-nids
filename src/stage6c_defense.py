# Stage 6 PILOT v3: a learned defender D and a fully adaptive attacker A6 that targets D.
#
# Defender D (per eps): standardised logistic regression on [msp, margin, mahalanobis_benign, knn5_benign]
# (score convention of src/stage6b_defense.py: higher = more suspicious), trained ONLY on validation data:
#   negatives  val benign rows predicted benign (a 20,000-row training sample)
#   positives  constrained A1 rows generated on VALIDATION source rows (the Stage 3c val attack-class rows used in
#              v1/v2), same eps, valid by the independent checker, flipped to benign
#   threshold  5% FPR on the HELD-OUT val benign rows (all val benign predicted benign minus the training sample)
# Test rows are never used for training or calibration. Test negatives = the same 1,000 genuine benign test rows as
# v2; positives = adversarial Stage 4 pilot rows predicted benign.
# Attacker A6 (white-box, knows D; constrained, same projection and independent checker):
#   maximise min(margin_benign, kappa = 10) - lambda * relu(D_logit - thr_D + 0.5), lambda in {1, 10, 100},
#   plus a normalised-gradient variant (unit-norm gradients of min(margin, 10) and -D_logit, weight 0.5).
#   D is differentiable end to end: softmax, top-2 logits, Mahalanobis (benign mean, shared covariance) and kNN-5 to
#   benign train rows (cdist + topk).
#   Ablations A6_msp and A6_knn: the same attack against a single standardised baseline score (lambda 10).
# A5 (kNN-only attacker of v2, lambda 100) is regenerated with the v2 code and seeds (v2 did not store its rows).
# Consistency is reported as an EXPLORATORY detector: its direction is chosen on validation (val benign vs A1 on
# val source rows), its threshold is the 5% FPR point on val benign.
#
# Usage:
#   .venv\Scripts\python.exe -m src.stage6c_defense
import copy
import json
import math
import time

import numpy as np
import pandas as pd
import torch
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler

from src import config as cfgmod
from src.config import load_config
from src.preprocess import Preprocessor
from src.stage3_explain import load_ftt
from src.stage4_attack import STAGE4, make_objective, pgd, rollout_cls
from src.stage4_constraints import TorchPreprocessor, check_validity, derive, load_spec_csv, make_context, project
from src.stage6_defense import BENIGN, STAGE6, Baselines, cv_gain, make_rows
from src.stage6b_defense import STAGE6B, a5_attack, auroc, boot, detector_scores
from src.utils import finish_run, resolve, start_run

STAGE6C = {
    "kappa_a6": 10.0,
    "a6_lambdas": [1.0, 10.0, 100.0],
    "a6_norm_w": 0.5,
    "a6_safety": 0.5,
    "a6_single_lambda": 10.0,
    "d_train_neg": 20000,
    "val_cons_neg": 5000,
}
FEATS_D = ["msp", "margin", "mahalanobis_benign", "knn5_benign"]


# ---------------------------------------------------------------- differentiable baseline features

def feats_torch(lg, z, base):
    # Same definitions as stage6b detector_scores, differentiable: (B, 4) in FEATS_D order.
    p = torch.softmax(lg, -1)
    top2 = lg.topk(2, dim=1).values
    d = z - base.mu_t[BENIGN]
    maha = ((d @ base.P) * d).sum(1).clamp_min(1e-12).sqrt()
    knn = torch.cdist(z, base.ref[BENIGN]).topk(base.k, largest=False).values.mean(1)
    return torch.stack([-p.max(1).values, -(top2[:, 0] - top2[:, 1]), maha, knn], 1)


@torch.no_grad()
def feats_np(model, Z, base, bs=4096):
    out = []
    for i in range(0, Z.shape[0], bs):
        z = Z[i:i + bs]
        out.append(feats_torch(model(z), z, base).cpu().numpy())
    return pd.DataFrame(np.concatenate(out), columns=FEATS_D)


class Defender:
    # Logistic regression on standardised features; decision = logit; also usable on torch tensors.
    def __init__(self, Xneg, Xpos, Xheld, fpr, dev):
        X = np.r_[Xneg, Xpos]
        y = np.r_[np.zeros(len(Xneg)), np.ones(len(Xpos))]
        self.sc = StandardScaler().fit(X)
        self.lr = LogisticRegression(max_iter=5000, class_weight="balanced").fit(self.sc.transform(X), y)
        self.thr = float(np.quantile(self.decision(Xheld), 1 - fpr))
        t = lambda a: torch.tensor(a, dtype=torch.float32, device=dev)
        self.mu, self.sd = t(self.sc.mean_), t(self.sc.scale_)
        self.w, self.b = t(self.lr.coef_[0]), float(self.lr.intercept_[0])

    def decision(self, X):
        return self.lr.decision_function(self.sc.transform(np.asarray(X)))

    def torch_decision(self, F):
        return ((F - self.mu) / self.sd) @ self.w + self.b


# ---------------------------------------------------------------- attacks

def a1_on(model, S, tp, eps, c, dev, seed):
    # Constrained A1 (stage4_attack objective and PGD) on arbitrary source rows.
    ctx = make_context(S.R, S.feats, S.floors, eps, S.frozen, dev)

    def fwd(xv):
        z, attns = model(tp(derive(xv, ctx)).float(), return_attn=True)
        return z, rollout_cls(attns)

    obj = make_objective("A1", S.y, S.y, None, None, c, 0.0)
    x = pgd(fwd, lambda d: project(d, ctx)[0], ctx.lo, ctx.hi, ctx.budget, obj, c["steps"], c["alpha0"], c["restarts"],
            torch.Generator(device=dev).manual_seed(seed))
    with torch.no_grad():
        return x.cpu().numpy(), tp(x).float()


def a6_attack(model, S, tp, eps, c, lam, norm_w, score_fn, thr, dev, seed):
    # maximise min(e, kappa) - lam * relu(score - thr + safety); normalised variant mixes unit-norm gradients of
    # min(e, kappa) and -score. score_fn(lg, z) -> (B,) differentiable detector score (higher = suspicious).
    ctx = make_context(S.R, S.feats, S.floors, eps, S.frozen, dev)
    gen = torch.Generator(device=dev).manual_seed(seed)
    kappa, safety = c["kappa_a6"], c["a6_safety"]

    def terms(xv):
        z = tp(derive(xv, ctx)).float()
        lg = model(z)
        return lg[:, BENIGN] - lg[:, 1:].amax(1), score_fn(lg, z)

    best_key = torch.full((S.R.shape[0],), float("-inf"), dtype=torch.float64, device=dev)
    best_x = None
    for run in range(1 + c["restarts"]):
        delta = torch.zeros_like(ctx.lo) if run == 0 else \
            ctx.lo + (ctx.hi - ctx.lo) * torch.rand(ctx.lo.shape, generator=gen, device=dev, dtype=torch.float64)
        for step in range(c["steps"] + 1):
            x, _ = project(delta, ctx)
            xv = x.detach().requires_grad_(True)
            e, s = terms(xv)
            obj = torch.clamp(e, max=kappa) - lam * torch.relu(s - thr + safety)
            succ = e > 0
            key = (1e7 * (succ & (s < thr)) + 1e6 * succ).double() + obj.detach().double()
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
                g2, = torch.autograd.grad((-s).sum(), xv)
                g1, g2 = g1.double() * ctx.budget, g2.double() * ctx.budget
                g = (1 - norm_w) * g1 / g1.norm(dim=1, keepdim=True).clamp_min(1e-30) + \
                    norm_w * g2 / g2.norm(dim=1, keepdim=True).clamp_min(1e-30)
            a_t = c["alpha0"] * 0.5 * (1 + math.cos(math.pi * step / c["steps"]))
            delta = torch.minimum(torch.maximum(delta + a_t * torch.sign(g), ctx.lo), ctx.hi)
    with torch.no_grad():
        return best_x.cpu().numpy(), tp(best_x).float()


# ---------------------------------------------------------------- main

def main():
    cfgmod.DEFAULTS.setdefault("stage6c", {**copy.deepcopy(STAGE4), **copy.deepcopy(STAGE6), **copy.deepcopy(STAGE6B),
                                           **copy.deepcopy(STAGE6C)})
    cfg, overrides = load_config("Stage 6 pilot v3: learned defender and adaptive attacker.")
    c = cfg["stage6c"]
    out = resolve("results/stage6/pilot_v3")
    out.mkdir(parents=True, exist_ok=True)
    t0 = time.time()
    run_dir, log = start_run("stage6c_pilot_v3", cfg, overrides)
    dev = torch.device("cuda")
    s1, s2, s3 = resolve("results/stage1"), resolve("results/stage2"), resolve("results/stage3")
    feats = json.loads((s1 / "features.json").read_text(encoding="utf-8"))
    lm = json.loads((s1 / "label_map.json").read_text(encoding="utf-8"))
    classes = [k for k, _ in sorted(lm.items(), key=lambda kv: kv[1])]
    pp = Preprocessor.load(s2 / "preprocess.json")
    tp = TorchPreprocessor(pp, dev)
    spec = load_spec_csv()
    floors = spec.set_index("feature")["zero_floor_scale"].astype(float).to_dict()
    # ---- same row selection and random stream as v1 / v2 (val sample, baselines, test benign negatives)
    rng = np.random.default_rng(c["rng_seed"] + 6000)
    sel = pd.read_parquet(s3 / "selection.parquet")
    pos = np.flatnonzero((sel["class"] != "benign").to_numpy() & (sel["rank_in_class"] < c["n_per_class"]).to_numpy())
    pos = pos[np.lexsort((sel["rank_in_class"].to_numpy()[pos], sel["label_id"].to_numpy()[pos]))]
    test = pd.read_parquet(s1 / "test.parquet", columns=feats + ["label_id"])
    Rte = test.iloc[sel["test_row"].to_numpy()[pos]][feats].to_numpy(np.float64).copy()
    yte = sel["label_id"].to_numpy()[pos].astype(np.int64)
    Zben_te_all = test.iloc[np.flatnonzero(test["label_id"].to_numpy() == BENIGN)][feats]
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
    model, _ = load_ftt(s2 / f"ftt_seed{c['seeds'][0]}.pt", dev)
    for p in model.parameters():
        p.requires_grad_(False)
    s = c["seeds"][0]
    with torch.no_grad():
        pv = torch.cat([model(Zben_va[i:i + 8192]).argmax(1) for i in range(0, len(Zben_va), 8192)])
    Zben_va = Zben_va[pv == BENIGN]
    Zte_b = torch.from_numpy(pp.transform(Zben_te_all)).to(dev)
    with torch.no_grad():
        pt = torch.cat([model(Zte_b[i:i + 8192]).argmax(1) for i in range(0, len(Zte_b), 8192)]).cpu().numpy()
    pick = rng.choice(np.flatnonzero(pt == BENIGN), size=c["n_test_benign"], replace=False)   # identical to v2
    Zte_neg = Zte_b[pick]
    del Zte_b
    # ---- baseline features and single-baseline thresholds (all val benign predicted benign, as v2)
    Fvb = feats_np(model, Zben_va, base)
    thr_single = {f: float(np.quantile(Fvb[f], 0.95)) for f in FEATS_D}
    mu_single = {f: float(Fvb[f].mean()) for f in FEATS_D}
    sd_single = {f: float(Fvb[f].std()) for f in FEATS_D}
    Fneg_te = feats_np(model, Zte_neg, base)
    rng_d = np.random.default_rng(c["rng_seed"] + 7000)
    perm = rng_d.permutation(len(Fvb))
    tr_idx, held_idx = perm[:c["d_train_neg"]], perm[c["d_train_neg"]:]
    # Validation benign sample for the exploratory consistency direction / threshold.
    cons_idx = held_idx[:c["val_cons_neg"]]
    sc_vb_cons, _ = detector_scores(model, Zben_va[torch.from_numpy(cons_idx).to(dev)], base, c)
    sc_te_neg, _ = detector_scores(model, Zte_neg, base, c)
    cons_cols = [k for k in sc_te_neg.columns if k.startswith("cons_")]
    att_rows, det_rows, d_rows, cons_rows, cv_rows = [], [], [], [], []
    for eps in c["eps"]:
        # ---- A1 on validation source rows -> D's positives
        ta = time.time()
        Xva1, zva1 = a1_on(model, Sva, tp, eps, c, dev, c["rng_seed"] + 7100 + int(eps * 100))
        vval, _, _ = check_validity(Xva1, Rva, feats, spec, eps, Sva.cls_rows)
        with torch.no_grad():
            flip_va = (model(zva1).argmax(1) == BENIGN).cpu().numpy() & vval
        Fva_pos = feats_np(model, zva1[torch.from_numpy(np.flatnonzero(flip_va)).to(dev)], base)
        D = Defender(Fvb.iloc[tr_idx][FEATS_D].to_numpy(), Fva_pos[FEATS_D].to_numpy(),
                     Fvb.iloc[held_idx][FEATS_D].to_numpy(), c["fpr"], dev)
        val_auc = auroc(D.decision(Fvb.iloc[held_idx].to_numpy()), D.decision(Fva_pos.to_numpy()))
        d_rows.append({"eps": eps, "val_source_rows": len(yva), "val_A1_valid": float(vval.mean()),
                       "val_A1_invalid_rows": int((~vval).sum()), "val_positives": int(flip_va.sum()),
                       "val_neg_train": len(tr_idx), "val_neg_heldout": len(held_idx), "thr_D_logit": D.thr,
                       "val_heldout_auroc": val_auc,
                       **{f"coef_{f}": float(w) for f, w in zip(FEATS_D, D.lr.coef_[0])},
                       "intercept": float(D.lr.intercept_[0]), "seconds": round(time.time() - ta, 1)})
        log.info(f"eps {eps}: D trained on {len(tr_idx)} val benign + {int(flip_va.sum())} val A1 positives; "
                 f"held-out val AUROC {val_auc:.3f}; threshold {D.thr:.3f}")
        # Exploratory consistency: direction and threshold from validation only.
        sc_va_pos, _ = detector_scores(model, zva1[torch.from_numpy(np.flatnonzero(flip_va)).to(dev)], base, c)
        direction = {k: (1.0 if auroc(sc_vb_cons[k].to_numpy(), sc_va_pos[k].to_numpy()) >= 0.5 else -1.0)
                     for k in cons_cols}
        thr_cons = {k: float(np.quantile(direction[k] * sc_vb_cons[k], 0.95)) for k in cons_cols}
        val_auc_cons = {k: auroc(direction[k] * sc_vb_cons[k].to_numpy(), direction[k] * sc_va_pos[k].to_numpy())
                        for k in cons_cols}
        # ---- attacks on the test pilot rows
        a1 = np.load(resolve("results/stage5") / f"seed{s}" / f"adv_raw_constrained_A1_eps{eps:g}.npz")
        i1 = np.array([np.flatnonzero(a1["pos"] == q)[0] for q in pos])
        score_D = lambda lg, z: D.torch_decision(feats_torch(lg, z, base))
        j = {f: FEATS_D.index(f) for f in FEATS_D}
        single = lambda f: (lambda lg, z: (feats_torch(lg, z, base)[:, j[f]] - mu_single[f]) / sd_single[f])
        thr_std = {f: (thr_single[f] - mu_single[f]) / sd_single[f] for f in FEATS_D}
        attacks = [("A1", None),
                   ("A5_lam100", ("A5", 100.0, None, c["rng_seed"] + 900 + int(eps * 100) + (7 if eps == 0.5 else 16)))]
        attacks += [(f"A6_lam{l:g}", ("A6", l, None, score_D, D.thr)) for l in c["a6_lambdas"]]
        attacks += [(f"A6norm_w{c['a6_norm_w']:g}", ("A6", 0.0, c["a6_norm_w"], score_D, D.thr))]
        attacks += [("A6_msp_only", ("A6", c["a6_single_lambda"], None, single("msp"), thr_std["msp"])),
                    ("A6_knn_only", ("A6", c["a6_single_lambda"], None, single("knn5_benign"), thr_std["knn5_benign"]))]
        for name, sp in attacks:
            ta = time.time()
            if sp is None:
                X = a1["x_adv"][i1]
                with torch.no_grad():
                    z = tp(torch.from_numpy(X).to(dev)).float()
            elif sp[0] == "A5":
                X, z = a5_attack(model, Ste, tp, eps, c, sp[1], sp[2], thr_single["knn5_benign"], base, dev, sp[3])
            else:
                X, z = a6_attack(model, Ste, tp, eps, c, sp[1], sp[2], sp[3], sp[4], dev,
                                 c["rng_seed"] + 7300 + int(eps * 100) + len(att_rows))
            valid, viol, _ = check_validity(X, Rte, feats, spec, eps, Ste.cls_rows)
            with torch.no_grad():
                lg = model(z)
            flip = (lg.argmax(1) == BENIGN).cpu().numpy()
            margin = (lg[:, BENIGN] - lg[:, 1:].amax(1)).cpu().numpy()
            Fp = feats_np(model, z, base)
            dD = D.decision(Fp.to_numpy())
            und = {"D": flip & (dD < D.thr), **{f: flip & (Fp[f].to_numpy() < thr_single[f]) for f in FEATS_D}}
            best_single = min(FEATS_D, key=lambda f: und[f].mean())
            a1_succ = [r["evasion_success"] for r in att_rows if r["eps"] == eps and r["attack"] == "A1"]
            att_rows.append({"eps": eps, "attack": name, "evasion_success": float(flip.mean()),
                             "drop_vs_A1": (a1_succ[0] - flip.mean()) if a1_succ else 0.0,
                             "succ_undet_D": float(und["D"].mean()),
                             **{f"succ_undet_{f}": float(und[f].mean()) for f in FEATS_D},
                             "best_single_baseline": best_single,
                             "succ_undet_best_single": float(und[best_single].mean()),
                             "median_margin_flipped": float(np.median(margin[flip])) if flip.any() else np.nan,
                             "validity": float(valid.mean()), "invalid_rows": int((~valid).sum()),
                             "violations": ";".join(k for k, v in viol.items() if v.any()),
                             "seconds": round(time.time() - ta, 1)})
            # D detection on test: negatives = 1,000 benign test rows, positives = flipped rows.
            neg_d, pos_d = D.decision(Fneg_te.to_numpy()), dD[flip]
            (alo, ahi), (tlo, thi) = boot(neg_d, pos_d, D.thr, c["n_boot"], np.random.default_rng(c["rng_seed"] + 66))
            det_rows.append({"eps": eps, "attack": name, "detector": "D (logistic, 4 baselines)", "n_neg": len(neg_d),
                             "n_pos": int(flip.sum()), "auroc": auroc(neg_d, pos_d), "auroc_lo": alo, "auroc_hi": ahi,
                             "tpr_at_5fpr": float((pos_d >= D.thr).mean()) if flip.any() else np.nan,
                             "tpr_lo": tlo, "tpr_hi": thi, "test_fpr": float((neg_d >= D.thr).mean())})
            for f in FEATS_D:
                n_, p_ = Fneg_te[f].to_numpy(), Fp[f].to_numpy()[flip]
                (alo, ahi), (tlo, thi) = boot(n_, p_, thr_single[f], c["n_boot"], np.random.default_rng(c["rng_seed"] + 66))
                det_rows.append({"eps": eps, "attack": name, "detector": f, "n_neg": len(n_), "n_pos": int(flip.sum()),
                                 "auroc": auroc(n_, p_), "auroc_lo": alo, "auroc_hi": ahi,
                                 "tpr_at_5fpr": float((p_ >= thr_single[f]).mean()) if flip.any() else np.nan,
                                 "tpr_lo": tlo, "tpr_hi": thi, "test_fpr": float((n_ >= thr_single[f]).mean())})
            # Exploratory consistency (A1 and A6 family only).
            if (name == "A1" or name.startswith("A6")) and flip.sum() >= 2:
                sc_p, _ = detector_scores(model, z[torch.from_numpy(np.flatnonzero(flip)).to(dev)], base, c)
                for k in cons_cols:
                    n_, p_ = direction[k] * sc_te_neg[k].to_numpy(), direction[k] * sc_p[k].to_numpy()
                    cons_rows.append({"eps": eps, "attack": name, "detector": k, "direction_from_val": direction[k],
                                      "val_auroc": val_auc_cons[k], "test_auroc": auroc(n_, p_),
                                      "test_tpr_at_5fpr": float((p_ >= thr_cons[k]).mean()) if len(p_) else np.nan,
                                      "test_fpr": float((n_ >= thr_cons[k]).mean())})
                negF = pd.concat([Fneg_te, sc_te_neg[cons_cols]], axis=1)
                posF = pd.concat([Fp[flip].reset_index(drop=True), sc_p[cons_cols]], axis=1)
                g = cv_gain(negF, posF, FEATS_D, cons_cols, c, c["rng_seed"])
                if g:
                    cv_rows.append({"eps": eps, "attack": name, **g})
            log.info(f"eps {eps} {name}: success {flip.mean():.3f} valid {valid.mean():.3f} "
                     f"succ_undet_D {und['D'].mean():.3f} TPR_D {det_rows[-5]['tpr_at_5fpr']}")
    for nm, rows in (("attacks", att_rows), ("detection", det_rows), ("defender", d_rows),
                     ("consistency_exploratory", cons_rows), ("cv_gain_exploratory", cv_rows)):
        pd.DataFrame(rows).to_csv(out / f"s6c_{nm}.csv", index=False)
    pd.DataFrame([{"feature": f, "thr_val_benign_95pct": thr_single[f], "mean_val_benign": mu_single[f],
                   "sd_val_benign": sd_single[f]} for f in FEATS_D]).to_csv(out / "s6c_single_thresholds.csv", index=False)
    pd.set_option("display.width", 250)
    log.info("Defender:\n" + pd.DataFrame(d_rows).round(3).to_string(index=False))
    log.info("Attacks:\n" + pd.DataFrame(att_rows).round(3).to_string(index=False))
    finish_run(run_dir, log, {"seed": s}, t0)


if __name__ == "__main__":
    main()
