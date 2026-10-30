# Stage 6 FINAL: learned defenders D and D_plus against adaptive attackers, 3 seeds, eps 0.5 and 1.0.
#
# Defenders (per seed and eps, trained ONLY on validation data):
#   D       standardised logistic regression on [msp, margin, mahalanobis_benign, knn5_benign] (as in v3)
#   D_plus  D's features plus the 12 attention-attribution consistency scores, each oriented in the validated
#           direction (feature = consistency, i.e. high consistency = suspicious), L2-regularised (C = 1)
#   negatives  20,000 val benign rows predicted benign (training) and 50,000 other ones (held out, threshold at 5% FPR)
#   positives  constrained A1 rows on the 1,000 val attack-class source rows of v1 to v3 (valid and flipped)
# Attackers on the Stage 5 rows (150 per attack class per seed, all if fewer), constrained, same projection and
# independent checker, 100 steps, 1 restart, kappa = 10 on the benign margin:
#   A1   the Stage 5 rows
#   A6   against D: hinge lambda {1, 100} on relu(D_logit - thr + 0.5), and the normalised variant (w = 0.5)
#   A7   against D_plus with the same settings. The 12 consistency features are not differentiable (ranks, IG), so
#        the attacker uses a surrogate: the rollout vs |grad x input| cosine (differentiable with double backprop,
#        one of the 12 features itself) and replaces every consistency feature by its linear fit on that cosine
#        (fitted on validation; the R^2 of every fit is reported to justify the choice).
#   Ablations (seed 0 only): A6_msp_only and A6_knn_only (lambda 10) as in v3.
# Test negatives: 1,000 genuine benign test rows predicted benign by that seed.
# Report per seed and pooled: success, success AND undetected under D and D_plus, AUROC and TPR at 5% FPR with
# bootstrap CIs, cost vs A1, per source class, and a PAIRED D vs D_plus bootstrap of the difference (same rows).
# No cross-validation on test rows.
# Stop conditions: any invalid constrained row, D val AUROC < 0.9, runtime estimate > 5 h.
# Resumable: every (seed, eps, attack) result is written to results/stage6/final/seed{s}/ and skipped on restart.
#
# Usage:
#   .venv\Scripts\python.exe -m src.stage6d_final                       # full run
#   .venv\Scripts\python.exe -m src.stage6d_final --set stage6d.smoke=true
import copy
import json
import math
import pickle
import sys
import time
from datetime import datetime

import numpy as np
import pandas as pd
import torch
from scipy.stats import spearmanr
from sklearn.linear_model import LinearRegression, LogisticRegression
from sklearn.preprocessing import StandardScaler

from src import config as cfgmod
from src.config import load_config
from src.preprocess import Preprocessor
from src.stage3_explain import load_ftt
from src.stage4_attack import STAGE4, rollout_cls
from src.stage4_constraints import TorchPreprocessor, check_validity, derive, load_spec_csv, make_context, project
from src.stage5_fullrun import keep_awake, md
from src.stage6_defense import BENIGN, STAGE6, Baselines, make_rows
from src.stage6b_defense import STAGE6B, auroc, boot, detector_scores
from src.stage6c_defense import FEATS_D, STAGE6C, a1_on, feats_np, feats_torch
from src.utils import ROOT, finish_run, resolve, start_run

STAGE6D = {
    "smoke": False,
    "seeds": [0, 1, 2],
    "n_per_class": 150,
    "n_heldout": 50000,
    "lambdas": [1.0, 100.0],
    "norm_w": 0.5,
    "ablation_seed": 0,
    "max_hours": 5.0,
    "min_val_auroc": 0.9,
    "C_plus": 1.0,
    "out_dir": "results/stage6/final",
    "progress_log": True,
    # Rows per attack chunk (A7 needs double backprop through attention; 8 GB VRAM).
    "attack_chunk": 500,
}
SURR = "cons_rollout_gxi_cos"
PROGRESS = ROOT / "PROGRESS.md"


class StopRun(Exception):
    pass


# ---------------------------------------------------------------- defenders

class LinearDefender:
    # Standardised L2 logistic regression; decision() on numpy, torch_decision() on torch features.
    def __init__(self, cols, Xneg, Xpos, Xheld, fpr, C, dev):
        self.cols = cols
        X = np.r_[Xneg, Xpos]
        y = np.r_[np.zeros(len(Xneg)), np.ones(len(Xpos))]
        self.sc = StandardScaler().fit(X)
        self.lr = LogisticRegression(C=C, max_iter=5000, class_weight="balanced").fit(self.sc.transform(X), y)
        self.thr = float(np.quantile(self.decision(Xheld), 1 - fpr))
        t = lambda a: torch.tensor(a, dtype=torch.float32, device=dev)
        self.mu, self.sd, self.w = t(self.sc.mean_), t(self.sc.scale_), t(self.lr.coef_[0])
        self.b = float(self.lr.intercept_[0])

    def decision(self, X):
        return self.lr.decision_function(self.sc.transform(np.asarray(X, dtype=np.float64)))

    def torch_decision(self, F):
        return ((F - self.mu) / self.sd) @ self.w + self.b


def plus_features(sc):
    # D_plus feature table: the 4 baselines and the 12 consistency scores in the validated direction
    # (detector_scores returns -consistency; the feature is +consistency, high = suspicious).
    cons = [k for k in sc.columns if k.startswith("cons_")]
    return pd.concat([sc[FEATS_D], -sc[cons]], axis=1), FEATS_D + cons


# ---------------------------------------------------------------- attacks

def sub_rows(S, idx):
    # Row subset of a make_rows() object (attacks are per-row independent, so they can run in chunks).
    T = type("Sub", (), {})()
    T.R, T.frozen, T.feats, T.floors = S.R[idx], S.frozen[idx], S.feats, S.floors
    return T


def attack_loop(model, S, tp, eps, c, lam, norm_w, score_terms, thr, dev, seed):
    # Runs attack_chunk() on row chunks of at most c["attack_chunk"] rows (bounds the memory of double backprop).
    xs, zs = [], []
    n = S.R.shape[0]
    for i in range(0, n, c["attack_chunk"]):
        idx = np.arange(i, min(i + c["attack_chunk"], n))
        x, z = attack_chunk(model, sub_rows(S, idx), tp, eps, c, lam, norm_w, score_terms, thr, dev, seed + i)
        xs.append(x), zs.append(z)
    return np.concatenate(xs), torch.cat(zs)


def attack_chunk(model, S, tp, eps, c, lam, norm_w, score_terms, thr, dev, seed):
    # Constrained PGD: maximise min(e, kappa) - lam * relu(score - thr + safety); normalised variant mixes the
    # unit-norm gradients of min(e, kappa) and -score. score_terms(xv, ctx) -> (e, score), both differentiable.
    ctx = make_context(S.R, S.feats, S.floors, eps, S.frozen, dev)
    gen = torch.Generator(device=dev).manual_seed(seed)
    kappa, safety = c["kappa_a6"], c["a6_safety"]
    best_key = torch.full((S.R.shape[0],), float("-inf"), dtype=torch.float64, device=dev)
    best_x = None
    for run in range(1 + c["restarts"]):
        delta = torch.zeros_like(ctx.lo) if run == 0 else \
            ctx.lo + (ctx.hi - ctx.lo) * torch.rand(ctx.lo.shape, generator=gen, device=dev, dtype=torch.float64)
        for step in range(c["steps"] + 1):
            x, _ = project(delta, ctx)
            xv = x.detach().requires_grad_(True)
            e, s = score_terms(xv, ctx)
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


def terms_D(model, tp, base, D):
    def f(xv, ctx):
        z = tp(derive(xv, ctx)).float()
        lg = model(z)
        return lg[:, BENIGN] - lg[:, 1:].amax(1), D.torch_decision(feats_torch(lg, z, base))
    return f


def terms_Dplus(model, tp, base, Dp, surr):
    # Baseline features exact; each consistency feature replaced by its validation linear fit on the
    # differentiable rollout vs |grad x input| cosine (benign logit, double backprop).
    a, k = surr

    def f(xv, ctx):
        z = tp(derive(xv, ctx)).float()
        lg, attns = model(z, return_attn=True)
        roll = rollout_cls(attns)
        gb, = torch.autograd.grad(lg[:, BENIGN].sum(), z, create_graph=True)
        gxi = (z * gb).abs()
        A = roll / roll.sum(1, keepdim=True)
        B = gxi / gxi.sum(1, keepdim=True).clamp_min(1e-12)
        cos = (A * B).sum(1) / (A.norm(dim=1) * B.norm(dim=1)).clamp_min(1e-12)
        F = torch.cat([feats_torch(lg, z, base), a[None, :] + k[None, :] * cos[:, None]], 1)
        return lg[:, BENIGN] - lg[:, 1:].amax(1), Dp.torch_decision(F)
    return f


def single_terms(model, tp, base, f_idx, mu, sd):
    def f(xv, ctx):
        z = tp(derive(xv, ctx)).float()
        lg = model(z)
        return lg[:, BENIGN] - lg[:, 1:].amax(1), (feats_torch(lg, z, base)[:, f_idx] - mu) / sd
    return f


# ---------------------------------------------------------------- statistics

def paired_boot(nD, pD, tD, nP, pP, tP, n, rng):
    # Paired bootstrap of (D_plus - D) for TPR at each detector's own threshold and for AUROC: the same resampled
    # negative and positive rows are used for both detectors.
    if len(pD) == 0:
        return {}
    dt, da = [], []
    for _ in range(n):
        ni, pi = rng.integers(0, len(nD), len(nD)), rng.integers(0, len(pD), len(pD))
        dt.append((pP[pi] >= tP).mean() - (pD[pi] >= tD).mean())
        da.append(auroc(nP[ni], pP[pi]) - auroc(nD[ni], pD[pi]))
    return {"tpr_diff": float((pP >= tP).mean() - (pD >= tD).mean()), "tpr_diff_lo": np.percentile(dt, 2.5),
            "tpr_diff_hi": np.percentile(dt, 97.5), "auroc_diff": auroc(nP, pP) - auroc(nD, pD),
            "auroc_diff_lo": np.percentile(da, 2.5), "auroc_diff_hi": np.percentile(da, 97.5)}


# ---------------------------------------------------------------- per-seed driver

def load_common(c, dev):
    G = type("G", (), {})()
    s1, s2, s3 = resolve("results/stage1"), resolve("results/stage2"), resolve("results/stage3")
    G.s2 = s2
    G.feats = json.loads((s1 / "features.json").read_text(encoding="utf-8"))
    lm = json.loads((s1 / "label_map.json").read_text(encoding="utf-8"))
    G.classes = [k for k, _ in sorted(lm.items(), key=lambda kv: kv[1])]
    G.pp = Preprocessor.load(s2 / "preprocess.json")
    G.tp = TorchPreprocessor(G.pp, dev)
    G.spec = load_spec_csv()
    G.floors = G.spec.set_index("feature")["zero_floor_scale"].astype(float).to_dict()
    rng = np.random.default_rng(c["rng_seed"] + 6000)                  # same stream as v1 to v3
    sel = pd.read_parquet(s3 / "selection.parquet")
    pos = np.flatnonzero((sel["class"] != "benign").to_numpy() & (sel["rank_in_class"] < c["n_per_class"]).to_numpy())
    G.pos = pos[np.lexsort((sel["rank_in_class"].to_numpy()[pos], sel["label_id"].to_numpy()[pos]))]
    test = pd.read_parquet(s1 / "test.parquet", columns=G.feats + ["label_id"])
    G.Rte = test.iloc[sel["test_row"].to_numpy()[G.pos]][G.feats].to_numpy(np.float64).copy()
    G.yte = sel["label_id"].to_numpy()[G.pos].astype(np.int64)
    G.Zben_te = torch.from_numpy(G.pp.transform(test[test["label_id"] == BENIGN][G.feats])).to(dev)
    del test
    vs = pd.read_parquet(resolve("results/stage3c") / "val_selection.parquet")
    vs = vs[vs.label_id != BENIGN]
    vpick = np.concatenate([rng.choice(g.index.to_numpy(), size=min(c["val_per_class"], len(g)), replace=False)
                            for _, g in vs.groupby("label_id")])
    val = pd.read_parquet(s1 / "val.parquet", columns=G.feats + ["label_id"])
    G.Rva = val.iloc[vs.loc[vpick, "val_row"].to_numpy()][G.feats].to_numpy(np.float64).copy()
    G.yva = vs.loc[vpick, "label_id"].to_numpy().astype(np.int64)
    G.Zben_va = torch.from_numpy(G.pp.transform(val[val["label_id"] == BENIGN][G.feats])).to(dev)
    del val
    train = pd.read_parquet(s1 / "train.parquet", columns=G.feats + ["label_id"])
    Ztr, ytr = G.pp.transform(train), train["label_id"].to_numpy()
    del train
    G.base = Baselines(Ztr, ytr, len(G.classes), c, dev, rng)
    del Ztr
    G.Ste = make_rows(G.Rte, G.yte, G.classes, G.feats, G.spec, G.floors, G.tp, dev)
    G.Sva = make_rows(G.Rva, G.yva, G.classes, G.feats, G.spec, G.floors, G.tp, dev)
    return G


def check_valid_or_stop(X, R, G, eps, S, what, log):
    valid, viol, _ = check_validity(X, R, G.feats, G.spec, eps, S.cls_rows)
    if not valid.all():
        bad = [k for k, v in viol.items() if v.any()]
        log.error(f"STOP: {int((~valid).sum())} invalid constrained rows in {what} ({bad})")
        raise StopRun(f"invalid rows in {what}: {bad}")
    return valid


def run_seed(c, s, G, dev, log, out):
    sdir = out / f"seed{s}"
    sdir.mkdir(parents=True, exist_ok=True)
    model, _ = load_ftt(G.s2 / f"ftt_seed{s}.pt", dev)
    for p in model.parameters():
        p.requires_grad_(False)
    rng_s = np.random.default_rng(c["rng_seed"] + 8000 + s)
    # ---- negatives: val benign (train sample + held-out) and test benign, all predicted benign by this seed
    with torch.no_grad():
        pv = torch.cat([model(G.Zben_va[i:i + 8192]).argmax(1) for i in range(0, len(G.Zben_va), 8192)])
        pt = torch.cat([model(G.Zben_te[i:i + 8192]).argmax(1) for i in range(0, len(G.Zben_te), 8192)])
    va_ok = torch.nonzero(pv == BENIGN).squeeze(1).cpu().numpy()
    perm = rng_s.permutation(va_ok)
    tr_idx, held_idx = perm[:c["d_train_neg"]], perm[c["d_train_neg"]:c["d_train_neg"] + c["n_heldout"]]
    te_idx = rng_s.choice(torch.nonzero(pt == BENIGN).squeeze(1).cpu().numpy(), size=c["n_test_benign"], replace=False)
    fneg = sdir / "negatives.pkl"
    if fneg.exists():
        NEG = pickle.loads(fneg.read_bytes())
    else:
        t0 = time.time()
        NEG = {k: detector_scores(model, G.Zben_va[torch.from_numpy(ix).to(dev)] if k != "test" else
                                  G.Zben_te[torch.from_numpy(ix).to(dev)], G.base, c)[0]
               for k, ix in (("val_train", tr_idx), ("val_held", held_idx), ("test", te_idx))}
        fneg.write_bytes(pickle.dumps(NEG))
        log.info(f"seed {s}: negatives scored in {time.time() - t0:.0f} s")
    rows_def, results = [], {}
    for eps in c["eps"]:
        fdef = sdir / f"defenders_eps{eps:g}.pkl"
        if fdef.exists():
            DEF = pickle.loads(fdef.read_bytes())
        else:
            t0 = time.time()
            Xv, zv = a1_on(model, G.Sva, G.tp, eps, c, dev, c["rng_seed"] + 7100 + int(eps * 100) + 10 * s)
            check_valid_or_stop(Xv, G.Rva, G, eps, G.Sva, f"seed {s} eps {eps} val A1", log)
            with torch.no_grad():
                flip = (model(zv).argmax(1) == BENIGN)
            posv, _ = detector_scores(model, zv[flip], G.base, c)
            Ftr, cols = plus_features(NEG["val_train"])
            Fho, _ = plus_features(NEG["val_held"])
            Fpv, _ = plus_features(posv)
            D = LinearDefender(FEATS_D, Ftr[FEATS_D].to_numpy(), Fpv[FEATS_D].to_numpy(), Fho[FEATS_D].to_numpy(),
                               c["fpr"], 1e6, dev)          # effectively unregularised, as v3
            Dp = LinearDefender(cols, Ftr[cols].to_numpy(), Fpv[cols].to_numpy(), Fho[cols].to_numpy(),
                                c["fpr"], c["C_plus"], dev)
            # Surrogate maps for A7: each consistency feature ~ a + k * surrogate cosine (validation rows).
            Fall = pd.concat([Ftr, Fpv], ignore_index=True)
            cons_cols = cols[len(FEATS_D):]
            a_, k_, r2, rho = [], [], {}, {}
            for col in cons_cols:
                lrm = LinearRegression().fit(Fall[[SURR]].to_numpy(), Fall[col].to_numpy())
                a_.append(lrm.intercept_), k_.append(lrm.coef_[0])
                r2[col] = lrm.score(Fall[[SURR]].to_numpy(), Fall[col].to_numpy())
                rho[col] = spearmanr(Fall[SURR], Fall[col]).statistic
            DEF = {"D": D, "Dp": Dp, "surr": (np.array(a_), np.array(k_)), "r2": r2, "rho": rho,
                   "val_auroc_D": auroc(D.decision(Fho[FEATS_D]), D.decision(Fpv[FEATS_D])),
                   "val_auroc_Dp": auroc(Dp.decision(Fho[cols]), Dp.decision(Fpv[cols])),
                   "n_val_pos": int(flip.sum()), "val_valid": 1.0, "seconds": round(time.time() - t0, 1)}
            fdef.write_bytes(pickle.dumps(DEF))
        D, Dp = DEF["D"], DEF["Dp"]
        for obj in (D, Dp):                                  # tensors back on the current device after unpickling
            obj.mu, obj.sd, obj.w = obj.mu.to(dev), obj.sd.to(dev), obj.w.to(dev)
        if DEF["val_auroc_D"] < c["min_val_auroc"]:
            raise StopRun(f"seed {s} eps {eps}: D val AUROC {DEF['val_auroc_D']:.3f} < {c['min_val_auroc']}")
        rows_def.append({"seed": s, "eps": eps, "n_val_pos": DEF["n_val_pos"], "val_auroc_D": DEF["val_auroc_D"],
                         "val_auroc_Dplus": DEF["val_auroc_Dp"], "thr_D": D.thr, "thr_Dplus": Dp.thr,
                         **{f"D_coef_{f}": w for f, w in zip(FEATS_D, D.lr.coef_[0])},
                         "Dplus_coef_abs_sum_consistency": float(np.abs(Dp.lr.coef_[0][len(FEATS_D):]).sum()),
                         "Dplus_coef_abs_sum_baselines": float(np.abs(Dp.lr.coef_[0][:len(FEATS_D)]).sum()),
                         **{f"surr_r2_{k}": v for k, v in DEF["r2"].items()},
                         **{f"surr_rho_{k}": v for k, v in DEF["rho"].items()}})
        Fneg_te, cols = plus_features(NEG["test"])
        nD, nP = D.decision(Fneg_te[FEATS_D]), Dp.decision(Fneg_te[cols])
        s5 = resolve("results/stage5") / f"seed{s}"
        a1 = np.load(s5 / f"adv_raw_constrained_A1_eps{eps:g}.npz")
        i1 = np.array([np.flatnonzero(a1["pos"] == q)[0] for q in G.pos])
        surr = tuple(torch.tensor(v, dtype=torch.float32, device=dev) for v in DEF["surr"])
        attacks = [("A1", None)]
        for lam in c["lambdas"]:
            attacks.append((f"A6_lam{lam:g}", (terms_D(model, G.tp, G.base, D), D.thr, lam, None)))
        attacks.append((f"A6norm_w{c['norm_w']:g}", (terms_D(model, G.tp, G.base, D), D.thr, 0.0, c["norm_w"])))
        for lam in c["lambdas"]:
            attacks.append((f"A7_lam{lam:g}", (terms_Dplus(model, G.tp, G.base, Dp, surr), Dp.thr, lam, None)))
        attacks.append((f"A7norm_w{c['norm_w']:g}", (terms_Dplus(model, G.tp, G.base, Dp, surr), Dp.thr, 0.0, c["norm_w"])))
        if s == c["ablation_seed"]:
            Fvb = feats_np(model, G.Zben_va[torch.from_numpy(held_idx).to(dev)], G.base)
            for f in ("msp", "knn5_benign"):
                mu, sd, thr = float(Fvb[f].mean()), float(Fvb[f].std()), float(np.quantile(Fvb[f], 0.95))
                attacks.append((f"A6_{f.split('_')[0]}_only",
                                (single_terms(model, G.tp, G.base, FEATS_D.index(f), mu, sd), (thr - mu) / sd,
                                 c["a6_single_lambda"], None)))
        for name, spec_a in attacks:
            fres = sdir / f"rows_eps{eps:g}_{name}.parquet"
            if fres.exists():
                results[(eps, name)] = pd.read_parquet(fres)
                continue
            t0 = time.time()
            if spec_a is None:
                X = a1["x_adv"][i1]
                with torch.no_grad():
                    z = G.tp(torch.from_numpy(X).to(dev)).float()
            else:
                fn, thr, lam, nw = spec_a
                X, z = attack_loop(model, G.Ste, G.tp, eps, c, lam, nw, fn, thr, dev,
                                   c["rng_seed"] + 7300 + int(eps * 100) + 17 * s + sum(map(ord, name)))
            check_valid_or_stop(X, G.Rte, G, eps, G.Ste, f"seed {s} eps {eps} {name}", log)
            with torch.no_grad():
                lg = model(z)
            flip = (lg.argmax(1) == BENIGN).cpu().numpy()
            sc, _ = detector_scores(model, z, G.base, c)
            F, _ = plus_features(sc)
            df = pd.DataFrame({"seed": s, "eps": eps, "attack": name, "pos": G.pos, "class": [G.classes[k] for k in G.yte],
                               "flipped": flip, "margin": (lg[:, BENIGN] - lg[:, 1:].amax(1)).cpu().numpy(),
                               "score_D": D.decision(F[FEATS_D]), "score_Dplus": Dp.decision(F[cols]),
                               "thr_D": D.thr, "thr_Dplus": Dp.thr, "valid": True,
                               "seconds": time.time() - t0})
            df.to_parquet(fres, index=False)
            results[(eps, name)] = df
            log.info(f"seed {s} eps {eps} {name}: success {flip.mean():.3f}, succ_undet D "
                     f"{(flip & (df.score_D < D.thr)).mean():.3f} D_plus {(flip & (df.score_Dplus < Dp.thr)).mean():.3f} "
                     f"[{time.time() - t0:.0f} s]")
        results[(eps, "_neg")] = pd.DataFrame({"seed": s, "eps": eps, "score_D": nD, "score_Dplus": nP})
    del model
    torch.cuda.empty_cache()
    return results, pd.DataFrame(rows_def)


# ---------------------------------------------------------------- summaries

def summarise(results_by_seed, defs, c, label):
    # Tables (DataFrames) for one seed or pooled over seeds.
    att, det, pair, cls_rows = [], [], [], []
    rng = np.random.default_rng(c["rng_seed"] + 99)
    names = sorted({k[1] for R in results_by_seed for k in R if k[1] != "_neg"},
                   key=lambda n: (n != "A1", n.startswith("A6_"), n))
    for eps in c["eps"]:
        neg = pd.concat([R[(eps, "_neg")] for R in results_by_seed], ignore_index=True)
        a1succ = None
        for name in names:
            parts = [R[(eps, name)] for R in results_by_seed if (eps, name) in R]
            if not parts:
                continue
            r = pd.concat(parts, ignore_index=True)
            fl = r.flipped.to_numpy()
            undD = fl & (r.score_D < r.thr_D).to_numpy()
            undP = fl & (r.score_Dplus < r.thr_Dplus).to_numpy()
            succ = fl.mean()
            if name == "A1":
                a1succ = succ
            seeds_in = sorted(r.seed.unique())
            nsub = neg[neg.seed.isin(seeds_in)]
            att.append({"eps": eps, "attack": name, "seeds": ",".join(map(str, seeds_in)), "n_rows": len(r),
                        "evasion_success": succ, "cost_vs_A1": (a1succ - succ) if a1succ is not None else np.nan,
                        "succ_undet_D": undD.mean(), "succ_undet_Dplus": undP.mean(),
                        "median_margin_flipped": float(np.median(r.margin[fl])) if fl.any() else np.nan,
                        "invalid_rows": 0})
            # Thresholds are seed-specific: compare each score with its own seed's threshold by centring.
            pD = (r.score_D - r.thr_D).to_numpy()[fl]
            pP = (r.score_Dplus - r.thr_Dplus).to_numpy()[fl]
            thrD = {sd: t for sd, t in zip(r.seed, r.thr_D)}
            thrP = {sd: t for sd, t in zip(r.seed, r.thr_Dplus)}
            nD = (nsub.score_D - nsub.seed.map(thrD)).to_numpy()
            nP = (nsub.score_Dplus - nsub.seed.map(thrP)).to_numpy()
            for dn, n_, p_ in (("D", nD, pD), ("D_plus", nP, pP)):
                (alo, ahi), (tlo, thi) = boot(n_, p_, 0.0, c["n_boot"], rng)
                det.append({"eps": eps, "attack": name, "detector": dn, "n_neg": len(n_), "n_pos": len(p_),
                            "auroc": auroc(n_, p_), "auroc_lo": alo, "auroc_hi": ahi,
                            "tpr_at_5fpr": float((p_ >= 0).mean()) if len(p_) else np.nan, "tpr_lo": tlo,
                            "tpr_hi": thi, "test_fpr": float((n_ >= 0).mean())})
            pb = paired_boot(nD, pD, 0.0, nP, pP, 0.0, c["n_boot"], rng)
            if pb:
                pair.append({"eps": eps, "attack": name, "n_pos": len(pD), **pb})
            for cn, g in r.groupby("class"):
                f_ = g.flipped.to_numpy()
                cls_rows.append({"eps": eps, "attack": name, "class": cn, "n": len(g), "success": f_.mean(),
                                 "succ_undet_D": (f_ & (g.score_D < g.thr_D).to_numpy()).mean(),
                                 "succ_undet_Dplus": (f_ & (g.score_Dplus < g.thr_Dplus).to_numpy()).mean(),
                                 "tpr_D": float((g.score_D >= g.thr_D)[f_].mean()) if f_.any() else np.nan,
                                 "tpr_Dplus": float((g.score_Dplus >= g.thr_Dplus)[f_].mean()) if f_.any() else np.nan})
    return {"attacks": pd.DataFrame(att), "detection": pd.DataFrame(det), "paired": pd.DataFrame(pair),
            "per_class": pd.DataFrame(cls_rows), "defenders": defs}


def report(T, title):
    f = lambda d: d.round(3)
    ci = lambda m, lo, hi: "" if pd.isna(m) else (f"{m:.3f}" if pd.isna(lo) else f"{m:.3f} [{lo:.3f}, {hi:.3f}]")
    out = [f"\n---\n\n## {datetime.now():%Y-%m-%d %H:%M}: {title}\n"]
    dfn = T["defenders"]
    keep = [k for k in dfn.columns if not k.startswith(("surr_r2", "surr_rho"))]
    out.append("**Defenders (validation only)**\n\n" + md(f(dfn[keep])))
    r2 = dfn[["seed", "eps"] + [k for k in dfn.columns if k.startswith("surr_r2")]]
    r2.columns = [k.replace("surr_r2_cons_", "") for k in r2.columns]
    out.append("\n**A7 surrogate fit: R^2 of each consistency feature on the rollout vs grad x input cosine "
               "(validation rows)**\n\n" + md(f(r2)))
    rho = dfn[["seed", "eps"] + [k for k in dfn.columns if k.startswith("surr_rho")]]
    rho.columns = [k.replace("surr_rho_cons_", "") for k in rho.columns]
    out.append("\nSpearman correlation of each consistency feature with the surrogate (validation rows):\n\n" + md(f(rho)))
    out.append("\n**Attacks: success, success AND undetected, cost vs A1 (0 invalid rows everywhere; the run stops "
               "otherwise)**\n\n" + md(f(T["attacks"])))
    d = T["detection"].copy()
    d["AUROC [95% CI]"] = [ci(r.auroc, r.auroc_lo, r.auroc_hi) for r in d.itertuples()]
    d["TPR at 5% FPR [95% CI]"] = [ci(r.tpr_at_5fpr, r.tpr_lo, r.tpr_hi) for r in d.itertuples()]
    out.append("\n**Detection on test (negatives = 1,000 benign test rows per seed)**\n\n"
               + md(d[["eps", "attack", "detector", "n_pos", "AUROC [95% CI]", "TPR at 5% FPR [95% CI]", "test_fpr"]].round(3)))
    p = T["paired"].copy()
    if len(p):
        p["TPR D_plus - D [95% CI]"] = [ci(r.tpr_diff, r.tpr_diff_lo, r.tpr_diff_hi) for r in p.itertuples()]
        p["AUROC D_plus - D [95% CI]"] = [ci(r.auroc_diff, r.auroc_diff_lo, r.auroc_diff_hi) for r in p.itertuples()]
        out.append("\n**Paired comparison D_plus minus D (same rows, paired bootstrap)**\n\n"
                   + md(p[["eps", "attack", "n_pos", "TPR D_plus - D [95% CI]", "AUROC D_plus - D [95% CI]"]]))
    pc = T["per_class"]
    main_att = pc[pc.attack.isin(["A1", "A6_lam1", "A7_lam1"])]
    out.append("\n**Per source class (A1, A6 lambda 1, A7 lambda 1)**\n\n" + md(f(main_att)))
    return "\n".join(out)


def append_progress(text, c, log):
    if c["progress_log"]:
        with open(PROGRESS, "a", encoding="utf-8") as fh:
            fh.write(text.rstrip() + "\n")
    else:
        log.info("[progress entry, not written]\n" + text[:3000])


# ---------------------------------------------------------------- main

def estimate_hours(c, G, dev, log):
    # Measure per-step attack costs and scoring throughput at full size, then extrapolate.
    model, _ = load_ftt(G.s2 / f"ftt_seed{c['seeds'][0]}.pt", dev)
    for p in model.parameters():
        p.requires_grad_(False)
    cc = dict(c, steps=2, restarts=0)
    D = type("Dd", (), {})()
    D.torch_decision = lambda F: F.sum(1)
    T = {}
    for nm, fn in (("A6", terms_D(model, G.tp, G.base, D)),
                   ("A7", terms_Dplus(model, G.tp, G.base, D, (torch.zeros(12, device=dev), torch.ones(12, device=dev))))):
        torch.cuda.synchronize()
        t0 = time.time()
        attack_loop(model, G.Ste, G.tp, 1.0, cc, 1.0, None, fn, 0.0, dev, 0)
        torch.cuda.synchronize()
        T[nm] = (time.time() - t0) / 3
    torch.cuda.synchronize()
    t0 = time.time()
    detector_scores(model, G.Ste.z0, G.base, c)
    T["score_row"] = (time.time() - t0) / len(G.pos)
    steps = (c["steps"] + 1) * (1 + c["restarts"])
    n_att = len(c["lambdas"]) + 1
    per_seed = (c["d_train_neg"] + c["n_heldout"] + c["n_test_benign"]) * T["score_row"] \
        + len(c["eps"]) * (steps * T["A6"] * 1000 / len(G.pos)                   # val A1 on 1,000 rows
                           + n_att * steps * (T["A6"] + T["A7"])
                           + (2 * n_att + 1) * len(G.pos) * T["score_row"])
    total = len(c["seeds"]) * per_seed + len(c["eps"]) * 2 * steps * T["A6"]
    del model
    torch.cuda.empty_cache()
    log.info(f"Measured: A6 step {T['A6']:.3f} s, A7 step {T['A7']:.3f} s ({len(G.pos)} rows), scoring "
             f"{1000 * T['score_row']:.2f} ms/row")
    return total / 3600, T


def main():
    cfgmod.DEFAULTS.setdefault("stage6d", {**copy.deepcopy(STAGE4), **copy.deepcopy(STAGE6), **copy.deepcopy(STAGE6B),
                                           **copy.deepcopy(STAGE6C), **copy.deepcopy(STAGE6D)})
    cfg, overrides = load_config("Stage 6 final.")
    c = cfg["stage6d"]
    if c["smoke"]:
        c.update(seeds=[0], n_per_class=3, steps=2, val_per_class=5, d_train_neg=2000, n_heldout=3000,
                 n_test_benign=300, n_boot=20, progress_log=False, out_dir=c["out_dir"] + "_smoke")
    out = resolve(c["out_dir"])
    out.mkdir(parents=True, exist_ok=True)
    t_start = time.time()
    run_dir, log = start_run("stage6d_smoke" if c["smoke"] else "stage6d_final", cfg, overrides)
    dev = torch.device("cuda")
    keep_awake(True)
    try:
        G = load_common(c, dev)
        hours, T = estimate_hours(c, G, dev, log)
        log.info(f"RUNTIME ESTIMATE: {hours:.2f} h for seeds {c['seeds']} (limit {c['max_hours']} h)")
        (out / "runtime_estimate.txt").write_text(f"{hours:.3f} h\n{T}\n", encoding="utf-8")
        if hours > c["max_hours"]:
            raise StopRun(f"runtime estimate {hours:.2f} h > {c['max_hours']} h")
        all_res, all_def = [], []
        for s in c["seeds"]:
            ts = time.time()
            R, dfn = run_seed(c, s, G, dev, log, out)
            all_res.append(R), all_def.append(dfn)
            T1 = summarise([R], dfn, c, f"seed {s}")
            for k, v in T1.items():
                v.to_csv(out / f"seed{s}" / f"summary_{k}.csv", index=False)
            marker = out / f"seed{s}" / "progress_logged"
            if not marker.exists():
                append_progress(report(T1, f"Stage 6 FINAL: seed {s} finished ({(time.time() - ts) / 60:.0f} min this "
                                           f"session)"), c, log)
                marker.write_text("1", encoding="utf-8")
        TP = summarise(all_res, pd.concat(all_def, ignore_index=True), c, "pooled")
        for k, v in TP.items():
            v.to_csv(out / f"summary_pooled_{k}.csv", index=False)
        marker = out / "final_progress_logged"
        if not marker.exists() and len(all_res) == len(c["seeds"]):
            append_progress(report(TP, f"Stage 6 FINAL: pooled over seeds {c['seeds']}"), c, log)
            marker.write_text("1", encoding="utf-8")
    except StopRun as e:
        log.error(f"STOPPED: {e}")
        append_progress(f"\n---\n\n## {datetime.now():%Y-%m-%d %H:%M}: Stage 6 FINAL stopped\n\nStop condition: {e}\n",
                        c, log)
        finish_run(run_dir, log, {"stopped": str(e)}, t_start)
        keep_awake(False)
        sys.exit(2)
    keep_awake(False)
    finish_run(run_dir, log, {"seeds": c["seeds"]}, t_start)


if __name__ == "__main__":
    main()
