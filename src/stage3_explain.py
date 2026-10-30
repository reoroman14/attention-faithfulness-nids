# Stage 3: attention vs attribution agreement and faithfulness on CLEAN test data (no attack yet).
# Read-only on results/stage1 and results/stage2. Writes raw float16 tensors to results/stage3/,
# summary tables to results/stage3_*.csv and a run dir under results/runs/.
# A pilot (--set stage3.pilot=true) writes everything under results/stage3/pilot/ instead.
#
# Explanation set (main): test rows correctly classified by every FT-T seed, up to n_per_class per class,
# allocated proportionally over days, fixed sampling seed. Target = predicted-class logit, which equals
# the label for every selected row (and is the same class for every seed, so seeds are comparable).
# Robustness set (per seed): the same size cap, drawn from the rows THAT seed classifies correctly. It reuses
# a random subset of the main rows and adds rows correct for that seed but not for all seeds, in proportion
# to their share (all of them when the class has at most n_per_class seed-correct rows). Only the added
# rows need new attributions.
#
# Attributions per FT-T seed (72 features; the CLS self-weight is never ranked):
#   attn_last     last-layer CLS row, head-averaged
#   attn_mean     CLS row averaged over heads and layers
#   attn_rollout  attention rollout (Abnar and Zuidema 2020): 0.5 A + 0.5 I, row-normalised, multiplied over layers
#   attn_cls_all  every layer x head CLS row (stored, not ranked)
#   ig_zero       Integrated Gradients, baseline = train mean (zeros in preprocessed space)
#   ig_benign     Integrated Gradients averaged over K sampled benign train rows as baselines
#   gshap         GradientSHAP with a class-stratified background of train rows
#   occlusion     leave-one-feature-out: f(x) - f(x with feature j set to the train mean)
#   treeshap_xgb  XGBoost TreeSHAP (pred_contribs), cross-model reference (agreement only)
#   abs_x         trivial reference: |x - 0| in preprocessed space (distance from the train mean)
#   random        a random ranking per row
# Signed methods are ranked by |attribution| (primary, same meaning as attention) and, separately, by the
# signed value (suffix _signed).
#
# Faithfulness: target-class probability (primary) and logit margin (target logit minus best other logit,
# secondary) under MoRF deletion, LeRF deletion and MoRF insertion, with two masking baselines (train mean,
# one sampled benign train row per explained row).
# f0 normalisation: p_norm = (p - p_empty) / (p_full - p_empty), where p_full = p(x) and p_empty = p(all
# features masked); skipped for rows where p_full - p_empty < norm_min_gap. For benign rows a fully masked
# input is itself classified benign, so normalised scores are not meaningful and are reported as NaN with a
# flag. Headline numbers are macro averages over the 10 attack classes.
#
# Usage:
#   .venv\Scripts\python.exe -m src.stage3_explain --set stage3.pilot=true   # pilot: 20 rows/class, seed 0
#   .venv\Scripts\python.exe -m src.stage3_explain                           # full run
import copy
import json
import math
import time
from contextlib import contextmanager
from itertools import combinations

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import xgboost as xgb
from captum.attr import GradientShap, IntegratedGradients
from scipy.stats import hypergeom, rankdata

from src import config as cfgmod
from src.config import load_config
from src.models.ft_transformer import FeatureTokenizer, FTTransformer
from src.preprocess import Preprocessor
from src.stage2_train import predict_torch
from src.utils import RESULTS, finish_run, resolve, start_run

# Stage 3 defaults, registered into the shared config at runtime so --set stage3.<key>=... works.
STAGE3 = {
    "pilot": False,
    "sel_seed": 2026,
    "n_per_class": 500,
    "min_per_class": 50,
    "seeds": [0, 1, 2],
    "pilot_n_per_class": 20,
    "pilot_seeds": [0],
    "topk": [1, 3, 5, 10, 20, 36],
    # AOPC averages p(x) - p(x with k features removed) over k = 0..aopc_steps.
    "aopc_steps": 10,
    "ig_steps": 50,
    "ig_benign_k": 10,
    "gs_background": 100,
    "gs_n_samples": 50,
    "gs_stdev": 0.0,
    # Rows per IG / GradientSHAP call, and interpolation points per captum forward/backward.
    "grad_batch": 32,
    "ig_internal_batch": 800,
    # Max model inputs per no-grad forward (curves, occlusion); keeps peak memory well under 8 GB.
    "eval_rows": 4096,
    "attn_batch": 256,
    "norm_min_gap": 0.05,
    # Rows per class for the parameter-randomisation sanity check.
    "rand_n_per_class": 50,
    "pilot_rand_n_per_class": 10,
    "treeshap": True,
    "robustness": True,
}
BENIGN = "benign"
ATTN = ["attn_last", "attn_mean", "attn_rollout"]
SIGNED = ["ig_zero", "ig_benign", "gshap", "occlusion"]
# Parameter randomisation levels: (name, modules re-initialised). The classifier head is kept intact
# except in head_only. Cascade levels re-seed identically and start at block3, so block3 (and block2)
# get the same random weights at every cascade level.
RAND_LEVELS = [("head_only", ["head"]),
               ("cascade_top1", ["block3"]),
               ("cascade_top2", ["block3", "block2"]),
               ("cascade_top3", ["block3", "block2", "block1"]),
               ("cascade_top3_tokenizer", ["block3", "block2", "block1", "tokenizer"]),
               ("indep_block1", ["block1"]),
               ("indep_block2", ["block2"]),
               ("indep_block3", ["block3"])]
HEAD_NOTE = "trivially unchanged for attention: attention does not depend on the classifier head"


# ---------------------------------------------------------------- helpers

@contextmanager
def timed(timing, method, seed, n):
    torch.cuda.synchronize()
    torch.cuda.reset_peak_memory_stats()
    t0 = time.time()
    yield
    torch.cuda.synchronize()
    dt = time.time() - t0
    timing.append({"seed": str(seed), "method": method, "n_rows": n, "seconds": round(dt, 2),
                   "ms_per_row": round(1000 * dt / max(n, 1), 3),
                   "peak_gpu_mem_gb": round(torch.cuda.max_memory_allocated() / 2**30, 3)})


def load_ftt(path, dev):
    ck = torch.load(path, map_location=dev, weights_only=False)
    model = FTTransformer(**ck["model_kwargs"]).to(dev)
    model.load_state_dict(ck["state_dict"])
    return model.eval(), ck


@torch.no_grad()
def target_out(model, X, target, rows):
    # Target-class logit and probability, fp32, in chunks.
    lg, pr = [], []
    for i in range(0, X.shape[0], rows):
        out = model(X[i:i + rows])
        t = target[i:i + rows, None]
        lg.append(out.gather(1, t).squeeze(1))
        pr.append(torch.softmax(out, -1).gather(1, t).squeeze(1))
    return torch.cat(lg), torch.cat(pr)


def class_rows(vals, y, classes, normalised=False, **key):
    # Mean / std / n per class over finite values, then unweighted means over the attack classes
    # (headline) and over all classes. Normalised metrics are not reported for benign (flagged).
    out, means = [], {}
    for ci, name in enumerate(classes):
        v = vals[y == ci]
        v = v[np.isfinite(v)]
        flag = ""
        if normalised and name == BENIGN:
            v = v[:0]
            flag = "not meaningful: fully masked input is classified benign, normalisation denominator about 0"
        mu = float(v.mean()) if len(v) else np.nan
        out.append({**key, "class": name, "mean": mu, "std": float(v.std(ddof=1)) if len(v) > 1 else np.nan,
                    "n": len(v), "flag": flag})
        if len(v):
            means[name] = mu
    for agg, members in (("macro_attack_classes", [k for k in means if k != BENIGN]),
                         ("macro_all_classes", list(means))):
        m = [means[k] for k in members]
        out.append({**key, "class": agg, "mean": float(np.mean(m)) if m else np.nan,
                    "std": float(np.std(m, ddof=1)) if len(m) > 1 else np.nan, "n": len(m), "flag": ""})
    return out


def ranking(score, rng):
    # Feature order per row, highest score first; exact ties are broken randomly (fixed rng).
    return np.lexsort((rng.random(score.shape), -score), axis=-1)


def centred_ranks(a):
    r = rankdata(a, axis=1)
    r -= r.mean(1, keepdims=True)
    return r, np.sqrt((r ** 2).sum(1))


def spearman_from_ranks(ra, rb):
    (a, na), (b, nb) = ra, rb
    den = na * nb
    return np.divide((a * b).sum(1), den, out=np.full(len(a), np.nan), where=den > 0)


def rowwise_spearman(a, b):
    return spearman_from_ranks(centred_ranks(a), centred_ranks(b))


def topk_mask(order, k):
    m = np.zeros(order.shape, dtype=bool)
    np.put_along_axis(m, order[:, :k], True, axis=1)
    return m


def jaccard(order_a, order_b, k):
    ma, mb = topk_mask(order_a, k), topk_mask(order_b, k)
    return (ma & mb).sum(1) / (ma | mb).sum(1)


def chance_jaccard(n_feat, k):
    # Expected Jaccard of two independent uniformly random k-subsets of n_feat features.
    i = np.arange(0, k + 1)
    return float((hypergeom(n_feat, k, k).pmf(i) * i / (2 * k - i)).sum())


def agreement_rows(scores, y, classes, rng, topk, **key):
    # Spearman and top-k Jaccard for every pair of methods (scores are non-negative importances).
    F = next(iter(scores.values())).shape[1]
    orders = {k: ranking(v, rng) for k, v in scores.items()}
    ranks = {k: centred_ranks(v) for k, v in scores.items()}
    rows = []
    for a, b in combinations(scores, 2):
        rows += class_rows(spearman_from_ranks(ranks[a], ranks[b]), y, classes, **key, method_a=a, method_b=b,
                           metric="spearman", chance=0.0)
        for k in topk:
            rows += class_rows(jaccard(orders[a], orders[b], k), y, classes, **key, method_a=a, method_b=b,
                               metric=f"jaccard@{k}", chance=chance_jaccard(F, k))
    return rows


# ---------------------------------------------------------------- selection

def select_rows(y, days, preds, classes, n_per_class, min_per_class, rng_seed):
    # Rows correct for every seed; per class up to n_per_class, allocated over days by largest remainder.
    all_ok = np.stack([p == y for p in preds.values()]).all(0)
    rng = np.random.default_rng(rng_seed)
    picked_all, rank_all, counts = [], [], []
    for ci, name in enumerate(classes):
        surv = np.flatnonzero((y == ci) & all_ok)
        n_take = min(n_per_class, len(surv))
        dvals, dcounts = np.unique(days[surv], return_counts=True)
        quota = dcounts / max(dcounts.sum(), 1) * n_take
        alloc = np.floor(quota).astype(int)
        for i in np.argsort(-(quota - alloc), kind="stable")[: n_take - alloc.sum()]:
            alloc[i] += 1
        parts = [rng.choice(surv[days[surv] == d], size=a, replace=False) for d, a in zip(dvals, alloc)]
        picked = rng.permutation(np.concatenate(parts)) if parts else np.array([], dtype=int)
        picked_all.append(picked)
        rank_all.append(np.arange(len(picked)))
        counts.append({"class": name, "test_support": int((y == ci).sum()),
                       **{f"correct_seed{s}": int(((y == ci) & (p == y)).sum()) for s, p in preds.items()},
                       "correct_all_seeds": len(surv), "selected": len(picked),
                       "below_min": len(surv) < min_per_class,
                       "survivors_by_day": ";".join(f"{d}:{n}" for d, n in zip(dvals, dcounts)),
                       "selected_by_day": ";".join(f"{d}:{a}" for d, a in zip(dvals, alloc))})
    return np.concatenate(picked_all), np.concatenate(rank_all), pd.DataFrame(counts)


def robustness_rows(y_te, preds, s, sel, rank_in_class, classes, n_per_class, rng):
    # Per-seed robustness set: positions of reused main rows plus new test rows (correct for seed s only).
    all_ok = np.stack([p == y_te for p in preds.values()]).all(0)
    ok_s = preds[s] == y_te
    y_sel = y_te[sel]
    keep, extra, recs = [], [], []
    for ci, name in enumerate(classes):
        cls = y_te == ci
        E = np.flatnonzero(cls & ok_s & ~all_ok)
        n_s = int((cls & ok_s).sum())
        main_pos = np.flatnonzero(y_sel == ci)
        main_pos = main_pos[np.argsort(rank_in_class[main_pos])]   # random order from the main selection
        if n_s <= n_per_class:
            n_e, n_m = len(E), len(main_pos)
        else:
            n_e = min(len(E), max(n_per_class - len(main_pos), int(round(n_per_class * len(E) / n_s))))
            n_m = min(len(main_pos), n_per_class - n_e)
        keep.append(main_pos[:n_m])
        extra.append(rng.choice(E, size=n_e, replace=False) if n_e else np.array([], dtype=int))
        recs.append({"seed": s, "class": name, "correct_seed": n_s, "correct_all_seeds": int((cls & all_ok).sum()),
                     "seed_only_correct": len(E), "main_rows_reused": n_m, "extra_rows_added": n_e})
    return np.concatenate(keep), np.concatenate(extra), pd.DataFrame(recs)


# ---------------------------------------------------------------- attributions

@torch.no_grad()
def attention_variants(model, X, bs):
    out = {k: [] for k in ("attn_cls_all", "attn_last", "attn_mean", "attn_rollout")}
    max_err = 0.0
    for i in range(0, X.shape[0], bs):
        _, attns = model(X[i:i + bs], return_attn=True)
        A = torch.stack(attns, 1)                          # (B, L, H, T, T)
        max_err = max(max_err, float((A.sum(-1) - 1).abs().max()))
        Ah = A.mean(2)                                     # head average, (B, L, T, T)
        T = Ah.shape[-1]
        eye = torch.eye(T, device=X.device)
        R = eye.expand(Ah.shape[0], T, T)
        for layer in range(Ah.shape[1]):
            Al = 0.5 * Ah[:, layer] + 0.5 * eye             # residual connection
            R = (Al / Al.sum(-1, keepdim=True)) @ R
        out["attn_cls_all"].append(A[:, :, :, 0, :].cpu())  # CLS rows incl. self-weight, stored only
        out["attn_last"].append(Ah[:, -1, 0, 1:].cpu())     # token 0 (CLS) excluded from ranking
        out["attn_mean"].append(Ah[:, :, 0, 1:].mean(1).cpu())
        out["attn_rollout"].append(R[:, 0, 1:].cpu())
    return {k: torch.cat(v).numpy() for k, v in out.items()}, max_err


def integrated_gradients(model, X, target, baselines, c):
    # Baselines (K, F); attributions and |delta| averaged over the K baselines (K = 1 for the train mean).
    # rel = |delta| / |f(x) - f(b)| per baseline, averaged.
    ig = IntegratedGradients(model)
    N, K = X.shape[0], baselines.shape[0]
    attr = torch.zeros_like(X)
    abs_delta = torch.zeros(N, device=X.device)
    rel = torch.zeros(N, device=X.device)
    fx, _ = target_out(model, X, target, c["eval_rows"])
    for k in range(K):
        b = baselines[k:k + 1]
        with torch.no_grad():
            fb = model(b)[0][target]
        for i in range(0, N, c["grad_batch"]):
            sl = slice(i, i + c["grad_batch"])
            xb = X[sl]
            a, d = ig.attribute(xb, baselines=b.expand_as(xb), target=target[sl], n_steps=c["ig_steps"],
                                internal_batch_size=c["ig_internal_batch"], return_convergence_delta=True)
            attr[sl] += a.detach() / K
            abs_delta[sl] += d.detach().abs() / K
            rel[sl] += d.detach().abs() / (fx[sl] - fb[sl]).abs().clamp_min(1e-12) / K
    return attr.cpu().numpy(), abs_delta.cpu().numpy(), rel.cpu().numpy()


def gradient_shap(model, X, target, background, c, seed):
    gs = GradientShap(model)
    torch.manual_seed(seed)
    attr, delta = [], []
    for i in range(0, X.shape[0], c["grad_batch"]):
        sl = slice(i, i + c["grad_batch"])
        a, d = gs.attribute(X[sl], baselines=background, target=target[sl], n_samples=c["gs_n_samples"],
                            stdevs=c["gs_stdev"], return_convergence_delta=True)
        attr.append(a.detach())
        delta.append(d.detach().view(a.shape[0], -1).mean(1).abs())
    return torch.cat(attr).cpu().numpy(), torch.cat(delta).cpu().numpy()


@torch.no_grad()
def occlusion(model, X, target, base, c):
    # Leave-one-feature-out: f(x) - f(x with feature j replaced by base_j), target logit.
    N, F = X.shape
    fx, _ = target_out(model, X, target, c["eval_rows"])
    idx = torch.arange(F, device=X.device)
    rows = max(1, c["eval_rows"] // F)
    out = torch.empty_like(X)
    for i in range(0, N, rows):
        xb, bb, t = X[i:i + rows], base[i:i + rows], target[i:i + rows]
        B = xb.shape[0]
        Xr = xb.unsqueeze(1).repeat(1, F, 1)
        Xr[:, idx, idx] = bb
        lg = model(Xr.view(-1, F)).view(B, F, -1)
        out[i:i + B] = fx[i:i + B, None] - lg.gather(2, t.view(B, 1, 1).expand(B, F, 1)).squeeze(2)
    return out.cpu().numpy()


def all_attributions(model, X, target, s, c, ig_ben_base, gs_bg, gs_seed, timing, tag=""):
    # Every FT-T attribution for rows X; returns (attributions, diagnostics, attention max row-sum error).
    N, F = X.shape
    attr = {}
    with timed(timing, f"{tag}attention (all variants)", s, N):
        att, max_err = attention_variants(model, X, c["attn_batch"])
    attr.update(att)
    with timed(timing, f"{tag}ig_zero", s, N):
        attr["ig_zero"], dz, rz = integrated_gradients(model, X, target, torch.zeros(1, F, device=X.device), c)
    with timed(timing, f"{tag}ig_benign (K={c['ig_benign_k']})", s, N):
        attr["ig_benign"], db, rb = integrated_gradients(model, X, target, ig_ben_base, c)
    with timed(timing, f"{tag}gshap (bg={len(gs_bg)}, n={c['gs_n_samples']})", s, N):
        attr["gshap"], dg = gradient_shap(model, X, target, gs_bg, c, gs_seed)
    with timed(timing, f"{tag}occlusion (LOFO, train mean)", s, N):
        attr["occlusion"] = occlusion(model, X, target, torch.zeros_like(X), c)
    diag = {"ig_zero_abs_delta": dz, "ig_zero_rel_delta": rz, "ig_benign_abs_delta": db,
            "ig_benign_rel_delta": rb, "gshap_abs_delta": dg}
    return attr, diag, max_err


def importance_scores(attr, X_np, rng, treeshap=None):
    # Non-negative importances used for agreement and stability (signed methods by |a|).
    sc = {k: attr[k] for k in ATTN}
    sc.update({k: np.abs(attr[k]) for k in SIGNED})
    if treeshap is not None:
        sc["treeshap_xgb"] = np.abs(treeshap)
    sc["abs_x"] = np.abs(X_np)
    sc["random"] = rng.random(X_np.shape)
    return sc


# ---------------------------------------------------------------- faithfulness

@torch.no_grad()
def perturbation_curves(model, X, base, order, target, c):
    # Target-class probability and logit margin (target logit minus best other logit) for k = 0..F features
    # removed (deletion) or inserted (insertion). MoRF = most relevant first; LeRF = least relevant first.
    N, F = X.shape
    pos = torch.empty_like(order)
    pos.scatter_(1, order, torch.arange(F, device=X.device).expand(N, F))  # rank position of each feature
    ks = torch.arange(F + 1, device=X.device).view(1, F + 1, 1)
    rows = max(1, c["eval_rows"] // (F + 1))
    names = ("del_morf", "del_lerf", "ins_morf")
    out = {f"{k}{v}": torch.empty(N, F + 1) for k in names for v in ("", "_margin")}
    for i in range(0, N, rows):
        xb, bb = X[i:i + rows, None, :], base[i:i + rows, None, :]
        p, t = pos[i:i + rows, None, :], target[i:i + rows]
        B = xb.shape[0]
        morf = p < ks                      # feature is among the first k of the ranking
        lerf = p >= F - ks                 # feature is among the last k of the ranking
        for name, Xk in zip(names, (torch.where(morf, bb, xb), torch.where(lerf, bb, xb), torch.where(morf, xb, bb))):
            lg = model(Xk.reshape(-1, F)).view(B, F + 1, -1)
            ti = t.view(B, 1, 1).expand(B, F + 1, 1)
            lt = lg.gather(2, ti).squeeze(2)
            other = lg.scatter(2, ti, float("-inf")).amax(2)
            out[name][i:i + B] = torch.softmax(lg, -1).gather(2, ti).squeeze(2).cpu()
            out[f"{name}_margin"][i:i + B] = (lt - other).cpu()
    return {k: v.numpy().astype(np.float64) for k, v in out.items()}


def curve_metrics(cur, topk, aopc_steps, min_gap):
    dm, dl, im = cur["del_morf"], cur["del_lerf"], cur["ins_morf"]
    F = dm.shape[1] - 1
    p_full, p_empty = dm[:, 0], dm[:, F]
    gap = p_full - p_empty
    ok = gap >= min_gap

    def norm(a):
        return np.divide(a - p_empty[:, None], gap[:, None], out=np.full(a.shape, np.nan), where=ok[:, None])

    def auc(a):
        return np.trapezoid(a, dx=1 / F, axis=1)

    m = {"p_full": p_full, "p_empty": p_empty, "norm_ok": ok.astype(float),
         "del_auc": auc(dm), "ins_auc": auc(im), "del_auc_norm": auc(norm(dm)), "ins_auc_norm": auc(norm(im))}
    m["ins_minus_del_auc"] = m["ins_auc"] - m["del_auc"]
    # Secondary: logit margin curves (raw AUC, no normalisation).
    m["margin_full"] = cur["del_morf_margin"][:, 0]
    m["margin_empty"] = cur["del_morf_margin"][:, F]
    m["del_auc_margin"] = auc(cur["del_morf_margin"])
    m["ins_auc_margin"] = auc(cur["ins_morf_margin"])
    m["ins_minus_del_auc_margin"] = m["ins_auc_margin"] - m["del_auc_margin"]
    gap_safe = np.where(ok, gap, np.nan)
    for k in topk:
        m[f"comp@{k}"] = p_full - dm[:, k]
        m[f"suff@{k}"] = p_full - im[:, k]
        m[f"comp@{k}_norm"] = m[f"comp@{k}"] / gap_safe
        m[f"suff@{k}_norm"] = m[f"suff@{k}"] / gap_safe
    L = aopc_steps
    m["aopc_morf"] = (p_full[:, None] - dm[:, :L + 1]).mean(1)
    m["aopc_lerf"] = (p_full[:, None] - dl[:, :L + 1]).mean(1)
    m["aopc_gap"] = m["aopc_morf"] - m["aopc_lerf"]
    m["aopc_gap_norm"] = m["aopc_gap"] / gap_safe
    return m


# ---------------------------------------------------------------- parameter randomisation

def reinit(module):
    for sub in module.modules():
        if isinstance(sub, (nn.Linear, nn.LayerNorm)):
            sub.reset_parameters()
        elif isinstance(sub, FeatureTokenizer):
            bound = 1 / math.sqrt(sub.weight.shape[1])
            for p in (sub.weight, sub.bias, sub.cls):
                nn.init.uniform_(p, -bound, bound)


def randomised(model, parts, seed):
    # Copy of the model with the named parts re-initialised (Adebayo et al. 2018 style).
    m = copy.deepcopy(model)
    mods = {"head": m.head, "tokenizer": m.tokenizer, **{f"block{i + 1}": b for i, b in enumerate(m.blocks)}}
    torch.manual_seed(seed)
    with torch.no_grad():
        for p in parts:
            reinit(mods[p])
    return m.eval()


# ---------------------------------------------------------------- main

def main():
    cfgmod.DEFAULTS.setdefault("stage3", copy.deepcopy(STAGE3))
    cfgmod.DEFAULTS["paths"].setdefault("stage3_dir", "results/stage3")
    cfg, overrides = load_config("Stage 3: attention vs attribution agreement on clean test data.")
    c = cfg["stage3"]
    pilot = bool(c["pilot"])
    n_per_class = c["pilot_n_per_class"] if pilot else c["n_per_class"]
    run_seeds = c["pilot_seeds"] if pilot else c["seeds"]
    rand_n = c["pilot_rand_n_per_class"] if pilot else c["rand_n_per_class"]
    topk = c["topk"]
    out_dir = resolve(cfg["paths"]["stage3_dir"])
    if pilot:
        out_dir = out_dir / "pilot"
    out_dir.mkdir(parents=True, exist_ok=True)
    csv_dir, csv_pre = (out_dir, "stage3_") if pilot else (RESULTS, "stage3_")
    t_start = time.time()
    run_dir, log = start_run("stage3_pilot" if pilot else "stage3_explain", cfg, overrides)
    dev = torch.device("cuda")
    timing, sanity, faith, agree, robust, stab, rand_rows, robust_counts = [], [], [], [], [], [], [], []

    def sane(name, seed, value, expected, passed):
        sanity.append({"check": name, "seed": seed, "value": value, "expected": expected, "pass": bool(passed)})
        log.info(f"[sanity] seed={seed} {name}: {value} (expected {expected}) -> {'PASS' if passed else 'FAIL'}")

    def ig_sanity(diag, s, tag=""):
        for name in ("ig_zero", "ig_benign"):
            r, d = diag[f"{name}_rel_delta"], diag[f"{name}_abs_delta"]
            # Reported as is (50 steps); the pilot showed the error halves per doubling of steps.
            sane(f"{tag}{name} completeness |delta| / |f(x)-f(b)| (median, p95, max)", s,
                 f"{np.median(r):.2e}, {np.quantile(r, 0.95):.2e}, {r.max():.2e}; max |delta| {d.max():.2e}",
                 "reported as is (50 steps)", True)

    # ---- data (read-only)
    s1, s2 = resolve(cfg["paths"]["stage1_dir"]), resolve(cfg["paths"]["stage2_dir"])
    feats = json.loads((s1 / "features.json").read_text(encoding="utf-8"))
    label_map = json.loads((s1 / "label_map.json").read_text(encoding="utf-8"))
    classes = [k for k, _ in sorted(label_map.items(), key=lambda kv: kv[1])]
    F, C = len(feats), len(classes)
    pp = Preprocessor.load(s2 / "preprocess.json")
    assert pp.features == feats
    test = pd.read_parquet(s1 / "test.parquet", columns=feats + ["label_id", "day"])
    Xte = pp.transform(test)
    y_te, d_te = test["label_id"].to_numpy().astype(np.int64), test["day"].to_numpy()

    # ---- selection: every trained seed must be correct, even when only seed 0 is explained (pilot)
    all_seeds = cfg["stage2"]["ftt"]["seeds"]
    logits_full, preds_full = {}, {}
    Xte_t = torch.from_numpy(Xte).to(dev)
    for s in all_seeds:
        m, ck = load_ftt(s2 / f"ftt_seed{s}.pt", dev)
        assert ck["features"] == feats and ck["classes"] == classes
        logits_full[s] = predict_torch(m, Xte_t, ck["config"]["stage2"]["ftt"]["eval_batch_size"]).cpu().numpy()
        preds_full[s] = logits_full[s].argmax(1)
        del m
    del Xte_t
    torch.cuda.empty_cache()
    sel, rank_in_class, counts = select_rows(y_te, d_te, preds_full, classes, n_per_class, c["min_per_class"],
                                             c["sel_seed"])
    _, _, counts_full = select_rows(y_te, d_te, preds_full, classes, c["n_per_class"], c["min_per_class"],
                                    c["sel_seed"])
    counts["selected_full_run"] = counts_full["selected"].to_numpy()
    N = len(sel)
    y = y_te[sel]
    Xs_np = Xte[sel]
    Xs = torch.from_numpy(Xs_np).to(dev)
    target = torch.from_numpy(y).to(dev)
    log.info(f"Selected {N} rows:\n{counts.to_string(index=False)}")

    # ---- XGBoost predictions and TreeSHAP (cross-model reference)
    bst = xgb.Booster()
    bst.load_model(s2 / "xgb.json")
    bst.set_param({"device": "cuda"})

    def treeshap_for(X_np, yy):
        contrib = bst.predict(xgb.DMatrix(X_np), pred_contribs=True)   # (N, C, F+1), bias last
        return contrib[np.arange(len(yy)), yy, :F]

    xgb_pred = bst.predict(xgb.DMatrix(Xs_np)).argmax(1)
    counts["xgb_correct_selected"] = [int(((y == ci) & (xgb_pred == y)).sum()) for ci in range(C)]
    treeshap = None
    if c["treeshap"]:
        with timed(timing, "treeshap_xgb", "xgb", N):
            treeshap = treeshap_for(Xs_np, y)
        np.savez_compressed(out_dir / "treeshap_xgb.npz", treeshap=treeshap.astype(np.float16))
    counts.to_csv(csv_dir / f"{csv_pre}selection_counts.csv", index=False)

    sel_df = pd.DataFrame({"test_row": sel, "label_id": y, "class": [classes[i] for i in y], "day": d_te[sel],
                           "rank_in_class": rank_in_class,
                           **{f"pred_seed{s}": preds_full[s][sel] for s in all_seeds}, "xgb_pred": xgb_pred})
    sel_df.to_parquet(out_dir / "selection.parquet", index=False)
    np.save(out_dir / "selection_logits.npy", np.stack([logits_full[s][sel] for s in all_seeds], 1))  # (N, S, C)
    del logits_full

    # ---- baselines from the train split (fixed rng)
    train = pd.read_parquet(s1 / "train.parquet", columns=feats + ["label_id"])
    ytr = train["label_id"].to_numpy()
    rng = np.random.default_rng(c["sel_seed"] + 1)
    ben = np.flatnonzero(ytr == classes.index(BENIGN))
    mask_rows = rng.choice(ben, size=N, replace=False)
    ig_rows = rng.choice(ben, size=c["ig_benign_k"], replace=False)
    per, extra = divmod(c["gs_background"], C)
    bg_rows = np.concatenate([rng.choice(np.flatnonzero(ytr == ci), size=per + (extra if ci == 0 else 0),
                                         replace=False) for ci in range(C)])
    np.savez(out_dir / "baseline_rows.npz", mask_benign_train_rows=mask_rows, ig_benign_train_rows=ig_rows,
             gshap_background_train_rows=bg_rows)
    to_t = lambda idx: torch.from_numpy(pp.transform(train.iloc[idx])).to(dev)
    mask_bases = {"train_mean": torch.zeros_like(Xs), "benign_sample": to_t(mask_rows)}
    ig_ben_base, gs_bg = to_t(ig_rows), to_t(bg_rows)
    del train

    rand_sub = np.flatnonzero(rank_in_class < rand_n)
    abs_scores = {}       # seed -> method -> (N, F) importances, kept for cross-seed stability
    for s in run_seeds:
        model, _ = load_ftt(s2 / f"ftt_seed{s}.pt", dev)
        lg, pr = target_out(model, Xs, target, c["eval_rows"])
        with torch.no_grad():
            pred = torch.cat([model(Xs[i:i + c["eval_rows"]]).argmax(1) for i in range(0, N, c["eval_rows"])])
        sane("selected rows predicted as label", s, int((pred == target).sum()), N, bool((pred == target).all()))

        attr, diag, max_err = all_attributions(model, Xs, target, s, c, ig_ben_base, gs_bg, c["sel_seed"] + s,
                                               timing)
        sane("attention max |row sum - 1|", s, f"{max_err:.2e}", "< 1e-5", max_err < 1e-5)
        ig_sanity(diag, s)
        sane("gshap mean |convergence delta| (50-sample estimate, informative only)", s,
             f"{diag['gshap_abs_delta'].mean():.3e}", "informative", True)
        np.savez_compressed(out_dir / f"attr_seed{s}.npz", **{k: v.astype(np.float16) for k, v in attr.items()},
                            **diag, target_logit=lg.cpu().numpy(), target_prob=pr.cpu().numpy())

        # Rankings for faithfulness: attention as is, signed methods by |a| (primary) and signed value,
        # plus the trivial |x| reference and random.
        rng_s = np.random.default_rng(c["sel_seed"] + 100 + s)
        sc = importance_scores(attr, Xs_np, rng_s)
        abs_scores[s] = sc
        rank_scores = dict(sc)
        rank_scores.update({f"{k}_signed": attr[k] for k in SIGNED})
        orders = {k: ranking(v, rng_s) for k, v in rank_scores.items()}

        # ---- faithfulness curves
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
                        faith += class_rows(vals, y, classes, normalised="_norm" in metric, seed=s,
                                            mask_baseline=bname, method=mname, metric=metric)
        np.savez_compressed(out_dir / f"curves_seed{s}.npz", **curves_store)
        del curves_store

        # ---- agreement on the main set (all pairs of non-negative importances)
        ag = dict(sc)
        if treeshap is not None:
            ag["treeshap_xgb"] = np.abs(treeshap)
        rows = agreement_rows(ag, y, classes, rng_s, topk, seed=s)
        agree += rows
        robust += [{**r, "subset": "all_seeds_correct"} for r in rows]

        # ---- robustness set: rows THIS seed classifies correctly (reuses main attributions)
        if c["robustness"]:
            rng_r = np.random.default_rng(c["sel_seed"] + 400 + s)
            keep_pos, extra_rows, rc = robustness_rows(y_te, preds_full, s, sel, rank_in_class, classes,
                                                       n_per_class, rng_r)
            robust_counts.append(rc)
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
                ig_sanity(ediag, s, tag="robustness extra: ")
                es = importance_scores(ea, Xe_np, rng_r, treeshap_for(Xe_np, ye) if treeshap is not None else None)
                np.savez_compressed(out_dir / f"robustness_extra_seed{s}.npz", test_row=extra_rows,
                                    **{k: v.astype(np.float16) for k, v in ea.items()}, **ediag)
                sub_scores = {k: np.concatenate([v, es[k]]) for k, v in sub_scores.items()}
            y_r = np.concatenate([y[keep_pos], ye])
            sub_scores["random"] = rng_r.random((len(y_r), F))
            np.savez(out_dir / f"robustness_rows_seed{s}.npz", main_positions=keep_pos, extra_test_rows=extra_rows)
            robust += [{**r, "subset": "seed_correct"} for r in
                       agreement_rows(sub_scores, y_r, classes, rng_r, topk, seed=s)]

        # ---- parameter randomisation sanity check (attention and IG)
        sub = rand_sub
        Xr, tr = Xs[sub], target[sub]
        with timed(timing, f"randomisation ({len(RAND_LEVELS)} levels)", s, len(sub)):
            rstore = {}
            for li, (level, parts) in enumerate(RAND_LEVELS):
                seed_l = c["sel_seed"] + 200 + s + (0 if level.startswith("cascade") else 10 * (li + 1))
                rm = randomised(model, parts, seed_l)
                ra, _ = attention_variants(rm, Xr, c["attn_batch"])
                rig, _, _ = integrated_gradients(rm, Xr, tr, torch.zeros(1, F, device=dev), c)
                pairs = [(k, ra[k], attr[k][sub]) for k in ATTN] + \
                        [("ig_zero", np.abs(rig), np.abs(attr["ig_zero"][sub])),
                         ("ig_zero_signed", rig, attr["ig_zero"][sub])]
                for mname, new, orig in pairs:
                    note = HEAD_NOTE if level == "head_only" and mname in ATTN else ""
                    rand_rows += class_rows(rowwise_spearman(orig, new), y[sub], classes, seed=s, level=level,
                                            randomised=";".join(parts), method=mname,
                                            metric="spearman_vs_original", note=note)
                    for k in topk:
                        rand_rows += class_rows(jaccard(ranking(orig, rng_s), ranking(new, rng_s), k), y[sub],
                                                classes, seed=s, level=level, randomised=";".join(parts),
                                                method=mname, metric=f"jaccard@{k}_vs_original", note=note)
                rstore.update({f"{level}/{k}": ra[k].astype(np.float16) for k in ATTN})
                rstore[f"{level}/ig_zero"] = rig.astype(np.float16)
                del rm
            np.savez_compressed(out_dir / f"randomisation_seed{s}.npz", rows=sub, **rstore)
        del model
        torch.cuda.empty_cache()

    # ---- cross-seed stability (abs_x is identical across seeds by construction and is left out)
    for sa, sb in combinations(run_seeds, 2):
        rng_p = np.random.default_rng(c["sel_seed"] + 300)
        for mname in abs_scores[sa]:
            if mname == "abs_x":
                continue
            a, b = abs_scores[sa][mname], abs_scores[sb][mname]
            stab += class_rows(rowwise_spearman(a, b), y, classes, method=mname, seed_a=sa, seed_b=sb,
                               metric="spearman")
            for k in topk:
                stab += class_rows(jaccard(ranking(a, rng_p), ranking(b, rng_p), k), y, classes, method=mname,
                                   seed_a=sa, seed_b=sb, metric=f"jaccard@{k}")

    faith_df, agree_df, rand_df = pd.DataFrame(faith), pd.DataFrame(agree), pd.DataFrame(rand_rows)
    stab_df, robust_df = pd.DataFrame(stab), pd.DataFrame(robust)

    # ---- random-baseline sanity: agreement near chance, AOPC MoRF-LeRF gap near 0
    macro = agree_df[agree_df["class"] == "macro_all_classes"]
    for s in run_seeds:
        r = macro[(macro.seed == s) & ((macro.method_a == "random") | (macro.method_b == "random"))]
        sp = r[r.metric == "spearman"]["mean"]
        sane("random vs every method: macro Spearman (max |.|)", s, f"{sp.abs().max():.3f}", "about 0",
             sp.abs().max() < 0.05)
        j5 = r[r.metric == "jaccard@5"]["mean"]
        sane("random vs every method: macro Jaccard@5 (min, max)", s, f"{j5.min():.3f}, {j5.max():.3f}",
             f"about {chance_jaccard(F, 5):.3f}", (j5 - chance_jaccard(F, 5)).abs().max() < 0.03)
        g = faith_df[(faith_df.seed == s) & (faith_df.method == "random") & (faith_df.metric == "aopc_gap")
                     & (faith_df["class"] == "macro_all_classes")]
        sane("random ranking: macro AOPC MoRF-LeRF gap per baseline", s,
             "; ".join(f"{b} {v:+.4f}" for b, v in zip(g.mask_baseline, g["mean"])), "about 0",
             g["mean"].abs().max() < 0.02)

    outputs = {"faithfulness": faith_df, "agreement": agree_df, "agreement_robustness": robust_df,
               "stability": stab_df, "randomisation": rand_df, "sanity": pd.DataFrame(sanity),
               "timing": pd.DataFrame(timing)}
    if robust_counts:
        outputs["robustness_counts"] = pd.concat(robust_counts, ignore_index=True)
    for name, df in outputs.items():
        df.to_csv(csv_dir / f"{csv_pre}{name}.csv", index=False)

    # ---- console summary (headline = macro over the 10 attack classes, mean over seeds)
    pd.set_option("display.width", 250)
    pd.set_option("display.max_columns", 40)
    H = "macro_attack_classes"
    log.info("Timing:\n" + pd.DataFrame(timing).to_string(index=False))
    log.info("Sanity:\n" + pd.DataFrame(sanity)[["check", "seed", "value", "pass"]].to_string(index=False))
    keep = ["del_auc", "ins_auc", "aopc_gap", "comp@5", "comp@20", "suff@5", "suff@20", "suff@36",
            "del_auc_margin", "ins_auc_margin", "del_auc_norm", "ins_auc_norm"]
    fm = faith_df[(faith_df["class"] == H) & faith_df.metric.isin(keep)]
    for b, g in fm.groupby("mask_baseline"):
        log.info(f"Faithfulness, mask baseline {b} ({H}, mean over seeds):\n"
                 + g.groupby(["method", "metric"])["mean"].mean().unstack()[keep].round(4).to_string())
    am = agree_df[(agree_df["class"] == H) & agree_df.method_a.isin(ATTN) & ~agree_df.method_b.isin(ATTN)
                  & agree_df.metric.isin(["spearman", "jaccard@5", "jaccard@10"])]
    log.info(f"Agreement attention vs others ({H}, mean over seeds):\n"
             + am.groupby(["method_a", "method_b", "metric"])["mean"].mean().unstack().round(3).to_string())
    ax = agree_df[(agree_df["class"] == H) & (agree_df.method_b == "abs_x") & (agree_df.metric == "spearman")]
    log.info(f"Spearman with |x| ({H}, mean over seeds):\n"
             + ax.groupby("method_a")["mean"].mean().round(3).to_string())
    if len(robust_df):
        rb = robust_df[(robust_df["class"] == H) & robust_df.method_a.isin(ATTN) & ~robust_df.method_b.isin(ATTN)
                       & (robust_df.metric == "spearman")]
        log.info(f"Robustness, Spearman attention vs others ({H}, mean over seeds):\n"
                 + rb.groupby(["method_a", "method_b", "subset"])["mean"].mean().unstack().round(3).to_string())
    if len(stab_df):
        sm = stab_df[(stab_df["class"] == H) & stab_df.metric.isin(["spearman", "jaccard@5"])]
        log.info(f"Cross-seed stability ({H}, mean over seed pairs):\n"
                 + sm.groupby(["method", "metric"])["mean"].mean().unstack().round(3).to_string())
    rmm = rand_df[(rand_df["class"] == H) & (rand_df.metric == "spearman_vs_original")]
    log.info(f"Parameter randomisation, Spearman vs original ({H}, mean over seeds; head_only is trivially 1 "
             f"for attention):\n" + rmm.groupby(["method", "level"])["mean"].mean().unstack()[
                 [lv for lv, _ in RAND_LEVELS]].round(3).to_string())
    if pilot:
        per_row = pd.DataFrame(timing).query("seed != 'xgb'")["seconds"].sum() / N
        n_full = int(counts["selected_full_run"].sum())
        log.info(f"Pilot: {per_row * 1000:.1f} ms per row per seed; full run {n_full} rows x {len(c['seeds'])} "
                 f"seeds -> about {per_row * n_full * len(c['seeds']) / 60:.0f} min")
    finish_run(run_dir, log, {"pilot": pilot, "n_rows": N, "seeds": run_seeds,
                              "sanity_all_pass": all(r["pass"] for r in sanity)}, t_start)


if __name__ == "__main__":
    main()
