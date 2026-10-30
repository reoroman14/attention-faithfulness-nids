# Stage 4 and 5: white-box attacks on the FT-Transformer and faithfulness of explanations under attack.
#
# Targets: each FT-T seed, attacked on its own weights. Rows: the Stage 3 explanation set (test rows correct for
# all 3 seeds), attack classes only, the first n rows per class in the Stage 3 random order (rank_in_class).
#
# Threat models
#   constrained     PGD on the attacker-controlled RAW features (src/stage4_constraints.py), mapped through the
#                   differentiable preprocessing to the model; projection (repair + backtracking) after every
#                   step; every final row is checked by the independent validity checker.
#   unconstrained   REFERENCE ONLY, NOT PROTOCOL-VALID: L-inf PGD directly on the standardised input,
#                   |delta| <= eps (std units), clipped to +-10. Its rows are mapped back to raw space and run
#                   through the same checker to show how invalid they are.
# Attacks (100 PGD steps, cosine step-size schedule, zero init + 1 random restart, best iterate per row)
#   A0   random control: one uniform draw in the same box, same projection
#   A1   evasion: maximise logit_benign - max attack logit (success: predicted benign)
#   A1b  targeted attack-to-attack: maximise logit_t - max other logit, t from TARGET_MAP (success: predicted t)
#   A2   prediction-preserving attention manipulation: minimise rollout mass on the clean top-k |IG zero| features
#        minus mass on the bottom-k, with a hinge keeping margin(y) >= tau (success: still y with margin >= tau
#        and top-k rollout mass at most (1 - mass_drop) of clean)
#   A3   joint: min(A1 objective, kappa) - lambda * (A2 attention term), lambda in a3_lambdas (success: as A1)
#   All gradients are first-order through the model and attention rollout; no IG inside any loss.
# Evaluation per setting: success, validity, raw-space perturbation; explanation shift (clean vs adversarial)
# for attn_rollout / attn_mean / attn_last / IG zero / occlusion (IG and occlusion recomputed on the adversarial
# input); faithfulness exactly as Stage 3 (deletion / insertion AUC, AOPC gap, comprehensiveness / sufficiency)
# plus the same metrics on the logit margin. Target: original class y for A1 and A3 (margin to the original
# attack class); adversarial predicted class for A0, A1b and A2 when it is an attack class, else y. Rows
# predicted benign get raw metrics only (normalised metrics NaN) and are flagged.
#
# Usage:
#   .venv\Scripts\python.exe -m src.stage4_attack --set stage4.pilot=true   # pilot: 20 rows/class, seed 0
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
from src.stage3_explain import (attention_variants, class_rows, curve_metrics, integrated_gradients, jaccard,
                                load_ftt, occlusion, perturbation_curves, ranking, rowwise_spearman)
from src.stage4_constraints import (CONTROLS, DERIVED, TorchPreprocessor, check_validity, derive, freeze_mask,
                                    load_spec_csv, make_context, preprocessing_equivalence, project)
from src.utils import RESULTS, finish_run, resolve, start_run

STAGE4 = {
    "pilot": False,
    "seeds": [0, 1, 2],
    "pilot_seeds": [0],
    "n_per_class": 150,
    "pilot_n_per_class": 20,
    "eps": [0.02, 0.05, 0.1, 0.25, 0.5],
    "steps": 100,
    # Step size in box units (constrained: fraction of the per-feature budget; unconstrained: fraction of eps),
    # cosine-decayed from alpha0 to 0 over the steps.
    "alpha0": 0.25,
    "restarts": 1,
    "a2_k": 10,
    "a2_tau": 1.0,
    "a2_penalty": 10.0,
    "a2_mass_drop": 0.5,
    "a3_lambdas": [0.1, 1.0, 10.0],
    "a3_kappa": 1.0,
    "constrained_attacks": ["A0", "A1", "A1b", "A2", "A3"],
    "unconstrained_attacks": ["A0", "A1", "A1b", "A2"],
    "topk": [5, 10, 20],
    "aopc_steps": 10,
    "ig_steps": 50,
    "grad_batch": 32,
    "ig_internal_batch": 800,
    "eval_rows": 4096,
    "attn_batch": 256,
    "norm_min_gap": 0.05,
    "rng_seed": 4026,
}
# A1b source -> target attack class (fixed, documented). Pairs stay inside a plausible confusion neighbourhood
# so the flipped prediction is still an attack class and faithfulness stays well defined.
TARGET_MAP = {"ddos": "dos_hulk", "dos_goldeneye": "dos_slowhttptest", "dos_hulk": "dos_slowloris",
              "dos_slowhttptest": "dos_slowloris", "dos_slowloris": "dos_slowhttptest",
              "ftp_patator": "ssh_patator", "ssh_patator": "ftp_patator", "portscan": "dos_hulk",
              "webattack": "bot", "bot": "webattack"}
BENIGN = "benign"
METHODS = ["attn_rollout", "attn_mean", "attn_last", "ig_zero", "occlusion", "abs_x", "random"]
SHIFT_METHODS = ["attn_rollout", "attn_mean", "attn_last", "ig_zero", "occlusion"]
KEEP_METRICS = ["p_full", "margin_full", "del_auc", "ins_auc", "aopc_gap", "del_auc_norm", "ins_auc_norm",
                "aopc_gap_norm", "del_auc_margin", "ins_auc_margin", "aopc_gap_margin"]


# ---------------------------------------------------------------- objectives

def other_max(z, c):
    return z.scatter(1, c[:, None], float("-inf")).amax(1)


def rollout_cls(attns):
    # Differentiable attention rollout (same as Stage 3): head mean, 0.5 A + 0.5 I, row-normalised, CLS row.
    A = torch.stack(attns, 1).mean(2)
    T = A.shape[-1]
    eye = torch.eye(T, device=A.device, dtype=A.dtype)
    R = eye.expand(A.shape[0], T, T)
    for layer in range(A.shape[1]):
        Al = 0.5 * A[:, layer] + 0.5 * eye
        R = (Al / Al.sum(-1, keepdim=True)) @ R
    return R[:, 0, 1:]


def attn_term(roll, top, bot):
    # Rollout mass (normalised over the 72 feature tokens) on the clean top-k minus the bottom-k features.
    m = roll / roll.sum(1, keepdim=True)
    return m.gather(1, top).sum(1) - m.gather(1, bot).sum(1), m.gather(1, top).sum(1)


def make_objective(kind, y, t, top, bot, c, lam):
    # Returns f(logits, roll) -> (objective to maximise, selection key); larger key = better iterate.
    def f(z, roll):
        if kind == "A1":
            o = z[:, 0] - z[:, 1:].amax(1)
            return o, o
        if kind == "A1b":
            o = z.gather(1, t[:, None]).squeeze(1) - other_max(z, t)
            return o, o
        L, _ = attn_term(roll, top, bot)
        if kind == "A2":
            marg = z.gather(1, y[:, None]).squeeze(1) - other_max(z, y)
            o = -L - c["a2_penalty"] * torch.relu(c["a2_tau"] - marg)
            return o, (marg >= c["a2_tau"]).double() * 1e3 - L
        if kind == "A3":
            e = z[:, 0] - z[:, 1:].amax(1)
            o = torch.clamp(e, max=c["a3_kappa"]) - lam * L
            return o, (e > 0).double() * 1e3 + o
        raise ValueError(kind)
    return f


def pgd(fwd, proj, lo, hi, grad_scale, objective, steps, alpha0, restarts, gen):
    # Sign-gradient ascent on delta inside [lo, hi]; every iterate is projected and evaluated, and the best
    # projected iterate per row (by key) over all steps and restarts is returned.
    best_key = torch.full((lo.shape[0],), float("-inf"), dtype=torch.float64, device=lo.device)
    best_x = None
    for run in range(1 + restarts):
        delta = torch.zeros_like(lo) if run == 0 else lo + (hi - lo) * torch.rand(lo.shape, generator=gen,
                                                                                   device=lo.device, dtype=lo.dtype)
        for step in range(steps + 1):
            x = proj(delta)
            xv = x.detach().requires_grad_(True)
            z, roll = fwd(xv)
            obj, key = objective(z, roll)
            key = key.detach().double()
            better = key > best_key
            best_key = torch.where(better, key, best_key)
            best_x = x.detach().clone() if best_x is None else torch.where(better[:, None], x.detach(), best_x)
            if step == steps:
                break
            g, = torch.autograd.grad(obj.sum(), xv)
            a = alpha0 * 0.5 * (1 + math.cos(math.pi * step / steps))
            delta = torch.minimum(torch.maximum(delta + a * torch.sign(g.double() * grad_scale), lo), hi)
    return best_x


# ---------------------------------------------------------------- evaluation helpers

def faith_metrics(cur, topk, aopc_steps, min_gap):
    # Stage 3 metrics (target probability) plus the same comprehensiveness / sufficiency / AOPC on the margin.
    m = curve_metrics(cur, topk, aopc_steps, min_gap)
    dm, dl, im = cur["del_morf_margin"], cur["del_lerf_margin"], cur["ins_morf_margin"]
    full = dm[:, 0]
    for k in topk:
        m[f"comp_margin@{k}"] = full - dm[:, k]
        m[f"suff_margin@{k}"] = full - im[:, k]
    L = aopc_steps
    m["aopc_gap_margin"] = (full[:, None] - dm[:, :L + 1]).mean(1) - (full[:, None] - dl[:, :L + 1]).mean(1)
    return m


def explain(model, Z, target, c):
    att, _ = attention_variants(model, Z, c["attn_batch"])
    ig, ig_d, _ = integrated_gradients(model, Z, target, torch.zeros(1, Z.shape[1], device=Z.device), c)
    occ = occlusion(model, Z, target, torch.zeros_like(Z), c)
    return {"attn_rollout": att["attn_rollout"], "attn_mean": att["attn_mean"], "attn_last": att["attn_last"],
            "ig_zero": ig, "occlusion": occ}, ig_d


def importance(expl, Z_np, rng):
    sc = {k: expl[k] for k in ("attn_rollout", "attn_mean", "attn_last")}
    sc["ig_zero"], sc["occlusion"] = np.abs(expl["ig_zero"]), np.abs(expl["occlusion"])
    sc["abs_x"] = np.abs(Z_np)
    sc["random"] = rng.random(Z_np.shape)
    return sc


def faithfulness(model, Z, target, sc, bases, c, rng):
    # Per-row metrics for every ranking and masking baseline: {(baseline, method): {metric: (N,)}}.
    out = {}
    for mname in METHODS:
        order = torch.from_numpy(ranking(sc[mname], rng)).to(Z.device)
        for bname, base in bases.items():
            cur = perturbation_curves(model, Z, base, order, target, c)
            out[(bname, mname)] = faith_metrics(cur, c["topk"], c["aopc_steps"], c["norm_min_gap"])
    return out


def rel_perturbation(X, R, floors, mask):
    d = np.abs(X - R)[:, mask]
    den = np.where(R[:, mask] != 0, np.abs(R[:, mask]), floors[mask][None, :])
    rel = d / den
    return rel.mean(1), rel.max(1)


# ---------------------------------------------------------------- main

def main():
    cfgmod.DEFAULTS.setdefault("stage4", copy.deepcopy(STAGE4))
    cfgmod.DEFAULTS["paths"].setdefault("stage3_dir", "results/stage3")
    cfgmod.DEFAULTS["paths"].setdefault("stage4_dir", "results/stage4")
    cfg, overrides = load_config("Stage 4/5: constrained attacks and faithfulness under attack.")
    c = cfg["stage4"]
    pilot = bool(c["pilot"])
    n_per_class = c["pilot_n_per_class"] if pilot else c["n_per_class"]
    run_seeds = c["pilot_seeds"] if pilot else c["seeds"]
    out_dir = resolve(cfg["paths"]["stage4_dir"]) / ("pilot" if pilot else "")
    out_dir.mkdir(parents=True, exist_ok=True)
    pre = "stage4_pilot_" if pilot else "stage4_"
    t_start = time.time()
    run_dir, log = start_run("stage4_pilot" if pilot else "stage4_attack", cfg, overrides)
    dev = torch.device("cuda")
    sanity, timing = [], []

    def sane(name, seed, value, expected, passed):
        sanity.append({"check": name, "seed": seed, "value": value, "expected": expected, "pass": bool(passed)})
        log.info(f"[sanity] seed={seed} {name}: {value} (expected {expected}) -> {'PASS' if passed else 'FAIL'}")

    # ---- data (read-only)
    s1, s2, s3 = (resolve(cfg["paths"][k]) for k in ("stage1_dir", "stage2_dir", "stage3_dir"))
    feats = json.loads((s1 / "features.json").read_text(encoding="utf-8"))
    label_map = json.loads((s1 / "label_map.json").read_text(encoding="utf-8"))
    classes = [k for k, _ in sorted(label_map.items(), key=lambda kv: kv[1])]
    F = len(feats)
    pp = Preprocessor.load(s2 / "preprocess.json")
    tp = TorchPreprocessor(pp, dev)
    spec = load_spec_csv()
    floors_s = spec.set_index("feature")["zero_floor_scale"].astype(float)
    floors = floors_s.loc[feats].to_numpy()
    ctrl_mask = np.isin(feats, CONTROLS)
    sel = pd.read_parquet(s3 / "selection.parquet")
    pos = np.flatnonzero((sel["class"] != BENIGN).to_numpy() & (sel["rank_in_class"] < n_per_class).to_numpy())
    pos = pos[np.lexsort((sel["rank_in_class"].to_numpy()[pos], sel["label_id"].to_numpy()[pos]))]
    N = len(pos)
    y_np = sel["label_id"].to_numpy()[pos].astype(np.int64)
    cls_rows = [classes[k] for k in y_np]
    test = pd.read_parquet(s1 / "test.parquet", columns=feats)
    R = test.iloc[sel["test_row"].to_numpy()[pos]].to_numpy(np.float64).copy()
    Z0_np = pp.transform(pd.DataFrame(R, columns=feats))
    with torch.no_grad():
        z0_64 = tp(torch.from_numpy(R).to(dev))
    Z0 = z0_64.float()
    eq = float((Z0.cpu().numpy() - Z0_np).__abs__().max())
    sane("torch preprocessing of the selected raw rows == src/preprocess.py (max abs diff)", "-", f"{eq:.2e}",
         "0", eq == 0)
    log.info(f"Rows: {N} ({pd.Series(cls_rows).value_counts().to_dict()})")

    # Preprocessing equivalence on 10k random test rows (written to the output dir).
    rng_eq = np.random.default_rng(c["rng_seed"])
    Xeq = test.to_numpy(np.float64)[rng_eq.choice(len(test), 10000, replace=False)]
    m0, _ = load_ftt(s2 / f"ftt_seed{run_seeds[0]}.pt", dev)
    eqd = preprocessing_equivalence(pp, tp, m0, Xeq, dev)
    pd.DataFrame([eqd]).to_csv(out_dir / f"{pre}preprocessing_equivalence.csv", index=False)
    sane("preprocessing equivalence on 10k test rows (max abs diff float64 / float32 / logits)", run_seeds[0],
         f"{eqd['max_abs_diff_float64']:.2e} / {eqd['max_abs_diff_float32_output']:.2e} / "
         f"{eqd['max_abs_diff_logits']:.2e}", "< 1e-12 / 0 / 0", eqd["max_abs_diff_float64"] < 1e-12)
    del m0, test

    # Masking baselines exactly as in Stage 3 (train mean = zeros; one benign train row per explained row).
    brow = np.load(s3 / "baseline_rows.npz")["mask_benign_train_rows"][pos]
    train = pd.read_parquet(s1 / "train.parquet", columns=feats)
    Zb = torch.from_numpy(pp.transform(train.iloc[brow])).to(dev)
    del train
    bases = {"train_mean": torch.zeros_like(Z0), "benign_sample": Zb}

    # Clean relation failures (extractor quirks) freeze the involved controls.
    _, _, clean_fail = check_validity(R, R, feats, spec, 0.0, cls_rows)
    frozen = freeze_mask(R, feats, cls_rows, clean_fail)
    pd.DataFrame([{"relation": k, "clean_rows_failing": int(v.sum())} for k, v in clean_fail.items()]).to_csv(
        out_dir / f"{pre}clean_relation_failures.csv", index=False)
    fr_df = pd.DataFrame(frozen[:, ctrl_mask], columns=np.array(feats)[ctrl_mask])
    fr_df["class"] = cls_rows
    fr_df.groupby("class").mean().round(3).to_csv(out_dir / f"{pre}frozen_share_by_class.csv")
    log.info(f"Clean relation failures: { {k: int(v.sum()) for k, v in clean_fail.items() if v.sum()} }")

    y = torch.from_numpy(y_np).to(dev)
    t_np = np.array([classes.index(TARGET_MAP[cn]) for cn in cls_rows])
    t = torch.from_numpy(t_np).to(dev)
    attacks_c = [a for a in c["constrained_attacks"] if a != "A3"] + \
        ([f"A3_lam{lam:g}" for lam in c["a3_lambdas"]] if "A3" in c["constrained_attacks"] else [])
    rows_all, faith_all = [], []

    for s in run_seeds:
        model, _ = load_ftt(s2 / f"ftt_seed{s}.pt", dev)
        for p in model.parameters():
            p.requires_grad_(False)
        with torch.no_grad():
            lg0 = model(Z0)
        sane("clean rows predicted as label", s, int((lg0.argmax(1) == y).sum()), N, bool((lg0.argmax(1) == y).all()))
        stored = np.load(s3 / f"attr_seed{s}.npz")
        clean_expl, _ = explain(model, Z0, y, {**c, "attn_batch": c["attn_batch"]})
        d_roll = float(np.abs(clean_expl["attn_rollout"] - stored["attn_rollout"][pos].astype(np.float32)).max())
        ig_st = stored["ig_zero"][pos].astype(np.float32)
        # Relative to each row's largest |IG| (float16 storage has a coarse absolute resolution at large values).
        d_ig = float((np.abs(clean_expl["ig_zero"] - ig_st).max(1) / np.abs(ig_st).max(1).clip(min=1e-6)).max())
        sane("clean rollout recomputed vs Stage 3 stored float16 (max abs diff)", s, f"{d_roll:.2e}", "< 1e-3",
             d_roll < 1e-3)
        sane("clean IG zero recomputed vs Stage 3 stored float16 (max abs diff / row max |IG|)", s, f"{d_ig:.2e}",
             "< 1e-3", d_ig < 1e-3)
        # Clean explanations: the Stage 3 stored IG / occlusion (exactly what Stage 3 evaluated).
        clean_expl["ig_zero"] = stored["ig_zero"][pos].astype(np.float32)
        clean_expl["occlusion"] = stored["occlusion"][pos].astype(np.float32)
        rng_c = np.random.default_rng(c["rng_seed"] + s)
        clean_sc = importance(clean_expl, Z0.cpu().numpy(), rng_c)
        clean_orders = {k: ranking(v, np.random.default_rng(c["rng_seed"] + 7)) for k, v in clean_sc.items()}
        k = c["a2_k"]
        ig_order = ranking(np.abs(clean_expl["ig_zero"]), np.random.default_rng(c["rng_seed"] + 11))
        top = torch.from_numpy(ig_order[:, :k].copy()).to(dev)
        bot = torch.from_numpy(ig_order[:, -k:].copy()).to(dev)
        with torch.no_grad():
            _, at0 = model(Z0, return_attn=True)
            _, top_mass0 = attn_term(rollout_cls(at0), top, bot)
        top_mass0 = top_mass0.cpu().numpy()
        with timed(timing, s, "clean", "faithfulness", 0.0, N):
            cf = faithfulness(model, Z0, y, clean_sc, bases, c, np.random.default_rng(c["rng_seed"] + 13))
        # Sanity: the clean deletion curve for rollout reproduces Stage 3 (stored float16).
        st_auc = np.trapezoid(np.load(s3 / f"curves_seed{s}.npz")["train_mean/attn_rollout/del_morf"][pos].astype(
            np.float64), dx=1 / F, axis=1)
        d_auc = float(np.abs(st_auc - cf[("train_mean", "attn_rollout")]["del_auc"]).max())
        sane("clean rollout deletion AUC vs Stage 3 stored curves (max abs diff)", s, f"{d_auc:.2e}", "< 2e-3",
             d_auc < 2e-3)
        base_info = {"seed": s, "pos": pos, "test_row": sel["test_row"].to_numpy()[pos], "class": cls_rows, "y": y_np}
        faith_all.append(flatten_faith(cf, {**base_info, "mode": "clean", "attack": "clean", "eps": 0.0}))

        def fwd_raw(ctx):
            def f(xv):
                z, attns = model(tp(derive(xv, ctx)).float(), return_attn=True)
                return z, rollout_cls(attns)
            return f

        def fwd_z(dv):
            z, attns = model(dv.float(), return_attn=True)
            return z, rollout_cls(attns)

        for eps in c["eps"]:
            ctx = make_context(R, feats, floors_s.to_dict(), eps, frozen, dev)
            settings = [("constrained", a) for a in attacks_c] + [("unconstrained", a) for a in c["unconstrained_attacks"]]
            for mode, atk in settings:
                kind = atk.split("_")[0]
                lam = float(atk.split("lam")[1]) if kind == "A3" else 0.0
                gen = torch.Generator(device=dev).manual_seed(c["rng_seed"] + 1000 * s + int(eps * 1e4)
                                                             + sum(map(ord, mode + atk)))
                with timed(timing, s, f"{mode}/{atk}", "attack", eps, N):
                    if mode == "constrained":
                        proj = lambda d: project(d, ctx)[0]
                        if kind == "A0":
                            u = ctx.lo + (ctx.hi - ctx.lo) * torch.rand(ctx.lo.shape, generator=gen, device=dev,
                                                                        dtype=torch.float64)
                            x_adv = proj(u)
                        else:
                            obj = make_objective(kind, y, t, top, bot, c, lam)
                            x_adv = pgd(fwd_raw(ctx), proj, ctx.lo, ctx.hi, ctx.budget, obj, c["steps"], c["alpha0"],
                                        c["restarts"], gen)
                        with torch.no_grad():
                            z_adv = tp(x_adv).float()
                        X_adv = x_adv.cpu().numpy()
                    else:
                        lo = torch.full_like(z0_64, -eps)
                        hi = torch.full_like(z0_64, eps)
                        proj = lambda d: (z0_64 + d).clamp(-pp.clip, pp.clip)
                        if kind == "A0":
                            zz = proj(lo + (hi - lo) * torch.rand(lo.shape, generator=gen, device=dev,
                                                                  dtype=torch.float64))
                        else:
                            obj = make_objective(kind, y, t, top, bot, c, lam)
                            zz = pgd(fwd_z, lambda d: proj(d), lo, hi, torch.ones_like(lo), obj, c["steps"],
                                     c["alpha0"] * eps, c["restarts"], gen)
                        z_adv = zz.float()
                        with torch.no_grad():
                            X_adv = torch.where(zz == z0_64, torch.from_numpy(R).to(dev), tp.inverse(zz)).cpu().numpy()
                with timed(timing, s, f"{mode}/{atk}", "evaluate", eps, N):
                    rec, fa = evaluate(model, z_adv, X_adv, R, Z0, y, t, top, bot, top_mass0, kind, c, bases,
                                       clean_sc, clean_orders, spec, feats, floors, ctrl_mask, eps, cls_rows,
                                       np.random.default_rng(c["rng_seed"] + 17 + s))
                info = {**base_info, "mode": mode, "attack": atk, "eps": eps}
                rec = pd.DataFrame({**info, **rec})
                rows_all.append(rec)
                faith_all.append(flatten_faith(fa, info, extra={"eval_target": rec["eval_target"].to_numpy(),
                                                                 "benign_pred": rec["benign_pred"].to_numpy(),
                                                                 "success": rec["success"].to_numpy()}))
                log.info(f"seed={s} eps={eps} {mode:13s} {atk:9s} success={rec['success'].mean():.3f} "
                         f"valid={rec['valid'].mean():.3f} rel_pert={rec['rel_pert_mean_all'].mean():.4f} "
                         f"n_changed={rec['n_feat_changed'].mean():.1f} [{timing[-2]['seconds']:.1f}s + "
                         f"{timing[-1]['seconds']:.1f}s]")
                if mode == "constrained":
                    np.savez_compressed(out_dir / f"adv_raw_seed{s}_{atk}_eps{eps:g}.npz", x_adv=X_adv.astype(np.float64),
                                        pos=pos)
        del model
        torch.cuda.empty_cache()

    rows_df = pd.concat(rows_all, ignore_index=True)
    faith_df = pd.concat(faith_all, ignore_index=True)
    rows_df.to_parquet(out_dir / f"{pre}rows.parquet", index=False)
    faith_df.to_parquet(out_dir / f"{pre}faith_rows.parquet", index=False)
    summarise(rows_df, faith_df, classes, c, out_dir, pre, log, timing)
    pd.DataFrame(sanity).to_csv(out_dir / f"{pre}sanity.csv", index=False)
    pd.DataFrame(timing).to_csv(out_dir / f"{pre}timing.csv", index=False)
    log.info("Sanity:\n" + pd.DataFrame(sanity)[["check", "seed", "value", "pass"]].to_string(index=False))
    if pilot:
        tdf = pd.DataFrame(timing)
        per_row = tdf["seconds"].sum() / N / len(run_seeds)
        n_full = int(sum(min(c["n_per_class"], int((sel["class"] == cn).sum())) for cn in set(sel["class"]) if cn != BENIGN))
        log.info(f"Pilot: {per_row:.2f} s per row per seed for all settings; full run {n_full} rows x "
                 f"{len(c['seeds'])} seeds -> about {per_row * n_full * len(c['seeds']) / 3600:.1f} h")
    finish_run(run_dir, log, {"pilot": pilot, "n_rows": N, "seeds": run_seeds,
                              "sanity_all_pass": all(r["pass"] for r in sanity)}, t_start)


class timed:
    def __init__(self, timing, seed, setting, phase, eps, n):
        self.t, self.info = timing, {"seed": seed, "setting": setting, "phase": phase, "eps": eps, "n_rows": n}

    def __enter__(self):
        torch.cuda.synchronize()
        torch.cuda.reset_peak_memory_stats()
        self.t0 = time.time()

    def __exit__(self, *a):
        torch.cuda.synchronize()
        self.t.append({**self.info, "seconds": round(time.time() - self.t0, 2),
                       "peak_gpu_mem_gb": round(torch.cuda.max_memory_allocated() / 2**30, 3)})


def evaluate(model, z_adv, X_adv, R, Z0, y, t, top, bot, top_mass0, kind, c, bases, clean_sc, clean_orders, spec,
             feats, floors, ctrl_mask, eps, cls_rows, rng):
    with torch.no_grad():
        lg = model(z_adv)
        _, at = model(z_adv, return_attn=True)
        L, top_mass = attn_term(rollout_cls(at), top, bot)
    pred = lg.argmax(1)
    marg_y = (lg.gather(1, y[:, None]).squeeze(1) - other_max(lg, y)).cpu().numpy()
    pred_np, y_np = pred.cpu().numpy(), y.cpu().numpy()
    tm = top_mass.cpu().numpy()
    if kind == "A0":
        success = pred_np != y_np
    elif kind in ("A1", "A3"):
        success = pred_np == 0
    elif kind == "A1b":
        success = pred_np == t.cpu().numpy()
    else:
        success = (pred_np == y_np) & (marg_y >= c["a2_tau"]) & (tm <= (1 - c["a2_mass_drop"]) * top_mass0)
    # Explanation / faithfulness target.
    if kind in ("A1", "A3"):
        target = y
    else:
        target = torch.where(pred > 0, pred, y)
    valid, viol, _ = check_validity(X_adv, R, feats, spec, eps, cls_rows)
    rp_all, rl_all = rel_perturbation(X_adv, R, floors, np.ones(len(feats), bool))
    rp_c, rl_c = rel_perturbation(X_adv, R, floors, ctrl_mask)
    der = np.isin(feats, DERIVED)
    rp_d, _ = rel_perturbation(X_adv, R, floors, der)
    expl, ig_d = explain(model, z_adv, target, c)
    Zn = z_adv.cpu().numpy()
    sc = importance(expl, Zn, rng)
    rec = {"adv_pred": pred_np, "success": success, "flipped": pred_np != y_np, "benign_pred": pred_np == 0,
           "eval_target": target.cpu().numpy(), "a1b_target": t.cpu().numpy(), "margin_y": marg_y,
           "prob_y": torch.softmax(lg, -1).gather(1, y[:, None]).squeeze(1).cpu().numpy(),
           "valid": valid, "rel_pert_mean_all": rp_all, "rel_pert_linf_all": rl_all, "rel_pert_mean_ctrl": rp_c,
           "rel_pert_linf_ctrl": rl_c, "rel_change_mean_derived": rp_d,
           "n_feat_changed": (X_adv != R).sum(1), "n_ctrl_changed": (X_adv != R)[:, ctrl_mask].sum(1),
           "z_linf": (z_adv - Z0).abs().amax(1).cpu().numpy(),
           "top_mass_clean": top_mass0, "top_mass_adv": tm, "attn_term_adv": L.cpu().numpy(),
           "ig_abs_delta": ig_d}
    for name, v in viol.items():
        rec[f"viol_{name}"] = v
    for mname in SHIFT_METHODS:
        rec[f"shift_spearman_{mname}"] = rowwise_spearman(clean_sc[mname], sc[mname])
        rec[f"shift_jaccard5_{mname}"] = jaccard(clean_orders[mname], ranking(sc[mname], rng), 5)
    fa = faithfulness(model, z_adv, target, sc, bases, c, rng)
    return rec, fa


def flatten_faith(fa, info, extra=None):
    # Long-ish table: one row per (row, baseline, method), metric columns.
    parts = []
    for (bname, mname), mets in fa.items():
        d = {k: v for k, v in info.items()}
        d.update(extra or {})
        d.update({"mask_baseline": bname, "method": mname})
        d.update({k: v for k, v in mets.items() if not k.startswith("norm_ok")})
        parts.append(pd.DataFrame(d))
    return pd.concat(parts, ignore_index=True)


# ---------------------------------------------------------------- summaries

def summarise(rows_df, faith_df, classes, c, out_dir, pre, log, timing):
    H = "macro_attack_classes"
    pd.set_option("display.width", 250)
    pd.set_option("display.max_columns", 40)
    pd.set_option("display.max_rows", 400)
    keys = ["mode", "attack", "eps"]
    tdf = pd.DataFrame(timing)
    tsum = tdf[tdf.phase.isin(["attack", "evaluate"])].pivot_table(index=["setting", "eps"], columns="phase",
                                                                    values=["seconds", "peak_gpu_mem_gb"], aggfunc="sum")
    tsum.columns = [f"{a}_{b}" for a, b in tsum.columns]
    tsum = tsum.reset_index()
    tsum["mode"] = tsum["setting"].str.split("/").str[0]
    tsum["attack"] = tsum["setting"].str.split("/").str[1]
    # Attack table: pooled and macro-over-class success, validity, perturbation size.
    g = rows_df.groupby(keys)
    att = g.agg(n=("success", "size"), success_rate=("success", "mean"), validity_rate=("valid", "mean"),
                rel_pert_mean_all=("rel_pert_mean_all", "mean"), rel_pert_mean_ctrl=("rel_pert_mean_ctrl", "mean"),
                rel_pert_linf_ctrl=("rel_pert_linf_ctrl", "mean"), rel_change_mean_derived=("rel_change_mean_derived", "mean"),
                n_feat_changed=("n_feat_changed", "mean"), n_ctrl_changed=("n_ctrl_changed", "mean"),
                z_linf=("z_linf", "mean"), benign_pred_rate=("benign_pred", "mean"),
                top_mass_ratio=("top_mass_adv", "mean")).reset_index()
    macro = rows_df.groupby(keys + ["class"])["success"].mean().groupby(keys).mean().rename("success_macro")
    att = att.merge(macro.reset_index(), on=keys)
    att["top_mass_ratio"] = att["top_mass_ratio"] / rows_df.groupby(keys)["top_mass_clean"].mean().to_numpy()
    att = att.merge(tsum[["mode", "attack", "eps", "seconds_attack", "seconds_evaluate", "peak_gpu_mem_gb_attack",
                          "peak_gpu_mem_gb_evaluate"]], on=keys, how="left")
    att["note"] = np.where(att["mode"] == "unconstrained", "reference only, NOT protocol-valid", "")
    att.to_csv(out_dir / f"{pre}attack_summary.csv", index=False)
    per_class = rows_df.groupby(keys + ["class"]).agg(n=("success", "size"), success_rate=("success", "mean"),
                                                       validity_rate=("valid", "mean"),
                                                       rel_pert_mean_all=("rel_pert_mean_all", "mean"),
                                                       n_feat_changed=("n_feat_changed", "mean"),
                                                       adv_pred_mode=("adv_pred", lambda v: classes[np.bincount(v).argmax()])
                                                       ).reset_index()
    per_class.to_csv(out_dir / f"{pre}attack_per_class.csv", index=False)
    vcols = [k for k in rows_df.columns if k.startswith("viol_")]
    rows_df.groupby(keys)[vcols].mean().reset_index().to_csv(out_dir / f"{pre}validity_detail.csv", index=False)
    log.info("Attack summary (pooled over classes and seeds):\n" + att[
        keys + ["success_rate", "success_macro", "validity_rate", "rel_pert_mean_ctrl", "rel_pert_linf_ctrl",
                "n_feat_changed", "n_ctrl_changed", "top_mass_ratio", "seconds_attack", "seconds_evaluate",
                "peak_gpu_mem_gb_evaluate"]].round(4).to_string(index=False))

    # Explanation shift (clean vs adversarial), per class and macro, overall and split by success.
    y = rows_df["y"].to_numpy()
    shift = []
    for kk, gdf in rows_df.groupby(keys):
        for subset, m in subsets(gdf):
            for mname in SHIFT_METHODS:
                for met in ("spearman", "jaccard5"):
                    shift += class_rows(gdf[f"shift_{met}_{mname}"].to_numpy()[m], gdf["y"].to_numpy()[m], classes,
                                        mode=kk[0], attack=kk[1], eps=kk[2], subset=subset, method=mname, metric=met)
    shift_df = pd.DataFrame(shift)
    shift_df.to_csv(out_dir / f"{pre}explanation_shift.csv", index=False)
    sm = shift_df[(shift_df["class"] == H) & (shift_df.subset == "all") & (shift_df.metric == "spearman")]
    log.info("Explanation shift, Spearman clean vs adversarial (macro over attack classes):\n"
             + sm.pivot_table(index=["mode", "attack", "eps"], columns="method", values="mean").round(3).to_string())

    # Faithfulness: adversarial value and change vs clean (same row, same ranking method and baseline).
    metrics = [m for m in faith_df.columns if m in KEEP_METRICS or m.startswith(("comp", "suff"))]
    metrics = [m for m in metrics if any(m.endswith(f"@{k}") or m.endswith(f"@{k}_norm") for k in c["topk"])
               or m in KEEP_METRICS]
    clean = faith_df[faith_df.attack == "clean"].set_index(["seed", "pos", "mask_baseline", "method"])[metrics]
    adv = faith_df[faith_df.attack != "clean"].copy()
    # Normalised metrics are not reported for benign-predicted rows (flagged).
    for m in metrics:
        if m.endswith("_norm"):
            adv.loc[adv["benign_pred"].astype(bool), m] = np.nan
    cl = clean.loc[pd.MultiIndex.from_frame(adv[["seed", "pos", "mask_baseline", "method"]])].to_numpy()
    delta = adv[metrics].to_numpy() - cl
    dcols = [f"d_{m}" for m in metrics]
    adv[dcols] = delta
    fr, cmp_rows = [], []
    cl_df = faith_df[faith_df.attack == "clean"]
    for (b, mname), gdf in cl_df.groupby(["mask_baseline", "method"]):
        for m in metrics:
            fr += class_rows(gdf[m].to_numpy(), gdf["y"].to_numpy(), classes, mode="clean", attack="clean", eps=0.0,
                             subset="all", mask_baseline=b, method=mname, metric=m, quantity="value")
    macro_d = {}
    for kk, gdf in adv.groupby(keys + ["mask_baseline", "method"]):
        for subset, msk in subsets(gdf):
            for m in metrics:
                for q, col in (("value", m), ("delta_vs_clean", f"d_{m}")):
                    r = class_rows(gdf[col].to_numpy()[msk], gdf["y"].to_numpy()[msk], classes, mode=kk[0],
                                   attack=kk[1], eps=kk[2], subset=subset, mask_baseline=kk[3], method=kk[4],
                                   metric=m, quantity=q)
                    if m.endswith("_norm"):
                        for row in r:
                            row["flag"] = "benign-predicted rows excluded (normalisation not meaningful)"
                    fr += r
                    if q == "delta_vs_clean":
                        macro_d[kk + (subset, m)] = (r[-2]["mean"], int(msk.sum()))
    faith_tab = pd.DataFrame(fr)
    faith_tab.to_csv(out_dir / f"{pre}faithfulness.csv", index=False)
    # The comparison that matters: change under each attack minus the change under A0 (same mode, eps, method).
    for (mode, atk, eps, b, mname, subset, m), (v, n) in macro_d.items():
        if atk == "A0":
            continue
        ref = macro_d.get((mode, "A0", eps, b, mname, "all", m))
        cmp_rows.append({"mode": mode, "attack": atk, "eps": eps, "mask_baseline": b, "method": mname,
                         "subset": subset, "metric": m, "n_rows": n, "delta_attack": v,
                         "delta_A0": ref[0] if ref else np.nan,
                         "excess_vs_A0": v - ref[0] if ref else np.nan})
    cmp_df = pd.DataFrame(cmp_rows)
    cmp_df.to_csv(out_dir / f"{pre}faithfulness_vs_A0.csv", index=False)
    show = cmp_df[(cmp_df["mode"] == "constrained") & (cmp_df.mask_baseline == "train_mean")
                  & cmp_df.metric.isin(["del_auc", "aopc_gap", "del_auc_margin", "aopc_gap_margin"])
                  & cmp_df.method.isin(["attn_rollout", "ig_zero", "random"])]
    log.info("Faithfulness change vs clean minus A0 change (constrained, train-mean masking, macro over attack "
             "classes):\n" + show.pivot_table(index=["attack", "subset", "eps"], columns=["metric", "method"],
                                              values="excess_vs_A0").round(3).to_string())


def subsets(gdf):
    # Row subsets for reporting: all rows; for evasion / targeted attacks also flipped vs not flipped.
    atk = gdf["attack"].iloc[0]
    out = [("all", np.ones(len(gdf), bool))]
    if atk.startswith(("A1", "A3")):
        s = gdf["success"].astype(bool).to_numpy()
        out += [("flipped", s), ("not_flipped", ~s)]
    return out


if __name__ == "__main__":
    main()
