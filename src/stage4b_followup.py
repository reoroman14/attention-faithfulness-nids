# Stage 4 pilot follow-up: diagnostics and a small pilot extension (seed 0, the same 20 rows per attack class).
# Read-only on every earlier output; writes to results/stage4/pilot_followup/ only.
#
# Parts (select with --set "stage4b.parts=[...]")
#   binding      which constraint type binds at the final A1 iterate (eps 0.25 / 0.5), and an UNCONSTRAINED A1
#                ablation restricted to feature groups (standardised eps 0.25 / 0.5)
#   intbudget    integer-budget artefact (share of integer controls with budget < 1) and a constrained variant
#                with count-feature budget max(eps |r|, 1); A0, A1, A3 lambda 1
#   eps_ext      constrained eps 1.0 and 2.0 for A0, A1, A2, A3 lambda 1 / 10
#   stiffness    (a) gradient norms of the attention term vs the margin term along A3 lambda 1
#                (b) pure attention attack (no prediction constraint) on rollout and attn_mean, mass ratio vs eps
#   sensitivity  (c) input sensitivity of attention on the clean Stage 3 set (all 3 seeds, stored attributions)
#   static       (d) STATIC attention baseline (train-set class-mean rollout / attn_mean) in the Stage 3
#                faithfulness metrics on clean data (all 3 seeds)
#   reporting    re-reports the pilot faithfulness with margin metrics primary for flipped rows, and the A1b
#                destination shares (no recomputation)
#   cost         measured component costs at full-run size and full-run configurations under a time budget
#
# Usage:
#   .venv\Scripts\python.exe -m src.stage4b_followup
import copy
import json
import time

import numpy as np
import pandas as pd
import torch

from src import config as cfgmod
from src.config import load_config
from src.preprocess import Preprocessor
from src.stage3_explain import (attention_variants, class_rows, curve_metrics, integrated_gradients, load_ftt,
                                occlusion, perturbation_curves, ranking, rowwise_spearman)
from src.stage4_attack import STAGE4, TARGET_MAP, attn_term, make_objective, other_max, pgd, rollout_cls
from src.stage4_constraints import (CONTROLS, DERIVED, SPEC, TorchPreprocessor, check_validity, derive, freeze_mask,
                                    load_spec_csv, make_context, project, repair)
from src.utils import finish_run, resolve, start_run

STAGE4B = {
    "parts": ["binding", "intbudget", "eps_ext", "stiffness", "sensitivity", "static", "reporting", "cost"],
    "n_per_class": 20,
    "seed": 0,
    "binding_eps": [0.25, 0.5],
    "ablation_eps": [0.25, 0.5],
    "intbudget_eps": [0.02, 0.05, 0.1, 0.25, 0.5],
    # Integer count features that get budget max(eps |r|, 1) in the variant.
    "count_features": ["fwd_pkt_cnt", "fwd_non_empty_pkt_cnt", "flag_ack", "fwd_flag_psh"],
    "ext_eps": [1.0, 2.0],
    "stiff_eps": [0.25, 0.5],
    "pure_eps": [0.02, 0.05, 0.1, 0.25, 0.5, 1.0, 2.0],
    "static_train_rows_per_class": 2000,
    "stage3_topk": [1, 3, 5, 10, 20, 36],
    "cost_steps": 10,
    "cost_n_per_class": 150,
    "budget_hours": 3.0,
}
BENIGN = "benign"
STD_CTRL = ["fwd_pkt_len_std", "fwd_iat_std", "iat_std"]
FAMILY_CTRL = ["fwd_pkt_cnt", "fwd_pkt_len_tot", "fwd_pkt_len_max", "fwd_pkt_len_min", "fwd_non_empty_pkt_cnt",
               "flow_duration", "fwd_iat_tot", "fwd_iat_max", "fwd_iat_min", "iat_max", "iat_min"]
COUPLING_CTRL = ["fwd_pkt_hdr_len_tot", "fwd_flag_psh", "flag_ack", "fwd_tcp_init_win_bytes"]
MARGIN_METRICS = ["del_auc_margin", "ins_auc_margin", "aopc_gap_margin"] + \
    [f"{a}_margin@{k}" for a in ("comp", "suff") for k in (5, 10, 20)]
PROB_METRICS = ["del_auc", "ins_auc", "aopc_gap", "del_auc_norm", "ins_auc_norm", "aopc_gap_norm"] + \
    [f"{a}@{k}" for a in ("comp", "suff") for k in (5, 10, 20)]
FLIPPED_LIKE = {"flipped", "to_benign", "to_target", "to_other_attack"}


# ---------------------------------------------------------------- shared setup

class Setup:
    pass


def setup(c, dev, n_per_class, seed):
    S = Setup()
    s1, s2, s3 = resolve("results/stage1"), resolve("results/stage2"), resolve("results/stage3")
    S.s1, S.s2, S.s3 = s1, s2, s3
    S.feats = json.loads((s1 / "features.json").read_text(encoding="utf-8"))
    lm = json.loads((s1 / "label_map.json").read_text(encoding="utf-8"))
    S.classes = [k for k, _ in sorted(lm.items(), key=lambda kv: kv[1])]
    S.pp = Preprocessor.load(s2 / "preprocess.json")
    S.tp = TorchPreprocessor(S.pp, dev)
    S.spec = load_spec_csv()
    S.floors = S.spec.set_index("feature")["zero_floor_scale"].astype(float).to_dict()
    S.ctrl_mask = np.isin(S.feats, CONTROLS)
    S.idx = {f: k for k, f in enumerate(S.feats)}
    S.sel = pd.read_parquet(s3 / "selection.parquet")
    sel = S.sel
    pos = np.flatnonzero((sel["class"] != BENIGN).to_numpy() & (sel["rank_in_class"] < n_per_class).to_numpy())
    S.pos = pos[np.lexsort((sel["rank_in_class"].to_numpy()[pos], sel["label_id"].to_numpy()[pos]))]
    S.y_np = sel["label_id"].to_numpy()[S.pos].astype(np.int64)
    S.cls_rows = [S.classes[k] for k in S.y_np]
    test = pd.read_parquet(s1 / "test.parquet", columns=S.feats)
    S.R = test.iloc[sel["test_row"].to_numpy()[S.pos]].to_numpy(np.float64).copy()
    with torch.no_grad():
        S.z0_64 = S.tp(torch.from_numpy(S.R).to(dev))
    S.Z0 = S.z0_64.float()
    _, _, cf = check_validity(S.R, S.R, S.feats, S.spec, 0.0, S.cls_rows)
    S.frozen = freeze_mask(S.R, S.feats, S.cls_rows, cf)
    S.y = torch.from_numpy(S.y_np).to(dev)
    S.t = torch.from_numpy(np.array([S.classes.index(TARGET_MAP[cn]) for cn in S.cls_rows])).to(dev)
    S.model, _ = load_ftt(s2 / f"ftt_seed{seed}.pt", dev)
    for p in S.model.parameters():
        p.requires_grad_(False)
    ig = np.load(s3 / f"attr_seed{seed}.npz")["ig_zero"][S.pos].astype(np.float32)
    order = ranking(np.abs(ig), np.random.default_rng(c["rng_seed"] + 11))   # same tie-break as the pilot
    S.top = torch.from_numpy(order[:, :c["a2_k"]].copy()).to(dev)
    S.bot = torch.from_numpy(order[:, -c["a2_k"]:].copy()).to(dev)
    with torch.no_grad():
        _, at0 = S.model(S.Z0, return_attn=True)
    S.mass0 = {w: mass(attn_vec(at0, w), S.top).cpu().numpy() for w in ("rollout", "mean")}
    S.dev = dev
    return S


def attn_vec(attns, which):
    # CLS row over the 72 feature tokens: rollout, or the mean over layers and heads (attn_mean).
    if which == "rollout":
        return rollout_cls(attns)
    return torch.stack(attns, 1).mean((1, 2))[:, 0, 1:]


def mass(a, top):
    return (a / a.sum(1, keepdim=True)).gather(1, top).sum(1)


def fwd_raw(S, ctx, which="rollout"):
    def f(xv):
        z, attns = S.model(S.tp(derive(xv, ctx)).float(), return_attn=True)
        return z, attn_vec(attns, which)
    return f


def fwd_z(S, which="rollout"):
    def f(zv):
        z, attns = S.model(zv.float(), return_attn=True)
        return z, attn_vec(attns, which)
    return f


def gen_for(c, *tag):
    return torch.Generator(device="cuda").manual_seed(c["rng_seed"] + sum(map(ord, "/".join(map(str, tag)))))


def pure_attention_objective(top):
    # Minimise the normalised attention mass on the clean top-k |IG| features; no prediction requirement.
    def f(z, a):
        o = -mass(a, top)
        return o, o
    return f


def run_constrained(S, c, kind, eps, lam=0.0, ctx_mod=None, which="rollout", objective=None):
    ctx = make_context(S.R, S.feats, S.floors, eps, S.frozen, S.dev)
    if ctx_mod is not None:
        ctx_mod(ctx)
    proj = lambda d: project(d, ctx)[0]
    gen = gen_for(c, "c", kind, eps, lam, which)
    if kind == "A0":
        u = ctx.lo + (ctx.hi - ctx.lo) * torch.rand(ctx.lo.shape, generator=gen, device=S.dev, dtype=torch.float64)
        x = proj(u)
    else:
        obj = objective or make_objective(kind, S.y, S.t, S.top, S.bot, c, lam)
        x = pgd(fwd_raw(S, ctx, which), proj, ctx.lo, ctx.hi, ctx.budget, obj, c["steps"], c["alpha0"],
                c["restarts"], gen)
    with torch.no_grad():
        z = S.tp(x).float()
    return x, z, ctx


def run_unconstrained(S, c, kind, eps, lam=0.0, which="rollout", objective=None):
    lo, hi = torch.full_like(S.z0_64, -eps), torch.full_like(S.z0_64, eps)
    clip = S.pp.clip
    proj = lambda d: (S.z0_64 + d).clamp(-clip, clip)
    gen = gen_for(c, "u", kind, eps, lam, which)
    obj = objective or make_objective(kind, S.y, S.t, S.top, S.bot, c, lam)
    zz = pgd(fwd_z(S, which), proj, lo, hi, torch.ones_like(lo), obj, c["steps"], c["alpha0"] * eps, c["restarts"], gen)
    return zz.float()


def validity(S, X, eps, budget=None):
    # Independent checker; with a budget override (variant budgets) only the budget check is recomputed.
    valid, viol, _ = check_validity(X, S.R, S.feats, S.spec, eps, S.cls_rows)
    if budget is not None:
        D = np.abs(X - S.R)
        viol["over_budget"] = (S.ctrl_mask[None, :] & (D > budget * (1 + 1e-6) + 1e-9)).any(1)
        bad = np.zeros(len(X), dtype=bool)
        for v in viol.values():
            bad |= v
        valid = ~bad
    return valid, viol


@torch.no_grad()
def light_eval(S, c, z, kind):
    lg, attns = S.model(z, return_attn=True)
    pred = lg.argmax(1)
    marg = (lg.gather(1, S.y[:, None]).squeeze(1) - other_max(lg, S.y)).cpu().numpy()
    p, yv = pred.cpu().numpy(), S.y_np
    mr = {w: mass(attn_vec(attns, w), S.top).cpu().numpy() / S.mass0[w] for w in ("rollout", "mean")}
    if kind == "A0":
        succ = p != yv
    elif kind in ("A1", "A3"):
        succ = p == 0
    elif kind == "A1b":
        succ = p == S.t.cpu().numpy()
    elif kind == "A2":
        succ = (p == yv) & (marg >= c["a2_tau"]) & (mr["rollout"] <= 1 - c["a2_mass_drop"])
    else:
        succ = np.full(len(p), np.nan)
    return {"adv_pred": p, "success": succ, "flipped": p != yv, "benign_pred": p == 0, "margin_y": marg,
            "mass_ratio_rollout": mr["rollout"], "mass_ratio_attn_mean": mr["mean"]}


def per_class_table(S, rec, **key):
    df = pd.DataFrame({"class": S.cls_rows, **{k: v for k, v in rec.items() if np.ndim(v) == 1}})
    num = [k for k in df.columns if k != "class" and df[k].dtype != object]
    g = df.groupby("class")[num].mean().reset_index()
    macro = g[num].mean().to_frame().T
    macro["class"] = "macro_attack_classes"
    pooled = df[num].mean().to_frame().T
    pooled["class"] = "pooled"
    out = pd.concat([g, macro, pooled], ignore_index=True)
    for k, v in key.items():
        out.insert(0, k, v)
    return out


# ---------------------------------------------------------------- part 1: binding + group ablation

def pgd_instrumented(fwd, proj_scale, lo, hi, grad_scale, objective, steps, alpha0, restarts, gen):
    # As stage4_attack.pgd, but also returns the latent delta and the gradient at each row's best iterate.
    best_key = torch.full((lo.shape[0],), float("-inf"), dtype=torch.float64, device=lo.device)
    best_x = best_d = best_g = None
    for run in range(1 + restarts):
        delta = torch.zeros_like(lo) if run == 0 else lo + (hi - lo) * torch.rand(lo.shape, generator=gen,
                                                                                   device=lo.device, dtype=lo.dtype)
        for step in range(steps + 1):
            x = proj_scale(delta)
            xv = x.detach().requires_grad_(True)
            z, a = fwd(xv)
            obj, key = objective(z, a)
            g, = torch.autograd.grad(obj.sum(), xv)
            key = key.detach().double()
            b = key > best_key
            best_key = torch.where(b, key, best_key)
            if best_x is None:
                best_x, best_d, best_g = x.detach().clone(), delta.clone(), g.detach().clone()
            else:
                best_x = torch.where(b[:, None], x.detach(), best_x)
                best_d = torch.where(b[:, None], delta, best_d)
                best_g = torch.where(b[:, None], g.detach(), best_g)
            if step == steps:
                break
            a_t = alpha0 * 0.5 * (1 + np.cos(np.pi * step / steps))
            delta = torch.minimum(torch.maximum(delta + a_t * torch.sign(g.double() * grad_scale), lo), hi)
    return best_x, best_d, best_g


@torch.no_grad()
def binding_types(S, ctx, delta, grad):
    # Per row and constraint type: does it bind at this iterate (the gradient pushes against it, or the
    # projection changed the proposal)? Only non-frozen controls count, except "frozen" itself.
    I = S.idx
    F = len(S.feats)
    col = lambda names: torch.tensor(np.isin(S.feats, names), device=S.dev)
    ctrl = col(CONTROLS)
    frozen = torch.from_numpy(S.frozen).to(S.dev) & ctrl[None, :]
    live = ctrl[None, :] & ~frozen
    d = torch.minimum(torch.maximum(delta, ctx.lo), ctx.hi)
    gd = grad * ctx.budget
    addonly = col([f for f in CONTROLS if SPEC[f]["direction"] == "add-only"])
    either = col([f for f in CONTROLS if SPEC[f]["direction"] == "either"])
    integer = col([f for f in CONTROLS if SPEC[f]["integer"]])
    at_hi = live & (d >= ctx.hi - 1e-9) & (gd > 0)
    at_lo = live & (d <= ctx.lo + 1e-9) & (gd < 0)
    out = {"budget_box": (at_hi | (at_lo & either[None, :])).any(1),
           "add_only": (at_lo & addonly[None, :]).any(1),
           "frozen_class_or_rule": (frozen & (grad != 0)).any(1)}
    change = d * ctx.budget
    out["integrality_blocked"] = (live & integer[None, :] & (change.abs() > 1e-9) & (change.abs() < 1)).any(1)
    xp = ctx.r + change
    xt = torch.where(integer[None, :], ctx.r + torch.trunc(xp - ctx.r), xp)
    xr = repair(xp, ctx)
    adj = live & ((xr - xt).abs() > 1e-9 * (1 + ctx.r.abs()))
    out["std_bounds"] = (adj & col(STD_CTRL)[None, :]).any(1)
    out["mean_feasibility_family"] = (adj & col(FAMILY_CTRL)[None, :]).any(1)
    out["coupling_hdr_psh_ack_window"] = (adj & col(COUPLING_CTRL)[None, :]).any(1)
    _, scale = project(d, ctx)
    out["backtracking"] = scale < 1
    proj_any = out["std_bounds"] | out["mean_feasibility_family"] | out["coupling_hdr_psh_ack_window"] | \
        out["integrality_blocked"] | out["backtracking"]
    out["projection_binding_any"] = proj_any
    out["any_binding"] = proj_any | out["budget_box"] | out["add_only"]
    # Per control: share of rows where that control sits at its budget box in the gradient direction.
    per_ctrl = {S.feats[k]: (at_hi | (at_lo & either[None, :]))[:, k].cpu().numpy() for k in range(F) if S.feats[k] in CONTROLS}
    return {k: v.cpu().numpy() for k, v in out.items()}, per_ctrl


def group_masks(S):
    f = np.array(S.feats)
    ctrl = np.isin(f, CONTROLS)
    bwd = np.char.startswith(f.astype(str), "bwd_")
    flags = np.isin(f, ["flag_syn", "flag_fin", "flag_rst"])
    return {"a_controls_only": (ctrl, False),
            "b_controls_derived_recomputed": (ctrl, True),
            "c_plus_bwd": (ctrl | bwd, True),
            "d_plus_syn_fin_rst": (ctrl | bwd | flags, True),
            "e_all_72": (np.ones(len(f), bool), False)}


def run_group_ablation(S, c, group, eps):
    # UNCONSTRAINED A1 in standardised space restricted to a feature group. With recompute=True the perturbed
    # features are mapped back to raw (non-negative where the clean value is), derived features are
    # recomputed from their bases, and the row is re-preprocessed.
    msk, recompute = group_masks(S)[group]
    m = torch.from_numpy(msk).to(S.dev)
    lo = torch.where(m, -eps, 0.0).to(torch.float64).expand_as(S.z0_64).contiguous()
    hi = torch.where(m, eps, 0.0).to(torch.float64).expand_as(S.z0_64).contiguous()
    clip = S.pp.clip
    proj = lambda d: (S.z0_64 + d).clamp(-clip, clip)
    if recompute:
        ctx = make_context(S.R, S.feats, S.floors, 0.0, S.frozen, S.dev)
        r = ctx.r

        def to_model(zv):
            raw = torch.where(m[None, :], torch.maximum(S.tp.inverse(zv), torch.minimum(r, torch.zeros_like(r))), r)
            return S.tp(derive(raw, ctx)).float()
    else:
        to_model = lambda zv: zv.float()

    def fwd(zv):
        z, attns = S.model(to_model(zv), return_attn=True)
        return z, rollout_cls(attns)

    obj = make_objective("A1", S.y, S.t, S.top, S.bot, c, 0.0)
    zz = pgd(fwd, proj, lo, hi, torch.ones_like(lo), obj, c["steps"], c["alpha0"] * eps, c["restarts"],
             gen_for(c, "abl", group, eps))
    with torch.no_grad():
        return to_model(zz)


def part_binding(S, c, out, log):
    rows, per_ctrl_rows, abl = [], [], []
    for eps in c["binding_eps"]:
        ctx = make_context(S.R, S.feats, S.floors, eps, S.frozen, S.dev)
        obj = make_objective("A1", S.y, S.t, S.top, S.bot, c, 0.0)
        x, d, g = pgd_instrumented(fwd_raw(S, ctx), lambda dd: project(dd, ctx)[0], ctx.lo, ctx.hi, ctx.budget,
                                   obj, c["steps"], c["alpha0"], c["restarts"], gen_for(c, "bind", eps))
        with torch.no_grad():
            ev = light_eval(S, c, S.tp(x).float(), "A1")
        bt, pc = binding_types(S, ctx, d, g)
        rows.append(per_class_table(S, {**bt, "a1_success": ev["success"]}, eps=eps))
        per_ctrl_rows.append(per_class_table(S, pc, eps=eps))
        log.info(f"binding eps={eps}: A1 success {ev['success'].mean():.3f}")
    bind = pd.concat(rows, ignore_index=True)
    bind.to_csv(out / "s4f_binding_by_class.csv", index=False)
    pd.concat(per_ctrl_rows, ignore_index=True).to_csv(out / "s4f_binding_budget_box_per_control.csv", index=False)
    for eps in c["ablation_eps"]:
        for group in group_masks(S):
            z = run_group_ablation(S, c, group, eps)
            ev = light_eval(S, c, z, "A1")
            abl.append(per_class_table(S, {"success": ev["success"]}, eps_std=eps, group=group))
            log.info(f"ablation eps={eps} {group}: A1 success {ev['success'].mean():.3f}")
    abl_df = pd.concat(abl, ignore_index=True)
    abl_df.to_csv(out / "s4f_group_ablation.csv", index=False)
    return bind, abl_df


# ---------------------------------------------------------------- part 2: integer budget artefact

def part_intbudget(S, c, out, log):
    I = S.idx
    int_ctrl = [f for f in CONTROLS if SPEC[f]["integer"]]
    share = []
    for eps in c["intbudget_eps"]:
        ctx = make_context(S.R, S.feats, S.floors, eps, S.frozen, S.dev)
        B = ctx.budget.cpu().numpy()
        live = ~S.frozen
        for name, feats in (("all_integer_controls", int_ctrl), ("count_features", c["count_features"])):
            k = [I[f] for f in feats]
            blocked = (B[:, k] < 1) & live[:, k]
            denom = live[:, k].sum(1).clip(min=1)
            rec = {"share_budget_below_1": blocked.sum(1) / denom,
                   **{f"below1_{f}": np.where(live[:, I[f]], B[:, I[f]] < 1, np.nan) for f in feats}}
            share.append(per_class_table(S, rec, eps=eps, feature_set=name))
    share_df = pd.concat(share, ignore_index=True)
    share_df.to_csv(out / "s4f_integer_budget_share.csv", index=False)

    def mod(ctx):
        for f in c["count_features"]:
            k = I[f]
            ctx.budget[:, k] = torch.maximum(ctx.budget[:, k], torch.ones_like(ctx.budget[:, k]))

    res = []
    for eps in c["intbudget_eps"]:
        for atk, kind, lam in (("A0", "A0", 0.0), ("A1", "A1", 0.0), ("A3_lam1", "A3", 1.0)):
            for variant, cm in (("original", None), ("count_budget_min1", mod)):
                x, z, ctx = run_constrained(S, c, kind, eps, lam, ctx_mod=cm)
                X = x.cpu().numpy()
                v, viol = validity(S, X, eps, ctx.budget.cpu().numpy())
                ev = light_eval(S, c, z, kind)
                cnt_changed = (X[:, [I[f] for f in c["count_features"]]] != S.R[:, [I[f] for f in c["count_features"]]]).any(1)
                res.append(per_class_table(S, {"success": ev["success"], "valid": v, "count_feature_changed": cnt_changed},
                                           eps=eps, attack=atk, variant=variant))
                log.info(f"intbudget eps={eps} {atk} {variant}: success {ev['success'].mean():.3f} valid {v.mean():.3f}"
                         f" violations {[k for k, vv in viol.items() if vv.any()]}")
    res_df = pd.concat(res, ignore_index=True)
    res_df.to_csv(out / "s4f_integer_budget_variant.csv", index=False)
    return share_df, res_df


# ---------------------------------------------------------------- part 3: eps 1.0 and 2.0

def part_eps_ext(S, c, out, log):
    res = []
    for eps in c["ext_eps"]:
        for atk, kind, lam in (("A0", "A0", 0.0), ("A1", "A1", 0.0), ("A2", "A2", 0.0), ("A3_lam1", "A3", 1.0),
                               ("A3_lam10", "A3", 10.0)):
            x, z, ctx = run_constrained(S, c, kind, eps, lam)
            X = x.cpu().numpy()
            v, viol = validity(S, X, eps)
            ev = light_eval(S, c, z, kind)
            res.append(per_class_table(S, {**{k: ev[k] for k in ("success", "benign_pred", "mass_ratio_rollout",
                                                                  "mass_ratio_attn_mean")},
                                           "valid": v, "n_ctrl_changed": (X != S.R)[:, S.ctrl_mask].sum(1)},
                                       eps=eps, attack=atk))
            log.info(f"eps_ext eps={eps} {atk}: success {ev['success'].mean():.3f} valid {v.mean():.3f} "
                     f"violations {[k for k, vv in viol.items() if vv.any()]}")
    df = pd.concat(res, ignore_index=True)
    df.to_csv(out / "s4f_eps_extension.csv", index=False)
    return df


# ---------------------------------------------------------------- part 4: attention stiffness

def gradnorm_trace(S, c, eps, mode):
    # A3 lambda 1 trajectory; at every step the gradients of the margin term e = logit_benign - max attack logit
    # and of the attention term L (rollout mass top-k minus bottom-k) are taken separately, in step units
    # (constrained: d/d delta = grad x budget over live controls; unconstrained: d/d z).
    lam, kappa = 1.0, c["a3_kappa"]
    if mode == "constrained":
        ctx = make_context(S.R, S.feats, S.floors, eps, S.frozen, S.dev)
        proj = lambda d: project(d, ctx)[0]
        fwd, lo, hi, scale = fwd_raw(S, ctx), ctx.lo, ctx.hi, ctx.budget
        a0 = c["alpha0"]
    else:
        lo, hi = torch.full_like(S.z0_64, -eps), torch.full_like(S.z0_64, eps)
        proj = lambda d: (S.z0_64 + d).clamp(-S.pp.clip, S.pp.clip)
        fwd, scale, a0 = fwd_z(S), torch.ones_like(lo), c["alpha0"] * eps
    delta = torch.zeros_like(lo)
    recs = []
    for step in range(c["steps"]):
        xv = proj(delta).detach().requires_grad_(True)
        z, roll = fwd(xv)
        e = z[:, 0] - z[:, 1:].amax(1)
        L, _ = attn_term(roll, S.top, S.bot)
        ge, = torch.autograd.grad(e.sum(), xv, retain_graph=True)
        gl, = torch.autograd.grad(L.sum(), xv)
        ge, gl = (ge.double() * scale), (gl.double() * scale)
        ne, nl = ge.norm(dim=1), gl.norm(dim=1)
        cos = (ge * gl).sum(1) / (ne * nl).clamp_min(1e-30)
        # Coordinates whose combined step sign is decided by the attention term (it flips the margin sign).
        comb = torch.where((e < kappa)[:, None], ge, torch.zeros_like(ge)) - lam * gl
        flips = ((torch.sign(comb) != torch.sign(ge)) & (ge != 0)).double().sum(1) / (ge != 0).double().sum(1).clamp_min(1)
        recs.append(pd.DataFrame({"step": step, "class": S.cls_rows, "norm_margin": ne.cpu().numpy(),
                                  "norm_attn": nl.cpu().numpy(), "cos_margin_attn": cos.cpu().numpy(),
                                  "share_sign_set_by_attn": flips.cpu().numpy(), "e": e.detach().cpu().numpy()}))
        a_t = a0 * 0.5 * (1 + np.cos(np.pi * step / c["steps"]))
        delta = torch.minimum(torch.maximum(delta + a_t * torch.sign(comb), lo), hi)
    df = pd.concat(recs, ignore_index=True)
    df["ratio_attn_over_margin"] = df["norm_attn"] / df["norm_margin"].clip(lower=1e-30)
    df["mode"], df["eps"] = mode, eps
    return df


def part_stiffness(S, c, out, log):
    traces = []
    for mode in ("constrained", "unconstrained"):
        for eps in c["stiff_eps"]:
            traces.append(gradnorm_trace(S, c, eps, mode))
    tr = pd.concat(traces, ignore_index=True)
    q = lambda p: (lambda v: v.quantile(p))
    gsum = tr.groupby(["mode", "eps"]).agg(ratio_median=("ratio_attn_over_margin", "median"),
                                           ratio_p10=("ratio_attn_over_margin", q(0.1)),
                                           ratio_p90=("ratio_attn_over_margin", q(0.9)),
                                           norm_margin_median=("norm_margin", "median"),
                                           norm_attn_median=("norm_attn", "median"),
                                           cos_median=("cos_margin_attn", "median"),
                                           share_sign_set_by_attn=("share_sign_set_by_attn", "mean")).reset_index()
    gsum.to_csv(out / "s4f_gradnorm_summary.csv", index=False)
    tr.groupby(["mode", "eps", "class"])[["ratio_attn_over_margin", "share_sign_set_by_attn"]].median().reset_index() \
        .to_csv(out / "s4f_gradnorm_by_class.csv", index=False)
    log.info("Gradient norms, attention term vs margin term (A3 lambda 1):\n" + gsum.round(4).to_string(index=False))
    pure = []
    for which in ("rollout", "mean"):
        obj = pure_attention_objective(S.top)
        for eps in c["pure_eps"]:
            x, z, ctx = run_constrained(S, c, "PA", eps, which=which, objective=obj)
            v, _ = validity(S, x.cpu().numpy(), eps)
            ev = light_eval(S, c, z, "PA")
            pure.append(per_class_table(S, {"mass_ratio": ev[f"mass_ratio_{'rollout' if which == 'rollout' else 'attn_mean'}"],
                                            "flipped": ev["flipped"], "valid": v}, mode="constrained", eps=eps,
                                        attention=which))
            z = run_unconstrained(S, c, "PA", eps, which=which, objective=obj)
            ev = light_eval(S, c, z, "PA")
            pure.append(per_class_table(S, {"mass_ratio": ev[f"mass_ratio_{'rollout' if which == 'rollout' else 'attn_mean'}"],
                                            "flipped": ev["flipped"], "valid": np.zeros(len(S.pos))},
                                        mode="unconstrained", eps=eps, attention=which))
            log.info(f"pure attention {which} eps={eps}: mass ratio constrained {pure[-2]['mass_ratio'].iloc[-1]:.3f}, "
                     f"unconstrained {pure[-1]['mass_ratio'].iloc[-1]:.3f}")
    pure_df = pd.concat(pure, ignore_index=True)
    pure_df.to_csv(out / "s4f_pure_attention_attack.csv", index=False)
    return gsum, pure_df


def part_sensitivity(S, c, out, log):
    # Clean Stage 3 set (all rows, all classes), stored attributions, every seed. Each row's importances are
    # normalised to sum 1 (|.| for signed methods). Class means include the row itself (n >= 144 per class).
    sel = S.sel
    y = sel["label_id"].to_numpy()
    rows, var_rows = [], []
    for s in (0, 1, 2):
        st = np.load(S.s3 / f"attr_seed{s}.npz")
        for m in ("attn_rollout", "attn_mean", "attn_last", "ig_zero", "occlusion", "gshap"):
            A = np.abs(st[m].astype(np.float64))
            A = A / A.sum(1, keepdims=True).clip(min=1e-30)
            M = np.stack([A[y == ci].mean(0) for ci in range(len(S.classes))])
            My = M[y]
            cos = (A * My).sum(1) / (np.linalg.norm(A, axis=1) * np.linalg.norm(My, axis=1)).clip(min=1e-30)
            sp = rowwise_spearman(A, My)
            rows += class_rows(cos, y, S.classes, seed=s, method=m, metric="cosine_row_vs_class_mean")
            rows += class_rows(sp, y, S.classes, seed=s, method=m, metric="spearman_row_vs_class_mean")
            ss_tot = ((A - A.mean()) ** 2).sum()
            ss_feat = ((A - A.mean(0)) ** 2).sum()
            ss_cf = ((A - My) ** 2).sum()
            var_rows.append({"seed": s, "method": m, "class": "all",
                             "share_feature_identity": 1 - ss_feat / ss_tot,
                             "share_class_x_feature": 1 - ss_cf / ss_tot, "share_row_specific": ss_cf / ss_tot})
            for ci, cn in enumerate(S.classes):
                Ac = A[y == ci]
                tot_c = ((Ac - Ac.mean()) ** 2).sum()
                var_rows.append({"seed": s, "method": m, "class": cn, "share_feature_identity": np.nan,
                                 "share_class_x_feature": 1 - ((Ac - M[ci]) ** 2).sum() / tot_c,
                                 "share_row_specific": ((Ac - M[ci]) ** 2).sum() / tot_c})
    sens, var = pd.DataFrame(rows), pd.DataFrame(var_rows)
    sens.to_csv(out / "s4f_attention_sensitivity.csv", index=False)
    var.to_csv(out / "s4f_attention_variance_share.csv", index=False)
    v = var[var["class"] == "all"].groupby("method")[["share_feature_identity", "share_class_x_feature",
                                                      "share_row_specific"]].mean()
    s_ = sens[sens["class"] == "macro_attack_classes"].groupby(["method", "metric"])["mean"].mean().unstack()
    log.info("Variance shares (all rows, mean over seeds):\n" + v.round(3).to_string())
    log.info("Row vs class-mean similarity (macro attack classes, mean over seeds):\n" + s_.round(3).to_string())
    return sens, var


def part_static(S, c, out, log):
    # STATIC attention: train-set class-mean rollout (and attn_mean) of the true class, identical for every row of
    # the class, evaluated with the Stage 3 masking baselines and metrics on the clean Stage 3 set.
    sel = S.sel
    y = sel["label_id"].to_numpy().astype(np.int64)
    test = pd.read_parquet(S.s1 / "test.parquet", columns=S.feats)
    Xs = torch.from_numpy(S.pp.transform(test.iloc[sel["test_row"].to_numpy()])).to(S.dev)
    del test
    train = pd.read_parquet(S.s1 / "train.parquet", columns=S.feats + ["label_id"])
    ytr = train["label_id"].to_numpy()
    rng = np.random.default_rng(c["rng_seed"] + 500)
    per = c["static_train_rows_per_class"]
    tr_idx = {ci: rng.choice(np.flatnonzero(ytr == ci), size=min(per, int((ytr == ci).sum())), replace=False)
              for ci in range(len(S.classes))}
    mask_rows = np.load(S.s3 / "baseline_rows.npz")["mask_benign_train_rows"]
    bases = {"train_mean": torch.zeros_like(Xs), "benign_sample": torch.from_numpy(S.pp.transform(train.iloc[mask_rows])).to(S.dev)}
    Ztr = {ci: torch.from_numpy(S.pp.transform(train.iloc[ix])).to(S.dev) for ci, ix in tr_idx.items()}
    del train
    target = torch.from_numpy(y).to(S.dev)
    c3 = {"eval_rows": c["eval_rows"]}
    faith, sim = [], []
    for s in (0, 1, 2):
        model, _ = load_ftt(S.s2 / f"ftt_seed{s}.pt", S.dev)
        static = {"rollout": [], "attn_mean": []}
        for ci in range(len(S.classes)):
            att, _ = attention_variants(model, Ztr[ci], c["attn_batch"])
            static["rollout"].append(att["attn_rollout"].mean(0))
            static["attn_mean"].append(att["attn_mean"].mean(0))
        dyn, _ = attention_variants(model, Xs, c["attn_batch"])
        scores = {"static_rollout": np.stack(static["rollout"])[y], "static_attn_mean": np.stack(static["attn_mean"])[y],
                  "attn_rollout": dyn["attn_rollout"], "attn_mean": dyn["attn_mean"]}
        for k in ("rollout", "attn_mean"):
            dk = "attn_rollout" if k == "rollout" else "attn_mean"
            sim += class_rows(rowwise_spearman(scores[f"static_{k}"], scores[dk]), y, S.classes, seed=s,
                              method=f"static_{k}", metric="spearman_static_vs_dynamic")
        rng_s = np.random.default_rng(c["rng_seed"] + 600 + s)
        for mname, sc in scores.items():
            order = torch.from_numpy(ranking(sc, rng_s)).to(S.dev)
            for bname, base in bases.items():
                cur = perturbation_curves(model, Xs, base, order, target, c3)
                for metric, vals in curve_metrics(cur, c["stage3_topk"], c["aopc_steps"], c["norm_min_gap"]).items():
                    faith += class_rows(vals, y, S.classes, normalised="_norm" in metric, seed=s, mask_baseline=bname,
                                        method=mname, metric=metric)
        log.info(f"static baseline seed {s} done")
        del model
        torch.cuda.empty_cache()
    fdf, sdf = pd.DataFrame(faith), pd.DataFrame(sim)
    fdf.to_csv(out / "s4f_static_attention_faithfulness.csv", index=False)
    sdf.to_csv(out / "s4f_static_vs_dynamic_similarity.csv", index=False)
    # Comparison with Stage 3 (same code path; Stage 3 rows for the other methods).
    s3 = pd.read_csv(resolve("results/stage3_faithfulness.csv"))
    keep = ["del_auc", "ins_auc", "aopc_gap", "comp@5", "suff@20", "del_auc_margin", "ins_auc_margin"]
    H = "macro_attack_classes"
    a = fdf[(fdf["class"] == H) & fdf.metric.isin(keep)].groupby(["mask_baseline", "method", "metric"])["mean"].mean()
    b = s3[(s3["class"] == H) & s3.metric.isin(keep) & s3.method.isin(
        ["ig_zero", "occlusion", "abs_x", "attn_rollout", "attn_mean", "attn_last", "random"])].groupby(
        ["mask_baseline", "method", "metric"])["mean"].mean()
    b.index = b.index.set_levels([f"stage3:{m}" for m in b.index.levels[1]], level=1)
    cmp = pd.concat([a, b]).unstack("metric")[keep].reset_index()
    cmp.to_csv(out / "s4f_static_attention_comparison.csv", index=False)
    log.info("Static attention baseline vs dynamic (macro attack classes, mean over seeds):\n"
             + cmp.round(4).to_string(index=False))
    return cmp


# ---------------------------------------------------------------- part 5: reporting (re-analysis of the pilot)

def part_reporting(S, c, out, log):
    P = resolve("results/stage4/pilot")
    rows = pd.read_parquet(P / "stage4_pilot_rows.parquet")
    faith = pd.read_parquet(P / "stage4_pilot_faith_rows.parquet")
    ben = 0

    def subset(r):
        a = r["attack"]
        if a.startswith(("A1_", "A3")) or a == "A1":
            return "flipped" if r["adv_pred"] == ben else "not_flipped"
        if a == "A1b":
            if r["adv_pred"] == r["a1b_target"]:
                return "to_target"
            if r["adv_pred"] == ben:
                return "to_benign"
            return "unchanged" if r["adv_pred"] == r["y"] else "to_other_attack"
        return "all_only"

    rows["subset"] = rows.apply(subset, axis=1)
    keys = ["seed", "pos", "mode", "attack", "eps"]
    # A1b destinations.
    a1b = rows[rows.attack == "A1b"]
    dest = (a1b.groupby(["mode", "eps"])["subset"].value_counts(normalize=True).unstack(fill_value=0).reset_index())
    dest_c = (a1b.groupby(["mode", "eps", "class"])["subset"].value_counts(normalize=True).unstack(fill_value=0)
              .reset_index())
    dest.to_csv(out / "s4f_a1b_destinations.csv", index=False)
    dest_c.to_csv(out / "s4f_a1b_destinations_by_class.csv", index=False)
    log.info("A1b destination shares:\n" + dest.round(3).to_string(index=False))
    metrics = [m for m in MARGIN_METRICS + PROB_METRICS if m in faith.columns]
    adv = faith[faith.attack != "clean"].merge(rows[keys + ["subset"]], on=keys, how="left")
    for m in metrics:
        if m.endswith("_norm"):
            adv.loc[adv["benign_pred"].astype(bool), m] = np.nan
    clean = faith[faith.attack == "clean"].set_index(["seed", "pos", "mask_baseline", "method"])[metrics]
    cl = clean.loc[pd.MultiIndex.from_frame(adv[["seed", "pos", "mask_baseline", "method"]])].to_numpy()
    D = adv[metrics].to_numpy() - cl
    recs = []
    for kk, g in adv.groupby(["mode", "attack", "eps", "mask_baseline", "method"]):
        idx = g.index.to_numpy()
        subs = [("all", np.ones(len(g), bool))] + [(s_, (g["subset"] == s_).to_numpy())
                                                   for s_ in sorted(g["subset"].unique()) if s_ != "all_only"]
        for sname, msk in subs:
            if msk.sum() == 0:
                continue
            flipped_like = sname in FLIPPED_LIKE
            for j, m in enumerate(metrics):
                is_margin = m in MARGIN_METRICS
                role = ("primary" if is_margin else "secondary") if flipped_like else \
                       ("primary" if not is_margin else "secondary")
                pos_in = np.flatnonzero(msk)
                vals = g[m].to_numpy()[pos_in]
                dvals = D[adv.index.get_indexer(idx[pos_in]), j]
                yv = g["y"].to_numpy()[pos_in]
                for q, v in (("value", vals), ("delta_vs_clean", dvals)):
                    r = class_rows(v, yv, S.classes, mode=kk[0], attack=kk[1], eps=kk[2], mask_baseline=kk[3],
                                   method=kk[4], subset=sname, metric=m, role=role, quantity=q)
                    recs.append(r[-2])           # macro over attack classes only (per-class rows are in the pilot CSVs)
    tab = pd.DataFrame(recs)
    tab = tab.rename(columns={"mean": "macro_mean", "std": "macro_std_over_classes", "n": "n_classes"})
    a0 = tab[(tab.attack == "A0") & (tab.subset == "all") & (tab.quantity == "delta_vs_clean")].set_index(
        ["mode", "eps", "mask_baseline", "method", "metric"])["macro_mean"]
    k = pd.MultiIndex.from_frame(tab[["mode", "eps", "mask_baseline", "method", "metric"]])
    tab["A0_delta"] = a0.reindex(k).to_numpy()
    tab["excess_vs_A0"] = np.where(tab.quantity == "delta_vs_clean", tab["macro_mean"] - tab["A0_delta"], np.nan)
    tab.to_csv(out / "s4f_faithfulness_primary_secondary.csv", index=False)
    show = tab[(tab["mode"] == "constrained") & (tab.mask_baseline == "train_mean") & (tab.eps == 0.5)
               & (tab.quantity == "delta_vs_clean") & (tab.role == "primary")
               & tab.method.isin(["attn_rollout", "ig_zero", "occlusion", "random"])
               & tab.metric.isin(["aopc_gap_margin", "del_auc_margin", "aopc_gap", "del_auc"])]
    log.info("Primary metrics, change vs clean minus A0 (constrained, eps 0.5, train-mean):\n"
             + show.pivot_table(index=["attack", "subset"], columns=["metric", "method"], values="excess_vs_A0")
             .round(3).to_string())
    return tab, dest


# ---------------------------------------------------------------- part 6: cost

def part_cost(S, c, out, log, dev):
    # Component costs measured at full-run size (n_per_class = 150 -> all classes capped), seed 0.
    Sf = setup(c, dev, c["cost_n_per_class"], c["seed"])
    N = len(Sf.pos)

    def tm(fn):
        torch.cuda.synchronize()
        torch.cuda.reset_peak_memory_stats()
        t0 = time.time()
        r = fn()
        torch.cuda.synchronize()
        return time.time() - t0, torch.cuda.max_memory_allocated() / 2**30, r

    cs = dict(c)
    cs["steps"], cs["restarts"] = c["cost_steps"], 0
    meas = {}
    t, mem, _ = tm(lambda: run_constrained(Sf, cs, "A1", 0.5))
    meas["pgd_step_constrained"] = (t / (cs["steps"] + 1), mem)
    t, mem, _ = tm(lambda: run_unconstrained(Sf, cs, "A1", 0.5))
    meas["pgd_step_unconstrained"] = (t / (cs["steps"] + 1), mem)
    Z = Sf.Z0
    t, mem, _ = tm(lambda: attention_variants(Sf.model, Z, c["attn_batch"]))
    meas["attention"] = (t, mem)
    t, mem, _ = tm(lambda: integrated_gradients(Sf.model, Z, Sf.y, torch.zeros(1, Z.shape[1], device=dev), c))
    meas["ig_zero"] = (t, mem)
    t, mem, _ = tm(lambda: occlusion(Sf.model, Z, Sf.y, torch.zeros_like(Z), c))
    meas["occlusion"] = (t, mem)
    order = torch.from_numpy(ranking(np.random.default_rng(0).random(Z.shape), np.random.default_rng(1))).to(dev)
    t, mem, _ = tm(lambda: perturbation_curves(Sf.model, Z, torch.zeros_like(Z), order, Sf.y, c))
    meas["curves_one_ranking_one_baseline"] = (t, mem)
    t, mem, _ = tm(lambda: check_validity(Sf.R, Sf.R, Sf.feats, Sf.spec, 0.5, Sf.cls_rows))
    meas["validity_check"] = (t, mem)
    mdf = pd.DataFrame([{"component": k, "seconds_at_N": v[0], "N_rows": N, "peak_gpu_gb": v[1]} for k, v in meas.items()])
    mdf.to_csv(out / "s4f_cost_components.csv", index=False)
    log.info("Measured component costs:\n" + mdf.round(3).to_string(index=False))
    T = {k: v[0] for k, v in meas.items()}
    steps_total = c["steps"] + 1

    def estimate(cfgx):
        # Seconds for one seed: per setting attack (PGD steps x runs, A0 free) + evaluation; plus clean once.
        ev_one = T["attention"] + T["ig_zero"] + T["occlusion"] + T["validity_check"]
        clean = ev_one + cfgx["n_rank"] * 2 * T["curves_one_ranking_one_baseline"]   # clean keeps both baselines
        tot = clean
        for mode, attacks in (("constrained", cfgx["constrained"]), ("unconstrained", cfgx["unconstrained"])):
            step = T[f"pgd_step_{mode}"]
            for a in attacks:
                for eps in cfgx[f"eps_{mode}"]:
                    att = 0.0 if a == "A0" else step * steps_total * (1 + cfgx["restarts"])
                    tot += att + ev_one + cfgx["n_rank"] * cfgx["n_base_adv"] * T["curves_one_ranking_one_baseline"]
        return tot * 3 / 3600

    E5 = [0.02, 0.05, 0.1, 0.25, 0.5]
    E7 = E5 + [1.0, 2.0]
    A3 = ["A3_lam0.1", "A3_lam1", "A3_lam10"]
    cands = [
        ("C0 as piloted", dict(constrained=["A0", "A1", "A1b", "A2"] + A3, unconstrained=["A0", "A1", "A1b", "A2"],
                               eps_constrained=E5, eps_unconstrained=E5, n_rank=7, n_base_adv=2, restarts=1)),
        ("C1 C0 + train-mean only for adversarial rows", dict(constrained=["A0", "A1", "A1b", "A2"] + A3,
                                                              unconstrained=["A0", "A1", "A1b", "A2"], eps_constrained=E5,
                                                              eps_unconstrained=E5, n_rank=7, n_base_adv=1, restarts=1)),
        ("C2 C1 + constrained eps 1.0 and 2.0", dict(constrained=["A0", "A1", "A1b", "A2"] + A3,
                                                     unconstrained=["A0", "A1", "A1b", "A2"], eps_constrained=E7,
                                                     eps_unconstrained=E5, n_rank=7, n_base_adv=1, restarts=1)),
        ("C3 C2 - A3 lambda 0.1 - constrained eps 0.02 - unconstrained eps 0.02",
         dict(constrained=["A0", "A1", "A1b", "A2", "A3_lam1", "A3_lam10"], unconstrained=["A0", "A1", "A1b", "A2"],
              eps_constrained=E7[1:], eps_unconstrained=E5[1:], n_rank=7, n_base_adv=1, restarts=1)),
        ("C4 C3 - unconstrained A0 and A1b",
         dict(constrained=["A0", "A1", "A1b", "A2", "A3_lam1", "A3_lam10"], unconstrained=["A1", "A2"],
              eps_constrained=E7[1:], eps_unconstrained=E5[1:], n_rank=7, n_base_adv=1, restarts=1)),
    ]
    est = pd.DataFrame([{"config": n, "hours_3_seeds": estimate(cf), **{k: str(v) for k, v in cf.items()}}
                        for n, cf in cands])
    est["under_budget"] = est["hours_3_seeds"] <= c["budget_hours"]
    est.to_csv(out / "s4f_fullrun_cost_options.csv", index=False)
    log.info("Full-run configurations (3 seeds, estimated hours):\n"
             + est[["config", "hours_3_seeds", "under_budget"]].round(2).to_string(index=False))
    del Sf
    torch.cuda.empty_cache()
    return mdf, est


# ---------------------------------------------------------------- main

def main():
    cfgmod.DEFAULTS.setdefault("stage4b", {**copy.deepcopy(STAGE4), **copy.deepcopy(STAGE4B)})
    cfg, overrides = load_config("Stage 4 pilot follow-up diagnostics.")
    c = cfg["stage4b"]
    if isinstance(c["parts"], str):
        # Also accept "a,b,c" (PowerShell 5.1 strips the quotes of a JSON list on the command line).
        c["parts"] = [p.strip() for p in c["parts"].strip("[]").split(",") if p.strip()]
    out =resolve("results/stage4/pilot_followup")
    out.mkdir(parents=True, exist_ok=True)
    t0 = time.time()
    run_dir, log = start_run("stage4b_followup", cfg, overrides)
    dev = torch.device("cuda")
    S = setup(c, dev, c["n_per_class"], c["seed"])
    log.info(f"Rows: {len(S.pos)}; parts {c['parts']}")
    timing = []
    for part in c["parts"]:
        t = time.time()
        if part == "cost":
            part_cost(S, c, out, log, dev)
        else:
            globals()[f"part_{part}"](S, c, out, log)
        timing.append({"part": part, "seconds": round(time.time() - t, 1)})
        log.info(f"part {part} done in {timing[-1]['seconds']} s")
    pd.DataFrame(timing).to_csv(out / "s4f_timing.csv", index=False)
    finish_run(run_dir, log, {"parts": c["parts"]}, t0)


if __name__ == "__main__":
    main()
