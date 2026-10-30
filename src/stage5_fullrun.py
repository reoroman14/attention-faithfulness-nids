# Stage 4/5 full run (approved configuration R1 with the 2026-10-05 changes).
#
# Per FT-T seed (sequentially 0, 1, 2; each attacked on its own weights), up to 150 rows per attack class from the
# Stage 3 explanation set:
#   constrained    A0, A1, A1b, A2 at eps {0.05, 0.1, 0.25, 0.5, 1, 2}
#                  A3 lambda 100 and 1000, and A3norm (unit-normalised margin and attention gradients, attention
#                  weight w = 0.5) at eps {0.25, 0.5, 1}
#   unconstrained  A1, A2 at standardised eps {0.05, 0.1, 0.25, 0.5}; A3 lambda 1000 at {0.25, 0.5}
#                  (REFERENCE ONLY, NOT PROTOCOL-VALID)
# Evaluation on adversarial rows: success, validity (independent checker), raw perturbation, top-10 rollout /
# attn_mean mass ratio (at the best and at the last PGD iterate), explanation shift, and Stage 3 faithfulness with
# train-mean masking for 9 rankings: attn_rollout, attn_mean, attn_last, ig_zero, occlusion, random, abs_x,
# static_rollout, static_ig (static = train-set class mean of the row's true class, fixed per class; vectors from
# src/stage4c_static_control.py). Clean rows: the same 9 rankings with both masking baselines.
# eps 2.0 is a stress setting; headline tables use eps <= 1.0 and always include A1 minus A0.
#
# Resume: every (seed, setting) is written to results/stage5/seed{s}/ as soon as it finishes and is skipped on
# restart. Per-seed summaries and a PROGRESS.md entry are written when a seed completes; the final cross-seed
# entry when all seeds are done. The process asks Windows not to sleep while it runs.
#
# Usage:
#   .venv\Scripts\python.exe -m src.stage5_fullrun
import copy
import ctypes
import json
import math
import time
from datetime import datetime

import numpy as np
import pandas as pd
import torch

from src import config as cfgmod
from src.config import load_config
from src.stage3_explain import class_rows, jaccard, ranking, rowwise_spearman
from src.stage4_attack import STAGE4, explain, faith_metrics, make_objective, other_max, rel_perturbation
from src.stage4_constraints import CONTROLS, DERIVED, make_context, project
from src.stage4b_followup import attn_vec, fwd_raw, fwd_z, mass, setup, validity
from src.stage3_explain import perturbation_curves
from src.utils import ROOT, finish_run, resolve, start_run

E6 = [0.05, 0.1, 0.25, 0.5, 1.0, 2.0]
E3 = [0.25, 0.5, 1.0]
U4 = [0.05, 0.1, 0.25, 0.5]
STAGE5 = {
    "seeds": [0, 1, 2],
    "n_per_class": 150,
    # (mode, attack name, eps list)
    "settings": [["constrained", "A0", E6], ["constrained", "A1", E6], ["constrained", "A1b", E6],
                 ["constrained", "A2", E6], ["constrained", "A3_lam100", E3], ["constrained", "A3_lam1000", E3],
                 ["constrained", "A3norm_w0.5", E3],
                 ["unconstrained", "A1", U4], ["unconstrained", "A2", U4], ["unconstrained", "A3_lam1000", [0.25, 0.5]]],
    "adv_rankings": ["attn_rollout", "attn_mean", "attn_last", "ig_zero", "occlusion", "random", "abs_x",
                     "static_rollout", "static_ig"],
    "clean_mask_baselines": ["train_mean", "benign_sample"],
    "adv_mask_baselines": ["train_mean"],
    "headline_max_eps": 1.0,
    "a3_key_offset": 1e7,
    "out_dir": "results/stage5",
    # Append per-seed and final entries to PROGRESS.md (off for smoke tests).
    "progress_log": True,
}
SHIFT = ["attn_rollout", "attn_mean", "attn_last", "ig_zero", "occlusion"]
MARGIN_M = ["del_auc_margin", "ins_auc_margin", "aopc_gap_margin", "comp_margin@5", "comp_margin@10",
            "comp_margin@20", "suff_margin@5", "suff_margin@10", "suff_margin@20"]
PROB_M = ["p_full", "margin_full", "del_auc", "ins_auc", "aopc_gap", "del_auc_norm", "ins_auc_norm", "aopc_gap_norm",
          "comp@5", "comp@10", "comp@20", "suff@5", "suff@10", "suff@20"]
FLIPPED_LIKE = {"flipped", "to_benign", "to_target", "to_other_attack"}
PROGRESS = ROOT / "PROGRESS.md"


# ---------------------------------------------------------------- keep-awake (Windows)

def keep_awake(on):
    # ES_CONTINUOUS | ES_SYSTEM_REQUIRED while running; ES_CONTINUOUS alone restores the default.
    try:
        ctypes.windll.kernel32.SetThreadExecutionState(0x80000000 | (0x00000001 if on else 0))
    except (AttributeError, OSError):
        pass


# ---------------------------------------------------------------- attacks

def parse(atk):
    kind = atk.split("_")[0]
    lam = float(atk.split("lam")[1]) if "lam" in atk else 0.0
    w = float(atk.split("_w")[1]) if "norm_w" in atk else None
    return kind, lam, w


def objective(S, c, kind, lam, w):
    # Returns f(z, a) -> (list of terms to ascend, key). One term: plain sign-gradient ascent. Two terms (A3norm):
    # each term's gradient is scaled to unit L2 norm per row and they are mixed with weights (1 - w, w).
    kappa, off = c["a3_kappa"], c["a3_key_offset"]
    if kind == "A3":
        def f(z, a):
            e = z[:, 0] - z[:, 1:].amax(1)
            L = attn_L(S, a)
            o = torch.clamp(e, max=kappa) - lam * L
            return [o], torch.where(e > 0, off + o, o)
        return f
    if kind == "A3norm":
        def f(z, a):
            e = z[:, 0] - z[:, 1:].amax(1)
            L = attn_L(S, a)
            return [torch.clamp(e, max=kappa), -L], torch.where(e > 0, off - L, e)
        return f
    base = make_objective(kind, S.y, S.t, S.top, S.bot, c, lam)
    return lambda z, a: (lambda o, k: ([o], k))(*base(z, a))


def attn_L(S, a):
    m = a / a.sum(1, keepdim=True)
    return m.gather(1, S.top).sum(1) - m.gather(1, S.bot).sum(1)


def pgd2(fwd, proj, lo, hi, gscale, obj, steps, alpha0, restarts, gen, w=None):
    # Sign-gradient PGD as in stage4_attack.pgd; returns the best projected iterate per row and the last iterate
    # of the run that produced it.
    best_key = torch.full((lo.shape[0],), float("-inf"), dtype=torch.float64, device=lo.device)
    best_x = best_last = None
    for run in range(1 + restarts):
        delta = torch.zeros_like(lo) if run == 0 else lo + (hi - lo) * torch.rand(lo.shape, generator=gen,
                                                                                   device=lo.device, dtype=lo.dtype)
        run_best = torch.zeros(lo.shape[0], dtype=torch.bool, device=lo.device)
        for step in range(steps + 1):
            x = proj(delta)
            xv = x.detach().requires_grad_(True)
            z, a = fwd(xv)
            terms, key = obj(z, a)
            key = key.detach().double()
            b = key > best_key
            run_best |= b
            best_key = torch.where(b, key, best_key)
            best_x = x.detach().clone() if best_x is None else torch.where(b[:, None], x.detach(), best_x)
            if step == steps:
                break
            if len(terms) == 1:
                g, = torch.autograd.grad(terms[0].sum(), xv)
                g = g.double() * gscale
            else:
                g1, = torch.autograd.grad(terms[0].sum(), xv, retain_graph=True)
                g2, = torch.autograd.grad(terms[1].sum(), xv)
                g1, g2 = g1.double() * gscale, g2.double() * gscale
                g1 = g1 / g1.norm(dim=1, keepdim=True).clamp_min(1e-30)
                g2 = g2 / g2.norm(dim=1, keepdim=True).clamp_min(1e-30)
                g = (1 - w) * g1 + w * g2
            a_t = alpha0 * 0.5 * (1 + math.cos(math.pi * step / steps))
            delta = torch.minimum(torch.maximum(delta + a_t * torch.sign(g), lo), hi)
        last = x.detach()
        best_last = last.clone() if best_last is None else torch.where(run_best[:, None], last, best_last)
    return best_x, best_last


def run_setting(S, c, mode, atk, eps):
    kind, lam, w = parse(atk)
    gen = torch.Generator(device=S.dev).manual_seed(c["rng_seed"] + 1000 * S.seed + int(eps * 1e4)
                                                     + sum(map(ord, mode + atk)))
    if mode == "constrained":
        ctx = make_context(S.R, S.feats, S.floors, eps, S.frozen, S.dev)
        proj = lambda d: project(d, ctx)[0]
        if kind == "A0":
            u = ctx.lo + (ctx.hi - ctx.lo) * torch.rand(ctx.lo.shape, generator=gen, device=S.dev, dtype=torch.float64)
            x = proj(u)
            last = x
        else:
            x, last = pgd2(fwd_raw(S, ctx), proj, ctx.lo, ctx.hi, ctx.budget, objective(S, c, kind, lam, w),
                           c["steps"], c["alpha0"], c["restarts"], gen, w)
        with torch.no_grad():
            z, z_last = S.tp(x).float(), S.tp(last).float()
        X = x.cpu().numpy()
    else:
        lo, hi = torch.full_like(S.z0_64, -eps), torch.full_like(S.z0_64, eps)
        proj = lambda d: (S.z0_64 + d).clamp(-S.pp.clip, S.pp.clip)
        zz, zl = pgd2(fwd_z(S), proj, lo, hi, torch.ones_like(lo), objective(S, c, kind, lam, w), c["steps"],
                      c["alpha0"] * eps, c["restarts"], gen, w)
        z, z_last = zz.float(), zl.float()
        with torch.no_grad():
            X = torch.where(zz == S.z0_64, torch.from_numpy(S.R).to(S.dev), S.tp.inverse(zz)).cpu().numpy()
    return kind, z, z_last, X


# ---------------------------------------------------------------- evaluation

def importances(S, expl, Zn, rng):
    sc = {k: expl[k] for k in ("attn_rollout", "attn_mean", "attn_last")}
    sc["ig_zero"], sc["occlusion"] = np.abs(expl["ig_zero"]), np.abs(expl["occlusion"])
    sc["abs_x"] = np.abs(Zn)
    sc["random"] = rng.random(Zn.shape)
    sc["static_rollout"], sc["static_ig"] = S.static_roll, S.static_ig
    return sc


def faith(S, c, Z, target, sc, base_names, rng):
    out = []
    for mname in c["adv_rankings"]:
        order = torch.from_numpy(ranking(sc[mname], rng)).to(S.dev)
        for b in base_names:
            cur = perturbation_curves(S.model, Z, S.bases[b], order, target, c)
            m = faith_metrics(cur, c["topk"], c["aopc_steps"], c["norm_min_gap"])
            out.append(pd.DataFrame({"pos": S.pos, "mask_baseline": b, "method": mname,
                                     **{k: v for k, v in m.items() if k in MARGIN_M + PROB_M}}))
    return pd.concat(out, ignore_index=True)


@torch.no_grad()
def masses(S, z):
    _, at = S.model(z, return_attn=True)
    return {w: mass(attn_vec(at, w), S.top).cpu().numpy() / S.mass0[w] for w in ("rollout", "mean")}


def evaluate(S, c, kind, mode, eps, z, z_last, X, rng):
    with torch.no_grad():
        lg = S.model(z)
    pred = lg.argmax(1)
    p, yv, tv = pred.cpu().numpy(), S.y_np, S.t.cpu().numpy()
    marg = (lg.gather(1, S.y[:, None]).squeeze(1) - other_max(lg, S.y)).cpu().numpy()
    mb, ml = masses(S, z), masses(S, z_last)
    if kind == "A0":
        succ = p != yv
    elif kind in ("A1", "A3", "A3norm"):
        succ = p == 0
    elif kind == "A1b":
        succ = p == tv
    else:
        succ = (p == yv) & (marg >= c["a2_tau"])            # A2: prediction preserved with margin >= tau
    a2_flag = (p == yv) & (marg >= c["a2_tau"]) & (mb["rollout"] <= 1 - c["a2_mass_drop"])
    if kind in ("A1", "A3", "A3norm"):
        sub = np.where(p == 0, "flipped", "not_flipped")
    elif kind == "A1b":
        sub = np.select([p == tv, p == 0, p == yv], ["to_target", "to_benign", "unchanged"], "to_other_attack")
    else:
        sub = np.where(p == yv, "unchanged", np.where(p == 0, "to_benign", "to_other_attack"))
    target = S.y if kind in ("A1", "A3", "A3norm") else torch.where(pred > 0, pred, S.y)
    valid, viol = validity(S, X, eps)
    allf = np.ones(len(S.feats), bool)
    rp_all, _ = rel_perturbation(X, S.R, S.floor_arr, allf)
    rp_c, rl_c = rel_perturbation(X, S.R, S.floor_arr, S.ctrl_mask)
    expl, _ = explain(S.model, z, target, c)
    sc = importances(S, expl, z.cpu().numpy(), rng)
    rec = {"pos": S.pos, "class": S.cls_rows, "y": yv, "a1b_target": tv, "adv_pred": p, "success": succ,
           "subset": sub, "a2_drop50_flag": a2_flag, "benign_pred": p == 0, "eval_target": target.cpu().numpy(),
           "margin_y": marg, "valid": valid, "rel_pert_mean_all": rp_all, "rel_pert_mean_ctrl": rp_c,
           "rel_pert_linf_ctrl": rl_c, "n_feat_changed": (X != S.R).sum(1),
           "n_ctrl_changed": (X != S.R)[:, S.ctrl_mask].sum(1),
           "mass_ratio_rollout": mb["rollout"], "mass_ratio_attn_mean": mb["mean"],
           "mass_ratio_rollout_last": ml["rollout"], "mass_ratio_attn_mean_last": ml["mean"]}
    for k, v in viol.items():
        rec[f"viol_{k}"] = v
    for m in SHIFT:
        rec[f"shift_spearman_{m}"] = rowwise_spearman(S.clean_sc[m], sc[m])
        rec[f"shift_jaccard5_{m}"] = jaccard(S.clean_orders[m], ranking(sc[m], rng), 5)
    fa = faith(S, c, z, target, sc, c["adv_mask_baselines"], rng)
    return pd.DataFrame(rec), fa


# ---------------------------------------------------------------- per-seed driver

def seed_setup(c, s, dev, log):
    S = setup(c, dev, c["n_per_class"], s)
    S.seed = s
    S.floor_arr = np.array([S.floors[f] for f in S.feats])
    sv = np.load(resolve("results/stage4") / f"static_vectors_seed{s}.npz")
    S.static_roll, S.static_ig = sv["static_rollout"][S.y_np], sv["static_ig"][S.y_np]
    # Masking baselines exactly as Stage 3.
    brow = np.load(S.s3 / "baseline_rows.npz")["mask_benign_train_rows"][S.pos]
    train = pd.read_parquet(S.s1 / "train.parquet", columns=S.feats)
    S.bases = {"train_mean": torch.zeros_like(S.Z0),
               "benign_sample": torch.from_numpy(S.pp.transform(train.iloc[brow])).to(dev)}
    del train
    return S


def clean_part(S, c, sdir, log):
    f = sdir / "clean_faith.parquet"
    st = np.load(S.s3 / f"attr_seed{S.seed}.npz")
    expl, _ = explain(S.model, S.Z0, S.y, c)
    expl["ig_zero"] = st["ig_zero"][S.pos].astype(np.float32)        # exactly what Stage 3 evaluated
    expl["occlusion"] = st["occlusion"][S.pos].astype(np.float32)
    S.clean_sc = importances(S, expl, S.Z0.cpu().numpy(), np.random.default_rng(c["rng_seed"] + S.seed))
    S.clean_orders = {k: ranking(v, np.random.default_rng(c["rng_seed"] + 7)) for k, v in S.clean_sc.items()}
    if not f.exists():
        fa = faith(S, c, S.Z0, S.y, S.clean_sc, c["clean_mask_baselines"], np.random.default_rng(c["rng_seed"] + 13))
        fa.to_parquet(f, index=False)
        log.info(f"seed {S.seed}: clean faithfulness written")


def run_seed(c, s, dev, log, out):
    sdir = out / f"seed{s}"
    sdir.mkdir(parents=True, exist_ok=True)
    S = seed_setup(c, s, dev, log)
    with torch.no_grad():
        ok = bool((S.model(S.Z0).argmax(1) == S.y).all())
    log.info(f"seed {s}: {len(S.pos)} rows, clean all predicted as label: {ok}")
    clean_part(S, c, sdir, log)
    for mode, atk, eps_list in c["settings"]:
        for eps in eps_list:
            tag = f"{mode}_{atk}_eps{eps:g}"
            if (sdir / f"rows_{tag}.parquet").exists():
                continue
            t0 = time.time()
            torch.cuda.reset_peak_memory_stats()
            kind, z, z_last, X = run_setting(S, c, mode, atk, eps)
            t_att = time.time() - t0
            rows, fa = evaluate(S, c, kind, mode, eps, z, z_last, X,
                                np.random.default_rng(c["rng_seed"] + 17 + s + int(eps * 1000)))
            info = {"seed": s, "mode": mode, "attack": atk, "eps": eps}
            for k, v in info.items():
                rows.insert(0, k, v)
                fa.insert(0, k, v)
            rows["t_attack_s"], rows["t_eval_s"] = t_att, time.time() - t0 - t_att
            rows["peak_gpu_gb"] = torch.cuda.max_memory_allocated() / 2**30
            fa.to_parquet(sdir / f"faith_{tag}.parquet", index=False)
            rows.to_parquet(sdir / f"rows_{tag}.parquet", index=False)     # written last = completion marker
            if mode == "constrained":
                np.savez_compressed(sdir / f"adv_raw_{tag}.npz", x_adv=X, pos=S.pos)
            log.info(f"seed {s} {tag}: success {rows['success'].mean():.3f} valid {rows['valid'].mean():.3f} "
                     f"mass ratio {rows['mass_ratio_rollout'].mean():.3f} [{t_att:.0f}s + {time.time() - t0 - t_att:.0f}s]")
    del S
    torch.cuda.empty_cache()


# ---------------------------------------------------------------- summaries

def load_seed(sdir):
    rows = pd.concat([pd.read_parquet(p) for p in sorted(sdir.glob("rows_*.parquet"))], ignore_index=True)
    fa = pd.concat([pd.read_parquet(p) for p in sorted(sdir.glob("faith_*.parquet"))], ignore_index=True)
    clean = pd.read_parquet(sdir / "clean_faith.parquet")
    return rows, fa, clean


def macro(df, val, by):
    # Mean per class, then unweighted mean over the attack classes present.
    g = df.groupby(by + ["class"])[val].mean()
    return g.groupby(level=list(range(len(by)))).mean()


def tables(rows, fa, clean, classes, c):
    # Returns a dict of DataFrames (macro over attack classes; mean over seeds when several seeds are present).
    T = {}
    rows = rows.copy()
    rows["success"] = rows["success"].astype(float)
    k = ["mode", "attack", "eps"]
    per_seed = lambda val: macro(rows, val, ["seed"] + k).groupby(level=[1, 2, 3])
    def ms(val):
        g = per_seed(val)
        return g.mean(), g.std()
    succ, succ_sd = ms("success")
    T["success"] = pd.DataFrame({"success": succ, "success_sd_seeds": succ_sd, "validity": ms("valid")[0],
                                 "a2_drop50_flag": ms("a2_drop50_flag")[0],
                                 "mass_ratio_rollout": ms("mass_ratio_rollout")[0],
                                 "mass_ratio_rollout_last": ms("mass_ratio_rollout_last")[0],
                                 "mass_ratio_attn_mean": ms("mass_ratio_attn_mean")[0],
                                 "benign_pred": ms("benign_pred")[0], "rel_pert_ctrl": ms("rel_pert_mean_ctrl")[0],
                                 "n_ctrl_changed": ms("n_ctrl_changed")[0]}).reset_index()
    a1 = rows[(rows.attack == "A1") & (rows["mode"] == "constrained")]
    T["a1_per_class"] = a1.groupby(["class", "eps"])["success"].mean().unstack().reset_index()
    a1b = rows[rows.attack == "A1b"]
    T["a1b_dest"] = a1b.groupby(["mode", "eps"])["subset"].value_counts(normalize=True).unstack(fill_value=0).reset_index()
    sh = {f"{m}": macro(rows, f"shift_spearman_{m}", k) for m in SHIFT}
    T["shift"] = pd.DataFrame(sh).reset_index()
    # Faithfulness change vs clean (same row, method, train-mean masking), primary metrics by row status.
    metrics = MARGIN_M + PROB_M
    cl = clean[clean.mask_baseline == "train_mean"].set_index(["seed", "pos", "method"])[metrics]
    fa = fa.merge(rows[["seed", "pos", "mode", "attack", "eps", "class", "subset", "benign_pred"]],
                  on=["seed", "pos", "mode", "attack", "eps"], how="left")
    for m in metrics:
        if m.endswith("_norm"):
            fa.loc[fa["benign_pred"].astype(bool), m] = np.nan
    d = fa[metrics].to_numpy() - cl.loc[pd.MultiIndex.from_frame(fa[["seed", "pos", "method"]])].to_numpy()
    dfa = fa[["seed", "mode", "attack", "eps", "class", "subset", "method"]].copy()
    dfa[metrics] = d
    allrows = dfa.assign(subset="all")
    dd = pd.concat([allrows, dfa[dfa.subset.isin(FLIPPED_LIKE | {"not_flipped", "unchanged"})]], ignore_index=True)
    key = ["mode", "attack", "eps", "subset", "method"]
    m_delta = dd.groupby(["seed"] + key + ["class"])[metrics].mean().groupby(["seed"] + key).mean() \
        .groupby(key).mean().reset_index()
    a0 = m_delta[(m_delta.attack == "A0") & (m_delta.subset == "all")].set_index(["mode", "eps", "method"])[metrics]
    ex = m_delta.copy()
    ref = a0.reindex(pd.MultiIndex.from_frame(ex[["mode", "eps", "method"]])).to_numpy()
    ex[metrics] = ex[metrics].to_numpy() - ref
    T["faith_delta"] = m_delta
    T["faith_excess_vs_A0"] = ex
    n = dd.groupby(key)["seed"].size().rename("n_row_obs").reset_index()
    T["faith_excess_vs_A0"] = T["faith_excess_vs_A0"].merge(n, on=key, how="left")
    T["clean_faith"] = macro(clean.assign(**{"class": clean["pos"].map(dict(zip(rows["pos"], rows["class"])))}),
                             "del_auc", ["mask_baseline", "method"]).rename("del_auc").to_frame().join(
        macro(clean.assign(**{"class": clean["pos"].map(dict(zip(rows["pos"], rows["class"])))}),
              "aopc_gap", ["mask_baseline", "method"]).rename("aopc_gap")).join(
        macro(clean.assign(**{"class": clean["pos"].map(dict(zip(rows["pos"], rows["class"])))}),
              "del_auc_margin", ["mask_baseline", "method"]).rename("del_auc_margin")).reset_index()
    t = rows.groupby(["seed", "mode", "attack", "eps"])[["t_attack_s", "t_eval_s", "peak_gpu_gb"]].first()
    T["timing"] = t.groupby(level=[1, 2, 3]).mean().reset_index()
    return T


def md(df, floatfmt="{:.3f}"):
    # Minimal markdown table (no external dependency).
    cols = [str(x) for x in df.columns]
    lines = ["| " + " | ".join(cols) + " |", "|" + "---|" * len(cols)]
    for r in df.itertuples(index=False):
        cells = []
        for v in r:
            if isinstance(v, (float, np.floating)):
                cells.append("" if np.isnan(v) else floatfmt.format(v))
            else:
                cells.append(str(v))
        lines.append("| " + " | ".join(cells) + " |")
    return "\n".join(lines)


def report(T, c, title, n_seeds):
    H = c["headline_max_eps"]
    s = T["success"]
    out = [f"\n---\n\n## {datetime.now():%Y-%m-%d %H:%M}: {title}\n"]
    out.append(f"Seeds included: {n_seeds}. Macro over the 10 attack classes"
               f"{', mean over seeds' if n_seeds > 1 else ''}. Headline tables use eps <= {H}; eps 2.0 is the stress "
               "setting (last table). Unconstrained rows are reference only, NOT protocol-valid (standardised eps).\n")
    cons = s[(s["mode"] == "constrained") & (s.eps <= H)]
    piv = lambda d, v: d.pivot_table(index="attack", columns="eps", values=v).reset_index()
    out.append("**Success rate (constrained)**\n\n" + md(piv(cons, "success")))
    out.append("\n**Validity (constrained, independent checker)**\n\n" + md(piv(cons, "validity")))
    a = piv(cons, "success")
    a0 = a[a.attack == "A0"].iloc[0]
    a1 = a[a.attack == "A1"].iloc[0]
    diff = pd.DataFrame([{"row": "A1 minus A0", **{e: a1[e] - a0[e] for e in a.columns if e != "attack"}}])
    out.append("\n**A1 minus A0 (success)**\n\n" + md(diff))
    out.append("\n**Top-10 rollout mass ratio adv/clean, best iterate (primary for A2) | last iterate**\n\n"
               + md(piv(cons, "mass_ratio_rollout")) + "\n\nLast PGD iterate:\n\n" + md(piv(cons, "mass_ratio_rollout_last"))
               + "\n\nattn_mean mass ratio (best iterate):\n\n" + md(piv(cons, "mass_ratio_attn_mean"))
               + "\n\nA2 secondary flag (>= 50% rollout mass drop, prediction kept with margin >= 1):\n\n"
               + md(piv(cons[cons.attack == "A2"], "a2_drop50_flag")))
    un = s[s["mode"] == "unconstrained"]
    out.append("\n**Unconstrained reference (success | mass ratio | validity)**\n\n"
               + md(un[["attack", "eps", "success", "mass_ratio_rollout", "mass_ratio_rollout_last", "validity"]]))
    pc = T["a1_per_class"]
    out.append("\n**A1 success per class (constrained)**\n\n"
               + md(pc[["class"] + [e for e in pc.columns if e != "class" and e <= H]]))
    out.append("\n**A1b destinations**\n\n" + md(T["a1b_dest"]))
    sh = T["shift"]
    out.append("\n**Explanation shift, Spearman clean vs adversarial (eps 0.5 and 1.0)**\n\n"
               + md(sh[sh.eps.isin([0.5, 1.0]) | ((sh["mode"] == "unconstrained") & (sh.eps == 0.5))]))
    ex = T["faith_excess_vs_A0"]
    meths = c["adv_rankings"]
    for title_, sel, cols in (("A1 all rows (A1 minus A0), probability metrics",
                         (ex.attack == "A1") & (ex.subset == "all"), ["del_auc", "aopc_gap", "comp@10", "suff@10"]),
                        ("A1 flipped rows (primary: margin metrics)", (ex.attack == "A1") & (ex.subset == "flipped"),
                         ["del_auc_margin", "aopc_gap_margin", "comp_margin@10", "suff_margin@10"]),
                        ("A1 not-flipped rows (primary: probability metrics)",
                         (ex.attack == "A1") & (ex.subset == "not_flipped"), ["del_auc", "aopc_gap", "comp@10", "suff@10"]),
                        ("A2 all rows (prediction kept; probability metrics)", (ex.attack == "A2") & (ex.subset == "all"),
                         ["del_auc", "aopc_gap", "comp@10", "suff@10"]),
                        ("A3 lambda 1000 flipped rows (margin metrics)",
                         (ex.attack == "A3_lam1000") & (ex.subset == "flipped"), ["del_auc_margin", "aopc_gap_margin"]),
                        ("A3norm w 0.5 flipped rows (margin metrics)",
                         (ex.attack == "A3norm_w0.5") & (ex.subset == "flipped"), ["del_auc_margin", "aopc_gap_margin"])):
        d = ex[sel & (ex["mode"] == "constrained") & ex.eps.isin([0.5, 1.0]) & ex.method.isin(meths)]
        if len(d):
            out.append(f"\n**Faithfulness change vs clean minus A0 change, {title_} (constrained, train-mean masking, "
                       "eps 0.5 / 1.0)**\n\n" + md(d[["eps", "method", "n_row_obs"] + cols]))
    cf = T["clean_faith"]
    out.append("\n**Clean faithfulness on these rows (reference)**\n\n" + md(cf))
    st = s[(s["mode"] == "constrained") & (s.eps == 2.0)]
    out.append("\n**Stress setting eps 2.0 (constrained)**\n\n"
               + md(st[["attack", "success", "validity", "mass_ratio_rollout", "mass_ratio_rollout_last"]]))
    tm = T["timing"]
    out.append(f"\nRuntime: attack {tm.t_attack_s.sum() / 60:.1f} min, evaluation {tm.t_eval_s.sum() / 60:.1f} min per seed "
               f"(mean over seeds); peak GPU {tm.peak_gpu_gb.max():.2f} GB.\n")
    return "\n".join(out)


def append_progress(text):
    with open(PROGRESS, "a", encoding="utf-8") as f:
        f.write(text.rstrip() + "\n")


# ---------------------------------------------------------------- main

def main():
    cfgmod.DEFAULTS.setdefault("stage5", {**copy.deepcopy(STAGE4), **copy.deepcopy(STAGE5)})
    cfg, overrides = load_config("Stage 4/5 full run.")
    c = cfg["stage5"]
    if not c["progress_log"]:
        global append_progress
        append_progress = lambda text: log.info("[progress entry, not written]\n" + text)
    out = resolve(c["out_dir"])
    out.mkdir(parents=True, exist_ok=True)
    t0 = time.time()
    run_dir, log = start_run("stage5_fullrun", cfg, overrides)
    dev = torch.device("cuda")
    keep_awake(True)
    try:
        lm = json.loads(resolve("results/stage1/label_map.json").read_text(encoding="utf-8"))
        classes = [k for k, _ in sorted(lm.items(), key=lambda kv: kv[1])]
        done = []
        for s in c["seeds"]:
            ts = time.time()
            run_seed(c, s, dev, log, out)
            done.append(s)
            sdir = out / f"seed{s}"
            rows, fa, clean = load_seed(sdir)
            T = tables(rows.assign(seed=s), fa.assign(seed=s), clean.assign(seed=s), classes, c)
            for k, v in T.items():
                v.to_csv(sdir / f"summary_{k}.csv", index=False)
            marker = sdir / "progress_logged"
            if not marker.exists():
                append_progress(report(T, c, f"Stage 5 full run: seed {s} finished "
                                             f"({(time.time() - ts) / 60:.0f} min this session)", 1))
                marker.write_text("1", encoding="utf-8")
            log.info(f"seed {s} finished and logged")
        R, Fa, Cl = [], [], []
        for s in done:
            r, f, cl = load_seed(out / f"seed{s}")
            R.append(r), Fa.append(f), Cl.append(cl.assign(seed=s))
        T = tables(pd.concat(R, ignore_index=True), pd.concat(Fa, ignore_index=True), pd.concat(Cl, ignore_index=True),
                   classes, c)
        for k, v in T.items():
            v.to_csv(out / f"summary_all_seeds_{k}.csv", index=False)
        marker = out / "final_progress_logged"
        if not marker.exists() and len(done) == len(c["seeds"]):
            append_progress(report(T, c, "Stage 5 full run: all seeds, final", len(done)))
            marker.write_text("1", encoding="utf-8")
    finally:
        keep_awake(False)
    finish_run(run_dir, log, {"seeds": c["seeds"]}, t0)


if __name__ == "__main__":
    main()
