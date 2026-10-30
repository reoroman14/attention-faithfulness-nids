# Stage 6e: a stronger attacker against D_plus (eps 1.0, seeds 0 and 2).
#
# The saved Stage 6 FINAL defenders (results/stage6/final/seed{s}/defenders_eps1.pkl), test negatives
# (negatives.pkl: 1,000 genuine benign test rows per seed) and source rows (Stage 5 rows, 150 per attack class) are
# reused unchanged; nothing is retrained.
#
# Part 1  oracle union over the existing attacks (A1, A6 x 3, A7 x 3): a source row counts as evaded under a detector
#         if ANY candidate flipped it to benign AND stayed below that detector's threshold. This is an attacker with
#         query access to the detector that keeps the best of 7 candidates (10 once A8 is added).
# Part 2  A8: white-box PGD against D_plus with differentiable stand-ins for ALL 12 consistency features
#         ({rollout, attn_mean} x {grad x input, IG} x {Spearman, top-10 Jaccard, cosine}):
#           Spearman  Pearson correlation of pairwise-sigmoid soft ranks (per-row standardised values, tau 0.1)
#           Jaccard   soft top-10 membership sigmoid((v - t) / tau), t = midpoint of the 10th and 11th largest
#                     values; soft intersection over soft union; plus 1e-3 x soft Spearman (the tie-breaker of the
#                     true feature)
#           cosine    exact
#           IG        8-step midpoint Riemann IG from the train-mean baseline (0), with double backprop
#         Loss: min(margin_benign, 10) - lambda x relu(D_plus_logit - thr + 0.5), lambda in {1, 100}, plus the
#         normalised-gradient variant (w 0.5), exactly as A6 / A7. Evaluation always uses the TRUE 12 features
#         (src/stage6b_defense.detector_scores, IG with 20 steps).
# Stand-in quality: R^2 and Spearman of each stand-in vs the true feature on seed 0 VALIDATION rows (val benign rows
# predicted benign plus A1 rows on val source rows that flip), reported before the attack.
# Stop conditions: any invalid constrained row; runtime estimate > 3 h; OOM at chunk size 50.
#
# Usage:
#   .venv\Scripts\python.exe -m src.stage6e_strong
import copy
import math
import pickle
import sys
import time
from datetime import datetime

import numpy as np
import pandas as pd
import torch
from scipy.stats import spearmanr
from sklearn.linear_model import LinearRegression

from src import config as cfgmod
from src.config import load_config
from src.stage3_explain import load_ftt
from src.stage4_attack import STAGE4, rollout_cls
from src.stage4_constraints import derive
from src.stage5_fullrun import keep_awake, md
from src.stage6_defense import BENIGN, STAGE6
from src.stage6b_defense import STAGE6B, auroc, boot, detector_scores
from src.stage6c_defense import FEATS_D, STAGE6C, a1_on, feats_torch
from src.stage6d_final import (STAGE6D, LinearDefender, StopRun, attack_loop, check_valid_or_stop,  # noqa: F401
                               load_common, paired_boot, plus_features)
from src.utils import ROOT, finish_run, resolve, start_run

STAGE6E = {
    "seeds_e": [0, 2],
    "eps_e": 1.0,
    "ig_steps_loss": 8,
    "tau": 0.1,
    "chunk_candidates": [250, 100, 50],
    "max_hours_e": 3.0,
    "n_val_benign_check": 2000,
    "estimate_rows": 500,
    "estimate_steps": 4,
    "out_dir_e": "results/stage6/strong",
    "src_dir_e": "results/stage6/final",
    # Append PROGRESS.md entries (off for smoke tests).
    "progress_log_e": True,
}
OLD = ["A1", "A6_lam1", "A6_lam100", "A6norm_w0.5", "A7_lam1", "A7_lam100", "A7norm_w0.5"]
PROGRESS = ROOT / "PROGRESS.md"


# ---------------------------------------------------------------- differentiable consistency features

def soft_rank(v, tau):
    # Pairwise-sigmoid soft ranks of per-row standardised values (n = 72, O(n^2) is cheap).
    v = (v - v.mean(1, keepdim=True)) / v.std(1, keepdim=True).clamp_min(1e-12)
    d = (v[:, :, None] - v[:, None, :]) / tau
    return torch.sigmoid(d).sum(2) + 0.5


def soft_spearman(a, b, tau):
    ra, rb = soft_rank(a, tau), soft_rank(b, tau)
    ra, rb = ra - ra.mean(1, keepdim=True), rb - rb.mean(1, keepdim=True)
    return (ra * rb).sum(1) / (ra.norm(dim=1) * rb.norm(dim=1)).clamp_min(1e-12)


def soft_topk(v, k, tau):
    # Soft top-k membership: sigmoid((v - t) / tau) on standardised values, t between the k-th and (k+1)-th largest.
    v = (v - v.mean(1, keepdim=True)) / v.std(1, keepdim=True).clamp_min(1e-12)
    top = v.topk(k + 1, dim=1).values
    t = 0.5 * (top[:, k - 1] + top[:, k])
    return torch.sigmoid((v - t[:, None]) / tau)


def soft_jaccard(a, b, k, tau):
    ma, mb = soft_topk(a, k, tau), soft_topk(b, k, tau)
    inter = (ma * mb).sum(1)
    return inter / (ma + mb - ma * mb).sum(1).clamp_min(1e-12)


def cosine(a, b):
    return (a * b).sum(1) / (a.norm(dim=1) * b.norm(dim=1)).clamp_min(1e-12)


def ig_steps(model, z, cls, m):
    # m-step midpoint Riemann IG from the zero baseline, differentiable (create_graph).
    acc = torch.zeros_like(z)
    for i in range(m):
        alpha = (i + 0.5) / m
        zi = alpha * z
        lg = model(zi)
        g, = torch.autograd.grad(lg.gather(1, cls[:, None]).sum(), zi, create_graph=True)
        acc = acc + g
    return z * acc / m


def standin_features(model, z, cls, base, c):
    # (lg, F) with F = [4 baselines, 12 consistency stand-ins] in D_plus column order:
    # for attention in (rollout, attnmean), attribution in (gxi, ig): spearman, jacc10, cos (+consistency).
    lg, attns = model(z, return_attn=True)
    A = torch.stack(attns, 1)
    ro = rollout_cls(attns)
    me = A.mean((1, 2))[:, 0, 1:]
    g, = torch.autograd.grad(lg.gather(1, cls[:, None]).sum(), z, create_graph=True)
    gxi = (z * g).abs()
    ig = ig_steps(model, z, cls, c["ig_steps_loss"]).abs()
    nrm = lambda t: t / t.sum(1, keepdim=True).clamp_min(1e-12)
    feats = []
    for att in (nrm(ro), nrm(me)):
        for attr in (nrm(gxi), nrm(ig)):
            sp = soft_spearman(att, attr, c["tau"])
            feats += [sp, soft_jaccard(att, attr, 10, c["tau"]) + 1e-3 * sp, cosine(att, attr)]
    return lg, torch.cat([feats_torch(lg, z, base), torch.stack(feats, 1)], 1)


def terms_A8(model, tp, base, Dp, c):
    def f(xv, ctx):
        z = tp(derive(xv, ctx)).float()
        cls = torch.full((z.shape[0],), BENIGN, dtype=torch.long, device=z.device)
        lg, F = standin_features(model, z, cls, base, c)
        return lg[:, BENIGN] - lg[:, 1:].amax(1), Dp.torch_decision(F)
    return f


# ---------------------------------------------------------------- helpers

def load_defenders(src, s, eps, dev):
    DEF = pickle.loads((src / f"seed{s}" / f"defenders_eps{eps:g}.pkl").read_bytes())
    for k in ("D", "Dp"):
        o = DEF[k]
        o.mu, o.sd, o.w = o.mu.to(dev), o.sd.to(dev), o.w.to(dev)
    return DEF


def union_eval(rows, names):
    # Per source row: evaded and undetected by ANY of the named candidates, for D and for D_plus.
    sub = rows[rows.attack.isin(names)]
    uD = (sub.flipped & (sub.score_D < sub.thr_D)).groupby([sub.seed, sub.pos]).any()
    uP = (sub.flipped & (sub.score_Dplus < sub.thr_Dplus)).groupby([sub.seed, sub.pos]).any()
    fl = sub.flipped.groupby([sub.seed, sub.pos]).any()
    cls = sub.groupby(["seed", "pos"])["class"].first()
    return pd.DataFrame({"union_flipped": fl, "union_undet_D": uD, "union_undet_Dplus": uP, "class": cls}).reset_index()


def paired_union(u, n, rng):
    # Paired bootstrap of mean(undet_Dplus) - mean(undet_D) over source rows, resampled within each seed.
    d = []
    groups = [g for _, g in u.groupby("seed")]
    for _ in range(n):
        parts = [g.iloc[rng.integers(0, len(g), len(g))] for g in groups]
        b = pd.concat(parts)
        d.append(b.union_undet_Dplus.mean() - b.union_undet_D.mean())
    return u.union_undet_Dplus.mean() - u.union_undet_D.mean(), np.percentile(d, 2.5), np.percentile(d, 97.5)


def attack_tables(R, negs, c):
    # Per attack: success, success AND undetected, TPR at 5% FPR (with CI), and paired D_plus - D.
    rng = np.random.default_rng(c["rng_seed"] + 991)
    att, det, pair = [], [], []
    a1 = R[R.attack == "A1"].flipped.mean()
    for name, r in R.groupby("attack", sort=False):
        fl = r.flipped.to_numpy()
        att.append({"attack": name, "seeds": ",".join(map(str, sorted(r.seed.unique()))), "n_rows": len(r),
                    "evasion_success": fl.mean(), "cost_vs_A1": a1 - fl.mean(),
                    "succ_undet_D": (fl & (r.score_D < r.thr_D)).mean(),
                    "succ_undet_Dplus": (fl & (r.score_Dplus < r.thr_Dplus)).mean(), "invalid_rows": 0})
        pD = (r.score_D - r.thr_D).to_numpy()[fl]
        pP = (r.score_Dplus - r.thr_Dplus).to_numpy()[fl]
        ns = negs[negs.seed.isin(r.seed.unique())]
        nD, nP = (ns.score_D - ns.thr_D).to_numpy(), (ns.score_Dplus - ns.thr_Dplus).to_numpy()
        for dn, n_, p_ in (("D", nD, pD), ("D_plus", nP, pP)):
            (alo, ahi), (tlo, thi) = boot(n_, p_, 0.0, c["n_boot"], rng)
            det.append({"attack": name, "detector": dn, "n_pos": len(p_), "auroc": auroc(n_, p_), "auroc_lo": alo,
                        "auroc_hi": ahi, "tpr_at_5fpr": float((p_ >= 0).mean()) if len(p_) else np.nan,
                        "tpr_lo": tlo, "tpr_hi": thi, "test_fpr": float((n_ >= 0).mean())})
        pb = paired_boot(nD, pD, 0.0, nP, pP, 0.0, c["n_boot"], rng)
        if pb:
            pair.append({"attack": name, "n_pos": len(pD), **pb})
    return pd.DataFrame(att), pd.DataFrame(det), pd.DataFrame(pair)


def report(R, negs, c, title, seeds, standin=None):
    ci = lambda m, lo, hi: "" if pd.isna(m) else (f"{m:.3f}" if pd.isna(lo) else f"{m:.3f} [{lo:.3f}, {hi:.3f}]")
    att, det, pair = attack_tables(R, negs, c)
    out = [f"\n---\n\n## {datetime.now():%Y-%m-%d %H:%M}: {title}\n",
           f"eps 1.0; seeds {seeds}; source rows = Stage 5 rows (150 per attack class); negatives = the saved 1,000 "
           "benign test rows per seed; saved Stage 6 FINAL defenders and thresholds (nothing retrained). A1, A6 and A7 "
           "rows are the saved Stage 6 FINAL rows; A8 rows are new.\n"]
    if standin is not None:
        out.append("**A8 stand-in quality (seed 0 VALIDATION rows: val benign predicted benign + val A1 rows that flip; "
                   "stand-ins computed with the same target class as the true features)**\n\n" + md(standin.round(3)))
    out.append("\n**Attacks: success, success AND undetected, cost vs A1 (0 invalid rows)**\n\n" + md(att.round(3)))
    d = det.copy()
    d["AUROC [95% CI]"] = [ci(r.auroc, r.auroc_lo, r.auroc_hi) for r in d.itertuples()]
    d["TPR at 5% FPR [95% CI]"] = [ci(r.tpr_at_5fpr, r.tpr_lo, r.tpr_hi) for r in d.itertuples()]
    out.append("\n**Detection on the attack rows (TPR among flipped rows, threshold from validation)**\n\n"
               + md(d[["attack", "detector", "n_pos", "AUROC [95% CI]", "TPR at 5% FPR [95% CI]", "test_fpr"]].round(3)))
    if len(pair):
        p = pair.copy()
        p["TPR D_plus - D [95% CI]"] = [ci(r.tpr_diff, r.tpr_diff_lo, r.tpr_diff_hi) for r in p.itertuples()]
        p["AUROC D_plus - D [95% CI]"] = [ci(r.auroc_diff, r.auroc_diff_lo, r.auroc_diff_hi) for r in p.itertuples()]
        out.append("\n**Paired D_plus minus D (same rows)**\n\n"
                   + md(p[["attack", "n_pos", "TPR D_plus - D [95% CI]", "AUROC D_plus - D [95% CI]"]]))
    rng = np.random.default_rng(c["rng_seed"] + 992)
    urows = []
    for lab, names in (("union of 7 (A1, A6 x 3, A7 x 3)", OLD),
                       ("union of 10 (+ A8 x 3)", OLD + [n for n in R.attack.unique() if n.startswith("A8")])):
        u = union_eval(R, names)
        dlt, lo, hi = paired_union(u, c["n_boot"], rng)
        urows.append({"candidates": lab, "n_source_rows": len(u), "union_flipped": u.union_flipped.mean(),
                      "succ_undet_union_D": u.union_undet_D.mean(), "succ_undet_union_Dplus": u.union_undet_Dplus.mean(),
                      "Dplus - D [95% CI]": ci(dlt, lo, hi)})
        if "10" in lab:
            u10 = u
    out.append("\n**Oracle union (an attacker with query access to the detector that keeps the best candidate per "
               "row)**\n\n" + md(pd.DataFrame(urows).round(3)))
    pc = u10.groupby("class")[["union_flipped", "union_undet_D", "union_undet_Dplus"]].mean().reset_index()
    out.append("\nUnion of 10, per source class:\n\n" + md(pc.round(3)))
    a8 = R[R.attack == "A8_lam1"]
    if len(a8):
        g = a8.groupby("class")
        pa = pd.DataFrame({"n": g.size(), "success": g.flipped.mean(),
                           "succ_undet_D": g.apply(lambda x: (x.flipped & (x.score_D < x.thr_D)).mean()),
                           "succ_undet_Dplus": g.apply(lambda x: (x.flipped & (x.score_Dplus < x.thr_Dplus)).mean()),
                           "tpr_D": g.apply(lambda x: (x.score_D >= x.thr_D)[x.flipped].mean()),
                           "tpr_Dplus": g.apply(lambda x: (x.score_Dplus >= x.thr_Dplus)[x.flipped].mean())}).reset_index()
        out.append("\n**A8 lambda 1, per source class**\n\n" + md(pa.round(3)))
    return "\n".join(out), att, det, pair, pd.DataFrame(urows)


def append_progress(text):
    with open(PROGRESS, "a", encoding="utf-8") as fh:
        fh.write(text.rstrip() + "\n")


# ---------------------------------------------------------------- main

def main():
    cfgmod.DEFAULTS.setdefault("stage6e", {**copy.deepcopy(STAGE4), **copy.deepcopy(STAGE6), **copy.deepcopy(STAGE6B),
                                           **copy.deepcopy(STAGE6C), **copy.deepcopy(STAGE6D), **copy.deepcopy(STAGE6E)})
    cfg, overrides = load_config("Stage 6e: stronger attacker against D_plus.")
    c = cfg["stage6e"]
    eps = c["eps_e"]
    src, out = resolve(c["src_dir_e"]), resolve(c["out_dir_e"])
    out.mkdir(parents=True, exist_ok=True)
    t_start = time.time()
    run_dir, log = start_run("stage6e_strong", cfg, overrides)
    dev = torch.device("cuda")
    keep_awake(True)
    global append_progress
    if not c["progress_log_e"]:
        append_progress = lambda text: log.info("[progress entry, not written]\n" + text[:4000])
    part1_txt = ""
    try:
        # ---- Part 1: oracle union over the 7 saved attacks (no new attack)
        rng1 = np.random.default_rng(c["rng_seed"] + 990)
        old_all = pd.concat([pd.read_parquet(src / f"seed{s}" / f"rows_eps{eps:g}_{n}.parquet")
                             for s in c["seeds_e"] for n in OLD], ignore_index=True)
        p1 = []
        for lab, seeds in [(f"seed {s}", [s]) for s in c["seeds_e"]] + [("pooled", c["seeds_e"])]:
            u = union_eval(old_all[old_all.seed.isin(seeds)], OLD)
            dlt, lo, hi = paired_union(u, c["n_boot"], rng1)
            p1.append({"seeds": lab, "n_source_rows": len(u), "union_flipped": u.union_flipped.mean(),
                       "succ_undet_union_D": u.union_undet_D.mean(), "succ_undet_union_Dplus": u.union_undet_Dplus.mean(),
                       "Dplus_minus_D": dlt, "ci_lo": lo, "ci_hi": hi})
        p1 = pd.DataFrame(p1)
        p1.to_csv(out / "part1_union7.csv", index=False)
        part1_txt = ("\n**Part 1: oracle union of the 7 saved attacks (query access to the detector, best of 7 per "
                     "row)**\n\n" + md(p1.round(3)) + "\n")
        log.info("Part 1 union of 7:\n" + p1.round(3).to_string(index=False))
        G = load_common(c, dev)
        variants =[("A8_lam1", 1.0, None), ("A8_lam100", 100.0, None), (f"A8norm_w{c['norm_w']:g}", 0.0, c["norm_w"])]
        # ---- stand-in quality on seed 0 validation rows
        f_si = out / "standin_quality.csv"
        model0, _ = load_ftt(G.s2 / "ftt_seed0.pt", dev)
        for p in model0.parameters():
            p.requires_grad_(False)
        if f_si.exists():
            standin = pd.read_csv(f_si)
        else:
            rng = np.random.default_rng(c["rng_seed"] + 9100)
            with torch.no_grad():
                pv = torch.cat([model0(G.Zben_va[i:i + 8192]).argmax(1) for i in range(0, len(G.Zben_va), 8192)])
            vb = rng.choice(torch.nonzero(pv == BENIGN).squeeze(1).cpu().numpy(), c["n_val_benign_check"], replace=False)
            Xv, zv = a1_on(model0, G.Sva, G.tp, eps, c, dev, c["rng_seed"] + 9200)
            with torch.no_grad():
                fl = model0(zv).argmax(1) == BENIGN
            Z = torch.cat([G.Zben_va[torch.from_numpy(vb).to(dev)], zv[fl]])
            true, _ = detector_scores(model0, Z, G.base, c)
            Ftrue, cols = plus_features(true)
            st = []
            for i in range(0, len(Z), 100):
                zb = Z[i:i + 100].clone().requires_grad_(True)
                with torch.no_grad():
                    cls = model0(zb).argmax(1)
                _, F = standin_features(model0, zb, cls, G.base, c)
                st.append(F.detach().cpu().numpy())
            S = np.concatenate(st)
            rows = []
            for j, col in enumerate(cols[len(FEATS_D):]):
                x, y = S[:, len(FEATS_D) + j], Ftrue[col].to_numpy()
                rows.append({"feature": col.replace("cons_", ""),
                             "r2": LinearRegression().fit(x[:, None], y).score(x[:, None], y),
                             "spearman": spearmanr(x, y).statistic})
            standin = pd.DataFrame(rows)
            standin["n_rows"] = len(Z)
            standin.to_csv(f_si, index=False)
        med = float(standin.r2.median())
        log.info(f"Stand-in quality (seed 0 val):\n{standin.round(3).to_string(index=False)}\nmedian R^2 {med:.3f}"
                 + ("  (BELOW 0.5: reported, continuing as instructed)" if med < 0.5 else ""))
        # ---- chunk size (OOM test) and runtime estimate on the first 500 rows of A8_lam1, seed 0
        DEF0 = load_defenders(src, 0, eps, dev)
        fn0 = terms_A8(model0, G.tp, G.base, DEF0["Dp"], c)
        sub = np.arange(min(c["estimate_rows"], len(G.pos)))
        Ssub = type("Sub", (), {})()
        Ssub.R, Ssub.frozen, Ssub.feats, Ssub.floors = G.Ste.R[sub], G.Ste.frozen[sub], G.Ste.feats, G.Ste.floors
        chunk = None
        for ch in c["chunk_candidates"]:
            try:
                torch.cuda.empty_cache()
                torch.cuda.reset_peak_memory_stats()
                cc = dict(c, steps=c["estimate_steps"], restarts=0, attack_chunk=ch)
                torch.cuda.synchronize()
                t0 = time.time()
                attack_loop(model0, Ssub, G.tp, eps, cc, 1.0, None, fn0, DEF0["Dp"].thr, dev, 0)
                torch.cuda.synchronize()
                per_step_500 = (time.time() - t0) / (c["estimate_steps"] + 1)
                chunk, peak = ch, torch.cuda.max_memory_allocated() / 2**30
                break
            except torch.cuda.OutOfMemoryError:
                log.info(f"OOM at chunk size {ch}")
                torch.cuda.empty_cache()
        if chunk is None:
            raise StopRun("IG-8 with double backprop runs out of memory at chunk size 50")
        c["attack_chunk"] = chunk
        steps_total = (c["steps"] + 1) * (1 + c["restarts"])
        per_variant = per_step_500 * steps_total * len(G.pos) / len(sub)
        hours = (per_variant * len(variants) * len(c["seeds_e"]) + len(c["seeds_e"]) * 4 * 60) / 3600
        log.info(f"RUNTIME ESTIMATE: chunk {chunk} (peak {peak:.2f} GB), {per_step_500:.2f} s per step on {len(sub)} "
                 f"rows -> {per_variant / 60:.1f} min per variant per seed -> {hours:.2f} h total (limit "
                 f"{c['max_hours_e']} h)")
        (out / "runtime_estimate.txt").write_text(f"{hours:.3f} h, chunk {chunk}, {per_step_500:.3f} s/step/500 rows\n",
                                                  encoding="utf-8")
        if hours > c["max_hours_e"]:
            raise StopRun(f"runtime estimate {hours:.2f} h > {c['max_hours_e']} h")
        del model0
        # ---- per seed
        R_all, N_all = [], []
        for s in c["seeds_e"]:
            sdir = out / f"seed{s}"
            sdir.mkdir(parents=True, exist_ok=True)
            DEF = load_defenders(src, s, eps, dev)
            D, Dp = DEF["D"], DEF["Dp"]
            NEG = pickle.loads((src / f"seed{s}" / "negatives.pkl").read_bytes())
            Fn, cols = plus_features(NEG["test"])
            negs = pd.DataFrame({"seed": s, "score_D": D.decision(Fn[FEATS_D]), "score_Dplus": Dp.decision(Fn[cols]),
                                 "thr_D": D.thr, "thr_Dplus": Dp.thr})
            old = pd.concat([pd.read_parquet(src / f"seed{s}" / f"rows_eps{eps:g}_{n}.parquet") for n in OLD],
                            ignore_index=True)
            model, _ = load_ftt(G.s2 / f"ftt_seed{s}.pt", dev)
            for p in model.parameters():
                p.requires_grad_(False)
            new = []
            for name, lam, nw in variants:
                f = sdir / f"rows_eps{eps:g}_{name}.parquet"
                if f.exists():
                    new.append(pd.read_parquet(f))
                    continue
                t0 = time.time()
                X, z = attack_loop(model, G.Ste, G.tp, eps, c, lam, nw, terms_A8(model, G.tp, G.base, Dp, c), Dp.thr,
                                   dev, c["rng_seed"] + 9300 + 17 * s + sum(map(ord, name)))
                check_valid_or_stop(X, G.Rte, G, eps, G.Ste, f"seed {s} {name}", log)
                with torch.no_grad():
                    lg = model(z)
                flip = (lg.argmax(1) == BENIGN).cpu().numpy()
                sc, _ = detector_scores(model, z, G.base, c)
                F, _ = plus_features(sc)
                df = pd.DataFrame({"seed": s, "eps": eps, "attack": name, "pos": G.pos,
                                   "class": [G.classes[k] for k in G.yte], "flipped": flip,
                                   "margin": (lg[:, BENIGN] - lg[:, 1:].amax(1)).cpu().numpy(),
                                   "score_D": D.decision(F[FEATS_D]), "score_Dplus": Dp.decision(F[cols]),
                                   "thr_D": D.thr, "thr_Dplus": Dp.thr, "valid": True, "seconds": time.time() - t0})
                df.to_parquet(f, index=False)
                new.append(df)
                log.info(f"seed {s} {name}: success {flip.mean():.3f}, succ_undet D "
                         f"{(flip & (df.score_D < D.thr)).mean():.3f} D_plus {(flip & (df.score_Dplus < Dp.thr)).mean():.3f}"
                         f" [{time.time() - t0:.0f} s]")
            del model
            torch.cuda.empty_cache()
            R = pd.concat([old] + new, ignore_index=True)
            R_all.append(R), N_all.append(negs)
            marker = sdir / "progress_logged"
            if not marker.exists():
                txt, att, det, pair, uni = report(R, negs, c, f"Stage 6e: strong attacker A8, seed {s}", [s],
                                                  standin if s == c["seeds_e"][0] else None)
                for nm, t in (("attacks", att), ("detection", det), ("paired", pair), ("union", uni)):
                    t.to_csv(sdir / f"summary_{nm}.csv", index=False)
                append_progress(txt)
                marker.write_text("1", encoding="utf-8")
        marker = out / "final_progress_logged"
        if not marker.exists():
            txt, att, det, pair, uni = report(pd.concat(R_all, ignore_index=True), pd.concat(N_all, ignore_index=True), c,
                                              "Stage 6e: strong attacker A8, pooled over seeds 0 and 2", c["seeds_e"])
            for nm, t in (("attacks", att), ("detection", det), ("paired", pair), ("union", uni)):
                t.to_csv(out / f"summary_pooled_{nm}.csv", index=False)
            append_progress(txt)
            marker.write_text("1", encoding="utf-8")
    except StopRun as e:
        log.error(f"STOPPED: {e}")
        append_progress(f"\n---\n\n## {datetime.now():%Y-%m-%d %H:%M}: Stage 6e stopped\n\nStop condition: {e}\n"
                        + part1_txt)
        finish_run(run_dir, log, {"stopped": str(e)}, t_start)
        keep_awake(False)
        sys.exit(2)
    keep_awake(False)
    finish_run(run_dir, log, {"seeds": c["seeds_e"]}, t_start)


if __name__ == "__main__":
    main()
