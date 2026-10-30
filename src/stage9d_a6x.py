# Stage 9d (Task 40): A6x, the A6 attacker on D with per-step selection by the true D score, both datasets, eps 1;
# plus two CPU analyses of the saved rows (A8x rows against D; controls minus D_plus under A8x).
#
# Decisions and the outcome rule: handoff/38_decisions_A6x.md (written before this file). Nothing existing is modified;
# every output goes to results/stage9/a6x/ (and results/stage9/a8x/summary_extra.csv for the CPU analyses).
#
# A6x = A6 (src/stage6d_final.py attack_chunk with terms_D, the saved D) with one change: the best iterate of a row is
# kept by the three-class key of A8x (Amendment 1 of handoff/36_decisions_A8x.md): (1) flipped and true D score below
# thr - 1e-3, (2) flipped, (3) not flipped; lowest true D score within (1) and (2), highest benign margin within (3).
# The in-loop D score is exact (feats_torch on the same forward pass). Amendment 1 (chunk size) of the decisions file:
# A6x runs at the Stage A partition, chunk 500 over all test source rows (G.Ste), with the A6 attack seed + chunk start
# as the generator seed of the restart noise, exactly as A6; the union uses the common rows only.
#
# Phases:
#   extra    CPU: (a) A8x rows against D vs A6 rows against D; (b) controls (Stage B) minus D_plus under A8x, union.
#   sanity   (s1) selection off (A6 key) reproduces the saved A6_lam1 rows of seed 0 on the first 200 common rows to
#            1e-5 (chunk 500); (s2) the true D features and decision inside A6x equal the saved values on the same rows.
#   pilot    all test source rows, 5 steps, lambda 1, seed 0, chunk 500, then scoring of the common rows; shared < 0.5 GB, reserved < 6.5 GB,
#            projected run time <= 2 h.
#   attack   18 runs (3 settings x 3 seeds x 2 datasets), resumable per file, --max-hours.
#   summary  unions: headline D+ (A1 + A8x x 3) minus D (A1 + A6x x 3), V0/V1/V2, per setting, sensitivity, A6x - A6.
# A watchdog polls the shared GPU memory counter every 60 s (10 s in the pilot) and stops on one value above 1 GB.
#
# Usage:
#   .venv\Scripts\python.exe -m src.stage9d_a6x {extra,sanity,pilot,attack,summary} [--years 2017 2018] [--max-hours 3]
import argparse
import math
import os
import sys
import time
import zlib

import numpy as np
import pandas as pd
import torch

from src.stage4_constraints import make_context, project
from src.stage6_defense import BENIGN
from src.stage6c_defense import FEATS_D, feats_torch
from src.stage6d_final import LinearDefender, sub_rows, terms_D  # noqa: F401  (pickles)
from src.stage9_dplus_ablation import OUT as OUT9, SEEDS, Log, StopRun, check_valid, keep_pos, load_defs, load_model, \
    load_set
from src.stage9b_dplus_adaptive import A6N, EPS, OUT as OUT9B, SETTINGS, boot_diff, cfg, load_G, union_ind
from src.stage9c_a8x import OUT as OUT8X, Watchdog, score_rows, shared_usage_gb
from src.utils import ROOT

OUT = ROOT / "results" / "stage9" / "a6x"
YEARS = ["2017", "2018"]
A6XN = [f"A6x_{t}" for t, _, _ in SETTINGS]
A8XN = [f"A8x_{t}" for t, _, _ in SETTINGS]
A6_OF = dict(zip(A6XN, A6N))                 # A6x name -> A6 name (attack seed and restart noise)
CONTROLS = ["c0", "c1", "c2", "a1", "a2"]
CTL_TYPE = {"c0": "exact", "c1": "exact", "c2": "exact", "a1": "stand-in", "a2": "stand-in"}
CHUNK = 500                                  # Amendment 1: the Stage A chunk of A6 (attack_chunk 2017, a6_chunk 2018)
TOL = 1e-5
TOL_MAHA_REL, TOL_DEC_ABS = 1e-3, 1e-3
SEL_MARGIN = 1e-3
PILOT_STEPS = 5
LIM_SHARED_GB, LIM_RESERVED_GB, STOP_SHARED_GB = 0.5, 6.5, 1.0
MAX_RUN_S = 2 * 3600


class Watchdog1(Watchdog):
    # As stage9c Watchdog, but one poll above 1 GB stops the process (handoff/38_decisions_A6x.md).
    def run(self):
        while True:
            sh = self.sample()
            if sh > STOP_SHARED_GB:
                self.log(f"STOP: shared GPU memory {sh:.2f} GB > {STOP_SHARED_GB} GB (phase {self.phase})")
                print(f"STOPPED: shared GPU memory above {STOP_SHARED_GB} GB", flush=True)
                os._exit(3)
            time.sleep(self.every)


# ---------------------------------------------------------------- the A6x attack

def a6_seed(c, s, a6_name):
    return c["rng_seed"] + 7300 + int(EPS * 100) + 17 * s + sum(map(ord, a6_name))


def attack_chunk_6x(model, S, tp, c, lam, norm_w, terms, thr, dev, seed, select="A6x"):
    # Constrained PGD of stage6d attack_chunk (A6 objective, terms_D, same generator and restart noise).
    # select "A6": Stage A key; "A6x": three-class key of A8x on the true D score (exact in the loop).
    ctx = make_context(S.R, S.feats, S.floors, EPS, S.frozen, dev)
    gen = torch.Generator(device=dev).manual_seed(seed)
    kappa, safety = c["kappa_a6"], c["a6_safety"]
    best_key = torch.full((S.R.shape[0],), float("-inf"), dtype=torch.float64, device=dev)
    best_s = torch.full((S.R.shape[0],), float("nan"), dtype=torch.float64, device=dev)
    best_x = None
    for run in range(1 + c["restarts"]):
        delta = torch.zeros_like(ctx.lo) if run == 0 else             ctx.lo + (ctx.hi - ctx.lo) * torch.rand(ctx.lo.shape, generator=gen, device=dev, dtype=torch.float64)
        for step in range(c["steps"] + 1):
            x, _ = project(delta, ctx)
            xv = x.detach().requires_grad_(True)
            e, s = terms(xv, ctx)
            obj = torch.clamp(e, max=kappa) - lam * torch.relu(s - thr + safety)
            if select == "A6":
                succ = e > 0
                key = (1e7 * (succ & (s < thr)) + 1e6 * succ).double() + obj.detach().double()
                key = key.detach()
            else:
                succ = e.detach() > 0
                sd = s.detach().double()
                und = succ & (sd < thr - SEL_MARGIN)
                key = torch.where(succ, 1e6 + 1e6 * und.double() - sd.clamp(-1e5, 1e5), e.detach().double())
            b = key > best_key
            best_key = torch.where(b, key, best_key)
            best_s = torch.where(b, s.detach().double(), best_s)
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
        return best_x.cpu().numpy(), tp(best_x).float(), best_s.cpu().numpy()


def run_6x(model, G, c, D, lam, nw, s, a6_name, dev, select="A6x", upto=None):
    # attack_loop of stage6d over all test source rows G.Ste in chunks of CHUNK, chunk seed = A6 seed + chunk start
    # (Stage A). `upto`: attack only the chunks that contain test rows < upto (sanity). Returns all attacked rows.
    xs, zs, ss = [], [], []
    fn = terms_D(model, G.tp, G.base, D)
    N = G.Ste.R.shape[0] if upto is None else upto
    for i in range(0, N, CHUNK):
        idx = np.arange(i, min(i + CHUNK, G.Ste.R.shape[0]))
        x, z, sc = attack_chunk_6x(model, sub_rows(G.Ste, idx), G.tp, c, lam, nw, fn, D.thr, dev,
                                   a6_seed(c, s, a6_name) + i, select)
        xs.append(x), zs.append(z), ss.append(sc)
        torch.cuda.empty_cache()
    return np.concatenate(xs), torch.cat(zs), np.concatenate(ss)


# ---------------------------------------------------------------- sanity checks

def sanity(year, dev, log):
    c = cfg(year)
    G = load_G(year, c, dev)
    s = 0
    model = load_model(year, s, dev)
    DEF = load_defs(year, s, EPS, dev)
    D = DEF["D"]
    sdir9 = OUT9 / year / f"seed{s}"
    rows = np.arange(200)
    k = G.common[rows]
    out = []

    def rec(chk, val, tol=TOL, kind="abs"):
        out.append({"year": year, "check": chk, "value": float(val), "tol": tol, "tol_kind": kind,
                    "pass": bool(val <= tol)})
        log(f"{year} sanity {chk}: {val:.3e} ({kind} tol {tol:g}; {'PASS' if val <= tol else 'FAIL'})")

    Xa6 = np.load(sdir9 / "att_eps1_A6_lam1_X.npz")
    assert (Xa6["pos"] == G.pos).all()
    Xsaved = Xa6["X"][k]
    # (s1) selection off (Stage A key), Stage A partition: reproduces the saved A6_lam1 rows
    X1, _, _ = run_6x(model, G, c, D, 1.0, None, s, "A6_lam1", dev, select="A6", upto=int(k.max()) + 1)
    X1 = X1[k]
    d1 = np.abs(X1 - Xsaved).max()
    rec(f"(s1) selection off, chunk {CHUNK}: max |X - saved A6_lam1 X|", d1)
    log(f"{year} sanity (s1, info) rows with any difference > 1e-5: {int((np.abs(X1 - Xsaved) > TOL).any(1).sum())}")
    # (s2) true D features and decision inside A6x (batches of 50) on the saved A6_lam1 rows vs the saved values
    V = load_set(sdir9 / "att_eps1_A6_lam1.npz")
    sc_saved = V["sc"].iloc[k].reset_index(drop=True)
    dec_saved = pd.read_parquet(sdir9 / "scores_eps1_A6_lam1.parquet")["score_D"].to_numpy()[k]
    Fi, si = [], []
    with torch.no_grad():
        z = G.tp(torch.from_numpy(Xsaved).to(dev)).float()
    for i in range(0, len(rows), 50):
        zc = z[i:i + 50]
        with torch.no_grad():
            lg = model(zc)
            F = feats_torch(lg, zc, G.base)
            Fi.append(F.double().cpu().numpy()), si.append(D.torch_decision(F).double().cpu().numpy())
    Fi, si = np.concatenate(Fi), np.concatenate(si)
    for j, col in enumerate(FEATS_D):
        dj = np.abs(Fi[:, j] - sc_saved[col].to_numpy())
        if col == "mahalanobis_benign":
            log(f"{year} sanity (s2, info) {col} max |inside - saved| absolute: {dj.max():.3e}")
            rec(f"(s2) {col} max relative |inside - saved|", (dj / np.abs(sc_saved[col].to_numpy()).clip(1e-30)).max(),
                TOL_MAHA_REL, "rel")
        else:
            rec(f"(s2) {col} max |inside - saved|", dj.max())
    rec("(s2) D decision max |inside - saved|", np.abs(si - dec_saved).max(), TOL_DEC_ABS, "abs (Amendment 1)")
    fl = V["pred"][k] == BENIGN
    u0, u1, us = fl & (si < D.thr), fl & (si < D.thr - SEL_MARGIN), fl & (dec_saved < D.thr)
    log(f"{year} sanity (margin, info) rows flipped {int(fl.sum())}; undetected: inside no margin {int(u0.sum())}, "
        f"inside with margin {int(u1.sum())}, saved evaluation {int(us.sum())}; differ margin vs no margin "
        f"{int((u0 != u1).sum())}, inside no margin vs saved {int((u0 != us).sum())}")
    df = pd.DataFrame(out)
    (OUT / year).mkdir(parents=True, exist_ok=True)
    df.to_csv(OUT / year / "sanity.csv", index=False)
    if not df["pass"].all():
        raise StopRun(f"sanity check failed on {year}: {df[~df['pass']].check.tolist()}")
    return df


# ---------------------------------------------------------------- pilot

def pilot(year, dev, log, wd):
    c = cfg(year)
    G = load_G(year, c, dev)
    s = 0
    model = load_model(year, s, dev)
    DEF = load_defs(year, s, EPS, dev)
    rows = np.arange(len(G.common))
    cp = {**c, "steps": PILOT_STEPS, "restarts": 0}
    n_all = G.Ste.R.shape[0]
    torch.cuda.empty_cache()
    torch.cuda.reset_peak_memory_stats()
    wd.phase = f"pilot {year} attack"
    sh0 = shared_usage_gb()
    t0 = time.time()
    X, z, _ = run_6x(model, G, cp, DEF["D"], 1.0, None, s, "A6_lam1", dev)
    torch.cuda.synchronize()
    ta = time.time() - t0
    sh_a = shared_usage_gb()
    check_valid(X, G.Rte, G, EPS, G.Ste, f"{year} pilot")
    z = z[torch.as_tensor(G.common, device=z.device)]
    wd.phase = f"pilot {year} scoring"
    t1 = time.time()
    score_rows(model, z, G, c, DEF, {"pos": G.pos[G.common][rows]})
    torch.cuda.synchronize()
    ts = time.time() - t1
    sh_s = shared_usage_gb()
    res = torch.cuda.max_memory_reserved() / 2 ** 30
    alc = torch.cuda.max_memory_allocated() / 2 ** 30
    shared_max = max(x for x in (sh0, sh_a, sh_s, wd.max_shared) if x == x)
    per_step = ta / (PILOT_STEPS + 1)
    run_s = per_step * (c["steps"] + 1) * (1 + c["restarts"]) + ts
    rec = {"year": year, "rows_attacked": n_all, "rows_scored": len(rows), "chunk": CHUNK, "steps": PILOT_STEPS,
           "attack_s": ta, "scoring_s": ts,
           "sec_per_iter_all_rows": per_step, "max_reserved_gb": res, "max_allocated_gb": alc,
           "shared_gb_start": sh0, "shared_gb_after_attack": sh_a, "shared_gb_after_scoring": sh_s,
           "shared_gb_max": shared_max, "est_min_per_run": run_s / 60,
           "est_hours_9_runs": run_s * len(SETTINGS) * len(SEEDS) / 3600, "valid_rows": len(X),
           "pass_shared": shared_max < LIM_SHARED_GB, "pass_reserved": res < LIM_RESERVED_GB,
           "pass_run_time": run_s <= MAX_RUN_S}
    log(f"{year} pilot: {rec}")
    pd.DataFrame([rec]).to_csv(OUT / year / "pilot.csv", index=False)
    if not (rec["pass_shared"] and rec["pass_reserved"]):
        raise StopRun(f"pilot memory limits failed on {year}: shared max {shared_max:.2f} GB, reserved {res:.2f} GB")
    if not rec["pass_run_time"]:
        raise StopRun(f"pilot projects {run_s / 3600:.2f} h per run on {year}, above 2 h")
    return rec


# ---------------------------------------------------------------- attack

def attack(year, dev, log, t_end, wd):
    c = cfg(year)
    G = load_G(year, c, dev)
    rows = np.arange(len(G.common))
    pos = G.pos[G.common]
    cls_names = [G.classes[k] for k in G.yte[G.common]]
    for (tag, lam, nw), name in zip(SETTINGS, A6XN):
        for s in SEEDS:
            sdir = OUT / year / f"seed{s}"
            sdir.mkdir(parents=True, exist_ok=True)
            p = sdir / f"rows_{name}.parquet"
            if p.exists():
                continue
            wd.phase = f"{year} seed {s} {name}"
            model = load_model(year, s, dev)
            DEF = load_defs(year, s, EPS, dev)
            t0 = time.time()
            X, z, s_sel = run_6x(model, G, c, DEF["D"], lam, nw, s, A6_OF[name], dev)
            check_valid(X, G.Rte, G, EPS, G.Ste, f"{year} seed {s} {name}")
            X, z, s_sel = X[G.common], z[torch.as_tensor(G.common, device=z.device)], s_sel[G.common]
            ta = time.time() - t0
            df = score_rows(model, z, G, c, DEF, {"seed": s, "eps": EPS, "attack": name, "pos": pos,
                                                   "class": cls_names, "valid": True, "score_D_sel": s_sel})
            df["seconds_attack"], df["seconds_total"] = ta, time.time() - t0
            np.savez_compressed(sdir / f"X_{name}.npz", X=X, pos=pos)
            df.to_parquet(p, index=False)
            u = df.flipped & (df.score_D < df.thr_D)
            log(f"{year} seed {s} {name}: success {df.flipped.mean():.3f}, succ+undet D {u.mean():.3f} "
                f"[attack {ta:.0f} s, total {time.time() - t0:.0f} s, max reserved "
                f"{torch.cuda.max_memory_reserved() / 2 ** 30:.2f} GB]")
            del model
            torch.cuda.empty_cache()
            if time.time() - t0 > MAX_RUN_S:
                raise StopRun(f"run {year} seed {s} {name} took {time.time() - t0:.0f} s, above 2 h")
            if time.time() > t_end:
                raise StopRun("runtime cap reached")


# ---------------------------------------------------------------- CPU: rows and unions

def base_rows(year, s):
    return pd.read_parquet(OUT9B / year / f"seed{s}" / "rows_base.parquet")


def a8x_rows(year, s):
    return [pd.read_parquet(OUT8X / year / f"seed{s}" / f"rows_{n}.parquet") for n in A8XN]


def boot_rows(U, a, b, year, tag):
    # rows of (per seed, pooled): mean(b) - mean(a) with the paired bootstrap of stage9b (within seed).
    res = []
    for sl, sd in [(f"seed {q}", [q]) for q in SEEDS] + [("pooled", SEEDS)]:
        u = U[U.seed.isin(sd)]
        rng = np.random.default_rng(zlib.crc32(f"{year}|{tag}|{sl}".encode()))
        lo, hi = boot_diff(u[a].to_numpy(), u[b].to_numpy(), u.seed.to_numpy(), 1000, rng)
        res.append({"seeds": sl, "n_rows": len(u), f"undet_{a}": u[a].mean(), f"undet_{b}": u[b].mean(),
                    "diff": u[b].mean() - u[a].mean(), "lo": lo, "hi": hi})
    return res


def extra(log):
    rows_a, rows_b = [], []
    for year in YEARS:
        R = []
        for s in SEEDS:
            R.append(base_rows(year, s))
            R += a8x_rows(year, s)
            for ctl in CONTROLS:
                R += [pd.read_parquet(OUT9B / year / f"seed{s}" / f"rows_B_{ctl}_{t}.parquet") for t, _, _ in SETTINGS]
        R = pd.concat(R, ignore_index=True)
        # (a) A8x rows against D vs A6 rows against D, V1, per setting and seed
        Rv = R[R.pos.isin(keep_pos(year, "V1"))]
        for (tag, _, _), n6, n8 in zip(SETTINGS, A6N, A8XN):
            for sl, sd in [(f"seed {q}", [q]) for q in SEEDS] + [("pooled", SEEDS)]:
                rec = {"part": "a", "year": year, "rowset": "V1", "setting": tag, "seeds": sl}
                for lab, nm in (("A6", n6), ("A8x", n8)):
                    r = Rv[(Rv.attack == nm) & Rv.seed.isin(sd)]
                    und = r.flipped & (r.score_D < r.thr_D)
                    rec.update({f"{lab}_n_source": len(r), f"{lab}_success": r.flipped.mean(),
                                f"{lab}_undetD_share_of_flipped": und.sum() / max(int(r.flipped.sum()), 1),
                                f"{lab}_undetD_share_of_source": und.mean()})
                    if lab == "A8x":
                        und_p = r.flipped & (r.score_Dplus < r.thr_Dplus)
                        rec["A8x_undetDplus_share_of_source"] = und_p.mean()
                rows_a.append(rec)
        # (b) control (A1 + B_ctl x 3) minus D_plus (A1 + A8x x 3), V0/V1/V2
        for rv in ("V0", "V1", "V2"):
            Rv = R[R.pos.isin(keep_pos(year, rv))]
            up = union_ind(Rv, "Dplus", ["A1"] + A8XN)
            ud = union_ind(Rv, "D", ["A1"] + A6N)
            for ctl in CONTROLS:
                uc = union_ind(Rv, ctl, ["A1"] + [f"B_{ctl}_{t}" for t, _, _ in SETTINGS])
                U = pd.concat([up, uc, ud], axis=1).reset_index()
                for r in boot_rows(U, "Dplus", ctl, year, f"{rv}|{ctl}|minus Dplus A8x"):
                    rows_b.append({"part": "b", "year": year, "rowset": rv, "control": ctl,
                                   "attacker_control": CTL_TYPE[ctl], "attacker_Dplus": "A8x (exact objective)",
                                   "undet_D": U.loc[U.seed.isin(SEEDS if r["seeds"] == "pooled" else
                                                                [int(r["seeds"][-1])]), "D"].mean(),
                                   "undet_control": r[f"undet_{ctl}"], "undet_Dplus": r["undet_Dplus"],
                                   "control_minus_Dplus": r["diff"], "lo": r["lo"], "hi": r["hi"],
                                   "seeds": r["seeds"], "n_rows": r["n_rows"]})
    A, B = pd.DataFrame(rows_a), pd.DataFrame(rows_b)
    p = OUT8X / "summary_extra.csv"
    assert not p.exists(), f"{p} exists; not overwritten"
    pd.concat([A, B], ignore_index=True).to_csv(p, index=False)
    log(f"summary_extra written: {len(A)} rows (a), {len(B)} rows (b)")


def summary(log):
    urows = []
    for year in YEARS:
        R = []
        for s in SEEDS:
            R.append(base_rows(year, s))
            R += a8x_rows(year, s)
            R += [pd.read_parquet(OUT / year / f"seed{s}" / f"rows_{n}.parquet") for n in A6XN]
        R = pd.concat(R, ignore_index=True)
        for rv in ("V0", "V1", "V2"):
            Rv = R[R.pos.isin(keep_pos(year, rv))]
            combos = [("headline: D A1+A6x x3, D+ A1+A8x x3", "D", ["A1"] + A6XN, "Dplus", ["A1"] + A8XN),
                      ("A8x report: D A1+A6x3, D+ A1+A8x x3", "D", ["A1"] + A6N, "Dplus", ["A1"] + A8XN),
                      ("sensitivity (unequal effort): D A1+A6x3+A6x x3, D+ A1+A8x x3", "D", ["A1"] + A6N + A6XN,
                       "Dplus", ["A1"] + A8XN),
                      ("selection effect on D: D A1+A6x x3 minus D A1+A6x3", "D", ["A1"] + A6N, "D",
                       ["A1"] + A6XN)]
            combos += [(f"per setting {t}: D A1+A6x_{t}, D+ A1+A8x_{t}", "D", ["A1", A6XN[i]], "Dplus",
                        ["A1", A8XN[i]]) for i, (t, _, _) in enumerate(SETTINGS)]
            for lab, da, na, db, nb in combos:
                U = pd.concat([union_ind(Rv, da, na).rename("a"), union_ind(Rv, db, nb).rename("b")],
                              axis=1).reset_index()
                for r in boot_rows(U, "a", "b", year, f"{rv}|{lab}"):
                    urows.append({"year": year, "rowset": rv, "union": lab, "seeds": r["seeds"],
                                  "n_rows": r["n_rows"], "undet_first": r["undet_a"], "undet_second": r["undet_b"],
                                  "second_minus_first": r["diff"], "lo": r["lo"], "hi": r["hi"]})
    U = pd.DataFrame(urows)
    U.to_csv(OUT / "summary_union.csv", index=False)
    log(f"summary written: {len(U)} union rows")


# ---------------------------------------------------------------- main

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("phase", choices=["extra", "sanity", "pilot", "attack", "summary"])
    ap.add_argument("--years", nargs="+", default=YEARS)
    ap.add_argument("--max-hours", type=float, default=3.0)
    a = ap.parse_args()
    OUT.mkdir(parents=True, exist_ok=True)
    log = Log(OUT / f"log_{a.phase}.txt")
    t_end = time.time() + a.max_hours * 3600
    log(f"start {a.phase} years {a.years} chunk {CHUNK} cap {a.max_hours} h pid {os.getpid()}")
    t0 = time.time()
    try:
        if a.phase == "extra":
            extra(log)
        elif a.phase == "summary":
            summary(log)
        else:
            dev = torch.device("cuda")
            wd = Watchdog1(log, OUT / "memlog.csv", every=10 if a.phase == "pilot" else 60)
            wd.phase = a.phase
            wd.start()
            for year in a.years:
                (OUT / year).mkdir(parents=True, exist_ok=True)
                if a.phase == "sanity":
                    sanity(year, dev, log)
                elif a.phase == "pilot":
                    pilot(year, dev, log, wd)
                elif a.phase == "attack":
                    attack(year, dev, log, t_end, wd)
    except StopRun as e:
        log(f"STOPPED: {e}")
        print(f"STOPPED: {e}")
        sys.exit(2)
    log(f"finished {a.phase} in {time.time() - t0:.0f} s")
    print(f"finished {a.phase} in {time.time() - t0:.0f} s")


if __name__ == "__main__":
    main()
