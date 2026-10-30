# Stage 9c (Task 37): A8x, an exact-objective adaptive attacker on D_plus, both datasets, eps 1.
#
# Decisions and the outcome rule: handoff/36_decisions_A8x.md (written before this file). Nothing existing is modified;
# every output goes to results/stage9/a8x/{2017,2018}/seed{s}/.
#
# A8x = A8 (src/stage6e_strong.py, terms_A8 / standin_features) with three changes:
#   - IG in the loss: 20-step Gauss-Legendre IG from the zero baseline, the nodes, weights and summation order of
#     captum 0.9.0 IntegratedGradients (method gausslegendre), double backprop through all 20 steps (A8: 8-step midpoint).
#   - Soft-rank temperature annealed geometrically from 0.1 to 0.03 over the steps of each restart (A8: fixed 0.1).
#   - Best-iterate selection by the TRUE D_plus decision: at every step the true 16 D_plus features are computed from the
#     same forward tensors (rollout, layer-mean attention, |grad x input| and IG-20 of the benign class, which is the
#     predicted class on every row that can count), with the evaluation functions (cons12, Baselines.scores); a row
#     keeps the iterate predicted benign with the lowest true D_plus score, else the one with the highest benign margin.
# Settings, rows, seeds as Stage B (src/stage9b_dplus_adaptive.py): PGD of stage6d attack_chunk (100 steps, 1 restart,
# kappa, safety, cosine step size), lambda 1, 100 and the normalised variant w 0.5, the 7 common classes, chunk 50.
#
# Phases:
#   sanity   (a) A8x with IG-8 midpoint and tau 0.1 vs A8 (loss terms and stand-in features), (b) IG-20 Gauss-Legendre
#            vs captum IntegratedGradients, (c) true D_plus features inside A8x vs the saved D_plus features; 200 rows of
#            the saved A6_lam1 set (rows predicted benign), seed 0, per dataset. Tolerance 1e-5; a failure stops.
#   pilot    all common rows, 5 steps, lambda 1, seed 0, chunk 50, then the scoring phase; logs max_memory_reserved and
#            the Windows counter "GPU Process Memory Shared Usage" of this process; go only if shared < 0.5 GB and
#            reserved < 6.5 GB.
#   attack   full runs, lambda 1 for seeds 0-2 first, then lambda 100 and norm_w0.5 (resumable per file, --max-hours);
#            interim line after the three 2017 lambda 1 runs.
#   summary  equal-effort unions (headline, sensitivity, V0/V2, per setting) and paired TPR differences.
# A watchdog thread polls the shared-usage counter every 60 s during every phase and stops the process on 3 consecutive
# values above 1 GB.
#
# Usage:
#   .venv\Scripts\python.exe -m src.stage9c_a8x {sanity,pilot,attack,summary} [--years 2017 2018] [--max-hours 5]
import argparse
import math
import os
import subprocess
import sys
import threading
import time
import zlib

import numpy as np
import pandas as pd
import torch

from src.stage3_explain import integrated_gradients
from src.stage4_attack import rollout_cls
from src.stage4_constraints import derive, make_context, project
from src.stage6_defense import BENIGN, make_rows, norm_abs
from src.stage6c_defense import FEATS_D, feats_torch
from src.stage6d_final import LinearDefender, paired_boot, plus_features, sub_rows  # noqa: F401  (pickles)
from src.stage6e_strong import cosine, ig_steps, soft_jaccard, soft_spearman, standin_features, terms_A8
from src.stage9_dplus_ablation import OUT as OUT9, SEEDS, Log, StopRun, check_valid, cons12, keep_pos, load_defs, \
    load_model, load_set, vectors
from src.stage9b_dplus_adaptive import A6N, A8N, A8_DIR, EPS, OUT as OUT9B, SETTINGS, boot_diff, cfg, load_G, union_ind
from src.utils import ROOT

OUT = ROOT / "results" / "stage9" / "a8x"
YEARS = ["2017", "2018"]
A8XN = [f"A8x_{t}" for t, _, _ in SETTINGS]
CHUNK = 50
IG_STEPS = 20
TAU0, TAU1 = 0.1, 0.03
TOL = 1e-5
TOL_MAHA_REL, TOL_DEC_ABS = 1e-3, 1e-3      # Amendment 1 (handoff/36_decisions_A8x.md), check (c)
SEL_MARGIN = 1e-3                          # Amendment 1: flipped and undetected for selection only below thr - 1e-3
PILOT_STEPS = 5
LIM_SHARED_GB, LIM_RESERVED_GB, STOP_SHARED_GB, STOP_POLLS = 0.5, 6.5, 1.0, 3


# ---------------------------------------------------------------- shared GPU memory watchdog (Windows counter)

def shared_usage_gb(pid=None):
    pid = pid or os.getpid()
    cmd = (f"$s = (Get-Counter '\\GPU Process Memory(pid_{pid}_*)\\Shared Usage' -ErrorAction SilentlyContinue)"
           ".CounterSamples; if ($s) { ($s | Measure-Object CookedValue -Sum).Sum } else { 0 }")
    try:
        r = subprocess.run(["powershell", "-NoProfile", "-Command", cmd], capture_output=True, text=True, timeout=60)
        return float(r.stdout.strip() or 0) / 2 ** 30
    except Exception:
        return float("nan")


class Watchdog(threading.Thread):
    # Polls the shared usage every `every` s into memlog.csv; 3 consecutive values above 1 GB stop the process.
    def __init__(self, log, path, every=60):
        super().__init__(daemon=True)
        self.log, self.path, self.every = log, path, every
        self.max_shared, self.n_high, self.phase = 0.0, 0, ""
        self.path.parent.mkdir(parents=True, exist_ok=True)
        if not self.path.exists():
            self.path.write_text("time,phase,shared_gb,reserved_gb,allocated_gb\n", encoding="utf-8")

    def sample(self):
        sh = shared_usage_gb()
        rs = torch.cuda.memory_reserved() / 2 ** 30 if torch.cuda.is_initialized() else 0.0
        al = torch.cuda.memory_allocated() / 2 ** 30 if torch.cuda.is_initialized() else 0.0
        with open(self.path, "a", encoding="utf-8") as f:
            f.write(f"{time.strftime('%Y-%m-%d %H:%M:%S')},{self.phase},{sh:.3f},{rs:.3f},{al:.3f}\n")
        if sh == sh:
            self.max_shared = max(self.max_shared, sh)
        return sh

    def run(self):
        while True:
            sh = self.sample()
            self.n_high = self.n_high + 1 if sh > STOP_SHARED_GB else 0
            if self.n_high >= STOP_POLLS:
                self.log(f"STOP: shared GPU memory {sh:.2f} GB > {STOP_SHARED_GB} GB for {STOP_POLLS} polls "
                         f"(phase {self.phase})")
                print(f"STOPPED: shared GPU memory above {STOP_SHARED_GB} GB", flush=True)
                os._exit(3)
            time.sleep(self.every)


# ---------------------------------------------------------------- A8x features

def gl_nodes(n):
    # captum gauss_legendre_builders: alphas and step sizes on [0, 1].
    x, w = np.polynomial.legendre.leggauss(n)
    return (0.5 * (1 + x)).tolist(), (0.5 * w).tolist()


def ig_gl(model, z, cls, n, create_graph=True):
    # IG from the zero baseline with captum's Gauss-Legendre rule and summation order; differentiable if create_graph.
    alphas, w = gl_nodes(n)
    base = torch.zeros_like(z)
    xs = torch.cat([base + a * (z - base) for a in alphas], 0)
    if not create_graph:
        xs = xs.detach().requires_grad_(True)
    lg = model(xs)
    g, = torch.autograd.grad(lg.gather(1, cls.repeat(n)[:, None]).sum(), xs, create_graph=create_graph)
    sg = g.contiguous().view(n, -1) * torch.tensor(w).float().view(n, 1).to(g.device)
    return (z - base) * sg.view(n, *z.shape).sum(0)


def nrm(t):
    return t / t.sum(1, keepdim=True).clamp_min(1e-12)


def features_x(model, z, cls, base, tau, ig_mode):
    # As stage6e standin_features, with the IG rule selectable; also returns the raw tensors for the true features.
    lg, attns = model(z, return_attn=True)
    A = torch.stack(attns, 1)
    ro = rollout_cls(attns)
    me = A.mean((1, 2))[:, 0, 1:]
    g, = torch.autograd.grad(lg.gather(1, cls[:, None]).sum(), z, create_graph=True)
    gxi = (z * g).abs()
    ig = (ig_steps(model, z, cls, 8) if ig_mode == "mid8" else ig_gl(model, z, cls, IG_STEPS)).abs()
    feats = []
    for att in (nrm(ro), nrm(me)):
        for attr in (nrm(gxi), nrm(ig)):
            sp = soft_spearman(att, attr, tau)
            feats += [sp, soft_jaccard(att, attr, 10, tau) + 1e-3 * sp, cosine(att, attr)]
    return lg, torch.cat([feats_torch(lg, z, base), torch.stack(feats, 1)], 1), (ro, me, z * g, ig)


@torch.no_grad()
def true_frame(lg, ro, me, gxi, ig, z, base):
    # The detector_scores columns used by D and D_plus, with the evaluation functions (stage9 vectors), benign class.
    ro, me = ro.float().cpu().numpy(), me.float().cpu().numpy()
    p = torch.softmax(lg, -1)
    top2 = lg.topk(2, dim=1).values
    ben = torch.zeros(z.shape[0], dtype=torch.long, device=z.device)
    maha_b, knn_b = base.scores(z, ben)
    s = {"msp": -p.max(1).values.cpu().numpy(), "margin": -(top2[:, 0] - top2[:, 1]).cpu().numpy(),
         "mahalanobis_benign": maha_b, "knn5_benign": knn_b}
    s.update(cons12({"rollout": norm_abs(ro), "attnmean": norm_abs(me)},
                    {"gxi": norm_abs(gxi.float().cpu().numpy()), "ig20": norm_abs(ig.float().cpu().numpy())}))
    return pd.DataFrame(s)


def true_dplus(lg, raw, z, base, Dp):
    F, cols = plus_features(true_frame(lg.detach(), *(t.detach() for t in raw), z.detach(), base))
    return Dp.decision(F[cols].to_numpy()), F[cols].to_numpy()


def tau_at(step, steps):
    return TAU0 * (TAU1 / TAU0) ** (step / max(steps - 1, 1))


def terms_x(model, tp, base, Dp, ig_mode="gl20"):
    def f(xv, ctx, tau):
        z = tp(derive(xv, ctx)).float()
        cls = torch.full((z.shape[0],), BENIGN, dtype=torch.long, device=z.device)
        lg, F, raw = features_x(model, z, cls, base, tau, ig_mode)
        s_true, _ = true_dplus(lg, raw, z, base, Dp)
        return lg[:, BENIGN] - lg[:, 1:].amax(1), Dp.torch_decision(F), torch.from_numpy(s_true).to(z.device)
    return f


# ---------------------------------------------------------------- attack (stage6d attack_chunk with true selection)

def attack_chunk_x(model, S, tp, c, lam, norm_w, terms, thr, dev, seed):
    # Constrained PGD of stage6d attack_chunk: maximise min(e, kappa) - lam relu(s - thr + safety) (normalised variant
    # mixes the unit-norm gradients of min(e, kappa) and -s), s = stand-in D_plus decision with tau annealed per
    # restart. Selection: predicted benign (e > 0) with the lowest TRUE D_plus decision, else the highest margin e.
    ctx = make_context(S.R, S.feats, S.floors, EPS, S.frozen, dev)
    gen = torch.Generator(device=dev).manual_seed(seed)
    kappa, safety = c["kappa_a6"], c["a6_safety"]
    best_key = torch.full((S.R.shape[0],), float("-inf"), dtype=torch.float64, device=dev)
    best_s = torch.full((S.R.shape[0],), float("nan"), dtype=torch.float64, device=dev)
    best_x = None
    for run in range(1 + c["restarts"]):
        delta = torch.zeros_like(ctx.lo) if run == 0 else \
            ctx.lo + (ctx.hi - ctx.lo) * torch.rand(ctx.lo.shape, generator=gen, device=dev, dtype=torch.float64)
        for step in range(c["steps"] + 1):
            x, _ = project(delta, ctx)
            xv = x.detach().requires_grad_(True)
            e, s, s_true = terms(xv, ctx, tau_at(step, c["steps"]))
            obj = torch.clamp(e, max=kappa) - lam * torch.relu(s - thr + safety)
            succ = e.detach() > 0
            # Amendment 1: classes (1) flipped and true score < thr - SEL_MARGIN, (2) flipped, (3) not flipped;
            # lowest true score within (1) and (2), highest margin within (3).
            und = succ & (s_true < thr - SEL_MARGIN)
            key = torch.where(succ, 1e6 + 1e6 * und.double() - s_true.clamp(-1e5, 1e5), e.detach().double())
            b = key > best_key
            best_key = torch.where(b, key, best_key)
            best_s = torch.where(b, s_true, best_s)
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


def run_x(model, S, G, c, Dp, lam, nw, seed, dev):
    xs, zs, ss = [], [], []
    fn = terms_x(model, G.tp, G.base, Dp)
    for i in range(0, S.R.shape[0], CHUNK):
        idx = np.arange(i, min(i + CHUNK, S.R.shape[0]))
        x, z, s = attack_chunk_x(model, sub_rows(S, idx), G.tp, c, lam, nw, fn, Dp.thr, dev, seed + i)
        xs.append(x), zs.append(z), ss.append(s)
        torch.cuda.empty_cache()
    return np.concatenate(xs), torch.cat(zs), np.concatenate(ss)


def attack_seed(s, tag):
    return 4026 + 9700 + 17 * s + zlib.crc32(f"A8x|{tag}".encode()) % 1000


def score_rows(model, z, G, c, DEF, meta):
    torch.cuda.empty_cache()
    V = vectors(model, z, G.base, c)
    F, cols = plus_features(V["sc"])
    flipped = V["pred"] == BENIGN
    return pd.DataFrame({**meta, "flipped": flipped,
                         "score_D": DEF["D"].decision(F[FEATS_D].to_numpy()), "thr_D": DEF["D"].thr,
                         "score_Dplus": DEF["Dp"].decision(F[cols].to_numpy()), "thr_Dplus": DEF["Dp"].thr})


# ---------------------------------------------------------------- sanity checks

def sanity(year, dev, log):
    c = cfg(year)
    G = load_G(year, c, dev)
    s = 0
    model = load_model(year, s, dev)
    DEF = load_defs(year, s, EPS, dev)
    sdir9 = OUT9 / year / f"seed{s}"
    # 200 rows predicted benign from the saved A6 sets of seed 0, in the order A6_lam1, A6norm_w0.5, A6_lam100
    # (2018 A6_lam1 has fewer than 200 such rows).
    Xs, igs, scs, svs = [], [], [], []
    for nm in ("A6_lam1", "A6norm_w0.5", "A6_lam100"):
        Vn = load_set(sdir9 / f"att_eps1_{nm}.npz")
        m = np.flatnonzero(Vn["pred"] == BENIGN)[:200 - sum(len(x) for x in Xs)]
        Xs.append(np.load(sdir9 / f"att_eps1_{nm}_X.npz")["X"][m]), igs.append(Vn["ig"][m])
        scs.append(Vn["sc"].iloc[m]), svs.append(pd.read_parquet(sdir9 / f"scores_eps1_{nm}.parquet")["score_Dplus"]
                                                 .to_numpy()[m])
        log(f"{year} sanity rows: {len(m)} from {nm}")
        if sum(len(x) for x in Xs) == 200:
            break
    V9 = {"ig": np.concatenate(igs), "sc": pd.concat(scs, ignore_index=True)}
    rows = np.arange(200)
    assert sum(len(x) for x in Xs) == 200
    with torch.no_grad():
        z = G.tp(torch.from_numpy(np.concatenate(Xs)).to(dev)).float()
    cls = torch.full((len(rows),), BENIGN, dtype=torch.long, device=dev)
    out = []

    def rec(chk, val, tol=TOL, kind="abs"):
        out.append({"year": year, "check": chk, "value": float(val), "tol": tol, "tol_kind": kind,
                    "pass": bool(val <= tol)})
        log(f"{year} sanity {chk}: {val:.3e} ({kind} tol {tol:g}; {'PASS' if val <= tol else 'FAIL'})")

    # (a) IG-8 midpoint, tau 0.1: A8x == A8 (stand-in features and loss terms), in chunks of CHUNK rows
    ca = {**c, "ig_steps_loss": 8, "tau": 0.1}
    da, dl = [], {1.0: [], 100.0: []}
    for i in range(0, len(rows), CHUNK):
        zb, cb = z[i:i + CHUNK], cls[i:i + CHUNK]
        za, zx = zb.clone().requires_grad_(True), zb.clone().requires_grad_(True)
        lg_a, F_a = standin_features(model, za, cb, G.base, ca)
        lg_x, F_x, _ = features_x(model, zx, cb, G.base, 0.1, "mid8")
        da.append((F_x - F_a).abs().max().item())
        e_a, s_a = lg_a[:, BENIGN] - lg_a[:, 1:].amax(1), DEF["Dp"].torch_decision(F_a)
        e_x, s_x = lg_x[:, BENIGN] - lg_x[:, 1:].amax(1), DEF["Dp"].torch_decision(F_x)
        for lam in dl:
            oa = torch.clamp(e_a, max=c["kappa_a6"]) - lam * torch.relu(s_a - DEF["Dp"].thr + c["a6_safety"])
            ox = torch.clamp(e_x, max=c["kappa_a6"]) - lam * torch.relu(s_x - DEF["Dp"].thr + c["a6_safety"])
            dl[lam].append((ox - oa).abs().max().item())
        del za, zx, lg_a, F_a, lg_x, F_x
        torch.cuda.empty_cache()
    rec("(a) stand-in features max |A8x - A8|", max(da))
    for lam, v in dl.items():
        rec(f"(a) loss lambda {lam:g} max |A8x - A8|", max(v))
    # (b) IG-20 Gauss-Legendre vs captum IntegratedGradients (benign = predicted class on these rows)
    assert c["ig_steps_det"] == IG_STEPS
    ig_x = np.concatenate([ig_gl(model, z[i:i + CHUNK].clone(), cls[i:i + CHUNK], IG_STEPS, create_graph=False)
                           .detach().cpu().numpy() for i in range(0, len(rows), CHUNK)])
    ig_c, _, _ = integrated_gradients(model, z, cls, torch.zeros(1, z.shape[1], device=dev),
                                      {**c, "ig_steps": c["ig_steps_det"]})
    rec("(b) IG-20 raw max |A8x - captum|", np.abs(ig_x - ig_c).max())
    rec("(b) IG-20 normalised max |A8x - captum|", np.abs(norm_abs(ig_x) - norm_abs(ig_c)).max())
    log(f"{year} sanity (b, info) IG-20 normalised max |A8x - saved set|: "
        f"{np.abs(norm_abs(ig_x) - V9['ig'][rows]).max():.3e}")
    # (c) true D_plus features inside A8x vs the saved D_plus features of the same rows
    Ft, st = [], []
    for i in range(0, len(rows), CHUNK):
        zc = z[i:i + CHUNK].clone().requires_grad_(True)
        lg, F, raw = features_x(model, zc, cls[i:i + CHUNK], G.base, TAU1, "gl20")
        s_true, Fi = true_dplus(lg, raw, zc, G.base, DEF["Dp"])
        Ft.append(Fi), st.append(s_true)
        del zc, lg, F, raw
        torch.cuda.empty_cache()
    Ft, s_true = np.concatenate(Ft), np.concatenate(st)
    Fs, cols = plus_features(V9["sc"].iloc[rows].reset_index(drop=True))
    d = np.abs(Ft - Fs[cols].to_numpy())
    for j, col in enumerate(cols):
        if col == "mahalanobis_benign":     # Amendment 1: relative tolerance 1e-3
            rel = d[:, j] / np.abs(Fs[cols].to_numpy()[:, j]).clip(1e-30)
            log(f"{year} sanity (c, info) {col} max |inside - saved| absolute: {d[:, j].max():.3e}")
            rec(f"(c) {col} max relative |inside - saved|", rel.max(), TOL_MAHA_REL, "rel")
        else:
            rec(f"(c) {col} max |inside - saved|", d[:, j].max())
    sv = np.concatenate(svs)
    rec("(c) D_plus decision max |inside - saved|", np.abs(s_true - sv).max(), TOL_DEC_ABS, "abs (Amendment 1)")
    # Amendment 1, effect of the selection margin on these rows (all predicted benign): undetected without margin
    # (true score < thr) vs with margin (< thr - SEL_MARGIN), and vs the saved evaluation (saved score < thr).
    thr = DEF["Dp"].thr
    u0, u1, us = s_true < thr, s_true < thr - SEL_MARGIN, sv < thr
    log(f"{year} sanity (margin, info) rows undetected: inside no margin {int(u0.sum())}, inside with margin "
        f"{int(u1.sum())}, saved evaluation {int(us.sum())}; differ margin vs no margin {int((u0 != u1).sum())}, "
        f"inside no margin vs saved {int((u0 != us).sum())}, inside with margin vs saved {int((u1 != us).sum())}")
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
    S = G.Sc
    cp = {**c, "steps": PILOT_STEPS, "restarts": 0}
    torch.cuda.empty_cache()
    torch.cuda.reset_peak_memory_stats()
    wd.phase = f"pilot {year} attack"
    sh0 = shared_usage_gb()
    t0 = time.time()
    X, z, _ = run_x(model, S, G, cp, DEF["Dp"], 1.0, None, attack_seed(s, "pilot"), dev)
    torch.cuda.synchronize()
    ta = time.time() - t0
    sh_a = shared_usage_gb()
    check_valid(X, S.R, G, EPS, S, f"{year} pilot")
    wd.phase = f"pilot {year} scoring"
    t1 = time.time()
    df = score_rows(model, z, G, c, DEF, {"pos": G.pos[G.common][rows]})
    torch.cuda.synchronize()
    ts = time.time() - t1
    sh_s = shared_usage_gb()
    res = torch.cuda.max_memory_reserved() / 2 ** 30
    alc = torch.cuda.max_memory_allocated() / 2 ** 30
    shared_max = max(x for x in (sh0, sh_a, sh_s, wd.max_shared) if x == x)
    n_iter = PILOT_STEPS + 1
    per_step = ta / n_iter
    full_iter = (c["steps"] + 1) * (1 + c["restarts"])
    run_s = per_step * full_iter + ts
    rec = {"year": year, "rows": len(rows), "chunk": CHUNK, "steps": PILOT_STEPS, "attack_s": ta, "scoring_s": ts,
           "sec_per_iter_all_rows": per_step, "max_reserved_gb": res, "max_allocated_gb": alc,
           "shared_gb_start": sh0, "shared_gb_after_attack": sh_a, "shared_gb_after_scoring": sh_s,
           "shared_gb_max": shared_max, "est_min_per_run": run_s / 60,
           "est_hours_9_runs": run_s * len(SETTINGS) * len(SEEDS) / 3600, "valid_rows": len(X),
           "pass_shared": shared_max < LIM_SHARED_GB, "pass_reserved": res < LIM_RESERVED_GB}
    log(f"{year} pilot: {rec}")
    pd.DataFrame([rec]).to_csv(OUT / year / "pilot.csv", index=False)
    if not (rec["pass_shared"] and rec["pass_reserved"]):
        raise StopRun(f"pilot memory limits failed on {year}: shared max {shared_max:.2f} GB, reserved {res:.2f} GB")
    return rec


# ---------------------------------------------------------------- attack

def interim(log):
    # Lambda 1 only, 2017, V1: per seed, D_plus union over A1 + A8x_lam1 minus D union over A1 + A6_lam1.
    year = "2017"
    parts = []
    for s in SEEDS:
        base = pd.read_parquet(OUT9B / year / f"seed{s}" / "rows_base.parquet")
        x = pd.read_parquet(OUT / year / f"seed{s}" / "rows_A8x_lam1.parquet")
        R = pd.concat([base, x], ignore_index=True)
        R = R[R.pos.isin(keep_pos(year, "V1"))]
        dD = union_ind(R, "D", ["A1", "A6_lam1"]).mean()
        dP = union_ind(R, "Dplus", ["A1", "A8x_lam1"]).mean()
        parts.append(f"seed {s} {dP - dD:+.3f}")
    line = (f"{time.strftime('%Y-%m-%d %H:%M')} 2017 V1, lambda 1 only, D+ union (A1 + A8x_lam1) minus D union "
            f"(A1 + A6_lam1): " + ", ".join(parts))
    with open(OUT / "interim.txt", "a", encoding="utf-8") as f:
        f.write(line + "\n")
    log(f"interim: {line}")


def attack(year, dev, log, t_end, wd):
    c = cfg(year)
    G = load_G(year, c, dev)
    S = G.Sc
    pos = G.pos[G.common]
    cls_names = [G.classes[k] for k in G.yte[G.common]]
    for tag, lam, nw in SETTINGS:
        for s in SEEDS:
            sdir = OUT / year / f"seed{s}"
            sdir.mkdir(parents=True, exist_ok=True)
            name = f"A8x_{tag}"
            p = sdir / f"rows_{name}.parquet"
            if p.exists():
                continue
            wd.phase = f"{year} seed {s} {name}"
            model = load_model(year, s, dev)
            DEF = load_defs(year, s, EPS, dev)
            t0 = time.time()
            X, z, s_sel = run_x(model, S, G, c, DEF["Dp"], lam, nw, attack_seed(s, tag), dev)
            check_valid(X, S.R, G, EPS, S, f"{year} seed {s} {name}")
            ta = time.time() - t0
            df = score_rows(model, z, G, c, DEF, {"seed": s, "eps": EPS, "attack": name, "pos": pos,
                                                   "class": cls_names, "valid": True, "score_Dplus_sel": s_sel})
            df["seconds_attack"], df["seconds_total"] = ta, time.time() - t0
            np.savez_compressed(sdir / f"X_{name}.npz", X=X, pos=pos)
            df.to_parquet(p, index=False)
            u = df.flipped & (df.score_Dplus < df.thr_Dplus)
            log(f"{year} seed {s} {name}: success {df.flipped.mean():.3f}, succ+undet Dplus {u.mean():.3f} "
                f"[attack {ta:.0f} s, total {time.time() - t0:.0f} s, max reserved "
                f"{torch.cuda.max_memory_reserved() / 2 ** 30:.2f} GB]")
            del model
            torch.cuda.empty_cache()
            if year == "2017" and tag == "lam1" and s == SEEDS[-1]:
                interim(log)
            if time.time() > t_end:
                raise StopRun("runtime cap reached")


# ---------------------------------------------------------------- summary (CPU)

def summary(log):
    urows, trows, prow = [], [], []
    for year in YEARS:
        R = []
        for s in SEEDS:
            sdir = OUT / year / f"seed{s}"
            base = pd.read_parquet(OUT9B / year / f"seed{s}" / "rows_base.parquet")
            R.append(base)
            R += [pd.read_parquet(sdir / f"rows_{n}.parquet") for n in A8XN if (sdir / f"rows_{n}.parquet").exists()]
            for nm in A8N:
                a8 = pd.read_parquet(A8_DIR[year] / f"seed{s}" / f"rows_eps1_{nm}.parquet")
                a8 = a8[a8.pos.isin(set(base.pos))][["seed", "pos", "class", "flipped", "score_D", "score_Dplus",
                                                     "thr_D", "thr_Dplus"]].copy()
                a8["attack"] = nm
                R.append(a8)
        R = pd.concat(R, ignore_index=True)
        if not R.attack.isin(A8XN).any():
            continue
        for rv in ("V0", "V1", "V2"):
            Rv = R[R.pos.isin(keep_pos(year, rv))]
            combos = [("headline: D A1+A6x3, D+ A1+A8x x3", ["A1"] + A6N, ["A1"] + A8XN),
                      ("Stage A: D A1+A6x3, D+ A1+A8x3", ["A1"] + A6N, ["A1"] + A8N),
                      ("sensitivity (unequal effort): D A1+A6x3, D+ A1+A8x3+A8x x3", ["A1"] + A6N,
                       ["A1"] + A8N + A8XN)]
            combos += [(f"per setting {t}: D A1+A6_{t}, D+ A1+A8x_{t}", ["A1", A6N[i]], ["A1", A8XN[i]])
                       for i, (t, _, _) in enumerate(SETTINGS)]
            for lab, nD, nP in combos:
                U = pd.concat([union_ind(Rv, "D", nD), union_ind(Rv, "Dplus", nP)], axis=1).reset_index()
                for sl, sd in [(f"seed {q}", [q]) for q in SEEDS] + [("pooled", SEEDS)]:
                    u = U[U.seed.isin(sd)]
                    rng = np.random.default_rng(zlib.crc32(f"{year}|{rv}|{lab}|{sl}".encode()))
                    lo, hi = boot_diff(u.D.to_numpy(), u.Dplus.to_numpy(), u.seed.to_numpy(), 1000, rng)
                    urows.append({"year": year, "rowset": rv, "union": lab, "seeds": sl, "n_rows": len(u),
                                  "undet_D": u.D.mean(), "undet_Dplus": u.Dplus.mean(),
                                  "Dplus_minus_D": u.Dplus.mean() - u.D.mean(), "lo": lo, "hi": hi})
            negs = pd.concat([pd.read_parquet(OUT9 / year / f"seed{s}" / "scores_neg_test_eps1.parquet")
                              for s in SEEDS], ignore_index=True)
            cen = lambda d, x: (x[f"score_{d}"] - x[f"thr_{d}"]).to_numpy()
            for nm in A8XN + A8N:
                r = Rv[Rv.attack == nm]
                f = r.flipped.to_numpy()
                pb = paired_boot(cen("D", negs), cen("D", r)[f], 0.0, cen("Dplus", negs), cen("Dplus", r)[f], 0.0,
                                 1000, np.random.default_rng(zlib.crc32(f"{year}|{rv}|{nm}|tpr".encode())))
                trows.append({"year": year, "rowset": rv, "attack": nm, "n_flipped": int(f.sum()),
                              "success": float(f.mean()),
                              "undet_Dplus": float((r.flipped & (r.score_Dplus < r.thr_Dplus)).mean()), **pb})
    U, T = pd.DataFrame(urows), pd.DataFrame(trows)
    U.to_csv(OUT / "summary_union.csv", index=False)
    T.to_csv(OUT / "summary_tpr.csv", index=False)
    log(f"summary written: {len(U)} union rows, {len(T)} TPR rows")


# ---------------------------------------------------------------- main

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("phase", choices=["sanity", "pilot", "attack", "summary"])
    ap.add_argument("--years", nargs="+", default=YEARS)
    ap.add_argument("--max-hours", type=float, default=5.0)
    a = ap.parse_args()
    OUT.mkdir(parents=True, exist_ok=True)
    log = Log(OUT / f"log_{a.phase}.txt")
    t_end = time.time() + a.max_hours * 3600
    log(f"start {a.phase} years {a.years} chunk {CHUNK} cap {a.max_hours} h pid {os.getpid()}")
    t0 = time.time()
    try:
        if a.phase == "summary":
            summary(log)
        else:
            dev = torch.device("cuda")
            wd = Watchdog(log, OUT / "memlog.csv", every=10 if a.phase == "pilot" else 60)
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
