# Stage 9b (Task 33): Stage B of the D_plus ablation, adaptive attackers against the controls, both datasets, eps 1.
#
# Plan: handoff/32_restructure_plan.md (section 4); decisions and the pre-registered rule: handoff/33_decisions.md.
# Nothing existing is modified; every output goes to results/stage9/adaptive/{2017,2018}/seed{s}/ (smoke runs to
# results/stage9/adaptive_smoke/).
#
# Controls attacked (Stage A definitions in src/stage9_dplus_ablation.py, detectors refit and checked by gate0b):
#   a1   D's 4 scores + 12 consistency scores between a FIXED partner attention and the row's own attribution. The
#        partner is the clean (unattacked) source row given by a derangement within the true class (oracle label),
#        seeded by "a1fixed|year|seed"; attacker and evaluation use the same vector.
#   a2   as a1, with one fixed draw per source row from the clean benign bank (bank_benign of Stage A), seeded by
#        "a2fixed|year|seed"; attacker and evaluation use the same vector.
#   b    12 consistency scores between {occlusion, GradientSHAP} and {|gxi|, |IG|}. Stand-ins: occlusion exact
#        (training-mean masking, differentiable), GradientSHAP with m = 8 samples (fresh seeded draws per step),
#        IG with 8 steps, soft ranks; evaluation with the true features (GradientSHAP 50 samples, IG 20 steps).
#   c0   D's 4 scores with C = 1 (exact gradients, A6-type).
#   c1   D's 4 scores + 12 N(0, 1) noise columns, fixed per source row ("c1fixed|year|seed"; constant for the attacker).
#   c2   D's 4 scores + the 12 logit and distance statistics of Stage A (exact gradients, A6-type).
# Attackers: the A6/A8 objective of src/stage6d_final.attack_chunk (min(margin, 10) - lambda relu(score - thr + 0.5);
# lambda 1, 100 and the normalised variant w 0.5), 100 steps, 1 restart, eps 1, against each control's threshold.
# Attributions inside the loss use the benign class (as A8); evaluation always uses the true features.
# 2017 rows: the 7 common classes only (1,044 of 1,494 source rows per seed); 2018: all 985 rows.
#
# Phases:
#   gate0b           refit every Stage A control on the saved sets; thresholds vs the saved thr_* (1e-5), val AUROC vs
#                    stageA_defenders.csv (1e-9), A6_lam1 rescoring vs the saved scores (1e-5); saves the controls.
#   standin_quality  R^2 and Spearman of every differentiable control feature vs the true feature (2,000 held-out
#                    benign val rows + the flipped val A1 positives, per seed), with cls = predicted class.
#   pilot            per control and dataset (seed 0): 4 PGD steps on 500 rows, chunk 100 then 50; runtime estimate.
#   attack           the full runs (resumable per file, --max-hours cap).
#   summary          equal-effort unions (pre-registered headline) and paired TPR differences.
#   smoke            attack + summary on 20 rows per class, 2 steps, seed 0, into adaptive_smoke/.
#
# Usage:
#   .venv\Scripts\python.exe -m src.stage9b_dplus_adaptive gate0b [--years 2017 2018]
#   .venv\Scripts\python.exe -m src.stage9b_dplus_adaptive {standin_quality,pilot,smoke,attack,summary} [--controls ...]
import argparse
import pickle
import sys
import time
import zlib

import numpy as np
import pandas as pd
import torch
from scipy.stats import spearmanr
from sklearn.linear_model import LinearRegression

from src.stage4_constraints import derive
from src.stage6_defense import BENIGN
from src.stage6b_defense import auroc
from src.stage6c_defense import FEATS_D, feats_torch
from src.stage6d_final import LinearDefender, attack_chunk, paired_boot, plus_features, sub_rows
from src.stage6d_final import load_common as load_common17
from src.stage6_defense import make_rows
from src.stage6e_strong import cosine, ig_steps, soft_jaccard, soft_spearman
from src.stage7e_defense_2018 import load_common7
from src.stage9_dplus_ablation import COMMON, DS, OUT as OUT9, SEEDS, TOL, VARIANTS, Log, StopRun, check_valid, \
    config, cons12_chunked, derangement, gs_background, keep_pos, load_defs, load_model, load_set, stats12, \
    variant_features, vectors, vectors_b
from src.utils import ROOT

RES = ROOT / "results"
OUT = RES / "stage9" / "adaptive"
OUT_SMOKE = RES / "stage9" / "adaptive_smoke"
EPS = 1.0
CONTROLS = ["c0", "c1", "c2", "a1", "a2", "b"]
DETS = ["D", "Dplus"] + CONTROLS
SETTINGS = [("lam1", 1.0, None), ("lam100", 100.0, None), ("norm_w0.5", 0.0, 0.5)]
A6N = ["A6_lam1", "A6_lam100", "A6norm_w0.5"]
A8N = ["A8_lam1", "A8_lam100", "A8norm_w0.5"]
A8_DIR = {"2017": RES / "stage6" / "strong", "2018": RES / "stage7" / "stage7e"}
GS_M = 8
B_CAP_HOURS = 12.0
VRAM_GB = 7.5          # RTX 4060 Laptop, 8 GB; a larger peak means the allocation spilled into system memory


def cfg(year):
    c = config(year)
    c.setdefault("tau", 0.1)
    c.setdefault("ig_steps_loss", 8)
    return c


def load_G(year, c, dev):
    G = load_common17(c, dev) if year == "2017" else load_common7(c, dev)
    assert set(COMMON) <= set(G.classes), G.classes
    G.common = np.flatnonzero(np.isin(np.array(G.classes)[G.yte], COMMON))    # indices into G.pos (Stage A order)
    G.Sc = make_rows(G.Rte[G.common], G.yte[G.common], G.classes, G.feats, G.spec, G.floors, G.tp, dev)
    return G


def det_to(o, dev):
    o.mu, o.sd, o.w = o.mu.to(dev), o.sd.to(dev), o.w.to(dev)
    return o


def load_detectors(year, s, dev):
    DEF = load_defs(year, s, EPS, dev)
    ctl = pickle.loads((OUT / year / f"seed{s}" / "controls_eps1.pkl").read_bytes())
    dets = {"D": DEF["D"], "Dplus": DEF["Dp"]}
    dets.update({k: det_to(ctl[k], dev) for k in CONTROLS})
    return dets


def banks_of(sdir):
    bb, bv = load_set(sdir / "bank_benign.npz"), load_set(sdir / "bank_valsrc.npz")
    banks = {BENIGN: (bb["ro"], bb["me"])}
    for k in np.unique(bv["pred"]):
        if int(k) != BENIGN:
            m = bv["pred"] == k
            banks[int(k)] = (bv["ro"][m], bv["me"][m])
    return banks


def c1_noise(year, s, n):
    return np.random.default_rng(zlib.crc32(f"c1fixed|{year}|{s}".encode())).standard_normal((n, 12))


def consts_for(year, s, G, model, c, dev):
    # Fixed per-row constants on the common source rows (cached): a1 partner attention, a2 bank draw, c1 noise.
    p = OUT / year / f"seed{s}" / "consts.npz"
    if p.exists():
        z = np.load(p)
        return {k: z[k] for k in z.files}
    V = vectors(model, G.Sc.z0, G.base, c, attn_only=True)
    idx = derangement(G.yte[G.common], np.random.default_rng(zlib.crc32(f"a1fixed|{year}|{s}".encode())))
    bb = load_set(OUT9 / year / f"seed{s}" / "bank_benign.npz")
    j = np.random.default_rng(zlib.crc32(f"a2fixed|{year}|{s}".encode())).integers(0, len(bb["ro"]), len(idx))
    out = {"a1_ro": V["ro"][idx], "a1_me": V["me"][idx], "a1_partner": idx, "a2_ro": bb["ro"][j], "a2_me": bb["me"][j],
           "a2_draw": j, "c1": c1_noise(year, s, len(idx)), "pos": G.pos[G.common]}
    p.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(p, **out)
    return out


# ---------------------------------------------------------------- true features (numpy) for every detector

def all_features(V, set_name, year, s, banks, K, batch):
    # V: a Stage A style set on the common rows (ro, me, gxi, ig, occ, gs, lg, pred, sc, ytrue); K: consts (same rows).
    FT = variant_features(V, set_name, year, s, EPS, banks, batch)
    base4 = FT["D"]
    B = {"gxi": V["gxi"], "ig20": V["ig"]}
    for k in ("a1", "a2"):
        cons = cons12_chunked({"rollout": K[f"{k}_ro"], "attnmean": K[f"{k}_me"]}, B, batch)
        FT[k] = np.c_[base4, -cons.to_numpy()]
    FT["c1"] = np.c_[base4, K["c1"]]
    return {k: FT[k] for k in DETS}


def score_frame(FT, dets, meta):
    return pd.DataFrame({**meta, **{f"score_{k}": dets[k].decision(FT[k]) for k in DETS},
                         **{f"thr_{k}": dets[k].thr for k in DETS}})


# ---------------------------------------------------------------- differentiable control features

def nrm(t):
    return t / t.sum(1, keepdim=True).clamp_min(1e-12)


def cons_standin(As, Bs, tau):
    # Column order of cons12 / variant_features: attention-side outer, attribution-side inner; Spearman,
    # top-10 Jaccard + 1e-3 Spearman, cosine (positive orientation, as -cons12).
    f = []
    for a in As:
        for b in Bs:
            sp = soft_spearman(a, b, tau)
            f += [sp, soft_jaccard(a, b, 10, tau) + 1e-3 * sp, cosine(a, b)]
    return torch.stack(f, 1)


def own_attr(model, z, cls, c):
    lg = model(z)
    g, = torch.autograd.grad(lg.gather(1, cls[:, None]).sum(), z, create_graph=True)
    return lg, nrm((z * g).abs()), nrm(ig_steps(model, z, cls, c["ig_steps_loss"]).abs())


def occl_torch(model, z, cls, lg):
    # Leave-one-feature-out with training-mean masking (0 in the preprocessed space), target logit, differentiable.
    B, F = z.shape
    keep = 1.0 - torch.eye(F, device=z.device, dtype=z.dtype)
    lo = model((z[:, None, :] * keep[None]).reshape(-1, F)).view(B, F, -1)
    return lg.gather(1, cls[:, None]) - lo.gather(2, cls.view(B, 1, 1).expand(B, F, 1)).squeeze(2)


def gshap_torch(model, z, cls, bg, m, gen):
    # GradientSHAP estimator of gshap_fast with m samples, differentiable (double backprop).
    B, F = z.shape
    idx = torch.randint(0, bg.shape[0], (B, m), generator=gen, device=z.device)
    a = torch.rand(B, m, 1, generator=gen, device=z.device)
    base = bg[idx]
    xs = (base + a * (z[:, None] - base)).reshape(-1, F)
    lg = model(xs)
    g, = torch.autograd.grad(lg.gather(1, cls.repeat_interleave(m)[:, None]).sum(), xs, create_graph=True)
    return (g.view(B, m, F) * (z[:, None] - base)).mean(1)


def stats12_torch(lg, z, base):
    # Same 12 statistics as stage9 stats12 (Mahalanobis and 5-NN of the predicted class, class fixed per step).
    logp = torch.log_softmax(lg, -1)
    p = logp.exp()
    top = lg.topk(3, dim=1).values
    pred = lg.argmax(1).detach()
    d = z - base.mu_t[pred]
    maha = ((d @ base.P) * d).sum(1).clamp_min(1e-12).sqrt()
    knn = torch.zeros(z.shape[0], device=z.device, dtype=z.dtype)
    for k in pred.unique().tolist():
        m = pred == k
        knn = knn.index_put((m.nonzero().squeeze(1),),
                            torch.cdist(z[m], base.ref[k]).topk(base.k, largest=False).values.mean(1))
    return torch.stack([-(p * logp).sum(1), -torch.logsumexp(lg, 1), lg.norm(dim=1), top[:, 0], top[:, 1],
                        top[:, 1] - top[:, 2], lg[:, BENIGN], lg[:, 1:].amax(1), lg.std(1, unbiased=False),
                        p[:, BENIGN], maha, knn], 1)


def control_feats(kind, model, z, cls, base, c, K, gs_bg, gen):
    # (logits, feature matrix in the column order of the control's detector); K = per-row constants (torch).
    if kind in ("c0", "c1", "c2"):
        lg = model(z)
        b4 = feats_torch(lg, z, base)
        extra = {"c0": None, "c1": K.get("c1"), "c2": stats12_torch(lg, z, base) if kind == "c2" else None}[kind]
        return lg, b4 if extra is None else torch.cat([b4, extra], 1)
    lg, gxi, ig = own_attr(model, z, cls, c)
    if kind in ("a1", "a2"):
        As = (K[f"{kind}_ro"], K[f"{kind}_me"])
    else:
        As = (nrm(occl_torch(model, z, cls, lg).abs()), nrm(gshap_torch(model, z, cls, gs_bg, GS_M, gen).abs()))
    return lg, torch.cat([feats_torch(lg, z, base), cons_standin(As, (gxi, ig), c["tau"])], 1)


def terms_control(kind, model, tp, base, det, c, K, gs_bg, gen):
    def f(xv, ctx):
        z = tp(derive(xv, ctx)).float()
        cls = torch.full((z.shape[0],), BENIGN, dtype=torch.long, device=z.device)
        lg, F = control_feats(kind, model, z, cls, base, c, K, gs_bg, gen)
        return lg[:, BENIGN] - lg[:, 1:].amax(1), det.torch_decision(F)
    return f


def k_torch(K, idx, dev):
    return {k: torch.from_numpy(np.asarray(K[k][idx], dtype=np.float32)).to(dev)
            for k in ("a1_ro", "a1_me", "a2_ro", "a2_me", "c1")}


def run_attack(kind, model, S, rows, G, c, det, K, gs_bg, lam, nw, seed, chunk, dev):
    # attack_loop of stage6d with per-chunk constants: rows = indices into the common rows (S = G.Sc).
    xs, zs = [], []
    for i in range(0, len(rows), chunk):
        idx = rows[i:i + chunk]
        gen = torch.Generator(device=dev).manual_seed(seed + 7919 + i)
        fn = terms_control(kind, model, G.tp, G.base, det, c, k_torch(K, idx, dev), gs_bg, gen)
        x, z = attack_chunk(model, sub_rows(S, idx), G.tp, EPS, c, lam, nw, fn, det.thr, dev, seed + i)
        xs.append(x), zs.append(z)
    return np.concatenate(xs), torch.cat(zs)


def attack_seed(kind, s, name):
    return 4026 + 9500 + 17 * s + sum(map(ord, f"{kind}|{name}"))


# ---------------------------------------------------------------- gate0b

def gate0b(year, dev, log):
    c = cfg(year)
    rows = []
    defs = pd.read_csv(OUT9 / year / "stageA_defenders.csv")
    for s in SEEDS:
        sdir9 = OUT9 / year / f"seed{s}"
        banks = banks_of(sdir9)
        Ntr, Nho, Nte = (load_set(sdir9 / f"neg_{k}.npz") for k in ("val_train", "val_held", "test"))
        for eps in c["eps"]:
            t0 = time.time()
            Vp = load_set(sdir9 / f"pos_eps{eps:g}.npz")
            FT = {k: variant_features(V, k, year, s, eps, banks, c["score_batch"])
                  for k, V in (("neg_val_train", Ntr), ("neg_val_held", Nho), ("neg_test", Nte), ("pos", Vp))}
            DEF = load_defs(year, s, eps, dev)
            dets = {"D": DEF["D"], "Dplus": DEF["Dp"]}
            for v in VARIANTS:
                cols = [f"f{i}" for i in range(FT["pos"][v].shape[1])]
                dets[v] = LinearDefender(cols, FT["neg_val_train"][v], FT["pos"][v], FT["neg_val_held"][v], c["fpr"],
                                         c["C_plus"], dev)
            sv1 = pd.read_parquet(sdir9 / f"scores_eps{eps:g}_A1.parquet")
            sv6 = pd.read_parquet(sdir9 / f"scores_eps{eps:g}_A6_lam1.parquet")
            V6 = load_set(sdir9 / f"att_eps{eps:g}_A6_lam1.npz")
            F6 = variant_features(V6, "A6_lam1", year, s, eps, banks, c["score_batch"])
            for v, o in dets.items():
                thr_saved = float(sv1[f"thr_{v}"].iloc[0])
                d = defs[(defs.seed == s) & (defs.eps == eps) & (defs.detector == v)].iloc[0]
                au = auroc(o.decision(FT["neg_val_held"][v]), o.decision(FT["pos"][v]))
                dsc = float(np.abs(o.decision(F6[v]) - sv6[f"score_{v}"].to_numpy()).max())
                for chk, val, tol in (("threshold |diff| vs saved thr", abs(o.thr - thr_saved), TOL["thr"]),
                                      ("val AUROC |diff| vs stageA_defenders.csv", abs(au - d.val_auroc), TOL["auroc"]),
                                      ("A6_lam1 rescoring max |diff| vs saved score", dsc, TOL["score"])):
                    rows.append({"year": year, "seed": s, "eps": eps, "detector": v, "check": chk, "value": val,
                                 "tol": tol, "pass": bool(val <= tol)})
            ok = all(r["pass"] for r in rows if r["seed"] == s and r["eps"] == eps)
            log(f"{year} seed {s} eps {eps}: gate0b {'PASS' if ok else 'FAIL'} [{time.time() - t0:.0f} s]")
            if eps == EPS:
                ctl = {}
                for v in VARIANTS:
                    o = dets[v]
                    o.mu, o.sd, o.w = o.mu.cpu(), o.sd.cpu(), o.w.cpu()
                    ctl[v] = o
                (OUT / year / f"seed{s}").mkdir(parents=True, exist_ok=True)
                (OUT / year / f"seed{s}" / "controls_eps1.pkl").write_bytes(pickle.dumps(ctl))
    df = pd.DataFrame(rows)
    df.to_csv(OUT / year / "gate0b.csv", index=False)
    if not df["pass"].all():
        raise StopRun(f"gate0b failed on {year}: {df[~df['pass']].head(10).to_dict('records')}")
    return df


# ---------------------------------------------------------------- stand-in quality

def standin_quality(year, dev, log, seeds):
    c = cfg(year)
    G = load_G(year, c, dev)
    gs_bg = gs_background(year, dev)
    out = []
    for s in seeds:
        t0 = time.time()
        sdir9 = OUT9 / year / f"seed{s}"
        model = load_model(year, s, dev)
        banks = banks_of(sdir9)
        ix = np.load(sdir9 / "indices.npz")
        Nho, Vp = load_set(sdir9 / "neg_val_held.npz"), load_set(sdir9 / "pos_eps1.npz")
        nn = 2000
        Vn = {k: (v.iloc[:nn].reset_index(drop=True) if k == "sc" else v[:nn]) for k, v in Nho.items() if k != "gs_impl"}
        Vn["ytrue"] = np.zeros(nn, dtype=np.int64)
        V = {k: (pd.concat([Vn[k], Vp[k]], ignore_index=True) if k == "sc" else np.concatenate([Vn[k], Vp[k]]))
             for k in ("ro", "me", "gxi", "ig", "occ", "gs", "lg", "pred", "sc", "ytrue")}
        Xp = np.load(sdir9 / "pos_eps1_X.npz")
        with torch.no_grad():
            Z = torch.cat([G.Zben_va[torch.from_numpy(ix["held"][:nn]).to(dev)],
                           G.tp(torch.from_numpy(Xp["X_all"][Xp["flip"]]).to(dev)).float()])
        assert len(Z) == len(V["pred"])
        n = len(Z)
        # constants for this check set: a1 partner within the set (true class of positives, one group for negatives),
        # a2 one benign-bank draw per row, c1 noise
        idx = derangement(V["ytrue"], np.random.default_rng(zlib.crc32(f"sq|a1|{year}|{s}".encode())))
        bro, bme = banks[BENIGN]
        j = np.random.default_rng(zlib.crc32(f"sq|a2|{year}|{s}".encode())).integers(0, len(bro), n)
        K = {"a1_ro": V["ro"][idx], "a1_me": V["me"][idx], "a2_ro": bro[j], "a2_me": bme[j], "c1": c1_noise(year, s, n),
             "pos": np.arange(n)}
        true = all_features(V, "standin_check", year, s, banks, K, c["score_batch"])
        for kind in CONTROLS:
            tk = time.time()
            st = []
            gen = torch.Generator(device=dev).manual_seed(1234 + s)
            ch = 50 if kind == "b" else 200
            for i in range(0, n, ch):
                zb = Z[i:i + ch].clone().requires_grad_(True)
                cls = torch.from_numpy(V["pred"][i:i + ch]).long().to(dev)
                _, F = control_feats(kind, model, zb, cls, G.base, c, k_torch(K, np.arange(i, min(i + ch, n)), dev),
                                     gs_bg, gen)
                st.append(F.detach().cpu().numpy())
            S = np.concatenate(st).astype(np.float64)
            T = true[kind]
            names = FEATS_D + ([] if kind == "c0" else [f"x{q}" for q in range(12)])
            for q, nm in enumerate(names):
                x, y = S[:, q], T[:, q]
                r2 = LinearRegression().fit(x[:, None], y).score(x[:, None], y) if np.std(x) > 0 and np.std(y) > 0 \
                    else np.nan
                out.append({"year": year, "seed": s, "control": kind, "col": q, "feature": nm, "r2": r2,
                            "spearman": spearmanr(x, y).statistic, "max_abs_diff": float(np.abs(x - y).max()),
                            "n_rows": n})
            log(f"{year} seed {s} {kind}: stand-ins on {n} rows in {time.time() - tk:.0f} s")
        log(f"{year} seed {s}: stand-in quality in {time.time() - t0:.0f} s")
        del model
        torch.cuda.empty_cache()
    return pd.DataFrame(out)


# ---------------------------------------------------------------- pilot (timing and memory)

def pilot(year, dev, log, controls, n_rows=500, steps=4):
    c = cfg(year)
    G = load_G(year, c, dev)
    gs_bg = gs_background(year, dev)
    s = 0
    model = load_model(year, s, dev)
    dets = load_detectors(year, s, dev)
    K = consts_for(year, s, G, model, c, dev)
    rows = np.arange(min(n_rows, len(G.common)))
    n_full = len(G.common)
    out = []
    for kind in controls:
        rec = {"year": year, "control": kind, "rows": len(rows), "steps_timed": steps + 1, "n_rows_full": n_full}
        for ch in (100, 50):
            try:
                torch.cuda.empty_cache()
                torch.cuda.reset_peak_memory_stats()
                cc = dict(c, steps=steps, restarts=0)
                torch.cuda.synchronize()
                t0 = time.time()
                X, z = run_attack(kind, model, G.Sc, rows, G, cc, dets[kind], K, gs_bg, 1.0, None, 0, ch, dev)
                torch.cuda.synchronize()
                peak = torch.cuda.max_memory_allocated() / 2**30
                if peak > VRAM_GB:
                    # Windows spills past the physical VRAM into system memory without an OOM (about 8x slower):
                    # treated as an OOM, the next chunk size is tried.
                    log(f"{year} {kind}: peak {peak:.2f} GB > {VRAM_GB} GB at chunk {ch} (spill), treated as OOM")
                    rec[f"spill_{ch}"] = round(peak, 2)
                    continue
                rec.update({"chunk": ch, "sec_per_step_500": (time.time() - t0) / (steps + 1) * 500 / len(rows),
                            "peak_gb": peak})
                break
            except torch.cuda.OutOfMemoryError:
                log(f"{year} {kind}: OOM at chunk {ch}")
                rec[f"oom_{ch}"] = True
                torch.cuda.empty_cache()
        if "chunk" not in rec:
            out.append(rec)
            continue
        check_valid(X, G.Sc.R[rows], G, EPS, make_rows(G.Sc.R[rows], G.Sc.y_np[rows], G.classes, G.feats, G.spec,
                                                         G.floors, G.tp, dev), f"pilot {year} {kind}")
        # true-feature scoring time of the pilot rows
        torch.cuda.synchronize()
        t0 = time.time()
        V = vectors(model, z, G.base, c)
        V.update(vectors_b(model, z, V["pred"], c, gs_bg, c["rng_seed"] + 9900))
        torch.cuda.synchronize()
        sc_sec = (time.time() - t0) * n_full / len(rows)
        steps_full = (c["steps"] + 1) * (1 + c["restarts"])
        per_run = rec["sec_per_step_500"] * steps_full * n_full / 500
        rec.update({"valid": True, "sec_per_run_est": per_run, "score_sec_per_set_est": sc_sec,
                    "hours_full_est": (per_run + sc_sec) * len(SETTINGS) * len(SEEDS) / 3600})
        log(f"{year} {kind}: chunk {rec['chunk']}, {rec['sec_per_step_500']:.2f} s/step/500 rows, peak "
            f"{rec['peak_gb']:.2f} GB -> {per_run / 60:.1f} min per run, {rec['hours_full_est']:.2f} h for 9 runs")
        out.append(rec)
    del model
    torch.cuda.empty_cache()
    return pd.DataFrame(out)


# ---------------------------------------------------------------- attack (full or smoke)

def attack(year, dev, log, t_end, controls, seeds, out_root, rows_per_class=None, steps=None, chunks=None):
    c = cfg(year)
    if steps is not None:
        c["steps"] = steps
    G = load_G(year, c, dev)
    gs_bg = gs_background(year, dev)
    if rows_per_class:
        yc = G.yte[G.common]
        rows = np.concatenate([np.flatnonzero(yc == k)[:rows_per_class] for k in np.unique(yc)])
    else:
        rows = np.arange(len(G.common))
    S = make_rows(G.Sc.R[rows], G.Sc.y_np[rows], G.classes, G.feats, G.spec, G.floors, G.tp, dev)
    for s in seeds:
        sdir = out_root / year / f"seed{s}"
        sdir.mkdir(parents=True, exist_ok=True)
        model = load_model(year, s, dev)
        dets = load_detectors(year, s, dev)
        K = consts_for(year, s, G, model, c, dev)
        banks = banks_of(OUT9 / year / f"seed{s}")
        Kr = {k: v[rows] for k, v in K.items() if k in ("a1_ro", "a1_me", "a2_ro", "a2_me", "c1")}
        pos = G.pos[G.common][rows]
        cls_names = [G.classes[k] for k in G.yte[G.common][rows]]
        for kind in controls:
            ch = (chunks or {}).get(kind, 100)
            for tag, lam, nw in SETTINGS:
                name = f"B_{kind}_{tag}"
                p = sdir / f"rows_{name}.parquet"
                if p.exists():
                    continue
                t0 = time.time()
                X, z = run_attack(kind, model, G.Sc, rows, G, c, dets[kind], K, gs_bg, lam, nw,
                                  attack_seed(kind, s, tag), ch, dev)
                check_valid(X, S.R, G, EPS, S, f"{year} seed {s} {name}")
                ta = time.time() - t0
                V = vectors(model, z, G.base, c)
                V["ytrue"] = G.yte[G.common][rows]
                V.update(vectors_b(model, z, V["pred"], c, gs_bg, c["rng_seed"] + 9900 + 100 * s))
                FT = all_features(V, name, year, s, banks, Kr, c["score_batch"])
                df = score_frame(FT, dets, {"seed": s, "eps": EPS, "attack": name, "control": kind, "pos": pos,
                                            "class": cls_names, "flipped": V["pred"] == BENIGN, "valid": True})
                df["seconds_attack"], df["seconds_total"] = ta, time.time() - t0
                np.savez_compressed(sdir / f"X_{name}.npz", X=X, pos=pos)
                df.to_parquet(p, index=False)
                u = df.flipped & (df[f"score_{kind}"] < df[f"thr_{kind}"])
                log(f"{year} seed {s} {name}: success {df.flipped.mean():.3f}, succ+undet {kind} {u.mean():.3f} "
                    f"[attack {ta:.0f} s, total {time.time() - t0:.0f} s]")
                if time.time() > t_end:
                    raise StopRun("runtime cap reached")
        # A1 and A6 rows of Stage A rescored with the Stage B definitions of a1, a2, c1 (CPU features)
        p = sdir / "rows_base.parquet"
        if not p.exists():
            parts = []
            for name in ["A1"] + A6N:
                V9 = load_set(OUT9 / year / f"seed{s}" / f"att_eps1_{name}.npz")
                m = G.common[rows]
                V = {k: (v.iloc[m].reset_index(drop=True) if k == "sc" else v[m]) for k, v in V9.items()
                     if k != "gs_impl"}
                FT = all_features(V, name, year, s, banks, Kr, c["score_batch"])
                parts.append(score_frame(FT, dets, {"seed": s, "eps": EPS, "attack": name, "control": "",
                                                    "pos": pos, "class": cls_names, "flipped": V["pred"] == BENIGN,
                                                    "valid": True}))
            pd.concat(parts, ignore_index=True).to_parquet(p, index=False)
        del model
        torch.cuda.empty_cache()


# ---------------------------------------------------------------- summary (CPU)

def union_ind(R, det, names):
    sub = R[R.attack.isin(names)]
    u = (sub.flipped & (sub[f"score_{det}"] < sub[f"thr_{det}"])).groupby([sub.seed, sub.pos]).any()
    return u.rename(det)


def boot_diff(a, b, seeds, n, rng):
    # Paired bootstrap of mean(b) - mean(a) over source rows, resampled within seed.
    tot = np.zeros(n)
    N = 0
    for s in np.unique(seeds):
        d = (b[seeds == s].astype(float) - a[seeds == s].astype(float))
        idx = rng.integers(0, len(d), (n, len(d)))
        tot += d[idx].sum(1)
        N += len(d)
    t = tot / N
    return float(np.percentile(t, 2.5)), float(np.percentile(t, 97.5))


def summary(out_root, controls, seeds, log):
    urows, trows = [], []
    for year in DS:
        R = []
        for s in seeds:
            sdir = out_root / year / f"seed{s}"
            if not (sdir / "rows_base.parquet").exists():
                continue
            base = pd.read_parquet(sdir / "rows_base.parquet")
            R.append(base)
            R += [pd.read_parquet(f) for f in sorted(sdir.glob("rows_B_*.parquet"))]
            pos = set(base.pos)
            for nm in A8N:
                a8 = pd.read_parquet(A8_DIR[year] / f"seed{s}" / f"rows_eps1_{nm}.parquet")
                a8 = a8[a8.pos.isin(pos)][["seed", "pos", "class", "flipped", "score_D", "score_Dplus", "thr_D",
                                           "thr_Dplus"]].copy()
                a8["attack"] = nm
                R.append(a8)
        if not R:
            continue
        R = pd.concat(R, ignore_index=True)
        for rv in ("V0", "V1", "V2"):
            Rv = R[R.pos.isin(keep_pos(year, rv))]
            U = pd.concat([union_ind(Rv, "D", ["A1"] + A6N), union_ind(Rv, "Dplus", ["A1"] + A8N)] +
                          [union_ind(Rv, k, ["A1"] + [f"B_{k}_{t}" for t, _, _ in SETTINGS]) for k in controls
                           if Rv.attack.str.startswith(f"B_{k}_").any()], axis=1).reset_index()
            for k in [x for x in controls if x in U.columns]:
                for lab, sd in [(f"seed {q}", [q]) for q in seeds] + [("pooled", list(seeds))]:
                    u = U[U.seed.isin(sd)]
                    rng = np.random.default_rng(zlib.crc32(f"{year}|{rv}|{k}|{lab}".encode()))
                    rec = {"year": year, "rowset": rv, "control": k, "seeds": lab, "n_rows": len(u),
                           "undet_D": u.D.mean(), "undet_Dplus": u.Dplus.mean(), f"undet_control": u[k].mean(),
                           "control_minus_D": u[k].mean() - u.D.mean(),
                           "control_minus_Dplus": u[k].mean() - u.Dplus.mean(),
                           "Dplus_minus_D": u.Dplus.mean() - u.D.mean()}
                    rec["cmD_lo"], rec["cmD_hi"] = boot_diff(u.D.to_numpy(), u[k].to_numpy(), u.seed.to_numpy(), 1000, rng)
                    rec["cmDp_lo"], rec["cmDp_hi"] = boot_diff(u.Dplus.to_numpy(), u[k].to_numpy(), u.seed.to_numpy(),
                                                               1000, rng)
                    urows.append(rec)
            # secondary: paired TPR difference, control minus D, on the control's own adaptive rows (flipped rows)
            for k in controls:
                for t, _, _ in SETTINGS:
                    r = Rv[Rv.attack == f"B_{k}_{t}"]
                    if not len(r):
                        continue
                    negs = pd.concat([pd.read_parquet(OUT9 / year / f"seed{s}" / "scores_neg_test_eps1.parquet")
                                      for s in seeds], ignore_index=True)
                    f = r.flipped.to_numpy()
                    cen = lambda d, x: (x[f"score_{d}"] - x[f"thr_{d}"]).to_numpy()
                    pb = paired_boot(cen("D", negs), cen("D", r)[f], 0.0, cen(k, negs), cen(k, r)[f], 0.0, 1000,
                                     np.random.default_rng(zlib.crc32(f"{year}|{rv}|{k}|{t}|tpr".encode())))
                    trows.append({"year": year, "rowset": rv, "control": k, "attack": f"B_{k}_{t}",
                                  "n_flipped": int(f.sum()), "success": float(f.mean()), **pb})
    U, T = pd.DataFrame(urows), pd.DataFrame(trows)
    U.to_csv(out_root / "summary_union.csv", index=False)
    T.to_csv(out_root / "summary_tpr.csv", index=False)
    log(f"summary written: {len(U)} union rows, {len(T)} TPR rows")
    return U, T


# ---------------------------------------------------------------- main

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("phase", choices=["gate0b", "standin_quality", "pilot", "attack", "summary", "smoke"])
    ap.add_argument("--years", nargs="+", default=list(DS))
    ap.add_argument("--controls", nargs="+", default=CONTROLS)
    ap.add_argument("--seeds", nargs="+", type=int, default=SEEDS)
    ap.add_argument("--max-hours", type=float, default=4.0)
    a = ap.parse_args()
    root = OUT_SMOKE if a.phase == "smoke" else OUT
    root.mkdir(parents=True, exist_ok=True)
    log = Log(root / f"log_{a.phase}.txt")
    t_end = time.time() + a.max_hours * 3600
    dev = torch.device("cuda")
    log(f"start {a.phase} years {a.years} controls {a.controls} seeds {a.seeds} cap {a.max_hours} h")
    t0 = time.time()
    try:
        if a.phase == "summary":
            summary(OUT, a.controls, a.seeds, log)
        for year in a.years:
            (OUT / year).mkdir(parents=True, exist_ok=True)
            if a.phase == "gate0b":
                df = gate0b(year, dev, log)
                log(f"{year}: gate0b PASS ({len(df)} checks, max threshold diff "
                    f"{df[df.check.str.startswith('threshold')].value.max():.1e})")
            elif a.phase == "standin_quality":
                df = standin_quality(year, dev, log, a.seeds)
                df.to_csv(OUT / year / "standin_quality.csv", index=False)
            elif a.phase == "pilot":
                df = pilot(year, dev, log, a.controls)
                df.to_csv(OUT / year / "pilot.csv", index=False)
            elif a.phase == "smoke":
                attack(year, dev, log, t_end, a.controls, [0], OUT_SMOKE, rows_per_class=20, steps=2)
            elif a.phase == "attack":
                pl = OUT / year / "pilot.csv"
                chunks = pd.read_csv(pl).dropna(subset=["chunk"]).set_index("control")["chunk"].astype(int).to_dict() \
                    if pl.exists() else None
                attack(year, dev, log, t_end, a.controls, a.seeds, OUT, chunks=chunks)
        if a.phase == "smoke":
            summary(OUT_SMOKE, a.controls, [0], log)
    except StopRun as e:
        log(f"STOPPED: {e}")
        print(f"STOPPED: {e}")
        sys.exit(2)
    log(f"finished {a.phase} in {time.time() - t0:.0f} s")
    print(f"finished {a.phase} in {time.time() - t0:.0f} s")


if __name__ == "__main__":
    main()
