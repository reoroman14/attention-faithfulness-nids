# Stage 9 (Task 19): D_plus ablation, Gate 0 and Stage A+ (non-adaptive), both datasets, 3 seeds, eps 0.5 and 1.0.
#
# Plan: handoff/18_ablation_plan.md, with the decisions of Task 19. Nothing existing is modified; every output goes to
# results/stage9/ablation/{2017,2018}/seed{s}/ (and a run log there).
#
# Reproduction (Gate 0): the configuration is read from the saved run config of the published runs
# (results/runs/20261005-170553_stage6d_final/config.json for 2017, results/runs/20261006-144518_stage7e_defense_2018/
# config.json for 2018), the data and baselines are built with the original load_common functions, and the negatives,
# validation positives and A6 rows are regenerated with the original random streams and seeds:
#   negatives   rng_seed + 8000 + s (val train 20,000 / held-out 50,000 / test 1,000, all predicted benign)
#   positives   constrained A1 on the val source rows, seed rng_seed + 7100 + int(100 eps) + 10 s, valid and flipped
#   A6          attack_loop with the saved D, seed rng_seed + 7300 + int(100 eps) + 17 s + sum(map(ord, name))
# Gate 0 passes when (per dataset, seed, eps) n_val_pos equals the saved value, the refitted D and D_plus reproduce the
# saved thresholds and validation AUROC, the A1 test rows re-scored with the refitted D_plus reproduce the saved
# score_Dplus, and (per dataset, seed) the 18 recomputed detector scores of the 71,000 negatives (12 consistency scores
# from cons12 on the row's own attention) equal the saved negatives.pkl, all to float tolerance.
#
# Detectors (all trained on the same regenerated positives and negatives; StandardScaler, class_weight balanced,
# threshold at 5% FPR on the same 50,000 held-out negatives):
#   D        as published (saved object, C = 1e6)          Dplus  as published (saved object, C = 1)
#   a1, a1s  D_plus features with the attention (rollout and layer mean together) of another row of the same pool:
#            pool = (set, true class); a fixed cyclic derangement of a random permutation (seed 0; a1s: seed 1)
#   a2, a2s  attention drawn from a fixed bank of clean validation attention vectors of the row's predicted class:
#            benign bank = 20,000 val benign rows predicted benign that are neither training nor held-out negatives;
#            other predicted classes = the clean val source rows predicted as that class (benign bank if none)
#   b        12 consistency scores between attributions only: {occlusion, GradientSHAP} x {|gxi|, |IG20|} x
#            {Spearman, top-10 Jaccard, cosine}, occlusion with training-mean masking, GradientSHAP with the Stage 3
#            settings (100 training background rows of Stage 3, 50 samples), both for the predicted class
#   c0       D's 4 features with C = 1
#   c1       D's 4 features + 12 N(0, 1) noise features (fixed seed per row set)
#   c2       D's 4 features + 12 logit and baseline statistics (STATS below)
#   All of a, b, c use C = C_plus = 1.
#
# Usage:
#   .venv\Scripts\python.exe -m src.stage9_dplus_ablation gate0  [--years 2017 2018] [--max-hours 4]
#   .venv\Scripts\python.exe -m src.stage9_dplus_ablation stageA [--years 2017 2018] [--max-hours 4]
#   .venv\Scripts\python.exe -m src.stage9_dplus_ablation summary
import argparse
import copy
import json
import pickle
import sys
import time
import zlib

import numpy as np
import pandas as pd
import torch

from src.stage3_explain import STAGE3, gradient_shap, integrated_gradients, jaccard, load_ftt, occlusion, ranking, \
    rowwise_spearman
from src.stage4_constraints import check_validity
from src.stage6_defense import BENIGN, attention_and_logits, grad_x_input, norm_abs
from src.stage6b_defense import auroc
from src.stage6c_defense import FEATS_D, a1_on
from src.stage6d_final import LinearDefender, attack_loop, paired_boot, plus_features, terms_D
from src.stage6d_final import load_common as load_common17
from src.stage7e_defense_2018 import load_common7
from src.utils import ROOT

RES = ROOT / "results"
OUT = RES / "stage9" / "ablation"
DS = {
    "2017": {"cfg": RES / "runs" / "20261005-170553_stage6d_final" / "config.json", "key": "stage6d",
             "def": RES / "stage6" / "final", "a1": RES / "stage5", "model": lambda s: RES / "stage2" / f"ftt_seed{s}.pt",
             "train": RES / "stage1" / "train.parquet", "feats": RES / "stage1" / "features.json",
             "bl": RES / "stage3" / "baseline_rows.npz", "chunk": "attack_chunk",
             "sel": RES / "stage3" / "selection.parquet"},
    "2018": {"cfg": RES / "runs" / "20261006-144518_stage7e_defense_2018" / "config.json", "key": "stage7e",
             "def": RES / "stage7" / "stage7e", "a1": RES / "stage7" / "stage7d",
             "model": lambda s: RES / "stage7" / f"seed{s}" / "ftt.pt",
             "train": RES / "stage7" / "data" / "train.parquet", "feats": RES / "stage7" / "data" / "features.json",
             "bl": RES / "stage7" / "stage7c" / "raw" / "baseline_rows.npz", "chunk": "a6_chunk",
             "sel": RES / "stage7" / "stage7c" / "raw" / "selection.parquet"},
}
SEEDS = [0, 1, 2]
A6N = ["A6_lam1", "A6_lam100", "A6norm_w0.5"]
ATT = ["A1"] + A6N
VARIANTS = ["a1", "a1s", "a2", "a2s", "b", "c0", "c1", "c2"]
DETS = ["D", "Dplus"] + VARIANTS
STATS = ["entropy", "energy", "logit_l2", "logit_top1", "logit_top2", "margin_23", "logit_benign", "logit_max_attack",
         "logit_std", "p_benign", "mahalanobis_pred", "knn5_pred"]
TOL = {"score": 1e-5, "thr": 1e-5, "auroc": 1e-9}
BANK_N = 20000
COMMON = ["bot", "ddos", "dos_goldeneye", "dos_hulk", "dos_slowloris", "ssh_patator", "webattack"]
V2_DROP = ["dos_slowloris", "webattack"]
RNG_SEED = 4026


class StopRun(Exception):
    pass


class Log:
    def __init__(self, path):
        self.f = open(path, "a", encoding="utf-8")

    def __call__(self, msg):
        line = f"[{time.strftime('%Y-%m-%d %H:%M:%S')}] {msg}"
        self.f.write(line + "\n")
        self.f.flush()


def config(year):
    j = json.loads(DS[year]["cfg"].read_text(encoding="utf-8"))
    c = copy.deepcopy(j["config"][DS[year]["key"]])
    for k in ("gs_background", "gs_n_samples", "gs_stdev"):
        c.setdefault(k, STAGE3[k])
    return c


def load_model(year, s, dev):
    model, _ = load_ftt(DS[year]["model"](s), dev)
    for p in model.parameters():
        p.requires_grad_(False)
    return model


def load_defs(year, s, eps, dev):
    DEF = pickle.loads((DS[year]["def"] / f"seed{s}" / f"defenders_eps{eps:g}.pkl").read_bytes())
    for k in ("D", "Dp"):
        o = DEF[k]
        o.mu, o.sd, o.w = o.mu.to(dev), o.sd.to(dev), o.w.to(dev)
    return DEF


# ---------------------------------------------------------------- scores and vectors

def cons12(A, B, prefix="cons"):
    # The consistency loop of stage6b_defense.detector_scores (one call per score chunk, rng 0 per call, same order):
    # for a in A (dict of normalised importances), b in B: -Spearman, -(top-10 Jaccard + 1e-3 Spearman), -cosine.
    s = {}
    rng = np.random.default_rng(0)
    for an, Av in A.items():
        for bn, Bv in B.items():
            sp = rowwise_spearman(Av, Bv)
            s[f"{prefix}_{an}_{bn}_spearman"] = -sp
            s[f"{prefix}_{an}_{bn}_jacc10"] = -(jaccard(ranking(Av, rng), ranking(Bv, rng), 10) + 1e-3 * sp)
            s[f"{prefix}_{an}_{bn}_cos"] = -(Av * Bv).sum(1) / np.clip(np.linalg.norm(Av, axis=1) * np.linalg.norm(Bv, axis=1),
                                                                     1e-30, None)
    return s


def cons12_chunked(A, B, batch, prefix="cons"):
    n = len(next(iter(A.values())))
    parts = [pd.DataFrame(cons12({k: v[i:i + batch] for k, v in A.items()}, {k: v[i:i + batch] for k, v in B.items()},
                                 prefix)) for i in range(0, n, batch)]
    return pd.concat(parts, ignore_index=True)


def vectors(model, Z, base, c, attn_only=False):
    # stage6b_defense.detector_scores, chunk by chunk, plus the normalised vectors, logits and predictions.
    parts, V = [], {k: [] for k in ("ro", "me", "gxi", "ig", "lg", "pred")}
    for i in range(0, Z.shape[0], c["score_batch"]):
        zb = Z[i:i + c["score_batch"]]
        lg, ro, me = attention_and_logits(model, zb)
        pred = lg.argmax(1)
        V["ro"].append(norm_abs(ro).astype(np.float32)), V["me"].append(norm_abs(me).astype(np.float32))
        V["lg"].append(lg.float().cpu().numpy()), V["pred"].append(pred.cpu().numpy())
        if attn_only:
            continue
        p = torch.softmax(lg, -1)
        top2 = lg.topk(2, dim=1).values
        gxi = norm_abs(grad_x_input(model, zb, pred))
        ig, _, _ = integrated_gradients(model, zb, pred, torch.zeros(1, zb.shape[1], device=zb.device),
                                        {**c, "ig_steps": c["ig_steps_det"]})
        ig = norm_abs(ig)
        maha, knn = base.scores(zb, pred)
        ben = torch.zeros_like(pred)
        maha_b, knn_b = base.scores(zb, ben)
        s = {"msp": -p.max(1).values.cpu().numpy(), "margin": -(top2[:, 0] - top2[:, 1]).cpu().numpy(),
             "mahalanobis": maha, "knn5": knn, "mahalanobis_benign": maha_b, "knn5_benign": knn_b}
        s.update(cons12({"rollout": norm_abs(ro), "attnmean": norm_abs(me)}, {"gxi": gxi, "ig20": ig}))
        parts.append(pd.DataFrame(s))
        V["gxi"].append(gxi.astype(np.float32)), V["ig"].append(ig.astype(np.float32))
    out = {k: np.concatenate(v) for k, v in V.items() if v}
    if parts:
        out["sc"] = pd.concat(parts, ignore_index=True)
    return out


def gshap_fast(model, X, target, bg, n, gen, rows=64):
    # GradientSHAP with stdevs 0, the estimator of captum GradientShap (stage3_explain.gradient_shap): per row, n
    # samples of a uniformly drawn background row b and alpha ~ U(0, 1); attribution = mean of
    # grad_target(b + alpha (x - b)) * (x - b). Vectorised over rows x samples (captum with grad_batch 128 needs about
    # 7 GB and runs at about 45 ms per row on this 8 GB GPU; this version about 4 ms per row at 3.5 GB). Checked
    # against captum on 2017 seed 0 positives: per-row Spearman of |attr| 0.772 vs 0.784 between two captum seeds.
    out, F = [], X.shape[1]
    for i in range(0, X.shape[0], rows):
        xb, t = X[i:i + rows], target[i:i + rows]
        B = xb.shape[0]
        idx = torch.randint(0, bg.shape[0], (B, n), generator=gen, device=X.device)
        a = torch.rand(B, n, 1, generator=gen, device=X.device)
        base = bg[idx]
        xs = (base + a * (xb[:, None] - base)).reshape(-1, F).requires_grad_(True)
        lg = model(xs)
        gr, = torch.autograd.grad(lg.gather(1, t.repeat_interleave(n)[:, None]).sum(), xs)
        out.append((gr.view(B, n, F) * (xb[:, None] - base)).mean(1).detach())
    return torch.cat(out).cpu().numpy()


def vectors_b(model, Z, pred, c, gs_bg, gs_seed):
    # Occlusion (training-mean masking) and GradientSHAP (Stage 3 settings: 100 background rows, 50 samples, stdevs 0)
    # for the predicted class, normalised |.|.
    occ, gs = [], []
    pr = torch.from_numpy(pred).to(Z.device)
    gen = torch.Generator(device=Z.device).manual_seed(gs_seed)
    for i in range(0, Z.shape[0], c["score_batch"]):
        zb, pb = Z[i:i + c["score_batch"]], pr[i:i + c["score_batch"]]
        occ.append(norm_abs(occlusion(model, zb, pb, torch.zeros_like(zb), c)).astype(np.float32))
        gs.append(norm_abs(gshap_fast(model, zb, pb, gs_bg, c["gs_n_samples"], gen)).astype(np.float32))
    return {"occ": np.concatenate(occ), "gs": np.concatenate(gs), "gs_impl": np.array("gshap_fast64")}


def save_set(path, V):
    arr = {k: v for k, v in V.items() if k != "sc"}
    np.savez_compressed(path, **arr)
    if "sc" in V:
        V["sc"].to_parquet(path.with_suffix(".scores.parquet"), index=False)


def load_set(path):
    z = np.load(path)
    V = {k: z[k] for k in z.files}
    p = path.with_suffix(".scores.parquet")
    if p.exists():
        V["sc"] = pd.read_parquet(p)
    return V


def concat_sets(*Vs):
    keys = set.intersection(*[set(V) for V in Vs])
    out = {k: (pd.concat([V[k] for V in Vs], ignore_index=True) if k == "sc" else np.concatenate([V[k] for V in Vs]))
           for k in keys}
    return out


# ---------------------------------------------------------------- per-seed preparation shared by both phases

def negatives_idx(c, s, G, model):
    rng_s = np.random.default_rng(c["rng_seed"] + 8000 + s)
    with torch.no_grad():
        pv = torch.cat([model(G.Zben_va[i:i + 8192]).argmax(1) for i in range(0, len(G.Zben_va), 8192)])
        pt = torch.cat([model(G.Zben_te[i:i + 8192]).argmax(1) for i in range(0, len(G.Zben_te), 8192)])
    va_ok = torch.nonzero(pv == BENIGN).squeeze(1).cpu().numpy()
    perm = rng_s.permutation(va_ok)
    a, b = c["d_train_neg"], c["d_train_neg"] + c["n_heldout"]
    tr_idx, held_idx = perm[:a], perm[a:b]
    te_idx = rng_s.choice(torch.nonzero(pt == BENIGN).squeeze(1).cpu().numpy(), size=c["n_test_benign"], replace=False)
    bank_idx = perm[b:b + BANK_N]
    return tr_idx, held_idx, te_idx, bank_idx


def a1_test_X(year, s, eps, G):
    a1 = np.load(DS[year]["a1"] / f"seed{s}" / f"adv_raw_constrained_A1_eps{eps:g}.npz")
    i1 = np.array([np.flatnonzero(a1["pos"] == q)[0] for q in G.pos])
    return a1["x_adv"][i1]


def check_valid(X, R, G, eps, S, what):
    valid, viol, _ = check_validity(X, R, G.feats, G.spec, eps, S.cls_rows)
    if not valid.all():
        raise StopRun(f"invalid rows in {what}: {int((~valid).sum())} ({[k for k, v in viol.items() if v.any()]})")


# ---------------------------------------------------------------- Gate 0

def gate0(year, dev, log, t_end):
    c = config(year)
    G = load_common17(c, dev) if year == "2017" else load_common7(c, dev)
    rows = []
    for s in SEEDS:
        sdir = OUT / year / f"seed{s}"
        sdir.mkdir(parents=True, exist_ok=True)
        model = load_model(year, s, dev)
        tr_idx, held_idx, te_idx, bank_idx = negatives_idx(c, s, G, model)
        np.savez(sdir / "indices.npz", tr=tr_idx, held=held_idx, te=te_idx, bank=bank_idx)
        NEG = pickle.loads((DS[year]["def"] / f"seed{s}" / "negatives.pkl").read_bytes())
        t0 = time.time()
        for k, ix, Zs in (("val_train", tr_idx, G.Zben_va), ("val_held", held_idx, G.Zben_va), ("test", te_idx, G.Zben_te)):
            V = vectors(model, Zs[torch.from_numpy(ix).to(dev)], G.base, c)
            save_set(sdir / f"neg_{k}.npz", V)
            saved = NEG[k]
            cols = list(saved.columns)
            assert list(V["sc"].columns) == cols, (list(V["sc"].columns), cols)
            d = np.abs(V["sc"][cols].to_numpy(np.float64) - saved[cols].to_numpy(np.float64)).max(0)
            cons_cols = [i for i, cn in enumerate(cols) if cn.startswith("cons_")]
            rows.append({"year": year, "seed": s, "eps": np.nan, "check": f"negatives {k}: 18 scores max |diff|",
                         "value": float(d.max()), "cons12_max_diff": float(d[cons_cols].max()), "n": len(saved),
                         "pass": bool(d.max() <= TOL["score"])})
            log(f"{year} seed {s} {k}: max |diff| all {d.max():.2e}, cons12 {d[cons_cols].max():.2e}")
        Vb = vectors(model, G.Zben_va[torch.from_numpy(bank_idx).to(dev)], G.base, c, attn_only=True)
        save_set(sdir / "bank_benign.npz", Vb)
        Va = vectors(model, G.Sva.z0, G.base, c, attn_only=True)
        save_set(sdir / "bank_valsrc.npz", Va)
        log(f"{year} seed {s}: negatives and banks in {time.time() - t0:.0f} s (bank {len(bank_idx)} rows)")
        for eps in c["eps"]:
            t0 = time.time()
            DEF = load_defs(year, s, eps, dev)
            Xv, zv = a1_on(model, G.Sva, G.tp, eps, c, dev, c["rng_seed"] + 7100 + int(eps * 100) + 10 * s)
            check_valid(Xv, G.Rva, G, eps, G.Sva, f"{year} seed {s} eps {eps} val A1")
            with torch.no_grad():
                flip = (model(zv).argmax(1) == BENIGN).cpu().numpy()
            Vp = vectors(model, zv[torch.from_numpy(np.flatnonzero(flip)).to(dev)], G.base, c)
            Vp["ytrue"] = G.yva[flip]
            save_set(sdir / f"pos_eps{eps:g}.npz", Vp)
            np.savez_compressed(sdir / f"pos_eps{eps:g}_X.npz", X_all=Xv, flip=flip, y=G.yva)
            rows.append({"year": year, "seed": s, "eps": eps, "check": "n_val_pos regenerated vs saved",
                         "value": int(flip.sum()), "saved": DEF["n_val_pos"], "pass": int(flip.sum()) == DEF["n_val_pos"]})
            Ftr, cols = plus_features(pd.read_parquet(sdir / "neg_val_train.scores.parquet"))
            Fho, _ = plus_features(pd.read_parquet(sdir / "neg_val_held.scores.parquet"))
            Fpv, _ = plus_features(Vp["sc"])
            D = LinearDefender(FEATS_D, Ftr[FEATS_D].to_numpy(), Fpv[FEATS_D].to_numpy(), Fho[FEATS_D].to_numpy(),
                               c["fpr"], 1e6, dev)
            Dp = LinearDefender(cols, Ftr[cols].to_numpy(), Fpv[cols].to_numpy(), Fho[cols].to_numpy(), c["fpr"],
                                c["C_plus"], dev)
            for nm, obj, sv, au, sau in (("D", D, DEF["D"], auroc(D.decision(Fho[FEATS_D]), D.decision(Fpv[FEATS_D])),
                                          DEF["val_auroc_D"]),
                                         ("Dplus", Dp, DEF["Dp"], auroc(Dp.decision(Fho[cols]), Dp.decision(Fpv[cols])),
                                          DEF["val_auroc_Dp"])):
                rows.append({"year": year, "seed": s, "eps": eps, "check": f"{nm} refit threshold |diff|",
                             "value": abs(obj.thr - sv.thr), "saved": sv.thr, "pass": abs(obj.thr - sv.thr) <= TOL["thr"]})
                rows.append({"year": year, "seed": s, "eps": eps, "check": f"{nm} refit val AUROC |diff|",
                             "value": abs(au - sau), "saved": sau, "pass": abs(au - sau) <= TOL["auroc"]})
            X1 = a1_test_X(year, s, eps, G)
            with torch.no_grad():
                z1 = G.tp(torch.from_numpy(X1).to(dev)).float()
            V1 = vectors(model, z1, G.base, c)
            V1["ytrue"] = G.yte
            save_set(sdir / f"att_eps{eps:g}_A1.npz", V1)
            F1, _ = plus_features(V1["sc"])
            saved = pd.read_parquet(DS[year]["def"] / f"seed{s}" / f"rows_eps{eps:g}_A1.parquet").set_index("pos") \
                .loc[G.pos]
            d = float(np.abs(Dp.decision(F1[cols]) - saved.score_Dplus.to_numpy()).max())
            rows.append({"year": year, "seed": s, "eps": eps, "check": "A1 test score_Dplus (refitted D_plus) max |diff|",
                         "value": d, "pass": d <= TOL["score"]})
            fl = (V1["pred"] == BENIGN)
            rows.append({"year": year, "seed": s, "eps": eps, "check": "A1 test flipped equal to saved",
                         "value": float((fl == saved.flipped.to_numpy()).mean()), "pass": bool((fl == saved.flipped.to_numpy()).all())})
            log(f"{year} seed {s} eps {eps}: positives {int(flip.sum())} (saved {DEF['n_val_pos']}), thr D "
                f"{abs(D.thr - DEF['D'].thr):.1e}, D+ {abs(Dp.thr - DEF['Dp'].thr):.1e}, A1 score_Dplus {d:.1e} "
                f"[{time.time() - t0:.0f} s]")
            if time.time() > t_end:
                raise StopRun("runtime cap reached")
        del model
        torch.cuda.empty_cache()
    return pd.DataFrame(rows)


# ---------------------------------------------------------------- Stage A+

def gs_background(year, dev):
    feats = json.loads(DS[year]["feats"].read_text(encoding="utf-8"))
    from src.preprocess import Preprocessor
    pp = Preprocessor.load(RES / "stage2" / "preprocess.json" if year == "2017" else RES / "stage7" / "preprocess.json")
    rows = np.load(DS[year]["bl"])["gshap_background_train_rows"]
    train = pd.read_parquet(DS[year]["train"], columns=feats)
    return torch.from_numpy(pp.transform(train.iloc[rows])).to(dev)


def stats12(V, base_sc):
    lg = V["lg"].astype(np.float64)
    p = np.exp(lg - lg.max(1, keepdims=True))
    p /= p.sum(1, keepdims=True)
    srt = np.sort(lg, 1)[:, ::-1]
    lse = lg.max(1) + np.log(np.exp(lg - lg.max(1, keepdims=True)).sum(1))
    return np.c_[-(p * np.log(np.clip(p, 1e-300, None))).sum(1), -lse, np.linalg.norm(lg, axis=1), srt[:, 0], srt[:, 1],
                 srt[:, 1] - srt[:, 2], lg[:, BENIGN], lg[:, 1:].max(1), lg.std(1), p[:, BENIGN],
                 base_sc["mahalanobis"].to_numpy(), base_sc["knn5"].to_numpy()]


def derangement(groups, rng):
    # Index map: each row takes the vector of another row of its group (cyclic shift of a random permutation).
    idx = np.arange(len(groups))
    for g in np.unique(groups):
        m = np.flatnonzero(groups == g)
        if len(m) > 1:
            p = rng.permutation(m)
            idx[p] = np.roll(p, -1)
    return idx


def variant_features(V, set_name, year, s, eps, banks, batch):
    # Feature matrix (n, 16) for every variant on one row set (V must contain sc, ro, me, gxi, ig, occ, gs, lg, pred).
    F, cols = plus_features(V["sc"])
    base4 = F[FEATS_D].to_numpy()
    out = {"D": base4, "Dplus": F[cols].to_numpy(), "c0": base4}
    groups = V["ytrue"] if "ytrue" in V else np.zeros(len(base4), dtype=np.int64)
    B = {"gxi": V["gxi"], "ig20": V["ig"]}
    for nm, sd in (("a1", 0), ("a1s", 1)):
        idx = derangement(groups, np.random.default_rng(zlib.crc32(f"a1|{year}|{s}|{eps}|{set_name}|{sd}".encode())))
        cons = cons12_chunked({"rollout": V["ro"][idx], "attnmean": V["me"][idx]}, B, batch)
        out[nm] = np.c_[base4, -cons.to_numpy()]
    for nm, sd in (("a2", 0), ("a2s", 1)):
        rng = np.random.default_rng(zlib.crc32(f"a2|{year}|{s}|{eps}|{set_name}|{sd}".encode()))
        ro, me = np.empty_like(V["ro"]), np.empty_like(V["me"])
        for k in np.unique(V["pred"]):
            m = np.flatnonzero(V["pred"] == k)
            bro, bme = banks.get(int(k), banks[BENIGN])
            j = rng.integers(0, len(bro), len(m))
            ro[m], me[m] = bro[j], bme[j]
        cons = cons12_chunked({"rollout": ro, "attnmean": me}, B, batch)
        out[nm] = np.c_[base4, -cons.to_numpy()]
    cons = cons12_chunked({"occl": V["occ"], "gshap": V["gs"]}, B, batch)
    out["b"] = np.c_[base4, -cons.to_numpy()]
    rng = np.random.default_rng(zlib.crc32(f"c1|{year}|{s}|{eps}|{set_name}".encode()))
    out["c1"] = np.c_[base4, rng.standard_normal((len(base4), 12))]
    out["c2"] = np.c_[base4, stats12(V, V["sc"])]
    return out


def stage_a(year, dev, log, t_end):
    c = config(year)
    G = load_common17(c, dev) if year == "2017" else load_common7(c, dev)
    gs_bg = gs_background(year, dev)
    agree, defrows = [], []
    for s in SEEDS:
        sdir = OUT / year / f"seed{s}"
        model = load_model(year, s, dev)
        t0 = time.time()
        # b vectors for every stored set (computed once, appended to the set file)
        sets = ["neg_val_train", "neg_val_held", "neg_test"] + [f"pos_eps{e:g}" for e in c["eps"]] + \
               [f"att_eps{e:g}_A1" for e in c["eps"]]
        ix = np.load(sdir / "indices.npz")
        for k in sets:
            p = sdir / f"{k}.npz"
            V = load_set(p)
            if "occ" in V and "gs_impl" in V:
                continue
            if k.startswith("neg_"):
                Zs = G.Zben_te if k == "neg_test" else G.Zben_va
                idx = {"neg_val_train": ix["tr"], "neg_val_held": ix["held"], "neg_test": ix["te"]}[k]
                Z = Zs[torch.from_numpy(idx).to(dev)]
            elif k.startswith("pos_"):
                e = float(k.split("eps")[1])
                Xp = np.load(sdir / f"{k}_X.npz")
                with torch.no_grad():
                    Z = G.tp(torch.from_numpy(Xp["X_all"][Xp["flip"]]).to(dev)).float()
            else:
                e = float(k.split("eps")[1].split("_")[0])
                with torch.no_grad():
                    Z = G.tp(torch.from_numpy(a1_test_X(year, s, e, G)).to(dev)).float()
            tb = time.time()
            V.update(vectors_b(model, Z, V["pred"], c, gs_bg, c["rng_seed"] + 9900 + 100 * s))
            save_set(p, V)
            log(f"{year} seed {s} {k}: b vectors for {len(V['pred'])} rows in {time.time() - tb:.0f} s")
            if time.time() > t_end:
                raise StopRun("runtime cap reached")
        log(f"{year} seed {s}: occlusion and GradientSHAP vectors in {time.time() - t0:.0f} s")
        # A6 re-run with the saved D (original seeds), X saved, vectors (incl. b)
        for eps in c["eps"]:
            DEF = load_defs(year, s, eps, dev)
            D = DEF["D"]
            specs = [(f"A6_lam{lam:g}", lam, None) for lam in c["lambdas"]] + [(f"A6norm_w{c['norm_w']:g}", 0.0, c["norm_w"])]
            for name, lam, nw in specs:
                p = sdir / f"att_eps{eps:g}_{name}.npz"
                if p.exists():
                    continue
                t0 = time.time()
                X, z = attack_loop(model, G.Ste, G.tp, eps, dict(c, attack_chunk=c[DS[year]["chunk"]]), lam, nw,
                                   terms_D(model, G.tp, G.base, D), D.thr, dev,
                                   c["rng_seed"] + 7300 + int(eps * 100) + 17 * s + sum(map(ord, name)))
                check_valid(X, G.Rte, G, eps, G.Ste, f"{year} seed {s} eps {eps} {name}")
                ta = time.time() - t0
                np.savez_compressed(sdir / f"att_eps{eps:g}_{name}_X.npz", X=X, pos=G.pos)
                V = vectors(model, z, G.base, c)
                V["ytrue"] = G.yte
                V.update(vectors_b(model, z, V["pred"], c, gs_bg, c["rng_seed"] + 9900 + 100 * s))
                save_set(p, V)
                log(f"{year} seed {s} eps {eps} {name}: attack {ta:.0f} s, vectors {time.time() - t0 - ta:.0f} s")
                if time.time() > t_end:
                    raise StopRun("runtime cap reached")
        # banks for a2
        bb, bv = load_set(sdir / "bank_benign.npz"), load_set(sdir / "bank_valsrc.npz")
        banks = {BENIGN: (bb["ro"], bb["me"])}
        for k in np.unique(bv["pred"]):
            if int(k) != BENIGN:
                m = bv["pred"] == k
                banks[int(k)] = (bv["ro"][m], bv["me"][m])
        # fit and score
        t0 = time.time()
        Ntr, Nho, Nte = (load_set(sdir / f"neg_{k}.npz") for k in ("val_train", "val_held", "test"))
        for eps in c["eps"]:
            DEF = load_defs(year, s, eps, dev)
            Vp = load_set(sdir / f"pos_eps{eps:g}.npz")
            FT = {k: variant_features(V, k, year, s, eps, banks, c["score_batch"])
                  for k, V in (("neg_val_train", Ntr), ("neg_val_held", Nho), ("neg_test", Nte), ("pos", Vp))}
            dets = {"D": DEF["D"], "Dplus": DEF["Dp"]}
            for v in VARIANTS:
                cols = [f"f{i}" for i in range(FT["pos"][v].shape[1])]
                dets[v] = LinearDefender(cols, FT["neg_val_train"][v], FT["pos"][v], FT["neg_val_held"][v], c["fpr"],
                                         c["C_plus"], dev)
            for v, obj in dets.items():
                defrows.append({"year": year, "seed": s, "eps": eps, "detector": v, "n_val_pos": len(Vp["pred"]),
                                "n_features": FT["pos"][v].shape[1], "C": 1e6 if v == "D" else c["C_plus"],
                                "thr": obj.thr, "val_auroc": auroc(obj.decision(FT["neg_val_held"][v]),
                                                                   obj.decision(FT["pos"][v])),
                                "test_fpr": float((obj.decision(FT["neg_test"][v]) >= obj.thr).mean())})
            neg = pd.DataFrame({"seed": s, "eps": eps, **{f"score_{v}": o.decision(FT["neg_test"][v]) for v, o in dets.items()},
                                **{f"thr_{v}": o.thr for v, o in dets.items()}})
            neg.to_parquet(sdir / f"scores_neg_test_eps{eps:g}.parquet", index=False)
            for name in ATT:
                V = load_set(sdir / f"att_eps{eps:g}_{name}.npz")
                Fa = variant_features(V, name, year, s, eps, banks, c["score_batch"])
                df = pd.DataFrame({"seed": s, "eps": eps, "attack": name, "pos": G.pos,
                                   "class": [G.classes[k] for k in G.yte], "flipped": V["pred"] == BENIGN,
                                   **{f"score_{v}": o.decision(Fa[v]) for v, o in dets.items()},
                                   **{f"thr_{v}": o.thr for v, o in dets.items()}})
                df.to_parquet(sdir / f"scores_eps{eps:g}_{name}.parquet", index=False)
                if name != "A1":
                    sv = pd.read_parquet(DS[year]["def"] / f"seed{s}" / f"rows_eps{eps:g}_{name}.parquet") \
                        .set_index("pos").loc[G.pos]
                    agree.append({"year": year, "seed": s, "eps": eps, "attack": name,
                                  "flipped_agree": float((df.flipped.to_numpy() == sv.flipped.to_numpy()).mean()),
                                  "success_rerun": float(df.flipped.mean()), "success_saved": float(sv.flipped.mean()),
                                  "max_abs_diff_score_D": float(np.abs(df.score_D.to_numpy() - sv.score_D.to_numpy()).max()),
                                  "max_abs_diff_score_Dplus": float(np.abs(df.score_Dplus.to_numpy() - sv.score_Dplus.to_numpy()).max()),
                                  "rows_identical_D": float((np.abs(df.score_D.to_numpy() - sv.score_D.to_numpy()) <= TOL["score"]).mean())})
        log(f"{year} seed {s}: variants fitted and scored in {time.time() - t0:.0f} s")
        del model
        torch.cuda.empty_cache()
    return pd.DataFrame(agree), pd.DataFrame(defrows)


# ---------------------------------------------------------------- summary (CPU)

def keep_pos(year, variant):
    sel = pd.read_parquet(DS[year]["sel"])
    nd = pd.read_parquet(RES / "stage8" / f"neardup_rows_{year}.parquet").set_index("test_row")["nn_dist_exact"]
    dist = nd.reindex(sel["test_row"].to_numpy()).to_numpy()
    k = sel["class"].isin(COMMON).to_numpy().copy()
    if variant == "V1":
        k &= ~(dist < 1e-3)
    elif variant == "V2":
        k &= ~sel["class"].isin(V2_DROP).to_numpy()
    return set(np.flatnonzero(k))


def summary():
    out_rows, pair_rows = [], []
    for year in DS:
        R = pd.concat([pd.read_parquet(f) for f in sorted((OUT / year).glob("seed*/scores_eps*_*.parquet"))],
                      ignore_index=True)
        N = pd.concat([pd.read_parquet(f) for f in sorted((OUT / year).glob("seed*/scores_neg_test_eps*.parquet"))],
                      ignore_index=True)
        for rv in ("V0", "V1", "V2"):
            kp = keep_pos(year, rv)
            Rv = R[R.pos.isin(kp)]
            for eps in sorted(Rv.eps.unique()):
                ns = N[N.eps == eps]
                for det in DETS:
                    fpr = [(g[f"score_{det}"] >= g[f"thr_{det}"]).mean() for _, g in ns.groupby("seed")]
                    rec = {"year": year, "rowset": rv, "eps": eps, "detector": det, "test_fpr": float(np.mean(fpr)),
                           "test_fpr_lo": min(fpr), "test_fpr_hi": max(fpr)}
                    for a in ATT:
                        r = Rv[(Rv.eps == eps) & (Rv.attack == a)]
                        f = r.flipped.to_numpy()
                        hit = (r[f"score_{det}"] >= r[f"thr_{det}"]).to_numpy()
                        per = [((g[f"score_{det}"] >= g[f"thr_{det}"]).to_numpy()[g.flipped.to_numpy()]).mean()
                               for _, g in r.groupby("seed") if g.flipped.any()]
                        rec[f"tpr_{a}"] = float(hit[f].mean()) if f.any() else np.nan
                        rec[f"tpr_{a}_lo"], rec[f"tpr_{a}_hi"] = (min(per), max(per)) if per else (np.nan, np.nan)
                        rec[f"n_pos_{a}"] = int(f.sum())
                    out_rows.append(rec)
                for det in VARIANTS + ["Dplus"]:
                    for a in ATT:
                        r = Rv[(Rv.eps == eps) & (Rv.attack == a)]
                        f = r.flipped.to_numpy()
                        cen = lambda d, x: (x[f"score_{d}"] - x[f"thr_{d}"]).to_numpy()
                        for lab, lo_d, hi_d in (("variant_minus_D", "D", det), ("Dplus_minus_variant", det, "Dplus")):
                            if lab == "Dplus_minus_variant" and det == "Dplus":
                                continue
                            rng = np.random.default_rng(zlib.crc32(f"{year}|{rv}|{eps}|{det}|{a}|{lab}".encode()))
                            pb = paired_boot(cen(lo_d, ns), cen(lo_d, r)[f], 0.0, cen(hi_d, ns), cen(hi_d, r)[f], 0.0,
                                             1000, rng)
                            per = []
                            for sd, g in r.groupby("seed"):
                                fg = g.flipped.to_numpy()
                                if fg.any():
                                    per.append((cen(hi_d, g)[fg] >= 0).mean() - (cen(lo_d, g)[fg] >= 0).mean())
                            pair_rows.append({"year": year, "rowset": rv, "eps": eps, "detector": det, "attack": a,
                                              "comparison": lab, "n_pos": int(f.sum()), **pb,
                                              "seed_lo": min(per) if per else np.nan,
                                              "seed_hi": max(per) if per else np.nan})
    T = pd.DataFrame(out_rows)
    P = pd.DataFrame(pair_rows)
    T.to_csv(OUT / "summary_detection.csv", index=False)
    P.to_csv(OUT / "summary_paired.csv", index=False)
    defs = pd.concat([pd.read_csv(f) for f in sorted(OUT.glob("*/stageA_defenders.csv"))], ignore_index=True)
    g = defs.groupby(["year", "eps", "detector"]).agg(val_auroc=("val_auroc", "mean"), val_auroc_lo=("val_auroc", "min"),
                                                       val_auroc_hi=("val_auroc", "max"), n_val_pos=("n_val_pos", "sum"))
    g.reset_index().to_csv(OUT / "summary_defenders.csv", index=False)
    # Gate A -> B: A6_lam1, eps 1, V1; interval of Dplus - variant or variant - D excludes 0 on at least one dataset.
    x = P[(P.rowset == "V1") & (P.eps == 1.0) & (P.attack == "A6_lam1")]
    gate = []
    for det in VARIANTS:
        rec = {"variant": det}
        hit = False
        for year in DS:
            for lab in ("variant_minus_D", "Dplus_minus_variant"):
                y = x[(x.year == year) & (x.detector == det) & (x.comparison == lab)].iloc[0]
                ex = (y.tpr_diff_lo > 0) or (y.tpr_diff_hi < 0)
                rec[f"{year}_{lab}"] = f"{y.tpr_diff:.3f} [{y.tpr_diff_lo:.3f}, {y.tpr_diff_hi:.3f}]"
                hit |= ex
        rec["gate_pass"] = hit
        gate.append(rec)
    pd.DataFrame(gate).to_csv(OUT / "gate_A_to_B.csv", index=False)
    print("summary written")


# ---------------------------------------------------------------- main

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("phase", choices=["gate0", "stageA", "summary"])
    ap.add_argument("--years", nargs="+", default=list(DS))
    ap.add_argument("--max-hours", type=float, default=4.0)
    a = ap.parse_args()
    OUT.mkdir(parents=True, exist_ok=True)
    if a.phase == "summary":
        summary()
        return
    log = Log(OUT / f"log_{a.phase}.txt")
    t_start = time.time()
    t_end = t_start + a.max_hours * 3600
    dev = torch.device("cuda")
    log(f"start {a.phase} {a.years} cap {a.max_hours} h")
    try:
        for year in a.years:
            ty = time.time()
            if a.phase == "gate0":
                if (OUT / year / "gate0.csv").exists():
                    log(f"{year}: gate0.csv exists, skipped")
                    continue
                df = gate0(year, dev, log, t_end)
                df.to_csv(OUT / year / "gate0.csv", index=False)
                log(f"{year}: gate 0 {'PASS' if df['pass'].all() else 'FAIL'} ({int((~df['pass']).sum())} failed checks) "
                    f"[{time.time() - ty:.0f} s]")
            else:
                g = pd.concat([pd.read_csv(f) for f in sorted(OUT.glob("*/gate0.csv"))], ignore_index=True)
                if not (len(g) and g["pass"].all() and set(g.year.astype(str)) >= set(a.years)):
                    raise StopRun("Gate 0 has not passed for every dataset; Stage A not started")
                ag, dfs = stage_a(year, dev, log, t_end)
                ag.to_csv(OUT / year / "a6_rerun_agreement.csv", index=False)
                dfs.to_csv(OUT / year / "stageA_defenders.csv", index=False)
                log(f"{year}: stage A done [{time.time() - ty:.0f} s]")
    except StopRun as e:
        log(f"STOPPED: {e}")
        sys.exit(2)
    log(f"finished {a.phase} in {time.time() - t_start:.0f} s")


if __name__ == "__main__":
    main()
