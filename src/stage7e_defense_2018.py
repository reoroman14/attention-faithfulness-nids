# Stage 7e: learned defenders D and D_plus against adaptive attackers on LycoS-Unicas-IDS2018 (Stage 6 FINAL and
# Stage 6e equivalent). Replicates src/stage6d_final.py and src/stage6e_strong.py with no methodological change and
# no tuning; only the data, checkpoints, class list and source rows change. A7 is skipped on 2018 (as instructed).
#
# Imported unchanged: LinearDefender, plus_features, attack_loop, terms_D, single_terms, paired_boot, summarise
# (stage6d_final); terms_A8, standin_features, union_eval, paired_union (stage6e_strong); detector_scores, auroc,
# boot (stage6b_defense); a1_on, feats_np, FEATS_D (stage6c_defense); Baselines, make_rows (stage6_defense); the
# projection and independent checker (stage4_constraints). Copied with 2018 paths: load_common, run_seed and the
# stand-in quality check.
#
# Defenders per seed and eps, trained ONLY on validation:
#   D       standardised class-balanced logistic regression on [msp, margin, mahalanobis_benign, knn5_benign]
#   D_plus  D's features plus the 12 consistency scores (validated direction), L2 (C = 1)
#   negatives  20,000 val benign rows predicted benign (training) + 50,000 others (held out, threshold at 5% FPR)
#   positives  constrained A1 rows on val attack-class source rows (Stage 7c VAL explanation set, up to 100 per class,
#              same sampling stream as 2017) that are valid and flipped
# Attackers on the Stage 7d source rows (150 per attack class; webattack all 85; dos_slowloris only rows with exact
# nearest-train distance >= 1e-3), same constrained projection and checker, 100 steps, 1 restart, kappa 10:
#   A1   the Stage 7d A1 rows (results/stage7/stage7d/seed{s}/adv_raw_constrained_A1_eps*.npz)
#   A6   against D: lambda 1, lambda 100, normalised w 0.5 (attack chunk 500, as 2017)
#   A8   against D_plus with differentiable stand-ins for all 12 consistency features (soft Spearman, soft top-10
#        Jaccard, exact cosine, IG with 8 steps in the loss): lambda 1, lambda 100, normalised w 0.5 (chunk 100)
#   Ablations (seed 0 only): A6_msp_only, A6_knn5_only (lambda 10)
# Budget floors: the Stage 7d 2018 floors (results/stage7/stage7d/stage7d_constraints_2018_floors.csv), so the
# source rows and the A1 rows are exactly those of Stage 7d.
# Test negatives: 1,000 genuine benign test rows predicted benign by that seed.
# Stand-in quality (R^2 and Spearman of each stand-in vs the true feature) on each seed's VALIDATION rows is computed
# and printed before any attack runs; a median R^2 below 0.5 is reported, not a stop.
# Stop conditions: any invalid constrained row (val A1, stand-in val A1, every test attack); D val AUROC < 0.9 for any
# seed and eps; runtime estimate > 5 h (from A6_lam1 and A8_lam1 at eps 1.0 on seed 0, printed before the rest).
# Resumable per seed and per (eps, attack). PROGRESS.md is not written by the script (entries go to report files).
#
# Usage:
#   .venv\Scripts\python.exe -m src.stage7e_defense_2018
#   smoke test: --set stage7e.smoke=true
import copy
import json
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
from src.preprocess import Preprocessor
from src.stage3_explain import load_ftt
from src.stage4_attack import STAGE4
from src.stage4_constraints import TorchPreprocessor, check_validity, load_spec_csv
from src.stage5_fullrun import keep_awake, md
from src.stage6_defense import BENIGN, STAGE6, Baselines, make_rows
from src.stage6b_defense import STAGE6B, auroc, detector_scores
from src.stage6c_defense import FEATS_D, STAGE6C, a1_on, feats_np
from src.stage6d_final import (STAGE6D, LinearDefender, attack_loop, paired_boot, plus_features, single_terms,
                               summarise, terms_D)
from src.stage6e_strong import STAGE6E, paired_union, standin_features, terms_A8, union_eval
from src.utils import ROOT, finish_run, resolve, start_run

STAGE7E = {
    "smoke": False,
    "out_dir": "results/stage7/stage7e",
    "a6_chunk": 500,
    "a8_chunk": 100,
    "max_hours": 5.0,
    "probe_eps": 1.0,
}
D7 = ROOT / "results" / "stage7"
C7 = D7 / "stage7c" / "raw"
S7D = D7 / "stage7d"
COMMON = ["bot", "ddos", "dos_goldeneye", "dos_hulk", "dos_slowloris", "ssh_patator", "webattack"]
A6N = ["A6_lam1", "A6_lam100", "A6norm_w0.5"]
A8N = ["A8_lam1", "A8_lam100", "A8norm_w0.5"]
EQ_D, EQ_P = ["A1"] + A6N, ["A1"] + A8N


class StopRun(Exception):
    pass


# ---------------------------------------------------------------- data (copy of stage6d_final.load_common, 2018)

def load_common7(c, dev):
    G = type("G", (), {})()
    G.feats = json.loads((D7 / "data" / "features.json").read_text(encoding="utf-8"))
    lm = json.loads((D7 / "data" / "label_map.json").read_text(encoding="utf-8"))
    G.classes = [k for k, _ in sorted(lm.items(), key=lambda kv: kv[1])]
    G.pp = Preprocessor.load(D7 / "preprocess.json")
    G.tp = TorchPreprocessor(G.pp, dev)
    G.spec = load_spec_csv(S7D / "stage7d_constraints_2018_floors.csv")
    G.floors = G.spec.set_index("feature")["zero_floor_scale"].astype(float).to_dict()
    rng = np.random.default_rng(c["rng_seed"] + 6000)                  # same stream as 2017
    sel = pd.read_parquet(C7 / "selection.parquet")
    G.pos = np.load(S7D / "stage7d_source_positions.npy")
    if c["smoke"]:
        G.pos = np.concatenate([G.pos[sel["label_id"].to_numpy()[G.pos] == k][:c["n_per_class"]]
                                for k in range(1, len(G.classes))])
    test = pd.read_parquet(D7 / "data" / "test.parquet", columns=G.feats + ["label_id"])
    G.Rte = test.iloc[sel["test_row"].to_numpy()[G.pos]][G.feats].to_numpy(np.float64).copy()
    G.yte = sel["label_id"].to_numpy()[G.pos].astype(np.int64)
    G.Zben_te = torch.from_numpy(G.pp.transform(test[test["label_id"] == BENIGN][G.feats])).to(dev)
    del test
    vs = pd.read_parquet(C7 / "val_selection.parquet")
    vs = vs[vs.label_id != BENIGN]
    vpick = np.concatenate([rng.choice(g.index.to_numpy(), size=min(c["val_per_class"], len(g)), replace=False)
                            for _, g in vs.groupby("label_id")])
    val = pd.read_parquet(D7 / "data" / "val.parquet", columns=G.feats + ["label_id"])
    G.Rva = val.iloc[vs.loc[vpick, "val_row"].to_numpy()][G.feats].to_numpy(np.float64).copy()
    G.yva = vs.loc[vpick, "label_id"].to_numpy().astype(np.int64)
    G.Zben_va = torch.from_numpy(G.pp.transform(val[val["label_id"] == BENIGN][G.feats])).to(dev)
    del val
    train = pd.read_parquet(D7 / "data" / "train.parquet", columns=G.feats + ["label_id"])
    Ztr, ytr = G.pp.transform(train), train["label_id"].to_numpy()
    del train
    G.base = Baselines(Ztr, ytr, len(G.classes), c, dev, rng)
    del Ztr
    G.Ste = make_rows(G.Rte, G.yte, G.classes, G.feats, G.spec, G.floors, G.tp, dev)
    G.Sva = make_rows(G.Rva, G.yva, G.classes, G.feats, G.spec, G.floors, G.tp, dev)
    return G


def load_model(s, dev):
    model, _ = load_ftt(D7 / f"seed{s}" / "ftt.pt", dev)
    for p in model.parameters():
        p.requires_grad_(False)
    return model


def check_valid_or_stop(X, R, G, eps, S, what, log):
    valid, viol, _ = check_validity(X, R, G.feats, G.spec, eps, S.cls_rows)
    if not valid.all():
        bad = [k for k, v in viol.items() if v.any()]
        log.error(f"STOP: {int((~valid).sum())} invalid constrained rows in {what} ({bad})")
        raise StopRun(f"invalid rows in {what}: {int((~valid).sum())} rows, checks {bad}")
    return valid


# ---------------------------------------------------------------- stand-in quality (copy of the stage6e check)

def standin_quality(model, G, c, dev, s, log):
    rng = np.random.default_rng(c["rng_seed"] + 9100 + s)               # seed 0: the 2017 stream
    with torch.no_grad():
        pv = torch.cat([model(G.Zben_va[i:i + 8192]).argmax(1) for i in range(0, len(G.Zben_va), 8192)])
    ok = torch.nonzero(pv == BENIGN).squeeze(1).cpu().numpy()
    vb = rng.choice(ok, min(c["n_val_benign_check"], len(ok)), replace=False)
    eps = 1.0
    Xv, zv = a1_on(model, G.Sva, G.tp, eps, c, dev, c["rng_seed"] + 9200 + s)
    check_valid_or_stop(Xv, G.Rva, G, eps, G.Sva, f"seed {s} stand-in val A1", log)
    with torch.no_grad():
        fl = model(zv).argmax(1) == BENIGN
    Z = torch.cat([G.Zben_va[torch.from_numpy(vb).to(dev)], zv[fl]])
    true, _ = detector_scores(model, Z, G.base, c)
    Ftrue, cols = plus_features(true)
    st = []
    for i in range(0, len(Z), 100):
        zb = Z[i:i + 100].clone().requires_grad_(True)
        with torch.no_grad():
            cls = model(zb).argmax(1)
        _, F = standin_features(model, zb, cls, G.base, c)
        st.append(F.detach().cpu().numpy())
    S = np.concatenate(st)
    rows = []
    for j, col in enumerate(cols[len(FEATS_D):]):
        x, y = S[:, len(FEATS_D) + j], Ftrue[col].to_numpy()
        rows.append({"seed": s, "feature": col.replace("cons_", ""),
                     "r2": LinearRegression().fit(x[:, None], y).score(x[:, None], y),
                     "spearman": spearmanr(x, y).statistic, "n_rows": len(Z),
                     "n_val_benign": len(vb), "n_val_a1_flipped": int(fl.sum())})
    return pd.DataFrame(rows)


# ---------------------------------------------------------------- per seed (copy of stage6d_final.run_seed, 2018)

def prepare_seed(c, s, G, model, dev, log, sdir):
    # Negatives (val train / val held-out / test) and the two defenders per eps.
    rng_s = np.random.default_rng(c["rng_seed"] + 8000 + s)
    with torch.no_grad():
        pv = torch.cat([model(G.Zben_va[i:i + 8192]).argmax(1) for i in range(0, len(G.Zben_va), 8192)])
        pt = torch.cat([model(G.Zben_te[i:i + 8192]).argmax(1) for i in range(0, len(G.Zben_te), 8192)])
    va_ok = torch.nonzero(pv == BENIGN).squeeze(1).cpu().numpy()
    perm = rng_s.permutation(va_ok)
    tr_idx, held_idx = perm[:c["d_train_neg"]], perm[c["d_train_neg"]:c["d_train_neg"] + c["n_heldout"]]
    te_idx = rng_s.choice(torch.nonzero(pt == BENIGN).squeeze(1).cpu().numpy(), size=c["n_test_benign"], replace=False)
    times = {}
    fneg = sdir / "negatives.pkl"
    if fneg.exists():
        NEG = pickle.loads(fneg.read_bytes())
    else:
        t0 = time.time()
        NEG = {k: detector_scores(model, G.Zben_va[torch.from_numpy(ix).to(dev)] if k != "test" else
                                  G.Zben_te[torch.from_numpy(ix).to(dev)], G.base, c)[0]
               for k, ix in (("val_train", tr_idx), ("val_held", held_idx), ("test", te_idx))}
        fneg.write_bytes(pickle.dumps(NEG))
        times["negatives"] = time.time() - t0
        log.info(f"seed {s}: negatives scored in {times['negatives']:.0f} s")
    DEFS, rows_def = {}, []
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
                               c["fpr"], 1e6, dev)
            Dp = LinearDefender(cols, Ftr[cols].to_numpy(), Fpv[cols].to_numpy(), Fho[cols].to_numpy(),
                                c["fpr"], c["C_plus"], dev)
            DEF = {"D": D, "Dp": Dp, "val_auroc_D": auroc(D.decision(Fho[FEATS_D]), D.decision(Fpv[FEATS_D])),
                   "val_auroc_Dp": auroc(Dp.decision(Fho[cols]), Dp.decision(Fpv[cols])),
                   "n_val_src": len(G.yva), "n_val_pos": int(flip.sum()), "val_valid": 1.0,
                   "seconds": round(time.time() - t0, 1)}
            fdef.write_bytes(pickle.dumps(DEF))
            times[f"defenders_eps{eps:g}"] = time.time() - t0
        D, Dp = DEF["D"], DEF["Dp"]
        for obj in (D, Dp):
            obj.mu, obj.sd, obj.w = obj.mu.to(dev), obj.sd.to(dev), obj.w.to(dev)
        if DEF["val_auroc_D"] < c["min_val_auroc"]:
            raise StopRun(f"seed {s} eps {eps}: D val AUROC {DEF['val_auroc_D']:.3f} < {c['min_val_auroc']}")
        DEFS[eps] = DEF
        rows_def.append({"seed": s, "eps": eps, "n_val_src": DEF["n_val_src"], "n_val_pos": DEF["n_val_pos"],
                         "val_auroc_D": DEF["val_auroc_D"], "val_auroc_Dplus": DEF["val_auroc_Dp"], "thr_D": D.thr,
                         "thr_Dplus": Dp.thr, **{f"D_coef_{f}": w for f, w in zip(FEATS_D, D.lr.coef_[0])},
                         "Dplus_coef_abs_sum_consistency": float(np.abs(Dp.lr.coef_[0][len(FEATS_D):]).sum()),
                         "Dplus_coef_abs_sum_baselines": float(np.abs(Dp.lr.coef_[0][:len(FEATS_D)]).sum())})
    return NEG, DEFS, pd.DataFrame(rows_def), held_idx, times


def attack_list(c, s, G, model, DEF, held_idx, dev):
    D, Dp = DEF["D"], DEF["Dp"]
    att = [("A1", None)]
    for lam in c["lambdas"]:
        att.append((f"A6_lam{lam:g}", (terms_D(model, G.tp, G.base, D), D.thr, lam, None, c["a6_chunk"])))
    att.append((f"A6norm_w{c['norm_w']:g}", (terms_D(model, G.tp, G.base, D), D.thr, 0.0, c["norm_w"], c["a6_chunk"])))
    for lam in c["lambdas"]:
        att.append((f"A8_lam{lam:g}", (terms_A8(model, G.tp, G.base, Dp, c), Dp.thr, lam, None, c["a8_chunk"])))
    att.append((f"A8norm_w{c['norm_w']:g}", (terms_A8(model, G.tp, G.base, Dp, c), Dp.thr, 0.0, c["norm_w"],
                                             c["a8_chunk"])))
    if s == c["ablation_seed"]:
        Fvb = feats_np(model, G.Zben_va[torch.from_numpy(held_idx).to(dev)], G.base)
        for f in ("msp", "knn5_benign"):
            mu, sd, thr = float(Fvb[f].mean()), float(Fvb[f].std()), float(np.quantile(Fvb[f], 0.95))
            att.append((f"A6_{f.split('_')[0]}_only", (single_terms(model, G.tp, G.base, FEATS_D.index(f), mu, sd),
                                                       (thr - mu) / sd, c["a6_single_lambda"], None, c["a6_chunk"])))
    return att


def run_attack(c, s, G, model, DEF, eps, name, spec_a, sdir, dev, log):
    fres = sdir / f"rows_eps{eps:g}_{name}.parquet"
    if fres.exists():
        return pd.read_parquet(fres), False
    D, Dp = DEF["D"], DEF["Dp"]
    t0 = time.time()
    torch.cuda.reset_peak_memory_stats()
    if spec_a is None:
        a1 = np.load(S7D / f"seed{s}" / f"adv_raw_constrained_A1_eps{eps:g}.npz")
        i1 = np.array([np.flatnonzero(a1["pos"] == q)[0] for q in G.pos])
        X = a1["x_adv"][i1]
        with torch.no_grad():
            z = G.tp(torch.from_numpy(X).to(dev)).float()
    else:
        fn, thr, lam, nw, chunk = spec_a
        X, z = attack_loop(model, G.Ste, G.tp, eps, dict(c, attack_chunk=chunk), lam, nw, fn, thr, dev,
                           c["rng_seed"] + 7300 + int(eps * 100) + 17 * s + sum(map(ord, name)))
    check_valid_or_stop(X, G.Rte, G, eps, G.Ste, f"seed {s} eps {eps} {name}", log)
    with torch.no_grad():
        lg = model(z)
    flip = (lg.argmax(1) == BENIGN).cpu().numpy()
    sc, _ = detector_scores(model, z, G.base, c)
    F, cols = plus_features(sc)
    df = pd.DataFrame({"seed": s, "eps": eps, "attack": name, "pos": G.pos, "class": [G.classes[k] for k in G.yte],
                       "flipped": flip, "margin": (lg[:, BENIGN] - lg[:, 1:].amax(1)).cpu().numpy(),
                       "score_D": D.decision(F[FEATS_D]), "score_Dplus": Dp.decision(F[cols]),
                       "thr_D": D.thr, "thr_Dplus": Dp.thr, "valid": True, "seconds": time.time() - t0,
                       "peak_gpu_gb": torch.cuda.max_memory_allocated() / 2**30})
    df.to_parquet(fres, index=False)
    log.info(f"seed {s} eps {eps} {name}: success {flip.mean():.3f}, succ_undet D "
             f"{(flip & (df.score_D < D.thr)).mean():.3f} D_plus {(flip & (df.score_Dplus < Dp.thr)).mean():.3f} "
             f"[{time.time() - t0:.0f} s, peak {df.peak_gpu_gb.iloc[0]:.2f} GB]")
    return df, True


def neg_frame(NEG, DEF, s, eps):
    D, Dp = DEF["D"], DEF["Dp"]
    Fn, cols = plus_features(NEG["test"])
    return pd.DataFrame({"seed": s, "eps": eps, "score_D": D.decision(Fn[FEATS_D]), "score_Dplus": Dp.decision(Fn[cols]),
                         "thr_D": D.thr, "thr_Dplus": Dp.thr})


# ---------------------------------------------------------------- union and comparison helpers

def equal_effort(rows, n_boot, rng, setD=EQ_D, setP=EQ_P):
    # Per source row: D evaded by any of setD (flipped and below D's threshold), D_plus by any of setP.
    uD = union_eval(rows, setD)[["seed", "pos", "class", "union_flipped", "union_undet_D"]]
    uP = union_eval(rows, setP)[["seed", "pos", "union_flipped", "union_undet_Dplus"]]
    u = uD.merge(uP, on=["seed", "pos"], suffixes=("_Dset", "_Dplusset"))
    dlt, lo, hi = paired_union(u, n_boot, rng)
    return u, {"n_source_rows": len(u), "union_flipped_D_candidates": u.union_flipped_Dset.mean(),
               "union_flipped_Dplus_candidates": u.union_flipped_Dplusset.mean(),
               "succ_undet_union_D": u.union_undet_D.mean(), "succ_undet_union_Dplus": u.union_undet_Dplus.mean(),
               "Dplus_minus_D": dlt, "ci_lo": lo, "ci_hi": hi}


def all_union(rows, names, n_boot, rng):
    u = union_eval(rows, names)
    dlt, lo, hi = paired_union(u, n_boot, rng)
    return u, {"n_source_rows": len(u), "union_flipped": u.union_flipped.mean(),
               "succ_undet_union_D": u.union_undet_D.mean(), "succ_undet_union_Dplus": u.union_undet_Dplus.mean(),
               "Dplus_minus_D": dlt, "ci_lo": lo, "ci_hi": hi}


def rng_str(vals):
    v = np.asarray([x for x in vals if pd.notna(x)], float)
    return "" if not len(v) else (f"{v[0]:.3f}" if len(v) == 1 else f"{v.min():.3f} to {v.max():.3f}")


def with_spread(pooled, per_seed, keys, cols):
    # Adds "<col> seed range" (min to max over the per-seed tables) to the pooled table.
    out = pooled.copy()
    d = pd.concat(per_seed, ignore_index=True)
    for col in cols:
        g = d.groupby(keys)[col].apply(lambda x: rng_str(x.tolist())).rename(f"{col} seed range").reset_index()
        out = out.merge(g, on=keys, how="left")
    return out


def headline(R, negs, n_boot, rng, eps=1.0):
    # The side-by-side headline quantities for one dataset (rows R and test negatives with thresholds).
    R = R[R.eps == eps] if "eps" in R else R
    q = {}
    seeds = sorted(R.seed.unique())

    def undet(r, det):
        return (r.flipped & (r[f"score_{det}"] < r[f"thr_{det}"])).mean()

    a6 = {n: undet(R[R.attack == n], "D") for n in A6N if (R.attack == n).any()}
    a8 = {n: undet(R[R.attack == n], "Dplus") for n in A8N if (R.attack == n).any()}
    best6, best8 = max(a6, key=a6.get), (max(a8, key=a8.get) if a8 else None)
    per = lambda n, det: [undet(R[(R.attack == n) & (R.seed == s)], det) for s in seeds
                          if ((R.attack == n) & (R.seed == s)).any()]
    q["succ_undet_D under strongest A6"] = (a6[best6], rng_str(per(best6, "D")), best6)
    if best8:
        q["succ_undet_Dplus under strongest A8"] = (a8[best8], rng_str(per(best8, "Dplus")), best8)
    for fam, names in (("A6", A6N), ("A8", A8N)):
        for n in names:
            r = R[R.attack == n]
            if not len(r):
                continue
            fl = r.flipped.to_numpy()
            pD, pP = (r.score_D - r.thr_D).to_numpy()[fl], (r.score_Dplus - r.thr_Dplus).to_numpy()[fl]
            ns = negs[negs.seed.isin(r.seed.unique())]
            nD, nP = (ns.score_D - ns.thr_D).to_numpy(), (ns.score_Dplus - ns.thr_Dplus).to_numpy()
            pb = paired_boot(nD, pD, 0.0, nP, pP, 0.0, n_boot, rng)
            ps = []
            for s in sorted(r.seed.unique()):
                rs = r[r.seed == s]
                f_ = rs.flipped.to_numpy()
                if f_.any():
                    ps.append((rs.score_Dplus >= rs.thr_Dplus)[f_].mean() - (rs.score_D >= rs.thr_D)[f_].mean())
            if pb:
                q[f"TPR D_plus - D on {n} rows"] = (pb["tpr_diff"], rng_str(ps),
                                                    f"CI [{pb['tpr_diff_lo']:.3f}, {pb['tpr_diff_hi']:.3f}], n_pos {len(pD)}")
    if all((R.attack == n).any() for n in EQ_P):
        Re = R[R.seed.isin(sorted(R[R.attack.isin(A8N)].seed.unique()))]
        u, d = equal_effort(Re, n_boot, rng)
        ps_D = [u[u.seed == s].union_undet_D.mean() for s in sorted(u.seed.unique())]
        ps_P = [u[u.seed == s].union_undet_Dplus.mean() for s in sorted(u.seed.unique())]
        ps_d = [b - a for a, b in zip(ps_D, ps_P)]
        q["equal-effort union: succ_undet D (A1 + A6 x 3)"] = (d["succ_undet_union_D"], rng_str(ps_D), f"seeds {[int(x) for x in sorted(u.seed.unique())]}")
        q["equal-effort union: succ_undet D_plus (A1 + A8 x 3)"] = (d["succ_undet_union_Dplus"], rng_str(ps_P), "")
        q["equal-effort union: D_plus - D"] = (d["Dplus_minus_D"], rng_str(ps_d), f"CI [{d['ci_lo']:.3f}, {d['ci_hi']:.3f}]")
    return q


def load_2017(classes_keep, dev):
    # Saved 2017 Stage 6 FINAL (A1, A6, seeds 0-2) and Stage 6e (A8, seeds 0 and 2) rows at eps 1, with the test
    # negatives scored by the saved eps-1 defenders.
    fin, strong = resolve("results/stage6/final"), resolve("results/stage6/strong")
    R, N = [], []
    for s in (0, 1, 2):
        for n in ["A1"] + A6N:
            R.append(pd.read_parquet(fin / f"seed{s}" / f"rows_eps1_{n}.parquet"))
        for n in A8N:
            f = strong / f"seed{s}" / f"rows_eps1_{n}.parquet"
            if f.exists():
                R.append(pd.read_parquet(f))
        DEF = pickle.loads((fin / f"seed{s}" / "defenders_eps1.pkl").read_bytes())
        NEG = pickle.loads((fin / f"seed{s}" / "negatives.pkl").read_bytes())
        N.append(neg_frame(NEG, DEF, s, 1.0))
    R = pd.concat(R, ignore_index=True)
    if classes_keep is not None:
        R = R[R["class"].isin(classes_keep)]
    return R, pd.concat(N, ignore_index=True)


# ---------------------------------------------------------------- reports

def ci(m, lo, hi):
    return "" if pd.isna(m) else (f"{m:.3f}" if pd.isna(lo) else f"{m:.3f} [{lo:.3f}, {hi:.3f}]")


def build_tables(results_list, defs, c, seeds):
    # summarise() pooled over the given seeds plus the per-seed tables for the spread columns.
    TP = summarise(results_list, defs, c, "pooled")
    if len(results_list) > 1:
        per = [summarise([R], defs[defs.seed == s], c, f"seed {s}") for R, s in zip(results_list, seeds)]
        TP["attacks"] = with_spread(TP["attacks"], [p["attacks"] for p in per], ["eps", "attack"],
                                    ["evasion_success", "succ_undet_D", "succ_undet_Dplus"])
        TP["detection"] = with_spread(TP["detection"], [p["detection"] for p in per], ["eps", "attack", "detector"],
                                      ["auroc", "tpr_at_5fpr", "test_fpr"])
        TP["paired"] = with_spread(TP["paired"], [p["paired"] for p in per], ["eps", "attack"],
                                   ["tpr_diff", "auroc_diff"])
    return TP


def union_tables(rows, c, seeds):
    rng = np.random.default_rng(c["rng_seed"] + 992)
    eq, al, pc = [], [], []
    for eps in c["eps"]:
        r = rows[rows.eps == eps]
        for lab, sd in [(f"seed {s}", [s]) for s in seeds] + ([("pooled", seeds)] if len(seeds) > 1 else []):
            rr = r[r.seed.isin(sd)]
            u, d = equal_effort(rr, c["n_boot"], rng)
            eq.append({"eps": eps, "seeds": lab, **d})
            names = ["A1"] + A6N + A8N
            ua, da = all_union(rr, names, c["n_boot"], rng)
            al.append({"eps": eps, "seeds": lab, "candidates": "A1 + A6 x 3 + A8 x 3 (both detectors face all 7)", **da})
            if lab == ("pooled" if len(seeds) > 1 else f"seed {seeds[0]}"):
                g = u.groupby("class")
                t = pd.DataFrame({"n": g.size() / len(sd), "union_flipped_A1_A6": g.union_flipped_Dset.mean(),
                                  "union_flipped_A1_A8": g.union_flipped_Dplusset.mean(),
                                  "union_undet_D": g.union_undet_D.mean(),
                                  "union_undet_Dplus": g.union_undet_Dplus.mean()}).reset_index()
                t.insert(0, "eps", eps)
                pc.append(t)
    eqd, ald, pcd = pd.DataFrame(eq), pd.DataFrame(al), pd.concat(pc, ignore_index=True)
    pcd["flag"] = np.where(pcd.n < 100, "n < 100", "")
    return eqd, ald, pcd


def a8_per_class(rows):
    out = []
    for (eps, name), r in rows[rows.attack.isin(["A1", "A6_lam1", "A8_lam1"])].groupby(["eps", "attack"]):
        for cn, x in r.groupby("class"):
            f_ = x.flipped.to_numpy()
            out.append({"eps": eps, "attack": name, "class": cn, "n_rows": len(x), "n_per_seed": len(x) // x.seed.nunique(),
                        "success": f_.mean(), "succ_undet_D": (f_ & (x.score_D < x.thr_D).to_numpy()).mean(),
                        "succ_undet_Dplus": (f_ & (x.score_Dplus < x.thr_Dplus).to_numpy()).mean(),
                        "tpr_D": float((x.score_D >= x.thr_D)[f_].mean()) if f_.any() else np.nan,
                        "tpr_Dplus": float((x.score_Dplus >= x.thr_Dplus)[f_].mean()) if f_.any() else np.nan,
                        "seed_range_succ_undet_D": rng_str([(g.flipped & (g.score_D < g.thr_D)).mean() for _, g in x.groupby("seed")]),
                        "seed_range_succ_undet_Dplus": rng_str([(g.flipped & (g.score_Dplus < g.thr_Dplus)).mean() for _, g in x.groupby("seed")]),
                        "flag": "n < 100" if len(x) // x.seed.nunique() < 100 else ""})
    return pd.DataFrame(out)


def report7(T, eq, al, pcu, pca, standin, title, seeds):
    f = lambda d: d.round(3)
    out = [f"\n---\n\n## {datetime.now():%Y-%m-%d %H:%M}: {title}\n"]
    out.append(f"Seeds {seeds}. Source rows: Stage 7d rows (150 per attack class, webattack 85, dos_slowloris non-near-"
               "duplicate rows only); negatives: 1,000 genuine benign test rows per seed; defenders trained on "
               "validation only. Pooled tables carry the per-seed range (min to max) where several seeds are "
               "included. 0 invalid constrained rows everywhere (the run stops otherwise).\n")
    if standin is not None:
        sm = standin.groupby("feature", sort=False).agg(r2=("r2", "mean"), spearman=("spearman", "mean")).reset_index()
        rr = standin.groupby("feature", sort=False).agg(
            r2_range=("r2", lambda x: rng_str(x.tolist())), spearman_range=("spearman", lambda x: rng_str(x.tolist())))
        sm = sm.merge(rr.reset_index(), on="feature")
        med = standin.groupby("seed").r2.median()
        out.append("**A8 stand-in quality (VALIDATION rows: val benign predicted benign + val A1 eps 1 rows that flip; "
                   f"median R^2 per seed {', '.join(f'{s}: {v:.3f}' for s, v in med.items())})**\n\n" + md(f(sm)))
    dfn = T["defenders"]
    out.append("\n**Defenders (validation only)**\n\n" + md(f(dfn)))
    out.append("\n**Attacks: success, success AND undetected, cost vs A1**\n\n" + md(f(T["attacks"])))
    d = T["detection"].copy()
    d["AUROC [95% CI]"] = [ci(r.auroc, r.auroc_lo, r.auroc_hi) for r in d.itertuples()]
    d["TPR at 5% FPR [95% CI]"] = [ci(r.tpr_at_5fpr, r.tpr_lo, r.tpr_hi) for r in d.itertuples()]
    extra = [k for k in d.columns if k.endswith("seed range")]
    out.append("\n**Detection on test (negatives = 1,000 benign test rows per seed; TPR among flipped rows)**\n\n"
               + md(d[["eps", "attack", "detector", "n_pos", "AUROC [95% CI]", "TPR at 5% FPR [95% CI]", "test_fpr"]
                      + extra].round(3)))
    p = T["paired"].copy()
    if len(p):
        p["TPR D_plus - D [95% CI]"] = [ci(r.tpr_diff, r.tpr_diff_lo, r.tpr_diff_hi) for r in p.itertuples()]
        p["AUROC D_plus - D [95% CI]"] = [ci(r.auroc_diff, r.auroc_diff_lo, r.auroc_diff_hi) for r in p.itertuples()]
        extra = [k for k in p.columns if k.endswith("seed range")]
        out.append("\n**Paired D_plus minus D (same rows, paired bootstrap)**\n\n"
                   + md(p[["eps", "attack", "n_pos", "TPR D_plus - D [95% CI]", "AUROC D_plus - D [95% CI]"] + extra]))
    e = eq.copy()
    e["D_plus - D [95% CI]"] = [ci(r.Dplus_minus_D, r.ci_lo, r.ci_hi) for r in e.itertuples()]
    out.append("\n**PRIMARY: equal attacker effort, oracle union per defender** (D attacked by A1 + A6 x 3, D_plus by "
               "A1 + A8 x 3; a source row counts as evaded if ANY candidate aimed at that defender flips it to benign "
               "and stays below that defender's threshold; attacker with query access; paired bootstrap over source "
               "rows within seed)\n\n" + md(f(e.drop(columns=["Dplus_minus_D", "ci_lo", "ci_hi"]))))
    a = al.copy()
    a["D_plus - D [95% CI]"] = [ci(r.Dplus_minus_D, r.ci_lo, r.ci_hi) for r in a.itertuples()]
    out.append("\n**Secondary: union of all 7 candidates under each detector.** This comparison is ASYMMETRIC: the A6 "
               "candidates are aimed at D and the A8 candidates at D_plus, so each detector also faces candidates that "
               "were not optimised against it.\n\n" + md(f(a.drop(columns=["Dplus_minus_D", "ci_lo", "ci_hi"]))))
    out.append("\n**Per source class: equal-effort union** (n = rows per seed)\n\n" + md(f(pcu)))
    out.append("\n**Per source class: A1, A6 lambda 1, A8 lambda 1** (n_per_seed = rows per seed)\n\n" + md(f(pca)))
    return "\n".join(out)


# ---------------------------------------------------------------- main

def main():
    cfgmod.DEFAULTS.setdefault("stage7e", {**copy.deepcopy(STAGE4), **copy.deepcopy(STAGE6), **copy.deepcopy(STAGE6B),
                                           **copy.deepcopy(STAGE6C), **copy.deepcopy(STAGE6D), **copy.deepcopy(STAGE6E),
                                           **copy.deepcopy(STAGE7E)})
    cfg, overrides = load_config("Stage 7e: defenders and adaptive attackers on LycoS-Unicas-IDS2018.")
    c = cfg["stage7e"]
    if c["smoke"]:
        c.update(n_per_class=3, steps=2, val_per_class=5, d_train_neg=2000, n_heldout=3000, n_test_benign=300,
                 n_boot=20, n_val_benign_check=200, out_dir=c["out_dir"] + "_smoke")
    out = resolve(c["out_dir"])
    out.mkdir(parents=True, exist_ok=True)
    t_start = time.time()
    run_dir, log = start_run("stage7e_defense_2018", cfg, overrides)
    dev = torch.device("cuda")
    keep_awake(True)
    seeds = c["seeds"]
    try:
        G = load_common7(c, dev)
        log.info(f"{len(G.pos)} test source rows, {len(G.yva)} val source rows; classes {G.classes}")
        # ---- stand-in quality on every seed's validation rows, BEFORE any attack
        f_si = out / "standin_quality.csv"
        if f_si.exists():
            standin = pd.read_csv(f_si)
        else:
            parts = []
            for s in seeds:
                m = load_model(s, dev)
                parts.append(standin_quality(m, G, c, dev, s, log))
                del m
                torch.cuda.empty_cache()
            standin = pd.concat(parts, ignore_index=True)
            standin.to_csv(f_si, index=False)
        med = standin.groupby("seed").r2.median()
        msg = ("A8 STAND-IN QUALITY (validation rows), before the attacks:\n" + standin.round(3).to_string(index=False)
               + f"\nmedian R^2 per seed: {med.round(3).to_dict()}"
               + ("  (BELOW 0.5 for at least one seed: reported, continuing as instructed)" if (med < 0.5).any() else ""))
        log.info(msg)
        print(msg, flush=True)

        all_res, all_def, all_rows = [], [], []
        for si, s in enumerate(seeds):
            ts = time.time()
            sdir = out / f"seed{s}"
            sdir.mkdir(parents=True, exist_ok=True)
            model = load_model(s, dev)
            NEG, DEFS, dfn, held_idx, ptimes = prepare_seed(c, s, G, model, dev, log, sdir)
            R = {}
            lists = {eps: attack_list(c, s, G, model, DEFS[eps], held_idx, dev) for eps in c["eps"]}
            if si == 0:
                # ---- runtime estimate from A6_lam1 and A8_lam1 (chunk 100) at eps 1.0, printed before the rest
                pe = c["probe_eps"]
                spec = dict(lists[pe])
                t = {}
                for name in ("A6_lam1", "A8_lam1"):
                    df, _ = run_attack(c, s, G, model, DEFS[pe], pe, name, spec[name], sdir, dev, log)
                    R[(pe, name)] = df
                    t[name] = float(df.seconds.iloc[0])
                t_neg = ptimes.get("negatives", 0.0)
                t_def = sum(v for k, v in ptimes.items() if k.startswith("defenders"))
                n_eps = len(c["eps"])
                per_seed = t_neg + t_def + n_eps * (3 * t["A6_lam1"] + 3 * t["A8_lam1"] + 0.1 * t["A6_lam1"])
                ablation = n_eps * 2 * t["A6_lam1"]
                done = time.time() - t_start
                hours = (done + len(seeds) * per_seed + ablation - (t["A6_lam1"] + t["A8_lam1"])) / 3600
                if not ptimes:
                    hours = (done + len(seeds) * per_seed + ablation) / 3600
                msg = (f"RUNTIME ESTIMATE: negatives {t_neg:.0f} s, defenders {t_def:.0f} s, A6_lam1 {t['A6_lam1']:.0f} s, "
                       f"A8_lam1 (chunk {c['a8_chunk']}) {t['A8_lam1']:.0f} s per setting on seed {s}, eps {pe}; full run "
                       f"about {hours:.2f} h (limit {c['max_hours']} h)")
                log.info(msg)
                print(msg, flush=True)
                (out / "runtime_estimate.txt").write_text(msg + "\n", encoding="utf-8")
                if hours > c["max_hours"]:
                    raise StopRun(msg)
            for eps in c["eps"]:
                for name, spec_a in lists[eps]:
                    if (eps, name) in R:
                        continue
                    R[(eps, name)], _ = run_attack(c, s, G, model, DEFS[eps], eps, name, spec_a, sdir, dev, log)
                R[(eps, "_neg")] = neg_frame(NEG, DEFS[eps], s, eps)
            del model
            torch.cuda.empty_cache()
            rows = pd.concat([v for k, v in R.items() if k[1] != "_neg"], ignore_index=True)
            all_res.append(R), all_def.append(dfn), all_rows.append(rows)
            T1 = build_tables([R], dfn, c, [s])
            eq, al, pcu = union_tables(rows, c, [s])
            pca = a8_per_class(rows)
            for k, v in list(T1.items()) + [("union_equal_effort", eq), ("union_all", al), ("per_class_union", pcu),
                                            ("per_class_attacks", pca)]:
                v.to_csv(sdir / f"summary_{k}.csv", index=False)
            (sdir / "seed_time_s.txt").write_text(f"{time.time() - ts:.1f}\n", encoding="utf-8")
            (out / f"stage7e_report_seed{s}.md").write_text(
                report7(T1, eq, al, pcu, pca, standin[standin.seed == s],
                        f"Stage 7e (2018): seed {s} ({(time.time() - ts) / 60:.0f} min this session)", [s]),
                encoding="utf-8")
            log.info(f"seed {s} finished in {(time.time() - ts) / 60:.1f} min")

        # ---- pooled
        defs = pd.concat(all_def, ignore_index=True)
        rows = pd.concat(all_rows, ignore_index=True)
        TP = build_tables(all_res, defs, c, seeds)
        eq, al, pcu = union_tables(rows, c, seeds)
        pca = a8_per_class(rows)
        for k, v in list(TP.items()) + [("union_equal_effort", eq), ("union_all", al), ("per_class_union", pcu),
                                        ("per_class_attacks", pca)]:
            v.to_csv(out / f"summary_pooled_{k}.csv", index=False)
        pooled_txt = report7(TP, eq, al, pcu, pca, standin, f"Stage 7e (2018): pooled over seeds {seeds}", seeds)
        # ---- 2017 vs 2018 headline (eps 1.0)
        negs18 = pd.concat([R[(1.0, "_neg")] for R in all_res], ignore_index=True)
        rng = np.random.default_rng(c["rng_seed"] + 995)
        h18 = headline(rows, negs18, c["n_boot"], rng)
        R17c, N17 = load_2017(COMMON, dev)
        h17c = headline(R17c, N17, c["n_boot"], rng)
        R17a, _ = load_2017(None, dev)
        h17a = headline(R17a, N17, c["n_boot"], rng)
        keys = list(dict.fromkeys(list(h17a) + list(h18)))
        cmp = []
        for k in keys:
            a, b, cc_ = h17a.get(k, (np.nan, "", "")), h17c.get(k, (np.nan, "", "")), h18.get(k, (np.nan, "", ""))
            cmp.append({"quantity (eps 1.0)": k, "2017 all 10 classes": a[0], "2017 range": a[1], "2017 note": a[2],
                        "2017 common 7": b[0], "2017 common 7 range": b[1], "2017 common 7 note": b[2],
                        "2018": cc_[0], "2018 range": cc_[1], "2018 note": cc_[2],
                        "same sign (2017 common 7 vs 2018)": bool(np.sign(b[0]) == np.sign(cc_[0])) if pd.notna(b[0]) and pd.notna(cc_[0]) else ""})
        cmp = pd.DataFrame(cmp)
        cmp.to_csv(out / "stage7e_cmp2017_headline.csv", index=False)
        cmp_txt = ("\n**2017 vs 2018 headline (eps 1.0)**. 2017: Stage 6 FINAL rows (A1, A6; seeds 0, 1, 2) and Stage 6e "
                   "rows (A8; seeds 0 and 2 only), scored with the saved 2017 defenders; recomputed on all 10 classes and "
                   "on the 7 common classes. 'range' = per-seed min to max; the equal-effort union uses the seeds where A8 "
                   "exists (2017: 0 and 2).\n\n" + md(cmp, "{:.3f}"))
        (out / "stage7e_report_pooled.md").write_text(pooled_txt + "\n" + cmp_txt + "\n", encoding="utf-8")
        rt = pd.DataFrame([{"seed": s, "seed_time_s_last_session": float((out / f"seed{s}" / "seed_time_s.txt").read_text()),
                            "attack_s_total": float(r.groupby(["eps", "attack"]).seconds.first().sum()),
                            "peak_gpu_gb": float(r.peak_gpu_gb.max())} for s, r in zip(seeds, all_rows)])
        rt.to_csv(out / "stage7e_runtime.csv", index=False)
        log.info(f"done; total {(time.time() - t_start) / 3600:.2f} h")
    except StopRun as e:
        log.error(f"STOPPED: {e}")
        (out / "stage7e_STOPPED.md").write_text(f"## {datetime.now():%Y-%m-%d %H:%M}: Stage 7e stopped\n\n"
                                                f"Stop condition: {e}\n", encoding="utf-8")
        finish_run(run_dir, log, {"stopped": str(e)}, t_start)
        keep_awake(False)
        sys.exit(2)
    keep_awake(False)
    finish_run(run_dir, log, {"seeds": seeds}, t_start)


if __name__ == "__main__":
    main()
