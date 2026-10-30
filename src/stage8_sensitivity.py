# Stage 8a3: near-duplicate sensitivity re-analysis, CPU only, from saved result files (no retraining, no new attacks).
#
# Near-duplicate flag of a test row: exact distance to its nearest train row < 1e-3 in the dataset's standardised space
# (results/stage8/neardup_rows_{2017,2018}.parquet, Stage 8a2). Every saved source row carries its position in the
# explanation-set selection ("pos"), which maps to its test-row index (results/stage3/selection.parquet for 2017,
# results/stage7/stage7c/raw/selection.parquet for 2018) and from there to the flag.
#
# Variants, 7 common classes (V2: 5 classes):
#   V0  all source rows (must reproduce the Stage 8a tables T5 to T9; checked, the run stops on a mismatch)
#   V1  near-duplicate source rows removed in both datasets
#   V2  dos_slowloris and webattack removed entirely in both datasets
# Rows are filtered after the fact: the attacks, the models and the defenders (trained on validation) are unchanged.
# Quantities: Stage 5 tables (stage5_fullrun.tables, as T5 / T6), defender tables (as T7 / T9), paired TPR / AUROC
# D_plus minus D per attack (stage6d_final.paired_boot), equal-effort union with the exact bootstrap schedule of T8.
# Defender negatives: the 1,000 benign test rows of each seed are recovered by replaying the Stage 6 FINAL / 7e
# sampling (same rng stream, model predictions recomputed on the CPU) and verified against the saved detector scores
# before their near-duplicate flags are used.
#
# Outputs: results/stage8/sensitivity_*.csv.
# Usage: set CUDA_VISIBLE_DEVICES= (empty) and run .venv\Scripts\python.exe -m src.stage8_sensitivity
import copy
import io
import json
import pickle
import sys
import time

import numpy as np
import pandas as pd
import torch

# Saved defender pickles hold CUDA tensors; load every storage on the CPU (this analysis never uses the GPU).
torch.storage._load_from_bytes = lambda b: torch.load(io.BytesIO(b), map_location="cpu", weights_only=False)

from src.preprocess import Preprocessor  # noqa: E402
from src.stage3_explain import load_ftt  # noqa: E402
from src.stage4_attack import STAGE4  # noqa: E402
from src.stage5_fullrun import STAGE5, load_seed, tables  # noqa: E402
from src.stage6_defense import BENIGN  # noqa: E402
from src.stage6d_final import LinearDefender, paired_boot  # noqa: E402

# The 2017 defenders were pickled while src/stage6d_final.py ran as __main__; resolve that class name here.
sys.modules["__main__"].LinearDefender = LinearDefender
from src.stage7e_defense_2018 import a8_per_class, equal_effort, neg_frame, union_tables  # noqa: E402
from src.utils import ROOT  # noqa: E402

sys.path.insert(0, str(ROOT / "paper"))
import data_v2 as D  # noqa: E402

OUT = ROOT / "results" / "stage8"
RES = ROOT / "results"
S7 = RES / "stage7"
THR = 1e-3
SEEDS = [0, 1, 2]
COMMON = D.COMMON
V2_DROP = ["dos_slowloris", "webattack"]
RNG_SEED = 4026
C5 = {**copy.deepcopy(STAGE4), **copy.deepcopy(STAGE5)}
C7E = {"rng_seed": RNG_SEED, "eps": [0.5, 1.0], "n_boot": 1000}
P = {"2017": {"sel": RES / "stage3" / "selection.parquet", "s5": RES / "stage5", "def": RES / "stage6" / "final",
              "test": RES / "stage1" / "test.parquet", "val": RES / "stage1" / "val.parquet",
              "pp": RES / "stage2" / "preprocess.json", "feats": RES / "stage1" / "features.json",
              "model": lambda s: RES / "stage2" / f"ftt_seed{s}.pt"},
     "2018": {"sel": S7 / "stage7c" / "raw" / "selection.parquet", "s5": S7 / "stage7d", "def": S7 / "stage7e",
              "test": S7 / "data" / "test.parquet", "val": S7 / "data" / "val.parquet", "pp": S7 / "preprocess.json",
              "feats": S7 / "data" / "features.json", "model": lambda s: S7 / f"seed{s}" / "ftt.pt"}}


class StopRun(Exception):
    pass


def log(msg):
    print(msg, flush=True)


# ---------------------------------------------------------------- flags

def flags(year):
    sel = pd.read_parquet(P[year]["sel"])
    nd = pd.read_parquet(RES / "stage8" / f"neardup_rows_{year}.parquet").set_index("test_row")["nn_dist_exact"]
    tr = sel["test_row"].to_numpy()
    dist = nd.reindex(tr).to_numpy()
    if np.isnan(dist).any():
        raise StopRun(f"{year}: {int(np.isnan(dist).sum())} selection rows have no near-duplicate distance")
    return pd.DataFrame({"pos": np.arange(len(sel)), "test_row": tr, "class_sel": sel["class"].to_numpy(),
                         "nn_dist": dist, "neardup": dist < THR})


def verify_flags(F, log_rows):
    # Counts on every saved source-row set, and agreement with neardup_2017_stage5_source_rows.csv.
    ref = pd.read_csv(OUT / "neardup_2017_stage5_source_rows.csv").set_index("class")
    sets = {"2017": {"Stage 5 rows": RES / "stage5" / "seed0" / "rows_constrained_A1_eps1.parquet",
                     "Stage 6 FINAL rows": RES / "stage6" / "final" / "seed0" / "rows_eps1_A6_lam1.parquet",
                     "Stage 6e rows": RES / "stage6" / "strong" / "seed0" / "rows_eps1_A8_lam1.parquet"},
            "2018": {"Stage 7d rows": S7 / "stage7d" / "seed0" / "rows_constrained_A1_eps1.parquet",
                     "Stage 7e rows": S7 / "stage7e" / "seed0" / "rows_eps1_A8_lam1.parquet"}}
    out = []
    for yr, d in sets.items():
        f = F[yr].set_index("pos")
        pos_sets = []
        for name, path in d.items():
            r = pd.read_parquet(path, columns=["pos", "class"])
            if not (f.loc[r.pos, "class_sel"].to_numpy() == r["class"].to_numpy()).all():
                raise StopRun(f"{yr} {name}: class of the stored pos does not match the selection")
            pos_sets.append(set(r.pos))
            g = r.assign(nd=f.loc[r.pos, "neardup"].to_numpy()).groupby("class").nd.agg(["size", "sum"])
            for cn, x in g.iterrows():
                out.append({"dataset": yr, "row_set": name, "class": cn, "n_source_rows": int(x["size"]),
                            "n_neardup": int(x["sum"])})
            out.append({"dataset": yr, "row_set": name, "class": "ALL", "n_source_rows": int(g["size"].sum()),
                        "n_neardup": int(g["sum"].sum())})
        if any(p != pos_sets[0] for p in pos_sets):
            raise StopRun(f"{yr}: the source-row sets of the attack and defender stages differ")
    V = pd.DataFrame(out)
    s5 = V[(V.dataset == "2017") & (V.row_set == "Stage 5 rows")].set_index("class")
    for cn in ref.index:
        a, b = int(s5.loc[cn, "n_neardup"]), int(ref.loc[cn, "n_lt_1e3"])
        log_rows.append({"check": f"2017 Stage 5 near-duplicate count, {cn}", "value": a, "expected": b, "pass": a == b})
    s7 = V[(V.dataset == "2018") & (V.row_set == "Stage 7d rows") & (V["class"] == "dos_slowloris")]
    log_rows.append({"check": "2018 Stage 7d dos_slowloris near-duplicate count (filtered by design)",
                     "value": int(s7.n_neardup.iloc[0]), "expected": 0, "pass": int(s7.n_neardup.iloc[0]) == 0})
    if not all(r["pass"] for r in log_rows):
        raise StopRun("near-duplicate mapping could not be verified: " +
                      "; ".join(f"{r['check']} {r['value']} vs {r['expected']}" for r in log_rows if not r["pass"]))
    return V


def keep_pos(F, year, variant):
    f = F[year]
    k = f.class_sel.isin(COMMON)
    if variant == "V1":
        k &= ~f.neardup
    elif variant == "V2":
        k &= ~f.class_sel.isin(V2_DROP)
    return set(f.pos[k])


# ---------------------------------------------------------------- Stage 5 quantities (T5 / T6)

def stage5_long(year, F, cache):
    out, Tv0 = [], {}
    for s in SEEDS:
        if (year, s) not in cache:
            cache[(year, s)] = load_seed(P[year]["s5"] / f"seed{s}")
        rows, fa, clean = cache[(year, s)]
        for v in ("V0", "V1", "V2"):
            kp = keep_pos(F, year, v)
            r = rows[rows.pos.isin(kp)]
            T = tables(r.assign(seed=s), fa[fa.pos.isin(kp)].assign(seed=s), clean[clean.pos.isin(kp)].assign(seed=s),
                       None, C5)
            if v == "V0":
                Tv0[s] = T
            su, ex = T["success"], T["faith_excess_vs_A0"]
            for _, x in su.iterrows():
                for col in ("success", "mass_ratio_rollout", "mass_ratio_attn_mean", "validity"):
                    out.append({"dataset": year, "variant": v, "seed": s, "quantity": col, "mode": x["mode"],
                                "attack": x.attack, "eps": x.eps, "method": "", "value": x[col]})
            for e in sorted(su.eps.unique()):
                c_ = su[(su["mode"] == "constrained") & (su.eps == e)].set_index("attack")["success"]
                if "A1" in c_ and "A0" in c_:
                    out.append({"dataset": year, "variant": v, "seed": s, "quantity": "A1_minus_A0_success",
                                "mode": "constrained", "attack": "A1-A0", "eps": e, "method": "", "value": c_["A1"] - c_["A0"]})
            x = ex[(ex["mode"] == "constrained") & (ex.attack == "A1") & (ex.subset == "flipped")]
            for _, y in x.iterrows():
                out.append({"dataset": year, "variant": v, "seed": s, "quantity": "aopc_gap_margin_flipped_minus_A0",
                            "mode": "constrained", "attack": "A1", "eps": y.eps, "method": y.method,
                            "value": y.aopc_gap_margin})
            nrow = r[(r.attack == "A0") & (r.eps == r.eps.min())].shape[0]
            out.append({"dataset": year, "variant": v, "seed": s, "quantity": "n_source_rows", "mode": "", "attack": "",
                        "eps": np.nan, "method": "", "value": nrow})
    return pd.DataFrame(out), Tv0


def check_stage5_v0(year, Tv0, checks):
    # V0 against the per-seed tables that produced T5 / T6.
    for s in SEEDS:
        if year == "2018":
            fs, fe = S7 / "stage7d" / f"seed{s}" / "summary_success.csv", S7 / "stage7d" / f"seed{s}" / "summary_faith_excess_vs_A0.csv"
        else:
            fs, fe = S7 / "stage7d" / f"cmp2017_seed{s}_success.csv", S7 / "stage7d" / f"cmp2017_seed{s}_faith_excess_vs_A0.csv"
        a, b = Tv0[s]["success"], pd.read_csv(fs)
        m = a.merge(b, on=["mode", "attack", "eps"], suffixes=("", "_ref"))
        d0 = float((m["success"] - m["success_ref"]).abs().max())
        d1 = float((m["mass_ratio_rollout"] - m["mass_ratio_rollout_ref"]).abs().max())
        a, b = Tv0[s]["faith_excess_vs_A0"], pd.read_csv(fe)
        m = a.merge(b, on=["mode", "attack", "eps", "subset", "method"], suffixes=("", "_ref"))
        d2 = float((m["aopc_gap_margin"] - m["aopc_gap_margin_ref"]).abs().max())
        checks.append({"check": f"{year} seed {s} V0 success vs T5 source (max abs diff, {len(m)} merged rows)",
                       "value": d0, "expected": "0 (< 1e-12)", "pass": d0 < 1e-12})
        checks.append({"check": f"{year} seed {s} V0 mass ratio vs T6 source (max abs diff; float32 values via CSV)",
                       "value": d1, "expected": "< 1e-6", "pass": d1 < 1e-6})
        checks.append({"check": f"{year} seed {s} V0 margin AOPC gap excess vs T6 source (max abs diff; via CSV)",
                       "value": d2, "expected": "< 1e-6", "pass": d2 < 1e-6})


# ---------------------------------------------------------------- defenders (T7 to T9)

def negatives(year):
    out = []
    for s in SEEDS:
        sd = P[year]["def"] / f"seed{s}"
        NEG = pickle.loads((sd / "negatives.pkl").read_bytes())
        for eps in (0.5, 1.0):
            DEF = pickle.loads((sd / f"defenders_eps{eps:g}.pkl").read_bytes())
            out.append(neg_frame(NEG, DEF, s, eps).assign(neg_idx=np.arange(len(NEG["test"]))))
    return pd.concat(out, ignore_index=True)


def defender_quantities(year, F, R, negs, checks):
    rows_out, pair_out, cls_out = [], [], []
    for v in ("V0", "V1", "V2"):
        kp = keep_pos(F, year, v)
        Rv = R[R.pos.isin(kp)]
        t = D.defender_table(Rv)
        rows_out.append(t.assign(dataset=year, variant=v))
        for s in SEEDS:
            ts = D.defender_table(Rv[Rv.seed == s])
            if len(ts):
                rows_out.append(ts.assign(dataset=year, variant=v, seed_only=s))
        rng = np.random.default_rng(RNG_SEED + 995)
        for atk in ["A1"] + D.A6N + D.A8N:
            r = Rv[(Rv.eps == 1.0) & (Rv.attack == atk)]
            if not len(r):
                continue
            fl = r.flipped.to_numpy()
            pD, pP = (r.score_D - r.thr_D).to_numpy()[fl], (r.score_Dplus - r.thr_Dplus).to_numpy()[fl]
            ns = negs[(negs.eps == 1.0) & negs.seed.isin(r.seed.unique())]
            nD, nP = (ns.score_D - ns.thr_D).to_numpy(), (ns.score_Dplus - ns.thr_Dplus).to_numpy()
            pb = paired_boot(nD, pD, 0.0, nP, pP, 0.0, 1000, rng)
            per = []
            for s in sorted(r.seed.unique()):
                x = r[r.seed == s]
                f_ = x.flipped.to_numpy()
                if f_.any():
                    per.append((x.score_Dplus >= x.thr_Dplus).to_numpy()[f_].mean() - (x.score_D >= x.thr_D).to_numpy()[f_].mean())
            pair_out.append({"dataset": year, "variant": v, "attack": atk, "n_pos": len(pD), **pb,
                             "tpr_diff_seed_lo": min(per) if per else np.nan, "tpr_diff_seed_hi": max(per) if per else np.nan})
        R1 = Rv[Rv.eps == 1.0]
        if year == "2018":
            pc = a8_per_class(Rv)
        else:
            pc = pd.concat([a8_per_class(R1[R1.attack.isin(["A1", "A6_lam1"])]), a8_per_class(R1[R1.attack == "A8_lam1"])],
                           ignore_index=True)
        cls_out.append(pc.assign(dataset=year, variant=v))
    T = pd.concat(rows_out, ignore_index=True)
    PC = pd.concat(cls_out, ignore_index=True)
    # V0 checks against the Stage 8a / 7e sources of T7 and T9.
    v0 = T[(T.variant == "V0") & T.seed_only.isna()] if "seed_only" in T else T[T.variant == "V0"]
    ref = D.defender_table(R)
    m = v0.merge(ref, on="attack", suffixes=("", "_ref"))
    d = max(float((m[c] - m[f"{c}_ref"]).abs().max()) for c in ("success", "undet_D", "undet_Dplus"))
    checks.append({"check": f"{year} V0 defender table vs T7 (max abs diff)", "value": d, "expected": 0.0, "pass": d < 1e-12})
    if year == "2018":
        for s in SEEDS:
            sa = pd.read_csv(S7 / "stage7e" / f"seed{s}" / "summary_attacks.csv")
            sa = sa[sa.eps == 1.0].set_index("attack")
            x = T[(T.variant == "V0") & (T.seed_only == s)].set_index("attack")
            mp = {"undet_D": "succ_undet_D", "undet_Dplus": "succ_undet_Dplus"}
            dd = max(float((x[c] - sa.loc[x.index, mp[c]]).abs().max()) for c in mp)
            checks.append({"check": f"2018 seed {s} V0 succ_undet vs results/stage7/stage7e/seed{s}/summary_attacks.csv",
                           "value": dd, "expected": 0.0, "pass": dd < 1e-12})
        ref_pc = pd.read_csv(S7 / "stage7e" / "summary_pooled_per_class_attacks.csv")
    else:
        ref_pc, _ = D.per_class_attacks("2017")
    keys = ["eps", "attack", "class"]
    m = PC[PC.variant == "V0"].merge(ref_pc, on=keys, suffixes=("", "_ref"))
    d = max(float((m[c] - m[f"{c}_ref"]).abs().max()) for c in ("success", "succ_undet_D", "succ_undet_Dplus"))
    checks.append({"check": f"{year} V0 per-class table vs T9 source (max abs diff)", "value": d, "expected": 0.0,
                   "pass": d < 1e-12 and len(m) == len(ref_pc[ref_pc.attack.isin(PC.attack.unique())])})
    return T, pd.DataFrame(pair_out), PC


def unions(year, F, R, checks):
    out = []
    for v in ("V0", "V1", "V2"):
        Rv = R[R.pos.isin(keep_pos(F, year, v))]
        if year == "2018":
            eq, _, _ = union_tables(Rv, C7E, SEEDS)                   # the Stage 7e schedule (T8 2018 source)
            out.append(eq.assign(dataset=year, variant=v))
        else:
            Re = Rv[Rv.eps == 1.0]
            seeds = sorted(Re[Re.attack.isin(D.A8N)].seed.unique())
            Re = Re[Re.seed.isin(seeds)]
            rng = np.random.default_rng(RNG_SEED + 992)                # the Stage 8a schedule (T8 2017 source)
            for lab, sd in [(f"seed {int(s)}", [s]) for s in seeds] + [("pooled", seeds)]:
                _, d = equal_effort(Re[Re.seed.isin(sd)], 1000, rng)
                out.append(pd.DataFrame([{"eps": 1.0, "seeds": lab, **d, "dataset": year, "variant": v}]))
    U = pd.concat(out, ignore_index=True)
    ref, _, _ = D.equal_effort_table(year)
    m = U[(U.variant == "V0") & (U.eps == 1.0)].merge(ref, on=["eps", "seeds"], suffixes=("", "_ref"))
    d = max(float((m[c] - m[f"{c}_ref"]).abs().max()) for c in ("succ_undet_union_D", "succ_undet_union_Dplus",
                                                                 "Dplus_minus_D", "ci_lo", "ci_hi"))
    checks.append({"check": f"{year} V0 equal-effort union vs T8 (incl. CI, max abs diff)", "value": d, "expected": 0.0,
                   "pass": d < 1e-12 and len(m) == len(ref)})
    return U


# ---------------------------------------------------------------- defender negatives

def recover_negatives(year, negs_scores, checks):
    # Replays the Stage 6 FINAL / 7e sampling of the 1,000 benign test negatives per seed (rng_seed + 8000 + s:
    # permutation of the val benign rows predicted benign, then a choice among the test benign rows predicted benign).
    feats = json.loads(P[year]["feats"].read_text(encoding="utf-8"))
    pp = Preprocessor.load(P[year]["pp"])
    te = pd.read_parquet(P[year]["test"], columns=feats + ["label_id"])
    va = pd.read_parquet(P[year]["val"], columns=feats + ["label_id"])
    ben_te = np.flatnonzero(te.label_id.to_numpy() == BENIGN)
    Zte = torch.from_numpy(pp.transform(te.iloc[ben_te]))
    Zva = torch.from_numpy(pp.transform(va[va.label_id == BENIGN]))
    del te, va
    nd = pd.read_parquet(RES / "stage8" / f"neardup_rows_{year}.parquet").set_index("test_row")["nn_dist_exact"]
    out = []
    for s in SEEDS:
        model, _ = load_ftt(P[year]["model"](s), torch.device("cpu"))
        with torch.no_grad():
            pv = torch.cat([model(Zva[i:i + 8192]).argmax(1) for i in range(0, len(Zva), 8192)]).numpy()
            lt = torch.cat([model(Zte[i:i + 8192]) for i in range(0, len(Zte), 8192)])
        pt = lt.argmax(1).numpy()
        rng_s = np.random.default_rng(RNG_SEED + 8000 + s)
        rng_s.permutation(np.flatnonzero(pv == BENIGN))
        te_idx = rng_s.choice(np.flatnonzero(pt == BENIGN), size=1000, replace=False)
        NEG = pickle.loads((P[year]["def"] / f"seed{s}" / "negatives.pkl").read_bytes())
        msp = -torch.softmax(lt[te_idx], -1).max(1).values.numpy()
        d = float(np.abs(msp - NEG["test"]["msp"].to_numpy()).max())
        checks.append({"check": f"{year} seed {s} recovered test negatives: max |msp - saved msp| (CPU recompute)",
                       "value": d, "expected": "< 1e-4", "pass": d < 1e-4})
        if d >= 1e-4:
            raise StopRun(f"{year} seed {s}: the benign test negatives could not be recovered (max msp diff {d:.2e})")
        tr = ben_te[te_idx]
        out.append(pd.DataFrame({"dataset": year, "seed": s, "neg_idx": np.arange(1000), "test_row": tr,
                                 "nn_dist": nd.reindex(tr).to_numpy(), "neardup": nd.reindex(tr).to_numpy() < THR}))
    N = pd.concat(out, ignore_index=True)
    return negs_scores.merge(N, on=["seed", "neg_idx"], how="left")


def negative_tables(year, NN, R, F):
    rows = []
    for (s, eps), g in list(NN.groupby(["seed", "eps"])) + [((-1, e), NN[NN.eps == e]) for e in (0.5, 1.0)]:
        rec = {"dataset": year, "seed": "pooled" if s == -1 else s, "eps": eps, "n_neg": len(g),
               "n_neardup": int(g.neardup.sum()), "share_neardup": float(g.neardup.mean())}
        for sub, x in (("neardup", g[g.neardup]), ("non_neardup", g[~g.neardup])):
            rec[f"n_{sub}"] = len(x)
            rec[f"fpr_D_{sub}"] = float((x.score_D >= x.thr_D).mean()) if len(x) else np.nan
            rec[f"fpr_Dplus_{sub}"] = float((x.score_Dplus >= x.thr_Dplus).mean()) if len(x) else np.nan
        rows.append(rec)
    tp = []
    f = F[year].set_index("pos")
    R1 = R[(R.eps == 1.0) & R.pos.isin(keep_pos(F, year, "V0"))].assign(neardup=lambda x: f.loc[x.pos, "neardup"].to_numpy())
    for atk, r in R1.groupby("attack"):
        for sub, x in (("neardup", r[r.neardup]), ("non_neardup", r[~r.neardup])):
            fl = x.flipped.to_numpy()
            tp.append({"dataset": year, "attack": atk, "eps": 1.0, "source_rows": sub, "n_rows": len(x),
                       "n_flipped": int(fl.sum()),
                       "tpr_D": float((x.score_D >= x.thr_D).to_numpy()[fl].mean()) if fl.any() else np.nan,
                       "tpr_Dplus": float((x.score_Dplus >= x.thr_Dplus).to_numpy()[fl].mean()) if fl.any() else np.nan})
    return pd.DataFrame(rows), pd.DataFrame(tp)


# ---------------------------------------------------------------- main

def main():
    t0 = time.time()
    if torch.cuda.is_available():
        log("note: CUDA is visible; this analysis runs on the CPU only (set CUDA_VISIBLE_DEVICES= to hide it)")
    checks = []
    try:
        F = {yr: flags(yr) for yr in ("2017", "2018")}
        V = verify_flags(F, checks)
        V.to_csv(OUT / "sensitivity_flags.csv", index=False)
        log(f"flags verified [{time.time() - t0:.0f} s]")
        cache, S5, DEFT, PAIR, PCL, UN, NEGT, TPF = {}, [], [], [], [], [], [], []
        for yr in ("2017", "2018"):
            L, Tv0 = stage5_long(yr, F, cache)
            check_stage5_v0(yr, Tv0, checks)
            S5.append(L)
            log(f"{yr}: Stage 5 quantities done [{time.time() - t0:.0f} s]")
            R, _ = D.defender_rows(yr)
            negs = negatives(yr)
            T, Pd, PC = defender_quantities(yr, F, R, negs, checks)
            DEFT.append(T), PAIR.append(Pd), PCL.append(PC)
            UN.append(unions(yr, F, R, checks))
            log(f"{yr}: defender quantities and unions done [{time.time() - t0:.0f} s]")
            NN = recover_negatives(yr, negs, checks)
            nt, tp = negative_tables(yr, NN, R, F)
            NEGT.append(nt), TPF.append(tp)
            log(f"{yr}: negatives done [{time.time() - t0:.0f} s]")
            bad = [c for c in checks if not c["pass"]]
            if bad:
                raise StopRun("; ".join(f"{c['check']}: {c['value']}" for c in bad))
        outs = {"stage5": pd.concat(S5, ignore_index=True), "defenders": pd.concat(DEFT, ignore_index=True),
                "paired": pd.concat(PAIR, ignore_index=True), "per_class": pd.concat(PCL, ignore_index=True),
                "union": pd.concat(UN, ignore_index=True), "negatives": pd.concat(NEGT, ignore_index=True),
                "tpr_by_flag": pd.concat(TPF, ignore_index=True), "checks": pd.DataFrame(checks)}
        for k, v in outs.items():
            v.to_csv(OUT / f"sensitivity_{k}.csv", index=False)
        log(pd.DataFrame(checks).to_string(index=False))
        log(f"done in {time.time() - t0:.0f} s")
    except StopRun as e:
        pd.DataFrame(checks).to_csv(OUT / "sensitivity_checks.csv", index=False)
        log(f"STOPPED: {e}")
        sys.exit(2)


if __name__ == "__main__":
    main()
