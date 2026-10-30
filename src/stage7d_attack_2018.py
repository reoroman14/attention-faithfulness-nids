# Stage 7d: constrained attacks and faithfulness under attack on LycoS-Unicas-IDS2018 (Stage 4/5 equivalent).
# Replicates the final 2017 Stage 5 pipeline (src/stage5_fullrun.py, after the user-approved _clamp_std fix in
# src/stage4_constraints.py) with no methodological change and no tuning. No defenders.
#
# Imported unchanged: the attack settings and PGD (stage5_fullrun.run_setting / pgd2 / objective), the projection
# and the independent numpy validity checker (stage4_constraints), the evaluation (stage5_fullrun.evaluate: success,
# validity, raw perturbation, top-10 rollout / attn_mean mass ratio, explanation shift, faithfulness with 9 rankings
# incl. the static class-mean rankings) and the summary tables (stage5_fullrun.tables). Copied with 2018 paths:
# the per-seed setup, the clean-faithfulness part and the per-seed driver.
#
# What changes (data, checkpoints, class list), and what that forces:
#   source rows   Stage 7c explanation set (test rows correct for all 3 seeds), attack classes only, the first
#                 150 per class in the Stage 7c random order (rank_in_class), as in Stage 5. Exceptions:
#                 - webattack has 85 rows: all are used;
#                 - dos_slowloris: only rows whose exact (direct-difference) nearest-train distance is >= 1e-3;
#                 - every candidate row that violates a Stage 4 invariant on its clean values (any relation of
#                   stage4_constraints.relations, or a non-integral integer feature) is excluded.
#                 A class with fewer than 20 valid source rows is reported and left out.
#   budget floor  the rule is unchanged (B = eps |r|; eps x train median of non-zero |values| when r = 0); the
#                 median is taken on the 2018 train split. The spec CSV is otherwise used as is; the 2018 floors
#                 are passed to the projection and to the checker through an in-memory copy of the spec.
#   A1b targets   2017 TARGET_MAP where its target class exists on 2018; otherwise the nearest other attack class
#                 by Euclidean distance between class means of the 2018 train split in standardised space.
#   static        train class-mean vectors from Stage 7c (results/stage7/stage7c/raw/static_vectors_seed{s}.npz).
#
# Outputs: results/stage7/stage7d/seed{s}/ with the Stage 5 file layout (rows_*.parquet, faith_*.parquet,
# adv_raw_*.npz, clean_faith.parquet, summary_*.csv), summary_all_seeds_*.csv, stage7d_*.csv and report markdown.
# Resumable per seed and per setting (rows_*.parquet is the completion marker). PROGRESS.md is not written by the
# script.
# Stop conditions: any invalid constrained row (stops, writes stage7d_STOPPED.md); runtime estimate above 7 h
# (estimated from one setting on seed 0 before the rest of the run).
#
# Usage:
#   .venv\Scripts\python.exe -m src.stage7d_attack_2018
#   smoke test: --set stage7d.n_per_class=4 --set stage7d.steps=2 --set stage7d.out_dir=<scratch dir>
import copy
import json
import sys
import time
from datetime import datetime

import numpy as np
import pandas as pd
import torch

from src import config as cfgmod
from src.config import load_config
from src.preprocess import Preprocessor
from src.stage3_explain import load_ftt, ranking
from src.stage4_attack import STAGE4, TARGET_MAP, explain
from src.stage4_constraints import (CLASS_FAMILY, CONTROLS, FAMILY_FREEZE, TorchPreprocessor, check_validity,
                                    freeze_mask, load_spec_csv, preprocessing_equivalence, relations,
                                    zero_floor_scales)
from src.stage4b_followup import Setup, attn_vec, mass
from src.stage5_fullrun import (STAGE5, evaluate, faith, importances, keep_awake, load_seed, md, run_setting,
                                tables)
from src.utils import ROOT, finish_run, resolve, start_run

STAGE7D = {
    "out_dir": "results/stage7/stage7d",
    "min_rows": 20,
    "max_hours": 7.0,
    "neardup_threshold": 1e-3,
    # Setting timed on seed 0 for the runtime estimate (it is one of the run's own settings, kept by resume).
    "probe_setting": ["constrained", "A1", 0.5],
    # Relative attack cost of a setting vs one plain PGD setting (A3norm: two backward passes, as in Stage 5).
    "a3norm_cost": 1.4,
}
D7 = ROOT / "results" / "stage7"
C7 = D7 / "stage7c" / "raw"
BENIGN = "benign"
COMMON = ["bot", "ddos", "dos_goldeneye", "dos_hulk", "dos_slowloris", "ssh_patator", "webattack"]
SLOW = "dos_slowloris"


class StopRun(Exception):
    pass


# ---------------------------------------------------------------- one-off preparation (all seeds)

def a1b_targets(classes, Ztr, ytr):
    # 2017 map where the target exists; else nearest other attack class by class-mean distance (standardised).
    mu = {c: Ztr[ytr == k].astype(np.float64).mean(0) for k, c in enumerate(classes)}
    rows = []
    for c in classes:
        if c == BENIGN:
            continue
        t17 = TARGET_MAP.get(c)
        d = {o: float(np.linalg.norm(mu[c] - mu[o])) for o in classes if o not in (c, BENIGN)}
        if t17 in classes:
            t, rule = t17, "2017 target (exists on 2018)"
        else:
            t, rule = min(d, key=d.get), f"2017 target {t17} absent: nearest attack class by class-mean distance"
        rows.append({"class": c, "target_2017": t17, "target_2018": t, "rule": rule,
                     "class_mean_distance": d[t],
                     "distances": "; ".join(f"{o} {v:.2f}" for o, v in sorted(d.items(), key=lambda kv: kv[1]))})
    return pd.DataFrame(rows)


def source_rows(c, feats, classes, spec):
    # Candidate rows = attack-class rows of the Stage 7c selection. Returns (pos, per-class counts, per-relation
    # violation counts). pos = positions in the Stage 7c selection, sorted by (label, rank_in_class).
    sel = pd.read_parquet(C7 / "selection.parquet")
    test = pd.read_parquet(D7 / "data" / "test.parquet", columns=feats)
    cand = np.flatnonzero((sel["class"] != BENIGN).to_numpy())
    R = test.iloc[sel["test_row"].to_numpy()[cand]].to_numpy(np.float64)
    rel = relations(R, R, feats)
    integer = (spec.set_index("feature").loc[feats, "integer"] == "yes").to_numpy()
    nonint = (integer[None, :] & ~np.isclose(R, np.round(R), atol=1e-6)).any(1)
    inv_bad = nonint.copy()
    for v in rel.values():
        inv_bad |= ~v
    cls = sel["class"].to_numpy()[cand]
    rank = sel["rank_in_class"].to_numpy()[cand]
    nd = sel["neardup_lt_1e-3"].to_numpy()[cand].astype(bool)
    neardup_excl = (cls == SLOW) & nd
    ok = ~inv_bad & ~neardup_excl
    pos, counts, viol = [], [], []
    for cn in [k for k in classes if k != BENIGN]:
        m = cls == cn
        elig = np.flatnonzero(m & ok)
        take = elig[np.argsort(rank[elig], kind="stable")][:c["n_per_class"]]
        n_used = len(take) if len(take) >= c["min_rows"] else 0
        if n_used:
            pos.append(cand[take])
        counts.append({"class": cn, "selection_rows": int(m.sum()),
                       "excluded_neardup_lt_1e-3": int((m & neardup_excl).sum()),
                       "invariant_violators_all_candidates": int((m & inv_bad).sum()),
                       "invariant_violators_after_neardup_rule": int((m & inv_bad & ~neardup_excl).sum()),
                       "eligible": int(len(elig)), "used": n_used,
                       "below_min_rows": len(take) < c["min_rows"], "flag": "n < 100" if 0 < n_used < 100 else ""})
        viol.append({"class": cn, **{k: int((m & ~v).sum()) for k, v in rel.items()},
                     "integer_features_nonintegral": int((m & nonint).sum())})
    pos = np.concatenate(pos)
    lab = sel["label_id"].to_numpy()[pos]
    pos = pos[np.lexsort((sel["rank_in_class"].to_numpy()[pos], lab))]
    return pos, pd.DataFrame(counts), pd.DataFrame(viol)


# ---------------------------------------------------------------- per-seed setup (copy of stage4b setup + stage5
# seed_setup with 2018 paths)

def seed_setup7(c, dev, seed, pos, spec, floors, tmap):
    S = Setup()
    S.feats = json.loads((D7 / "data" / "features.json").read_text(encoding="utf-8"))
    lm = json.loads((D7 / "data" / "label_map.json").read_text(encoding="utf-8"))
    S.classes = [k for k, _ in sorted(lm.items(), key=lambda kv: kv[1])]
    S.pp = Preprocessor.load(D7 / "preprocess.json")
    S.tp = TorchPreprocessor(S.pp, dev)
    S.spec = spec
    S.floors = floors
    S.ctrl_mask = np.isin(S.feats, CONTROLS)
    S.idx = {f: k for k, f in enumerate(S.feats)}
    S.sel = pd.read_parquet(C7 / "selection.parquet")
    S.pos = pos
    S.y_np = S.sel["label_id"].to_numpy()[S.pos].astype(np.int64)
    S.cls_rows = [S.classes[k] for k in S.y_np]
    test = pd.read_parquet(D7 / "data" / "test.parquet", columns=S.feats)
    S.R = test.iloc[S.sel["test_row"].to_numpy()[S.pos]].to_numpy(np.float64).copy()
    del test
    with torch.no_grad():
        S.z0_64 = S.tp(torch.from_numpy(S.R).to(dev))
    S.Z0 = S.z0_64.float()
    _, _, cf = check_validity(S.R, S.R, S.feats, S.spec, 0.0, S.cls_rows)
    S.clean_fail = cf
    S.frozen = freeze_mask(S.R, S.feats, S.cls_rows, cf)
    S.y = torch.from_numpy(S.y_np).to(dev)
    S.t = torch.from_numpy(np.array([S.classes.index(tmap[cn]) for cn in S.cls_rows])).to(dev)
    S.model, _ = load_ftt(D7 / f"seed{seed}" / "ftt.pt", dev)
    for p in S.model.parameters():
        p.requires_grad_(False)
    ig = np.load(C7 / f"attr_seed{seed}.npz")["ig_zero"][S.pos].astype(np.float32)
    order = ranking(np.abs(ig), np.random.default_rng(c["rng_seed"] + 11))
    S.top = torch.from_numpy(order[:, :c["a2_k"]].copy()).to(dev)
    S.bot = torch.from_numpy(order[:, -c["a2_k"]:].copy()).to(dev)
    with torch.no_grad():
        _, at0 = S.model(S.Z0, return_attn=True)
    S.mass0 = {w: mass(attn_vec(at0, w), S.top).cpu().numpy() for w in ("rollout", "mean")}
    S.dev = dev
    S.seed = seed
    S.floor_arr = np.array([S.floors[f] for f in S.feats])
    sv = np.load(C7 / f"static_vectors_seed{seed}.npz")
    S.static_roll, S.static_ig = sv["static_rollout"][S.y_np], sv["static_ig"][S.y_np]
    brow = np.load(C7 / "baseline_rows.npz")["mask_benign_train_rows"][S.pos]
    train = pd.read_parquet(D7 / "data" / "train.parquet", columns=S.feats)
    S.bases = {"train_mean": torch.zeros_like(S.Z0),
               "benign_sample": torch.from_numpy(S.pp.transform(train.iloc[brow])).to(dev)}
    del train
    return S


def clean_part7(S, c, sdir, log):
    # Copy of stage5_fullrun.clean_part with the Stage 7c attribution store.
    f = sdir / "clean_faith.parquet"
    st = np.load(C7 / f"attr_seed{S.seed}.npz")
    expl, _ = explain(S.model, S.Z0, S.y, c)
    expl["ig_zero"] = st["ig_zero"][S.pos].astype(np.float32)
    expl["occlusion"] = st["occlusion"][S.pos].astype(np.float32)
    S.clean_sc = importances(S, expl, S.Z0.cpu().numpy(), np.random.default_rng(c["rng_seed"] + S.seed))
    S.clean_orders = {k: ranking(v, np.random.default_rng(c["rng_seed"] + 7)) for k, v in S.clean_sc.items()}
    if not f.exists():
        fa = faith(S, c, S.Z0, S.y, S.clean_sc, c["clean_mask_baselines"], np.random.default_rng(c["rng_seed"] + 13))
        fa.to_parquet(f, index=False)
        log.info(f"seed {S.seed}: clean faithfulness written")


def do_setting(S, c, sdir, mode, atk, eps, log):
    # Copy of the stage5_fullrun.run_seed inner block, plus the invalid-row stop condition.
    tag = f"{mode}_{atk}_eps{eps:g}"
    if (sdir / f"rows_{tag}.parquet").exists():
        return None
    t0 = time.time()
    torch.cuda.reset_peak_memory_stats()
    kind, z, z_last, X = run_setting(S, c, mode, atk, eps)
    t_att = time.time() - t0
    rows, fa = evaluate(S, c, kind, mode, eps, z, z_last, X,
                        np.random.default_rng(c["rng_seed"] + 17 + S.seed + int(eps * 1000)))
    info = {"seed": S.seed, "mode": mode, "attack": atk, "eps": eps}
    for k, v in info.items():
        rows.insert(0, k, v)
        fa.insert(0, k, v)
    rows["t_attack_s"], rows["t_eval_s"] = t_att, time.time() - t0 - t_att
    rows["peak_gpu_gb"] = torch.cuda.max_memory_allocated() / 2**30
    if mode == "constrained" and not rows["valid"].all():
        bad = rows[~rows["valid"]]
        vcols = [k for k in rows.columns if k.startswith("viol_") and bad[k].any()]
        bad.to_parquet(sdir / f"INVALID_rows_{tag}.parquet", index=False)
        np.savez_compressed(sdir / f"INVALID_adv_raw_{tag}.npz", x_adv=X, pos=S.pos)
        raise StopRun(f"seed {S.seed} {tag}: {len(bad)} invalid constrained rows (positions "
                      f"{bad['pos'].tolist()[:20]}); failing checks: {vcols}")
    fa.to_parquet(sdir / f"faith_{tag}.parquet", index=False)
    rows.to_parquet(sdir / f"rows_{tag}.parquet", index=False)        # written last = completion marker
    if mode == "constrained":
        np.savez_compressed(sdir / f"adv_raw_{tag}.npz", x_adv=X, pos=S.pos)
    log.info(f"seed {S.seed} {tag}: success {rows['success'].mean():.3f} valid {rows['valid'].mean():.3f} "
             f"mass ratio {rows['mass_ratio_rollout'].mean():.3f} [{t_att:.0f}s + {time.time() - t0 - t_att:.0f}s]")
    return rows


# ---------------------------------------------------------------- reporting helpers

def cell(vals):
    v = np.asarray(vals, dtype=float)
    v = v[~np.isnan(v)]
    if not len(v):
        return ""
    if len(v) == 1:
        return f"{v[0]:.3f}"
    return f"{v.mean():.3f} ({v.min():.3f} to {v.max():.3f})"


def spread(dfs, keys):
    # Per-seed tables with identical keys -> one table of "mean (min to max over seeds)" strings.
    d = pd.concat([x.assign(_seed=i) for i, x in enumerate(dfs)], ignore_index=True)
    vals = [k for k in d.columns if k not in keys and k != "_seed" and pd.api.types.is_numeric_dtype(d[k])]
    out = []
    for k, g in d.groupby(keys, sort=False, dropna=False):
        k = k if isinstance(k, tuple) else (k,)
        out.append({**dict(zip(keys, k)), **{v: cell(g[v].tolist()) for v in vals}})
    return pd.DataFrame(out)


def wide(df, index, col, val):
    return df.pivot_table(index=index, columns=col, values=val, aggfunc="first").reset_index()


def a1_minus_a0(T):
    s = T["success"]
    s = s[s["mode"] == "constrained"].pivot_table(index="attack", columns="eps", values="success")
    return pd.DataFrame([{"row": "A1 minus A0", **(s.loc["A1"] - s.loc["A0"]).to_dict()}])


def constraint_cost(T):
    # Constrained A1 / A2 (relative raw budget eps) next to the unconstrained reference (standardised eps).
    s = T["success"]
    return s[s.attack.isin(["A1", "A2", "A3_lam1000"])][["mode", "attack", "eps", "success", "validity",
                                                         "mass_ratio_rollout", "rel_pert_ctrl", "n_ctrl_changed"]]


def report7(Ts, title, c, extra_top=""):
    # Stage 5 report tables; with several seeds every number is "mean (min to max over seeds)".
    H = c["headline_max_eps"]
    keys_s = ["mode", "attack", "eps"]
    S_ = spread([T["success"].drop(columns=["success_sd_seeds"]) for T in Ts], keys_s)
    cons = S_[(S_["mode"] == "constrained") & (S_.eps <= H)]
    out = [f"\n---\n\n## {datetime.now():%Y-%m-%d %H:%M}: {title}\n", extra_top]
    out.append(f"Macro over the attack classes present (7 on 2018). Seeds: {len(Ts)}"
               f"{'; every cell is mean (min to max over seeds)' if len(Ts) > 1 else ''}. Headline tables use "
               f"eps <= {H}; eps 2.0 is the stress setting. Unconstrained rows are reference only, NOT protocol-valid "
               "(standardised eps).\n")
    out.append("**Success rate (constrained)**\n\n" + md(wide(cons, "attack", "eps", "success")))
    out.append("\n**Validity (constrained, independent checker)**\n\n" + md(wide(cons, "attack", "eps", "validity")))
    d = spread([a1_minus_a0(T) for T in Ts], ["row"])
    out.append("\n**A1 minus A0 (success)**\n\n" + md(d[["row"] + [e for e in d.columns if e != "row" and e <= H]]))
    out.append("\n**Top-10 rollout mass ratio adv/clean, best iterate (primary for A2)**\n\n"
               + md(wide(cons, "attack", "eps", "mass_ratio_rollout"))
               + "\n\nLast PGD iterate:\n\n" + md(wide(cons, "attack", "eps", "mass_ratio_rollout_last"))
               + "\n\nattn_mean mass ratio (best iterate):\n\n" + md(wide(cons, "attack", "eps", "mass_ratio_attn_mean"))
               + "\n\nA2 secondary flag (>= 50% rollout mass drop, prediction kept with margin >= 1):\n\n"
               + md(wide(cons[cons.attack == "A2"], "attack", "eps", "a2_drop50_flag")))
    un = S_[S_["mode"] == "unconstrained"]
    out.append("\n**Unconstrained reference (success | mass ratio | validity)**\n\n"
               + md(un[["attack", "eps", "success", "mass_ratio_rollout", "mass_ratio_rollout_last", "validity"]]))
    cc = spread([constraint_cost(T) for T in Ts], keys_s)
    out.append("\n**Constraint cost: constrained (relative raw budget) vs unconstrained reference (standardised eps)**"
               "\n\n" + md(cc))
    pc = spread([T["a1_per_class"] for T in Ts], ["class"])
    out.append("\n**A1 success per class (constrained)**\n\n"
               + md(pc[["class"] + [e for e in pc.columns if e != "class" and e <= H]]))
    out.append("\n**A1b destinations**\n\n" + md(spread([T["a1b_dest"] for T in Ts], ["mode", "eps"])))
    sh = spread([T["shift"] for T in Ts], keys_s)
    out.append("\n**Explanation shift, Spearman clean vs adversarial (eps 0.5 and 1.0)**\n\n"
               + md(sh[sh.eps.isin([0.5, 1.0]) | ((sh["mode"] == "unconstrained") & (sh.eps == 0.5))]))
    ex = spread([T["faith_excess_vs_A0"] for T in Ts], ["mode", "attack", "eps", "subset", "method"])
    meths = c["adv_rankings"]
    for title_, atk, sub, cols in (("A1 all rows (A1 minus A0), probability metrics", "A1", "all",
                                    ["del_auc", "aopc_gap", "comp@10", "suff@10"]),
                                   ("A1 flipped rows (primary: margin metrics)", "A1", "flipped",
                                    ["del_auc_margin", "aopc_gap_margin", "comp_margin@10", "suff_margin@10"]),
                                   ("A1 not-flipped rows (primary: probability metrics)", "A1", "not_flipped",
                                    ["del_auc", "aopc_gap", "comp@10", "suff@10"]),
                                   ("A2 all rows (prediction kept; probability metrics)", "A2", "all",
                                    ["del_auc", "aopc_gap", "comp@10", "suff@10"]),
                                   ("A3 lambda 1000 flipped rows (margin metrics)", "A3_lam1000", "flipped",
                                    ["del_auc_margin", "aopc_gap_margin"]),
                                   ("A3norm w 0.5 flipped rows (margin metrics)", "A3norm_w0.5", "flipped",
                                    ["del_auc_margin", "aopc_gap_margin"])):
        d = ex[(ex.attack == atk) & (ex.subset == sub) & (ex["mode"] == "constrained") & ex.eps.isin([0.5, 1.0])
               & ex.method.isin(meths)]
        if len(d):
            out.append(f"\n**Faithfulness change vs clean minus A0 change, {title_} (constrained, train-mean "
                       "masking, eps 0.5 / 1.0; n_row_obs per seed)**\n\n" + md(d[["eps", "method", "n_row_obs"] + cols]))
    out.append("\n**Clean faithfulness on these rows (reference)**\n\n"
               + md(spread([T["clean_faith"] for T in Ts], ["mask_baseline", "method"])))
    st = S_[(S_["mode"] == "constrained") & (S_.eps == 2.0)]
    out.append("\n**Stress setting eps 2.0 (constrained)**\n\n"
               + md(st[["attack", "success", "validity", "mass_ratio_rollout", "mass_ratio_rollout_last"]]))
    tm = [T["timing"] for T in Ts]
    out.append(f"\nRuntime per seed: attack {cell([t.t_attack_s.sum() / 60 for t in tm])} min, evaluation "
               f"{cell([t.t_eval_s.sum() / 60 for t in tm])} min; peak GPU {max(t.peak_gpu_gb.max() for t in tm):.2f} GB.\n")
    return "\n".join(out)


def frozen_share(R, feats, cls_rows, frozen):
    ctrl = np.isin(feats, CONTROLS)
    i = {f: k for k, f in enumerate(feats)}
    rows = []
    for cn in sorted(set(cls_rows), key=cls_rows.index):
        m = np.array(cls_rows) == cn
        fam = FAMILY_FREEZE.get(CLASS_FAMILY[cn], [])
        fam_mask = np.zeros(len(feats), bool)
        fam_mask[[i[f] for f in fam]] = True
        fz = frozen[m][:, ctrl]
        rows.append({"class": cn, "family": CLASS_FAMILY[cn], "n_rows": int(m.sum()), "n_controls": int(ctrl.sum()),
                     "family_frozen_controls": int(fam_mask[ctrl].sum()),
                     "share_controls_frozen_mean": float(fz.mean()),
                     "share_controls_frozen_other_reasons": float((fz & ~fam_mask[ctrl][None, :]).mean()),
                     "share_rows_any_frozen": float(fz.any(1).mean()),
                     "share_rows_all_frozen": float(fz.all(1).mean())})
    return pd.DataFrame(rows)


def frozen_2017(n_per_class):
    # Same quantities on the 2017 Stage 5 rows (stage4b_followup.setup row logic, no model needed).
    s1, s3 = resolve("results/stage1"), resolve("results/stage3")
    feats = json.loads((s1 / "features.json").read_text(encoding="utf-8"))
    lm = json.loads((s1 / "label_map.json").read_text(encoding="utf-8"))
    classes = [k for k, _ in sorted(lm.items(), key=lambda kv: kv[1])]
    sel = pd.read_parquet(s3 / "selection.parquet")
    pos = np.flatnonzero((sel["class"] != BENIGN).to_numpy() & (sel["rank_in_class"] < n_per_class).to_numpy())
    pos = pos[np.lexsort((sel["rank_in_class"].to_numpy()[pos], sel["label_id"].to_numpy()[pos]))]
    cls_rows = [classes[k] for k in sel["label_id"].to_numpy()[pos]]
    test = pd.read_parquet(s1 / "test.parquet", columns=feats)
    R = test.iloc[sel["test_row"].to_numpy()[pos]].to_numpy(np.float64)
    spec = load_spec_csv()
    _, _, cf = check_validity(R, R, feats, spec, 0.0, cls_rows)
    fr = freeze_mask(R, feats, cls_rows, cf)
    inv = np.zeros(len(R), bool)
    for v in cf.values():
        inv |= v
    out = frozen_share(R, feats, cls_rows, fr)
    out["clean_invariant_violators"] = [int(inv[np.array(cls_rows) == cn].sum()) for cn in out["class"]]
    return out


def common_tables(sdir_root, seeds, classes_keep, c):
    # Per-seed Stage 5 tables restricted to the given classes (for the 2017 vs 2018 comparison).
    Ts = []
    for s in seeds:
        rows, fa, clean = load_seed(sdir_root / f"seed{s}")
        rows = rows[rows["class"].isin(classes_keep)]
        keep = set(rows["pos"])
        fa, clean = fa[fa["pos"].isin(keep)], clean[clean["pos"].isin(keep)]
        Ts.append(tables(rows.assign(seed=s), fa.assign(seed=s), clean.assign(seed=s), None, c))
    return Ts


def key_quantities(Ts):
    # Headline quantities of the Stage 5 final reading, one value per seed.
    def succ(T, mode, atk, eps, col="success"):
        s = T["success"]
        v = s[(s["mode"] == mode) & (s.attack == atk) & (s.eps == eps)][col]
        return float(v.iloc[0]) if len(v) else np.nan

    def exc(T, atk, sub, eps, meth, col):
        e = T["faith_excess_vs_A0"]
        v = e[(e["mode"] == "constrained") & (e.attack == atk) & (e.subset == sub) & (e.eps == eps) & (e.method == meth)][col]
        return float(v.iloc[0]) if len(v) else np.nan

    def shift(T, atk, eps, meth):
        s = T["shift"]
        v = s[(s["mode"] == "constrained") & (s.attack == atk) & (s.eps == eps)][meth]
        return float(v.iloc[0]) if len(v) else np.nan

    def dest(T, eps, sub):
        d = T["a1b_dest"]
        v = d[(d["mode"] == "constrained") & (d.eps == eps)]
        return float(v[sub].iloc[0]) if len(v) and sub in v else np.nan

    Q = []
    for e in (0.25, 0.5, 1.0):
        Q.append((f"A1 success, eps {e}", lambda T, e=e: succ(T, "constrained", "A1", e)))
        Q.append((f"A0 success, eps {e}", lambda T, e=e: succ(T, "constrained", "A0", e)))
        Q.append((f"A1 minus A0 success, eps {e}",
                  lambda T, e=e: succ(T, "constrained", "A1", e) - succ(T, "constrained", "A0", e)))
    Q.append(("A1 success, eps 2.0 (stress)", lambda T: succ(T, "constrained", "A1", 2.0)))
    Q.append(("A1b reaches target, eps 1.0", lambda T: dest(T, 1.0, "to_target")))
    Q.append(("A1b goes to benign, eps 1.0", lambda T: dest(T, 1.0, "to_benign")))
    for e in (0.25, 0.5, 1.0):
        Q.append((f"A2 top-10 rollout mass ratio, best iterate, eps {e}",
                  lambda T, e=e: succ(T, "constrained", "A2", e, "mass_ratio_rollout")))
    Q.append(("A2 attn_mean mass ratio, eps 1.0", lambda T: succ(T, "constrained", "A2", 1.0, "mass_ratio_attn_mean")))
    Q.append(("A2 drop >= 50% flag, eps 1.0", lambda T: succ(T, "constrained", "A2", 1.0, "a2_drop50_flag")))
    Q.append(("A3 lambda 1000 mass ratio, eps 1.0",
              lambda T: succ(T, "constrained", "A3_lam1000", 1.0, "mass_ratio_rollout")))
    Q.append(("A3 lambda 1000 success, eps 1.0", lambda T: succ(T, "constrained", "A3_lam1000", 1.0)))
    Q.append(("A3 lambda 1000 minus A1 success, eps 1.0",
              lambda T: succ(T, "constrained", "A3_lam1000", 1.0) - succ(T, "constrained", "A1", 1.0)))
    Q.append(("A3norm w 0.5 mass ratio, eps 1.0",
              lambda T: succ(T, "constrained", "A3norm_w0.5", 1.0, "mass_ratio_rollout")))
    Q.append(("unconstrained A1 success, std eps 0.5", lambda T: succ(T, "unconstrained", "A1", 0.5)))
    Q.append(("unconstrained A2 mass ratio, std eps 0.5",
              lambda T: succ(T, "unconstrained", "A2", 0.5, "mass_ratio_rollout")))
    Q.append(("unconstrained A1 validity, std eps 0.5", lambda T: succ(T, "unconstrained", "A1", 0.5, "validity")))
    for m in ("attn_rollout", "attn_mean", "ig_zero", "occlusion", "attn_last"):
        Q.append((f"shift Spearman under A1 eps 1.0: {m}", lambda T, m=m: shift(T, "A1", 1.0, m)))
        Q.append((f"shift Spearman under A0 eps 1.0: {m}", lambda T, m=m: shift(T, "A0", 1.0, m)))
    for m in ("ig_zero", "occlusion", "static_ig", "attn_rollout", "attn_mean", "abs_x", "static_rollout", "random"):
        Q.append((f"A1 flipped, eps 1.0, margin AOPC gap minus A0: {m}",
                  lambda T, m=m: exc(T, "A1", "flipped", 1.0, m, "aopc_gap_margin")))
    for m in ("attn_rollout", "ig_zero", "occlusion", "static_rollout"):
        Q.append((f"A1 not-flipped, eps 1.0, del AUC minus A0: {m}",
                  lambda T, m=m: exc(T, "A1", "not_flipped", 1.0, m, "del_auc")))
        Q.append((f"A2 all rows, eps 1.0, del AUC minus A0: {m}",
                  lambda T, m=m: exc(T, "A2", "all", 1.0, m, "del_auc")))
    return Q


def compare(T17, T18):
    rows = []
    for (lab, f) in key_quantities(T17):
        a = np.array([f(T) for T in T17], float)
        b = np.array([f(T) for T in T18], float)
        ma, mb = np.nanmean(a) if np.isfinite(a).any() else np.nan, np.nanmean(b) if np.isfinite(b).any() else np.nan
        rows.append({"quantity": lab, "mean_2017": ma, "seed_range_2017": cell(a).split(" (")[-1].rstrip(")"),
                     "mean_2018": mb, "seed_range_2018": cell(b).split(" (")[-1].rstrip(")"),
                     "same_sign": bool(np.sign(ma) == np.sign(mb)), "ratio_2018_over_2017": mb / ma if ma else np.nan})
    return pd.DataFrame(rows)


def per_class_compare(T17, T18, eps_list=(0.5, 1.0)):
    out = []
    for yr, Ts in (("2017", T17), ("2018", T18)):
        d = pd.concat([T["a1_per_class"].assign(_s=i) for i, T in enumerate(Ts)])
        for cn, g in d.groupby("class"):
            for e in eps_list:
                out.append({"class": cn, "eps": e, "year": yr, "value": cell(g[e].tolist()) if e in g else ""})
    p = pd.DataFrame(out).pivot_table(index=["class", "eps"], columns="year", values="value", aggfunc="first")
    return p.reset_index()


# ---------------------------------------------------------------- main

def main():
    cfgmod.DEFAULTS.setdefault("stage7d", {**copy.deepcopy(STAGE4), **copy.deepcopy(STAGE5), **copy.deepcopy(STAGE7D)})
    cfg, overrides = load_config("Stage 7d: Stage 4/5 constrained attacks on LycoS-Unicas-IDS2018.")
    c = cfg["stage7d"]
    out = resolve(c["out_dir"])
    out.mkdir(parents=True, exist_ok=True)
    t_start = time.time()
    run_dir, log = start_run("stage7d_attack_2018", cfg, overrides)
    dev = torch.device("cuda")
    keep_awake(True)
    sanity = []

    def sane(name, seed, value, expected, passed):
        sanity.append({"check": name, "seed": seed, "value": value, "expected": expected, "pass": bool(passed)})
        log.info(f"[sanity] seed={seed} {name}: {value} (expected {expected}) -> {'PASS' if passed else 'FAIL'}")

    try:
        feats = json.loads((D7 / "data" / "features.json").read_text(encoding="utf-8"))
        lm = json.loads((D7 / "data" / "label_map.json").read_text(encoding="utf-8"))
        classes = [k for k, _ in sorted(lm.items(), key=lambda kv: kv[1])]
        spec17 = load_spec_csv()
        assert set(spec17["feature"]) == set(feats)
        fam = {cn: CLASS_FAMILY[cn] for cn in classes}
        log.info(f"class families: {fam}")
        # ---- 2018 budget floors (same rule, 2018 train split) and A1b targets
        pp = Preprocessor.load(D7 / "preprocess.json")
        train = pd.read_parquet(D7 / "data" / "train.parquet", columns=feats + ["label_id"])
        floors = zero_floor_scales(train, feats)
        spec = spec17.copy()
        spec["zero_floor_scale"] = spec["feature"].map(floors)
        fl = spec17[["feature", "role"]].assign(zero_floor_scale_2017=spec17["zero_floor_scale"].astype(float),
                                                zero_floor_scale_2018=spec["zero_floor_scale"].astype(float))
        fl["ratio_2018_over_2017"] = fl["zero_floor_scale_2018"] / fl["zero_floor_scale_2017"]
        fl.to_csv(out / "stage7d_budget_floors.csv", index=False)
        spec.to_csv(out / "stage7d_constraints_2018_floors.csv", index=False)
        tm = a1b_targets(classes, pp.transform(train), train["label_id"].to_numpy())
        tm.to_csv(out / "stage7d_a1b_targets.csv", index=False)
        tmap = dict(zip(tm["class"], tm["target_2018"]))
        log.info(f"A1b targets 2018: {tmap}")
        del train
        # ---- source rows
        pos, counts, viol = source_rows(c, feats, classes, spec)
        counts.to_csv(out / "stage7d_source_counts.csv", index=False)
        viol.to_csv(out / "stage7d_source_invariant_violations.csv", index=False)
        log.info(f"source rows:\n{counts.to_string(index=False)}")
        small = counts.loc[counts.below_min_rows, "class"].tolist()
        if small:
            log.error(f"STOP CONDITION (class level): fewer than {c['min_rows']} valid source rows for {small}; "
                      "these classes are left out, the others continue")
        np.save(out / "stage7d_source_positions.npy", pos)

        seeds = c["seeds"]
        probe = tuple(c["probe_setting"])
        settings = [(m, a, e) for m, a, el in c["settings"] for e in el]
        for si, s in enumerate(seeds):
            sdir = out / f"seed{s}"
            sdir.mkdir(parents=True, exist_ok=True)
            ts = time.time()
            S = seed_setup7(c, dev, s, pos, spec, floors, tmap)
            with torch.no_grad():
                ok = (S.model(S.Z0).argmax(1) == S.y).cpu().numpy()
            sane("clean source rows predicted as label (through the torch preprocessing)", s, f"{ok.sum()} / {len(ok)}",
                 "all", ok.all())
            eq = preprocessing_equivalence(S.pp, S.tp, S.model, S.R, dev)
            sane("preprocessing equivalence numpy vs torch (max |diff| float64, argmax agreement)", s,
                 f"{eq['max_abs_diff_float64']:.1e}, {eq['argmax_agreement']:.4f}", "about 0, 1.0",
                 eq["argmax_agreement"] == 1.0 and eq["max_abs_diff_float64"] < 1e-6)
            v0, _, _ = check_validity(S.R, S.R, S.feats, S.spec, 0.0, S.cls_rows)
            nfail = sum(int(v.sum()) for v in S.clean_fail.values())
            sane("clean source rows valid under the checker at eps 0 (and no clean relation failures)", s,
                 f"{int(v0.sum())} / {len(v0)}; clean relation failures {nfail}", "all; 0", v0.all() and nfail == 0)
            if si == 0:
                fz = frozen_share(S.R, S.feats, S.cls_rows, S.frozen)
                fz.to_csv(out / "stage7d_frozen_share.csv", index=False)
            t_clean0 = time.time()
            clean_part7(S, c, sdir, log)
            t_clean = time.time() - t_clean0
            order = settings
            if si == 0 and probe in settings:
                order = [probe] + [x for x in settings if x != probe]
            for mode, atk, eps in order:
                r = do_setting(S, c, sdir, mode, atk, eps, log)
                if si == 0 and (mode, atk, eps) == probe:
                    pf = sdir / f"rows_{mode}_{atk}_eps{eps:g}.parquet"
                    rr = r if r is not None else pd.read_parquet(pf)
                    t_att, t_ev = float(rr["t_attack_s"].iloc[0]), float(rr["t_eval_s"].iloc[0])
                    n_pgd = sum(1 for m_, a_, _ in settings if not (m_ == "constrained" and a_ == "A0")
                                and not a_.startswith("A3norm"))
                    n_norm = sum(1 for _, a_, _ in settings if a_.startswith("A3norm"))
                    per_seed = t_clean + (n_pgd + c["a3norm_cost"] * n_norm) * t_att + len(settings) * t_ev
                    est_h = (len(seeds) * per_seed + (time.time() - t_start) - t_clean - t_att - t_ev) / 3600
                    msg = (f"RUNTIME ESTIMATE from {mode} {atk} eps {eps:g} on seed {s}: attack {t_att:.0f} s, "
                           f"evaluation {t_ev:.0f} s per setting, clean part {t_clean:.0f} s; {len(settings)} settings "
                           f"per seed ({n_pgd} PGD, {n_norm} A3norm at {c['a3norm_cost']}x); "
                           f"full run about {est_h:.2f} h")
                    log.info(msg)
                    print(msg, flush=True)
                    (out / "stage7d_runtime_estimate.txt").write_text(msg + "\n", encoding="utf-8")
                    sane("runtime estimate for the full run (h)", s, f"{est_h:.2f}", f"<= {c['max_hours']}",
                         est_h <= c["max_hours"])
                    if est_h > c["max_hours"]:
                        raise StopRun(msg)
            rows, fa, clean = load_seed(sdir)
            T = tables(rows.assign(seed=s), fa.assign(seed=s), clean.assign(seed=s), classes, c)
            for k, v in T.items():
                v.to_csv(sdir / f"summary_{k}.csv", index=False)
            (sdir / "seed_time_s.txt").write_text(f"{time.time() - ts:.1f}\n", encoding="utf-8")
            log.info(f"seed {s} finished ({(time.time() - ts) / 60:.1f} min this session)")
            del S
            torch.cuda.empty_cache()

        # ---- pooled summaries (Stage 5 layout) and the reports
        R_, F_, C_ = [], [], []
        for s in seeds:
            r, f, cl = load_seed(out / f"seed{s}")
            R_.append(r), F_.append(f), C_.append(cl.assign(seed=s))
        Tall = tables(pd.concat(R_, ignore_index=True), pd.concat(F_, ignore_index=True),
                      pd.concat(C_, ignore_index=True), classes, c)
        for k, v in Tall.items():
            v.to_csv(out / f"summary_all_seeds_{k}.csv", index=False)
        T18 = [tables(r.assign(seed=s), f.assign(seed=s), cl, classes, c) for s, r, f, cl in zip(seeds, R_, F_, C_)]
        T17 = common_tables(resolve("results/stage5"), seeds, COMMON, c)
        cmp = compare(T17, T18)
        cmp.to_csv(out / "stage7d_cmp2017_key_quantities.csv", index=False)
        pcc = per_class_compare(T17, T18)
        pcc.to_csv(out / "stage7d_cmp2017_a1_per_class.csv", index=False)
        fz17 = frozen_2017(c["n_per_class"])
        fz17.to_csv(out / "stage7d_frozen_share_2017.csv", index=False)
        fz = pd.read_csv(out / "stage7d_frozen_share.csv")
        # 2017 Stage 5 tables on the 7 common classes, same report layout (for side-by-side reading)
        for s, T in zip(seeds, T17):
            for k, v in T.items():
                v.to_csv(out / f"cmp2017_seed{s}_{k}.csv", index=False)
        n_cls = R_[0][(R_[0].attack == "A0") & (R_[0].eps == 0.05)].groupby("class").size().rename("n_rows").reset_index()
        n_cls["flag"] = np.where(n_cls.n_rows < 100, "n < 100", "")
        top = (f"**Source rows per class** (identical for every seed)\n\n{md(counts)}\n\n"
               f"Rows per class in every setting:\n\n{md(n_cls)}\n\n"
               f"Clean invariant violations among the candidate rows, per relation:\n\n{md(viol)}\n\n"
               f"**A1b targets**\n\n{md(tm[['class', 'target_2017', 'target_2018', 'rule', 'class_mean_distance']])}\n\n"
               f"**Frozen share by class (2018 source rows)**\n\n{md(fz)}\n\n"
               f"2017 Stage 5 rows (common classes shown with the others):\n\n{md(fz17)}\n\n"
               f"**Budget floors (zero_floor_scale), controls only, 2017 vs 2018 train median**\n\n"
               f"{md(fl[fl.role == 'control'][['feature', 'zero_floor_scale_2017', 'zero_floor_scale_2018', 'ratio_2018_over_2017']], '{:.4g}')}\n\n"
               f"**Sanity checks**\n\n{md(pd.DataFrame(sanity))}\n")
        for s, T in zip(seeds, T18):
            (out / f"stage7d_report_seed{s}.md").write_text(report7([T], f"Stage 7d (2018): seed {s}", c),
                                                            encoding="utf-8")
        pooled = report7(T18, "Stage 7d (2018): all seeds, pooled", c, top)
        cmp_txt = (f"\n**2017 vs 2018, the 7 common classes** (2017 recomputed from results/stage5/seed*/ on the same "
                   f"7 classes; mean over seeds and per-seed range)\n\n{md(cmp)}\n\n"
                   f"A1 success per class, mean (min to max over seeds):\n\n{md(pcc)}\n")
        (out / "stage7d_report_pooled.md").write_text(pooled + cmp_txt, encoding="utf-8")
        (out / "stage7d_report_2017_common7.md").write_text(
            report7(T17, "Stage 5 (2017) recomputed on the 7 common classes, all seeds", c), encoding="utf-8")
        pd.DataFrame(sanity).to_csv(out / "stage7d_sanity.csv", index=False)
        rt = pd.DataFrame([{"seed": s, "seed_time_s_last_session": float((out / f"seed{s}" / "seed_time_s.txt").read_text()),
                            "attack_s": T["timing"].t_attack_s.sum(), "eval_s": T["timing"].t_eval_s.sum(),
                            "peak_gpu_gb": T["timing"].peak_gpu_gb.max()} for s, T in zip(seeds, T18)])
        rt.to_csv(out / "stage7d_runtime.csv", index=False)
        log.info(f"done; reports in {out}")
    except StopRun as e:
        log.error(f"STOPPED: {e}")
        (out / "stage7d_STOPPED.md").write_text(f"## {datetime.now():%Y-%m-%d %H:%M}: Stage 7d stopped\n\n"
                                                f"Stop condition: {e}\n", encoding="utf-8")
        pd.DataFrame(sanity).to_csv(out / "stage7d_sanity.csv", index=False)
        finish_run(run_dir, log, {"stopped": str(e)}, t_start)
        keep_awake(False)
        sys.exit(2)
    finally:
        keep_awake(False)
    finish_run(run_dir, log, {"seeds": seeds}, t_start)


if __name__ == "__main__":
    main()
