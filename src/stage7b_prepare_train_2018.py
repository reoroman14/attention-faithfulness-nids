# Stage 7b: prepare LycoS-Unicas-IDS2018 and train the FT-Transformer (3 seeds) and XGBoost, with the Stage 2
# recipe unchanged (no tuning on 2018). No explanations, no attacks.
#
# Decisions (fixed by the user):
#   features   the 72 features of results/stage4_constraints.csv in that order; flag_SYN renamed to flag_syn in the
#              loaded frame only; dst_port, ip_prot, flag_urg, fwd_flag_urg, bwd_flag_urg dropped
#   cleaning   (1) remove every FTP-Patator and DoS Slowhttptest row (indistinguishable in the features);
#              (2) remove every remaining feature vector that carries more than one (merged) class label;
#              (3) drop exact duplicate rows on the 72 features, keeping the first in file order; all before any split
#   classes    DDoS LOIC-HTTP / HOIC / LOIC-UDP -> ddos; the three web attacks -> webattack; others as in 2017
#   cap        benign <= 1,000,000, each attack class <= 300,000, random with a fixed seed, after cleaning and dedup
#   split      stratified random 60 / 20 / 20, fixed seed (no timestamps exist)
#   model      Stage 2 preprocessing recipe refit on the 2018 train split; Stage 2 FT-T config, class weighting and
#              early stopping; seeds 0, 1, 2; Stage 2 XGBoost once
# Duplicates and conflicts use 64-bit row hashes on the float32 feature values; every hash collision group is
# verified value by value (as Stage 1 did).
# Stop conditions: test macro-F1 < 0.80 for any seed; a class with < 100 train rows; runtime estimate > 3 h.
# Resumable: data split, preprocessing, every seed and XGBoost are skipped if their outputs exist.
#
# Usage:
#   .venv\Scripts\python.exe -m src.stage7b_prepare_train_2018
import copy
import json
import sys
import time
from datetime import datetime

import numpy as np
import pandas as pd
import pyarrow.parquet as pq
import torch
from sklearn.model_selection import train_test_split

from src import config as cfgmod
from src.metrics import confusion_long, per_class_report
from src.preprocess import Preprocessor
from src.stage2_train import make_class_weights, predict_torch, predict_xgb, train_ftt, train_xgb
from src.stage4_constraints import CLASS_FAMILY, FAMILY_FREEZE, load_spec_csv
from src.stage5_fullrun import md
from src.utils import ROOT, finish_run, resolve, start_run

PQ = ROOT / "data" / "lycos-unicas-ids2018" / "parquet" / "LycoS-Unicas-IDS2018.parquet"
OUT = ROOT / "results" / "stage7"
DATA = OUT / "data"
SEED = 2026
SEEDS = [0, 1, 2]
REMOVE = ["FTP-Patator", "DoS Slowhttptest"]
MERGE = {"Benign": "benign", "Bot": "bot", "DDoS LOIC-HTTP": "ddos", "DDoS HOIC": "ddos", "DDoS LOIC-UDP": "ddos",
         "DoS GoldenEye": "dos_goldeneye", "DoS Hulk": "dos_hulk", "DoS Slowloris": "dos_slowloris",
         "SSH-Patator": "ssh_patator", "Web Attack - Brute Force": "webattack", "Web Attack - XSS": "webattack",
         "Web Attack - Sql Injection": "webattack", "FTP-Patator": "ftp_patator", "DoS Slowhttptest": "dos_slowhttptest"}
CAP_BENIGN, CAP_ATTACK = 1_000_000, 300_000
MIN_F1, MIN_TRAIN, MAX_HOURS = 0.80, 100, 3.0


class StopRun(Exception):
    pass


# ---------------------------------------------------------------- hashing with value-by-value verification

def row_hash(X):
    return pd.util.hash_pandas_object(pd.DataFrame(X), index=False).to_numpy()


def verified_groups(X, h):
    # Group ids for identical rows: 64-bit hash, then every rows-sharing-a-hash group is checked value by value.
    # Returns (group id per row, number of hash collisions found and split).
    order = np.argsort(h, kind="stable")
    hs = h[order]
    gid_sorted = np.cumsum(np.r_[True, hs[1:] != hs[:-1]]) - 1
    collisions = 0
    # Within each multi-row hash group, compare rows to the group's first row.
    starts = np.flatnonzero(np.r_[True, hs[1:] != hs[:-1]])
    sizes = np.diff(np.r_[starts, len(hs)])
    next_id = gid_sorted[-1] + 1 if len(hs) else 0
    for s, n in zip(starts[sizes > 1], sizes[sizes > 1]):
        idx = order[s:s + n]
        same = (X[idx] == X[idx[0]]).all(1)
        if not same.all():
            collisions += 1
            # Re-group exactly with a unique on the rows of this (rare) collision group.
            _, inv = np.unique(X[idx], axis=0, return_inverse=True)
            for k in np.unique(inv)[1:]:
                gid_sorted[s:s + n][inv == k] = next_id
                next_id += 1
    gid = np.empty_like(gid_sorted)
    gid[order] = gid_sorted
    return gid, collisions


# ---------------------------------------------------------------- data preparation

def prepare(log):
    DATA.mkdir(parents=True, exist_ok=True)
    if all((DATA / f"{s}.parquet").exists() for s in ("train", "val", "test")):
        return json.loads((DATA / "prep_summary.json").read_text(encoding="utf-8"))
    spec = load_spec_csv()
    feats = spec["feature"].tolist()
    src_cols = ["flag_SYN" if f == "flag_syn" else f for f in feats]
    t0 = time.time()
    tbl = pq.read_table(PQ, columns=src_cols + ["label"])
    X = np.column_stack([tbl.column(c).to_numpy().astype(np.float32) for c in src_cols])
    raw = tbl.column("label").to_pandas().astype(str).to_numpy()
    del tbl
    file_row = np.arange(len(raw))
    log.info(f"loaded {len(raw):,} rows x {X.shape[1]} features in {time.time() - t0:.0f} s")
    steps = []

    def count(step, lab_raw, keep):
        s = pd.Series(lab_raw[keep]).value_counts()
        for l_, n in s.items():
            steps.append({"step": step, "raw_label": l_, "class": MERGE[l_], "rows": int(n)})

    allrows = np.ones(len(raw), bool)
    count("1 raw", raw, allrows)
    keep = ~np.isin(raw, REMOVE)
    count("2 after removing FTP-Patator and DoS Slowhttptest", raw, keep)
    cls = np.array([MERGE[l_] for l_ in raw])
    # Conflicts: feature vectors (72 features) carrying more than one merged class label.
    idx = np.flatnonzero(keep)
    h = row_hash(X[idx])
    gid, coll1 = verified_groups(X[idx], h)
    lab_codes, lab_names = pd.factorize(cls[idx])
    pairs = np.unique(np.stack([gid, lab_codes], 1), axis=0)
    n_lab = np.bincount(pairs[:, 0], minlength=gid.max() + 1)
    conflict = n_lab[gid] > 1
    conf_groups = int((n_lab > 1).sum())
    keep2 = np.zeros(len(raw), bool)
    keep2[idx[~conflict]] = True
    count("3 after removing conflicting feature vectors", raw, keep2)
    # Exact duplicates on the 72 features: keep the first in file order.
    idx2 = np.flatnonzero(keep2)
    gid2 = gid[~conflict]
    first = np.zeros(len(idx2), bool)
    _, first_pos = np.unique(gid2, return_index=True)
    first[first_pos] = True
    keep3 = np.zeros(len(raw), bool)
    keep3[idx2[first]] = True
    count("4 after dropping exact duplicates", raw, keep3)
    # Cap per merged class (random, fixed seed).
    rng = np.random.default_rng(SEED)
    keep4 = np.zeros(len(raw), bool)
    for c in sorted(set(cls[keep3])):
        ids = np.flatnonzero(keep3 & (cls == c))
        cap = CAP_BENIGN if c == "benign" else CAP_ATTACK
        keep4[rng.choice(ids, size=cap, replace=False) if len(ids) > cap else ids] = True
    count("5 after cap", raw, keep4)
    # Split: stratified 60 / 20 / 20 on the merged class.
    ids = np.flatnonzero(keep4)
    classes = sorted(set(cls[ids]))
    label_map = {c: i for i, c in enumerate(classes)}
    y = np.array([label_map[c] for c in cls[ids]])
    tr, rest = train_test_split(ids, test_size=0.4, stratify=y, random_state=SEED)
    y_rest = np.array([label_map[c] for c in cls[rest]])
    va, te = train_test_split(rest, test_size=0.5, stratify=y_rest, random_state=SEED)
    for name, part in (("train", tr), ("val", va), ("test", te)):
        part = np.sort(part)
        df = pd.DataFrame(X[part], columns=feats)
        df["label_id"] = [label_map[c] for c in cls[part]]
        df["class"] = cls[part]
        df["raw_label"] = raw[part]
        df["file_row"] = file_row[part]
        df.to_parquet(DATA / f"{name}.parquet", index=False)
        count(f"6 split {name}", raw, np.isin(file_row, part))
    (DATA / "features.json").write_text(json.dumps(feats), encoding="utf-8")
    (DATA / "label_map.json").write_text(json.dumps(label_map, indent=2), encoding="utf-8")
    pd.DataFrame(steps).to_csv(OUT / "stage7b_counts_by_step.csv", index=False)
    summ = {"n_raw": int(len(raw)), "conflict_groups": conf_groups, "hash_collisions_split": coll1,
            "classes": classes, "seconds": round(time.time() - t0, 1)}
    (DATA / "prep_summary.json").write_text(json.dumps(summ, indent=2), encoding="utf-8")
    # File-order check (item 3), on the raw file in its original order.
    order_check(raw, X, feats, log)
    log.info(f"prepared splits in {time.time() - t0:.0f} s")
    return summ


def order_check(raw, X, feats, log):
    # Labels in contiguous blocks? Monotone columns? Report only.
    change = raw[1:] != raw[:-1]
    runs = pd.Series(raw).groupby((np.r_[True, change]).cumsum()).agg(["first", "size"])
    per = runs.groupby("first")["size"].agg(n_runs="size", longest_run="max", rows="sum").reset_index()
    p = pd.Series(raw).value_counts(normalize=True)
    # Expected number of runs if the file were shuffled: 1 + sum over adjacent pairs of P(labels differ).
    exp_changes = (len(raw) - 1) * (1 - float((p ** 2).sum()))
    per["longest_run_share_of_class"] = per["longest_run"] / per["rows"]
    mono = []
    for j, f in enumerate(feats):
        d = np.diff(X[:, j].astype(np.float64))
        mono.append({"feature": f, "share_non_decreasing_steps": float((d >= 0).mean()),
                     "share_strictly_increasing_steps": float((d > 0).mean())})
    mono = pd.DataFrame(mono).sort_values("share_strictly_increasing_steps", ascending=False)
    per.to_csv(OUT / "stage7b_order_label_runs.csv", index=False)
    mono.to_csv(OUT / "stage7b_order_monotone.csv", index=False)
    (OUT / "stage7b_order_summary.json").write_text(json.dumps(
        {"label_changes": int(change.sum()), "expected_label_changes_if_shuffled": exp_changes,
         "n_runs": int(len(runs))}, indent=2), encoding="utf-8")
    log.info(f"file order: {int(change.sum()):,} label changes (shuffled expectation {exp_changes:,.0f})")


# ---------------------------------------------------------------- leakage check

@torch.no_grad()
def nn_dist(Q, T, dev, qb=256, tb=262144):
    # Euclidean distance of each query row to its nearest train row (GPU, chunked).
    Tt = torch.from_numpy(T).to(dev)
    out = []
    for i in range(0, len(Q), qb):
        q = torch.from_numpy(Q[i:i + qb]).to(dev)
        best = torch.full((len(q),), float("inf"), device=dev)
        for j in range(0, len(Tt), tb):
            best = torch.minimum(best, torch.cdist(q, Tt[j:j + tb]).min(1).values)
        out.append(best.cpu().numpy())
    del Tt
    torch.cuda.empty_cache()
    return np.concatenate(out)


def leakage(name, Xtr, Xte, yte, classes, dev, rng, n=2000):
    rows = []
    for k, c in enumerate(classes):
        idx = np.flatnonzero(yte == k)
        if len(idx) == 0:
            continue
        pick = rng.choice(idx, size=min(n, len(idx)), replace=False)
        d = nn_dist(Xte[pick], Xtr, dev)
        rows.append({"dataset": name, "class": c, "n_test_rows": len(pick), "median": float(np.median(d)),
                     "p5": float(np.percentile(d, 5)), "share_exact_0": float((d == 0).mean()),
                     "share_below_0.01": float((d < 0.01).mean())})
    return rows


# ---------------------------------------------------------------- main

def main():
    cfg = copy.deepcopy(cfgmod.DEFAULTS)                      # Stage 2 settings, unchanged
    OUT.mkdir(parents=True, exist_ok=True)
    t_start = time.time()
    run_dir, log = start_run("stage7b_prepare_train_2018", cfg, [])
    dev = torch.device("cuda")
    entry = []
    try:
        # ---- class-family check (stage4_constraints, unchanged)
        fam = {c: CLASS_FAMILY.get(c, "MISSING") for c in sorted(set(MERGE.values()) - set(MERGE[r] for r in REMOVE))}
        fam_txt = ", ".join(f"{c} -> {f}" + (f" (frozen: {len(FAMILY_FREEZE[f])} controls)" if f in FAMILY_FREEZE else "")
                            for c, f in fam.items())
        if "MISSING" in fam.values():
            raise StopRun(f"class names missing from stage4_constraints.CLASS_FAMILY: {fam}")
        prep = prepare(log)
        feats = json.loads((DATA / "features.json").read_text(encoding="utf-8"))
        label_map = json.loads((DATA / "label_map.json").read_text(encoding="utf-8"))
        classes = [c for c, _ in sorted(label_map.items(), key=lambda kv: kv[1])]
        K = len(classes)
        data = {s: pd.read_parquet(DATA / f"{s}.parquet") for s in ("train", "val", "test")}
        ntr = data["train"]["label_id"].value_counts()
        small = [classes[k] for k in range(K) if ntr.get(k, 0) < MIN_TRAIN]
        if small:
            raise StopRun(f"classes with fewer than {MIN_TRAIN} train rows: {small}")
        # ---- preprocessing (refit on 2018 train, Stage 2 recipe)
        ppf = OUT / "preprocess.json"
        if ppf.exists():
            pp = Preprocessor.load(ppf)
        else:
            pp = Preprocessor.fit(data["train"], feats, cfg["stage2"]["preprocess"]["log1p_skew_threshold"],
                                  cfg["stage2"]["preprocess"]["clip"])
            pp.save(ppf)
        Xs = {s: pp.transform(data[s]) for s in data}
        y = {s: data[s]["label_id"].to_numpy().astype(np.int64) for s in data}
        # ---- runtime estimate (2017: 61 s per epoch on 906,981 train rows; at most 20 epochs per seed)
        est_h = (len(SEEDS) * cfg["stage2"]["ftt"]["max_epochs"] * 61 * len(y["train"]) / 906_981 + 15 * 60) / 3600
        log.info(f"RUNTIME ESTIMATE (upper bound, 20 epochs per seed): {est_h:.2f} h")
        if est_h > MAX_HOURS:
            raise StopRun(f"runtime estimate {est_h:.2f} h > {MAX_HOURS} h")
        class_w, counts = make_class_weights(y["train"], K, cfg["stage2"]["class_weight"])
        Xt = {s: torch.from_numpy(Xs[s]).to(dev) for s in Xs}
        yt = {s: torch.from_numpy(y[s]).to(dev) for s in y}
        meta = {"features": feats, "classes": classes, "preprocess_path": "results/stage7/preprocess.json",
                "config": cfg, "dataset": "LycoS-Unicas-IDS2018"}
        reps, confs, infos = [], [], []
        for s in SEEDS:
            sd = OUT / f"seed{s}"
            sd.mkdir(exist_ok=True)
            fi = sd / "info.json"
            if fi.exists():
                info = json.loads(fi.read_text(encoding="utf-8"))
                reps.append(pd.read_csv(sd / "per_class.csv"))
                confs.append(pd.read_csv(sd / "confusion.csv"))
            else:
                t0 = time.time()
                model, curves, info = train_ftt(s, Xt["train"], yt["train"], Xt["val"], yt["val"], K, class_w, None,
                                                cfg, log, sd / "ftt.pt", meta)
                info["train_time_s"] = round(time.time() - t0, 1)
                pd.DataFrame(curves).to_csv(sd / "curves.csv", index=False)
                rs, cs = [], []
                for split in ("val", "test"):
                    pred = predict_torch(model, Xt[split], cfg["stage2"]["ftt"]["eval_batch_size"]).argmax(1).cpu().numpy()
                    r = per_class_report(y[split], pred, classes).assign(model="ftt", seed=s, split=split)
                    cm, _ = confusion_long(y[split], pred, classes)
                    rs.append(r), cs.append(cm.assign(model="ftt", seed=s, split=split))
                    info[f"{split}_macro_f1"] = float(r.loc[r["class"] == "macro_avg", "f1"].iloc[0])
                pd.concat(rs).to_csv(sd / "per_class.csv", index=False)
                pd.concat(cs).to_csv(sd / "confusion.csv", index=False)
                fi.write_text(json.dumps(info, indent=2), encoding="utf-8")
                reps.append(pd.concat(rs)), confs.append(pd.concat(cs))
                del model
                torch.cuda.empty_cache()
            infos.append(info)
            log.info(f"seed {s}: test macro-F1 {info['test_macro_f1']:.4f} ({info.get('train_time_s', 0):.0f} s)")
            if info["test_macro_f1"] < MIN_F1:
                raise StopRun(f"seed {s} test macro-F1 {info['test_macro_f1']:.4f} < {MIN_F1}")
        # ---- XGBoost (once)
        fx = OUT / "xgb_info.json"
        if fx.exists():
            xinfo = json.loads(fx.read_text(encoding="utf-8"))
            reps.append(pd.read_csv(OUT / "xgb_per_class.csv"))
            confs.append(pd.read_csv(OUT / "xgb_confusion.csv"))
        else:
            bst, xinfo, curve = train_xgb(Xs["train"], y["train"], class_w[y["train"]], Xs["val"], y["val"], K, cfg, log)
            bst[: bst.best_iteration + 1].save_model(OUT / "xgb.json")
            rs, cs = [], []
            for split in ("val", "test"):
                pred = predict_xgb(bst, Xs[split])
                r = per_class_report(y[split], pred, classes).assign(model="xgb", seed=0, split=split)
                cm, _ = confusion_long(y[split], pred, classes)
                rs.append(r), cs.append(cm.assign(model="xgb", seed=0, split=split))
                xinfo[f"{split}_macro_f1_eval"] = float(r.loc[r["class"] == "macro_avg", "f1"].iloc[0])
            pd.concat(rs).to_csv(OUT / "xgb_per_class.csv", index=False)
            pd.concat(cs).to_csv(OUT / "xgb_confusion.csv", index=False)
            fx.write_text(json.dumps(xinfo, indent=2), encoding="utf-8")
            reps.append(pd.concat(rs)), confs.append(pd.concat(cs))
        rep = pd.concat(reps, ignore_index=True)
        conf = pd.concat(confs, ignore_index=True)
        rep.to_csv(OUT / "stage7b_per_class.csv", index=False)
        conf.to_csv(OUT / "stage7b_confusion.csv", index=False)
        # ---- leakage check (2018 vs saved 2017)
        fl = OUT / "stage7b_leakage.csv"
        if not fl.exists():
            rng = np.random.default_rng(SEED + 1)
            rows = leakage("2018", Xs["train"], Xs["test"], y["test"], classes, dev, rng)
            s1 = resolve("results/stage1")
            f17 = json.loads((s1 / "features.json").read_text(encoding="utf-8"))
            lm17 = json.loads((s1 / "label_map.json").read_text(encoding="utf-8"))
            c17 = [c for c, _ in sorted(lm17.items(), key=lambda kv: kv[1])]
            pp17 = Preprocessor.load(resolve("results/stage2/preprocess.json"))
            tr17 = pd.read_parquet(s1 / "train.parquet", columns=f17 + ["label_id"])
            te17 = pd.read_parquet(s1 / "test.parquet", columns=f17 + ["label_id"])
            rows += leakage("2017", pp17.transform(tr17), pp17.transform(te17), te17["label_id"].to_numpy(), c17, dev, rng)
            pd.DataFrame(rows).to_csv(fl, index=False)
        leak = pd.read_csv(fl)
        entry = build_entry(prep, classes, rep, conf, infos, xinfo, leak, fam_txt, data, cfg)
        with open(ROOT / "PROGRESS.md", "a", encoding="utf-8") as fh:
            fh.write(entry.rstrip() + "\n")
        (OUT / "stage7b_progress_entry.md").write_text(entry, encoding="utf-8")
    except StopRun as e:
        log.error(f"STOPPED: {e}")
        with open(ROOT / "PROGRESS.md", "a", encoding="utf-8") as fh:
            fh.write(f"\n---\n\n## {datetime.now():%Y-%m-%d %H:%M}: Stage 7b stopped\n\nStop condition: {e}\n")
        finish_run(run_dir, log, {"stopped": str(e)}, t_start)
        sys.exit(2)
    finish_run(run_dir, log, {"seeds": SEEDS}, t_start)


def build_entry(prep, classes, rep, conf, infos, xinfo, leak, fam_txt, data, cfg):
    steps = pd.read_csv(OUT / "stage7b_counts_by_step.csv")
    by_cls = steps.pivot_table(index="class", columns="step", values="rows", aggfunc="sum", fill_value=0).astype(int)
    by_raw = steps[steps.step.str[0].isin(list("12345"))].pivot_table(index="raw_label", columns="step", values="rows",
                                                                       aggfunc="sum", fill_value=0).astype(int)
    for t in (by_cls, by_raw):
        t.loc["TOTAL"] = t.sum()
    te = rep[rep.split == "test"]
    ftt = te[te.model == "ftt"]
    g = ftt.groupby("class", sort=False)
    tab = pd.DataFrame({"precision mean": g.precision.mean(), "recall mean": g.recall.mean(), "F1 mean": g.f1.mean(),
                        "F1 range": g.f1.min().round(4).astype(str) + " to " + g.f1.max().round(4).astype(str),
                        "support": g.support.first()})
    xg = te[te.model == "xgb"].set_index("class")
    tab["XGB precision"], tab["XGB recall"], tab["XGB F1"] = xg.precision, xg.recall, xg.f1
    tab = tab.reset_index()
    cmt = conf[(conf.split == "test")]
    cm_f = cmt[cmt.model == "ftt"].groupby(["true", "pred"])["count"].mean().unstack().reindex(index=classes, columns=classes)
    cm_x = cmt[cmt.model == "xgb"].groupby(["true", "pred"])["count"].sum().unstack().reindex(index=classes, columns=classes)
    osum = json.loads((OUT / "stage7b_order_summary.json").read_text(encoding="utf-8"))
    runs = pd.read_csv(OUT / "stage7b_order_label_runs.csv")
    mono = pd.read_csv(OUT / "stage7b_order_monotone.csv").head(5)
    lk = leak.pivot_table(index="class", columns="dataset", values=["median", "p5", "share_exact_0"])
    lk.columns = [f"{a} {b}" for a, b in lk.columns]
    tt = [i.get("train_time_s", np.nan) for i in infos]
    n_tr = len(data["train"])
    # Stage 3 / 3c estimates scale with the explanation-set size (2017: 4,766 rows, 3 seeds, 3,064 s and 1,709 s).
    n3 = int(sum(min(500, int((data["test"]["label_id"] == k).sum())) for k in range(len(classes))))
    s3 = 3064 / 4766 * n3 / 3600
    s3c = 1709 / (4766 + 4858) * (2 * n3) / 3600
    mf = [i["test_macro_f1"] for i in infos]
    L = [f"\n---\n\n## {datetime.now():%Y-%m-%d %H:%M}: Stage 7b, LycoS-Unicas-IDS2018 prepared and FT-T trained (3 seeds) + XGBoost\n"]
    L.append(f"""**Files (all new):** `src/stage7b_prepare_train_2018.py`; `results/stage7/`:
- `data/{{train,val,test}}.parquet`, `data/features.json`, `data/label_map.json`
- `preprocess.json`, `seed{{0,1,2}}/` (`ftt.pt`, `curves.csv`, `per_class.csv`, `confusion.csv`, `info.json`), `xgb.json`
- `stage7b_*.csv`, `stage7b_order_*`

**Dataset limitation (recorded as decided)**
- Every FTP-Patator (190,300) and DoS Slowhttptest (105,550) row was removed. The two classes are indistinguishable in the 72 features: all 52 label-conflicting feature vectors contain both, and they have only 54 and 56 distinct vectors.
- LycoS-Unicas-IDS2018 is therefore used with 8 classes; results do not cover FTP brute force or slow-HTTP DoS on 2018.
""")
    L.append(f"""**Setup (decisions fixed by the user; nothing tuned on 2018)**
- Features: the 72 of `results/stage4_constraints.csv`, same order. `flag_SYN` renamed to `flag_syn` in the loaded frame only. Dropped: `dst_port`, `ip_prot`, `flag_urg`, `fwd_flag_urg`, `bwd_flag_urg`.
- Cleaning before any split, in this order:
  - remove the two classes
  - remove every feature vector with more than one MERGED class label ({prep['conflict_groups']} groups)
  - drop exact duplicates on the 72 features, keeping the first in file order
- Duplicates and conflicts use 64-bit row hashes with value-by-value verification: {prep['hash_collisions_split']} hash collisions found.
- Classes (merged): {', '.join(classes)}.
- Cap after cleaning: benign <= 1,000,000, each attack <= 300,000, random with seed {SEED}.
- Split: stratified random 60 / 20 / 20, seed {SEED}. No temporal split, since there are no timestamps.
- Preprocessing refit on the 2018 train split with the Stage 2 recipe (log1p for skew > 2.0 on non-negative features, standardise, clip +-10).
- FT-T, class weights (sqrt-inverse), AdamW, early stopping on val macro-F1 (patience 5, at most 20 epochs), bf16 training: all as in Stage 2, seeds 0, 1, 2. XGBoost as in Stage 2, once.
- Class families in `src/stage4_constraints.py` (checked without editing): {fam_txt}. All names resolve; the slow-DoS freeze applies to dos_slowloris. portscan and the two removed classes do not occur.

**1. Counts per class at every step**

{md(by_cls.reset_index())}

By original label (steps 1 to 5):

{md(by_raw.reset_index())}
""")
    L.append(f"""**2. Test results** (FT-T test macro-F1 per seed: {', '.join(f'{v:.4f}' for v in mf)}; mean {np.mean(mf):.4f}, range {min(mf):.4f} to {max(mf):.4f}. Val macro-F1: {', '.join(f"{i['val_macro_f1']:.4f}" for i in infos)}. XGBoost test macro-F1 {xinfo['test_macro_f1_eval']:.4f}, val {xinfo['val_macro_f1_eval']:.4f}, best iteration {xinfo['best_iteration']}.)

{md(tab.round(4))}

Confusion matrix on test, FT-T (mean over 3 seeds; rows = true, columns = predicted):

{md(cm_f.round(1).reset_index().rename(columns={'true': 'true / pred'}))}

Confusion matrix on test, XGBoost:

{md(cm_x.reset_index().rename(columns={'true': 'true / pred'}))}
""")
    L.append(f"""**3. Does the file row order look time-ordered?** (report only, not used)
- Label changes between adjacent rows: {osum['label_changes']:,}. If the file were shuffled, about {osum['expected_label_changes_if_shuffled']:,.0f} would be expected.
- Per label: number of contiguous runs and the longest run:

{md(runs.round(4))}

- Columns with the most strictly increasing adjacent steps. A shuffled file gives about 0.5 for continuous columns; a counter-like timestamp proxy would give close to 1.

{md(mono.round(4))}
""")
    L.append(f"""**4. Near-duplicate leakage:** distance of each test row to its nearest train row in standardised space, per dataset with its own preprocessing.
- Up to 2,000 random test rows per class.
- 2017 uses the saved Stage 1 split and the Stage 2 preprocess.json.

{md(lk.reset_index().round(4))}

**5. Training time and estimates**
- FT-T training time per seed: {', '.join(f'{t:.0f} s' for t in tt)}. Best epochs {', '.join(str(i['best_epoch']) for i in infos)}; peak GPU memory {max(i['peak_gpu_mem_gb'] for i in infos):.2f} GB. {n_tr:,} train rows, vs 906,981 in 2017.
- XGBoost: {xinfo['train_time_s']} s, {xinfo['rounds_run']} rounds.
- Stage 3 on 2018:
  - explanation set of up to 500 rows per class, all seeds correct: at most {n3:,} rows, vs 4,766 in 2017
  - estimate about {s3:.2f} h for 3 seeds (2017 rate 3,064 s for 4,766 rows x 3 seeds)
- Stage 3c on 2018: about {s3c:.2f} h, scaled from 2017 (1,709 s for 4,766 test + 4,858 val rows).

**Next**
- Stopped after Stage 7b as instructed; waiting for review.
""")
    return "\n".join(L)


if __name__ == "__main__":
    main()
