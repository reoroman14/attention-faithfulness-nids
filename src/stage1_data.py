# Stage 1b: clean, relabel, deduplicate and split LYCOS-IDS2017.
# Input : data/lycos-ids2017/<Day>-WorkingHours.pcap_lycos.csv  (made by src.run_labelling)
# Output: results/stage1/cleaned/<Day>.parquet   cleaned rows before dedup (float32 features)
#         results/stage1/{train,val,test}.parquet, features.json, label_map.json
#         results/stage1_*.csv                      reports
#
# Usage: .venv\Scripts\python.exe -m src.stage1_data [--set stage1.split.gap_frac=0.02]
import json
import math
import time
from datetime import datetime, timezone

import numpy as np
import pandas as pd
import pyarrow as pa
import pyarrow.parquet as pq

from src.config import load_config
from src.utils import RESULTS, ROOT, finish_run, resolve, start_run

META_DTYPES = {"flow_id": "string", "src_addr": "string", "dst_addr": "string",
               "src_port": "int32", "dst_port": "int32", "ip_prot": "int16", "timestamp": "int64"}


def normalise_columns(cols):
    # Fixes header quirks: stray whitespace and the lone upper-case 'flag_SYN'.
    return [c.strip().lower() for c in cols]


def clean_chunk(chunk, day, cfg, feat_cols, stats):
    s1 = cfg["stage1"]
    chunk.columns = normalise_columns(chunk.columns)

    # Coerce any non-numeric feature column (e.g. values with a leading space) to float.
    for c in feat_cols:
        if not pd.api.types.is_numeric_dtype(chunk[c]):
            txt = chunk[c].astype("string").str.strip()
            num = pd.to_numeric(txt, errors="coerce")
            failed = num.isna() & txt.notna() & ~txt.str.lower().isin(["", "nan", "inf", "-inf",
                                                                      "infinity", "-infinity", "+infinity"])
            stats["coerced_columns"].add(c)
            stats["coerce_failures"] += int(failed.sum())
            chunk[c] = num

    label_orig = chunk["label"].astype("string").str.strip()
    label = label_orig.replace(s1["label_merge"])

    # Drop excluded labels (heartbleed) first so they are not counted elsewhere.
    keep = ~(label_orig.isin(s1["drop_labels"]) | label.isin(s1["drop_labels"]))
    for lab, n in label_orig[~keep].value_counts().items():
        stats["dropped_label"][lab] = stats["dropped_label"].get(lab, 0) + int(n)

    feats = chunk[feat_cols].astype("float64")
    arr = feats.to_numpy()
    has_inf = np.isinf(arr).any(axis=1)
    has_nan = np.isnan(arr).any(axis=1)
    bad = (has_inf | has_nan) & keep.to_numpy()
    for lab, n in label[bad].value_counts().items():
        stats["dropped_nan_inf"][lab] = stats["dropped_nan_inf"].get(lab, 0) + int(n)
    stats["rows_with_inf"] += int((has_inf & keep.to_numpy()).sum())
    stats["rows_with_nan"] += int((has_nan & keep.to_numpy()).sum())

    # float32 overflow check: values finite in float64 but too large for float32.
    ok = keep.to_numpy() & ~bad
    f32_overflow = (np.abs(arr[ok]) > np.finfo(np.float32).max).any(axis=1)
    stats["float32_overflow_rows"] += int(f32_overflow.sum())

    out = feats[ok].astype("float32").reset_index(drop=True)
    meta = chunk.loc[ok, list(META_DTYPES)].reset_index(drop=True)
    for c, dt in META_DTYPES.items():
        meta[c] = meta[c].astype(dt)
    out = pd.concat([meta, out], axis=1)
    out["day"] = day
    out["label_orig"] = label_orig[ok].to_numpy()
    out["label"] = label[ok].to_numpy()
    return out


def stage_clean(cfg, log):
    # Read each labelled CSV in chunks, clean, and stream to results/stage1/cleaned/<Day>.parquet.
    s1 = cfg["stage1"]
    labelled_dir = resolve(cfg["paths"]["labelled_dir"])
    cleaned_dir = resolve(cfg["paths"]["stage1_dir"]) / "cleaned"
    cleaned_dir.mkdir(parents=True, exist_ok=True)

    raw_rows, clean_rows = [], []
    feat_cols = None
    for day in s1["days"]:
        path = labelled_dir / f"{day}-WorkingHours.pcap_lycos.csv"
        header = normalise_columns(pd.read_csv(path, nrows=0, encoding="ISO-8859-1").columns)
        cols_here = [c for c in header if c not in s1["identifier_columns"] and c != "label"]
        if feat_cols is None:
            feat_cols = cols_here
        elif cols_here != feat_cols:
            raise RuntimeError(f"{day}: feature columns differ from previous days")

        stats = {"coerced_columns": set(), "coerce_failures": 0, "dropped_label": {},
                 "dropped_nan_inf": {}, "rows_with_inf": 0, "rows_with_nan": 0,
                 "float32_overflow_rows": 0}
        n_read, n_written, writer, schema = 0, 0, None, None
        out_path = cleaned_dir / f"{day}.parquet"
        raw_counts = {}
        for chunk in pd.read_csv(path, chunksize=s1["chunk_rows"], encoding="ISO-8859-1",
                                 low_memory=False):
            n_read += len(chunk)
            for lab, n in chunk["label"].value_counts().items():
                raw_counts[lab] = raw_counts.get(lab, 0) + int(n)
            out = clean_chunk(chunk, day, cfg, feat_cols, stats)
            table = pa.Table.from_pandas(out, preserve_index=False)
            if writer is None:
                schema = table.schema
                writer = pq.ParquetWriter(out_path, schema)
            writer.write_table(table.cast(schema))
            n_written += len(out)
        writer.close()

        for lab, n in sorted(raw_counts.items()):
            raw_rows.append({"day": day, "label_orig": lab, "n": n})
        clean_rows.append({
            "day": day, "rows_read": n_read, "rows_written": n_written,
            "dropped_labels": json.dumps(stats["dropped_label"]),
            "dropped_nan_inf": int(sum(stats["dropped_nan_inf"].values())),
            "dropped_nan_inf_by_class": json.dumps(stats["dropped_nan_inf"]),
            "rows_with_inf": stats["rows_with_inf"], "rows_with_nan": stats["rows_with_nan"],
            "float32_overflow_rows": stats["float32_overflow_rows"],
            "coerced_columns": ";".join(sorted(stats["coerced_columns"])),
            "coerce_failures": stats["coerce_failures"],
        })
        log.info(f"{day}: read {n_read}, kept {n_written}, nan/inf dropped "
                 f"{sum(stats['dropped_nan_inf'].values())}, label-dropped {stats['dropped_label']}, "
                 f"coerced cols {sorted(stats['coerced_columns']) or 'none'}")

    pd.DataFrame(raw_rows).to_csv(RESULTS / "stage1_label_counts_raw.csv", index=False)
    pd.DataFrame(clean_rows).to_csv(RESULTS / "stage1_cleaning_report.csv", index=False)
    return feat_cols, pd.DataFrame(raw_rows)


def choose_features(cfg, all_feat_cols, df, log):
    # Model features = numeric flow features minus known-constant columns (+ optional dst_port/ip_prot).
    s1 = cfg["stage1"]
    rows = []
    for c in s1["drop_constant_columns"]:
        mn, mx = float(df[c].min()), float(df[c].max())
        rows.append({"feature": c, "kept": False, "reason": "configured constant column",
                     "min_all": mn, "max_all": mx})
        if mn != mx:
            log.warning(f"{c} is NOT constant on the cleaned data (min={mn}, max={mx}); dropped anyway per config")
    feats = [c for c in all_feat_cols if c not in s1["drop_constant_columns"]]
    if s1["include_ip_prot"]:
        feats = ["ip_prot"] + feats
    if s1["include_dst_port"]:
        feats = ["dst_port"] + feats
    for c in s1["identifier_columns"]:
        if c not in feats:
            rows.append({"feature": c, "kept": False, "reason": "identifier (excluded)"})
    return feats, rows


def deduplicate(df, feats, cfg, log):
    # Exact duplicates on the model feature columns. Earliest occurrence (by timestamp) is kept.
    # Rows are hashed (64-bit) for speed; every hash-duplicate is then compared value-by-value
    # with its kept row so hash collisions cannot remove distinct rows.
    X = df[feats].to_numpy(dtype=np.float32) + np.float32(0.0)  # folds -0.0 into 0.0
    h = pd.util.hash_pandas_object(pd.DataFrame(X), index=False).to_numpy()
    labels = df["label"].to_numpy()

    report = pd.DataFrame({"label": df["label"].value_counts()}).rename(columns={"label": "before_dedup"})
    report.index.name = "class"

    conflict = np.zeros(len(df), dtype=bool)
    conflict_pairs = pd.DataFrame()
    if cfg["stage1"]["dedup_drop_conflicts"]:
        g = pd.DataFrame({"h": h, "label": labels})
        n_lab = g.groupby("h")["label"].nunique()
        bad_h = n_lab.index[n_lab.to_numpy() > 1]
        conflict = g["h"].isin(bad_h).to_numpy()
        pairs = (g[conflict].groupby("h")["label"].apply(lambda s: " | ".join(sorted(set(s))))
                 .value_counts().rename("n_distinct_vectors"))
        conflict_pairs = pairs.reset_index().rename(columns={"index": "labels", "label": "labels"})
        log.info(f"Conflicting feature vectors (same features, different labels): {len(bad_h)} distinct, "
                 f"{int(conflict.sum())} rows")

    keep_pool = ~conflict
    hs = pd.Series(h[keep_pool])
    dup_local = hs.duplicated(keep="first").to_numpy()
    # Collision check: compare each duplicate with the first row sharing its hash.
    idx_pool = np.flatnonzero(keep_pool)
    first_of_hash = pd.Series(np.arange(len(hs))).groupby(hs.to_numpy()).transform("first").to_numpy()
    d = np.flatnonzero(dup_local)
    same = np.ones(len(d), dtype=bool)
    for start in range(0, len(d), 200000):
        sl = d[start:start + 200000]
        same[start:start + 200000] = (X[idx_pool[sl]] == X[idx_pool[first_of_hash[sl]]]).all(axis=1)
    n_collisions = int((~same).sum())
    if n_collisions:
        log.warning(f"{n_collisions} hash collisions found; those rows are kept")
    dup = np.zeros(len(df), dtype=bool)
    dup[idx_pool[d[same]]] = True

    report["dropped_conflict"] = pd.Series(labels[conflict]).value_counts()
    report["dropped_duplicate"] = pd.Series(labels[dup]).value_counts()
    report = report.fillna(0).astype(int)
    report["after_dedup"] = report["before_dedup"] - report["dropped_conflict"] - report["dropped_duplicate"]
    report["removed_pct"] = (100 * (1 - report["after_dedup"] / report["before_dedup"])).round(2)
    total = report.sum(numeric_only=True)
    total["removed_pct"] = round(100 * (1 - total["after_dedup"] / total["before_dedup"]), 2)
    report.loc["TOTAL"] = total
    log.info(f"Dedup: {int(total['before_dedup'])} -> {int(total['after_dedup'])} rows "
             f"({total['removed_pct']}% removed, collisions={n_collisions})")
    return df[~(conflict | dup)].reset_index(drop=True), report, conflict_pairs, n_collisions


def temporal_cut(df, group_keys, cfg):
    # For every group: stable sort by timestamp, then train | gap | val | gap | test.
    sp = cfg["stage1"]["split"]
    f_tr, f_va, f_te = sp["fractions"]
    f_sum = f_tr + f_va + f_te
    ts = df["timestamp"].to_numpy()
    assign = np.full(len(df), "", dtype=object)
    for _, idx in df.groupby(group_keys, sort=True).indices.items():
        order = idx[np.argsort(ts[idx], kind="stable")]
        n = len(order)
        g = int(math.ceil(sp["gap_frac"] * n)) if sp["gap_frac"] > 0 else 0
        usable = n - 2 * g
        n_tr = int(round(usable * f_tr / f_sum))
        n_va = int(round(usable * f_va / f_sum))
        a, b = n_tr, n_tr + g
        c, d = b + n_va, b + n_va + g
        assign[order[:a]] = "train"
        assign[order[a:b]] = "gap"
        assign[order[b:c]] = "val"
        assign[order[c:d]] = "gap"
        assign[order[d:]] = "test"
    return assign


def split_per_class_temporal(df, cfg, log):
    return temporal_cut(df, ["label"], cfg), ["label"]


def split_per_group_temporal(df, cfg, log):
    # Groups are (original label, day); the final merged class is only used after splitting.
    df["split_group"] = df["label_orig"].replace(cfg["stage1"]["split"]["group_label_merge"])
    keys = ["split_group", "day"]
    groups = df.groupby(keys).size()
    log.info(f"Split groups (original label x day): {len(groups)}")
    return temporal_cut(df, keys, cfg), keys


def split_cross_day(df, cfg, log):
    cd = cfg["stage1"]["split"]["cross_day"]
    assign = np.full(len(df), "unused", dtype=object)
    for split in ("train", "val", "test"):
        assign[df["day"].isin(cd[split]).to_numpy()] = split
    return assign, ["label", "day"]


def to_utc(us):
    return datetime.fromtimestamp(int(us) / 1e6, tz=timezone.utc).strftime("%Y-%m-%d %H:%M:%S")


def main():
    cfg, overrides = load_config("Stage 1: clean, dedup and split LYCOS-IDS2017.")
    t0 = time.time()
    run_dir, log = start_run("stage1_data", cfg, overrides)
    s1 = cfg["stage1"]
    out_dir = resolve(cfg["paths"]["stage1_dir"])
    out_dir.mkdir(parents=True, exist_ok=True)

    # 1. Clean (chunked) -> per-day parquet.
    all_feat_cols, raw_counts = stage_clean(cfg, log)
    log.info(f"Raw numeric feature columns: {len(all_feat_cols)}")

    # 2. Load cleaned data, order by time.
    df = pd.concat([pd.read_parquet(out_dir / "cleaned" / f"{d}.parquet") for d in s1["days"]],
                   ignore_index=True)
    df = df.sort_values("timestamp", kind="stable").reset_index(drop=True)
    log.info(f"Cleaned rows: {len(df)}; memory {df.memory_usage(deep=True).sum() / 2**30:.2f} GiB")

    # 3. Feature selection (identifiers / dst_port excluded unless configured).
    feats, feat_rows = choose_features(cfg, all_feat_cols, df, log)

    # 4. Deduplicate on the feature columns.
    df, dedup_report, conflict_pairs, n_coll = deduplicate(df, feats, cfg, log)
    dedup_report.to_csv(RESULTS / "stage1_dedup_report.csv")
    if len(conflict_pairs):
        conflict_pairs.to_csv(RESULTS / "stage1_dedup_conflicts.csv", index=False)

    # 5. Split.
    mode = s1["split"]["mode"]
    if mode == "per_group_temporal":
        assign, range_keys = split_per_group_temporal(df, cfg, log)
    elif mode == "per_class_temporal":
        assign, range_keys = split_per_class_temporal(df, cfg, log)
    elif mode == "cross_day":
        assign, range_keys = split_cross_day(df, cfg, log)
    else:
        raise ValueError(f"Unknown split mode {mode}")
    df["split"] = assign
    gap_counts = df.loc[df["split"].eq("gap"), "label"].value_counts()

    # 6. Drop features constant on train.
    tr_mask = df["split"].eq("train").to_numpy()
    if s1["drop_train_constant"]:
        tr = df.loc[tr_mask, feats]
        const = [c for c in feats if tr[c].min() == tr[c].max()]
        for c in const:
            feat_rows.append({"feature": c, "kept": False, "reason": "constant on train",
                              "min_all": float(df[c].min()), "max_all": float(df[c].max())})
        feats = [c for c in feats if c not in const]
        if const:
            log.info(f"Dropped train-constant features: {const}")
    tr = df.loc[tr_mask, feats]
    for c in feats:
        feat_rows.append({"feature": c, "kept": True, "reason": "model feature",
                          "min_all": float(df[c].min()), "max_all": float(df[c].max()),
                          "train_min": float(tr[c].min()), "train_max": float(tr[c].max()),
                          "train_n_negative": int((tr[c] < 0).sum())})
    feat_df = pd.DataFrame(feat_rows)
    feat_df.to_csv(RESULTS / "stage1_features.csv", index=False)
    log.info(f"Model features: {len(feats)}")

    # 7. Label ids: benign = 0, then alphabetical.
    classes = ["benign"] + sorted(c for c in df["label"].unique() if c != "benign")
    label_map = {c: i for i, c in enumerate(classes)}
    df["label_id"] = df["label"].map(label_map).astype("int16")

    # 8. Write splits (float32 features + label + metadata).
    keep_cols = list(META_DTYPES) + ["day", "label_orig", "label", "label_id", "split"]
    if "split_group" in df:
        keep_cols.append("split_group")
    keep_cols = [c for c in keep_cols if c not in feats] + feats
    for split in ("train", "val", "test"):
        part = df.loc[df["split"].eq(split), keep_cols].reset_index(drop=True)
        part.to_parquet(out_dir / f"{split}.parquet", index=False)
        log.info(f"Wrote {split}.parquet: {len(part)} rows")
    (out_dir / "features.json").write_text(json.dumps(feats, indent=2), encoding="utf-8")
    (out_dir / "label_map.json").write_text(json.dumps(label_map, indent=2), encoding="utf-8")

    # 9. Reports.
    raw_merged = raw_counts.assign(label=raw_counts["label_orig"].replace(s1["label_merge"]))
    raw_by_class = raw_merged.groupby("label")["n"].sum()
    clean_df = pd.read_csv(RESULTS / "stage1_cleaning_report.csv")
    nan_by_class = {}
    for js in clean_df["dropped_nan_inf_by_class"]:
        for k, v in json.loads(js).items():
            nan_by_class[k] = nan_by_class.get(k, 0) + v
    split_counts = df.groupby(["label", "split"]).size().unstack(fill_value=0)
    counts = pd.DataFrame(index=sorted(set(raw_by_class.index), key=lambda c: (c != "benign", c)))
    counts.index.name = "class"
    counts["raw_labelled"] = raw_by_class
    excluded = raw_merged["label_orig"].isin(s1["drop_labels"]) | raw_merged["label"].isin(s1["drop_labels"])
    counts["dropped_label_excluded"] = raw_merged[excluded].groupby("label")["n"].sum()
    counts["dropped_nan_inf"] = pd.Series(nan_by_class)
    counts["after_clean"] = dedup_report["before_dedup"]
    counts["dropped_conflict"] = dedup_report["dropped_conflict"]
    counts["dropped_duplicate"] = dedup_report["dropped_duplicate"]
    counts["after_dedup"] = dedup_report["after_dedup"]
    counts["dedup_removed_pct"] = dedup_report["removed_pct"]
    counts["gap_dropped"] = gap_counts
    for split in ("train", "val", "test"):
        counts[split] = split_counts[split] if split in split_counts else 0
    counts = counts.fillna(0)
    num = counts.columns.drop("dedup_removed_pct")
    counts[num] = counts[num].astype(int)
    counts["label_id"] = [label_map.get(c, -1) for c in counts.index]
    tot = counts[num].sum()
    counts.loc["TOTAL", num] = tot
    counts.loc["TOTAL", "dedup_removed_pct"] = round(100 * (1 - tot["after_dedup"] / tot["after_clean"]), 2)
    counts.loc["TOTAL", "label_id"] = -1
    counts[num] = counts[num].astype(int)
    counts["label_id"] = counts["label_id"].astype(int)
    counts.to_csv(RESULTS / "stage1_class_counts.csv")
    log.info("Class counts:\n" + counts.to_string())

    # Time ranges per split group and split; within each group the splits must be in time order.
    used = df[df["split"].isin(["train", "val", "test"])]
    tr_rows = []
    for key, g in used.groupby(range_keys + ["split"]):
        key = dict(zip(range_keys + ["split"], key))
        tr_rows.append({**key, "class": g["label"].iloc[0], "n": len(g),
                        "label_orig_counts": json.dumps(g["label_orig"].value_counts().to_dict()),
                        "t_min_utc": to_utc(g["timestamp"].min()), "t_max_utc": to_utc(g["timestamp"].max()),
                        "t_min_us": int(g["timestamp"].min()), "t_max_us": int(g["timestamp"].max()),
                        "days": ";".join(sorted(g["day"].unique()))})
    ranges = pd.DataFrame(tr_rows)
    order = {"train": 0, "val": 1, "test": 2}
    ranges["_o"] = ranges["split"].map(order)
    ranges = ranges.sort_values(range_keys + ["_o"]).drop(columns="_o").reset_index(drop=True)
    ranges["ordered_after_prev_split"] = None
    if mode in ("per_group_temporal", "per_class_temporal"):
        for _, g in ranges.groupby(range_keys):
            prev_max = None
            for i, r in g.iterrows():
                ranges.loc[i, "ordered_after_prev_split"] = (prev_max is None) or bool(r["t_min_us"] > prev_max)
                prev_max = r["t_max_us"]
    ranges.to_csv(RESULTS / "stage1_split_time_ranges.csv", index=False)

    # Class counts per split per day (final merged classes), plus gap rows.
    by_day = (df.groupby(["label", "day", "split"]).size().unstack("split", fill_value=0)
              .reindex(columns=["train", "val", "test", "gap"], fill_value=0))
    by_day["total"] = by_day.sum(axis=1)
    day_order = {d: i for i, d in enumerate(s1["days"])}
    by_day = by_day.reset_index()
    by_day["_c"] = by_day["label"].ne("benign").astype(int)
    by_day["_d"] = by_day["day"].map(day_order)
    by_day = (by_day.sort_values(["_c", "label", "_d"]).drop(columns=["_c", "_d"])
              .rename(columns={"label": "class"}))
    by_day.to_csv(RESULTS / "stage1_class_counts_by_day.csv", index=False)
    log.info("Class counts per split per day:\n" + by_day.to_string(index=False))

    for name in ["stage1_class_counts.csv", "stage1_dedup_report.csv", "stage1_features.csv",
                 "stage1_cleaning_report.csv", "stage1_label_counts_raw.csv", "stage1_split_time_ranges.csv",
                 "stage1_class_counts_by_day.csv"]:
        (run_dir / name).write_bytes((RESULTS / name).read_bytes())

    finish_run(run_dir, log, {
        "n_features": len(feats), "classes": label_map, "hash_collisions": n_coll,
        "rows_after_clean": int(tot["after_clean"]), "rows_after_dedup": int(tot["after_dedup"]),
        "train": int(tot["train"]), "val": int(tot["val"]), "test": int(tot["test"]),
        "unordered_splits": int((ranges["ordered_after_prev_split"] == False).sum()),
    }, t0)


if __name__ == "__main__":
    main()
