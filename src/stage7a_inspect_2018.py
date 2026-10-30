# Stage 7a: inspect and convert LycoS-Unicas-IDS2018 (no training, no attacks, no split, no mapping decisions).
#
# Phase "convert" (one chunked pass over the 5.5 GB CSV, never loaded whole):
#   - header, inferred dtypes, non-numeric columns (values that fail numeric coercion are counted)
#   - float32 parquet in data/lycos-unicas-ids2018/parquet/ (label kept as a dictionary-encoded string)
#   - per-column NaN / inf / negative / zero counts, min / max, non-integer counts, float32 precision loss
#     (integral values with |x| > 2^24 that float32 cannot store exactly)
#   - label counts
#   - 64-bit row hashes (all columns; and the 75 non-identifier numeric columns) for duplicate and
#     label-conflict counts, as in Stage 1 (Stage 1 additionally verified value by value; here 64-bit hashes only)
#   - the Stage 4 invariants (src/stage4_constraints.relations, unchanged) on every row, after a case-only
#     column-name normalisation (flag_SYN -> flag_syn); integrality / non-negativity of the 72 features per the
#     Stage 4 spec
# Phase "report": comparisons (paper Table 1, results/stage4_constraints.csv, results/stage2/preprocess.json),
# runtime and disk estimates, PROGRESS.md entry.
#
# Usage:
#   .venv\Scripts\python.exe -m src.stage7a_inspect_2018 --phase convert
#   .venv\Scripts\python.exe -m src.stage7a_inspect_2018 --phase report
import argparse
import json
import time
from datetime import datetime
from pathlib import Path

import numpy as np
import pandas as pd
import pyarrow as pa
import pyarrow.parquet as pq

from src.stage4_constraints import load_spec_csv, relations
from src.utils import ROOT, resolve

CSV = ROOT / "data" / "lycos-unicas-ids2018" / "LycoS-Unicas-IDS2018.csv"
PQ_DIR = ROOT / "data" / "lycos-unicas-ids2018" / "parquet"
OUT = ROOT / "results" / "stage7" / "inspect"
CHUNK = 1_000_000
LABEL = "label"
F32_EXACT = 2 ** 24
# Paper Table 1, LycoS-Unicas-IDS2018 column (arXiv 2402.10974v1).
PAPER = {"Benign": 10_000_000, "DoS GoldenEye": 26_861, "DoS Slowloris": 10_274, "DoS Hulk": 1_802_966,
         "DoS Slowhttptest": 105_550, "DDoS LOIC-HTTP": 289_328, "DDoS LOIC-UDP": 2_382, "DDoS HOIC": 1_074_379,
         "SSH-Patator": 92_648, "FTP-Patator": 190_300, "Bot": 96_154, "Web Attack - Brute Force": 260,
         "Web Attack - XSS": 116, "Web Attack - Sql Injection": 50}


def convert():
    OUT.mkdir(parents=True, exist_ok=True)
    PQ_DIR.mkdir(parents=True, exist_ok=True)
    spec = load_spec_csv()
    feats72 = spec["feature"].tolist()
    t0 = time.time()
    header = pd.read_csv(CSV, nrows=0).columns.tolist()
    num_cols = [c for c in header if c != LABEL]
    lower = {c: c.lower() for c in num_cols}
    # Case-only normalisation for the invariant check (identical names apart from letter case).
    case_map = {c: lower[c] for c in num_cols if lower[c] in feats72}
    feats_present = [f for f in feats72 if f in case_map.values()]
    ident = ["dst_port", "ip_prot"]
    hash_cols = [c for c in num_cols if c not in ident]
    is_int = {r.feature: r.integer == "yes" for r in spec.itertuples()}
    S = {c: dict(nan=0, inf=0, neg=0, zero=0, nonint=0, f32_loss=0, min=np.inf, max=-np.inf, coerced=0)
         for c in num_cols}
    dtypes_first, object_cols = None, set()
    label_counts = {}
    h_all, h_feat, labels_all = [], [], []
    rel_fail, rel_n = {}, 0
    writer, n_rows = None, 0
    reader = pd.read_csv(CSV, chunksize=CHUNK, low_memory=False)
    for k, ch in enumerate(reader):
        if dtypes_first is None:
            dtypes_first = {c: str(t) for c, t in ch.dtypes.items()}
        for c in num_cols:
            if ch[c].dtype == object:
                object_cols.add(c)
                before = ch[c].notna()
                ch[c] = pd.to_numeric(ch[c].astype(str).str.strip(), errors="coerce")
                S[c]["coerced"] += int((before & ch[c].isna()).sum())
        X = ch[num_cols].to_numpy(np.float64)
        fin = np.isfinite(X)
        for j, c in enumerate(num_cols):
            v = X[:, j]
            f = fin[:, j]
            s = S[c]
            s["nan"] += int(np.isnan(v).sum())
            s["inf"] += int(np.isinf(v).sum())
            vf = v[f]
            if len(vf):
                s["min"], s["max"] = min(s["min"], float(vf.min())), max(s["max"], float(vf.max()))
            s["neg"] += int((vf < 0).sum())
            s["zero"] += int((vf == 0).sum())
            integral = vf == np.round(vf)
            s["nonint"] += int((~integral).sum())
            s["f32_loss"] += int((integral & (np.abs(vf) > F32_EXACT) & (vf.astype(np.float32).astype(np.float64) != vf)).sum())
        lab = ch[LABEL].astype(str)
        for l_, n_ in lab.value_counts().items():
            label_counts[l_] = label_counts.get(l_, 0) + int(n_)
        # Parquet (float32 numerics, dictionary-encoded label).
        df32 = pd.DataFrame(X.astype(np.float32), columns=num_cols)
        df32[LABEL] = pd.Categorical(lab)
        tbl = pa.Table.from_pandas(df32, preserve_index=False)
        tbl = tbl.cast(pa.schema([pa.field(c, pa.float32()) for c in num_cols] +
                                 [pa.field(LABEL, pa.dictionary(pa.int32(), pa.string()))]))
        if writer is None:
            writer = pq.ParquetWriter(PQ_DIR / "LycoS-Unicas-IDS2018.parquet", tbl.schema, compression="zstd")
        writer.write_table(tbl)
        # Row hashes on the stored float32 values (exact duplicates; feature-only duplicates / label conflicts).
        h_all.append(pd.util.hash_pandas_object(df32, index=False).to_numpy())
        h_feat.append(pd.util.hash_pandas_object(df32[hash_cols], index=False).to_numpy())
        labels_all.append(lab.to_numpy())
        # Stage 4 invariants on the clean (finite) rows, float64 values, case-normalised names.
        ok_rows = fin.all(1)
        if feats_present == feats72:
            Xi = ch.loc[ok_rows, list(case_map)].rename(columns=case_map)[feats72].to_numpy(np.float64)
            rel = relations(Xi, Xi, feats72)
            for name, holds in rel.items():
                rel_fail[name] = rel_fail.get(name, 0) + int((~holds).sum())
            ii = [feats72.index(f) for f in feats72 if is_int[f]]
            nonint = ~np.isclose(Xi[:, ii], np.round(Xi[:, ii]), atol=1e-6)
            rel_fail["integer_features_nonintegral"] = rel_fail.get("integer_features_nonintegral", 0) + int(nonint.any(1).sum())
            rel_n += len(Xi)
        n_rows += len(ch)
        print(f"chunk {k}: {n_rows:,} rows, {time.time() - t0:.0f} s", flush=True)
    writer.close()
    ha, hf = np.concatenate(h_all), np.concatenate(h_feat)
    labs = np.concatenate(labels_all)
    np.save(OUT / "row_hash_all.npy", ha)
    np.save(OUT / "row_hash_features.npy", hf)
    # Duplicates: exact (all columns incl. identifiers and label) and on the 75 non-identifier numeric columns.
    _, inv_a, cnt_a = np.unique(ha, return_inverse=True, return_counts=True)
    dup_exact = int(len(ha) - len(cnt_a))
    uf, inv_f, cnt_f = np.unique(hf, return_inverse=True, return_counts=True)
    dup_feat_rows = int(len(hf) - len(uf))
    # Label conflicts: feature-hash groups with more than one label.
    lab_codes, lab_names = pd.factorize(labs)
    pairs = np.unique(np.stack([inv_f, lab_codes], 1), axis=0)
    n_lab_per_group = np.bincount(pairs[:, 0], minlength=len(uf))
    conflict_groups = np.flatnonzero(n_lab_per_group > 1)
    conflict_rows = int(np.isin(inv_f, conflict_groups).sum())
    per_label = []
    for li, ln in enumerate(lab_names):
        m = lab_codes == li
        g = inv_f[m]
        per_label.append({"label": ln, "rows": int(m.sum()), "unique_feature_vectors": int(len(np.unique(g))),
                          "rows_in_conflict_groups": int(np.isin(g, conflict_groups).sum())})
    stats = {"n_rows": n_rows, "seconds": time.time() - t0, "header": header, "dtypes_first_chunk": dtypes_first,
             "object_columns": sorted(object_cols), "label_counts": label_counts, "columns": S,
             "dup_exact_rows": dup_exact, "dup_feature_rows": dup_feat_rows,
             "conflict_feature_groups": int(len(conflict_groups)), "conflict_rows": conflict_rows,
             "per_label_dedup": per_label, "hash_columns": hash_cols, "case_map": case_map,
             "invariant_rows": rel_n, "invariant_failures": rel_fail,
             "parquet_bytes": (PQ_DIR / "LycoS-Unicas-IDS2018.parquet").stat().st_size,
             "csv_bytes": CSV.stat().st_size}
    (OUT / "stats.json").write_text(json.dumps(stats, indent=1, default=float), encoding="utf-8")
    print(f"done: {n_rows:,} rows in {time.time() - t0:.0f} s")


# ---------------------------------------------------------------- report

def md(df):
    cols = [str(c) for c in df.columns]
    out = ["| " + " | ".join(cols) + " |", "|" + "---|" * len(cols)]
    for r in df.itertuples(index=False):
        out.append("| " + " | ".join(f"{v:,}" if isinstance(v, (int, np.integer)) else
                                      (f"{v:.4g}" if isinstance(v, float) else str(v)) for v in r) + " |")
    return "\n".join(out)


def dir_bytes(p):
    return sum(f.stat().st_size for f in Path(p).rglob("*") if f.is_file()) if Path(p).exists() else 0


def report(label_map_json):
    st = json.loads((OUT / "stats.json").read_text(encoding="utf-8"))
    spec = load_spec_csv()
    feats72 = spec["feature"].tolist()
    pre = json.loads(resolve("results/stage2/preprocess.json").read_text(encoding="utf-8"))["features"]
    header = st["header"]
    num_cols = [c for c in header if c != LABEL]
    L = [f"\n---\n\n## {datetime.now():%Y-%m-%d %H:%M}: Stage 7a, inspect and convert LycoS-Unicas-IDS2018 "
         "(no training, no attacks)\n"]
    L.append(f"""**Files (all new; nothing existing modified)**
- Code: `src/stage7a_inspect_2018.py`
- Parquet: `data/lycos-unicas-ids2018/parquet/LycoS-Unicas-IDS2018.parquet` (float32 numerics, dictionary-encoded `label`, zstd)
- `results/stage7/inspect/`: `stats.json`, `columns.csv`, `class_counts.csv`, `feature_comparison.csv`, `invariants.csv`, `row_hash_*.npy`

**Commands:** `.venv\\Scripts\\python.exe -m src.stage7a_inspect_2018 --phase convert`, then `--phase report`

**1. Header, dtypes, conversion**
- {st['n_rows']:,} rows and {len(header)} columns: {len(num_cols)} numeric plus `label`. One chunked pass (1,000,000 rows per chunk) took {st['seconds'] / 60:.1f} min.
- CSV {st['csv_bytes'] / 2**30:.2f} GiB, parquet {st['parquet_bytes'] / 2**30:.2f} GiB.
- Non-numeric (object) columns in any chunk: {', '.join(st['object_columns']) or 'none'}. Values that failed numeric coercion: {sum(v['coerced'] for v in st['columns'].values()):,}.
- Inferred dtypes (first chunk): {', '.join(f'{t} x{n}' for t, n in pd.Series(st['dtypes_first_chunk']).value_counts().items())}.
- **No identifier columns besides `dst_port` and `ip_prot`.** There is no flow id, source/destination address, source port or timestamp. Stage 1 dropped the identifiers and used the timestamp for the temporal split; that split cannot be reproduced from this file.
- Float32 storage: integral values above 2^24 that float32 cannot store exactly: {sum(v['f32_loss'] for v in st['columns'].values()):,} values in total (LYCOS-IDS2017 was stored the same way; see `columns.csv`).
""")
    # Class counts vs paper Table 1 (label strings matched to the paper names in label_map_json).
    lc = st["label_counts"]
    # Default: exact string match between file labels and the paper's class names.
    lm = json.loads(label_map_json) if label_map_json else {k: k for k in lc if k in PAPER}
    rows = []
    for lab, n in sorted(lc.items(), key=lambda kv: -kv[1]):
        pname = lm.get(lab, "")
        exp = PAPER.get(pname, np.nan)
        rows.append({"label (file)": lab, "rows": int(n), "paper name": pname or "(no match)",
                     "paper Table 1": "" if np.isnan(exp) else f"{int(exp):,}",
                     "difference": "" if np.isnan(exp) else f"{int(n - exp):+,}"})
    matched = set(lm.values())
    for pname, exp in PAPER.items():
        if pname not in matched:
            rows.append({"label (file)": "(absent)", "rows": 0, "paper name": pname, "paper Table 1": f"{exp:,}",
                         "difference": f"{-exp:+,}"})
    cdf = pd.DataFrame(rows)
    cdf.to_csv(OUT / "class_counts.csv", index=False)
    L.append(f"**2. Class counts vs paper Table 1** (total {st['n_rows']:,}; paper total {sum(PAPER.values()):,}). "
             "Label strings were matched to the paper's class names by exact string equality.\n\n" + md(cdf))
    # Feature comparison.
    fc = []
    lower_hdr = {c.lower(): c for c in num_cols}
    for i, f in enumerate(feats72):
        if f in num_cols:
            status = "identical"
        elif f in lower_hdr:
            status = f"differs in letter case only ({lower_hdr[f]})"
        else:
            status = "missing in 2018"
        fc.append({"feature (2017 spec)": f, "position 2017 spec": i,
                   "position 2018 header": num_cols.index(lower_hdr[f]) if f in lower_hdr else "",
                   "status vs spec": status,
                   "in preprocess.json": "yes" if f in pre else "no",
                   "preprocess.json position": pre.index(f) if f in pre else ""})
    for c in num_cols:
        if c.lower() not in feats72:
            fc.append({"feature (2017 spec)": "", "position 2017 spec": "", "position 2018 header": num_cols.index(c),
                       "status vs spec": f"new in 2018 / not in the 72-feature spec: {c}",
                       "in preprocess.json": "yes" if c in pre else "no", "preprocess.json position": ""})
    fdf = pd.DataFrame(fc)
    fdf.to_csv(OUT / "feature_comparison.csv", index=False)
    n_ident = int((fdf["status vs spec"] == "identical").sum())
    case_only = fdf[fdf["status vs spec"].str.startswith("differs")]
    missing = fdf[fdf["status vs spec"] == "missing in 2018"]
    new = fdf[fdf["status vs spec"].str.startswith("new")]
    order_same = [c.lower() for c in num_cols if c.lower() in feats72] == feats72
    L.append(f"""
**3. Feature comparison** (candidates only; no mapping chosen)
- Against `results/stage4_constraints.csv` (72 features):
  - identical names: {n_ident}
  - case-only difference: {len(case_only)} ({', '.join(case_only['status vs spec'])})
  - missing in 2018: {len(missing)}{(' (' + ', '.join(missing['feature (2017 spec)']) + ')') if len(missing) else ''}
  - in the 2018 header but not among the 72: {len(new)} ({', '.join(c.split(': ')[1] for c in new['status vs spec'])})
- Order of the 72 within the 2018 header, after case normalisation: {'identical to the 2017 spec order' if order_same else 'DIFFERENT from the 2017 spec order'}.
- Against `results/stage2/preprocess.json` (the 72-feature list `src/preprocess.py` loads; the module itself has no hard-coded list): the same 72 names in the same order as the spec ({'yes' if pre == feats72 else 'NO'}).
- Columns 2017 Stage 1 dropped and their 2018 counterparts:
  - `flag_urg`, `fwd_flag_urg`, `bwd_flag_urg` (constant 0 in 2017): present in 2018 (see constant-column check)
  - `dst_port`, `ip_prot` (excluded identifiers): present
  - `flow_id`, `src_addr`, `src_port`, `dst_addr`, `timestamp`: absent
- Candidate mapping for review: the case-only renaming `flag_SYN -> flag_syn` would make all 72 names identical to 2017. Not applied to the parquet, which keeps the original names.

Full per-feature table: `results/stage7/inspect/feature_comparison.csv`.
""")
    # Data quality.
    cols = pd.DataFrame([{"column": c, **{k: v for k, v in st["columns"][c].items()}} for c in num_cols])
    cols["constant"] = cols["min"] == cols["max"]
    cols.to_csv(OUT / "columns.csv", index=False)
    must_nonneg = [f for f in feats72 if f not in ("iat_min", "fwd_tcp_init_win_bytes", "bwd_tcp_init_win_bytes")]
    neg = cols[(cols.neg > 0)][["column", "neg", "min"]]
    neg["must be non-negative (2017 spec)"] = ["yes" if c.lower() in must_nonneg else "no (negative allowed in 2017: clock / -1 window)" for c in neg.column]
    L.append(f"""**4. Data quality**
- NaN cells: {int(cols.nan.sum()):,} (columns with any: {', '.join(cols[cols.nan > 0].column) or 'none'}).
- inf cells: {int(cols.inf.sum()):,} (columns with any: {', '.join(cols[cols.inf > 0].column) or 'none'}).
- Constant columns: {', '.join(f"{r.column} (= {r.min:g})" for r in cols[cols.constant].itertuples()) or 'none'}.
- Exact duplicate rows (all {len(header)} columns incl. `dst_port`, `ip_prot`, `label`): {st['dup_exact_rows']:,} ({100 * st['dup_exact_rows'] / st['n_rows']:.2f}%).
- Duplicates on the {len(st['hash_columns'])} non-identifier numeric columns (the Stage 1 dedup key, before dropping the 3 urg columns): {st['dup_feature_rows']:,} rows ({100 * st['dup_feature_rows'] / st['n_rows']:.2f}%).
- Label conflicts (identical non-identifier features, more than one label): {st['conflict_feature_groups']:,} feature vectors covering {st['conflict_rows']:,} rows.
- Hashes are 64-bit on the stored float32 values. Stage 1 also verified duplicates value by value; not done here.

Negative values:

{md(neg) if len(neg) else 'none'}

Per label: rows, unique feature vectors (non-identifier columns), rows in conflict groups:

{md(pd.DataFrame(st['per_label_dedup']).sort_values('rows', ascending=False))}

Full per-column table (NaN, inf, negatives, zeros, non-integral counts, min, max, float32 precision loss): `results/stage7/inspect/columns.csv`.
""")
    # Invariants.
    inv = pd.DataFrame([{"invariant": k, "violating_rows": v, "violation_rate": v / max(st["invariant_rows"], 1)}
                        for k, v in st["invariant_failures"].items()])
    inv.to_csv(OUT / "invariants.csv", index=False)
    L.append(f"""**5. Stage 4 invariants on 2018**
- Computed on {st['invariant_rows']:,} rows with all values finite, using `relations()` from `src/stage4_constraints.py` unchanged. Columns were case-normalised (`flag_SYN -> flag_syn`) only for this check.
- Tolerances are as in Stage 4 (rtol 1e-4; std bounds rtol 1e-3, atol 1e-2).
- Invariant groups:
  - `fwd_len_family`: min <= mean <= max, mean feasibility, std bounds, n = 1 / n = 0 cases
  - `fwd_non_empty`: non-empty count vs count / min / total
  - `fwd_psh`, `flag_psh_sum`, `ack`: ACK / PSH relations
  - `header`: header bytes vs header min x count, <= 60 per packet
  - `pkt_len_family`: flow-level max / min / mean / pooled variance / std
  - `fwd_iat_family`, `iat_family`: IAT summary statistics, totals, std bounds
  - `no_bwd_timing`: flow IAT = forward IAT without backward packets
  - `rates`, `down_up`, `subflow`, `window`, `fixed_timing_bounds`
  - `integer_features_nonintegral`: any spec-integer feature non-integral

{md(inv)}
""")
    # Runtime and disk estimates.
    pl = {k: v for k, v in lc.items()}
    ben_lab = [k for k, v in lm.items() if v == "Benign"]
    n_ben = sum(pl.get(b, 0) for b in ben_lab)
    n_att = st["n_rows"] - n_ben
    n_sub = min(n_ben, 1_000_000) + n_att
    r17 = 1_542_503
    scale = n_sub / r17
    sizes = {d: dir_bytes(resolve(f"results/{d}")) for d in ("stage1", "stage2", "stage3", "stage3c", "stage4", "stage5", "stage6")}
    L.append(f"""**6. Runtime and disk estimates for the existing pipeline on the subsample you specified** (all benign capped at 1,000,000, plus all {n_att:,} attack rows: {n_sub:,} rows before any dedup)
- Basis: measured LYCOS-IDS2017 times on 1,542,503 deduplicated rows (Stage 2 full run 2,215 s for 3 FT-T seeds + XGB + day-confound; Stage 3 3,064 s; Stage 3c 1,709 s; Stage 5 full run 4.3 h; Stage 6 FINAL 82 min; Stage 6e 67 min). Scale factor in rows: {scale:.2f}x.
- Estimates assume the same model, class structure and per-class row caps. Dedup will shrink the subsample by an unknown amount (see item 4).

| stage | 2017 measured | scales with | 2018 estimate per seed |
|---|---|---|---|
| Stage 1 (load, dedup, split) | about 3 min | rows | {3 * scale:.0f} min (one-off, not per seed) |
| Stage 2 FT-T training | about 10 to 12 min per seed (61 s per epoch on 906,981 train rows, 4 to 7 epochs + patience 5) | train rows | about {61 * scale * 0.6 * 1542503 / 906981 / 60 * 10:.0f} to {61 * scale * 0.6 * 1542503 / 906981 / 60 * 12:.0f} min (60% train share, 10 to 12 epochs) |
| Stage 2 XGBoost | about 2 min | train rows | about {2 * scale:.0f} min (once) |
| Stage 3 explanations | about 17 min per seed | rows per class (500 cap) x classes | about {17 * 14 / 11:.0f} min (14 classes vs 11) |
| Stage 3c diagnostics | about 9.5 min per seed | rows per class x classes | about {9.5 * 14 / 11:.0f} min |
| Stage 4/5 attacks (R1 config) | 1.4 h per seed | source rows (150 per attack class) | about {1.4 * 13 / 10:.1f} h (13 attack classes vs 10) |
| Stage 6 FINAL | about 27 min per seed | source rows, val benign | about {27 * 13 / 10:.0f} min |
| Stage 6e | about 33 min per seed | source rows | about {33 * 13 / 10:.0f} min |

- The attack-class count above assumes every 2018 attack label stays a separate class. Merging (e.g. the three web attacks) is a decision for review.
- Disk:
  - CSV 5.14 GiB (existing); parquet {st['parquet_bytes'] / 2**30:.2f} GiB (written)
  - 2017 result folders: {', '.join(f'{k} {v / 2**30:.2f} GiB' for k, v in sizes.items())}; total {sum(sizes.values()) / 2**30:.2f} GiB
  - 2018 stages 1 to 6 estimate: about {(sizes['stage1'] * scale + sum(v for k, v in sizes.items() if k != 'stage1') * 12 / 10) / 2**30:.1f} GiB. Stage 1 scales with rows; the rest with source rows per class.
  - Free disk on C: at the time of writing: {__import__('shutil').disk_usage('C:/').free / 2**30:.0f} GiB.
""")
    L.append("""**Open decisions for review (not taken here)**
- Feature mapping: accept the case-only rename `flag_SYN -> flag_syn` or not; whether to keep `dst_port` / `ip_prot` excluded and to drop the three urg columns as in 2017.
- Split: no timestamp is available, so the per-group temporal split of Stage 1 is impossible from this file. Alternatives (random stratified, or obtaining timestamps from the LycoS-Unicas release) need a decision.
- Subsample: benign cap, and whether to deduplicate before or after subsampling.
- Class structure: keep or merge the three web-attack classes and the DDoS variants.
- Preprocessing: refit on 2018 (log1p / standardise) or reuse the 2017 preprocess.json for cross-dataset work.

**Next**
- Stopped after Stage 7a as instructed; waiting for review.
""")
    text = "\n".join(L)
    (OUT / "progress_entry.md").write_text(text, encoding="utf-8")
    with open(ROOT / "PROGRESS.md", "a", encoding="utf-8") as fh:
        fh.write(text.rstrip() + "\n")
    print(text[:3000])


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--phase", choices=["convert", "report"], required=True)
    ap.add_argument("--label-map", default="", help="JSON dict: file label -> paper Table 1 class name")
    a = ap.parse_args()
    convert() if a.phase == "convert" else report(a.label_map)


if __name__ == "__main__":
    main()
