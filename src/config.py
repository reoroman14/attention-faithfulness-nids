# Default configuration for all stages.
# Override any value from the command line with:
#   --set section.key=value      (value parsed as JSON when possible, else kept as a string)
# Example:
#   python -m src.stage1_data --set stage1.split.gap_frac=0.02 --set seed=7
import argparse
import copy
import json

DAYS = ["Monday", "Tuesday", "Wednesday", "Thursday", "Friday"]

DEFAULTS = {
    "seed": 2026,
    "paths": {
        "pcap_lycos_dir": "data/pcap_lycos",
        "labelling_script": "data/labelling.py",
        "labelled_dir": "data/lycos-ids2017",
        "reference_labelling_log": "data/python_logs/labelling_info.log",
        "reference_create_log": "data/python_logs/create_datasets_info.log",
        # labelling.py is run with this folder as its working directory so it cannot
        # overwrite data/python_logs/labelling_info.log (it opens its log with mode "w").
        "labelling_workdir": "results/stage1/_labelling_work",
        "stage1_dir": "results/stage1",
        "stage2_dir": "results/stage2",
    },
    "labelling": {
        # labelling.py asks for the number of worker processes on stdin.
        "n_processes": 3,
        # Delete the extracted (unlabelled) CSVs from the work dir after a successful run.
        "keep_extracted_csv": False,
    },
    "stage1": {
        "days": DAYS,
        "chunk_rows": 200000,
        # Columns that identify a flow (5-tuple, id, time). Never used as model features.
        "identifier_columns": ["flow_id", "src_addr", "src_port", "dst_addr",
                               "dst_port", "ip_prot", "timestamp"],
        # Ablation switches: add these identifier columns back into the feature set.
        "include_dst_port": False,
        "include_ip_prot": False,
        # Columns reported constant (all zero) in data/python_logs/create_datasets_info.log.
        "drop_constant_columns": ["flag_urg", "fwd_flag_urg", "bwd_flag_urg"],
        # Also drop any other feature that is constant on the training split.
        "drop_train_constant": True,
        "label_merge": {
            "webattack_bruteforce": "webattack",
            "webattack_xss": "webattack",
            "webattack_sql_injection": "webattack",
        },
        # Matched against both the original and the merged label.
        # heartbleed: 11 rows; webattack_sql_injection: 12 rows that all land in test (after XSS in time).
        "drop_labels": ["heartbleed", "webattack_sql_injection"],
        # Feature vectors that occur with more than one label are removed entirely.
        "dedup_drop_conflicts": True,
        "split": {
            # "per_group_temporal": each (original label, day) group sorted by timestamp and cut
            #     into train/gap/val/gap/test; sub-labels are merged into final classes afterwards.
            # "per_class_temporal": same cut, but per final (merged) class across all days.
            # "cross_day": whole days assigned to splits (see cross_day below).
            "mode": "per_group_temporal",
            # Original labels folded together before grouping (too few rows to split on their own).
            "group_label_merge": {"webattack_sql_injection": "webattack_xss"},
            "fractions": [0.6, 0.2, 0.2],
            # Rows discarded at each train/val and val/test boundary, as a fraction of the group size.
            "gap_frac": 0.01,
            "cross_day": {
                "train": ["Monday", "Tuesday", "Wednesday"],
                "val": ["Thursday"],
                "test": ["Friday"],
            },
        },
    },
    "stage2": {
        # Which parts to run: "ftt", "xgb", "day_confound".
        "models": ["ftt", "xgb", "day_confound"],
        "preprocess": {
            # log1p is applied to features that are non-negative on train with skewness above this.
            "log1p_skew_threshold": 2.0,
            # Standardised values are clipped to [-clip, clip] (protects the transformer from
            # extreme out-of-range val/test values). Set to null to disable.
            "clip": 10.0,
        },
        # Loss weights per class: "none", "sqrt_inv" (sqrt inverse frequency) or "inv".
        # Weights are normalised so the average weight per training sample is 1.
        "class_weight": "sqrt_inv",
        # "none" = uniform shuffling; "balanced" = sample rows with probability proportional to class weight.
        "sampler": "none",
        "ftt": {
            "seeds": [0, 1, 2],
            "d_model": 96,
            "n_blocks": 3,
            "n_heads": 4,
            "ffn_factor": 4 / 3,
            "attn_dropout": 0.1,
            "ffn_dropout": 0.1,
            "residual_dropout": 0.0,
            "batch_size": 1024,
            "eval_batch_size": 8192,
            "lr": 3e-4,
            "weight_decay": 1e-5,
            "warmup_steps": 500,
            # One epoch is ~60 s on the RTX 4060 Laptop; val macro-F1 is ~0.97 after epoch 1.
            "max_epochs": 20,
            # Early stopping on validation macro-F1.
            "patience": 5,
            # The loss is checked for NaN/inf every this many steps (each check forces a GPU sync).
            "finite_check_every": 50,
            "grad_clip": 1.0,
            # bf16 autocast for training only; evaluation and attribution run in fp32.
            "amp": True,
        },
        "xgb": {
            "seed": 0,
            "device": "cuda",
            "max_depth": 8,
            "eta": 0.1,
            "subsample": 0.8,
            "colsample_bytree": 0.8,
            "min_child_weight": 1.0,
            "num_boost_round": 2000,
            # Early stopping on validation macro-F1 (custom metric, maximised).
            "early_stopping_rounds": 50,
        },
        "day_confound": {
            "seed": 0,
            "device": "cuda",
            "max_depth": 6,
            "eta": 0.2,
            "num_boost_round": 300,
        },
    },
}


def _set_nested(cfg, dotted_key, raw_value):
    # Walk the dotted path and overwrite the final key; unknown keys are an error.
    keys = dotted_key.split(".")
    node = cfg
    for k in keys[:-1]:
        if k not in node or not isinstance(node[k], dict):
            raise KeyError(f"Unknown config section: {dotted_key}")
        node = node[k]
    if keys[-1] not in node:
        raise KeyError(f"Unknown config key: {dotted_key}")
    try:
        value = json.loads(raw_value)
    except json.JSONDecodeError:
        value = raw_value
    node[keys[-1]] = value


def load_config(description, argv=None):
    # Returns (config dict, list of override strings) after applying --set overrides.
    parser = argparse.ArgumentParser(description=description)
    parser.add_argument("--set", action="append", default=[], metavar="KEY=VALUE",
                        help="Override a config value, e.g. --set stage1.split.gap_frac=0.02")
    args = parser.parse_args(argv)
    cfg = copy.deepcopy(DEFAULTS)
    for item in args.set:
        if "=" not in item:
            raise ValueError(f"--set expects KEY=VALUE, got {item!r}")
        key, value = item.split("=", 1)
        _set_nested(cfg, key.strip(), value.strip())
    return cfg, args.set
