# Shared helpers: run directories, logging, seeding and environment capture.
import csv
import json
import logging
import platform
import random
import sys
import time
from datetime import datetime
from importlib import metadata
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
RESULTS = ROOT / "results"
RUN_INDEX = RESULTS / "run_log.csv"

PACKAGES = ["numpy", "pandas", "pyarrow", "scikit-learn", "scipy", "torch",
            "captum", "shap", "xgboost"]


def resolve(path_str):
    # Config paths are relative to the project root.
    p = Path(path_str)
    return p if p.is_absolute() else ROOT / p


def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    if "torch" in sys.modules:
        import torch
        torch.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)


def env_info():
    info = {"python": sys.version, "platform": platform.platform(), "packages": {}}
    for pkg in PACKAGES:
        try:
            info["packages"][pkg] = metadata.version(pkg)
        except metadata.PackageNotFoundError:
            info["packages"][pkg] = None
    # Only query the GPU if torch was already imported by the stage.
    if "torch" in sys.modules:
        import torch
        info["cuda_available"] = torch.cuda.is_available()
        if torch.cuda.is_available():
            info["gpu"] = torch.cuda.get_device_name(0)
    return info


def start_run(stage, cfg, overrides):
    # Creates results/runs/<timestamp>_<stage>/ with config.json and log.txt,
    # appends a line to results/run_log.csv and returns (run_dir, logger).
    stamp = datetime.now().strftime("%Y%m%d-%H%M%S")
    run_dir = RESULTS / "runs" / f"{stamp}_{stage}"
    run_dir.mkdir(parents=True, exist_ok=False)
    set_seed(cfg["seed"])
    record = {
        "stage": stage,
        "started": datetime.now().isoformat(timespec="seconds"),
        "command": " ".join([Path(sys.executable).name] + sys.argv),
        "overrides": overrides,
        "seed": cfg["seed"],
        "config": cfg,
        "env": env_info(),
    }
    (run_dir / "config.json").write_text(json.dumps(record, indent=2), encoding="utf-8")

    logger = logging.getLogger(stage)
    logger.setLevel(logging.INFO)
    logger.handlers.clear()
    fmt = logging.Formatter("[%(levelname)s] %(asctime)s - %(message)s", "%Y-%m-%d %H:%M:%S")
    for h in (logging.StreamHandler(sys.stdout),
              logging.FileHandler(run_dir / "log.txt", encoding="utf-8")):
        h.setFormatter(fmt)
        logger.addHandler(h)

    new_index = not RUN_INDEX.exists()
    with open(RUN_INDEX, "a", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        if new_index:
            w.writerow(["started", "stage", "seed", "run_dir", "command"])
        w.writerow([record["started"], stage, cfg["seed"],
                    run_dir.relative_to(ROOT).as_posix(), record["command"]])
    logger.info(f"Run dir: {run_dir.relative_to(ROOT)}  seed={cfg['seed']}")
    return run_dir, logger


def finish_run(run_dir, logger, summary, t0):
    summary = dict(summary)
    summary["elapsed_s"] = round(time.time() - t0, 1)
    summary["finished"] = datetime.now().isoformat(timespec="seconds")
    (run_dir / "summary.json").write_text(json.dumps(summary, indent=2, default=str),
                                          encoding="utf-8")
    logger.info(f"Finished in {summary['elapsed_s']} s")
