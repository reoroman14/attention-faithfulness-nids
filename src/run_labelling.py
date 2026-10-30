# Stage 1a: run data/labelling.py UNMODIFIED, then verify its output against the
# reference log data/python_logs/labelling_info.log.
#
# labelling.py resolves every path relative to the current working directory:
#   ./pcap_lycos/<Day>-WorkingHours.pcap_lycos.csv   input (unzipped)
#   ./lycos-ids2017/                                  output
#   ./python_logs/labelling_info.log                  log, opened with mode "w"
# Running it with cwd=data/ would overwrite the reference log, so it runs inside a
# separate work dir. Finished outputs are moved into data/lycos-ids2017/.
# Existing labelled files are never overwritten: if all five exist, labelling is
# skipped and only the verification is run.
#
# Usage: .venv\Scripts\python.exe -m src.run_labelling [--set labelling.n_processes=3]
import re
import shutil
import subprocess
import sys
import time
import zipfile

import pandas as pd

from src.config import load_config
from src.utils import RESULTS, finish_run, resolve, start_run

DAY_ABBR = {"Mon": "Monday", "Tue": "Tuesday", "Wed": "Wednesday", "Thu": "Thursday", "Fri": "Friday"}


def csv_name(day):
    return f"{day}-WorkingHours.pcap_lycos.csv"


def parse_reference_counts(log_path):
    # Lines look like: "... - Tue_thread - Lycos ftp_patator: 4003"
    pat = re.compile(r"(Mon|Tue|Wed|Thu|Fri)_thread - Lycos (\S+): (\d+)")
    rows = []
    for line in log_path.read_text(encoding="utf-8", errors="replace").splitlines():
        m = pat.search(line)
        if m:
            rows.append((DAY_ABBR[m.group(1)], m.group(2), int(m.group(3))))
    return pd.DataFrame(rows, columns=["day", "label", "reference"])


def parse_reference_shapes(log_path):
    # Lines look like: "Loaded CSV ./lycos-ids2017/Monday-WorkingHours.pcap_lycos.csv - shape: [367181,83]"
    pat = re.compile(r"Loaded CSV \./lycos-ids2017/(\w+)-WorkingHours\.pcap_lycos\.csv - shape: \[(\d+),(\d+)\]")
    out = {}
    for line in log_path.read_text(encoding="utf-8", errors="replace").splitlines():
        m = pat.search(line)
        if m:
            out[m.group(1)] = (int(m.group(2)), int(m.group(3)))
    return out


def run_labelling(cfg, run_dir, log):
    paths = cfg["paths"]
    days = cfg["stage1"]["days"]
    labelled_dir = resolve(paths["labelled_dir"])
    targets = [labelled_dir / csv_name(d) for d in days]
    present = [t.exists() for t in targets]
    if all(present):
        log.info("All labelled CSVs already exist in data/lycos-ids2017/ - skipping labelling.py, verifying only.")
        return False
    if any(present):
        raise RuntimeError("Some labelled CSVs already exist in data/lycos-ids2017/; refusing to overwrite. "
                           f"Present: {[t.name for t, p in zip(targets, present) if p]}")

    work = resolve(paths["labelling_workdir"])
    if work.exists():
        raise RuntimeError(f"Work dir {work} already exists; remove it or change paths.labelling_workdir.")
    for sub in ("pcap_lycos", "lycos-ids2017", "python_logs"):
        (work / sub).mkdir(parents=True)

    # Unzip the LycoSTand CSVs into the work dir (data/pcap_lycos/ itself stays untouched).
    for day in days:
        zpath = resolve(paths["pcap_lycos_dir"]) / f"{day}-WorkingHours.pcap_lycos.zip"
        with zipfile.ZipFile(zpath) as z:
            z.extract(csv_name(day), work / "pcap_lycos")
        log.info(f"Extracted {zpath.name}")

    script = resolve(paths["labelling_script"])
    n_proc = int(cfg["labelling"]["n_processes"])
    log.info(f"Running {script} (unmodified) with cwd={work}, answering '{n_proc}' to the thread prompt")
    t0 = time.time()
    proc = subprocess.run([sys.executable, str(script)], cwd=work, input=f"{n_proc}\n",
                          capture_output=True, text=True)
    (run_dir / "labelling_stdout.txt").write_text(proc.stdout, encoding="utf-8")
    (run_dir / "labelling_stderr.txt").write_text(proc.stderr, encoding="utf-8")
    log.info(f"labelling.py exit code {proc.returncode} after {time.time() - t0:.1f} s")
    if proc.returncode != 0:
        raise RuntimeError(f"labelling.py failed, see {run_dir / 'labelling_stderr.txt'}")

    new_log = work / "python_logs" / "labelling_info.log"
    if new_log.exists():
        shutil.copy2(new_log, run_dir / "labelling_info_rerun.log")
    for day, target in zip(days, targets):
        src = work / "lycos-ids2017" / csv_name(day)
        if not src.exists():
            raise RuntimeError(f"labelling.py did not produce {src}")
        shutil.move(str(src), str(target))
        log.info(f"Moved {src.name} -> {target.relative_to(resolve('.'))}")
    if not cfg["labelling"]["keep_extracted_csv"]:
        shutil.rmtree(work)
        log.info(f"Removed work dir {work}")
    return True


def verify(cfg, run_dir, log):
    paths = cfg["paths"]
    labelled_dir = resolve(paths["labelled_dir"])
    ref = parse_reference_counts(resolve(paths["reference_labelling_log"]))
    ref_shapes = parse_reference_shapes(resolve(paths["reference_create_log"]))

    rows, shape_rows = [], []
    for day in cfg["stage1"]["days"]:
        path = labelled_dir / csv_name(day)
        n_cols = len(pd.read_csv(path, nrows=0, encoding="ISO-8859-1").columns)
        labels = pd.read_csv(path, usecols=["label"], encoding="ISO-8859-1")["label"]
        for lab, n in labels.value_counts().items():
            rows.append((day, lab, int(n)))
        ref_n, ref_c = ref_shapes.get(day, (None, None))
        shape_rows.append({"day": day, "label": "__TOTAL_ROWS__", "reference": ref_n,
                           "rerun": len(labels), "ref_cols": ref_c, "rerun_cols": n_cols})
        log.info(f"{day}: {len(labels)} rows, {n_cols} columns")

    got = pd.DataFrame(rows, columns=["day", "label", "rerun"])
    check = ref.merge(got, on=["day", "label"], how="outer")
    check = pd.concat([check, pd.DataFrame(shape_rows)], ignore_index=True)
    check["diff"] = check["rerun"] - check["reference"]
    check["match"] = check["diff"].eq(0)
    if "ref_cols" in check:
        cols_ok = check["ref_cols"].isna() | check["ref_cols"].eq(check["rerun_cols"])
        check["match"] = check["match"] & cols_ok
    out = RESULTS / "stage1_labelling_check.csv"
    check.to_csv(out, index=False)
    check.to_csv(run_dir / out.name, index=False)
    n_bad = int((~check["match"]).sum())
    log.info(f"Wrote {out.relative_to(resolve('.'))}: {len(check)} rows, {n_bad} mismatches")
    for _, r in check[~check["match"]].iterrows():
        log.warning(f"MISMATCH {r['day']} {r['label']}: reference={r['reference']} rerun={r['rerun']}")
    return n_bad, check


def main():
    cfg, overrides = load_config("Run data/labelling.py unmodified and verify its output.")
    t0 = time.time()
    run_dir, log = start_run("stage1a_labelling", cfg, overrides)
    ran = run_labelling(cfg, run_dir, log)
    n_bad, check = verify(cfg, run_dir, log)
    finish_run(run_dir, log, {"labelling_ran": ran, "mismatches": n_bad,
                              "total_rows": int(check.loc[check["label"] == "__TOTAL_ROWS__", "rerun"].sum())}, t0)
    sys.exit(1 if n_bad else 0)


if __name__ == "__main__":
    main()
