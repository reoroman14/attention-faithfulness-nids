# Builds the two flat submission folders of the Scientific Reports version (Task 46):
#   C:\attn_submission_srep     main.tex, references.bib, f_pipeline.pdf, f_headline.pdf
#   C:\attn_submission_srep_si  supplementary.tex, references.bib and the 4 Supplementary figure PDFs
# Every \input is inlined, full-line comments are dropped, \graphicspath is dropped (figures sit next to the .tex), and
# the Nature-style reference list (numbered by first citation in that document) replaces the %%BIBLIOGRAPHY%% line.
# The Supplementary tables and figures are written in the order of srep_common.py, so their automatic numbering equals
# the numbering macros. Refuses to write if a target holds files it did not write, if a protected folder is the target,
# or if the flat text has an \input left, a non-ASCII character or an unescaped text-mode underscore, or if
# checks_srep.check() reports any violation. Files that will be overwritten are first copied to
# scratch\srep_backups\<time>.
#
# Usage (from the project root): .venv\Scripts\python.exe paper_srep\gen\build_srep.py
import hashlib
import re
import shutil
from datetime import datetime, timedelta, timezone

import srep_common as S
import make_bib_srep as B
import checks_srep
import underscore_check as U                      # paper_v3/gen/underscore_check.py

U.SAFE_ARG = U.SAFE_ARG + ("bibitem",)
MAIN_DIR = S.Path("C:/attn_submission_srep")
SI_DIR = S.Path("C:/attn_submission_srep_si")
PROTECTED = [S.Path(p) for p in ("C:/attn_submission_v3", "C:/attn_submission_cyse", "C:/attn_submission",
                                 "C:/attn_release_clean")]
BACKUP = S.ROOT / "scratch" / "srep_backups"
CAPTIONS = {"f_mass_ratio": "capMassRatio", "f_evasion_eps": "capEvasion", "f_detection_tpr": "capDetection",
            "f_per_class_heatmap": "capPerClass"}


def inline(text):
    def rep(m):
        f = S.SREP / m.group(1)
        if not f.suffix:
            f = f.with_suffix(".tex")
        return inline(f.read_text(encoding="utf-8")).rstrip("\n")
    return re.sub(r"\\input\{([^}]*)\}", rep, text)


def flatten(text):
    flat = inline(text)
    lines = [l for l in flat.splitlines() if not l.lstrip().startswith("%") or l.strip() == "%%BIBLIOGRAPHY%%"]
    lines = [l for l in lines if not l.startswith(r"\graphicspath")]
    out, blank = [], False
    for l in lines:
        if not l.strip():
            if blank:
                continue
            blank = True
        else:
            blank = False
        out.append(l)
    flat = "\n".join(out).strip() + "\n"
    bib = B.thebibliography(B.cite_order(flat), S.SREP / "references.bib")
    assert flat.count("%%BIBLIOGRAPHY%%") == 1
    flat = flat.replace("%%BIBLIOGRAPHY%%", bib)
    assert "\\input" not in flat
    assert all(ord(c) < 128 for c in flat), "non-ASCII character"
    us = U.find(flat)
    assert not us, f"unescaped text-mode underscores: {us}"
    return flat


def si_source():
    t = (S.SREP / "supplementary.tex").read_text(encoding="utf-8")
    tabs = "\n".join(f"\\input{{generated/si_{n}.tex}}\n\\clearpage" for n, _ in S.SUPP_TABLES)
    figs = "\n".join(
        "\\begin{figure}[p]\n\\centering\n"
        f"\\includegraphics[width=\\textwidth]{{{n}}}\n\\caption{{\\{CAPTIONS[n]}}}\n\\label{{{n}}}\n\\end{{figure}}\n\\clearpage"
        for n, _ in S.SUPP_FIGURES)
    t = t.replace("%%SUPPTABLES%%", tabs).replace("%%SUPPFIGURES%%", figs)
    return t.replace(r"\input{decisions.tex}", "\\input{decisions.tex}\n\\input{si/fig_captions.tex}")


def write(target, files):
    # files: {name: bytes}
    assert target.resolve() not in [p.resolve() for p in PROTECTED]
    target.mkdir(exist_ok=True)
    extra = [p.name for p in target.iterdir() if p.name not in files]
    assert not extra, f"{target} holds other files: {extra}"
    old = [p for p in target.iterdir() if p.is_file()]
    if old:
        stamp = datetime.now(timezone(timedelta(hours=3))).strftime("%Y%m%d_%H%M%S")
        bdir = BACKUP / stamp / target.name
        bdir.mkdir(parents=True, exist_ok=True)
        for p in old:
            shutil.copy2(p, bdir / p.name)
        print(f"backup of {len(old)} files: {bdir}")
    for name, data in files.items():
        (target / name).write_bytes(data)
    for p in sorted(target.iterdir()):
        print(f"{target.name}\\{p.name}  {hashlib.md5(p.read_bytes()).hexdigest()}  {p.stat().st_size} bytes")


def main():
    bib = (S.SREP / "references.bib").read_bytes()
    main_flat = flatten((S.SREP / "main.tex").read_text(encoding="utf-8"))
    si_flat = flatten(si_source())
    bad = checks_srep.check(main_flat, si_flat)
    assert not bad, f"build stopped, {len(bad)} check violations: {bad}"
    files = {"main.tex": main_flat.encode("ascii"), "references.bib": bib}
    for n, _ in S.MAIN_FIGURES:
        files[f"{n}.pdf"] = (S.GEN / f"{n}.pdf").read_bytes()
    write(MAIN_DIR, files)
    files = {"supplementary.tex": si_flat.encode("ascii"), "references.bib": bib}
    for n, _ in S.SUPP_FIGURES:
        files[f"{n}.pdf"] = (S.GEN / f"{n}.pdf").read_bytes()
    write(SI_DIR, files)


if __name__ == "__main__":
    main()
