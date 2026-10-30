# Builds the flat submission folder C:\attn_submission_v3 from paper_v3: one main.tex with every \input inlined
# (numbers, sections, tables), references.bib and the figure PDFs, side by side, nothing else. Full-line comments are
# dropped; \graphicspath is dropped because the figures sit next to main.tex. Only files that main.tex references are
# copied. Refuses to write anywhere else and stops if the target holds files it did not write, or if the flat file has
# an unescaped underscore in text mode (underscore_check.py).
#
# Usage (from the project root): .venv\Scripts\python.exe paper_v3\gen\build.py
import hashlib
import re
import shutil
from pathlib import Path

from common import V3
import underscore_check

TARGET = Path("C:/attn_submission_v3")
PROTECTED = [Path(p) for p in ("C:/attn_submission_cyse", "C:/attn_submission", "C:/attn_release_clean")]


def inline(text, base):
    def rep(m):
        f = base / m.group(1)
        if not f.suffix:
            f = f.with_suffix(".tex")
        return inline(f.read_text(encoding="utf-8"), base).rstrip("\n")
    return re.sub(r"\\input\{([^}]*)\}", rep, text)


def main():
    assert TARGET.resolve() not in [p.resolve() for p in PROTECTED]
    flat = inline((V3 / "main.tex").read_text(encoding="utf-8"), V3)
    lines = [l for l in flat.splitlines() if not l.lstrip().startswith("%")]
    lines = [l for l in lines if not l.startswith(r"\graphicspath")]
    out, blank = [], False
    for l in lines:                                    # collapse runs of blank lines
        if not l.strip():
            if blank:
                continue
            blank = True
        else:
            blank = False
        out.append(l)
    flat = "\n".join(out).strip() + "\n"
    us = underscore_check.find(flat)
    assert not us, f"unescaped text-mode underscores (Missing $ inserted): {us}"
    figs = sorted(set(re.findall(r"\\includegraphics(?:\[[^]]*\])?\{([^}]*)\}", flat)))
    TARGET.mkdir(exist_ok=True)
    allowed = {"main.tex", "references.bib"} | {f + ".pdf" for f in figs}
    extra = [p.name for p in TARGET.iterdir() if p.name not in allowed]
    assert not extra, f"target holds other files: {extra}"
    (TARGET / "main.tex").write_bytes(flat.encode("ascii"))
    shutil.copyfile(V3 / "references.bib", TARGET / "references.bib")
    for f in figs:
        shutil.copyfile(V3 / "generated" / f"{f}.pdf", TARGET / f"{f}.pdf")
    for p in sorted(TARGET.iterdir()):
        print(f"{p.name}  {hashlib.md5(p.read_bytes()).hexdigest()}  {p.stat().st_size} bytes")


if __name__ == "__main__":
    main()
