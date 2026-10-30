# Static checks of the restructured manuscript (Task 42); no LaTeX installation is needed.
#   1. paper/lint_numbers.py, applied to paper_v3/sections (same classification of numeric literals; same \ref check
#      against the labels in sections, main.tex and generated/t_*.tex; same dash check).
#   2. On the flat C:\attn_submission_v3\main.tex: non-ASCII characters, em / en dashes and "--", unresolved \ref,
#      duplicate \label, \cite keys missing from references.bib, bib entries never cited, macros that are used but
#      neither defined in the file nor used in the compiled CYSE main.tex (candidates for undefined control sequences),
#      figure files, wording that the task forbids, unescaped underscores in text mode (underscore_check.py; also run
#      on every file in sections, generated and main.tex).
# Exit code 1 if anything is flagged.
#
# Usage (from the project root): .venv\Scripts\python.exe paper_v3\gen\checks_v3.py
import re
import sys
from pathlib import Path

from common import OLD, V3

import lint_numbers as L                                # paper/lint_numbers.py, imported; its paths are redirected
import underscore_check

TARGET = Path("C:/attn_submission_v3")
CYSE = Path("C:/attn_submission_cyse/main.tex")
FORBIDDEN = ["complementary signal", "slightly fewer", "at most small", "helped little", "much less", "robust",
             "defence", "defense", "resists", "survives"]
BRACE_ARG_OK = {"begin", "end"}


def lint_sections():
    L.P = V3
    L.SECTIONS = sorted((V3 / "sections").glob("*.tex"))
    flags, params, years, seeds = L.scan()
    unresolved, used = L.refs()
    dd = L.dashes()
    print(f"[lint] sections: {', '.join(f.name for f in L.SECTIONS)}")
    print(f"[lint] math-mode parameters (allowed, listed): {len(params)}")
    for f, i, n, ctx in params:
        print(f"         {f}:{i}: {n} in {ctx}")
    print(f"[lint] year labels {len(years)}; seed indices {len(seeds)}")
    print(f"[lint] FLAGGED numeric literals outside macros: {len(flags)}")
    for f, i, n, ctx in flags:
        print(f"         {f}:{i}: {n} | {ctx}")
    print(f"[lint] references used {len(used)}; unresolved {len(unresolved)}: {unresolved}")
    print(f"[lint] em / en dashes: {len(dd)}")
    return len(flags) + len(unresolved) + len(dd)


def flat_checks():
    t = (TARGET / "main.tex").read_text(encoding="ascii")
    bad = 0
    nonascii = [c for c in t if ord(c) > 127]
    dashes = re.findall(r"--+", t)
    print(f"[flat] lines {len(t.splitlines())}; non-ASCII characters {len(nonascii)}; '--' or longer {len(dashes)}")
    bad += len(nonascii) + len(dashes)
    labels = re.findall(r"\\label\{([^}]*)\}", t)
    dup = sorted({l for l in labels if labels.count(l) > 1})
    refs = set(re.findall(r"\\ref\{([^}]*)\}", t))
    unres = sorted(refs - set(labels))
    print(f"[flat] labels {len(labels)} (duplicates {dup}); \\ref keys {len(refs)}, unresolved {unres}")
    bad += len(dup) + len(unres)
    cites = set()
    for m in re.finditer(r"\\cite[a-zA-Z]*\{([^}]*)\}", t):
        cites |= {c.strip() for c in m.group(1).split(",")}
    bib = (TARGET / "references.bib").read_text(encoding="utf-8")
    keys = re.findall(r"@\w+\{([^,\s]+),", bib)
    missing = sorted(cites - set(keys))
    unused = sorted(set(keys) - cites)
    print(f"[flat] \\cite keys {len(cites)}; missing from bib {missing}; bib entries {len(keys)}; never cited {unused}")
    bad += len(missing)
    defined = set(re.findall(r"\\newcommand\{\\([A-Za-z]+)\}", t))
    used = set(re.findall(r"\\([A-Za-z]+)", t))
    cyse = set(re.findall(r"\\([A-Za-z]+)", CYSE.read_text(encoding="utf-8")))
    unknown = sorted(used - defined - cyse)
    print(f"[flat] macros defined {len(defined)}; used control words {len(used)}; not defined here and not used in the "
          f"compiled CYSE file: {unknown}")
    figs = re.findall(r"\\includegraphics(?:\[[^]]*\])?\{([^}]*)\}", t)
    nofile = [f for f in figs if not (TARGET / f"{f}.pdf").exists()]
    print(f"[flat] figures {figs}; missing files {nofile}")
    bad += len(nofile)
    us = underscore_check.find(t)
    print(f"[flat] unescaped text-mode underscores: {len(us)}")
    for ln, l in us:
        print(f"         line {ln}: {l}")
    bad += len(us)
    for f in sorted((V3 / "sections").glob("*.tex")) + sorted((V3 / "generated").glob("*.tex")) + [V3 / "main.tex"]:
        for ln, l in underscore_check.find(f.read_text(encoding="utf-8")):
            print(f"         source {f.relative_to(V3)}:{ln}: {l}")
            bad += 1
    hits = []
    for w in FORBIDDEN:
        for m in re.finditer(w, t, re.I):
            line = t[:m.start()].count("\n") + 1
            hits.append((w, line, t.splitlines()[line - 1].strip()[:150]))
    print(f"[flat] forbidden wording hits: {len(hits)}")
    for h in hits:
        print(f"         '{h[0]}' line {h[1]}: {h[2]}")
    content = sorted(p.name for p in TARGET.iterdir())
    print(f"[flat] folder content: {content}; subfolders: {[p.name for p in TARGET.iterdir() if p.is_dir()]}")
    for k in ("Declaration", "Acknowledg"):
        print(f"[flat] '{k}' occurrences: {len(re.findall(k, t, re.I))}")
    auth = re.findall(r"\\author(?:\[[^]]*\])?\{[^}]*\}", t)
    print(f"[flat] author commands: {auth} (anonymised if the only one is an empty \\author{{}})")
    bad += 0 if auth == ["\\author{}"] else 1
    return bad, hits, unknown


if __name__ == "__main__":
    n = lint_sections()
    b, hits, unknown = flat_checks()
    sys.exit(1 if n or b else 0)
