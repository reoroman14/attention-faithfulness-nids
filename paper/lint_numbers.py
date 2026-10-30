# Lints paper/sections/*.tex:
#   1. every numeric literal outside macros, \ref / \label / \cite / \citeTODO / \input / \texttt / \reviewcheck arguments,
#      the column count of \multicolumn, a Claude model name with its version, and comments, classified as math-mode parameter (allowed, listed for review),
#      dataset year label (2017 / 2018), seed index, or FLAG (must be a macro if it is a result);
#   2. every \ref resolves against \label in the sections, main.tex and generated/t_*.tex;
#   3. no em dash (U+2014, or --- in LaTeX) and no en dash character.
# Exit code 1 if anything is flagged.
#
# Usage (from the project root): .venv\Scripts\python.exe paper\lint_numbers.py
import re
import sys
from pathlib import Path

P = Path(__file__).resolve().parent
SECTIONS = sorted((P / "sections").glob("*.tex"))
SKIP_ARGS = r"\\(?:ref|label|cite[a-zA-Z]*|citeTODO|input|texttt|reviewcheck|includegraphics|TODO)\*?\{[^{}]*\}"
NUM = re.compile(r"(?<![A-Za-z0-9_\\.])(\d+(?:[.,]\d+)*)(?![A-Za-z0-9])")
MATH = re.compile(r"\$[^$]*\$|\\\(.*?\\\)")


def strip_comments(line):
    return re.sub(r"(?<!\\)%.*$", "", line)


def scan():
    flags, params, labels_years, seeds = [], [], [], []
    for f in SECTIONS:
        for i, raw in enumerate(f.read_text(encoding="utf-8").splitlines(), 1):
            line = strip_comments(raw)
            line = re.sub(SKIP_ARGS, " ", line)
            line = re.sub(r"\\multicolumn\{\d+\}\{[^{}]*\}", " ", line)       # table structure, not a number
            line = re.sub(r"Claude (?:Sonnet|Opus|Haiku) \d+(?:\.\d+)*", " ", line)  # model name (AI declaration)
            line = re.sub(r"\\[A-Za-z]+", " ", line)                   # macro names (letters only)
            for m in MATH.finditer(line):
                for n in NUM.finditer(m.group(0)):
                    params.append((f.name, i, n.group(1), m.group(0).strip()))
            text = MATH.sub(" ", line)
            for n in NUM.finditer(text):
                tok, start = n.group(1), n.start(1)
                before = text[max(0, start - 40):start]
                if tok in ("2017", "2018"):
                    labels_years.append((f.name, i, tok))
                elif re.search(r"seeds?\s+(?:\d+\s*,\s*)*(?:\d+\s+and\s+)?$", before) or re.search(r"seed\s*$", before):
                    seeds.append((f.name, i, tok))
                else:
                    flags.append((f.name, i, tok, raw.strip()[:140]))
    return flags, params, labels_years, seeds


def refs():
    used, defined = {}, set()
    for f in SECTIONS + [P / "main.tex"] + sorted((P / "generated").glob("t_*.tex")):
        t = "\n".join(strip_comments(l) for l in f.read_text(encoding="utf-8").splitlines())
        defined |= set(re.findall(r"\\label\{([^}]*)\}", t))
        if f.parent.name == "sections":
            for r in re.findall(r"\\ref\{([^}]*)\}", t):
                used.setdefault(r, set()).add(f.name)
    return {r: sorted(fs) for r, fs in used.items() if r not in defined}, sorted(used)


def dashes():
    out = []
    for f in SECTIONS:
        for i, l in enumerate(f.read_text(encoding="utf-8").splitlines(), 1):
            if "\u2014" in l or "\u2013" in l or re.search(r"(?<!-)---(?!-)", strip_comments(l)):
                out.append((f.name, i, l.strip()[:100]))
    return out


def main():
    flags, params, years, seeds = scan()
    unresolved, used = refs()
    dd = dashes()
    print(f"Scanned {len(SECTIONS)} files: {', '.join(f.name for f in SECTIONS)}")
    print(f"\nMath-mode parameters (allowed; review): {len(params)}")
    for f, i, n, ctx in params:
        print(f"  {f}:{i}: {n}   in {ctx}")
    print(f"\nDataset year labels (allowed): {len(years)}; seed indices (allowed): {len(seeds)}")
    for f, i, n in seeds:
        print(f"  seed index {f}:{i}: {n}")
    print(f"\nFLAGGED numeric literals outside macros: {len(flags)}")
    for f, i, n, ctx in flags:
        print(f"  {f}:{i}: {n}   | {ctx}")
    print(f"\nReferences used: {len(used)}; unresolved: {len(unresolved)}")
    for r, fs in unresolved.items():
        print(f"  UNRESOLVED \\ref{{{r}}} in {', '.join(fs)}")
    print(f"\nEm / en dashes: {len(dd)}")
    for f, i, l in dd:
        print(f"  {f}:{i}: {l}")
    sys.exit(1 if flags or unresolved or dd else 0)


if __name__ == "__main__":
    main()
