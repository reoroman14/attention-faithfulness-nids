# References of the Scientific Reports version (Task 46).
#   1. write_bib(): paper_srep/references.bib = paper_v3/references.bib without the entries in srep_common.REF_CUTS
#      (each checked first for an affiliation, note or any text linking it to asu.edu.jo; a linked entry is kept), and
#      with the repository entry pointing to the address given in the named version of the earlier submission
#      (C:\attn_submission\delete\main.tex) instead of the anonymisation placeholder.
#   2. thebibliography(): a Nature-style reference list (numbered by first citation, at most five authors, otherwise
#      the first author and "et al.") written as LaTeX into the flat document, so no .bst, .cls or .bbl file is needed.
#      Used by build_srep.py for main.tex and supplementary.tex separately.
#
# Usage (from the project root): .venv\Scripts\python.exe paper_srep\gen\make_bib_srep.py   (writes references.bib)
import re
from pathlib import Path

import srep_common as S

NAMED = Path("C:/attn_submission/delete/main.tex")
ASU = re.compile(r"asu\.edu\.jo|applied science (private )?university", re.I)


def entries(text):
    # [(type, key, raw, {field: value})] in file order; values keep their inner braces.
    out = []
    for m in re.finditer(r"@(\w+)\{([^,\s]+),", text):
        i, d = text.index("{", m.start()), 0
        for j in range(i, len(text)):
            d += {"{": 1, "}": -1}.get(text[j], 0)
            if d == 0:
                break
        raw = text[m.start():j + 1]
        fields, body, k = {}, text[m.end():j], 0
        for f in re.finditer(r"(\w+)\s*=\s*", body):
            if f.start() < k:
                continue
            v0 = f.end()
            if body[v0] == "{":
                dd = 0
                for e in range(v0, len(body)):
                    dd += {"{": 1, "}": -1}.get(body[e], 0)
                    if dd == 0:
                        break
                fields[f.group(1).lower()] = body[v0 + 1:e]
                k = e + 1
            else:
                e = body.find(",", v0)
                e = len(body) if e < 0 else e
                fields[f.group(1).lower()] = body[v0:e].strip().strip('"')
                k = e
        out.append((m.group(1).lower(), m.group(2), raw, fields))
    return out


def repo_url():
    m = re.search(r"repository address is \\url\{([^}]*)\}", NAMED.read_text(encoding="utf-8"))
    return m.group(1)


def write_bib():
    src = (S.V3 / "references.bib").read_text(encoding="utf-8")
    report, keep = [], []
    for typ, key, raw, f in entries(src):
        if key in S.REF_CUTS:
            if ASU.search(raw):
                report.append(f"{key}: linked to asu.edu.jo in the entry, KEPT")
                keep.append(raw)
            else:
                report.append(f"{key}: no affiliation, note or text linking it to asu.edu.jo; cut")
            continue
        if key == "repo_attn_nids":
            raw = raw.replace("{Link withheld for double-anonymous review}", "{\\url{" + repo_url() + "}}")
        keep.append(raw)
    (S.SREP / "references.bib").write_text("\n\n".join(keep) + "\n", encoding="utf-8")
    (S.GEN / "ref_cuts_check.txt").write_text("\n".join(report) + "\n", encoding="utf-8")
    print("\n".join(report))
    print(f"wrote references.bib: {len(keep)} entries")


# ---------------------------------------------------------------- Nature style

def _initials(given):
    out = []
    for tok in given.split():
        parts = tok.split("-")
        ini = []
        for p in parts:
            if p.startswith("{"):
                d = 0
                for e, c in enumerate(p):
                    d += {"{": 1, "}": -1}.get(c, 0)
                    if d == 0:
                        break
                ini.append(p[:e + 1] + ".")
            elif p:
                ini.append(p[0] + ".")
        out.append("-".join(ini))
    return " ".join(out)


def _name(n):
    n = n.strip()
    if n.startswith("{") and n.endswith("}") and "," not in n:
        return n                                            # corporate author
    if "," in n:
        last, given = [x.strip() for x in n.split(",", 1)]
    else:
        toks = n.split()
        last, given = toks[-1], " ".join(toks[:-1])
    return f"{last}, {_initials(given)}".rstrip(", ")


def authors(s):
    names = [_name(a) for a in re.split(r"\s+and\s+", s.strip())]
    if len(names) > 5:
        return names[0] + " et al."
    if len(names) == 1:
        return names[0]
    return ", ".join(names[:-1]) + " \\& " + names[-1]


def _clean(v):
    v = re.sub(r"\s+", " ", v).strip()
    return v.replace("--", "-")


def fmt(typ, f):
    au = authors(f["author"]) + " " if "author" in f else ""
    title = _clean(f["title"]).rstrip(".")
    sep = " " if title.endswith(("?", "!")) else ". "     # no full stop after a title ending in ? or !
    year = f.get("year", "")
    pages = _clean(f.get("pages", ""))
    if typ == "article":
        vol = f" \\textbf{{{f['volume']}}}" if "volume" in f else ""
        pg = f", {pages}" if pages else ""
        return f"{au}{title}{sep}\\emph{{{_clean(f['journal'])}}}{vol}{pg} ({year})."
    if typ == "inproceedings":
        pg = f" {pages}" if pages else ""
        pub = f"{_clean(f['publisher'])}, " if "publisher" in f else ""
        return f"{au}{title}{sep}In \\emph{{{_clean(f['booktitle'])}}}{pg} ({pub}{year})."
    extra = []
    if "eprint" in f:
        extra.append(f"Preprint at arXiv:{f['eprint']}")
    if "howpublished" in f:
        extra.append(_clean(f["howpublished"]))
    if "note" in f and "eprint" not in f:
        extra.append(_clean(f["note"]))
    return f"{au}{title}{sep}{'. '.join(extra)} ({year})."


def thebibliography(order, bibfile):
    ents = {k: (t, f) for t, k, _, f in entries(Path(bibfile).read_text(encoding="utf-8"))}
    missing = [k for k in order if k not in ents]
    assert not missing, missing
    lines = [r"\begin{thebibliography}{99}"]
    for k in order:
        lines.append(f"\\bibitem{{{k}}} {fmt(*ents[k])}")
    lines.append(r"\end{thebibliography}")
    return "\n".join(lines)


def cite_order(text):
    text = "\n".join(re.sub(r"(?<!\\)%.*$", "", l) for l in text.splitlines())
    order = []
    for m in re.finditer(r"\\cite[a-zA-Z]*\{([^}]*)\}", text):
        for k in m.group(1).split(","):
            k = k.strip()
            if k not in order:
                order.append(k)
    return order


if __name__ == "__main__":
    S.GEN.mkdir(parents=True, exist_ok=True)
    write_bib()
