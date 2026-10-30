# Checks of the Scientific Reports version (Task 46). check(main_flat, si_flat) is called by build_srep.py before
# anything is written (the build fails on any violation); run as a script it checks the built folders.
#   Limits of the journal: main text <= 4,500 words (Introduction, Results, Discussion), abstract <= 200, title <= 20
#   words and one sentence without a colon, <= 8 display items, each legend <= 350 words, no colour in tables, no
#   footnotes, reference count (about 55 expected).
#   Consistency: every \ref and \cite resolves; numbering macros agree with the order of the tables, figures and first
#   citations; the caveats of plan 42 section 3.2 are in the main text.
#   Text rules: underscore check, no em / en dashes or "--", ASCII only, number lint (paper/lint_numbers.py on
#   paper_srep/sections and si), forbidden wording.
# Word counting rules (as in handoff/42_srep_plan.md section 1.1): comments, floats, \input, \label and \cite dropped;
# each \ref, macro use and inline math group counts as one word; tokens without a letter or digit are not counted.
#
# Usage (from the project root): .venv\Scripts\python.exe paper_srep\gen\checks_srep.py
import re
import sys

import srep_common as S
import underscore_check as U
from common import OLD                            # noqa: F401  (puts paper/ on sys.path)
import lint_numbers as L

U.SAFE_ARG = tuple(dict.fromkeys(U.SAFE_ARG + ("bibitem",)))
SEC = S.SREP / "sections"
LIMITS = {"main": 4500, "abstract": 200, "title": 20, "items": 8, "legend": 350}
REF_EXPECTED = 55
FORBIDDEN = ["complementary signal", "slightly fewer", "at most small", "helped little", "much less", "robust",
             "defence", "defense", "resists", "survives", "pre-registered", "preregistered", "significant",
             "changed no row"]
# Caveats that must stay in the main text (plan 42, section 3.2): (name, regex over the main-text sources).
CAVEATS = [("three seeds, opposite-sign seeds", r"\\axHeadOppSeedsPhrase\{\} of the opposite sign"),
           ("intervals within seed, no seed variance", r"do not capture (variation between seeds|seed variance)"),
           ("surrogate gradients for the rank scores", r"surrogate gradients[^.]*\\nRankScores"),
           ("a1 and a2 stand-in attackers", r"a1 and a2[^.]*stand-in"),
           ("b not attacked and why", r"b was not attacked adaptively[^.]*memory of the graphics card"),
           ("2018 floor", r"near a floor|close to the floor"),
           ("A6 versus A8x optimiser strength not measured", r"optimiser strength remains unmeasured|"
                                                           r"strong an attack on D as A8x is on D\$_\+\$ was not measured"),
           ("no mechanism tested", r"did not test why D\$_\+\$")]


def strip_comments(t):
    return "\n".join(re.sub(r"(?<!\\)%.*$", "", l) for l in t.splitlines())


def words(t):
    t = strip_comments(t)
    for env in ("figure", "figure*", "table", "table*"):
        e = re.escape(env)
        t = re.sub(r"\\begin\{" + e + r"\}.*?\\end\{" + e + r"\}", " ", t, flags=re.S)
    t = re.sub(r"\\input\{[^}]*\}", " ", t)
    t = re.sub(r"\\(label|cite[a-z]*)\{[^}]*\}", " ", t)
    t = re.sub(r"~?\\ref\{[^}]*\}", " REF", t)
    t = re.sub(r"\$[^$]*\$", " M ", t)
    t = re.sub(r"\\(section|subsection|subsubsection|paragraph)\*?\{([^}]*)\}", r" \2 ", t)
    t = re.sub(r"\\(begin|end)\{[^}]*\}", " ", t)
    t = re.sub(r"\\item(\[[^]]*\])?", " ", t)
    t = re.sub(r"\\[A-Za-z]+(\{\})?", " MACRO ", t)
    t = re.sub(r"[{}\\~]", " ", t)
    return len([w for w in t.split() if re.search(r"[A-Za-z0-9]", w)])


def brace_arg(t, i):
    d = 0
    for k in range(i, len(t)):
        d += {"{": 1, "}": -1}.get(t[k], 0) if t[k - 1] != "\\" else 0
        if d == 0:
            return t[i + 1:k]
    return t[i + 1:]


def counts():
    c = {}
    for name in ("intro", "results", "discussion", "methods", "abstract"):
        t = (SEC / f"{name}.tex").read_text(encoding="utf-8")
        c[name] = words(t)
        if name == "results":
            parts = re.split(r"\\subsection\*\{", strip_comments(t))
            c["results: opening"] = words(parts[0])
            for i, p in enumerate(parts[1:], 1):
                c[f"results: subsection {i}"] = words("\\subsection*{" + p)
    c["main text"] = c["intro"] + c["results"] + c["discussion"]
    return c


def legends():
    out = []
    t = strip_comments((SEC / "legends.tex").read_text(encoding="utf-8"))
    for part in re.split(r"\\noindent\\textbf\{Figure~", t)[1:]:
        name = part.split(".", 1)[0]
        out.append((f"Figure {name}", words(part)))
    for n, mac in S.MAIN_TABLES:
        tt = (S.GEN / f"main_{n}.tex").read_text(encoding="utf-8")
        cap = brace_arg(tt, tt.index(r"\caption{") + len(r"\caption"))
        note = re.search(r"\\parbox\{\\linewidth\}\{\\scriptsize (.*)\}\s*\\end\{table\}", tt, re.S)
        out.append((f"Table {n}", words(cap) + (words(note.group(1)) if note else 0)))
    return out


def title_check():
    t = re.search(r"\\newcommand\{\\paperTitle\}\{(.*)\}", (S.SREP / "decisions.tex").read_text(encoding="utf-8")).group(1)
    n1, n2 = len(t.split()), len(t.replace("-", " ").split())
    one_sentence = not re.search(r"[:;.?!]", t)
    return t, n1, n2, one_sentence


def structure(flat):
    # Static stand-ins for a compile (no LaTeX on this machine): unescaped braces balance, every \begin{env} is closed
    # by the matching \end{env} in order, every \newcommand name is defined once, and $ signs pair up per paragraph.
    t = strip_comments(flat)
    t = re.sub(r"\\verb(.)(.*?)\1", " ", t)
    problems = []
    depth = 0
    for i, ch in enumerate(t):
        if ch in "{}" and (i == 0 or t[i - 1] != "\\" or (i > 1 and t[i - 2] == "\\")):
            depth += 1 if ch == "{" else -1
            if depth < 0:
                problems.append(f"unmatched }} at line {t[:i].count(chr(10)) + 1}")
                depth = 0
    if depth:
        problems.append(f"{depth} unclosed {{")
    stack = []
    for m in re.finditer(r"\\(begin|end)\{([^}]*)\}", t):
        if m.group(1) == "begin":
            stack.append((m.group(2), t[:m.start()].count("\n") + 1))
        elif not stack or stack[-1][0] != m.group(2):
            problems.append(f"\\end{{{m.group(2)}}} at line {t[:m.start()].count(chr(10)) + 1} does not match "
                            f"{stack[-1] if stack else 'nothing'}")
        else:
            stack.pop()
    problems += [f"unclosed \\begin{{{e}}} at line {ln}" for e, ln in stack]
    names = re.findall(r"\\newcommand\{\\([A-Za-z]+)\}", t)
    problems += [f"\\{n} defined twice" for n in sorted({n for n in names if names.count(n) > 1})]
    for k, para in enumerate(re.split(r"\n\s*\n", t)):
        if len(re.findall(r"(?<!\\)\$", para)) % 2:
            problems.append(f"odd number of $ in paragraph starting: {para.strip()[:60]!r}")
    return problems


def lint():
    L.P = S.SREP
    L.SECTIONS = sorted(SEC.glob("*.tex")) + sorted((S.SREP / "si").glob("*.tex"))
    flags, params, years, seeds = L.scan()
    dd = L.dashes()
    return flags, params, dd


def check(main_flat, si_flat, verbose=True):
    bad = []
    say = print if verbose else (lambda *a: None)
    c = counts()
    say("[words] " + "; ".join(f"{k} {v}" for k, v in c.items()))
    if c["main text"] > LIMITS["main"]:
        bad.append(f"main text {c['main text']} > {LIMITS['main']}")
    if c["abstract"] > LIMITS["abstract"]:
        bad.append(f"abstract {c['abstract']} > {LIMITS['abstract']}")
    t, n1, n2, one = title_check()
    say(f"[title] {n1} words ({n2} with hyphens split); one sentence without colon: {one}")
    if max(n1, n2) > LIMITS["title"] or not one:
        bad.append("title")
    lg = legends()
    say("[legends] " + "; ".join(f"{k} {v}" for k, v in lg))
    bad += [f"legend {k} {v} > {LIMITS['legend']}" for k, v in lg if v > LIMITS["legend"]]
    items = len(re.findall(r"\\includegraphics", main_flat)) + len(re.findall(r"\\begin\{table\*?\}", main_flat))
    say(f"[items] display items in main.tex: {items}")
    if items > LIMITS["items"]:
        bad.append(f"display items {items}")
    for name, flat in (("main", main_flat), ("si", si_flat)):
        tables = re.findall(r"\\begin\{table\*?\}.*?\\end\{table\*?\}", flat, re.S)
        col = [x for x in tables if re.search(r"\\(cellcolor|rowcolor|textcolor|color)\b", x)]
        foot = re.findall(r"\\(footnote|thanks)\b", flat)
        labels = re.findall(r"\\label\{([^}]*)\}", flat)
        dup = sorted({x for x in labels if labels.count(x) > 1})
        refs = set(re.findall(r"\\ref\{([^}]*)\}", flat))
        unres = sorted(refs - set(labels))
        cites = set()
        for m in re.finditer(r"\\cite[a-zA-Z]*\{([^}]*)\}", flat):
            cites |= {k.strip() for k in m.group(1).split(",")}
        items_b = re.findall(r"\\bibitem\{([^}]*)\}", flat)
        miss = sorted(cites - set(items_b))
        nonascii = sum(ord(ch) > 127 for ch in flat)
        dashes = len(re.findall(r"--+", flat)) + flat.count("\u2014") + flat.count("\u2013")
        us = U.find(flat)
        inputs = flat.count("\\input")
        balance = structure(flat)
        say(f"[{name}] brace and environment structure: {balance or 'balanced'}")
        if balance:
            bad.append(f"{name}: structure {balance}")
        say(f"[{name}] tables {len(tables)}, with colour {len(col)}; footnotes {len(foot)}; labels {len(labels)} "
            f"(duplicates {dup}); refs {len(refs)}, unresolved {unres}; cites {len(cites)}, missing from list {miss}; "
            f"references {len(items_b)}; non-ASCII {nonascii}; dashes {dashes}; underscores {len(us)}; \\input {inputs}")
        for ln, l in us:
            say(f"         line {ln}: {l}")
        bad += [f"{name}: {k}" for k, v in (("colour", col), ("footnotes", foot), ("duplicate labels", dup),
                                             ("unresolved refs", unres), ("missing cites", miss), ("underscores", us))
                if v]
        bad += [f"{name}: {k}" for k, v in (("non-ASCII", nonascii), ("dashes", dashes), ("input", inputs)) if v]
        # Wording rules apply to the authors' text, not to the titles of cited works in the reference list.
        own = re.sub(r"\\begin\{thebibliography\}.*?\\end\{thebibliography\}",
                     lambda mm: "\n" * mm.group(0).count("\n"), flat, flags=re.S)
        hits = [(w, own[:m.start()].count("\n") + 1) for w in FORBIDDEN for m in re.finditer(w, own, re.I)]
        say(f"[{name}] forbidden wording hits: {hits}")
        bad += [f"{name}: wording {h}" for h in hits]
        if name == "main":
            say(f"[refs] references in main.tex: {len(items_b)} (expected {REF_EXPECTED})")
            if len(items_b) != REF_EXPECTED:
                bad.append(f"reference count {len(items_b)}")
            order = re.findall(r"\\label\{(t_[^}]*)\}", flat)
            if order != [n for n, _ in S.MAIN_TABLES]:
                bad.append(f"main table order {order}")
            figs = re.findall(r"\\includegraphics(?:\[[^]]*\])?\{([^}]*)\}", flat)
            if figs != [n for n, _ in S.MAIN_FIGURES]:
                bad.append(f"main figure order {figs}")
            body = flat[flat.index("\\section*{Introduction}"):flat.index("\\begin{thebibliography}")]
            for group in (S.MAIN_TABLES, S.MAIN_FIGURES):
                first = [body.find("\\" + mac) for _, mac in group]
                if -1 in first or first != sorted(first):
                    bad.append(f"first citation order of {[m for _, m in group]}: {first}")
            say(f"[numbering] main tables {order}; figures {figs}")
        else:
            order = re.findall(r"\\label\{(t_[^}]*)\}", flat[flat.index("Supplementary Tables"):])
            figs = re.findall(r"\\includegraphics(?:\[[^]]*\])?\{([^}]*)\}", flat)
            if order != [n for n, _ in S.SUPP_TABLES]:
                bad.append(f"SI table order {order}")
            if figs != [n for n, _ in S.SUPP_FIGURES]:
                bad.append(f"SI figure order {figs}")
            say(f"[numbering] SI tables {len(order)} in srep_common order: {order == [n for n, _ in S.SUPP_TABLES]}; "
                f"figures {figs}")
    src = "\n".join(strip_comments((SEC / f"{n}.tex").read_text(encoding="utf-8")) for n in
                    ("intro", "results", "discussion"))
    for name, rx in CAVEATS:
        ok = re.search(rx, src) is not None
        say(f"[caveat] {name}: {'present' if ok else 'MISSING'}")
        if not ok:
            bad.append(f"caveat {name}")
    flags, params, dd = lint()
    say(f"[lint] math-mode parameters {len(params)}; FLAGGED numeric literals {len(flags)}; dashes {len(dd)}")
    for f, i, n, ctx in flags:
        say(f"         {f}:{i}: {n} | {ctx}")
    bad += [f"lint {f}:{i}: {n}" for f, i, n, _ in flags] + [f"dash {x}" for x in dd]
    say(f"[result] violations: {len(bad)}")
    for b in bad:
        say(f"         {b}")
    return bad


if __name__ == "__main__":
    m = (S.Path("C:/attn_submission_srep") / "main.tex").read_text(encoding="ascii")
    s = (S.Path("C:/attn_submission_srep_si") / "supplementary.tex").read_text(encoding="ascii")
    for d, want in ((S.Path("C:/attn_submission_srep"), {"main.tex", "references.bib"} |
                     {f"{n}.pdf" for n, _ in S.MAIN_FIGURES}),
                    (S.Path("C:/attn_submission_srep_si"), {"supplementary.tex", "references.bib"} |
                     {f"{n}.pdf" for n, _ in S.SUPP_FIGURES})):
        have = {p.name for p in d.iterdir()}
        sub = [p.name for p in d.iterdir() if p.is_dir()]
        print(f"[folder] {d}: {sorted(have)}; subfolders {sub}; as required: {have == want and not sub}")
        if have != want or sub:
            sys.exit(1)
    sys.exit(1 if check(m, s) else 0)
