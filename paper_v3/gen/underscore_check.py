# Unescaped underscores in text mode (Task 43). A bare "_" outside math gives "Missing $ inserted".
# Removed before the search: comments; math ($...$, $$...$$, \(...\), \[...\], math environments, \ensuremath{...});
# the arguments of \label, \ref, \eqref, \pageref, \cite*, \url, \href (first argument), \input, \include,
# \includegraphics, \bibliography, \bibliographystyle, \begin, \end; \verb|...|. \texttt is NOT exempt: its argument
# is text mode, so an underscore there must be written \_ as well. Escaped underscores (\_) are allowed.
import re

SAFE_ARG = ("label", "ref", "eqref", "pageref", "autoref", "cref", "Cref", "url", "href", "input", "include",
            "includegraphics", "bibliography", "bibliographystyle", "begin", "end", "cite", "citep", "citet",
            "citealp", "citeauthor", "citeyear")
MATH_ENVS = ("equation", "equation*", "align", "align*", "gather", "gather*", "multline", "multline*", "math",
             "displaymath", "eqnarray", "eqnarray*")


def _blank(s, i, j):
    """Replace s[i:j] by spaces, keeping newlines so that line numbers stay valid."""
    return s[:i] + re.sub(r"[^\n]", " ", s[i:j]) + s[j:]


def _brace_end(s, i):
    """s[i] == '{'; index after the matching '}'."""
    d = 0
    k = i
    while k < len(s):
        c = s[k]
        if c == "\\":
            k += 2
            continue
        if c == "{":
            d += 1
        elif c == "}":
            d -= 1
            if d == 0:
                return k + 1
        k += 1
    return len(s)


def strip(t):
    t = re.sub(r"(?<!\\)%[^\n]*", lambda m: " " * len(m.group(0)), t)
    t = re.sub(r"\\verb(.)(.*?)\1", lambda m: " " * len(m.group(0)), t)
    for env in MATH_ENVS:
        e = re.escape(env)
        t = re.sub(r"\\begin\{" + e + r"\}.*?\\end\{" + e + r"\}", lambda m: re.sub(r"[^\n]", " ", m.group(0)), t,
                   flags=re.S)
    for pat in (r"\$\$.*?\$\$", r"\\\[.*?\\\]", r"\\\(.*?\\\)", r"(?<!\\)\$(?:\\.|[^$\\])*?\$"):
        t = re.sub(pat, lambda m: re.sub(r"[^\n]", " ", m.group(0)), t, flags=re.S)
    # \ensuremath{...} and the arguments of the safe commands (optional [..] arguments skipped)
    names = "|".join(re.escape(n) for n in SAFE_ARG + ("ensuremath",))
    pos = 0
    rx = re.compile(r"\\(" + names + r")\*?(?![A-Za-z])\s*(\[[^\]]*\]\s*)?")
    while True:
        m = rx.search(t, pos)
        if not m:
            break
        k = m.end()
        if k < len(t) and t[k] == "{":
            j = _brace_end(t, k)
            t = _blank(t, m.start(), j)
            pos = j
        else:
            pos = k
    return t


def find(t):
    """List of (line number, line text) for each unescaped text-mode underscore."""
    s = strip(t)
    lines = t.splitlines()
    out = []
    for m in re.finditer(r"(?<!\\)_", s):
        ln = s[:m.start()].count("\n") + 1
        out.append((ln, lines[ln - 1].strip()[:160]))
    return out
