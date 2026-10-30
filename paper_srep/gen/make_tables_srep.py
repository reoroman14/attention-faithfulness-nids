# Tables of the Scientific Reports version (Task 46), written to paper_srep/generated.
#   - Every table of paper_v3/gen/make_tables_v3.py is regenerated (imported, output redirected; code unchanged) and
#     compared with paper_v3/generated (generated/v3_tables_check.txt).
#   - t_attn_flip (main Table 2): t_attn_attack and t_flip merged as panels a and b of one table, built from the same
#     rows that the two v3 functions produce.
#   - Cross-references in table notes are rewritten for the document that prints the table: in main.tex the v3
#     appendices become Supplementary Notes and Methods; in supplementary.tex the main-text tables and sections become
#     "of the main text". Results: generated/main_<name>.tex and generated/si_<name>.tex.
#   - No colour is used in any table.
# Read-only on results/, paper/ and paper_v3/.
#
# Usage (from the project root): .venv\Scripts\python.exe paper_srep\gen\make_tables_srep.py
import re

import srep_common as S
import make_tables_v3 as T                        # paper_v3/gen/make_tables_v3.py, output redirected by srep_common

V3_TABLES = ["t_data", "t_clean", "t_attn_attack", "t_flip", "t_attacks", "t_union_adaptive", "t_per_class_v1",
             "t_neardup_compact", "t_union_a8", "t_a8x_setting", "t_a8x_tpr", "t_runs", "t_standin", "t_impl"]

MAIN_REPL = [(r"Appendix~\ref{sec:supp}", r"Supplementary Note~\suppNoteResults"),
             (r"Appendix~\ref{sec:sens}", r"Supplementary Note~\suppNoteNeardup"),
             (r"Section~\ref{sec:sens}", r"Supplementary Note~\suppNoteNeardup"),
             (r"Section~\ref{sec:methods}", "Methods"),
             (r"(Section~\ref{sec:limitations})", "(Methods)")]
SI_REPL = [(r"Appendix~\ref{sec:supp}", r"Supplementary Note~\suppNoteResults"),
           (r"Appendix~\ref{sec:sens}", r"Supplementary Note~\suppNoteNeardup"),
           (r"Section~\ref{sec:sens}", r"Supplementary Note~\suppNoteNeardup"),
           (r"(Section~\ref{sec:a6x})", "(Methods of the main text)"),
           (r"Table~\ref{t_attacks}", r"Table~\mainTabAttacks{} of the main text"),
           (r"Table~\ref{t_union_adaptive}", r"Table~\mainTabUnion{} of the main text")]


def attn_flip():
    # Capture the arguments of the two v3 tables, then write one table with two panels.
    cap = {}
    orig = T.write

    def grab(name, caption, label, colspec, header, rows, sources, notes=(), **kw):
        cap[name] = dict(caption=caption, colspec=colspec, header=header, rows=rows, sources=sources, notes=notes)
    T.write = grab
    try:
        T.t_attn_attack()
        T.t_flip()
    finally:
        T.write = orig
    a, b = cap["t_attn_attack"], cap["t_flip"]
    ha = [a["header"]] if isinstance(a["header"][0], str) else a["header"]
    panels = [(r"a, " + a["caption"].rstrip("."), a["colspec"], ha, a["rows"]),
              (r"b, " + b["caption"].rstrip("."), b["colspec"], b["header"], b["rows"])]
    notes = ["a: " + " ".join(a["notes"]), "b: " + " ".join(b["notes"])]
    T.write("t_attn_flip", "Attention under protocol-constrained attacks (a) and faithfulness after an evasion by row "
            "status (b), V1.", "t_attn_flip", None, None, None, a["sources"] + b["sources"], notes,
            panels=panels, resize="shrink", tabcolsep=4)


def rewrite(name, repl, prefix):
    t = (S.GEN / f"{name}.tex").read_text(encoding="utf-8")
    for old, new in repl:
        t = t.replace(old, new)
    left = re.findall(r"\\ref\{(sec:[^}]*)\}", t)
    assert not left, (name, left)
    (S.GEN / f"{prefix}_{name}.tex").write_text(t, encoding="utf-8")


def check_v3():
    lines = []
    for name in V3_TABLES + T.OLD_KEEP:
        a = [l for l in (S.GEN / f"{name}.tex").read_text(encoding="utf-8").splitlines() if not l.startswith("% Generated")]
        b = [l for l in (S.V3 / "generated" / f"{name}.tex").read_text(encoding="utf-8").splitlines()
             if not l.startswith("% Generated")]
        lines.append(f"{name}.tex: {'identical to paper_v3' if a == b else 'DIFFERENT from paper_v3'}")
    (S.GEN / "v3_tables_check.txt").write_text("\n".join(lines) + "\n", encoding="utf-8")
    print(f"v3 tables regenerated: {len(lines)}; identical: {sum('identical' in l for l in lines)}")
    return all("identical" in l for l in lines)


if __name__ == "__main__":
    S.GEN.mkdir(parents=True, exist_ok=True)
    T.old()
    for fn in V3_TABLES:
        getattr(T, fn)()
    assert check_v3()
    attn_flip()
    for name, _ in S.MAIN_TABLES:
        rewrite(name, MAIN_REPL, "main")
    for name, _ in S.SUPP_TABLES:
        rewrite(name, SI_REPL, "si")
    colour = [p.name for p in S.GEN.glob("*_t_*.tex") if re.search(r"\\(cellcolor|rowcolor|textcolor|color)\b",
                                                                     p.read_text(encoding="utf-8"))]
    assert not colour, colour
    print(f"wrote {len(S.MAIN_TABLES)} main tables and {len(S.SUPP_TABLES)} Supplementary tables; colour in tables: 0")
