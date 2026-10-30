# Figures of the Scientific Reports version (Task 46), vector PDF in paper_srep/generated.
#   - f_evasion_eps, f_mass_ratio, f_detection_tpr, f_per_class_heatmap, f_pipeline: paper_v3/gen/make_figures_v3.py
#     (imported, output redirected; code unchanged).
#   - f_headline: the v3 headline figure with one more row, the A6x comparison (D+ under A1 + A8x minus D under
#     A1 + A6x). Rows 1 and 2 read results/stage9/a8x/summary_union.csv, row 3 results/stage9/a6x/summary_union.csv.
# Read-only on results/.
#
# Usage (from the project root): .venv\Scripts\python.exe paper_srep\gen\make_figures_srep.py [--png <dir>]
import argparse

import matplotlib.pyplot as plt
import pandas as pd

import srep_common as S
import make_figures_v3 as F                       # paper_v3/gen/make_figures_v3.py, output redirected by srep_common

mf, mf2 = F.mf, F.mf2
SLOTS, INK, INK2, REF_DARK, FULL = mf.SLOTS, mf.INK, mf.INK2, mf.REF_DARK, mf.FULL


def series(yr):
    # (label, colour, marker, pooled row, seed rows) with columns diff, lo, hi.
    a8 = pd.read_csv(F.A8X / "summary_union.csv")
    a6 = pd.read_csv(S.ROOT / "results" / "stage9" / "a6x" / "summary_union.csv")
    a8 = a8[(a8.rowset == "V1") & (a8.year == yr)].rename(columns={"Dplus_minus_D": "diff"})
    a6 = a6[(a6.rowset == "V1") & (a6.year == yr)].rename(columns={"second_minus_first": "diff"})
    out = []
    for df, lab, name, col, mk in ((a8, "Stage A:", "A8 (stand-ins); D faces A6", SLOTS[0], "o"),
                                   (a8, "headline:", "A8x (exact objective); D faces A6", SLOTS[1], "D"),
                                   (a6, "headline:", "A8x (exact objective); D faces A6x", SLOTS[2], "s")):
        x = df[df.union.str.startswith(lab)]
        assert x.union.nunique() == 1, (yr, lab)
        p, s = x[x.seeds == "pooled"].iloc[0], x[x.seeds != "pooled"].sort_values("seeds")
        assert len(s) == 3
        out.append((name, col, mk, p, s))
    return out


def headline(png):
    fig, axes = plt.subplots(1, 2, figsize=(FULL, 2.45), sharey=True)
    n = 3
    for ax, yr, title in zip(axes, (2017, 2018), ("LYCOS-IDS2017 (V1)", "LycoS-Unicas-IDS2018")):
        rows = series(yr)
        for i, (name, col, mk, p, s) in enumerate(rows):
            y = n - 1 - i
            ax.plot([p.lo, p.hi], [y, y], color=col, lw=2.0, solid_capstyle="butt", zorder=2)
            ax.scatter([p["diff"]], [y], s=42, marker=mk, color=col, edgecolor="white", linewidth=1.2, zorder=4)
            ax.scatter(s["diff"], [y - 0.22] * len(s), s=20, marker=mk, facecolor="white", edgecolor=col,
                       linewidth=1.0, zorder=3)
            groups = []
            for _, r in s.sort_values("diff").iterrows():
                if groups and r["diff"] - groups[-1][-1][0] < 0.008:
                    groups[-1].append((r["diff"], r.seeds.replace("seed ", "s")))
                else:
                    groups.append([(r["diff"], r.seeds.replace("seed ", "s"))])
            for g in groups:
                ax.text(sum(v for v, _ in g) / len(g), y - 0.40, ", ".join(sorted(k for _, k in g)), ha="center",
                        va="top", fontsize=5.3, color=INK2)
            ax.text(p["diff"], y + 0.12, f"{p['diff']:+.3f} [{p.lo:+.3f}, {p.hi:+.3f}]", ha="center", va="bottom",
                    fontsize=6.0, color=INK)
        ax.axvline(0, color=REF_DARK, lw=0.8, ls=":", zorder=1)
        ax.set_title(title)
        ax.set_yticks(list(range(n - 1, -1, -1)))
        ax.set_yticklabels([r[0] for r in rows])
        ax.set_ylim(-0.75, n - 0.45)
        ax.set_xlabel(r"D$_+$ minus D, successful and undetected share")
        ax.grid(axis="y", visible=False)
    lo = min(a.get_xlim()[0] for a in axes)
    hi = max(a.get_xlim()[1] for a in axes)
    for a in axes:
        a.set_xlim(lo, hi)
    mf.save(fig, "f_headline", png)


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--png", default=None)
    args = ap.parse_args()
    S.GEN.mkdir(parents=True, exist_ok=True)
    mf.setup()
    for fn in (mf2.f2, mf2.f3, mf2.f4, mf2.f5):
        fn(args.png)
    F.pipeline(args.png)
    headline(args.png)
