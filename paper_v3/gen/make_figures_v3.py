# Figures of the restructured manuscript (Task 42), vector PDF in paper_v3/generated, in the style of
# paper/make_figures.py (same palette, fonts and sizes).
#   - f_evasion_eps, f_mass_ratio, f_detection_tpr, f_per_class_heatmap: paper/make_figures_v2.py, imported and run
#     unchanged with the output redirected.
#   - f_pipeline: as in paper/make_figures_v2.py f1, with the attackers added in this version.
#   - f_headline (new, central figure): D+ minus D under the equal-effort union, V1, eps 1, per dataset, for the
#     stand-in attacker A8 and the exact-objective attacker A8x; pooled value with the 95% interval and the three
#     per-seed values. Read from results/stage9/a8x/summary_union.csv.
# Read-only on results/.
#
# Usage (from the project root): .venv\Scripts\python.exe paper_v3\gen\make_figures_v3.py [--png <dir>]
import argparse

import matplotlib.pyplot as plt
import pandas as pd
from matplotlib.patches import FancyArrowPatch, FancyBboxPatch

from common import A8X, GEN, old_figures

mf, mf2 = old_figures()
SLOTS, INK, INK2, REF_DARK, FULL, AXIS = mf.SLOTS, mf.INK, mf.INK2, mf.REF_DARK, mf.FULL, mf.AXIS


def pipeline(png):
    fig, ax = plt.subplots(figsize=(FULL, 2.3))
    ax.set_axis_off()
    ax.set_xlim(0, 10)
    ax.set_ylim(0, 3.2)
    boxes = {
        "data": (0.0, 1.2, "Datasets\nLYCOS-IDS2017,\nLycoS-Unicas-\nIDS2018"),
        "model": (2.02, 1.2, "Models\nFT-Transformer\n(3 seeds), XGBoost"),
        "clean": (4.04, 2.15, "Clean explanations\nattention vs IG,\nGradientSHAP,\nocclusion, TreeSHAP"),
        "attack": (4.04, 0.25, "Constrained attacks\nA0, A1, A1b, A2, A3\n(projection + checker)"),
        "faith": (6.06, 2.15, "Agreement and\nfaithfulness\n(deletion, insertion)"),
        "uatt": (6.06, 0.25, "Faithfulness\nunder attack\n(mass ratio, margins)"),
        "def": (8.08, 1.2, "Detectors D, D$_+$,\ncontrols vs A6, A6x,\nA8, A8x, own attackers\n(equal-effort union)"),
    }
    w, h = 1.9, 0.95
    for k, (x, y, t) in boxes.items():
        col = SLOTS[0] if k in ("clean", "faith") else SLOTS[1] if k in ("attack", "uatt") else SLOTS[2] if k == "def" else REF_DARK
        ax.add_patch(FancyBboxPatch((x, y), w, h, boxstyle="round,pad=0.02,rounding_size=0.08", fc="white", ec=col, lw=1.2))
        ax.text(x + w / 2, y + h / 2, t, ha="center", va="center", fontsize=6.0, color=INK)
    for a, b in [("data", "model"), ("model", "clean"), ("model", "attack"), ("clean", "faith"), ("attack", "uatt"),
                 ("faith", "def"), ("uatt", "def")]:
        xa, ya, _ = boxes[a]
        xb, yb, _ = boxes[b]
        ax.add_patch(FancyArrowPatch((xa + w, ya + h / 2), (xb, yb + h / 2), arrowstyle="-|>", mutation_scale=8,
                                     color=INK2, lw=0.9, shrinkA=2, shrinkB=2))
    mf.save(fig, "f_pipeline", png)


def headline(png):
    U = pd.read_csv(A8X / "summary_union.csv")
    U = U[U.rowset == "V1"]
    rows = [("Stage A:", "A8 (stand-ins)", SLOTS[0], "o"), ("headline:", "A8x (exact objective)", SLOTS[1], "D")]
    fig, axes = plt.subplots(1, 2, figsize=(FULL, 1.75), sharey=True)
    for ax, yr, title in zip(axes, (2017, 2018), ("LYCOS-IDS2017 (V1)", "LycoS-Unicas-IDS2018")):
        for i, (lab, name, col, mk) in enumerate(rows):
            x = U[(U.year == yr) & U.union.str.startswith(lab)]
            p, s = x[x.seeds == "pooled"].iloc[0], x[x.seeds != "pooled"].sort_values("seeds")
            y = 1 - i
            ax.plot([p.lo, p.hi], [y, y], color=col, lw=2.0, solid_capstyle="butt", zorder=2)
            ax.scatter([p.Dplus_minus_D], [y], s=42, marker=mk, color=col, edgecolor="white", linewidth=1.2, zorder=4,
                       label="pooled, 95% CI" if i == 0 else None)
            ax.scatter(s.Dplus_minus_D, [y - 0.22] * len(s), s=20, marker=mk, facecolor="white", edgecolor=col,
                       linewidth=1.0, zorder=3)
            # Seed labels; seeds closer than 0.008 share one label so that labels do not overlap.
            groups = []
            for _, r in s.sort_values("Dplus_minus_D").iterrows():
                if groups and r.Dplus_minus_D - groups[-1][-1][0] < 0.008:
                    groups[-1].append((r.Dplus_minus_D, r.seeds.replace("seed ", "s")))
                else:
                    groups.append([(r.Dplus_minus_D, r.seeds.replace("seed ", "s"))])
            for g in groups:
                ax.text(sum(v for v, _ in g) / len(g), y - 0.42, ", ".join(sorted(n for _, n in g)), ha="center",
                        va="top", fontsize=5.5, color=INK2)
            ax.text(p.Dplus_minus_D, y + 0.13, f"{p.Dplus_minus_D:+.3f} [{p.lo:+.3f}, {p.hi:+.3f}]", ha="center", va="bottom",
                    fontsize=6.2, color=INK)
        ax.axvline(0, color=REF_DARK, lw=0.8, ls=":", zorder=1)
        ax.set_title(title)
        ax.set_yticks([1, 0])
        ax.set_yticklabels([r[1] for r in rows])
        ax.set_ylim(-0.75, 1.5)
        ax.set_xlabel(r"D$_+$ minus D, successful and undetected share")
        ax.grid(axis="y", visible=False)
    lo = min(a.get_xlim()[0] for a in axes)
    hi = max(a.get_xlim()[1] for a in axes)
    for a in axes:
        a.set_xlim(lo, hi)
    axes[1].text(0.99, -0.42, "filled: pooled with 95% CI; open: seeds 0, 1, 2 (s0, s1, s2)\n"
                 "right of 0: D$_+$ leaves more rows undetected", transform=axes[1].transAxes, ha="right", va="top",
                 fontsize=5.8, color=INK2)
    mf.save(fig, "f_headline", png)


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--png", default=None)
    args = ap.parse_args()
    GEN.mkdir(parents=True, exist_ok=True)
    mf.setup()
    for fn in (mf2.f2, mf2.f3, mf2.f4, mf2.f5):
        fn(args.png)
    pipeline(args.png)
    headline(args.png)
