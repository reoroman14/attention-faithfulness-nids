# Stage 8a figures F1 to F5 (vector PDF) in the style of paper/make_figures.py (same palette, fonts and sizes; the
# palette is colour-blind validated, and identity never relies on colour alone: markers and line styles differ too).
# Read-only on results/ through paper/data_v2.py. The existing paper/make_figures.py is unchanged.
#
# Usage (from the project root): .venv\Scripts\python.exe paper\make_figures_v2.py [--png <dir>]
import argparse

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.colors import LinearSegmentedColormap
from matplotlib.patches import FancyArrowPatch, FancyBboxPatch

import data_v2 as D
from make_figures import AXIS, COL, FULL, INK, INK2, OUT, REF_DARK, REF_LIGHT, SLOTS, save, setup

YEARS = ("2017", "2018")
YLAB = {"2017": "LYCOS-IDS2017 (7 common classes)", "2018": "LycoS-Unicas-IDS2018"}
# attack -> (label, colour, marker, linestyle)
ATT = {"A0": ("A0 random control", REF_DARK, "s", ":"), "A1": ("A1 evasion", SLOTS[0], "o", "-"),
       "A1b": ("A1b targeted", SLOTS[1], "^", "--"), "A2": ("A2 attention only", SLOTS[2], "o", "-"),
       "A3_lam100": (r"A3 $\lambda=100$", SLOTS[6], "^", "--"), "A3_lam1000": (r"A3 $\lambda=1000$", SLOTS[7], "v", "--"),
       "A3norm_w0.5": ("A3 normalised", SLOTS[5], "D", "-.")}


# ---------------------------------------------------------------- F1 pipeline

def f1(png):
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
        "def": (8.08, 1.2, "Defenders D, D$_+$\nvs adaptive A6, A8\n(equal-effort union)"),
    }
    w, h = 1.9, 0.95
    for k, (x, y, t) in boxes.items():
        col = SLOTS[0] if k in ("clean", "faith") else SLOTS[1] if k in ("attack", "uatt") else SLOTS[2] if k == "def" else REF_DARK
        ax.add_patch(FancyBboxPatch((x, y), w, h, boxstyle="round,pad=0.02,rounding_size=0.08", fc="white", ec=col, lw=1.2))
        ax.text(x + w / 2, y + h / 2, t, ha="center", va="center", fontsize=6.0, color=INK)
    arrows = [("data", "model"), ("model", "clean"), ("model", "attack"), ("clean", "faith"), ("attack", "uatt"),
              ("faith", "def"), ("uatt", "def")]
    for a, b in arrows:
        xa, ya, _ = boxes[a]
        xb, yb, _ = boxes[b]
        ax.add_patch(FancyArrowPatch((xa + w, ya + h / 2), (xb, yb + h / 2), arrowstyle="-|>", mutation_scale=8,
                                     color=INK2, lw=0.9, shrinkA=2, shrinkB=2))
    save(fig, "f_pipeline", png)


# ---------------------------------------------------------------- F2 evasion vs eps

def f2(png):
    fig, axes = plt.subplots(1, 2, figsize=(FULL, 2.3), sharey=True)
    for ax, yr in zip(axes, YEARS):
        d, _ = D.success_long(yr)
        d = d[d["mode"] == "constrained"]
        for atk in ("A0", "A1", "A1b"):
            g = D.agg_seeds(d[d.attack == atk], ["eps"], "success").sort_values("eps")
            lab, col, mk, ls = ATT[atk]
            ax.plot(g.eps, g["mean"], color=col, marker=mk, ms=3.5, ls=ls, label=lab)
            ax.fill_between(g.eps, g.lo, g.hi, color=col, alpha=0.18, lw=0)
        ax.set_xscale("log")
        ax.set_xticks([0.05, 0.1, 0.25, 0.5, 1, 2], ["0.05", "0.1", "0.25", "0.5", "1", "2"])
        ax.set_xlabel(r"Relative budget $\epsilon$")
        ax.set_title(YLAB[yr], loc="left")
        ax.set_ylim(0, 1)
    axes[0].set_ylabel("Evasion success")
    axes[1].legend(loc="upper left")
    save(fig, "f_evasion_eps", png)


# ---------------------------------------------------------------- F3 mass ratio vs eps

def f3(png):
    fig, axes = plt.subplots(1, 3, figsize=(FULL, 2.3), sharey=True, gridspec_kw={"width_ratios": [1, 1, 0.8]})
    for ax, yr in zip(axes[:2], YEARS):
        d, _ = D.success_long(yr)
        d = d[(d["mode"] == "constrained") & (d.eps <= 1.0)]
        for atk in ("A2", "A3_lam100", "A3_lam1000", "A3norm_w0.5"):
            g = D.agg_seeds(d[d.attack == atk], ["eps"], "mass_ratio_rollout").sort_values("eps")
            lab, col, mk, ls = ATT[atk]
            ax.plot(g.eps, g["mean"], color=col, marker=mk, ms=3.5, ls=ls, label=lab)
            ax.fill_between(g.eps, g.lo, g.hi, color=col, alpha=0.18, lw=0)
        ax.set_xscale("log")
        ax.set_xticks([0.05, 0.1, 0.25, 0.5, 1], ["0.05", "0.1", "0.25", "0.5", "1"])
        ax.set_xlabel(r"Relative budget $\epsilon$ (constrained)")
        ax.set_title(YLAB[yr], loc="left")
    ax = axes[2]
    for yr, col, mk in (("2017", SLOTS[0], "o"), ("2018", SLOTS[1], "s")):
        d, _ = D.success_long(yr)
        for atk, ls in (("A2", "-"), ("A3_lam1000", "--")):
            g = D.agg_seeds(d[(d["mode"] == "unconstrained") & (d.attack == atk)], ["eps"], "mass_ratio_rollout").sort_values("eps")
            ax.plot(g.eps, g["mean"], color=col, marker=mk, ms=3.5, ls=ls, label=f"{yr} {ATT[atk][0].split(' ')[0]}"
                    + (r" $\lambda=1000$" if atk != "A2" else ""))
            ax.fill_between(g.eps, g.lo, g.hi, color=col, alpha=0.15, lw=0)
    ax.set_xlabel(r"Standardised $\epsilon$ (unconstrained)")
    ax.set_title("Unconstrained reference", loc="left")
    ax.legend(loc="lower left")
    for a in axes:
        a.axhline(1.0, color=AXIS, lw=0.8, zorder=0)
    axes[0].set_ylabel("Top-10 rollout mass ratio\n(adversarial / clean)")
    axes[0].set_ylim(0.6, 1.05)
    axes[1].legend(loc="lower left")
    save(fig, "f_mass_ratio", png)


# ---------------------------------------------------------------- F4 D vs D_plus TPR

def f4(png):
    attacks = ["A1"] + D.A6N + D.A8N
    fig, axes = plt.subplots(1, 2, figsize=(FULL, 2.4), sharey=True)
    lo_all = 1.0
    for ax, yr in zip(axes, YEARS):
        R, _ = D.defender_rows(yr)
        t = D.defender_table(R, 1.0, attacks).set_index("attack")
        x = np.arange(len(attacks))
        for dx, det, lab, col, mk in ((-0.13, "D", "D", SLOTS[0], "o"), (0.13, "Dplus", r"D$_+$", SLOTS[1], "s")):
            v = t.reindex(attacks)
            y, lo, hi = v[f"tpr_{det}"].to_numpy(), v[f"tpr_{det}_lo"].to_numpy(), v[f"tpr_{det}_hi"].to_numpy()
            ax.errorbar(x + dx, y, yerr=[y - lo, hi - y], fmt=mk, color=col, ms=4, capsize=2, lw=0.9, label=lab)
            lo_all = min(lo_all, np.nanmin(lo))
        ax.set_xticks(x, [D.ATTACK_NAMES[a].replace(r"$\lambda{=}", r"$\lambda=") for a in attacks], rotation=30, ha="right")
        ax.axvspan(0.5, 3.5, color=SLOTS[0], alpha=0.05, lw=0)
        ax.axvspan(3.5, 6.5, color=SLOTS[1], alpha=0.05, lw=0)
        a8 = sorted(int(s) for s in R[R.attack.isin(D.A8N)].seed.unique())
        partial = a8 != sorted(int(s) for s in R.seed.unique())
        ax.set_title(YLAB[yr] + (f" (A8: seeds {', '.join(map(str, a8))})" if partial else ""), loc="left")
    axes[0].set_ylim(max(0.0, lo_all - 0.05), 1.01)
    axes[0].set_ylabel(r"TPR at 5% FPR ($\epsilon = 1$)")
    axes[1].legend(loc="lower left")
    save(fig, "f_detection_tpr", png)


# ---------------------------------------------------------------- F5 per-class heatmap

def f5(png):
    cols, M, N = [], [], {}
    for yr in YEARS:
        d, _ = D.per_class_union(yr)
        d = d.set_index("class").reindex(D.COMMON)
        for det, lab in (("union_undet_D", "D"), ("union_undet_Dplus", r"D$_+$")):
            cols.append(f"{yr}\n{lab}")
            M.append(d[det].to_numpy())
        N[yr] = d["n"].to_numpy()
    M = np.array(M).T
    cmap = LinearSegmentedColormap.from_list("seq", ["#f7f7f4", "#9cc3ef", "#2a78d6", "#184f95"])
    fig, ax = plt.subplots(figsize=(COL, 2.9))
    ax.grid(False)
    vmax = max(0.05, float(np.nanmax(M)))
    im = ax.imshow(M, cmap=cmap, vmin=0, vmax=vmax, aspect="auto")
    for i in range(M.shape[0]):
        for j in range(M.shape[1]):
            v = M[i, j]
            ax.text(j, i, "n/a" if np.isnan(v) else f"{v:.2f}", ha="center", va="center", fontsize=6.2,
                    color="white" if v > 0.6 * vmax else INK)
    ylabels = [D.CLASS_NAMES[c] + ("*" if N["2018"][k] < 100 else "") for k, c in enumerate(D.COMMON)]
    ax.set_yticks(range(len(D.COMMON)), ylabels)
    ax.set_xticks(range(len(cols)), cols)
    ax.tick_params(length=0)
    for sp in ax.spines.values():
        sp.set_visible(False)
    ax.axvline(1.5, color="white", lw=2)
    cb = fig.colorbar(im, ax=ax, fraction=0.05, pad=0.03)
    cb.set_label("Success AND undetected\n(equal-effort union, $\\epsilon = 1$)")
    cb.outline.set_visible(False)
    seeds, partial = {}, False
    for yr in YEARS:
        R, _ = D.defender_rows(yr)
        a8 = sorted(int(s) for s in R[R.attack.isin(D.A8N)].seed.unique())
        partial |= a8 != sorted(int(s) for s in R.seed.unique())
        seeds[yr] = ", ".join(map(str, a8))
    txt = (f"Seeds: 2017 {seeds['2017']}; 2018 {seeds['2018']}"
           + (" (seeds with A8 runs)" if partial else "") + ". * fewer than 100 source rows per seed.")
    ax.text(0, -0.2, txt, transform=ax.transAxes, fontsize=6, color=INK2, va="top")
    save(fig, "f_per_class_heatmap", png)


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--png", default=None, help="also write PNG previews to this directory")
    args = ap.parse_args()
    OUT.mkdir(parents=True, exist_ok=True)
    setup()
    failed = []
    for fn in (f1, f2, f3, f4, f5):
        try:
            fn(args.png)
        except Exception as e:                       # report and continue with the other figures
            failed.append(f"{fn.__name__}: {type(e).__name__}: {e}")
            print(f"FAILED {fn.__name__}: {e}")
    if failed:
        print("\n".join(failed))
