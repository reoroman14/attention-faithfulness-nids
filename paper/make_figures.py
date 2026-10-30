# Generates paper figures (PDF) from results/ into paper/generated/. Read-only on results/.
#
# Colour: the validated reference categorical palette, slots taken in fixed order; each method keeps
# the same colour in every figure (colour follows the entity). Three slots (aqua, yellow, magenta) sit
# below 3:1 contrast on white, so identity never relies on colour alone: every figure has a legend,
# and line style encodes the method family (attention = solid, attribution = dashed,
# references = grey dotted / dash-dot). Diverging heatmap: blue <-> red with a grey midpoint at 0.
# Sizes target a two-column journal: column width 3.5 in, full width 7.1 in, 7 to 8 pt text.
#
# Usage (from the project root): .venv\Scripts\python.exe paper\make_figures.py [--png <dir>]
#   --png also writes PNG previews (for checking layout; not used by the paper).
import argparse
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402
from matplotlib.colors import LinearSegmentedColormap  # noqa: E402

ROOT = Path(__file__).resolve().parents[1]
RESULTS = ROOT / "results"
OUT = Path(__file__).resolve().parent / "generated"
H = "macro_attack_classes"
SEEDS = [0, 1, 2]

# Reference categorical palette (light mode), fixed slot order.
SLOTS = ["#2a78d6", "#eb6834", "#1baf7a", "#eda100", "#e87ba4", "#008300", "#4a3aa7", "#e34948"]
INK, INK2, MUTED, GRID, AXIS = "#0b0b0b", "#52514e", "#898781", "#e1e0d9", "#c3c2b7"
REF_DARK, REF_LIGHT = "#52514e", "#a8a69f"

# method -> (label, colour, linestyle)
STYLE = {
    "attn_last": ("Attention, last layer", SLOTS[0], "-"),
    "attn_mean": ("Attention, layer mean", SLOTS[1], "-"),
    "attn_rollout": ("Attention rollout", SLOTS[2], "-"),
    "ig_zero": ("IG, train mean", SLOTS[3], "--"),
    "gshap": ("GradientSHAP", SLOTS[4], "--"),
    "occlusion": ("Occlusion", SLOTS[5], "--"),
    "ig_benign": ("IG, benign", SLOTS[6], "--"),
    "gradattn_rollout": ("Gradient-weighted attn.", SLOTS[7], "-"),
    "treeshap_xgb": ("TreeSHAP (XGB)", INK2, (0, (3, 1, 1, 1))),
    "abs_x": ("|x| reference", REF_DARK, ":"),
    "random": ("Random", REF_LIGHT, "-."),
    "ig_zero_signed": ("IG, train mean (signed)", SLOTS[3], ":"),
}
CLASS_NAMES = {
    "bot": "Bot", "ddos": "DDoS", "dos_goldeneye": "GoldenEye", "dos_hulk": "Hulk",
    "dos_slowhttptest": "Slowhttptest", "dos_slowloris": "Slowloris", "ftp_patator": "FTP-Patator",
    "portscan": "PortScan", "ssh_patator": "SSH-Patator", "webattack": "Web attack",
}
COL, FULL = 3.5, 7.1


def setup():
    plt.rcParams.update({
        "font.size": 7.5, "axes.titlesize": 8, "axes.labelsize": 7.5, "xtick.labelsize": 7,
        "ytick.labelsize": 7, "legend.fontsize": 6.8, "axes.edgecolor": AXIS, "axes.linewidth": 0.6,
        "axes.labelcolor": INK, "xtick.color": INK2, "ytick.color": INK2, "text.color": INK,
        "axes.spines.top": False, "axes.spines.right": False, "axes.grid": True, "grid.color": GRID,
        "grid.linewidth": 0.5, "axes.axisbelow": True, "lines.linewidth": 1.4, "legend.frameon": False,
        "pdf.fonttype": 42, "ps.fonttype": 42, "savefig.bbox": "tight", "savefig.pad_inches": 0.02,
        "figure.facecolor": "white", "axes.facecolor": "white",
    })


def save(fig, name, png_dir):
    fig.savefig(OUT / f"{name}.pdf")
    if png_dir:
        fig.savefig(Path(png_dir) / f"{name}.png", dpi=200)
    plt.close(fig)
    print(f"wrote generated/{name}.pdf")


# ---------------------------------------------------------------- deletion / insertion curves

def curves(png_dir):
    sel = pd.read_parquet(RESULTS / "stage3" / "selection.parquet")
    y = sel["class"].to_numpy()
    attack = [c for c in CLASS_NAMES]
    methods = ["attn_last", "attn_mean", "attn_rollout", "ig_zero", "gshap", "occlusion", "abs_x", "random"]
    for base in ("train_mean", "benign_sample"):
        # Macro over attack classes of the per-class mean curve, then mean over seeds.
        mc = {m: {k: [] for k in ("del_morf", "ins_morf")} for m in methods}
        for s in SEEDS:
            z = np.load(RESULTS / "stage3" / f"curves_seed{s}.npz")
            for m in methods:
                for k in ("del_morf", "ins_morf"):
                    cur = z[f"{base}/{m}/{k}"].astype(np.float64)
                    mc[m][k].append(np.mean([cur[y == c].mean(0) for c in attack], axis=0))
        fig, axes = plt.subplots(1, 2, figsize=(FULL, 2.3), sharey=True)
        for ax, k, title in ((axes[0], "del_morf", "Deletion (most relevant first)"),
                             (axes[1], "ins_morf", "Insertion (most relevant first)")):
            for m in methods:
                c = np.mean(mc[m][k], axis=0)
                lab, col, ls = STYLE[m]
                ax.plot(np.arange(len(c)), c, color=col, ls=ls, label=lab)
            ax.set_title(title, loc="left")
            ax.set_xlabel("Features masked / inserted (of 72)")
            ax.set_xlim(0, 72)
            ax.set_ylim(0, 1.02)
            ax.set_xticks([0, 12, 24, 36, 48, 60, 72])
        axes[0].set_ylabel("Target-class probability")
        axes[1].legend(loc="center left", bbox_to_anchor=(1.01, 0.5))
        save(fig, f"fig_deletion_insertion_{base}", png_dir)


# ---------------------------------------------------------------- agreement heatmap

def heatmap(png_dir):
    a = pd.concat([pd.read_csv(RESULTS / "stage3_agreement.csv"),
                   pd.read_csv(RESULTS / "stage3c_gradattn_agreement.csv")], ignore_index=True)
    a = a[(a["class"] == H) & (a.metric == "spearman")]
    v = a.groupby(["method_a", "method_b"])["mean"].mean()
    order = ["attn_last", "attn_mean", "attn_rollout", "gradattn_rollout", "ig_zero", "ig_benign", "gshap",
             "occlusion", "treeshap_xgb", "abs_x", "random"]
    n = len(order)
    M = np.full((n, n), np.nan)
    for i, p in enumerate(order):
        for j, q in enumerate(order):
            if i == j:
                M[i, j] = 1.0
            elif (p, q) in v.index:
                M[i, j] = M[j, i] = v[(p, q)]
            elif (q, p) in v.index:
                M[i, j] = M[j, i] = v[(q, p)]
    M[np.triu_indices(n, 1)] = np.nan          # lower triangle only
    cmap = LinearSegmentedColormap.from_list("div", ["#c4302f", "#e34948", "#f0efec", "#2a78d6", "#184f95"])
    cmap.set_bad("white")
    fig, ax = plt.subplots(figsize=(COL, 3.25))
    ax.grid(False)
    im = ax.imshow(M, cmap=cmap, vmin=-1, vmax=1)
    for i in range(n):
        for j in range(i + 1):
            val = M[i, j]
            t = "0.00" if abs(val) < 0.005 else f"{val:.2f}"
            ax.text(j, i, t, ha="center", va="center", fontsize=5.6, color="white" if abs(val) > 0.6 else INK)
    labels = [STYLE[m][0] for m in order]
    ax.set_xticks(range(n), labels, rotation=55, ha="right")
    ax.set_yticks(range(n), labels)
    for sp in ax.spines.values():
        sp.set_visible(False)
    ax.tick_params(length=0)
    cb = fig.colorbar(im, ax=ax, fraction=0.04, pad=0.02)
    cb.set_label("Spearman (macro over attack classes)")
    cb.outline.set_visible(False)
    save(fig, "fig_agreement_heatmap", png_dir)


# ---------------------------------------------------------------- parameter randomisation

def randomisation(png_dir):
    r = pd.read_csv(RESULTS / "stage3_randomisation.csv")
    r = r[(r["class"] == H) & (r.metric == "spearman_vs_original")]
    v = r.groupby(["method", "level"])["mean"].mean()
    casc = ["cascade_top1", "cascade_top2", "cascade_top3", "cascade_top3_tokenizer"]
    casc_lab = ["Original", "B3", "B3, B2", "B3 to B1", "+ tokenizer"]
    indep = ["indep_block1", "indep_block2", "indep_block3"]
    methods = ["attn_last", "attn_mean", "attn_rollout", "ig_zero", "ig_zero_signed"]
    fig, axes = plt.subplots(1, 2, figsize=(FULL, 2.2), gridspec_kw={"width_ratios": [1.5, 1]}, sharey=True)
    for m in methods:
        lab, col, ls = STYLE[m]
        lab = lab + (" (abs.)" if m == "ig_zero" else "")
        ys = [1.0] + [v[(m, lv)] for lv in casc]
        axes[0].plot(range(len(ys)), ys, color=col, ls=ls, marker="o", ms=3.5, label=lab)
        axes[1].plot(range(3), [v[(m, lv)] for lv in indep], color=col, ls="none", marker="o", ms=4.5)
    axes[0].set_xticks(range(len(casc_lab)), casc_lab)
    axes[0].set_title("Cascading randomisation, from the top (head kept)", loc="left")
    axes[0].set_ylabel("Spearman with original ranking")
    axes[1].set_xticks(range(3), ["B1 only", "B2 only", "B3 only"])
    axes[1].set_xlim(-0.5, 2.5)
    axes[1].set_title("Independent block randomisation", loc="left")
    for ax in axes:
        ax.axhline(0, color=AXIS, lw=0.8, zorder=0)
        ax.set_ylim(-0.2, 1.05)
    axes[1].legend(*axes[0].get_legend_handles_labels(), loc="center left", bbox_to_anchor=(1.01, 0.5))
    save(fig, "fig_randomisation", png_dir)


# ---------------------------------------------------------------- per-class bars

def per_class(png_dir):
    f = pd.read_csv(RESULTS / "stage3_faithfulness.csv")
    f = f[(f.mask_baseline == "train_mean") & f["class"].isin(CLASS_NAMES)]
    v = f.groupby(["metric", "method", "class"])["mean"].mean()
    methods = ["attn_rollout", "ig_zero", "occlusion", "abs_x", "random"]
    classes = list(CLASS_NAMES)
    x = np.arange(len(classes))
    w = 0.16
    fig, axes = plt.subplots(2, 1, figsize=(FULL, 3.6), sharex=True)
    for ax, metric, ylab in ((axes[0], "del_auc", "Deletion AUC\n(lower is better)"),
                             (axes[1], "aopc_gap", "AOPC gap\n(higher is better)")):
        for i, m in enumerate(methods):
            lab, col, _ = STYLE[m]
            ax.bar(x + (i - 2) * w, [v[(metric, m, c)] for c in classes], width=w * 0.88, color=col, label=lab,
                   edgecolor="white", linewidth=0.4)
        ax.set_ylabel(ylab)
        ax.axhline(0, color=AXIS, lw=0.8)
        ax.grid(axis="x", visible=False)
    axes[1].set_xticks(x, [CLASS_NAMES[c] for c in classes], rotation=20, ha="right")
    axes[0].legend(ncol=5, loc="lower left", bbox_to_anchor=(0, 1.02))
    save(fig, "fig_per_class", png_dir)


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--png", default=None, help="also write PNG previews to this directory")
    args = ap.parse_args()
    OUT.mkdir(parents=True, exist_ok=True)
    setup()
    curves(args.png)
    heatmap(args.png)
    randomisation(args.png)
    per_class(args.png)
