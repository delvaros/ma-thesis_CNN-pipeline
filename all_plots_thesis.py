"""
Thesis plots, if they can be of use to somebody...
Includes custom color palette.
"""

# ============================================================
# Imports
# ============================================================
import time

start_time = time.perf_counter()
from pathlib import Path
import os
import random

import numpy as np
import pandas as pd

import matplotlib as mpl
import matplotlib.colors as mcolors
import matplotlib.pyplot as plt
from matplotlib.ticker import FuncFormatter, FixedLocator, ScalarFormatter

import seaborn as sns
import openslide
from itertools import combinations

from scipy import stats
from scipy.stats import ttest_ind, spearmanr, mannwhitneyu

from sklearn.metrics import (
    roc_auc_score,
    roc_curve,
    auc,
    precision_recall_curve,
    average_precision_score,
    confusion_matrix,
)
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler

import statsmodels.api as sm
import statsmodels.formula.api as smf

from tbparse import SummaryReader

from helper_scripts.compressed_sampler import SampleManager
import score_analysis as sa

# ============================================================
# Output directory & save helper
# ============================================================
OUT_DIR = Path("thesis_plots")
OUT_DIR.mkdir(exist_ok=True)


def save_svg(fig, name):
    """Save a matplotlib figure as SVG to OUT_DIR. Uses rcParams['savefig.bbox']."""
    fig.savefig(OUT_DIR / f"{name}.svg", format="svg", bbox_inches=None)


# ============================================================
# Color palette & rcParams (single source of truth)
# ============================================================
LABEL = {
    "negative": "#68A6F2",
    "positive": "#C92B45",
}

CASE_LABEL = {
    "negative": "#2260AB",
    "positive": "#FD7A1C",
}

MARKER = {
    "p15": "#62A7E0",
    "p16": "#336397",
    "p21": "#122C4D",
}

TISSUE = {
    "epi": "#4A3B6B",
    "fat": "#E0B469",
    "stroma": "#379068FF",
    "tdlu": "#DD86E0",
}

HE_SEQ = mcolors.LinearSegmentedColormap.from_list(
    "he_seq",
    ["#FBEDD8", "#E8C0D0", "#C97BA0", "#8E4B70", "#4A3B6B"],
    N=256,
)
HE_DIV = mcolors.LinearSegmentedColormap.from_list(
    "he_div",
    ["#427F8D", "#FFFFFF", "#912B96"],
    N=256,
)
CORR_DIV = mcolors.LinearSegmentedColormap.from_list(
    "corr_div",
    ["#427F8D", "#FFFFFF", "#912B96"],
    N=256,
)

CORR_COL = {"pos": "#912B96", "neg": "#427F8D"}

for cm in (HE_SEQ, HE_DIV, CORR_DIV):
    try:
        mpl.colormaps.unregister(cm.name)
    except (KeyError, ValueError):
        pass
    mpl.colormaps.register(cm)

mpl.rcParams.update(
    {
        "font.family": "sans-serif",
        "font.sans-serif": ["Helvetica", "TeX Gyre Heros", "Arial", "DejaVu Sans"],
        "font.size": 9,
        "axes.titlesize": 10,
        "axes.labelsize": 9,
        "xtick.labelsize": 8,
        "ytick.labelsize": 8,
        "legend.fontsize": 8,
        "figure.titlesize": 11,
        "axes.spines.top": False,
        "axes.spines.right": False,
        "axes.linewidth": 0.8,
        "xtick.major.width": 0.8,
        "ytick.major.width": 0.8,
        "pdf.fonttype": 42,
        "ps.fonttype": 42,
        "savefig.dpi": 300,
        "savefig.bbox": "tight",
        "image.cmap": "he_seq",
    }
)
sns.set_palette(list(LABEL.values()))

# Two consistent override sizes for in-plot text only
FONT_ANNOT = 7  # bracket text, effect-size labels
FONT_CELL = 9  # heatmap cell annotations


# ============================================================
# Constants
# ============================================================
DAB_COLS = ["dab_mean", "dab_median", "dab_max", "dab_p90", "dab_positive_fraction"]
MORPH_COLS = ["area", "convexity", "aspect"]
SCORE_COLS = ["mean_p15", "mean_p16", "mean_p21"]


# ============================================================
# Data loading
# ============================================================
# Samplers (full IHC datasets, per marker)
p15_sampler = SampleManager(Path("/allsamplers_p15"))
p15_sampler.load_samples()
p16_sampler = SampleManager(Path("/allsamplers_p16"))
p16_sampler.load_samples()
p21_sampler = SampleManager(Path("/allsamplers_p21"))
p21_sampler.load_samples()

# Validation samplers (per marker)
p15_val_sampler = SampleManager("/model_training/models/train_datasets/val_sampler_v1")
p15_val_sampler.load_samples()
p16_val_sampler = SampleManager("/model_training/models/train_datasets/val_sampler_v47")
p16_val_sampler.load_samples()
p21_val_sampler = SampleManager("/model_training/models/train_datasets/val_sampler_v8")
p21_val_sampler.load_samples()

# Morphometric dataframes (per marker)
morph_df_p15 = pd.read_csv("/results_Apr26_v2-morph.csv")
morph_df_p16 = pd.read_csv("/results_Apr26_v2-morph.csv")
morph_df_p21 = pd.read_csv("/results_Apr26_v2-morph.csv")

# KTB prediction tables
table_dir = Path("/KTB_prediction_tables/")
p15_p16_p21_IR_tissue_df = pd.read_csv(
    table_dir / "p15_p16_p21_IR_tissue.csv", sep=";"
).drop(columns="Unnamed: 0")
IR_mar20_df = pd.read_csv(table_dir / "nusp-ir-mar20_2026-s50-cvp256-mapped.csv")
RS_mar20_df = pd.read_csv(table_dir / "nusp-rs-mar20_2026-s50-cvp256-mapped.csv")
ATVR_feb28_df = pd.read_csv(table_dir / "nusp-atvr-feb28-s50-cvp256-mapped.csv")
OX_apr13_df = pd.read_csv(table_dir / "nusp-ox-apr13_2026-s50-e0-cvp256-mapped.csv")
ANTI_feb28_df = pd.read_csv(table_dir / "nusp-anti-feb28-s50-e0-cvp256-mapped.csv")

# TensorBoard scalars (named tb_dfs to avoid shadowing sample_dfs later)
log_dirs = {
    "p15": "/model_training/models/v1/tensorboard",
    "p16": "/model_training/models/v47/tensorboard",
    "p21": "/model_training/models/v8/tensorboard",
}
tb_dfs = {marker: SummaryReader(path).scalars for marker, path in log_dirs.items()}


# ============================================================
# Build per-nucleus samples dataframes from samplers
# ============================================================
def _build_samples_df(sampler):
    keys, dabs, labels = [], [], []
    for i, slide in enumerate(sampler.sample_xs):
        keys.extend(sampler.sample_keys[i])
        for cell in slide:
            dabs.append(cell["dab_intensity"])
            labels.append(cell["senescence_label"])
    df = pd.concat(
        [pd.DataFrame({"key": keys, "label": labels}), pd.DataFrame(dabs)],
        axis=1,
    )
    df["wsi"] = df["key"].str.split(" ").str[0]
    return df


p15_samples_df = _build_samples_df(p15_sampler)
p16_samples_df = _build_samples_df(p16_sampler)
p21_samples_df = _build_samples_df(p21_sampler)

# -------- TODO: merge model predictions into samples_dfs --------
# path definitions
DATASET = "p15"
MODEL = "v1"
base_path = Path(f"/ktb_ihc_{DATASET}/model_training/")
p15_pred_df = pd.read_csv((base_path / f"eval/nusp-{MODEL}-virchow_{DATASET}-pred.csv"))

# path definitions
DATASET = "p16"
MODEL = "v47"
base_path = Path(f"/ktb_ihc_{DATASET}/model_training/")
p16_pred_df = pd.read_csv((base_path / f"eval/nusp-v47-virchow.csv"))

# path definitions
DATASET = "p21"
MODEL = "v8"
base_path = Path(f"/ktb_ihc_{DATASET}/model_training/")
p21_pred_df = pd.read_csv((base_path / f"eval/nusp-{MODEL}-virchow_{DATASET}-pred.csv"))


p15_samples_df = p15_samples_df.merge(p15_pred_df, on="key")
p16_samples_df = p16_samples_df.merge(p16_pred_df, on="key")
p21_samples_df = p21_samples_df.merge(p21_pred_df, on="key")
# ------------------------------------------------------------------

# Merge Age (from prediction tables) into each samples df
age_lookup = p15_p16_p21_IR_tissue_df[["wsi", "Age"]].drop_duplicates()
p15_age_samples_df = p15_samples_df.merge(age_lookup, on="wsi", how="left").dropna(
    subset=["Age"]
)
p16_age_samples_df = p16_samples_df.merge(age_lookup, on="wsi", how="left").dropna(
    subset=["Age"]
)
p21_age_samples_df = p21_samples_df.merge(age_lookup, on="wsi", how="left").dropna(
    subset=["Age"]
)

sample_dfs = {
    "p15": p15_samples_df,
    "p16": p16_samples_df,
    "p21": p21_samples_df,
}

sample_age_dfs = {
    "p15": p15_age_samples_df,
    "p16": p16_age_samples_df,
    "p21": p21_age_samples_df,
}

# Validation subset: filter sample_dfs by WSIs in the val samplers
val_samplers = {"p15": p15_val_sampler, "p16": p16_val_sampler, "p21": p21_val_sampler}
val_wsis = {
    marker: {key.split(" ")[0] for key in sampler.sample_keys}
    for marker, sampler in val_samplers.items()
}
sample_dfs_val = {
    marker: df[df["wsi"].isin(val_wsis[marker])].copy()
    for marker, df in sample_dfs.items()
}

# IHC training cohort = WSIs that appear in any of the three samples dfs
IHC_50_wsi_keys = set()
for df in sample_dfs.values():
    IHC_50_wsi_keys.update(df["wsi"].unique())

IHC_51_predictions_df = p15_p16_p21_IR_tissue_df[
    p15_p16_p21_IR_tissue_df.wsi.isin(IHC_50_wsi_keys)
]

# Convenience structures for the dab-vs-morph heatmap
sources = [
    ("p15", p15_samples_df, morph_df_p15),
    ("p16", p16_samples_df, morph_df_p16),
    ("p21", p21_samples_df, morph_df_p21),
]
morph_dfs = {"p15": morph_df_p15, "p16": morph_df_p16, "p21": morph_df_p21}


# ============================================================
# Helper: build tidy dataframe (per-nucleus, with morphometrics)
# ============================================================
def build_tidy(sampler, morph_df, marker, morph_cols=("area", "convexity", "aspect")):
    rows = []
    for cells, keys in zip(sampler.sample_xs, sampler.sample_keys):
        for cell, key in zip(cells, keys):
            rows.append(
                {
                    "marker": marker,
                    "key": key,
                    "wsi": key.rsplit("_", 2)[0],
                    "dab_mean": cell["dab_intensity"]["dab_mean"],
                    "senescence_label": cell["senescence_label"],
                }
            )
    d = pd.DataFrame(rows)
    return d.merge(morph_df[["key"] + list(morph_cols)], on="key", how="left")


tidy = pd.concat(
    [
        build_tidy(p15_sampler, morph_df_p15, "p15"),
        build_tidy(p16_sampler, morph_df_p16, "p16"),
        build_tidy(p21_sampler, morph_df_p21, "p21"),
    ],
    ignore_index=True,
)


# ============================================================
# Effect size helpers
# ============================================================
def cliffs_delta(x, y):
    u, p = mannwhitneyu(x, y, alternative="two-sided")
    delta = 2 * u / (len(x) * len(y)) - 1
    return delta, p


def delta_label(delta):
    a = abs(delta)
    if a < 0.147:
        return "negligible"
    if a < 0.330:
        return "small"
    if a < 0.474:
        return "medium"
    return "large"


# ============================================================
# Section 1 — Dataset description
# ============================================================
def plot_nuclei_per_slide(tidy, markers=("p15", "p16", "p21"), savename=None):
    fig, axes = plt.subplots(
        2,
        3,
        figsize=(8, 5),
        sharex="col",
        gridspec_kw={"height_ratios": [2, 1]},
    )
    for col, m in enumerate(markers):
        sub = tidy[tidy["marker"] == m]
        counts = sub.groupby(["wsi", "senescence_label"]).size().unstack(fill_value=0)
        total = counts.sum(axis=1).sort_values(ascending=False)
        counts = counts.loc[total.index]
        frac_pos = (
            counts[1] / total if 1 in counts else pd.Series(0, index=counts.index)
        )

        ax_top = axes[0, col]
        bottom = np.zeros(len(counts))
        for label, color in zip([0, 1], LABEL.values()):
            if label in counts:
                descr = "pos" if label == 1 else "neg"
                ax_top.bar(
                    range(len(counts)),
                    counts[label],
                    bottom=bottom,
                    color=color,
                    label=descr,
                    width=1.0,
                )
                bottom += counts[label].values
        ax_top.set_title(m)
        ax_top.set_ylabel("n nuclei" if col == 0 else "")

        ax_bot = axes[1, col]
        ax_bot.bar(
            range(len(frac_pos)), frac_pos.values, color=LABEL["positive"], width=1.0
        )
        ax_bot.axhline(
            frac_pos.mean(),
            color="black",
            ls="--",
            lw=0.8,
            label=f"mean={frac_pos.mean():.2%}",
        )
        ax_bot.set_ylabel("frac. positive" if col == 0 else "")
        ax_bot.set_xlabel("slide (sorted by total)")
        ax_bot.set_xticks([])

    axes[1, 0].legend(frameon=False, loc="upper right")
    axes[1, 1].legend(frameon=False, loc="upper right")
    axes[1, 2].legend(frameon=False, loc="upper left")
    axes[0, -1].legend(loc="upper right", frameon=False)
    plt.tight_layout()
    if savename:
        save_svg(fig, savename)
    plt.show()


def plot_cohort_composition_ihc(
    df, savename=None, age_col="Age", tissue_col="tissue", wsi_col="wsi"
):
    """Cohort plot for the IHC training cohort (no case labels available)."""
    slides = df.groupby(wsi_col).agg({age_col: "first"}).reset_index()

    fig, axes = plt.subplots(1, 2, figsize=(8, 3.5))

    sub = slides.dropna(subset=[age_col])
    sns.histplot(data=sub, x=age_col, bins=30, color=LABEL["negative"], ax=axes[0])
    axes[0].set_title("Age distribution")
    axes[0].set_xlabel("Age (years)")

    tissue_counts = (
        df[df[tissue_col] != "both"]
        .groupby(tissue_col)
        .size()
        .sort_values(ascending=False)
    )
    tissue_colors = [TISSUE.get(t, "#888") for t in tissue_counts.index]
    axes[1].bar(tissue_counts.index, tissue_counts.values, color=tissue_colors)
    axes[1].set_title("Nuclei per tissue type")
    axes[1].set_ylabel("Number of nuclei")
    axes[1].tick_params(axis="x", rotation=30)
    axes[1].yaxis.set_major_formatter(
        FuncFormatter(
            lambda v, _: (
                f"{v/1e6:.1f}M"
                if v >= 1e6
                else (f"{int(v/1e3)}k" if v >= 1e3 else f"{int(v)}")
            )
        )
    )
    plt.tight_layout()
    if savename:
        save_svg(fig, savename)
    plt.show()


def plot_dab_distribution(
    tidy,
    markers=("p15", "p16", "p21"),
    dab_threshold=0.25,
    neg_pct=90,
    pos_pct=30,
    log_y=False,
    savename=None,
):
    fig, axes = plt.subplots(1, 3, figsize=(8, 3), sharey=False)
    for ax, m in zip(axes, markers):
        sub = tidy[tidy["marker"] == m]["dab_mean"].dropna()
        neg = sub[sub < dab_threshold]
        pos = sub[sub >= dab_threshold]
        neg_cut = np.percentile(neg, neg_pct)
        pos_cut = np.percentile(pos, pos_pct)

        bins = np.linspace(0, sub.quantile(0.999), 100)
        ax.hist(
            neg,
            bins=bins,
            color=LABEL["negative"],
            alpha=1,
            label="neg",
            edgecolor="white",
            linewidth=0.1,
        )
        ax.hist(
            pos,
            bins=bins,
            color=LABEL["positive"],
            alpha=1,
            label="pos",
            edgecolor="white",
            linewidth=0.1,
        )

        ax.axvline(dab_threshold, color="black", ls="-", lw=1, label="thr")
        ax.axvline(neg_cut, color="black", ls="--", lw=1, label=f"neg p{neg_pct}")
        ax.axvline(pos_cut, color="dimgray", ls=":", lw=1, label=f"pos p{pos_pct}")

        ax.set_title(m)
        ax.set_xlabel("DAB mean")
        ax.set_ylabel(("count (log)" if log_y else "count") if m == markers[0] else "")

        if log_y:
            ax.set_yscale("log")
            fmt = ScalarFormatter()
            fmt.set_scientific(False)
            ymax = ax.get_ylim()[1]
            ticks = [
                t
                for t in [10, 100, 500, 1000, 3000, 10_000, 20_000, 40_000, 80_000]
                if t <= ymax
            ]
            ax.yaxis.set_major_locator(FixedLocator(ticks))
            ax.yaxis.set_major_formatter(fmt)
            ax.yaxis.set_minor_formatter(plt.NullFormatter())

    axes[2].legend(frameon=False, loc="upper right")
    plt.tight_layout()
    if savename:
        save_svg(fig, savename)
    plt.show()


def pad_image_square(img, border=5):
    """Add `border` px black margin, then pad shorter dim to make it square."""
    img = np.asarray(img)
    is_rgb = img.ndim == 3
    pad_spec = ((border, border), (border, border)) + (((0, 0),) if is_rgb else ())
    img = np.pad(img, pad_spec, constant_values=0)

    h, w = img.shape[:2]
    if h != w:
        diff = abs(h - w)
        a, b = diff // 2, diff - diff // 2
        sq_pad = ((0, 0), (a, b)) if h > w else ((a, b), (0, 0))
        if is_rgb:
            sq_pad = sq_pad + ((0, 0),)
        img = np.pad(img, sq_pad, constant_values=0)
    return img


def plot_sampler_triplets(
    sampler,
    pX_dir,
    marker="",
    n=3,
    wsi_idx=None,
    margin=20,
    border=5,
    seed=104,
    savename=True,
):
    rng = random.Random(seed)
    pX_dir = Path(pX_dir)

    if wsi_idx is None:
        wsi_idx = rng.randrange(len(sampler.sample_xs))
    nuclei = sampler.sample_xs[wsi_idx]
    keys = sampler.sample_keys[wsi_idx]
    wsi_name = keys[0].split(" ")[0]

    candidates = [
        f
        for f in pX_dir.glob(f"{wsi_name}*")
        if f.suffix.lower() in (".svs", ".ndpi", ".tif", ".tiff")
    ]
    if not candidates:
        raise FileNotFoundError(f'No pX slide for "{wsi_name}" in {pX_dir}')
    slide = openslide.OpenSlide(str(candidates[0]))

    nuc_indices = rng.sample(range(len(nuclei)), n)

    # 3 rows = image type, n cols = nuclei
    fig, axes = plt.subplots(
        3, n, figsize=(2.3, 2.3), gridspec_kw={"wspace": 0.03, "hspace": 0.03}
    )
    if n == 1:
        axes = axes[:, None]

    row_labels = ["mask", "H&E", marker or "pX"]

    for col, idx in enumerate(nuc_indices):
        cell = nuclei[idx]

        # mask
        axes[0, col].imshow(
            pad_image_square(cell["bbox_bitmap"], border),
            cmap="gray",
            interpolation="nearest",
        )

        # H&E
        axes[1, col].imshow(pad_image_square(cell["bbox_pixels"], border))

        # pX
        bbox_pX = np.array(cell["bbox_pX"])
        ys, xs = bbox_pX[:, 0], bbox_pX[:, 1]
        x1, y1 = int(xs.min() - margin), int(ys.min() - margin)
        x2, y2 = int(xs.max() + margin), int(ys.max() + margin)
        pX_img = np.array(
            slide.read_region((x1, y1), 0, (x2 - x1, y2 - y1)).convert("RGB")
        )
        axes[2, col].imshow(pad_image_square(pX_img, border))

        # axes[0, col].set_title(f'nuc {idx}', fontsize=9)

    # remove ticks + spines, keep ylabels intact
    for ax in axes.flat:
        ax.set_xticks([])
        ax.set_yticks([])
        for spine in ax.spines.values():
            spine.set_visible(False)

    # row labels on the left of each row
    for row_idx, label in enumerate(row_labels):
        axes[row_idx, 0].set_ylabel(
            label, rotation=0, ha="right", va="center", fontsize=12, labelpad=12
        )

    fig.suptitle(f"WSI: {wsi_name}", fontsize=11, y=0.92)  # {marker}  |
    plt.tight_layout()
    if savename:
        savename = f"fig1_{marker}_example"
        save_svg(fig, savename)
    slide.close()


# ============================================================
# Section 2 — Morphometric analyses
# ============================================================
def plot_morph_comparison(
    tidy,
    markers=("p15", "p16", "p21"),
    metrics=("area", "convexity", "aspect"),
    savename=None,
):
    fig, axes = plt.subplots(1, len(metrics), figsize=(8, 3))
    y_positions = {m: len(markers) - 1 - i for i, m in enumerate(markers)}

    for ax, metric in zip(axes, metrics):
        for m in markers:
            sub = tidy[tidy["marker"] == m].dropna(subset=[metric, "senescence_label"])
            neg = sub[sub["senescence_label"] == 0][metric]
            pos = sub[sub["senescence_label"] == 1][metric]
            y = y_positions[m]
            if len(neg) < 2 or len(pos) < 2:
                continue

            ax.errorbar(
                neg.mean(),
                y,
                xerr=neg.sem(),
                fmt="o",
                color=LABEL["negative"],
                markersize=5,
                capsize=3,
                elinewidth=1.2,
                mew=1.2,
                zorder=3,
            )
            ax.errorbar(
                pos.mean(),
                y,
                xerr=pos.sem(),
                fmt="o",
                color=LABEL["positive"],
                markersize=5,
                capsize=3,
                elinewidth=1.2,
                mew=1.2,
                zorder=3,
            )

            delta, p = cliffs_delta(neg, pos)
            sig = (
                "***" if p < 0.001 else "**" if p < 0.01 else "*" if p < 0.05 else "ns"
            )
            annot = f"{sig}\n δ = {delta:+.2f}"

            x_lo, x_hi = sorted([neg.mean(), pos.mean()])
            bracket_y = y + 0.22
            ax.plot(
                [x_lo, x_lo, x_hi, x_hi],
                [y + 0.06, bracket_y, bracket_y, y + 0.06],
                color="gray",
                lw=0.7,
                zorder=2,
            )
            ax.text(
                (x_lo + x_hi) / 2,
                bracket_y + 0.03,
                annot,
                ha="center",
                va="bottom",
                fontsize=FONT_ANNOT,
                color="gray",
                style="italic",
            )

        if metric == "convexity":
            ax.xaxis.set_major_locator(FixedLocator([0.9575, 0.958, 0.9585]))

        ax.set_yticks(list(y_positions.values()))
        ax.set_yticklabels(list(y_positions.keys()))
        ax.set_xlabel(metric.capitalize())
        ax.set_ylim(-0.6, len(markers) - 1 + 0.7)

    handles = [
        plt.Line2D(
            [],
            [],
            marker="o",
            color=LABEL["negative"],
            ls="",
            markersize=5,
            label="neg",
        ),
        plt.Line2D(
            [],
            [],
            marker="o",
            color=LABEL["positive"],
            ls="",
            markersize=5,
            label="pos",
        ),
    ]
    axes[-1].legend(handles=handles, frameon=False, loc="upper right")
    plt.tight_layout()
    if savename:
        save_svg(fig, savename)
    plt.show()


def plot_metric_distributions(
    tidy,
    markers=("p15", "p16", "p21"),
    metrics=("area", "convexity", "aspect"),
    clip_p=0,
    clip_q=0.99,
    savename=None,
):
    fig, axes = plt.subplots(len(markers), len(metrics), figsize=(7, 5), sharex="col")

    x_limits = {
        m: (
            tidy[m].dropna().quantile(clip_p if m != "convexity" else 0.01),
            tidy[m].dropna().quantile(clip_q),
        )
        for m in metrics
    }

    for i, marker in enumerate(markers):
        for j, metric in enumerate(metrics):
            ax = axes[i, j]
            sub = tidy[tidy["marker"] == marker].dropna(
                subset=[metric, "senescence_label"]
            )
            binsize = {"area": 10, "convexity": 0.001}.get(metric, 0.05)

            sns.histplot(
                data=sub,
                x=metric,
                hue="senescence_label",
                palette={0: LABEL["negative"], 1: LABEL["positive"]},
                stat="density",
                common_norm=False,
                element="step",
                fill=True,
                alpha=0.4,
                ax=ax,
                binwidth=binsize,
                legend=(i == 0 and j == len(metrics) - 1),
            )
            ax.set_xlim(x_limits[metric])
            # Row label = marker — keep at default body font for consistency
            ax.set_ylabel(
                marker if j == 0 else "",
                rotation=0,
                ha="right",
                va="center",
                labelpad=18,
            )
            ax.set_xlabel(metric.capitalize() if i == len(markers) - 1 else "")
            ax.tick_params(left=False, labelleft=False)

    leg = axes[0, -1].get_legend()
    if leg is not None:
        for t, label in zip(leg.get_texts(), ["neg", "pos"]):
            t.set_text(label)
        leg.set_title("")
        leg.set_frame_on(False)

    plt.tight_layout()
    if savename:
        save_svg(fig, savename)
    plt.show()


def plot_dab_morph_heatmap(
    sources, dab_cols=DAB_COLS, morph_cols=MORPH_COLS, savename=None
):
    fig, axes = plt.subplots(1, 3, figsize=(20, 4))
    for ax, (marker, samp, morph) in zip(axes, sources):
        merged = samp[["key"] + dab_cols].merge(
            morph[["key"] + morph_cols], on="key", how="inner"
        )
        rows = []
        for m in morph_cols:
            for d in dab_cols:
                valid = merged[[m, d]].dropna()
                rho, _ = spearmanr(valid[m], valid[d])
                rows.append({"morph": m, "dab": d, "rho": rho})
        cdf = pd.DataFrame(rows)
        mat = cdf.pivot(index="morph", columns="dab", values="rho")[dab_cols].loc[
            morph_cols
        ]

        sns.heatmap(
            mat,
            annot=True,
            fmt=".2f",
            cmap="corr_div",
            vmin=-0.4,
            vmax=0.4,
            center=0,
            square=True,
            linewidths=0.5,
            linecolor="white",
            annot_kws={"fontsize": FONT_CELL},
            cbar=False,
            ax=ax,
        )
        ax.set_title(marker)
        ax.set_xticklabels(
            [
                t.get_text()
                .replace("dab_", "")
                .replace("positive_fraction", "pos_frac")
                for t in ax.get_xticklabels()
            ],
            rotation=30,
            ha="right",
        )
        ax.set_yticklabels(ax.get_yticklabels(), rotation=0)
        ax.set_xlabel("")
        ax.set_ylabel("")
        ax.tick_params(left=False, bottom=False)

    sm = plt.cm.ScalarMappable(cmap="corr_div", norm=plt.Normalize(vmin=-0.4, vmax=0.4))
    sm.set_array([])
    fig.colorbar(sm, ax=axes, shrink=0.6, pad=0.02, label="Spearman ρ")
    if savename:
        save_svg(fig, savename)
    plt.show()


def plot_pca_morphometrics(
    tidy, morph_dfs, feature_cols=None, sample_n=None, savename=None
):
    if feature_cols is None:
        feature_cols = [
            "area",
            "perimeter",
            "hull_area",
            "hull_perimeter",
            "convexity",
            "solidity",
            "circularity",
            "aspect",
        ]

    markers = list(morph_dfs.keys())
    fig, axes = plt.subplots(1, len(markers), figsize=(8, 3))

    for ax, marker in zip(axes, markers):
        labels_df = tidy.loc[tidy["marker"] == marker, ["key", "senescence_label"]]
        data = labels_df.merge(
            morph_dfs[marker][["key"] + feature_cols], on="key"
        ).dropna()

        if sample_n is not None:
            data = data.groupby("senescence_label", group_keys=False).apply(
                lambda d: d.sample(min(len(d), sample_n), random_state=42)
            )

        X = StandardScaler().fit_transform(data[feature_cols].values)
        pca = PCA(n_components=2).fit(X)
        Z = pca.transform(X)
        y = data["senescence_label"].values

        for label, color in [(0, LABEL["negative"]), (1, LABEL["positive"])]:
            mask = y == label
            ax.scatter(
                Z[mask, 0],
                Z[mask, 1],
                s=8,
                alpha=0.4,
                color=color,
                edgecolor="none",
                label=f'{"pos" if label else "neg"}',
                rasterized=True,
            )

        v1, v2 = pca.explained_variance_ratio_ * 100
        ax.set_xlabel(f"PC1 ({v1:.1f}%)")
        ax.set_ylabel(f"PC2 ({v2:.1f}%)")
        ax.set_title(marker)
        ax.axhline(0, color="gray", lw=0.4, alpha=0.5)
        ax.axvline(0, color="gray", lw=0.4, alpha=0.5)

    axes[-1].legend(frameon=False, loc="best")
    plt.tight_layout()
    if savename:
        save_svg(fig, savename)
    plt.show()


# ============================================================
# Section 3 — Training & prediction
# ============================================================
def plot_training_curves(
    tb_dfs,
    tag_map,
    ncols=4,
    smooth_window=5,
    best_model_tag="BestModel/saved",
    savename=None,
):
    best_steps = {
        marker: (
            df.loc[(df["tag"] == best_model_tag) & (df["value"] == 1), "step"].max()
            if ((df["tag"] == best_model_tag) & (df["value"] == 1)).any()
            else None
        )
        for marker, df in tb_dfs.items()
    }

    metrics = list(tag_map.keys())
    nrows = (len(metrics) + ncols - 1) // ncols
    fig, axes = plt.subplots(nrows, ncols, figsize=(16, 3 * nrows), squeeze=False)
    axes = axes.flatten()

    for ax, (display_name, tag) in zip(axes, tag_map.items()):
        for marker, df in tb_dfs.items():
            sub = df[df["tag"] == tag].sort_values("step")
            if len(sub) == 0:
                continue
            steps = sub["step"].values
            y = sub["value"].values
            if smooth_window:
                y = pd.Series(y).rolling(smooth_window, min_periods=1).mean().values

            best_step = best_steps.get(marker)
            if best_step is None or best_step >= steps.max():
                ax.plot(steps, y, color=MARKER[marker], label=marker, lw=1.4)
            else:
                solid = steps <= best_step
                dotted = steps >= best_step
                ax.plot(
                    steps[solid], y[solid], color=MARKER[marker], label=marker, lw=1.4
                )
                ax.plot(steps[dotted], y[dotted], color=MARKER[marker], ls=":", lw=1.4)
                ax.plot(
                    best_step,
                    y[steps == best_step][0],
                    marker="o",
                    markersize=5,
                    markerfacecolor="white",
                    markeredgecolor=MARKER[marker],
                    markeredgewidth=1.2,
                    zorder=5,
                )

        ax.set_title(display_name)
        ax.set_xlabel("Step")
        ax.grid(alpha=0.3, lw=0.5)

    for ax in axes[len(metrics) :]:
        ax.axis("off")
    axes[min(len(metrics) - 1, ncols - 1)].legend(frameon=False, loc="upper right")
    plt.tight_layout()
    if savename:
        save_svg(fig, savename)
    plt.show()


def plot_roc_pr(dfs, label_col="label", pred_col="prediction", savename=None):
    fig, axes = plt.subplots(1, 2, figsize=(8, 3))
    for marker, df in dfs.items():
        y_true = df[label_col].astype(int)
        y_pred = df[pred_col]
        prevalence = y_true.mean()

        fpr, tpr, _ = roc_curve(y_true, y_pred)
        axes[0].plot(
            fpr,
            tpr,
            color=MARKER[marker],
            lw=1.5,
            label=f"{marker} (AUC = {auc(fpr, tpr):.3f})",
        )

        prec, rec, _ = precision_recall_curve(y_true, y_pred)
        axes[1].plot(
            rec,
            prec,
            color=MARKER[marker],
            lw=1.5,
            label=f"{marker} (AP = {average_precision_score(y_true, y_pred):.3f})",
        )
        axes[1].axhline(prevalence, color=MARKER[marker], lw=0.6, ls=":", alpha=0.6)

    axes[0].plot([0, 1], [0, 1], "k--", lw=0.6, alpha=0.4)
    axes[0].set(xlabel="FPR", ylabel="TPR", title="ROC", xlim=(0, 1), ylim=(0, 1.02))
    axes[1].set(
        xlabel="Recall",
        ylabel="Precision",
        title="Precision-Recall",
        xlim=(0, 1),
        ylim=(0, 1.02),
    )
    for ax in axes:
        ax.legend(frameon=False, loc="lower right" if ax is axes[0] else "best")
    plt.tight_layout()
    if savename:
        save_svg(fig, savename)
    plt.show()


def plot_confusion_matrices(
    dfs, label_col="label", pred_col="prediction", threshold=0.5, savename=None
):
    fig, axes = plt.subplots(1, 3, figsize=(8, 3))
    for ax, (marker, df) in zip(axes, dfs.items()):
        y_true = df[label_col].astype(int)
        y_pred = (df[pred_col] >= threshold).astype(int)

        cm = confusion_matrix(y_true, y_pred)
        cm_pct = cm / cm.sum(axis=1, keepdims=True) * 100
        annot = np.array(
            [
                [f"{cm[i, j]:,}\n({cm_pct[i, j]:.1f}%)" for j in range(2)]
                for i in range(2)
            ]
        )
        sns.heatmap(
            cm_pct,
            annot=annot,
            fmt="",
            cmap="he_seq",
            vmin=0,
            vmax=100,
            square=True,
            cbar=False,
            xticklabels=["Pred neg", "Pred pos"],
            yticklabels=["True neg", "True pos"],
            annot_kws={"fontsize": FONT_CELL},
            ax=ax,
        )
        ax.set_title(f"{marker}  (thr = {threshold})")
    plt.tight_layout()
    if savename:
        save_svg(fig, savename)
    plt.show()


def plot_slide_level(
    dfs,
    x_col="dab_mean",
    x_label="Mean DAB intensity (per slide)",
    label_col="label",
    pred_col="prediction",
    wsi_col="wsi",
    savename=None,
):
    """Per-slide aggregation: mean predicted probability vs `x_col` mean per slide."""
    fig, axes = plt.subplots(1, 3, figsize=(7.5, 3.5))
    for ax, (marker, df) in zip(axes, dfs.items()):
        slide_agg = df.groupby(wsi_col).agg(
            mean_pred=(pred_col, "mean"),
            mean_x=(x_col, "mean"),
            frac_true_pos=(label_col, "mean"),
            n_nuclei=(label_col, "size"),
        )
        rho = slide_agg[["mean_pred", "mean_x"]].corr(method="spearman").iloc[0, 1]

        ax.scatter(
            slide_agg["mean_x"],
            slide_agg["mean_pred"],
            s=slide_agg["n_nuclei"] / slide_agg["n_nuclei"].max() * 80 + 5,
            color=MARKER[marker],
            alpha=0.6,
            edgecolor="none",
        )
        sns.regplot(
            data=slide_agg,
            x="mean_x",
            y="mean_pred",
            scatter=False,
            color="black",
            line_kws={"lw": 1.2},
            ax=ax,
        )

        ax.set_xlabel(x_label)
        if ax is axes[0]:
            ax.set_ylabel("Mean predicted probability (per slide)")
        ax.set_title(f"{marker}  (ρ = {rho:.2f}")
    plt.tight_layout()
    if savename:
        save_svg(fig, savename)
    plt.show()


# ============================================================
# Section 4 — KomenTissueBank prediction
# ============================================================
def plot_cohort_composition_ktb(
    df,
    savename=None,
    age_col="Age",
    case_col="case",
    tissue_col="tissue",
    wsi_col="wsi",
):
    slides = (
        df.groupby(wsi_col).agg({age_col: "first", case_col: "first"}).reset_index()
    )

    fig, axes = plt.subplots(1, 2, figsize=(7.5, 3.5))

    sub = slides.dropna(subset=[age_col])
    sns.histplot(
        data=sub,
        x=age_col,
        bins=30,
        hue=case_col,
        palette={True: CASE_LABEL["positive"], False: CASE_LABEL["negative"]},
        multiple="stack",
        ax=axes[0],
    )
    axes[0].set_title("Age distribution")
    axes[0].set_xlabel("Age (years)")

    tissue_counts = (
        df[df[tissue_col] != "both"]
        .groupby(tissue_col)
        .size()
        .sort_values(ascending=False)
    )
    tissue_colors = [TISSUE.get(t, "#888") for t in tissue_counts.index]
    axes[1].bar(tissue_counts.index, tissue_counts.values, color=tissue_colors)
    axes[1].set_title("Nuclei per tissue type")
    axes[1].set_ylabel("Number of nuclei")
    axes[1].tick_params(axis="x", rotation=30)
    axes[1].yaxis.set_major_formatter(
        FuncFormatter(
            lambda v, _: (
                f"{v/1e6:.1f}M"
                if v >= 1e6
                else (f"{int(v/1e3)}k" if v >= 1e3 else f"{int(v)}")
            )
        )
    )
    plt.tight_layout()
    if savename:
        save_svg(fig, savename)
    plt.show()


# Build df_pt_tissue (one row per wsi × tissue)
body = (
    p15_p16_p21_IR_tissue_df.groupby(["wsi", "tissue"])
    .agg(
        mean_p15=("score_p15", "mean"),
        mean_p16=("score_p16", "mean"),
        mean_p21=("score_p21", "mean"),
        mean_IR=("score_IR_legacy", "mean"),
        median_p15=("score_p15", "median"),
        median_p16=("score_p16", "median"),
        median_p21=("score_p21", "median"),
        median_IR=("score_IR_legacy", "median"),
        n_tiles=("score_p16", "size"),
    )
    .reset_index()
)
meta = (
    p15_p16_p21_IR_tissue_df.groupby("wsi")
    .agg(Age=("Age", "first"), case=("case", "first"))
    .reset_index()
)
df_pt_tissue = body.merge(meta, on="wsi")
df_pt_tissue = df_pt_tissue[df_pt_tissue["tissue"] != "both"]

# Optional: merge new prediction tables
new_dfs = {
    "IR_mar20": IR_mar20_df,
    "RS_mar20": RS_mar20_df,
    "ATVR_feb28": ATVR_feb28_df,
    "OX_apr13": OX_apr13_df,
    "ANTI_feb28": ANTI_feb28_df,
}
all_merged_pt_tissue = df_pt_tissue.copy()
for name, d in new_dfs.items():
    all_merged_pt_tissue = all_merged_pt_tissue.merge(
        d[["code", "tissue", "sen"]].rename(columns={"sen": f"sen_{name}"}),
        left_on=["wsi", "tissue"],
        right_on=["code", "tissue"],
        how="left",
    ).drop(columns="code")


def plot_tissue_pointplot(df_pt_tissue, score_col, savename=None):
    """Per-tissue case-vs-control point plot for one score column."""
    p_values = {}
    for tissue in df_pt_tissue["tissue"].unique():
        subset = df_pt_tissue[df_pt_tissue["tissue"] == tissue]
        cases = subset["case"].unique()
        if len(cases) == 2:
            g1 = subset[subset["case"] == cases[0]][score_col]
            g2 = subset[subset["case"] == cases[1]][score_col]
            _, p = ttest_ind(g1, g2, equal_var=False)
            p_values[tissue] = p

    fig, ax = plt.subplots(figsize=(3.5, 4))
    sns.pointplot(
        data=df_pt_tissue,
        x="tissue",
        y=score_col,
        hue="case",
        join=False,
        errorbar="ci",
        dodge=True,
        ax=ax,
        palette={True: CASE_LABEL["positive"], False: CASE_LABEL["negative"]},
    )

    x_labels = [t.get_text() for t in ax.get_xticklabels()]
    tissue_ymax = {t: [] for t in x_labels}
    for collection in ax.collections:
        for x, y in collection.get_offsets():
            tissue = x_labels[int(round(x))]
            tissue_ymax[tissue].append(y)

    for i, tissue in enumerate(x_labels):
        if tissue in p_values and tissue_ymax[tissue]:
            y = max(tissue_ymax[tissue])
            ax.text(
                i,
                y + 0.006 * (max(map(max, tissue_ymax.values()))),
                f"p = {p_values[tissue]:.3g}",
                ha="center",
                va="bottom",
                color="black",
            )

    ax.set_title(score_col)
    plt.tight_layout()
    if savename:
        save_svg(fig, savename)
    plt.show()


def plot_score_age_pairplot(df_pt_tissue, score_cols=SCORE_COLS, savename=None):
    """Appendix figure: pairwise scatter of scores + Age, colored by tissue."""
    g = sns.pairplot(
        df_pt_tissue,
        vars=list(score_cols) + ["Age"],
        hue="tissue",
        palette=TISSUE,
        kind="scatter",
        plot_kws={"alpha": 0.4, "s": 10},
    )
    if savename:
        g.fig.savefig(OUT_DIR / f"{savename}.svg", format="svg")
    plt.show()


def or_analysis(df, score_cols, method="median", covariates=("Age",), case_col="case"):
    """Logistic regression OR for each (score × tissue)."""
    covariates = list(covariates)
    rows = []
    for score in score_cols:
        sub_all = df.dropna(subset=[score, case_col, "tissue"] + covariates).copy()
        for tt in sorted(sub_all["tissue"].unique()):
            tdfo = sub_all[sub_all["tissue"] == tt].copy()

            if method == "quartile":
                tdfo["_q"] = pd.qcut(tdfo[score], 4, labels=False, duplicates="drop")
                tdfo = tdfo[tdfo["_q"].isin([0, 3])]
                tdfo["_pred"] = (tdfo["_q"] == 3).astype(int)
            elif method == "decile":
                tdfo["_d"] = pd.qcut(tdfo[score], 10, labels=False, duplicates="drop")
                dmax = tdfo["_d"].max()
                tdfo = tdfo[tdfo["_d"].isin([0, dmax])]
                tdfo["_pred"] = (tdfo["_d"] == dmax).astype(int)
            else:
                tdfo["_pred"] = pd.qcut(tdfo[score], 2, labels=False, duplicates="drop")

            X = sm.add_constant(tdfo[["_pred"] + covariates])
            y = tdfo[case_col].astype(int)
            try:
                fit = sm.Logit(y, X).fit(disp=False)
                or_val = np.exp(fit.params["_pred"])
                ci_lo, ci_hi = np.exp(fit.conf_int().loc["_pred"])
                p = fit.pvalues["_pred"]
            except Exception:
                or_val = ci_lo = ci_hi = p = np.nan

            rows.append(
                {
                    "score": score,
                    "tissue": tt,
                    "method": method,
                    "n": len(tdfo),
                    "n_case": int(y.sum()),
                    "OR": or_val,
                    "CI_low": ci_lo,
                    "CI_high": ci_hi,
                    "p": p,
                }
            )
    return pd.DataFrame(rows)


def plot_or_forest(results_df, savename=None):
    scores = list(results_df["score"].unique())
    fig, axes = plt.subplots(
        1, len(scores), figsize=(3 * len(scores), 3.5), sharex=True, squeeze=False
    )
    for ax, score in zip(axes[0], scores):
        sub = results_df[results_df["score"] == score].reset_index(drop=True)
        ax.errorbar(
            sub["OR"],
            range(len(sub)),
            xerr=[sub["OR"] - sub["CI_low"], sub["CI_high"] - sub["OR"]],
            fmt="o",
            color="black",
            capsize=3,
        )
        ax.axvline(1, color="gray", linestyle="--", lw=1)
        ax.set_yticks(range(len(sub)))
        ax.set_yticklabels(sub["tissue"])

        for i, (or_val, p) in enumerate(zip(sub["OR"], sub["p"])):
            p_text = "p < 0.001" if p < 0.001 else f"p = {p:.3g}"
            ax.text(
                or_val,
                i - 0.2,
                p_text,
                ha="center",
                va="top",
                fontsize=FONT_ANNOT,
                color="gray",
                style="italic",
            )

        ax.set_ylim(-0.7, len(sub) - 0.5)
        ax.set_title(score)
        ax.set_xlabel("Odds Ratio")

    plt.tight_layout()
    if savename:
        save_svg(fig, savename)
    plt.show()


def plot_marker_corr_lollipop(
    df_pt_tissue,
    markers=("mean_p15", "mean_p16", "mean_p21"),
    exclude_tissues=("both",),
    alpha=0.05,
    ncols=2,
    savename=None,
):
    """Spearman correlation between own marker models, one panel per tissue."""
    tissues = [
        t for t in sorted(df_pt_tissue["tissue"].unique()) if t not in exclude_tissues
    ]
    pairs = list(combinations(markers, 2))
    label_of = lambda m: m.replace("mean_", "").replace("sen_", "")
    pair_labels = [f"{label_of(a)} – {label_of(b)}" for a, b in pairs]

    nrows = int(np.ceil(len(tissues) / ncols))
    fig, axes = plt.subplots(
        nrows,
        ncols,
        figsize=(3 * ncols, 1.6 + 0.4 * len(pairs) * nrows),
        squeeze=False,
        sharex=True,
        sharey=True,
    )
    axes_flat = axes.flatten()

    for ax, t in zip(axes_flat, tissues):
        sub = df_pt_tissue[df_pt_tissue["tissue"] == t][list(markers)].dropna()
        rhos, ps = [], []
        for a, b in pairs:
            r, p = spearmanr(sub[a], sub[b])
            rhos.append(r)
            ps.append(p)

        colors = [CORR_COL["pos"] if r > 0 else CORR_COL["neg"] for r in rhos]
        ax.hlines(pair_labels, 0, rhos, color=colors, lw=2)
        ax.scatter(
            rhos,
            pair_labels,
            c=colors,
            s=[60 if p < alpha else 20 for p in ps],
            edgecolor="black",
            lw=0.5,
            zorder=3,
        )
        ax.axvline(0, color="gray", lw=0.8)
        ax.set_xlim(-1, 1)
        ax.set_title(f"{t}")
        ax.set_xlabel("Spearman ρ")

    for ax in axes_flat[len(tissues) :]:
        ax.axis("off")

    plt.tight_layout()
    if savename:
        save_svg(fig, savename)
    plt.show()


def plot_supervisor_corr_heatmap(
    df,
    my_markers=("mean_p15", "mean_p16", "mean_p21"),
    supervisor_markers=(
        "mean_IR",
        "sen_IR_mar20",
        "sen_RS_mar20",
        "sen_ATVR_feb28",
        "sen_OX_apr13",
        "sen_ANTI_feb28",
    ),
    exclude_tissues=("both",),
    savename=None,
):
    """Heatmap per my marker: rows = supervisor models, cols = tissues."""
    tissues = [t for t in sorted(df["tissue"].unique()) if t not in exclude_tissues]
    label_of = lambda m: m.replace("mean_", "").replace("sen_", "")

    fig, axes = plt.subplots(1, len(my_markers), figsize=(4.5 * len(my_markers), 4))

    for ax, my_marker in zip(axes, my_markers):
        rho_mat = np.full((len(supervisor_markers), len(tissues)), np.nan)
        for i, sup in enumerate(supervisor_markers):
            for j, t in enumerate(tissues):
                sub = df[df["tissue"] == t][[my_marker, sup]].dropna()
                if len(sub) >= 5:
                    rho_mat[i, j], _ = spearmanr(sub[my_marker], sub[sup])

        sns.heatmap(
            rho_mat,
            annot=True,
            fmt=".2f",
            cmap="corr_div",
            vmin=-0.4,
            vmax=0.4,
            center=0,
            square=True,
            linewidths=0.5,
            linecolor="white",
            annot_kws={"fontsize": FONT_CELL},
            xticklabels=tissues,
            yticklabels=[label_of(s) for s in supervisor_markers],
            cbar=False,
            ax=ax,
        )
        ax.set_title(label_of(my_marker))
        ax.tick_params(left=False, bottom=False)
        ax.set_xticklabels(ax.get_xticklabels(), rotation=0)
        ax.set_yticklabels(ax.get_yticklabels(), rotation=0)

    sm = plt.cm.ScalarMappable(cmap="corr_div", norm=plt.Normalize(vmin=-0.4, vmax=0.4))
    sm.set_array([])
    fig.colorbar(sm, ax=axes, shrink=0.8, pad=0.02, label="Spearman ρ")

    if savename:
        save_svg(fig, savename)
    plt.show()


def plot_supervisor_corr_lollipop(
    df,
    my_markers=("mean_p15", "mean_p16", "mean_p21"),
    supervisor_markers=(
        "mean_IR",
        "sen_IR_mar20",
        "sen_RS_mar20",
        "sen_ATVR_feb28",
        "sen_OX_apr13",
        "sen_ANTI_feb28",
    ),
    exclude_tissues=("both",),
    savename=None,
):
    """Same data as plot_supervisor_corr_heatmap but as a dodged lollipop."""
    tissues = [t for t in sorted(df["tissue"].unique()) if t not in exclude_tissues]
    label_of = lambda m: m.replace("mean_", "").replace("sen_", "")

    # vertical dodge so tissues don't overlap within a supervisor row
    dodge_span = 0.6
    offsets = np.linspace(-dodge_span / 2, dodge_span / 2, len(tissues))

    fig, axes = plt.subplots(
        1, len(my_markers), figsize=(4.5 * len(my_markers), 4), sharex=True, sharey=True
    )

    for ax, my_marker in zip(axes, my_markers):
        for i, sup in enumerate(supervisor_markers):
            for j, t in enumerate(tissues):
                sub = df[df["tissue"] == t][[my_marker, sup]].dropna()
                if len(sub) < 5:
                    continue
                r, _ = spearmanr(sub[my_marker], sub[sup])
                y_pos = i + offsets[j]
                color = TISSUE.get(t, "#888")
                ax.hlines(y_pos, 0, r, color=color, lw=1.5, alpha=0.8)
                ax.scatter(
                    r, y_pos, color=color, s=45, edgecolor="black", lw=0.5, zorder=3
                )

        ax.set_yticks(range(len(supervisor_markers)))
        ax.set_yticklabels([label_of(s) for s in supervisor_markers])
        ax.axvline(0, color="gray", lw=0.8)
        ax.set_xlim(-1, 1)
        ax.set_xlabel("Spearman ρ")
        ax.set_title(label_of(my_marker))

    handles = [
        plt.Line2D(
            [],
            [],
            marker="o",
            color=TISSUE.get(t, "#888"),
            ls="",
            markersize=6,
            label=t,
            markeredgecolor="black",
            markeredgewidth=0.5,
        )
        for t in tissues
    ]
    axes[-1].legend(
        handles=handles,
        frameon=False,
        loc="center left",
        bbox_to_anchor=(1.02, 0.5),
        title="tissue",
        title_fontsize=8,
    )

    plt.tight_layout()
    if savename:
        save_svg(fig, savename)
    plt.show()


# ============================================================
# Plot generation calls
# ============================================================
if __name__ == "__main__":

    # -------- Figure 1: Dataset description --------
    plot_nuclei_per_slide(tidy, savename="fig1_nuclei_per_slide")
    plot_cohort_composition_ihc(
        IHC_51_predictions_df, savename="fig1_cohort_composition"
    )
    plot_dab_distribution(tidy, savename="fig1_dab_distribution")
    plot_sampler_triplets(p15_sampler, "/ktb_ihc_p15/p15_cropped/", marker="p15")
    plot_sampler_triplets(
        p16_sampler,
        "/ktb_ihc_p16/p16_cropped/",
        marker="p16",
        seed=24,
        margin=10,
    )
    plot_sampler_triplets(
        p21_sampler,
        "/ktb_ihc_p21/p21_cropped/",
        marker="p21",
        seed=67,
        margin=10,
    )

    # -------- Figure 2: Morphometric analyses --------
    plot_morph_comparison(tidy, savename="fig2_morph_comparison")
    plot_metric_distributions(tidy, savename="fig2_metric_distributions")
    plot_dab_morph_heatmap(sources, savename="fig2_dab_morph_heatmap")
    plot_pca_morphometrics(tidy, morph_dfs, savename="fig2_pca")

    # -------- Figure 3: Training & prediction --------
    tag_map = {
        "AUC": "Metrics/AUC",
        "F1": "Metrics/F1",
        "Sensitivity": "Metrics/Sensitivity",
        "Specificity": "Metrics/Specificity",
    }
    plot_training_curves(tb_dfs, tag_map, savename="fig3_training_curves")

    # The next four require a "prediction" column on each samples df (see TODO above).
    plot_roc_pr(sample_dfs_val, savename="fig3_roc_pr_val")
    plot_confusion_matrices(sample_dfs_val, savename="fig3_confusion_matrices_val")
    plot_slide_level(
        sample_dfs,
        x_col="dab_mean",
        x_label="Mean DAB intensity (per slide)",
        savename="fig3_slide_level_dab",
    )

    plot_slide_level(
        sample_age_dfs, x_col="Age", x_label="Age", savename="fig3_slide_level_age"
    )

    # -------- Figure 4: KomenTissueBank prediction --------
    plot_cohort_composition_ktb(
        p15_p16_p21_IR_tissue_df, savename="fig4_cohort_composition"
    )
    plot_tissue_pointplot(df_pt_tissue, "mean_p15", savename="fig4_pointplot_p15")
    plot_tissue_pointplot(df_pt_tissue, "mean_p16", savename="fig4_pointplot_p16")
    plot_tissue_pointplot(df_pt_tissue, "mean_p21", savename="fig4_pointplot_p21")

    # Appendix
    plot_score_age_pairplot(df_pt_tissue, savename="appendix_score_age_pairplot")

    # OR forest plots — three methods
    for method in ("median", "quartile", "decile"):
        results = or_analysis(df_pt_tissue, SCORE_COLS, method=method)
        plot_or_forest(results, savename=f"fig4_or_forest_{method}")

    # Marker–marker correlations
    plot_marker_corr_lollipop(df_pt_tissue, savename="fig4_marker_corr_lollipop")

    # My markers vs supervisor models — heatmap and lollipop versions
    plot_supervisor_corr_heatmap(
        all_merged_pt_tissue, savename="fig4_supervisor_corr_heatmap"
    )
    plot_supervisor_corr_lollipop(
        all_merged_pt_tissue, savename="fig4_supervisor_corr_lollipop"
    )

    print(f"Total time taken: {(start_time - time.perf_counter())/60} min")
