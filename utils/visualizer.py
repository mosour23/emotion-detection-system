"""
=============================================================================
MODULE: visualizer.py
PURPOSE: Data Distribution and Model Performance Visualisations
DESCRIPTION:
    Provides chart-generating functions used during EDA and post-training
    evaluation. All charts are saved to disk as high-resolution PNGs and
    also returned as Matplotlib Figure objects for embedding in reports.
=============================================================================
"""

import logging
from pathlib import Path
from typing import Optional

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")                          # headless / server-safe backend
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import seaborn as sns
from wordcloud import WordCloud

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Consistent colour palette (matches EMOTION_COLOR_MAP in data_loader.py)
# ---------------------------------------------------------------------------
PALETTE = {
    "sadness":  "#5B8FDE",
    "joy":      "#F4D03F",
    "love":     "#E91E8C",
    "anger":    "#E74C3C",
    "fear":     "#8E44AD",
    "surprise": "#1ABC9C",
}
DEFAULT_COLORS = list(PALETTE.values())

sns.set_theme(style="whitegrid", font_scale=1.1)


def _savefig(fig: plt.Figure, path: str) -> None:
    """Save figure and close it to free memory."""
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    logger.info("Figure saved → %s", path)


# =============================================================================
# 1. Class distribution bar chart
# =============================================================================

def plot_class_distribution(
    df: pd.DataFrame,
    label_col: str = "emotion",
    save_path: Optional[str] = None,
) -> plt.Figure:
    """
    Bar chart of emotion label frequencies with percentage annotations.

    Parameters
    ----------
    df        : pd.DataFrame – dataset with emotion labels
    label_col : str          – column name for emotion labels
    save_path : str | None   – optional output file path
    """
    counts = df[label_col].value_counts().reset_index()
    counts.columns = ["emotion", "count"]
    counts["pct"] = counts["count"] / counts["count"].sum() * 100

    fig, ax = plt.subplots(figsize=(10, 5))
    colors = [PALETTE.get(e, "#7F8C8D") for e in counts["emotion"]]
    bars = ax.bar(counts["emotion"], counts["count"], color=colors, edgecolor="white", linewidth=1.5)

    for bar, pct in zip(bars, counts["pct"]):
        ax.text(
            bar.get_x() + bar.get_width() / 2,
            bar.get_height() + 15,
            f"{pct:.1f}%",
            ha="center", va="bottom", fontsize=10, fontweight="bold",
        )

    ax.set_title("Emotion Class Distribution", fontsize=14, fontweight="bold", pad=15)
    ax.set_xlabel("Emotion", fontsize=12)
    ax.set_ylabel("Sample Count", fontsize=12)
    ax.set_ylim(0, counts["count"].max() * 1.15)
    plt.tight_layout()

    if save_path:
        _savefig(fig, save_path)
    return fig


# =============================================================================
# 2. Confusion matrix heatmap
# =============================================================================

def plot_confusion_matrix(
    cm: list | np.ndarray,
    emotion_names: list[str],
    save_path: Optional[str] = None,
    normalise: bool = True,
) -> plt.Figure:
    """
    Annotated confusion matrix heatmap.

    Parameters
    ----------
    cm            : 2-D array-like – raw confusion matrix
    emotion_names : list[str]      – class label strings
    save_path     : str | None
    normalise     : bool           – show row-normalised percentages
    """
    cm_arr = np.array(cm, dtype=float)
    title = "Confusion Matrix"

    if normalise:
        row_sums = cm_arr.sum(axis=1, keepdims=True)
        cm_arr = np.where(row_sums == 0, 0, cm_arr / row_sums)
        title += " (Row-Normalised)"
        fmt = ".2f"
        vmin, vmax = 0.0, 1.0
    else:
        fmt = ".0f"
        vmin, vmax = None, None

    fig, ax = plt.subplots(figsize=(9, 7))
    sns.heatmap(
        cm_arr,
        annot=True,
        fmt=fmt,
        xticklabels=emotion_names,
        yticklabels=emotion_names,
        cmap="Blues",
        linewidths=0.5,
        linecolor="white",
        vmin=vmin,
        vmax=vmax,
        ax=ax,
    )
    ax.set_title(title, fontsize=14, fontweight="bold", pad=12)
    ax.set_xlabel("Predicted Label", fontsize=12)
    ax.set_ylabel("True Label",      fontsize=12)
    plt.tight_layout()

    if save_path:
        _savefig(fig, save_path)
    return fig


# =============================================================================
# 3. Per-class F1 radar / bar chart
# =============================================================================

def plot_per_class_f1(
    per_class_f1: dict[str, float],
    save_path: Optional[str] = None,
) -> plt.Figure:
    """
    Horizontal bar chart of per-class F1 scores.

    Parameters
    ----------
    per_class_f1 : dict[str, float] – {emotion: f1_score}
    save_path    : str | None
    """
    emotions = list(per_class_f1.keys())
    scores   = list(per_class_f1.values())
    colors   = [PALETTE.get(e, "#7F8C8D") for e in emotions]

    fig, ax = plt.subplots(figsize=(8, 5))
    bars = ax.barh(emotions, scores, color=colors, edgecolor="white", height=0.6)

    for bar, score in zip(bars, scores):
        ax.text(
            score + 0.005, bar.get_y() + bar.get_height() / 2,
            f"{score:.3f}", va="center", fontsize=10,
        )

    ax.set_xlim(0, 1.05)
    ax.set_xlabel("F1 Score", fontsize=12)
    ax.set_title("Per-Class F1 Scores", fontsize=14, fontweight="bold")
    ax.axvline(0.7, color="grey", linestyle="--", linewidth=1, label="0.70 threshold")
    ax.legend(fontsize=9)
    plt.tight_layout()

    if save_path:
        _savefig(fig, save_path)
    return fig


# =============================================================================
# 4. Text length distribution
# =============================================================================

def plot_text_length_distribution(
    df: pd.DataFrame,
    text_col: str = "text",
    label_col: str = "emotion",
    save_path: Optional[str] = None,
) -> plt.Figure:
    """
    KDE / violin plot of text token length by emotion class.

    Parameters
    ----------
    df        : pd.DataFrame
    text_col  : str – raw text column
    label_col : str – emotion label column
    save_path : str | None
    """
    df2 = df.copy()
    df2["token_count"] = df2[text_col].fillna("").apply(lambda t: len(t.split()))

    fig, axes = plt.subplots(1, 2, figsize=(14, 5))

    # ── Left: overall histogram ──
    axes[0].hist(df2["token_count"], bins=40, color="#5B8FDE", edgecolor="white")
    axes[0].set_title("Overall Token-Length Distribution", fontweight="bold")
    axes[0].set_xlabel("Token Count")
    axes[0].set_ylabel("Frequency")

    # ── Right: box-plot by emotion ──
    order = df2[label_col].value_counts().index.tolist()
    palette = {e: PALETTE.get(e, "#7F8C8D") for e in order}
    sns.boxplot(
        data=df2, x=label_col, y="token_count",
        order=order, palette=palette, ax=axes[1],
    )
    axes[1].set_title("Token Length by Emotion Class", fontweight="bold")
    axes[1].set_xlabel("Emotion")
    axes[1].set_ylabel("Token Count")
    axes[1].tick_params(axis="x", rotation=15)

    plt.tight_layout()

    if save_path:
        _savefig(fig, save_path)
    return fig


# =============================================================================
# 5. Word cloud per emotion
# =============================================================================

def plot_wordclouds(
    df: pd.DataFrame,
    text_col: str = "clean_text",
    label_col: str = "emotion",
    save_path: Optional[str] = None,
    max_words: int = 80,
) -> plt.Figure:
    """
    Grid of word clouds, one per emotion class.

    Parameters
    ----------
    df        : pd.DataFrame (must contain a cleaned text column)
    text_col  : str
    label_col : str
    save_path : str | None
    max_words : int
    """
    emotions = sorted(df[label_col].unique())
    n = len(emotions)
    cols = 3
    rows = (n + cols - 1) // cols

    fig, axes = plt.subplots(rows, cols, figsize=(cols * 5, rows * 4))
    axes = axes.flatten()

    for i, emotion in enumerate(emotions):
        corpus = " ".join(df.loc[df[label_col] == emotion, text_col].fillna("").tolist())
        wc = WordCloud(
            width=400,
            height=300,
            background_color="white",
            max_words=max_words,
            colormap="tab10",
            random_state=42,
        ).generate(corpus if corpus.strip() else "no data")

        axes[i].imshow(wc, interpolation="bilinear")
        axes[i].axis("off")
        axes[i].set_title(emotion.capitalize(), fontsize=13, fontweight="bold",
                           color=PALETTE.get(emotion, "#333333"))

    # Hide unused axes
    for j in range(i + 1, len(axes)):
        axes[j].axis("off")

    fig.suptitle("Top Words per Emotion Class", fontsize=16, fontweight="bold", y=1.01)
    plt.tight_layout()

    if save_path:
        _savefig(fig, save_path)
    return fig


# =============================================================================
# 6. Metrics summary dashboard
# =============================================================================

def plot_metrics_summary(
    metrics: dict,
    save_path: Optional[str] = None,
) -> plt.Figure:
    """
    Single-page dashboard: overall metrics bar + per-class F1 + confusion matrix.

    Parameters
    ----------
    metrics   : dict – output from EmotionModelTrainer.evaluate()
    save_path : str | None
    """
    fig = plt.figure(figsize=(16, 10))
    gs  = gridspec.GridSpec(2, 2, figure=fig, hspace=0.4, wspace=0.35)

    # ── Top-left: overall metrics ──
    ax1 = fig.add_subplot(gs[0, 0])
    overall = {
        "Accuracy":    metrics["accuracy"],
        "Precision":   metrics["precision"],
        "Sensitivity": metrics.get("sensitivity", metrics["recall"]),
        "Specificity": metrics.get("specificity", 0.0),
        "Recall":      metrics["recall"],
        "F1 (wtd)":    metrics["f1_weighted"],
    }
    bars = ax1.bar(
        list(overall.keys()), list(overall.values()),
        color=["#3498DB", "#2ECC71", "#E67E22", "#E74C3C", "#F39C12", "#9B59B6"],
        edgecolor="white",
    )
    for bar, val in zip(bars, overall.values()):
        ax1.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.005,
                 f"{val:.3f}", ha="center", va="bottom", fontsize=9, fontweight="bold")
    ax1.set_ylim(0, 1.1)
    ax1.set_title("Overall Metrics", fontweight="bold")
    ax1.set_ylabel("Score")
    ax1.tick_params(axis="x", rotation=15)

    # ── Top-right: per-class F1 ──
    ax2 = fig.add_subplot(gs[0, 1])
    pf1 = metrics["per_class_f1"]
    emotions = list(pf1.keys())
    scores   = list(pf1.values())
    colors   = [PALETTE.get(e, "#7F8C8D") for e in emotions]
    h_bars = ax2.barh(emotions, scores, color=colors, height=0.5)
    for bar, score in zip(h_bars, scores):
        ax2.text(score + 0.005, bar.get_y() + bar.get_height() / 2,
                 f"{score:.3f}", va="center", fontsize=9)
    ax2.set_xlim(0, 1.1)
    ax2.set_title("Per-Class F1", fontweight="bold")
    ax2.set_xlabel("F1 Score")

    # ── Bottom: confusion matrix ──
    ax3 = fig.add_subplot(gs[1, :])
    cm_arr   = np.array(metrics["confusion_matrix"], dtype=float)
    row_sums = cm_arr.sum(axis=1, keepdims=True)
    cm_norm  = np.where(row_sums == 0, 0, cm_arr / row_sums)
    emotion_names = metrics["emotion_names"]

    sns.heatmap(
        cm_norm, annot=True, fmt=".2f",
        xticklabels=emotion_names, yticklabels=emotion_names,
        cmap="Blues", linewidths=0.4, linecolor="white", ax=ax3,
    )
    ax3.set_title("Normalised Confusion Matrix", fontweight="bold")
    ax3.set_xlabel("Predicted")
    ax3.set_ylabel("True")

    fig.suptitle("Emotion Detection – Model Evaluation Dashboard", fontsize=16, fontweight="bold")

    if save_path:
        _savefig(fig, save_path)
    return fig
