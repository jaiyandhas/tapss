"""
visualization/heatmaps.py

Layer importance heatmaps and protection map visualisations.

Produces publication-quality matplotlib figures showing:
  - Which transformer layers carry the most important parameters
  - How protection is distributed across layers
  - Side-by-side comparison of multiple importance methods
"""
from __future__ import annotations

import logging
import os
from typing import Optional

import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
import numpy as np
import seaborn as sns
import pandas as pd

from saliency.base import ImportanceScores
from saliency.rankings import RankingResults

logger = logging.getLogger(__name__)

# Research-quality style
plt.rcParams.update({
    "font.family": "DejaVu Sans",
    "axes.facecolor": "#0d1117",
    "figure.facecolor": "#0d1117",
    "text.color": "#c9d1d9",
    "axes.labelcolor": "#c9d1d9",
    "axes.edgecolor": "#30363d",
    "xtick.color": "#8b949e",
    "ytick.color": "#8b949e",
    "grid.color": "#21262d",
    "grid.alpha": 0.7,
})


def _sort_layers(layer_keys: list[str]) -> list[str]:
    """Sort layer keys numerically (layer_0, layer_1, ..., other)."""
    numbered = sorted(
        [k for k in layer_keys if k.startswith("layer_")],
        key=lambda k: int(k.split("_")[1]),
    )
    other = [k for k in layer_keys if not k.startswith("layer_")]
    return numbered + other


def plot_layer_importance_heatmap(
    scores: ImportanceScores,
    title: str = "Layer-wise Parameter Importance",
    save_path: Optional[str] = None,
    show: bool = False,
) -> plt.Figure:
    """
    Plot a heatmap of mean importance score per transformer layer.

    Each row is a transformer layer; the colour intensity represents
    the mean TAPSS score of parameters in that layer.

    Parameters
    ----------
    scores : ImportanceScores
        Importance scores from any estimator.
    title : str
        Figure title.
    save_path : str | None
        If provided, save the figure to this path.
    show : bool
        If True, call plt.show().

    Returns
    -------
    matplotlib.figure.Figure
    """
    layer_scores = scores.layer_aggregated()
    layers = _sort_layers(list(layer_scores.keys()))
    values = np.array([[layer_scores[l] for l in layers]])  # 1 × num_layers

    fig, ax = plt.subplots(figsize=(max(10, len(layers) * 0.8), 3.5))
    im = ax.imshow(
        values,
        aspect="auto",
        cmap="plasma",
        vmin=0.0,
        vmax=1.0,
    )

    ax.set_xticks(range(len(layers)))
    ax.set_xticklabels(layers, rotation=45, ha="right", fontsize=9)
    ax.set_yticks([0])
    ax.set_yticklabels([scores.method], fontsize=10)

    # Annotate cells
    for j, val in enumerate(values[0]):
        ax.text(
            j, 0, f"{val:.2f}",
            ha="center", va="center",
            color="white" if val < 0.7 else "black",
            fontsize=8, fontweight="bold",
        )

    plt.colorbar(im, ax=ax, label="Mean Importance Score", shrink=0.8)
    ax.set_title(title, fontsize=13, fontweight="bold", color="#58a6ff", pad=10)
    plt.tight_layout()

    if save_path:
        os.makedirs(os.path.dirname(save_path) or ".", exist_ok=True)
        fig.savefig(save_path, dpi=150, bbox_inches="tight")
        logger.info(f"Layer importance heatmap saved: {save_path}")

    if show:
        plt.show()

    return fig


def plot_protection_map(
    rankings: RankingResults,
    title: str = "Parameter Protection Map",
    save_path: Optional[str] = None,
    show: bool = False,
) -> plt.Figure:
    """
    Visualise which parameters are protected, broken down by layer.

    For each layer: bar showing total parameters vs protected parameters.

    Parameters
    ----------
    rankings : RankingResults
    title : str
    save_path : str | None
    show : bool

    Returns
    -------
    matplotlib.figure.Figure
    """
    df = rankings.as_dataframe()
    if df.empty:
        fig, ax = plt.subplots()
        ax.text(0.5, 0.5, "No data", ha="center", va="center")
        return fig

    layer_stats = (
        df.groupby("layer")
        .agg(
            total=("param_name", "count"),
            protected=("is_protected", "sum"),
        )
        .reset_index()
    )

    layers = _sort_layers(layer_stats["layer"].tolist())
    layer_stats = layer_stats.set_index("layer").loc[layers].reset_index()

    total = layer_stats["total"].values
    protected = layer_stats["protected"].values.astype(float)
    unprotected = total - protected

    x = np.arange(len(layers))
    width = 0.6

    fig, ax = plt.subplots(figsize=(max(10, len(layers) * 0.9), 5))
    bars_u = ax.bar(x, unprotected, width, label="Unprotected", color="#3b82f6", alpha=0.85)
    bars_p = ax.bar(
        x, protected, width, bottom=unprotected, label="Protected", color="#f59e0b", alpha=0.9
    )

    ax.set_xticks(x)
    ax.set_xticklabels(layers, rotation=45, ha="right", fontsize=9)
    ax.set_ylabel("Number of Parameters", fontsize=11)
    ax.set_title(title, fontsize=13, fontweight="bold", color="#58a6ff", pad=10)
    ax.legend(framealpha=0.3)
    ax.grid(axis="y", alpha=0.3)

    # Annotate protection percentage
    for xi, (tot, prot) in enumerate(zip(total, protected)):
        pct = 100.0 * prot / max(tot, 1)
        if pct > 0:
            ax.text(xi, tot + 0.5, f"{pct:.0f}%", ha="center", fontsize=8, color="#fbbf24")

    plt.tight_layout()

    if save_path:
        os.makedirs(os.path.dirname(save_path) or ".", exist_ok=True)
        fig.savefig(save_path, dpi=150, bbox_inches="tight")
        logger.info(f"Protection map saved: {save_path}")

    if show:
        plt.show()

    return fig


def plot_multi_method_heatmap(
    all_scores: dict[str, ImportanceScores],
    title: str = "Importance Score Comparison (All Methods)",
    save_path: Optional[str] = None,
    show: bool = False,
) -> plt.Figure:
    """
    Compare layer-wise importance across multiple methods in one heatmap.

    Rows = methods, Columns = transformer layers.

    Parameters
    ----------
    all_scores : dict[str, ImportanceScores]
        {method_name: scores} mapping.
    title, save_path, show : standard plotting kwargs.

    Returns
    -------
    matplotlib.figure.Figure
    """
    # Collect all layer keys
    all_layers: set[str] = set()
    method_layer_scores: dict[str, dict[str, float]] = {}

    for method, scores in all_scores.items():
        layer_agg = scores.layer_aggregated()
        method_layer_scores[method] = layer_agg
        all_layers.update(layer_agg.keys())

    layers = _sort_layers(list(all_layers))
    methods = list(all_scores.keys())

    # Build 2D matrix: methods × layers
    matrix = np.zeros((len(methods), len(layers)))
    for i, method in enumerate(methods):
        for j, layer in enumerate(layers):
            matrix[i, j] = method_layer_scores[method].get(layer, 0.0)

    fig, ax = plt.subplots(figsize=(max(12, len(layers) * 0.9), max(4, len(methods) * 0.8 + 2)))
    im = ax.imshow(matrix, aspect="auto", cmap="plasma", vmin=0.0, vmax=1.0)

    ax.set_xticks(range(len(layers)))
    ax.set_xticklabels(layers, rotation=45, ha="right", fontsize=9)
    ax.set_yticks(range(len(methods)))
    ax.set_yticklabels(methods, fontsize=10)

    # Annotate
    for i in range(len(methods)):
        for j in range(len(layers)):
            val = matrix[i, j]
            ax.text(
                j, i, f"{val:.2f}",
                ha="center", va="center",
                color="white" if val < 0.65 else "black",
                fontsize=7,
            )

    plt.colorbar(im, ax=ax, label="Mean Importance Score", shrink=0.8)
    ax.set_title(title, fontsize=13, fontweight="bold", color="#58a6ff", pad=10)
    plt.tight_layout()

    if save_path:
        os.makedirs(os.path.dirname(save_path) or ".", exist_ok=True)
        fig.savefig(save_path, dpi=150, bbox_inches="tight")
        logger.info(f"Multi-method heatmap saved: {save_path}")

    if show:
        plt.show()

    return fig
