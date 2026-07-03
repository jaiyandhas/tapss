"""
visualization/distributions.py

Distribution plots for TAPSS importance scores, gradients, and activations.

Includes:
  - Importance score histograms (with protection threshold marked)
  - Gradient distribution across layers
  - Activation magnitude distribution
  - Score scatter: TAPSS score vs individual component scores
"""
from __future__ import annotations

import logging
import os
from typing import Optional

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns

from saliency.base import ImportanceScores
from saliency.rankings import RankingResults

logger = logging.getLogger(__name__)


def plot_importance_histogram(
    scores: ImportanceScores,
    protection_threshold: Optional[float] = None,
    title: str = "Parameter Importance Score Distribution",
    save_path: Optional[str] = None,
    show: bool = False,
) -> plt.Figure:
    """
    Plot a histogram of parameter importance scores.

    If protection_threshold is provided, marks the threshold line
    separating protected from unprotected parameters.

    Parameters
    ----------
    scores : ImportanceScores
    protection_threshold : float | None
        Importance score value above which parameters are protected.
    title, save_path, show : standard args.

    Returns
    -------
    matplotlib.figure.Figure
    """
    values = list(scores.scores.values())

    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    fig.patch.set_facecolor("#0d1117")

    # ── Left: Full distribution ──
    ax = axes[0]
    ax.set_facecolor("#0d1117")
    n, bins, patches = ax.hist(values, bins=50, color="#3b82f6", alpha=0.8, edgecolor="#1e3a5f")

    # Colour bars above threshold differently
    if protection_threshold is not None:
        for patch, left in zip(patches, bins[:-1]):
            if left >= protection_threshold:
                patch.set_facecolor("#f59e0b")
                patch.set_alpha(0.9)
        ax.axvline(
            x=protection_threshold,
            color="#ef4444",
            linestyle="--",
            linewidth=2,
            label=f"Protection threshold ({protection_threshold:.2f})",
        )
        ax.legend(framealpha=0.3, fontsize=9)

    ax.set_xlabel("Importance Score", fontsize=11, color="#c9d1d9")
    ax.set_ylabel("Count", fontsize=11, color="#c9d1d9")
    ax.set_title(f"{title}\nMethod: {scores.method}", fontsize=11, color="#58a6ff")
    ax.tick_params(colors="#8b949e")
    for spine in ax.spines.values():
        spine.set_edgecolor("#30363d")

    # ── Right: Cumulative distribution ──
    ax2 = axes[1]
    ax2.set_facecolor("#0d1117")
    sorted_vals = np.sort(values)
    cdf = np.arange(1, len(sorted_vals) + 1) / len(sorted_vals)
    ax2.plot(sorted_vals, cdf, color="#58a6ff", linewidth=2)
    ax2.fill_between(sorted_vals, cdf, alpha=0.15, color="#3b82f6")

    if protection_threshold is not None:
        ax2.axvline(x=protection_threshold, color="#ef4444", linestyle="--", linewidth=2)
        # Mark fraction of protected params
        protected_frac = sum(v >= protection_threshold for v in values) / len(values)
        ax2.axhline(
            y=1.0 - protected_frac,
            color="#f59e0b",
            linestyle=":",
            linewidth=1.5,
            label=f"Protected: {protected_frac * 100:.1f}%",
        )
        ax2.legend(framealpha=0.3, fontsize=9)

    ax2.set_xlabel("Importance Score", fontsize=11, color="#c9d1d9")
    ax2.set_ylabel("Cumulative Fraction", fontsize=11, color="#c9d1d9")
    ax2.set_title("Cumulative Distribution", fontsize=11, color="#58a6ff")
    ax2.tick_params(colors="#8b949e")
    for spine in ax2.spines.values():
        spine.set_edgecolor("#30363d")

    plt.suptitle(
        f"TAPSS Score Distribution — {scores.method}",
        fontsize=13,
        fontweight="bold",
        color="#c9d1d9",
        y=1.02,
    )
    plt.tight_layout()

    if save_path:
        os.makedirs(os.path.dirname(save_path) or ".", exist_ok=True)
        fig.savefig(save_path, dpi=150, bbox_inches="tight", facecolor=fig.get_facecolor())
        logger.info(f"Importance histogram saved: {save_path}")

    if show:
        plt.show()

    return fig


def plot_gradient_distribution(
    scores: ImportanceScores,
    title: str = "Gradient Magnitude Distribution by Layer",
    save_path: Optional[str] = None,
    show: bool = False,
) -> plt.Figure:
    """
    Box/violin plot of importance scores grouped by transformer layer.

    Shows how gradient magnitude varies across the network depth.

    Parameters
    ----------
    scores : ImportanceScores
    title, save_path, show : standard args.

    Returns
    -------
    matplotlib.figure.Figure
    """
    # Group scores by layer
    layer_data: dict[str, list[float]] = {}
    for name, score in scores.scores.items():
        parts = name.split(".")
        layer_key = "other"
        for i, part in enumerate(parts):
            if part == "layer" and i + 1 < len(parts) and parts[i + 1].isdigit():
                layer_key = f"L{parts[i + 1]}"
                break
        layer_data.setdefault(layer_key, []).append(score)

    # Sort layers
    layers = sorted(
        [k for k in layer_data if k.startswith("L")],
        key=lambda k: int(k[1:]),
    ) + [k for k in layer_data if not k.startswith("L")]

    data = [layer_data[l] for l in layers if l in layer_data]
    layers = [l for l in layers if l in layer_data]

    fig, ax = plt.subplots(figsize=(max(10, len(layers) * 0.9), 6))
    fig.patch.set_facecolor("#0d1117")
    ax.set_facecolor("#0d1117")

    bp = ax.violinplot(
        data,
        positions=range(len(layers)),
        showmeans=True,
        showmedians=True,
        showextrema=True,
    )

    # Style violin bodies
    for body in bp["bodies"]:
        body.set_facecolor("#3b82f6")
        body.set_alpha(0.7)
        body.set_edgecolor("#58a6ff")

    for part in ["cmeans", "cmedians", "cmins", "cmaxes", "cbars"]:
        if part in bp:
            bp[part].set_color("#f59e0b" if part == "cmeans" else "#c9d1d9")

    ax.set_xticks(range(len(layers)))
    ax.set_xticklabels(layers, fontsize=10, color="#8b949e")
    ax.set_xlabel("Transformer Layer", fontsize=12, color="#c9d1d9")
    ax.set_ylabel("Importance Score", fontsize=12, color="#c9d1d9")
    ax.set_title(title, fontsize=13, fontweight="bold", color="#58a6ff", pad=10)
    ax.tick_params(colors="#8b949e")
    ax.grid(axis="y", alpha=0.3, color="#30363d")
    for spine in ax.spines.values():
        spine.set_edgecolor("#30363d")

    plt.tight_layout()

    if save_path:
        os.makedirs(os.path.dirname(save_path) or ".", exist_ok=True)
        fig.savefig(save_path, dpi=150, bbox_inches="tight", facecolor=fig.get_facecolor())
        logger.info(f"Gradient distribution saved: {save_path}")

    if show:
        plt.show()

    return fig


def plot_activation_distribution(
    scores: ImportanceScores,
    title: str = "Activation Magnitude Distribution",
    save_path: Optional[str] = None,
    show: bool = False,
) -> plt.Figure:
    """Alias of plot_gradient_distribution for activation scores."""
    return plot_gradient_distribution(scores, title=title, save_path=save_path, show=show)


def plot_score_scatter(
    tapss_scores: ImportanceScores,
    component_scores: dict[str, ImportanceScores],
    title: str = "TAPSS Score vs Component Scores",
    save_path: Optional[str] = None,
    show: bool = False,
) -> plt.Figure:
    """
    Scatter plots: TAPSS combined score vs each individual component score.

    Shows how well each component correlates with the combined TAPSS score.
    Useful for understanding which component drives TAPSS decisions.

    Parameters
    ----------
    tapss_scores : ImportanceScores
        Combined TAPSS scores.
    component_scores : dict[str, ImportanceScores]
        Individual component scores (gradient, perturbation, etc.)
    title, save_path, show : standard args.

    Returns
    -------
    matplotlib.figure.Figure
    """
    n_components = len(component_scores)
    if n_components == 0:
        fig, ax = plt.subplots()
        ax.text(0.5, 0.5, "No component scores", ha="center", va="center")
        return fig

    ncols = min(2, n_components)
    nrows = (n_components + 1) // 2

    fig, axes = plt.subplots(nrows, ncols, figsize=(7 * ncols, 5 * nrows))
    fig.patch.set_facecolor("#0d1117")
    axes_flat = np.array(axes).flatten() if n_components > 1 else [axes]

    param_names = list(tapss_scores.scores.keys())
    tapss_vals = np.array([tapss_scores.scores[n] for n in param_names])

    colors = ["#3b82f6", "#f59e0b", "#10b981", "#ef4444", "#8b5cf6"]

    for ax_idx, (method_name, comp_scores) in enumerate(component_scores.items()):
        ax = axes_flat[ax_idx]
        ax.set_facecolor("#0d1117")

        comp_vals = np.array([comp_scores.scores.get(n, 0.0) for n in param_names])
        color = colors[ax_idx % len(colors)]

        ax.scatter(comp_vals, tapss_vals, alpha=0.3, s=6, color=color)

        # Correlation annotation
        corr = float(np.corrcoef(comp_vals, tapss_vals)[0, 1])
        ax.text(
            0.05, 0.92, f"r = {corr:.3f}",
            transform=ax.transAxes,
            fontsize=11, color="#c9d1d9",
            bbox=dict(facecolor="#21262d", alpha=0.8, edgecolor="#30363d"),
        )

        ax.set_xlabel(f"{method_name} score", fontsize=10, color="#c9d1d9")
        ax.set_ylabel("TAPSS score", fontsize=10, color="#c9d1d9")
        ax.set_title(f"TAPSS vs {method_name}", fontsize=11, color="#58a6ff")
        ax.tick_params(colors="#8b949e")
        for spine in ax.spines.values():
            spine.set_edgecolor("#30363d")

    # Hide unused axes
    for ax_idx in range(n_components, len(axes_flat)):
        axes_flat[ax_idx].set_visible(False)

    plt.suptitle(title, fontsize=13, fontweight="bold", color="#c9d1d9", y=1.01)
    plt.tight_layout()

    if save_path:
        os.makedirs(os.path.dirname(save_path) or ".", exist_ok=True)
        fig.savefig(save_path, dpi=150, bbox_inches="tight", facecolor=fig.get_facecolor())
        logger.info(f"Score scatter saved: {save_path}")

    if show:
        plt.show()

    return fig
