"""
visualization/continual.py

Continual learning result visualisations.

Includes:
  - Forgetting comparison bar chart (all methods)
  - Task accuracy comparison (grouped bar)
  - Training history (loss/acc curves)
  - Radar chart (multi-metric method comparison)
  - Forgetting over training steps (if available)
"""
from __future__ import annotations

import logging
import os
from typing import Any, Optional

import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import numpy as np

logger = logging.getLogger(__name__)

# Dark theme constants
DARK_BG = "#0d1117"
TEXT_COLOR = "#c9d1d9"
ACCENT_BLUE = "#58a6ff"
ACCENT_AMBER = "#f59e0b"
ACCENT_GREEN = "#3fb950"
ACCENT_RED = "#f85149"
EDGE_COLOR = "#30363d"

METHOD_COLORS = {
    "tapss": "#3b82f6",
    "vanilla_lora": "#10b981",
    "naive_finetuning": "#ef4444",
    "ewc": "#f59e0b",
    "random_protection": "#8b5cf6",
}

DEFAULT_COLORS = ["#3b82f6", "#10b981", "#ef4444", "#f59e0b", "#8b5cf6", "#06b6d4"]


def _get_color(method: str, idx: int) -> str:
    return METHOD_COLORS.get(method.lower(), DEFAULT_COLORS[idx % len(DEFAULT_COLORS)])


def _apply_dark_theme(fig: plt.Figure, axes) -> None:
    fig.patch.set_facecolor(DARK_BG)
    ax_list = np.array(axes).flatten() if hasattr(axes, "__iter__") else [axes]
    for ax in ax_list:
        ax.set_facecolor(DARK_BG)
        ax.tick_params(colors="#8b949e")
        ax.xaxis.label.set_color(TEXT_COLOR)
        ax.yaxis.label.set_color(TEXT_COLOR)
        ax.title.set_color(ACCENT_BLUE)
        for spine in ax.spines.values():
            spine.set_edgecolor(EDGE_COLOR)
        ax.grid(alpha=0.3, color=EDGE_COLOR)


def plot_forgetting_over_methods(
    results: list[Any],  # list[CLResult]
    title: str = "Catastrophic Forgetting by Method",
    save_path: Optional[str] = None,
    show: bool = False,
) -> plt.Figure:
    """
    Horizontal bar chart comparing catastrophic forgetting across methods.

    TAPSS is highlighted in a distinct colour.
    Lower forgetting = better.

    Parameters
    ----------
    results : list[CLResult]
    title, save_path, show : standard args.

    Returns
    -------
    matplotlib.figure.Figure
    """
    methods = [r.method_name for r in results]
    forgetting = [r.forgetting for r in results]
    colors = [_get_color(m, i) for i, m in enumerate(methods)]

    # Sort by forgetting (ascending = best first)
    order = np.argsort(forgetting)
    methods = [methods[i] for i in order]
    forgetting = [forgetting[i] for i in order]
    colors = [colors[i] for i in order]

    fig, ax = plt.subplots(figsize=(10, max(4, len(methods) * 0.7 + 1)))
    _apply_dark_theme(fig, ax)

    bars = ax.barh(methods, forgetting, color=colors, alpha=0.85, edgecolor=EDGE_COLOR)

    # Annotate bar values
    for bar, val in zip(bars, forgetting):
        ax.text(
            val + 0.002, bar.get_y() + bar.get_height() / 2,
            f"{val:.4f}",
            va="center", ha="left", fontsize=9, color=TEXT_COLOR,
        )

    ax.set_xlabel("Catastrophic Forgetting (lower = better)", fontsize=11)
    ax.set_title(title, fontsize=13, fontweight="bold", pad=10)
    ax.axvline(x=0, color=EDGE_COLOR, linewidth=0.8)

    # Legend for TAPSS
    if "tapss" in [m.lower() for m in methods]:
        ax.annotate(
            "◀ TAPSS",
            xy=(0, 0), xycoords="axes fraction",
            fontsize=9, color="#3b82f6", ha="right",
        )

    plt.tight_layout()

    if save_path:
        os.makedirs(os.path.dirname(save_path) or ".", exist_ok=True)
        fig.savefig(save_path, dpi=150, bbox_inches="tight", facecolor=DARK_BG)
        logger.info(f"Forgetting chart saved: {save_path}")

    if show:
        plt.show()

    return fig


def plot_task_accuracy_comparison(
    results: list[Any],
    title: str = "Task Accuracy Comparison",
    save_path: Optional[str] = None,
    show: bool = False,
) -> plt.Figure:
    """
    Grouped bar chart: Task A post-accuracy vs Task B accuracy, per method.

    Parameters
    ----------
    results : list[CLResult]
    title, save_path, show : standard args.

    Returns
    -------
    matplotlib.figure.Figure
    """
    methods = [r.method_name for r in results]
    task_a_post = [r.task_a_post_accuracy for r in results]
    task_b = [r.task_b_accuracy for r in results]

    x = np.arange(len(methods))
    width = 0.35

    fig, ax = plt.subplots(figsize=(max(10, len(methods) * 1.5), 6))
    _apply_dark_theme(fig, ax)

    bars_a = ax.bar(x - width / 2, task_a_post, width, label="Task A (post-B)", color="#3b82f6", alpha=0.85)
    bars_b = ax.bar(x + width / 2, task_b, width, label="Task B", color="#10b981", alpha=0.85)

    # Annotate
    for bar in list(bars_a) + list(bars_b):
        h = bar.get_height()
        ax.text(
            bar.get_x() + bar.get_width() / 2, h + 0.005,
            f"{h:.3f}", ha="center", va="bottom", fontsize=8, color=TEXT_COLOR,
        )

    ax.set_xticks(x)
    ax.set_xticklabels(methods, rotation=20, ha="right", fontsize=10)
    ax.set_ylim(0, min(1.15, max(max(task_a_post), max(task_b)) + 0.15))
    ax.set_ylabel("Accuracy", fontsize=11)
    ax.set_title(title, fontsize=13, fontweight="bold", pad=10)
    ax.legend(framealpha=0.3, fontsize=10)
    ax.yaxis.set_major_formatter(plt.FuncFormatter(lambda y, _: f"{y:.0%}"))

    plt.tight_layout()

    if save_path:
        os.makedirs(os.path.dirname(save_path) or ".", exist_ok=True)
        fig.savefig(save_path, dpi=150, bbox_inches="tight", facecolor=DARK_BG)
        logger.info(f"Task accuracy comparison saved: {save_path}")

    if show:
        plt.show()

    return fig


def plot_training_history(
    history: Any,  # TrainingHistory
    method_name: str = "",
    task: str = "Task",
    save_path: Optional[str] = None,
    show: bool = False,
) -> plt.Figure:
    """
    Plot training loss and accuracy curves over epochs.

    Parameters
    ----------
    history : TrainingHistory
    method_name : str
    task : str
    save_path, show : standard args.

    Returns
    -------
    matplotlib.figure.Figure
    """
    epochs = list(range(1, len(history.train_losses) + 1))

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))
    _apply_dark_theme(fig, [ax1, ax2])

    # Loss curves
    ax1.plot(epochs, history.train_losses, color="#3b82f6", linewidth=2, label="Train")
    ax1.plot(epochs, history.val_losses, color="#f59e0b", linewidth=2, linestyle="--", label="Val")
    if any(l > 0 for l in history.additional_losses):
        ax1.plot(
            epochs, history.additional_losses,
            color="#8b5cf6", linewidth=1.5, linestyle=":",
            label="Reg loss",
        )
    ax1.set_xlabel("Epoch", fontsize=11)
    ax1.set_ylabel("Loss", fontsize=11)
    ax1.set_title(f"{method_name} — {task} Loss", fontsize=12, fontweight="bold")
    ax1.legend(framealpha=0.3)

    # Accuracy curves
    ax2.plot(epochs, history.train_accuracies, color="#3b82f6", linewidth=2, label="Train")
    ax2.plot(epochs, history.val_accuracies, color="#f59e0b", linewidth=2, linestyle="--", label="Val")
    ax2.set_xlabel("Epoch", fontsize=11)
    ax2.set_ylabel("Accuracy", fontsize=11)
    ax2.set_title(f"{method_name} — {task} Accuracy", fontsize=12, fontweight="bold")
    ax2.set_ylim(0, 1.05)
    ax2.yaxis.set_major_formatter(plt.FuncFormatter(lambda y, _: f"{y:.0%}"))
    ax2.legend(framealpha=0.3)

    plt.tight_layout()

    if save_path:
        os.makedirs(os.path.dirname(save_path) or ".", exist_ok=True)
        fig.savefig(save_path, dpi=150, bbox_inches="tight", facecolor=DARK_BG)
        logger.info(f"Training history saved: {save_path}")

    if show:
        plt.show()

    return fig


def plot_radar_chart(
    results: list[Any],
    title: str = "Multi-metric Method Comparison",
    save_path: Optional[str] = None,
    show: bool = False,
) -> plt.Figure:
    """
    Radar (spider) chart comparing methods across multiple metrics.

    Metrics shown:
      - Task A retention (1 - forgetting)
      - Task B accuracy
      - Average accuracy
      - Backward transfer (normalised)
      - Parameter efficiency (1 - trainable_params fraction)

    Parameters
    ----------
    results : list[CLResult]
    title, save_path, show : standard args.

    Returns
    -------
    matplotlib.figure.Figure
    """
    if not results:
        fig, ax = plt.subplots()
        ax.text(0.5, 0.5, "No results", ha="center", va="center")
        return fig

    metric_names = [
        "Task A\nRetention",
        "Task B\nAccuracy",
        "Average\nAccuracy",
        "BWT\n(normed)",
        "Param\nEfficiency",
    ]
    N = len(metric_names)
    angles = np.linspace(0, 2 * np.pi, N, endpoint=False).tolist()
    angles += angles[:1]  # close the polygon

    # Normalise BWT to [0, 1]
    all_bwt = [r.backward_transfer for r in results]
    bwt_min, bwt_max = min(all_bwt), max(all_bwt)

    # Normalise trainable params (fewer = better efficiency)
    all_params = [r.num_trainable_params_task_b for r in results]
    max_params = max(all_params) if max(all_params) > 0 else 1

    fig, ax = plt.subplots(figsize=(9, 9), subplot_kw={"polar": True})
    fig.patch.set_facecolor(DARK_BG)
    ax.set_facecolor("#0d1117")
    ax.spines["polar"].set_color(EDGE_COLOR)

    for i, r in enumerate(results):
        bwt_norm = (r.backward_transfer - bwt_min) / max(bwt_max - bwt_min, 1e-5)
        param_eff = 1.0 - (r.num_trainable_params_task_b / max_params)

        values = [
            1.0 - r.forgetting,  # Task A retention (higher = better)
            r.task_b_accuracy,
            r.average_accuracy,
            bwt_norm,
            max(0.0, min(1.0, param_eff)),
        ]
        values += values[:1]

        color = _get_color(r.method_name, i)
        ax.plot(angles, values, linewidth=2, linestyle="solid", color=color, label=r.method_name)
        ax.fill(angles, values, alpha=0.1, color=color)

    ax.set_xticks(angles[:-1])
    ax.set_xticklabels(metric_names, size=10, color=TEXT_COLOR)
    ax.set_ylim(0, 1)
    ax.set_yticks([0.2, 0.4, 0.6, 0.8, 1.0])
    ax.set_yticklabels(["0.2", "0.4", "0.6", "0.8", "1.0"], size=7, color="#8b949e")
    ax.tick_params(colors="#8b949e")
    ax.grid(color=EDGE_COLOR, linestyle="--", alpha=0.5)

    legend = ax.legend(
        loc="upper right",
        bbox_to_anchor=(1.3, 1.1),
        framealpha=0.2,
        labelcolor=TEXT_COLOR,
        fontsize=10,
    )

    ax.set_title(title, fontsize=13, fontweight="bold", color=ACCENT_BLUE, pad=25)
    plt.tight_layout()

    if save_path:
        os.makedirs(os.path.dirname(save_path) or ".", exist_ok=True)
        fig.savefig(save_path, dpi=150, bbox_inches="tight", facecolor=DARK_BG)
        logger.info(f"Radar chart saved: {save_path}")

    if show:
        plt.show()

    return fig
