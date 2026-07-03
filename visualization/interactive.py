"""
visualization/interactive.py

Interactive Plotly visualisations for TAPSS experiments.

All functions return Plotly figures that can be:
  - Displayed in Jupyter notebooks (fig.show())
  - Exported as standalone HTML (fig.write_html(...))
  - Exported as static PNG/SVG (fig.write_image(...) if kaleido installed)
"""
from __future__ import annotations

import logging
import os
from typing import Any, Optional

import numpy as np
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots

from saliency.base import ImportanceScores
from saliency.rankings import RankingResults

logger = logging.getLogger(__name__)

# Dark theme for all Plotly figures
PLOTLY_TEMPLATE = "plotly_dark"
ACCENT_BLUE = "#58a6ff"


def interactive_importance_scatter(
    rankings: RankingResults,
    title: str = "Parameter Importance — Interactive Explorer",
    save_path: Optional[str] = None,
) -> go.Figure:
    """
    Interactive scatter: Rank vs Score, coloured by layer, sized by num_elements.

    Hovering shows full parameter name, layer, score, and protection status.

    Parameters
    ----------
    rankings : RankingResults
    title : str
    save_path : str | None

    Returns
    -------
    plotly.graph_objects.Figure
    """
    df = rankings.as_dataframe()
    if df.empty:
        return go.Figure()

    fig = px.scatter(
        df,
        x="rank",
        y="importance_score",
        color="layer",
        size=np.clip(df["num_elements"].values, 100, 100_000),
        size_max=20,
        symbol="is_protected",
        symbol_map={True: "diamond", False: "circle"},
        hover_data=["param_name", "param_type", "num_elements", "is_protected"],
        title=title,
        labels={
            "rank": "Rank (1 = most important)",
            "importance_score": "TAPSS Score",
            "layer": "Layer",
        },
        template=PLOTLY_TEMPLATE,
    )

    fig.update_layout(
        title_font_size=16,
        title_font_color=ACCENT_BLUE,
        hovermode="closest",
        legend_title_text="Layer",
        height=600,
    )

    if save_path:
        os.makedirs(os.path.dirname(save_path) or ".", exist_ok=True)
        fig.write_html(save_path)
        logger.info(f"Interactive scatter saved: {save_path}")

    return fig


def interactive_layer_heatmap(
    all_scores: dict[str, ImportanceScores],
    title: str = "Layer Importance Heatmap (All Methods)",
    save_path: Optional[str] = None,
) -> go.Figure:
    """
    Interactive heatmap: methods × transformer layers.

    Each cell shows the mean importance score for that method × layer combination.

    Parameters
    ----------
    all_scores : dict[str, ImportanceScores]
        {method_name: scores} mapping.
    title, save_path : standard args.

    Returns
    -------
    plotly.graph_objects.Figure
    """
    all_layers: set[str] = set()
    method_layer_scores: dict[str, dict[str, float]] = {}

    for method, scores in all_scores.items():
        layer_agg = scores.layer_aggregated()
        method_layer_scores[method] = layer_agg
        all_layers.update(layer_agg.keys())

    # Sort layers
    layers = sorted(
        [l for l in all_layers if l.startswith("layer_")],
        key=lambda l: int(l.split("_")[1]),
    ) + [l for l in all_layers if not l.startswith("layer_")]

    methods = list(all_scores.keys())
    matrix = []
    for method in methods:
        row = [method_layer_scores[method].get(l, 0.0) for l in layers]
        matrix.append(row)

    fig = go.Figure(
        data=go.Heatmap(
            z=matrix,
            x=layers,
            y=methods,
            colorscale="Plasma",
            zmin=0,
            zmax=1,
            text=[[f"{v:.3f}" for v in row] for row in matrix],
            texttemplate="%{text}",
            hoverongaps=False,
            colorbar=dict(title="Mean Score"),
        )
    )

    fig.update_layout(
        title=dict(text=title, font=dict(size=16, color=ACCENT_BLUE)),
        template=PLOTLY_TEMPLATE,
        height=max(300, len(methods) * 60 + 150),
        xaxis_title="Transformer Layer",
        yaxis_title="Importance Method",
    )

    if save_path:
        os.makedirs(os.path.dirname(save_path) or ".", exist_ok=True)
        fig.write_html(save_path)
        logger.info(f"Interactive heatmap saved: {save_path}")

    return fig


def interactive_comparison_bar(
    results: list[Any],  # list[CLResult]
    title: str = "Continual Learning Results — Method Comparison",
    save_path: Optional[str] = None,
) -> go.Figure:
    """
    Grouped interactive bar chart comparing all CL metrics across methods.

    Parameters
    ----------
    results : list[CLResult]
    title, save_path : standard args.

    Returns
    -------
    plotly.graph_objects.Figure
    """
    methods = [r.method_name for r in results]
    metrics = {
        "Task A (Post)": [r.task_a_post_accuracy for r in results],
        "Task B": [r.task_b_accuracy for r in results],
        "Avg Accuracy": [r.average_accuracy for r in results],
        "Forgetting (×−1)": [-r.forgetting for r in results],  # negate so higher = better
    }

    fig = go.Figure()

    colors = ["#3b82f6", "#10b981", "#8b5cf6", "#ef4444"]
    for color, (metric_name, values) in zip(colors, metrics.items()):
        fig.add_trace(
            go.Bar(
                name=metric_name,
                x=methods,
                y=values,
                marker_color=color,
                opacity=0.85,
                text=[f"{v:.3f}" for v in values],
                textposition="outside",
            )
        )

    fig.update_layout(
        title=dict(text=title, font=dict(size=16, color=ACCENT_BLUE)),
        barmode="group",
        template=PLOTLY_TEMPLATE,
        yaxis_title="Score",
        xaxis_title="Method",
        legend_title_text="Metric",
        height=550,
        yaxis=dict(range=[-0.1, 1.15]),
    )

    if save_path:
        os.makedirs(os.path.dirname(save_path) or ".", exist_ok=True)
        fig.write_html(save_path)
        logger.info(f"Interactive comparison bar saved: {save_path}")

    return fig


def interactive_radar(
    results: list[Any],  # list[CLResult]
    title: str = "Multi-metric Radar — All Methods",
    save_path: Optional[str] = None,
) -> go.Figure:
    """
    Interactive Plotly radar chart for multi-metric method comparison.

    Parameters
    ----------
    results : list[CLResult]
    title, save_path : standard args.

    Returns
    -------
    plotly.graph_objects.Figure
    """
    metrics = ["Task A Retention", "Task B Accuracy", "Avg Accuracy", "BWT (normed)", "Param Efficiency"]

    all_bwt = [r.backward_transfer for r in results]
    bwt_min, bwt_max = min(all_bwt), max(all_bwt)
    all_params = [r.num_trainable_params_task_b for r in results]
    max_params = max(all_params) if max(all_params) > 0 else 1

    colors = ["#3b82f6", "#10b981", "#ef4444", "#f59e0b", "#8b5cf6", "#06b6d4"]

    fig = go.Figure()

    for i, r in enumerate(results):
        bwt_norm = (r.backward_transfer - bwt_min) / max(bwt_max - bwt_min, 1e-5)
        param_eff = max(0.0, 1.0 - r.num_trainable_params_task_b / max_params)

        values = [
            max(0.0, 1.0 - r.forgetting),
            r.task_b_accuracy,
            r.average_accuracy,
            min(1.0, max(0.0, bwt_norm)),
            param_eff,
        ]

        fig.add_trace(
            go.Scatterpolar(
                r=values + [values[0]],
                theta=metrics + [metrics[0]],
                fill="toself",
                fillcolor=colors[i % len(colors)],
                opacity=0.15,
                line=dict(color=colors[i % len(colors)], width=2),
                name=r.method_name,
            )
        )

    fig.update_layout(
        polar=dict(
            radialaxis=dict(visible=True, range=[0, 1], color="#8b949e"),
            angularaxis=dict(color="#c9d1d9"),
            bgcolor="#0d1117",
        ),
        title=dict(text=title, font=dict(size=16, color=ACCENT_BLUE)),
        template=PLOTLY_TEMPLATE,
        height=600,
        showlegend=True,
    )

    if save_path:
        os.makedirs(os.path.dirname(save_path) or ".", exist_ok=True)
        fig.write_html(save_path)
        logger.info(f"Interactive radar saved: {save_path}")

    return fig
