"""
saliency/rankings.py

Parameter ranking and serialisation.

Takes ImportanceScores and produces a structured ParameterRanking table
with layer, parameter name, score, rank, and protection status.
Exports to CSV, JSON, and interactive Plotly HTML.
"""
from __future__ import annotations

import json
import logging
import os
from dataclasses import dataclass, asdict, field
from typing import Optional

import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go

from saliency.base import ImportanceScores

logger = logging.getLogger(__name__)


@dataclass
class ParameterRanking:
    """
    Ranked entry for a single parameter tensor.

    Attributes
    ----------
    param_name : str
        Fully-qualified parameter name (e.g. "distilbert.transformer.layer.0.attention.q_lin.weight").
    layer : str
        Inferred transformer layer key (e.g. "layer_0", "other").
    importance_score : float
        Normalised importance score in [0, 1].
    rank : int
        Rank by importance (1 = most important).
    num_elements : int
        Total number of scalar elements in this parameter tensor.
    is_protected : bool
        Whether this parameter is flagged for protection.
    method : str
        Saliency method that produced this score.
    """

    param_name: str
    layer: str
    importance_score: float
    rank: int
    num_elements: int
    is_protected: bool
    method: str

    def param_type(self) -> str:
        """Infer parameter type (weight, bias, other) from name."""
        if self.param_name.endswith(".weight"):
            return "weight"
        elif self.param_name.endswith(".bias"):
            return "bias"
        return "other"


@dataclass
class RankingResults:
    """
    Complete ranking output for all parameters.

    Attributes
    ----------
    rankings : list[ParameterRanking]
        All ranked parameters, sorted by rank ascending.
    method : str
        Saliency method used.
    protected_names : set[str]
        Parameter names flagged as protected.
    metadata : dict
        Experiment metadata (model name, dataset, etc.).
    """

    rankings: list[ParameterRanking]
    method: str
    protected_names: set[str] = field(default_factory=set)
    metadata: dict = field(default_factory=dict)

    def as_dataframe(self) -> pd.DataFrame:
        """Convert rankings to a pandas DataFrame."""
        rows = [asdict(r) for r in self.rankings]
        df = pd.DataFrame(rows)
        if not df.empty:
            df["param_type"] = df["param_name"].apply(
                lambda n: "weight" if n.endswith(".weight") else
                          "bias" if n.endswith(".bias") else "other"
            )
        return df

    def top_k(self, k: int) -> list[ParameterRanking]:
        """Return the k most important parameters."""
        return sorted(self.rankings, key=lambda r: r.rank)[:k]

    def layer_summary(self) -> pd.DataFrame:
        """Aggregate importance by layer."""
        df = self.as_dataframe()
        if df.empty:
            return df
        return (
            df.groupby("layer")
            .agg(
                mean_score=("importance_score", "mean"),
                max_score=("importance_score", "max"),
                num_params=("param_name", "count"),
                num_protected=("is_protected", "sum"),
            )
            .reset_index()
            .sort_values("mean_score", ascending=False)
        )

    def save_csv(self, path: str) -> None:
        """Save full ranking table to CSV."""
        os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
        df = self.as_dataframe()
        df.to_csv(path, index=False)
        logger.info(f"Rankings saved to CSV: {path}")

    def save_json(self, path: str) -> None:
        """Save ranking metadata and top-100 entries to JSON."""
        os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
        payload = {
            "method": self.method,
            "metadata": self.metadata,
            "total_params": len(self.rankings),
            "total_protected": len(self.protected_names),
            "top_100": [asdict(r) for r in self.top_k(100)],
            "layer_summary": self.layer_summary().to_dict(orient="records"),
        }
        with open(path, "w") as f:
            json.dump(payload, f, indent=2)
        logger.info(f"Rankings saved to JSON: {path}")

    def save_html(self, path: str) -> None:
        """Generate and save an interactive Plotly ranking visualisation."""
        os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
        df = self.as_dataframe()
        if df.empty:
            return

        # ── Scatter: Rank vs Score coloured by layer ──
        fig1 = px.scatter(
            df,
            x="rank",
            y="importance_score",
            color="layer",
            symbol="is_protected",
            hover_data=["param_name", "num_elements", "param_type"],
            title=f"Parameter Importance Ranking — Method: {self.method}",
            labels={"rank": "Rank (1 = most important)", "importance_score": "TAPSS Score"},
            template="plotly_dark",
        )

        # ── Bar: Layer-level mean importance ──
        layer_df = self.layer_summary()
        fig2 = px.bar(
            layer_df,
            x="layer",
            y="mean_score",
            color="num_protected",
            title="Layer-Level Mean Importance",
            labels={"layer": "Layer", "mean_score": "Mean TAPSS Score"},
            template="plotly_dark",
        )

        # ── Histogram: Score distribution ──
        fig3 = px.histogram(
            df,
            x="importance_score",
            color="is_protected",
            nbins=50,
            title="Importance Score Distribution",
            labels={"importance_score": "TAPSS Score"},
            template="plotly_dark",
        )

        # Write all three charts to a single HTML file
        from plotly.subplots import make_subplots
        import plotly.io as pio

        html_content = (
            "<html><head><title>TAPSS Parameter Rankings</title></head><body>"
            "<h1 style='font-family:monospace;color:#ccc;'>TAPSS Parameter Rankings</h1>"
            + fig1.to_html(full_html=False)
            + fig2.to_html(full_html=False)
            + fig3.to_html(full_html=False)
            + "</body></html>"
        )
        with open(path, "w") as f:
            f.write(html_content)

        logger.info(f"Interactive ranking plots saved to HTML: {path}")


def _infer_layer(param_name: str) -> str:
    """Extract transformer layer key from a parameter name."""
    parts = param_name.split(".")
    for i, part in enumerate(parts):
        if part == "layer" and i + 1 < len(parts) and parts[i + 1].isdigit():
            return f"layer_{parts[i + 1]}"
    return "other"


def build_rankings(
    scores: ImportanceScores,
    model: "torch.nn.Module",
    protection_percent: float = 20.0,
    extra_metadata: Optional[dict] = None,
) -> RankingResults:
    """
    Build a RankingResults from ImportanceScores.

    Parameters
    ----------
    scores : ImportanceScores
        Normalised importance scores from any estimator.
    model : nn.Module
        The model (used to get parameter element counts).
    protection_percent : float
        Top-N% of parameters to flag as protected.
    extra_metadata : dict | None
        Additional metadata to embed in the results.

    Returns
    -------
    RankingResults
    """
    import torch.nn as nn

    # Build param name → num_elements map
    param_sizes: dict[str, int] = {
        name: param.numel()
        for name, param in model.named_parameters()
        if param.requires_grad
    }

    # Sort by score descending to assign ranks
    sorted_items = sorted(scores.scores.items(), key=lambda x: x[1], reverse=True)
    num_to_protect = max(1, int(len(sorted_items) * protection_percent / 100.0))
    protected_names = {name for name, _ in sorted_items[:num_to_protect]}

    rankings = []
    for rank, (name, score) in enumerate(sorted_items, start=1):
        rankings.append(
            ParameterRanking(
                param_name=name,
                layer=_infer_layer(name),
                importance_score=round(score, 6),
                rank=rank,
                num_elements=param_sizes.get(name, 0),
                is_protected=(name in protected_names),
                method=scores.method,
            )
        )

    return RankingResults(
        rankings=rankings,
        method=scores.method,
        protected_names=protected_names,
        metadata=extra_metadata or {},
    )
