"""
saliency/base.py

Base types for importance estimation: ImportanceScores dataclass and
ImportanceEstimator abstract class.
"""
from __future__ import annotations

import logging
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import Any

import numpy as np
import torch
from torch.utils.data import DataLoader

logger = logging.getLogger(__name__)


@dataclass
class ImportanceScores:
    """
    Holds per-parameter importance scores from a single estimator run.

    Scores are a dict of {param_name: float}, normalised to [0, 1].
    metadata stores anything extra (raw scores, timing, component breakdown).
    """

    scores: dict[str, float]
    method: str
    metadata: dict[str, Any] = field(default_factory=dict)

    def __len__(self) -> int:
        return len(self.scores)

    def top_k(self, k: int) -> list[tuple[str, float]]:
        """Return the k most important parameters, sorted descending."""
        sorted_items = sorted(self.scores.items(), key=lambda x: x[1], reverse=True)
        return sorted_items[:k]

    def top_k_percent(self, pct: float) -> list[tuple[str, float]]:
        """Return the top pct% most important parameters."""
        k = max(1, int(len(self.scores) * pct / 100.0))
        return self.top_k(k)

    def as_numpy(self, param_names: list[str] | None = None) -> np.ndarray:
        """Return scores as a numpy array, optionally ordered by param_names."""
        if param_names is None:
            param_names = list(self.scores.keys())
        return np.array([self.scores.get(n, 0.0) for n in param_names])

    def layer_aggregated(self) -> dict[str, float]:
        """
        Aggregate scores per transformer layer.

        Layer is inferred from parameter names containing 'layer.N'.
        Returns dict mapping layer_key → mean importance score.
        """
        layer_scores: dict[str, list[float]] = {}
        for name, score in self.scores.items():
            parts = name.split(".")
            layer_key = "other"
            for i, part in enumerate(parts):
                if part == "layer" and i + 1 < len(parts) and parts[i + 1].isdigit():
                    layer_key = f"layer_{parts[i + 1]}"
                    break
            layer_scores.setdefault(layer_key, []).append(score)
        return {k: float(np.mean(v)) for k, v in layer_scores.items()}

    @staticmethod
    def normalise(raw: dict[str, float]) -> dict[str, float]:
        """Min-max normalise scores to [0, 1]."""
        if not raw:
            return raw
        values = np.array(list(raw.values()), dtype=float)
        vmin, vmax = values.min(), values.max()
        if vmax - vmin < 1e-10:
            # All scores identical — assign uniform importance
            return {k: 0.5 for k in raw}
        return {k: float((v - vmin) / (vmax - vmin)) for k, v in raw.items()}

    @staticmethod
    def combine(
        score_dicts: list[dict[str, float]],
        weights: list[float],
    ) -> dict[str, float]:
        """
        Weighted combination of multiple normalised score dicts.

        All dicts must share the same keys.
        Weights need not sum to 1 (they are renormalised internally).
        """
        if not score_dicts:
            return {}
        total_weight = sum(weights)
        keys = list(score_dicts[0].keys())
        combined = {}
        for k in keys:
            combined[k] = sum(
                w * sd.get(k, 0.0) for sd, w in zip(score_dicts, weights)
            ) / total_weight
        return combined


class ImportanceEstimator(ABC):
    """
    Abstract base for importance estimators. Subclasses implement compute().
    """

    def __init__(self, cfg: Any = None):
        self.cfg = cfg
        self.logger = logging.getLogger(self.__class__.__name__)

    @property
    @abstractmethod
    def name(self) -> str:
        """Human-readable name for this estimator."""
        ...

    @abstractmethod
    def compute(
        self,
        model: torch.nn.Module,
        dataloader: DataLoader,
        device: torch.device,
    ) -> ImportanceScores:
        """
        Compute parameter importance scores.

        Parameters
        ----------
        model : nn.Module
            Model to analyse. Must have parameters with requires_grad=True.
        dataloader : DataLoader
            Calibration data (small subset of Task A training data).
        device : torch.device
            Device to run computation on.

        Returns
        -------
        ImportanceScores
            Normalised importance scores for all named parameters.
        """
        ...

    def _iter_batches(
        self,
        dataloader: DataLoader,
        max_batches: int | None,
        device: torch.device,
    ):
        """Helper generator that yields batches moved to device, up to max_batches."""
        for i, batch in enumerate(dataloader):
            if max_batches is not None and i >= max_batches:
                break
            yield {k: v.to(device) for k, v in batch.items() if isinstance(v, torch.Tensor)}

    def _log_progress(self, step: int, total: int | None = None) -> None:
        """Log progress during calibration."""
        if total is not None:
            self.logger.debug(f"Calibration batch {step}/{total}")
        else:
            self.logger.debug(f"Calibration batch {step}")
