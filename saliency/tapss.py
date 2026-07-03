"""
saliency/tapss.py

Combined TAPSS importance score — weighted sum of the four component estimators.

Each component is normalised to [0,1] before combining so the weights are
actually comparable. Weights are read from config or DEFAULT_WEIGHTS and
renormalised to sum to 1.

The idea: no single signal is reliable on its own. Gradient magnitude is noisy;
perturbation is slow; activation is coarse. Combining them should be more
stable than any individual method — that's the hypothesis the ablations will test.
"""
from __future__ import annotations

import logging
import time
from typing import Any

import torch
import torch.nn as nn
from torch.utils.data import DataLoader

from saliency.base import ImportanceEstimator, ImportanceScores
from saliency.gradient import GradientMagnitudeEstimator
from saliency.activation import ActivationFrequencyEstimator
from saliency.perturbation import PerturbationSensitivityEstimator
from saliency.layer_contribution import LayerContributionEstimator

logger = logging.getLogger(__name__)

# Default TAPSS weights (must be positive; renormalised internally)
DEFAULT_WEIGHTS = {
    "gradient": 0.40,
    "perturbation": 0.30,
    "activation": 0.20,
    "layer_contribution": 0.10,
}


class TAPSSEstimator(ImportanceEstimator):
    """
    Runs all component estimators and combines them into a single TAPSS score.

    Args:
        cfg: experiment config (reads weights from cfg.saliency.weights if present)
        weights: override weights dict; if None falls back to cfg then DEFAULT_WEIGHTS
        skip_perturbation: skip the slow O(params*batches) perturbation step
    """

    def __init__(
        self,
        cfg: Any = None,
        weights: dict[str, float] | None = None,
        skip_perturbation: bool = False,
    ):
        super().__init__(cfg)
        self.skip_perturbation = skip_perturbation

        # Resolve weights
        if weights is not None:
            self.weights = weights
        elif cfg is not None and hasattr(cfg, "saliency") and cfg.saliency is not None and cfg.saliency.get("weights", None) is not None:
            self.weights = dict(cfg.saliency.weights)
        else:
            self.weights = dict(DEFAULT_WEIGHTS)

        # Renormalise weights to sum to 1
        total = sum(self.weights.values())
        self.weights = {k: v / total for k, v in self.weights.items()}

        logger.info(f"[TAPSS] Weights: {self.weights}")

        # Instantiate sub-estimators
        self._gradient_est = GradientMagnitudeEstimator(cfg)
        self._activation_est = ActivationFrequencyEstimator(cfg)
        self._perturbation_est = PerturbationSensitivityEstimator(cfg)
        self._layer_est = LayerContributionEstimator(cfg)

    @property
    def name(self) -> str:
        return "tapss"

    def compute(
        self,
        model: nn.Module,
        dataloader: DataLoader,
        device: torch.device,
    ) -> ImportanceScores:
        """Run each enabled component estimator and return the weighted combination."""
        logger.info("[TAPSS] Starting combined importance estimation...")
        t0 = time.time()

        component_scores: dict[str, dict[str, float]] = {}

        # --- Gradient magnitude ---
        if self.weights.get("gradient", 0.0) > 0:
            logger.info("[TAPSS] Running: Gradient Magnitude")
            grad_result = self._gradient_est.compute(model, dataloader, device)
            component_scores["gradient"] = grad_result.scores

        # --- Activation frequency ---
        if self.weights.get("activation", 0.0) > 0:
            logger.info("[TAPSS] Running: Activation Frequency")
            act_result = self._activation_est.compute(model, dataloader, device)
            component_scores["activation"] = act_result.scores

        # --- Perturbation sensitivity ---
        if self.weights.get("perturbation", 0.0) > 0 and not self.skip_perturbation:
            logger.info("[TAPSS] Running: Perturbation Sensitivity")
            pert_result = self._perturbation_est.compute(model, dataloader, device)
            component_scores["perturbation"] = pert_result.scores
        elif self.skip_perturbation:
            logger.warning("[TAPSS] Perturbation estimator skipped (skip_perturbation=True).")
            # Redistribute its weight to gradient
            extra = self.weights.pop("perturbation", 0.0)
            self.weights["gradient"] = self.weights.get("gradient", 0.0) + extra

        # --- Layer contribution ---
        if self.weights.get("layer_contribution", 0.0) > 0:
            logger.info("[TAPSS] Running: Layer Contribution")
            layer_result = self._layer_est.compute(model, dataloader, device)
            component_scores["layer_contribution"] = layer_result.scores

        if not component_scores:
            logger.warning("[TAPSS] No component estimators ran!")
            return ImportanceScores(
                scores={n: 0.0 for n, p in model.named_parameters() if p.requires_grad},
                method=self.name,
            )

        # --- Weighted combination ---
        # Collect all parameter names from any component
        all_param_names = list(next(iter(component_scores.values())).keys())

        active_weights = {k: v for k, v in self.weights.items() if k in component_scores}
        total_weight = sum(active_weights.values())

        combined: dict[str, float] = {}
        for param_name in all_param_names:
            score = 0.0
            for method_name, w in active_weights.items():
                score += w * component_scores[method_name].get(param_name, 0.0)
            combined[param_name] = score / total_weight

        # Final normalisation of the combined score
        final_scores = ImportanceScores.normalise(combined)

        elapsed = time.time() - t0
        logger.info(
            f"[TAPSS] Done in {elapsed:.1f}s. "
            f"Components: {list(component_scores.keys())}. "
            f"Params scored: {len(final_scores)}."
        )

        return ImportanceScores(
            scores=final_scores,
            method=self.name,
            metadata={
                "component_scores": component_scores,
                "weights": self.weights,
                "elapsed_seconds": elapsed,
                "skip_perturbation": self.skip_perturbation,
            },
        )
