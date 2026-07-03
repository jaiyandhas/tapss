"""
saliency/layer_contribution.py

Method 4 — Layer-wise Contribution Estimator.

Estimates parameter importance aggregated at the transformer layer level,
using the product of gradient magnitude and activation magnitude (a.k.a.
gradient × activation, or "Taylor criterion").

Rationale
---------
The Taylor decomposition of loss change suggests that the first-order
approximation of removing a feature h is:

  ΔL ≈ -h · (∂L/∂h)

i.e. the product of activation and its gradient. Large product = removing
this feature would change the loss a lot = the feature is important.

This method aggregates this criterion per transformer layer, then broadcasts
the layer-level importance back to individual parameters within each layer.

Reference
---------
Molchanov et al., "Pruning Convolutional Neural Networks for Resource
Efficient Inference", ICLR 2017.
"""
from __future__ import annotations

import logging
import time
from collections import defaultdict
from typing import Any

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader

from saliency.base import ImportanceEstimator, ImportanceScores

logger = logging.getLogger(__name__)


class LayerContributionEstimator(ImportanceEstimator):
    """
    Estimates importance using the Taylor criterion (grad × activation)
    aggregated per transformer layer.

    Layer-level scores are then broadcast to individual parameters
    within each layer.
    """

    def __init__(
        self,
        cfg: Any = None,
        num_calibration_batches: int = 50,
        aggregation: str = "mean",
    ):
        super().__init__(cfg)
        if cfg is not None and hasattr(cfg, "saliency"):
            self.num_calibration_batches = cfg.saliency.get(
                "num_calibration_batches", num_calibration_batches
            )
            self.aggregation = cfg.saliency.get("aggregation", aggregation)
        else:
            self.num_calibration_batches = num_calibration_batches
            self.aggregation = aggregation

    @property
    def name(self) -> str:
        return "layer_contribution"

    def _extract_layer_index(self, param_name: str) -> str:
        """Extract the layer key from a parameter name."""
        parts = param_name.split(".")
        for i, part in enumerate(parts):
            if part == "layer" and i + 1 < len(parts) and parts[i + 1].isdigit():
                return f"layer_{parts[i + 1]}"
        return "other"

    def compute(
        self,
        model: nn.Module,
        dataloader: DataLoader,
        device: torch.device,
    ) -> ImportanceScores:
        """
        Compute layer-wise Taylor-criterion importance scores.

        Returns
        -------
        ImportanceScores
            Per-parameter scores based on their layer's aggregate importance.
        """
        logger.info(
            f"[LayerContribution] Computing Taylor criterion over "
            f"{self.num_calibration_batches} batches (agg={self.aggregation})..."
        )
        t0 = time.time()

        model.eval()

        # Accumulate grad × activation per named parameter
        param_taylor: dict[str, list[float]] = defaultdict(list)

        # Register hooks to capture activations
        activation_store: dict[str, torch.Tensor] = {}

        hooks = []
        for name, module in model.named_modules():
            if isinstance(module, nn.Linear):
                def _make_hook(mod_name):
                    def hook(module, input, output):
                        if isinstance(output, torch.Tensor):
                            activation_store[mod_name] = output.detach()
                    return hook
                hooks.append(module.register_forward_hook(_make_hook(name)))

        num_batches_processed = 0

        for batch in self._iter_batches(dataloader, self.num_calibration_batches, device):
            model.zero_grad()
            activation_store.clear()

            outputs = model(
                input_ids=batch.get("input_ids"),
                attention_mask=batch.get("attention_mask"),
                labels=batch.get("labels"),
            )

            if hasattr(outputs, "loss") and outputs.loss is not None:
                loss = outputs.loss
            else:
                loss = nn.CrossEntropyLoss()(outputs.logits, batch["labels"])

            loss.backward()

            # For each parameter, compute |grad| × mean(|activation|) of its module
            for param_name, param in model.named_parameters():
                if not param.requires_grad or param.grad is None:
                    continue

                # Get the parent module's activation
                parent_module = ".".join(param_name.split(".")[:-1])
                act_magnitude = 0.0
                if parent_module in activation_store:
                    act_magnitude = float(activation_store[parent_module].abs().mean().item())

                grad_magnitude = float(param.grad.data.abs().mean().item())
                taylor_score = grad_magnitude * (1.0 + act_magnitude)  # +1 avoids zero
                param_taylor[param_name].append(taylor_score)

            num_batches_processed += 1

        # Cleanup hooks
        for h in hooks:
            h.remove()
        model.zero_grad()

        if num_batches_processed == 0:
            logger.warning("[LayerContribution] No batches processed!")
            return ImportanceScores(
                scores={n: 0.0 for n, p in model.named_parameters() if p.requires_grad},
                method=self.name,
            )

        # Aggregate per-parameter scores
        agg_fn = {"mean": np.mean, "max": np.max, "sum": np.sum}.get(
            self.aggregation, np.mean
        )
        param_scores: dict[str, float] = {
            name: float(agg_fn(scores))
            for name, scores in param_taylor.items()
        }

        # Aggregate to layer level
        layer_scores: dict[str, list[float]] = defaultdict(list)
        for param_name, score in param_scores.items():
            layer_key = self._extract_layer_index(param_name)
            layer_scores[layer_key].append(score)

        layer_aggregated: dict[str, float] = {
            k: float(np.mean(v)) for k, v in layer_scores.items()
        }

        # Broadcast layer-level score to each parameter in that layer
        raw_scores: dict[str, float] = {}
        for param_name in param_scores:
            layer_key = self._extract_layer_index(param_name)
            # Use layer average × individual param score (weighted combination)
            raw_scores[param_name] = (
                0.7 * param_scores[param_name]
                + 0.3 * layer_aggregated.get(layer_key, 0.0)
            )

        normalised = ImportanceScores.normalise(raw_scores)

        elapsed = time.time() - t0
        logger.info(
            f"[LayerContribution] Done in {elapsed:.1f}s. "
            f"Layer groups: {len(layer_aggregated)}. "
            f"Params scored: {len(normalised)}."
        )

        return ImportanceScores(
            scores=normalised,
            method=self.name,
            metadata={
                "raw_scores": raw_scores,
                "layer_aggregated": layer_aggregated,
                "aggregation": self.aggregation,
                "num_batches": num_batches_processed,
                "elapsed_seconds": elapsed,
            },
        )
