"""
models/registry.py

Model factory for TAPSS experiments.
Supports DistilBERT, BERT-base, and RoBERTa-base for sequence classification.
"""
from __future__ import annotations

import logging
from typing import Any

import torch
from transformers import (
    AutoModelForSequenceClassification,
    AutoTokenizer,
    DistilBertForSequenceClassification,
    BertForSequenceClassification,
    RobertaForSequenceClassification,
)

from models.wrappers import TAPSSModel

logger = logging.getLogger(__name__)

# Map from config name → HuggingFace model name
_MODEL_NAME_MAP: dict[str, str] = {
    "distilbert": "distilbert-base-uncased",
    "bert": "bert-base-uncased",
    "bert-base": "bert-base-uncased",
    "roberta": "roberta-base",
    "roberta-base": "roberta-base",
}

# Map from model name → default LoRA target modules
LORA_TARGET_MODULES: dict[str, list[str]] = {
    "distilbert-base-uncased": ["q_lin", "v_lin"],
    "bert-base-uncased": ["query", "value"],
    "roberta-base": ["query", "value"],
}


class ModelRegistry:
    """
    Registry for instantiating models used in TAPSS experiments.
    Centralises model loading and ensures consistent configurations.
    """

    @staticmethod
    def resolve_model_name(name: str) -> str:
        """Resolve a short alias to the full HuggingFace model name."""
        return _MODEL_NAME_MAP.get(name, name)

    @staticmethod
    def get_lora_target_modules(model_name: str) -> list[str]:
        """Return the default LoRA target modules for a model."""
        resolved = ModelRegistry.resolve_model_name(model_name)
        if resolved not in LORA_TARGET_MODULES:
            logger.warning(
                f"No default LoRA target modules for {resolved!r}. "
                "Using ['query', 'value'] as fallback."
            )
            return ["query", "value"]
        return LORA_TARGET_MODULES[resolved]

    @staticmethod
    def load(
        name: str,
        num_labels: int,
        cache_dir: str = ".cache/models",
    ) -> tuple[Any, Any]:
        """
        Load a HuggingFace model and tokenizer by name.

        Parameters
        ----------
        name : str
            Model name or short alias (e.g. "distilbert", "bert-base-uncased").
        num_labels : int
            Number of classification labels.
        cache_dir : str
            Cache directory for downloaded model weights.

        Returns
        -------
        (model, tokenizer) : tuple
        """
        resolved = ModelRegistry.resolve_model_name(name)
        logger.info(f"Loading model: {resolved} (num_labels={num_labels})")

        tokenizer = AutoTokenizer.from_pretrained(resolved, cache_dir=cache_dir)
        model = AutoModelForSequenceClassification.from_pretrained(
            resolved,
            num_labels=num_labels,
            cache_dir=cache_dir,
            ignore_mismatched_sizes=True,
        )

        logger.info(
            f"Model loaded: {resolved} "
            f"| Parameters: {sum(p.numel() for p in model.parameters()):,}"
        )
        return model, tokenizer


def get_model_and_tokenizer(
    cfg: Any,
    num_labels: int,
    device: torch.device | None = None,
) -> tuple[TAPSSModel, Any]:
    """
    High-level factory: load model from config, wrap in TAPSSModel, move to device.

    Parameters
    ----------
    cfg : DictConfig or object with .model.name and .model.cache_dir
    num_labels : int
        Number of output classes.
    device : torch.device | None
        Target device. Defaults to CUDA if available.

    Returns
    -------
    (tapss_model, tokenizer)
    """
    if device is None:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    if hasattr(cfg, "model") and cfg.model is not None:
        model_name = cfg.model.get("name", "distilbert-base-uncased")
        cache_dir = cfg.model.get("cache_dir", ".cache/models")
    else:
        model_name = getattr(cfg, "name", "distilbert-base-uncased")
        cache_dir = ".cache/models"

    raw_model, tokenizer = ModelRegistry.load(model_name, num_labels, cache_dir)
    wrapped = TAPSSModel(raw_model, model_name=model_name)
    wrapped.to(device)

    return wrapped, tokenizer
