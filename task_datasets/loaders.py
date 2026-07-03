"""
datasets/loaders.py

HuggingFace dataset loading and subsetting for TAPSS experiments.
All datasets are loaded with reproducible random subsets.
"""
from __future__ import annotations

import logging
from typing import Optional

from omegaconf import DictConfig

import datasets as _hf_datasets  # HuggingFace datasets library
from .base import TaskDataset, TaskDatasetConfig
from .preprocessing import tokenize_dataset

logger = logging.getLogger(__name__)


# Registry of known dataset configurations.
# These mirror the YAML configs but allow programmatic access.
DATASET_REGISTRY: dict[str, dict] = {
    "ag_news": {
        "hf_name": "fancyzhx/ag_news",
        "hf_config": None,
        "text_column": "text",
        "label_column": "label",
        "num_labels": 4,
        "label_names": ["World", "Sports", "Business", "Sci/Tech"],
    },
    "sst2": {
        "hf_name": "nyu-mll/glue",
        "hf_config": "sst2",
        "text_column": "sentence",
        "label_column": "label",
        "num_labels": 2,
        "label_names": ["Negative", "Positive"],
    },
    "imdb": {
        "hf_name": "stanfordnlp/imdb",
        "hf_config": None,
        "text_column": "text",
        "label_column": "label",
        "num_labels": 2,
        "label_names": ["Negative", "Positive"],
    },
    "dbpedia": {
        "hf_name": "lhoestq/dbpedia_14",
        "hf_config": None,
        "text_column": "content",
        "label_column": "label",
        "num_labels": 14,
        "label_names": [
            "Company", "Educational Institution", "Artist", "Athlete",
            "Office Holder", "Mean of Transportation", "Building",
            "Natural Place", "Village", "Animal", "Plant", "Album",
            "Film", "Written Work",
        ],
    },
}


def _build_config_from_registry(
    name: str,
    train_size: int = 5000,
    val_size: int = 1000,
    test_size: int = 1000,
    cache_dir: str = ".cache/datasets",
) -> TaskDatasetConfig:
    """Build a TaskDatasetConfig from the registry."""
    if name not in DATASET_REGISTRY:
        raise ValueError(
            f"Unknown dataset: {name!r}. Available: {list(DATASET_REGISTRY.keys())}"
        )
    meta = DATASET_REGISTRY[name]
    return TaskDatasetConfig(
        name=name,
        hf_name=meta["hf_name"],
        hf_config=meta["hf_config"],
        text_column=meta["text_column"],
        label_column=meta["label_column"],
        num_labels=meta["num_labels"],
        label_names=meta["label_names"],
        train_size=train_size,
        val_size=val_size,
        test_size=test_size,
        cache_dir=cache_dir,
    )


def _config_from_omegaconf(cfg: DictConfig) -> TaskDatasetConfig:
    """Build a TaskDatasetConfig from a Hydra DictConfig."""
    return TaskDatasetConfig(
        name=cfg.name,
        hf_name=cfg.hf_name,
        hf_config=cfg.get("hf_config", None),
        text_column=cfg.text_column,
        label_column=cfg.label_column,
        num_labels=cfg.num_labels,
        label_names=list(cfg.label_names),
        train_size=cfg.get("train_size", 5000),
        val_size=cfg.get("val_size", 1000),
        test_size=cfg.get("test_size", 1000),
        cache_dir=cfg.get("cache_dir", ".cache/datasets"),
    )


def load_task_dataset(
    name_or_cfg,
    tokenizer,
    batch_size: int = 16,
    eval_batch_size: int = 32,
    seed: int = 42,
    num_workers: int = 0,
    train_size: Optional[int] = None,
    val_size: Optional[int] = None,
    test_size: Optional[int] = None,
    cache_dir: str = ".cache/datasets",
) -> TaskDataset:
    """
    Load and prepare a task dataset for a TAPSS experiment.

    Parameters
    ----------
    name_or_cfg : str | DictConfig
        Dataset name (registry key) or a Hydra DictConfig for the dataset.
    tokenizer :
        HuggingFace tokenizer instance.
    batch_size : int
        Training DataLoader batch size.
    eval_batch_size : int
        Evaluation DataLoader batch size.
    seed : int
        Random seed for reproducible subset selection.
    num_workers : int
        DataLoader workers (0 for main process only).
    train_size / val_size / test_size : int | None
        Override config subset sizes.
    cache_dir : str
        Cache directory for HuggingFace datasets.

    Returns
    -------
    TaskDataset
    """
    # Build config
    if isinstance(name_or_cfg, str):
        cfg = _build_config_from_registry(
            name_or_cfg,
            train_size=train_size or 5000,
            val_size=val_size or 1000,
            test_size=test_size or 1000,
            cache_dir=cache_dir,
        )
    else:
        cfg = _config_from_omegaconf(name_or_cfg)
        if train_size is not None:
            cfg.train_size = train_size
        if val_size is not None:
            cfg.val_size = val_size
        if test_size is not None:
            cfg.test_size = test_size

    logger.info(f"Loading dataset: {cfg.name} from HuggingFace ({cfg.hf_name})")

    # Load raw HuggingFace dataset
    raw = _hf_datasets.load_dataset(
        cfg.hf_name,
        cfg.hf_config,
        cache_dir=cfg.cache_dir,
    )

    # Handle datasets with no validation split or unlabelled test splits (e.g. GLUE sst2)
    train_raw = raw["train"]
    if cfg.name == "sst2" or (hasattr(cfg, "hf_config") and cfg.hf_config == "sst2") or (hasattr(cfg, "hf_name") and "glue" in cfg.hf_name):
        # glue validation split has labels, test does not. Split validation.
        split = raw["validation"].train_test_split(test_size=0.5, seed=seed)
        val_raw = split["train"]
        test_raw = split["test"]
    elif "validation" in raw:
        val_raw = raw["validation"]
        test_raw = raw.get("test", val_raw)
    elif "test" in raw:
        # Split test into val + test if no validation split
        split = raw["test"].train_test_split(test_size=0.5, seed=seed)
        val_raw = split["train"]
        test_raw = split["test"]
    else:
        split = train_raw.train_test_split(test_size=0.1, seed=seed)
        train_raw = split["train"]
        val_raw = split["test"]
        test_raw = val_raw

    # Subsample for fast experiments
    def subsample(ds, n: int, seed: int):
        n = min(n, len(ds))
        return ds.shuffle(seed=seed).select(range(n))

    train_raw = subsample(train_raw, cfg.train_size, seed)
    val_raw = subsample(val_raw, cfg.val_size, seed + 1)
    test_raw = subsample(test_raw, cfg.test_size, seed + 2)

    logger.info(
        f"Subsets — train: {len(train_raw)}, val: {len(val_raw)}, test: {len(test_raw)}"
    )

    # Tokenize
    train_tok, val_tok, test_tok = tokenize_dataset(
        train_raw=train_raw,
        val_raw=val_raw,
        test_raw=test_raw,
        tokenizer=tokenizer,
        text_column=cfg.text_column,
        label_column=cfg.label_column,
        max_length=tokenizer.model_max_length if tokenizer.model_max_length <= 512 else 128,
    )

    # Build DataLoaders
    from torch.utils.data import DataLoader

    train_loader = DataLoader(
        train_tok,
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,
        pin_memory=True,
    )
    val_loader = DataLoader(
        val_tok,
        batch_size=eval_batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=True,
    )
    test_loader = DataLoader(
        test_tok,
        batch_size=eval_batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=True,
    )

    return TaskDataset(
        config=cfg,
        train_loader=train_loader,
        val_loader=val_loader,
        test_loader=test_loader,
        train_dataset=train_tok,
        val_dataset=val_tok,
        test_dataset=test_tok,
    )
