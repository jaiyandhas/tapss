"""
datasets/base.py

Base dataclasses and interfaces for TAPSS task datasets.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Optional

from torch.utils.data import DataLoader


@dataclass
class TaskDatasetConfig:
    """Configuration for a single classification task dataset."""

    name: str
    hf_name: str
    hf_config: Optional[str]
    text_column: str
    label_column: str
    num_labels: int
    label_names: list[str]
    train_size: int = 5000
    val_size: int = 1000
    test_size: int = 1000
    cache_dir: str = ".cache/datasets"


@dataclass
class TaskDataset:
    """
    A fully-prepared task dataset ready for training/evaluation.

    Holds tokenized HuggingFace datasets and DataLoaders.
    """

    config: TaskDatasetConfig
    train_loader: DataLoader
    val_loader: DataLoader
    test_loader: DataLoader
    train_dataset: object = field(repr=False)
    val_dataset: object = field(repr=False)
    test_dataset: object = field(repr=False)

    @property
    def name(self) -> str:
        return self.config.name

    @property
    def num_labels(self) -> int:
        return self.config.num_labels

    @property
    def label_names(self) -> list[str]:
        return self.config.label_names

    def __repr__(self) -> str:
        return (
            f"TaskDataset(name={self.name!r}, num_labels={self.num_labels}, "
            f"train={len(self.train_dataset)}, val={len(self.val_dataset)}, "
            f"test={len(self.test_dataset)})"
        )
