"""
datasets/preprocessing.py

Tokenization and dataset formatting for TAPSS experiments.
Converts raw HuggingFace datasets into PyTorch-ready format.
"""
from __future__ import annotations

import logging
from typing import Any

import datasets as _hf_datasets

logger = logging.getLogger(__name__)


def tokenize_dataset(
    train_raw: _hf_datasets.Dataset,
    val_raw: _hf_datasets.Dataset,
    test_raw: _hf_datasets.Dataset,
    tokenizer: Any,
    text_column: str,
    label_column: str,
    max_length: int = 128,
) -> tuple[_hf_datasets.Dataset, _hf_datasets.Dataset, _hf_datasets.Dataset]:
    """
    Tokenize raw text datasets and format them for PyTorch.

    Parameters
    ----------
    train_raw, val_raw, test_raw :
        Raw HuggingFace datasets with at least text_column and label_column.
    tokenizer :
        HuggingFace tokenizer.
    text_column : str
        Name of the text field.
    label_column : str
        Name of the label field.
    max_length : int
        Maximum token length.

    Returns
    -------
    (train_tok, val_tok, test_tok) : tuple of tokenized datasets
    """
    def _tokenize(batch):
        encoding = tokenizer(
            batch[text_column],
            truncation=True,
            padding="max_length",
            max_length=max_length,
        )
        encoding["labels"] = batch[label_column]
        return encoding

    # Determine which columns to remove (keep only model inputs)
    def _process(ds: _hf_datasets.Dataset) -> _hf_datasets.Dataset:
        cols_to_remove = [
            c for c in ds.column_names if c not in ("input_ids", "attention_mask", "labels")
        ]
        tokenized = ds.map(
            _tokenize,
            batched=True,
            remove_columns=cols_to_remove,
            desc="Tokenizing",
        )
        tokenized.set_format("torch", columns=["input_ids", "attention_mask", "labels"])
        return tokenized

    logger.info("Tokenizing train split")
    train_tok = _process(train_raw)

    logger.info("Tokenizing val split")
    val_tok = _process(val_raw)

    logger.info("Tokenizing test split")
    test_tok = _process(test_raw)

    return train_tok, val_tok, test_tok
