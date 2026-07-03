"""
evaluation/tracker.py

SQLite-backed experiment tracker for TAPSS.

Saves all experiment runs to a local SQLite database, enabling:
  - Reproducibility: full config + seed saved per run
  - Comparison: query across multiple runs
  - Resumption: check if a config was already run

Schema
------
experiments (
    id INTEGER PRIMARY KEY,
    run_name TEXT,
    method TEXT,
    model TEXT,
    task_a TEXT,
    task_b TEXT,
    timestamp TEXT,
    seed INTEGER,
    config_json TEXT,        -- full Hydra config as JSON
    task_a_pre_acc REAL,
    task_a_post_acc REAL,
    task_b_acc REAL,
    forgetting REAL,
    avg_acc REAL,
    backward_transfer REAL,
    train_time_s REAL,
    trainable_params INTEGER,
    output_dir TEXT,
    notes TEXT
)
"""
from __future__ import annotations

import json
import logging
import os
import sqlite3
from datetime import datetime
from typing import Any, Optional

import pandas as pd

logger = logging.getLogger(__name__)

_CREATE_TABLE_SQL = """
CREATE TABLE IF NOT EXISTS experiments (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    run_name TEXT NOT NULL,
    method TEXT,
    model TEXT,
    task_a TEXT,
    task_b TEXT,
    timestamp TEXT,
    seed INTEGER,
    config_json TEXT,
    task_a_pre_acc REAL,
    task_a_post_acc REAL,
    task_b_acc REAL,
    forgetting REAL,
    avg_acc REAL,
    backward_transfer REAL,
    train_time_s REAL,
    trainable_params INTEGER,
    output_dir TEXT,
    notes TEXT
);
"""

_INSERT_SQL = """
INSERT INTO experiments (
    run_name, method, model, task_a, task_b, timestamp, seed, config_json,
    task_a_pre_acc, task_a_post_acc, task_b_acc, forgetting, avg_acc,
    backward_transfer, train_time_s, trainable_params, output_dir, notes
) VALUES (
    :run_name, :method, :model, :task_a, :task_b, :timestamp, :seed, :config_json,
    :task_a_pre_acc, :task_a_post_acc, :task_b_acc, :forgetting, :avg_acc,
    :backward_transfer, :train_time_s, :trainable_params, :output_dir, :notes
);
"""


class ExperimentTracker:
    """
    Lightweight SQLite experiment tracker for TAPSS runs.

    Automatically saves configuration, metrics, and metadata to a
    local database file, enabling reproducibility and comparison.

    Usage
    -----
    >>> tracker = ExperimentTracker("outputs/experiments.db")
    >>> tracker.log(result, cfg, seed=42, notes="First run")
    >>> df = tracker.load_all()
    """

    def __init__(self, db_path: str = "outputs/experiments.db"):
        self.db_path = db_path
        os.makedirs(os.path.dirname(db_path) or ".", exist_ok=True)
        self._init_db()
        logger.info(f"[ExperimentTracker] Initialised at {db_path}")

    def _init_db(self) -> None:
        """Create the experiments table if it doesn't exist."""
        with sqlite3.connect(self.db_path) as conn:
            conn.execute(_CREATE_TABLE_SQL)

    def log(
        self,
        result: Any,              # CLResult
        cfg: Any = None,
        seed: int = 42,
        notes: str = "",
    ) -> int:
        """
        Log a CLResult to the database.

        Parameters
        ----------
        result : CLResult
        cfg : DictConfig | None
            Hydra config to serialize.
        seed : int
        notes : str
            Optional human-readable notes.

        Returns
        -------
        int
            Row ID of the inserted record.
        """
        config_json = ""
        if cfg is not None:
            try:
                from omegaconf import OmegaConf
                config_json = OmegaConf.to_yaml(cfg)
            except Exception:
                config_json = str(cfg)

        row = {
            "run_name": result.method_name,
            "method": result.method_name,
            "model": result.model_name,
            "task_a": result.task_a_name,
            "task_b": result.task_b_name,
            "timestamp": datetime.utcnow().isoformat(),
            "seed": seed,
            "config_json": config_json,
            "task_a_pre_acc": result.task_a_pre_accuracy,
            "task_a_post_acc": result.task_a_post_accuracy,
            "task_b_acc": result.task_b_accuracy,
            "forgetting": result.forgetting,
            "avg_acc": result.average_accuracy,
            "backward_transfer": result.backward_transfer,
            "train_time_s": result.total_time_seconds,
            "trainable_params": result.num_trainable_params_task_b,
            "output_dir": getattr(result, "output_dir", ""),
            "notes": notes,
        }

        with sqlite3.connect(self.db_path) as conn:
            cursor = conn.execute(_INSERT_SQL, row)
            row_id = cursor.lastrowid

        logger.info(f"[ExperimentTracker] Logged run '{result.method_name}' (id={row_id})")
        return row_id

    def load_all(self) -> pd.DataFrame:
        """Load all experiment records as a DataFrame."""
        with sqlite3.connect(self.db_path) as conn:
            df = pd.read_sql_query("SELECT * FROM experiments ORDER BY timestamp DESC", conn)
        return df

    def load_by_method(self, method: str) -> pd.DataFrame:
        """Load all runs for a specific method."""
        with sqlite3.connect(self.db_path) as conn:
            df = pd.read_sql_query(
                "SELECT * FROM experiments WHERE method = ? ORDER BY timestamp DESC",
                conn,
                params=(method,),
            )
        return df

    def get_best_by_forgetting(self) -> pd.DataFrame:
        """Return the best run per method (lowest forgetting)."""
        df = self.load_all()
        if df.empty:
            return df
        return (
            df.sort_values("forgetting")
            .groupby("method")
            .first()
            .reset_index()
        )

    def summary(self) -> str:
        """Print a summary of all logged experiments."""
        df = self.load_all()
        if df.empty:
            return "No experiments logged yet."
        summary_df = df[["method", "forgetting", "task_b_acc", "avg_acc", "timestamp"]].copy()
        summary_df["forgetting"] = summary_df["forgetting"].round(4)
        summary_df["task_b_acc"] = summary_df["task_b_acc"].round(4)
        summary_df["avg_acc"] = summary_df["avg_acc"].round(4)
        return summary_df.to_string(index=False)
