"""
evaluation/tables.py

Auto-generate comparison tables from CLResult lists.

Produces:
  - Pandas DataFrames for in-memory analysis
  - LaTeX tables for paper drafts
  - HTML tables for reports
  - Console-formatted Rich tables for CLI output
"""
from __future__ import annotations

import io
import logging
from typing import Any

import numpy as np
import pandas as pd
from rich.console import Console
from rich.table import Table

logger = logging.getLogger(__name__)


def build_comparison_table(results: list[Any]) -> pd.DataFrame:
    """
    Build a comparison DataFrame from a list of CLResult objects.

    Parameters
    ----------
    results : list[CLResult]
        Results from multiple methods (TAPSS + baselines).

    Returns
    -------
    pd.DataFrame
        Columns: method, task_a_pre_acc, task_a_post_acc, task_b_acc,
                 forgetting, avg_acc, backward_transfer, train_time_s,
                 trainable_params.
    """
    rows = []
    for r in results:
        rows.append({
            "Method": r.method_name,
            "Task A (Pre)": round(r.task_a_pre_accuracy, 4),
            "Task A (Post)": round(r.task_a_post_accuracy, 4),
            "Task B": round(r.task_b_accuracy, 4),
            "Forgetting ↓": round(r.forgetting, 4),
            "Avg Acc ↑": round(r.average_accuracy, 4),
            "BWT": round(r.backward_transfer, 4),
            "Time (s)": round(r.total_time_seconds, 1),
            "Trainable Params": r.num_trainable_params_task_b,
        })

    df = pd.DataFrame(rows)

    # Sort: lower forgetting = better
    if len(df) > 0:
        df = df.sort_values("Forgetting ↓", ascending=True).reset_index(drop=True)

    return df


def format_results_table(df: pd.DataFrame) -> str:
    """
    Format a comparison DataFrame as a Rich console table string.

    Parameters
    ----------
    df : pd.DataFrame
        Output of build_comparison_table().

    Returns
    -------
    str
        Rich-formatted string (for printing to console).
    """
    console = Console(record=True, width=120)
    table = Table(
        title="TAPSS Continual Learning Comparison",
        show_header=True,
        header_style="bold cyan",
        border_style="dim",
    )

    # Add columns
    for col in df.columns:
        justify = "right" if col not in ("Method",) else "left"
        table.add_column(col, justify=justify, style="white")

    # Add rows
    for _, row in df.iterrows():
        style = ""
        if row["Method"] == "tapss":
            style = "bold green"
        table.add_row(*[str(v) for v in row.values], style=style)

    console.print(table)
    return console.export_text()


def to_latex(df: pd.DataFrame, caption: str = "Continual Learning Results") -> str:
    """
    Export comparison table as a LaTeX table (for paper drafts).

    Parameters
    ----------
    df : pd.DataFrame
    caption : str

    Returns
    -------
    str
        LaTeX table string.
    """
    latex = df.to_latex(
        index=False,
        float_format="%.4f",
        caption=caption,
        label="tab:cl_results",
        bold_rows=False,
        escape=True,
    )
    return latex


def to_html(df: pd.DataFrame, title: str = "Results") -> str:
    """
    Export comparison table as an HTML table with inline styling.

    Parameters
    ----------
    df : pd.DataFrame
    title : str

    Returns
    -------
    str
        HTML string.
    """
    styles = """
    <style>
    table { border-collapse: collapse; font-family: monospace; font-size: 13px; }
    th { background: #1a1a2e; color: #e0e0e0; padding: 8px 12px; }
    td { padding: 6px 12px; border-bottom: 1px solid #333; }
    tr:hover { background: #16213e; }
    .best { color: #4ade80; font-weight: bold; }
    </style>
    """
    html = df.to_html(index=False, border=0, classes="tapss-table")
    return f"<html><head>{styles}</head><body><h2>{title}</h2>{html}</body></html>"


def highlight_best(df: pd.DataFrame) -> pd.DataFrame:
    """
    Return a copy of df with the best value in each metric column highlighted
    by adding '*' to the value string.

    'Best' means:
      - Lowest forgetting
      - Highest Task B accuracy
      - Highest Average Accuracy
    """
    df = df.copy()

    # Best = lowest forgetting
    if "Forgetting ↓" in df.columns:
        best_idx = df["Forgetting ↓"].idxmin()
        df.loc[best_idx, "Forgetting ↓"] = f"{df.loc[best_idx, 'Forgetting ↓']}*"

    # Best = highest task B
    if "Task B" in df.columns:
        # Only numeric entries
        best_idx = df["Task B"].astype(float).idxmax()
        df.loc[best_idx, "Task B"] = f"{df.loc[best_idx, 'Task B']}*"

    return df
