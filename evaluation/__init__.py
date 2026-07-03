"""
evaluation/__init__.py
"""
from evaluation.metrics import (
    CLMetrics,
    compute_forgetting,
    compute_backward_transfer,
    compute_average_accuracy,
    evaluate_model,
)
from evaluation.tables import build_comparison_table, format_results_table
from evaluation.tracker import ExperimentTracker

__all__ = [
    "CLMetrics",
    "compute_forgetting",
    "compute_backward_transfer",
    "compute_average_accuracy",
    "evaluate_model",
    "build_comparison_table",
    "format_results_table",
    "ExperimentTracker",
]
