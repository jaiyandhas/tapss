"""
visualization/__init__.py
"""
from visualization.heatmaps import (
    plot_layer_importance_heatmap,
    plot_protection_map,
)
from visualization.distributions import (
    plot_importance_histogram,
    plot_gradient_distribution,
    plot_activation_distribution,
    plot_score_scatter,
)
from visualization.continual import (
    plot_forgetting_over_methods,
    plot_task_accuracy_comparison,
    plot_training_history,
    plot_radar_chart,
)
from visualization.interactive import (
    interactive_importance_scatter,
    interactive_layer_heatmap,
    interactive_comparison_bar,
    interactive_radar,
)
from visualization.report import generate_html_report

__all__ = [
    "plot_layer_importance_heatmap",
    "plot_protection_map",
    "plot_importance_histogram",
    "plot_gradient_distribution",
    "plot_activation_distribution",
    "plot_score_scatter",
    "plot_forgetting_over_methods",
    "plot_task_accuracy_comparison",
    "plot_training_history",
    "plot_radar_chart",
    "interactive_importance_scatter",
    "interactive_layer_heatmap",
    "interactive_comparison_bar",
    "interactive_radar",
    "generate_html_report",
]
