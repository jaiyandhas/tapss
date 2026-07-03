"""
continual_learning/__init__.py
"""
from continual_learning.pipeline import ContinualLearningPipeline, CLResult
from continual_learning.baselines import (
    BaselineMethod,
    VanillaLoRABaseline,
    NaiveFinetuningBaseline,
    RandomProtectionBaseline,
)
from continual_learning.ewc import EWCBaseline

__all__ = [
    "ContinualLearningPipeline",
    "CLResult",
    "BaselineMethod",
    "VanillaLoRABaseline",
    "NaiveFinetuningBaseline",
    "RandomProtectionBaseline",
    "EWCBaseline",
]
