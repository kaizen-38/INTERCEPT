"""
INTERCEPT: Intervention Reinforcement Control for Epidemic Prevention Transmission.

A reinforcement learning framework for optimal intervention timing in network
cascade control using Group Relative Policy Optimization (GRPO).

Core Modules:
    - intercept_env: Independent Cascade environment
    - intercept_grpo: Policy network and GRPO trainer
    - intercept_baselines: Centrality-based baseline strategies

Example:
    >>> from src.intercept_env import IndependentCascadeEnv, IndependentCascadeConfig
    >>> from src.intercept_grpo import TemporalGRPOPolicy, GRPOTrainer
    >>> from src.intercept_baselines import compare_all_baselines
"""

from src.intercept_env import (
    IndependentCascadeEnv,
    IndependentCascadeConfig,
    NodeState,
)
from src.intercept_grpo import (
    TemporalGRPOPolicy,
    GraphEncoder,
    GRPOTrainer,
    GRPOConfig,
)
from src.intercept_baselines import (
    BaselineStrategy,
    BaselineResults,
    RandomBaseline,
    DegreeBaseline,
    PageRankBaseline,
    BetweennessBaseline,
    ClosenessBaseline,
    KShellBaseline,
    compare_all_baselines,
    evaluate_baseline_strategy,
)

__version__ = "0.1.0"
__author__ = "kaizen-38"

__all__ = [
    # Environment
    "IndependentCascadeEnv",
    "IndependentCascadeConfig",
    "NodeState",
    # Policy and Training
    "TemporalGRPOPolicy",
    "GraphEncoder",
    "GRPOTrainer",
    "GRPOConfig",
    # Baselines
    "BaselineStrategy",
    "BaselineResults",
    "RandomBaseline",
    "DegreeBaseline",
    "PageRankBaseline",
    "BetweennessBaseline",
    "ClosenessBaseline",
    "KShellBaseline",
    "compare_all_baselines",
    "evaluate_baseline_strategy",
]

