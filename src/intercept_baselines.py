"""
Baseline Intervention Strategies for Cascade Control.

This module implements centrality-based heuristic strategies for node
intervention in cascade control problems. These serve as baselines for
comparison with learned policies.

All baselines follow the same interface: given a network and budget,
select the top-k nodes according to some centrality measure and apply
immediate interventions (delay=0).

Implemented Strategies:
    - RandomBaseline: Uniform random node selection
    - DegreeBaseline: Highest-degree nodes (hub targeting)
    - PageRankBaseline: Highest PageRank nodes
    - BetweennessBaseline: Highest betweenness centrality
    - ClosenessBaseline: Highest closeness centrality
    - KShellBaseline: Highest k-shell/core number

Example:
    >>> from src.intercept_baselines import compare_all_baselines
    >>> import networkx as nx
    >>> 
    >>> graph = nx.barabasi_albert_graph(200, 3)
    >>> config = IndependentCascadeConfig(intervention_budget=10)
    >>> results = compare_all_baselines(graph, config, n_trials=100)

References:
    - Freeman, L. C. (1978). Centrality in social networks.
    - Kitsak, M. et al. (2010). Identification of influential spreaders in 
      complex networks. Nature Physics.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict, List

import networkx as nx
import numpy as np

from src.intercept_env import IndependentCascadeEnv, IndependentCascadeConfig


@dataclass
class BaselineResults:
    """Results from evaluating a baseline intervention strategy.
    
    Attributes:
        strategy_name: Human-readable name of the strategy
        mean_infected: Mean final infection count across trials
        std_infected: Standard deviation of infection counts
        median_infected: Median infection count
        min_infected: Minimum infection count observed
        max_infected: Maximum infection count observed
        mean_protected: Mean number of successfully protected nodes
        intervention_nodes: List of node IDs selected for intervention
        all_infections: Raw list of final infection counts per trial
    """
    strategy_name: str
    mean_infected: float
    std_infected: float
    median_infected: float
    min_infected: float
    max_infected: float
    mean_protected: float
    intervention_nodes: List[int]
    all_infections: List[int]


class BaselineStrategy:
    """Abstract base class for intervention strategies.
    
    Subclasses must implement select_nodes() to define the node
    selection criterion.
    
    Attributes:
        graph: The NetworkX graph
        n_nodes: Number of nodes in the graph
    
    Args:
        graph: NetworkX Graph to operate on
    """

    def __init__(self, graph: nx.Graph):
        self.graph = graph
        self.n_nodes = graph.number_of_nodes()

    def select_nodes(self, budget: int) -> List[int]:
        """Select nodes for intervention.
        
        Args:
            budget: Maximum number of nodes to select
        
        Returns:
            List of node IDs to intervene on
        
        Raises:
            NotImplementedError: Must be implemented by subclass
        """
        raise NotImplementedError

    def get_name(self) -> str:
        """Return human-readable strategy name."""
        return self.__class__.__name__


class RandomBaseline(BaselineStrategy):
    """Random node selection baseline.
    
    Selects nodes uniformly at random. Useful as a lower-bound baseline.
    """

    def select_nodes(self, budget: int) -> List[int]:
        return list(np.random.choice(self.n_nodes, size=min(budget, self.n_nodes), replace=False))

    def get_name(self) -> str:
        return "Random"


class DegreeBaseline(BaselineStrategy):
    """Degree centrality baseline.
    
    Selects highest-degree nodes (network hubs). Effective in scale-free
    networks where hubs are key spreaders.
    """

    def select_nodes(self, budget: int) -> List[int]:
        degrees = dict(self.graph.degree())
        sorted_nodes = sorted(degrees.items(), key=lambda x: x[1], reverse=True)
        return [node for node, _ in sorted_nodes[:budget]]

    def get_name(self) -> str:
        return "Degree Centrality"


class PageRankBaseline(BaselineStrategy):
    """PageRank centrality baseline.
    
    Selects nodes with highest PageRank. Captures both local connectivity
    and global importance in the network.
    """

    def select_nodes(self, budget: int) -> List[int]:
        pr = nx.pagerank(self.graph)
        sorted_nodes = sorted(pr.items(), key=lambda x: x[1], reverse=True)
        return [node for node, _ in sorted_nodes[:budget]]

    def get_name(self) -> str:
        return "PageRank"


class BetweennessBaseline(BaselineStrategy):
    """Betweenness centrality baseline.
    
    Selects nodes with highest betweenness centrality (nodes that lie on
    many shortest paths). Effective for disrupting information flow.
    """

    def select_nodes(self, budget: int) -> List[int]:
        bc = nx.betweenness_centrality(self.graph)
        sorted_nodes = sorted(bc.items(), key=lambda x: x[1], reverse=True)
        return [node for node, _ in sorted_nodes[:budget]]

    def get_name(self) -> str:
        return "Betweenness"


class ClosenessBaseline(BaselineStrategy):
    """Closeness centrality baseline.
    
    Selects nodes with highest closeness centrality (nodes with short
    average distance to all others). Targets central network positions.
    """

    def select_nodes(self, budget: int) -> List[int]:
        cc = nx.closeness_centrality(self.graph)
        sorted_nodes = sorted(cc.items(), key=lambda x: x[1], reverse=True)
        return [node for node, _ in sorted_nodes[:budget]]

    def get_name(self) -> str:
        return "Closeness"


class KShellBaseline(BaselineStrategy):
    """K-shell decomposition baseline.
    
    Selects nodes in the innermost k-shell (highest core number).
    Based on the influential spreader identification method from
    Kitsak et al. (2010).
    """

    def select_nodes(self, budget: int) -> List[int]:
        core_numbers = nx.core_number(self.graph)
        sorted_nodes = sorted(core_numbers.items(), key=lambda x: x[1], reverse=True)
        return [node for node, _ in sorted_nodes[:budget]]

    def get_name(self) -> str:
        return "K-Shell"


def evaluate_baseline_strategy(
    graph: nx.Graph,
    strategy: BaselineStrategy,
    env_config: IndependentCascadeConfig,
    n_trials: int = 100,
    verbose: bool = True,
) -> BaselineResults:
    """Evaluate a baseline strategy over multiple trials.
    
    Runs the strategy on the given network multiple times with different
    random seeds to estimate performance statistics.
    
    Args:
        graph: Network to evaluate on
        strategy: Baseline strategy instance
        env_config: Environment configuration
        n_trials: Number of evaluation episodes
        verbose: Whether to print progress
    
    Returns:
        BaselineResults with performance statistics
    """
    if verbose:
        print(f"Evaluating {strategy.get_name()}... ", end="", flush=True)

    # Pre-compute intervention nodes
    intervention_nodes = strategy.select_nodes(env_config.intervention_budget)

    all_infections = []
    all_protected = []

    for trial in range(n_trials):
        # Create fresh environment for each trial
        env = IndependentCascadeEnv(graph, env_config)
        state = env.reset()

        # Apply all interventions immediately (delay=0)
        interventions_applied = 0
        for node_id in intervention_nodes:
            if state["node_mask"][node_id] > 0:  # Can intervene
                action = {"node_id": node_id, "delay": 0}
                state, _, _, _ = env.step(action)
                interventions_applied += 1
                if interventions_applied >= env_config.intervention_budget:
                    break

        # Run cascade to completion
        done = False
        max_steps = env_config.max_steps * 2  # Safety limit
        steps = 0
        while not done and steps < max_steps:
            # No more interventions, just let cascade evolve
            action = {"node_id": -1, "delay": 0}  # Invalid action = no-op
            state, _, done, _ = env.step(action)
            steps += 1

        all_infections.append(state["n_infected"])
        all_protected.append(state["n_protected"])

    results = BaselineResults(
        strategy_name=strategy.get_name(),
        mean_infected=float(np.mean(all_infections)),
        std_infected=float(np.std(all_infections)),
        median_infected=float(np.median(all_infections)),
        min_infected=int(np.min(all_infections)),
        max_infected=int(np.max(all_infections)),
        mean_protected=float(np.mean(all_protected)),
        intervention_nodes=intervention_nodes,
        all_infections=all_infections,
    )

    if verbose:
        print(f"✓ {results.mean_infected:.1f} ± {results.std_infected:.1f} infected")

    return results


def compare_all_baselines(
    graph: nx.Graph,
    env_config: IndependentCascadeConfig,
    n_trials: int = 100,
) -> Dict[str, BaselineResults]:
    """Compare all baseline strategies on a network.
    
    Evaluates each baseline strategy and returns results sorted by
    performance.
    
    Args:
        graph: Network to evaluate on
        env_config: Environment configuration
        n_trials: Number of trials per strategy
    
    Returns:
        Dictionary mapping strategy name to BaselineResults
    """
    strategies = [
        RandomBaseline(graph),
        DegreeBaseline(graph),
        PageRankBaseline(graph),
        BetweennessBaseline(graph),
        ClosenessBaseline(graph),
        KShellBaseline(graph),
    ]

    print("=" * 70)
    print("BASELINE COMPARISON")
    print("=" * 70)
    print(f"Network: {graph.number_of_nodes()} nodes, {graph.number_of_edges()} edges")
    print(f"Budget: {env_config.intervention_budget} interventions")
    print(f"Infection prob: {env_config.infection_prob}")
    print(f"Trials: {n_trials}")
    print("=" * 70)
    print()

    results = {}
    for strategy in strategies:
        result = evaluate_baseline_strategy(graph, strategy, env_config, n_trials)
        results[strategy.get_name()] = result

    print()
    print("=" * 70)
    print("SUMMARY")
    print("=" * 70)

    # Sort by performance (lower is better)
    sorted_results = sorted(results.items(), key=lambda x: x[1].mean_infected)

    for name, result in sorted_results:
        print(f"{name:20s}: {result.mean_infected:6.1f} ± {result.std_infected:5.1f} infected")

    print("=" * 70)

    return results


def visualize_baseline_comparison(
    results: Dict[str, BaselineResults],
    save_path: str = "figures/baseline_comparison.png",
) -> None:
    """Create visualization of baseline comparison results.
    
    Generates a two-panel figure with bar chart and box plots
    showing strategy performance.
    
    Args:
        results: Dictionary of baseline results
        save_path: Path to save the figure
    """
    import matplotlib.pyplot as plt
    from pathlib import Path

    Path(save_path).parent.mkdir(parents=True, exist_ok=True)

    fig, axes = plt.subplots(1, 2, figsize=(14, 6))

    # Sort by performance
    sorted_items = sorted(results.items(), key=lambda x: x[1].mean_infected)
    names = [name for name, _ in sorted_items]
    means = [r.mean_infected for _, r in sorted_items]
    stds = [r.std_infected for _, r in sorted_items]

    # Color palette
    colors = ['#e74c3c', '#3498db', '#2ecc71', '#f39c12', '#9b59b6', '#1abc9c']

    # Panel 1: Bar chart with error bars
    ax = axes[0]
    x = np.arange(len(names))
    bars = ax.bar(
        x, means, yerr=stds, capsize=5, alpha=0.8,
        color=colors[:len(names)], edgecolor='black', linewidth=1.5
    )

    ax.set_ylabel('Mean Final Infected Nodes', fontsize=12, fontweight='bold')
    ax.set_title('Baseline Strategy Comparison', fontsize=14, fontweight='bold')
    ax.set_xticks(x)
    ax.set_xticklabels(names, rotation=25, ha='right', fontsize=10)
    ax.grid(axis='y', alpha=0.3)

    # Add value labels on bars
    for bar, mean, std in zip(bars, means, stds):
        height = bar.get_height()
        ax.text(
            bar.get_x() + bar.get_width() / 2., height + std,
            f'{mean:.1f}', ha='center', va='bottom', fontsize=10, fontweight='bold'
        )

    # Panel 2: Box plots
    ax = axes[1]
    data = [r.all_infections for _, r in sorted_items]
    bp = ax.boxplot(data, labels=names, patch_artist=True)

    for patch, color in zip(bp['boxes'], colors[:len(names)]):
        patch.set_facecolor(color)
        patch.set_alpha(0.7)

    ax.set_ylabel('Final Infected Nodes', fontsize=12, fontweight='bold')
    ax.set_title('Distribution of Results', fontsize=14, fontweight='bold')
    ax.set_xticklabels(names, rotation=25, ha='right', fontsize=10)
    ax.grid(axis='y', alpha=0.3)

    plt.tight_layout()
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    print(f"\n✓ Saved baseline comparison to {save_path}")
    plt.close()


if __name__ == "__main__":
    print("Testing baseline strategies on BA network")
    print()

    # Create test network
    graph = nx.barabasi_albert_graph(200, 3, seed=42)

    # Environment config
    env_config = IndependentCascadeConfig(
        infection_prob=0.05,
        initial_infected_count=3,
        intervention_budget=10,
        max_steps=40,
        intervention_cost=0.1,
        delay_penalty=0.01,
        seed=None,
    )

    # Compare all baselines
    results = compare_all_baselines(graph, env_config, n_trials=50)

    # Visualize
    visualize_baseline_comparison(results)

    print("\n✓ Baseline evaluation complete!")
