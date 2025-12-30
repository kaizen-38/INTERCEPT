"""
Multi-Network Experiment Runner.

This module runs INTERCEPT and baseline evaluations across multiple
network types to assess generalization performance.

Evaluates on:
    - Barabási-Albert networks (various sizes)
    - Erdős-Rényi networks (various sizes)
    - Watts-Strogatz networks (various sizes)
    - Real-world SNAP datasets (if available)

Usage:
    $ python -m src.run_multi_network_experiments \\
        --checkpoint results/intercept_grpo_*/checkpoints/checkpoint_group_0100.pt \\
        --n-trials 50 \\
        --output-dir results/multi_network

Output:
    - all_results.json: Complete results for all networks
    - multi_network_comparison.png: Grouped bar chart
    - performance_heatmap.png: Heatmap visualization
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Dict

import matplotlib.pyplot as plt
import numpy as np
import torch

from src.network_datasets import get_all_test_networks, print_network_statistics
from src.intercept_env import IndependentCascadeConfig
from src.intercept_baselines import compare_all_baselines, BaselineResults
from src.evaluate_intercept import load_trained_policy, evaluate_grpo_policy


def run_experiments_on_network(
    network_name: str,
    graph,
    env_config: IndependentCascadeConfig,
    checkpoint_path: str | Path | None = None,
    n_trials: int = 50,
    device: str = "cpu",
) -> Dict[str, BaselineResults]:
    """Run all experiments on a single network.
    
    Evaluates all baseline strategies and optionally the trained
    INTERCEPT policy on the given network.
    
    Args:
        network_name: Name for display
        graph: NetworkX graph
        env_config: Environment configuration
        checkpoint_path: Path to INTERCEPT checkpoint (optional)
        n_trials: Evaluation episodes per strategy
        device: Device for policy inference
    
    Returns:
        Dict mapping strategy name to BaselineResults
    """
    print("\n" + "=" * 70)
    print(f"NETWORK: {network_name}")
    print("=" * 70)
    print_network_statistics(graph, network_name)
    print()

    # Run baselines
    baseline_results = compare_all_baselines(graph, env_config, n_trials)

    # If checkpoint provided, also evaluate GRPO
    if checkpoint_path:
        print()
        policy = load_trained_policy(checkpoint_path, device)
        grpo_results = evaluate_grpo_policy(
            policy, graph, env_config, n_trials, device
        )
        all_results = {**baseline_results, "INTERCEPT (GRPO)": grpo_results}
    else:
        all_results = baseline_results

    return all_results


def create_multi_network_comparison_plot(
    all_network_results: Dict[str, Dict[str, BaselineResults]],
    save_path: str = "figures/multi_network_comparison.png",
) -> None:
    """Create grouped bar chart comparing strategies across networks.
    
    Args:
        all_network_results: Dict[network_name -> Dict[strategy_name -> results]]
        save_path: Path to save figure
    """
    Path(save_path).parent.mkdir(parents=True, exist_ok=True)

    networks = list(all_network_results.keys())
    strategies = list(next(iter(all_network_results.values())).keys())

    fig, ax = plt.subplots(figsize=(16, 8))

    x = np.arange(len(networks))
    width = 0.12  # Width of each bar

    colors = {
        "Random": "#e74c3c",
        "Degree Centrality": "#3498db",
        "PageRank": "#2ecc71",
        "Betweenness": "#f39c12",
        "Closeness": "#9b59b6",
        "K-Shell": "#1abc9c",
        "INTERCEPT (GRPO)": "#e67e22",
    }

    for i, strategy in enumerate(strategies):
        means = [
            all_network_results[net][strategy].mean_infected 
            for net in networks
        ]
        
        offset = (i - len(strategies) / 2) * width
        bars = ax.bar(
            x + offset, means, width, label=strategy,
            color=colors.get(strategy, "#95a5a6"),
            alpha=0.8, edgecolor="black", linewidth=0.5
        )

        # Highlight GRPO
        if strategy == "INTERCEPT (GRPO)":
            for bar in bars:
                bar.set_linewidth(2.5)
                bar.set_edgecolor("#c0392b")

    ax.set_ylabel("Mean Final Infected Nodes", fontsize=13, fontweight="bold")
    ax.set_title("INTERCEPT Performance Across Network Types", 
                 fontsize=15, fontweight="bold")
    ax.set_xticks(x)
    ax.set_xticklabels(networks, rotation=35, ha="right", fontsize=10)
    ax.legend(fontsize=9, loc="upper left", ncol=2)
    ax.grid(axis="y", alpha=0.3)

    plt.tight_layout()
    plt.savefig(save_path, dpi=300, bbox_inches="tight")
    print(f"\n✓ Saved multi-network comparison to {save_path}")
    plt.close()


def create_heatmap_comparison(
    all_network_results: Dict[str, Dict[str, BaselineResults]],
    save_path: str = "figures/performance_heatmap.png",
) -> None:
    """Create heatmap showing relative performance.
    
    Normalizes results per network to show relative performance
    of each strategy.
    
    Args:
        all_network_results: Results dictionary
        save_path: Path to save figure
    """
    Path(save_path).parent.mkdir(parents=True, exist_ok=True)

    networks = list(all_network_results.keys())
    strategies = list(next(iter(all_network_results.values())).keys())

    # Build matrix: rows=networks, cols=strategies
    matrix = np.zeros((len(networks), len(strategies)))

    for i, network in enumerate(networks):
        for j, strategy in enumerate(strategies):
            matrix[i, j] = all_network_results[network][strategy].mean_infected

    # Normalize by network (to show relative performance)
    matrix_normalized = matrix / matrix.max(axis=1, keepdims=True)

    fig, ax = plt.subplots(figsize=(12, 8))
    im = ax.imshow(matrix_normalized, cmap="RdYlGn_r", aspect="auto")

    # Set ticks
    ax.set_xticks(np.arange(len(strategies)))
    ax.set_yticks(np.arange(len(networks)))
    ax.set_xticklabels(strategies, rotation=45, ha="right", fontsize=10)
    ax.set_yticklabels(networks, fontsize=10)

    # Add values
    for i in range(len(networks)):
        for j in range(len(strategies)):
            ax.text(
                j, i, f"{matrix[i, j]:.0f}",
                ha="center", va="center", color="black", fontsize=8
            )

    ax.set_title(
        "Performance Across Networks (lower is better)",
        fontsize=14, fontweight="bold", pad=20
    )

    cbar = plt.colorbar(im, ax=ax)
    cbar.set_label("Normalized Infections", rotation=270, labelpad=20)

    plt.tight_layout()
    plt.savefig(save_path, dpi=300, bbox_inches="tight")
    print(f"✓ Saved performance heatmap to {save_path}")
    plt.close()


def save_all_results(
    all_network_results: Dict[str, Dict[str, BaselineResults]],
    output_path: str | Path,
) -> None:
    """Save all results to JSON file.
    
    Args:
        all_network_results: Complete results dictionary
        output_path: Path for output JSON
    """
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    results_dict = {}
    for network, strategies in all_network_results.items():
        results_dict[network] = {}
        for strategy, result in strategies.items():
            results_dict[network][strategy] = {
                "mean_infected": result.mean_infected,
                "std_infected": result.std_infected,
                "median_infected": result.median_infected,
                "min_infected": result.min_infected,
                "max_infected": result.max_infected,
            }

    with open(output_path, "w") as f:
        json.dump(results_dict, f, indent=2)

    print(f"✓ Saved all results to {output_path}")


def main() -> None:
    """Command-line entry point."""
    parser = argparse.ArgumentParser(
        description="Run INTERCEPT experiments across multiple network types"
    )
    parser.add_argument(
        "--checkpoint",
        type=str,
        default=None,
        help="Path to GRPO checkpoint (optional)",
    )
    parser.add_argument(
        "--n-trials",
        type=int,
        default=50,
        help="Number of trials per network",
    )
    parser.add_argument(
        "--networks",
        type=str,
        nargs="+",
        default=None,
        help="Specific networks to test (default: standard set)",
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        default="results/multi_network",
        help="Output directory",
    )
    parser.add_argument(
        "--device",
        type=str,
        default="cuda" if torch.cuda.is_available() else "cpu",
        help="Device for policy inference",
    )

    args = parser.parse_args()

    # Load networks
    all_networks = get_all_test_networks(include_snap=True)

    # Filter if specific networks requested
    if args.networks:
        all_networks = {
            k: v for k, v in all_networks.items() 
            if k in args.networks
        }

    # Default test set (smaller networks for faster testing)
    test_networks = {
        "BA-100": all_networks.get("BA-100"),
        "BA-200": all_networks.get("BA-200"),
        "ER-100": all_networks.get("ER-100"),
        "ER-200": all_networks.get("ER-200"),
        "WS-100": all_networks.get("WS-100"),
        "WS-200": all_networks.get("WS-200"),
    }
    # Remove None values
    test_networks = {k: v for k, v in test_networks.items() if v is not None}

    # Environment config (same for all networks)
    env_config = IndependentCascadeConfig(
        infection_prob=0.05,
        initial_infected_count=3,
        intervention_budget=10,
        max_steps=40,
        intervention_cost=0.1,
        delay_penalty=0.01,
        seed=None,
    )

    # Run experiments
    all_results = {}

    for network_name, graph in test_networks.items():
        results = run_experiments_on_network(
            network_name=network_name,
            graph=graph,
            env_config=env_config,
            checkpoint_path=args.checkpoint,
            n_trials=args.n_trials,
            device=args.device,
        )
        all_results[network_name] = results

    # Save and visualize
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    save_all_results(all_results, output_dir / "all_results.json")
    create_multi_network_comparison_plot(
        all_results, str(output_dir / "multi_network_comparison.png")
    )
    create_heatmap_comparison(
        all_results, str(output_dir / "performance_heatmap.png")
    )

    # Print summary
    print("\n" + "=" * 70)
    print("SUMMARY: Best Method Per Network")
    print("=" * 70)

    for network_name, strategies in all_results.items():
        best = min(strategies.items(), key=lambda x: x[1].mean_infected)
        print(
            f"{network_name:15s}: {best[0]:20s} "
            f"({best[1].mean_infected:.1f} infected)"
        )

    print("=" * 70)
    print(f"\n✓ All experiments complete! Results in {output_dir}")


if __name__ == "__main__":
    main()
