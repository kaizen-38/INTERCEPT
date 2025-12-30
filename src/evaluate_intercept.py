"""
Evaluation Script for INTERCEPT Policies.

This module provides utilities for evaluating trained INTERCEPT policies
and comparing them against baseline intervention strategies.

Key Functions:
    - load_trained_policy: Load a policy checkpoint
    - evaluate_grpo_policy: Run evaluation episodes
    - compare_grpo_vs_baselines: Full comparison pipeline

Usage:
    Command line:
    
    $ python -m src.evaluate_intercept \\
        --checkpoint results/intercept_grpo_*/checkpoints/checkpoint_group_0100.pt \\
        --n-trials 100 \\
        --include-no-timing
    
    Programmatic:
    
    >>> from src.evaluate_intercept import compare_grpo_vs_baselines
    >>> results = compare_grpo_vs_baselines(checkpoint_path, graph, env_config)

Output:
    - Prints ranked comparison of all strategies
    - Saves results to JSON and PNG visualization
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any, Dict

import networkx as nx
import numpy as np
import torch

from src.intercept_env import IndependentCascadeEnv, IndependentCascadeConfig
from src.intercept_grpo import TemporalGRPOPolicy
from src.intercept_baselines import (
    compare_all_baselines,
    BaselineResults,
    visualize_baseline_comparison,
)


def load_trained_policy(
    checkpoint_path: str | Path,
    device: str = "cpu",
) -> TemporalGRPOPolicy:
    """Load a trained policy from checkpoint.
    
    Automatically infers network architecture from saved weights.
    
    Args:
        checkpoint_path: Path to .pt checkpoint file
        device: Device to load model onto
    
    Returns:
        Loaded TemporalGRPOPolicy in eval mode
    
    Raises:
        FileNotFoundError: If checkpoint doesn't exist
    """
    checkpoint_path = Path(checkpoint_path)

    if not checkpoint_path.exists():
        raise FileNotFoundError(f"Checkpoint not found: {checkpoint_path}")

    checkpoint = torch.load(checkpoint_path, map_location=device)
    state_dict = checkpoint["policy_state_dict"]

    # Infer architecture from saved weights
    hidden_dim = state_dict["encoder.fc1.weight"].shape[0]
    n_time_bins = state_dict["time_scorer.3.weight"].shape[0]
    node_feat_dim = state_dict["encoder.fc1.weight"].shape[1]

    print(
        f"Detected architecture: node_feat_dim={node_feat_dim}, "
        f"hidden_dim={hidden_dim}, n_time_bins={n_time_bins}"
    )

    policy = TemporalGRPOPolicy(
        node_feat_dim=node_feat_dim,
        n_time_bins=n_time_bins,
        hidden_dim=hidden_dim,
    ).to(device)
    policy.load_state_dict(state_dict)
    policy.eval()

    print(f"✓ Loaded policy from {checkpoint_path}")
    return policy


def evaluate_grpo_policy(
    policy: TemporalGRPOPolicy,
    graph: nx.Graph,
    env_config: IndependentCascadeConfig,
    n_trials: int = 100,
    device: str = "cpu",
    deterministic: bool = True,
    force_zero_delay: bool = False,
    label: str = "INTERCEPT (GRPO)",
) -> BaselineResults:
    """Evaluate a trained GRPO policy over multiple trials.
    
    Runs the policy on the given network multiple times and computes
    performance statistics.
    
    Args:
        policy: Trained TemporalGRPOPolicy
        graph: Network to evaluate on
        env_config: Environment configuration
        n_trials: Number of evaluation episodes
        device: Device for inference
        deterministic: Use argmax instead of sampling
        force_zero_delay: Override policy delay to 0 (timing ablation)
        label: Name for this evaluation run
    
    Returns:
        BaselineResults with performance statistics
    """
    print(f"Evaluating {label}... ", end="", flush=True)

    all_infections = []
    all_protected = []
    intervention_nodes_list = []

    policy.eval()

    for _ in range(n_trials):
        env = IndependentCascadeEnv(graph, env_config)
        state = env.reset()
        done = False
        steps = 0
        max_steps = env_config.max_steps * 2

        trial_interventions = []
        remaining_budget_prev = env_config.intervention_budget

        while not done and steps < max_steps:
            # Prepare tensors
            node_features = torch.tensor(
                state["node_features"], dtype=torch.float32, device=device
            ).unsqueeze(0)
            adj_matrix = torch.tensor(
                state["adj_matrix"], dtype=torch.float32, device=device
            ).unsqueeze(0)
            node_mask = torch.tensor(
                state["node_mask"], dtype=torch.float32, device=device
            ).unsqueeze(0)

            # Get action from policy
            with torch.no_grad():
                sample = policy.sample_action(
                    node_features, adj_matrix, node_mask, deterministic=deterministic
                )

            node_id = int(sample["node_id"][0].item())
            delay = int(sample["delay"][0].item())

            if force_zero_delay:
                delay = 0

            action = {"node_id": node_id, "delay": delay}
            state, _, done, info = env.step(action)

            # Track interventions
            remaining_budget = info.get("remaining_budget", remaining_budget_prev)
            if remaining_budget < remaining_budget_prev and node_id >= 0:
                trial_interventions.append(node_id)
            remaining_budget_prev = remaining_budget

            steps += 1

        all_infections.append(state["n_infected"])
        all_protected.append(state["n_protected"])
        intervention_nodes_list.append(trial_interventions)

    # Find most commonly selected nodes
    from collections import Counter
    all_nodes_flat = [n for trial in intervention_nodes_list for n in trial]
    most_common = Counter(all_nodes_flat).most_common(env_config.intervention_budget)
    intervention_nodes = [node for node, _ in most_common]

    results = BaselineResults(
        strategy_name=label,
        mean_infected=float(np.mean(all_infections)),
        std_infected=float(np.std(all_infections)),
        median_infected=float(np.median(all_infections)),
        min_infected=int(np.min(all_infections)),
        max_infected=int(np.max(all_infections)),
        mean_protected=float(np.mean(all_protected)),
        intervention_nodes=intervention_nodes,
        all_infections=all_infections,
    )

    print(f"✓ {results.mean_infected:.1f} ± {results.std_infected:.1f} infected")
    return results


def compare_grpo_vs_baselines(
    checkpoint_path: str | Path,
    graph: nx.Graph,
    env_config: IndependentCascadeConfig,
    n_trials: int = 100,
    device: str = "cpu",
    include_no_timing: bool = False,
) -> Dict[str, BaselineResults]:
    """Full comparison of GRPO policy against all baselines.
    
    Loads the trained policy, evaluates all baselines, and produces
    a ranked comparison.
    
    Args:
        checkpoint_path: Path to policy checkpoint
        graph: Network for evaluation
        env_config: Environment configuration
        n_trials: Trials per strategy
        device: Device for policy inference
        include_no_timing: Also evaluate with delay forced to 0
    
    Returns:
        Dictionary mapping strategy name to results
    """
    print("=" * 70)
    print("INTERCEPT: GRPO vs Baselines Comparison")
    print("=" * 70)
    print()

    # Load trained policy
    policy = load_trained_policy(checkpoint_path, device)
    print()

    # Run baseline evaluation
    baseline_results = compare_all_baselines(graph, env_config, n_trials)
    print()

    # Evaluate GRPO
    grpo_results = evaluate_grpo_policy(
        policy,
        graph,
        env_config,
        n_trials=n_trials,
        device=device,
        deterministic=True,
        force_zero_delay=False,
        label="INTERCEPT (GRPO)",
    )

    all_results: Dict[str, BaselineResults] = {
        **baseline_results,
        "INTERCEPT (GRPO)": grpo_results,
    }

    # Optional timing ablation
    if include_no_timing:
        grpo_no_timing = evaluate_grpo_policy(
            policy,
            graph,
            env_config,
            n_trials=n_trials,
            device=device,
            deterministic=True,
            force_zero_delay=True,
            label="INTERCEPT (no timing)",
        )
        all_results["INTERCEPT (no timing)"] = grpo_no_timing

    # Print final comparison
    print()
    print("=" * 70)
    print("FINAL COMPARISON (sorted by performance)")
    print("=" * 70)

    sorted_results = sorted(all_results.items(), key=lambda x: x[1].mean_infected)
    for rank, (name, result) in enumerate(sorted_results, 1):
        marker = "🥇" if rank == 1 else "🥈" if rank == 2 else "🥉" if rank == 3 else "  "
        print(f"{marker} {rank}. {name:20s}: {result.mean_infected:6.1f} ± {result.std_infected:5.1f} infected")

    # Compute improvement over best baseline
    best_baseline = min(baseline_results.values(), key=lambda x: x.mean_infected)
    grpo_improvement = best_baseline.mean_infected - grpo_results.mean_infected
    grpo_improvement_pct = 100 * grpo_improvement / best_baseline.mean_infected

    print()
    print("=" * 70)
    print("KEY RESULT")
    print("=" * 70)
    print(f"Best baseline: {best_baseline.strategy_name} ({best_baseline.mean_infected:.1f} infected)")
    print(f"GRPO:          {grpo_results.mean_infected:.1f} infected")
    print(f"Improvement:   {grpo_improvement:.1f} nodes ({grpo_improvement_pct:.1f}%)")

    if include_no_timing:
        nt = all_results["INTERCEPT (no timing)"]
        delta = nt.mean_infected - grpo_results.mean_infected
        pct = 100 * delta / nt.mean_infected if nt.mean_infected > 0 else 0.0
        print()
        print("Timing ablation:")
        print(f"  INTERCEPT (no timing): {nt.mean_infected:.1f} infected")
        print(f"  INTERCEPT (full):      {grpo_results.mean_infected:.1f} infected")
        print(f"  Gain from timing:      {delta:.1f} nodes ({pct:.1f}%)")
    print("=" * 70)

    return all_results


def save_comparison_results(
    results: Dict[str, BaselineResults],
    output_path: str | Path,
) -> None:
    """Save comparison results to JSON file.
    
    Args:
        results: Dictionary of baseline results
        output_path: Path for output JSON
    """
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    results_dict: Dict[str, Any] = {}
    for name, result in results.items():
        results_dict[name] = {
            "mean_infected": result.mean_infected,
            "std_infected": result.std_infected,
            "median_infected": result.median_infected,
            "min_infected": result.min_infected,
            "max_infected": result.max_infected,
            "mean_protected": result.mean_protected,
        }

    with open(output_path, "w") as f:
        json.dump(results_dict, f, indent=2)

    print(f"✓ Saved results to {output_path}")


def main() -> None:
    """Command-line entry point for evaluation."""
    parser = argparse.ArgumentParser(
        description="Evaluate INTERCEPT policy against baselines"
    )
    parser.add_argument(
        "--checkpoint",
        type=str,
        required=True,
        help="Path to checkpoint file",
    )
    parser.add_argument(
        "--n-trials",
        type=int,
        default=100,
        help="Number of evaluation trials",
    )
    parser.add_argument(
        "--device",
        type=str,
        default="cuda" if torch.cuda.is_available() else "cpu",
        help="Device to run on",
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        default="results/evaluation",
        help="Output directory for results",
    )
    parser.add_argument(
        "--include-no-timing",
        action="store_true",
        help="Also evaluate INTERCEPT with delay forced to 0",
    )

    args = parser.parse_args()

    # Create evaluation environment
    graph = nx.barabasi_albert_graph(80, 3, seed=42)
    env_config = IndependentCascadeConfig(
        infection_prob=0.05,
        initial_infected_count=3,
        intervention_budget=10,
        max_steps=40,
        intervention_cost=0.1,
        delay_penalty=0.01,
        seed=None,
    )

    # Run comparison
    results = compare_grpo_vs_baselines(
        checkpoint_path=args.checkpoint,
        graph=graph,
        env_config=env_config,
        n_trials=args.n_trials,
        device=args.device,
        include_no_timing=args.include_no_timing,
    )

    # Save outputs
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    save_comparison_results(results, output_dir / "comparison_results.json")
    visualize_baseline_comparison(results, str(output_dir / "grpo_vs_baselines.png"))

    print(f"\n✓ Evaluation complete! Results saved to {output_dir}")


if __name__ == "__main__":
    import sys

    if len(sys.argv) == 1:
        print("Example usage:")
        print(
            "  python -m src.evaluate_intercept "
            "--checkpoint results/intercept_grpo_ba80_.../checkpoints/checkpoint_group_0100.pt "
            "--include-no-timing"
        )
    main()
