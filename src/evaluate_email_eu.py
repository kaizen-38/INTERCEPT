"""
Evaluation on SNAP Email-Eu-core Network.

This module evaluates INTERCEPT policies on the real-world Email-Eu-core
network from the Stanford SNAP dataset collection.

The Email-Eu-core network is derived from email communications within a
large European research institution. It has ~1000 nodes and ~25000 edges.

Usage:
    $ python -m src.evaluate_email_eu \\
        --checkpoint results/intercept_grpo_*/checkpoints/checkpoint_group_0100.pt \\
        --n-trials 100 \\
        --include-no-timing

Prerequisites:
    Download the Email-Eu-core dataset from SNAP and place email-Eu-core.txt
    in the data/snap/ directory.

References:
    Leskovec, J., Kleinberg, J., & Faloutsos, C. (2007). Graph evolution:
    Densification and shrinking diameters. ACM TKDD.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import networkx as nx
import numpy as np
import torch

from src.network_datasets import download_snap_dataset, print_network_statistics
from src.intercept_env import IndependentCascadeConfig
from src.evaluate_intercept import (
    load_trained_policy,
    evaluate_grpo_policy,
    save_comparison_results,
)
from src.intercept_baselines import compare_all_baselines, visualize_baseline_comparison


def evaluate_on_email_eu(
    checkpoint: str,
    n_trials: int = 100,
    device: str = "cpu",
    include_no_timing: bool = True,
    output_dir: str = "results/email_eu_eval",
):
    """Full evaluation pipeline for the SNAP Email-Eu-core network.
    
    Loads the Email-Eu-core network, evaluates all baselines and the
    trained INTERCEPT policy, and saves results.
    
    Args:
        checkpoint: Path to trained INTERCEPT checkpoint
        n_trials: Number of evaluation episodes
        device: Device for policy inference
        include_no_timing: Include timing ablation (delay=0)
        output_dir: Directory for output files
    
    Returns:
        Dict mapping strategy name to BaselineResults
    
    Raises:
        RuntimeError: If Email-Eu-core dataset not found
    """
    # Load SNAP Email-Eu-core graph
    print("Loading Email-Eu-core dataset...")
    graph = download_snap_dataset("email-Eu-core")
    if graph is None:
        raise RuntimeError(
            "Email-Eu-core dataset not found. "
            "Place email-Eu-core.txt in data/snap/"
        )

    print_network_statistics(graph, "Email-Eu-core")

    # Environment config (adjusted for larger network)
    env_config = IndependentCascadeConfig(
        infection_prob=0.05,       # Standard transmission rate
        initial_infected_count=5,  # More seeds for larger network
        intervention_budget=20,    # Larger budget for larger network
        max_steps=60,              # Longer episodes
        intervention_cost=0.1,
        delay_penalty=0.01,
        seed=None,
    )

    # Load trained policy
    print(f"\nLoading policy from {checkpoint}...")
    policy = load_trained_policy(checkpoint, device=device)
    policy.eval()

    # Evaluate baselines
    print("\nEvaluating baselines on Email-Eu-core...")
    baseline_results = compare_all_baselines(
        graph=graph,
        env_config=env_config,
        n_trials=n_trials,
    )

    # Evaluate INTERCEPT
    print("\nEvaluating INTERCEPT (GRPO) on Email-Eu-core...")
    grpo_results = evaluate_grpo_policy(
        policy=policy,
        graph=graph,
        env_config=env_config,
        n_trials=n_trials,
        device=device,
        deterministic=True,
        force_zero_delay=False,
        label="INTERCEPT (GRPO)",
    )

    results = {**baseline_results, "INTERCEPT (GRPO)": grpo_results}

    # Optional timing ablation
    if include_no_timing:
        print("\nEvaluating timing ablation: INTERCEPT (no timing)...")
        no_timing_results = evaluate_grpo_policy(
            policy=policy,
            graph=graph,
            env_config=env_config,
            n_trials=n_trials,
            device=device,
            deterministic=True,
            force_zero_delay=True,
            label="INTERCEPT (no timing)",
        )
        results["INTERCEPT (no timing)"] = no_timing_results

    # Save results
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    save_comparison_results(results, output_dir / "email_eu_results.json")
    visualize_baseline_comparison(
        results,
        str(output_dir / "email_eu_vs_baselines.png"),
    )

    # Print final summary
    print("\n" + "=" * 70)
    print("FINAL RESULTS (Email-Eu-core)")
    print("=" * 70)
    for name, res in sorted(results.items(), key=lambda x: x[1].mean_infected):
        print(f"{name:25s}: {res.mean_infected:6.2f} ± {res.std_infected:5.2f}")
    print("=" * 70 + "\n")

    return results


def main() -> None:
    """Command-line entry point."""
    parser = argparse.ArgumentParser(
        description="Evaluate INTERCEPT on SNAP Email-Eu-core network"
    )
    parser.add_argument(
        "--checkpoint",
        type=str,
        required=True,
        help="Path to GRPO checkpoint",
    )
    parser.add_argument(
        "--n-trials",
        type=int,
        default=100,
        help="Number of evaluation episodes",
    )
    parser.add_argument(
        "--device",
        type=str,
        default="cuda" if torch.cuda.is_available() else "cpu",
        help="Device (cpu/cuda)",
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        default="results/email_eu_eval",
        help="Output directory",
    )
    parser.add_argument(
        "--include-no-timing",
        action="store_true",
        help="Include timing ablation analysis",
    )

    args = parser.parse_args()

    evaluate_on_email_eu(
        checkpoint=args.checkpoint,
        n_trials=args.n_trials,
        device=args.device,
        include_no_timing=args.include_no_timing,
        output_dir=args.output_dir,
    )


if __name__ == "__main__":
    main()
