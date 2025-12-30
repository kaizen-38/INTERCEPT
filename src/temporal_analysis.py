"""
Temporal Analysis for INTERCEPT Policies.

This module analyzes the temporal decision-making patterns learned by
INTERCEPT policies, specifically examining:
    - Distribution of chosen delay values
    - Relationship between timestep and delay
    - Relationship between node degree and delay

These analyses help understand whether the policy has learned meaningful
timing strategies or defaults to immediate intervention.

Usage:
    $ python -m src.temporal_analysis \\
        --checkpoint results/intercept_grpo_*/checkpoints/checkpoint_group_0100.pt \\
        --n-episodes 200 \\
        --output-dir figures/temporal_analysis
"""

from __future__ import annotations

import argparse
from pathlib import Path
from typing import Dict, List

import matplotlib.pyplot as plt
import networkx as nx
import numpy as np
import torch

from src.intercept_env import IndependentCascadeEnv, IndependentCascadeConfig
from src.evaluate_intercept import load_trained_policy
from src.intercept_grpo import TemporalGRPOPolicy


def make_ba80_env(seed: int | None = None) -> tuple[nx.Graph, IndependentCascadeEnv]:
    """Create BA(80,3) environment for analysis.
    
    Args:
        seed: Random seed (None for random)
    
    Returns:
        Tuple of (graph, environment)
    """
    graph = nx.barabasi_albert_graph(80, 3, seed=seed)
    config = IndependentCascadeConfig(
        infection_prob=0.05,
        initial_infected_count=3,
        intervention_budget=10,
        max_steps=40,
        intervention_cost=0.1,
        delay_penalty=0.01,
        seed=seed,
    )
    env = IndependentCascadeEnv(graph, config)
    return graph, env


def collect_temporal_stats(
    policy: TemporalGRPOPolicy,
    n_episodes: int,
    device: str = "cpu",
) -> Dict[str, List[float]]:
    """Collect temporal decision statistics from policy rollouts.
    
    Runs episodes and records the (timestep, delay, degree) tuple
    for each intervention decision.
    
    Args:
        policy: Trained INTERCEPT policy
        n_episodes: Number of episodes to collect
        device: Device for inference
    
    Returns:
        Dict with keys "delays", "timesteps", "degrees" containing
        lists of values for each intervention
    """
    delays: List[int] = []
    timesteps: List[int] = []
    degrees: List[int] = []

    policy.eval()

    for ep in range(n_episodes):
        graph, env = make_ba80_env(seed=None)
        degree_dict = dict(graph.degree())

        state = env.reset()
        t = 0
        done = False
        remaining_budget_prev = env.config.intervention_budget

        while not done and t < env.config.max_steps * 2:
            node_features = torch.tensor(
                state["node_features"], dtype=torch.float32, device=device
            ).unsqueeze(0)
            adj_matrix = torch.tensor(
                state["adj_matrix"], dtype=torch.float32, device=device
            ).unsqueeze(0)
            node_mask = torch.tensor(
                state["node_mask"], dtype=torch.float32, device=device
            ).unsqueeze(0)

            with torch.no_grad():
                sample = policy.sample_action(
                    node_features, adj_matrix, node_mask, deterministic=False
                )

            node_id = int(sample["node_id"][0].item())
            delay = int(sample["delay"][0].item())

            action = {"node_id": node_id, "delay": delay}
            state, _, done, info = env.step(action)

            remaining_budget = info.get("remaining_budget", remaining_budget_prev)

            # Record only when an actual intervention was scheduled
            if remaining_budget < remaining_budget_prev and node_id >= 0:
                delays.append(delay)
                timesteps.append(t)
                degrees.append(degree_dict.get(node_id, 0))

            remaining_budget_prev = remaining_budget
            t += 1

        if (ep + 1) % 50 == 0:
            print(f"  Collected {ep + 1}/{n_episodes} episodes...")

    return {"delays": delays, "timesteps": timesteps, "degrees": degrees}


def plot_delay_histogram(delays: List[int], output_path: Path) -> None:
    """Plot histogram of chosen delay values.
    
    Args:
        delays: List of delay values
        output_path: Path to save figure
    """
    plt.figure(figsize=(8, 5))
    
    min_delay = min(delays + [0])
    max_delay = max(delays + [0])
    bins = range(min_delay, max_delay + 2)
    
    plt.hist(delays, bins=bins, align="left", edgecolor="black", 
             alpha=0.7, color="#3498db")
    plt.xlabel("Chosen Delay (time bins)", fontsize=12)
    plt.ylabel("Count", fontsize=12)
    plt.title("Distribution of Intervention Delays", fontsize=14, fontweight="bold")
    plt.grid(True, alpha=0.3, axis="y")
    
    # Add statistics annotation
    mean_delay = np.mean(delays)
    plt.axvline(mean_delay, color="red", linestyle="--", linewidth=2, 
                label=f"Mean: {mean_delay:.2f}")
    plt.legend(fontsize=10)
    
    plt.tight_layout()
    output_path.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(output_path, dpi=300, bbox_inches="tight")
    print(f"✓ Saved delay histogram to {output_path}")
    plt.close()


def plot_timestep_vs_delay(
    timesteps: List[int],
    delays: List[int],
    output_path: Path,
) -> None:
    """Plot timestep vs chosen delay scatter.
    
    This reveals whether the policy adjusts timing based on
    how far into the episode it is.
    
    Args:
        timesteps: List of timestep values
        delays: List of corresponding delay values
        output_path: Path to save figure
    """
    plt.figure(figsize=(8, 5))
    plt.scatter(timesteps, delays, alpha=0.4, s=25, c="#2ecc71", edgecolor="none")
    plt.xlabel("Timestep", fontsize=12)
    plt.ylabel("Chosen Delay", fontsize=12)
    plt.title("Intervention Timing: Timestep vs. Delay", fontsize=14, fontweight="bold")
    plt.grid(True, alpha=0.3)
    
    # Add trend line
    if len(timesteps) > 10:
        z = np.polyfit(timesteps, delays, 1)
        p = np.poly1d(z)
        x_line = np.linspace(min(timesteps), max(timesteps), 100)
        plt.plot(x_line, p(x_line), "r--", linewidth=2, alpha=0.8, 
                 label=f"Trend (slope={z[0]:.3f})")
        plt.legend(fontsize=10)
    
    plt.tight_layout()
    output_path.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(output_path, dpi=300, bbox_inches="tight")
    print(f"✓ Saved timestep vs delay plot to {output_path}")
    plt.close()


def plot_degree_vs_delay(
    degrees: List[int],
    delays: List[int],
    output_path: Path,
) -> None:
    """Plot node degree vs chosen delay scatter.
    
    This reveals whether the policy uses different timing
    strategies for high-degree vs low-degree nodes.
    
    Args:
        degrees: List of node degree values
        delays: List of corresponding delay values
        output_path: Path to save figure
    """
    plt.figure(figsize=(8, 5))
    plt.scatter(degrees, delays, alpha=0.4, s=25, c="#e74c3c", edgecolor="none")
    plt.xlabel("Node Degree", fontsize=12)
    plt.ylabel("Chosen Delay", fontsize=12)
    plt.title("Intervention Timing: Node Degree vs. Delay", fontsize=14, fontweight="bold")
    plt.grid(True, alpha=0.3)
    
    # Add trend line
    if len(degrees) > 10:
        z = np.polyfit(degrees, delays, 1)
        p = np.poly1d(z)
        x_line = np.linspace(min(degrees), max(degrees), 100)
        plt.plot(x_line, p(x_line), "b--", linewidth=2, alpha=0.8,
                 label=f"Trend (slope={z[0]:.4f})")
        plt.legend(fontsize=10)
    
    plt.tight_layout()
    output_path.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(output_path, dpi=300, bbox_inches="tight")
    print(f"✓ Saved degree vs delay plot to {output_path}")
    plt.close()


def print_temporal_summary(stats: Dict[str, List[float]]) -> None:
    """Print summary statistics for temporal analysis.
    
    Args:
        stats: Dictionary with delays, timesteps, degrees lists
    """
    delays = stats["delays"]
    timesteps = stats["timesteps"]
    degrees = stats["degrees"]
    
    print("\n" + "=" * 60)
    print("TEMPORAL ANALYSIS SUMMARY")
    print("=" * 60)
    print(f"Total interventions analyzed: {len(delays)}")
    print()
    print("Delay Statistics:")
    print(f"  Mean delay:   {np.mean(delays):.2f}")
    print(f"  Std delay:    {np.std(delays):.2f}")
    print(f"  Min delay:    {min(delays)}")
    print(f"  Max delay:    {max(delays)}")
    print(f"  % immediate:  {100 * sum(1 for d in delays if d == 0) / len(delays):.1f}%")
    print()
    print("Correlations:")
    if len(delays) > 10:
        timestep_corr = np.corrcoef(timesteps, delays)[0, 1]
        degree_corr = np.corrcoef(degrees, delays)[0, 1]
        print(f"  Timestep-Delay correlation: {timestep_corr:.3f}")
        print(f"  Degree-Delay correlation:   {degree_corr:.3f}")
    print("=" * 60)


def main() -> None:
    """Command-line entry point."""
    parser = argparse.ArgumentParser(
        description="Analyze temporal decision patterns in INTERCEPT policies"
    )
    parser.add_argument(
        "--checkpoint",
        type=str,
        required=True,
        help="Path to trained INTERCEPT checkpoint",
    )
    parser.add_argument(
        "--n-episodes",
        type=int,
        default=200,
        help="Number of evaluation episodes",
    )
    parser.add_argument(
        "--device",
        type=str,
        default="cuda" if torch.cuda.is_available() else "cpu",
        help="Device for inference",
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        default="figures/temporal_analysis",
        help="Directory for output figures",
    )
    args = parser.parse_args()

    device = args.device
    checkpoint_path = args.checkpoint

    print(f"Loading policy from {checkpoint_path}...")
    policy = load_trained_policy(checkpoint_path, device=device)

    print(f"\nCollecting statistics from {args.n_episodes} episodes...")
    stats = collect_temporal_stats(policy, n_episodes=args.n_episodes, device=device)

    # Generate plots
    out_dir = Path(args.output_dir)
    plot_delay_histogram(stats["delays"], out_dir / "delay_histogram.png")
    plot_timestep_vs_delay(stats["timesteps"], stats["delays"], 
                           out_dir / "timestep_vs_delay.png")
    plot_degree_vs_delay(stats["degrees"], stats["delays"], 
                         out_dir / "degree_vs_delay.png")
    
    # Print summary
    print_temporal_summary(stats)


if __name__ == "__main__":
    main()
