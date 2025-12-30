"""
Cascade Visualization Utilities.

This module provides tools for visualizing how different intervention
strategies affect cascade progression over time.

Usage:
    $ python -m src.visualize_cascades \\
        --checkpoint results/intercept_grpo_*/checkpoints/checkpoint_group_0100.pt \\
        --output figures/cascade_example.png

The output shows cascade curves for:
    - No intervention (baseline)
    - Betweenness centrality strategy
    - INTERCEPT (learned policy)
"""

from __future__ import annotations

import argparse
from pathlib import Path
from typing import Dict, List

import matplotlib.pyplot as plt
import networkx as nx
import torch

from src.intercept_env import IndependentCascadeEnv, IndependentCascadeConfig
from src.intercept_grpo import TemporalGRPOPolicy
from src.evaluate_intercept import load_trained_policy


def make_ba80_env(seed: int | None = 42) -> tuple[nx.Graph, IndependentCascadeEnv]:
    """Create BA(80,3) environment with standard config.
    
    Args:
        seed: Random seed for reproducibility
    
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


def run_no_intervention(env: IndependentCascadeEnv) -> List[int]:
    """Simulate cascade with no interventions.
    
    Args:
        env: Cascade environment
    
    Returns:
        List of infection counts at each timestep
    """
    state = env.reset()
    infected_history = [state["n_infected"]]

    done = False
    steps = 0
    max_steps = env.config.max_steps * 2

    while not done and steps < max_steps:
        action = {"node_id": -1, "delay": 0}
        state, _, done, _ = env.step(action)
        infected_history.append(state["n_infected"])
        steps += 1

    return infected_history


def run_betweenness_strategy(
    graph: nx.Graph,
    env: IndependentCascadeEnv,
) -> List[int]:
    """Run cascade with betweenness centrality intervention.
    
    Intervenes on top-k betweenness nodes with immediate interventions.
    
    Args:
        graph: Network graph
        env: Cascade environment
    
    Returns:
        List of infection counts at each timestep
    """
    state = env.reset()
    infected_history = [state["n_infected"]]

    k = env.config.intervention_budget

    # Pre-compute betweenness ranking
    bet = nx.betweenness_centrality(graph)
    ranked_nodes = sorted(bet.keys(), key=lambda n: bet[n], reverse=True)
    top_k = ranked_nodes[:k]

    done = False
    steps = 0
    max_steps = env.config.max_steps * 2

    while not done and steps < max_steps:
        if steps < k:
            node_id = int(top_k[steps])
        else:
            node_id = -1

        action = {"node_id": node_id, "delay": 0}
        state, _, done, _ = env.step(action)
        infected_history.append(state["n_infected"])
        steps += 1

    return infected_history


def run_intercept_policy(
    env: IndependentCascadeEnv,
    policy: TemporalGRPOPolicy,
    device: str = "cpu",
    deterministic: bool = True,
) -> List[int]:
    """Run cascade with INTERCEPT policy.
    
    Args:
        env: Cascade environment
        policy: Trained INTERCEPT policy
        device: Device for inference
        deterministic: Use argmax instead of sampling
    
    Returns:
        List of infection counts at each timestep
    """
    policy.eval()
    state = env.reset()
    infected_history = [state["n_infected"]]

    done = False
    steps = 0
    max_steps = env.config.max_steps * 2

    while not done and steps < max_steps:
        node_features = torch.tensor(
            state["node_features"], dtype=torch.float32, device=device
        ).unsqueeze(0)
        adj_matrix = torch.tensor(
            state["adj_matrix"], dtype=torch.float32, device=device
        ).unsqueeze(0)
        node_mask = torch.tensor(
            state["node_mask"], dtype=torch.float32, device=device
        ).unsqueeze(0)

        sample = policy.sample_action(
            node_features, adj_matrix, node_mask, deterministic=deterministic
        )
        node_id = int(sample["node_id"][0].item())
        delay = int(sample["delay"][0].item())

        action = {"node_id": node_id, "delay": delay}
        state, _, done, _ = env.step(action)
        infected_history.append(state["n_infected"])
        steps += 1

    return infected_history


def plot_cascades(
    histories: Dict[str, List[int]],
    output_path: str | Path | None = None,
) -> None:
    """Plot cascade progression curves.
    
    Creates a line plot showing infection count over time for
    each intervention strategy.
    
    Args:
        histories: Dict mapping strategy name to infection history
        output_path: Path to save figure (shows plot if None)
    """
    plt.figure(figsize=(10, 6))
    
    colors = {
        "No intervention": "#e74c3c",
        "Betweenness": "#3498db", 
        "INTERCEPT (GRPO)": "#2ecc71",
    }
    
    for name, series in histories.items():
        color = colors.get(name, "#95a5a6")
        plt.plot(series, marker="o", markersize=4, label=name, 
                 linewidth=2, color=color, alpha=0.8)

    plt.xlabel("Timestep", fontsize=12)
    plt.ylabel("Number of Infected Nodes", fontsize=12)
    plt.title("Cascade Progression Under Different Intervention Strategies", 
              fontsize=14, fontweight="bold")
    plt.legend(fontsize=11, loc="upper left")
    plt.grid(True, alpha=0.3)
    plt.tight_layout()

    if output_path is not None:
        output_path = Path(output_path)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        plt.savefig(output_path, dpi=300, bbox_inches="tight")
        print(f"✓ Saved cascade plot to {output_path}")
    else:
        plt.show()
    
    plt.close()


def main() -> None:
    """Command-line entry point."""
    parser = argparse.ArgumentParser(
        description="Visualize cascade progression under different strategies"
    )
    parser.add_argument(
        "--checkpoint",
        type=str,
        required=True,
        help="Path to trained INTERCEPT checkpoint",
    )
    parser.add_argument(
        "--device",
        type=str,
        default="cuda" if torch.cuda.is_available() else "cpu",
        help="Device for inference",
    )
    parser.add_argument(
        "--output",
        type=str,
        default="figures/cascade_example.png",
        help="Output figure path",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=123,
        help="Random seed for reproducible visualization",
    )
    args = parser.parse_args()

    device = args.device

    # Create environments with same seed for fair comparison
    graph, env_no_int = make_ba80_env(seed=args.seed)
    _, env_bet = make_ba80_env(seed=args.seed)
    _, env_grpo = make_ba80_env(seed=args.seed)

    # Load policy
    policy = load_trained_policy(args.checkpoint, device=device)

    # Run three strategies
    print("Running cascade simulations...")
    histories = {
        "No intervention": run_no_intervention(env_no_int),
        "Betweenness": run_betweenness_strategy(graph, env_bet),
        "INTERCEPT (GRPO)": run_intercept_policy(env_grpo, policy, device=device),
    }

    # Print final results
    print("\nFinal infection counts:")
    for name, hist in histories.items():
        print(f"  {name}: {hist[-1]} infected")

    plot_cascades(histories, args.output)


if __name__ == "__main__":
    main()
