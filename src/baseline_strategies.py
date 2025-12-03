"""
baseline_strategies.py

Baseline intervention strategies and evaluation utilities for the
TemporalCascadeEnv.

Includes:
    - No-intervention baseline
    - Random policy
    - Degree centrality policy
    - PageRank policy
    - Betweenness centrality policy
    - Closeness centrality policy

Each policy decides which nodes to treat at each timestep, given the
current environment state.

Example usage (from project root):

    python -m src.baseline_strategies

This will:
    - Run all baselines for several episodes
    - Print mean / std / min / max infections
    - Save a bar plot comparing baselines
    - Save an example network + timeline plot for one baseline run
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Callable, Dict, List, Tuple

import matplotlib.pyplot as plt
import networkx as nx
import numpy as np

from .environment import TemporalCascadeEnv


# ---------------------------------------------------------------------
# Types
# ---------------------------------------------------------------------

Action = Tuple[int, int]  # (node_id, delay)
Policy = Callable[[Dict], List[Action]]  # takes obs, returns actions


@dataclass
class BaselineResult:
    infections: np.ndarray  # total infections across episodes
    mean: float
    std: float
    min: float
    max: float


# ---------------------------------------------------------------------
# Policy factories
# ---------------------------------------------------------------------

def _treatable_nodes(env: TemporalCascadeEnv, obs: Dict) -> List[int]:
    """Nodes we are allowed to treat: SUSCEPTIBLE or EXPOSED."""
    state = obs["state"]
    treatable_states = {env.SUSCEPTIBLE, env.EXPOSED}
    return [int(i) for i, s in enumerate(state) if int(s) in treatable_states]


def make_no_intervention_policy(env: TemporalCascadeEnv, rng: np.random.Generator) -> Policy:
    """Never intervene. Useful as a baseline."""
    def policy(obs: Dict) -> List[Action]:
        return []
    return policy


def make_random_policy(env: TemporalCascadeEnv, rng: np.random.Generator) -> Policy:
    """Pick random treatable nodes each step."""
    budget = env.intervention_budget

    def policy(obs: Dict) -> List[Action]:
        nodes = _treatable_nodes(env, obs)
        if not nodes or budget <= 0:
            return []

        k = min(budget, len(nodes))
        chosen = rng.choice(nodes, size=k, replace=False)
        return [(int(n), 0) for n in chosen]  # delay=0 (immediate)
    return policy


def make_degree_policy(env: TemporalCascadeEnv, rng: np.random.Generator) -> Policy:
    """Always target highest-degree nodes (hubs)."""
    budget = env.intervention_budget
    degrees = dict(env.G.degree())
    ranking = sorted(degrees.keys(), key=lambda n: degrees[n], reverse=True)
    already_targeted = set()

    def policy(obs: Dict) -> List[Action]:
        nodes = _treatable_nodes(env, obs)
        if not nodes or budget <= 0:
            return []

        nodes_set = set(nodes)
        # pick high-degree nodes that are treatable and not yet targeted
        candidates = [
            int(n)
            for n in ranking
            if (n in nodes_set) and (n not in already_targeted)
        ]
        chosen = candidates[:budget]
        already_targeted.update(chosen)
        return [(int(n), 0) for n in chosen]
    return policy


def make_pagerank_policy(env: TemporalCascadeEnv, rng: np.random.Generator) -> Policy:
    """Pick nodes with highest PageRank."""
    budget = env.intervention_budget
    pr = nx.pagerank(env.G)
    ranking = sorted(pr.keys(), key=lambda n: pr[n], reverse=True)
    already_targeted = set()

    def policy(obs: Dict) -> List[Action]:
        nodes = _treatable_nodes(env, obs)
        if not nodes or budget <= 0:
            return []

        nodes_set = set(nodes)
        candidates = [
            int(n)
            for n in ranking
            if (n in nodes_set) and (n not in already_targeted)
        ]
        chosen = candidates[:budget]
        already_targeted.update(chosen)
        return [(int(n), 0) for n in chosen]
    return policy


def make_betweenness_policy(env: TemporalCascadeEnv, rng: np.random.Generator) -> Policy:
    """Pick nodes with highest betweenness centrality (bridges)."""
    budget = env.intervention_budget
    bc = nx.betweenness_centrality(env.G)
    ranking = sorted(bc.keys(), key=lambda n: bc[n], reverse=True)
    already_targeted = set()

    def policy(obs: Dict) -> List[Action]:
        nodes = _treatable_nodes(env, obs)
        if not nodes or budget <= 0:
            return []

        nodes_set = set(nodes)
        candidates = [
            int(n)
            for n in ranking
            if (n in nodes_set) and (n not in already_targeted)
        ]
        chosen = candidates[:budget]
        already_targeted.update(chosen)
        return [(int(n), 0) for n in chosen]
    return policy


def make_closeness_policy(env: TemporalCascadeEnv, rng: np.random.Generator) -> Policy:
    """Pick nodes with highest closeness centrality (close to everyone)."""
    budget = env.intervention_budget
    cc = nx.closeness_centrality(env.G)
    ranking = sorted(cc.keys(), key=lambda n: cc[n], reverse=True)
    already_targeted = set()

    def policy(obs: Dict) -> List[Action]:
        nodes = _treatable_nodes(env, obs)
        if not nodes or budget <= 0:
            return []

        nodes_set = set(nodes)
        candidates = [
            int(n)
            for n in ranking
            if (n in nodes_set) and (n not in already_targeted)
        ]
        chosen = candidates[:budget]
        already_targeted.update(chosen)
        return [(int(n), 0) for n in chosen]
    return policy


# ---------------------------------------------------------------------
# Evaluation utilities
# ---------------------------------------------------------------------

def run_single_episode(
    policy_factory: Callable[[TemporalCascadeEnv, np.random.Generator], Policy],
    env_kwargs: Dict,
    seed: int,
) -> Tuple[int, TemporalCascadeEnv, Dict]:
    """
    Run one episode with a given policy factory.

    Returns:
        total_infected, env, final_obs
    """
    env = TemporalCascadeEnv(seed=seed, **env_kwargs)
    obs = env.reset()
    rng = np.random.default_rng(seed)

    policy = policy_factory(env, rng)
    done = False
    last_obs = obs

    while not done:
        actions = policy(last_obs)
        last_obs, reward, done, info = env.step(actions)

    stats = env.get_episode_stats()
    total_infected = stats["final_state"]["total_infected"]
    return total_infected, env, last_obs


def evaluate_policy(
    name: str,
    policy_factory: Callable[[TemporalCascadeEnv, np.random.Generator], Policy],
    env_kwargs: Dict,
    n_runs: int = 50,
    base_seed: int = 0,
) -> BaselineResult:
    """Evaluate one baseline across many runs."""
    totals: List[int] = []

    for i in range(n_runs):
        seed = base_seed + i
        total_infected, env, _ = run_single_episode(policy_factory, env_kwargs, seed)
        totals.append(total_infected)

    arr = np.asarray(totals, dtype=float)
    return BaselineResult(
        infections=arr,
        mean=float(arr.mean()),
        std=float(arr.std(ddof=1)) if len(arr) > 1 else 0.0,
        min=float(arr.min()),
        max=float(arr.max()),
    )


def evaluate_all_baselines(
    env_kwargs: Dict,
    n_runs: int = 50,
    base_seed: int = 0,
) -> Dict[str, BaselineResult]:
    """Evaluate all baseline strategies and return a result dict."""
    policies = {
        "No Interventions": make_no_intervention_policy,
        "Random":           make_random_policy,
        "Degree":           make_degree_policy,
        "PageRank":         make_pagerank_policy,
        "Betweenness":      make_betweenness_policy,
        "Closeness":        make_closeness_policy,
    }

    results: Dict[str, BaselineResult] = {}
    for idx, (name, factory) in enumerate(policies.items()):
        print(f"Evaluating {name} ...")
        res = evaluate_policy(
            name,
            factory,
            env_kwargs,
            n_runs=n_runs,
            base_seed=base_seed + idx * 1000,
        )
        results[name] = res
        print(
            f"  {name:15s}: "
            f"mean={res.mean:.2f}, std={res.std:.2f}, "
            f"min={res.min:.0f}, max={res.max:.0f}"
        )

    return results


# ---------------------------------------------------------------------
# Visualization helpers
# ---------------------------------------------------------------------

STATE_COLORS = {
    0: "lightgray",  # SUSCEPTIBLE
    1: "gold",       # EXPOSED
    2: "red",        # INFECTIOUS
    3: "dodgerblue", # TREATED
    4: "green",      # RECOVERED
}


def plot_baseline_bar(
    results: Dict[str, BaselineResult],
    out_path: str = "baseline_comparison.png",
) -> None:
    """Bar chart of mean infections (+/- std) for each baseline."""
    names = list(results.keys())
    means = [results[n].mean for n in names]
    stds = [results[n].std for n in names]

    x = np.arange(len(names))
    plt.figure(figsize=(8, 4))
    plt.bar(x, means, yerr=stds, capsize=5)
    plt.xticks(x, names, rotation=45, ha="right")
    plt.ylabel("Final infections (mean ± std)")
    plt.title("Baseline comparison")
    plt.tight_layout()
    plt.savefig(out_path, bbox_inches="tight")
    plt.close()
    print(f"Saved {out_path}")


def plot_example_network(
    env: TemporalCascadeEnv,
    obs: Dict,
    out_path: str = "baseline_example_network.png",
) -> None:
    """Draw final network colored by node state."""
    G = env.G
    state = obs["state"]
    pos = nx.spring_layout(G, seed=0)

    node_colors = [STATE_COLORS[int(s)] for s in state]

    plt.figure(figsize=(6, 6))
    nx.draw_networkx(
        G,
        pos=pos,
        node_color=node_colors,
        with_labels=False,
        node_size=70,
        edge_color="lightgray",
    )
    plt.title("Final node states (S/E/I/T/R)")
    plt.axis("off")
    plt.tight_layout()
    plt.savefig(out_path, bbox_inches="tight")
    plt.close()
    print(f"Saved {out_path}")


def plot_example_timeline(
    env: TemporalCascadeEnv,
    out_path: str = "baseline_example_timeline.png",
) -> None:
    """Timeline of state counts for the last episode of env."""
    history = env.history
    t = history["t"]

    plt.figure(figsize=(7, 4))
    plt.plot(t, history["susceptible"], label="S")
    plt.plot(t, history["exposed"],     label="E")
    plt.plot(t, history["infectious"],  label="I")
    plt.plot(t, history["treated"],     label="T")
    plt.plot(t, history["recovered"],   label="R")
    plt.xlabel("Time step")
    plt.ylabel("# nodes")
    plt.legend()
    plt.title("Cascade over time")
    plt.tight_layout()
    plt.savefig(out_path, bbox_inches="tight")
    plt.close()
    print(f"Saved {out_path}")


# ---------------------------------------------------------------------
# Demo / main
# ---------------------------------------------------------------------

def main():
    # Environment configuration for baseline comparison
    env_kwargs = dict(
        n_nodes=200,
        graph_type="ba",
        graph_kwargs={"m": 3},
        infection_prob=0.15,
        initial_infected=3,
        max_steps=20,
        intervention_budget=10,
        cost_per_treatment=0.1,
        cascade_model="ic",
        regen_graph_each_reset=True,
    )

    n_runs = 50
    print(f"Running {n_runs} runs per baseline...\n")
    results = evaluate_all_baselines(env_kwargs, n_runs=n_runs, base_seed=0)

    # Print a compact table
    print("\nSummary (final infections):")
    print("Baseline         | mean   ± std   | min   | max")
    print("-----------------|---------------|-------|-----")
    for name, res in results.items():
        print(
            f"{name:15s} | "
            f"{res.mean:5.1f} ± {res.std:5.1f} | "
            f"{res.min:5.1f} | {res.max:5.1f}"
        )

    # Plot bar chart
    plot_baseline_bar(results, out_path="baseline_comparison.png")

    # Also run one example episode with the best-looking baseline (smallest mean)
    best_name = min(results.keys(), key=lambda n: results[n].mean)
    print(f"\nBest baseline according to mean infections: {best_name}")

    # For the example run, reuse the same env_kwargs, new seed
    factory_map = {
        "No Interventions": make_no_intervention_policy,
        "Random":           make_random_policy,
        "Degree":           make_degree_policy,
        "PageRank":         make_pagerank_policy,
        "Betweenness":      make_betweenness_policy,
        "Closeness":        make_closeness_policy,
    }
    example_factory = factory_map[best_name]
    total_infected, env, final_obs = run_single_episode(
        example_factory,
        env_kwargs,
        seed=12345,
    )
    print(f"Example run for {best_name}: total_infected = {total_infected}")

    # Visualize final network + timeline for this example
    plot_example_network(env, final_obs, out_path=f"{best_name.lower()}_example_network.png")
    plot_example_timeline(env, out_path=f"{best_name.lower()}_example_timeline.png")


if __name__ == "__main__":
    main()
