"""Service layer – wraps existing src/ modules for the API.

Keeps route handlers thin; all domain logic lives here.
"""

from __future__ import annotations

import json
from dataclasses import asdict
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import networkx as nx
import numpy as np

from src.intercept_baselines import (
    BaselineResults,
    BaselineStrategy,
    BetweennessBaseline,
    ClosenessBaseline,
    DegreeBaseline,
    KShellBaseline,
    PageRankBaseline,
    RandomBaseline,
    evaluate_baseline_strategy,
)
from src.intercept_env import IndependentCascadeConfig, IndependentCascadeEnv, NodeState
from src.network_datasets import (
    create_barabasi_albert,
    create_erdos_renyi,
    create_watts_strogatz,
)

from apps.api.schemas import (
    BaselineCompareRequest,
    BaselineCompareResponse,
    GraphEdge,
    NetworkType,
    NodePosition,
    SimulationRequest,
    SimulationResponse,
    StepSnapshot,
    StrategyName,
    StrategyResult,
)

REPO_ROOT = Path(__file__).resolve().parent.parent.parent


# ── Graph helpers ─────────────────────────────────────────────────────

def _make_graph(net_type: NetworkType, n: int, seed: int | None = None) -> nx.Graph:
    builders = {
        NetworkType.BA: lambda: create_barabasi_albert(n, 3, seed=seed),
        NetworkType.ER: lambda: create_erdos_renyi(n, max(0.03, 6.0 / n), seed=seed),
        NetworkType.WS: lambda: create_watts_strogatz(n, 6, 0.1, seed=seed),
    }
    return builders[net_type]()


def _graph_layout(G: nx.Graph) -> Dict[int, Tuple[float, float]]:
    return nx.spring_layout(G, seed=42, iterations=50)


def _strategy_for(name: StrategyName, G: nx.Graph) -> BaselineStrategy | None:
    mapping: dict[StrategyName, type[BaselineStrategy]] = {
        StrategyName.RANDOM: RandomBaseline,
        StrategyName.DEGREE: DegreeBaseline,
        StrategyName.PAGERANK: PageRankBaseline,
        StrategyName.BETWEENNESS: BetweennessBaseline,
        StrategyName.CLOSENESS: ClosenessBaseline,
        StrategyName.KSHELL: KShellBaseline,
    }
    cls = mapping.get(name)
    if cls is None:
        return None
    return cls(G)


# ── Artifact loading ──────────────────────────────────────────────────

def _load_json(rel_path: str) -> Dict[str, Any] | None:
    path = REPO_ROOT / rel_path
    if not path.exists():
        return None
    with path.open() as f:
        return json.load(f)


def load_evaluation_artifact() -> Dict[str, Any] | None:
    return _load_json("results/evaluation/comparison_results.json")


def load_multi_network_artifact() -> Dict[str, Any] | None:
    return _load_json("results/multi_network/all_results.json")


def load_email_eu_artifact() -> Dict[str, Any] | None:
    return _load_json("results/email_eu_eval/email_eu_results.json")


# ── Simulation ────────────────────────────────────────────────────────

def run_simulation(req: SimulationRequest) -> SimulationResponse:
    G = _make_graph(req.network_type, req.n_nodes, seed=req.seed)
    layout = _graph_layout(G)

    env_cfg = IndependentCascadeConfig(
        infection_prob=req.infection_prob,
        initial_infected_count=req.initial_infected,
        intervention_budget=req.budget,
        max_steps=req.max_steps,
        seed=req.seed,
    )
    env = IndependentCascadeEnv(G, env_cfg)
    state = env.reset()

    strategy = _strategy_for(req.strategy, G)
    intervention_nodes: List[int] = []
    if strategy is not None:
        intervention_nodes = strategy.select_nodes(req.budget)

    nodes = [NodePosition(x=float(layout[i][0]), y=float(layout[i][1]))
             for i in range(G.number_of_nodes())]
    edges = [GraphEdge(source=int(u), target=int(v)) for u, v in G.edges()]

    timeline: List[StepSnapshot] = []
    timeline.append(StepSnapshot(
        t=0,
        node_states=env.node_state.tolist(),
        n_infected=int(state["n_infected"]),
        n_protected=int(state["n_protected"]),
    ))

    intervention_idx = 0
    done = False
    step = 0
    max_sim_steps = req.max_steps * 2

    while not done and step < max_sim_steps:
        action_node: int | None = None
        action_delay: int | None = None

        if intervention_idx < len(intervention_nodes) and env.remaining_budget > 0:
            nid = intervention_nodes[intervention_idx]
            if state["node_mask"][nid] > 0:
                delay = 0 if req.force_zero_delay else 0
                action = {"node_id": nid, "delay": delay}
                action_node = nid
                action_delay = delay
                intervention_idx += 1
            else:
                action = {"node_id": -1, "delay": 0}
                intervention_idx += 1
        else:
            action = {"node_id": -1, "delay": 0}

        state, _, done, info = env.step(action)
        step += 1

        timeline.append(StepSnapshot(
            t=step,
            node_states=env.node_state.tolist(),
            n_infected=int(state["n_infected"]),
            n_protected=int(state["n_protected"]),
            action_node=action_node,
            action_delay=action_delay,
        ))

    net_info = (
        f"{req.network_type.value.upper()}({req.n_nodes}) · "
        f"{G.number_of_edges()} edges · p={req.infection_prob}"
    )

    return SimulationResponse(
        nodes=nodes,
        edges=edges,
        timeline=timeline,
        strategy=req.strategy.value,
        network_info=net_info,
    )


# ── Baseline comparison ──────────────────────────────────────────────

def run_baseline_comparison(req: BaselineCompareRequest) -> BaselineCompareResponse:
    G = _make_graph(req.network_type, req.n_nodes, seed=req.seed)

    env_cfg = IndependentCascadeConfig(
        infection_prob=req.infection_prob,
        initial_infected_count=req.initial_infected,
        intervention_budget=req.budget,
        max_steps=req.max_steps,
        seed=None,
    )

    strategies: list[BaselineStrategy] = [
        RandomBaseline(G),
        DegreeBaseline(G),
        PageRankBaseline(G),
        BetweennessBaseline(G),
        ClosenessBaseline(G),
        KShellBaseline(G),
    ]

    results: Dict[str, StrategyResult] = {}
    for strat in strategies:
        br: BaselineResults = evaluate_baseline_strategy(
            G, strat, env_cfg, n_trials=req.n_trials, verbose=False
        )
        results[br.strategy_name] = StrategyResult(
            mean_infected=br.mean_infected,
            std_infected=br.std_infected,
            median_infected=br.median_infected,
            min_infected=br.min_infected,
            max_infected=br.max_infected,
            mean_protected=br.mean_protected,
        )

    net_info = (
        f"{req.network_type.value.upper()}({req.n_nodes}) · "
        f"{G.number_of_edges()} edges · {req.n_trials} trials"
    )

    return BaselineCompareResponse(results=results, network_info=net_info)
