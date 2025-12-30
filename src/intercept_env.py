"""
Independent Cascade Environment for Network Intervention.

This module implements a discrete-time Independent Cascade (IC) model with
intervention capabilities for reinforcement learning experiments on network
cascade control.

The environment models disease/information spread on a network where:
- Infected nodes attempt to infect susceptible neighbors each timestep
- An agent can protect nodes by scheduling interventions
- The goal is to minimize total infections given a limited intervention budget

Example:
    >>> import networkx as nx
    >>> from src.intercept_env import IndependentCascadeEnv, IndependentCascadeConfig
    >>> 
    >>> graph = nx.barabasi_albert_graph(100, 3)
    >>> config = IndependentCascadeConfig(infection_prob=0.05, intervention_budget=10)
    >>> env = IndependentCascadeEnv(graph, config)
    >>> 
    >>> state = env.reset()
    >>> action = {"node_id": 5, "delay": 0}
    >>> next_state, reward, done, info = env.step(action)

References:
    - Kempe, D., Kleinberg, J., & Tardos, É. (2003). Maximizing the spread of 
      influence through a social network. KDD.
"""

from __future__ import annotations

from dataclasses import dataclass
from enum import IntEnum
from typing import Any, Dict, List, Mapping, Optional, Tuple

import networkx as nx
import numpy as np


class NodeState(IntEnum):
    """Discrete states a node can be in during cascade simulation.
    
    Attributes:
        SUSCEPTIBLE: Node has not been infected and can be infected
        INFECTED: Node is currently infected and can spread infection
        PROTECTED: Node has been vaccinated/protected and cannot be infected
    """
    SUSCEPTIBLE = 0
    INFECTED = 1
    PROTECTED = 2


@dataclass
class IndependentCascadeConfig:
    """Configuration parameters for the Independent Cascade environment.
    
    Attributes:
        infection_prob: Probability of transmission along each edge per timestep.
            Typical values range from 0.01 to 0.15 depending on network density.
        initial_infected_count: Number of seed infections at episode start.
            These are selected uniformly at random.
        intervention_budget: Maximum number of nodes that can be protected.
            This is the agent's resource constraint.
        max_steps: Maximum episode length. Episode also ends when no infected
            nodes remain or budget is exhausted.
        intervention_cost: Per-intervention cost subtracted from reward.
            Used to penalize excessive intervention.
        delay_penalty: Per-timestep penalty for delayed interventions.
            Encourages timely action when appropriate.
        seed: Random seed for reproducibility. None for random initialization.
    """
    infection_prob: float = 0.05
    initial_infected_count: int = 3
    intervention_budget: int = 10
    max_steps: int = 40
    intervention_cost: float = 0.1
    delay_penalty: float = 0.01
    seed: Optional[int] = None


class IndependentCascadeEnv:
    """
    Independent Cascade environment with intervention actions.
    
    This environment simulates disease/information spread on a network graph
    using the Independent Cascade model. An RL agent can intervene by protecting
    nodes, either immediately or with a scheduled delay.
    
    Action Format:
        Actions are dictionaries with two keys:
        - "node_id" (int): Index of node to protect, in range [0, num_nodes).
          Use -1 or invalid index for no-op (let cascade evolve without intervention).
        - "delay" (int): Number of timesteps before intervention takes effect.
          0 means immediate protection.
    
    Observation Format:
        Observations are dictionaries containing:
        - "node_features" (np.ndarray): Shape (N, F) where F=5 by default.
          Features are [one_hot_state(3), normalized_degree(1), timestep(1)].
        - "adj_matrix" (np.ndarray): Shape (N, N), the adjacency matrix.
        - "node_mask" (np.ndarray): Shape (N,), 1.0 for nodes valid for intervention.
        - "t" (int): Current timestep.
        - "n_infected" (int): Number of currently infected nodes.
        - "n_protected" (int): Number of protected nodes.
    
    Reward Structure:
        - -1.0 per new infection
        - -intervention_cost per intervention scheduled
        - -delay_penalty * delay for delayed interventions
    
    Attributes:
        graph: The NetworkX graph representing the network.
        config: Environment configuration parameters.
        num_nodes: Number of nodes in the graph.
        current_step: Current timestep in the episode.
        remaining_budget: Number of interventions still available.
        node_state: Array of NodeState values for each node.
    
    Example:
        >>> env = IndependentCascadeEnv(graph, config)
        >>> state = env.reset()
        >>> while True:
        ...     action = {"node_id": policy(state), "delay": 0}
        ...     state, reward, done, info = env.step(action)
        ...     if done:
        ...         break
    """

    def __init__(self, graph: nx.Graph, config: IndependentCascadeConfig) -> None:
        """Initialize the Independent Cascade environment.
        
        Args:
            graph: A NetworkX Graph object representing the network topology.
                Must contain at least one node.
            config: Configuration dataclass with environment parameters.
        
        Raises:
            ValueError: If the graph contains no nodes.
        """
        if graph.number_of_nodes() == 0:
            raise ValueError("Graph must contain at least one node.")

        self.graph = graph
        self.config = config

        self.num_nodes: int = self.graph.number_of_nodes()
        self._adj_matrix: np.ndarray = nx.to_numpy_array(self.graph, dtype=np.float32)

        self.rng = np.random.default_rng(config.seed)

        self.current_step: int = 0
        self.remaining_budget: int = 0
        self.node_state: np.ndarray  # (num_nodes,)
        self._scheduled_interventions: Dict[int, List[int]]

        self._degree_cache: np.ndarray = self._compute_normalized_degree()

    def reset(self) -> Dict[str, Any]:
        """Reset the environment to initial state.
        
        Initializes a new episode with random seed infections and full
        intervention budget.
        
        Returns:
            Initial observation dictionary.
        """
        self.current_step = 0
        self.remaining_budget = self.config.intervention_budget

        self.node_state = np.full(self.num_nodes, NodeState.SUSCEPTIBLE, dtype=np.int8)

        initial_infected_count = min(self.config.initial_infected_count, self.num_nodes)
        if initial_infected_count > 0:
            initially_infected = self.rng.choice(
                self.num_nodes, size=initial_infected_count, replace=False
            )
            self.node_state[initially_infected] = NodeState.INFECTED

        self._scheduled_interventions = {}

        return self._build_observation()

    def step(
        self,
        action: Mapping[str, int],
    ) -> Tuple[Dict[str, Any], float, bool, Dict[str, Any]]:
        """Execute one environment step.
        
        Processes the agent's action, applies scheduled interventions,
        propagates infections, and returns the new state.
        
        Args:
            action: Dictionary with "node_id" (int) and "delay" (int).
                node_id should be in [0, num_nodes) for valid intervention,
                or -1/invalid for no-op.
        
        Returns:
            Tuple of (observation, reward, done, info):
            - observation: New state observation dictionary
            - reward: Scalar reward (negative = bad)
            - done: Whether episode has ended
            - info: Additional information dict with keys:
                - "new_infections": Number of new infections this step
                - "remaining_budget": Remaining intervention budget
        """
        reward = 0.0
        info: Dict[str, Any] = {}

        node_id = int(action.get("node_id", -1))
        delay = int(action.get("delay", 0))
        delay_clipped = max(delay, 0)

        # Schedule intervention if valid
        if self._can_schedule_intervention(node_id):
            intervention_step = self.current_step + delay_clipped
            self._scheduled_interventions.setdefault(intervention_step, []).append(node_id)
            self.remaining_budget -= 1
            reward -= self.config.intervention_cost
            reward -= self.config.delay_penalty * float(delay_clipped)

        # Apply any interventions scheduled for this timestep
        self._apply_scheduled_interventions_for_current_step()

        # Propagate infections
        new_infections = self._propagate_infections_one_step()
        reward -= float(new_infections)

        self.current_step += 1
        done = self._is_terminal()

        obs = self._build_observation()
        info["new_infections"] = new_infections
        info["remaining_budget"] = self.remaining_budget

        return obs, reward, done, info

    def _compute_normalized_degree(self) -> np.ndarray:
        """Compute z-score normalized node degrees.
        
        Returns:
            Array of normalized degrees, shape (num_nodes,).
        """
        degrees = np.array(
            [self.graph.degree(node_idx) for node_idx in range(self.num_nodes)],
            dtype=np.float32,
        )
        mean = degrees.mean()
        std = degrees.std()
        if std < 1e-6:
            return np.zeros_like(degrees, dtype=np.float32)
        return (degrees - mean) / (std + 1e-6)

    def _can_schedule_intervention(self, node_id: int) -> bool:
        """Check if an intervention can be scheduled for a node.
        
        Args:
            node_id: Index of node to check.
        
        Returns:
            True if intervention is valid and can be scheduled.
        """
        if not (0 <= node_id < self.num_nodes):
            return False
        if self.remaining_budget <= 0:
            return False
        if self.node_state[node_id] == NodeState.PROTECTED:
            return False
        return True

    def _apply_scheduled_interventions_for_current_step(self) -> None:
        """Apply all interventions scheduled for the current timestep."""
        nodes_to_protect = self._scheduled_interventions.pop(self.current_step, [])
        for node_id in nodes_to_protect:
            self.node_state[node_id] = NodeState.PROTECTED

    def _propagate_infections_one_step(self) -> int:
        """Execute one round of infection propagation.
        
        Each infected node attempts to infect each susceptible neighbor
        with probability infection_prob.
        
        Returns:
            Number of newly infected nodes.
        """
        newly_infected: List[int] = []

        infected_indices = np.where(self.node_state == NodeState.INFECTED)[0]
        for u in infected_indices:
            for v in self.graph.neighbors(u):
                if self.node_state[v] != NodeState.SUSCEPTIBLE:
                    continue
                if self.rng.random() < self.config.infection_prob:
                    newly_infected.append(v)

        if not newly_infected:
            return 0

        newly_infected = list(set(newly_infected))
        for v in newly_infected:
            if self.node_state[v] == NodeState.SUSCEPTIBLE:
                self.node_state[v] = NodeState.INFECTED

        return len(newly_infected)

    def _is_terminal(self) -> bool:
        """Check if the episode should end.
        
        Returns:
            True if no infected nodes remain, max steps reached,
            or budget exhausted.
        """
        no_infected_remaining = not np.any(self.node_state == NodeState.INFECTED)
        max_steps_reached = self.current_step >= self.config.max_steps
        no_budget_remaining = self.remaining_budget <= 0
        return no_infected_remaining or max_steps_reached or no_budget_remaining

    def _build_observation(self) -> Dict[str, Any]:
        """Construct the observation dictionary.
        
        Returns:
            Observation dict with node_features, adj_matrix, node_mask,
            timestep, and infection counts.
        """
        node_features = self._build_node_features()
        node_mask = self._build_node_mask()
        num_infected = int(np.sum(self.node_state == NodeState.INFECTED))
        num_protected = int(np.sum(self.node_state == NodeState.PROTECTED))

        return {
            "node_features": node_features,
            "adj_matrix": self._adj_matrix,
            "node_mask": node_mask,
            "t": self.current_step,
            "n_infected": num_infected,
            "n_protected": num_protected,
        }

    def _build_node_features(self) -> np.ndarray:
        """Build node feature matrix.
        
        Features per node (F=5):
        - One-hot encoding of state (3 dims)
        - Normalized degree (1 dim)
        - Current timestep (1 dim)
        
        Returns:
            Feature matrix of shape (num_nodes, 5).
        """
        one_hot = np.zeros((self.num_nodes, 3), dtype=np.float32)
        one_hot[np.arange(self.num_nodes), self.node_state] = 1.0

        time_feature = np.full(
            (self.num_nodes, 1), float(self.current_step), dtype=np.float32
        )

        features = np.concatenate(
            [one_hot, self._degree_cache[:, None], time_feature],
            axis=1,
        )
        return features

    def _build_node_mask(self) -> np.ndarray:
        """Build mask indicating which nodes can be intervened on.
        
        Returns:
            Binary mask of shape (num_nodes,), 1.0 for valid nodes.
        """
        mask = np.ones(self.num_nodes, dtype=np.float32)
        mask[self.node_state == NodeState.PROTECTED] = 0.0
        return mask


if __name__ == "__main__":
    print("=" * 60)
    print("IndependentCascadeEnv smoke test")
    print("=" * 60)

    G = nx.barabasi_albert_graph(50, 3)
    cfg = IndependentCascadeConfig(
        infection_prob=0.05,
        initial_infected_count=3,
        intervention_budget=5,
        max_steps=40,
        intervention_cost=0.1,
        delay_penalty=0.01,
        seed=0,
    )
    env = IndependentCascadeEnv(G, cfg)

    state = env.reset()
    print(
        f"reset: t={state['t']}, "
        f"infected={state['n_infected']}, "
        f"protected={state['n_protected']}"
    )
    print("node_features:", state["node_features"].shape)
    print("adj_matrix:", state["adj_matrix"].shape)

    done = False
    step = 0
    while not done and step < 10:
        step += 1
        action = {
            "node_id": int(np.random.randint(0, env.num_nodes)),
            "delay": int(np.random.randint(0, 3)),
        }
        state, reward, done, info = env.step(action)
        print(
            f"step={step:02d} "
            f"t={state['t']:02d} "
            f"infected={state['n_infected']:02d} "
            f"protected={state['n_protected']:02d} "
            f"new_inf={info['new_infections']:02d} "
            f"budget={info['remaining_budget']:02d} "
            f"reward={reward: .3f}"
        )

    print("\nSmoke test complete.")
