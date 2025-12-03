"""
environment.py - Enhanced Temporal Information Diffusion Environment

A realistic simulation environment for controlling information diffusion in networks.
Simulates disease/misinformation spread with temporal intervention capabilities.

Key Features:
- Realistic cascade dynamics (IC and LT models)
- Temporal interventions: (node, delay_time) pairs
- Detailed infection tracking with exposure history
- Rich state representation for GNN policies
- Intervention effectiveness varies by timing
- Multiple infection stages and recovery dynamics

States:
    0 = SUSCEPTIBLE (healthy, not exposed)
    1 = EXPOSED (infected but not yet infectious)
    2 = INFECTIOUS (actively spreading)
    3 = TREATED (intervention applied, recovering)
    4 = RECOVERED (immune/removed)

Action Space:
    List of (node_id, delay_timesteps) tuples
    - node_id: which node to treat
    - delay_timesteps: when to apply treatment (0 = immediate, 1+ = delayed)
"""

from __future__ import annotations

import random
from typing import Dict, List, Optional, Iterable, Tuple
from dataclasses import dataclass

import numpy as np
import networkx as nx


@dataclass
class InfectionEvent:
    """Records when and how a node got infected."""
    node_id: int
    timestep: int
    source_node: Optional[int] = None
    exposure_count: int = 0


@dataclass
class InterventionEvent:
    """Scheduled intervention to be applied at a future timestep."""
    node_id: int
    scheduled_time: int
    action_time: int
    effectiveness: float = 1.0


class TemporalCascadeEnv:
    """
    Enhanced cascade environment with temporal dynamics and delayed interventions.
    """
    
    # State constants
    SUSCEPTIBLE = 0
    EXPOSED = 1
    INFECTIOUS = 2
    TREATED = 3
    RECOVERED = 4
    
    def __init__(
        self,
        n_nodes: int = 200,
        graph_type: str = "ba",
        graph_kwargs: Optional[Dict] = None,
        infection_prob: float = 0.15,
        initial_infected: int = 3,
        max_steps: int = 30,
        intervention_budget: int = 5,
        cost_per_treatment: float = 0.1,
        cascade_model: str = "ic",
        # New temporal parameters
        exposure_period: int = 2,  # timesteps before becoming infectious
        infectious_period: int = 3,  # timesteps of being infectious
        treatment_delay_penalty: float = 0.1,  # effectiveness decreases per timestep delay
        early_intervention_bonus: float = 0.5,  # bonus for treating exposed nodes
        max_intervention_delay: int = 5,  # maximum delay allowed
        # Network dynamics
        regen_graph_each_reset: bool = True,
        seed: Optional[int] = None,
        # Optional externally provided graph
        graph: Optional[nx.Graph] = None,
    ):
        self.n_nodes = n_nodes
        self.graph_type = graph_type
        self.graph_kwargs = graph_kwargs or {}
        self.infection_prob = infection_prob
        self.initial_infected = initial_infected
        self.max_steps = max_steps
        self.intervention_budget = intervention_budget
        self.cost_per_treatment = cost_per_treatment
        self.cascade_model = cascade_model
        self.regen_graph_each_reset = regen_graph_each_reset
        
        # Temporal dynamics parameters
        self.exposure_period = exposure_period
        self.infectious_period = infectious_period
        self.treatment_delay_penalty = treatment_delay_penalty
        self.early_intervention_bonus = early_intervention_bonus
        self.max_intervention_delay = max_intervention_delay
        
        self.rng = np.random.default_rng(seed)
        if seed is not None:
            random.seed(seed)
        
        # If a graph is provided, use it; otherwise build one
        self._external_graph = graph is not None
        if graph is not None:
            self.G: nx.Graph = graph.copy()
            self.n_nodes = self.G.number_of_nodes()
        else:
            self.G: nx.Graph = self._build_graph()
        
        # Core state
        self.state: np.ndarray = np.zeros(self.n_nodes, dtype=np.int8)
        self.t: int = 0
        self.done: bool = False
        
        # Temporal tracking
        self.infection_times: Dict[int, int] = {}  # node -> timestep infected
        self.exposure_times: Dict[int, int] = {}   # node -> timestep exposed
        self.treatment_times: Dict[int, int] = {}  # node -> timestep treated
        self.exposure_counts: np.ndarray = np.zeros(self.n_nodes, dtype=np.int32)
        
        # Intervention queue: scheduled interventions
        self.intervention_queue: List[InterventionEvent] = []
        
        # History tracking
        self.infection_events: List[InfectionEvent] = []
        self.intervention_events: List[InterventionEvent] = []
        
        # Episode statistics
        self.history: Dict[str, List] = {}
        
        # For LT model
        self.lt_thresholds: Optional[np.ndarray] = None
        
        self.reset()
    
    # ---------- Graph Construction ----------
    
    def _build_graph(self) -> nx.Graph:
        """Build realistic network topology."""
        if self.graph_type == "ba":
            m = self.graph_kwargs.get("m", 3)
            G = nx.barabasi_albert_graph(self.n_nodes, m, seed=int(self.rng.integers(1e9)))
        elif self.graph_type == "er":
            p = self.graph_kwargs.get("p", 0.05)
            G = nx.erdos_renyi_graph(self.n_nodes, p, seed=int(self.rng.integers(1e9)))
        elif self.graph_type == "ws":
            k = self.graph_kwargs.get("k", 4)
            p = self.graph_kwargs.get("p", 0.3)
            G = nx.watts_strogatz_graph(self.n_nodes, k, p, seed=int(self.rng.integers(1e9)))
        else:
            raise ValueError(f"Unsupported graph_type: {self.graph_type}")
        
        # Ensure connectivity
        if not nx.is_connected(G):
            largest_cc = max(nx.connected_components(G), key=len)
            G = G.subgraph(largest_cc).copy()
            mapping = {node: i for i, node in enumerate(G.nodes())}
            G = nx.relabel_nodes(G, mapping)
            self.n_nodes = G.number_of_nodes()
        
        return G
    
    def _sample_initial_infected(self) -> List[int]:
        """Sample initial infected nodes (prefer high-degree nodes for realism)."""
        if self.rng.random() < 0.7:  # 70% chance to prefer high-degree nodes
            degrees = dict(self.G.degree())
            nodes = list(degrees.keys())
            weights = np.array([degrees[n] for n in nodes]) + 1  # avoid zero weights
            probs = weights / weights.sum()
            selected = self.rng.choice(
                nodes,
                size=min(self.initial_infected, self.n_nodes),
                replace=False,
                p=probs
            )
            return selected.tolist()
        else:
            return self.rng.choice(
                self.n_nodes,
                size=min(self.initial_infected, self.n_nodes),
                replace=False
            ).tolist()
    
    # ---------- Public API ----------
    
    def reset(self) -> Dict:
        """Reset environment for new episode."""
        if self.regen_graph_each_reset and not self._external_graph:
            self.G = self._build_graph()
        
        # Reset state
        self.state = np.zeros(self.n_nodes, dtype=np.int8)
        self.t = 0
        self.done = False
        
        # Reset tracking
        self.infection_times = {}
        self.exposure_times = {}
        self.treatment_times = {}
        self.exposure_counts = np.zeros(self.n_nodes, dtype=np.int32)
        self.intervention_queue = []
        self.infection_events = []
        self.intervention_events = []
        
        # LT model thresholds
        if self.cascade_model == "lt":
            self.lt_thresholds = self.rng.uniform(0.0, 1.0, size=self.n_nodes)
        
        # Initialize infections (start as EXPOSED)
        seeds = self._sample_initial_infected()
        for node in seeds:
            self.state[node] = self.EXPOSED
            self.exposure_times[node] = 0
            self.infection_events.append(
                InfectionEvent(node_id=node, timestep=0, source_node=None)
            )
        
        # Initialize history
        self.history = {
            "t": [0],
            "susceptible": [int((self.state == self.SUSCEPTIBLE).sum())],
            "exposed": [int((self.state == self.EXPOSED).sum())],
            "infectious": [int((self.state == self.INFECTIOUS).sum())],
            "treated": [int((self.state == self.TREATED).sum())],
            "recovered": [int((self.state == self.RECOVERED).sum())],
            "new_infections": [len(seeds)],
            "new_treatments": [0],
            "pending_interventions": [0],
            "actions": [[]],
        }
        
        return self._get_observation()
    
    def step(self, action: Optional[Iterable[Tuple[int, int]]] = None) -> Tuple[Dict, float, bool, Dict]:
        """
        Advance cascade by one timestep with temporal interventions.
        
        Parameters
        ----------
        action : List of (node_id, delay) tuples
            - node_id: which node to intervene on
            - delay: timesteps to wait before intervention (0 = immediate)
        
        Returns
        -------
        obs, reward, done, info
        """
        if self.done:
            raise RuntimeError("Episode finished. Call reset().")
        
        # Parse and validate actions
        action = list(action) if action is not None else []
        scheduled_now = []
        
        for item in action:
            if isinstance(item, (list, tuple)) and len(item) == 2:
                node_id, delay = item
            else:
                node_id = item
                delay = 0
            
            # Validate
            if not (0 <= node_id < self.n_nodes):
                continue
            delay = max(0, min(delay, self.max_intervention_delay))
            
            # Can only intervene on susceptible or exposed nodes
            if self.state[node_id] in [self.SUSCEPTIBLE, self.EXPOSED]:
                scheduled_now.append((node_id, delay))
        
        # Apply intervention budget
        if len(scheduled_now) > self.intervention_budget:
            scheduled_now = scheduled_now[:self.intervention_budget]
        
        # Schedule interventions
        new_interventions = []
        for node_id, delay in scheduled_now:
            # Calculate effectiveness based on delay
            effectiveness = 1.0 - (delay * self.treatment_delay_penalty)
            effectiveness = max(0.1, effectiveness)  # minimum 10% effectiveness
            
            # Bonus for early intervention on exposed nodes
            if self.state[node_id] == self.EXPOSED:
                effectiveness += self.early_intervention_bonus
                effectiveness = min(1.0, effectiveness)
            
            intervention = InterventionEvent(
                node_id=node_id,
                scheduled_time=self.t + delay,
                action_time=self.t,
                effectiveness=effectiveness
            )
            self.intervention_queue.append(intervention)
            new_interventions.append(intervention)
        
        # Process due interventions
        applied_interventions = self._apply_due_interventions()
        
        # Progress disease stages
        self._update_disease_progression()
        
        # Cascade dynamics
        new_infections = self._cascade_step()
        
        # Calculate reward
        reward = self._calculate_reward(
            new_infections=new_infections,
            interventions_applied=applied_interventions,
            interventions_scheduled=new_interventions
        )
        
        # Update tracking
        self.t += 1
        self._update_history(new_infections, len(applied_interventions))
        
        # Check termination
        active_infections = (
            (self.state == self.EXPOSED).sum() + 
            (self.state == self.INFECTIOUS).sum()
        )
        if self.t >= self.max_steps or active_infections == 0:
            self.done = True
        
        obs = self._get_observation()
        info = {
            "new_infections": new_infections,
            "interventions_applied": len(applied_interventions),
            "interventions_scheduled": len(new_interventions),
            "pending_interventions": len(self.intervention_queue),
            "active_infections": int(active_infections),
            "state_counts": {
                "susceptible": int((self.state == self.SUSCEPTIBLE).sum()),
                "exposed": int((self.state == self.EXPOSED).sum()),
                "infectious": int((self.state == self.INFECTIOUS).sum()),
                "treated": int((self.state == self.TREATED).sum()),
                "recovered": int((self.state == self.RECOVERED).sum()),
            }
        }
        
        return obs, reward, self.done, info
    
    # ---------- Disease Dynamics ----------
    
    def _update_disease_progression(self):
        """Progress nodes through disease stages."""
        # EXPOSED -> INFECTIOUS after exposure_period
        for node in range(self.n_nodes):
            if self.state[node] == self.EXPOSED:
                if node in self.exposure_times:
                    if self.t - self.exposure_times[node] >= self.exposure_period:
                        self.state[node] = self.INFECTIOUS
                        self.infection_times[node] = self.t
        
        # INFECTIOUS -> RECOVERED after infectious_period
        for node in range(self.n_nodes):
            if self.state[node] == self.INFECTIOUS:
                if node in self.infection_times:
                    if self.t - self.infection_times[node] >= self.infectious_period:
                        self.state[node] = self.RECOVERED
        
        # TREATED -> RECOVERED (treatment accelerates recovery)
        for node in range(self.n_nodes):
            if self.state[node] == self.TREATED:
                if node in self.treatment_times:
                    # Faster recovery when treated
                    treatment_duration = max(1, self.infectious_period // 2)
                    if self.t - self.treatment_times[node] >= treatment_duration:
                        self.state[node] = self.RECOVERED
    
    def _cascade_step(self) -> int:
        """Execute one step of cascade dynamics."""
        if self.cascade_model == "ic":
            return self._independent_cascade_step()
        elif self.cascade_model == "lt":
            return self._linear_threshold_step()
        else:
            raise ValueError(f"Unknown cascade model: {self.cascade_model}")
    
    def _independent_cascade_step(self) -> int:
        """Independent Cascade: each infectious node tries to infect neighbors."""
        newly_exposed = []
        
        infectious_nodes = np.where(self.state == self.INFECTIOUS)[0]
        
        for u in infectious_nodes:
            for v in self.G.neighbors(u):
                if self.state[v] == self.SUSCEPTIBLE:
                    # Track exposure attempts
                    self.exposure_counts[v] += 1
                    
                    # Infection probability (can be heterogeneous)
                    if self.rng.random() < self.infection_prob:
                        newly_exposed.append(v)
                        self.infection_events.append(
                            InfectionEvent(
                                node_id=v,
                                timestep=self.t,
                                source_node=u,
                                exposure_count=int(self.exposure_counts[v])
                            )
                        )
        
        # Apply new exposures
        newly_exposed = list(set(newly_exposed))
        for node in newly_exposed:
            self.state[node] = self.EXPOSED
            self.exposure_times[node] = self.t
        
        return len(newly_exposed)
    
    def _linear_threshold_step(self) -> int:
        """Linear Threshold: node infected when fraction of infected neighbors exceeds threshold."""
        if self.lt_thresholds is None:
            self.lt_thresholds = self.rng.uniform(0.0, 1.0, size=self.n_nodes)
        
        newly_exposed = []
        susceptible_nodes = np.where(self.state == self.SUSCEPTIBLE)[0]
        infectious_set = set(np.where(
            (self.state == self.INFECTIOUS) | (self.state == self.EXPOSED)
        )[0])
        
        for v in susceptible_nodes:
            neighbors = list(self.G.neighbors(v))
            if not neighbors:
                continue
            
            infected_neighbors = sum(1 for u in neighbors if u in infectious_set)
            frac_infected = infected_neighbors / len(neighbors)
            
            # Track exposure
            self.exposure_counts[v] += infected_neighbors
            
            if frac_infected >= self.lt_thresholds[v]:
                newly_exposed.append(v)
                source = next((u for u in neighbors if u in infectious_set), None)
                self.infection_events.append(
                    InfectionEvent(
                        node_id=v,
                        timestep=self.t,
                        source_node=source,
                        exposure_count=int(self.exposure_counts[v])
                    )
                )
        
        # Apply new exposures
        for node in newly_exposed:
            self.state[node] = self.EXPOSED
            self.exposure_times[node] = self.t
        
        return len(newly_exposed)
    
    # ---------- Intervention Management ----------
    
    def _apply_due_interventions(self) -> List[InterventionEvent]:
        """Apply interventions scheduled for current timestep."""
        applied = []
        remaining = []
        
        for intervention in self.intervention_queue:
            if intervention.scheduled_time <= self.t:
                node = intervention.node_id
                
                # Can still treat if not yet recovered
                if self.state[node] not in [self.RECOVERED, self.TREATED]:
                    # Apply treatment with effectiveness
                    if self.rng.random() < intervention.effectiveness:
                        self.state[node] = self.TREATED
                        self.treatment_times[node] = self.t
                        applied.append(intervention)
                        self.intervention_events.append(intervention)
                    else:
                        # Treatment failed, but still counts as applied
                        applied.append(intervention)
            else:
                remaining.append(intervention)
        
        self.intervention_queue = remaining
        return applied
    
    def _calculate_reward(
        self,
        new_infections: int,
        interventions_applied: List[InterventionEvent],
        interventions_scheduled: List[InterventionEvent]
    ) -> float:
        """Calculate reward balancing infection control and intervention costs."""
        # Penalty for new infections
        infection_penalty = float(new_infections)
        
        # Cost for interventions (higher cost for delayed interventions)
        intervention_cost = 0.0
        for intervention in interventions_scheduled:
            delay = intervention.scheduled_time - intervention.action_time
            delay_multiplier = 1.0 + (delay * 0.1)  # 10% more expensive per timestep delay
            intervention_cost += self.cost_per_treatment * delay_multiplier
        
        # Bonus for effective early interventions
        early_bonus = 0.0
        for intervention in interventions_applied:
            if intervention.effectiveness > 0.8:
                early_bonus += 0.05
        
        reward = -infection_penalty - intervention_cost + early_bonus
        return reward
    
    # ---------- Observation & History ----------
    
    def _get_observation(self) -> Dict:
        """
        Rich observation for GNN-based policies.
        
        Includes:
        - Temporal information (current time, disease stage durations)
        - Spatial information (graph structure)
        - Intervention information (pending interventions)
        """
        # Node features
        node_features = np.zeros((self.n_nodes, 7), dtype=np.float32)
        
        for i in range(self.n_nodes):
            node_features[i, 0] = self.state[i] / 4.0  # normalized state
            node_features[i, 1] = self.exposure_counts[i] / 10.0  # normalized exposure count
            
            # Time since exposure/infection (normalized)
            if i in self.exposure_times:
                node_features[i, 2] = min((self.t - self.exposure_times[i]) / 10.0, 1.0)
            if i in self.infection_times:
                node_features[i, 3] = min((self.t - self.infection_times[i]) / 10.0, 1.0)
            if i in self.treatment_times:
                node_features[i, 4] = min((self.t - self.treatment_times[i]) / 10.0, 1.0)
            
            # Neighborhood risk
            neighbors = list(self.G.neighbors(i))
            if neighbors:
                infectious_neighbors = sum(
                    1 for n in neighbors 
                    if self.state[n] in [self.EXPOSED, self.INFECTIOUS]
                )
                node_features[i, 5] = infectious_neighbors / len(neighbors)
            
            # Pending intervention flag
            node_features[i, 6] = float(any(
                iv.node_id == i for iv in self.intervention_queue
            ))
        
        n_infected = int(((self.state == self.EXPOSED) | (self.state == self.INFECTIOUS)).sum())
        
        return {
            "t": self.t,
            "state": self.state.copy(),
            "node_features": node_features,
            "graph": self.G,
            "pending_interventions": len(self.intervention_queue),
            "intervention_queue": [
                (iv.node_id, iv.scheduled_time) 
                for iv in self.intervention_queue
            ],
            "n_infected": n_infected,
        }
    
    def _update_history(self, new_infections: int, interventions_applied: int):
        """Update episode history."""
        self.history["t"].append(self.t)
        self.history["susceptible"].append(int((self.state == self.SUSCEPTIBLE).sum()))
        self.history["exposed"].append(int((self.state == self.EXPOSED).sum()))
        self.history["infectious"].append(int((self.state == self.INFECTIOUS).sum()))
        self.history["treated"].append(int((self.state == self.TREATED).sum()))
        self.history["recovered"].append(int((self.state == self.RECOVERED).sum()))
        self.history["new_infections"].append(new_infections)
        self.history["new_treatments"].append(interventions_applied)
        self.history["pending_interventions"].append(len(self.intervention_queue))
    
    def get_episode_stats(self) -> Dict:
        """Get comprehensive episode statistics."""
        return {
            "history": self.history,
            "infection_events": self.infection_events,
            "intervention_events": self.intervention_events,
            "final_state": {
                "total_infected": len(self.infection_events),
                "total_treated": len(self.intervention_events),
                "final_recovered": int((self.state == self.RECOVERED).sum()),
                "cascade_size": len([e for e in self.infection_events if e.source_node is not None]),
            }
        }


def demo_temporal_interventions():
    """Demo showing temporal intervention capabilities."""
    print("=" * 70)
    print("TEMPORAL INTERVENTION DEMO")
    print("=" * 70)
    print("Shows: delayed interventions, disease progression, intervention timing")
    print()
    
    env = TemporalCascadeEnv(
        n_nodes=100,
        graph_type="ba",
        graph_kwargs={"m": 3},
        infection_prob=0.2,
        initial_infected=2,
        max_steps=25,
        intervention_budget=3,
        cost_per_treatment=0.1,
        exposure_period=2,
        infectious_period=3,
        treatment_delay_penalty=0.1,
        early_intervention_bonus=0.3,
        max_intervention_delay=5,
        cascade_model="ic",
        seed=42,
    )
    
    obs = env.reset()
    print(f"Network: {env.G.number_of_nodes()} nodes, {env.G.number_of_edges()} edges")
    print(f"Initial exposed: {(obs['state'] == env.EXPOSED).sum()}")
    print(f"Disease stages: Exposed({env.exposure_period}t) -> Infectious({env.infectious_period}t) -> Recovered")
    print()
    
    done = False
    total_reward = 0.0
    rng = np.random.default_rng(42)
    
    print(f"{'Step':<5} {'Susc':<5} {'Expo':<5} {'Infec':<5} {'Treat':<6} {'Recov':<6} {'NewInf':<7} {'IntAppl':<8} {'Pending':<8} {'Reward':<8}")
    print("-" * 90)
    
    while not done:
        # Strategy: Mix of immediate and delayed interventions on exposed/susceptible nodes
        actions: List[Tuple[int, int]] = []
        
        # Target exposed nodes with immediate intervention (catch early)
        exposed = np.where(obs['state'] == env.EXPOSED)[0]
        if len(exposed) > 0 and env.t % 3 == 0:
            target = int(rng.choice(exposed))
            actions.append((target, 0))  # immediate
        
        # Target high-degree susceptible nodes with delayed intervention (preventive)
        susceptible = np.where(obs['state'] == env.SUSCEPTIBLE)[0]
        if len(susceptible) > 0 and env.t % 2 == 0:
            degrees = [env.G.degree(int(n)) for n in susceptible]
            high_degree_idx = int(np.argmax(degrees))
            target = int(susceptible[high_degree_idx])
            delay = int(rng.integers(1, 4))  # delayed intervention
            actions.append((target, delay))
        
        obs, reward, done, info = env.step(actions)
        total_reward += reward
        
        print(
            f"{env.t:<5} "
            f"{info['state_counts']['susceptible']:<5} "
            f"{info['state_counts']['exposed']:<5} "
            f"{info['state_counts']['infectious']:<5} "
            f"{info['state_counts']['treated']:<6} "
            f"{info['state_counts']['recovered']:<6} "
            f"{info['new_infections']:<7} "
            f"{info['interventions_applied']:<8} "
            f"{info['pending_interventions']:<8} "
            f"{reward:>7.2f}"
        )
    
    print()
    stats = env.get_episode_stats()
    print("Episode Summary:")
    print(f"  Total infections: {stats['final_state']['total_infected']}")
    print(f"  Total interventions: {stats['final_state']['total_treated']}")
    print(f"  Final recovered: {stats['final_state']['final_recovered']}")
    print(f"  Total reward: {total_reward:.2f}")
    print()


def demo_no_intervention_comparison():
    """Demo comparing no intervention vs temporal interventions."""
    print("=" * 70)
    print("COMPARISON: No Intervention vs Temporal Interventions")
    print("=" * 70)
    print()
    
    # Run without intervention
    env1 = TemporalCascadeEnv(
        n_nodes=100,
        graph_type="ba",
        infection_prob=0.18,
        initial_infected=2,
        max_steps=20,
        intervention_budget=0,  # No intervention
        seed=123,
    )
    
    obs1 = env1.reset()
    done1 = False
    while not done1:
        obs1, _, done1, _ = env1.step([])
    
    stats1 = env1.get_episode_stats()
    
    # Run with interventions
    env2 = TemporalCascadeEnv(
        n_nodes=100,
        graph_type="ba",
        infection_prob=0.18,
        initial_infected=2,
        max_steps=20,
        intervention_budget=3,
        seed=123,
    )
    
    obs2 = env2.reset()
    done2 = False
    
    while not done2:
        # Simple strategy: treat exposed nodes immediately
        exposed = np.where(obs2['state'] == env2.EXPOSED)[0]
        actions = [(int(n), 0) for n in exposed[:3]]
        obs2, _, done2, _ = env2.step(actions)
    
    stats2 = env2.get_episode_stats()
    
    print("No Intervention:")
    print(f"  Total infections: {stats1['final_state']['total_infected']}")
    print(f"  Final recovered: {stats1['final_state']['final_recovered']}")
    print()
    
    print("With Temporal Interventions:")
    print(f"  Total infections: {stats2['final_state']['total_infected']}")
    print(f"  Interventions applied: {stats2['final_state']['total_treated']}")
    print(f"  Final recovered: {stats2['final_state']['final_recovered']}")
    prevented = stats1['final_state']['total_infected'] - stats2['final_state']['total_infected']
    print(f"  Infections prevented: {prevented}")
    print()


__all__ = ["TemporalCascadeEnv", "InfectionEvent", "InterventionEvent"]
