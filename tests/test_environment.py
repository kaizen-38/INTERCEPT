"""
Unit tests for the Independent Cascade environment.

Run with: pytest tests/ -v
"""

import pytest
import networkx as nx
import numpy as np

from src.intercept_env import (
    IndependentCascadeEnv,
    IndependentCascadeConfig,
    NodeState,
)


class TestEnvironmentCreation:
    """Tests for environment initialization."""

    def test_create_with_valid_graph(self):
        """Environment should initialize with a valid graph."""
        G = nx.barabasi_albert_graph(100, 3)
        config = IndependentCascadeConfig(infection_prob=0.05)
        env = IndependentCascadeEnv(G, config)
        
        assert env.num_nodes == 100
        assert env.graph == G

    def test_create_with_empty_graph_raises(self):
        """Environment should raise error for empty graph."""
        G = nx.Graph()
        config = IndependentCascadeConfig()
        
        with pytest.raises(ValueError, match="at least one node"):
            IndependentCascadeEnv(G, config)

    def test_adjacency_matrix_shape(self):
        """Adjacency matrix should have correct shape."""
        G = nx.barabasi_albert_graph(50, 3)
        env = IndependentCascadeEnv(G, IndependentCascadeConfig())
        
        assert env._adj_matrix.shape == (50, 50)


class TestReset:
    """Tests for environment reset."""

    def test_reset_returns_observation(self):
        """Reset should return valid observation dict."""
        G = nx.barabasi_albert_graph(50, 3)
        env = IndependentCascadeEnv(G, IndependentCascadeConfig())
        
        state = env.reset()
        
        assert "node_features" in state
        assert "adj_matrix" in state
        assert "node_mask" in state
        assert "t" in state
        assert "n_infected" in state
        assert "n_protected" in state

    def test_reset_initializes_infections(self):
        """Reset should create initial infections."""
        G = nx.barabasi_albert_graph(50, 3)
        config = IndependentCascadeConfig(initial_infected_count=5)
        env = IndependentCascadeEnv(G, config)
        
        state = env.reset()
        
        assert state["n_infected"] == 5
        assert state["n_protected"] == 0
        assert state["t"] == 0

    def test_reset_node_features_shape(self):
        """Node features should have correct shape."""
        G = nx.barabasi_albert_graph(50, 3)
        env = IndependentCascadeEnv(G, IndependentCascadeConfig())
        
        state = env.reset()
        
        assert state["node_features"].shape == (50, 5)  # 3 one-hot + degree + time

    def test_reset_restores_budget(self):
        """Reset should restore full intervention budget."""
        G = nx.barabasi_albert_graph(50, 3)
        config = IndependentCascadeConfig(intervention_budget=10)
        env = IndependentCascadeEnv(G, config)
        
        env.reset()
        
        assert env.remaining_budget == 10


class TestStep:
    """Tests for environment step."""

    def test_step_returns_tuple(self):
        """Step should return (obs, reward, done, info)."""
        G = nx.barabasi_albert_graph(50, 3)
        env = IndependentCascadeEnv(G, IndependentCascadeConfig())
        env.reset()
        
        result = env.step({"node_id": 0, "delay": 0})
        
        assert len(result) == 4
        state, reward, done, info = result
        assert isinstance(state, dict)
        assert isinstance(reward, float)
        assert isinstance(done, bool)
        assert isinstance(info, dict)

    def test_step_increments_time(self):
        """Step should increment timestep."""
        G = nx.barabasi_albert_graph(50, 3)
        env = IndependentCascadeEnv(G, IndependentCascadeConfig())
        env.reset()
        
        state, _, _, _ = env.step({"node_id": 0, "delay": 0})
        
        assert state["t"] == 1

    def test_valid_intervention_decreases_budget(self):
        """Valid intervention should decrease remaining budget."""
        G = nx.barabasi_albert_graph(50, 3)
        config = IndependentCascadeConfig(intervention_budget=10)
        env = IndependentCascadeEnv(G, config)
        env.reset()
        
        env.step({"node_id": 0, "delay": 0})
        
        assert env.remaining_budget == 9

    def test_invalid_node_id_no_op(self):
        """Invalid node_id should not use budget."""
        G = nx.barabasi_albert_graph(50, 3)
        config = IndependentCascadeConfig(intervention_budget=10)
        env = IndependentCascadeEnv(G, config)
        env.reset()
        
        env.step({"node_id": -1, "delay": 0})
        
        assert env.remaining_budget == 10

    def test_delayed_intervention(self):
        """Delayed intervention should protect node later."""
        G = nx.path_graph(10)  # Simple chain for predictability
        config = IndependentCascadeConfig(
            infection_prob=0.0,  # No spread
            initial_infected_count=1,
            intervention_budget=5,
        )
        env = IndependentCascadeEnv(G, config)
        env.reset()
        
        # Schedule intervention with delay=2
        env.step({"node_id": 5, "delay": 2})
        assert env.node_state[5] != NodeState.PROTECTED  # Not yet
        
        env.step({"node_id": -1, "delay": 0})  # t=1
        assert env.node_state[5] != NodeState.PROTECTED  # Still not
        
        env.step({"node_id": -1, "delay": 0})  # t=2, intervention applied
        assert env.node_state[5] == NodeState.PROTECTED


class TestTermination:
    """Tests for episode termination."""

    def test_terminates_when_max_steps_reached(self):
        """Episode should end at max_steps."""
        G = nx.barabasi_albert_graph(50, 3)
        config = IndependentCascadeConfig(
            infection_prob=0.0,  # No spread
            initial_infected_count=1,
            max_steps=5,
        )
        env = IndependentCascadeEnv(G, config)
        env.reset()
        
        for _ in range(4):
            _, _, done, _ = env.step({"node_id": -1, "delay": 0})
            assert not done
        
        _, _, done, _ = env.step({"node_id": -1, "delay": 0})
        assert done

    def test_terminates_when_budget_exhausted(self):
        """Episode should end when budget is exhausted."""
        G = nx.barabasi_albert_graph(50, 3)
        config = IndependentCascadeConfig(
            intervention_budget=2,
            max_steps=100,
        )
        env = IndependentCascadeEnv(G, config)
        env.reset()
        
        env.step({"node_id": 0, "delay": 0})
        _, _, done, _ = env.step({"node_id": 1, "delay": 0})
        
        assert done


class TestInfectionPropagation:
    """Tests for infection spreading."""

    def test_infection_spreads_with_prob_one(self):
        """With p=1, infection should spread to all neighbors."""
        G = nx.star_graph(5)  # Center connected to 5 leaves
        config = IndependentCascadeConfig(
            infection_prob=1.0,
            initial_infected_count=0,
            intervention_budget=0,
            max_steps=10,
            seed=42,
        )
        env = IndependentCascadeEnv(G, config)
        env.reset()
        
        # Manually infect center node
        env.node_state[0] = NodeState.INFECTED
        
        # Step should spread to all neighbors
        state, _, _, _ = env.step({"node_id": -1, "delay": 0})
        
        # All nodes should be infected
        assert state["n_infected"] == 6

    def test_protection_blocks_infection(self):
        """Protected nodes should not get infected."""
        G = nx.path_graph(3)  # 0 -- 1 -- 2
        config = IndependentCascadeConfig(
            infection_prob=1.0,
            initial_infected_count=0,
            intervention_budget=1,
            max_steps=10,
        )
        env = IndependentCascadeEnv(G, config)
        env.reset()
        
        # Infect node 0
        env.node_state[0] = NodeState.INFECTED
        
        # Protect node 1
        env.step({"node_id": 1, "delay": 0})
        
        # Node 1 should be protected and node 2 should stay susceptible
        assert env.node_state[1] == NodeState.PROTECTED
        
        # Step again - infection cannot pass through protected node
        env.step({"node_id": -1, "delay": 0})
        
        assert env.node_state[2] == NodeState.SUSCEPTIBLE


class TestNodeMask:
    """Tests for node intervention mask."""

    def test_protected_nodes_masked(self):
        """Protected nodes should have mask=0."""
        G = nx.barabasi_albert_graph(10, 2)
        config = IndependentCascadeConfig(intervention_budget=5)
        env = IndependentCascadeEnv(G, config)
        state = env.reset()
        
        # Protect a node
        state, _, _, _ = env.step({"node_id": 3, "delay": 0})
        
        assert state["node_mask"][3] == 0.0

    def test_unprotected_nodes_valid(self):
        """Non-protected nodes should have mask=1."""
        G = nx.barabasi_albert_graph(10, 2)
        env = IndependentCascadeEnv(G, IndependentCascadeConfig())
        state = env.reset()
        
        # Most nodes should be valid (not yet protected)
        assert state["node_mask"].sum() > 0


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
