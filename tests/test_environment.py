"""
Unit tests for the IC environment.

Run with:
    pytest tests/
"""

import networkx as nx

from src.teammate_env import IndependentCascadeEnvironment


def test_environment_creation():
    G = nx.barabasi_albert_graph(100, 3)
    env = IndependentCascadeEnvironment(G, infection_prob=0.15)
    assert env.n_nodes == 100


def test_reset():
    G = nx.barabasi_albert_graph(50, 3)
    env = IndependentCascadeEnvironment(G, infection_prob=0.15)
    state = env.reset()
    assert state["n_infected"] > 0
    assert state["node_features"].shape[0] == 50


def test_cascade_step():
    G = nx.barabasi_albert_graph(50, 3)
    env = IndependentCascadeEnvironment(G, infection_prob=0.15)
    env.reset()
    state, reward, done, info = env.step()
    assert isinstance(reward, (int, float))
    assert isinstance(done, bool)
