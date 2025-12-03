"""
teammate_env.py

Thin wrapper around TemporalCascadeEnv exposing a simpler
IndependentCascadeEnvironment used in the unit tests.

The constructor takes a pre-built NetworkX graph G and an
infection probability, and internally reuses the rich
TemporalCascadeEnv dynamics with the Independent Cascade model.
"""

from typing import Optional

import networkx as nx

from .environment import TemporalCascadeEnv


class IndependentCascadeEnvironment(TemporalCascadeEnv):
    """Simple IC environment wrapper used in unit tests.

    It reuses TemporalCascadeEnv but:
    - takes a pre-built graph G in the constructor
    - disables graph regeneration between episodes
    - uses the Independent Cascade model only
    - does not require an action argument in step()
    """

    def __init__(
        self,
        G: nx.Graph,
        infection_prob: float = 0.15,
        initial_infected: int = 3,
        max_steps: int = 30,
        seed: Optional[int] = None,
    ):
        super().__init__(
            n_nodes=G.number_of_nodes(),
            graph_type="ba",               # ignored because we pass a graph object
            graph_kwargs=None,
            infection_prob=infection_prob,
            initial_infected=initial_infected,
            max_steps=max_steps,
            intervention_budget=0,         # no interventions in the basic tests
            cost_per_treatment=0.0,
            cascade_model="ic",
            exposure_period=1,
            infectious_period=3,
            treatment_delay_penalty=0.1,
            early_intervention_bonus=0.0,
            max_intervention_delay=0,
            regen_graph_each_reset=False,
            seed=seed,
            graph=G,
        )


__all__ = ["IndependentCascadeEnvironment"]
