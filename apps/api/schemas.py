"""Pydantic schemas for API request/response validation.

All API contracts are defined here for single-source-of-truth typing.
The OpenAPI spec generated from these is used to produce the TS client.
"""

from __future__ import annotations

from enum import Enum
from typing import Dict, List, Optional

from pydantic import BaseModel, Field


# ── Shared enums ──────────────────────────────────────────────────────

class NetworkType(str, Enum):
    BA = "ba"
    ER = "er"
    WS = "ws"


class StrategyName(str, Enum):
    RANDOM = "random"
    DEGREE = "degree"
    PAGERANK = "pagerank"
    BETWEENNESS = "betweenness"
    CLOSENESS = "closeness"
    KSHELL = "kshell"
    INTERCEPT = "intercept"


# ── Artifact responses ────────────────────────────────────────────────

class StrategyResult(BaseModel):
    mean_infected: float
    std_infected: float
    median_infected: float
    min_infected: int
    max_infected: int
    mean_protected: Optional[float] = None


class ArtifactResponse(BaseModel):
    """Generic artifact: strategy_name -> metrics."""
    data: Dict[str, StrategyResult]


class MultiNetworkResponse(BaseModel):
    """Multi-network artifact: network_name -> strategy_name -> metrics."""
    data: Dict[str, Dict[str, StrategyResult]]


# ── Simulation ────────────────────────────────────────────────────────

class SimulationRequest(BaseModel):
    network_type: NetworkType = NetworkType.BA
    n_nodes: int = Field(default=50, ge=10, le=500)
    infection_prob: float = Field(default=0.05, gt=0.0, le=1.0)
    initial_infected: int = Field(default=3, ge=1, le=50)
    budget: int = Field(default=5, ge=1, le=50)
    max_steps: int = Field(default=30, ge=5, le=200)
    strategy: StrategyName = StrategyName.DEGREE
    force_zero_delay: bool = True
    seed: Optional[int] = None


class NodePosition(BaseModel):
    x: float
    y: float


class GraphEdge(BaseModel):
    source: int
    target: int


class StepSnapshot(BaseModel):
    t: int
    node_states: List[int]
    n_infected: int
    n_protected: int
    action_node: Optional[int] = None
    action_delay: Optional[int] = None


class SimulationResponse(BaseModel):
    nodes: List[NodePosition]
    edges: List[GraphEdge]
    timeline: List[StepSnapshot]
    strategy: str
    network_info: str


# ── Baseline comparison ───────────────────────────────────────────────

class BaselineCompareRequest(BaseModel):
    network_type: NetworkType = NetworkType.BA
    n_nodes: int = Field(default=80, ge=10, le=500)
    infection_prob: float = Field(default=0.05, gt=0.0, le=1.0)
    initial_infected: int = Field(default=3, ge=1, le=50)
    budget: int = Field(default=10, ge=1, le=50)
    max_steps: int = Field(default=40, ge=5, le=200)
    n_trials: int = Field(default=30, ge=5, le=200)
    seed: Optional[int] = 42


class BaselineCompareResponse(BaseModel):
    results: Dict[str, StrategyResult]
    network_info: str


# ── Health ────────────────────────────────────────────────────────────

class HealthResponse(BaseModel):
    status: str = "ok"
    version: str = "0.1.0"
