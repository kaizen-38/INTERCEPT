"""INTERCEPT API — FastAPI server wrapping the RL cascade-control framework.

Run:  uvicorn apps.api.main:app --reload --port 8000
Docs: http://localhost:8000/docs
"""

from __future__ import annotations

from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware

from apps.api.schemas import (
    ArtifactResponse,
    BaselineCompareRequest,
    BaselineCompareResponse,
    HealthResponse,
    MultiNetworkResponse,
    SimulationRequest,
    SimulationResponse,
    StrategyResult,
)
from apps.api.services import (
    load_email_eu_artifact,
    load_evaluation_artifact,
    load_multi_network_artifact,
    run_baseline_comparison,
    run_simulation,
)

app = FastAPI(
    title="INTERCEPT API",
    version="0.1.0",
    description="Cascade control with GNN-based intervention policies.",
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


# ── Health ────────────────────────────────────────────────────────────

@app.get("/api/health", response_model=HealthResponse)
def health() -> HealthResponse:
    return HealthResponse()


# ── Artifact endpoints (read-only demo data) ─────────────────────────

def _parse_artifact(raw: dict | None) -> dict[str, StrategyResult]:
    if raw is None:
        raise HTTPException(status_code=404, detail="Artifact not found on disk.")
    return {k: StrategyResult(**v) for k, v in raw.items()}


def _parse_multi_artifact(raw: dict | None) -> dict[str, dict[str, StrategyResult]]:
    if raw is None:
        raise HTTPException(status_code=404, detail="Artifact not found on disk.")
    return {
        net: {strat: StrategyResult(**v) for strat, v in strats.items()}
        for net, strats in raw.items()
    }


@app.get("/api/artifacts/evaluation", response_model=ArtifactResponse)
def get_evaluation_artifact() -> ArtifactResponse:
    return ArtifactResponse(data=_parse_artifact(load_evaluation_artifact()))


@app.get("/api/artifacts/multi-network", response_model=MultiNetworkResponse)
def get_multi_network_artifact() -> MultiNetworkResponse:
    return MultiNetworkResponse(data=_parse_multi_artifact(load_multi_network_artifact()))


@app.get("/api/artifacts/email-eu", response_model=ArtifactResponse)
def get_email_eu_artifact() -> ArtifactResponse:
    return ArtifactResponse(data=_parse_artifact(load_email_eu_artifact()))


# ── Simulation ────────────────────────────────────────────────────────

@app.post("/api/simulate", response_model=SimulationResponse)
def simulate(req: SimulationRequest) -> SimulationResponse:
    return run_simulation(req)


# ── Baseline comparison ──────────────────────────────────────────────

@app.post("/api/compare/baselines", response_model=BaselineCompareResponse)
def compare_baselines(req: BaselineCompareRequest) -> BaselineCompareResponse:
    return run_baseline_comparison(req)
