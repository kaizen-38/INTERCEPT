"""
INTERCEPT Training Script using GRPO.

This script trains the INTERCEPT policy using Group Relative Policy
Optimization (GRPO) on the Independent Cascade environment.

Training Configuration:
    - Environment: BA(80, 3) graph with infection_prob=0.05
    - Objective: Minimize final number of infected nodes
    - Policy: TemporalGRPOPolicy with GNN encoder
    - Training: GRPO with PPO-style clipping

The training loop:
    1. Collect group_size trajectories using current policy
    2. Compute group-relative advantages from final infection counts
    3. Update policy with clipped surrogate objective
    4. Log metrics and save checkpoints periodically

Usage:
    Run from project root:
    
    $ python -m src.train_intercept
    
    Or with custom settings:
    
    $ python -m src.train_intercept --total-groups 200 --group-size 32

Output:
    Training outputs are saved to results/intercept_grpo_ba80_<timestamp>/
    including:
    - train_config.json: Full configuration
    - metrics.jsonl: Per-update training metrics
    - checkpoints/: Model checkpoints

Example:
    >>> from src.train_intercept import train_intercept_grpo, TrainConfig
    >>> 
    >>> config = TrainConfig(total_groups=100, group_size=16)
    >>> run_dir = train_intercept_grpo(make_env, config)
"""

from __future__ import annotations

import json
from dataclasses import asdict, dataclass
from datetime import datetime
from pathlib import Path
from typing import Any, Callable, Dict, List

import networkx as nx
import numpy as np
import torch

from src.intercept_env import IndependentCascadeConfig, IndependentCascadeEnv
from src.intercept_grpo import GRPOConfig, GRPOTrainer, TemporalGRPOPolicy


@dataclass
class TrainConfig:
    """Training configuration for INTERCEPT.
    
    Attributes:
        node_feature_dim: Dimension of node features (must match env)
        n_time_bins: Number of delay discretization bins
        hidden_dim: GNN hidden layer dimension
        total_groups: Number of GRPO update iterations
        group_size: Trajectories per GRPO group
        max_steps_per_trajectory: Maximum steps per episode
        gamma: Discount factor (1.0 for final-return objective)
        learning_rate: Adam optimizer learning rate
        clip_epsilon: PPO clipping parameter
        entropy_coef: Entropy bonus coefficient
        max_grad_norm: Gradient clipping threshold
        weight_decay: L2 regularization weight
        log_root: Directory for training outputs
        save_every_groups: Checkpoint frequency
        device: Training device (cuda/cpu)
        seed: Random seed for reproducibility
        graph_n: Number of nodes in training graph
        graph_m: BA model attachment parameter
        infection_prob: Cascade infection probability
        budget: Intervention budget
        max_env_steps: Maximum environment steps
    """
    # Policy / model
    node_feature_dim: int = 5
    n_time_bins: int = 5
    hidden_dim: int = 64

    # GRPO training
    total_groups: int = 100
    group_size: int = 16
    max_steps_per_trajectory: int = 40
    gamma: float = 1.0

    # Optimization
    learning_rate: float = 2e-4
    clip_epsilon: float = 0.2
    entropy_coef: float = 0.003
    max_grad_norm: float = 1.0
    weight_decay: float = 1e-4

    # Logging
    log_root: Path = Path("results")
    save_every_groups: int = 20

    # Device and reproducibility
    device: str = "cuda" if torch.cuda.is_available() else "cpu"
    seed: int | None = 0

    # Environment settings
    graph_n: int = 80
    graph_m: int = 3
    infection_prob: float = 0.05
    budget: int = 10
    max_env_steps: int = 50


def create_log_dir(cfg: TrainConfig) -> Path:
    """Create timestamped directory for training outputs.
    
    Args:
        cfg: Training configuration
    
    Returns:
        Path to created run directory
    """
    ts = datetime.now().strftime("%Y%m%d_%H%M%S")
    run_dir = cfg.log_root / f"intercept_grpo_ba80_{ts}"
    (run_dir / "checkpoints").mkdir(parents=True, exist_ok=True)
    return run_dir


def save_checkpoint(
    run_dir: Path,
    group_idx: int,
    policy: TemporalGRPOPolicy,
    trainer: GRPOTrainer,
    extra: Dict[str, Any] | None = None,
) -> None:
    """Save training checkpoint.
    
    Args:
        run_dir: Training run directory
        group_idx: Current training iteration
        policy: Policy network to save
        trainer: Trainer with optimizer state
        extra: Additional metadata to save
    """
    extra = extra or {}
    ckpt_path = run_dir / "checkpoints" / f"checkpoint_group_{group_idx:04d}.pt"
    torch.save(
        {
            "group_idx": group_idx,
            "policy_state_dict": policy.state_dict(),
            "optimizer_state_dict": trainer.optimizer.state_dict(),
            "trainer_stats": trainer.stats,
            "extra": extra,
        },
        ckpt_path,
    )


def rollout_final_objective(
    env: IndependentCascadeEnv,
    policy: TemporalGRPOPolicy,
    cfg: TrainConfig,
) -> Dict[str, Any]:
    """Collect a trajectory with final-infection objective.
    
    The return is defined as negative final infection count, which
    directly matches the evaluation metric used for baselines.
    
    Args:
        env: Cascade environment instance
        policy: Policy network for action selection
        cfg: Training configuration
    
    Returns:
        Trajectory dictionary with keys:
        - "states": List of state observations
        - "actions": List of actions taken
        - "log_probs": List of action log probabilities
        - "total_return": Negative final infection count
        - "final_infected": Raw final infection count
    """
    device = cfg.device

    state = env.reset()
    done = False

    states: List[Dict[str, Any]] = []
    actions: List[Dict[str, int]] = []
    log_probs: List[float] = []

    steps = 0
    interventions_used = 0

    while not done and steps < cfg.max_env_steps:
        steps += 1

        # If no valid nodes or budget exhausted, let cascade run
        if state["node_mask"].sum() <= 0 or interventions_used >= cfg.budget:
            action = {"node_id": -1, "delay": 0}
            next_state, _, done, _ = env.step(action)
            state = next_state
            continue

        # Prepare tensors for policy
        node_features = torch.tensor(
            state["node_features"], dtype=torch.float32, device=device
        ).unsqueeze(0)
        adj_matrix = torch.tensor(
            state["adj_matrix"], dtype=torch.float32, device=device
        ).unsqueeze(0)
        node_mask = torch.tensor(
            state["node_mask"], dtype=torch.float32, device=device
        ).unsqueeze(0)

        # Sample action from policy
        sample = policy.sample_action(node_features, adj_matrix, node_mask)

        node_id = int(sample["node_id"][0].item())
        # Force immediate intervention for this training phase
        delay = 0
        log_prob = float(sample["log_prob"][0].item())

        action = {"node_id": node_id, "delay": delay}
        next_state, _, done, _ = env.step(action)

        states.append(state)
        actions.append(action)
        log_probs.append(log_prob)

        interventions_used += 1
        state = next_state

    # Return is negative final infection count
    final_infected = int(state["n_infected"])
    total_return = -float(final_infected)

    return {
        "states": states,
        "actions": actions,
        "log_probs": log_probs,
        "total_return": total_return,
        "final_infected": final_infected,
    }


def train_intercept_grpo(
    env_factory: Callable[[], IndependentCascadeEnv],
    cfg: TrainConfig,
) -> Path:
    """Main GRPO training loop.
    
    Trains the INTERCEPT policy using Group Relative Policy Optimization.
    Logs metrics and saves checkpoints throughout training.
    
    Args:
        env_factory: Callable that creates fresh environment instances
        cfg: Training configuration
    
    Returns:
        Path to the training run directory
    """
    # Set random seeds
    if cfg.seed is not None:
        np.random.seed(cfg.seed)
        torch.manual_seed(cfg.seed)
        if cfg.device.startswith("cuda"):
            torch.cuda.manual_seed_all(cfg.seed)

    run_dir = create_log_dir(cfg)
    metrics_path = run_dir / "metrics.jsonl"

    # Save configuration
    cfg_dict = asdict(cfg)
    for k, v in list(cfg_dict.items()):
        if isinstance(v, Path):
            cfg_dict[k] = str(v)
    with (run_dir / "train_config.json").open("w") as f:
        json.dump(cfg_dict, f, indent=2)

    # Initialize policy and trainer
    policy = TemporalGRPOPolicy(
        node_feat_dim=cfg.node_feature_dim,
        n_time_bins=cfg.n_time_bins,
        hidden_dim=cfg.hidden_dim,
    ).to(cfg.device)

    grpo_cfg = GRPOConfig(
        learning_rate=cfg.learning_rate,
        group_size=cfg.group_size,
        clip_epsilon=cfg.clip_epsilon,
        entropy_coef=cfg.entropy_coef,
        max_grad_norm=cfg.max_grad_norm,
        weight_decay=cfg.weight_decay,
    )
    trainer = GRPOTrainer(policy, grpo_cfg)

    # Print training info
    print("=" * 70)
    print("INTERCEPT – GRPO Training")
    print("=" * 70)
    print(f"Device:         {cfg.device}")
    print(f"Run dir:        {run_dir}")
    print(f"Groups:         {cfg.total_groups}")
    print(f"Group size:     {cfg.group_size}")
    print(f"Graph:          BA({cfg.graph_n}, {cfg.graph_m})")
    print(f"Infection prob: {cfg.infection_prob}")
    print(f"Budget:         {cfg.budget}")
    print("=" * 70 + "\n")

    total_env_episodes = 0

    with metrics_path.open("a") as f_metrics:
        for group_idx in range(1, cfg.total_groups + 1):
            # Collect trajectories
            trajectories: List[Dict[str, Any]] = []

            for _ in range(cfg.group_size):
                env = env_factory()
                traj = rollout_final_objective(env, policy, cfg)
                trajectories.append(traj)
                total_env_episodes += 1

            # GRPO update
            metrics = trainer.update_policy(trajectories)
            metrics["group_idx"] = group_idx
            metrics["total_env_episodes"] = total_env_episodes

            # Compute infection statistics
            mean_final_infected = float(
                np.mean([t["final_infected"] for t in trajectories])
            )
            std_final_infected = float(
                np.std([t["final_infected"] for t in trajectories])
            )

            # Log to console
            print(
                f"[group {group_idx:03d}/{cfg.total_groups:03d}] "
                f"mean_return={metrics['mean_return']:.3f} "
                f"std_return={metrics['std_return']:.3f} "
                f"mean_final_inf={mean_final_infected:.2f} "
                f"policy_loss={metrics['policy_loss']:.3f} "
                f"entropy={metrics['entropy']:.3f}"
            )

            # Log to file
            out_row = dict(metrics)
            out_row["mean_final_infected"] = mean_final_infected
            out_row["std_final_infected"] = std_final_infected
            f_metrics.write(json.dumps(out_row) + "\n")
            f_metrics.flush()

            # Save checkpoint
            if group_idx % cfg.save_every_groups == 0 or group_idx == cfg.total_groups:
                save_checkpoint(
                    run_dir,
                    group_idx,
                    policy,
                    trainer,
                    extra={"total_env_episodes": total_env_episodes},
                )

    print(f"\n[INTERCEPT] Training complete. Logs written to: {run_dir}")
    return run_dir


def make_default_env() -> IndependentCascadeEnv:
    """Create default environment for training.
    
    Uses BA(80, 3) graph with standard cascade parameters,
    aligned with baseline evaluation configuration.
    
    Returns:
        Fresh IndependentCascadeEnv instance
    """
    graph = nx.barabasi_albert_graph(80, 3)
    env_cfg = IndependentCascadeConfig(
        infection_prob=0.05,
        initial_infected_count=3,
        intervention_budget=10,
        max_steps=50,
        intervention_cost=0.1,
        delay_penalty=0.01,
        seed=None,
    )
    return IndependentCascadeEnv(graph, env_cfg)


if __name__ == "__main__":
    device = "cuda" if torch.cuda.is_available() else "cpu"

    train_cfg = TrainConfig(
        device=device,
        graph_n=80,
        graph_m=3,
        infection_prob=0.05,
        budget=10,
        total_groups=100,
        group_size=16,
        max_steps_per_trajectory=40,
        max_env_steps=50,
        learning_rate=2e-4,
        entropy_coef=0.003,
        seed=0,
    )

    run_dir = train_intercept_grpo(make_default_env, train_cfg)
    print(f"Run directory: {run_dir}")
