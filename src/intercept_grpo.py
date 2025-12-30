"""
Group Relative Policy Optimization (GRPO) for Network Intervention.

This module implements the policy network architecture and GRPO training
algorithm for learning optimal intervention strategies in cascade control.

The key components are:
- GraphEncoder: GCN-style message passing over the network adjacency
- TemporalGRPOPolicy: Factored policy outputting (node, delay) actions
- GRPOTrainer: Group Relative Policy Optimization with PPO-style clipping

GRPO computes advantages by ranking returns within a group of trajectories,
eliminating the need for a learned value function while maintaining stable
policy gradients.

Example:
    >>> from src.intercept_grpo import TemporalGRPOPolicy, GRPOTrainer, GRPOConfig
    >>> 
    >>> policy = TemporalGRPOPolicy(node_feat_dim=5, n_time_bins=5, hidden_dim=64)
    >>> trainer = GRPOTrainer(policy, GRPOConfig(group_size=16))
    >>> 
    >>> # Collect trajectories and update
    >>> metrics = trainer.update_policy(trajectory_batch)

References:
    - Schulman, J. et al. (2017). Proximal Policy Optimization Algorithms.
    - Kipf, T. N., & Welling, M. (2017). Semi-Supervised Classification with 
      Graph Convolutional Networks.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict, List, Tuple

import numpy as np
import torch
import torch.nn as nn
from torch.distributions import Categorical


class GraphEncoder(nn.Module):
    """Two-layer GCN-style encoder for graph-structured data.
    
    Performs message passing over the adjacency matrix to produce
    node embeddings that capture local network structure.
    
    The forward pass applies:
        h1 = LayerNorm(ReLU(A @ (X @ W1)))
        h2 = LayerNorm(ReLU(A @ (h1 @ W2)))
    
    Attributes:
        fc1: First linear transformation
        fc2: Second linear transformation  
        norm1: Layer normalization after first layer
        norm2: Layer normalization after second layer
        dropout: Dropout for regularization
    
    Args:
        node_feat_dim: Dimension of input node features
        hidden_dim: Hidden layer dimension (default: 64)
        dropout: Dropout probability (default: 0.0)
    """

    def __init__(self, node_feat_dim: int, hidden_dim: int = 64, dropout: float = 0.0) -> None:
        super().__init__()
        self.fc1 = nn.Linear(node_feat_dim, hidden_dim)
        self.fc2 = nn.Linear(hidden_dim, hidden_dim)
        self.norm1 = nn.LayerNorm(hidden_dim)
        self.norm2 = nn.LayerNorm(hidden_dim)
        self.dropout = nn.Dropout(dropout)

    def forward(
        self,
        node_features: torch.Tensor,  # (B, N, F)
        adj_matrix: torch.Tensor,  # (B, N, N)
    ) -> torch.Tensor:
        """Forward pass through the graph encoder.
        
        Args:
            node_features: Node feature tensor of shape (batch, nodes, features)
            adj_matrix: Adjacency matrix of shape (batch, nodes, nodes)
        
        Returns:
            Node embeddings of shape (batch, nodes, hidden_dim)
        """
        x = self.fc1(node_features)
        x = torch.bmm(adj_matrix, x)
        x = self.norm1(x)
        x = torch.relu(x)
        x = self.dropout(x)

        x = self.fc2(x)
        x = torch.bmm(adj_matrix, x)
        x = self.norm2(x)
        x = torch.relu(x)
        x = self.dropout(x)

        return x


class TemporalGRPOPolicy(nn.Module):
    """Factored policy network for (node, delay) action selection.
    
    The policy uses a GNN encoder to embed nodes, then applies separate
    heads for node selection and delay prediction. The delay is conditioned
    on the selected node.
    
    Architecture:
        1. GraphEncoder produces node embeddings H ∈ R^{B×N×D}
        2. Node scorer: H → R^{B×N} (node selection logits)
        3. Time scorer: H → R^{B×N×T} (delay logits per node)
    
    During action sampling:
        1. Sample node from Categorical(node_logits)
        2. Get time_logits for selected node
        3. Sample delay from Categorical(time_logits[node])
    
    Attributes:
        n_time_bins: Number of discrete delay options
        encoder: GraphEncoder for node embeddings
        node_scorer: MLP for node selection logits
        time_scorer: MLP for delay logits
    
    Args:
        node_feat_dim: Dimension of input node features
        n_time_bins: Number of delay discretization bins (default: 5)
        hidden_dim: Hidden layer dimension (default: 64)
        dropout: Dropout probability (default: 0.1)
    """

    def __init__(
        self,
        node_feat_dim: int,
        n_time_bins: int = 5,
        hidden_dim: int = 64,
        dropout: float = 0.1,
    ) -> None:
        super().__init__()
        self.n_time_bins = n_time_bins

        self.encoder = GraphEncoder(node_feat_dim, hidden_dim, dropout=dropout)

        self.node_scorer = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, 1),
        )

        self.time_scorer = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, n_time_bins),
        )

    def forward(
        self,
        node_features: torch.Tensor,  # (B, N, F)
        adj_matrix: torch.Tensor,  # (B, N, N)
        node_mask: torch.Tensor | None = None,  # (B, N)
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """Compute node and time logits for action selection.
        
        Args:
            node_features: Node features, shape (batch, nodes, features)
            adj_matrix: Adjacency matrix, shape (batch, nodes, nodes)
            node_mask: Optional mask for valid nodes, shape (batch, nodes).
                Masked nodes receive -inf logits.
        
        Returns:
            Tuple of (node_logits, time_logits):
            - node_logits: Shape (batch, nodes)
            - time_logits: Shape (batch, nodes, n_time_bins)
        """
        embeddings = self.encoder(node_features, adj_matrix)  # (B, N, H)

        node_logits = self.node_scorer(embeddings).squeeze(-1)  # (B, N)
        if node_mask is not None:
            node_logits = node_logits.masked_fill(~node_mask.bool(), float("-inf"))

        time_logits = self.time_scorer(embeddings)  # (B, N, T)
        return node_logits, time_logits

    @torch.no_grad()
    def sample_action(
        self,
        node_features: torch.Tensor,  # (B, N, F)
        adj_matrix: torch.Tensor,  # (B, N, N)
        node_mask: torch.Tensor | None = None,  # (B, N)
        deterministic: bool = False,
    ) -> Dict[str, torch.Tensor]:
        """Sample an action from the policy.
        
        Args:
            node_features: Node features, shape (batch, nodes, features)
            adj_matrix: Adjacency matrix, shape (batch, nodes, nodes)
            node_mask: Optional mask for valid nodes
            deterministic: If True, select argmax instead of sampling
        
        Returns:
            Dictionary with keys:
            - "node_id": Selected node indices, shape (batch,)
            - "delay": Selected delays, shape (batch,)
            - "log_prob": Log probability of action, shape (batch,)
            - "entropy": Policy entropy, shape (batch,)
        """
        device = next(self.parameters()).device
        node_features = node_features.to(device)
        adj_matrix = adj_matrix.to(device)
        if node_mask is not None:
            node_mask = node_mask.to(device)

        node_logits, time_logits = self.forward(node_features, adj_matrix, node_mask)

        # Sample or select node
        node_dist = Categorical(logits=node_logits)
        if deterministic:
            node_id = node_logits.argmax(dim=-1)
        else:
            node_id = node_dist.sample()
        node_log_prob = node_dist.log_prob(node_id)
        node_entropy = node_dist.entropy()

        # Get time logits for selected node and sample delay
        batch_idx = torch.arange(node_id.shape[0], device=device)
        selected_time_logits = time_logits[batch_idx, node_id]  # (B, T)

        time_dist = Categorical(logits=selected_time_logits)
        if deterministic:
            delay = selected_time_logits.argmax(dim=-1)
        else:
            delay = time_dist.sample()
        time_log_prob = time_dist.log_prob(delay)
        time_entropy = time_dist.entropy()

        # Combined log prob and entropy
        log_prob = node_log_prob + time_log_prob
        entropy = node_entropy + time_entropy

        return {
            "node_id": node_id,      # (B,)
            "delay": delay,          # (B,)
            "log_prob": log_prob,    # (B,)
            "entropy": entropy,      # (B,)
        }


@dataclass
class GRPOConfig:
    """Configuration for GRPO training.
    
    Attributes:
        learning_rate: Adam optimizer learning rate
        group_size: Number of trajectories per GRPO update group
        clip_epsilon: PPO clipping parameter for policy ratio
        entropy_coef: Coefficient for entropy bonus in loss
        max_grad_norm: Maximum gradient norm for clipping
        weight_decay: L2 regularization weight
    """
    learning_rate: float = 3e-4
    group_size: int = 8
    clip_epsilon: float = 0.2
    entropy_coef: float = 0.01
    max_grad_norm: float = 1.0
    weight_decay: float = 1e-4


class GRPOTrainer:
    """
    Group Relative Policy Optimization trainer.
    
    GRPO is a policy gradient method that computes advantages by ranking
    trajectory returns within a group, rather than using a value function.
    This simplifies training while maintaining stable updates.
    
    The algorithm:
    1. Collect group_size trajectories
    2. Rank trajectories by return to compute advantages
    3. Apply PPO-style clipped surrogate objective
    4. Add entropy bonus for exploration
    
    Trajectory Format:
        Each trajectory dict should contain:
        - "states": List of state dicts with node_features, adj_matrix, node_mask
        - "actions": List of action dicts with node_id, delay
        - "log_probs": List of log probabilities (from old policy)
        - "total_return": Scalar total return for the trajectory
    
    Attributes:
        policy: The TemporalGRPOPolicy being trained
        cfg: GRPOConfig with hyperparameters
        optimizer: AdamW optimizer
        stats: Dictionary tracking training statistics
    
    Args:
        policy: Policy network to train
        config: Optional GRPOConfig (uses defaults if None)
    
    Example:
        >>> trainer = GRPOTrainer(policy, GRPOConfig(group_size=16))
        >>> for _ in range(num_updates):
        ...     trajectories = collect_trajectories(policy, env, group_size=16)
        ...     metrics = trainer.update_policy(trajectories)
    """

    def __init__(self, policy: TemporalGRPOPolicy, config: GRPOConfig | None = None) -> None:
        self.policy = policy
        self.cfg = config or GRPOConfig()

        self.optimizer = torch.optim.AdamW(
            self.policy.parameters(),
            lr=self.cfg.learning_rate,
            weight_decay=self.cfg.weight_decay,
        )

        self.stats: Dict[str, Any] = {
            "iterations": 0,
            "total_samples": 0,
        }

    @staticmethod
    def compute_relative_advantages(returns: torch.Tensor) -> torch.Tensor:
        """Compute rank-based relative advantages.
        
        Advantages are computed as normalized ranks of returns within
        the group, providing a simple baseline-free advantage estimate.
        
        Args:
            returns: Tensor of trajectory returns, shape (group_size,)
        
        Returns:
            Normalized advantages, shape (group_size,)
        """
        ranks = returns.argsort().argsort().float()
        return (ranks - ranks.mean()) / (ranks.std() + 1e-8)

    def update_policy(self, trajectory_batch: List[Dict[str, Any]]) -> Dict[str, float]:
        """Perform one GRPO policy update.
        
        Args:
            trajectory_batch: List of trajectory dicts, length must equal group_size.
                Each trajectory contains states, actions, log_probs, total_return.
        
        Returns:
            Dictionary of training metrics:
            - "policy_loss": Clipped surrogate loss
            - "entropy": Mean policy entropy
            - "mean_return": Mean trajectory return
            - "std_return": Std of returns
            - "min_return", "max_return": Return range
            - "mean_advantage": Mean computed advantage
        
        Raises:
            AssertionError: If batch size doesn't match group_size
        """
        assert len(trajectory_batch) == self.cfg.group_size, (
            f"Expected group_size={self.cfg.group_size}, "
            f"got {len(trajectory_batch)}"
        )

        device = next(self.policy.parameters()).device

        # Compute relative advantages from returns
        returns = torch.tensor(
            [traj["total_return"] for traj in trajectory_batch],
            dtype=torch.float32,
            device=device,
        )
        traj_advantages = self.compute_relative_advantages(returns)

        # Flatten all steps from all trajectories
        all_states: List[Dict[str, Any]] = []
        all_actions: List[Dict[str, int]] = []
        all_old_log_probs: List[float] = []
        all_advantages: List[float] = []

        for i, traj in enumerate(trajectory_batch):
            adv_i = traj_advantages[i].item()
            states_i = traj["states"]
            actions_i = traj["actions"]
            log_probs_i = traj["log_probs"]

            assert len(states_i) == len(actions_i) == len(log_probs_i), "Trajectory lengths mismatch"

            for state, action, lp in zip(states_i, actions_i, log_probs_i):
                all_states.append(state)
                all_actions.append(action)
                all_old_log_probs.append(lp)
                all_advantages.append(adv_i)

        old_log_probs = torch.tensor(all_old_log_probs, dtype=torch.float32, device=device)
        advantages = torch.tensor(all_advantages, dtype=torch.float32, device=device)

        # Recompute log probs under current policy
        self.policy.train()
        new_log_probs: List[torch.Tensor] = []
        entropies: List[torch.Tensor] = []

        for state, action in zip(all_states, all_actions):
            node_features = torch.tensor(state["node_features"], dtype=torch.float32, device=device).unsqueeze(0)
            adj_matrix = torch.tensor(state["adj_matrix"], dtype=torch.float32, device=device).unsqueeze(0)
            node_mask = torch.tensor(state["node_mask"], dtype=torch.float32, device=device).unsqueeze(0)

            node_logits, time_logits = self.policy(node_features, adj_matrix, node_mask)

            # Node log prob
            node_dist = Categorical(logits=node_logits)
            node_idx = torch.tensor([action["node_id"]], dtype=torch.long, device=device)
            node_log_prob = node_dist.log_prob(node_idx)
            node_entropy = node_dist.entropy()

            # Time log prob
            time_logits_selected = time_logits[0, action["node_id"]]  # (T,)
            time_dist = Categorical(logits=time_logits_selected)
            delay_idx = torch.tensor(action["delay"], dtype=torch.long, device=device)
            time_log_prob = time_dist.log_prob(delay_idx)
            time_entropy = time_dist.entropy()

            total_log_prob = node_log_prob + time_log_prob
            entropy = node_entropy + time_entropy

            new_log_probs.append(total_log_prob.squeeze(0))
            entropies.append(entropy.squeeze(0))

        new_log_probs_t = torch.stack(new_log_probs)
        entropies_t = torch.stack(entropies)

        # PPO clipped objective
        ratio = torch.exp(new_log_probs_t - old_log_probs)
        surr1 = ratio * advantages
        surr2 = torch.clamp(
            ratio, 1.0 - self.cfg.clip_epsilon, 1.0 + self.cfg.clip_epsilon
        ) * advantages
        policy_loss = -torch.min(surr1, surr2).mean()
        entropy_loss = -entropies_t.mean()
        total_loss = policy_loss + self.cfg.entropy_coef * entropy_loss

        # Gradient update
        self.optimizer.zero_grad()
        total_loss.backward()
        nn.utils.clip_grad_norm_(self.policy.parameters(), self.cfg.max_grad_norm)
        self.optimizer.step()

        # Update stats
        self.stats["iterations"] += 1
        self.stats["total_samples"] += len(all_states)

        return {
            "policy_loss": float(policy_loss.item()),
            "entropy": float(entropies_t.mean().item()),
            "mean_return": float(returns.mean().item()),
            "std_return": float(returns.std().item()),
            "min_return": float(returns.min().item()),
            "max_return": float(returns.max().item()),
            "mean_advantage": float(advantages.mean().item()),
        }


def _test_grpo_components() -> None:
    """Smoke test for GRPO components."""
    print("=" * 60)
    print("Testing GRPO Components")
    print("=" * 60)

    torch.manual_seed(0)
    np.random.seed(0)

    batch_size = 2
    n_nodes = 50
    feat_dim = 4
    n_time_bins = 5

    policy = TemporalGRPOPolicy(node_feat_dim=feat_dim, n_time_bins=n_time_bins, hidden_dim=32)
    device = "cuda" if torch.cuda.is_available() else "cpu"
    policy.to(device)

    print(f"✓ Policy created: {sum(p.numel() for p in policy.parameters())} params")

    node_features = torch.randn(batch_size, n_nodes, feat_dim, device=device)
    adj_matrix = torch.eye(n_nodes, device=device).unsqueeze(0).repeat(batch_size, 1, 1)
    node_mask = torch.ones(batch_size, n_nodes, device=device)

    node_logits, time_logits = policy(node_features, adj_matrix, node_mask)
    print(f"✓ Forward pass: node_logits={node_logits.shape}, time_logits={time_logits.shape}")

    sample = policy.sample_action(node_features, adj_matrix, node_mask)
    print(
        f"✓ Sampled action: node={sample['node_id'][0].item()}, "
        f"delay={sample['delay'][0].item()}, "
        f"log_prob={sample['log_prob'][0].item():.3f}"
    )

    trainer = GRPOTrainer(policy, GRPOConfig(group_size=4))
    print("✓ Trainer created")

    dummy_trajectories: List[Dict[str, Any]] = []
    for _ in range(4):
        traj_len = 5
        states, actions, log_probs = [], [], []
        for _t in range(traj_len):
            state = {
                "node_features": torch.randn(n_nodes, feat_dim).cpu().numpy(),
                "adj_matrix": torch.eye(n_nodes).cpu().numpy(),
                "node_mask": torch.ones(n_nodes).cpu().numpy(),
            }
            states.append(state)
            actions.append(
                {
                    "node_id": int(np.random.randint(n_nodes)),
                    "delay": int(np.random.randint(n_time_bins)),
                }
            )
            log_probs.append(float(np.random.randn()))

        dummy_trajectories.append(
            {
                "states": states,
                "actions": actions,
                "log_probs": log_probs,
                "total_return": float(np.random.randn() * 10.0),
            }
        )

    metrics = trainer.update_policy(dummy_trajectories)
    print("✓ Policy update successful")
    print(f"  Policy loss: {metrics['policy_loss']:.3f}")
    print(f"  Mean return: {metrics['mean_return']:.3f}")
    print(f"  Entropy: {metrics['entropy']:.3f}")

    print("\n" + "=" * 60)
    print("ALL TESTS PASSED! ✓")
    print("=" * 60)
    print("\nYou can now integrate with the real environment.")


if __name__ == "__main__":
    _test_grpo_components()
