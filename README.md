# INTERCEPT

**INT**ervention **E**nforcement **R**einforcement **C**ontrol for **E**pidemic **P**revention **T**ransmission

A reinforcement learning framework for optimal intervention timing in network cascade control using Group Relative Policy Optimization (GRPO).

---

## Overview

INTERCEPT addresses the problem of controlling information or disease cascades on networks through strategically timed interventions. Given a limited budget of interventions, the system learns *which* nodes to protect and *when* to apply protection to minimize total infections.

### Key Features

- **Graph Neural Network Policy**: Uses a GCN-style encoder to capture network structure and node states
- **Temporal Decision Making**: Jointly learns node selection and intervention timing
- **GRPO Training**: Group Relative Policy Optimization for stable, sample-efficient learning
- **Comprehensive Baselines**: Comparison against degree, PageRank, betweenness, closeness, and k-shell centrality heuristics
- **Multi-Network Evaluation**: Supports Barabási-Albert, Erdős-Rényi, Watts-Strogatz, and real-world SNAP datasets

---

## Installation

### Requirements

- Python 3.10+
- PyTorch 2.0+
- NetworkX 3.0+

### Setup

```bash
# Clone the repository
git clone https://github.com/kaizen-38/INTERCEPT.git
cd INTERCEPT

# Create virtual environment
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate

# Install dependencies
pip install -r requirements.txt
```

---

## Quick Start

### 1. Train a Policy

```bash
python -m src.train_intercept
```

This trains an INTERCEPT policy on BA(80,3) graphs with default hyperparameters. Training checkpoints are saved to `results/intercept_grpo_*/`.

### 2. Evaluate Against Baselines

```bash
python -m src.evaluate_intercept \
    --checkpoint results/intercept_grpo_ba80_*/checkpoints/checkpoint_group_0100.pt \
    --n-trials 100
```

### 3. Run Baseline Comparison Only

```bash
python -m src.intercept_baselines
```

---

## Project Structure

```
INTERCEPT/
├── src/
│   ├── intercept_env.py          # Independent Cascade environment
│   ├── intercept_grpo.py         # Policy network and GRPO trainer
│   ├── intercept_baselines.py    # Centrality-based baseline strategies
│   ├── train_intercept.py        # Main training script
│   ├── evaluate_intercept.py     # Evaluation and comparison
│   ├── evaluate_email_eu.py      # Evaluation on SNAP Email-Eu dataset
│   ├── network_datasets.py       # Network generators and loaders
│   ├── run_multi_network_experiments.py  # Cross-network experiments
│   ├── temporal_analysis.py      # Analysis of learned timing strategies
│   ├── visualize_cascades.py     # Cascade visualization utilities
│   ├── analyze_training.py       # Training diagnostics
│   └── plot_training.py          # Training curve visualization
├── tests/
│   └── test_environment.py       # Unit tests
├── data/
│   └── snap/                     # SNAP dataset files
├── results/                      # Training outputs and evaluations
├── figures/                      # Generated visualizations
├── requirements.txt
├── LICENSE
└── README.md
```

---

## Core Components

### Environment (`intercept_env.py`)

The `IndependentCascadeEnv` implements a discrete-time Independent Cascade model with intervention capabilities:

- **States**: Node features (infection status, degree, timestep) + adjacency matrix
- **Actions**: `(node_id, delay)` — select a node and schedule intervention timing
- **Dynamics**: Infected nodes spread to susceptible neighbors with probability `p`

```python
from src.intercept_env import IndependentCascadeEnv, IndependentCascadeConfig
import networkx as nx

graph = nx.barabasi_albert_graph(100, 3)
config = IndependentCascadeConfig(
    infection_prob=0.05,
    initial_infected_count=3,
    intervention_budget=10,
    max_steps=40,
)
env = IndependentCascadeEnv(graph, config)

state = env.reset()
action = {"node_id": 5, "delay": 2}
next_state, reward, done, info = env.step(action)
```

### Policy Network (`intercept_grpo.py`)

`TemporalGRPOPolicy` is a graph neural network that outputs:
1. **Node logits**: Which node to intervene on
2. **Delay logits**: When to apply the intervention (discretized into time bins)

```python
from src.intercept_grpo import TemporalGRPOPolicy, GRPOTrainer, GRPOConfig

policy = TemporalGRPOPolicy(
    node_feat_dim=5,
    n_time_bins=5,
    hidden_dim=64,
)

# Sample an action
sample = policy.sample_action(node_features, adj_matrix, node_mask)
# Returns: {"node_id": tensor, "delay": tensor, "log_prob": tensor, "entropy": tensor}
```

### GRPO Training

Group Relative Policy Optimization computes advantages by ranking returns within a group of trajectories, enabling stable learning without a value function:

```python
trainer = GRPOTrainer(policy, GRPOConfig(
    learning_rate=2e-4,
    group_size=16,
    clip_epsilon=0.2,
    entropy_coef=0.01,
))

# trajectories: List[Dict] with keys ["states", "actions", "log_probs", "total_return"]
metrics = trainer.update_policy(trajectories)
```

---

## Experiments

### Baseline Strategies

| Strategy | Description |
|----------|-------------|
| Random | Uniform random node selection |
| Degree Centrality | Highest-degree nodes (hubs) |
| PageRank | Highest PageRank scores |
| Betweenness | Highest betweenness centrality |
| Closeness | Highest closeness centrality |
| K-Shell | Highest k-shell decomposition |

### Supported Networks

| Network | Type | Nodes | Description |
|---------|------|-------|-------------|
| BA-{n} | Synthetic | n | Barabási-Albert scale-free |
| ER-{n} | Synthetic | n | Erdős-Rényi random |
| WS-{n} | Synthetic | n | Watts-Strogatz small-world |
| Email-Eu | Real | 1,005 | SNAP Email network |

### Running Multi-Network Experiments

```bash
python -m src.run_multi_network_experiments \
    --checkpoint results/intercept_grpo_*/checkpoints/checkpoint_group_0100.pt \
    --n-trials 50 \
    --output-dir results/multi_network
```

---

## Configuration

### Training Hyperparameters

| Parameter | Default | Description |
|-----------|---------|-------------|
| `total_groups` | 100 | Number of GRPO update steps |
| `group_size` | 16 | Trajectories per group |
| `learning_rate` | 2e-4 | Adam learning rate |
| `hidden_dim` | 64 | GNN hidden dimension |
| `n_time_bins` | 5 | Delay discretization bins |
| `entropy_coef` | 0.003 | Exploration bonus weight |
| `clip_epsilon` | 0.2 | PPO clipping parameter |

### Environment Parameters

| Parameter | Default | Description |
|-----------|---------|-------------|
| `infection_prob` | 0.05 | Per-edge transmission probability |
| `initial_infected_count` | 3 | Seed infections |
| `intervention_budget` | 10 | Maximum interventions |
| `max_steps` | 40-50 | Episode horizon |

---

## Results

After training, INTERCEPT typically achieves:

- **10-25% reduction** in final infections vs. best centrality baseline
- **Learned timing patterns**: Policy adapts intervention timing based on network state
- **Generalization**: Policies trained on BA graphs transfer to other topologies

See `results/` for detailed evaluation outputs and `figures/` for visualizations.

---

## Extending INTERCEPT

### Adding New Baselines

```python
# In src/intercept_baselines.py
class MyBaseline(BaselineStrategy):
    def select_nodes(self, budget: int) -> List[int]:
        # Your node selection logic
        return selected_nodes
    
    def get_name(self) -> str:
        return "My Strategy"
```

### Using Custom Networks

```python
from src.network_datasets import download_snap_dataset

# Load SNAP dataset (place file in data/snap/)
graph = download_snap_dataset("email-Eu-core")

# Or create custom network
import networkx as nx
graph = nx.read_edgelist("my_network.txt")
```

---

## Citation

If you use INTERCEPT in your research, please cite:

```bibtex
@software{intercept2025,
  title={INTERCEPT: Intervention Reinforcement Control for Epidemic Prevention Transmission},
  author={kaizen-38},
  year={2025},
  url={https://github.com/kaizen-38/INTERCEPT}
}
```

---

## License

MIT License — see [LICENSE](LICENSE) for details.

---

## Acknowledgments

- NetworkX for graph algorithms
- PyTorch for deep learning infrastructure
- SNAP project for real-world network datasets
