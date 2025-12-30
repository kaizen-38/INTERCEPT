"""
Training Curve Visualization.

Simple utility to plot training curves from INTERCEPT training runs.

Usage:
    $ python -m src.plot_training results/intercept_grpo_*/
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import matplotlib.pyplot as plt


def load_metrics(run_dir: str | Path) -> tuple[list, list]:
    """Load training metrics from a run directory.
    
    Args:
        run_dir: Path to training run directory
    
    Returns:
        Tuple of (group_indices, mean_returns)
    """
    run_dir = Path(run_dir)
    metrics_path = run_dir / "metrics.jsonl"
    
    groups, mean_returns = [], []

    with metrics_path.open() as f:
        for line in f:
            m = json.loads(line)
            groups.append(m["group_idx"])
            mean_returns.append(m["mean_return"])
    
    return groups, mean_returns


def plot_training_curve(
    run_dir: str | Path,
    save_path: str | Path | None = None,
) -> None:
    """Plot training curve from metrics file.
    
    Args:
        run_dir: Path to training run directory
        save_path: Path to save figure (shows plot if None)
    """
    groups, mean_returns = load_metrics(run_dir)

    plt.figure(figsize=(10, 6))
    plt.plot(groups, mean_returns, marker="o", markersize=3, 
             linewidth=1.5, color="#3498db")
    plt.xlabel("Group Index", fontsize=12)
    plt.ylabel("Mean Return (higher is better)", fontsize=12)
    plt.title("INTERCEPT – GRPO Training Curve", fontsize=14, fontweight="bold")
    plt.grid(True, alpha=0.3)
    plt.tight_layout()

    if save_path:
        save_path = Path(save_path)
        save_path.parent.mkdir(parents=True, exist_ok=True)
        plt.savefig(save_path, dpi=300, bbox_inches="tight")
        print(f"✓ Saved training curve to {save_path}")
    else:
        plt.show()
    
    plt.close()


def main() -> None:
    """Command-line entry point."""
    parser = argparse.ArgumentParser(
        description="Plot INTERCEPT training curves"
    )
    parser.add_argument(
        "run_dir",
        type=str,
        help="Path to training run directory",
    )
    parser.add_argument(
        "--output",
        type=str,
        default=None,
        help="Path to save figure (shows plot if not specified)",
    )

    args = parser.parse_args()

    plot_training_curve(args.run_dir, args.output)


if __name__ == "__main__":
    import sys
    
    if len(sys.argv) == 1:
        # Default: find most recent run
        results_dir = Path("results")
        if results_dir.exists():
            runs = sorted(results_dir.glob("intercept_grpo_*"))
            if runs:
                print(f"Plotting most recent run: {runs[-1]}")
                plot_training_curve(runs[-1])
            else:
                print("No training runs found.")
                print("Usage: python -m src.plot_training <run_dir>")
        else:
            print("Usage: python -m src.plot_training <run_dir>")
    else:
        main()
