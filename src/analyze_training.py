"""
Training Analysis and Diagnostics.

This module provides tools for analyzing INTERCEPT training runs,
diagnosing potential issues, and visualizing training progress.

Usage:
    $ python -m src.analyze_training results/intercept_grpo_*/

Output:
    - training_diagnosis.png: Diagnostic visualization
    - Console summary of issues and recommendations
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np


def analyze_training_run(run_dir: str | Path) -> None:
    """Analyze training metrics to diagnose issues.
    
    Loads training metrics, computes diagnostics, and generates
    a visualization highlighting potential problems.
    
    Args:
        run_dir: Path to training run directory
    """
    run_dir = Path(run_dir)
    metrics_path = run_dir / "metrics.jsonl"

    if not metrics_path.exists():
        print(f"Metrics file not found: {metrics_path}")
        return

    # Load metrics
    groups, returns, losses, entropies = [], [], [], []
    final_infected_list = []

    with open(metrics_path) as f:
        for line in f:
            m = json.loads(line)
            groups.append(m["group_idx"])
            returns.append(m["mean_return"])
            losses.append(m["policy_loss"])
            entropies.append(m["entropy"])
            if "mean_final_infected" in m:
                final_infected_list.append(m["mean_final_infected"])

    # Create diagnostic plots
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))

    # Plot 1: Returns over time
    ax = axes[0, 0]
    ax.plot(groups, returns, linewidth=2, color="#e74c3c")
    ax.axhline(0, color="black", linestyle="--", alpha=0.3)
    ax.set_xlabel("Training Group", fontsize=11)
    ax.set_ylabel("Mean Return", fontsize=11)
    ax.set_title("Training Returns (Should Increase)", fontsize=12, fontweight="bold")
    ax.grid(alpha=0.3)

    # Diagnose returns
    final_return = returns[-1]
    initial_return = returns[0]
    improvement = final_return - initial_return

    if improvement < 0:
        ax.text(
            0.5, 0.95, "❌ PROBLEM: Returns decreased!",
            transform=ax.transAxes, ha="center", va="top",
            bbox=dict(boxstyle="round", facecolor="red", alpha=0.7),
            fontsize=11, fontweight="bold"
        )
    elif improvement < 5:
        ax.text(
            0.5, 0.95, "⚠️  WARNING: Minimal improvement",
            transform=ax.transAxes, ha="center", va="top",
            bbox=dict(boxstyle="round", facecolor="orange", alpha=0.7),
            fontsize=11, fontweight="bold"
        )

    # Plot 2: Entropy over time
    ax = axes[0, 1]
    ax.plot(groups, entropies, linewidth=2, color="#3498db")
    ax.set_xlabel("Training Group", fontsize=11)
    ax.set_ylabel("Policy Entropy", fontsize=11)
    ax.set_title("Exploration (Entropy)", fontsize=12, fontweight="bold")
    ax.grid(alpha=0.3)

    # Diagnose entropy
    final_entropy = entropies[-1]
    if final_entropy < 0.5:
        ax.text(
            0.5, 0.95, "❌ PROBLEM: Collapsed to deterministic!",
            transform=ax.transAxes, ha="center", va="top",
            bbox=dict(boxstyle="round", facecolor="red", alpha=0.7),
            fontsize=10, fontweight="bold"
        )

    # Plot 3: Loss over time
    ax = axes[1, 0]
    ax.plot(groups, losses, linewidth=2, color="#2ecc71")
    ax.set_xlabel("Training Group", fontsize=11)
    ax.set_ylabel("Policy Loss", fontsize=11)
    ax.set_title("Training Loss", fontsize=12, fontweight="bold")
    ax.grid(alpha=0.3)

    # Plot 4: Diagnostic summary
    ax = axes[1, 1]
    ax.axis("off")

    # Calculate statistics
    return_trend = np.polyfit(groups, returns, 1)[0]
    entropy_trend = np.polyfit(groups, entropies, 1)[0]

    # Build diagnosis text
    diagnosis_lines = [
        "TRAINING DIAGNOSIS",
        "=" * 50,
        "",
        f"Training Steps: {len(groups)} groups",
        "",
        "Returns:",
        f"  Initial: {initial_return:.2f}",
        f"  Final:   {final_return:.2f}",
        f"  Change:  {improvement:.2f}",
        f"  Trend:   {'↗️ Increasing' if return_trend > 0 else '↘️ Decreasing'}",
        "",
        "Entropy:",
        f"  Initial: {entropies[0]:.3f}",
        f"  Final:   {final_entropy:.3f}",
        f"  Trend:   {'↗️ Increasing' if entropy_trend > 0 else '↘️ Decreasing'}",
    ]

    if final_infected_list:
        diagnosis_lines.extend([
            "",
            "Final Infections:",
            f"  Start: {final_infected_list[0]:.1f}",
            f"  End:   {final_infected_list[-1]:.1f}",
        ])

    diagnosis_lines.extend([
        "",
        "=" * 50,
        "ISSUES DETECTED:",
    ])

    # Identify issues
    issues = []
    if improvement < 5:
        issues.append("❌ Returns barely improved")
    if final_entropy < 0.5:
        issues.append("❌ Policy collapsed (no exploration)")
    if return_trend < 0:
        issues.append("❌ Performance degraded during training")
    if len(groups) < 100:
        issues.append("⚠️  Too few training steps")

    if not issues:
        issues.append("✅ No major issues detected")

    diagnosis_lines.extend(["  " + issue for issue in issues])

    diagnosis_lines.extend([
        "",
        "=" * 50,
        "RECOMMENDED FIXES:",
        "",
        "1. Increase training steps (500+ groups)",
        "2. Higher entropy coefficient (0.02+)",
        "3. Review reward shaping",
        "4. Larger hidden dim (128+)",
        "5. Lower learning rate (1e-4)",
    ])

    diagnosis = "\n".join(diagnosis_lines)

    ax.text(
        0.05, 0.95, diagnosis, transform=ax.transAxes,
        fontsize=9, verticalalignment="top", family="monospace",
        bbox=dict(boxstyle="round", facecolor="wheat", alpha=0.8)
    )

    plt.tight_layout()

    save_path = run_dir / "training_diagnosis.png"
    plt.savefig(save_path, dpi=300, bbox_inches="tight")
    print(f"✓ Saved diagnosis to {save_path}")
    plt.close()

    # Print to console
    print("\n" + "=" * 70)
    print("TRAINING ANALYSIS")
    print("=" * 70)
    print(f"Run directory: {run_dir}")
    print(f"\nReturns: {initial_return:.2f} → {final_return:.2f} (Δ {improvement:.2f})")
    print(f"Entropy: {entropies[0]:.3f} → {final_entropy:.3f}")
    print("\nIssues found:")
    for issue in issues:
        print(f"  {issue}")
    print("=" * 70)


def main() -> None:
    """Command-line entry point."""
    parser = argparse.ArgumentParser(
        description="Analyze INTERCEPT training runs"
    )
    parser.add_argument(
        "run_dir",
        type=str,
        help="Path to training run directory",
    )

    args = parser.parse_args()

    print("Analyzing training run...")
    analyze_training_run(args.run_dir)


if __name__ == "__main__":
    # Default to analyzing a specific run if no args
    import sys
    if len(sys.argv) == 1:
        # Try to find a recent run
        results_dir = Path("results")
        if results_dir.exists():
            runs = sorted(results_dir.glob("intercept_grpo_*"))
            if runs:
                print(f"Analyzing most recent run: {runs[-1]}")
                analyze_training_run(runs[-1])
            else:
                print("No training runs found in results/")
                print("Usage: python -m src.analyze_training <run_dir>")
        else:
            print("No results directory found.")
            print("Usage: python -m src.analyze_training <run_dir>")
    else:
        main()
