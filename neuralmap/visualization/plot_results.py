"""Build success-rate bar chart and decision-time box plot from eval CSVs."""
import argparse
import csv
import os
from typing import Dict, List

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from evaluation.stats import wilson_ci


def _load(csv_path: str):
    rewards, successes, times, steps = [], [], [], []
    with open(csv_path) as f:
        for row in csv.DictReader(f):
            rewards.append(float(row["reward"]))
            successes.append(int(row["success"]))
            times.append(float(row["mean_decision_ms"]))
            steps.append(int(row["steps"]))
    return rewards, successes, times, steps


def plot_results(eval_csvs: Dict[str, str], out_dir: str) -> None:
    os.makedirs(out_dir, exist_ok=True)
    names = list(eval_csvs.keys())
    success_rates, ci_lo, ci_hi, mean_times, all_times = [], [], [], [], []
    n_episodes = 0
    for name, path in eval_csvs.items():
        _, s, t, _ = _load(path)
        n_episodes = len(s) if not n_episodes else n_episodes
        p = sum(s) / len(s) if s else 0.0
        lo, hi = wilson_ci(p, len(s))
        success_rates.append(p); ci_lo.append(lo); ci_hi.append(hi)
        mean_times.append(sum(t) / len(t) if t else 0.0)
        all_times.append(t)

    # --- bar chart: success rate ---------------------------------------
    fig, ax = plt.subplots(figsize=(6.0, 3.8))
    x = list(range(len(names)))
    err_lo = [r - lo for r, lo in zip(success_rates, ci_lo)]
    err_hi = [hi - r for r, hi in zip(success_rates, ci_hi)]
    colors = ["#e07a3a", "#7a3ae0", "#3a7ec1", "#3aaf6e"][:len(names)]
    bars = ax.bar(x, success_rates, yerr=[err_lo, err_hi], capsize=7,
                  color=colors, edgecolor="black", linewidth=0.8)
    ax.set_xticks(x); ax.set_xticklabels(names)
    ax.set_ylabel("Success rate")
    ax.set_ylim(0, 1.05)
    for b, r in zip(bars, success_rates):
        ax.text(b.get_x() + b.get_width() / 2, r + 0.03, f"{r:.2f}",
                ha="center", fontsize=10, fontweight="bold")
    ax.grid(axis="y", linestyle=":", alpha=0.5)
    ax.set_title(f"Success rate ({n_episodes} episodes)")
    fig.tight_layout()
    p1 = os.path.join(out_dir, "success_rate.pdf")
    p1_png = os.path.join(out_dir, "success_rate.png")
    fig.savefig(p1, bbox_inches="tight"); fig.savefig(p1_png, bbox_inches="tight")
    plt.close(fig)
    print(f"saved {p1}, {p1_png}")

    # --- box plot: decision time ---------------------------------------
    fig, ax = plt.subplots(figsize=(6.0, 3.8))
    bp = ax.boxplot(all_times, tick_labels=names, patch_artist=True,
                    widths=0.5, showfliers=False)
    for patch, c in zip(bp["boxes"], colors):
        patch.set_facecolor(c); patch.set_alpha(0.7)
    ax.set_yscale("log")
    ax.set_ylabel("Decision time, ms (log scale)")
    ax.set_title("Per-decision wall-clock time")
    ax.grid(axis="y", which="both", linestyle=":", alpha=0.5)
    fig.tight_layout()
    p2 = os.path.join(out_dir, "decision_time.pdf")
    p2_png = os.path.join(out_dir, "decision_time.png")
    fig.savefig(p2, bbox_inches="tight"); fig.savefig(p2_png, bbox_inches="tight")
    plt.close(fig)
    print(f"saved {p2}, {p2_png}")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--inputs", nargs="+", required=True,
                    help="pairs name=path/to/eval.csv")
    ap.add_argument("--out-dir", required=True)
    a = ap.parse_args()
    inputs: Dict[str, str] = {}
    for pair in a.inputs:
        name, path = pair.split("=", 1)
        inputs[name] = path
    plot_results(inputs, a.out_dir)
