"""Plot a training-curve figure from a `training_log.csv`."""
import argparse
import csv

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt


def plot_training(csv_path: str, out_path: str, title: str = "") -> None:
    episodes, rewards, losses = [], [], []
    with open(csv_path) as f:
        reader = csv.DictReader(f)
        for row in reader:
            episodes.append(int(row["episode"]))
            rewards.append(float(row["reward"]))
            losses.append(float(row["avg_loss"]))

    fig, ax1 = plt.subplots(figsize=(6.2, 3.8))
    c1, c2 = "#3a7ec1", "#e07a3a"
    ax1.plot(episodes, rewards, color=c1, lw=1.0)
    ax1.set_xlabel("Training episode")
    ax1.set_ylabel("Episode reward", color=c1)
    ax1.tick_params(axis="y", labelcolor=c1)
    ax2 = ax1.twinx()
    ax2.plot(episodes, losses, color=c2, lw=1.0)
    ax2.set_ylabel("Avg. TD loss", color=c2)
    ax2.tick_params(axis="y", labelcolor=c2)
    if title:
        ax1.set_title(title)
    ax1.grid(linestyle=":", alpha=0.5)
    fig.tight_layout()
    fig.savefig(out_path, bbox_inches="tight")
    plt.close(fig)
    print(f"saved {out_path}")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--csv", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--title", default="")
    a = ap.parse_args()
    plot_training(a.csv, a.out, a.title)
