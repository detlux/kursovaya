"""Q-value heatmap: max_a Q_theta(o(x,y), a) over the grid, fixed landmark."""
import argparse

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import torch

from env.grid_env import GridEnv, flatten_obs
from planners.dqn_planner import QNetwork


def plot_qheatmap(weights_path: str, out_path: str,
                  grid: int = 8, view_radius: int = 2) -> None:
    """Place a synthetic landmark and probe the network for every cell."""
    env = GridEnv(grid_width=grid, grid_height=grid, num_agents=3,
                  view_radius=view_radius)
    net = QNetwork(env.obs_size, env.action_space_size, hidden=128)
    net.load_state_dict(torch.load(weights_path, map_location="cpu"))
    net.eval()

    lx, ly = grid - 1, 1
    v = [[0.0] * grid for _ in range(grid)]
    for x in range(grid):
        for y in range(grid):
            # set agent 0 at (x, y); other agents far away; one landmark at (lx, ly)
            env.agents = [(x, y), (0, 0), (grid - 1, grid - 1)]
            env.landmarks = [(lx, ly),
                             (grid - 1, 0), (0, grid - 1)]  # other landmarks far
            obs = env.observe(0)
            xv = torch.tensor(flatten_obs(obs), dtype=torch.float32)
            with torch.no_grad():
                q = net(xv.unsqueeze(0)).squeeze(0)
            v[y][x] = float(q.max())

    fig, ax = plt.subplots(figsize=(5.0, 4.4))
    im = ax.imshow(v, origin="lower", cmap="viridis")
    ax.scatter([lx], [ly], marker="*", s=260, c="red",
               edgecolor="white", linewidth=0.8, label="landmark")
    ax.set_xlabel("x"); ax.set_ylabel("y")
    ax.legend(loc="upper left", fontsize=8)
    fig.colorbar(im, ax=ax, label=r"$\max_a Q_\theta(s,a)$",
                 fraction=0.046, pad=0.04)
    fig.tight_layout()
    fig.savefig(out_path, bbox_inches="tight")
    plt.close(fig)
    print(f"saved {out_path}")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--weights", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--grid", type=int, default=8)
    a = ap.parse_args()
    plot_qheatmap(a.weights, a.out, grid=a.grid)
