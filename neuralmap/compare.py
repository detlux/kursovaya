"""End-to-end comparison pipeline.

Pipeline:
    1. (optional) train DQN          -> results/dqn/dqn_shared.pt
    2. (optional) train IQL          -> results/iql/iql_agent_{i}.pt
    3. evaluate A*, DQN, IQL on the same 50 seeds
    4. build training curves, success bars, decision-time boxes, Q-heatmap
    5. write results/summary.csv with the headline numbers

Usage:
    python compare.py                 # full run
    python compare.py --skip-train    # re-use existing weights
    python compare.py --only-dqn      # train DQN only
    python compare.py --only-iql      # train IQL only
"""
import argparse
import csv
import os
import statistics
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from env.grid_env import GridEnv
from evaluation.runner import run_episodes
from evaluation.stats import mann_whitney_u, wilson_ci, two_proportion_z
from planners.astar_planner import CentralizedAStarPlanner
from planners.dqn_planner import DQNPlanner
from planners.iql_planner import IQLPlanner
from visualization.plot_training import plot_training
from visualization.plot_results import plot_results
from visualization.plot_qheatmap import plot_qheatmap


# evaluation settings ------------------------------------------------------- #
EVAL_EPISODES = 50
EVAL_BASE_SEED = 1000
GRID = 8
NUM_AGENTS = 3
VIEW_RADIUS = 3
MAX_STEPS = 50


def _train_if_needed(skip: bool, only_dqn: bool = False,
                     only_iql: bool = False) -> None:
    if skip:
        return
    from train import _load_yaml
    from training.trainer import train_dqn_shared, train_iql
    if not only_iql:
        print("\n=== TRAIN DQN ===")
        cfg = _load_yaml("configs/dqn.yaml")
        train_dqn_shared(cfg, cfg.get("out_dir", "results/dqn"))
    if not only_dqn:
        print("\n=== TRAIN IQL ===")
        cfg = _load_yaml("configs/iql.yaml")
        train_iql(cfg, cfg.get("out_dir", "results/iql"))


def _evaluate_all():
    env = GridEnv(grid_width=GRID, grid_height=GRID, num_agents=NUM_AGENTS,
                  max_steps=MAX_STEPS, view_radius=VIEW_RADIUS)

    print("\n=== EVAL A* ===")
    astar = CentralizedAStarPlanner(num_agents=NUM_AGENTS,
                                    grid_width=GRID, grid_height=GRID)
    rec_a = run_episodes(env, [astar] * NUM_AGENTS,
                         EVAL_EPISODES, EVAL_BASE_SEED,
                         "results/astar/eval.csv")

    print("\n=== EVAL DQN ===")
    dqn = DQNPlanner(input_dim=env.obs_size,
                     num_actions=env.action_space_size,
                     weights_path="results/dqn/dqn_shared.pt", epsilon=0.0)
    rec_d = run_episodes(env, [dqn] * NUM_AGENTS,
                         EVAL_EPISODES, EVAL_BASE_SEED,
                         "results/dqn/eval.csv")

    print("\n=== EVAL IQL ===")
    iql_planners = [
        IQLPlanner(agent_id=i, input_dim=env.obs_size,
                   num_actions=env.action_space_size,
                   weights_path=f"results/iql/iql_agent_{i}.pt",
                   epsilon=0.0)
        for i in range(NUM_AGENTS)
    ]
    rec_i = run_episodes(env, iql_planners, EVAL_EPISODES, EVAL_BASE_SEED,
                         "results/iql/eval.csv")
    return rec_a, rec_d, rec_i


def _summary(rec_a, rec_d, rec_i) -> None:
    rows = []
    for name, rec in [("A*", rec_a), ("DQN", rec_d), ("IQL", rec_i)]:
        succ = sum(1 for r in rec if r.success) / len(rec)
        steps = statistics.mean(r.steps for r in rec)
        times = []
        for r in rec:
            times.extend(r.decision_times_ms)
        mean_t = statistics.mean(times) if times else 0.0
        med_t = statistics.median(times) if times else 0.0
        lo, hi = wilson_ci(succ, len(rec))
        rows.append((name, succ, lo, hi, steps, mean_t, med_t))

    with open("results/summary.csv", "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["planner", "success_rate", "ci_lo", "ci_hi",
                    "avg_episode_length", "mean_decision_ms",
                    "median_decision_ms"])
        for r in rows:
            w.writerow([r[0], f"{r[1]:.3f}", f"{r[2]:.3f}", f"{r[3]:.3f}",
                        f"{r[4]:.2f}", f"{r[5]:.3f}", f"{r[6]:.3f}"])

    print("\n=== SUMMARY ===")
    print(f"{'planner':<8} {'success':>9} {'CI':>16} "
          f"{'avg.len':>9} {'mean ms':>9}")
    for name, succ, lo, hi, steps, mean_t, _ in rows:
        print(f"{name:<8} {succ:>9.3f}  [{lo:.3f},{hi:.3f}]  "
              f"{steps:>9.2f} {mean_t:>9.3f}")

    # pairwise stats
    def times(rec):
        out = []
        for r in rec:
            out.extend(r.decision_times_ms)
        return out

    u, p = mann_whitney_u(times(rec_d), times(rec_a))
    print(f"\nMann-Whitney U  DQN vs A* on decision time: U={u:.1f}, p={p:.4g}")
    u, p = mann_whitney_u(times(rec_i), times(rec_a))
    print(f"Mann-Whitney U  IQL vs A* on decision time: U={u:.1f}, p={p:.4g}")
    z, p = two_proportion_z(sum(r.success for r in rec_d), len(rec_d),
                            sum(r.success for r in rec_a), len(rec_a))
    print(f"two-prop z       DQN vs A* on success:      z={z:.2f}, p={p:.4g}")
    z, p = two_proportion_z(sum(r.success for r in rec_i), len(rec_i),
                            sum(r.success for r in rec_a), len(rec_a))
    print(f"two-prop z       IQL vs A* on success:      z={z:.2f}, p={p:.4g}")


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--skip-train", action="store_true",
                    help="Skip training; use existing weights.")
    ap.add_argument("--only-dqn", action="store_true",
                    help="Train only DQN (skip IQL training).")
    ap.add_argument("--only-iql", action="store_true",
                    help="Train only IQL (skip DQN training, reuse existing).")
    a = ap.parse_args()

    os.makedirs("results", exist_ok=True)
    _train_if_needed(a.skip_train, a.only_dqn, a.only_iql)

    rec_a, rec_d, rec_i = _evaluate_all()
    _summary(rec_a, rec_d, rec_i)

    print("\n=== FIGURES ===")
    plot_training("results/dqn/training_log.csv",
                  "results/training_dqn.pdf", "DQN")
    plot_training("results/dqn/training_log.csv",
                  "results/training_dqn.png", "DQN")
    plot_training("results/iql/training_log.csv",
                  "results/training_iql.pdf", "IQL")
    plot_training("results/iql/training_log.csv",
                  "results/training_iql.png", "IQL")
    plot_results({"A*":   "results/astar/eval.csv",
                  "DQN":  "results/dqn/eval.csv",
                  "IQL":  "results/iql/eval.csv"},
                 out_dir="results")
    plot_qheatmap("results/dqn/dqn_shared.pt",
                  "results/qheatmap.pdf", grid=GRID, view_radius=VIEW_RADIUS)
    plot_qheatmap("results/dqn/dqn_shared.pt",
                  "results/qheatmap.png", grid=GRID, view_radius=VIEW_RADIUS)

    print("\nDone. See results/ for outputs.")


if __name__ == "__main__":
    main()
