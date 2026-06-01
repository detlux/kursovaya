"""Evaluate a single planner on GridEnv for N episodes.

Usage:
    python eval.py --algo astar --episodes 50 --out results/astar/eval.csv
    python eval.py --algo dqn   --weights results/dqn/dqn_shared.pt \
                   --episodes 50 --out results/dqn/eval.csv
    python eval.py --algo iql   --weights-dir results/iql \
                   --episodes 50 --out results/iql/eval.csv
"""
import argparse
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from env.grid_env import GridEnv
from evaluation.runner import run_episodes
from planners.astar_planner import CentralizedAStarPlanner
from planners.dqn_planner import DQNPlanner
from planners.iql_planner import IQLPlanner


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--algo", choices=["astar", "dqn", "iql"], required=True)
    ap.add_argument("--weights", help="DQN weights path", default=None)
    ap.add_argument("--weights-dir", help="IQL weights directory", default=None)
    ap.add_argument("--episodes", type=int, default=50)
    ap.add_argument("--base-seed", type=int, default=1000)
    ap.add_argument("--grid", type=int, default=8)
    ap.add_argument("--num-agents", type=int, default=3)
    ap.add_argument("--max-steps", type=int, default=50)
    ap.add_argument("--view-radius", type=int, default=3)
    ap.add_argument("--out", required=True)
    a = ap.parse_args()

    env = GridEnv(grid_width=a.grid, grid_height=a.grid,
                  num_agents=a.num_agents, max_steps=a.max_steps,
                  view_radius=a.view_radius)

    if a.algo == "astar":
        shared = CentralizedAStarPlanner(num_agents=env.num_agents,
                                         grid_width=env.grid_width,
                                         grid_height=env.grid_height)
        planners = [shared] * env.num_agents
    elif a.algo == "dqn":
        assert a.weights, "--weights required for DQN"
        shared = DQNPlanner(input_dim=env.obs_size,
                            num_actions=env.action_space_size,
                            weights_path=a.weights, epsilon=0.0)
        planners = [shared] * env.num_agents
    else:  # iql
        assert a.weights_dir, "--weights-dir required for IQL"
        planners = []
        for i in range(env.num_agents):
            p = IQLPlanner(agent_id=i,
                           input_dim=env.obs_size,
                           num_actions=env.action_space_size,
                           weights_path=os.path.join(a.weights_dir,
                                                     f"iql_agent_{i}.pt"),
                           epsilon=0.0)
            planners.append(p)

    print(f"== eval {a.algo}, {a.episodes} episodes ==")
    records = run_episodes(env, planners, a.episodes, a.base_seed, a.out)
    succ = sum(1 for r in records if r.success) / len(records)
    steps = sum(r.steps for r in records) / len(records)
    times = []
    for r in records:
        times.extend(r.decision_times_ms)
    mean_t = sum(times) / len(times) if times else 0.0
    print(f"  success rate     : {succ:.3f}")
    print(f"  avg episode len  : {steps:.2f}")
    print(f"  avg decision ms  : {mean_t:.3f}")
    print(f"  wrote {a.out}")


if __name__ == "__main__":
    main()
