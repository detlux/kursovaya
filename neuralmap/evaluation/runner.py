"""Run N evaluation episodes of any planner list on GridEnv."""
import csv
import os
from time import perf_counter
from typing import List

from env.grid_env import GridEnv
from evaluation.metrics import EpisodeMetrics


def run_episodes(env: GridEnv, planners, num_episodes: int,
                 base_seed: int, csv_path: str) -> List[EpisodeMetrics]:
    """Run `num_episodes`, each with seed = base_seed + i.

    Same seeds across planners ⇒ identical problem instances.
    """
    assert env.num_agents == len(planners), \
        "One planner per agent (use [shared]*N for shared-parameter)."
    os.makedirs(os.path.dirname(csv_path) or ".", exist_ok=True)
    out = open(csv_path, "w", newline=""); writer = csv.writer(out)
    writer.writerow(["episode", "seed", "steps", "reward",
                     "success", "mean_decision_ms"])

    records: List[EpisodeMetrics] = []
    for ep in range(num_episodes):
        seed = base_seed + ep
        obs = env.reset(seed=seed)
        for p in planners:
            p.reset()
        rec = EpisodeMetrics(episode_index=ep, seed=seed,
                             steps=0, total_reward=0.0, success=False)
        for _ in range(env.max_steps):
            actions: List[int] = []
            for i, planner in enumerate(planners):
                t0 = perf_counter()
                a = planner.plan(obs[i])
                rec.decision_times_ms.append((perf_counter() - t0) * 1000.0)
                actions.append(a)
            step = env.step(actions)
            rec.steps += 1; rec.total_reward += step.reward
            obs = step.observations
            if step.done:
                rec.success = step.info.get("success", False)
                break
        writer.writerow([rec.episode_index, rec.seed, rec.steps,
                         f"{rec.total_reward:.3f}", int(rec.success),
                         f"{rec.mean_decision_ms:.4f}"])
        records.append(rec)
    out.close()
    return records
