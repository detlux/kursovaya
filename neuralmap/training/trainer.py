"""Training loops for DQN (shared) and IQL (per-agent).

Both share the same skeleton: epsilon-greedy data collection, FIFO
replay, target network with periodic hard copy, mean-squared TD loss.
The difference is how many networks/buffers/optimizers exist.

CSV log columns:
    episode, reward, avg_loss, epsilon, success, steps
The CSV is later read by visualization/plot_training.py.
"""
from __future__ import annotations
import csv
import os
import random
import time
from typing import List

import torch
import torch.nn as nn
import torch.nn.functional as F

from env.grid_env import GridEnv, flatten_obs
from planners.dqn_planner import QNetwork
from training.replay_buffer import ReplayBuffer


# --------------------------------------------------------------------------- #
def _epsilon(step: int, eps_start: float, eps_end: float, eps_steps: int) -> float:
    if step >= eps_steps:
        return eps_end
    frac = step / eps_steps
    return eps_start + frac * (eps_end - eps_start)


def _select(net: nn.Module, obs_vec, num_actions, eps) -> int:
    if random.random() < eps:
        return random.randrange(num_actions)
    x = torch.tensor(obs_vec, dtype=torch.float32).unsqueeze(0)
    with torch.no_grad():
        q = net(x).squeeze(0)
    return int(torch.argmax(q).item())


def _td_loss(net, target, batch, gamma) -> torch.Tensor:
    s, a, r, s2, d = zip(*batch)
    s = torch.tensor(s, dtype=torch.float32)
    a = torch.tensor(a, dtype=torch.long).unsqueeze(1)
    r = torch.tensor(r, dtype=torch.float32).unsqueeze(1)
    s2 = torch.tensor(s2, dtype=torch.float32)
    d = torch.tensor(d, dtype=torch.float32).unsqueeze(1)
    q = net(s).gather(1, a)
    with torch.no_grad():
        q_next = target(s2).max(1, keepdim=True)[0]
        tgt = r + gamma * q_next * (1 - d)
    return F.mse_loss(q, tgt)


# --------------------------------------------------------------------------- #
def train_dqn_shared(cfg: dict, out_dir: str) -> str:
    """Train one Q-network shared across all agents."""
    os.makedirs(out_dir, exist_ok=True)
    random.seed(cfg["seed"]); torch.manual_seed(cfg["seed"])
    env = GridEnv(
        grid_width=cfg["grid_width"], grid_height=cfg["grid_height"],
        num_agents=cfg["num_agents"], max_steps=cfg["max_steps"],
        view_radius=cfg["view_radius"],
        collision_penalty=cfg["collision_penalty"],
        coverage_bonus=cfg.get("coverage_bonus", 1.0),
        success_bonus=cfg.get("success_bonus", 20.0),
        step_penalty=cfg.get("step_penalty", 0.1),
    )
    input_dim = env.obs_size; num_actions = env.action_space_size

    net = QNetwork(input_dim, num_actions, hidden=cfg["hidden_units"])
    target = QNetwork(input_dim, num_actions, hidden=cfg["hidden_units"])
    target.load_state_dict(net.state_dict())
    opt = torch.optim.Adam(net.parameters(), lr=cfg["lr"])
    buf = ReplayBuffer(capacity=cfg["buffer_size"], seed=cfg["seed"])

    log_path = os.path.join(out_dir, "training_log.csv")
    log = open(log_path, "w", newline=""); writer = csv.writer(log)
    writer.writerow(["episode", "reward", "avg_loss", "epsilon", "success", "steps"])

    step_count = 0
    t_start = time.time()
    for ep in range(cfg["episodes"]):
        obs = env.reset(seed=ep)
        ep_r = 0.0; losses: List[float] = []; success = False; steps = 0
        for _ in range(env.max_steps):
            eps = _epsilon(step_count, cfg["eps_start"], cfg["eps_end"], cfg["eps_steps"])
            actions: List[int] = []
            obs_vecs = [flatten_obs(o) for o in obs]
            for ov in obs_vecs:
                actions.append(_select(net, ov, num_actions, eps))
            step = env.step(actions)
            steps += 1; ep_r += step.reward
            success = success or step.info.get("success", False)
            next_vecs = [flatten_obs(o) for o in step.observations]
            for i in range(env.num_agents):
                buf.push(obs_vecs[i], actions[i], step.reward,
                         next_vecs[i], step.done)
            obs = step.observations
            step_count += 1

            if len(buf) >= cfg["batch_size"]:
                loss = _td_loss(net, target, buf.sample(cfg["batch_size"]),
                                cfg["gamma"])
                opt.zero_grad(); loss.backward()
                # gradient clipping for stability (deadly-triad mitigation)
                torch.nn.utils.clip_grad_norm_(net.parameters(),
                                                cfg.get("grad_clip", 10.0))
                opt.step()
                losses.append(loss.item())
            if step_count % cfg["target_update"] == 0:
                target.load_state_dict(net.state_dict())
            if step.done:
                break

        avg_loss = sum(losses) / len(losses) if losses else 0.0
        writer.writerow([ep, f"{ep_r:.3f}", f"{avg_loss:.5f}",
                         f"{eps:.3f}", int(success), steps])
        if ep % 25 == 0 or ep == cfg["episodes"] - 1:
            elapsed = time.time() - t_start
            print(f"  ep {ep:4d}  reward {ep_r:8.2f}  loss {avg_loss:.4f}  "
                  f"eps {eps:.3f}  success {int(success)}  ({elapsed:.0f}s)")

    log.close()
    weights = os.path.join(out_dir, "dqn_shared.pt")
    torch.save(net.state_dict(), weights)
    print(f"  saved {weights}")
    print(f"  saved {log_path}")
    return weights


# --------------------------------------------------------------------------- #
def train_iql(cfg: dict, out_dir: str) -> List[str]:
    """Train one INDEPENDENT Q-network per agent (IQL)."""
    os.makedirs(out_dir, exist_ok=True)
    random.seed(cfg["seed"]); torch.manual_seed(cfg["seed"])
    env = GridEnv(
        grid_width=cfg["grid_width"], grid_height=cfg["grid_height"],
        num_agents=cfg["num_agents"], max_steps=cfg["max_steps"],
        view_radius=cfg["view_radius"],
        collision_penalty=cfg["collision_penalty"],
        coverage_bonus=cfg.get("coverage_bonus", 1.0),
        success_bonus=cfg.get("success_bonus", 20.0),
        step_penalty=cfg.get("step_penalty", 0.1),
    )
    input_dim = env.obs_size; num_actions = env.action_space_size
    N = env.num_agents

    nets    = [QNetwork(input_dim, num_actions, hidden=cfg["hidden_units"]) for _ in range(N)]
    targets = [QNetwork(input_dim, num_actions, hidden=cfg["hidden_units"]) for _ in range(N)]
    for i in range(N):
        targets[i].load_state_dict(nets[i].state_dict())
    opts = [torch.optim.Adam(nets[i].parameters(), lr=cfg["lr"]) for i in range(N)]
    bufs = [ReplayBuffer(capacity=cfg["buffer_size"], seed=cfg["seed"] + i)
            for i in range(N)]

    log_path = os.path.join(out_dir, "training_log.csv")
    log = open(log_path, "w", newline=""); writer = csv.writer(log)
    writer.writerow(["episode", "reward", "avg_loss", "epsilon", "success", "steps"])

    step_count = 0
    t_start = time.time()
    for ep in range(cfg["episodes"]):
        obs = env.reset(seed=ep)
        ep_r = 0.0; losses: List[float] = []; success = False; steps = 0
        for _ in range(env.max_steps):
            eps = _epsilon(step_count, cfg["eps_start"], cfg["eps_end"], cfg["eps_steps"])
            obs_vecs = [flatten_obs(o) for o in obs]
            actions = [_select(nets[i], obs_vecs[i], num_actions, eps)
                       for i in range(N)]
            step = env.step(actions)
            steps += 1; ep_r += step.reward
            success = success or step.info.get("success", False)
            next_vecs = [flatten_obs(o) for o in step.observations]
            for i in range(N):
                bufs[i].push(obs_vecs[i], actions[i], step.reward,
                             next_vecs[i], step.done)
            obs = step.observations
            step_count += 1

            for i in range(N):
                if len(bufs[i]) >= cfg["batch_size"]:
                    loss = _td_loss(nets[i], targets[i],
                                    bufs[i].sample(cfg["batch_size"]),
                                    cfg["gamma"])
                    opts[i].zero_grad(); loss.backward()
                    torch.nn.utils.clip_grad_norm_(nets[i].parameters(),
                                                    cfg.get("grad_clip", 10.0))
                    opts[i].step()
                    losses.append(loss.item())
            if step_count % cfg["target_update"] == 0:
                for i in range(N):
                    targets[i].load_state_dict(nets[i].state_dict())
            if step.done:
                break

        avg_loss = sum(losses) / len(losses) if losses else 0.0
        writer.writerow([ep, f"{ep_r:.3f}", f"{avg_loss:.5f}",
                         f"{eps:.3f}", int(success), steps])
        if ep % 25 == 0 or ep == cfg["episodes"] - 1:
            elapsed = time.time() - t_start
            print(f"  ep {ep:4d}  reward {ep_r:8.2f}  loss {avg_loss:.4f}  "
                  f"eps {eps:.3f}  success {int(success)}  ({elapsed:.0f}s)")

    log.close()
    paths: List[str] = []
    for i in range(N):
        p = os.path.join(out_dir, f"iql_agent_{i}.pt")
        torch.save(nets[i].state_dict(), p); paths.append(p)
    print(f"  saved {N} IQL networks to {out_dir}")
    print(f"  saved {log_path}")
    return paths
