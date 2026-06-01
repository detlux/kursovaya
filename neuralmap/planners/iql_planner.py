"""IQL: Independent Q-Learning.

Each agent maintains its own Q-network and learns independently,
treating the other agents as part of the environment. No parameter
sharing is used and no centralised critic is involved. Independent
learners are subject to a non-stationarity effect: the environment
seen by one agent shifts whenever any other agent updates its policy.

An evaluation instantiates one IQLPlanner per agent, each loading
its own weight file `iql_agent_<i>.pt`.
"""
from typing import Any, Optional

import torch
import torch.nn.functional as F

from env.grid_env import flatten_obs
from planners.base import NeuralPlanner
from planners.dqn_planner import QNetwork


class IQLPlanner(NeuralPlanner):
    """One independent Q-network per agent.

    Parameters
    ----------
    agent_id : int
        Which agent this planner instance controls.
    input_dim, num_actions : int
        Observation vector length and action set size.
    weights_path : str, optional
        Path to this agent's `.pt` file (NOT shared across agents).
    epsilon : float
        Exploration probability at evaluation time (usually 0.0).
    device : str
    """

    def __init__(
        self,
        agent_id: int,
        input_dim: int,
        num_actions: int,
        weights_path: Optional[str] = None,
        epsilon: float = 0.0,
        device: str = "cpu",
    ):
        super().__init__(network=QNetwork(input_dim, num_actions),
                         device=device)
        self.agent_id = agent_id
        self.num_actions = num_actions
        self.epsilon = epsilon
        self.network.to(device)
        if weights_path is not None:
            self.load_weights(weights_path)

    def plan(self, observation: Any) -> int:
        flat = flatten_obs(observation)
        x = torch.tensor(flat, dtype=torch.float32, device=self.device)
        with torch.no_grad():
            q = self.network(x.unsqueeze(0)).squeeze(0)
        self.last_q_values = q.cpu().tolist()
        self.last_action_probs = F.softmax(q, dim=-1).cpu().tolist()
        if self.epsilon > 0 and torch.rand(1).item() < self.epsilon:
            return int(torch.randint(self.num_actions, (1,)).item())
        return int(torch.argmax(q).item())

    def info(self):
        meta = super().info()
        meta["agent_id"] = self.agent_id
        return meta
