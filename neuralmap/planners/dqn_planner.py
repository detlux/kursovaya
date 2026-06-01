"""DQN planner: a value-based neural planner with a shared Q-network.

All N agents query the same network on their own local observation.
This represents the parameter-sharing variant of independent
deep Q-learning. The Q-network is a small MLP that fits on CPU
training time.
"""
from typing import Any, Optional

import torch
import torch.nn as nn
import torch.nn.functional as F

from env.grid_env import flatten_obs
from planners.base import NeuralPlanner


class QNetwork(nn.Module):
    """Two-layer MLP Q-network."""

    def __init__(self, input_dim: int, num_actions: int, hidden: int = 128):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(input_dim, hidden),
            nn.ReLU(),
            nn.Linear(hidden, hidden),
            nn.ReLU(),
            nn.Linear(hidden, num_actions),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)


class DQNPlanner(NeuralPlanner):
    """Shared-parameter DQN planner.

    Parameters
    ----------
    input_dim, num_actions : int
        Observation vector length and discrete action set size.
    weights_path : str, optional
        If given, load pre-trained weights from this `.pt` file.
    epsilon : float
        Exploration probability. Set 0.0 at evaluation time.
    device : str
        'cpu' or 'cuda'.
    """

    def __init__(
        self,
        input_dim: int,
        num_actions: int,
        weights_path: Optional[str] = None,
        epsilon: float = 0.0,
        device: str = "cpu",
    ):
        super().__init__(network=QNetwork(input_dim, num_actions),
                         device=device)
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
