"""Abstract Planner interface.

Every planner — symbolic or neural — implements `plan(observation) -> int`.
This single-method contract is what allows classical A* and neural
DQN/IQL to be plugged into the same evaluation harness.
"""
from abc import ABC, abstractmethod
from typing import Any, Dict, Optional


class Planner(ABC):
    """Abstract base class for all planners."""

    @abstractmethod
    def plan(self, observation: Any) -> int:
        """Return a discrete action id given a local observation."""

    def reset(self) -> None:
        """Optional hook called at the start of every episode."""
        pass

    def info(self) -> Dict[str, Any]:
        """Optional metadata recorded by the logger."""
        return {"type": self.__class__.__name__}


class NeuralPlanner(Planner):
    """Planner backed by a torch.nn.Module.

    Subclasses must implement `plan`. Internal-state traces (Q-values,
    action probabilities, latest minibatch loss) are written to
    `last_*` attributes so the logger can record them every step.
    """

    def __init__(self, network=None, device: str = "cpu"):
        self.network = network
        self.device = device
        self.last_q_values: Optional[list] = None
        self.last_action_probs: Optional[list] = None
        self.last_loss: Optional[float] = None

    def load_weights(self, path: str) -> None:
        import torch
        state = torch.load(path, map_location=self.device)
        self.network.load_state_dict(state)
        self.network.eval()

    def info(self) -> Dict[str, Any]:
        meta = super().info()
        if self.network is not None:
            meta["num_parameters"] = sum(
                p.numel() for p in self.network.parameters()
            )
        return meta
