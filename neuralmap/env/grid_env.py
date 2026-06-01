"""Cooperative Navigation: a discrete grid environment.

N agents move on a W x H grid and try to cover N landmark cells.
Each agent has 5 actions: up, down, left, right, stay.

The shared per-step reward is

    R(s, a) = -step_penalty
              + coverage_bonus  * (# landmarks currently occupied)
              - collision_penalty * (# pairs of agents in same cell)
              + success_bonus    on the terminal step if all covered

An episode ends successfully once every landmark is occupied by at
least one agent, or after max_steps without success.
"""
from __future__ import annotations
from dataclasses import dataclass, field
from typing import List, Optional, Tuple
import random


# action ids
UP, DOWN, LEFT, RIGHT, STAY = 0, 1, 2, 3, 4
ACTIONS = (UP, DOWN, LEFT, RIGHT, STAY)
ACTION_DELTA = {
    UP:    (0, -1),
    DOWN:  (0, 1),
    LEFT:  (-1, 0),
    RIGHT: (1, 0),
    STAY:  (0, 0),
}


@dataclass
class StepResult:
    """Transition produced by GridEnv.step()."""
    observations: List[dict]
    reward: float
    done: bool
    info: dict = field(default_factory=dict)


class GridEnv:
    """Cooperative Navigation on a grid.

    Parameters
    ----------
    grid_width, grid_height : int
        Size of the rectangular grid.
    num_agents : int
        Number of agents = number of landmarks.
    max_steps : int
        Maximum episode length.
    view_radius : int
        Local observation half-window. Agent sees a
        (2*view_radius + 1) x (2*view_radius + 1) area around itself.
    collision_penalty : float
        Penalty added (negative) per colliding pair per step.
    """

    def __init__(
        self,
        grid_width: int = 8,
        grid_height: int = 8,
        num_agents: int = 3,
        max_steps: int = 50,
        view_radius: int = 2,
        collision_penalty: float = 0.5,
        coverage_bonus: float = 1.0,
        success_bonus: float = 20.0,
        step_penalty: float = 0.1,
    ):
        self.grid_width = grid_width
        self.grid_height = grid_height
        self._num_agents = num_agents
        self.max_steps = max_steps
        self.view_radius = view_radius
        self.collision_penalty = collision_penalty
        self.coverage_bonus = coverage_bonus
        self.success_bonus = success_bonus
        self.step_penalty = step_penalty

        self.agents: List[Tuple[int, int]] = []
        self.landmarks: List[Tuple[int, int]] = []
        self._t: int = 0
        self._rng = random.Random()

    # ---- properties ----------------------------------------------------
    @property
    def num_agents(self) -> int:
        return self._num_agents

    @property
    def action_space_size(self) -> int:
        return 5

    @property
    def obs_size(self) -> int:
        """Flattened observation vector length."""
        side = 2 * self.view_radius + 1
        return side * side * 3   # 3 channels: free, other_agent, landmark

    # ---- main API ------------------------------------------------------
    def reset(self, seed: Optional[int] = None) -> List[dict]:
        if seed is not None:
            self._rng = random.Random(seed)
        positions = self._sample_unique(self._num_agents * 2)
        self.agents = positions[: self._num_agents]
        self.landmarks = positions[self._num_agents:]
        self._t = 0
        return [self.observe(i) for i in range(self._num_agents)]

    def step(self, actions: List[int]) -> StepResult:
        assert len(actions) == self._num_agents
        new_positions = []
        for (x, y), a in zip(self.agents, actions):
            dx, dy = ACTION_DELTA[a]
            nx = max(0, min(self.grid_width - 1, x + dx))
            ny = max(0, min(self.grid_height - 1, y + dy))
            new_positions.append((nx, ny))
        self.agents = new_positions
        self._t += 1

        # shared reward — pure coverage-based, no distance shaping
        # 1) constant time penalty (encourages reaching the goal quickly)
        reward = -self.step_penalty
        # 2) coverage bonus: +bonus for every landmark currently occupied
        agent_set = set(self.agents)
        covered = sum(1 for l in self.landmarks if l in agent_set)
        reward += self.coverage_bonus * covered
        # 3) collision penalty
        for i in range(self._num_agents):
            for j in range(i + 1, self._num_agents):
                if self.agents[i] == self.agents[j]:
                    reward -= self.collision_penalty

        success = self._all_landmarks_covered()
        # 4) terminal success bonus to make goal salient
        if success:
            reward += self.success_bonus
        done = success or self._t >= self.max_steps
        return StepResult(
            observations=[self.observe(i) for i in range(self._num_agents)],
            reward=reward,
            done=done,
            info={"success": success, "t": self._t},
        )

    def observe(self, agent_id: int) -> dict:
        """Return a local observation for agent_id.

        The observation is a dict with:
          - 'local'      : 3D list of shape (side, side, 3) one-hot.
                           Channels: 0 = free cell, 1 = other agent, 2 = landmark.
                           Cells outside the grid are all zeros.
          - 'self_xy'    : (x, y) of this agent (global coordinates).
          - 'agents'     : list of all agent positions (for centralized A*).
          - 'landmarks'  : list of all landmark positions.
        """
        cx, cy = self.agents[agent_id]
        r = self.view_radius
        side = 2 * r + 1
        local = [[[0, 0, 0] for _ in range(side)] for _ in range(side)]
        for i in range(side):
            for j in range(side):
                gx, gy = cx + i - r, cy + j - r
                if not (0 <= gx < self.grid_width and 0 <= gy < self.grid_height):
                    continue
                local[i][j][0] = 1
                for k, (ax, ay) in enumerate(self.agents):
                    if k != agent_id and (ax, ay) == (gx, gy):
                        local[i][j][0] = 0
                        local[i][j][1] = 1
                if (gx, gy) in self.landmarks:
                    local[i][j][2] = 1
        return {
            "local": local,
            "self_xy": (cx, cy),
            "agents": list(self.agents),
            "landmarks": list(self.landmarks),
        }

    def is_terminal(self) -> bool:
        return self._all_landmarks_covered() or self._t >= self.max_steps

    # ---- helpers -------------------------------------------------------
    def _all_landmarks_covered(self) -> bool:
        a = set(self.agents)
        return all(l in a for l in self.landmarks)

    def _sample_unique(self, n: int) -> List[Tuple[int, int]]:
        chosen: set = set()
        while len(chosen) < n:
            chosen.add((
                self._rng.randrange(self.grid_width),
                self._rng.randrange(self.grid_height),
            ))
        return list(chosen)


def flatten_obs(obs: dict) -> List[float]:
    """Flatten obs['local'] into a 1D vector of length side*side*3."""
    flat: List[float] = []
    for row in obs["local"]:
        for cell in row:
            flat.extend(cell)
    return flat
