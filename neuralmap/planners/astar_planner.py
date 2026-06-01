"""Centralized A* planner: a classical reference baseline.

Treats the joint configuration of all agents as one search node.
Heuristic: the sum over landmarks of the Manhattan distance to the
nearest agent. Solution quality is high on small grids; the joint
action space scales as 5^N, which limits the method to small teams.
"""
import heapq
from itertools import product
from typing import Any, Dict, List, Tuple, Optional

from env.grid_env import ACTION_DELTA
from planners.base import Planner


def _manhattan(a: Tuple[int, int], b: Tuple[int, int]) -> int:
    return abs(a[0] - b[0]) + abs(a[1] - b[1])


class CentralizedAStarPlanner(Planner):
    """A single shared planner instance for all N agents.

    The first agent's call triggers the joint plan computation;
    subsequent agents in the same step retrieve their action from
    the cached joint trajectory.
    """

    def __init__(
        self,
        num_agents: int,
        grid_width: int,
        grid_height: int,
        collision_penalty: float = 1.0,
        max_expansions: int = 20000,
    ):
        self.num_agents = num_agents
        self.grid_width = grid_width
        self.grid_height = grid_height
        self.collision_penalty = collision_penalty
        self.max_expansions = max_expansions
        self._cached_joint_actions: List[Tuple[int, ...]] = []
        self._step: int = 0
        # joint action space enumerated once for speed
        self._joint_actions = list(product(range(5), repeat=num_agents))

    def reset(self) -> None:
        self._cached_joint_actions = []
        self._step = 0

    # --- search helpers ------------------------------------------------
    def _apply_joint(self, positions, joint_action):
        out = []
        for (x, y), a in zip(positions, joint_action):
            dx, dy = ACTION_DELTA[a]
            out.append((
                max(0, min(self.grid_width - 1, x + dx)),
                max(0, min(self.grid_height - 1, y + dy)),
            ))
        return tuple(out)

    def _heuristic(self, positions, landmarks):
        return sum(min(_manhattan(p, l) for p in positions) for l in landmarks)

    def _all_covered(self, positions, landmarks):
        a = set(positions)
        return all(l in a for l in landmarks)

    def _search(self, start: Tuple, landmarks: Tuple) -> Optional[List]:
        counter = 0
        open_q = []
        heapq.heappush(
            open_q,
            (self._heuristic(start, landmarks), 0, counter, start, ()),
        )
        best_g: Dict = {start: 0}
        expansions = 0
        while open_q and expansions < self.max_expansions:
            _, g, _, cur, path = heapq.heappop(open_q)
            if self._all_covered(cur, landmarks):
                return list(path)
            expansions += 1
            for ja in self._joint_actions:
                nxt = self._apply_joint(cur, ja)
                cols = 0
                for i in range(self.num_agents):
                    for j in range(i + 1, self.num_agents):
                        if nxt[i] == nxt[j]:
                            cols += 1
                tg = g + 1 + cols * self.collision_penalty
                if tg < best_g.get(nxt, float("inf")):
                    best_g[nxt] = tg
                    counter += 1
                    heapq.heappush(
                        open_q,
                        (tg + self._heuristic(nxt, landmarks),
                         tg, counter, nxt, path + (ja,)),
                    )
        return None  # search budget exhausted

    # --- Planner API ---------------------------------------------------
    def plan(self, observation: Any) -> int:
        # build/refresh cached plan when agent 0 is queried at a new step
        if self._step >= len(self._cached_joint_actions):
            start = tuple(observation["agents"])
            landmarks = tuple(observation["landmarks"])
            plan = self._search(start, landmarks)
            self._cached_joint_actions = plan or []
            self._step = 0

        if not self._cached_joint_actions or \
                self._step >= len(self._cached_joint_actions):
            return 4  # STAY

        joint = self._cached_joint_actions[self._step]
        sx, sy = observation["self_xy"]
        try:
            idx = observation["agents"].index((sx, sy))
        except ValueError:
            idx = 0
        action = joint[idx]
        if idx == self.num_agents - 1:
            self._step += 1
        return action
