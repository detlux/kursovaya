"""Standard FIFO experience replay buffer."""
import random
from collections import deque
from typing import List, Tuple


Transition = Tuple[List[float], int, float, List[float], bool]


class ReplayBuffer:
    """Bounded FIFO buffer of transitions (o, a, r, o', done)."""

    def __init__(self, capacity: int = 50_000, seed: int = 0):
        self.buf: "deque[Transition]" = deque(maxlen=capacity)
        self._rng = random.Random(seed)

    def __len__(self) -> int:
        return len(self.buf)

    def push(self, o, a, r, o_next, done) -> None:
        self.buf.append((o, a, r, o_next, done))

    def sample(self, batch_size: int):
        return self._rng.sample(self.buf, batch_size)
