"""Per-episode metrics container."""
from dataclasses import dataclass, field
from typing import List


@dataclass
class EpisodeMetrics:
    episode_index: int
    seed: int
    steps: int
    total_reward: float
    success: bool
    decision_times_ms: List[float] = field(default_factory=list)

    @property
    def mean_decision_ms(self) -> float:
        if not self.decision_times_ms:
            return 0.0
        return sum(self.decision_times_ms) / len(self.decision_times_ms)
