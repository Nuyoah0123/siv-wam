"""Explicit implementation choices for the paper's underspecified RL stage."""
from dataclasses import asdict, dataclass
import math


@dataclass(frozen=True)
class RLConfig:
    seed: int = 42
    updates: int = 30
    group_size: int = 8
    max_chunks: int = 4
    epochs: int = 2
    learning_rate: float = 0.0003
    clip_epsilon: float = 0.2
    kl_beta: float = 0.01
    target_kl: float = 0.1
    grad_clip: float = 1.0
    discount: float = 0.99
    dense_weight: float = 1.0
    sparse_weight: float = 2.0
    exploration_std: float = 0.25
    min_std: float = 0.01
    save_every: int = 5

    def __post_init__(self):
        for key in ('updates', 'max_chunks', 'epochs', 'save_every'):
            value = getattr(self, key)
            if not isinstance(value, int) or isinstance(value, bool) or value < 1:
                raise ValueError(f'{key} must be a positive integer')
        if not isinstance(self.seed, int) or self.seed < 0:
            raise ValueError('seed must be a nonnegative integer')
        if not isinstance(self.group_size, int) or self.group_size < 2:
            raise ValueError('GRPO requires group_size >= 2')
        for key, value in asdict(self).items():
            if not math.isfinite(value):
                raise ValueError(f'{key} must be finite')
        for key in ('learning_rate', 'target_kl', 'grad_clip', 'exploration_std', 'min_std'):
            if getattr(self, key) <= 0:
                raise ValueError(f'{key} must be positive')
        if not 0 < self.clip_epsilon < 1 or not 0 <= self.discount <= 1:
            raise ValueError('Invalid clipping coefficient or discount')
        if min(self.kl_beta, self.dense_weight, self.sparse_weight) < 0:
            raise ValueError('Reward weights and kl_beta must be nonnegative')
        if self.dense_weight + self.sparse_weight <= 0:
            raise ValueError('At least one reward weight must be positive')
