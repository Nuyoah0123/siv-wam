from dataclasses import dataclass
import math
import numpy as np
import torch
from .objectives import gaussian_log_prob


@dataclass
class Transition:
    features: torch.Tensor
    state: torch.Tensor
    next_state: torch.Tensor
    old_mean: torch.Tensor
    active: torch.Tensor
    dt: float
    std: float
    old_log_prob: torch.Tensor


@dataclass
class ActionChunk:
    actions: np.ndarray
    value_maps: torch.Tensor  # Head-camera future frames [T,H,W], excluding current frame.
    transitions: list[Transition]


@dataclass
class StepResult:
    observation: dict
    dense: float
    sparse: float
    done: bool
    success: bool
    executed_steps: int


@torch.no_grad()
def sample_transition(head, features, state, active, dt, std, generator):
    if not math.isfinite(dt) or dt >= 0 or std <= 0 or not math.isfinite(std):
        raise ValueError('Denoising requires dt < 0 and std > 0')
    features = features.detach().float()
    state = state.detach().float()
    active = active.to(device=state.device, dtype=torch.bool)
    mean = state + dt * head(features)
    noise = torch.randn(mean.shape, dtype=torch.float32, device=mean.device, generator=generator)
    nxt = torch.where(active, mean + std * noise, state)
    log_prob = gaussian_log_prob(nxt, mean, std, active)
    if not torch.isfinite(log_prob):
        raise FloatingPointError('Non-finite sampled log probability')
    # Cache frozen features rather than replaying a changing KV cache during optimization.
    record = Transition(features.cpu(), state.cpu(), nxt.cpu(), mean.cpu(), active.cpu(),
                        float(dt), float(std), log_prob.cpu())
    return nxt, record
