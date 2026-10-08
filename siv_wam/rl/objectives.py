"""GRPO over Gaussian action-denoising transitions, not an ODE marginal density."""
import math
import torch


def group_advantages(returns, epsilon=1e-6):
    values = torch.as_tensor(returns, dtype=torch.float32)
    if values.ndim != 1 or values.numel() < 2 or not torch.isfinite(values).all():
        raise ValueError('Expected at least two finite returns from the SAME initial-state group')
    # A constant-reward group provides no policy-gradient signal.
    return (values - values.mean()) / values.std(unbiased=False).clamp_min(epsilon)


def gaussian_log_prob(value, mean, std, active):
    if value.shape != mean.shape or active.shape != mean.shape or active.dtype != torch.bool:
        raise ValueError('Value, mean and boolean active mask must have identical shapes')
    if not active.any() or not math.isfinite(float(std)) or std <= 0:
        raise ValueError('Need active coordinates and strictly positive finite variance')
    terms = -0.5 * ((value.float() - mean.float()) / std).square() - math.log(std) - 0.5 * math.log(2 * math.pi)
    # Sum, never mean: this is the joint density of the active transition coordinates.
    return terms[active].sum()


def gaussian_kl(mean, reference_mean, std, active):
    return (0.5 * ((mean.float() - reference_mean.float()) / std).square())[active].sum()


def clipped_surrogate(new_log_prob, old_log_prob, advantage, clip_epsilon):
    log_ratio = new_log_prob - old_log_prob.detach()
    if not torch.isfinite(log_ratio).all() or (log_ratio.abs() > 30).any():
        raise FloatingPointError('Unstable importance ratio; reduce LR/exploration horizon')
    ratio = log_ratio.exp()
    advantage = torch.as_tensor(advantage, device=ratio.device).detach()
    objective = torch.minimum(ratio * advantage,
                              ratio.clamp(1 - clip_epsilon, 1 + clip_epsilon) * advantage)
    return -objective.mean(), ratio.detach()
