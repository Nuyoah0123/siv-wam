"""Small CPU environment for testing real optimizer updates, NOT a robotics benchmark."""
import numpy as np
import torch
from torch import nn
from .geometry import sample_value_map
from .trajectory import ActionChunk, StepResult, sample_transition


class SmokeActor:
    def __init__(self, config):
        self.config = config
        self.model = nn.Module()
        self.model.world = nn.Linear(3, 3, bias=False)
        self.model.mask_proj_out = nn.Linear(3, 3, bias=False)
        self.model.action_proj_out = nn.Linear(3, 2)
        with torch.no_grad():
            self.model.world.weight.copy_(torch.eye(3))
            self.model.mask_proj_out.weight.copy_(torch.eye(3))
            self.model.action_proj_out.weight.zero_()
            self.model.action_proj_out.bias.zero_()
        self.model.eval().requires_grad_(False)
        self.head = self.model.action_proj_out.requires_grad_(True)

    def reset(self, observation):
        pass

    @torch.no_grad()
    def sample(self, observation, generator, world_seed, deterministic=False):
        target = torch.as_tensor(observation['target'], dtype=torch.float32)
        features = self.model.world(torch.cat([target, torch.ones(1)])[None])
        cue = self.model.mask_proj_out(features)[0, :2]
        ys, xs = torch.meshgrid(torch.linspace(-1, 1, 65), torch.linspace(-1, 1, 65), indexing='ij')
        value_map = torch.exp(-((xs - cue[0]) ** 2 + (ys - cue[1]) ** 2) / (2 * 0.25 ** 2))
        state = torch.zeros(1, 2)
        if deterministic:
            action, trace = -self.head(features), []
        else:
            action, record = sample_transition(self.head, features, state, torch.ones_like(state, dtype=torch.bool),
                                                -1, self.config.exploration_std, generator)
            trace = [record]
        return ActionChunk(action.numpy(), value_map[None], trace)


class SmokeEnvironment:
    def reset(self, seed):
        rng = np.random.default_rng(seed)
        self.observation = {'target': np.array([0.5, -0.4], dtype=np.float32) + rng.uniform(-0.1, 0.1, 2)}
        return self.observation

    def step_chunk(self, chunk):
        action = torch.as_tensor(chunk.actions, dtype=torch.float32)
        pixels = (action + 1) * 32
        valid = ((action >= -1) & (action <= 1)).all(-1)
        dense = float(sample_value_map(chunk.value_maps[0], pixels, valid).mean())
        success = bool(np.linalg.norm(chunk.actions[0] - self.observation['target']) < 0.2)
        return StepResult(self.observation, dense, float(success), True, success, 1)

    def close(self):
        pass
