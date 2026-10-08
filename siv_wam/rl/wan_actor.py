"""Stage II adapter for the existing Wan world/value model and action projection.

Only the final action projection is trained. World and SIV generation, token features,
VAE and language encoders stay frozen. Each decision rebuilds a short context cache
from current RGB, predicted SIV and the previous commanded action chunk.
"""
from copy import deepcopy
import math
from pathlib import Path
import numpy as np
import torch
from .trajectory import ActionChunk, sample_transition


class WanStage2Actor:
    def __init__(self, settings, config, output_dir):
        from siv_wam.configs import SIV_CONFIGS
        from siv_wam.server import SIVWAMServer
        from siv_wam.utils import data_seq_to_patch
        self.to_patch = data_seq_to_patch
        self.config = config
        if not torch.cuda.is_available():
            raise RuntimeError('The Wan/RoboTwin backend requires CUDA; use smoke for CPU verification')
        if torch.distributed.is_initialized():
            raise RuntimeError('Stage II currently supports one process/GPU; do not launch through torchrun')
        job = deepcopy(SIV_CONFIGS[settings.get('config_name', 'robotwin_b2')])
        job.wan22_pretrained_model_name_or_path = str(Path(settings['pretrained_path']).expanduser().resolve())
        job.transformer_path = str(Path(settings['transformer_path']).expanduser().resolve())
        job.local_rank = int(settings.get('gpu', 0))
        job.rank = 0
        job.save_root = str(Path(output_dir).resolve())
        job.guidance_scale = job.action_guidance_scale = 1.0
        job.enable_offload = True
        job.attn_window = 8
        job.num_inference_steps = int(settings.get('world_steps', 25))
        job.action_num_inference_steps = int(settings.get('action_steps', 20))
        if min(job.num_inference_steps, job.action_num_inference_steps) < 1:
            raise ValueError('Denoising step counts must be positive')
        if job.frame_chunk_size != 2 or job.env_type != 'robotwin_tshape' or len(job.used_action_channel_ids) != 16:
            raise ValueError('This adapter requires RoboTwin T-layout, 2 latent frames and 16 effective EE channels')
        self.server = SIVWAMServer(job)
        self.model = self.server.transformer.eval().requires_grad_(False)
        # Inference attention has explicit context caching; use SDPA rather than training Flex masks.
        class InferenceSDPA(torch.nn.Module):
            def forward(self, q, k, v):
                return torch.nn.functional.scaled_dot_product_attention(
                    q.transpose(1, 2), k.transpose(1, 2), v.transpose(1, 2)).transpose(1, 2)
        for block in self.model.blocks:
            block.attn1.attn_op = InferenceSDPA()
            block.attn2.attn_op = InferenceSDPA()
        if not any(torch.count_nonzero(p).item() for p in self.model.mask_proj_out.parameters()):
            raise ValueError('SIV head is zero-initialized; supply a trained Stage I/SIV checkpoint')
        self.head = self.model.action_proj_out.float().requires_grad_(True)
        self.server.vae.eval().requires_grad_(False)
        self.server.text_encoder.eval().requires_grad_(False)
        if self.server.streaming_vae_half is not None:
            self.server.streaming_vae_half.vae.eval().requires_grad_(False)
        self.device, self.dtype = self.server.device, self.server.dtype
        self.job = job

    @torch.no_grad()
    def reset(self, observation):
        s = self.server
        s.text_encoder.to(self.device)
        try:
            s._reset(prompt=observation['prompt'])
        finally:
            s.text_encoder.to('cpu')
        self.history = torch.zeros(1, self.job.action_dim, 1, self.job.action_per_frame, 1,
                                   device=self.device, dtype=torch.float32)

    def _video_input(self, value, timestep, frame=0):
        inputs = self.server._prepare_latent_input(value.clone(), None, latent_t=timestep, frame_st_id=frame)
        return self.server._repeat_input_for_cfg(inputs['latent_res_lst'])

    @torch.no_grad()
    def _world_prediction(self, observation, seed):
        s = self.server
        images = observation['images']
        formatted = dict(zip(self.job.obs_cam_keys, [images[k] for k in ('head_camera', 'left_camera', 'right_camera')]))
        s.vae.to(self.device)
        s.streaming_vae_half.vae.to(self.device)
        try:
            s.streaming_vae.clear_cache()
            s.streaming_vae_half.clear_cache()
            current = s._encode_obs({'obs': [formatted]})
            s.streaming_vae.clear_cache()
            s.streaming_vae_half.clear_cache()
            blank = s._encode_obs({'obs': [{key: np.zeros_like(value) for key, value in formatted.items()}]})
            rng = torch.Generator(device=self.device).manual_seed(seed)
            shape = (1, current.shape[1], 2, current.shape[-2], current.shape[-1])
            video = torch.randn(shape, generator=rng, device=self.device, dtype=self.dtype)
            maps = torch.randn(shape, generator=rng, device=self.device, dtype=self.dtype)
            s.scheduler.set_timesteps(self.job.num_inference_steps)
            # No future RGB KV entries survive this call: this cache name is deliberately unallocated.
            for t in s.scheduler.timesteps:
                video[:, :, :1] = current
                maps[:, :, :1] = blank
                inputs = self._video_input(video, t)
                inputs['timesteps'][:, :1] = 0
                inputs['noisy_mask_latents'] = maps
                video_velocity, map_velocity = self.model(inputs, action_mode=False, cache_name='stage2_world')
                video_velocity = self.to_patch(self.job.patch_size, video_velocity, 2, shape[-2], shape[-1], batch_size=1)
                map_velocity = self.to_patch(self.job.patch_size, map_velocity, 2, shape[-2], shape[-1], batch_size=1)
                video = s.scheduler.step(video_velocity, t, video)
                maps = s.scheduler.step(map_velocity, t, maps)
            maps[:, :, :1] = blank
            # The provided encoder packs wrist latents ABOVE head latents; decode the head separately.
            head_height = self.job.height // 16
            head_maps = maps[:, :, :, -head_height:, :]
            mean = torch.as_tensor(s.vae.config.latents_mean, device=self.device).view(1, -1, 1, 1, 1)
            std = torch.as_tensor(s.vae.config.latents_std, device=self.device).view(1, -1, 1, 1, 1)
            decoded = s.vae.decode((head_maps * std + mean).to(self.dtype)).sample.float()
            # Three-channel map image -> scalar response; no per-map max normalization.
            value_maps = ((decoded[0] + 1) / 2).clamp(0, 1).mean(0)[1:].cpu()
            if value_maps.shape[0] < 1:
                raise RuntimeError('VAE produced no future SIV frames')
            return current, maps[:, :, 1:].clone(), value_maps
        finally:
            s.streaming_vae.clear_cache()
            s.streaming_vae_half.clear_cache()
            s.vae.to('cpu')
            s.streaming_vae_half.vae.to('cpu')

    @torch.no_grad()
    def _action_context(self, current, future_siv):
        s = self.server
        self.model.clear_cache(s.cache_name)
        self.model(self._video_input(current, 0), update_cache=2, cache_name=s.cache_name)
        # Insert only predicted SIV into the future-facing context, never future RGB tokens.
        self.model(self._video_input(future_siv, 0, frame=1), update_cache=2, cache_name=s.cache_name)

    @torch.no_grad()
    def sample(self, observation, generator, world_seed, deterministic=False):
        s = self.server
        current, future_siv, value_maps = self._world_prediction(observation, world_seed)
        self._action_context(current, future_siv)
        channels, per_frame = self.job.action_dim, self.job.action_per_frame
        state = torch.randn(1, channels, 2, per_frame, 1, device=self.device, generator=generator)
        state[:, :, :1] = self.history
        used = torch.tensor(self.job.used_action_channel_ids, device=self.device)
        active = torch.zeros(2 * per_frame, channels, device=self.device, dtype=torch.bool)
        active[per_frame:, used] = True
        channel_mask = torch.zeros(channels, dtype=torch.bool, device=self.device)
        channel_mask[used] = True
        state[:, ~channel_mask] = 0
        transitions = []
        s.action_scheduler.set_timesteps(self.job.action_num_inference_steps)
        sigmas = s.action_scheduler.sigmas.tolist() + [0.0]
        for index, timestep in enumerate(s.action_scheduler.timesteps):
            inputs = s._prepare_latent_input(None, state.clone().to(self.dtype), action_t=timestep,
                                             action_cond=self.history.to(self.dtype), frame_st_id=0)
            inputs = s._repeat_input_for_cfg(inputs['action_res_lst'])
            features = self.model(inputs, action_mode=True, cache_name=s.cache_name,
                                  return_action_features=True).reshape(2 * per_frame, -1).float()
            flat = state.permute(0, 2, 3, 4, 1).reshape(2 * per_frame, channels)
            dt = sigmas[index + 1] - sigmas[index]
            std = max(self.config.min_std, self.config.exploration_std * math.sqrt(-dt))
            if deterministic:
                nxt = torch.where(active, flat + dt * self.head(features), flat)
            else:
                nxt, record = sample_transition(self.head, features, flat, active, dt, std, generator)
                transitions.append(record)
            state = nxt.reshape(1, 2, per_frame, 1, channels).permute(0, 4, 1, 2, 3).contiguous()
        self.history = state[:, :, 1:].clone()
        actions = state[0, :, 1, :, 0].T
        q01 = torch.as_tensor(self.job.norm_stat['q01'], device=self.device)
        q99 = torch.as_tensor(self.job.norm_stat['q99'], device=self.device)
        physical = (actions + 1) / 2 * (q99 - q01 + 1e-6) + q01
        return ActionChunk(physical[:, used].cpu().numpy(), value_maps, transitions)
