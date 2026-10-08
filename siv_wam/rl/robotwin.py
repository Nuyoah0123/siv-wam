"""RoboTwin simulator adapter with an explicit 16D EE action/camera contract."""
from contextlib import contextmanager
from copy import deepcopy
import importlib
import os
from pathlib import Path
import random
import sys
import numpy as np
import torch
from .geometry import projected_reward
from .trajectory import StepResult


@contextmanager
def working_directory(path):
    previous = Path.cwd()
    os.chdir(path)
    try:
        yield
    finally:
        os.chdir(previous)


def _xyzw(quaternion, order):
    q = np.asarray(quaternion, dtype=np.float64)
    if order == 'wxyz':
        q = q[[1, 2, 3, 0]]
    elif order != 'xyzw':
        raise ValueError('Quaternion order must be explicitly xyzw or wxyz')
    norm = np.linalg.norm(q)
    if not np.isfinite(norm) or norm < 1e-8:
        raise ValueError('Invalid quaternion in action/initial end-effector pose')
    return q / norm


def decode_ee_action(action, initial_pose, model_order, environment_order, relative_to_initial=True):
    """Translation offsets are in world axes; quaternion composition is q_initial * q_delta."""
    action = np.asarray(action, dtype=np.float64)
    initial_pose = np.asarray(initial_pose, dtype=np.float64)
    if (action.shape != (16,) or initial_pose.shape != (16,)
            or not np.isfinite(action).all() or not np.isfinite(initial_pose).all()):
        raise ValueError('Expected finite [L_xyz,L_quat,L_grip,R_xyz,R_quat,R_grip]')
    result = action.copy()
    for start in (0, 8):
        q = _xyzw(action[start + 3:start + 7], model_order)
        if relative_to_initial:
            p = _xyzw(initial_pose[start + 3:start + 7], environment_order)
            xyz = p[3] * q[:3] + q[3] * p[:3] + np.cross(p[:3], q[:3])
            q = np.r_[xyz, p[3] * q[3] - p[:3] @ q[:3]]
            result[start:start + 3] += initial_pose[start:start + 3]
        if environment_order == 'wxyz':
            q = q[[3, 0, 1, 2]]
        elif environment_order != 'xyzw':
            raise ValueError('Unknown environment quaternion convention')
        result[start + 3:start + 7] = q
        result[start + 7] = np.clip(result[start + 7], 0, 1)
    return result


class RoboTwinEnvironment:
    def __init__(self, settings):
        import yaml
        self.settings = dict(settings)
        self.root = Path(settings['root']).expanduser().resolve()
        if not (self.root / 'envs').is_dir():
            raise FileNotFoundError('Set environment.root to an installed RoboTwin checkout with assets')
        self.task_name = settings['task']
        self.prompt = settings['instruction']
        self.model_order = settings['model_quaternion_order']
        self.environment_order = settings['environment_quaternion_order']
        if self.model_order not in {'xyzw', 'wxyz'} or self.environment_order not in {'xyzw', 'wxyz'}:
            raise ValueError('Explicit quaternion convention required')
        self.relative = settings.get('relative_to_initial', True)
        self.reward_arm = settings.get('reward_arm', 'both')
        if self.reward_arm not in {'left', 'right', 'both'}:
            raise ValueError('reward_arm must be left, right, or both')
        sys.path.insert(0, str(self.root))
        with working_directory(self.root):
            config_path = Path('task_config') / (settings['task_config'] + '.yml')
            self.args = yaml.safe_load(config_path.read_text(encoding='utf-8'))
            catalog = yaml.safe_load(Path('task_config/_embodiment_config.yml').read_text(encoding='utf-8'))
            cameras = yaml.safe_load(Path('task_config/_camera_config.yml').read_text(encoding='utf-8'))
            embodiment = self.args['embodiment']
            if len(embodiment) not in (1, 3):
                raise ValueError('RoboTwin embodiment must have one or three entries')
            left = catalog[embodiment[0]]['file_path']
            right = catalog[embodiment[0 if len(embodiment) == 1 else 1]]['file_path']
            self.args.update(left_robot_file=left, right_robot_file=right,
                             left_embodiment_config=yaml.safe_load((Path(left) / 'config.yml').read_text(encoding='utf-8')),
                             right_embodiment_config=yaml.safe_load((Path(right) / 'config.yml').read_text(encoding='utf-8')),
                             dual_arm_embodied=len(embodiment) == 1)
            if len(embodiment) == 3:
                self.args['embodiment_dis'] = embodiment[2]
            head = cameras[self.args['camera']['head_camera_type']]
            self.args.update(head_camera_h=head['h'], head_camera_w=head['w'], task_name=self.task_name,
                             eval_mode=True, render_freq=0, save_data=False, eval_video_save_dir=None,
                             policy_name='SIV-WAM-StageII')
            # Some RoboTwin task configs omit data_type entirely.  Stage II
            # needs RGB and end-effector poses, so make that contract explicit.
            self.args.setdefault('data_type', {})
            self.args['data_type'].update(rgb=True, endpose=True)
            task_module = importlib.import_module('envs.' + self.task_name)
            self.task_class = getattr(task_module, self.task_name)
        self.env = None

    def reset(self, seed):
        self.close()
        # Same seed for every candidate in a group; do not skip hard/failed scenes silently.
        random.seed(seed)
        np.random.seed(seed)
        with working_directory(self.root):
            self.env = self.task_class()
            self.env.setup_demo(now_ep_num=0, seed=seed, is_test=True, **deepcopy(self.args))
            self.env.set_instruction(instruction=self.prompt)
            obs = self.env.get_obs()
        pose = obs['endpose']
        self.initial_pose = np.r_[pose['left_endpose'], pose['left_gripper'],
                                  pose['right_endpose'], pose['right_gripper']]
        self.observation = self._observation(obs)
        return self.observation

    def _observation(self, raw):
        cameras = raw['observation']
        head = cameras['head_camera']
        return {'images': {key: cameras[key]['rgb'] for key in ('head_camera', 'left_camera', 'right_camera')},
                'K': np.asarray(head['intrinsic_cv']), 'T_cw': np.asarray(head['extrinsic_cv']),
                'prompt': self.prompt}

    def step_chunk(self, chunk):
        if chunk.actions.ndim != 2 or chunk.actions.shape[1] != 16 or len(chunk.actions) < 1:
            raise ValueError('Stage II RoboTwin adapter expects a nonempty [T,16] action chunk')
        if chunk.value_maps.ndim != 3 or len(chunk.value_maps) < 1:
            raise ValueError('Expected predicted head-camera SIV frames [T,H,W]')
        # Use the camera attached to the prediction context, not a later wrist pose.
        original_h, original_w = self.observation['images']['head_camera'].shape[:2]
        map_h, map_w = chunk.value_maps.shape[-2:]
        K = self.observation['K'].copy().astype(np.float32)
        # F.interpolate uses half-pixel centers for image resize.
        sx, sy = map_w / original_w, map_h / original_h
        K[0] *= sx
        K[1] *= sy
        K[0, 2] += (sx - 1) / 2
        K[1, 2] += (sy - 1) / 2
        T_cw = self.observation['T_cw']
        values, success = [], False
        with working_directory(self.root):
            for index, action in enumerate(chunk.actions):
                if self.env.take_action_cnt >= self.env.step_lim or self.env.eval_success:
                    break
                world_action = decode_ee_action(action, self.initial_pose, self.model_order,
                                                self.environment_order, self.relative)
                target = np.stack([world_action[:3], world_action[8:11]])
                if self.reward_arm != 'both':
                    target = target[[0 if self.reward_arm == 'left' else 1]]
                frame = min(index * len(chunk.value_maps) // len(chunk.actions), len(chunk.value_maps) - 1)
                response = projected_reward(chunk.value_maps[frame], target, K, T_cw)
                self.env.take_action(world_action, action_type='ee')
                values.append(float(response.mean()))
                success = bool(self.env.eval_success or self.env.check_success())
                if success:
                    break
            raw = self.env.get_obs()
            done = success or self.env.take_action_cnt >= self.env.step_lim
        self.observation = self._observation(raw)
        return StepResult(self.observation, float(np.mean(values)) if values else 0.0,
                          float(success), done, success, len(values))

    def close(self):
        if self.env is not None:
            with working_directory(self.root):
                self.env.close_env()
            self.env = None
