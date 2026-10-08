import os
from functools import partial
from multiprocessing import Pool
from pathlib import Path

import numpy as np
import packaging.version
import torch
from einops import rearrange
from lerobot.constants import HF_LEROBOT_HOME
from lerobot.datasets.compute_stats import aggregate_stats
from lerobot.datasets.lerobot_dataset import LeRobotDataset, LeRobotDatasetMetadata
from lerobot.datasets.utils import get_episode_data_index
from scipy.spatial.transform import Rotation as R
from torch.utils.data import DataLoader
from tqdm import tqdm


def recursive_find_file(directory, filename="info.json"):
    result = []
    try:
        for root, dirs, files in os.walk(directory):
            if filename in files:
                full_path = os.path.join(root, filename)
                result.append(full_path)
    except PermissionError:
        print(f"Error: can not access {directory}")
    except Exception as e:  # noqa: BLE001
        print(f"Error: {e}")
    return result


def construct_lerobot(
    repo_id,
    config,
):
    return LatentLeRobotDataset(
        repo_id=repo_id,
        config=config,
    )


def construct_lerobot_multi_processor(
    config,
    num_init_worker=None,
):
    if num_init_worker is None:
        # Dataset construction calls HuggingFace's parquet resolver. Keep
        # this pool bounded: the old default (128) can exhaust file
        # descriptors and overload the storage server when there are many
        # RoboTwin tasks.
        num_init_worker = getattr(config, "dataset_init_workers", None)
    if num_init_worker is None:
        num_init_worker = min(getattr(config, "load_worker", 8), os.cpu_count() or 1)
    num_init_worker = max(1, int(num_init_worker))
    datasets_out_lst = []
    construct_func = partial(
        construct_lerobot,
        config=config,
    )
    repo_list = recursive_find_file(config.dataset_path, "info.json")
    repo_list = [v.split("/meta/info.json")[0] for v in repo_list]
    with Pool(num_init_worker) as pool:
        datasets_out_lst = pool.map(construct_func, repo_list)

    return datasets_out_lst


def get_relative_pose(pose):
    if torch.is_tensor(pose):
        pose = pose.detach().cpu().numpy()

    rot = R.from_quat(pose[:, 3:7])
    first_rot = R.from_quat(np.tile(pose[:1, 3:7], (pose.shape[0], 1)))
    trans = pose[:, :3]
    relative_trans = trans - trans[0:1]

    relative_rot = first_rot.inv() * rot
    relative_quat = relative_rot.as_quat()

    relative_pose = np.concatenate([relative_trans, relative_quat], axis=1)
    return torch.from_numpy(relative_pose)


class MultiLatentLeRobotDataset(torch.utils.data.Dataset):
    def __init__(
        self,
        config,
        num_init_worker=None,
    ):
        self._datasets = construct_lerobot_multi_processor(
            config,
            num_init_worker,
        )
        self.item_id_to_dataset_id, self.acc_dset_num = (
            self._get_item_id_to_dataset_id()
        )

    def __len__(
        self,
    ):
        return sum(len(v) for v in self._datasets)

    def _get_item_id_to_dataset_id(self):
        item_id_to_dataset_id = {}
        acc_dset_num = {}
        acc_nums = [0]
        id = 0
        for dset_id, dset in enumerate(self._datasets):
            acc_nums.append(acc_nums[-1] + len(dset))
            for _ in range(len(dset)):
                item_id_to_dataset_id[id] = dset_id
                id += 1
        for did in range(len(self._datasets)):
            acc_dset_num[did] = acc_nums[did]
        return item_id_to_dataset_id, acc_dset_num

    def __getitem__(self, idx) -> dict:
        assert idx < len(self)
        cur_dset = self._datasets[self.item_id_to_dataset_id[idx]]
        local_idx = idx - self.acc_dset_num[self.item_id_to_dataset_id[idx]]
        return cur_dset[local_idx]


class LatentLeRobotDataset(LeRobotDataset):
    def __init__(
        self,
        repo_id,
        config=None,
    ):
        self.repo_id = repo_id
        self.root = HF_LEROBOT_HOME / repo_id
        self.image_transforms = None
        self.delta_timestamps = None
        self.episodes = None
        self.tolerance_s = 1e-4
        self.revision = "v2.1"
        self.video_backend = "pyav"
        self.delta_indices = None
        self.batch_encoding_size = 1
        self.episodes_since_last_encoding = 0
        self.image_writer = None
        self.episode_buffer = None
        self.root.mkdir(exist_ok=True, parents=True)
        self.meta = LeRobotDatasetMetadata(
            self.repo_id, self.root, self.revision, force_cache_sync=False
        )
        # Limit number of episodes per task for overfitting experiments.
        max_episodes = getattr(config, "max_episodes", None)
        if max_episodes is not None:
            total = self.meta.total_episodes
            self.episodes = list(range(min(max_episodes, total)))

        if self.episodes is not None and self.meta._version >= packaging.version.parse(
            "v2.1"
        ):
            episodes_stats = [
                self.meta.episodes_stats[ep_idx] for ep_idx in self.episodes
            ]
            self.stats = aggregate_stats(episodes_stats)

        self.hf_dataset = self.load_hf_dataset()
        self.episode_data_index = get_episode_data_index(
            self.meta.episodes, self.episodes
        )

        self.latent_path = Path(repo_id) / "latents"
        self.empty_emb = torch.load(config.empty_emb_path, weights_only=False)
        self.config = config

        from diffusers import AutoencoderKLWan

        vae_path = os.path.join(config.wan22_pretrained_model_name_or_path, "vae")
        vae_cfg = AutoencoderKLWan.load_config(vae_path)
        self._latent_mean = torch.tensor(vae_cfg["latents_mean"], dtype=torch.float32)
        self._latent_std = torch.tensor(vae_cfg["latents_std"], dtype=torch.float32)
        self.cfg_prob = config.cfg_prob
        self.used_video_keys = config.obs_cam_keys
        self.mask_cam_keys = getattr(config, "mask_cam_keys", None)
        self.q01 = np.array(config.norm_stat["q01"], dtype="float")[None]
        self.q99 = np.array(config.norm_stat["q99"], dtype="float")[None]
        self._hf_torch_view = self.hf_dataset.with_format(
            type="torch", columns=["action"], output_all_columns=False
        )
        self.parse_meta()

    def parse_meta(self):
        # Build a set of allowed episode indices for fast membership checks.
        allowed_episodes = set(self.episodes) if self.episodes is not None else None
        out = []
        # 遍历所有 episode， 每个 episode 会提供：episode_index、tasks、action_config
        for value in self.meta.episodes.values():
            episode_index = value["episode_index"]
            if allowed_episodes is not None and episode_index not in allowed_episodes:
                continue
            tasks = value["tasks"]
            action_config = value["action_config"]
            for acfg in action_config:
                cur_meta = {
                    "episode_index": episode_index,
                    "tasks": tasks,
                }
                cur_meta.update(acfg)

                max_frames = getattr(self.config, "max_episode_frames", None)
                if (
                    max_frames
                    and (acfg["end_frame"] - acfg["start_frame"]) > max_frames
                ):
                    # 避免单个样本过长
                    continue

                check_statu = self._check_meta(
                    cur_meta["start_frame"],
                    cur_meta["end_frame"],
                    cur_meta["episode_index"],
                )

                if check_statu:
                    out.append(cur_meta)
        # self.new_metas 包含多个episode，每个episode代表一个时间窗口
        self.new_metas = out

    def _check_meta(self, start_frame, end_frame, episode_index):
        # 检查当前窗口所需的 latent 文件是否存在。
        episode_chunk = self.meta.get_episode_chunk(episode_index)
        latent_path = Path(self.latent_path) / f"chunk-{episode_chunk:03d}"
        for key in self.used_video_keys:
            cur_path = latent_path / key
            latent_file = (
                cur_path / f"episode_{episode_index:06d}_{start_frame}_{end_frame}.pth"
            )
            if not os.path.exists(latent_file):
                return False
        if self.mask_cam_keys:
            for key in self.mask_cam_keys:
                cur_path = latent_path / key
                latent_file = (
                    cur_path
                    / f"episode_{episode_index:06d}_{start_frame}_{end_frame}.pth"
                )
                if not os.path.exists(latent_file):
                    return False
        return True

    def _get_global_idx(self, episode_index: int, local_index: int):
        """获取当前时间窗口下所在的 episode 的全局起始位置，然后加上局部起始位置，构成全局起始帧"""
        ep_start = self.episode_data_index["from"][episode_index]
        return local_index + ep_start

    def _get_range_hf_data(self, start_frame, end_frame):
        batch = self._hf_torch_view[start_frame:end_frame]
        return batch

    def _flatten_latent_dict(self, latent_dict):
        out = {}
        for key, value in latent_dict.items():
            for inner_key, inner_value in value.items():
                new_key = f"{key}.{inner_key}"
                out[new_key] = inner_value
        return out

    def _get_range_latent_data(self, start_frame, end_frame, episode_index):
        """
        latent 来自 .pth 文件
        action 不来自 .pth 文件，而来自 LeRobot/HF 数据表
        """
        episode_chunk = self.meta.get_episode_chunk(episode_index)
        latent_path = Path(self.latent_path) / f"chunk-{episode_chunk:03d}"
        out = {}
        all_keys = list(self.used_video_keys)
        if self.mask_cam_keys:
            all_keys += list(self.mask_cam_keys)
        for key in all_keys:
            cur_path = latent_path / key
            latent_file = (
                cur_path / f"episode_{episode_index:06d}_{start_frame}_{end_frame}.pth"
            )
            assert os.path.exists(latent_file)
            latent_data = torch.load(latent_file, weights_only=False)
            out[key] = latent_data

        return self._flatten_latent_dict(out)

    def _normalize_latent(self, latent):
        """Normalize raw VAE latent to match server's normalized space.
        Input/output: (..., C) format. Applies: (raw - mean) * (1/std)
        数据集读取出的 VAE latent 必须和模型推理阶段使用的 latent 空间保持一致。"""
        mean = self._latent_mean.to(latent.dtype)
        inv_std = (1.0 / self._latent_std).to(latent.dtype)
        return (latent - mean) * inv_std

    def _cat_video_latents(self, data_dict):
        latent_lst = []
        for key in self.used_video_keys:
            latent = data_dict[f"{key}.latent"]
            latent_num_frames = data_dict[f"{key}.latent_num_frames"]
            latent_height = data_dict[f"{key}.latent_height"]
            latent_width = data_dict[f"{key}.latent_width"]
            latent = rearrange(
                latent,
                "(f h w) c -> f h w c",
                f=latent_num_frames,
                h=latent_height,
                w=latent_width,
            ) # 恢复 latent 的时空结构
            latent = self._normalize_latent(latent)
            latent_lst.append(latent)
        if self.config.env_type == "robotwin_tshape":
            # 多相机拼接，cam_high、cam_wrist_left、cam_wrist_right
            wrist_latent = torch.cat(latent_lst[1:], dim=2)
            cat_latent = torch.cat([wrist_latent, latent_lst[0]], dim=1)
        else:
            cat_latent = torch.cat(latent_lst, dim=2)

        text_emb = data_dict[f"{self.used_video_keys[0]}.text_emb"]
        if torch.rand(1).item() < self.cfg_prob:
            text_emb = self.empty_emb

        out_dict = {
            "latents": cat_latent,
            "text_emb": text_emb,
        }
        return out_dict

    def _cat_mask_latents(self, data_dict):
        mask_lst = []
        for key in self.mask_cam_keys:
            latent = data_dict[f"{key}.latent"]
            latent_num_frames = data_dict[f"{key}.latent_num_frames"]
            latent_height = data_dict[f"{key}.latent_height"]
            latent_width = data_dict[f"{key}.latent_width"]
            latent = rearrange(
                latent,
                "(f h w) c -> f h w c",
                f=latent_num_frames,
                h=latent_height,
                w=latent_width,
            )
            latent = self._normalize_latent(latent)
            mask_lst.append(latent)
        if self.config.env_type == "robotwin_tshape":
            # The video stream always has three cameras, but the SIV target can
            # intentionally use only the fixed head camera.  Do not concatenate
            # an empty wrist list in that single-target configuration.
            if len(mask_lst) == 1:
                cat_mask = mask_lst[0]
            else:
                wrist_mask = torch.cat(mask_lst[1:], dim=2)
                cat_mask = torch.cat([wrist_mask, mask_lst[0]], dim=1)
        else:
            cat_mask = torch.cat(mask_lst, dim=2)
        return cat_mask

    def _action_post_process(
        self, local_start_frame, local_end_frame, latent_frame_ids, action
    ):
        act_shift = int(latent_frame_ids[0] - local_start_frame) # 视频 latent 的起始帧 - 原始视频的起始帧，动作偏移量
        frame_stride = latent_frame_ids[1] - latent_frame_ids[0] # 每个视频 latent 之间帧的跨度
        action = action[act_shift:] # 跳过原始视频与视频 latent 没有对齐的帧
        if (
            self.config.env_type == "robotwin_tshape"
        ):  ## TODO support get_relative_pose for other dataset, currently only support robotwin
            left_action = get_relative_pose(action[:, :7])
            right_action = get_relative_pose(action[:, 8:15])
            action = np.concatenate(
                [left_action, action[:, 7:8], right_action, action[:, 15:16]], axis=1
            ) # 左右机械臂动作拼接
        """
        补齐时间长度
        frame_stride * 4 是当前实现对 latent 时间压缩和动作 chunk 组织方式的约定。它意味着：
        一个 latent 时间步对应若干个原始动作时间步，模型动作输入需要预留历史或起始位置。
        """
        action = np.pad(
            action,
            pad_width=((frame_stride * 4, 0), (0, 0)),
            mode="constant",
            constant_values=0,
        )

        # 计算模型需要多少动作帧
        latent_frame_num = (len(latent_frame_ids) - 1) // 4 + 1
        required_action_num = latent_frame_num * frame_stride * 4

        action = action[:required_action_num] # 确保动作长度刚好匹配模型需要的时间范围。
        action_mask = np.ones_like(action, dtype="bool") # 构造动作有效掩码
        assert action.shape[0] == required_action_num

        action_paded = np.pad(
            action, ((0, 0), (0, 1)), mode="constant", constant_values=0
        ) # 在动作最后增加一个额外通道，通常用于补齐或表示未使用的动作维度。
        action_mask_padded = np.pad(
            action_mask, ((0, 0), (0, 1)), mode="constant", constant_values=0
        )

        """
        选择并重新排列动作通道
        - 数据集原始动作通道顺序；
        - 模型内部动作通道顺序；
        - 左右手和夹爪通道顺序；
        不一定完全一致，所以需要显式映射。
        """
        action_aligned = action_paded[:, self.config.inverse_used_action_channel_ids]
        action_mask_aligned = action_mask_padded[
            :, self.config.inverse_used_action_channel_ids
        ]
        # 基于分位数统计的归一化, 面对不同量纲的动作时更加稳定。
        action_aligned = (action_aligned - self.q01) / (
            self.q99 - self.q01 + 1e-6
        ) * 2.0 - 1.0
        action_aligned = rearrange(
            action_aligned, "(f n) c -> c f n 1", f=latent_frame_num
        )
        action_mask_aligned = rearrange(
            action_mask_aligned, "(f n) c -> c f n 1", f=latent_frame_num
        )
        action_aligned *= action_mask_aligned
        return torch.from_numpy(action_aligned).float(), torch.from_numpy(
            action_mask_aligned
        ).bool()

    def __getitem__(self, idx) -> dict:
        """
        根据一个样本索引 idx，找到一个有效的时间窗口。一个 episode 通常代表一条完整的机器人轨迹
        然后把这个窗口中的视频 latent、SIV latent、文本和动作整理成模型可以直接使用的格式。
        """
        idx = idx % len(self.new_metas) # 数据集会循环重复使用原始窗口。
        cur_meta = self.new_metas[idx] # 选择当前的样本时间窗口，一个 episode 中存在多个时间窗口
        episode_index = cur_meta["episode_index"] # 获取当前时间窗口所在的 episode 的索引值
        start_frame = cur_meta["start_frame"] # 获取当前时间窗口的起始帧
        end_frame = cur_meta["end_frame"] # 获取当前时间窗口的结束帧
        local_start_frame = start_frame # 定义为局部起始帧，仅针对 episode 内 LeRobot 时序数据的帧索引
        local_end_frame = end_frame # 定义为局部结束帧，仅针对 episode 内 LeRobot 时序数据的帧索引

        # 获取当前时间窗口内所有的视频 latent 和对应的 SIV mask latent
        ori_data_dict = self._get_range_latent_data(
            start_frame, end_frame, episode_index
        )

        # 获取每个latent的帧编号，latent 帧与原始图像帧通常不是一一对应，而是经过了时间下采样，latent 帧更少
        latent_frame_ids = ori_data_dict[f"{self.used_video_keys[0]}.frame_ids"]
        start_frame = self._get_global_idx(episode_index, start_frame) # 获取当前时间窗口的全局起始帧
        end_frame = self._get_global_idx(episode_index, end_frame) # 获取当前时间窗口的全局结束帧

        # action 使用全局索引，其存储在 LeRobot 的统一数据表中，多个 episode 可能连续拼接在一起。
        hf_data_frames = self._get_range_hf_data(start_frame, end_frame) # 从 LeRobot 数据集中读取这个窗口的 action。
        ori_data_dict.update(hf_data_frames) # 更新 ori_data_dict，现在包含视频 latent、SIV mask latent、action
        out_dict = self._cat_video_latents(ori_data_dict) # 构造视频输入，恢复 latent 时空结构，多相机拼接

        out_dict["actions"], out_dict["actions_mask"] = self._action_post_process(
            local_start_frame,
            local_end_frame,
            latent_frame_ids,
            ori_data_dict["action"],
        ) # 处理动作

        out_dict["latents"] = out_dict["latents"].permute(3, 0, 1, 2) # (F, H, W, C) -> (C, F, H, W) pytorch 常用格式
        if self.mask_cam_keys:
            out_dict["mask_latents"] = self._cat_mask_latents(ori_data_dict).permute(
                3, 0, 1, 2
            )
        return out_dict

    def __len__(self):
        repeat = getattr(self.config, "dataset_repeat", 1)
        return len(self.new_metas) * repeat


if __name__ == "__main__":
    from tqdm import tqdm

    from siv_wam.configs import SIV_CONFIGS

    dset = MultiLatentLeRobotDataset(SIV_CONFIGS["demo_train"])
    for key, value in dset[0].items():
        if isinstance(value, torch.Tensor):
            print(f"{key}: {value.shape} tensor")
        elif isinstance(value, np.ndarray):
            print(f"{key}: {value.shape} np")
        else:
            print(f"{key}: {value}")
    print(len(dset))
    dloader = DataLoader(
        dset,
        batch_size=1,
        shuffle=True,
        num_workers=32,
    )
    max_l = 0
    action_list = []
    for data in tqdm(dloader):
        _, _, F, H, W = data["latents"].shape
        max_l = max(max_l, F * H * W)
        action_list.append(data["actions"].flatten(2).permute(0, 2, 1).flatten(0, 1))
    action_all = torch.cat(action_list, dim=0)
    print(max_l)
    print(
        action_all.shape,
        action_all.mean(dim=0),
        action_all.min(dim=0)[0],
        action_all.max(dim=0)[0],
    )
