import os
import torch
from easydict import EasyDict

from .shared_config import siv_shared_cfg

siv_franka_cfg = EasyDict(__name__='Config: SIV-WAM franka')
siv_franka_cfg.update(siv_shared_cfg)
siv_shared_cfg.infer_mode = 'server'

siv_franka_cfg.wan22_pretrained_model_name_or_path = os.environ.get("SIV_WAM_PRETRAINED_PATH", "./assets/pretrained")

siv_franka_cfg.attn_window = 30
siv_franka_cfg.frame_chunk_size = 4
siv_franka_cfg.env_type = 'none'

siv_franka_cfg.height = 224
siv_franka_cfg.width = 320
siv_franka_cfg.action_dim = 30
siv_franka_cfg.action_per_frame = 20
siv_franka_cfg.obs_cam_keys = [
    'observation.images.cam_high', 'observation.images.cam_left_wrist',
    'observation.images.cam_right_wrist'
]
siv_franka_cfg.guidance_scale = 5
siv_franka_cfg.action_guidance_scale = 1

siv_franka_cfg.num_inference_steps = 5
siv_franka_cfg.video_exec_step = -1
siv_franka_cfg.action_num_inference_steps = 10

siv_franka_cfg.snr_shift = 5.0
siv_franka_cfg.action_snr_shift = 1.0

siv_franka_cfg.used_action_channel_ids = list(range(0, 7)) + list(range(
    28, 29)) + list(range(7, 14)) + list(range(29, 30))
inverse_used_action_channel_ids = [len(siv_franka_cfg.used_action_channel_ids)
                                   ] * siv_franka_cfg.action_dim
for i, j in enumerate(siv_franka_cfg.used_action_channel_ids):
    inverse_used_action_channel_ids[j] = i
siv_franka_cfg.inverse_used_action_channel_ids = inverse_used_action_channel_ids

siv_franka_cfg.action_norm_method = 'quantiles'
siv_franka_cfg.norm_stat = {
    "q01": [
        0.3051295876502991, -0.22647984325885773, 0.19957000017166138,
        -0.022680532187223434, -0.05553057789802551, -0.2693849802017212,
        -0.29341773986816405, 0.2935442328453064, -0.4431332051753998,
        0.21256473660469055, -0.7962440848350525, -0.40816226601600647,
        -0.28359392285346985, -0.44507765769958496
    ] + [0.] * 16,
    "q99": [
        0.7572150230407715, 0.47736290097236633, 0.6428080797195435,
        0.9835678935050964, 0.9927203059196472, 0.28041139245033264,
        0.47529348731040877, 0.7564866304397571, 0.04082797020673729,
        0.5355993628501885, 0.9976375699043274, 0.8973174452781656,
        0.6016915678977965, 0.5027598619461056
    ] + [0.] * 14 + [1.0, 1.0],
}
