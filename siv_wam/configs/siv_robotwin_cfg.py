import os

from easydict import EasyDict

from .shared_config import siv_shared_cfg

siv_robotwin_cfg = EasyDict(__name__='Config: SIV-WAM robotwin')
siv_robotwin_cfg.update(siv_shared_cfg)

siv_robotwin_cfg.wan22_pretrained_model_name_or_path = os.environ.get(
    "SIV_WAM_PRETRAINED_PATH",
    "./assets/pretrained",
)

siv_robotwin_cfg.attn_window = 72
siv_robotwin_cfg.frame_chunk_size = 2
siv_robotwin_cfg.env_type = 'robotwin_tshape'

siv_robotwin_cfg.height = 256
siv_robotwin_cfg.width = 320
siv_robotwin_cfg.action_dim = 30
siv_robotwin_cfg.action_per_frame = 16
siv_robotwin_cfg.obs_cam_keys = [
    'observation.images.cam_high', 'observation.images.cam_left_wrist',
    'observation.images.cam_right_wrist'
]
siv_robotwin_cfg.guidance_scale = 5
siv_robotwin_cfg.action_guidance_scale = 1

siv_robotwin_cfg.num_inference_steps = 25
siv_robotwin_cfg.video_exec_step = -1
siv_robotwin_cfg.action_num_inference_steps = 50

siv_robotwin_cfg.snr_shift = 5.0
siv_robotwin_cfg.action_snr_shift = 1.0

siv_robotwin_cfg.used_action_channel_ids = list(range(0, 7)) + list(
    range(28, 29)) + list(range(7, 14)) + list(range(29, 30))
inverse_used_action_channel_ids = [
    len(siv_robotwin_cfg.used_action_channel_ids)
] * siv_robotwin_cfg.action_dim
for i, j in enumerate(siv_robotwin_cfg.used_action_channel_ids):
    inverse_used_action_channel_ids[j] = i
siv_robotwin_cfg.inverse_used_action_channel_ids = inverse_used_action_channel_ids

siv_robotwin_cfg.action_norm_method = 'quantiles'
siv_robotwin_cfg.norm_stat = {
    "q01": [
        -0.06172713458538055, -3.6716461181640625e-05, -0.08783501386642456,
        -1, -1, -1, -1, -0.3547105032205582, -1.3113021850585938e-06,
        -0.11975435614585876, -1, -1, -1, -1
    ] + [0.] * 16,
    "q99": [
        0.3462600058317184, 0.39966784834861746, 0.14745532035827624, 1, 1, 1,
        1, 0.034201726913452024, 0.39142737388610793, 0.1792279863357542, 1, 1,
        1, 1
    ] + [0.] * 14 + [1.0, 1.0],
}
