import os

from easydict import EasyDict

from .siv_robotwin_cfg import siv_robotwin_cfg

siv_robotwin_mask_train_cfg = EasyDict(__name__='Config: SIV-WAM robotwin mask train')
siv_robotwin_mask_train_cfg.update(siv_robotwin_cfg)

_ROBOTWIN_LEROBOT_ROOT = os.environ.get(
    "SIV_WAM_DATASET_PATH",
    "./data/robotwin_lerobot",
)

siv_robotwin_mask_train_cfg.dataset_path = _ROBOTWIN_LEROBOT_ROOT
siv_robotwin_mask_train_cfg.empty_emb_path = os.path.join(
    _ROBOTWIN_LEROBOT_ROOT,
    "empty_emb.pt",
)
siv_robotwin_mask_train_cfg.enable_wandb = False
siv_robotwin_mask_train_cfg.load_worker = 8
siv_robotwin_mask_train_cfg.save_interval = 5000
siv_robotwin_mask_train_cfg.gc_interval = 50
siv_robotwin_mask_train_cfg.cfg_prob = 0.1

# Keep all three RGB/video cameras as model inputs, but supervise the SIV head
# from the fixed head camera only. Wrist views do not need contact projection.
_SIV_MASK_CAMERA = os.environ.get("SIV_WAM_MASK_CAMERA", "cam_high")
if _SIV_MASK_CAMERA != "cam_high":
    raise ValueError(
        "SIV-WAM heatmap supervision is fixed to cam_high; "
        f"got SIV_WAM_MASK_CAMERA={_SIV_MASK_CAMERA!r}"
    )
siv_robotwin_mask_train_cfg.mask_cam_keys = [
    f'observation.masks.{_SIV_MASK_CAMERA}',
]

siv_robotwin_mask_train_cfg.learning_rate = 1e-4
siv_robotwin_mask_train_cfg.beta1 = 0.9
siv_robotwin_mask_train_cfg.beta2 = 0.95
siv_robotwin_mask_train_cfg.weight_decay = 0.1
siv_robotwin_mask_train_cfg.warmup_steps = 500
siv_robotwin_mask_train_cfg.max_episode_frames = 500
siv_robotwin_mask_train_cfg.batch_size = 16
siv_robotwin_mask_train_cfg.gradient_accumulation_steps = 1
siv_robotwin_mask_train_cfg.num_steps = 50000
