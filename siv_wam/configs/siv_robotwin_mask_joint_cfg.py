import os

from easydict import EasyDict

from .siv_robotwin_cfg import siv_robotwin_cfg

# ── Full joint-diffusion mask training ──────────────────────────────────────
siv_robotwin_mask_joint_cfg = EasyDict(__name__='Config: SIV-WAM robotwin mask joint train')
siv_robotwin_mask_joint_cfg.update(siv_robotwin_cfg)

_ROBOTWIN_LEROBOT_ROOT = os.environ.get(
    "SIV_WAM_DATASET_PATH",
    "./data/robotwin_lerobot",
)

siv_robotwin_mask_joint_cfg.dataset_path = _ROBOTWIN_LEROBOT_ROOT
siv_robotwin_mask_joint_cfg.empty_emb_path = os.path.join(
    _ROBOTWIN_LEROBOT_ROOT,
    "empty_emb.pt",
)
siv_robotwin_mask_joint_cfg.enable_wandb  = False
siv_robotwin_mask_joint_cfg.load_worker   = 8
siv_robotwin_mask_joint_cfg.save_interval = 5000
siv_robotwin_mask_joint_cfg.gc_interval   = 50
siv_robotwin_mask_joint_cfg.cfg_prob      = 0.1

# Keep all three RGB/video cameras as model inputs, but supervise the SIV head
# from the fixed head camera only. Wrist views do not need contact projection.
_SIV_MASK_CAMERA = os.environ.get("SIV_WAM_MASK_CAMERA", "cam_high")
if _SIV_MASK_CAMERA != "cam_high":
    raise ValueError(
        "SIV-WAM heatmap supervision is fixed to cam_high; "
        f"got SIV_WAM_MASK_CAMERA={_SIV_MASK_CAMERA!r}"
    )
siv_robotwin_mask_joint_cfg.mask_cam_keys = [
    f'observation.masks.{_SIV_MASK_CAMERA}',
]

siv_robotwin_mask_joint_cfg.learning_rate = 1e-4
siv_robotwin_mask_joint_cfg.beta1         = 0.9
siv_robotwin_mask_joint_cfg.beta2         = 0.95
siv_robotwin_mask_joint_cfg.weight_decay  = 0.1
siv_robotwin_mask_joint_cfg.warmup_steps  = 500
siv_robotwin_mask_joint_cfg.max_episode_frames = 500
siv_robotwin_mask_joint_cfg.batch_size    = 8    # B2 joint seq; BS=8 safe for 95GB GPU
siv_robotwin_mask_joint_cfg.gradient_accumulation_steps = 2  # effective BS=16/GPU, 256 global
siv_robotwin_mask_joint_cfg.num_steps     = 100000
siv_robotwin_mask_joint_cfg.max_episodes  = None   # None = use all episodes


# ── Overfitting variant (B2 validation, 2 episodes) ─────────────────────────
siv_robotwin_mask_joint_overfit_cfg = EasyDict(__name__='Config: SIV-WAM robotwin mask joint overfit')
siv_robotwin_mask_joint_overfit_cfg.update(siv_robotwin_mask_joint_cfg)

siv_robotwin_mask_joint_overfit_cfg.dataset_path = os.path.join(
    _ROBOTWIN_LEROBOT_ROOT,
    "adjust_bottle-aloha-agilex_randomized_500-1000",
)
siv_robotwin_mask_joint_overfit_cfg.max_episodes   = 2    # 2 real episodes from this single task
siv_robotwin_mask_joint_overfit_cfg.dataset_repeat = 128  # 2 × 128 = 256 items → 16 per GPU (16 GPUs)
siv_robotwin_mask_joint_overfit_cfg.batch_size     = 16   # B2 uses ~2x mem; start conservative
siv_robotwin_mask_joint_overfit_cfg.num_steps      = 2000
siv_robotwin_mask_joint_overfit_cfg.save_interval  = 100
siv_robotwin_mask_joint_overfit_cfg.warmup_steps   = 50
siv_robotwin_mask_joint_overfit_cfg.load_worker    = 2
