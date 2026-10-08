#!/usr/bin/env python3
"""Overfitting check – joint video+mask denoising from checkpoint.

Key fixes vs the previous buggy version:
  1. Video is also denoised from pure noise (not teacher-forced with clean GT).
  2. lat_noisy = actual noisy video at timestep t (NOT clean GT).
  3. frame-0 pinned to clean GT for both video and mask (conditioning frame).
  4. Video uses forward_train (FlexAttn, same as training).
  5. Mask  uses forward_train_joint (SDPA fallback, joint sequence).
"""
from __future__ import annotations
import os
import sys, glob
from pathlib import Path
import torch
import torch.nn.functional as nnF
from einops import rearrange
from tqdm import tqdm
from diffusers.video_processor import VideoProcessor
from diffusers.utils import export_to_video

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from siv_wam.modules.utils import load_transformer, load_vae
from siv_wam.modules.model import FlexAttnFunc
from siv_wam.utils import FlowMatchScheduler, data_seq_to_patch, get_mesh_id, init_logger, logger
from siv_wam.configs import SIV_CONFIGS

CKPT_STEP = 2000
CKPT_DIR  = ROOT / f"joint_overfit_output/checkpoints/checkpoint_step_{CKPT_STEP}"
OUT_DIR   = ROOT / f"joint_overfit_infer_step{CKPT_STEP}"
DEVICE    = torch.device("cuda:0")
NUM_STEPS = 25
DTYPE     = torch.bfloat16
TASK_DIR = Path(
    os.environ.get(
        "ROBOTWIN_MASK_TASK_PATH",
        "./data/robotwin_lerobot/adjust_bottle-aloha-agilex_randomized_500-1000",
    )
)


# ── Latent helpers ─────────────────────────────────────────────────────────────
def denorm(vae, lat):
    m = torch.tensor(vae.config.latents_mean, device=lat.device, dtype=lat.dtype).view(1,-1,1,1,1)
    s = torch.tensor(vae.config.latents_std,  device=lat.device, dtype=lat.dtype).view(1,-1,1,1,1)
    return lat * s + m

def norm(vae, lat):
    m = torch.tensor(vae.config.latents_mean, device=lat.device, dtype=lat.dtype).view(1,-1,1,1,1)
    s = torch.tensor(vae.config.latents_std,  device=lat.device, dtype=lat.dtype).view(1,-1,1,1,1)
    return (lat - m) / s

@torch.no_grad()
def decode_save(vae, lat_norm, path, fps=10):
    raw = denorm(vae, lat_norm)
    dec = vae.decode(raw, return_dict=False)[0]
    proc = VideoProcessor(vae_scale_factor=1)
    frames = proc.postprocess_video(dec, output_type="np")[0]
    export_to_video(frames, str(path), fps=fps)
    logger.info(f"  saved {path}")


# ── Load episode latents directly from .pth files ─────────────────────────────
def load_episode_latents(episode_idx, vae):
    chunk_dir = TASK_DIR / "latents" / "chunk-000"
    ep_str    = f"episode_{episode_idx:06d}"

    def load_cam(key):
        files = sorted(glob.glob(str(chunk_dir / key / f"{ep_str}*.pth")))
        assert files, f"no pth: {key} ep{episode_idx}"
        d = torch.load(files[0], map_location="cpu")
        lat = rearrange(d['latent'].float(), '(f h w) c -> c f h w',
                        f=d['latent_num_frames'], h=d['latent_height'],
                        w=d['latent_width']).unsqueeze(0)
        te = d.get('text_emb')
        if te is not None and te.ndim == 2:
            te = te.unsqueeze(0)
        return norm(vae, lat), te, d.get('text', '')

    vid_high,  te, text = load_cam('observation.images.cam_high')
    vid_left,  _, _     = load_cam('observation.images.cam_left_wrist')
    vid_right, _, _     = load_cam('observation.images.cam_right_wrist')
    msk_high,  _, _     = load_cam('observation.masks.cam_high')
    msk_left,  _, _     = load_cam('observation.masks.cam_left_wrist')
    msk_right, _, _     = load_cam('observation.masks.cam_right_wrist')

    # T-shape (same layout as dataset)
    vid = torch.cat([torch.cat([vid_left, vid_right], dim=-1), vid_high], dim=-2)
    msk = torch.cat([torch.cat([msk_left, msk_right], dim=-1), msk_high], dim=-2)
    logger.info(f"  ep{episode_idx}: vid={vid.shape} msk={msk.shape} text={text[:60]!r}")
    return vid, msk, te


@torch.no_grad()
def run_joint(model, scheduler, gt_vid, gt_mask, text_emb, cfg, device, dtype, num_steps):
    """Jointly denoise video + mask from pure noise.

    Video  : uses forward_train  (frozen backbone + FlexAttn, pretrained proj_out)
    Mask   : uses forward_train_joint (frozen backbone + SDPA, trained mask_proj_out)

    frame-0 is always pinned to the clean GT (conditioning frame, t=0).
    """
    patch_size = cfg.patch_size
    B, C, F, H, W = gt_vid.shape
    action_dim   = cfg.action_dim        # 30
    apr          = cfg.action_per_frame  # 16
    F_a          = apr * F               # 32 for F=2

    gt_vid  = gt_vid.to(device, dtype)
    gt_mask = gt_mask.to(device, dtype)
    te      = text_emb.to(device, dtype)

    # Grids
    grid_lat = get_mesh_id(F // patch_size[0], H // patch_size[1], W // patch_size[2],
                           t=0, f_w=1, f_shift=0).to(device)[None].repeat(B, 1, 1)
    grid_act = get_mesh_id(F_a, 1, 1, t=1, f_w=1, f_shift=0, action=True
                           ).to(device)[None].repeat(B, 1, 1)
    act_zero = torch.zeros(B, action_dim, F_a, 1, 1, device=device, dtype=dtype)

    scheduler.set_timesteps(num_steps)
    # Pad with final t=0 step (same as reference inference)
    timesteps = nnF.pad(scheduler.timesteps, (0, 1), value=0)

    # Start from pure noise
    noisy_vid  = torch.randn_like(gt_vid)
    noisy_mask = torch.randn_like(gt_mask)

    for i, t in enumerate(tqdm(timesteps, desc="Joint denoise", ncols=80)):
        last = i == len(timesteps) - 1

        # ── Build per-frame timestep vectors ──────────────────────────────
        ts      = torch.full((B, F), float(t), device=device)
        ts_cond = torch.zeros_like(ts)
        ts[:, 0] = 0.0   # frame-0 always conditioned (t=0)

        # ── Pin frame-0 to clean GT ────────────────────────────────────────
        noisy_vid[:, :, 0:1]  = gt_vid[:, :, 0:1]
        noisy_mask[:, :, 0:1] = gt_mask[:, :, 0:1]

        # ── Shared dicts ────────────────────────────────────────────────────
        lat_dict = dict(
            noisy_latents = noisy_vid,    # noisy video (being denoised)
            latent        = gt_vid,        # clean GT (conditioning copy)
            timesteps     = ts,
            cond_timesteps = ts_cond,
            grid_id       = grid_lat,
            text_emb      = te,
        )
        act_dict = dict(
            noisy_latents  = act_zero,
            latent         = act_zero,
            timesteps      = torch.zeros(B, F_a, device=device),
            cond_timesteps = torch.zeros(B, F_a, device=device),
            grid_id        = grid_act,
            text_emb       = te,
            actions_mask   = torch.ones(B, F_a, 1, 1, device=device),
        )
        input_dict = dict(
            latent_dict          = lat_dict,
            action_dict          = act_dict,
            noisy_mask_latents   = noisy_mask,
            mask_targets         = torch.zeros_like(noisy_mask),
            # required by forward_train / FlexAttn:
            chunk_size           = F,
            window_size          = cfg.attn_window,
        )

        # ── Video prediction (forward_train) ────────────────────────────────
        vid_pred_seq, _ = model(input_dict, train_mode=True)

        # ── Mask prediction (forward_train_joint) ──────────────────────────
        _, _, _, mask_pred_seq = model(input_dict, joint_mode=True)

        if not last:
            vid_pred_vol  = data_seq_to_patch(patch_size, vid_pred_seq,  F, H, W, batch_size=B)
            mask_pred_vol = data_seq_to_patch(patch_size, mask_pred_seq, F, H, W, batch_size=B)
            noisy_vid  = scheduler.step(vid_pred_vol,  t, noisy_vid)
            noisy_mask = scheduler.step(mask_pred_vol, t, noisy_mask)

    return noisy_vid, noisy_mask


def main():
    init_logger()
    OUT_DIR.mkdir(parents=True, exist_ok=True)

    cfg = SIV_CONFIGS["robotwin_mask_joint"]

    logger.info(f"Loading model from {CKPT_DIR/'transformer'} ...")
    model = load_transformer(str(CKPT_DIR / "transformer"), torch_dtype=DTYPE, torch_device=DEVICE)
    model.eval()
    logger.info(f"  mask_proj_out present: {hasattr(model, 'mask_proj_out')}")

    vae = load_vae(str(Path(cfg.wan22_pretrained_model_name_or_path) / "vae"), DTYPE, DEVICE)
    vae.eval()

    scheduler = FlowMatchScheduler(shift=cfg.snr_shift, sigma_min=0.0, extra_one_step=True)

    for ep_idx in range(2):
        logger.info(f"\n{'='*60}\nEpisode {ep_idx}\n{'='*60}")
        ep_dir = OUT_DIR / f"episode_{ep_idx}"
        ep_dir.mkdir(parents=True, exist_ok=True)

        vid_lat, msk_lat, te = load_episode_latents(ep_idx, vae)
        F_use    = cfg.frame_chunk_size
        vid_chunk = vid_lat[:, :, :F_use]
        msk_chunk = msk_lat[:, :, :F_use]

        decode_save(vae, vid_chunk.to(DEVICE, DTYPE), ep_dir / "gt_video.mp4")
        decode_save(vae, msk_chunk.to(DEVICE, DTYPE), ep_dir / "gt_mask.mp4")

        logger.info(f"Joint inference ({NUM_STEPS} steps)…")
        pred_vid, pred_mask = run_joint(
            model, scheduler, vid_chunk, msk_chunk, te, cfg, DEVICE, DTYPE, NUM_STEPS)

        decode_save(vae, pred_vid,  ep_dir / "pred_video.mp4")
        decode_save(vae, pred_mask, ep_dir / "pred_mask.mp4")

        logger.info(f"  gt_mask   mean={msk_chunk.mean():.4f} std={msk_chunk.std():.4f}")
        logger.info(f"  pred_mask mean={pred_mask.mean():.4f} std={pred_mask.std():.4f}")
        logger.info(f"  gt_vid    mean={vid_chunk.mean():.4f} std={vid_chunk.std():.4f}")
        logger.info(f"  pred_vid  mean={pred_vid.mean():.4f}  std={pred_vid.std():.4f}")
        torch.cuda.empty_cache()

    logger.info(f"\nDone → {OUT_DIR}")


if __name__ == "__main__":
    main()
