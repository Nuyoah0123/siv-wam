#!/usr/bin/env python3
"""Inference script for the B2 joint-backbone mask head model.

Architecture (B2):
  - Backbone processes [vid_noisy | mask_noisy] together each denoising step.
  - proj_out      → video velocity (frozen)
  - mask_proj_out → mask velocity  (trainable)
  - KV-cache stores both vid AND mask token KVs for autoregressive generation.

Inference is fully autoregressive (T-shape KV-cache), identical in spirit to
the video-action server inference, extended to include mask tokens.
"""
from __future__ import annotations
import os
import sys, glob
from pathlib import Path
import torch
import torch.nn.functional as F
from einops import rearrange
from tqdm import tqdm
from diffusers.video_processor import VideoProcessor
from diffusers.utils import export_to_video

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from siv_wam.modules.utils import load_transformer, load_vae
from siv_wam.utils import FlowMatchScheduler, data_seq_to_patch, get_mesh_id, init_logger, logger
from siv_wam.configs import SIV_CONFIGS

# ── Configuration ────────────────────────────────────────────────────────────
CKPT_STEP = 1200
CKPT_DIR  = ROOT / f"uni-checkpoints/checkpoint_step_{CKPT_STEP}"
OUT_DIR   = ROOT / f"infer_mask_fullrun_step{CKPT_STEP}"
DEVICE    = torch.device("cuda:0")
NUM_STEPS = 25
DTYPE     = torch.bfloat16
GUIDANCE_SCALE = 5.0

TASK_DIR = Path(
    os.environ.get(
        "ROBOTWIN_MASK_TASK_PATH",
        "./data/robotwin_lerobot/adjust_bottle-aloha-agilex_randomized_500-1000",
    )
)
NUM_EPISODES = 2   # episodes to run inference on


# ── Latent helpers ────────────────────────────────────────────────────────────
def denorm(vae, lat):
    m = torch.tensor(vae.config.latents_mean, device=lat.device, dtype=lat.dtype).view(1, -1, 1, 1, 1)
    s = torch.tensor(vae.config.latents_std,  device=lat.device, dtype=lat.dtype).view(1, -1, 1, 1, 1)
    return lat * s + m

def norm(vae, lat):
    m = torch.tensor(vae.config.latents_mean, device=lat.device, dtype=lat.dtype).view(1, -1, 1, 1, 1)
    s = torch.tensor(vae.config.latents_std,  device=lat.device, dtype=lat.dtype).view(1, -1, 1, 1, 1)
    return (lat - m) / s

@torch.no_grad()
def decode_save(vae, lat_norm, path, fps=10):
    raw = denorm(vae, lat_norm)
    dec = vae.decode(raw, return_dict=False)[0]
    proc = VideoProcessor(vae_scale_factor=1)
    frames = proc.postprocess_video(dec, output_type="np")[0]
    export_to_video(frames, str(path), fps=fps)
    logger.info(f"  saved {path}")


# ── Load episode latents from pre-extracted .pth files ───────────────────────
def load_episode_latents(episode_idx, vae):
    chunk_dir = TASK_DIR / "latents" / "chunk-000"
    ep_str    = f"episode_{episode_idx:06d}"

    def load_cam(key):
        files = sorted(glob.glob(str(chunk_dir / key / f"{ep_str}*.pth")))
        assert files, f"no .pth found: {key} ep{episode_idx}"
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

    # T-shape: wrist pair side-by-side (left|right), then stacked above head
    vid = torch.cat([torch.cat([vid_left, vid_right], dim=-1), vid_high], dim=-2)
    msk = torch.cat([torch.cat([msk_left, msk_right], dim=-1), msk_high], dim=-2)
    logger.info(f"  ep{episode_idx}: vid={vid.shape} msk={msk.shape} text={text[:60]!r}")
    return vid, msk, te


# ── Single-chunk inference (mirrors server_style_infer_single_chunk) ──────────
@torch.no_grad()
def infer_single_chunk(
    model, scheduler,
    init_vid_latent, init_mask_latent,
    text_emb, neg_emb,
    patch_size, device, dtype,
    num_steps, guidance_scale,
    frame_chunk_size, H, W,
    frame_st_id=0,
    vid_cache_name="vid", mask_cache_name="mask",
):
    """Separate-pass inference (B2, two forward calls per denoising step).

    Pass 1 – video only (no mask tokens):
        backbone input: [vid_noisy]  (identical to the video-only inference path)
        output used:    proj_out → video velocity
        KV-cache:       vid_cache_name  (vid tokens only, same size as original)

    Pass 2 – joint [vid | mask]:
        backbone input: [vid_noisy | mask_noisy]
        output used:    mask_proj_out → mask velocity  (vid output discarded)
        KV-cache:       mask_cache_name  (doubled, stores vid+mask tokens)

    This guarantees the video branch is unaffected by mask tokens while the
    mask branch still benefits from cross-attention with video tokens.
    """
    use_cfg = guidance_scale > 1

    scheduler.set_timesteps(num_steps)
    timesteps = F.pad(scheduler.timesteps, (0, 1), mode='constant', value=0)

    vid_latents  = torch.randn(1, 48, frame_chunk_size, H, W, device=device, dtype=dtype)
    mask_latents = torch.randn(1, 48, frame_chunk_size, H, W, device=device, dtype=dtype)

    vid_cond  = init_vid_latent[:, :, 0:1].to(device, dtype) if frame_st_id == 0 else None
    mask_cond = (init_mask_latent[:, :, 0:1].to(device, dtype)
                 if (frame_st_id == 0 and init_mask_latent is not None) else None)

    for i, t in enumerate(tqdm(timesteps, desc=f"chunk fst={frame_st_id}", leave=False)):
        last_step = (i == len(timesteps) - 1)

        ts = torch.ones([frame_chunk_size], dtype=torch.float32, device=device) * float(t)
        grid = get_mesh_id(
            frame_chunk_size // patch_size[0],
            H // patch_size[1],
            W // patch_size[2],
            0, 1, frame_st_id,
        ).to(device)

        noisy_vid  = vid_latents.clone()
        noisy_mask = mask_latents.clone()
        if vid_cond is not None:
            noisy_vid[:, :, 0:1]  = vid_cond
            noisy_mask[:, :, 0:1] = mask_cond if mask_cond is not None else noisy_mask[:, :, 0:1]
            ts[0] = 0.0

        if use_cfg:
            noisy_vid_cfg  = noisy_vid.repeat(2, 1, 1, 1, 1)
            noisy_mask_cfg = noisy_mask.repeat(2, 1, 1, 1, 1)
            text_cfg       = torch.cat([text_emb, neg_emb], dim=0)
            grid_cfg       = grid[None].repeat(2, 1, 1)
            ts_cfg         = ts[None].repeat(2, 1)
        else:
            noisy_vid_cfg  = noisy_vid
            noisy_mask_cfg = noisy_mask
            text_cfg       = text_emb
            grid_cfg       = grid[None]
            ts_cfg         = ts[None]

        base_dict = dict(timesteps=ts_cfg, grid_id=grid_cfg, text_emb=text_cfg)

        # ── Pass 1: video only (video-only inference path, NO mask tokens) ───
        vid_input = dict(noisy_latents=noisy_vid_cfg, **base_dict)
        video_pred_seq = model(
            vid_input,
            update_cache=1 if last_step else 0,
            cache_name=vid_cache_name,
            action_mode=False,
        )
        # forward returns plain tensor in video-only mode

        # ── Pass 2: joint [vid | mask] → only use mask output ───────────────
        mask_input = dict(noisy_latents=noisy_vid_cfg,
                          noisy_mask_latents=noisy_mask_cfg, **base_dict)
        _, mask_pred_seq = model(
            mask_input,
            update_cache=1 if last_step else 0,
            cache_name=mask_cache_name,
            action_mode=False,
        )

        if not last_step:
            bs_out = 2 if use_cfg else 1

            video_pred = data_seq_to_patch(patch_size, video_pred_seq, frame_chunk_size, H, W, batch_size=bs_out)
            if use_cfg:
                video_pred = video_pred[1:] + guidance_scale * (video_pred[:1] - video_pred[1:])
            else:
                video_pred = video_pred[:1]
            vid_latents = scheduler.step(video_pred, t, vid_latents, return_dict=False)

            mask_pred = data_seq_to_patch(patch_size, mask_pred_seq, frame_chunk_size, H, W, batch_size=bs_out)
            if use_cfg:
                mask_pred = mask_pred[1:] + guidance_scale * (mask_pred[:1] - mask_pred[1:])
            else:
                mask_pred = mask_pred[:1]
            mask_latents = scheduler.step(mask_pred, t, mask_latents, return_dict=False)

        if vid_cond is not None:
            vid_latents[:, :, 0:1] = vid_cond
        if mask_cond is not None:
            mask_latents[:, :, 0:1] = mask_cond

    return vid_latents, mask_latents


# ── Autoregressive multi-chunk inference ─────────────────────────────────────
@torch.no_grad()
def autoregressive_infer(
    model, scheduler,
    init_vid_latent, init_mask_latent,
    text_emb, neg_emb,
    cfg, device, dtype, num_chunks=1,
):
    patch_size       = cfg.patch_size
    frame_chunk_size = cfg.frame_chunk_size
    H = init_vid_latent.shape[-2]
    W = init_vid_latent.shape[-1]
    use_cfg = cfg.guidance_scale > 1

    latent_token_per_chunk = (frame_chunk_size * H * W) // (
        patch_size[0] * patch_size[1] * patch_size[2])
    action_token_per_chunk = frame_chunk_size * cfg.action_per_frame

    # Pass 1 cache: video only (original size)
    model.create_empty_cache(
        "vid", cfg.attn_window,
        latent_token_per_chunk, action_token_per_chunk,
        dtype=dtype, device=device,
        batch_size=2 if use_cfg else 1,
        has_mask=False,
    )
    # Pass 2 cache: vid + mask tokens (doubled lat slots)
    model.create_empty_cache(
        "mask", cfg.attn_window,
        latent_token_per_chunk, action_token_per_chunk,
        dtype=dtype, device=device,
        batch_size=2 if use_cfg else 1,
        has_mask=True,
    )

    vid_chunks  = []
    mask_chunks = []
    for chunk_id in range(num_chunks):
        frame_st_id = chunk_id * frame_chunk_size
        logger.info(f"  chunk {chunk_id+1}/{num_chunks} (frame_st={frame_st_id})")
        v, m = infer_single_chunk(
            model, scheduler,
            init_vid_latent, init_mask_latent,
            text_emb, neg_emb,
            patch_size, device, dtype,
            cfg.num_inference_steps, cfg.guidance_scale,
            frame_chunk_size, H, W,
            frame_st_id=frame_st_id,
            vid_cache_name="vid", mask_cache_name="mask",
        )
        vid_chunks.append(v)
        mask_chunks.append(m)

    model.clear_cache("vid")
    model.clear_cache("mask")
    return torch.cat(vid_chunks, dim=2), torch.cat(mask_chunks, dim=2)


def main():
    init_logger()
    OUT_DIR.mkdir(parents=True, exist_ok=True)

    cfg = SIV_CONFIGS["robotwin_mask_joint"]

    logger.info(f"Loading model from {CKPT_DIR / 'transformer'} ...")
    model = load_transformer(str(CKPT_DIR / "transformer"), torch_dtype=DTYPE, torch_device=DEVICE)
    model.eval()
    logger.info(f"  mask_proj_out present: {hasattr(model, 'mask_proj_out')}")

    vae = load_vae(str(Path(cfg.wan22_pretrained_model_name_or_path) / "vae"), DTYPE, DEVICE)
    vae.eval()

    scheduler = FlowMatchScheduler(shift=cfg.snr_shift, sigma_min=0.0, extra_one_step=True)

    # Zero embedding for negative prompt (CFG)
    neg_emb = torch.zeros(1, 512, 4096, device=DEVICE, dtype=DTYPE)

    for ep_idx in range(NUM_EPISODES):
        logger.info(f"\n{'='*60}\nEpisode {ep_idx}\n{'='*60}")
        ep_dir = OUT_DIR / f"episode_{ep_idx}"
        ep_dir.mkdir(parents=True, exist_ok=True)

        vid_lat, msk_lat, te = load_episode_latents(ep_idx, vae)
        text_emb = te.to(DEVICE, DTYPE)

        total_lat_frames = vid_lat.shape[2]
        fcs = cfg.frame_chunk_size
        # Trim to a multiple of frame_chunk_size
        trimmed_frames = (total_lat_frames // fcs) * fcs
        if trimmed_frames == 0:
            trimmed_frames = fcs
        vid_full = vid_lat[:, :, :trimmed_frames].to(DEVICE, DTYPE)
        msk_full = msk_lat[:, :, :trimmed_frames].to(DEVICE, DTYPE)
        num_chunks = trimmed_frames // fcs
        logger.info(f"  total_lat_frames={total_lat_frames}, frame_chunk_size={fcs}, "
                    f"num_chunks={num_chunks}")

        # Save ground truth (full episode)
        decode_save(vae, vid_full, ep_dir / "gt_video.mp4")
        decode_save(vae, msk_full, ep_dir / "gt_mask.mp4")

        logger.info(f"Running inference ({NUM_STEPS} steps, guidance={GUIDANCE_SCALE}, "
                    f"{num_chunks} chunks)...")
        pred_vid, pred_mask = autoregressive_infer(
            model, scheduler,
            vid_full, msk_full,
            text_emb, neg_emb,
            cfg, DEVICE, DTYPE, num_chunks=num_chunks,
        )

        decode_save(vae, pred_vid,  ep_dir / "pred_video.mp4")
        decode_save(vae, pred_mask, ep_dir / "pred_mask.mp4")

        logger.info(f"  gt_vid    mean={vid_full.mean():.4f} std={vid_full.std():.4f}")
        logger.info(f"  pred_vid  mean={pred_vid.mean():.4f}  std={pred_vid.std():.4f}")
        logger.info(f"  gt_mask   mean={msk_full.mean():.4f} std={msk_full.std():.4f}")
        logger.info(f"  pred_mask mean={pred_mask.mean():.4f} std={pred_mask.std():.4f}")
        torch.cuda.empty_cache()

    logger.info(f"\nDone → {OUT_DIR}")


if __name__ == "__main__":
    main()
