#!/usr/bin/env python3
"""SIV-WAM image-to-video-action demo inference.

Pipeline (same as server.py generate()):
  1. Read the first frame(s) from raw MP4 videos for each camera.
  2. Resize & encode with the VAE → normalised latent (T-shape).
  3. Load the text instruction.
  4. Run B2 autoregressive denoising (separate vid / mask passes).
  5. Decode and save  demo_video.mp4  +  demo_mask.mp4.

Test episode is chosen from a task NOT used in the overfitting experiments
(adjust_bottle was used for overfit; we pick beat_block_hammer here).
"""
from __future__ import annotations
import os
import sys, glob
from pathlib import Path

import cv2
import numpy as np
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

# ── Config ────────────────────────────────────────────────────────────────────
CKPT_DIR = ROOT / "uni-checkpoints/checkpoint_step_1200/transformer"
OUT_DIR  = ROOT / "infer_demo_step1200"
DEVICE   = torch.device("cuda:0")
DTYPE    = torch.bfloat16
NUM_STEPS      = 25
GUIDANCE_SCALE = 5.0

# Test data — raw RoboTwin MP4, episode 0, beat_block_hammer task
TASK_VIDEO_DIR = Path(
    os.environ.get(
        "ROBOTWIN_DEMO_VIDEO_DIR",
        "./external/RoboTwin/data/beat_block_hammer/demo_clean_100/video",
    )
)
INST_FILE = Path(
    os.environ.get(
        "ROBOTWIN_DEMO_INST_FILE",
        "./external/RoboTwin/data/beat_block_hammer/demo_clean_100/instructions/episode0.json",
    )
)
EPISODE_ID     = 0
NUM_CHUNKS     = 9   # 18 latent frames / 2 per chunk

# T-shape camera dimensions (matching training config: height=256 width=320)
CAM_HIGH_H, CAM_HIGH_W   = 256, 320
WRIST_H, WRIST_W         = 128, 160   # half resolution wrists


# ── Video helpers ──────────────────────────────────────────────────────────────
def read_all_frames(mp4_path: Path) -> np.ndarray:
    """Return all frames as uint8 [T, H, W, 3] (RGB)."""
    cap = cv2.VideoCapture(str(mp4_path))
    frames = []
    while True:
        ret, f = cap.read()
        if not ret:
            break
        frames.append(cv2.cvtColor(f, cv2.COLOR_BGR2RGB))
    cap.release()
    return np.stack(frames, axis=0)  # [T, H, W, 3]


def resize_frames(frames: np.ndarray, h: int, w: int) -> torch.Tensor:
    """Resize [T, H, W, 3] uint8 → float32 tensor [1, 3, T, H, W] in [-1, 1]."""
    out = []
    for f in frames:
        img = cv2.resize(f, (w, h), interpolation=cv2.INTER_LINEAR)
        out.append(img)
    arr = np.stack(out, axis=0).astype(np.float32) / 255.0 * 2.0 - 1.0
    t = torch.from_numpy(arr)            # [T, H, W, 3]
    t = t.permute(3, 0, 1, 2).unsqueeze(0)  # [1, 3, T, H, W]
    return t


# ── VAE encode (normalised, T-shape) ─────────────────────────────────────────
def encode_tshape(vae, frames_high, frames_left, frames_right, device, dtype):
    """Encode T-shape video: [1, 48, T, 24, 20].

    Layout  (latent space, H×W = 24×20):
        rows 0-7   : left_wrist | right_wrist  (8 rows each, side-by-side)
        rows 8-23  : cam_high                  (16 rows)
    """
    lm = torch.tensor(vae.config.latents_mean, device=device, dtype=dtype).view(1, -1, 1, 1, 1)
    ls = torch.tensor(vae.config.latents_std,  device=device, dtype=dtype).view(1, -1, 1, 1, 1)

    @torch.no_grad()
    def enc(t):
        t = t.to(device, dtype)
        # VAE expects [B, C, T, H, W]; encode returns mean+logvar
        out = vae.encode(t).latent_dist.mean   # [1, 16, T, H, W]  (WanVAE uses 16-ch)
        # Normalise to match training (latents_mean / latents_std)
        return (out - lm[:, :out.shape[1]]) / ls[:, :out.shape[1]]

    lat_high  = enc(frames_high)   # [1, 16, T, 16, 20]
    lat_left  = enc(frames_left)   # [1, 16, T,  8, 10]
    lat_right = enc(frames_right)  # [1, 16, T,  8, 10]

    # side-by-side wrists → [1, 16, T, 8, 20]
    lat_wrists = torch.cat([lat_left, lat_right], dim=-1)
    # stack over height: wrists on top, high below → [1, 16, T, 24, 20]
    lat = torch.cat([lat_wrists, lat_high], dim=-2)

    # replicate across 48 channels (WanTransformer expects 48-ch latent via patch_embed)
    # Actually the VAE already produces 16 channels; the 48-ch in training is post-patchify.
    # We just return the 16-ch latent — load_vae/denorm will handle it.
    return lat   # [1, 16, T, 24, 20]


def encode_tshape_48ch(vae, frames_high, frames_left, frames_right, device, dtype):
    """Return normalised 16-ch latent compatible with infer_mask_v2 decode_save."""
    return encode_tshape(vae, frames_high, frames_left, frames_right, device, dtype)


# ── Latent denorm / decode ────────────────────────────────────────────────────
@torch.no_grad()
def decode_latent(vae, lat_norm, path, fps=10):
    """Denorm + VAE decode + save mp4. lat_norm: [1, 16, T, H, W]."""
    lm = torch.tensor(vae.config.latents_mean, device=lat_norm.device, dtype=lat_norm.dtype).view(1, -1, 1, 1, 1)
    ls = torch.tensor(vae.config.latents_std,  device=lat_norm.device, dtype=lat_norm.dtype).view(1, -1, 1, 1, 1)
    C = lat_norm.shape[1]
    raw = lat_norm * ls[:, :C] + lm[:, :C]
    dec = vae.decode(raw, return_dict=False)[0]
    proc = VideoProcessor(vae_scale_factor=1)
    frames = proc.postprocess_video(dec, output_type="np")[0]
    export_to_video(frames, str(path), fps=fps)
    logger.info(f"  saved {path}")


# ── Single-chunk denoising (B2 separate passes) ───────────────────────────────
@torch.no_grad()
def infer_chunk(
    model, scheduler,
    init_vid_latent, init_mask_latent,
    text_emb, neg_emb,
    patch_size, device, dtype,
    num_steps, guidance_scale,
    frame_chunk_size, H, W,
    frame_st_id, vid_cache, mask_cache,
):
    use_cfg = guidance_scale > 1
    scheduler.set_timesteps(num_steps)
    timesteps = F.pad(scheduler.timesteps, (0, 1), mode='constant', value=0)

    vid_lat  = torch.randn(1, init_vid_latent.shape[1],  frame_chunk_size, H, W, device=device, dtype=dtype)
    mask_lat = torch.randn(1, init_mask_latent.shape[1], frame_chunk_size, H, W, device=device, dtype=dtype)

    vid_cond  = init_vid_latent[:, :, 0:1].to(device, dtype)  if frame_st_id == 0 else None
    mask_cond = init_mask_latent[:, :, 0:1].to(device, dtype) if frame_st_id == 0 else None

    for i, t in enumerate(tqdm(timesteps, desc=f"chunk fst={frame_st_id}", leave=False)):
        last = (i == len(timesteps) - 1)
        ts   = torch.ones([frame_chunk_size], dtype=torch.float32, device=device) * float(t)
        grid = get_mesh_id(
            frame_chunk_size // patch_size[0],
            H // patch_size[1], W // patch_size[2],
            0, 1, frame_st_id,
        ).to(device)

        noisy_v = vid_lat.clone()
        noisy_m = mask_lat.clone()
        if vid_cond is not None:
            noisy_v[:, :, 0:1] = vid_cond
            noisy_m[:, :, 0:1] = mask_cond
            ts[0] = 0.0

        if use_cfg:
            nv  = noisy_v.repeat(2, 1, 1, 1, 1);  nm  = noisy_m.repeat(2, 1, 1, 1, 1)
            te  = torch.cat([text_emb, neg_emb], dim=0)
            g   = grid[None].repeat(2, 1, 1);     ts2 = ts[None].repeat(2, 1)
        else:
            nv, nm, te, g, ts2 = noisy_v, noisy_m, text_emb, grid[None], ts[None]

        base = dict(timesteps=ts2, grid_id=g, text_emb=te)

        # Pass 1 – video only (frozen video backbone path)
        # forward returns plain tensor (not tuple) in video-only mode
        vpred_seq = model(dict(noisy_latents=nv, **base),
                          update_cache=1 if last else 0,
                          cache_name=vid_cache, action_mode=False)
        # Pass 2 – joint [vid | mask] → only mask output used
        _, mpred_seq = model(dict(noisy_latents=nv, noisy_mask_latents=nm, **base),
                             update_cache=1 if last else 0,
                             cache_name=mask_cache, action_mode=False)

        if not last:
            bs = 2 if use_cfg else 1
            vp = data_seq_to_patch(patch_size, vpred_seq, frame_chunk_size, H, W, batch_size=bs)
            mp = data_seq_to_patch(patch_size, mpred_seq, frame_chunk_size, H, W, batch_size=bs)
            if use_cfg:
                vp = vp[1:] + guidance_scale * (vp[:1] - vp[1:])
                mp = mp[1:] + guidance_scale * (mp[:1] - mp[1:])
            else:
                vp, mp = vp[:1], mp[:1]
            vid_lat  = scheduler.step(vp, t, vid_lat,  return_dict=False)
            mask_lat = scheduler.step(mp, t, mask_lat, return_dict=False)

        if vid_cond  is not None: vid_lat[:, :, 0:1]  = vid_cond
        if mask_cond is not None: mask_lat[:, :, 0:1] = mask_cond

    return vid_lat, mask_lat


# ── Main ──────────────────────────────────────────────────────────────────────
def main():
    init_logger()
    OUT_DIR.mkdir(parents=True, exist_ok=True)

    cfg = SIV_CONFIGS["robotwin_mask_joint"]
    patch_size       = cfg.patch_size        # [1, 2, 2]
    frame_chunk_size = cfg.frame_chunk_size  # 2

    # ── load model ──
    logger.info(f"Loading model from {CKPT_DIR} ...")
    model = load_transformer(str(CKPT_DIR), torch_dtype=DTYPE, torch_device=DEVICE)
    model.eval()

    # ── load VAE ──
    vae = load_vae(str(Path(cfg.wan22_pretrained_model_name_or_path) / "vae"), DTYPE, DEVICE)
    vae.eval()

    scheduler = FlowMatchScheduler(shift=cfg.snr_shift, sigma_min=0.0, extra_one_step=True)
    neg_emb   = torch.zeros(1, 512, 4096, device=DEVICE, dtype=DTYPE)

    # ── load instruction ──
    import json
    inst_data = json.loads(INST_FILE.read_text())
    # instruction JSON has {'seen': [...], 'unseen': [...]}; pick first seen instruction
    if isinstance(inst_data, dict):
        instruction = inst_data.get('seen', inst_data.get('unseen', ['']))[ 0]
    elif isinstance(inst_data, list):
        instruction = inst_data[0]
    else:
        instruction = str(inst_data)
    logger.info(f"Instruction: {instruction[:80]!r}")

    # Load pre-encoded text embedding if available, else use zero (demo mode)
    lerobot_dir = Path(
        os.environ.get(
            "ROBOTWIN_BEAT_BLOCK_LEROBOT_DIR",
            "./data/robotwin_lerobot/beat_block_hammer-aloha-agilex_randomized_500-1000",
        )
    )
    te_files = sorted((lerobot_dir / "latents/chunk-000/observation.images.cam_high").glob(f"episode_{EPISODE_ID:06d}*.pth"))
    if te_files:
        d = torch.load(te_files[0], map_location="cpu")
        te = d.get('text_emb')
        if te is not None:
            te = te.unsqueeze(0) if te.ndim == 2 else te
            text_emb = te.to(DEVICE, DTYPE)
        else:
            text_emb = neg_emb.clone()
    else:
        text_emb = neg_emb.clone()
    logger.info(f"text_emb shape: {text_emb.shape}")

    # ── read raw video frames ──
    logger.info(f"Reading raw MP4 frames for episode {EPISODE_ID} ...")
    vid_high_name  = f"episode{EPISODE_ID}.mp4"
    vid_wrist_name = f"episode{EPISODE_ID}_wrist.mp4"
    mask_name      = f"episode{EPISODE_ID}_mask.mp4"
    head_mask_name = f"episode{EPISODE_ID}_headmask.mp4"

    frames_high  = read_all_frames(TASK_VIDEO_DIR / vid_high_name)   # [T, H, W, 3]
    frames_wrist = read_all_frames(TASK_VIDEO_DIR / vid_wrist_name)  # [T, H, W, 3] wrist is left|right side-by-side
    frames_mask  = read_all_frames(TASK_VIDEO_DIR / mask_name)       # [T, H, W, 3] wrist masks
    has_head_mask = (TASK_VIDEO_DIR / head_mask_name).exists()
    frames_head_mask = read_all_frames(TASK_VIDEO_DIR / head_mask_name) if has_head_mask else frames_high.copy()

    T = min(len(frames_high), len(frames_wrist), len(frames_mask))
    T = (T // frame_chunk_size) * frame_chunk_size
    if T == 0: T = frame_chunk_size
    frames_high  = frames_high[:T]
    frames_wrist = frames_wrist[:T]
    frames_mask  = frames_mask[:T]
    frames_head_mask = frames_head_mask[:T]
    logger.info(f"  T={T} frames, {T // frame_chunk_size} chunks")

    # wrist video: left|right are side-by-side along width axis → split at mid-width
    Ww = frames_wrist.shape[2]   # e.g. 1024
    frames_left  = frames_wrist[:, :, :Ww//2, :]   # left wrist
    frames_right = frames_wrist[:, :, Ww//2:, :]   # right wrist
    # mask wrist: same layout
    Wm = frames_mask.shape[2]
    frames_mask_left  = frames_mask[:, :, :Wm//2, :]
    frames_mask_right = frames_mask[:, :, Wm//2:, :]
    frames_mask_high  = frames_head_mask

    # ── resize and encode all frames via VAE ──
    logger.info("Encoding video frames with VAE ...")
    th  = resize_frames(frames_high,  CAM_HIGH_H, CAM_HIGH_W).to(DEVICE, DTYPE)  # [1,3,T,256,320]
    tl  = resize_frames(frames_left,  WRIST_H,    WRIST_W)   .to(DEVICE, DTYPE)  # [1,3,T,128,160]
    tr  = resize_frames(frames_right, WRIST_H,    WRIST_W)   .to(DEVICE, DTYPE)  # [1,3,T,128,160]
    tmh = resize_frames(frames_mask_high,  CAM_HIGH_H, CAM_HIGH_W).to(DEVICE, DTYPE)
    tml = resize_frames(frames_mask_left,  WRIST_H,    WRIST_W)   .to(DEVICE, DTYPE)
    tmr = resize_frames(frames_mask_right, WRIST_H,    WRIST_W)   .to(DEVICE, DTYPE)

    # Save GT frames for comparison
    vp = VideoProcessor(vae_scale_factor=1)
    export_to_video(vp.postprocess_video(th, output_type="np")[0], str(OUT_DIR / "gt_high.mp4"), fps=10)
    export_to_video(vp.postprocess_video(tmh, output_type="np")[0], str(OUT_DIR / "gt_mask_high.mp4"), fps=10)
    logger.info("  GT videos saved.")

    with torch.no_grad():
        vid_lat  = encode_tshape(vae, th, tl, tr, DEVICE, DTYPE)   # [1,16,T,24,20]
        mask_lat = encode_tshape(vae, tmh, tml, tmr, DEVICE, DTYPE)

    H = vid_lat.shape[-2]  # 24
    W = vid_lat.shape[-1]  # 20
    logger.info(f"  encoded latent: {vid_lat.shape}")

    # ── save GT decoded from latent ──
    decode_latent(vae, vid_lat,  OUT_DIR / "gt_video_from_lat.mp4")
    decode_latent(vae, mask_lat, OUT_DIR / "gt_mask_from_lat.mp4")

    # ── set up KV caches ──
    lat_tpc = (frame_chunk_size * H * W) // (patch_size[0] * patch_size[1] * patch_size[2])
    act_tpc = frame_chunk_size * cfg.action_per_frame
    use_cfg = GUIDANCE_SCALE > 1
    bs_cache = 2 if use_cfg else 1

    model.create_empty_cache("vid",  cfg.attn_window, lat_tpc, act_tpc,
                             dtype=DTYPE, device=DEVICE, batch_size=bs_cache, has_mask=False)
    model.create_empty_cache("mask", cfg.attn_window, lat_tpc, act_tpc,
                             dtype=DTYPE, device=DEVICE, batch_size=bs_cache, has_mask=True)

    # ── autoregressive inference ──
    num_chunks = T // frame_chunk_size
    logger.info(f"Running inference: {num_chunks} chunks × {NUM_STEPS} steps ...")
    vid_chunks, mask_chunks = [], []

    for cid in range(num_chunks):
        fst = cid * frame_chunk_size
        logger.info(f"  chunk {cid+1}/{num_chunks} (frame_st={fst})")
        v, m = infer_chunk(
            model, scheduler,
            vid_lat, mask_lat,
            text_emb, neg_emb,
            patch_size, DEVICE, DTYPE,
            NUM_STEPS, GUIDANCE_SCALE,
            frame_chunk_size, H, W,
            fst, "vid", "mask",
        )
        vid_chunks.append(v)
        mask_chunks.append(m)

    model.clear_cache("vid")
    model.clear_cache("mask")

    pred_vid  = torch.cat(vid_chunks,  dim=2)  # [1,16,T,24,20]
    pred_mask = torch.cat(mask_chunks, dim=2)

    # ── decode and save ──
    decode_latent(vae, pred_vid,  OUT_DIR / "demo_video.mp4")
    decode_latent(vae, pred_mask, OUT_DIR / "demo_mask.mp4")

    logger.info(f"\nDone → {OUT_DIR}")
    logger.info(f"  GT:        gt_video_from_lat.mp4  /  gt_mask_from_lat.mp4")
    logger.info(f"  Predicted: demo_video.mp4          /  demo_mask.mp4")


if __name__ == "__main__":
    main()
