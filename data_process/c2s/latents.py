"""把单视角 SIV map 编成 observation.masks.* latent（与 video latent 同 schema）。

编码约定完全对齐本机 wan_va/tools/extract_mask_latents.py：
  img[0,255] -> /255*2-1 -> 因果流式 encode_chunk([1,4,4,...]) -> mu=chunk(enc,2)[0]
  -> (mu - latents_mean) / latents_std -> (Fl,Hl,Wl,48) -> flatten (Fl*Hl*Wl,48)
磁盘上 video latent 是 normalized 的，所以 SIV latent 也必须 normalize。
"""

from __future__ import annotations

import os
from typing import Optional

import numpy as np
import torch


def siv_to_rgb(maps: np.ndarray, kinds: Optional[np.ndarray] = None) -> np.ndarray:
    """(T,H,W) float[0,1] -> (T,3,H,W) uint8。

    通道语义：R=grasp 接触，G=place/支撑接触，B=二者融合（强调交互点）。
    kinds 为 (T,) 的布尔数组时 True 表示该帧为 place 类接触。
    """
    T = maps.shape[0]
    rgb = np.zeros((T, 3, maps.shape[1], maps.shape[2]), dtype=np.float32)
    if kinds is None:
        rgb[:, 0] = maps
        rgb[:, 2] = maps
    else:
        grasp = ~np.asarray(kinds, dtype=bool)
        rgb[grasp, 0] = maps[grasp]
        rgb[~grasp, 1] = maps[~grasp]
        rgb[:, 2] = maps
    return (np.clip(rgb, 0.0, 1.0) * 255.0).astype(np.uint8)


class WanVAEEncoder:
    """Wan2.2 VAE 的因果流式编码器包装（与本地 WanVAEStreamingWrapper 等价）。"""

    def __init__(self, vae):
        self.vae = vae
        self.encoder = vae.encoder
        self.quant_conv = vae.quant_conv
        n = getattr(vae, "_cached_conv_counts", None)
        if n is not None:
            self.enc_conv_num = n["encoder"]
        else:
            self.enc_conv_num = sum(1 for m in self.encoder.modules()
                                    if m.__class__.__name__ == "WanCausalConv3d")
        self.clear_cache()

    def clear_cache(self):
        self.feat_cache = [None] * self.enc_conv_num

    def encode_chunk(self, x):
        # 与 wan_va/modules/utils.py::patchify / diffusers.patchify 逐行一致
        # （permute(0,1,6,4,2,3,5)），否则 latent 语义与已有 video latent 不一致
        ps = getattr(self.vae.config, "patch_size", None)
        if ps is not None and ps != 1:
            b, c, f, h, w = x.shape
            x = x.view(b, c, f, h // ps, ps, w // ps, ps)
            x = x.permute(0, 1, 6, 4, 2, 3, 5).contiguous()
            x = x.view(b, c * ps * ps, f, h // ps, w // ps)
        # feat_idx 每个 chunk 都从 0 开始（与参考实现一致），跨 chunk 复用会越界
        feat_idx = [0]
        out = self.encoder(x, feat_cache=self.feat_cache, feat_idx=feat_idx)
        return self.quant_conv(out)

    @torch.no_grad()
    def encode(self, rgb_uint8: np.ndarray, device="cuda", dtype=torch.bfloat16):
        """(T,3,H,W) uint8 -> normalized latent (Fl,Hl,Wl,C)。"""
        x = torch.from_numpy(rgb_uint8).float() / 255.0 * 2.0 - 1.0
        x = x.permute(1, 0, 2, 3).unsqueeze(0).to(device).to(dtype)
        T = x.shape[2]
        chunk_sizes = [1] + [4] * ((T - 1) // 4)
        self.clear_cache()
        encs, idx = [], 0
        for cs in chunk_sizes:
            encs.append(self.encode_chunk(x[:, :, idx:idx + cs]))
            idx += cs
        enc = torch.cat(encs, dim=2)
        mu, _ = torch.chunk(enc, 2, dim=1)
        mean = torch.tensor(self.vae.config.latents_mean, device=mu.device).view(1, -1, 1, 1, 1)
        std = torch.tensor(self.vae.config.latents_std, device=mu.device).view(1, -1, 1, 1, 1)
        mu = (mu.float() - mean) / std
        return mu[0].permute(1, 2, 3, 0).contiguous()


def load_vae(vae_path: str, device="cuda", dtype=torch.bfloat16):
    from diffusers import AutoencoderKLWan
    vae = AutoencoderKLWan.from_pretrained(vae_path, torch_dtype=dtype).to(device).eval()
    return vae


def save_mask_latent(out_path: str, latent_fhwc: torch.Tensor, video_latent: dict):
    """写成与同 episode 同相机的 video latent 完全同 schema 的 .pth。"""
    Fl, Hl, Wl, C = latent_fhwc.shape
    flat = latent_fhwc.reshape(Fl * Hl * Wl, C)
    out = {
        "latent": flat.to(torch.bfloat16).cpu(),
        "latent_num_frames": Fl, "latent_height": Hl, "latent_width": Wl,
        "video_num_frames": video_latent["video_num_frames"],
        "video_height": video_latent["video_height"],
        "video_width": video_latent["video_width"],
        "text_emb": video_latent["text_emb"], "text": video_latent["text"],
        "frame_ids": video_latent["frame_ids"],
        "start_frame": video_latent["start_frame"], "end_frame": video_latent["end_frame"],
        "fps": video_latent["fps"], "ori_fps": video_latent["ori_fps"],
    }
    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    torch.save(out, out_path)
    return out_path
