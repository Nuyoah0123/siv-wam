import math
from copy import deepcopy
from functools import partial
from typing import Callable, ClassVar  # noqa: UP035

import torch
import torch.nn as nn  # noqa: PLR0402
import torch.nn.functional as F
from diffusers.configuration_utils import ConfigMixin, register_to_config
from diffusers.models.attention import FeedForward
from diffusers.models.embeddings import (
    PixArtAlphaTextProjection,
    TimestepEmbedding,
    Timesteps,
)
from diffusers.models.modeling_utils import ModelMixin
from diffusers.models.normalization import FP32LayerNorm
from einops import rearrange
from torch.nn.attention.flex_attention import (
    BlockMask,
    _mask_mod_signature,
    and_masks,
    create_block_mask,
    flex_attention,
    or_masks,
)

try:
    from flash_attn_interface import flash_attn_func  # type: ignore
except Exception:  # noqa: BLE001
    try:
        from flash_attn import flash_attn_func
    except Exception:  # noqa: BLE001
        flash_attn_func = None

__all__ = ["WanTransformer3DModel"]


def custom_sdpa(q, k, v):
    out = F.scaled_dot_product_attention(
        q.transpose(1, 2), k.transpose(1, 2), v.transpose(1, 2)
    )
    return out.transpose(1, 2)


class FlexAttnFunc(nn.Module):
    flex_attn: ClassVar[Callable] = torch.compile(
        flex_attention,
        dynamic=True,
    )
    compiled_create_block_mask: ClassVar[Callable] = torch.compile(create_block_mask)
    attention_mask: ClassVar[BlockMask] = None
    cross_attention_mask: ClassVar[BlockMask] = None

    def __init__(
        self,
        is_cross=False,
    ) -> None:
        super().__init__()
        self.is_cross = is_cross

    def forward(
        self,
        query: torch.Tensor,
        key: torch.Tensor,
        value: torch.Tensor,
        dtype=torch.bfloat16,
    ) -> torch.Tensor:
        q_varlen = rearrange(query[0], "s n d -> 1 n s d")
        k_varlen = rearrange(key[0], "s n d -> 1 n s d")
        v_varlen = rearrange(value[0], "s n d -> 1 n s d")

        half_dtypes = (torch.float16, torch.bfloat16)
        assert dtype in half_dtypes

        def half(x):
            return x if x.dtype in half_dtypes else x.to(dtype)

        q_varlen = half(q_varlen)
        k_varlen = half(k_varlen)
        v_varlen = half(v_varlen)
        q_varlen = q_varlen.to(v_varlen.dtype)
        k_varlen = k_varlen.to(v_varlen.dtype)

        block_mask = (
            FlexAttnFunc.cross_attention_mask
            if self.is_cross
            else FlexAttnFunc.attention_mask
        )

        x_out = FlexAttnFunc.flex_attn(
            q_varlen,
            k_varlen,
            v_varlen,
            block_mask=block_mask,
            kernel_options={
                "BLOCK_M": 64,
                "BLOCK_N": 64,
                "BLOCK_M1": 32,
                "BLOCK_N1": 64,
                "BLOCK_M2": 64,
                "BLOCK_N2": 32,
            },
        )

        x_out = rearrange(x_out, "b n s d -> b s n d")
        return x_out

    @staticmethod
    @torch.no_grad()
    def init_mask(
        latent_shape,
        action_shape,
        padded_length,
        chunk_size,
        window_size,
        patch_size,
        device,
        mask_shape=None,  # B2: when set, mask tokens are inserted between vid_cond and act_noisy
    ):
        """
        根据视频、mask、action token 的身份、时间位置和 noisy/condition 状态，
        构造 Transformer 的注意力可见性规则。
        某个 query token能不能看到某个 key/value token
        """
        torch._inductor.config.realize_opcount_threshold = 100
        B, _, L_F, L_H, L_W = latent_shape
        _, _, A_F, A_H, A_W = action_shape

        # seq_id 表示的是 batch 样本身份，不是 video/mask/action 身份。
        latent_seq_id = (
            torch.arange(B)[:, None, None, None]
            .expand(
                -1, L_F // patch_size[0], L_H // patch_size[1], L_W // patch_size[2]
            )
            .flatten()
        )
        action_seq_id = (
            torch.arange(B)[:, None, None, None].expand(-1, A_F, A_H, A_W).flatten()
        )

        # 表示每个视频 token 来自哪个 latent 时间帧。
        latent_frame_id = (
            torch.arange(L_F)[None, :, None, None]
            .expand(B, -1, L_H // patch_size[1], L_W // patch_size[2])[None]
            .flatten()
        )
        action_frame_id = (
            torch.arange(A_F)[None, :, None, None]
            .expand(B, -1, A_H, A_W)[None]
            .flatten()
        )

        if mask_shape is not None:
            # B2 layout: [vid_noisy | vid_cond | mask_noisy | mask_cond | act_noisy | act_cond]
            # Mask and video may have different spatial token counts.
            # noise_ids: mask_noisy=0 (like vid_noisy), mask_cond=1 (like vid_cond).
            # This lets mask tokens cross-attend to video tokens in the same frame (noise2noise)
            # and to clean video/mask conditioning from earlier frames (noise2clean).
            """
            一个 token 最终能看到另一个 token，必须同时满足：
                1. 属于同一个 batch 样本
                2. 满足 noisy/clean 关系
                    clean -> clean: 
                3. 满足时间因果关系 (condition token 可以看到当前及更早时间块的 condition token，
                    但不能看到未来 condition token。)
                4. 没有超出 window_size
                5. 不是 padding token
            seq_ids 防止不同 batch 样本相互泄漏。
            frame_ids 控制时间顺序和时间块关系。
            noise_ids 区分 noisy token 与 condition token。
            video、mask、action 的交互主要发生在共同的 self-attention 中。
            text 通过独立的 cross-attention 注入，不等同于 video/action 之间的 self-attention。
            """
            _, _, M_F, M_H, M_W = mask_shape
            mask_seq_id = (
                torch.arange(B)[:, None, None, None]
                .expand(
                    -1, M_F // patch_size[0], M_H // patch_size[1], M_W // patch_size[2]
                )
                .flatten()
            )
            mask_frame_id = (
                torch.arange(M_F)[None, :, None, None]
                .expand(B, -1, M_H // patch_size[1], M_W // patch_size[2])
                .flatten()
            )
            seq_ids = torch.cat(
                [latent_seq_id] * 2 + [mask_seq_id] * 2 + [action_seq_id] * 2
            ) # 这里的 * 2 对应 noisy 和 condition 两份 token。
            """
            视频和 mask 使用偶数时间块, action 使用相邻的奇数时间块
            例如：chunk_size = 2
            视频和 mask 的 frame id 大致为：
            原始帧：       0   1   2   3   4   5
            视频/mask：    0   0   2   2   4   4
            动作的 frame id 会额外加 1：
            动作：         1   1   3   3   5   5"""
            vid_fid = latent_frame_id // chunk_size * 2
            mask_fid = mask_frame_id // chunk_size * 2
            act_fid = action_frame_id // chunk_size * 2 + 1
            frame_ids = torch.cat([vid_fid] * 2 + [mask_fid] * 2 + [act_fid] * 2)
            noise_ids = torch.cat(
                [
                    torch.zeros_like(latent_frame_id),  # vid_noisy   noise_id=0
                    torch.ones_like(latent_frame_id),  # vid_cond    noise_id=1
                    torch.zeros_like(mask_frame_id),  # mask_noisy  noise_id=0
                    torch.ones_like(mask_frame_id),  # mask_cond   noise_id=1
                    torch.zeros_like(action_frame_id),  # act_noisy
                    torch.ones_like(action_frame_id),  # act_cond
                ]
            )
        else:
            # Original layout: [vid_noisy | vid_cond | act_noisy | act_cond]
            seq_ids = torch.cat([latent_seq_id] * 2 + [action_seq_id] * 2)
            frame_ids = torch.cat(
                [latent_frame_id // chunk_size * 2] * 2
                + [action_frame_id // chunk_size * 2 + 1] * 2
            )
            noise_ids = torch.cat(
                [
                    torch.zeros_like(latent_frame_id),
                    torch.ones_like(latent_frame_id),
                    torch.zeros_like(action_frame_id),
                    torch.ones_like(action_frame_id),
                ]
            )

        seq_ids = F.pad(seq_ids, (0, padded_length), value=-1)
        frame_ids = F.pad(frame_ids, (0, padded_length), value=-1)
        noise_ids = F.pad(noise_ids, (0, padded_length), value=-1)

        mask_mod = FlexAttnFunc._get_mask_mod(
            seq_ids.long().to(device),
            frame_ids.long().to(device),
            noise_ids.long().to(device),
            window_size,
        )
        block_mask = FlexAttnFunc.compiled_create_block_mask(
            mask_mod, 1, 1, len(seq_ids), len(seq_ids), device=device, _compile=True
        )
        FlexAttnFunc.attention_mask = block_mask

        text_seq_ids = torch.arange(B)[:, None].expand(-1, 512).flatten()
        mask_mod_cross = FlexAttnFunc._get_cross_mask_mod(
            seq_ids.long().to(device), text_seq_ids.long().to(device)
        )
        block_mask_cross = FlexAttnFunc.compiled_create_block_mask(
            mask_mod_cross,
            1,
            1,
            len(seq_ids),
            len(text_seq_ids),
            device=device,
            _compile=True,
        )
        FlexAttnFunc.cross_attention_mask = block_mask_cross

    @staticmethod
    @torch.no_grad()
    def _get_cross_mask_mod(seq_ids, text_seq_ids):
        def seq_mask(
            b: torch.Tensor, h: torch.Tensor, q_idx: torch.Tensor, kv_idx: torch.Tensor
        ):
            return (
                (seq_ids[q_idx] == text_seq_ids[kv_idx])
                & (seq_ids[q_idx] >= 0)
                & (text_seq_ids[kv_idx] >= 0)
            )

        return seq_mask

    @staticmethod
    @torch.no_grad()
    def _get_mask_mod(seq_ids, frame_ids, noise_ids, window_size):
        def seq_mask(
            b: torch.Tensor, h: torch.Tensor, q_idx: torch.Tensor, kv_idx: torch.Tensor
        ):
            return (
                (seq_ids[q_idx] == seq_ids[kv_idx])
                & (seq_ids[q_idx] >= 0)
                & (seq_ids[kv_idx] >= 0)
            )

        def block_causal_mask(
            b: torch.Tensor, h: torch.Tensor, q_idx: torch.Tensor, kv_idx: torch.Tensor
        ):
            return frame_ids[kv_idx] <= frame_ids[q_idx]

        def block_causal_mask_exclude_self(
            b: torch.Tensor, h: torch.Tensor, q_idx: torch.Tensor, kv_idx: torch.Tensor
        ):
            return frame_ids[kv_idx] < frame_ids[q_idx]

        def block_self_mask(
            b: torch.Tensor, h: torch.Tensor, q_idx: torch.Tensor, kv_idx: torch.Tensor
        ):
            return frame_ids[kv_idx] == frame_ids[q_idx]

        def clean2clean_mask(
            b: torch.Tensor, h: torch.Tensor, q_idx: torch.Tensor, kv_idx: torch.Tensor
        ):
            # condition token 可以看到当前及更早时间块的 condition token，但不能看到未来 condition token。
            return (noise_ids[q_idx] == 1) & (noise_ids[kv_idx] == 1)

        def noise2clean_mask(
            b: torch.Tensor, h: torch.Tensor, q_idx: torch.Tensor, kv_idx: torch.Tensor
        ):
            """
            只能看更早时间的 clean token
            不能看同一时间块的 clean token
            不能看未来 clean token
            """
            return (noise_ids[q_idx] == 0) & (noise_ids[kv_idx] == 1)

        def noise2noise_mask(
            b: torch.Tensor, h: torch.Tensor, q_idx: torch.Tensor, kv_idx: torch.Tensor
        ):
            # noisy token 可以看到当前时间块内的 noisy token。
            return (noise_ids[q_idx] == 0) & (noise_ids[kv_idx] == 0)

        def block_window_mask(
            b: torch.Tensor,
            h: torch.Tensor,
            q_idx: torch.Tensor,
            kv_idx: torch.Tensor,
            window_size: int,
        ):
            # 模型使用的是一个局部时间窗口，而不是无限制地访问整个序列。
            return (frame_ids[q_idx] - frame_ids[kv_idx]).abs() <= window_size

        mask_list = []
        mask_list.append(and_masks(clean2clean_mask, block_causal_mask))
        mask_list.append(and_masks(noise2clean_mask, block_causal_mask_exclude_self))
        mask_list.append(and_masks(noise2noise_mask, block_self_mask))
        mask = or_masks(*mask_list)
        mask = and_masks(mask, seq_mask)
        mask = and_masks(mask, partial(block_window_mask, window_size=window_size))
        return mask


class WanTimeTextImageEmbedding(nn.Module):
    def __init__(
        self,
        dim,
        time_freq_dim,
        time_proj_dim,
        text_embed_dim,
        pos_embed_seq_len,
    ):
        super().__init__()

        self.timesteps_proj = Timesteps(
            num_channels=time_freq_dim, flip_sin_to_cos=True, downscale_freq_shift=0
        )
        self.time_embedder = TimestepEmbedding(
            in_channels=time_freq_dim, time_embed_dim=dim
        )
        self.act_fn = nn.SiLU()
        self.time_proj = nn.Linear(dim, time_proj_dim)
        self.text_embedder = PixArtAlphaTextProjection(
            text_embed_dim, dim, act_fn="gelu_tanh"
        )

    def forward(
        self,
        timestep: torch.Tensor,
        dtype=None,
    ):
        B, L = timestep.shape
        timestep = timestep.reshape(-1)
        timestep = self.timesteps_proj(timestep)
        # time_embedder_dtype = next(iter(self.time_embedder.parameters())).dtype
        time_embedder_dtype = self.time_embedder.linear_1.weight.dtype
        if timestep.dtype != time_embedder_dtype and time_embedder_dtype != torch.int8:
            timestep = timestep.to(time_embedder_dtype)
        temb = self.time_embedder(timestep).to(dtype=dtype)
        timestep_proj = self.time_proj(self.act_fn(temb))
        return temb.reshape(B, L, -1), timestep_proj.reshape(B, L, -1)


class WanRotaryPosEmbed(nn.Module):
    def __init__(
        self,
        attention_head_dim: int,
        patch_size,
        max_seq_len: int,
        theta: float = 10000.0,
    ):
        super().__init__()

        self.attention_head_dim = attention_head_dim
        self.patch_size = patch_size
        self.max_seq_len = max_seq_len
        self.theta = theta

        self.f_dim = self.attention_head_dim - 2 * (self.attention_head_dim // 3)
        self.h_dim = self.attention_head_dim // 3
        self.w_dim = self.attention_head_dim // 3

        # Precompute and register buffers
        f_freqs_base, h_freqs_base, w_freqs_base = self._precompute_freqs_base()
        self.f_freqs_base = f_freqs_base
        self.h_freqs_base = h_freqs_base
        self.w_freqs_base = w_freqs_base

    def _precompute_freqs_base(self):
        # freqs_base = 1.0 / (theta ** (2k / dim))
        f_freqs_base = 1.0 / (
            self.theta
            ** (
                torch.arange(0, self.f_dim, 2)[: (self.f_dim // 2)].double()
                / self.f_dim
            )
        )
        h_freqs_base = 1.0 / (
            self.theta
            ** (
                torch.arange(0, self.h_dim, 2)[: (self.h_dim // 2)].double()
                / self.h_dim
            )
        )
        w_freqs_base = 1.0 / (
            self.theta
            ** (
                torch.arange(0, self.w_dim, 2)[: (self.w_dim // 2)].double()
                / self.w_dim
            )
        )
        return f_freqs_base, h_freqs_base, w_freqs_base

    def forward(self, grid_ids):
        with torch.no_grad():
            f_freqs = grid_ids[:, 0, :].unsqueeze(-1) * self.f_freqs_base.to(
                grid_ids.device
            )
            h_freqs = grid_ids[:, 1, :].unsqueeze(-1) * self.h_freqs_base.to(
                grid_ids.device
            )
            w_freqs = grid_ids[:, 2, :].unsqueeze(-1) * self.w_freqs_base.to(
                grid_ids.device
            )
            freqs = torch.cat([f_freqs, h_freqs, w_freqs], dim=-1).float()
            freqs_cis = torch.polar(torch.ones_like(freqs), freqs)

        return freqs_cis


class WanAttention(torch.nn.Module):
    def __init__(
        self,
        dim,
        heads=8,
        dim_head=64,
        eps=1e-5,
        dropout=0.0,
        cross_attention_dim_head=None,
        attn_mode="torch",
    ):
        super().__init__()
        if attn_mode == "torch":
            self.attn_op = custom_sdpa
        elif attn_mode == "flashattn":
            self.attn_op = flash_attn_func
        elif attn_mode == "flex":
            self.attn_op = FlexAttnFunc(cross_attention_dim_head is not None)
        else:
            raise ValueError(
                f"Unsupported attention mode: {attn_mode}, only support torch and flashattn"
            )

        self.inner_dim = dim_head * heads
        self.heads = heads
        self.cross_attention_dim_head = cross_attention_dim_head
        self.kv_inner_dim = (
            self.inner_dim
            if cross_attention_dim_head is None
            else cross_attention_dim_head * heads
        )

        self.to_q = torch.nn.Linear(dim, self.inner_dim, bias=True)
        self.to_k = torch.nn.Linear(dim, self.kv_inner_dim, bias=True)
        self.to_v = torch.nn.Linear(dim, self.kv_inner_dim, bias=True)
        self.to_out = torch.nn.ModuleList(
            [
                torch.nn.Linear(self.inner_dim, dim, bias=True),
                torch.nn.Dropout(dropout),
            ]
        )
        self.norm_q = torch.nn.RMSNorm(
            dim_head * heads, eps=eps, elementwise_affine=True
        )
        self.norm_k = torch.nn.RMSNorm(
            dim_head * heads, eps=eps, elementwise_affine=True
        )
        self.attn_caches = {} if cross_attention_dim_head is None else None

    def clear_pred_cache(self, cache_name):
        if self.attn_caches is None:
            return
        cache = self.attn_caches[cache_name]
        is_pred = cache["is_pred"]
        cache["mask"][is_pred] = False

    def clear_cache(self, cache_name):
        if self.attn_caches is None:
            return
        self.attn_caches[cache_name] = None

    def init_kv_cache(
        self, cache_name, total_tolen, num_head, head_dim, device, dtype, batch_size
    ):
        if self.attn_caches is None:
            return
        self.attn_caches[cache_name] = {
            "k": torch.empty(
                [batch_size, total_tolen, num_head, head_dim],
                device=device,
                dtype=dtype,
            ),
            "v": torch.empty(
                [batch_size, total_tolen, num_head, head_dim],
                device=device,
                dtype=dtype,
            ),
            "id": torch.full((total_tolen,), -1, device=device),
            "mask": torch.zeros((total_tolen,), dtype=torch.bool, device=device),
            "is_pred": torch.zeros((total_tolen,), dtype=torch.bool, device=device),
        }

    def allocate_slots(self, cache_name, key_size):
        cache = self.attn_caches[cache_name]
        mask = cache["mask"]
        ids = cache["id"]
        free = (~mask).nonzero(as_tuple=False).squeeze(-1)

        if free.numel() < key_size:
            used = mask.nonzero(as_tuple=False).squeeze(-1)

            used_ids = ids[used]
            order = torch.argsort(used_ids)
            need = key_size - free.numel()
            to_free = used[order[:need]]

            mask[to_free] = False
            ids[to_free] = -1
            free = (~mask).nonzero(as_tuple=False).squeeze(-1)

        assert free.numel() >= key_size
        return free[:key_size]

    def _next_cache_id(self, cache_name):
        ids = self.attn_caches[cache_name]["id"]
        mask = self.attn_caches[cache_name]["mask"]

        if mask.any():
            return ids[mask].max() + 1
        else:
            return torch.tensor(0, device=ids.device, dtype=ids.dtype)

    def update_cache(self, cache_name, key, value, is_pred):
        cache = self.attn_caches[cache_name]

        key_size = key.shape[1]
        slots = self.allocate_slots(cache_name, key_size)

        new_id = self._next_cache_id(cache_name)

        cache["k"][:, slots] = key
        cache["v"][:, slots] = value
        cache["mask"][slots] = True
        cache["id"][slots] = new_id
        cache["is_pred"][slots] = is_pred
        return slots

    def restore_cache(self, cache_name, slots):
        self.attn_caches[cache_name]["mask"][slots] = False

    def forward(
        self,
        q,
        k,
        v,
        rotary_emb,
        update_cache=0,
        cache_name="pos",
    ):
        kv_cache = (
            self.attn_caches[cache_name]
            if (self.attn_caches is not None) and (cache_name in self.attn_caches)
            else None
        )

        query, key, value = self.to_q(q), self.to_k(k), self.to_v(v)
        query = self.norm_q(query)
        query = query.unflatten(2, (self.heads, -1))
        key = self.norm_k(key)
        key = key.unflatten(2, (self.heads, -1))
        value = value.unflatten(2, (self.heads, -1))
        if rotary_emb is not None:

            def apply_rotary_emb(x, freqs):
                x_out = torch.view_as_complex(
                    x.to(torch.float64).reshape(
                        x.shape[0], x.shape[1], x.shape[2], -1, 2
                    )
                )
                x_out = torch.view_as_real(x_out * freqs).flatten(3)
                return x_out.to(x.dtype)

            query = apply_rotary_emb(query, rotary_emb)
            key = apply_rotary_emb(key, rotary_emb)
        slots = None
        if kv_cache is not None and kv_cache["k"] is not None:
            slots = self.update_cache(
                cache_name, key, value, is_pred=(update_cache == 1)
            )
            key_pool = self.attn_caches[cache_name]["k"]
            value_pool = self.attn_caches[cache_name]["v"]
            mask = self.attn_caches[cache_name]["mask"]
            valid = mask.nonzero(as_tuple=False).squeeze(-1)
            key = key_pool[:, valid]
            value = value_pool[:, valid]

        hidden_states = self.attn_op(query, key, value)

        if update_cache == 0:  # noqa: SIM102
            if kv_cache is not None and kv_cache["k"] is not None:
                self.restore_cache(cache_name, slots)

        hidden_states = hidden_states.flatten(2, 3)
        hidden_states = hidden_states.type_as(query)
        hidden_states = self.to_out[0](hidden_states)
        hidden_states = self.to_out[1](hidden_states)
        return hidden_states


class WanTransformerBlock(nn.Module):
    def __init__(
        self,
        dim,
        ffn_dim,
        num_heads,
        cross_attn_norm=False,
        eps=1e-6,
        attn_mode: str = "flashattn",
    ):
        super().__init__()
        self.attn_mode = attn_mode

        # 1. Self-attention
        self.norm1 = FP32LayerNorm(dim, eps, elementwise_affine=False)
        self.attn1 = WanAttention(
            dim=dim,
            heads=num_heads,
            dim_head=dim // num_heads,
            eps=eps,
            cross_attention_dim_head=None,
            attn_mode=attn_mode,
        )

        # 2. Cross-attention
        self.attn2 = WanAttention(
            dim=dim,
            heads=num_heads,
            dim_head=dim // num_heads,
            eps=eps,
            cross_attention_dim_head=dim // num_heads,
            attn_mode=attn_mode,
        )
        self.norm2 = (
            FP32LayerNorm(dim, eps, elementwise_affine=True)
            if cross_attn_norm
            else nn.Identity()
        )

        # 3. Feed-forward
        self.ffn = FeedForward(dim, inner_dim=ffn_dim, activation_fn="gelu-approximate")
        self.norm3 = FP32LayerNorm(dim, eps, elementwise_affine=False)

        self.scale_shift_table = nn.Parameter(torch.randn(1, 6, dim) / dim**0.5)

    def forward(
        self,
        hidden_states,
        encoder_hidden_states,
        temb,
        rotary_emb,
        update_cache=0,
        cache_name="pos",
    ) -> torch.Tensor:
        temb_scale_shift_table = self.scale_shift_table[None] + temb.float()
        shift_msa, scale_msa, gate_msa, c_shift_msa, c_scale_msa, c_gate_msa = (
            rearrange(temb_scale_shift_table, "b l n c -> b n l c").chunk(6, dim=1)
        )
        shift_msa = shift_msa.squeeze(1)
        scale_msa = scale_msa.squeeze(1)
        gate_msa = gate_msa.squeeze(1)
        c_shift_msa = c_shift_msa.squeeze(1)
        c_scale_msa = c_scale_msa.squeeze(1)
        c_gate_msa = c_gate_msa.squeeze(1)
        # 1. Self-attention
        norm_hidden_states = (
            self.norm1(hidden_states.float()) * (1.0 + scale_msa) + shift_msa
        ).type_as(hidden_states)
        attn_output = self.attn1(
            norm_hidden_states,
            norm_hidden_states,
            norm_hidden_states,
            rotary_emb,
            update_cache=update_cache,
            cache_name=cache_name,
        )
        hidden_states = (hidden_states.float() + attn_output * gate_msa).type_as(
            hidden_states
        )

        # 2. Cross-attention
        norm_hidden_states = self.norm2(hidden_states.float()).type_as(hidden_states)
        attn_output = self.attn2(
            norm_hidden_states,
            encoder_hidden_states,
            encoder_hidden_states,
            None,
            update_cache=0,
            cache_name=cache_name,
        )
        hidden_states = hidden_states + attn_output

        # 3. Feed-forward
        norm_hidden_states = (
            self.norm3(hidden_states.float()) * (1.0 + c_scale_msa) + c_shift_msa
        ).type_as(hidden_states)

        ff_output = self.ffn(norm_hidden_states)

        hidden_states = (
            hidden_states.float() + ff_output.float() * c_gate_msa
        ).type_as(hidden_states)
        return hidden_states


class WanTransformer3DModel(ModelMixin, ConfigMixin):
    r"""
    TODO
    """

    _supports_gradient_checkpointing = True
    _skip_layerwise_casting_patterns = [  # noqa: RUF012
        # "patch_embedding",
        "patch_embedding_mlp",
        "condition_embedder",
        "condition_embedder_action",
        "norm",
    ]
    _no_split_modules = ["WanTransformerBlock"]  # noqa: RUF012
    _keep_in_fp32_modules = [  # noqa: RUF012
        "time_embedder",
        "scale_shift_table",
        "scale_shift_table_action",
        "norm1",
        "action_norm1",
        "text_norm1",
        "norm2",
        "action_norm2",
        "text_norm2",
        "norm3",
        "action_norm3",
        "text_norm3",
    ]
    _keys_to_ignore_on_load_unexpected = ["norm_added_q"]  # noqa: RUF012
    _repeated_blocks = ["WanTransformerBlock"]  # noqa: RUF012

    @register_to_config
    def __init__(
        self,
        patch_size=[1, 2, 2],  # noqa: B006
        num_attention_heads=24,
        attention_head_dim=128,
        in_channels=48,
        out_channels=48,
        action_dim=30,
        text_dim=4096,
        freq_dim=256,
        ffn_dim=14336,
        num_layers=30,
        cross_attn_norm=True,
        eps=1e-06,
        rope_max_seq_len=1024,
        pos_embed_seq_len=None,
        attn_mode="torch",
    ):
        r"""
        TODO
        """
        super().__init__()
        self.patch_size = patch_size
        self.num_attention_heads = num_attention_heads
        self.attention_head_dim = attention_head_dim
        inner_dim = num_attention_heads * attention_head_dim
        self.rope = WanRotaryPosEmbed(attention_head_dim, patch_size, rope_max_seq_len)
        self.patch_embedding_mlp = nn.Linear(
            in_channels * patch_size[0] * patch_size[1] * patch_size[2], inner_dim
        )
        self.action_embedder = nn.Linear(action_dim, inner_dim)
        self.condition_embedder = WanTimeTextImageEmbedding(
            dim=inner_dim,
            time_freq_dim=freq_dim,
            time_proj_dim=inner_dim * 6,
            text_embed_dim=text_dim,
            pos_embed_seq_len=pos_embed_seq_len,
        )
        self.condition_embedder_action = deepcopy(self.condition_embedder)

        self.blocks = nn.ModuleList(
            [
                WanTransformerBlock(
                    inner_dim,
                    ffn_dim,
                    num_attention_heads,
                    cross_attn_norm,
                    eps,
                    attn_mode=attn_mode,
                )
                for _ in range(num_layers)
            ]
        )

        self.norm_out = FP32LayerNorm(inner_dim, eps, elementwise_affine=False)
        self.proj_out = nn.Linear(inner_dim, out_channels * math.prod(patch_size))
        self.action_proj_out = nn.Linear(inner_dim, action_dim)
        self.scale_shift_table = nn.Parameter(
            torch.randn(1, 2, inner_dim) / inner_dim**0.5
        )

        # B2 mask head: parallel to proj_out, operates on mask-stream backbone features.
        # Backbone processes [vid_noisy | vid_cond | mask_noisy | mask_cond | act_noisy | act_cond]
        # together, so mask tokens attend to video tokens (cross-domain) at the same frame
        # and to clean video/mask from earlier frames (noise2clean).  This gives the mask head
        # the same information richness as the video head, eliminating the ~1.0 loss plateau
        # that occurred when mask_proj_out only saw video backbone features.
        self.mask_proj_out = nn.Linear(inner_dim, out_channels * math.prod(patch_size))
        nn.init.zeros_(self.mask_proj_out.weight)
        nn.init.zeros_(self.mask_proj_out.bias)

    def clear_cache(self, cache_name):
        for block in self.blocks:
            block.attn1.clear_cache(cache_name)

    def clear_pred_cache(self, cache_name):
        for block in self.blocks:
            block.attn1.clear_pred_cache(cache_name)

    def create_empty_cache(
        self,
        cache_name,
        attn_window,
        latent_token_per_chunk,
        action_token_per_chunk,
        device,
        dtype,
        batch_size,
        has_mask=False,
    ):
        # B2: mask tokens double the latent-stream KV entries in the cache.
        lat_slots = latent_token_per_chunk * (2 if has_mask else 1)
        total_tolen = (attn_window // 2) * lat_slots + (
            attn_window // 2
        ) * action_token_per_chunk
        for block in self.blocks:
            block.attn1.init_kv_cache(
                cache_name,
                total_tolen,
                self.num_attention_heads,
                self.attention_head_dim,
                device,
                dtype,
                batch_size,
            )

    def _input_embed(self, latents, input_type="latent"):
        if input_type == "latent":
            hidden_states = rearrange(
                latents,
                "b c (f p1) (h p2) (w p3) -> b (f h w) (c p1 p2 p3)",
                p1=self.patch_size[0],
                p2=self.patch_size[1],
                p3=self.patch_size[2],
            )
            hidden_states = self.patch_embedding_mlp(hidden_states)
        elif input_type == "action":
            hidden_states = rearrange(latents, "b c f h w -> b (f h w) c")
            hidden_states = self.action_embedder(hidden_states)
        elif input_type == "text":
            hidden_states = self.condition_embedder.text_embedder(latents)
        else:
            raise ValueError(f"Unsupported input type: {input_type}")
        return hidden_states

    def _time_embed(self, timesteps, H, W, dtype, action_mode=False):
        pach_scale_h, pach_scale_w = (
            (1, 1) if action_mode else (self.patch_size[1], self.patch_size[2])
        )
        latent_time_steps = torch.repeat_interleave(
            timesteps, (H // pach_scale_h) * (W // pach_scale_w), dim=1
        )  # L
        current_condition_embedder = (
            self.condition_embedder_action if action_mode else self.condition_embedder
        )
        temb, timestep_proj = current_condition_embedder(latent_time_steps, dtype=dtype)
        timestep_proj = timestep_proj.unflatten(2, (6, -1))  # B L 6 C
        return temb, timestep_proj

    def forward_train(self, input_dict):
        # The training interface accepts the original video/action pair and the
        # optional mask stream used by SIV training.  Keeping the mask stream
        # optional lets train.py and train_siv.py share one model entry point.
        has_mask = "mask_dict" in input_dict and input_dict["mask_dict"] is not None
        for key in ("noisy_latents", "latent"):
            input_dict["latent_dict"][key] = input_dict["latent_dict"][key].to(
                torch.bfloat16
            )
            input_dict["action_dict"][key] = input_dict["action_dict"][key].to(
                torch.bfloat16
            )
            if has_mask:
                input_dict["mask_dict"][key] = input_dict["mask_dict"][key].to(
                    torch.bfloat16
                )

        latent_dict = input_dict["latent_dict"]
        mask_dict = input_dict.get("mask_dict")
        action_dict = input_dict["action_dict"]
        batch_size = latent_dict["noisy_latents"].shape[0]

        # ── Embed the video/action streams and, when supplied, the mask streams ──
        vid_noisy_hs = self._input_embed(
            latent_dict["noisy_latents"], "latent"
        ).flatten(0, 1)[None]
        vid_cond_hs = self._input_embed(latent_dict["latent"], "latent").flatten(0, 1)[
            None
        ]
        act_noisy_hs = self._input_embed(
            action_dict["noisy_latents"], "action"
        ).flatten(0, 1)[None]
        act_cond_hs = self._input_embed(action_dict["latent"], "action").flatten(0, 1)[
            None
        ]
        text_hs = self._input_embed(latent_dict["text_emb"], "text").flatten(0, 1)[None]

        if has_mask:
            msk_noisy_hs = self._input_embed(
                mask_dict["noisy_latents"], "latent"
            ).flatten(0, 1)[None]
            msk_cond_hs = self._input_embed(mask_dict["latent"], "latent").flatten(
                0, 1
            )[None]
            # !B2 layout: [vid_noisy | vid_cond | mask_noisy | mask_cond | act_noisy | act_cond]
            stream_states = [
                vid_noisy_hs,
                vid_cond_hs,
                msk_noisy_hs,
                msk_cond_hs,
                act_noisy_hs,
                act_cond_hs,
            ]
        else:
            # Original layout: [vid_noisy | vid_cond | act_noisy | act_cond]
            stream_states = [vid_noisy_hs, vid_cond_hs, act_noisy_hs, act_cond_hs]
        hidden_states = torch.cat(stream_states, dim=1)

        # ── Positional embeddings ────────────────────────────────────────────
        latent_grid_id = latent_dict["grid_id"].permute(1, 0, 2).flatten(1)[None]
        mask_grid_id = None
        if has_mask:
            mask_grid_id = mask_dict["grid_id"].permute(1, 0, 2).flatten(1)[None]
        action_grid_id = action_dict["grid_id"].permute(1, 0, 2).flatten(1)[None]
        if has_mask:
            full_grid_id = torch.cat(
                [latent_grid_id] * 2 + [mask_grid_id] * 2 + [action_grid_id] * 2,
                dim=2,
            )
        else:
            full_grid_id = torch.cat([latent_grid_id] * 2 + [action_grid_id] * 2, dim=2)
        rotary_emb = self.rope(full_grid_id)[:, :, None]

        # ── Timestep conditioning ────────────────────────────────────────────
        # mask tokens share the same noise timestep as video tokens
        L_H = latent_dict["noisy_latents"].shape[-2]
        L_W = latent_dict["noisy_latents"].shape[-1]
        A_H = action_dict["noisy_latents"].shape[-2]
        A_W = action_dict["noisy_latents"].shape[-1]

        vid_ts = torch.cat(
            [
                latent_dict["timesteps"].flatten(0, 1),
                latent_dict["cond_timesteps"].flatten(0, 1),
            ]
        )[None]
        act_ts = torch.cat(
            [
                action_dict["timesteps"].flatten(0, 1),
                action_dict["cond_timesteps"].flatten(0, 1),
            ]
        )[None]
        latent_temb, latent_timestep_proj = self._time_embed(
            vid_ts, L_H, L_W, dtype=hidden_states.dtype, action_mode=False
        )
        if has_mask:
            msk_ts = torch.cat(
                [
                    mask_dict["timesteps"].flatten(0, 1),
                    mask_dict["cond_timesteps"].flatten(0, 1),
                ]
            )[None]
            M_H = mask_dict["noisy_latents"].shape[-2]
            M_W = mask_dict["noisy_latents"].shape[-1]
            mask_temb, mask_timestep_proj = self._time_embed(
                msk_ts, M_H, M_W, dtype=hidden_states.dtype, action_mode=False
            )
            latent_temb = torch.cat([latent_temb, mask_temb], dim=1)
            latent_timestep_proj = torch.cat(
                [latent_timestep_proj, mask_timestep_proj], dim=1
            )
        action_temb, action_timestep_proj = self._time_embed(
            act_ts, A_H, A_W, dtype=hidden_states.dtype, action_mode=True
        )

        temb = torch.cat([latent_temb, action_temb], dim=1)
        timestep_proj = torch.cat([latent_timestep_proj, action_timestep_proj], dim=1)

        # ── Padding ──────────────────────────────────────────────────────────
        total_length = hidden_states.shape[1]
        padded_length = (128 - total_length % 128) % 128
        hidden_states = F.pad(hidden_states, (0, 0, 0, padded_length))
        rotary_emb = F.pad(rotary_emb, (0, 0, 0, 0, 0, padded_length))
        temb = F.pad(temb, (0, 0, 0, padded_length))
        timestep_proj = F.pad(timestep_proj, (0, 0, 0, 0, 0, padded_length))

        split_list = [vid_noisy_hs.shape[1], vid_cond_hs.shape[1]]
        if has_mask:
            split_list.extend([msk_noisy_hs.shape[1], msk_cond_hs.shape[1]])
        split_list.extend([act_noisy_hs.shape[1], act_cond_hs.shape[1], padded_length])

        # ── FlexAttn mask ────────────────────────────────────────────────────
        FlexAttnFunc.init_mask(
            latent_dict["noisy_latents"].shape,
            action_dict["noisy_latents"].shape,
            padded_length,
            input_dict["chunk_size"],
            window_size=input_dict["window_size"],
            patch_size=self.patch_size,
            device=hidden_states.device,
            mask_shape=mask_dict["noisy_latents"].shape if has_mask else None,
        )

        # ── Backbone (frozen) ────────────────────────────────────────────────
        for block in self.blocks:
            hidden_states = block(
                hidden_states, text_hs, timestep_proj, rotary_emb, update_cache=False
            )

        # ── Final norm ───────────────────────────────────────────────────────
        temb_sst = self.scale_shift_table[None] + temb[:, :, None, ...]
        shift, scale = rearrange(temb_sst, "b l n c -> b n l c").chunk(2, dim=1)
        shift = shift.to(hidden_states.device).squeeze(1)
        scale = scale.to(hidden_states.device).squeeze(1)
        hidden_states = (
            self.norm_out(hidden_states.float()) * (1.0 + scale) + shift
        ).type_as(hidden_states)

        # ── Split and project ────────────────────────────────────────────────
        parts = torch.split(hidden_states, split_list, dim=1)
        vid_feat, _ = parts[:2]
        if has_mask:
            _, _, act_feat, _, _ = parts[2:]
            msk_feat = parts[2]
        else:
            act_feat, _, _ = parts[2:]

        latent_hidden_states = self.proj_out(vid_feat)
        latent_hidden_states = rearrange(
            latent_hidden_states,
            "1 (b l) (n c) -> b (l n) c",
            n=math.prod(self.patch_size),
            b=batch_size,
        )

        action_hidden_states = self.action_proj_out(act_feat)
        action_hidden_states = rearrange(
            action_hidden_states, "1 (b l) c -> b l c", b=batch_size
        )

        if not has_mask:
            return latent_hidden_states, action_hidden_states

        mask_hidden_states = self.mask_proj_out(msk_feat)
        mask_hidden_states = rearrange(
            mask_hidden_states,
            "1 (b l) (n c) -> b (l n) c",
            n=math.prod(self.patch_size),
            b=batch_size,
        )
        return latent_hidden_states, action_hidden_states, mask_hidden_states

    def forward(
        self,
        input_dict,
        update_cache=0,
        cache_name="pos",
        action_mode=False,
        train_mode=False,
        return_action_features=False,
    ):
        r"""
        Forward pass through the diffusion model

        Args:
            x (List[Tensor]):
                List of input video tensors, each with shape [C_in, F, H, W]
            t (Tensor):
                Diffusion timesteps tensor of shape [B]
            context (List[Tensor]):
                List of text embeddings each with shape [L, C]
            seq_len (`int`):
                Maximum sequence length for positional encoding
            y (List[Tensor], *optional*):
                Conditional video inputs for image-to-video mode, same shape as x

        Returns:
            List[Tensor]:
                List of denoised video tensors with original input shapes [C_out, F, H / 8, W / 8]
        """
        if train_mode:
            return self.forward_train(input_dict)
        if action_mode:  # action input emb
            latent_hidden_states = rearrange(
                input_dict["noisy_latents"], "b c f h w -> b (f h w) c"
            )
            latent_hidden_states = self.action_embedder(latent_hidden_states)  # B L1 C
            text_hidden_states = self.condition_embedder.text_embedder(
                input_dict["text_emb"]
            )
            latent_grid_id = input_dict["grid_id"]
            rotary_emb = self.rope(latent_grid_id)[:, :, None]
            latent_time_steps = torch.repeat_interleave(
                input_dict["timesteps"],
                input_dict["noisy_latents"].shape[-2]
                * input_dict["noisy_latents"].shape[-1],
                dim=1,
            )
            temb, timestep_proj = self.condition_embedder_action(
                latent_time_steps, dtype=latent_hidden_states.dtype
            )
            timestep_proj = timestep_proj.unflatten(2, (6, -1))
            for block in self.blocks:
                latent_hidden_states = block(
                    latent_hidden_states,
                    text_hidden_states,
                    timestep_proj,
                    rotary_emb,
                    update_cache=update_cache,
                    cache_name=cache_name,
                )
            temb_sst = self.scale_shift_table[None] + temb[:, :, None, ...]
            shift, scale = rearrange(temb_sst, "b l n c -> b n l c").chunk(2, dim=1)
            shift = shift.to(latent_hidden_states.device).squeeze(1)
            scale = scale.to(latent_hidden_states.device).squeeze(1)
            latent_hidden_states = (
                self.norm_out(latent_hidden_states.float()) * (1.0 + scale) + shift
            ).type_as(latent_hidden_states)
            if return_action_features:
                return latent_hidden_states
            latent_hidden_states = self.action_proj_out(latent_hidden_states)
            return latent_hidden_states

        # ── Video (+ optional mask) inference ────────────────────────────────
        # B2: when noisy_mask_latents is provided, process [vid | mask] together through
        # the backbone so mask tokens attend to video tokens in the same denoising step.
        noisy_mask = input_dict.get("noisy_mask_latents")

        vid_hs = self._input_embed(input_dict["noisy_latents"], "latent")  # [B, L, D]
        text_hidden_states = self.condition_embedder.text_embedder(
            input_dict["text_emb"]
        )

        if noisy_mask is not None:
            msk_hs = self._input_embed(
                noisy_mask.to(vid_hs.dtype), "latent"
            )  # [B, L, D]
            latent_hidden_states = torch.cat([vid_hs, msk_hs], dim=1)  # [B, 2L, D]
        else:
            latent_hidden_states = vid_hs

        latent_grid_id = input_dict["grid_id"]
        # grid_id is [B, 4, L] (f,h,w,t per token from get_mesh_id); sequence length is dim=2.
        # For [vid | mask] we duplicate positions along dim=2 (not dim=1 — that would break RoPE).
        if noisy_mask is not None:
            combined_grid_id = torch.cat([latent_grid_id, latent_grid_id], dim=2)
        else:
            combined_grid_id = latent_grid_id
        rotary_emb = self.rope(combined_grid_id)[:, :, None]

        L_H = input_dict["noisy_latents"].shape[-2]
        L_W = input_dict["noisy_latents"].shape[-1]
        spatial_per_frame = (L_H // self.patch_size[1]) * (L_W // self.patch_size[2])

        # Timestep: repeat for video; mask uses the same timestep.
        ts_vid = torch.repeat_interleave(
            input_dict["timesteps"], spatial_per_frame, dim=1
        )
        if noisy_mask is not None:
            latent_time_steps = torch.cat([ts_vid, ts_vid], dim=1)
        else:
            latent_time_steps = ts_vid
        temb, timestep_proj = self.condition_embedder(
            latent_time_steps, dtype=latent_hidden_states.dtype
        )
        timestep_proj = timestep_proj.unflatten(2, (6, -1))  # B, L, 6, C

        for block in self.blocks:
            latent_hidden_states = block(
                latent_hidden_states,
                text_hidden_states,
                timestep_proj,
                rotary_emb,
                update_cache=update_cache,
                cache_name=cache_name,
            )

        temb_sst = self.scale_shift_table[None] + temb[:, :, None, ...]
        shift, scale = rearrange(temb_sst, "b l n c -> b n l c").chunk(2, dim=1)
        shift = shift.to(latent_hidden_states.device).squeeze(1)
        scale = scale.to(latent_hidden_states.device).squeeze(1)
        latent_hidden_states = (
            self.norm_out(latent_hidden_states.float()) * (1.0 + scale) + shift
        ).type_as(latent_hidden_states)

        vid_L = vid_hs.shape[1]
        vid_feat = latent_hidden_states[:, :vid_L]

        vid_out = self.proj_out(vid_feat)
        vid_out = rearrange(
            vid_out, "b l (n c) -> b (l n) c", n=math.prod(self.patch_size)
        )

        if noisy_mask is not None:
            msk_feat = latent_hidden_states[:, vid_L:]
            msk_out = self.mask_proj_out(msk_feat)
            msk_out = rearrange(
                msk_out, "b l (n c) -> b (l n) c", n=math.prod(self.patch_size)
            )
            return vid_out, msk_out
        else:
            # Return just the video tensor for backward-compatibility with the
            # original server code which expects a plain tensor (not a tuple).
            return vid_out


if __name__ == "__main__":
    model = WanTransformer3DModel(
        patch_size=[1, 2, 2],
        num_attention_heads=24,
        attention_head_dim=128,
        in_channels=48,
        out_channels=48,
        action_dim=30,
        text_dim=4096,
        freq_dim=256,
        ffn_dim=14336,
        num_layers=30,
        cross_attn_norm=True,
        eps=1e-6,
        rope_max_seq_len=1024,
        pos_embed_seq_len=None,
        attn_mode="torch",
    )
    print(model)
