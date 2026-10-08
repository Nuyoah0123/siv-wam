import torch
from diffusers import AutoencoderKLWan
from transformers import (
    T5TokenizerFast,
    UMT5EncoderModel,
)

from .model import WanTransformer3DModel


def load_vae(
    vae_path,
    torch_dtype,
    torch_device,
):
    vae = AutoencoderKLWan.from_pretrained(
        vae_path,
        torch_dtype=torch_dtype,
    )
    return vae.to(torch_device)


def load_text_encoder(
    text_encoder_path,
    torch_dtype,
    torch_device,
):
    text_encoder = UMT5EncoderModel.from_pretrained(
        text_encoder_path,
        torch_dtype=torch_dtype,
    )
    return text_encoder.to(torch_device)


def load_tokenizer(tokenizer_path, ):
    tokenizer = T5TokenizerFast.from_pretrained(tokenizer_path, )
    return tokenizer


def load_transformer(
    transformer_path,
    torch_dtype,
    torch_device,
    **kwargs,
):
    """Load WanTransformer3DModel from a checkpoint directory.

    Extra keyword arguments (e.g. attn_mode="torch") are forwarded to
    from_pretrained and override values in the saved config.json.
    mask_proj_out is always zero-initialised when not present in the checkpoint
    so the model starts with neutral mask predictions.
    """
    model = WanTransformer3DModel.from_pretrained(
        transformer_path,
        torch_dtype=torch_dtype,
        **kwargs,
    )
    # from_pretrained uses low_cpu_mem_usage=True by default, which builds the
    # model on the meta device and then loads weights.  Any parameter NOT found
    # in the checkpoint (e.g. newly added mask_proj_out) stays on the meta
    # device and would cause "Cannot copy out of meta tensor" when .to() is
    # called.  Fix: replace meta tensors with real zero tensors before moving.
    for module in model.modules():
        for pname, param in list(module.named_parameters(recurse=False)):
            if param.is_meta:
                real = torch.nn.Parameter(
                    torch.zeros(param.shape, dtype=torch_dtype))
                setattr(module, pname, real)
        for bname, buf in list(module.named_buffers(recurse=False)):
            if buf.is_meta:
                real_buf = torch.zeros(buf.shape, dtype=torch_dtype)
                module.register_buffer(bname, real_buf)
    return model.to(torch_device)


def patchify(x, patch_size):
    if patch_size is None or patch_size == 1:
        return x
    batch_size, channels, frames, height, width = x.shape
    x = x.view(batch_size, channels, frames, height // patch_size, patch_size,
               width // patch_size, patch_size)
    x = x.permute(0, 1, 6, 4, 2, 3, 5).contiguous()
    x = x.view(batch_size, channels * patch_size * patch_size, frames,
               height // patch_size, width // patch_size)
    return x


class WanVAEStreamingWrapper:

    def __init__(self, vae_model):
        self.vae = vae_model
        self.encoder = vae_model.encoder
        self.quant_conv = vae_model.quant_conv

        if hasattr(self.vae, "_cached_conv_counts"):
            self.enc_conv_num = self.vae._cached_conv_counts["encoder"]
        else:
            count = 0
            for m in self.encoder.modules():
                if m.__class__.__name__ == "WanCausalConv3d":
                    count += 1
            self.enc_conv_num = count

        self.clear_cache()

    def clear_cache(self):
        self.feat_cache = [None] * self.enc_conv_num

    def encode_chunk(self, x_chunk):
        if hasattr(self.vae.config,
                   "patch_size") and self.vae.config.patch_size is not None:
            x_chunk = patchify(x_chunk, self.vae.config.patch_size)
        feat_idx = [0]
        out = self.encoder(x_chunk,
                           feat_cache=self.feat_cache,
                           feat_idx=feat_idx)
        enc = self.quant_conv(out)
        return enc
