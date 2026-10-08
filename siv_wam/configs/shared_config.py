import torch
from easydict import EasyDict

siv_shared_cfg = EasyDict()

siv_shared_cfg.host = '0.0.0.0'
siv_shared_cfg.port = 29536

siv_shared_cfg.param_dtype = torch.bfloat16
siv_shared_cfg.save_root = './outputs/train'

siv_shared_cfg.patch_size = (1, 2, 2)

siv_shared_cfg.enable_offload = True
