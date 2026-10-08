# Server config for the B2 finetuned model (mask head checkpoint)
import os
from pathlib import Path
from easydict import EasyDict
from .siv_robotwin_cfg import siv_robotwin_cfg

siv_robotwin_b2_cfg = EasyDict(__name__='Config: SIV-WAM robotwin B2 finetuned')
siv_robotwin_b2_cfg.update(siv_robotwin_cfg)

_REPO_ROOT = Path(__file__).resolve().parents[2]

# Override transformer path to load our B2 checkpoint
# VAE / tokenizer / text_encoder still come from the original pretrained model
siv_robotwin_b2_cfg.transformer_path = os.environ.get(
    "SIV_WAM_TRANSFORMER_PATH",
    str(
        _REPO_ROOT
        / "outputs"
        / "siv"
        / "checkpoints"
        / "checkpoint_step_100"
        / "transformer"
    ),
)

siv_robotwin_b2_cfg.infer_mode = 'server'
