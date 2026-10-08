from easydict import EasyDict
from .siv_franka_cfg import siv_franka_cfg

siv_franka_i2va_cfg = EasyDict(__name__='Config: SIV-WAM franka i2va')
siv_franka_i2va_cfg.update(siv_franka_cfg)

siv_franka_i2va_cfg.input_img_path = 'example/franka'
siv_franka_i2va_cfg.num_chunks_to_infer = 10
siv_franka_i2va_cfg.prompt = 'pick bunk'
siv_franka_i2va_cfg.infer_mode = 'i2va'