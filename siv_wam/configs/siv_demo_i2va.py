from easydict import EasyDict
from .siv_demo_cfg import siv_demo_cfg

siv_demo_i2va_cfg = EasyDict(__name__='Config: SIV-WAM demo i2va')
siv_demo_i2va_cfg.update(siv_demo_cfg)

siv_demo_i2va_cfg.input_img_path = 'example/demo'
siv_demo_i2va_cfg.num_chunks_to_infer = 10
siv_demo_i2va_cfg.prompt = 'Pick the green cube and place it inside the blue box'
siv_demo_i2va_cfg.infer_mode = 'i2va'