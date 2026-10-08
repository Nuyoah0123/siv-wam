from easydict import EasyDict
from .siv_robotwin_cfg import siv_robotwin_cfg

siv_robotwin_i2va_cfg = EasyDict(__name__='Config: SIV-WAM robotwin i2va')
siv_robotwin_i2va_cfg.update(siv_robotwin_cfg)

siv_robotwin_i2va_cfg.input_img_path = 'example/robotwin'
siv_robotwin_i2va_cfg.num_chunks_to_infer = 10
siv_robotwin_i2va_cfg.prompt = 'Grab the medium-sized white mug, rotate it, place it on the table, and hook it onto the smooth dark gray rack.'
siv_robotwin_i2va_cfg.infer_mode = 'i2va'