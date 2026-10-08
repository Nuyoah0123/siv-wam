import os

from easydict import EasyDict

from .siv_robotwin_cfg import siv_robotwin_cfg

siv_robotwin_train_cfg = EasyDict(__name__='Config: SIV-WAM robotwin train')
siv_robotwin_train_cfg.update(siv_robotwin_cfg)


siv_robotwin_train_cfg.dataset_path = os.environ.get("SIV_WAM_DATASET_PATH", "./data/robotwin_lerobot")
siv_robotwin_train_cfg.empty_emb_path = os.path.join(siv_robotwin_train_cfg.dataset_path, 'empty_emb.pt')
siv_robotwin_train_cfg.enable_wandb = False
siv_robotwin_train_cfg.load_worker = 16
siv_robotwin_train_cfg.save_interval = 1000
siv_robotwin_train_cfg.gc_interval = 50
siv_robotwin_train_cfg.cfg_prob = 0.1

# Training parameters
siv_robotwin_train_cfg.learning_rate = 1e-5
siv_robotwin_train_cfg.beta1 = 0.9
siv_robotwin_train_cfg.beta2 = 0.95
siv_robotwin_train_cfg.weight_decay = 0.1
siv_robotwin_train_cfg.warmup_steps = 10
siv_robotwin_train_cfg.batch_size = 1 
siv_robotwin_train_cfg.gradient_accumulation_steps = 1
siv_robotwin_train_cfg.num_steps = 50000 
