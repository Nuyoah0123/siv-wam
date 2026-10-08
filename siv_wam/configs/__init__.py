from .siv_demo_cfg import siv_demo_cfg
from .siv_demo_i2va import siv_demo_i2va_cfg
from .siv_demo_train_cfg import siv_demo_train_cfg
from .siv_franka_cfg import siv_franka_cfg
from .siv_franka_i2va import siv_franka_i2va_cfg
from .siv_robotwin_b2_cfg import siv_robotwin_b2_cfg
from .siv_robotwin_cfg import siv_robotwin_cfg
from .siv_robotwin_i2va import siv_robotwin_i2va_cfg
from .siv_robotwin_mask_joint_cfg import (
    siv_robotwin_mask_joint_cfg,
    siv_robotwin_mask_joint_overfit_cfg,
)
from .siv_robotwin_mask_train_cfg import siv_robotwin_mask_train_cfg
from .siv_robotwin_train_cfg import siv_robotwin_train_cfg

SIV_CONFIGS = {
    'robotwin': siv_robotwin_cfg,
    'franka': siv_franka_cfg,
    'robotwin_i2av': siv_robotwin_i2va_cfg,
    'franka_i2av': siv_franka_i2va_cfg,
    'robotwin_train': siv_robotwin_train_cfg,
    'demo': siv_demo_cfg,
    'demo_train': siv_demo_train_cfg,
    'demo_i2av': siv_demo_i2va_cfg,
    'robotwin_mask_train': siv_robotwin_mask_train_cfg,
    'robotwin_mask_joint': siv_robotwin_mask_joint_cfg,
    'robotwin_mask_joint_overfit': siv_robotwin_mask_joint_overfit_cfg,
    'robotwin_b2': siv_robotwin_b2_cfg,
}