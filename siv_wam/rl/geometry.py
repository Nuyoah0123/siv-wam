"""OpenCV camera convention: +Z forward, +X right, +Y down; world units are meters."""
import torch
import torch.nn.functional as F


def project_world(points, intrinsics, world_to_camera, image_size):
    points = torch.as_tensor(points, dtype=torch.float32)
    K = torch.as_tensor(intrinsics, dtype=torch.float32, device=points.device)
    T = torch.as_tensor(world_to_camera, dtype=torch.float32, device=points.device)
    h, w = image_size
    if points.ndim != 2 or points.shape[-1] != 3 or K.shape != (3, 3) or T.shape != (4, 4):
        raise ValueError('Expected points [N,3], K [3,3], T_cw [4,4]')
    if min(h, w) < 2 or not torch.isfinite(K).all() or not torch.isfinite(T).all():
        raise ValueError('Invalid camera calibration or image size')
    if not torch.allclose(T[3], T.new_tensor([0, 0, 0, 1]), atol=1e-5):
        raise ValueError('T_cw must be a homogeneous world-to-camera transform')
    camera = points @ T[:3, :3].T + T[:3, 3]
    projected = camera @ K.T
    pixels = projected[:, :2] / projected[:, 2:].clamp_min(1e-8)
    valid = (camera[:, 2] > 1e-6) & torch.isfinite(pixels).all(-1) & torch.isfinite(points).all(-1)
    valid &= (pixels[:, 0] >= 0) & (pixels[:, 0] <= w - 1)
    valid &= (pixels[:, 1] >= 0) & (pixels[:, 1] <= h - 1)
    return pixels, valid


def sample_value_map(value_map, pixels, valid):
    """Bilinear response, with behind-camera/out-of-frame targets receiving zero."""
    value_map = torch.as_tensor(value_map, dtype=torch.float32)
    if value_map.ndim != 2 or not torch.isfinite(value_map).all():
        raise ValueError('SIV map must be finite [H,W]')
    if (value_map < 0).any() or (value_map > 1).any():
        raise ValueError('SIV responses must be in [0,1]')
    pixels = torch.as_tensor(pixels, dtype=torch.float32, device=value_map.device)
    valid = torch.as_tensor(valid, dtype=torch.bool, device=value_map.device)
    h, w = value_map.shape
    if min(h, w) < 2 or pixels.shape != (valid.numel(), 2):
        raise ValueError('Invalid sampling shape')
    grid = torch.nan_to_num(pixels.clone(), nan=-1e6, posinf=1e6, neginf=-1e6)
    grid[:, 0] = 2 * grid[:, 0] / (w - 1) - 1
    grid[:, 1] = 2 * grid[:, 1] / (h - 1) - 1
    sampled = F.grid_sample(value_map[None, None], grid[None, None],
                            mode='bilinear', padding_mode='zeros', align_corners=True).flatten()
    return torch.where(valid, sampled, torch.zeros_like(sampled))


def projected_reward(value_map, targets, K, T_cw):
    pixels, valid = project_world(targets, K, T_cw, value_map.shape)
    return sample_value_map(value_map, pixels, valid)
