# Data processing

`data_process/` converts robot trajectories and contact information into Spatial Interaction Value (SIV) maps and, optionally, SIV latents consumed by the training code.

```text
LeRobot episode or RoboTwin contacts
        ↓
world-frame contact points
        ↓
world → camera → pixel projection
        ↓
visibility and depth checks
        ↓
Gaussian SIV heatmap per camera
        ↓
optional Wan VAE encoding
        ↓
observation.masks.<camera>
```

## Layout

```text
data_process/
├── c2s/
│   ├── contacts.py       contact sources and data structures
│   ├── geometry.py       camera transforms, projection, and sigma
│   ├── lerobot_io.py     LeRobot episode/video/latent I/O
│   ├── latents.py        SIV pseudo-RGB and VAE encoding
│   ├── seeds.py          episode/seed mapping helpers
│   ├── siv.py            SIV heatmap construction
│   ├── spec.py           camera, action, and path conventions
│   └── viz.py            map overlays and diagnostics
├── scripts/
│   ├── verify_projection.py
│   ├── export_siv_dataset.py
│   ├── batch_export.py
│   ├── build_contacts_sim.py
│   ├── build_siv_from_contacts.py
│   ├── build_siv_lerobot.py
│   ├── prepare_siv_train_export.py
│   ├── recover_replay_seeds.py
│   └── check_replay_alignment.py
├── visual_debug/
│   ├── debug_visualize_pipeline.py
│   └── validate_wrist_geometry.py
└── tests/
```

## Configuration

The code uses environment variables instead of server-specific absolute paths:

```bash
export SIV_WAM_DATASET_PATH=/path/to/robotwin_lerobot
export SIV_WAM_ROBOTWIN_ROOT=/path/to/RoboTwin
export SIV_WAM_VAE_PATH=/path/to/vae
```

The default action layout is the 16D dual-arm LeRobot interface:

```text
left_xyz(3), left_quaternion(4), left_gripper(1),
right_xyz(3), right_quaternion(4), right_gripper(1)
```

`data_process/c2s/spec.py` documents the camera calibration, quaternion convention, and fallback map sizes. Check the dataset's own metadata before using fallback values for a new robot or camera rig.

## Test and debug first

Run the synthetic tests before touching a real dataset:

```bash
cd data_process
python -m unittest discover -s tests -v
python scripts/verify_projection.py --help
python visual_debug/debug_visualize_pipeline.py --help
```

The offline `EEFContactEstimator` is a geometry/debugging proxy. It estimates likely finger-pad contacts from end-effector pose and gripper state; it is not a physical-engine contact label. For formal training, use `build_contacts_sim.py` with a validated RoboTwin installation or provide recorded contact JSONL.

## Typical data flow

```bash
# Inspect an episode and camera projection.
python scripts/verify_projection.py \
  --task_dir "$SIV_WAM_DATASET_PATH/<task>" \
  --episode 0 \
  --cam cam_high

# Generate raw SIV maps using the offline proxy.
python scripts/export_siv_dataset.py \
  --task_dir "$SIV_WAM_DATASET_PATH/<task>" \
  --episode 0 \
  --offline

# Render a visual diagnostic for selected frames.
python visual_debug/debug_visualize_pipeline.py \
  --task_dir "$SIV_WAM_DATASET_PATH/<task>" \
  --episode 0 \
  --cam cam_high \
  --frame_ids 40,64,81,101,121 \
  --out_dir ./outputs/visual_debug/episode_000000
```

Use each command's `--help` output as the source of truth for options because dataset versions may expose different metadata fields.

## Known boundaries

- Raw maps are generated per camera; multi-view T-pose packing happens in the training/data-loading path.
- Wrist-camera extrinsics must be validated with rendered overlays before producing formal labels.
- Depth rejection and camera coordinate conventions are part of label quality, not cosmetic visualization details.
- Large datasets, videos, latents, simulator assets, and VAE weights are intentionally ignored by Git.
