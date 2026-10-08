# Data generation visualization

This folder contains a standalone diagnostic for the contact-to-SIV pipeline. It renders projected contacts, the raw SIV heatmap, an RGB overlay, and numeric diagnostics without modifying the source data.

Run it from the repository root:

```bash
python data_process/visual_debug/debug_visualize_pipeline.py \
  --task_dir "$SIV_WAM_DATASET_PATH/<task>" \
  --episode 0 \
  --cam cam_high \
  --out_dir ./outputs/visual_debug/episode_000000 \
  --frame_ids 40,64,81,101,121
```

Each `frame_*_pipeline.png` contains RGB with projected contact points, a standalone SIV heatmap, an RGB/SIV overlay, and numeric diagnostics. The script also writes `pipeline_debug.mp4` and `summary.json`.

To pause after each selected frame:

```bash
python data_process/visual_debug/debug_visualize_pipeline.py \
  --task_dir "$SIV_WAM_DATASET_PATH/<task>" \
  --episode 0 \
  --cam cam_high \
  --out_dir ./outputs/visual_debug/live_episode_000000 \
  --frame_ids 64 \
  --live \
  --pause_each
```

The default source is the offline EEF contact proxy. To visualize recorded contacts instead, pass `--contacts path/to/ep000000_contacts.jsonl`.
