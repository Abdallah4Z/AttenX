# AttenX Getting Started

This guide covers no-env setup, dataset preparation, and a first successful training run.

## 1) Install Dependencies (No Env)

From the AttenX root:

```bash
cd /home/skyvision/AttenX
pip3 install --user -r requirements.txt
```

If your system enforces externally managed Python packages, use:

```bash
pip3 install --break-system-packages -r requirements.txt
```

Optional sanity checks:

```bash
python3 -m compileall -q code scripts train.py
python3 train.py --help | head -n 80
```

## 2) Dataset Setup (CUB)

Run the dataset script:

```bash
cd /home/skyvision/AttenX/data
./download_cub.sh
```

Expected structure (either of these is supported):

```text
data/
├── CUB_200_2011/
│   ├── images/
│   ├── images.txt
│   └── bounding_boxes.txt
├── train/                # optional direct split layout
│   ├── filenames.pickle
│   ├── captions.pickle
│   └── class_info.pickle
├── test/                 # optional direct split layout
│   ├── filenames.pickle
│   ├── captions.pickle
│   └── class_info.pickle
└── birds/                # layout created by download_cub.sh
  ├── captions.pickle
  ├── train/
  │   ├── filenames.pickle
  │   └── class_info.pickle
  └── test/
    ├── filenames.pickle
    └── class_info.pickle
```

Note: the loader automatically falls back to the `data/birds` layout when `data/train` and `data/test` are missing.

## 3) Configure Training

Main config file:

- `configs/train.yaml`

Useful keys:

- `data_dir`: dataset root (default `./data`)
- `batch_size`: global batch size seen by DataParallel
- `gpu_ids`: comma-separated GPU IDs (example: `6,7`)
- `use_amp`: mixed precision (`true` or `false`)
- `checkpoint_dir`, `log_dir`: output paths

## 4) Start Training (No Env Vars)

Default run with config file:

```bash
cd /home/skyvision/AttenX
python3 train.py --config configs/train.yaml
```

Direct 2-GPU run without CUDA environment variables:

```bash
python3 train.py --config configs/train.yaml --gpu-ids 6,7 --use-amp
```

Resume from latest checkpoint:

```bash
python3 train.py --config configs/train.yaml --resume
```

## 5) Common Overrides

Override a few options without editing YAML:

```bash
python3 train.py \
  --config configs/train.yaml \
  --epochs 150 \
  --batch-size 8 \
  --checkpoint-interval 5
```

## 6) Outputs

- Checkpoints: `checkpoints/` (or `checkpoint_dir` from config)
- Training logs: `logs/training/` (or `log_dir` from config)

If training is running correctly, you should see epoch progress, validation metrics, and checkpoint save messages.
