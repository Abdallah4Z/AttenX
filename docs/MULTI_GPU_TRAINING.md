# Multi-GPU Training (2x A6000)

This guide explains the exact commands to run AttenX on two GPUs.

## Current Multi-GPU Behavior

- `train.py` uses `torch.nn.DataParallel` when more than one GPU ID is provided.
- GPU IDs come from `--gpu-ids` or `gpu_ids` in `configs/train.yaml`.
- The `batch_size` value is global (split across selected GPUs by DataParallel).

## Recommended 2-GPU Commands (No Env Vars)

Use physical GPU IDs directly:

This is the simplest and recommended command when you do not want environment-variable setup.

```bash
cd /home/skyvision/AttenX
python3 train.py --config configs/train.yaml --gpu-ids 6,7 --use-amp
```

## Suggested Config for 2x48GB

In `configs/train.yaml`:

```yaml
gpu_ids: "6,7"
use_amp: true
batch_size: 8
num_workers: 8
```

Tune `batch_size` upward if memory headroom is available.

## Verify Both GPUs Are Used

During startup, logs should include a line like:

```text
Using DataParallel on 2 GPUs
```

Also monitor utilization:

```bash
watch -n 1 nvidia-smi
```

## Resume Training on 2 GPUs

```bash
cd /home/skyvision/AttenX
python3 train.py --config configs/train.yaml --gpu-ids 6,7 --resume --use-amp
```

## Troubleshooting

- If only one GPU is active:
  - confirm `--gpu-ids` has two IDs
  - confirm CUDA is available (`nvidia-smi` works)
  - check startup log for DataParallel message
- If out-of-memory occurs:
  - reduce `batch_size`
  - keep `--use-amp` enabled
  - reduce `image_size` only if necessary
