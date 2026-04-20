# AttenX

Enhanced attention-based GAN (AttnGAN-style) for text-to-image synthesis, with self-attention integration and discriminator spectral normalization experiments.

## Repository layout

- `code/` core model components (`generator`, `discriminator`, `encoder`, `losses`, `datasets`)
- `scripts/` utilities for generation, evaluation, issue/project management, and environment checks
- `docs/architecture/` architecture/math notes
- `docs/specifications/` implementation audits/specs
- `docs/planning/` project planning and phase breakdown
- `train.py` multi-epoch training pipeline with checkpointing, configurable D-steps, and DAMSM loss


## Current implementation status

- `G_NET` implemented with staged upsampling to `256x256` and a `SelfAttention` block
- `D_NET256` implemented with spectral normalization over convolution layers
- DAMSM loss fully integrated: supports pre-trained encoder loading, gamma_damsm weighting, and separate word/sentence loss logging
- `TextImageDataset` class implemented for CUB-200-2011 format: loads images with bounding box cropping, captions from pickle files, and applies configurable transforms
- `train.py` full multi-epoch trainer with:
  - DataLoader support with configurable batch size and workers
  - Checkpointing (periodic + latest for resume)
  - Configurable discriminator steps per generator (`--D-steps`)
  - Learning rate scheduling with `ReduceLROnPlateau` (`--lr-patience`, `--lr-factor`)
  - Validation loop with metric computation (`--validate-interval`)
  - Early stopping based on validation loss (`--patience-early-stop`, `--min-delta`)
  - Full DAMSM loss integration with gamma weighting

## Setup

Use Python 3.10+.

```bash
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

Or with PEP 621 metadata:

```bash
pip install .
```


## Quick checks

```bash
python3 scripts/verify_env.py
python3 -m compileall -q code scripts train.py
pytest --cov=code --cov=tests --cov-report=term-missing --cov-report=xml --cov-report=html --cov-fail-under=80
```


## Run examples

```bash
# Full multi-epoch training on CUB dataset
python3 train.py --epochs 100 --batch-size 4 --data-dir ./data --checkpoint-dir ./checkpoints --checkpoint-interval 10

# With learning rate scheduling (ReduceLROnPlateau) and early stopping
python3 train.py --epochs 200 --lr-patience 5 --lr-factor 0.5 --validate-interval 1 --patience-early-stop 10

# With custom discriminator steps per generator step
python3 train.py --epochs 100 --D-steps 2 --gamma-damsm 1.0

# Resume from checkpoint
python3 train.py --epochs 200 --resume --checkpoint-dir ./checkpoints

# With pre-trained DAMSM encoders
python3 train.py --epochs 100 --damsm-text-path path/to/text_encoder.pth --damsm-image-path path/to/image_encoder.pth

# Disable validation (train only)
python3 train.py --epochs 50 --validate-interval 0

# Run generation/evaluation scripts
python3 scripts/generate.py
python3 scripts/quantitative_evaluation.py
python3 scripts/qualitative_analysis.py
```

## Dataset format

The training pipeline expects CUB-200-2011 style data:

```
data/
├── CUB_200_2011/
│   ├── images.txt              # image filename list
│   ├── bounding_boxes.txt      # x y width height per image
│   └── images/
│       └── 001.Class/
│           └── image_0001.jpg
├── train/
│   ├── filenames.pickle        # list of image keys (e.g., "001.Class/sample_0001")
│   ├── captions.pickle         # list of caption arrays (10 captions per image, each as word indices)
│   └── class_info.pickle       # class IDs
└── test/ (same structure)
```

Captions should be pre-tokenized to word indices based on a vocabulary of size `vocab_size` (default 10000). The `TextImageDataset` class handles loading and batching.

## Training arguments reference

### Dataset & DataLoader
| Argument | Default | Description |
|----------|---------|-------------|
| `--data-dir` | `./data` | Dataset root directory |
| `--image-size` | `256` | Image size for training (square) |
| `--num-workers` | `4` | Number of DataLoader worker processes |

### Model architecture
| Argument | Default | Description |
|----------|---------|-------------|
| `--ngf` | `64` | Generator base channel count (scales with depth) |
| `--ndf` | `64` | Discriminator base channel count |
| `--nef` | `512` | Encoder feature dimension (text & image) |
| `--nhidden` | `256` | RNN hidden size for text encoder |
| `--nembed` | `256` | Word embedding dimension |
| `--vocab-size` | `10000` | Vocabulary size for token indices |

### Training hyperparameters
| Argument | Default | Description |
|----------|---------|-------------|
| `--epochs` | `100` | Total training epochs |
| `--batch-size` | `4` | Batch size per GPU |
| `--lr-g` | `1e-4` | Generator learning rate |
| `--lr-d` | `4e-4` | Discriminator learning rate |
| `--D-steps` | `1` | Number of D updates per G update |
| `--gamma-damsm` | `1.0` | Weight for DAMSM loss component |

### Learning rate scheduler
| Argument | Default | Description |
|----------|---------|-------------|
| `--lr-patience` | `5` | Epochs without improvement before reducing LR |
| `--lr-factor` | `0.5` | Factor by which to reduce LR on plateau |

### Validation & early stopping
| Argument | Default | Description |
|----------|---------|-------------|
| `--validate-interval` | `1` | Run validation every N epochs (0 = disabled) |
| `--patience-early-stop` | `10` | Epochs without improvement before early stopping |
| `--min-delta` | `1e-4` | Minimum change in validation loss to count as improvement |

### Checkpointing & logging
| Argument | Default | Description |
|----------|---------|-------------|
| `--checkpoint-dir` | `./checkpoints` | Directory to save model checkpoints |
| `--checkpoint-interval` | `10` | Save a checkpoint every N epochs |
| `--log-dir` | `./logs/training` | Directory for loss logs |
| `--resume` | `False` | Resume from latest checkpoint if set |

### Pretrained encoders
| Argument | Default | Description |
|----------|---------|-------------|
| `--damsm-text-path` | `None` | Path to pretrained DAMSM text encoder checkpoint |
| `--damsm-image-path` | `None` | Path to pretrained DAMSM image encoder checkpoint |

## Documentation index

- `docs/planning/PROJECT_PLAN.md`
- `docs/planning/TEAM_ASSIGNMENTS.md`
- `docs/planning/ISSUE_RESTRUCTURING_PLAN.md`
- `docs/planning/DEPENDENCY_GRAPH.md`
- `docs/architecture/self_attention_math.md`
- `docs/specifications/discriminator_audit.md`

## DAMSM Loss Integration

- DAMSM encoders (text and image) can be loaded from pre-trained checkpoints and are frozen during GAN training.
- The generator loss is:  
	$L_G = L_{GAN} + \gamma_{damsm} (L_{Words} + L_{Sent})$
- All loss components are logged separately: `D_Loss`, `G_Loss`, `G_GAN`, `L_Words`, `L_Sent`, `L_DAMSM`, `Gamma_DAMSM`.
- Word-level similarity is masked by caption length for correct averaging.

## Training Features

### Learning Rate Scheduling
Uses `ReduceLROnPlateau` to decay learning rates when validation loss plateaus:
- `--lr-patience`: epochs to wait before reducing LR (default: 5)
- `--lr-factor`: multiplier for LR reduction (default: 0.5)
- Schedulers step after each validation run

### Validation Loop
Periodic evaluation on a held-out validation set:
- `--validate-interval`: run validation every N epochs (default: 1)
- Computes D loss, G loss, and DAMSM loss on validation samples
- Validation losses are used for LR scheduling and early stopping

### Early Stopping
Stops training when validation loss fails to improve:
- `--patience-early-stop`: max epochs without improvement before stopping (default: 10)
- `--min-delta`: minimum change to qualify as improvement (default: 1e-4)
- Best checkpoint (lowest validation loss) is saved separately


## Testing & Continuous Integration

- All core modules (SelfAttention, G_NET, D_NET, TextDataset, losses, encoders) are covered by unit tests.
- Edge cases (missing images, empty captions) are explicitly tested.
- Spectral normalization is verified on all discriminator Conv2d layers.
- CI (GitHub Actions) runs pytest with coverage on every push/PR and enforces a minimum 80% coverage gate.
- Current coverage: 97%+ (see coverage reports in CI artifacts or run locally as above).

## Notes

- GitHub issue helper scripts require `PyGitHub` and a personal access token.
- Generated artifacts/logs are written under `results/`, `output/`, and `logs/`.
- Checkpoints are saved as `checkpoint_epoch_{:04d}.pth` plus `checkpoint_latest.pth` for resuming.
- For GPU training, ensure CUDA toolkit is installed; the code auto-detects CUDA availability.
