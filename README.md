# AttenX

Enhanced Attention-Based GAN for Fine-Grained Text-to-Image Synthesis.

**Paper**: AttenX: Enhanced Attention-Based GAN for Fine-Grained Text-to-Image Synthesis

## Three Variants

| Variant | Attention | Spectral Norm | Config |
|---------|-----------|---------------|--------|
| Baseline | None | No | `configs/baseline.yaml` |
| AttenX-SA | Single-head SA at 64x64 | Yes (all D) | `configs/attenx_sa.yaml` |
| AttenX-MHSA | Multi-head SA at 64x64 | Yes (all D) | `configs/attenx_mhsa.yaml` |

## Setup

```bash
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
pip install -e .
```

## Training

```bash
# Baseline
python3 -m attenx.scripts.train --config configs/baseline.yaml

# AttenX-SA
python3 -m attenx.scripts.train --config configs/attenx_sa.yaml

# AttenX-MHSA
python3 -m attenx.scripts.train --config configs/attenx_mhsa.yaml

# With DAMSM pretrained encoders
python3 -m attenx.scripts.train \
    --config configs/attenx_mhsa.yaml \
    --gamma-damsm 5.0 \
    --damsm-text-path ./checkpoints/damsm/text_encoder.pth \
    --damsm-image-path ./checkpoints/damsm/image_encoder.pth
```

## DAMSM Pretraining

```bash
python3 -m attenx.scripts.pretrain_damsm \
    --data-dir ./data \
    --epochs 5 \
    --batch-size 16 \
    --checkpoint-dir ./checkpoints/damsm
```

## Evaluation

```bash
python3 -m attenx.scripts.evaluate \
    --config configs/attenx_mhsa.yaml \
    --checkpoint ./checkpoints/attenx_mhsa/checkpoint_best.pth \
    --num-imgs 500
```

## Generation

```bash
python3 -m attenx.scripts.generate \
    --config configs/attenx_mhsa.yaml \
    --checkpoint ./checkpoints/attenx_mhsa/checkpoint_best.pth \
    --captions "a bright yellow bird with a black head and wings"
```

## Project Structure

```
attenx/                    # Core package
├── config.py              # AttenXConfig dataclass
├── data/
│   └── dataset.py         # TextImageDataset (CUB-200-2011)
├── losses/
│   ├── gan.py             # Adversarial loss
│   └── damsm.py           # DAMSM losses (words + sentence + KL)
├── models/
│   ├── attention.py       # SelfAttention, MultiHeadSelfAttention, CrossAttention
│   ├── conditioning.py    # ConditioningAugmentation
│   ├── generator.py       # G_NET with GenStage
│   ├── discriminator.py   # D_NET64/128/256 with spectral norm
│   └── encoders.py        # RNN_ENCODER, CNN_ENCODER
├── training/
│   ├── trainer.py         # Main training loop
│   └── validator.py       # Validation loop
├── utils/
│   ├── checkpointing.py   # Save/load checkpoints
│   └── logging.py         # LossLogger
└── scripts/
    ├── train.py           # Training entry point
    ├── pretrain_damsm.py  # DAMSM encoder pretraining
    ├── evaluate.py        # IS evaluation
    └── generate.py        # Image generation from text

configs/
├── baseline.yaml
├── attenx_sa.yaml
└── attenx_mhsa.yaml

data/                      # Dataset (CUB-200-2011)
```
