# AttenX — Enhanced Attention-Based GAN for Text-to-Image Synthesis

**AttenX** is a PyTorch framework for fine-grained text-to-image generation, built on the StackGAN++ / AttnGAN architecture. It introduces configurable self-attention mechanisms (single-head SAGAN-style or multi-head Transformer-style) at the 64×64 resolution stage of a cascaded generator, combined with word-level cross-attention (DAMSM) at every upsampling stage.

---

## Project Structure

```
attenx_refactored/
├── config.py                  # Central configuration dataclass (AttenXConfig)
├── configs/                   # YAML configuration presets
│   ├── attenx_mhsa.yaml       #   Multi-head self-attention + spectral norm
│   ├── attenx_sa.yaml         #   Single-head self-attention + spectral norm
│   └── baseline.yaml          #   No self-attention, no spectral norm
├── data/
│   └── dataset.py             # CUB-200-2011 text-image paired dataset
├── losses/
│   ├── damsm.py               # DAMSM losses (word-level, sentence-level, KL)
│   └── gan.py                 # GAN adversarial loss (BCE)
├── models/
│   ├── attention.py           # SelfAttention, MultiHeadSelfAttention, CrossAttention
│   ├── conditioning.py        # Conditioning Augmentation (StackGAN)
│   ├── discriminator.py       # Multi-scale discriminators (64/128/256)
│   ├── encoders.py            # RNN text encoder + CNN image encoder (DAMSM)
│   └── generator.py           # Cascaded generator with attention stages
├── scripts/
│   ├── train.py               # Training entry point
│   ├── generate.py            # Image generation from text
│   ├── evaluate.py            # Inception Score evaluation
│   └── pretrain_damsm.py      # DAMSM encoder pretraining
├── training/
│   ├── trainer.py             # Main training loop (AMP, early stopping, etc.)
│   └── validator.py           # Validation loop
├── utils/
│   ├── checkpointing.py       # Save/load model checkpoints
│   ├── collate.py             # Shared DataLoader collation function
│   └── logging.py             # Loss metric logger (JSON)
├── pyproject.toml             # Package metadata and entry points
└── requirements.txt           # Python dependencies
```

---

## File-by-File Explanation

### 1. `config.py` — Configuration

Defines `AttenXConfig`, a `@dataclass` that holds every hyperparameter: dataset paths, model dimensions (ngf, ndf, nef, etc.), training settings (learning rates, batch size, epochs), optimizer config, validation/early stopping parameters, and DAMSM encoder paths. Supports loading from YAML (`from_yaml`) and saving to YAML (`save`).

### 2. `configs/` — YAML Presets

Three predefined configurations:

| File | `attention_mode` | `use_sn` | Description |
|------|-------------------|----------|-------------|
| `attenx_mhsa.yaml` | `multi` | `true` | Multi-head self-attention + spectral norm |
| `attenx_sa.yaml` | `single` | `true` | Single-head self-attention + spectral norm |
| `baseline.yaml` | `none` | `false` | No self-attention, no spectral norm |

### 3. `data/dataset.py` — CUB-200-2011 Dataset

`TextImageDataset` loads bird images from the CUB-200-2011 dataset:
- Crops images using bounding box annotations (with 0.75× padding)
- Resizes to the target resolution (default 256×256)
- Normalizes pixel values to [-1, 1]
- Randomly selects 1 of 10 captions per image
- Returns `(image, caption_token_ids, caption_length, class_id, filename)`

### 4. `models/`

#### `attention.py`
Three attention modules used inside the generator:

| Class | Type | Description |
|-------|------|-------------|
| `SelfAttention` | Single-head non-local | SAGAN-style spatial attention with learnable gamma (init 0), residual connection |
| `MultiHeadSelfAttention` | Multi-head | Transformer-style with LayerNorm, QKV projection, h parallel heads |
| `CrossAttention` | Word-level | Queries from image features, keys/values from word embeddings |

#### `conditioning.py`
`ConditioningAugmentation` maps a sentence embedding to a latent code `c` by sampling from `N(mu, std)`, where `mu` and `logvar` are learned linear projections. Used to add stochasticity to the generation process (StackGAN).

#### `encoders.py`
Two encoders for the DAMSM:

| Class | Type | Description |
|-------|------|-------------|
| `RNN_ENCODER` | Bidirectional LSTM | Embeds word indices → word-level features + global sentence vector |
| `CNN_ENCODER` | Inception-v3 (frozen) | Extracts local feature maps (768-dim) + global image code (2048-dim), projects both to `nef` dim |

#### `generator.py`
`G_NET` — cascaded generator with 7 stages:
- **Stage 0**: 4×4 conv transpose from noise + conditioning code
- **Stages 1–4**: Upsampling (nearest 2× → conv3×3 → BN → ReLU → cross-attention → fusion)
- **Self-attention** injected at 64×64 resolution (after stage 4)
- **Stage 5**: Another upsampling stage to 128×128
- **Stage 6**: Final upsample to 256×256 + Tanh output
- Returns intermediate images at 64×64, 128×128, and 256×256 for multi-scale discriminator supervision

`GenStage` — single reusable upsampling block with cross-attention fusion and optional self-attention.

#### `discriminator.py`
Three conditional discriminators operating at different scales:

| Class | Input Size | Downsampling Path |
|-------|-----------|-------------------|
| `D_NET64` | 64×64 | 64 → 32 → 16 → 8 → 4 |
| `D_NET128` | 128×128 | 128 → 64 → 32 → 16 → 8 → 4 |
| `D_NET256` | 256×256 | 256 → 128 → 64 → 32 → 16 → 8 → 4 |

`D_GET_LOGITS` produces a realism score by concatenating sentence embeddings (spatially replicated) with image features.

Spectral normalization (`use_sn`) is applied conditionally.

### 5. `losses/`

#### `gan.py`
`build_gan_criterion()` returns `nn.BCELoss` for adversarial training.

#### `damsm.py`
Three loss functions:

| Function | Description |
|----------|-------------|
| `words_loss` | Word-level contrastive loss: attends over spatial image features for each word, computes cosine similarity, cross-modal contrastive ranking |
| `sent_loss` | Sentence-level contrastive loss: maximizes similarity for correct pairs, minimizes for incorrect pairs (forward + backward) |
| `KL_loss` | KL divergence between N(mu, std) and N(0, 1) for conditioning augmentation regularization |

### 6. `training/`

#### `trainer.py`
Main training orchestration:

| Component | Description |
|-----------|-------------|
| `_AMPHelper` | Eliminates duplicate AMP/non-AMP code by providing unified `autocast()`, `backward()`, `step()`, `update()` interface |
| `_set_seed` | Sets all RNG seeds for reproducibility |
| `_resolve_device` | Parses device string into `torch.device` |
| `_load_pretrained_encoder` | Loads pretrained DAMSM encoder weights |
| `_sort_captions` | Sorts captions by length for packed LSTM |
| `_make_real_scales` | Creates 64/128/256 multi-resolution real images |
| `_make_fake_scales` | Generates multi-resolution fake images |
| `_train_d_step` | Updates all discriminators on real + fake images |
| `_train_g_step` | Updates generator (adversarial + DAMSM + KL losses) |
| `train` | Full training loop with validation, early stopping, checkpointing |

Features:
- Automatic Mixed Precision (AMP) via `GradScaler`
- Learning rate scheduling (`ReduceLROnPlateau`)
- Early stopping with configurable patience
- Best/latest checkpoint saving
- Loss logging to JSON

#### `validator.py`
`validate()` runs a full validation epoch without gradients, computing average D loss, G loss, DAMSM loss, and KL loss across the validation set.

### 7. `utils/`

| File | Description |
|------|-------------|
| `collate.py` | Shared `collate_fn` for DataLoader: stacks images, pads captions, squeezes lengths |
| `checkpointing.py` | Dynamic save/load for any number of discriminators (not hardcoded to 3) |
| `logging.py` | `LossLogger` — accumulates step-wise metrics and saves to timestamped JSON |

### 8. `scripts/`

| Script | Entry | Description |
|--------|-------|-------------|
| `train.py` | `attenx-train` | Loads config + optional CLI overrides, launches training |
| `generate.py` | `attenx-generate` | Loads checkpoint, generates images from text captions |
| `evaluate.py` | `attenx-evaluate` | Generates images from test set, computes Inception Score |
| `pretrain_damsm.py` | `attenx-pretrain-damsm` | Pretrains text + image encoders with DAMSM loss |

---

## How It All Fits Together — The Full Pipeline

```
┌─────────────────────────────────────────────────────────────────┐
│                       1. DAMSM Pretraining                      │
│                                                                 │
│   pretrain_damsm.py                                             │
│       ├── TextImageDataset (train split)                        │
│       ├── RNN_ENCODER ←──→ words_loss + sent_loss ←── CNN_ENCODER│
│       └── Saves: damsm_epoch_*.pth                              │
│                                                                 │
├─────────────────────────────────────────────────────────────────┤
│                     2. GAN Training                             │
│                                                                 │
│   train.py → train()                                            │
│       ├── Load frozen RNN_ENCODER + CNN_ENCODER                 │
│       ├── G_NET (generator with attention)                      │
│       │     └── Stages: 4×4 → 8×8 → 16×16 → 32×32 → 64×64     │
│       │           → self-attn → 128×128 → 256×256              │
│       ├── D_NET64 / D_NET128 / D_NET256 (discriminators)       │
│       ├── Losses: adv_gan + gamma_damsm * (word_loss +         │
│       │           sent_loss) + lambda_kl * KL_loss             │
│       ├── Validation + early stopping                           │
│       └── Saves: checkpoint_best.pth / checkpoint_latest.pth   │
│                                                                 │
├─────────────────────────────────────────────────────────────────┤
│                   3. Evaluation / Generation                    │
│                                                                 │
│   evaluate.py → inception_score()                               │
│       ├── Generates images from test captions                   │
│       └── Computes Inception Score                              │
│                                                                 │
│   generate.py                                                    │
│       └── Generates images from user-provided text              │
│                                                                 │
└─────────────────────────────────────────────────────────────────┘
```

### Training Flow (detailed):

1. **Data Loading**: `TextImageDataset` loads CUB images + captions → `DataLoader` with `collate_fn`
2. **Text Encoding**: `RNN_ENCODER` converts word tokens → word-level features + sentence embedding (frozen)
3. **Generator Forward**: 
   - Noise `z` + sentence embedding `sent_emb` → Conditioning Augmentation → latent code `c`
   - `c + z` → progressive upsampling through 7 stages
   - Cross-attention with word embeddings at each stage
   - Self-attention at 64×64 resolution
   - Outputs: 64×64, 128×128, 256×256 images
4. **Discriminator Forward**: Each scale classifies real/fake with conditional sentence embedding
5. **Loss Computation**:
   - **Adversarial**: BCE between discriminator outputs and real/fake labels
   - **DAMSM**: Word-level (attended regions vs. words) + sentence-level (global contrastive)
   - **KL**: Regularization for conditioning augmentation
6. **Backward & Update**: Discriminator steps first (`D_steps`), then generator step
7. **Validation**: Every N epochs, evaluate on test set without gradients
8. **Checkpointing**: Save best (by validation loss) and latest model

---

## Usage

### Prerequisites
```bash
pip install -r requirements.txt
```

### Configuration
Edit one of the YAML files in `configs/` or create your own.

### Training
```bash
python -m attenx_refactored.scripts.train --config configs/attenx_mhsa.yaml
```

With CLI overrides:
```bash
python -m attenx_refactored.scripts.train \
    --config configs/attenx_mhsa.yaml \
    --epochs 100 \
    --batch-size 8 \
    --use-amp
```

### DAMSM Pretraining
```bash
python -m attenx_refactored.scripts.pretrain_damsm \
    --data-dir ./data \
    --epochs 10
```

### Evaluation (Inception Score)
```bash
python -m attenx_refactored.scripts.evaluate \
    --config configs/attenx_mhsa.yaml \
    --checkpoint checkpoints/attenx_mhsa/checkpoint_best.pth \
    --num-imgs 500
```

### Generation
```bash
python -m attenx_refactored.scripts.generate \
    --config configs/attenx_mhsa.yaml \
    --checkpoint checkpoints/attenx_mhsa/checkpoint_best.pth \
    --captions "a bird with blue wings" "a small yellow bird"
```

---

## Key Design Decisions

| Decision | Rationale |
|----------|-----------|
| **`_AMPHelper` class** | Eliminates ~80 lines of duplicated AMP/non-AMP branching code |
| **Shared `collate_fn`** | Single source of truth, avoids duplicate definitions across 3 scripts |
| **Dynamic checkpointing** | Works with any number of discriminators (not hardcoded to 3) |
| **Explicit `use_sn` config** | Spectral normalization controlled independently from attention mode (was implicitly coupled) |
| **No `__init__.py` files** | Python 3.10+ supports namespace packages (PEP 420); reduces boilerplate |
| **Multi-scale discriminators** | Three discriminators at 64/128/256 provide coarse-to-fine supervision |
| **Conditioning Augmentation** | Adds stochasticity, improves diversity of generated images |

---

## Dependencies

- Python ≥ 3.10
- PyTorch ≥ 2.0.0
- torchvision ≥ 0.15.0
- NumPy, pandas, Pillow, PyYAML
