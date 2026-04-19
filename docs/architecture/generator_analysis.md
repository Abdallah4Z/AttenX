# Issue #2: Multi-Stage Generator Architecture Analysis

## 1. Feature Flow Mapping
The AttenX generator follows a 7-stage hierarchical upsampling structure (Stage 0-6) producing 256x256 images from a 100-dimensional noise vector.

### Stage 0 ($F_0$): Initial Feature Map
- **Input:** Noise vector $z$ (100 dim).
- **Output:** $4 \times 4$ feature map, 1024 channels.
- **Layer:** `ConvTranspose2d(100, ngf*16, 4, 1, 0)` → BatchNorm → ReLU.

### Stage 1 ($F_1$): 4x4 → 8x8
- **Input:** Stage 0 output.
- **Output:** $8 \times 8$, 512 channels.
- **Layer:** `nn.Upsample(scale_factor=2, mode='nearest')` → Conv3x3.

### Stage 2 ($F_2$): 8x8 → 16x16
- **Input:** Stage 1 output.
- **Output:** $16 \times 16$, 256 channels.
- **Focus:** Coarse structure emergence.

### Stage 3 ($F_3$): 16x16 → 32x32
- **Input:** Stage 2 output.
- **Output:** $32 \times 32$, 128 channels.
- **Focus:** Shape refinement.

### Stage 4 ($F_4$): 32x32 → 64x64
- **Input:** Stage 3 output.
- **Output:** $64 \times 64$, 64 channels.
- **Self-Attention:** Applied here for global coherence.

### Stage 5 ($F_5$): 64x64 → 128x128
- **Input:** Attention output.
- **Output:** $128 \times 128$, 32 channels.
- **Focus:** Detail introduction.

### Stage 6 ($F_6$): 128x128 → 256x256
- **Input:** Stage 5 output.
- **Output:** $256 \times 256$, 16 channels.
- **Focus:** Fine textures, sharp edges, and detailed object features.

### Final Output
- **Layer:** Conv3x3(16, 3) → Tanh.
- **Output:** $256 \times 256$ RGB image.

## 2. Global vs. Local Transition
- **Global Structure:** Primarily determined in Stages 0-3.
- **Local Detail:** Dominates Stages 4-6 with self-attention at 64x64.
- **Observation:** If the self-attention mechanism at Stage 4 is weak, the later stages often produce "hallucinated" details that don't align with the text.

## 3. Upsampling vs. Deconvolution
- **AttenX:** Uses `nn.Upsample(scale_factor=2, mode='nearest')` followed by convolution (Stages 1-6).
- **Stage 0:** Uses `ConvTranspose2d` for initial 4x4 projection.
- **Analysis:** Nearest-neighbor upsampling is computationally efficient and avoids the checkerboard artifacts common with `ConvTranspose2d`.
