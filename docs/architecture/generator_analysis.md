# Issue #2: Multi-Stage Generator Architecture Analysis

## 1. Feature Flow Mapping
The AttenX generator follows a hierarchical forest structure ($F_0, F_1, F_2$) where each stage refines the image resolution and adds local details.

### Stage 0: $F_0$ (Initial Image)
- **Input:** Noise vector $z$ and global sentence embedding.
- **Output:** $64 \times 64$ hidden feature map.
- **Focus:** Global structure, coarse shapes, and primary color distributions.

### Stage 1: $F_1$ (Refinement 1)
- **Input:** $F_0$ output + word-level attention features.
- **Upsampling:** $64 \times 64 \rightarrow 128 \times 128$.
- **Mechanism:** Usually involves `nn.Upsample` followed by $3 \times 3$ convolutions to reduce checkerboard artifacts.

### Stage 2: $F_2$ (Refinement 2)
- **Input:** $F_1$ output + word-level attention features.
- **Upsampling:** $128 \times 128 \rightarrow 256 \times 256$.
- **Focus:** Fine textures, sharp edges, and detailed object features (e.g., bird feathers, eye highlights).

## 2. Global vs. Local Transition
- **Global Structure:** Primarily determined in $F_0$ and the early layers of $F_1$.
- **Local Detail:** Dominates the $F_2$ stage.
- **Observation:** If the attention mechanism in $F_1$ is weak, the $F_2$ stage often produces "hallucinated" details that don't align with the text, leading to anatomical inconsistencies.

## 3. Upsampling vs. Deconvolution
- **Original AttnGAN:** Uses `nn.Upsample(scale_factor=2, mode='nearest')` followed by a convolution.
- **Analysis:** Nearest-neighbor upsampling is computationally efficient and avoids the checkerboard artifacts common with `ConvTranspose2d` (Deconvolution), provided the subsequent convolution is properly trained.
