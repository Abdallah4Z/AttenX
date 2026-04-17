# Issue #15: Discriminator Layer Audit & Spectral Norm Specification

## 1. Discriminator Architectures ($D_0, D_1, D_2$)
All three discriminators share a similar downsampling structure but operate on different image scales.

### $D_0$ ($64 \times 64$)
- Layers: Conv4x4(stride 2) $\rightarrow$ Conv4x4(stride 2) $\rightarrow$ Conv4x4(stride 2) $\rightarrow$ Conv4x4(stride 2).
- Feature Maps: $32 \rightarrow 64 \rightarrow 128 \rightarrow 256 \rightarrow 512$.

### $D_1$ ($128 \times 128$)
- Layers: Similar to $D_0$ but with an additional initial downsampling block.
- Feature Maps: Up to 1024.

### $D_2$ ($256 \times 256$)
- Layers: Most complex, handling the highest resolution.
- Feature Maps: Up to 2048.

## 2. Spectral Normalization Targets
To stabilize the training of the GAN and prevent the Discriminator from becoming too "powerful" (leading to mode collapse), we will wrap all `nn.Conv2d` layers in the discriminator classes.

**Technical Specification:**
- **Module:** `torch.nn.utils.spectral_norm`
- **Application:** Wrap all Conv layers except the final output layer (though some research suggests wrapping that too).

## 3. Learning Rate Recommendations
- **Baseline LR:** $0.0002$ (Adam optimizer).
- **Adjustment:** With Spectral Normalization, we can safely increase the LR for the Discriminator to $0.0004$ (Two-Time Scale Update Rule - TTUR) to accelerate convergence.
