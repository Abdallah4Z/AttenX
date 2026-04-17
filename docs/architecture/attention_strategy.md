# Issue #14: Self-Attention Placement & Residual Strategy

## 1. Optimal Placement Evaluation
We evaluated three potential injection points for the Self-Attention module within the `G_Net` architecture.

| Option | Placement Stage | Pros | Cons |
|--------|-----------------|------|------|
| **A** | After $64 \times 64$ ($F_0$) | Captures global layout early. | Low resolution might miss critical feature relationships. |
| **B** | After $128 \times 128$ ($F_1$) | **Selected.** Balanced resolution for structural coherence. | Moderate computational overhead. |
| **C** | After $256 \times 256$ ($F_2$) | Very fine detail refinement. | Extremely high memory cost; too late for structural fix. |

**Decision:** Inject after the $128 \times 128$ stage to refine the transition between coarse layout and final high-resolution details.

## 2. Residual Connection Architecture
To ensure training stability, the self-attention output will be added to the input via a skip connection with a learnable scale parameter ($\gamma$).

$$ \text{Output} = \gamma \cdot \text{SelfAttention}(x) + x $$

- **Initialization:** $\gamma$ is initialized to 0, allowing the model to initially ignore the attention block and gradually learn to incorporate it.

## 3. Ablation Study Plan
1. **Control:** Original AttnGAN (No Self-Attention).
2. **Experiment 1:** Single-stage attention after $128 \times 128$.
3. **Experiment 2:** Dual-stage attention (after $64 \times 64$ and $128 \times 128$).
