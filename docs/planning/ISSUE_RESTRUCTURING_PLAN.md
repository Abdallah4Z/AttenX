# Issue Restructuring Plan for AttenX

## Overview
This document contains the restructured issues. The following issues have been split for better granularity, easier PR reviews, and incremental testing:

- **Issue #6** → Split into #6a and #6b
- **Issue #7** → Split into #7a and #7b
- **Issue #8** → Split into #8a, #8b, and #8c
- **Issue #9** → Split into #9a and #9b

---

## NEW ISSUES

---

### Issue #6a: Mathematical Formulation of Self-Attention (Q/K/V Transformations)

**Label:** Research and Design

**Objective:** Mathematically define the Query (Q), Key (K), and Value (V) transformations for the feature maps in the Non-Local Neural Network block.

**Subtasks:**
- Study the original AttnGAN attention mechanism and identify limitations.
- Define the mathematical formulation for Q, K, V transformations on feature maps of shape $(C, H, W)$.
- Determine the embedding dimension for Q/K/V (typically $C/2$ or $C/8$).
- Document how the attention map is computed: $Attention = softmax(Q^T K) V$.
- Analyze computational complexity and memory footprint for different feature map sizes.

**Acceptance Criteria (Definition of Done):**
- A mathematical specification document detailing the Q/K/V transformations.
- Computational complexity analysis included (FLOPs and memory for 64x64, 128x128, 256x256).

**Technical Details:**
- Feature map dimensions: Varies by stage (e.g., $C \times 64 \times 64$, $C \times 128 \times 128$).
- Embedding dimension for Q/K/V: To be determined via ablation.
- Core formula: $Attention(Q, K, V) = softmax(\frac{Q^T K}{\sqrt{d_k}}) V$.
- Output: Technical document ready for implementation reference.

**Dependencies:** Issue #1 (DAMSM Deep Dive), Issue #2 (Multi-Stage Generator)

---

### Issue #6b: Self-Attention Placement Strategy and Residual Connection Design

**Label:** Research and Design

**Objective:** Decide on the optimal placement of the self-attention block and design the residual connection architecture.

**Subtasks:**
- Evaluate placement options: After $64 \times 64$, after $128 \times 128$, or multiple stages.
- Design residual/skip connection to allow the model to bypass attention if necessary.
- Plan ablation study to compare: (a) No attention, (b) Single-stage attention, (c) Multi-stage attention.
- Create the final architectural diagram for the "Enhanced-AttnGAN."

**Acceptance Criteria (Definition of Done):**
- A finalized architectural diagram showing exact placement and residual connections.
- Ablation study plan documented with expected outcomes.

**Technical Details:**
- Primary candidate placement: After the $128 \times 128$ stage (balances compute cost and global structure enforcement).
- Residual connection pattern: $Output = Input + \alpha \times AttentionBlock(Input)$ where $\alpha$ is a learnable parameter.
- Diagram tool: Draw.io, TikZ, or similar.
- Target model: Enhanced-AttnGAN.

**Dependencies:** Issue #6a (Mathematical Formulation), Issue #2 (Multi-Stage Generator)

---

### Issue #7a: Discriminator Layer Audit and Spectral Normalization Plan

**Label:** Research and Design

**Objective:** Audit the three discriminators ($D_0, D_1, D_2$) and create a technical specification for applying Spectral Normalization.

**Subtasks:**
- Map the architecture of each discriminator ($D_0, D_1, D_2$): layer-by-layer breakdown.
- Identify all convolutional layers that require spectral normalization wrapping.
- Document current learning rates and training stability issues (e.g., mode collapse evidence).
- Research and recommend learning rate adjustments (spectral norm often permits higher LRs).
- Create a technical specification document for the discriminator overhaul.

**Acceptance Criteria (Definition of Done):**
- Layer-by-layer documentation for all three discriminators.
- Technical specification document identifying which layers get wrapped and recommended LR changes.

**Technical Details:**
- Discriminators: $D_0$ (64x64), $D_1$ (128x128), $D_2$ (256x256).
- Spectral Normalization: `torch.nn.utils.spectral_norm(layer)`.
- Focus: Convolutional layers only (linear layers typically excluded).
- Output: Spec document ready for implementation.

**Dependencies:** Issue #2 (Multi-Stage Generator - for understanding architecture)

---

### Issue #7b: Implement Spectral Normalization and Tune Learning Rates

**Label:** Development and Training

**Objective:** Apply Spectral Normalization to the discriminator layers and tune learning rates for stable training.

**Subtasks:**
- Wrap identified convolutional layers with `torch.nn.utils.spectral_norm` in all three discriminators.
- Update learning rate configuration based on Issue #7a recommendations.
- Run short training trials (50-100 iterations) to verify stability.
- Monitor discriminator loss curves and check for signs of mode collapse reduction.
- Document any issues or unexpected behavior.

**Acceptance Criteria (Definition of Done):**
- All three discriminators use spectral normalization.
- Short training trials complete without instability or mode collapse.
- Loss curves logged and compared to baseline (pre-spectral-norm).

**Technical Details:**
- Implementation: `torch.nn.utils.spectral_norm(conv_layer)`.
- LR tuning: Experiment with 1x-2x baseline LR.
- Validation: 50-100 iteration trials with loss curve monitoring.
- Metrics: D loss stability, G loss trend, visual sample quality.

**Dependencies:** Issue #7a (Discriminator Audit), Issue #3 (Environment Setup)

---

### Issue #8a: Implement SelfAttention Class in modules.py

**Label:** Development and Training

**Objective:** Implement a reusable `SelfAttention` module in `modules.py`.

**Subtasks:**
- Create the `SelfAttention` class with the following interface:
  - `__init__(self, in_channels)`: Initialize Q, K, V convolutions and output projection.
  - `forward(self, x)`: Compute attention map and return refined features.
- Implement the embedded space transformation:
  - $Q = Conv_Q(x)$, $K = Conv_K(x)$, $V = Conv_V(x)$.
  - Attention weights: $S = softmax(Q^T K)$.
  - Output: $O = Conv_{out}(Attention \times V)$.
- Add the learnable scaling parameter $\alpha$ (initialized to 0).
- Write unit tests verifying:
  - Output shape matches input shape.
  - Forward pass is deterministic.
  - Gradient flow is correct.

**Acceptance Criteria (Definition of Done):**
- `SelfAttention` class implemented and documented.
- Unit tests pass for shape, determinism, and gradient flow.
- Module can be imported and initialized without errors.

**Technical Details:**
- File: `modules.py` (or equivalent).
- Input/Output shape: $(B, C, H, W)$.
- Embedded dimension: Typically $C // 8$.
- Scaling parameter: `self.alpha = nn.Parameter(torch.zeros(1))`.
- PyTorch utilities: `nn.Conv2d`, `nn.functional.softmax`, `einsum` or matrix operations.

**Dependencies:** Issue #6a (Mathematical Formulation), Issue #6b (Placement Strategy)

---

### Issue #8b: Inject Self-Attention Layers into G_Net (Generator)

**Label:** Development and Training

**Objective:** Modify the `G_Net` generator to integrate self-attention layers at the designed stages.

**Subtasks:**
- Locate the upsampling stages in `G_Net` (64x64 → 128x128 → 256x256).
- Inject `SelfAttention` module after the stage specified in Issue #6b (likely after 128x128).
- Ensure proper feature dimension passing to/from attention block.
- Verify the model initializes correctly and `print(model)` shows new layers.
- Run a forward pass with dummy input to confirm no shape mismatches.

**Acceptance Criteria (Definition of Done):**
- `G_Net` updated with self-attention injection.
- Model initialization succeeds with correct layer hierarchy.
- Forward pass with dummy tensor $(B, C, 64, 64)$ completes without errors.

**Technical Details:**
- File: `model.py` (or `G_Net` definition).
- Injection point: As defined in Issue #6b (likely after 128x128 upsampling block).
- Validation: `print(model)` output, dummy forward pass, gradient check.
- Ensure existing skip connections remain intact.

**Dependencies:** Issue #8a (SelfAttention Class), Issue #6b (Placement Strategy)

---

### Issue #8c: Update Discriminators with Spectral Normalization

**Label:** Development and Training

**Objective:** Apply spectral normalization to the discriminator classes as specified in Issue #7a.

**Subtasks:**
- Modify each discriminator class ($D_0, D_1, D_2$) to wrap convolutional layers.
- Use `torch.nn.utils.spectral_norm` on identified layers from Issue #7a.
- Ensure discriminator initialization still works with existing training code.
- Run a forward pass on each discriminator with dummy image/text inputs.
- Verify gradient flow through spectral-normalized layers.

**Acceptance Criteria (Definition of Done):**
- All three discriminators use spectral normalization on conv layers.
- Forward passes succeed for $D_0, D_1, D_2$ with dummy inputs.
- Gradient flow confirmed through spectral-normalized layers.

**Technical Details:**
- Files: Discriminator definitions (likely `model.py` or separate file).
- Wrapping: `torch.nn.utils.spectral_norm(nn.Conv2d(...))`.
- Layers to wrap: All convolutional layers in $D_0, D_1, D_2$.
- Validation: Forward pass + backward pass (gradient check).

**Dependencies:** Issue #7a (Discriminator Audit), Issue #7b (Spectral Norm Planning)

---

### Issue #9a: Refactor Training Loop for Modified Forward Pass and DAMSM Compatibility

**Label:** Development and Training

**Objective:** Update `trainer.py` to handle the modified forward pass with self-attention and ensure DAMSM loss compatibility.

**Subtasks:**
- Refactor the forward pass in `trainer.py` to accommodate self-attention outputs.
- Ensure DAMSM loss calculation works with any new feature dimensions from attention.
- Update any shape assertions or hardcoded dimensions in the training loop.
- Verify that existing loss components (G loss, D loss, DAMSM loss) still compute correctly.
- Add sanity checks for tensor shapes at critical points in the loop.

**Acceptance Criteria (Definition of Done):**
- Training loop runs 5-10 iterations without crashes or memory leaks.
- All loss components compute correctly and produce finite values.
- No shape mismatches or runtime errors in forward/backward passes.

**Technical Details:**
- File: `trainer.py`.
- Key modifications: Forward pass flow, feature dimension handling, DAMSM compatibility.
- Loss components: $L_G$ (Generator), $L_D$ (Discriminator), $L_{DAMSM}$ (Deep Attentional Multimodal Similarity).
- Validation: Dry run with 5-10 iterations, shape sanity checks, finite loss values.

**Dependencies:** Issue #8b (G_Net Updates), Issue #8c (Discriminator Updates)

---

### Issue #9b: Implement Loss Component Logging and Monitoring

**Label:** Development and Training

**Objective:** Add comprehensive logging for individual loss components to enable training analysis.

**Subtasks:**
- Implement logging for each loss component:
  - $L_G$: Generator adversarial loss.
  - $L_D$: Discriminator adversarial loss.
  - $L_{Attn}$: Attention-specific loss (if applicable).
  - $L_{DAMSM}$: Text-image alignment loss.
- Configure logging frequency (e.g., every N iterations, every epoch).
- Set up visualization tools (e.g., TensorBoard, matplotlib plots).
- Create a training monitor script/dashboard for real-time analysis.
- Document expected loss behavior and warning signs (e.g., D loss → 0, G loss → ∞).

**Acceptance Criteria (Definition of Done):**
- All loss components logged and visualized.
- Training monitor script/dashboard functional.
- Documentation for interpreting loss curves and identifying issues.

**Technical Details:**
- Logging tools: TensorBoard, CSV logging, or custom visualization.
- Loss components: $L_G$, $L_D$, $L_{Attn}$, $L_{DAMSM}$.
- Frequency: Per-iteration and per-epoch logging.
- Output: Dashboard/script + documentation for loss interpretation.

**Dependencies:** Issue #9a (Training Loop Refactor)

---

## ISSUE DEPENDENCY GRAPH

```
Foundation Phase:
├── #1 (DAMSM Deep Dive) ──────────────────┐
├── #2 (Multi-Stage Generator) ────────────┤
└── #3 (Environment Setup) ────────────────┤
                                             ↓
Data & Baseline Phase:                       │
├── #4 (Dataset Preparation)                │
├── #5 (Baseline Execution)                 │
                                             ↓
Design Phase:                                │
├── #6a (Self-Attention Math) ←─ #1, #2     │
├── #6b (Attention Placement) ←─ #6a, #2    │
├── #7a (Discriminator Audit) ←─ #2         │
└── #7b (Spectral Norm Plan) ←─ #7a         │
                                             ↓
Implementation Phase:                        │
├── #8a (SelfAttention Class) ←─ #6a, #6b   │
├── #8b (G_Net Updates) ←─ #8a, #6b         │
├── #8c (Discriminator Updates) ←─ #7a, #7b │
                                             ↓
Training Phase:                              │
├── #9a (Training Loop Refactor) ←─ #8b, #8c│
├── #9b (Loss Logging) ←─ #9a               │
├── #10 (Hyperparameter Sweep) ←─ #9b        │
                                             ↓
Evaluation Phase:                            │
├── #11 (Qualitative Analysis) ←─ #10        │
└── #12 (Quantitative Evaluation) ←─ #10     │
```

## RECOMMENDED EXECUTION ORDER

1. **Phase 1 - Foundation & Data:** #1 → #2 → #3 → #4 → #5
2. **Phase 2 - Design:** #6a → #6b → #7a → #7b (can run in parallel after #2)
3. **Phase 3 - Implementation:** #8a → #8b → #8c (#8b and #8c can run in parallel after #8a)
4. **Phase 4 - Training:** #9a → #9b → #10
5. **Phase 5 - Evaluation:** #11 → #12

## BENEFITS OF THIS RESTRUCTURING

- **Smaller PRs:** Each issue now represents a focused, reviewable change.
- **Incremental Testing:** Units can be tested independently before integration.
- **Clear Dependencies:** Explicit dependency graph prevents blockers.
- **Better Parallelism:** Some issues can be worked on simultaneously.
- **Easier Debugging:** If something breaks, smaller scope = easier root cause identification.
- **Milestone Tracking:** Each phase has clear completion criteria.
