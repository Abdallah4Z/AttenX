#!/usr/bin/env python3
"""
Script to create restructured issues on GitHub.
Requires PyGitHub: pip install PyGitHub
Usage: python create_issues.py --token YOUR_GITHUB_TOKEN --repo Abdallah4Z/AttenX
"""

import argparse
from github import Github

ISSUES = [
    {
        "title": "DESIGN - Self-Attention Mathematical Formulation (Q/K/V Transformations)",
        "body": """**Objective:** Mathematically define the Query (Q), Key (K), and Value (V) transformations for the feature maps in the Non-Local Neural Network block.

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
- Feature map dimensions: Varies by stage (e.g., $C \\times 64 \\times 64$, $C \\times 128 \\times 128$).
- Embedding dimension for Q/K/V: To be determined via ablation.
- Core formula: $Attention(Q, K, V) = softmax(\\frac{Q^T K}{\\sqrt{d_k}}) V$.
- Output: Technical document ready for implementation reference.

**Dependencies:** Issue #1 (DAMSM Deep Dive), Issue #2 (Multi-Stage Generator)

**Parent Issue:** This is part of the original Issue #6 split.
""",
        "labels": ["Research and Design"]
    },
    {
        "title": "DESIGN - Self-Attention Placement Strategy & Residual Connection",
        "body": """**Objective:** Decide on the optimal placement of the self-attention block and design the residual connection architecture.

**Subtasks:**
- Evaluate placement options: After $64 \\times 64$, after $128 \\times 128$, or multiple stages.
- Design residual/skip connection to allow the model to bypass attention if necessary.
- Plan ablation study to compare: (a) No attention, (b) Single-stage attention, (c) Multi-stage attention.
- Create the final architectural diagram for the "Enhanced-AttnGAN."

**Acceptance Criteria (Definition of Done):**
- A finalized architectural diagram showing exact placement and residual connections.
- Ablation study plan documented with expected outcomes.

**Technical Details:**
- Primary candidate placement: After the $128 \\times 128$ stage (balances compute cost and global structure enforcement).
- Residual connection pattern: $Output = Input + \\alpha \\times AttentionBlock(Input)$ where $\\alpha$ is a learnable parameter.
- Diagram tool: Draw.io, TikZ, or similar.
- Target model: Enhanced-AttnGAN.

**Dependencies:** Issue #6a (Mathematical Formulation), Issue #2 (Multi-Stage Generator)

**Parent Issue:** This is part of the original Issue #6 split.
""",
        "labels": ["Research and Design"]
    },
    {
        "title": "DESIGN - Discriminator Layer Audit for Spectral Normalization",
        "body": """**Objective:** Audit the three discriminators ($D_0, D_1, D_2$) and create a technical specification for applying Spectral Normalization.

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

**Parent Issue:** This is part of the original Issue #7 split.
""",
        "labels": ["Research and Design"]
    },
    {
        "title": "IMPLEMENTATION - Apply Spectral Normalization & Tune Learning Rates",
        "body": """**Objective:** Apply Spectral Normalization to the discriminator layers and tune learning rates for stable training.

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

**Parent Issue:** This is part of the original Issue #7 split.
""",
        "labels": ["Development and Training"]
    },
    {
        "title": "IMPLEMENTATION - SelfAttention Module in modules.py",
        "body": """**Objective:** Implement a reusable `SelfAttention` module in `modules.py`.

**Subtasks:**
- Create the `SelfAttention` class with the following interface:
  - `__init__(self, in_channels)`: Initialize Q, K, V convolutions and output projection.
  - `forward(self, x)`: Compute attention map and return refined features.
- Implement the embedded space transformation:
  - $Q = Conv_Q(x)$, $K = Conv_K(x)$, $V = Conv_V(x)$.
  - Attention weights: $S = softmax(Q^T K)$.
  - Output: $O = Conv_{out}(Attention \\times V)$.
- Add the learnable scaling parameter $\\alpha$ (initialized to 0).
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

**Parent Issue:** This is part of the original Issue #8 split.
""",
        "labels": ["Development and Training"]
    },
    {
        "title": "IMPLEMENTATION - Inject Self-Attention into G_Net (Generator)",
        "body": """**Objective:** Modify the `G_Net` generator to integrate self-attention layers at the designed stages.

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

**Parent Issue:** This is part of the original Issue #8 split.
""",
        "labels": ["Development and Training"]
    },
    {
        "title": "IMPLEMENTATION - Update Discriminators with Spectral Normalization",
        "body": """**Objective:** Apply spectral normalization to the discriminator classes as specified in Issue #7a.

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

**Parent Issue:** This is part of the original Issue #8 split.
""",
        "labels": ["Development and Training"]
    },
    {
        "title": "IMPLEMENTATION - Refactor Training Loop for Self-Attention & DAMSM Compatibility",
        "body": """**Objective:** Update `trainer.py` to handle the modified forward pass with self-attention and ensure DAMSM loss compatibility.

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

**Parent Issue:** This is part of the original Issue #9 split.
""",
        "labels": ["Development and Training"]
    },
    {
        "title": "IMPLEMENTATION - Loss Component Logging & Monitoring",
        "body": """**Objective:** Add comprehensive logging for individual loss components to enable training analysis.

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

**Parent Issue:** This is part of the original Issue #9 split.
""",
        "labels": ["Development and Training"]
    }
]


def main():
    parser = argparse.ArgumentParser(description="Create restructured issues on GitHub")
    parser.add_argument("--token", required=True, help="GitHub personal access token")
    parser.add_argument("--repo", required=True, help="Repository name (e.g., Abdallah4Z/AttenX)")
    parser.add_argument("--dry-run", action="store_true", help="Print issues without creating them")
    args = parser.parse_args()

    if args.dry_run:
        print("🔍 DRY RUN MODE - Issues will be printed but NOT created\n")
        for i, issue in enumerate(ISSUES, 1):
            print(f"\n{'='*80}")
            print(f"Issue {i}: {issue['title']}")
            print(f"{'='*80}")
            print(f"Labels: {', '.join(issue['labels'])}")
            print(f"\nBody:\n{issue['body']}")
            print(f"{'='*80}\n")
        return

    print(f"🚀 Creating {len(ISSUES)} issues on {args.repo}...\n")
    
    gh = Github(args.token)
    repo = gh.get_repo(args.repo)

    for i, issue in enumerate(ISSUES, 1):
        print(f"Creating issue {i}/{len(ISSUES)}: {issue['title']}")
        try:
            new_issue = repo.create_issue(
                title=issue['title'],
                body=issue['body']
            )
            print(f"✅ Created: {new_issue.html_url}")
            
            # Add labels if they exist
            if issue['labels']:
                try:
                    new_issue.add_to_labels(*issue['labels'])
                    print(f"   Labels added: {', '.join(issue['labels'])}")
                except Exception as e:
                    print(f"   ⚠️  Could not add labels: {e}")
        except Exception as e:
            print(f"❌ Failed to create issue: {e}")
        print()

    print("✨ Done!")


if __name__ == "__main__":
    main()
