#!/usr/bin/env python3
"""
Script to create follow-up issues for the next stages of AttenX.
Usage: python create_followup_issues.py --token YOUR_GITHUB_TOKEN --repo Abdallah4Z/AttenX
"""

import argparse
from github import Github

FOLLOWUP_ISSUES = [
    {
        "title": "IMPLEMENTATION - Full DAMSM Loss Integration",
        "body": """**Objective:** Implement the actual DAMSM loss calculation within the training loop to provide fine-grained text-image alignment.

**Tasks:**
- Load pre-trained DAMSM (RNN and Inception) weights.
- Implement `cosine_similarity` between word embeddings and image sub-regions.
- Integrate the global sentence-image similarity loss.
- Balance the DAMSM loss weight ($\gamma_{damsm}$) against the GAN loss.

**Acceptance Criteria:**
- Training loop successfully computes and backpropagates DAMSM loss.
- Loss logging shows separate word-level and sentence-level similarity components.

**Priority:** High
**Label:** Development and Training
""",
        "labels": ["Development and Training", "Phase 4: Training", "priority: high"]
    },
    {
        "title": "INFRASTRUCTURE - Automated Text Tokenization & Embedding Pipeline",
        "body": """**Objective:** Automate the generation of caption pickles and text embeddings for new or custom datasets.

**Tasks:**
- Implement a script to tokenize captions using a vocabulary.
- Generate word-level and sentence-level embeddings using a pre-trained RNN.
- Export preprocessed data into the `.pickle` format expected by `TextDataset`.
- Support for CUB-200-2011 and COCO datasets.

**Acceptance Criteria:**
- A single script that converts raw images+captions into training-ready pickles.

**Priority:** Medium
**Label:** Environment and Baselines
""",
        "labels": ["Environment and Baselines", "Phase 1: Foundation", "priority: medium"]
    },
    {
        "title": "DEVELOPMENT - Unit Testing Suite for Neural Modules",
        "body": """**Objective:** Ensure the structural integrity and numerical stability of custom modules like `SelfAttention`.

**Tasks:**
- Write unit tests for `SelfAttention` (check output shapes, gradient flow, and $\gamma$ parameter learning).
- Write unit tests for `G_NET` and `D_NET` initializations.
- Test `TextDataset` with edge cases (missing images, empty captions).
- Verify Spectral Normalization is correctly applied to all target layers.

**Acceptance Criteria:**
- `pytest` suite passes with 100% success on all core modules.

**Priority:** Medium
**Label:** Development and Training
""",
        "labels": ["Development and Training", "Phase 3: Implementation", "priority: medium"]
    },
    {
        "title": "RESEARCH - Systematic Ablation Study Execution",
        "body": """**Objective:** Execute the ablation study planned in Issue #14 to statistically prove the benefit of the selected attention placement.

**Tasks:**
- Train 3 models: (1) Baseline AttnGAN, (2) AttenX with Stage 1 Attention, (3) AttenX with Stage 1+2 Attention.
- Compare IS and FID scores across all three versions.
- Perform visual comparison on the "Golden Prompts".
- Document results for the final research paper.

**Acceptance Criteria:**
- Final report/table showing the performance trade-offs for each attention configuration.

**Priority:** High
**Label:** Evaluation and Science
""",
        "labels": ["Evaluation and Science", "Phase 5: Evaluation", "priority: high"]
    },
    {
        "title": "DEPLOYMENT - Interactive Inference Interface",
        "body": """**Objective:** Create a user-friendly script for generating images from arbitrary text prompts using the trained weights.

**Tasks:**
- Implement `generate.py` that takes a text string as input.
- Load the best model checkpoint automatically.
- Generate and display the $64 \times 64$, $128 \times 128$, and $256 \times 256$ stages.
- Save result with a timestamped filename.

**Acceptance Criteria:**
- Users can run `python generate.py --text "a red bird"` and receive a high-res image.

**Priority:** Medium
**Label:** Development and Training
""",
        "labels": ["Development and Training", "Phase 5: Evaluation", "priority: medium"]
    }
]

def main():
    parser = argparse.ArgumentParser(description="Create follow-up issues.")
    parser.add_argument("--token", required=True, help="GitHub personal access token")
    parser.add_argument("--repo", required=True, help="Repository name")
    args = parser.parse_args()

    gh = Github(args.token)
    repo = gh.get_repo(args.repo)
    
    print(f"Creating 5 follow-up issues for {args.repo}...")
    
    for issue_data in FOLLOWUP_ISSUES:
        try:
            issue = repo.create_issue(
                title=issue_data["title"],
                body=issue_data["body"],
                labels=issue_data["labels"]
            )
            print(f"  ✅ Created Issue #{issue.number}: {issue.title}")
        except Exception as e:
            print(f"  ❌ Failed to create issue '{issue_data['title']}': {e}")

    print("\nNext stage roadmap successfully created!")

if __name__ == "__main__":
    main()
