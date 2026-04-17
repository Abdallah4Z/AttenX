#!/usr/bin/env python3
"""
Script to close completed issues and create a new follow-up issue on GitHub.
Requires PyGitHub: pip install PyGitHub
Usage: python manage_issues.py --token YOUR_GITHUB_TOKEN --repo Abdallah4Z/AttenX
"""

import argparse
from github import Github

# Issues that have been implemented and are ready to close
COMPLETED_ISSUES = [2, 4, 5, 10, 11, 12, 14, 15, 16, 18, 19, 21]

NEW_ISSUE = {
    "title": "IMPLEMENTATION - Unified Training Pipeline (train.py)",
    "body": """**Objective:** Integrate the newly developed model components (`code/generator.py`, `code/model.py`) and dataset loaders (`code/datasets.py`) into a single unified training script (`train.py`).

**Tasks:**
- Create `train.py` that imports `G_NET` and the three discriminators (`D_NET64`, `D_NET128`, `D_NET256`).
- Initialize models, optimizers (with TTUR learning rates as specified in Issue #15), and the `TextDataset`.
- Implement the forward and backward passes using the provided loss logging class (`scripts/loss_logging.py`).
- Implement periodic model checkpointing.

**Dependencies:** All Phase 3 Implementation issues.
**Priority:** High
""",
    "labels": ["Development and Training", "Phase 4: Training", "priority: high"]
}

def main():
    parser = argparse.ArgumentParser(description="Close completed issues and create new ones.")
    parser.add_argument("--token", required=True, help="GitHub personal access token")
    parser.add_argument("--repo", required=True, help="Repository name (e.g., Abdallah4Z/AttenX)")
    args = parser.parse_args()

    gh = Github(args.token)
    repo = gh.get_repo(args.repo)
    
    print(f"Connecting to repository: {args.repo}...")
    
    # 1. Close completed issues
    print("\n[1] Closing Completed Issues:")
    for issue_num in COMPLETED_ISSUES:
        try:
            issue = repo.get_issue(issue_num)
            if issue.state == 'open':
                issue.create_comment("Implementation complete. All scripts, components, and documentation have been merged into the `main` and `testing` branches.")
                issue.edit(state='closed')
                print(f"  ✅ Closed Issue #{issue_num}: {issue.title}")
            else:
                print(f"  ℹ️ Issue #{issue_num} is already closed.")
        except Exception as e:
            print(f"  ❌ Failed to close Issue #{issue_num}: {e}")

    # 2. Create new follow-up issue
    print("\n[2] Creating New Issue:")
    try:
        new_issue = repo.create_issue(
            title=NEW_ISSUE["title"],
            body=NEW_ISSUE["body"],
            labels=NEW_ISSUE["labels"]
        )
        print(f"  ✅ Created new issue: {new_issue.title} (Issue #{new_issue.number})")
    except Exception as e:
        print(f"  ❌ Failed to create new issue: {e}")
        
    print("\nOperation complete!")

if __name__ == "__main__":
    main()
