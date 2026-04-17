#!/usr/bin/env python3
"""
Script to assign the 5 follow-up issues to team members on GitHub.
Usage: python assign_followup_issues.py --token YOUR_GITHUB_TOKEN --repo Abdallah4Z/AttenX
"""

import argparse
from github import Github

# Assignments mapping
ASSIGNMENTS = {
    22: ["Abdallah4Z", "ahmedislamfarouk"],  # Implementation - Full DAMSM Loss
    23: ["ammarlhassan"],                    # Infrastructure - Tokenization
    24: ["mezo04"],                          # Development - Unit Testing
    25: ["fourarms4x4", "ahmedislamfarouk"], # Research - Ablation Study
    26: ["ammarlhassan", "mezo04"]           # Deployment - Inference Interface
}

def main():
    parser = argparse.ArgumentParser(description="Assign follow-up issues to team members.")
    parser.add_argument("--token", required=True, help="GitHub personal access token")
    parser.add_argument("--repo", required=True, help="Repository name")
    args = parser.parse_args()

    gh = Github(args.token)
    repo = gh.get_repo(args.repo)
    
    print(f"Assigning follow-up issues for {args.repo}...\n")
    
    for issue_num, assignees in ASSIGNMENTS.items():
        try:
            issue = repo.get_issue(issue_num)
            # Add assignees to the issue
            issue.add_to_assignees(*assignees)
            print(f"  ✅ Issue #{issue_num} ('{issue.title}') assigned to: {', '.join(assignees)}")
        except Exception as e:
            print(f"  ❌ Failed to assign Issue #{issue_num}: {e}")

    print("\nAll assignments complete!")

if __name__ == "__main__":
    main()
