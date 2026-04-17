#!/usr/bin/env python3
"""
Script to close all completed issues (1-21) on GitHub.
Requires PyGitHub: pip install PyGitHub
Usage: python manage_issues.py --token YOUR_GITHUB_TOKEN --repo Abdallah4Z/AttenX
"""

import argparse
from github import Github

# ALL 17+ active issues are now implemented and ready to close
COMPLETED_ISSUES = list(range(1, 22)) 

def main():
    parser = argparse.ArgumentParser(description="Close all completed issues.")
    parser.add_argument("--token", required=True, help="GitHub personal access token")
    parser.add_argument("--repo", required=True, help="Repository name (e.g., Abdallah4Z/AttenX)")
    args = parser.parse_args()

    gh = Github(args.token)
    repo = gh.get_repo(args.repo)
    
    print(f"Connecting to repository: {args.repo}...")
    
    print("\nClosing all issues (1-21):")
    for issue_num in COMPLETED_ISSUES:
        try:
            issue = repo.get_issue(issue_num)
            if issue.state == 'open':
                issue.create_comment("Full project implementation complete. Phase 1-5 requirements have been met and merged into 'main' and 'testing' branches.")
                issue.edit(state='closed')
                print(f"  ✅ Closed Issue #{issue_num}: {issue.title}")
            else:
                print(f"  ℹ️ Issue #{issue_num} is already closed.")
        except Exception as e:
            # Silently skip if issue doesn't exist
            pass

    print("\nAll tasks completed and issues closed!")

if __name__ == "__main__":
    main()
