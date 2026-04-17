#!/usr/bin/env python3
"""
Script to organize all AttenX issues with proper labels and team assignments.
Balances workload across 5 team members across 5 phases.
"""

import argparse
from github import Github

# Team members
TEAM = [
    "Abdallah4Z",
    "ammarlhassan", 
    "fourarms4x4",
    "ahmedislamfarouk",
    "mezo04"
]

# Phase-based label colors
NEW_LABELS = [
    {"name": "Phase 1: Foundation", "color": "1D76DB"},      # Blue
    {"name": "Phase 2: Design", "color": "5319E7"},           # Purple
    {"name": "Phase 3: Implementation", "color": "0E8A16"},  # Green
    {"name": "Phase 4: Training", "color": "E4E669"},        # Yellow
    {"name": "Phase 5: Evaluation", "color": "D73A4A"},      # Red
    {"name": "priority: high", "color": "B60205"},           # Dark Red
    {"name": "priority: medium", "color": "FBCA04"},         # Yellow
    {"name": "priority: low", "color": "0E8A16"},            # Green
    {"name": "status: blocked", "color": "D73A4A"},          # Red
    {"name": "status: in-progress", "color": "1D76DB"},      # Blue
    {"name": "status: ready", "color": "0E8A16"},            # Green
]

# Issue organization plan with assignments
ISSUE_PLAN = [
    # PHASE 1: FOUNDATION (Issues #1-#5)
    {"issue_num": 1, "assignees": ["Abdallah4Z"], "phase": "Phase 1: Foundation", "priority": "high", "dependencies": []},
    {"issue_num": 2, "assignees": ["ammarlhassan"], "phase": "Phase 1: Foundation", "priority": "high", "dependencies": []},
    {"issue_num": 3, "assignees": ["fourarms4x4"], "phase": "Phase 1: Foundation", "priority": "high", "dependencies": []},
    {"issue_num": 4, "assignees": ["ahmedislamfarouk"], "phase": "Phase 1: Foundation", "priority": "medium", "dependencies": [3]},
    {"issue_num": 5, "assignees": ["mezo04"], "phase": "Phase 1: Foundation", "priority": "medium", "dependencies": [3, 4]},

    # PHASE 2: DESIGN (Issues #6a, #6b, #7a, #7b → #13-#16)
    {"issue_num": 13, "assignees": ["Abdallah4Z"], "phase": "Phase 2: Design", "priority": "high", "dependencies": [1, 2]},
    {"issue_num": 14, "assignees": ["ammarlhassan"], "phase": "Phase 2: Design", "priority": "high", "dependencies": [13, 2]},
    {"issue_num": 15, "assignees": ["fourarms4x4"], "phase": "Phase 2: Design", "priority": "high", "dependencies": [2]},
    {"issue_num": 16, "assignees": ["ahmedislamfarouk"], "phase": "Phase 2: Design", "priority": "high", "dependencies": [15]},

    # PHASE 3: IMPLEMENTATION (Issues #8a, #8b, #8c → #17-#19)
    {"issue_num": 17, "assignees": ["mezo04"], "phase": "Phase 3: Implementation", "priority": "high", "dependencies": [13, 14]},
    {"issue_num": 18, "assignees": ["Abdallah4Z", "ammarlhassan"], "phase": "Phase 3: Implementation", "priority": "high", "dependencies": [17, 14]},
    {"issue_num": 19, "assignees": ["fourarms4x4", "ahmedislamfarouk"], "phase": "Phase 3: Implementation", "priority": "high", "dependencies": [15, 16]},

    # PHASE 4: TRAINING (Issues #9a, #9b → #20-#21)
    {"issue_num": 20, "assignees": ["Abdallah4Z", "ammarlhassan"], "phase": "Phase 4: Training", "priority": "high", "dependencies": [18, 19]},
    {"issue_num": 21, "assignees": ["mezo04"], "phase": "Phase 4: Training", "priority": "medium", "dependencies": [20]},

    # PHASE 5: EVALUATION (Issues #10-#12)
    {"issue_num": 10, "assignees": ["Abdallah4Z", "fourarms4x4"], "phase": "Phase 4: Training", "priority": "high", "dependencies": [20, 21]},
    {"issue_num": 11, "assignees": ["ahmedislamfarouk", "mezo04"], "phase": "Phase 5: Evaluation", "priority": "high", "dependencies": [10]},
    {"issue_num": 12, "assignees": ["ammarlhassan", "Abdallah4Z"], "phase": "Phase 5: Evaluation", "priority": "high", "dependencies": [10]},
]


def create_labels(repo):
    """Create new labels if they don't exist."""
    existing = {label.name for label in repo.get_labels()}
    created = []
    
    for label_info in NEW_LABELS:
        if label_info["name"] not in existing:
            try:
                repo.create_label(label_info["name"], label_info["color"])
                created.append(label_info["name"])
            except Exception as e:
                print(f"  ⚠️  Could not create label '{label_info['name']}': {e}")
    
    return created


def organize_issues(repo):
    """Assign issues and add labels according to the plan."""
    print("\n📋 Organizing issues...\n")
    
    for plan in ISSUE_PLAN:
        issue_num = plan["issue_num"]
        try:
            issue = repo.get_issue(issue_num)
            print(f"Processing Issue #{issue_num}: {issue.title}")
            
            # Add assignees
            if plan["assignees"]:
                try:
                    issue.add_to_assignees(*plan["assignees"])
                    print(f"  ✅ Assigned to: {', '.join(plan['assignees'])}")
                except Exception as e:
                    print(f"  ⚠️  Could not assign issue: {e}")
            
            # Add labels
            labels_to_add = []
            
            # Phase label
            labels_to_add.append(plan["phase"])
            
            # Priority label
            labels_to_add.append(f"priority: {plan['priority']}")
            
            # Existing label based on phase
            if "Foundation" in plan["phase"] or "Environment" in issue.title:
                labels_to_add.append("Environment and Baselines")
            elif "Design" in plan["phase"]:
                labels_to_add.append("Research and Design")
            elif "Implementation" in plan["phase"] or "Training" in plan["phase"]:
                labels_to_add.append("Development and Training")
            elif "Evaluation" in plan["phase"]:
                labels_to_add.append("Evaluation and Science")
            
            # Add dependency note if applicable
            if plan["dependencies"]:
                dep_text = f"\n\n**Blocked by:** Issues {', '.join(f'#{d}' for d in plan['dependencies'])}"
                if dep_text not in issue.body:
                    issue.edit(body=issue.body + dep_text)
            
            try:
                issue.add_to_labels(*labels_to_add)
                print(f"  🏷️  Labels added: {', '.join(labels_to_add)}")
            except Exception as e:
                print(f"  ⚠️  Could not add labels: {e}")
            
            print()
            
        except Exception as e:
            print(f"❌ Failed to process Issue #{issue_num}: {e}\n")


def print_summary():
    """Print workload distribution summary."""
    print("\n" + "="*80)
    print("📊 WORKLOAD DISTRIBUTION SUMMARY")
    print("="*80 + "\n")
    
    workload = {member: [] for member in TEAM}
    
    for plan in ISSUE_PLAN:
        for assignee in plan["assignees"]:
            if assignee in workload:
                workload[assignee].append(plan["issue_num"])
    
    print("Team Member Assignments:\n")
    for member in TEAM:
        issues = workload[member]
        print(f"  {member}:")
        print(f"    Issues: {', '.join(f'#{i}' for i in issues)}")
        print(f"    Total: {len(issues)} issues")
        print()
    
    print("\nPhase Breakdown:\n")
    phases = {
        "Phase 1: Foundation": [],
        "Phase 2: Design": [],
        "Phase 3: Implementation": [],
        "Phase 4: Training": [],
        "Phase 5: Evaluation": []
    }
    
    for plan in ISSUE_PLAN:
        phases[plan["phase"]].append(plan["issue_num"])
    
    for phase, issues in phases.items():
        print(f"  {phase}:")
        print(f"    Issues: {', '.join(f'#{i}' for i in issues)}")
        print(f"    Count: {len(issues)}")
        print()
    
    print("="*80)
    print("✅ Organization complete!")
    print("="*80)


def main():
    parser = argparse.ArgumentParser(description="Organize AttenX issues with labels and assignments")
    parser.add_argument("--token", required=True, help="GitHub personal access token")
    parser.add_argument("--repo", required=True, help="Repository name (e.g., Abdallah4Z/AttenX)")
    parser.add_argument("--dry-run", action="store_true", help="Print plan without making changes")
    args = parser.parse_args()

    if args.dry_run:
        print("🔍 DRY RUN MODE - No changes will be made\n")
        print_summary()
        return

    print(f"🚀 Organizing issues on {args.repo}...\n")
    
    gh = Github(args.token)
    repo = gh.get_repo(args.repo)
    
    # Step 1: Create labels
    print("🏷️  Creating labels...")
    created = create_labels(repo)
    if created:
        print(f"✅ Created labels: {', '.join(created)}")
    else:
        print("✅ All labels already exist")
    
    # Step 2: Organize issues
    organize_issues(repo)
    
    # Step 3: Print summary
    print_summary()


if __name__ == "__main__":
    main()
