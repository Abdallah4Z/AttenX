# AttenX Team Assignments & Project Organization

## 👥 Team Members

| Team Member | GitHub Username |
|-------------|----------------|
| **Abdallah** | @Abdallah4Z |
| **Ammar** | @ammarlhassan |
| **Four Arms** | @fourarms4x4 |
| **Ahmed** | @ahmedislamfarouk |
| **Mezo** | @mezo04 |

---

## 📊 Workload Distribution

### Individual Assignments

#### @Abdallah4Z (6 issues - Project Lead)
| Issue | Title | Phase | Priority | Status |
|-------|-------|-------|----------|--------|
| #1 | Deep Dive - AttnGAN Text-Encoder and DAMSM | Phase 1 | 🔴 High | 🔵 Ready |
| #13 | DESIGN - Self-Attention Mathematical Formulation | Phase 2 | 🔴 High | 🔵 Ready |
| #18 | IMPLEMENTATION - Inject Self-Attention into G_Net *(with @ammarlhassan)* | Phase 3 | 🔴 High | 🔵 Ready |
| #20 | IMPLEMENTATION - Refactor Training Loop *(with @ammarlhassan)* | Phase 4 | 🔴 High | 🔵 Ready |
| #10 | Training - Hyperparameter Sweep *(with @fourarms4x4)* | Phase 4 | 🔴 High | 🔵 Ready |
| #12 | Quantitative Evaluation - IS and FID *(with @ammarlhassan)* | Phase 5 | 🔴 High | 🔵 Ready |

#### @ammarlhassan (5 issues)
| Issue | Title | Phase | Priority | Status |
|-------|-------|-------|----------|--------|
| #2 | Deep Dive - Multi-Stage Generator Architecture | Phase 1 | 🔴 High | 🔵 Ready |
| #14 | DESIGN - Self-Attention Placement Strategy | Phase 2 | 🔴 High | 🔵 Ready |
| #18 | IMPLEMENTATION - Inject Self-Attention into G_Net *(with @Abdallah4Z)* | Phase 3 | 🔴 High | 🔵 Ready |
| #20 | IMPLEMENTATION - Refactor Training Loop *(with @Abdallah4Z)* | Phase 4 | 🔴 High | 🔵 Ready |
| #12 | Quantitative Evaluation - IS and FID *(with @Abdallah4Z)* | Phase 5 | 🔴 High | 🔵 Ready |

#### @fourarms4x4 (4 issues)
| Issue | Title | Phase | Priority | Status |
|-------|-------|-------|----------|--------|
| #3 | Environment Setup and Dependency Resolution | Phase 1 | 🔴 High | 🔵 Ready |
| #15 | DESIGN - Discriminator Layer Audit for Spectral Normalization | Phase 2 | 🔴 High | 🔵 Ready |
| #19 | IMPLEMENTATION - Update Discriminators *(with @ahmedislamfarouk)* | Phase 3 | 🔴 High | 🔵 Ready |
| #10 | Training - Hyperparameter Sweep *(with @Abdallah4Z)* | Phase 4 | 🔴 High | 🔵 Ready |

#### @ahmedislamfarouk (4 issues)
| Issue | Title | Phase | Priority | Status |
|-------|-------|-------|----------|--------|
| #4 | Dataset Preparation and Preprocessing | Phase 1 | 🟡 Medium | 🔵 Ready |
| #16 | IMPLEMENTATION - Apply Spectral Normalization & Tune LR | Phase 2 | 🔴 High | 🔵 Ready |
| #19 | IMPLEMENTATION - Update Discriminators *(with @fourarms4x4)* | Phase 3 | 🔴 High | 🔵 Ready |
| #11 | Qualitative Analysis - Visual Comparison *(with @mezo04)* | Phase 5 | 🔴 High | 🔵 Ready |

#### @mezo04 (4 issues)
| Issue | Title | Phase | Priority | Status |
|-------|-------|-------|----------|--------|
| #5 | Baseline Execution and Snapshotting | Phase 1 | 🟡 Medium | 🔵 Ready |
| #17 | IMPLEMENTATION - SelfAttention Module in modules.py | Phase 3 | 🔴 High | 🔵 Ready |
| #21 | IMPLEMENTATION - Loss Component Logging & Monitoring | Phase 4 | 🟡 Medium | 🔵 Ready |
| #11 | Qualitative Analysis - Visual Comparison *(with @ahmedislamfarouk)* | Phase 5 | 🔴 High | 🔵 Ready |

---

## 🏗️ Phase-Based Organization

### Phase 1: Foundation 🔵
**Goal:** Understand architecture, setup environment, establish baselines

| Issue | Assignee | Priority | Dependencies |
|-------|----------|----------|--------------|
| #1 DAMSM Deep Dive | @Abdallah4Z | 🔴 High | None |
| #2 Multi-Stage Generator | @ammarlhassan | 🔴 High | None |
| #3 Environment Setup | @fourarms4x4 | 🔴 High | None |
| #4 Dataset Preparation | @ahmedislamfarouk | 🟡 Medium | #3 |
| #5 Baseline Execution | @mezo04 | 🟡 Medium | #3, #4 |

**Start Date:** Immediately  
**Target Completion:** When all 5 issues are closed

---

### Phase 2: Design 🟣
**Goal:** Mathematical formulation and architectural planning

| Issue | Assignee | Priority | Dependencies |
|-------|----------|----------|--------------|
| #13 Self-Attention Math (Q/K/V) | @Abdallah4Z | 🔴 High | #1, #2 |
| #14 Attention Placement Strategy | @ammarlhassan | 🔴 High | #13, #2 |
| #15 Discriminator Layer Audit | @fourarms4x4 | 🔴 High | #2 |
| #16 Apply Spectral Normalization & Tune LR | @ahmedislamfarouk | 🔴 High | #15 |

**Start Date:** After Phase 1 completion  
**Target Completion:** When all 4 issues are closed

---

### Phase 3: Implementation 🟢
**Goal:** Code the new modules and integrate them

| Issue | Assignees | Priority | Dependencies |
|-------|-----------|----------|--------------|
| #17 SelfAttention Module | @mezo04 | 🔴 High | #13, #14 |
| #18 Inject Self-Attention into G_Net | @Abdallah4Z, @ammarlhassan | 🔴 High | #17, #14 |
| #19 Update Discriminators w/ Spectral Norm | @fourarms4x4, @ahmedislamfarouk | 🔴 High | #15, #16 |

**Start Date:** After Phase 2 completion  
**Note:** #18 and #19 can be worked on in parallel after #17 is done  
**Target Completion:** When all 3 issues are closed

---

### Phase 4: Training 🟡
**Goal:** Update training loop, add logging, run full training

| Issue | Assignees | Priority | Dependencies |
|-------|-----------|----------|--------------|
| #20 Refactor Training Loop | @Abdallah4Z, @ammarlhassan | 🔴 High | #18, #19 |
| #21 Loss Logging & Monitoring | @mezo04 | 🟡 Medium | #20 |
| #10 Hyperparameter Sweep | @Abdallah4Z, @fourarms4x4 | 🔴 High | #20, #21 |

**Start Date:** After Phase 3 completion  
**Target Completion:** When all 3 issues are closed

---

### Phase 5: Evaluation 🔴
**Goal:** Visual and statistical comparison for paper

| Issue | Assignees | Priority | Dependencies |
|-------|-----------|----------|--------------|
| #11 Qualitative Analysis (Visual) | @ahmedislamfarouk, @mezo04 | 🔴 High | #10 |
| #12 Quantitative Evaluation (IS/FID) | @ammarlhassan, @Abdallah4Z | 🔴 High | #10 |

**Start Date:** After Phase 4 completion  
**Note:** #11 and #12 can be worked on in parallel  
**Target Completion:** When both issues are closed

---

## 🏷️ Label System

### Phase Labels
- `Phase 1: Foundation` 🔵
- `Phase 2: Design` 🟣
- `Phase 3: Implementation` 🟢
- `Phase 4: Training` 🟡
- `Phase 5: Evaluation` 🔴

### Priority Labels
- `priority: high` 🔴 - Critical path items
- `priority: medium` 🟡 - Important but can be parallelized
- `priority: low` 🟢 - Nice to have

### Status Labels
- `status: ready` 🟢 - Ready to start
- `status: in-progress` 🔵 - Currently being worked on
- `status: blocked` 🔴 - Blocked by dependencies

### Category Labels
- `Research and Design` - Design/architecture issues
- `Development and Training` - Implementation/training issues
- `Evaluation and Science` - Analysis/evaluation issues
- `Environment and Baselines` - Setup/baseline issues

---

## 🔄 Dependency Graph

```
Phase 1 (Foundation)
    ↓
Phase 2 (Design)
    ↓
Phase 3 (Implementation)
    ↓
Phase 4 (Training)
    ↓
Phase 5 (Evaluation)
```

### Detailed Dependencies

```
#1, #2, #3 (can start immediately)
    ↓
#4 (needs #3)
    ↓
#5 (needs #3, #4)
    ↓
#13 (needs #1, #2)
#15 (needs #2)
    ↓
#14 (needs #13, #2)
#16 (needs #15)
    ↓
#17 (needs #13, #14)
    ↓
#18 (needs #17, #14)    #19 (needs #15, #16) [parallel]
    ↓
#20 (needs #18, #19)
    ↓
#21 (needs #20)
    ↓
#10 (needs #20, #21)
    ↓
#11, #12 (need #10) [parallel]
```

---

## 📋 Execution Timeline

```
Week 1-2: Phase 1 - Foundation
    ├─ @Abdallah4Z: #1
    ├─ @ammarlhassan: #2
    ├─ @fourarms4x4: #3
    ├─ @ahmedislamfarouk: #4 (after #3)
    └─ @mezo04: #5 (after #3, #4)

Week 3-4: Phase 2 - Design
    ├─ @Abdallah4Z: #13
    ├─ @ammarlhassan: #14 (after #13)
    ├─ @fourarms4x4: #15
    └─ @ahmedislamfarouk: #16 (after #15)

Week 5-6: Phase 3 - Implementation
    ├─ @mezo04: #17
    ├─ @Abdallah4Z + @ammarlhassan: #18 (after #17)
    └─ @fourarms4x4 + @ahmedislamfarouk: #19

Week 7-8: Phase 4 - Training
    ├─ @Abdallah4Z + @ammarlhassan: #20
    ├─ @mezo04: #21 (after #20)
    └─ @Abdallah4Z + @fourarms4x4: #10 (after #20, #21)

Week 9-10: Phase 5 - Evaluation
    ├─ @ahmedislamfarouk + @mezo04: #11
    └─ @ammarlhassan + @Abdallah4Z: #12
```

---

## 🎯 Getting Started

### For Everyone:
1. **Check your assigned issues** in the table above
2. **Review the issue details** on GitHub
3. **Update status** when you start working:
   - Change `status: ready` → `status: in-progress`
   - Add comments with your progress
4. **Create PRs** linking to the issue number
5. **Close the issue** when the acceptance criteria are met

### Best Practices:
- ✅ Comment on issues with daily/weekly progress updates
- ✅ Mention blockers early using `status: blocked` label
- ✅ Link PRs to issues using `Fixes #X` or `Related to #X`
- ✅ Review teammates's PRs when assigned to the same issue
- ✅ Ask questions in issue comments for visibility

---

## 📊 Progress Tracking

### Current Status: 🔵 Phase 1 - Ready to Start

| Phase | Status | Issues Complete | Total Issues |
|-------|--------|----------------|--------------|
| Phase 1: Foundation | 🔵 Ready | 0/5 | 5 |
| Phase 2: Design | ⏳ Pending | 0/4 | 4 |
| Phase 3: Implementation | ⏳ Pending | 0/3 | 3 |
| Phase 4: Training | ⏳ Pending | 0/3 | 3 |
| Phase 5: Evaluation | ⏳ Pending | 0/2 | 2 |

**Overall Progress:** 0/17 issues complete (0%)

---

## 🔗 Useful Links

- **Repository:** https://github.com/Abdallah4Z/AttenX
- **All Issues:** https://github.com/Abdallah4Z/AttenX/issues
- **Reorganization Script:** `scripts/organize_issues.py`
- **Detailed Plan:** `ISSUE_RESTRUCTURING_PLAN.md`
- **Dependency Graph:** `DEPENDENCY_GRAPH.md`
- **Restructuring Summary:** `RESTRUCTURING_SUMMARY.md`

---

## 📝 Notes

- Workload is balanced: 4-6 issues per person
- Critical path issues are marked as `priority: high`
- Collaborative issues (2+ assignees) should be coordinated between team members
- Phase transitions happen when ALL issues in current phase are complete
- Questions? Discuss in issue comments or team meetings

---

**Last Updated:** Auto-generated by `scripts/organize_issues.py`
