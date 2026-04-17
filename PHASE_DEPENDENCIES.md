# AttenX Phase Dependencies & Execution Order

## ⚠️ CRITICAL: Phase Execution Order

**You MUST complete phases in this exact order:**

```
Phase 1 🔵 → Phase 2 🟣 → Phase 3 🟢 → Phase 4 🟡 → Phase 5 🔴
```

**Why?** Each phase builds on the work from the previous phase. You cannot skip ahead!

---

## Phase 1: Foundation 🔵 (START HERE)

**What:** Understand the architecture, setup environment, establish baselines  
**Must Finish Before:** Phase 2 can begin  
**Parallel Work:** All 5 issues can be done simultaneously

| Issue | Assignees | Priority | Blocks |
|-------|-----------|----------|--------|
| **#1** DAMSM Deep Dive | @Abdallah4Z, @ammarlhassan | 🔴 High | #13, #15 |
| **#2** Multi-Stage Generator | @fourarms4x4, @ahmedislamfarouk | 🔴 High | #13, #14, #15 |
| **#3** Environment Setup | @mezo04, @Abdallah4Z | 🔴 High | #4, #5 |
| **#4** Dataset Preparation | @ammarlhassan, @fourarms4x4 | 🟡 Medium | #5 |
| **#5** Baseline Execution | @ahmedislamfarouk, @mezo04 | 🟡 Medium | Phase 2 |

### Phase 1 Dependencies:
```
#3 must finish first → enables #4
#4 must finish → enables #5
#1 and #2 can start immediately (no dependencies)
```

### Phase 1 Completion Criteria:
- ✅ All 5 issues closed
- ✅ Environment working (can run PyTorch on GPU)
- ✅ Dataset downloaded and preprocessed
- ✅ Baseline images generated from 10 standard prompts
- ✅ Technical documentation for DAMSM and Multi-Stage Generator complete

---

## Phase 2: Design 🟣 (AFTER Phase 1)

**What:** Mathematical formulation and architectural planning for self-attention and spectral normalization  
**Must Finish Before:** Phase 3 can begin  
**Prerequisites:** ALL Phase 1 issues must be closed

| Issue | Assignees | Priority | Needs From Phase 1 | Blocks |
|-------|-----------|----------|-------------------|--------|
| **#13** Self-Attention Math (Q/K/V) | @Abdallah4Z, @ammarlhassan, @fourarms4x4 | 🔴 High | #1, #2 | #14, #17 |
| **#14** Attention Placement Strategy | @ahmedislamfarouk, @mezo04 | 🔴 High | #2, #13 | #17, #18 |
| **#15** Discriminator Layer Audit | @Abdallah4Z, @fourarms4x4, @ahmedislamfarouk | 🔴 High | #2 | #16, #19 |
| **#16** Apply Spectral Norm & Tune LR | @ammarlhassan, @mezo04 | 🔴 High | #15 | #19 |

### Phase 2 Dependencies:
```
#13 needs #1 and #2 from Phase 1
#15 needs #2 from Phase 1

After Phase 1 is done:
  - #13 and #15 can start immediately
  - #14 waits for #13
  - #16 waits for #15
```

### Phase 2 Completion Criteria:
- ✅ Mathematical specification for Q/K/V transformations complete
- ✅ Architectural diagram showing self-attention placement
- ✅ Layer-by-layer discriminator documentation
- ✅ Spectral normalization implementation plan with LR recommendations
- ✅ All 4 issues closed

---

## Phase 3: Implementation 🟢 (AFTER Phase 2)

**What:** Code the self-attention module, inject into generator, update discriminators  
**Must Finish Before:** Phase 4 can begin  
**Prerequisites:** ALL Phase 2 issues must be closed

| Issue | Assignees | Priority | Needs From Phase 2 | Blocks |
|-------|-----------|----------|-------------------|--------|
| **#17** SelfAttention Module | @mezo04, @Abdallah4Z, @ammarlhassan | 🔴 High | #13, #14 | #18 |
| **#18** Inject Self-Attention into G_Net | @fourarms4x4, @ahmedislamfarouk, @Abdallah4Z | 🔴 High | #17, #14 | #20 |
| **#19** Update Discriminators w/ Spectral Norm | @ammarlhassan, @mezo04, @fourarms4x4 | 🔴 High | #15, #16 | #20 |

### Phase 3 Dependencies:
```
#17 needs #13 and #14 from Phase 2

After #17 is done:
  - #18 and #19 can run IN PARALLEL
  
#18 needs #17 (SelfAttention module must exist)
#19 needs #15 and #16 from Phase 2 (can start earlier if #15, #16 done)
```

### Phase 3 Completion Criteria:
- ✅ `SelfAttention` class implemented in `modules.py` with unit tests
- ✅ Generator (`G_Net`) updated with self-attention injection
- ✅ All 3 discriminators use spectral normalization
- ✅ Forward passes work without errors (dummy input test)
- ✅ All 3 issues closed

---

## Phase 4: Training 🟡 (AFTER Phase 3)

**What:** Update training loop, add loss logging, run full training with hyperparameter sweep  
**Must Finish Before:** Phase 5 can begin  
**Prerequisites:** ALL Phase 3 issues must be closed

| Issue | Assignees | Priority | Needs From Phase 3 | Blocks |
|-------|-----------|----------|-------------------|--------|
| **#20** Refactor Training Loop | @Abdallah4Z, @ammarlhassan, @mezo04 | 🔴 High | #18, #19 | #21, #10 |
| **#21** Loss Logging & Monitoring | @fourarms4x4, @ahmedislamfarouk | 🟡 Medium | #20 | #10 |
| **#10** Hyperparameter Sweep | @Abdallah4Z, @ahmedislamfarouk, @ammarlhassan | 🔴 High | #20, #21 | Phase 5 |

### Phase 4 Dependencies:
```
#20 needs #18 and #19 from Phase 3

After #20 is done:
  - #21 can start
  
#10 needs BOTH #20 and #21 to be complete
```

### Phase 4 Completion Criteria:
- ✅ Training loop handles self-attention outputs correctly
- ✅ DAMSM loss computes with new feature dimensions
- ✅ All loss components logged (L_G, L_D, L_Attn, L_DAMSM)
- ✅ Model trained for ~600 epochs to convergence
- ✅ Optimal checkpoint identified via Inception Score
- ✅ All 3 issues closed

---

## Phase 5: Evaluation 🔴 (AFTER Phase 4)

**What:** Visual and statistical comparison between baseline and enhanced model  
**Prerequisites:** ALL Phase 4 issues must be closed  
**This is the FINAL phase**

| Issue | Assignees | Priority | Needs From Phase 4 | 
|-------|-----------|----------|-------------------|
| **#11** Qualitative Analysis (Visual) | @mezo04, @ahmedislamfarouk, @ammarlhassan | 🔴 High | #10 |
| **#12** Quantitative Evaluation (IS/FID) | @Abdallah4Z, @fourarms4x4, @ahmedislamfarouk | 🔴 High | #10 |

### Phase 5 Dependencies:
```
#11 and #12 BOTH need #10 from Phase 4

#11 and #12 can run IN PARALLEL
```

### Phase 5 Completion Criteria:
- ✅ Side-by-side comparison grids (Original vs. Enhanced)
- ✅ Visual improvements documented (e.g., "Correct number of limbs")
- ✅ Inception Score (IS) calculated
- ✅ Fréchet Inception Distance (FID) calculated
- ✅ "Table 1" for paper with all metrics
- ✅ Both issues closed

---

## Visual Dependency Flow

```
PHASE 1: FOUNDATION
┌──────────────────────────────────────────────────┐
│  #1 ──┐                                          │
│  #2 ──┼──→ Can run in parallel                   │
│  #3 ──┤                                          │
│  #4 ←─┤  (#4 waits for #3)                      │
│  #5 ←─┘  (#5 waits for #3, #4)                  │
└──────────────────────────────────────────────────┘
                    ↓ (ALL must complete)
PHASE 2: DESIGN
┌──────────────────────────────────────────────────┐
│  #13 ← (#1, #2)  ──→ #14                        │
│  #15 ← (#2)        ──→ #16                      │
│                                                  │
│  Two parallel tracks!                            │
└──────────────────────────────────────────────────┘
                    ↓ (ALL must complete)
PHASE 3: IMPLEMENTATION
┌──────────────────────────────────────────────────┐
│  #17 ← (#13, #14)                                │
│     ↓                                            │
│  #18 ← (#17)          #19 ← (#15, #16)          │
│       ↘_______________↙                          │
│          (parallel after #17)                    │
└──────────────────────────────────────────────────┘
                    ↓ (ALL must complete)
PHASE 4: TRAINING
┌──────────────────────────────────────────────────┐
│  #20 ← (#18, #19)                                │
│     ↓                                            │
│  #21 ← (#20)                                     │
│     ↓                                            │
│  #10 ← (#20, #21)                                │
└──────────────────────────────────────────────────┘
                    ↓ (ALL must complete)
PHASE 5: EVALUATION
┌──────────────────────────────────────────────────┐
│  #11 ← (#10)                                     │
│  #12 ← (#10)                                     │
│       (can run in parallel)                      │
└──────────────────────────────────────────────────┘
                    ↓
                🎉 DONE!
```

---

## Can We Skip a Phase?

**NO!** ❌

Each phase depends on deliverables from the previous phase:

| If You Skip... | What Breaks |
|----------------|-------------|
| Phase 1 | No environment, no baseline, no understanding of architecture → Phase 2 impossible |
| Phase 2 | No mathematical spec, no design → Phase 3 implementation has no blueprint |
| Phase 3 | No self-attention module, no updated generators → Phase 4 has nothing to train |
| Phase 4 | No trained model → Phase 5 has nothing to evaluate |

---

## Quick Reference: "What Blocks What?"

```
Phase 1 Issues → Phase 2 Issues:
  #1 → #13
  #2 → #13, #14, #15
  #3 → #4, #5
  #4 → #5

Phase 2 Issues → Phase 3 Issues:
  #13 → #17
  #14 → #17, #18
  #15 → #19
  #16 → #19

Phase 3 Issues → Phase 4 Issues:
  #17 → #18
  #18 → #20
  #19 → #20

Phase 4 Issues → Phase 5 Issues:
  #20 → #21, #10
  #21 → #10
  #10 → #11, #12
```

---

## Team Coordination Tips

### Within Each Phase:
- ✅ **Pair Programming:** Each issue has 2-3 assignees - work together!
- ✅ **Code Review:** If assigned to same issue, review each other's PRs
- ✅ **Divide Subtasks:** Split the issue's subtasks between assignees

### Between Phases:
- 📢 **Phase Gate:** ALL issues in current phase must close before next phase starts
- 📝 **Documentation:** Document everything in issue comments for next phase team
- 🔄 **Handoff:** Consider a quick sync when transitioning between phases

---

**TL;DR:** Complete phases in order: **1 → 2 → 3 → 4 → 5**. No skipping! Each issue shows exactly which previous issues it depends on.
