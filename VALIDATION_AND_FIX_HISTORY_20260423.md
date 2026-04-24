# AttenX Recheck: Validation and Fix History

Date: 2026-04-23
Repo: AttenX_recheck_20260423

## 1) Requirement Source Used
Assignment text extracted and normalized in:
- requirements_extracted.txt

Extracted tasks:
1. Choose one listed text-to-image model, analyze model, state drawbacks.
2. Design enhanced model and implement it.
3. Test model to create different images.
4. Write scientific paper supported with results.

## 2) Validation Performed

### 2.1 Repository health checks
- scripts/verify_env.py: PASS
- python -m compileall -q code scripts train.py: PASS
- pytest -q: PASS

### 2.2 Artifact checks
- results/attenx contains generated image files.
- results/baseline contains baseline image files.
- output/evaluation contains quantitative_metrics.csv and ablation_results.csv.
- output/analysis contains qualitative_report.txt.
- Research/AttenX_Scientific_Paper.tex exists and references result images that resolve to existing files.

### 2.3 Initial issue discovered
Observed generated images appeared like random noise.

Root-cause analysis from code:
- scripts/baseline_inference.py used torch.randn(...) to create dummy images.
- scripts/generate.py previously allowed running without pretrained DAMSM text encoder (random text embeddings warning).
- scripts/generate.py used simplistic character-hash tokenizer stub, not guaranteed aligned with training vocabulary.
- permissive checkpoint loading (strict=False) could hide key mismatches.
- quantitative/qualitative scripts include mock/dummy logic comments.

## 3) Fixes Applied (Code)

### 3.1 scripts/generate.py
- Added strict generation guardrails:
  - requires valid generator checkpoint.
  - requires valid DAMSM text encoder checkpoint.
  - requires vocabulary mapping source (wordtoix path).
- Added robust wordtoix loader from metadata pickle.
- Replaced stub text hashing with vocabulary-based tokenization.
- Added controlled partial-load behavior via --allow-partial-load flag.
- Improved output conversion from tanh domain [-1, 1] to image uint8.
- Added fail-fast, actionable error messages.
- Added repo-root path injection to avoid import shadowing issue (`code` package import failure).

### 3.2 scripts/baseline_inference.py
- Removed random dummy-image baseline generation.
- Replaced with real inference path that delegates to strict generator pipeline.
- Added CLI requiring real checkpoint + DAMSM text encoder for baseline generation.

## 4) Post-fix Verification
- py_compile scripts/generate.py scripts/baseline_inference.py: PASS
- scripts/generate.py --help: PASS
- scripts/baseline_inference.py --help: PASS
- strict guardrail run without checkpoints: FAIL as intended with explicit checkpoint missing error.

## 5) Current Blockers (As of this report)

### 5.1 Missing data/checkpoints
Not present:
- data/CUB_200_2011
- data/birds
- data/train/*.pickle
- data/test/*.pickle
- checkpoints/checkpoint_latest.pth
- checkpoints/damsm/damsm_latest.pth

### 5.2 Dataset setup blocked by storage
Attempted Windows-native fallback for dataset download (curl + tar + gdown path).
Result:
- Drive D has Free=0 bytes.
- CUB archive download failed with curl write error due to no free disk space.

## 6) Recommended Next Steps (once storage is available)
1. Free significant disk space on D: (dataset + checkpoints + temporary archives).
2. Download/prepare dataset:
   - images: CUB_200_2011
   - metadata: birds_text.zip / birds/
   - run scripts/preprocess_cub_metadata.py ./data
3. Pretrain DAMSM (short bootstrap):
   - python scripts/pretrain_damsm.py --data-dir ./data --epochs 1 --batch-size 2 --num-workers 0 --output-dir ./checkpoints/damsm
4. Train generator to produce checkpoint:
   - python train.py --data-dir ./data --epochs <N> --batch-size <B> --checkpoint-dir ./checkpoints
5. Generate images with strict script:
   - python scripts/generate.py --checkpoint ./checkpoints/checkpoint_latest.pth --damsm-text-path ./checkpoints/damsm/damsm_latest.pth --wordtoix-path ./data/birds/captions.pickle

## 7) Status Summary
- Validation work: completed.
- Noise root-cause identification: completed.
- Code fixes for guardrails and baseline realism: completed.
- End-to-end high-quality image regeneration: blocked by missing data/checkpoints and zero free disk space.

## 8) GPU Runtime Hardening Update (2026-04-23)

### 8.1 Code updates for GPU execution
- train.py:
   - Added explicit `--device` argument (`auto`, `cpu`, `cuda`, `cuda:0`, ...).
   - Added strict CUDA-request validation with clear error message when CUDA is unavailable.
   - Added optimizer-state migration to active device during resume so checkpoint resume works on GPU.
   - Switched DataLoader pin_memory to be device-aware (`True` on CUDA, else `False`).
   - Updated cuDNN settings for faster CUDA training by default.
- scripts/pretrain_damsm.py:
   - Added explicit `--device` argument and strict CUDA-request validation.
   - Switched DataLoader pin_memory to be device-aware.
   - Updated cuDNN settings for faster CUDA training by default.
- scripts/generate.py:
   - Added strict device resolver for `--device` with fail-fast CUDA validation.
   - Added explicit startup print of selected device.

### 8.2 Validation results
- Syntax checks for updated files: PASS.
- Current configured Python environment reports `torch 2.11.0+cpu` and `cuda_available=False`.
- Conclusion: code paths are now GPU-ready, but the active Python environment still needs CUDA-enabled PyTorch to actually run on GPU.
