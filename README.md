# AttenX

Enhanced attention-based GAN (AttnGAN-style) for text-to-image synthesis, with self-attention integration and discriminator spectral normalization experiments.

## Repository layout

- `code/` core model components (`generator`, `discriminator`, `encoder`, `losses`, `datasets`)
- `scripts/` utilities for generation, evaluation, issue/project management, and environment checks
- `docs/architecture/` architecture/math notes
- `docs/specifications/` implementation audits/specs
- `docs/planning/` project planning and phase breakdown
- `train.py` single-step training/gradient sanity script

## Current implementation status

- `G_NET` implemented with staged upsampling to `256x256` and a `SelfAttention` block
- `D_NET256` implemented with spectral normalization over convolution layers
- DAMSM-like loss utilities present in `code/losses.py`
- Dataset loader scaffold available for CUB-style data in `code/datasets.py`

> Note: `train.py` is currently a mathematical/gradient sanity pass, not a full multi-epoch production trainer.

## Setup

Use Python 3.10+.

```bash
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

Or with PEP 621 metadata:

```bash
pip install .
```

## Quick checks

```bash
python3 scripts/verify_env.py
python3 -m compileall -q code scripts train.py
```

## Run examples

```bash
python3 train.py
python3 scripts/generate.py
python3 scripts/quantitative_evaluation.py
python3 scripts/qualitative_analysis.py
```

## Documentation index

- `docs/planning/PROJECT_PLAN.md`
- `docs/planning/TEAM_ASSIGNMENTS.md`
- `docs/planning/ISSUE_RESTRUCTURING_PLAN.md`
- `docs/planning/DEPENDENCY_GRAPH.md`
- `docs/architecture/self_attention_math.md`
- `docs/specifications/discriminator_audit.md`

## Notes

- GitHub issue helper scripts require `PyGitHub` and a personal access token.
- Generated artifacts/logs are written under `results/`, `output/`, and `logs/`.
