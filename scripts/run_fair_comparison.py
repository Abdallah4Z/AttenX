"""Run a fair three-way comparison on GPU 0 (or selected GPU).

Compares:
- baseline: AttnGAN-like (no generator self-attention, no discriminator spectral norm)
- attenx_sa: AttenX with single-head self-attention + spectral norm
- attenx_mhsa: AttenX with multi-head self-attention + spectral norm

All runs share identical training settings except the variant-specific knobs.
"""

import argparse
import os
import subprocess
import sys
from pathlib import Path


def _build_cmd(args, variant, ckpt_dir, log_dir, use_damsm):
    cmd = [
        sys.executable,
        "train.py",
        "--data-dir",
        args.data_dir,
        "--epochs",
        str(args.epochs),
        "--batch-size",
        str(args.batch_size),
        "--num-workers",
        str(args.num_workers),
        "--gpu-ids",
        "0",
        "--variant",
        variant,
        "--validate-interval",
        str(args.validate_interval),
        "--checkpoint-dir",
        str(ckpt_dir),
        "--log-dir",
        str(log_dir),
        "--seed",
        str(args.seed),
        "--checkpoint-interval",
        str(args.checkpoint_interval),
    ]

    if args.use_amp:
        cmd.append("--use-amp")

    if variant == "attenx_mhsa":
        cmd.extend(["--mhsa-heads", str(args.mhsa_heads)])

    if use_damsm:
        cmd.extend(["--damsm-text-path", args.damsm_text_path])
        cmd.extend(["--damsm-image-path", args.damsm_image_path])
    else:
        cmd.extend(["--gamma-damsm", "0.0"])

    return cmd


def main():
    parser = argparse.ArgumentParser(description="Fair 3-way AttenX comparison runner")
    parser.add_argument("--data-dir", type=str, default="./data")
    parser.add_argument("--epochs", type=int, default=50)
    parser.add_argument("--batch-size", type=int, default=4)
    parser.add_argument("--num-workers", type=int, default=4)
    parser.add_argument("--validate-interval", type=int, default=1)
    parser.add_argument("--checkpoint-interval", type=int, default=10)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--gpu-id", type=int, default=0)
    parser.add_argument("--base-output", type=str, default="./results/fair_comparison")
    parser.add_argument("--use-amp", action="store_true")
    parser.add_argument("--mhsa-heads", type=int, default=4)
    parser.add_argument("--damsm-text-path", type=str, default="./checkpoints/damsm/damsm_latest.pth")
    parser.add_argument("--damsm-image-path", type=str, default="./checkpoints/damsm/damsm_latest.pth")
    args = parser.parse_args()

    repo_root = Path(__file__).resolve().parent.parent
    base_output = Path(args.base_output)
    base_output.mkdir(parents=True, exist_ok=True)

    use_damsm = Path(args.damsm_text_path).is_file() and Path(args.damsm_image_path).is_file()
    if not use_damsm:
        print("DAMSM checkpoints not found. Running all variants with --gamma-damsm 0.0 for fairness.")

    variants = ["baseline", "attenx_sa", "attenx_mhsa"]
    env = os.environ.copy()
    env["CUDA_VISIBLE_DEVICES"] = str(args.gpu_id)

    commands_log = []

    for variant in variants:
        ckpt_dir = base_output / variant / "checkpoints"
        log_dir = base_output / variant / "logs"
        ckpt_dir.mkdir(parents=True, exist_ok=True)
        log_dir.mkdir(parents=True, exist_ok=True)

        cmd = _build_cmd(args, variant, ckpt_dir, log_dir, use_damsm)
        commands_log.append(" ".join(cmd))

        print(f"\n=== Running variant: {variant} ===")
        print("Command:", " ".join(cmd))
        subprocess.run(cmd, cwd=repo_root, env=env, check=True)

    commands_path = base_output / "commands_used.txt"
    commands_path.write_text("\n".join(commands_log), encoding="utf-8")
    print(f"\nCompleted all variants. Commands saved to: {commands_path}")


if __name__ == "__main__":
    main()
