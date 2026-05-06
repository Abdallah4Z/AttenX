"""
AttenX training script.

Loads a YAML configuration (with optional CLI overrides) and
launches the full adversarial training loop.

Usage:
    python -m attenx_refactored.scripts.train --config configs/attenx_mhsa.yaml
"""

import argparse
import os

from attenx_refactored.config import AttenXConfig
from attenx_refactored.training.trainer import train


def parse_args():
    """
    Parse command-line arguments for training.

    Returns:
        Parsed argparse.Namespace with config path and optional overrides.
    """
    parser = argparse.ArgumentParser(description="AttenX - Training")
    parser.add_argument("--config", type=str, required=True, help="Path to YAML config file")
    parser.add_argument("--data-dir", type=str, help="Override dataset directory")
    parser.add_argument("--attention-mode", type=str, choices=["none", "single", "multi"],
                        help="Override attention variant")
    parser.add_argument("--use-sn", action="store_true", help="Enable spectral normalization")
    parser.add_argument("--num-heads", type=int, help="Override attention heads")
    parser.add_argument("--epochs", type=int, help="Override number of epochs")
    parser.add_argument("--batch-size", type=int, help="Override batch size")
    parser.add_argument("--gamma-damsm", type=float, help="Override DAMSM loss weight")
    parser.add_argument("--lr-g", type=float, help="Override generator learning rate")
    parser.add_argument("--lr-d", type=float, help="Override discriminator learning rate")
    parser.add_argument("--seed", type=int, help="Override random seed")
    parser.add_argument("--device", type=str, help="Override device (cuda/cpu)")
    parser.add_argument("--use-amp", action="store_true", help="Enable mixed precision training")
    parser.add_argument("--checkpoint-dir", type=str, help="Override checkpoint directory")
    parser.add_argument("--resume", action="store_true", help="Resume from latest checkpoint")
    parser.add_argument("--damsm-text-path", type=str, help="Path to pretrained text encoder")
    parser.add_argument("--damsm-image-path", type=str, help="Path to pretrained image encoder")
    return parser.parse_args()


def main():
    """
    Main entry point: parse args, load config, and launch training.
    """
    args = parse_args()
    cfg = AttenXConfig.from_yaml(args.config)

    # Apply CLI overrides on top of YAML config
    overrides = {k: v for k, v in vars(args).items() if v is not None and k != "config"}
    for key, val in overrides.items():
        if hasattr(cfg, key):
            setattr(cfg, key, val)

    os.makedirs(cfg.checkpoint_dir, exist_ok=True)
    cfg.save(os.path.join(cfg.checkpoint_dir, "config.yaml"))

    train(cfg)


if __name__ == "__main__":
    main()
