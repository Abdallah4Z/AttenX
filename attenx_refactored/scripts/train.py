import argparse
import os

from attenx_refactored.config import AttenXConfig
from attenx_refactored.training.trainer import train


def parse_args():
    parser = argparse.ArgumentParser(description="AttenX - Training")
    parser.add_argument("--config", type=str, required=True, help="Path to YAML config file")
    parser.add_argument("--data-dir", type=str)
    parser.add_argument("--attention-mode", type=str, choices=["none", "single", "multi"])
    parser.add_argument("--use-sn", action="store_true")
    parser.add_argument("--num-heads", type=int)
    parser.add_argument("--epochs", type=int)
    parser.add_argument("--batch-size", type=int)
    parser.add_argument("--gamma-damsm", type=float)
    parser.add_argument("--lr-g", type=float)
    parser.add_argument("--lr-d", type=float)
    parser.add_argument("--seed", type=int)
    parser.add_argument("--device", type=str)
    parser.add_argument("--use-amp", action="store_true")
    parser.add_argument("--checkpoint-dir", type=str)
    parser.add_argument("--resume", action="store_true")
    parser.add_argument("--damsm-text-path", type=str)
    parser.add_argument("--damsm-image-path", type=str)
    return parser.parse_args()


def main():
    args = parse_args()
    cfg = AttenXConfig.from_yaml(args.config)

    overrides = {k: v for k, v in vars(args).items() if v is not None and k != "config"}
    for key, val in overrides.items():
        if hasattr(cfg, key):
            setattr(cfg, key, val)

    os.makedirs(cfg.checkpoint_dir, exist_ok=True)
    cfg.save(os.path.join(cfg.checkpoint_dir, "config.yaml"))

    train(cfg)


if __name__ == "__main__":
    main()
