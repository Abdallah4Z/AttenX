import argparse
import os

from attenx.config import AttenXConfig
from attenx.training.trainer import train


def parse_args():
    p = argparse.ArgumentParser(description="AttenX — Training")
    p.add_argument("--config", type=str, required=True,
                   help="Path to YAML config file")
    p.add_argument("--data-dir", type=str)
    p.add_argument("--attention-mode", type=str,
                   choices=["none", "single", "multi"])
    p.add_argument("--num-heads", type=int)
    p.add_argument("--epochs", type=int)
    p.add_argument("--batch-size", type=int)
    p.add_argument("--gamma-damsm", type=float)
    p.add_argument("--lr-g", type=float)
    p.add_argument("--lr-d", type=float)
    p.add_argument("--seed", type=int)
    p.add_argument("--device", type=str)
    p.add_argument("--use-amp", action="store_true")
    p.add_argument("--checkpoint-dir", type=str)
    p.add_argument("--resume", action="store_true")
    p.add_argument("--damsm-text-path", type=str)
    p.add_argument("--damsm-image-path", type=str)
    return p.parse_args()


if __name__ == "__main__":
    args = parse_args()
    cfg = AttenXConfig.from_yaml(args.config)

    overrides = {k: v for k, v in vars(args).items()
                 if v is not None and k != "config"}
    for key, val in overrides.items():
        if hasattr(cfg, key):
            setattr(cfg, key, val)

    os.makedirs(cfg.checkpoint_dir, exist_ok=True)
    cfg.save(os.path.join(cfg.checkpoint_dir, "config.yaml"))

    train(cfg)
