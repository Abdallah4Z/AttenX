from dataclasses import dataclass, fields
from typing import Optional

import yaml


@dataclass
class AttenXConfig:
    # Dataset
    data_dir: str = "./data"
    image_size: int = 256
    num_workers: int = 4

    # Model — attention variant
    attention_mode: str = "none"
    num_heads: int = 4
    ngf: int = 64
    ndf: int = 64
    nef: int = 512
    nhidden: int = 256
    nembed: int = 256
    vocab_size: int = 10000

    # Training
    epochs: int = 50
    batch_size: int = 4
    lr_g: float = 1e-4
    lr_d: float = 4e-4
    D_steps: int = 1
    gamma_damsm: float = 0.0
    lambda_kl: float = 2.0
    seed: int = 42
    device: str = "auto"
    use_amp: bool = False

    # LR scheduler
    lr_patience: int = 5
    lr_factor: float = 0.5

    # Validation / early stopping
    validate_interval: int = 1
    patience_early_stop: int = 10
    min_delta: float = 1e-4

    # Checkpoint / logging
    checkpoint_dir: str = "./checkpoints"
    log_dir: str = "./logs/training"
    resume: bool = False

    # Pretrained DAMSM encoders
    damsm_text_path: Optional[str] = None
    damsm_image_path: Optional[str] = None

    @classmethod
    def from_yaml(cls, path):
        with open(path) as f:
            d = yaml.safe_load(f)
        return cls(**{k: v for k, v in d.items() if hasattr(cls, k)})

    def save(self, path):
        with open(path, "w") as f:
            yaml.dump(self.to_dict(), f, default_flow_style=False)

    def to_dict(self):
        return {f.name: getattr(self, f.name) for f in fields(self)}
