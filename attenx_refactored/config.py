"""
AttenX configuration module.

Defines the central AttenXConfig dataclass that holds all
hyperparameters and settings for training, model architecture,
optimization, and checkpointing. Supports loading from YAML
and saving back to YAML.
"""

from dataclasses import dataclass, fields
from typing import Optional

import yaml


@dataclass
class AttenXConfig:
    """
    Central configuration for AttenX training and model setup.

    Attributes:
        data_dir: Path to dataset directory.
        image_size: Target image resolution (default 256).
        num_workers: DataLoader worker count.

        attention_mode: One of 'none', 'single', 'multi'.
        use_sn: Whether to apply spectral normalization to discriminators.
        num_heads: Number of attention heads for multi-head mode.
        ngf: Generator feature channel count.
        ndf: Discriminator feature channel count.
        nef: Encoder feature dimension (DAMSM).
        nhidden: Text encoder hidden dimension.
        nembed: Text encoder embedding dimension.
        vocab_size: Vocabulary size for text encoding.

        epochs: Total training epochs.
        batch_size: Samples per batch.
        lr_g: Generator learning rate.
        lr_d: Discriminator learning rate.
        D_steps: Discriminator updates per generator update.
        gamma_damsm: Weight for DAMSM loss in generator.
        lambda_kl: Weight for KL divergence loss.
        seed: Random seed for reproducibility.
        device: Device string ('auto', 'cuda', 'cpu').
        use_amp: Enable automatic mixed precision training.

        lr_patience: LR scheduler patience (epochs).
        lr_factor: LR decay factor.

        validate_interval: Run validation every N epochs.
        patience_early_stop: Early stopping patience.
        min_delta: Minimum improvement threshold for early stopping.

        checkpoint_dir: Directory to save model checkpoints.
        log_dir: Directory to save training logs.
        resume: Resume training from latest checkpoint.

        damsm_text_path: Path to pretrained text encoder weights.
        damsm_image_path: Path to pretrained image encoder weights.
    """

    data_dir: str = "./data"
    image_size: int = 256
    num_workers: int = 4

    attention_mode: str = "none"
    use_sn: bool = False
    num_heads: int = 4
    ngf: int = 64
    ndf: int = 64
    nef: int = 512
    nhidden: int = 256
    nembed: int = 256
    vocab_size: int = 10000

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

    lr_patience: int = 5
    lr_factor: float = 0.5

    validate_interval: int = 1
    patience_early_stop: int = 10
    min_delta: float = 1e-4

    checkpoint_dir: str = "./checkpoints"
    log_dir: str = "./logs/training"
    resume: bool = False

    damsm_text_path: Optional[str] = None
    damsm_image_path: Optional[str] = None

    @classmethod
    def from_yaml(cls, path):
        """
        Load configuration from a YAML file.

        Args:
            path: Path to the YAML config file.

        Returns:
            An AttenXConfig instance populated with YAML values.
        """
        with open(path) as f:
            d = yaml.safe_load(f)
        return cls(**{k: v for k, v in d.items() if hasattr(cls, k)})

    def save(self, path):
        """
        Save configuration to a YAML file.

        Args:
            path: Destination path for the YAML file.
        """
        with open(path, "w") as f:
            yaml.dump(self.to_dict(), f, default_flow_style=False)

    def to_dict(self):
        """Convert the config dataclass fields into a plain dictionary."""
        return {f.name: getattr(self, f.name) for f in fields(self)}
