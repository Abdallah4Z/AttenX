"""Configuration loader for AttenX training."""
import argparse
import os
from dataclasses import dataclass, fields
from typing import Optional, Any
import yaml


@dataclass
class TrainingConfig:
    """Training configuration dataclass."""
    # Dataset
    data_dir: str = "./data"
    image_size: int = 256
    num_workers: int = 4
    
    # Model Architecture
    ngf: int = 64  # Generator base channel count
    ndf: int = 64  # Discriminator base channel count
    nef: int = 512  # Text/Image encoder feature dim
    nhidden: int = 256  # RNN hidden size
    nembed: int = 256  # Word embedding dim
    vocab_size: int = 10000  # Vocabulary size
    
    # Training Hyperparameters
    epochs: int = 100
    batch_size: int = 4
    lr_g: float = 1e-4
    lr_d: float = 4e-4
    D_steps: int = 1
    gamma_damsm: float = 5.0
    lambda_kl: float = 2.0
    clip_grad: Optional[float] = None
    seed: int = 42
    use_amp: bool = False
    gpu_ids: str = ""
    
    # Learning Rate Scheduling
    lr_patience: int = 5
    lr_factor: float = 0.5
    
    # Validation & Early Stopping
    validate_interval: int = 1
    patience_early_stop: int = 10
    min_delta: float = 1e-4
    
    # Checkpointing & Logging
    checkpoint_dir: str = "./checkpoints"
    checkpoint_interval: int = 10
    log_dir: str = "./logs/training"
    resume: bool = False
    
    # Pretrained DAMSM Encoders (optional)
    damsm_text_path: Optional[str] = None
    damsm_image_path: Optional[str] = None
    
    @classmethod
    def from_yaml(cls, yaml_path: str) -> "TrainingConfig":
        """Load configuration from YAML file."""
        with open(yaml_path, "r") as f:
            config_dict = yaml.safe_load(f)
        return cls(**config_dict)
    
    @classmethod
    def from_args(cls, args: argparse.Namespace) -> "TrainingConfig":
        """Create configuration from argparse namespace."""
        config_dict = {}
        for field in fields(cls):
            if hasattr(args, field.name):
                config_dict[field.name] = getattr(args, field.name)
        return cls(**config_dict)
    
    def to_dict(self) -> dict:
        """Convert configuration to dictionary."""
        return {field.name: getattr(self, field.name) for field in fields(self)}
    
    def save_yaml(self, yaml_path: str):
        """Save configuration to YAML file."""
        os.makedirs(os.path.dirname(yaml_path), exist_ok=True)
        with open(yaml_path, "w") as f:
            yaml.dump(self.to_dict(), f, default_flow_style=False)


def create_arg_parser() -> argparse.ArgumentParser:
    """Create argument parser for backward compatibility."""
    parser = argparse.ArgumentParser(description="Multi-epoch AttenX training pipeline")
    
    # Dataset args
    parser.add_argument("--data-dir", type=str, default="./data", help="Dataset root directory")
    parser.add_argument("--image-size", type=int, default=256, help="Image size for training")
    parser.add_argument("--num-workers", type=int, default=4, help="DataLoader workers")
    
    # Model args
    parser.add_argument("--ngf", type=int, default=64, help="Generator base channel count")
    parser.add_argument("--ndf", type=int, default=64, help="Discriminator base channel count")
    parser.add_argument("--nef", type=int, default=512, help="Text/Image encoder feature dim")
    parser.add_argument("--nhidden", type=int, default=256, help="RNN hidden size")
    parser.add_argument("--nembed", type=int, default=256, help="Word embedding dim")
    parser.add_argument("--vocab-size", type=int, default=10000, help="Vocabulary size")
    
    # Training args
    parser.add_argument("--epochs", type=int, default=100, help="Number of training epochs")
    parser.add_argument("--batch-size", type=int, default=4, help="Batch size")
    parser.add_argument("--lr-g", type=float, default=1e-4, help="Generator learning rate")
    parser.add_argument("--lr-d", type=float, default=4e-4, help="Discriminator learning rate")
    parser.add_argument("--D-steps", type=int, default=1, help="Discriminator steps per generator step")
    parser.add_argument("--gamma-damsm", type=float, default=1.0, help="DAMSM loss weight")
    parser.add_argument("--clip-grad", type=float, default=None, help="Gradient clipping value")
    parser.add_argument("--seed", type=int, default=42, help="Random seed for reproducibility")
    parser.add_argument("--use-amp", action="store_true", help="Use automatic mixed precision")
    parser.add_argument("--gpu-ids", type=str, default="", help="Comma-separated CUDA GPU IDs to use, e.g. '0,1'")
    
    # LR Scheduler args
    parser.add_argument("--lr-patience", type=int, default=5, help="Patience for LR scheduler")
    parser.add_argument("--lr-factor", type=float, default=0.5, help="Factor to reduce LR by on plateau")
    
    # Validation & Early Stopping args
    parser.add_argument("--validate-interval", type=int, default=1, help="Run validation every N epochs")
    parser.add_argument("--patience-early-stop", type=int, default=10, help="Patience for early stopping")
    parser.add_argument("--min-delta", type=float, default=1e-4, help="Minimum improvement to count as progress")
    
    # Checkpoint/logging args
    parser.add_argument("--checkpoint-dir", type=str, default="./checkpoints", help="Checkpoint directory")
    parser.add_argument("--checkpoint-interval", type=int, default=10, help="Save checkpoint every N epochs")
    parser.add_argument("--log-dir", type=str, default="./logs/training", help="Log directory")
    parser.add_argument("--resume", action="store_true", help="Resume from latest checkpoint")
    
    # Pretrained DAMSM encoder paths
    parser.add_argument("--damsm-text-path", type=str, default=os.getenv("DAMSM_TEXT_ENCODER_PATH"), help="Pretrained DAMSM text encoder checkpoint")
    parser.add_argument("--damsm-image-path", type=str, default=os.getenv("DAMSM_IMAGE_ENCODER_PATH"), help="Pretrained DAMSM image encoder checkpoint")
    
    # Config file option
    parser.add_argument("--config", type=str, default=None, help="Path to YAML configuration file")
    
    return parser


def load_config() -> TrainingConfig:
    """Load configuration from command line args or YAML file."""
    parser = create_arg_parser()
    args = parser.parse_args()
    
    if args.config:
        # Load from YAML file, override with command line args
        config = TrainingConfig.from_yaml(args.config)
        # Override with command line arguments
        for field in fields(config):
            arg_value = getattr(args, field.name.replace("_", "-"))
            if arg_value != parser.get_default(field.name.replace("_", "-")):
                setattr(config, field.name, arg_value)
    else:
        # Load from command line args only
        config = TrainingConfig.from_args(args)
    
    return config