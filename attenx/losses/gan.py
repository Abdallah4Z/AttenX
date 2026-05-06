import torch
import torch.nn as nn


def build_gan_criterion(device):
    """Binary cross-entropy for adversarial loss."""
    return nn.BCELoss()
