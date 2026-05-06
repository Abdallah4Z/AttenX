"""
GAN adversarial loss builder.

Provides the binary cross-entropy criterion used for
discriminator and generator adversarial training.
"""

import torch.nn as nn


def build_gan_criterion(device):
    """
    Build the binary cross-entropy loss for adversarial training.

    Args:
        device: Target device (unused, kept for API compatibility).

    Returns:
        An nn.BCELoss instance.
    """
    return nn.BCELoss()
