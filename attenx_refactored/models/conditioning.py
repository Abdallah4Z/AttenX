"""
Conditioning Augmentation module.

Maps a sentence embedding to a latent conditioning vector by
sampling from a learned Gaussian distribution (StackGAN/StackGAN++).
The KL divergence between this distribution and N(0,1) acts as a
regularizer during training.
"""

import torch
import torch.nn as nn


class ConditioningAugmentation(nn.Module):
    """
    Conditioning Augmentation layer.

    Takes a sentence embedding and produces a latent code c
    by sampling from N(mu, std), where mu and std are learned
    linear projections of the sentence embedding.

    Args:
        emb_dim: Dimension of the input sentence embedding.
        nz: Dimension of the output latent code.
    """

    def __init__(self, emb_dim, nz):
        super().__init__()
        self.fc_mu = nn.Linear(emb_dim, nz)
        self.fc_logvar = nn.Linear(emb_dim, nz)

    def forward(self, sent_emb):
        """
        Sample a conditioning vector from the learned distribution.

        Args:
            sent_emb: Sentence embedding (B, emb_dim).

        Returns:
            Tuple of (c, mu, logvar) where c is the sampled code,
            mu is the mean, and logvar is the log-variance.
        """
        mu = self.fc_mu(sent_emb)
        logvar = self.fc_logvar(sent_emb)
        std = torch.exp(0.5 * logvar)
        eps = torch.randn_like(std)
        c = mu + eps * std
        return c, mu, logvar
