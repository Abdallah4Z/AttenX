import torch
import torch.nn as nn


class ConditioningAugmentation(nn.Module):
    """Conditioning Augmentation (StackGAN/StackGAN++).

    Maps a sentence embedding to a conditioning vector by
    sampling from a learned Gaussian distribution. The KL
    divergence between the learned distribution and N(0,1)
    is added as a regularizer during training.
    """

    def __init__(self, emb_dim, nz):
        super().__init__()
        self.fc_mu = nn.Linear(emb_dim, nz)
        self.fc_logvar = nn.Linear(emb_dim, nz)

    def forward(self, sent_emb):
        mu = self.fc_mu(sent_emb)
        logvar = self.fc_logvar(sent_emb)
        std = torch.exp(0.5 * logvar)
        eps = torch.randn_like(std)
        c = mu + eps * std
        return c, mu, logvar
