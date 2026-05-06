"""
DAMSM (Deep Attentional Multimodal Similarity Model) losses.

Implements word-level and sentence-level contrastive losses
for aligning image and text embeddings, plus the KL
divergence loss for conditioning augmentation.
"""

import torch
import torch.nn.functional as F


def cosine_similarity(x1, x2, dim=1, eps=1e-8):
    """
    Compute row-wise cosine similarity between two tensors.

    Args:
        x1: First tensor of shape (N, D).
        x2: Second tensor of shape (N, D).
        dim: Dimension along which to normalize.
        eps: Small epsilon for numerical stability.

    Returns:
        Cosine similarity values of shape (N,).
    """
    n1 = torch.norm(x1, dim=dim, keepdim=True).clamp_min(eps)
    n2 = torch.norm(x2, dim=dim, keepdim=True).clamp_min(eps)
    return ((x1 / n1) * (x2 / n2)).sum(dim=dim)


def words_loss(img_features, words_emb, labels, cap_lens, batch_size):
    """
    Word-level DAMSM contrastive loss.

    For each word in a caption, attends over spatial image features
    and computes similarity between the attended vector and the word
    embedding. Uses cross-modal contrastive learning.

    Args:
        img_features: Spatial image features (B, nef, H, W).
        words_emb: Word embedding sequence (B, seq_len, w_dim).
        labels: Unused (kept for API compatibility).
        cap_lens: Lengths of each caption (B,).
        batch_size: Batch size.

    Returns:
        Scalar word-level loss.
    """
    att_nef = img_features.size(1)
    img_features = img_features.view(batch_size, att_nef, -1)

    words_emb_t = words_emb.transpose(1, 2)
    s = torch.bmm(words_emb_t, img_features)
    s_norm = F.softmax(s, dim=2)

    img_features_t = img_features.transpose(1, 2)
    c = torch.bmm(s_norm, img_features_t)

    row_sim = cosine_similarity(c, words_emb_t, dim=2)

    seq_len = words_emb_t.size(1)
    device = row_sim.device
    steps = torch.arange(seq_len, device=device).unsqueeze(0)
    cap_lens = cap_lens.to(device).unsqueeze(1)
    mask = (steps < cap_lens).float()

    row_sim = torch.where(mask.bool(), row_sim, torch.zeros_like(row_sim))
    exp_row_sim = torch.exp(row_sim) * mask
    valid = mask.sum(dim=1).clamp_min(1.0)
    loss = -torch.log(exp_row_sim.sum(dim=1) / valid + 1e-8)

    return loss.mean()


def sent_loss(cnn_code, sent_emb, labels, batch_size):
    """
    Sentence-level DAMSM contrastive loss.

    Maximizes similarity between matching image-sentence pairs
    and minimizes similarity with all incorrect pairs in the batch.

    Args:
        cnn_code: Global image embeddings (B, nef).
        sent_emb: Global sentence embeddings (B, nef).
        labels: Unused (kept for API compatibility).
        batch_size: Batch size.

    Returns:
        Scalar sentence-level loss.
    """
    scores = torch.mm(cnn_code, sent_emb.t())
    diag = torch.diag(scores)

    cost_pos = (diag - scores).clamp(min=0)
    loss_fwd = cost_pos.mean()

    cost_pos_b = (diag - scores.t()).clamp(min=0)
    loss_bwd = cost_pos_b.mean()

    return loss_fwd + loss_bwd


def KL_loss(mu, logvar):
    """
    Kullback-Leibler divergence for conditioning augmentation.

    Regularizes the learned Gaussian distribution toward N(0, 1).

    Args:
        mu: Mean vector from conditioning augmentation.
        logvar: Log-variance vector from conditioning augmentation.

    Returns:
        Scalar KL divergence loss.
    """
    kld = mu.pow(2).add_(logvar.exp()).mul_(-1).add_(1).add_(logvar)
    return torch.mean(kld).mul_(-0.5)
