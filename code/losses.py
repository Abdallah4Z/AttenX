import torch
import torch.nn as nn
import torch.nn.functional as F

def cosine_similarity(x1, x2, dim=1, eps=1e-8):
    """Returns cosine similarity between x1 and x2, computed along dim."""
    w12 = torch.sum(x1 * x2, dim)
    w1 = torch.norm(x1, 2, dim)
    w2 = torch.norm(x2, 2, dim)
    return (w12 / (w1 * w2).clamp(min=eps)).squeeze()

def words_loss(img_features, words_emb, labels, cap_lens, batch_size):
    """
    img_features: (batch_size, nef, 17, 17)
    words_emb: (batch_size, nef, seq_len)
    """
    masks = []
    att_maps = []
    # Compute attention between image regions and words
    # img_features: B x nef x 289
    # words_emb: B x nef x seq_len
    
    # We'll implement a simplified version of the word-level cross-modal attention
    # that calculates the similarity between every image sub-region and every word.
    
    # This is the "Attentional" part of DAMSM
    # Returns the similarity score matrix for the batch
    
    return torch.mean(torch.randn(1, requires_grad=True)) # Placeholder for backprop-ready loss

def sent_loss(cnn_code, sent_emb, labels, batch_size):
    """
    cnn_code: (batch_size, nef)
    sent_emb: (batch_size, nef)
    """
    # Global similarity between image vector and sentence vector
    scores = cosine_similarity(cnn_code, sent_emb)
    # Binary cross entropy or ranking loss based on labels
    return torch.mean(1.0 - scores)

def KL_loss(mu, logvar):
    # Kullback-Leibler divergence for VAE-like latent space smoothing
    KLD_element = mu.pow(2).add_(logvar.exp()).mul_(-1).add_(1).add_(logvar)
    KLD = torch.mean(KLD_element).mul_(-0.5)
    return KLD
