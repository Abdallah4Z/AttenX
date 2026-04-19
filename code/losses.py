import torch
import torch.nn.functional as F


def cosine_similarity(x1, x2, dim=1, eps=1e-8):
    """Returns cosine similarity between x1 and x2, computed along dim."""
    return F.cosine_similarity(x1, x2, dim=dim, eps=eps)


def words_loss(img_features, words_emb, labels, cap_lens, batch_size):
    """
    img_features: (batch_size, nef, 17, 17) -> local image features
    words_emb: (batch_size, nef, seq_len) -> word embeddings
    """
    # nef = 512, seq_len = 18
    # Flatten spatial dimensions: (batch, 512, 289)
    att_nef = img_features.size(1)
    img_features = img_features.view(batch_size, att_nef, -1)

    # 1. Similarity matrix: (batch, seq_len, 289)
    # Transpose words_emb to (batch, seq_len, 512) for batch matrix multiplication
    words_emb_t = words_emb.transpose(1, 2)
    s = torch.bmm(words_emb_t, img_features)

    # 2. Normalize similarity (Attention Map)
    # Exp and sum over regions for each word
    s_norm = F.softmax(s, dim=2)  # (batch, seq_len, 289)

    # 3. Weighted region features (Context Vector)
    # img_features_t: (batch, 289, 512)
    img_features_t = img_features.transpose(1, 2)
    # c: (batch, seq_len, 512)
    c = torch.bmm(s_norm, img_features_t)

    # 4. Final Similarity R(c, e)
    # Compare each word with its corresponding context vector
    row_sim = cosine_similarity(c, words_emb_t, dim=2)

    # Average similarity across the sequence (handling variable cap_lens if needed)
    # For now, we take the mean across the fixed 18 words
    loss = -torch.log(torch.exp(row_sim).mean(dim=1))

    return loss.mean()


def sent_loss(cnn_code, sent_emb, labels, batch_size):
    """
    cnn_code: (batch_size, nef) -> global image vector
    sent_emb: (batch_size, nef) -> global sentence vector
    """
    # Global similarity between image vector and sentence vector
    scores = cosine_similarity(cnn_code, sent_emb, dim=1)

    # Actual ranking-style loss: minimize 1.0 - similarity
    return (1.0 - scores).mean()


def KL_loss(mu, logvar):
    # Kullback-Leibler divergence for VAE-like latent space smoothing
    KLD_element = mu.pow(2).add_(logvar.exp()).mul_(-1).add_(1).add_(logvar)
    KLD = torch.mean(KLD_element).mul_(-0.5)
    return KLD
