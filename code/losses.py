import torch
import torch.nn.functional as F


def cosine_similarity(x1, x2, dim=1, eps=1e-8):
    # Manual implementation to avoid NaN gradients from zero-norm vectors
    x1_norm = torch.norm(x1, dim=dim, keepdim=True).clamp_min(eps)
    x2_norm = torch.norm(x2, dim=dim, keepdim=True).clamp_min(eps)
    x1_unit = x1 / x1_norm
    x2_unit = x2 / x2_norm
    return (x1_unit * x2_unit).sum(dim=dim)


def words_loss(img_features, words_emb, labels, cap_lens, batch_size):
    """
    Word-level attentional DAMSM loss.

    img_features: (batch, nef, H, W)
    words_emb: (batch, nef, seq_len)
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

    # Zero out padding positions to avoid NaN from zero-norm vectors
    row_sim = torch.where(mask.bool(), row_sim, torch.zeros_like(row_sim))
    exp_row_sim = torch.exp(row_sim) * mask
    valid_counts = mask.sum(dim=1).clamp_min(1.0)
    loss = -torch.log(exp_row_sim.sum(dim=1) / valid_counts + 1e-8)

    return loss.mean()


def sent_loss(cnn_code, sent_emb, labels, batch_size):
    """
    Sentence-level DAMSM loss with contrastive ranking.
    For each image-sentence pair, maximizes similarity with the correct pair
    and minimizes similarity with all incorrect pairs in the batch.

    cnn_code: (batch, nef)
    sent_emb: (batch, nef)
    """
    scores = torch.mm(cnn_code, sent_emb.t())
    diag = torch.diag(scores)

    # Forward: given image, find correct sentence
    cost_pos = (diag - scores).clamp(min=0)
    loss_fwd = cost_pos.mean()

    # Backward: given sentence, find correct image
    cost_pos_b = (diag - scores.t()).clamp(min=0)
    loss_bwd = cost_pos_b.mean()

    return loss_fwd + loss_bwd


def KL_loss(mu, logvar):
    KLD_element = mu.pow(2).add_(logvar.exp()).mul_(-1).add_(1).add_(logvar)
    return torch.mean(KLD_element).mul_(-0.5)