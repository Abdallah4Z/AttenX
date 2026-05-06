import torch
import torch.nn.functional as F


def cosine_similarity(x1, x2, dim=1, eps=1e-8):
    n1 = torch.norm(x1, dim=dim, keepdim=True).clamp_min(eps)
    n2 = torch.norm(x2, dim=dim, keepdim=True).clamp_min(eps)
    return ((x1 / n1) * (x2 / n2)).sum(dim=dim)


def words_loss(img_features, words_emb, labels, cap_lens, batch_size):
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
    scores = torch.mm(cnn_code, sent_emb.t())
    diag = torch.diag(scores)

    cost_pos = (diag - scores).clamp(min=0)
    loss_fwd = cost_pos.mean()

    cost_pos_b = (diag - scores.t()).clamp(min=0)
    loss_bwd = cost_pos_b.mean()

    return loss_fwd + loss_bwd


def KL_loss(mu, logvar):
    kld = mu.pow(2).add_(logvar.exp()).mul_(-1).add_(1).add_(logvar)
    return torch.mean(kld).mul_(-0.5)
