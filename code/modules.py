import torch
import torch.nn as nn
import torch.nn.functional as F


class SelfAttention(nn.Module):
    def __init__(self, in_dim):
        super().__init__()
        self.query_conv = nn.Conv2d(in_dim, in_dim // 8, kernel_size=1)
        self.key_conv = nn.Conv2d(in_dim, in_dim // 8, kernel_size=1)
        self.value_conv = nn.Conv2d(in_dim, in_dim, kernel_size=1)
        self.gamma = nn.Parameter(torch.zeros(1))

    def forward(self, x):
        B, C, H, W = x.size()
        q = self.query_conv(x).view(B, -1, H * W).permute(0, 2, 1)
        k = self.key_conv(x).view(B, -1, H * W)
        energy = torch.bmm(q, k)
        attention = F.softmax(energy, dim=-1)
        v = self.value_conv(x).view(B, -1, H * W)
        out = torch.bmm(v, attention.permute(0, 2, 1))
        out = out.view(B, C, H, W)
        return self.gamma * out + x


class ConditioningAugmentation(nn.Module):
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


class CrossAttention(nn.Module):
    def __init__(self, h_dim, w_dim, attn_dim):
        super().__init__()
        self.query_conv = nn.Conv2d(h_dim, attn_dim, kernel_size=1)
        self.key_proj = nn.Linear(w_dim, attn_dim)
        self.value_proj = nn.Linear(w_dim, attn_dim)

    def forward(self, h, w):
        B, C, H, W = h.size()
        q = self.query_conv(h).view(B, -1, H * W).permute(0, 2, 1)
        k = self.key_proj(w)
        v = self.value_proj(w)
        s = torch.bmm(q, k.permute(0, 2, 1))
        s = F.softmax(s, dim=-1)
        c = torch.bmm(s, v)
        c = c.permute(0, 2, 1).view(B, -1, H, W)
        return c


class GenStage(nn.Module):
    def __init__(self, in_ch, out_ch, w_dim, attn_dim, use_self_attn=False):
        super().__init__()
        self.upsample = nn.Sequential(
            nn.Upsample(scale_factor=2, mode="nearest"),
            nn.Conv2d(in_ch, out_ch, 3, 1, 1, bias=False),
            nn.BatchNorm2d(out_ch),
            nn.ReLU(True),
        )
        self.cross_attn = CrossAttention(out_ch, w_dim, attn_dim)
        self.fuse = nn.Sequential(
            nn.Conv2d(out_ch + attn_dim, out_ch, 3, 1, 1, bias=False),
            nn.BatchNorm2d(out_ch),
            nn.ReLU(True),
        )
        self.self_attn = SelfAttention(out_ch) if use_self_attn else None

    def forward(self, h, w):
        h = self.upsample(h)
        ctx = self.cross_attn(h, w)
        h = self.fuse(torch.cat([h, ctx], dim=1))
        if self.self_attn is not None:
            h = self.self_attn(h)
        return h