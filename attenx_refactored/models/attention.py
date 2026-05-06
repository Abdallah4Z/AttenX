"""
Attention mechanism modules for AttenX.

Provides three attention variants used inside the generator:
  - SelfAttention: Single-head non-local (SAGAN-style).
  - MultiHeadSelfAttention: Multi-head self-attention (Transformer-style).
  - CrossAttention: Word-level cross-attention (AttnGAN DAMSM).
"""

import torch
import torch.nn as nn
import torch.nn.functional as F


class SelfAttention(nn.Module):
    """
    Single-head non-local self-attention (SAGAN-style).

    Injects long-range spatial dependencies by computing attention
    over all spatial positions. Uses 1x1 conv projections for
    query, key, value and a learnable gamma (initialized to 0)
    for residual scaling.

    Args:
        in_dim: Number of input feature channels.
    """

    def __init__(self, in_dim):
        super().__init__()
        self.query = nn.Conv2d(in_dim, in_dim // 8, kernel_size=1)
        self.key = nn.Conv2d(in_dim, in_dim // 8, kernel_size=1)
        self.value = nn.Conv2d(in_dim, in_dim, kernel_size=1)
        self.gamma = nn.Parameter(torch.zeros(1))

    def forward(self, x):
        """
        Apply self-attention to spatial feature map.

        Args:
            x: Input tensor of shape (B, C, H, W).

        Returns:
            Output tensor with same shape as input (residual connection).
        """
        B, C, H, W = x.shape
        q = self.query(x).view(B, -1, H * W).permute(0, 2, 1)
        k = self.key(x).view(B, -1, H * W)
        energy = torch.bmm(q, k)
        attn = F.softmax(energy, dim=-1)
        v = self.value(x).view(B, -1, H * W)
        out = torch.bmm(v, attn.permute(0, 2, 1))
        out = out.view(B, C, H, W)
        return self.gamma * out + x


class MultiHeadSelfAttention(nn.Module):
    """
    Multi-head self-attention (Vaswani et al. 2017).

    Reshapes the spatial feature map into a sequence of tokens,
    applies LayerNorm, then computes attention across h parallel
    heads. Output is concatenated and projected back.

    Args:
        embed_dim: Total embedding dimension (must be divisible by num_heads).
        num_heads: Number of parallel attention heads.
    """

    def __init__(self, embed_dim, num_heads=4):
        super().__init__()
        assert embed_dim % num_heads == 0
        self.num_heads = num_heads
        self.head_dim = embed_dim // num_heads
        self.scale = self.head_dim ** -0.5

        self.qkv = nn.Linear(embed_dim, embed_dim * 3)
        self.proj = nn.Linear(embed_dim, embed_dim)
        self.norm = nn.LayerNorm(embed_dim)

    def forward(self, x):
        """
        Apply multi-head self-attention.

        Args:
            x: Input tensor of shape (B, C, H, W).

        Returns:
            Output tensor with same shape as input (residual connection).
        """
        B, C, H, W = x.shape
        x_flat = x.view(B, C, -1).permute(0, 2, 1)
        x_norm = self.norm(x_flat)

        qkv = self.qkv(x_norm).reshape(B, -1, 3, self.num_heads, self.head_dim)
        qkv = qkv.permute(2, 0, 3, 1, 4)
        q, k, v = qkv[0], qkv[1], qkv[2]

        attn = (q @ k.transpose(-2, -1)) * self.scale
        attn = F.softmax(attn, dim=-1)

        out = (attn @ v).transpose(1, 2).reshape(B, -1, C)
        out = self.proj(out)
        out = out.permute(0, 2, 1).view(B, C, H, W)
        return out + x


class CrossAttention(nn.Module):
    """
    Word-level cross-attention (AttnGAN DAMSM).

    Queries come from the image feature map, keys and values
    come from word embeddings. Produces a context vector that
    captures fine-grained word-region alignment.

    Args:
        h_dim: Image feature channel dimension.
        w_dim: Word embedding dimension.
        attn_dim: Projection dimension for attention.
    """

    def __init__(self, h_dim, w_dim, attn_dim):
        super().__init__()
        self.query = nn.Conv2d(h_dim, attn_dim, kernel_size=1)
        self.key = nn.Linear(w_dim, attn_dim)
        self.value = nn.Linear(w_dim, attn_dim)

    def forward(self, h, w):
        """
        Compute cross-attention between image features and word embeddings.

        Args:
            h: Image feature map (B, C, H, W).
            w: Word embeddings (B, seq_len, w_dim).

        Returns:
            Context tensor (B, attn_dim, H, W).
        """
        B, C, H, W = h.shape
        q = self.query(h).view(B, -1, H * W).permute(0, 2, 1)
        k = self.key(w)
        v = self.value(w)
        s = torch.bmm(q, k.transpose(1, 2))
        attn = F.softmax(s, dim=-1)
        c = torch.bmm(attn, v)
        c = c.permute(0, 2, 1).view(B, -1, H, W)
        return c
