"""
Cascaded generator network (G_NET) for AttenX.

Progressively upsamples from 4x4 to 256x256 resolution. Each
stage applies nearest-neighbor upsampling, convolution, batch
norm, ReLU, cross-attention with word embeddings, and a fusion
conv. Self-attention (single or multi-head) is injected at the
64x64 resolution stage.
"""

import torch
import torch.nn as nn

from attenx_refactored.models.conditioning import ConditioningAugmentation
from attenx_refactored.models.attention import (
    SelfAttention,
    MultiHeadSelfAttention,
    CrossAttention,
)

_ATTN_MODULES = {
    "none": None,
    "single": SelfAttention,
    "multi": MultiHeadSelfAttention,
}


class GenStage(nn.Module):
    """
    Single upsampling stage with cross-attention fusion.

    Each stage: nearest-neighbor 2x up -> conv3x3 -> BN -> ReLU
    -> cross-attention with word embeddings -> fuse (conv3x3).
    Optionally followed by self-attention.

    Args:
        in_ch: Input channel count.
        out_ch: Output channel count after upsampling.
        w_dim: Word embedding dimension.
        attn_dim: Cross-attention projection dimension.
        use_self_attn: Whether to include self-attention at this stage.
    """

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
        self.self_attn = None
        if use_self_attn:
            self.self_attn = SelfAttention(out_ch)

    def forward(self, h, w):
        """
        Process features through one generation stage.

        Args:
            h: Input feature map (B, in_ch, H, W).
            w: Word embeddings for cross-attention (B, seq_len, w_dim).

        Returns:
            Output feature map (B, out_ch, H*2, W*2).
        """
        h = self.upsample(h)
        ctx = self.cross_attn(h, w)
        h = self.fuse(torch.cat([h, ctx], dim=1))
        if self.self_attn is not None:
            h = self.self_attn(h)
        return h


class G_NET(nn.Module):
    """
    Cascaded generator network (AttenX).

    Starts from a noise vector z and a conditioning code c, then
    progressively upsamples through 7 stages to produce 256x256
    images. Intermediate outputs at 64x64 and 128x128 are also
    returned for multi-scale discriminator supervision.

    Args:
        ngf: Base feature channel count for the generator.
        nz: Noise vector dimension.
        nef: Conditioning augmentation embedding dimension.
        word_dim: Word embedding dimension from text encoder.
        attention_mode: Type of self-attention ('none', 'single', 'multi').
        num_heads: Number of heads for multi-head attention.
    """

    def __init__(self, ngf=64, nz=100, nef=512, word_dim=512,
                 attention_mode="none", num_heads=4):
        super().__init__()
        self.ngf = ngf
        self.nz = nz
        self.attention_mode = attention_mode

        self.ca = ConditioningAugmentation(nef, nz)

        self.stage0 = nn.Sequential(
            nn.ConvTranspose2d(nz * 2, ngf * 16, 4, 1, 0, bias=False),
            nn.BatchNorm2d(ngf * 16),
            nn.ReLU(True),
        )

        self.stage1 = GenStage(ngf * 16, ngf * 8, word_dim, ngf)
        self.stage2 = GenStage(ngf * 8, ngf * 4, word_dim, ngf)
        self.stage3 = GenStage(ngf * 4, ngf * 2, word_dim, ngf)

        use_sa = attention_mode == "single"
        use_mhsa = attention_mode == "multi"
        self.stage4 = GenStage(ngf * 2, ngf, word_dim, ngf, use_self_attn=False)

        if use_sa:
            self.self_attn_64 = SelfAttention(ngf)
        elif use_mhsa:
            self.self_attn_64 = MultiHeadSelfAttention(ngf, num_heads=num_heads)
        else:
            self.self_attn_64 = None

        self.to_rgb_64 = nn.Conv2d(ngf, 3, 3, 1, 1)

        self.stage5 = GenStage(ngf, ngf // 2, word_dim, ngf // 2)
        self.to_rgb_128 = nn.Conv2d(ngf // 2, 3, 3, 1, 1)

        self.stage6 = nn.Sequential(
            nn.Upsample(scale_factor=2, mode="nearest"),
            nn.Conv2d(ngf // 2, ngf // 4, 3, 1, 1, bias=False),
            nn.BatchNorm2d(ngf // 4),
            nn.ReLU(True),
        )
        self.to_rgb = nn.Sequential(
            nn.Conv2d(ngf // 4, 3, 3, 1, 1),
            nn.Tanh(),
        )

    def forward(self, z, sent_emb, word_emb):
        """
        Generate images from noise and text embeddings.

        Args:
            z: Noise tensor (B, nz, 1, 1).
            sent_emb: Global sentence embedding (B, nef).
            word_emb: Word-level features (B, seq_len, word_dim).

        Returns:
            Tuple of (img_64, img_128, img_256, mu, logvar) where:
                img_64: 64x64 intermediate output.
                img_128: 128x128 intermediate output.
                img_256: Final 256x256 output.
                mu: Mean from conditioning augmentation.
                logvar: Log-variance from conditioning augmentation.
        """
        c, mu, logvar = self.ca(sent_emb)
        c = c.unsqueeze(-1).unsqueeze(-1)
        h = self.stage0(torch.cat([z, c], dim=1))

        h = self.stage1(h, word_emb)
        h = self.stage2(h, word_emb)
        h = self.stage3(h, word_emb)
        h = self.stage4(h, word_emb)

        if self.self_attn_64 is not None:
            h = self.self_attn_64(h)

        img_64 = torch.tanh(self.to_rgb_64(h))

        h = self.stage5(h, word_emb)
        img_128 = torch.tanh(self.to_rgb_128(h))

        h = self.stage6(h)
        img_256 = self.to_rgb(h)

        return img_64, img_128, img_256, mu, logvar
