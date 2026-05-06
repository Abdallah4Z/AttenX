"""
Multi-scale conditional discriminators for AttenX.

Three discriminators operate at 64x64, 128x128, and 256x256
resolutions. Each uses spectral normalization (optional) and
a conditional logit branch that concatenates the sentence
embedding with image features.
"""

import torch.nn as nn
from torch.nn.utils import spectral_norm


def conv3x3(in_planes, out_planes, stride=1):
    """
    Create a 3x3 convolution layer.

    Args:
        in_planes: Input channels.
        out_planes: Output channels.
        stride: Convolution stride.

    Returns:
        nn.Conv2d with kernel_size=3, padding=1, no bias.
    """
    return nn.Conv2d(in_planes, out_planes, kernel_size=3,
                     stride=stride, padding=1, bias=False)


def _maybe_sn(module, use_sn):
    """
    Conditionally wrap a module with spectral normalization.

    Args:
        module: PyTorch module.
        use_sn: If True, apply spectral_norm.

    Returns:
        The original or spectrally-normalized module.
    """
    return spectral_norm(module) if use_sn else module


class D_GET_LOGITS(nn.Module):
    """
    Conditional logit branch for multi-scale discriminators.

    Concatenates the sentence embedding (spatially replicated)
    with image features and produces a single scalar logit.

    Args:
        ndf: Base discriminator feature count.
        nef: Sentence embedding dimension.
        use_sn: Whether to use spectral normalization.
    """

    def __init__(self, ndf, nef, use_sn=True):
        super().__init__()
        self.joint_conv = nn.Sequential(
            _maybe_sn(conv3x3(ndf * 8 + nef, ndf * 8), use_sn),
            nn.LeakyReLU(0.2, inplace=True),
            _maybe_sn(nn.Conv2d(ndf * 8, 1, kernel_size=4, stride=4), use_sn),
            nn.Sigmoid(),
        )

    def forward(self, h_code, c_code):
        """
        Produce realism logit from image features and sentence code.

        Args:
            h_code: Image feature map (B, ndf*8, H, W).
            c_code: Sentence embedding (B, nef).

        Returns:
            Scalar logit per sample (B,).
        """
        c_code = c_code.view(-1, c_code.size(1), 1, 1)
        c_code = c_code.repeat(1, 1, h_code.size(2), h_code.size(3))
        h_c_code = torch.cat((h_code, c_code), 1)
        out = self.joint_conv(h_c_code)
        return out.view(-1)


class D_NET64(nn.Module):
    """
    Discriminator for 64x64 images.

    Downsampling: 64 -> 32 -> 16 -> 8 -> 4.

    Args:
        ndf: Base feature channel count.
        nef: Sentence embedding dimension.
        use_sn: Whether to use spectral normalization.
    """

    def __init__(self, ndf=64, nef=512, use_sn=True):
        super().__init__()
        self.encode_img = nn.Sequential(
            _maybe_sn(nn.Conv2d(3, ndf, 3, 2, 1, bias=False), use_sn),
            nn.LeakyReLU(0.2, inplace=True),
            _maybe_sn(nn.Conv2d(ndf, ndf * 2, 3, 2, 1, bias=False), use_sn),
            nn.BatchNorm2d(ndf * 2),
            nn.LeakyReLU(0.2, inplace=True),
            _maybe_sn(nn.Conv2d(ndf * 2, ndf * 4, 3, 2, 1, bias=False), use_sn),
            nn.BatchNorm2d(ndf * 4),
            nn.LeakyReLU(0.2, inplace=True),
            _maybe_sn(nn.Conv2d(ndf * 4, ndf * 8, 3, 2, 1, bias=False), use_sn),
            nn.BatchNorm2d(ndf * 8),
            nn.LeakyReLU(0.2, inplace=True),
        )
        self.get_logits = D_GET_LOGITS(ndf, nef, use_sn=use_sn)

    def forward(self, x, c_code):
        """
        Classify a 64x64 image as real or fake given conditioning.

        Args:
            x: Input image (B, 3, 64, 64).
            c_code: Sentence embedding (B, nef).

        Returns:
            Realism logits (B,).
        """
        h_code = self.encode_img(x)
        return self.get_logits(h_code, c_code)


class D_NET128(nn.Module):
    """
    Discriminator for 128x128 images.

    Downsampling: 128 -> 64 -> 32 -> 16 -> 8 -> 4.

    Args:
        ndf: Base feature channel count.
        nef: Sentence embedding dimension.
        use_sn: Whether to use spectral normalization.
    """

    def __init__(self, ndf=64, nef=512, use_sn=True):
        super().__init__()
        self.encode_img = nn.Sequential(
            _maybe_sn(nn.Conv2d(3, ndf, 3, 2, 1, bias=False), use_sn),
            nn.LeakyReLU(0.2, inplace=True),
            _maybe_sn(nn.Conv2d(ndf, ndf * 2, 3, 2, 1, bias=False), use_sn),
            nn.BatchNorm2d(ndf * 2),
            nn.LeakyReLU(0.2, inplace=True),
            _maybe_sn(nn.Conv2d(ndf * 2, ndf * 4, 3, 2, 1, bias=False), use_sn),
            nn.BatchNorm2d(ndf * 4),
            nn.LeakyReLU(0.2, inplace=True),
            _maybe_sn(nn.Conv2d(ndf * 4, ndf * 8, 3, 2, 1, bias=False), use_sn),
            nn.BatchNorm2d(ndf * 8),
            nn.LeakyReLU(0.2, inplace=True),
            _maybe_sn(nn.Conv2d(ndf * 8, ndf * 8, 3, 2, 1, bias=False), use_sn),
            nn.BatchNorm2d(ndf * 8),
            nn.LeakyReLU(0.2, inplace=True),
        )
        self.get_logits = D_GET_LOGITS(ndf, nef, use_sn=use_sn)

    def forward(self, x, c_code):
        """
        Classify a 128x128 image as real or fake given conditioning.

        Args:
            x: Input image (B, 3, 128, 128).
            c_code: Sentence embedding (B, nef).

        Returns:
            Realism logits (B,).
        """
        h_code = self.encode_img(x)
        return self.get_logits(h_code, c_code)


class D_NET256(nn.Module):
    """
    Discriminator for 256x256 images.

    Downsampling: 256 -> 128 -> 64 -> 32 -> 16 -> 8 -> 4.

    Args:
        ndf: Base feature channel count.
        nef: Sentence embedding dimension.
        use_sn: Whether to use spectral normalization.
    """

    def __init__(self, ndf=64, nef=512, use_sn=True):
        super().__init__()
        self.encode_img = nn.Sequential(
            _maybe_sn(nn.Conv2d(3, ndf, 3, 2, 1, bias=False), use_sn),
            nn.LeakyReLU(0.2, inplace=True),
            _maybe_sn(nn.Conv2d(ndf, ndf * 2, 3, 2, 1, bias=False), use_sn),
            nn.BatchNorm2d(ndf * 2),
            nn.LeakyReLU(0.2, inplace=True),
            _maybe_sn(nn.Conv2d(ndf * 2, ndf * 4, 3, 2, 1, bias=False), use_sn),
            nn.BatchNorm2d(ndf * 4),
            nn.LeakyReLU(0.2, inplace=True),
            _maybe_sn(nn.Conv2d(ndf * 4, ndf * 8, 3, 2, 1, bias=False), use_sn),
            nn.BatchNorm2d(ndf * 8),
            nn.LeakyReLU(0.2, inplace=True),
            _maybe_sn(nn.Conv2d(ndf * 8, ndf * 16, 3, 2, 1, bias=False), use_sn),
            nn.BatchNorm2d(ndf * 16),
            nn.LeakyReLU(0.2, inplace=True),
            _maybe_sn(nn.Conv2d(ndf * 16, ndf * 8, 3, 2, 1, bias=False), use_sn),
            nn.BatchNorm2d(ndf * 8),
            nn.LeakyReLU(0.2, inplace=True),
        )
        self.get_logits = D_GET_LOGITS(ndf, nef, use_sn=use_sn)

    def forward(self, x, c_code):
        """
        Classify a 256x256 image as real or fake given conditioning.

        Args:
            x: Input image (B, 3, 256, 256).
            c_code: Sentence embedding (B, nef).

        Returns:
            Realism logits (B,).
        """
        h_code = self.encode_img(x)
        return self.get_logits(h_code, c_code)
