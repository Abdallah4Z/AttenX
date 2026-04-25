import torch
import torch.nn as nn
from torch.nn.utils import spectral_norm


def conv3x3(in_planes, out_planes, stride=1):
    return nn.Conv2d(in_planes, out_planes, kernel_size=3, stride=stride, padding=1, bias=False)


def _sn(module, enabled=True):
    return spectral_norm(module) if enabled else module


class D_GET_LOGITS(nn.Module):
    def __init__(self, ndf, nef, bcondition=True, use_spectral_norm=True):
        super().__init__()
        self.df_dim = ndf
        self.ef_dim = nef
        self.bcondition = bcondition

        if bcondition:
            self.jointConv = nn.Sequential(
                _sn(conv3x3(ndf * 8 + nef, ndf * 8), use_spectral_norm),
                nn.LeakyReLU(0.2, inplace=True),
                _sn(nn.Conv2d(ndf * 8, 1, kernel_size=4, stride=4), use_spectral_norm),
                nn.Sigmoid(),
            )
        else:
            self.outlogits = nn.Sequential(
                _sn(nn.Conv2d(ndf * 8, 1, kernel_size=4, stride=4), use_spectral_norm),
                nn.Sigmoid(),
            )

    def forward(self, h_code, c_code=None):
        if self.bcondition and c_code is not None:
            c_code = c_code.view(-1, self.ef_dim, 1, 1)
            c_code = c_code.repeat(1, 1, h_code.size(2), h_code.size(3))
            h_c_code = torch.cat((h_code, c_code), 1)
            out = self.jointConv(h_c_code)
        else:
            out = self.outlogits(h_code)
        return out.view(-1)


class D_NET64(nn.Module):
    """Discriminator for 64x64 images."""

    def __init__(self, ndf, nef=512, use_spectral_norm=True):
        super().__init__()
        self.encode_img = nn.Sequential(
            _sn(nn.Conv2d(3, ndf, 3, 2, 1, bias=False), use_spectral_norm),       # 64 -> 32
            nn.LeakyReLU(0.2, inplace=True),
            _sn(nn.Conv2d(ndf, ndf * 2, 3, 2, 1, bias=False), use_spectral_norm),  # 32 -> 16
            nn.BatchNorm2d(ndf * 2),
            nn.LeakyReLU(0.2, inplace=True),
            _sn(nn.Conv2d(ndf * 2, ndf * 4, 3, 2, 1, bias=False), use_spectral_norm),  # 16 -> 8
            nn.BatchNorm2d(ndf * 4),
            nn.LeakyReLU(0.2, inplace=True),
            _sn(nn.Conv2d(ndf * 4, ndf * 8, 3, 2, 1, bias=False), use_spectral_norm),  # 8 -> 4
            nn.BatchNorm2d(ndf * 8),
            nn.LeakyReLU(0.2, inplace=True),
        )
        self.get_logits = D_GET_LOGITS(ndf, nef, bcondition=True, use_spectral_norm=use_spectral_norm)

    def forward(self, x, c_code):
        h_code = self.encode_img(x)
        return self.get_logits(h_code, c_code)


class D_NET128(nn.Module):
    """Discriminator for 128x128 images."""

    def __init__(self, ndf, nef=512, use_spectral_norm=True):
        super().__init__()
        self.encode_img = nn.Sequential(
            _sn(nn.Conv2d(3, ndf, 3, 2, 1, bias=False), use_spectral_norm),       # 128 -> 64
            nn.LeakyReLU(0.2, inplace=True),
            _sn(nn.Conv2d(ndf, ndf * 2, 3, 2, 1, bias=False), use_spectral_norm),  # 64 -> 32
            nn.BatchNorm2d(ndf * 2),
            nn.LeakyReLU(0.2, inplace=True),
            _sn(nn.Conv2d(ndf * 2, ndf * 4, 3, 2, 1, bias=False), use_spectral_norm),  # 32 -> 16
            nn.BatchNorm2d(ndf * 4),
            nn.LeakyReLU(0.2, inplace=True),
            _sn(nn.Conv2d(ndf * 4, ndf * 8, 3, 2, 1, bias=False), use_spectral_norm),  # 16 -> 8
            nn.BatchNorm2d(ndf * 8),
            nn.LeakyReLU(0.2, inplace=True),
            _sn(nn.Conv2d(ndf * 8, ndf * 8, 3, 2, 1, bias=False), use_spectral_norm),  # 8 -> 4
            nn.BatchNorm2d(ndf * 8),
            nn.LeakyReLU(0.2, inplace=True),
        )
        self.get_logits = D_GET_LOGITS(ndf, nef, bcondition=True, use_spectral_norm=use_spectral_norm)

    def forward(self, x, c_code):
        h_code = self.encode_img(x)
        return self.get_logits(h_code, c_code)


class D_NET256(nn.Module):
    """Discriminator for 256x256 images."""

    def __init__(self, ndf, nef=512, use_spectral_norm=True):
        super().__init__()
        self.encode_img = nn.Sequential(
            _sn(nn.Conv2d(3, ndf, 3, 2, 1, bias=False), use_spectral_norm),       # 256 -> 128
            nn.LeakyReLU(0.2, inplace=True),
            _sn(nn.Conv2d(ndf, ndf * 2, 3, 2, 1, bias=False), use_spectral_norm),  # 128 -> 64
            nn.BatchNorm2d(ndf * 2),
            nn.LeakyReLU(0.2, inplace=True),
            _sn(nn.Conv2d(ndf * 2, ndf * 4, 3, 2, 1, bias=False), use_spectral_norm),  # 64 -> 32
            nn.BatchNorm2d(ndf * 4),
            nn.LeakyReLU(0.2, inplace=True),
            _sn(nn.Conv2d(ndf * 4, ndf * 8, 3, 2, 1, bias=False), use_spectral_norm),  # 32 -> 16
            nn.BatchNorm2d(ndf * 8),
            nn.LeakyReLU(0.2, inplace=True),
            _sn(nn.Conv2d(ndf * 8, ndf * 16, 3, 2, 1, bias=False), use_spectral_norm),  # 16 -> 8
            nn.BatchNorm2d(ndf * 16),
            nn.LeakyReLU(0.2, inplace=True),
            _sn(nn.Conv2d(ndf * 16, ndf * 8, 3, 2, 1, bias=False), use_spectral_norm),  # 8 -> 4
            nn.BatchNorm2d(ndf * 8),
            nn.LeakyReLU(0.2, inplace=True),
        )
        self.get_logits = D_GET_LOGITS(ndf, nef, bcondition=True, use_spectral_norm=use_spectral_norm)

    def forward(self, x, c_code):
        h_code = self.encode_img(x)
        return self.get_logits(h_code, c_code)