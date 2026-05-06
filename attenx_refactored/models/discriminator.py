import torch.nn as nn
from torch.nn.utils import spectral_norm


def conv3x3(in_planes, out_planes, stride=1):
    return nn.Conv2d(in_planes, out_planes, kernel_size=3, stride=stride, padding=1, bias=False)


def _maybe_sn(module, use_sn):
    return spectral_norm(module) if use_sn else module


class D_GET_LOGITS(nn.Module):
    def __init__(self, ndf, nef, use_sn=True):
        super().__init__()
        self.joint_conv = nn.Sequential(
            _maybe_sn(conv3x3(ndf * 8 + nef, ndf * 8), use_sn),
            nn.LeakyReLU(0.2, inplace=True),
            _maybe_sn(nn.Conv2d(ndf * 8, 1, kernel_size=4, stride=4), use_sn),
            nn.Sigmoid(),
        )

    def forward(self, h_code, c_code):
        c_code = c_code.view(-1, c_code.size(1), 1, 1)
        c_code = c_code.repeat(1, 1, h_code.size(2), h_code.size(3))
        h_c_code = torch.cat((h_code, c_code), 1)
        out = self.joint_conv(h_c_code)
        return out.view(-1)


class D_NET64(nn.Module):
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
        h_code = self.encode_img(x)
        return self.get_logits(h_code, c_code)


class D_NET128(nn.Module):
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
        h_code = self.encode_img(x)
        return self.get_logits(h_code, c_code)


class D_NET256(nn.Module):
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
        h_code = self.encode_img(x)
        return self.get_logits(h_code, c_code)
