import torch
import torch.nn as nn
from torch.nn.utils import spectral_norm


def conv3x3(in_planes, out_planes, stride=1):
    "3x3 convolution with padding"
    return nn.Conv2d(in_planes, out_planes, kernel_size=3, stride=stride, padding=1, bias=False)


class D_GET_LOGITS(nn.Module):
    def __init__(self, ndf, nef, bcondition=True):
        super(D_GET_LOGITS, self).__init__()
        self.df_dim = ndf
        self.ef_dim = nef
        self.bcondition = bcondition

        # Classification head: (batch, ndf*8 + nef, 4, 4) -> (batch, 1)
        if bcondition:
            self.jointConv = nn.Sequential(
                spectral_norm(conv3x3(ndf * 8 + nef, ndf * 8)),
                nn.LeakyReLU(0.2, inplace=True),
                spectral_norm(nn.Conv2d(ndf * 8, 1, kernel_size=4, stride=4)),
                nn.Sigmoid(),
            )
        else:
            self.outlogits = nn.Sequential(
                spectral_norm(nn.Conv2d(ndf * 8, 1, kernel_size=4, stride=4)), nn.Sigmoid()
            )

    def forward(self, h_code, c_code=None):
        if self.bcondition and c_code is not None:
            # Condition image features with text embedding
            c_code = c_code.view(-1, self.ef_dim, 1, 1)
            c_code = c_code.repeat(1, 1, 4, 4)
            h_c_code = torch.cat((h_code, c_code), 1)
            out = self.jointConv(h_c_code)
        else:
            out = self.outlogits(h_code)
        return out.view(-1)


class D_NET256(nn.Module):
    def __init__(self, ndf, nef=512):
        super(D_NET256, self).__init__()
        self.encode_img = nn.Sequential(
            spectral_norm(nn.Conv2d(3, ndf, 3, 2, 1, bias=False)),  # 256 -> 128
            nn.LeakyReLU(0.2, inplace=True),
            spectral_norm(nn.Conv2d(ndf, ndf * 2, 3, 2, 1, bias=False)),  # 128 -> 64
            nn.BatchNorm2d(ndf * 2),
            nn.LeakyReLU(0.2, inplace=True),
            spectral_norm(nn.Conv2d(ndf * 2, ndf * 4, 3, 2, 1, bias=False)),  # 64 -> 32
            nn.BatchNorm2d(ndf * 4),
            nn.LeakyReLU(0.2, inplace=True),
            spectral_norm(nn.Conv2d(ndf * 4, ndf * 8, 3, 2, 1, bias=False)),  # 32 -> 16
            nn.BatchNorm2d(ndf * 8),
            nn.LeakyReLU(0.2, inplace=True),
            spectral_norm(nn.Conv2d(ndf * 8, ndf * 16, 3, 2, 1, bias=False)),  # 16 -> 8
            nn.BatchNorm2d(ndf * 16),
            nn.LeakyReLU(0.2, inplace=True),
            spectral_norm(nn.Conv2d(ndf * 16, ndf * 8, 3, 2, 1, bias=False)),  # 8 -> 4
            nn.BatchNorm2d(ndf * 8),
            nn.LeakyReLU(0.2, inplace=True),
        )
        # Conditioned Logit Head (ndf*8 = 512, nef = 512)
        self.get_logits = D_GET_LOGITS(ndf, nef, bcondition=True)

    def forward(self, x, c_code):
        h_code = self.encode_img(x)  # (batch, 512, 4, 4)
        return self.get_logits(h_code, c_code)
