import torch
import torch.nn as nn
from code.modules import ConditioningAugmentation, GenStage, SelfAttention


class G_NET(nn.Module):
    def __init__(self, ngf=64, nz=100, nef=512, nhidden=256, word_dim=512):
        super().__init__()
        self.nz = nz
        self.ngf = ngf

        self.ca = ConditioningAugmentation(nef, nz)

        self.stage0 = nn.Sequential(
            nn.ConvTranspose2d(nz * 2, ngf * 16, 4, 1, 0, bias=False),
            nn.BatchNorm2d(ngf * 16),
            nn.ReLU(True),
        )

        self.stage1 = GenStage(ngf * 16, ngf * 8, word_dim, ngf)
        self.stage2 = GenStage(ngf * 8, ngf * 4, word_dim, ngf)
        self.stage3 = GenStage(ngf * 4, ngf * 2, word_dim, ngf)
        self.stage4 = GenStage(ngf * 2, ngf, word_dim, ngf, use_self_attn=True)

        self.to_rgb_64 = nn.Conv2d(ngf, 3, 3, 1, 1, bias=False)

        self.stage5 = GenStage(ngf, ngf // 2, word_dim, ngf // 2)
        self.to_rgb_128 = nn.Conv2d(ngf // 2, 3, 3, 1, 1, bias=False)

        self.stage6 = nn.Sequential(
            nn.Upsample(scale_factor=2, mode="nearest"),
            nn.Conv2d(ngf // 2, ngf // 4, 3, 1, 1, bias=False),
            nn.BatchNorm2d(ngf // 4),
            nn.ReLU(True),
        )

        self.to_rgb = nn.Sequential(
            nn.Conv2d(ngf // 4, 3, 3, 1, 1, bias=False),
            nn.Tanh(),
        )

    def forward(self, z, sent_emb, word_emb):
        c, mu, logvar = self.ca(sent_emb)
        c = c.unsqueeze(-1).unsqueeze(-1)
        h = self.stage0(torch.cat([z, c], dim=1))

        h = self.stage1(h, word_emb)
        h = self.stage2(h, word_emb)
        h = self.stage3(h, word_emb)
        h = self.stage4(h, word_emb)

        img_64 = torch.tanh(self.to_rgb_64(h))

        h = self.stage5(h, word_emb)

        img_128 = torch.tanh(self.to_rgb_128(h))

        h = self.stage6(h)
        img_256 = self.to_rgb(h)

        return img_64, img_128, img_256, mu, logvar


if __name__ == "__main__":
    batch = 2
    nz = 100
    nef = 512
    nhidden = 256
    word_dim = nhidden * 2

    netG = G_NET(ngf=64, nz=nz, nef=nef, nhidden=nhidden, word_dim=word_dim)

    z = torch.randn(batch, nz, 1, 1)
    sent_emb = torch.randn(batch, nef)
    word_emb = torch.randn(batch, 18, word_dim)

    img_64, img_128, img_256, mu, logvar = netG(z, sent_emb, word_emb)
    print(f"img_64 shape:  {img_64.shape}")
    print(f"img_128 shape: {img_128.shape}")
    print(f"img_256 shape: {img_256.shape}")
    print(f"CA mu shape: {mu.shape}, logvar shape: {logvar.shape}")
    assert img_64.shape == (batch, 3, 64, 64)
    assert img_128.shape == (batch, 3, 128, 128)
    assert img_256.shape == (batch, 3, 256, 256)
    assert mu.shape == (batch, nz)
    assert logvar.shape == (batch, nz)
    print("Success: All shapes verified.")