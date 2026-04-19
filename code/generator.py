import torch
import torch.nn as nn
from code.modules import SelfAttention


class G_NET(nn.Module):
    def __init__(self, ngf=64):
        super(G_NET, self).__init__()
        self.ngf = ngf

        # Stage 0: 4x4
        self.stage0 = nn.Sequential(
            nn.ConvTranspose2d(100, ngf * 16, 4, 1, 0, bias=False),
            nn.BatchNorm2d(ngf * 16),
            nn.ReLU(True),
        )

        # Stage 1: 4x4 -> 8x8
        self.stage1 = nn.Sequential(
            nn.Upsample(scale_factor=2, mode="nearest"),
            nn.Conv2d(ngf * 16, ngf * 8, 3, 1, 1, bias=False),
            nn.BatchNorm2d(ngf * 8),
            nn.ReLU(True),
        )

        # Stage 2: 8x8 -> 16x16
        self.stage2 = nn.Sequential(
            nn.Upsample(scale_factor=2, mode="nearest"),
            nn.Conv2d(ngf * 8, ngf * 4, 3, 1, 1, bias=False),
            nn.BatchNorm2d(ngf * 4),
            nn.ReLU(True),
        )

        # Stage 3: 16x16 -> 32x32
        self.stage3 = nn.Sequential(
            nn.Upsample(scale_factor=2, mode="nearest"),
            nn.Conv2d(ngf * 4, ngf * 2, 3, 1, 1, bias=False),
            nn.BatchNorm2d(ngf * 2),
            nn.ReLU(True),
        )

        # Stage 4: 32x32 -> 64x64
        self.stage4 = nn.Sequential(
            nn.Upsample(scale_factor=2, mode="nearest"),
            nn.Conv2d(ngf * 2, ngf, 3, 1, 1, bias=False),
            nn.BatchNorm2d(ngf),
            nn.ReLU(True),
        )

        # INJECTED: Self-Attention at 64x64 stage (Optimized for VRAM)
        self.attn_stage = SelfAttention(ngf)

        # Stage 5: 64x64 -> 128x128
        self.stage5 = nn.Sequential(
            nn.Upsample(scale_factor=2, mode="nearest"),
            nn.Conv2d(ngf, ngf // 2, 3, 1, 1, bias=False),
            nn.BatchNorm2d(ngf // 2),
            nn.ReLU(True),
        )

        # Stage 6: 128x128 -> 256x256
        self.stage6 = nn.Sequential(
            nn.Upsample(scale_factor=2, mode="nearest"),
            nn.Conv2d(ngf // 2, ngf // 4, 3, 1, 1, bias=False),
            nn.BatchNorm2d(ngf // 4),
            nn.ReLU(True),
        )

        # Final image output: 256x256
        self.to_rgb = nn.Sequential(nn.Conv2d(ngf // 4, 3, 3, 1, 1, bias=False), nn.Tanh())

    def forward(self, z):
        h = self.stage0(z)
        h = self.stage1(h)
        h = self.stage2(h)
        h = self.stage3(h)
        h = self.stage4(h)
        h = self.attn_stage(h)  # Self-Attention Applied at 64x64
        h = self.stage5(h)
        h = self.stage6(h)
        out_img = self.to_rgb(h)
        return out_img


if __name__ == "__main__":
    netG = G_NET()
    dummy_input = torch.randn(1, 100, 1, 1)
    output = netG(dummy_input)
    print(f"Generator Initialized. Output shape: {output.shape}")
    assert output.shape == (1, 3, 256, 256), "Error: Output must be 256x256!"
    print("Success: Spatial dimensions verified at 256x256.")
