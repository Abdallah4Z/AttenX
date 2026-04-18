import torch
import torch.nn as nn
from code.modules import SelfAttention

class G_NET(nn.Module):
    def __init__(self, ngf=64):
        super(G_NET, self).__init__()
        self.ngf = ngf
        
        # Stage 0: 64x64
        self.stage0 = nn.Sequential(
            nn.ConvTranspose2d(100, ngf * 8, 4, 1, 0, bias=False),
            nn.BatchNorm2d(ngf * 8),
            nn.ReLU(True)
        )
        
        # Stage 1: 128x128
        self.stage1 = nn.Sequential(
            nn.Upsample(scale_factor=2, mode='nearest'),
            nn.Conv2d(ngf * 8, ngf * 4, 3, 1, 1, bias=False),
            nn.BatchNorm2d(ngf * 4),
            nn.ReLU(True)
        )
        
        # INJECTED: Self-Attention after 128x128 stage
        self.attn_stage1 = SelfAttention(ngf * 4)
        
        # Stage 2: 256x256
        self.stage2 = nn.Sequential(
            nn.Upsample(scale_factor=2, mode='nearest'),
            nn.Conv2d(ngf * 4, ngf * 2, 3, 1, 1, bias=False),
            nn.BatchNorm2d(ngf * 2),
            nn.ReLU(True)
        )
        
        # Final image output
        self.to_rgb = nn.Sequential(
            nn.Conv2d(ngf * 2, 3, 3, 1, 1, bias=False),
            nn.Tanh()
        )

    def forward(self, z):
        h0 = self.stage0(z)
        h1 = self.stage1(h0)
        h1_attn = self.attn_stage1(h1) # Self-Attention Applied
        h2 = self.stage2(h1_attn)
        out_img = self.to_rgb(h2)
        return out_img

if __name__ == "__main__":
    # Test initialization and forward pass
    netG = G_NET()
    print("Generator initialized successfully.")
    print(netG)
    
    dummy_input = torch.randn(1, 100, 1, 1)
    output = netG(dummy_input)
    print(f"Forward pass successful. Output shape: {output.shape}")
