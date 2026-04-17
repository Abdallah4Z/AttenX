import torch
import torch.nn as nn

class SelfAttention(nn.Module):
    """
    Self-Attention module for enhancing global coherence.
    Reference: Non-local Neural Networks
    """
    def __init__(self, in_dim):
        super(SelfAttention, self).__init__()
        self.chanel_in = in_dim
        
        self.query_conv = nn.Conv2d(in_channels=in_dim, out_channels=in_dim // 8, kernel_size=1)
        self.key_conv = nn.Conv2d(in_channels=in_dim, out_channels=in_dim // 8, kernel_size=1)
        self.value_conv = nn.Conv2d(in_channels=in_dim, out_channels=in_dim, kernel_size=1)
        self.gamma = nn.Parameter(torch.zeros(1))

        self.softmax = nn.Softmax(dim=-1)

    def forward(self, x):
        """
        inputs :
            x : input feature maps( B X C X H X W)
        returns :
            out : self attention value + input feature
            attention: B X N X N (N is H*W)
        """
        m_batchsize, C, height, width = x.size()
        proj_query = self.query_conv(x).view(m_batchsize, -1, width * height).permute(0, 2, 1) # B X CX(N)
        proj_key = self.key_conv(x).view(m_batchsize, -1, width * height) # B X C X (*W*H)
        energy = torch.bmm(proj_query, proj_key) # transpose check
        attention = self.softmax(energy) # BX (N) X (N) 
        proj_value = self.value_conv(x).view(m_batchsize, -1, width * height) # B X C X N

        out = torch.bmm(proj_value, attention.permute(0, 2, 1))
        out = out.view(m_batchsize, C, height, width)

        out = self.gamma * out + x
        return out

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

    def forward(self, z):
        h0 = self.stage0(z)
        h1 = self.stage1(h0)
        h1_attn = self.attn_stage1(h1) # Self-Attention Applied
        h2 = self.stage2(h1_attn)
        return h2

if __name__ == "__main__":
    # Test initialization and forward pass
    netG = G_NET()
    print("Generator initialized successfully.")
    print(netG)
    
    dummy_input = torch.randn(1, 100, 1, 1)
    output = netG(dummy_input)
    print(f"Forward pass successful. Output shape: {output.shape}")
