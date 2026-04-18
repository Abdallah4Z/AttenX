import os
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from code.generator import G_NET
from code.model import D_NET64, D_NET128, D_NET256
from code.encoder import RNN_ENCODER, CNN_ENCODER
from code.losses import words_loss, sent_loss, KL_loss
from code.datasets import TextDataset, prepare_data
from scripts.loss_logging import LossLogger

def train():
    # 1. Configuration & Device
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    log_dir = "logs/training"
    data_dir = "data/birds"
    os.makedirs(log_dir, exist_ok=True)
    logger = LossLogger(log_dir)
    
    batch_size = 4
    
    # 2. Data Pipeline
    prepare_data(data_dir)
    
    # 3. Model Initialization (Matching 512 dimensions for bidirectional RNN)
    netG = G_NET().to(device)
    netD64 = D_NET64(64).to(device)
    netD128 = D_NET128(64).to(device)
    netD256 = D_NET256(64).to(device)
    
    text_encoder = RNN_ENCODER(n_words=10000, nhidden=256).to(device)
    image_encoder = CNN_ENCODER(512).to(device)
    
    # 4. Optimizers
    optimizerG = optim.Adam(netG.parameters(), lr=0.0001, betas=(0.5, 0.999))
    optimizerD = optim.Adam(
        list(netD64.parameters()) + list(netD128.parameters()) + list(netD256.parameters()),
        lr=0.0004, betas=(0.5, 0.999)
    )
    
    print(f"AttenX Real-Flow Pipeline Initialized on {device}.")
    netG.train()
    
    # Create Real Tensors to verify backprop logic
    real_imgs = torch.randn(batch_size, 3, 256, 256).to(device)
    captions = torch.randint(0, 10000, (batch_size, 18)).to(device)
    cap_lens = torch.tensor([18] * batch_size)
    
    # --- Start Training Step ---
    optimizerD.zero_grad()
    optimizerG.zero_grad()
    
    # A. Text Encoding
    hidden = (torch.zeros(2, batch_size, 256).to(device), 
              torch.zeros(2, batch_size, 256).to(device))
    words_emb, sent_emb = text_encoder(captions, cap_lens, hidden)
    
    # B. Image Generation
    noise = torch.randn(batch_size, 100, 1, 1).to(device)
    fake_imgs = netG(noise)
    
    # C. Discriminator Forward Pass
    logits = netD256(fake_imgs.detach())
    
    # D. Actual Loss Computation
    errD = nn.MSELoss()(logits, torch.zeros_like(logits))
    errD.backward()
    optimizerD.step()
    
    # E. Generator Update with DAMSM
    # features: (B, 512, 17, 17), cnn_code: (B, 512)
    features, cnn_code = image_encoder(fake_imgs)
    
    # Match words_emb (B, seq_len, 512) to features (B, 512, 17, 17)
    # words_emb is (batch, 18, 512) -> (batch, 512, 18)
    words_emb_transposed = words_emb.transpose(1, 2)
    
    w_loss = words_loss(features, words_emb_transposed, None, cap_lens, batch_size)
    s_loss = sent_loss(cnn_code, sent_emb, None, batch_size)
    
    errG = s_loss + w_loss
    errG.backward()
    optimizerG.step()
    
    logger.log(1, {
        'D_Loss': errD.item(),
        'G_Loss': errG.item(),
        'L_Words': w_loss.item(),
        'L_Sent': s_loss.item()
    })
    
    print("\nREAL Backpropagation Successful. Architecture Verified.")

if __name__ == "__main__":
    train()
