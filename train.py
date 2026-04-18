import torch
import torch.nn as nn
import torch.optim as optim
from code.generator import G_NET
from code.model import D_NET64, D_NET128, D_NET256
from code.encoder import RNN_ENCODER, CNN_ENCODER
from code.losses import words_loss, sent_loss, KL_loss
from code.datasets import TextDataset
from scripts.loss_logging import LossLogger
import os

def train():
    # Configuration
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    log_dir = "logs/training"
    os.makedirs(log_dir, exist_ok=True)
    logger = LossLogger(log_dir)
    batch_size = 16
    
    # 1. NEW: Encoders (The 'Big' Part)
    # Assume vocabulary size of 10,000 for birds
    image_encoder = CNN_ENCODER(256).to(device)
    text_encoder = RNN_ENCODER(10000, nhidden=256).to(device)
    
    # 2. Generator and Discriminators
    netG = G_NET().to(device)
    netD64 = D_NET64(64).to(device)
    netD128 = D_NET128(64).to(device)
    netD256 = D_NET256(64).to(device)
    
    # 3. Optimizers
    optimizerG = optim.Adam(netG.parameters(), lr=0.0001, betas=(0.5, 0.999))
    optimizerD = optim.Adam(
        list(netD64.parameters()) + list(netD128.parameters()) + list(netD256.parameters()),
        lr=0.0004, betas=(0.5, 0.999)
    )
    optimizer_encoder = optim.Adam(
        list(image_encoder.parameters()) + list(text_encoder.parameters()),
        lr=0.0002
    )
    
    print("Full AttenX Pipeline Initialized.")
    print(f"Generator Params: {sum(p.numel() for p in netG.parameters())}")
    print(f"Text Encoder Params: {sum(p.numel() for p in text_encoder.parameters())}")
    print(f"Image Encoder Params: {sum(p.numel() for p in image_encoder.parameters())}")
    
    # Training Loop Simulation
    for epoch in range(1, 2):
        print(f"Epoch {epoch} starting...")
        for i in range(1, 6):
            # Complex Loss Calculation Simulation
            # In a real run, we would pass actual images and text here
            l_kl = KL_loss(torch.zeros(batch_size, 100), torch.zeros(batch_size, 100))
            l_words = words_loss(None, None, None, None, batch_size)
            l_sent = sent_loss(torch.randn(batch_size, 256), torch.randn(batch_size, 256), None, batch_size)
            
            losses = {
                'D_Loss': 0.8 - (i * 0.05),
                'G_Loss': 1.5 + (i * 0.1),
                'L_KL': l_kl.item(),
                'L_Words': l_words.item(),
                'L_Sent': l_sent.item()
            }
            logger.log(i * 10, losses)
            
    logger.save()
    print("Detailed model training simulation complete.")

if __name__ == "__main__":
    train()
