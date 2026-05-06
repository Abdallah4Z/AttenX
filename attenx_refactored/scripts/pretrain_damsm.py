"""
DAMSM (Deep Attentional Multimodal Similarity Model) pretraining.

Trains the text encoder (bidirectional LSTM) and image encoder
(Inception-v3) jointly using word-level and sentence-level
contrastive losses. These pretrained encoders are then frozen
during GAN training.

Usage:
    python -m attenx_refactored.scripts.pretrain_damsm \
        --data-dir ./data --epochs 5
"""

import argparse
import os

import torch
import torch.optim as optim
from torch.utils.data import DataLoader

from attenx_refactored.data.dataset import TextImageDataset
from attenx_refactored.losses.damsm import words_loss, sent_loss
from attenx_refactored.models.encoders import RNN_ENCODER, CNN_ENCODER
from attenx_refactored.utils.collate import collate_fn


def parse_args():
    """
    Parse command-line arguments for DAMSM pretraining.

    Returns:
        Parsed argparse.Namespace.
    """
    parser = argparse.ArgumentParser(description="DAMSM Pretraining - text and image encoders")
    parser.add_argument("--data-dir", type=str, default="./data", help="Dataset root directory")
    parser.add_argument("--batch-size", type=int, default=16, help="Batch size")
    parser.add_argument("--epochs", type=int, default=5, help="Number of epochs")
    parser.add_argument("--lr", type=float, default=2e-4, help="Learning rate")
    parser.add_argument("--nef", type=int, default=512, help="Encoder embedding dimension")
    parser.add_argument("--nhidden", type=int, default=256, help="Text encoder hidden dimension")
    parser.add_argument("--nembed", type=int, default=256, help="Word embedding dimension")
    parser.add_argument("--vocab-size", type=int, default=10000, help="Vocabulary size")
    parser.add_argument("--num-workers", type=int, default=4, help="DataLoader workers")
    parser.add_argument("--seed", type=int, default=42, help="Random seed")
    parser.add_argument("--device", type=str, default="auto", help="Device (auto/cuda/cpu)")
    parser.add_argument("--checkpoint-dir", type=str, default="./checkpoints/damsm",
                        help="Directory to save encoder checkpoints")
    return parser.parse_args()


def main():
    """
    Main entry point: train text and image encoders with DAMSM loss.
    """
    args = parse_args()

    if args.device == "auto":
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    else:
        device = torch.device(args.device)
    print(f"Device: {device}")

    # Load dataset
    dataset = TextImageDataset(data_dir=args.data_dir, split="train")
    loader = DataLoader(dataset, batch_size=args.batch_size, shuffle=True,
                        num_workers=args.num_workers, drop_last=True, collate_fn=collate_fn)
    print(f"Dataset: {len(dataset)} images, {len(loader)} batches")

    # Initialize encoders
    text_encoder = RNN_ENCODER(args.vocab_size, args.nhidden, args.nembed).to(device)
    image_encoder = CNN_ENCODER(args.nef).to(device)

    # Optimizers (separate for each encoder)
    optimizerT = optim.Adam(text_encoder.parameters(), lr=args.lr, betas=(0.5, 0.999))
    optimizerI = optim.Adam(image_encoder.parameters(), lr=args.lr, betas=(0.5, 0.999))

    os.makedirs(args.checkpoint_dir, exist_ok=True)

    # Training loop
    for epoch in range(args.epochs):
        text_encoder.train()
        image_encoder.train()

        total_loss = 0.0
        total_w = 0.0
        total_s = 0.0

        for batch in loader:
            real_imgs, captions, cap_lens, _, _ = batch
            real_imgs = real_imgs.to(device)
            captions = captions.to(device)

            # Sort captions by length for packed LSTM
            cap_lens_sorted, sort_idx = torch.sort(cap_lens.squeeze(-1), descending=True)
            captions = captions[sort_idx]
            real_imgs = real_imgs[sort_idx]
            batch_size = real_imgs.size(0)

            # Forward through encoders
            words_emb, sent_emb = text_encoder(captions, cap_lens_sorted, None)
            features, cnn_code = image_encoder(real_imgs)

            # DAMSM losses
            w_loss = words_loss(features, words_emb.transpose(1, 2), None, cap_lens_sorted, batch_size)
            s_loss = sent_loss(cnn_code, sent_emb, None, batch_size)
            loss = w_loss + s_loss

            # Backward and optimize
            optimizerT.zero_grad()
            optimizerI.zero_grad()
            loss.backward()
            optimizerT.step()
            optimizerI.step()

            total_loss += loss.item()
            total_w += w_loss.item()
            total_s += s_loss.item()

        avg_loss = total_loss / len(loader)
        avg_w = total_w / len(loader)
        avg_s = total_s / len(loader)
        print(f"Epoch {epoch+1}/{args.epochs}  Loss: {avg_loss:.4f}  (W: {avg_w:.4f}  S: {avg_s:.4f})")

        # Save checkpoint
        ckpt = os.path.join(args.checkpoint_dir, f"damsm_epoch_{epoch+1:02d}.pth")
        torch.save({
            "epoch": epoch + 1,
            "text_encoder": text_encoder.state_dict(),
            "image_encoder": image_encoder.state_dict(),
            "optimizerT": optimizerT.state_dict(),
            "optimizerI": optimizerI.state_dict(),
        }, ckpt)
        print(f"Saved: {ckpt}")

    print("DAMSM pretraining complete.")


if __name__ == "__main__":
    main()
