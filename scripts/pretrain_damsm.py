"""Pretrain DAMSM encoders (text + image) using contrastive losses.

This must be run BEFORE the main GAN training. The learned encoders produce
word/sentence embeddings that the AttenX generator will use for text conditioning.

Usage:
    python scripts/pretrain_damsm.py --data-dir ./data --epochs 50 --batch-size 16
"""
import argparse
import os
import random
import numpy as np
import sys

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from torch.nn.utils.rnn import pad_sequence

# Ensure local repository root is searched before stdlib module names.
REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if REPO_ROOT not in sys.path:
    sys.path.insert(0, REPO_ROOT)

from code.datasets import TextImageDataset
from code.encoder import RNN_ENCODER, CNN_ENCODER
from code.losses import words_loss, sent_loss


def collate_text_image(batch):
    """Collate function to pad variable-length captions within a batch."""
    images, captions, cap_lens, cls_ids, keys = zip(*batch)
    images = torch.stack(images, 0)
    cap_lens = torch.stack(cap_lens, 0).squeeze(-1)
    captions = pad_sequence(captions, batch_first=True, padding_value=0)
    return images, captions, cap_lens, cls_ids, keys


def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
        torch.backends.cudnn.deterministic = False
        torch.backends.cudnn.benchmark = True


def resolve_device(device_arg):
    if device_arg == "auto":
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")
    if device_arg.startswith("cuda") and not torch.cuda.is_available():
        raise RuntimeError(
            "CUDA was requested but is not available in this Python environment. "
            "Install a CUDA-enabled PyTorch build or use --device cpu."
        )
    return torch.device(device_arg)


def save_encoders(text_encoder, image_encoder, output_dir, epoch):
    os.makedirs(output_dir, exist_ok=True)
    torch.save({
        "epoch": epoch,
        "text_encoder": text_encoder.state_dict(),
        "image_encoder": image_encoder.state_dict(),
    }, os.path.join(output_dir, f"damsm_epoch_{epoch:04d}.pth"))
    torch.save({
        "epoch": epoch,
        "text_encoder": text_encoder.state_dict(),
        "image_encoder": image_encoder.state_dict(),
    }, os.path.join(output_dir, "damsm_latest.pth"))
    print(f"DAMSM checkpoint saved at epoch {epoch + 1}")


def pretrain(args):
    set_seed(args.seed)
    device = resolve_device(args.device)
    print(f"Device: {device}")
    pin_memory = device.type == "cuda"

    dataset = TextImageDataset(data_dir=args.data_dir, split="train", image_size=256)
    loader = DataLoader(dataset, batch_size=args.batch_size, shuffle=True,
                        num_workers=args.num_workers, pin_memory=pin_memory, drop_last=True,
                        collate_fn=collate_text_image)
    print(f"Dataset size: {len(dataset)} | Batches: {len(loader)}")

    text_encoder = RNN_ENCODER(n_words=args.vocab_size, nhidden=args.nhidden, nembed=args.nembed).to(device)
    image_encoder = CNN_ENCODER(nef=args.nef).to(device)

    for p in image_encoder.parameters():
        if hasattr(p, 'requires_grad'):
            p.requires_grad = True

    optimizer = optim.Adam(
        list(text_encoder.parameters()) + list(image_encoder.parameters()),
        lr=args.lr, betas=(0.5, 0.999),
    )
    scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=10, gamma=0.5)

    start_epoch = 0
    if args.resume:
        latest_path = os.path.join(args.output_dir, "damsm_latest.pth")
        if os.path.isfile(latest_path):
            state = torch.load(latest_path, map_location=device)
            text_encoder.load_state_dict(state["text_encoder"])
            image_encoder.load_state_dict(state["image_encoder"])
            start_epoch = state.get("epoch", 0)
            print(f"Resumed DAMSM from epoch {start_epoch}")

    os.makedirs(args.output_dir, exist_ok=True)
    best_loss = float("inf")

    for epoch in range(start_epoch, args.epochs):
        text_encoder.train()
        image_encoder.train()
        epoch_loss = 0.0
        n_batches = 0

        for batch in loader:
            imgs, captions, cap_lens, _, _ = batch
            imgs = imgs.to(device)
            captions = captions.to(device)
            cap_lens = cap_lens.to(device).squeeze(-1)
            batch_size = imgs.size(0)

            cap_lens, sort_idx = torch.sort(cap_lens, descending=True)
            captions = captions[sort_idx]
            imgs = imgs[sort_idx]

            optimizer.zero_grad()

            hidden = None
            words_emb, sent_emb = text_encoder(captions, cap_lens, hidden)

            features, cnn_code = image_encoder(imgs)

            w_loss = words_loss(features, words_emb.transpose(1, 2), None, cap_lens, batch_size)
            s_loss = sent_loss(cnn_code, sent_emb, None, batch_size)
            total_loss = w_loss + s_loss

            total_loss.backward()
            torch.nn.utils.clip_grad_norm_(
                list(text_encoder.parameters()) + list(image_encoder.parameters()),
                max_norm=5.0,
            )
            optimizer.step()

            epoch_loss += total_loss.item()
            n_batches += 1

        scheduler.step()
        avg_loss = epoch_loss / n_batches
        print(f"Epoch {epoch + 1}/{args.epochs} | DAMSM Loss: {avg_loss:.4f}")

        if avg_loss < best_loss:
            best_loss = avg_loss
            save_encoders(text_encoder, image_encoder, args.output_dir, epoch)

        if (epoch + 1) % args.save_interval == 0:
            save_encoders(text_encoder, image_encoder, args.output_dir, epoch)

    print("DAMSM pretraining complete. Final encoders saved.")


def main():
    parser = argparse.ArgumentParser(description="Pretrain DAMSM encoders")
    parser.add_argument("--data-dir", type=str, default="./data")
    parser.add_argument("--epochs", type=int, default=50)
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument("--lr", type=float, default=2e-4)
    parser.add_argument("--nef", type=int, default=512)
    parser.add_argument("--nhidden", type=int, default=256)
    parser.add_argument("--nembed", type=int, default=256)
    parser.add_argument("--vocab-size", type=int, default=10000)
    parser.add_argument("--num-workers", type=int, default=4)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--device", type=str, default="auto")
    parser.add_argument("--output-dir", type=str, default="./checkpoints/damsm")
    parser.add_argument("--save-interval", type=int, default=10)
    parser.add_argument("--resume", action="store_true")
    args = parser.parse_args()
    pretrain(args)


if __name__ == "__main__":
    main()