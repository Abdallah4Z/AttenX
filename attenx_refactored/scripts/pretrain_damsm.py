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
    parser = argparse.ArgumentParser(description="DAMSM Pretraining")
    parser.add_argument("--data-dir", type=str, default="./data")
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument("--epochs", type=int, default=5)
    parser.add_argument("--lr", type=float, default=2e-4)
    parser.add_argument("--nef", type=int, default=512)
    parser.add_argument("--nhidden", type=int, default=256)
    parser.add_argument("--nembed", type=int, default=256)
    parser.add_argument("--vocab-size", type=int, default=10000)
    parser.add_argument("--num-workers", type=int, default=4)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--device", type=str, default="auto")
    parser.add_argument("--checkpoint-dir", type=str, default="./checkpoints/damsm")
    return parser.parse_args()


def main():
    args = parse_args()

    if args.device == "auto":
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    else:
        device = torch.device(args.device)
    print(f"Device: {device}")

    dataset = TextImageDataset(data_dir=args.data_dir, split="train")
    loader = DataLoader(dataset, batch_size=args.batch_size, shuffle=True,
                        num_workers=args.num_workers, drop_last=True, collate_fn=collate_fn)
    print(f"Dataset: {len(dataset)} images, {len(loader)} batches")

    text_encoder = RNN_ENCODER(args.vocab_size, args.nhidden, args.nembed).to(device)
    image_encoder = CNN_ENCODER(args.nef).to(device)

    optimizerT = optim.Adam(text_encoder.parameters(), lr=args.lr, betas=(0.5, 0.999))
    optimizerI = optim.Adam(image_encoder.parameters(), lr=args.lr, betas=(0.5, 0.999))

    os.makedirs(args.checkpoint_dir, exist_ok=True)

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

            cap_lens_sorted, sort_idx = torch.sort(cap_lens.squeeze(-1), descending=True)
            captions = captions[sort_idx]
            real_imgs = real_imgs[sort_idx]
            batch_size = real_imgs.size(0)

            words_emb, sent_emb = text_encoder(captions, cap_lens_sorted, None)
            features, cnn_code = image_encoder(real_imgs)

            w_loss = words_loss(features, words_emb.transpose(1, 2), None, cap_lens_sorted, batch_size)
            s_loss = sent_loss(cnn_code, sent_emb, None, batch_size)
            loss = w_loss + s_loss

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
