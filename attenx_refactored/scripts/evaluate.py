import argparse
import os

import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader
from torchvision.models import inception_v3

from attenx_refactored.config import AttenXConfig
from attenx_refactored.data.dataset import TextImageDataset
from attenx_refactored.models.encoders import RNN_ENCODER
from attenx_refactored.models.generator import G_NET
from attenx_refactored.utils.collate import collate_fn


def inception_score(images, batch_size=32, splits=10):
    device = images.device
    model = inception_v3(pretrained=True, transform_input=False).to(device)
    model.eval()

    n = images.size(0)
    preds = []
    with torch.no_grad():
        for i in range(0, n, batch_size):
            batch = images[i:i + batch_size]
            if batch.size(-1) != 299:
                batch = F.interpolate(batch, size=(299, 299), mode="bilinear", align_corners=False)
            logits = model(batch)
            preds.append(F.softmax(logits, dim=-1).cpu())
    preds = torch.cat(preds, 0)

    scores = []
    split_size = n // splits
    for i in range(splits):
        part = preds[i * split_size: (i + 1) * split_size]
        kl = part * (part.log() - part.mean(0, keepdim=True).log())
        kl = kl.sum(1).mean().exp()
        scores.append(kl.item())

    mean = torch.tensor(scores).mean().item()
    std = torch.tensor(scores).std().item()
    return mean, std


def parse_args():
    parser = argparse.ArgumentParser(description="AttenX - Evaluation")
    parser.add_argument("--config", type=str, required=True)
    parser.add_argument("--checkpoint", type=str, required=True)
    parser.add_argument("--num-imgs", type=int, default=500)
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument("--splits", type=int, default=10)
    parser.add_argument("--device", type=str, default="auto")
    return parser.parse_args()


def main():
    args = parse_args()
    cfg = AttenXConfig.from_yaml(args.config)

    if args.device == "auto":
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    else:
        device = torch.device(args.device)

    word_dim = cfg.nhidden * 2
    nz = 100

    netG = G_NET(ngf=cfg.ngf, nz=nz, nef=cfg.nef, word_dim=word_dim,
                 attention_mode=cfg.attention_mode, num_heads=cfg.num_heads).to(device)
    state = torch.load(args.checkpoint, map_location=device)
    netG.load_state_dict(state["netG"])
    netG.eval()
    print(f"Loaded generator from {args.checkpoint}")

    text_encoder = RNN_ENCODER(cfg.vocab_size, cfg.nhidden, cfg.nembed).to(device)
    text_encoder.eval()

    dataset = TextImageDataset(data_dir=cfg.data_dir, split="test")
    loader = DataLoader(dataset, batch_size=1, shuffle=True, collate_fn=collate_fn)

    all_imgs = []
    count = 0
    with torch.no_grad():
        for batch in loader:
            if count >= args.num_imgs:
                break
            _, captions, cap_lens, _, _ = batch
            captions = captions.to(device)
            cap_lens_sorted, sort_idx = torch.sort(cap_lens.squeeze(-1), descending=True)
            captions = captions[sort_idx]
            words_emb, sent_emb = text_encoder(captions, cap_lens_sorted, None)

            noise = torch.randn(1, nz, 1, 1, device=device)
            _, _, img_256, _, _ = netG(noise, sent_emb, words_emb)
            all_imgs.append(img_256.cpu())
            count += 1

    all_imgs = torch.cat(all_imgs, 0)
    all_imgs = (all_imgs + 1) / 2
    print(f"Generated {all_imgs.size(0)} images")

    is_mean, is_std = inception_score(all_imgs, args.batch_size, args.splits)
    print(f"Inception Score: {is_mean:.4f} +/- {is_std:.4f}")


if __name__ == "__main__":
    main()
