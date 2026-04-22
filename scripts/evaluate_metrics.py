"""Compute FID and Inception Score (IS) for generated images.

Requires: torchvision, scipy, numpy
Install:  pip install scipy scikit-learn

Usage:
    python scripts/evaluate_metrics.py --checkpoint ./checkpoints/checkpoint_latest.pth \
        --data-dir ./data --num-images 1000 --batch-size 16
"""
import argparse
import os
import random
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision import models
from scipy import linalg


def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


class InceptionFeatureExtractor(nn.Module):
    def __init__(self):
        super().__init__()
        inception = models.inception_v3(pretrained=True, aux_logits=True)
        inception.eval()
        for p in inception.parameters():
            p.requires_grad = False
        self.inception = inception

    def forward(self, x):
        if x.shape[-1] != 299:
            x = F.interpolate(x, size=(299, 299), mode="bilinear", align_corners=False)
        return self.inception(x)


def compute_inception_score(probs, splits=10):
    scores = []
    N = probs.shape[0]
    split_size = N // splits
    for k in range(splits):
        part = probs[k * split_size : (k + 1) * split_size]
        py = np.mean(part, axis=0)
        kl_div = part * (np.log(part + 1e-16) - np.log(py + 1e-16))
        kl_div = np.mean(np.sum(kl_div, axis=1))
        scores.append(np.exp(kl_div))
    return float(np.mean(scores)), float(np.std(scores))


def compute_fid(mu_real, sigma_real, mu_fake, sigma_fake):
    diff = mu_real - mu_fake
    covmean, _ = linalg.sqrtm(sigma_real @ sigma_fake, disp=False)
    if np.iscomplexobj(covmean):
        covmean = covmean.real
    fid = diff @ diff + np.trace(sigma_real + sigma_fake - 2 * covmean)
    return float(fid)


@torch.no_grad()
def extract_features(dataloader, inception, device, num_images):
    features_list = []
    logits_list = []
    count = 0
    for batch in dataloader:
        imgs = batch[0].to(device) if isinstance(batch, (list, tuple)) else batch.to(device)
        remaining = num_images - count
        if remaining <= 0:
            break
        if imgs.size(0) > remaining:
            imgs = imgs[:remaining]
        out = inception(imgs)
        if isinstance(out, tuple):
            logits = out[0]
        else:
            logits = out
        probs = F.softmax(logits, dim=1).cpu().numpy()
        logits_list.append(probs)
        pool_features = logits
        features_list.append(pool_features)
        count += imgs.size(0)
        if count >= num_images:
            break
    all_probs = np.concatenate(logits_list, axis=0)[:num_images]
    return all_probs


def generate_images(netG, text_encoder, dataloader, device, num_images, nz):
    netG.eval()
    inception = InceptionFeatureExtractor().to(device)
    inception.eval()

    all_fake_probs = []
    count = 0

    for batch in dataloader:
        if count >= num_images:
            break

        imgs, captions, cap_lens, _, _ = batch
        batch_size = min(imgs.size(0), num_images - count)
        if batch_size <= 0:
            break

        imgs = imgs[:batch_size].to(device)
        captions = captions[:batch_size].to(device)
        cap_lens = cap_lens[:batch_size].to(device).squeeze(-1)

        cap_lens, sort_idx = torch.sort(cap_lens, descending=True)
        captions = captions[sort_idx]

        hidden = None
        words_emb, sent_emb = text_encoder(captions, cap_lens, hidden)
        word_emb = words_emb

        noise = torch.randn(batch_size, nz, 1, 1, device=device)
        _, _, img_256, _, _ = netG(noise, sent_emb, word_emb)

        probs = inception(img_256)
        if isinstance(probs, tuple):
            probs = probs[0]
        probs = F.softmax(probs, dim=1).cpu().numpy()
        all_fake_probs.append(probs)
        count += batch_size

    all_fake_probs = np.concatenate(all_fake_probs, axis=0)[:num_images]
    return all_fake_probs


def main():
    parser = argparse.ArgumentParser(description="Evaluate FID and IS")
    parser.add_argument("--checkpoint", type=str, required=True, help="Path to generator checkpoint")
    parser.add_argument("--data-dir", type=str, default="./data")
    parser.add_argument("--num-images", type=int, default=500)
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument("--num-workers", type=int, default=4)
    parser.add_argument("--device", type=str, default="auto")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--vocab-size", type=int, default=10000)
    parser.add_argument("--nhidden", type=int, default=256)
    parser.add_argument("--nembed", type=int, default=256)
    parser.add_argument("--nef", type=int, default=512)
    parser.add_argument("--ngf", type=int, default=64)
    parser.add_argument("--nz", type=int, default=100)
    parser.add_argument("--damsm-text-path", type=str, default=None)
    args = parser.parse_args()

    set_seed(args.seed)
    if args.device == "auto":
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    else:
        device = torch.device(args.device)

    from code.generator import G_NET
    from code.encoder import RNN_ENCODER

    word_dim = args.nhidden * 2
    netG = G_NET(ngf=args.ngf, nz=args.nz, nef=args.nef, nhidden=args.nhidden, word_dim=word_dim).to(device)
    text_encoder = RNN_ENCODER(n_words=args.vocab_size, nhidden=args.nhidden, nembed=args.nembed).to(device)

    state = torch.load(args.checkpoint, map_location=device)
    netG.load_state_dict(state["netG"], strict=False)
    print(f"Loaded G from {args.checkpoint}")

    if args.damsm_text_path:
        ts = torch.load(args.damsm_text_path, map_location=device)
        if isinstance(ts, dict) and "text_encoder" in ts:
            text_encoder.load_state_dict(ts["text_encoder"], strict=False)
        elif isinstance(ts, dict) and "state_dict" in ts:
            text_encoder.load_state_dict(ts["state_dict"], strict=False)
        else:
            text_encoder.load_state_dict(ts, strict=False)
        print(f"Loaded text encoder from {args.damsm_text_path}")

    text_encoder.eval()
    netG.eval()

    from code.datasets import TextImageDataset
    from torch.utils.data import DataLoader

    dataset = TextImageDataset(data_dir=args.data_dir, split="test", image_size=256)
    loader = DataLoader(dataset, batch_size=args.batch_size, shuffle=False,
                       num_workers=args.num_workers, pin_memory=True)

    print(f"Computing metrics over {args.num_images} images...")
    fake_probs = generate_images(netG, text_encoder, loader, device, args.num_images, args.nz)

    is_mean, is_std = compute_inception_score(fake_probs)
    print(f"\n=== Results ===")
    print(f"Inception Score: {is_mean:.4f} +/- {is_std:.4f}")
    print(f"(FID requires real statistics; use a separate script with pre-computed real stats)")
    print(f"For full FID computation, first extract real dataset statistics using:")
    print(f"  python scripts/extract_real_stats.py --data-dir {args.data_dir}")


if __name__ == "__main__":
    main()