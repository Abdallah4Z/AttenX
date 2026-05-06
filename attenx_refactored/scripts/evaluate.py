"""
AttenX evaluation script (Inception Score).

Loads a trained generator and generates images from test set
captions, then computes the Inception Score (IS) as a quality
and diversity metric.

Usage:
    python -m attenx_refactored.scripts.evaluate \
        --config configs/attenx_mhsa.yaml \
        --checkpoint checkpoints/attenx_mhsa/checkpoint_best.pth \
        --num-imgs 500
"""

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
    """
    Compute Inception Score (IS) for a set of generated images.

    Uses a pretrained Inception-v3 model to classify images,
    then computes KL divergence between conditional and marginal
    label distributions.

    Args:
        images: Tensor of generated images (N, 3, H, W) in [0, 1].
        batch_size: Batch size for Inception forward passes.
        splits: Number of splits for mean/std computation.

    Returns:
        Tuple of (mean IS, std IS).
    """
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
    """
    Parse command-line arguments for evaluation.

    Returns:
        Parsed argparse.Namespace.
    """
    parser = argparse.ArgumentParser(description="AttenX - Evaluate with Inception Score")
    parser.add_argument("--config", type=str, required=True, help="Path to YAML config file")
    parser.add_argument("--checkpoint", type=str, required=True, help="Path to generator checkpoint")
    parser.add_argument("--num-imgs", type=int, default=500, help="Number of images to generate")
    parser.add_argument("--batch-size", type=int, default=16, help="Batch size for IS computation")
    parser.add_argument("--splits", type=int, default=10, help="Number of splits for IS stats")
    parser.add_argument("--device", type=str, default="auto", help="Device (auto/cuda/cpu)")
    return parser.parse_args()


def main():
    """
    Main entry point: generate images and compute Inception Score.
    """
    args = parse_args()
    cfg = AttenXConfig.from_yaml(args.config)

    if args.device == "auto":
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    else:
        device = torch.device(args.device)

    word_dim = cfg.nhidden * 2
    nz = 100

    # Load generator
    netG = G_NET(ngf=cfg.ngf, nz=nz, nef=cfg.nef, word_dim=word_dim,
                 attention_mode=cfg.attention_mode, num_heads=cfg.num_heads).to(device)
    state = torch.load(args.checkpoint, map_location=device)
    netG.load_state_dict(state["netG"])
    netG.eval()
    print(f"Loaded generator from {args.checkpoint}")

    # Load text encoder
    text_encoder = RNN_ENCODER(cfg.vocab_size, cfg.nhidden, cfg.nembed).to(device)
    text_encoder.eval()

    # Load test dataset for captions
    dataset = TextImageDataset(data_dir=cfg.data_dir, split="test")
    loader = DataLoader(dataset, batch_size=1, shuffle=True, collate_fn=collate_fn)

    # Generate images from test captions
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
    all_imgs = (all_imgs + 1) / 2  # Denormalize to [0, 1]
    print(f"Generated {all_imgs.size(0)} images")

    # Compute and report Inception Score
    is_mean, is_std = inception_score(all_imgs, args.batch_size, args.splits)
    print(f"Inception Score: {is_mean:.4f} +/- {is_std:.4f}")


if __name__ == "__main__":
    main()
