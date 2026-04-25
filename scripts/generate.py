"""Generate images from text prompts using a trained AttenX model.

Usage:
    python scripts/generate.py --checkpoint ./checkpoints/checkpoint_latest.pth \
        --damsm-text-path ./checkpoints/damsm/damsm_latest.pth \
        --prompts "a yellow bird" "a red cardinal"
"""
import argparse
import os
import numpy as np
import torch
from PIL import Image

from code.generator import G_NET
from code.encoder import RNN_ENCODER


def tokenize_prompt(text, vocab_size=10000, seq_len=18):
    """Simple character-level tokenization for inference without a pre-built vocab.

    In production, use the same tokenizer used during dataset preparation.
    This stub hashes characters to token IDs within vocab range.
    """
    tokens = []
    for ch in text.lower():
        tokens.append(ord(ch) % vocab_size)
    while len(tokens) < seq_len:
        tokens.append(0)
    tokens = tokens[:seq_len]
    return torch.LongTensor(tokens), torch.LongTensor([min(len(text.split()), seq_len)])


def generate_images(prompts, checkpoint_path, damsm_text_path=None,
                    ngf=64, nef=512, nhidden=256, nembed=256,
                    vocab_size=10000, nz=100, output_dir="results/attenx",
                    seed=42, device_str="auto"):
    """Generate images from text prompts using the AttenX generator."""
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu") if device_str == "auto" else torch.device(device_str)

    word_dim = nhidden * 2
    netG = G_NET(ngf=ngf, nz=nz, nef=nef, nhidden=nhidden, word_dim=word_dim).to(device)
    text_encoder = RNN_ENCODER(n_words=vocab_size, nhidden=nhidden, nembed=nembed).to(device)

    state = torch.load(checkpoint_path, map_location=device)
    netG.load_state_dict(state["netG"], strict=False)
    print(f"Loaded generator from {checkpoint_path}")

    if damsm_text_path and os.path.isfile(damsm_text_path):
        ts = torch.load(damsm_text_path, map_location=device)
        if isinstance(ts, dict) and "text_encoder" in ts:
            text_encoder.load_state_dict(ts["text_encoder"], strict=False)
        elif isinstance(ts, dict) and "state_dict" in ts:
            text_encoder.load_state_dict(ts["state_dict"], strict=False)
        else:
            text_encoder.load_state_dict(ts, strict=False)
        print(f"Loaded text encoder from {damsm_text_path}")
    else:
        print("WARNING: No pretrained text encoder provided. Using random text embeddings.")
        print("         Images will NOT be semantically meaningful without pretrained DAMSM encoders.")
        print("         Run 'python scripts/pretrain_damsm.py' first to train the text encoder.")

    text_encoder.eval()
    netG.eval()

    torch.manual_seed(seed)

    os.makedirs(output_dir, exist_ok=True)

    with torch.no_grad():
        for i, prompt in enumerate(prompts):
            print(f"  [{i+1}/{len(prompts)}] Generating: '{prompt}'")

            caption, cap_len = tokenize_prompt(prompt, vocab_size)
            caption = caption.unsqueeze(0).to(device)
            cap_len = cap_len.cpu()

            hidden = None
            words_emb, sent_emb = text_encoder(caption, cap_len, hidden)
            word_emb = words_emb

            noise = torch.randn(1, nz, 1, 1, device=device)
            _, _, img_256, _, _ = netG(noise, sent_emb, word_emb)

            img = img_256.squeeze(0).cpu().float().numpy()
            img = (img - img.min()) / (img.max() - img.min() + 1e-8)
            img = (img * 255).astype(np.uint8)
            img = np.transpose(img, (1, 2, 0))

            pil_img = Image.fromarray(img)
            filename = f"attenx_output_{i+1:03d}.png"
            save_path = os.path.join(output_dir, filename)
            pil_img.save(save_path, "PNG")

            file_size = os.path.getsize(save_path) / 1024
            print(f"       -> Saved ({file_size:.1f} KB) to {save_path}")

    print(f"\nDone. {len(prompts)} images saved to {output_dir}/")


if __name__ == "__main__":
    default_prompts = [
        "a bright yellow bird with a black head and wings",
        "a small red bird with a short beak and long tail",
        "a large blue bird sitting on a tree branch",
        "a white bird with black spots on its wings",
        "a green bird with a red chest and yellow eyes",
    ]

    parser = argparse.ArgumentParser(description="Generate images from text prompts")
    parser.add_argument("--checkpoint", type=str, required=True, help="Path to generator checkpoint")
    parser.add_argument("--damsm-text-path", type=str, default=None, help="Path to pretrained text encoder")
    parser.add_argument("--prompts", nargs="+", default=None, help="Text prompts")
    parser.add_argument("--output-dir", type=str, default="results/attenx")
    parser.add_argument("--ngf", type=int, default=64)
    parser.add_argument("--nef", type=int, default=512)
    parser.add_argument("--nhidden", type=int, default=256)
    parser.add_argument("--nembed", type=int, default=256)
    parser.add_argument("--vocab-size", type=int, default=10000)
    parser.add_argument("--nz", type=int, default=100)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    prompts = args.prompts if args.prompts else default_prompts
    generate_images(
        prompts=prompts,
        checkpoint_path=args.checkpoint,
        damsm_text_path=args.damsm_text_path,
        ngf=args.ngf, nef=args.nef, nhidden=args.nhidden, nembed=args.nembed,
        vocab_size=args.vocab_size, nz=args.nz,
        output_dir=args.output_dir, seed=args.seed,
    )