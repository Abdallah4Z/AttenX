import argparse
import os

import torch
from torchvision.utils import save_image

from attenx_refactored.config import AttenXConfig
from attenx_refactored.models.encoders import RNN_ENCODER
from attenx_refactored.models.generator import G_NET


def parse_args():
    parser = argparse.ArgumentParser(description="AttenX - Generation")
    parser.add_argument("--config", type=str, required=True)
    parser.add_argument("--checkpoint", type=str, required=True)
    parser.add_argument("--captions", type=str, nargs="+", required=True)
    parser.add_argument("--output-dir", type=str, default="./generated")
    parser.add_argument("--nz", type=int, default=100)
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
    netG = G_NET(ngf=cfg.ngf, nz=args.nz, nef=cfg.nef, word_dim=word_dim,
                 attention_mode=cfg.attention_mode, num_heads=cfg.num_heads).to(device)
    state = torch.load(args.checkpoint, map_location=device)
    netG.load_state_dict(state["netG"])
    netG.eval()

    text_encoder = RNN_ENCODER(cfg.vocab_size, cfg.nhidden, cfg.nembed).to(device)
    text_encoder.eval()

    os.makedirs(args.output_dir, exist_ok=True)

    with torch.no_grad():
        for i, caption in enumerate(args.captions):
            tokens = torch.randint(0, cfg.vocab_size, (1, 18), device=device)
            cap_len = torch.tensor([18], device=device)
            words_emb, sent_emb = text_encoder(tokens, cap_len, None)

            noise = torch.randn(1, args.nz, 1, 1, device=device)
            _, _, img_256, _, _ = netG(noise, sent_emb, words_emb)

            img = (img_256 + 1) / 2
            path = os.path.join(args.output_dir, f"sample_{i:03d}.png")
            save_image(img, path)
            print(f"Saved: {path}")


if __name__ == "__main__":
    main()
