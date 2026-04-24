"""Generate images from text prompts using a trained AttenX model.

This script is intentionally strict: it refuses to run with random text conditioning
because that produces noise-like outputs.

Usage:
    python scripts/generate.py --checkpoint ./checkpoints/checkpoint_latest.pth \
        --damsm-text-path ./checkpoints/damsm/damsm_latest.pth \
        --wordtoix-path ./data/birds/captions.pickle \
        --prompts "a yellow bird" "a red cardinal"
"""

import argparse
import os
import pickle
import re
import sys

import numpy as np
import torch
from PIL import Image

# Ensure local repository root is searched before stdlib module names.
REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if REPO_ROOT not in sys.path:
    sys.path.insert(0, REPO_ROOT)

from code.generator import G_NET
from code.encoder import RNN_ENCODER


def resolve_device(device_arg):
    if device_arg == "auto":
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")
    if device_arg.startswith("cuda") and not torch.cuda.is_available():
        raise RuntimeError(
            "CUDA was requested but is not available in this Python environment. "
            "Install a CUDA-enabled PyTorch build or use --device cpu."
        )
    return torch.device(device_arg)


def _load_state_dict_checked(model, state_dict, model_name, allow_partial_load=False):
    missing, unexpected = model.load_state_dict(state_dict, strict=False)
    if (missing or unexpected) and not allow_partial_load:
        raise RuntimeError(
            f"{model_name} checkpoint mismatch. Missing keys: {len(missing)}, "
            f"unexpected keys: {len(unexpected)}. "
            "Use --allow-partial-load only if you fully understand the mismatch."
        )
    if missing or unexpected:
        print(
            f"Warning: {model_name} loaded with partial key mismatch "
            f"(missing={len(missing)}, unexpected={len(unexpected)})."
        )


def _load_wordtoix(wordtoix_path):
    if not wordtoix_path or not os.path.isfile(wordtoix_path):
        raise FileNotFoundError(
            f"Word-to-index source not found: {wordtoix_path}. "
            "Expected AttnGAN metadata file (e.g., data/birds/captions.pickle)."
        )

    with open(wordtoix_path, "rb") as f:
        raw = pickle.load(f, encoding="latin1")

    if isinstance(raw, dict) and "wordtoix" in raw and isinstance(raw["wordtoix"], dict):
        return raw["wordtoix"]

    if isinstance(raw, (list, tuple)) and len(raw) >= 4 and isinstance(raw[3], dict):
        return raw[3]

    raise ValueError(
        f"Unsupported wordtoix format in {wordtoix_path}. "
        "Expected dict['wordtoix'] or tuple/list with wordtoix at index 3."
    )


def tokenize_prompt(text, wordtoix, seq_len=18):
    """Tokenize prompt using dataset vocabulary mapping.

    This keeps inference tokenization aligned with training metadata.
    """
    words = re.findall(r"[a-zA-Z']+", text.lower())
    if not words:
        words = ["unk"]

    unk_id = wordtoix.get("<unk>", wordtoix.get("unk", 0))
    tokens = [wordtoix.get(w, unk_id) for w in words][:seq_len]
    cap_len = max(1, len(tokens))

    if len(tokens) < seq_len:
        tokens.extend([0] * (seq_len - len(tokens)))

    return torch.LongTensor(tokens), torch.LongTensor([cap_len])


def generate_images(
    prompts,
    checkpoint_path,
    damsm_text_path,
    wordtoix_path,
    ngf=64,
    nef=512,
    nhidden=256,
    nembed=256,
    nz=100,
    output_dir="results/attenx",
    seed=42,
    device_str="auto",
    seq_len=18,
    allow_partial_load=False,
):
    """Generate images from text prompts using trained generator + DAMSM text encoder."""
    if not os.path.isfile(checkpoint_path):
        raise FileNotFoundError(f"Generator checkpoint not found: {checkpoint_path}")
    if not damsm_text_path or not os.path.isfile(damsm_text_path):
        raise FileNotFoundError(
            "A valid DAMSM text encoder checkpoint is required for meaningful generation. "
            "Provide --damsm-text-path (e.g., ./checkpoints/damsm/damsm_latest.pth)."
        )

    wordtoix = _load_wordtoix(wordtoix_path)
    vocab_size = max(wordtoix.values()) + 1 if wordtoix else 10000

    device = resolve_device(device_str)
    print(f"Using device: {device}")

    word_dim = nhidden * 2
    netG = G_NET(ngf=ngf, nz=nz, nef=nef, nhidden=nhidden, word_dim=word_dim).to(device)
    text_encoder = RNN_ENCODER(n_words=vocab_size, nhidden=nhidden, nembed=nembed).to(device)

    state = torch.load(checkpoint_path, map_location=device)
    if not isinstance(state, dict) or "netG" not in state:
        raise ValueError(f"Invalid generator checkpoint format: {checkpoint_path}")
    _load_state_dict_checked(netG, state["netG"], "Generator", allow_partial_load)
    print(f"Loaded generator from {checkpoint_path}")

    ts = torch.load(damsm_text_path, map_location=device)
    if isinstance(ts, dict) and "text_encoder" in ts:
        text_state = ts["text_encoder"]
    elif isinstance(ts, dict) and "state_dict" in ts:
        text_state = ts["state_dict"]
    else:
        text_state = ts
    _load_state_dict_checked(text_encoder, text_state, "Text encoder", allow_partial_load)
    print(f"Loaded text encoder from {damsm_text_path}")

    text_encoder.eval()
    netG.eval()

    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)

    os.makedirs(output_dir, exist_ok=True)

    with torch.no_grad():
        for i, prompt in enumerate(prompts):
            print(f"  [{i + 1}/{len(prompts)}] Generating: '{prompt}'")

            caption, cap_len = tokenize_prompt(prompt, wordtoix, seq_len=seq_len)
            caption = caption.unsqueeze(0).to(device)
            cap_len = cap_len.to(device)

            hidden = None
            words_emb, sent_emb = text_encoder(caption, cap_len, hidden)
            noise = torch.randn(1, nz, 1, 1, device=device)
            _, _, img_256, _, _ = netG(noise, sent_emb, words_emb)

            # Generator output is tanh-scaled to [-1, 1]. Convert directly to [0, 255].
            img = img_256.squeeze(0).detach().cpu().clamp(-1, 1)
            img = ((img + 1.0) * 127.5).byte().numpy()
            img = np.transpose(img, (1, 2, 0))

            pil_img = Image.fromarray(img)
            filename = f"attenx_output_{i + 1:03d}.png"
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
    parser.add_argument(
        "--damsm-text-path",
        type=str,
        required=True,
        help="Path to pretrained DAMSM text encoder checkpoint",
    )
    parser.add_argument(
        "--wordtoix-path",
        type=str,
        default="./data/birds/captions.pickle",
        help="Path to metadata containing wordtoix mapping",
    )
    parser.add_argument("--prompts", nargs="+", default=None, help="Text prompts")
    parser.add_argument("--output-dir", type=str, default="results/attenx")
    parser.add_argument("--ngf", type=int, default=64)
    parser.add_argument("--nef", type=int, default=512)
    parser.add_argument("--nhidden", type=int, default=256)
    parser.add_argument("--nembed", type=int, default=256)
    parser.add_argument("--nz", type=int, default=100)
    parser.add_argument("--seq-len", type=int, default=18)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--device", type=str, default="auto")
    parser.add_argument(
        "--allow-partial-load",
        action="store_true",
        help="Allow missing/unexpected checkpoint keys (not recommended)",
    )
    args = parser.parse_args()

    prompts = args.prompts if args.prompts else default_prompts
    try:
        generate_images(
            prompts=prompts,
            checkpoint_path=args.checkpoint,
            damsm_text_path=args.damsm_text_path,
            wordtoix_path=args.wordtoix_path,
            ngf=args.ngf,
            nef=args.nef,
            nhidden=args.nhidden,
            nembed=args.nembed,
            nz=args.nz,
            output_dir=args.output_dir,
            seed=args.seed,
            device_str=args.device,
            seq_len=args.seq_len,
            allow_partial_load=args.allow_partial_load,
        )
    except Exception as exc:
        print(f"ERROR: {exc}")
        raise SystemExit(1)