"""Generate baseline outputs using a real baseline checkpoint.

This script no longer produces random placeholder images.
It delegates generation to scripts.generate with strict checkpoint/text-encoder checks.
"""

import argparse

try:
    from scripts.generate import generate_images
except Exception:
    from generate import generate_images


DEFAULT_PROMPTS = [
    "a bright yellow bird with a black head and wings",
    "a small red bird with a short beak and long tail",
    "a large blue bird sitting on a tree branch",
    "a white bird with black spots on its wings",
    "a green bird with a red chest and yellow eyes",
    "a brown bird with a long curved beak",
    "a black and white bird with a sharp beak",
    "a colorful bird with a long tail and blue feathers",
    "a small bird with a yellow belly and grey wings",
    "a grey bird with a red crest on its head",
]


def run_baseline_inference(args):
    print("Establishing control group from a real baseline checkpoint...")
    prompts = args.prompts if args.prompts else DEFAULT_PROMPTS
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


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Run baseline checkpoint inference (no random placeholders).")
    parser.add_argument("--checkpoint", type=str, required=True, help="Path to baseline generator checkpoint")
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
    parser.add_argument("--output-dir", type=str, default="results/baseline")
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
    cli_args = parser.parse_args()
    run_baseline_inference(cli_args)
