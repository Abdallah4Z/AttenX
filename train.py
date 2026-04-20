import argparse
import os
from code.datasets import TextImageDataset
from code.encoder import CNN_ENCODER, RNN_ENCODER
from code.generator import G_NET
from code.losses import sent_loss, words_loss
from code.model import D_NET256

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader

from scripts.loss_logging import LossLogger


def _load_pretrained_if_available(model, checkpoint_path, name):
    """Load pretrained checkpoint into a model if a valid path is provided."""
    if not checkpoint_path:
        return False
    if not os.path.isfile(checkpoint_path):
        raise FileNotFoundError(f"{name} checkpoint not found: {checkpoint_path}")

    state = torch.load(checkpoint_path, map_location="cpu")
    if isinstance(state, dict) and "state_dict" in state:
        state = state["state_dict"]

    model.load_state_dict(state, strict=False)
    print(f"Loaded pretrained {name} weights from: {checkpoint_path}")
    return True


def save_checkpoint(checkpoint_dir, epoch, netG, netD, optimizerG, optimizerD, **extra):
    """Save training checkpoint."""
    os.makedirs(checkpoint_dir, exist_ok=True)
    checkpoint_path = os.path.join(checkpoint_dir, f"checkpoint_epoch_{epoch:04d}.pth")
    state = {
        "epoch": epoch,
        "netG": netG.state_dict(),
        "netD": netD.state_dict(),
        "optimizerG": optimizerG.state_dict(),
        "optimizerD": optimizerD.state_dict(),
        **extra,
    }
    torch.save(state, checkpoint_path)
    print(f"Checkpoint saved: {checkpoint_path}")

    # Also save latest as separate file for easy resume
    latest_path = os.path.join(checkpoint_dir, "checkpoint_latest.pth")
    torch.save(state, latest_path)


def load_checkpoint(checkpoint_dir, netG, netD, optimizerG=None, optimizerD=None):
    """Load latest checkpoint if available."""
    latest_path = os.path.join(checkpoint_dir, "checkpoint_latest.pth")
    if not os.path.isfile(latest_path):
        print(f"No checkpoint found at {latest_path}")
        return 0

    state = torch.load(latest_path, map_location="cpu")
    netG.load_state_dict(state["netG"])
    netD.load_state_dict(state["netD"])
    if optimizerG is not None:
        optimizerG.load_state_dict(state["optimizerG"])
    if optimizerD is not None:
        optimizerD.load_state_dict(state["optimizerD"])

    start_epoch = state.get("epoch", 0)
    print(f"Resumed from checkpoint epoch {start_epoch}")
    return start_epoch


def train(args):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    # Setup logging
    logger = LossLogger(args.log_dir)

    # Create checkpoint directory
    os.makedirs(args.checkpoint_dir, exist_ok=True)

    # 1. Dataset & DataLoader
    dataset = TextImageDataset(
        data_dir=args.data_dir,
        split="train",
        image_size=args.image_size,
    )
    dataloader = DataLoader(
        dataset,
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=args.num_workers,
        pin_memory=True,
        drop_last=True,
    )
    print(f"Dataset size: {len(dataset)} | Batches per epoch: {len(dataloader)}")

    # 2. Models Initialization
    netG = G_NET(ngf=args.ngf).to(device)
    netD = D_NET256(ndf=args.ndf, nef=args.nef).to(device)
    text_encoder = RNN_ENCODER(
        n_words=args.vocab_size, nhidden=args.nhidden, nembed=args.nembed
    ).to(device)
    image_encoder = CNN_ENCODER(nef=args.nef).to(device)

    # Load pretrained DAMSM encoders (frozen)
    _load_pretrained_if_available(
        text_encoder, args.damsm_text_path, "DAMSM text encoder"
    )
    _load_pretrained_if_available(
        image_encoder, args.damsm_image_path, "DAMSM image encoder"
    )
    text_encoder.eval()
    image_encoder.eval()
    for p in text_encoder.parameters():
        p.requires_grad = False
    for p in image_encoder.parameters():
        p.requires_grad = False

    # 3. Optimizers
    optimizerG = optim.Adam(netG.parameters(), lr=args.lr_g, betas=(0.5, 0.999))
    optimizerD = optim.Adam(netD.parameters(), lr=args.lr_d, betas=(0.5, 0.999))

    # 4. Optionally resume from checkpoint
    start_epoch = 0
    if args.resume:
        start_epoch = load_checkpoint(
            args.checkpoint_dir, netG, netD, optimizerG, optimizerD
        )

    # 5. Training Loop
    criterion = nn.BCELoss()
    global_step = start_epoch * len(dataloader)

    for epoch in range(start_epoch, args.epochs):
        netG.train()
        netD.train()

        for batch_idx, batch in enumerate(dataloader):
            real_imgs, captions, cap_lens, _, _ = batch
            real_imgs = real_imgs.to(device)
            captions = captions.to(device)
            cap_lens = cap_lens.to(device).squeeze(-1)

            batch_size = real_imgs.size(0)

            # Sort captions by length for LSTM packing
            cap_lens, sort_idx = torch.sort(cap_lens, descending=True)
            captions = captions[sort_idx]

            # Text encoding
            hidden = None  # Let LSTM initialize hidden state
            words_emb, sent_emb = text_encoder(captions, cap_lens, hidden)
            # words_emb: (batch, seq_len, nhidden*2)
            # sent_emb: (batch, nhidden*2)

            # Detach sent_emb for discriminator updates
            sent_emb_detached = sent_emb.detach()

            # ------------------------
            # Discriminator Update
            # ------------------------
            for d_step in range(args.D_steps):
                optimizerD.zero_grad()

                # Real images
                real_logits = netD(real_imgs, sent_emb_detached)
                errD_real = criterion(real_logits, torch.ones_like(real_logits))

                # Fake images
                noise = torch.randn(batch_size, 100, 1, 1, device=device)
                fake_imgs = netG(noise)
                fake_logits = netD(fake_imgs.detach(), sent_emb_detached)
                errD_fake = criterion(fake_logits, torch.zeros_like(fake_logits))

                errD = errD_real + errD_fake
                errD.backward()
                optimizerD.step()

            # ------------------------
            # Generator Update
            # ------------------------
            optimizerG.zero_grad()

            # GAN loss
            fake_imgs = netG(noise)
            gan_logits = netD(fake_imgs, sent_emb.detach())
            g_gan_loss = criterion(gan_logits, torch.ones_like(gan_logits))

            # DAMSM loss on generated images
            features, cnn_code = image_encoder(fake_imgs)
            w_loss = words_loss(
                features, words_emb.transpose(1, 2), None, cap_lens, batch_size
            )
            s_loss = sent_loss(cnn_code, sent_emb, None, batch_size)
            damsm_loss = w_loss + s_loss

            errG = g_gan_loss + args.gamma_damsm * damsm_loss
            errG.backward()
            optimizerG.step()

            # Logging
            global_step += 1
            logger.log(
                global_step,
                {
                    "D_Loss": errD.item(),
                    "G_Loss": errG.item(),
                    "G_GAN": g_gan_loss.item(),
                    "L_Words": w_loss.item(),
                    "L_Sent": s_loss.item(),
                    "L_DAMSM": damsm_loss.item(),
                    "Gamma_DAMSM": args.gamma_damsm,
                },
            )

        # End of epoch
        print(f"\nEpoch {epoch + 1}/{args.epochs} complete.")

        # Save checkpoint every N epochs
        if (epoch + 1) % args.checkpoint_interval == 0:
            save_checkpoint(
                args.checkpoint_dir,
                epoch + 1,
                netG,
                netD,
                optimizerG,
                optimizerD,
            )

        # Always save latest checkpoint
        save_checkpoint(
            args.checkpoint_dir,
            epoch + 1,
            netG,
            netD,
            optimizerG,
            optimizerD,
        )

    # Save final log
    logger.save()
    print("Training complete.")


def parse_args():
    parser = argparse.ArgumentParser(description="Multi-epoch AttenX training pipeline")
    # Dataset args
    parser.add_argument(
        "--data-dir", type=str, default="./data", help="Dataset root directory"
    )
    parser.add_argument(
        "--image-size", type=int, default=256, help="Image size for training"
    )
    parser.add_argument("--num-workers", type=int, default=4, help="DataLoader workers")

    # Model args
    parser.add_argument(
        "--ngf", type=int, default=64, help="Generator base channel count"
    )
    parser.add_argument(
        "--ndf", type=int, default=64, help="Discriminator base channel count"
    )
    parser.add_argument(
        "--nef", type=int, default=512, help="Text/Image encoder feature dim"
    )
    parser.add_argument("--nhidden", type=int, default=256, help="RNN hidden size")
    parser.add_argument("--nembed", type=int, default=256, help="Word embedding dim")
    parser.add_argument("--vocab-size", type=int, default=10000, help="Vocabulary size")

    # Training args
    parser.add_argument(
        "--epochs", type=int, default=100, help="Number of training epochs"
    )
    parser.add_argument("--batch-size", type=int, default=4, help="Batch size")
    parser.add_argument(
        "--lr-g", type=float, default=1e-4, help="Generator learning rate"
    )
    parser.add_argument(
        "--lr-d", type=float, default=4e-4, help="Discriminator learning rate"
    )
    parser.add_argument(
        "--D-steps", type=int, default=1, help="Discriminator steps per generator step"
    )
    parser.add_argument(
        "--gamma-damsm", type=float, default=1.0, help="DAMSM loss weight"
    )

    # Checkpoint/logging args
    parser.add_argument(
        "--checkpoint-dir",
        type=str,
        default="./checkpoints",
        help="Checkpoint directory",
    )
    parser.add_argument(
        "--checkpoint-interval",
        type=int,
        default=10,
        help="Save checkpoint every N epochs",
    )
    parser.add_argument(
        "--log-dir", type=str, default="./logs/training", help="Log directory"
    )
    parser.add_argument(
        "--resume", action="store_true", help="Resume from latest checkpoint"
    )

    # Pretrained DAMSM encoder paths
    parser.add_argument(
        "--damsm-text-path",
        type=str,
        default=os.getenv("DAMSM_TEXT_ENCODER_PATH"),
        help="Pretrained DAMSM text encoder checkpoint",
    )
    parser.add_argument(
        "--damsm-image-path",
        type=str,
        default=os.getenv("DAMSM_IMAGE_ENCODER_PATH"),
        help="Pretrained DAMSM image encoder checkpoint",
    )

    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()
    train(args)
