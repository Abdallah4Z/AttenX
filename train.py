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
from torch.optim.lr_scheduler import ReduceLROnPlateau
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


def run_validation(netG, netD, text_encoder, image_encoder, val_loader, device, args):
    """Run validation and compute average losses."""
    netG.eval()
    netD.eval()

    val_D_loss = 0.0
    val_G_loss = 0.0
    val_damsm_loss = 0.0
    n_batches = 0
    criterion = nn.BCELoss()

    with torch.no_grad():
        for batch in val_loader:
            real_imgs, captions, cap_lens, _, _ = batch
            real_imgs = real_imgs.to(device)
            captions = captions.to(device)
            cap_lens = cap_lens.to(device).squeeze(-1)
            batch_size = real_imgs.size(0)

            # Sort captions by length for LSTM packing
            cap_lens, sort_idx = torch.sort(cap_lens, descending=True)
            captions = captions[sort_idx]

            # Text encoding
            hidden = None
            words_emb, sent_emb = text_encoder(captions, cap_lens, hidden)
            sent_emb_detached = sent_emb.detach()

            # Discriminator validation loss
            real_logits = netD(real_imgs, sent_emb_detached)
            errD_real = criterion(real_logits, torch.ones_like(real_logits))

            noise = torch.randn(batch_size, 100, 1, 1, device=device)
            fake_imgs = netG(noise)
            fake_logits = netD(fake_imgs, sent_emb_detached)
            errD_fake = criterion(fake_logits, torch.zeros_like(fake_logits))
            errD_val = errD_real + errD_fake

            # Generator validation loss
            gan_logits = netD(fake_imgs, sent_emb.detach())
            g_gan_loss = criterion(gan_logits, torch.ones_like(gan_logits))

            features, cnn_code = image_encoder(fake_imgs)
            w_loss = words_loss(
                features, words_emb.transpose(1, 2), None, cap_lens, batch_size
            )
            s_loss = sent_loss(cnn_code, sent_emb, None, batch_size)
            damsm_loss_val = w_loss + s_loss

            errG_val = g_gan_loss + args.gamma_damsm * damsm_loss_val

            val_D_loss += errD_val.item()
            val_G_loss += errG_val.item()
            val_damsm_loss += damsm_loss_val.item()
            n_batches += 1

    netG.train()
    netD.train()

    avg_D = val_D_loss / n_batches
    avg_G = val_G_loss / n_batches
    avg_damsm = val_damsm_loss / n_batches

    return avg_D, avg_G, avg_damsm


def train(args):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    # Setup logging
    logger = LossLogger(args.log_dir)

    # Create checkpoint directory
    os.makedirs(args.checkpoint_dir, exist_ok=True)

    # 1. Datasets & DataLoaders
    train_dataset = TextImageDataset(
        data_dir=args.data_dir,
        split="train",
        image_size=args.image_size,
    )
    train_loader = DataLoader(
        train_dataset,
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=args.num_workers,
        pin_memory=True,
        drop_last=True,
    )
    print(f"Train dataset size: {len(train_dataset)} | Batches: {len(train_loader)}")

    val_loader = None
    if args.validate_interval > 0:
        val_dataset = TextImageDataset(
            data_dir=args.data_dir,
            split="test",
            image_size=args.image_size,
        )
        val_loader = DataLoader(
            val_dataset,
            batch_size=args.batch_size,
            shuffle=False,
            num_workers=args.num_workers,
            pin_memory=True,
            drop_last=False,
        )
        print(
            f"Validation dataset size: {len(val_dataset)} | Batches: {len(val_loader)}"
        )

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

    # 4. LR Schedulers
    schedulerG = ReduceLROnPlateau(
        optimizerG,
        mode="min",
        factor=args.lr_factor,
        patience=args.lr_patience,
        verbose=True,
    )
    schedulerD = ReduceLROnPlateau(
        optimizerD,
        mode="min",
        factor=args.lr_factor,
        patience=args.lr_patience,
        verbose=True,
    )

    # 5. Optionally resume from checkpoint
    start_epoch = 0
    best_val_loss = float("inf")
    if args.resume:
        start_epoch = load_checkpoint(
            args.checkpoint_dir, netG, netD, optimizerG, optimizerD
        )
        latest_path = os.path.join(args.checkpoint_dir, "checkpoint_latest.pth")
        if os.path.isfile(latest_path):
            state = torch.load(latest_path, map_location="cpu")
            if "schedulerG" in state:
                schedulerG.load_state_dict(state["schedulerG"])
            if "schedulerD" in state:
                schedulerD.load_state_dict(state["schedulerD"])
            best_val_loss = state.get("best_val_loss", float("inf"))
            print(f"Resumed best_val_loss: {best_val_loss:.4f}")

    # 6. Early stopping setup
    early_stop_counter = 0

    # 7. Training Loop
    criterion = nn.BCELoss()
    global_step = start_epoch * len(train_loader)

    for epoch in range(start_epoch, args.epochs):
        netG.train()
        netD.train()

        for batch_idx, batch in enumerate(train_loader):
            real_imgs, captions, cap_lens, _, _ = batch
            real_imgs = real_imgs.to(device)
            captions = captions.to(device)
            cap_lens = cap_lens.to(device).squeeze(-1)

            batch_size = real_imgs.size(0)

            # Sort captions by length for LSTM packing
            cap_lens, sort_idx = torch.sort(cap_lens, descending=True)
            captions = captions[sort_idx]

            # Text encoding
            hidden = None
            words_emb, sent_emb = text_encoder(captions, cap_lens, hidden)
            sent_emb_detached = sent_emb.detach()

            # ------------------------
            # Discriminator Update
            # ------------------------
            for d_step in range(args.D_steps):
                optimizerD.zero_grad()

                real_logits = netD(real_imgs, sent_emb_detached)
                errD_real = criterion(real_logits, torch.ones_like(real_logits))

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

            fake_imgs = netG(noise)
            gan_logits = netD(fake_imgs, sent_emb.detach())
            g_gan_loss = criterion(gan_logits, torch.ones_like(gan_logits))

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

        print(f"\nEpoch {epoch + 1}/{args.epochs} complete.")

        # Validation
        if val_loader and (epoch + 1) % args.validate_interval == 0:
            print(f"Running validation for epoch {epoch + 1}...")
            val_D, val_G, val_damsm = run_validation(
                netG, netD, text_encoder, image_encoder, val_loader, device, args
            )
            val_loss = (
                val_G  # Use generator validation loss for scheduling/early stopping
            )
            print(
                f"Validation — D: {val_D:.4f} | G: {val_G:.4f} | DAMSM: {val_damsm:.4f}"
            )

            # LR scheduler step
            schedulerG.step(val_loss)
            schedulerD.step(val_loss)

            # Early stopping check
            if val_loss < best_val_loss - args.min_delta:
                best_val_loss = val_loss
                early_stop_counter = 0
                # Save best model
                save_checkpoint(
                    args.checkpoint_dir,
                    epoch + 1,
                    netG,
                    netD,
                    optimizerG,
                    optimizerD,
                    best_val_loss=best_val_loss,
                    schedulerG=schedulerG.state_dict(),
                    schedulerD=schedulerD.state_dict(),
                )
                print(f"  New best validation loss: {best_val_loss:.4f}")
            else:
                early_stop_counter += 1
                print(
                    f"  No improvement for {early_stop_counter}/"
                    f"{args.patience_early_stop} epochs"
                )
                if early_stop_counter >= args.patience_early_stop:
                    print(f"Early stopping triggered after epoch {epoch + 1}")
                    break

        # Save checkpoint every N epochs
        if (epoch + 1) % args.checkpoint_interval == 0:
            save_checkpoint(
                args.checkpoint_dir,
                epoch + 1,
                netG,
                netD,
                optimizerG,
                optimizerD,
                best_val_loss=best_val_loss,
                schedulerG=schedulerG.state_dict(),
                schedulerD=schedulerD.state_dict(),
            )

        # Always save latest checkpoint
        save_checkpoint(
            args.checkpoint_dir,
            epoch + 1,
            netG,
            netD,
            optimizerG,
            optimizerD,
            best_val_loss=best_val_loss,
            schedulerG=schedulerG.state_dict(),
            schedulerD=schedulerD.state_dict(),
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

    # LR Scheduler args
    parser.add_argument(
        "--lr-patience",
        type=int,
        default=5,
        help="Patience for LR scheduler (epochs without improvement)",
    )
    parser.add_argument(
        "--lr-factor", type=float, default=0.5, help="Factor to reduce LR by on plateau"
    )

    # Validation & Early Stopping args
    parser.add_argument(
        "--validate-interval",
        type=int,
        default=1,
        help="Run validation every N epochs (0 to disable)",
    )
    parser.add_argument(
        "--patience-early-stop",
        type=int,
        default=10,
        help="Patience for early stopping (epochs without improvement)",
    )
    parser.add_argument(
        "--min-delta",
        type=float,
        default=1e-4,
        help="Minimum improvement to count as progress",
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
