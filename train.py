import argparse
import os
import random
import numpy as np
from code.datasets import TextImageDataset
from code.encoder import CNN_ENCODER, RNN_ENCODER
from code.generator import G_NET
from code.losses import sent_loss, words_loss, KL_loss
from code.model import D_NET64, D_NET128, D_NET256

import torch
import torch.nn as nn
import torch.optim as optim
from torch.optim.lr_scheduler import ReduceLROnPlateau
from torch.utils.data import DataLoader
from torch.cuda.amp import autocast, GradScaler

from scripts.loss_logging import LossLogger


def set_seed(seed: int):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False


def _load_pretrained_if_available(model, checkpoint_path, name):
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


def save_checkpoint(checkpoint_dir, epoch, netG, netsD, optimizerG, optimizersD, **extra):
    os.makedirs(checkpoint_dir, exist_ok=True)
    checkpoint_path = os.path.join(checkpoint_dir, f"checkpoint_epoch_{epoch:04d}.pth")
    state = {
        "epoch": epoch,
        "netG": netG.state_dict(),
        "netD0": netsD[0].state_dict(),
        "netD1": netsD[1].state_dict(),
        "netD2": netsD[2].state_dict(),
        "optimizerG": optimizerG.state_dict(),
        "optimizerD0": optimizersD[0].state_dict(),
        "optimizerD1": optimizersD[1].state_dict(),
        "optimizerD2": optimizersD[2].state_dict(),
        **extra,
    }
    torch.save(state, checkpoint_path)
    print(f"Checkpoint saved: {checkpoint_path}")
    latest_path = os.path.join(checkpoint_dir, "checkpoint_latest.pth")
    torch.save(state, latest_path)


def load_checkpoint(checkpoint_dir, netG, netsD, optimizerG=None, optimizersD=None):
    latest_path = os.path.join(checkpoint_dir, "checkpoint_latest.pth")
    if not os.path.isfile(latest_path):
        print(f"No checkpoint found at {latest_path}")
        return 0
    state = torch.load(latest_path, map_location="cpu")
    netG.load_state_dict(state["netG"])
    netsD[0].load_state_dict(state["netD0"])
    netsD[1].load_state_dict(state["netD1"])
    netsD[2].load_state_dict(state["netD2"])
    if optimizerG is not None:
        optimizerG.load_state_dict(state["optimizerG"])
    if optimizersD is not None:
        for i, optD in enumerate(optimizersD):
            if f"optimizerD{i}" in state:
                optD.load_state_dict(state[f"optimizerD{i}"])
    start_epoch = state.get("epoch", 0)
    print(f"Resumed from checkpoint epoch {start_epoch}")
    return start_epoch


def _discriminator_step(netD, real_imgs_scale, fake_imgs_scale, sent_emb,
                         criterion, device, batch_size, nz, netG=None,
                         sent_emb_for_G=None, word_emb_for_G=None,
                         fake_imgs_scale_detached=None):
    if fake_imgs_scale_detached is not None:
        real_logits = netD(real_imgs_scale, sent_emb)
        errD_real = criterion(real_logits, torch.ones_like(real_logits))
        fake_logits = netD(fake_imgs_scale_detached, sent_emb)
        errD_fake = criterion(fake_logits, torch.zeros_like(fake_logits))
        return errD_real + errD_fake
    real_logits = netD(real_imgs_scale, sent_emb)
    errD_real = criterion(real_logits, torch.ones_like(real_logits))
    fake_logits = netD(fake_imgs_scale.detach(), sent_emb)
    errD_fake = criterion(fake_logits, torch.zeros_like(fake_logits))
    return errD_real + errD_fake


def run_validation(netG, netsD, text_encoder, image_encoder, val_loader, device, args):
    netG.eval()
    for netD in netsD:
        netD.eval()
    word_dim = args.nhidden * 2
    nz = 100

    val_D_loss = 0.0
    val_G_loss = 0.0
    val_damsm_loss = 0.0
    val_kl_loss = 0.0
    n_batches = 0
    criterion = nn.BCELoss()

    with torch.no_grad():
        for batch in val_loader:
            real_imgs, captions, cap_lens, _, _ = batch
            real_imgs = real_imgs.to(device)
            captions = captions.to(device)
            cap_lens = cap_lens.to(device).squeeze(-1)
            batch_size = real_imgs.size(0)

            cap_lens_sorted, sort_idx = torch.sort(cap_lens, descending=True)
            captions_sorted = captions[sort_idx]

            hidden = None
            words_emb, sent_emb = text_encoder(captions_sorted, cap_lens_sorted, hidden)
            sent_emb_detached = sent_emb.detach()
            word_emb = words_emb.detach()

            real_64 = nn.functional.interpolate(real_imgs, size=(64, 64), mode="bilinear", align_corners=False)
            real_128 = nn.functional.interpolate(real_imgs, size=(128, 128), mode="bilinear", align_corners=False)

            errD_total = torch.tensor(0.0, device=device)
            for netD, real_scale in zip(netsD, [real_64, real_128, real_imgs]):
                real_logits = netD(real_scale, sent_emb_detached)
                errD_real = criterion(real_logits, torch.ones_like(real_logits))
                errD_fake_val = torch.tensor(0.0, device=device)

            noise = torch.randn(batch_size, nz, 1, 1, device=device)
            img_64, img_128, img_256, mu, logvar = netG(noise, sent_emb_detached, word_emb)
            fake_imgs_all = [img_64, img_128, img_256]

            errD_total = torch.tensor(0.0, device=device)
            for netD, real_scale in zip(netsD, [real_64, real_128, real_imgs]):
                fake_detached = fake_imgs_all[netsD.index(netD)].detach()
                real_logits = netD(real_scale, sent_emb_detached)
                errD_real = criterion(real_logits, torch.ones_like(real_logits))
                fake_logits = netD(fake_detached, sent_emb_detached)
                errD_fake = criterion(fake_logits, torch.zeros_like(fake_logits))
                errD_total = errD_total + errD_real + errD_fake

            noise = torch.randn(batch_size, nz, 1, 1, device=device)
            img_64, img_128, img_256, mu, logvar = netG(noise, sent_emb, word_emb)
            fake_imgs_all = [img_64, img_128, img_256]

            g_gan_loss = torch.tensor(0.0, device=device)
            for netD, fake_scale in zip(netsD, fake_imgs_all):
                logits = netD(fake_scale, sent_emb)
                g_gan_loss = g_gan_loss + criterion(logits, torch.ones_like(logits))

            kl = KL_loss(mu, logvar)
            features, cnn_code = image_encoder(img_256)
            w_loss = words_loss(features, words_emb.transpose(1, 2), None, cap_lens_sorted, batch_size)
            s_loss = sent_loss(cnn_code, sent_emb, None, batch_size)
            damsm_loss = w_loss + s_loss

            errG_val = g_gan_loss + args.gamma_damsm * damsm_loss + args.lambda_kl * kl

            val_D_loss += errD_total.item()
            val_G_loss += errG_val.item()
            val_damsm_loss += damsm_loss.item()
            val_kl_loss += kl.item()
            n_batches += 1

    netG.train()
    for netD in netsD:
        netD.train()

    return (val_D_loss / n_batches, val_G_loss / n_batches,
            val_damsm_loss / n_batches, val_kl_loss / n_batches)


def train(args):
    set_seed(args.seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    scaler = None
    if args.use_amp and torch.cuda.is_available():
        try:
            from torch.cuda.amp import autocast, GradScaler
            scaler = GradScaler()
            print("Using Automatic Mixed Precision (AMP)")
        except ImportError:
            print("AMP not available, falling back to FP32")
            scaler = None

    logger = LossLogger(args.log_dir)
    os.makedirs(args.checkpoint_dir, exist_ok=True)

    train_dataset = TextImageDataset(data_dir=args.data_dir, split="train", image_size=args.image_size)
    train_loader = DataLoader(
        train_dataset, batch_size=args.batch_size, shuffle=True,
        num_workers=args.num_workers, pin_memory=True, drop_last=True,
    )
    print(f"Train dataset size: {len(train_dataset)} | Batches: {len(train_loader)}")

    val_loader = None
    if args.validate_interval > 0:
        val_dataset = TextImageDataset(data_dir=args.data_dir, split="test", image_size=args.image_size)
        val_loader = DataLoader(
            val_dataset, batch_size=args.batch_size, shuffle=False,
            num_workers=args.num_workers, pin_memory=True, drop_last=False,
        )
        print(f"Validation dataset size: {len(val_dataset)} | Batches: {len(val_loader)}")

    word_dim = args.nhidden * 2
    nz = 100

    netG = G_NET(ngf=args.ngf, nz=nz, nef=args.nef, nhidden=args.nhidden, word_dim=word_dim).to(device)
    netD0 = D_NET64(ndf=args.ndf, nef=args.nef).to(device)
    netD1 = D_NET128(ndf=args.ndf, nef=args.nef).to(device)
    netD2 = D_NET256(ndf=args.ndf, nef=args.nef).to(device)
    netsD = [netD0, netD1, netD2]

    text_encoder = RNN_ENCODER(n_words=args.vocab_size, nhidden=args.nhidden, nembed=args.nembed).to(device)
    image_encoder = CNN_ENCODER(nef=args.nef).to(device)

    _load_pretrained_if_available(text_encoder, args.damsm_text_path, "DAMSM text encoder")
    _load_pretrained_if_available(image_encoder, args.damsm_image_path, "DAMSM image encoder")
    text_encoder.eval()
    image_encoder.eval()
    for p in text_encoder.parameters():
        p.requires_grad = False
    for p in image_encoder.parameters():
        p.requires_grad = False

    optimizerG = optim.Adam(netG.parameters(), lr=args.lr_g, betas=(0.5, 0.999))
    optimizersD = [
        optim.Adam(netD0.parameters(), lr=args.lr_d, betas=(0.5, 0.999)),
        optim.Adam(netD1.parameters(), lr=args.lr_d, betas=(0.5, 0.999)),
        optim.Adam(netD2.parameters(), lr=args.lr_d, betas=(0.5, 0.999)),
    ]

    schedulerG = ReduceLROnPlateau(optimizerG, mode="min", factor=args.lr_factor, patience=args.lr_patience, verbose=True)
    schedulersD = [
        ReduceLROnPlateau(opt, mode="min", factor=args.lr_factor, patience=args.lr_patience, verbose=True)
        for opt in optimizersD
    ]

    start_epoch = 0
    best_val_loss = float("inf")
    if args.resume:
        start_epoch = load_checkpoint(args.checkpoint_dir, netG, netsD, optimizerG, optimizersD)
        latest_path = os.path.join(args.checkpoint_dir, "checkpoint_latest.pth")
        if os.path.isfile(latest_path):
            state = torch.load(latest_path, map_location="cpu")
            best_val_loss = state.get("best_val_loss", float("inf"))
            print(f"Resumed best_val_loss: {best_val_loss:.4f}")

    early_stop_counter = 0
    criterion = nn.BCELoss()
    global_step = start_epoch * len(train_loader)

    for epoch in range(start_epoch, args.epochs):
        netG.train()
        for netD in netsD:
            netD.train()

        for batch_idx, batch in enumerate(train_loader):
            real_imgs, captions, cap_lens, _, _ = batch
            real_imgs = real_imgs.to(device)
            captions = captions.to(device)
            cap_lens = cap_lens.to(device).squeeze(-1)
            batch_size = real_imgs.size(0)

            cap_lens, sort_idx = torch.sort(cap_lens, descending=True)
            captions = captions[sort_idx]

            hidden = None
            words_emb, sent_emb = text_encoder(captions, cap_lens, hidden)
            sent_emb_detached = sent_emb.detach()
            word_emb = words_emb.detach()

            real_64 = nn.functional.interpolate(real_imgs, size=(64, 64), mode="bilinear", align_corners=False)
            real_128 = nn.functional.interpolate(real_imgs, size=(128, 128), mode="bilinear", align_corners=False)
            real_scales = [real_64, real_128, real_imgs]

            # Discriminator steps
            for d_step in range(args.D_steps):
                for netD, optD, real_scale in zip(netsD, optimizersD, real_scales):
                    optD.zero_grad()

                    if scaler:
                        with autocast():
                            real_logits = netD(real_scale, sent_emb_detached)
                            errD_real = criterion(real_logits, torch.ones_like(real_logits))

                            noise = torch.randn(batch_size, nz, 1, 1, device=device)
                            img_64, img_128, img_256, _, _ = netG(noise, sent_emb_detached, word_emb)
                            fake_scales = [img_64, img_128, img_256]
                            fake_idx = netsD.index(netD)
                            fake_logits = netD(fake_scales[fake_idx].detach(), sent_emb_detached)
                            errD_fake = criterion(fake_logits, torch.zeros_like(fake_logits))
                            errD = errD_real + errD_fake

                        scaler.scale(errD).backward()
                        if args.clip_grad:
                            scaler.unscale_(optD)
                            torch.nn.utils.clip_grad_norm_(netD.parameters(), args.clip_grad)
                        scaler.step(optD)
                        scaler.update()
                    else:
                        real_logits = netD(real_scale, sent_emb_detached)
                        errD_real = criterion(real_logits, torch.ones_like(real_logits))

                        noise = torch.randn(batch_size, nz, 1, 1, device=device)
                        img_64, img_128, img_256, _, _ = netG(noise, sent_emb_detached, word_emb)
                        fake_scales = [img_64, img_128, img_256]
                        fake_idx = netsD.index(netD)
                        fake_logits = netD(fake_scales[fake_idx].detach(), sent_emb_detached)
                        errD_fake = criterion(fake_logits, torch.zeros_like(fake_logits))
                        errD = errD_real + errD_fake

                        errD.backward()
                        if args.clip_grad:
                            torch.nn.utils.clip_grad_norm_(netD.parameters(), args.clip_grad)
                        optD.step()

            # Generator step
            optimizerG.zero_grad()
            noise = torch.randn(batch_size, nz, 1, 1, device=device)

            if scaler:
                with autocast():
                    img_64, img_128, img_256, mu, logvar = netG(noise, sent_emb, word_emb)
                    fake_scales = [img_64, img_128, img_256]

                    g_gan_loss = torch.tensor(0.0, device=device)
                    for netD, fake_scale in zip(netsD, fake_scales):
                        logits = netD(fake_scale, sent_emb)
                        g_gan_loss = g_gan_loss + criterion(logits, torch.ones_like(logits))

                    kl = KL_loss(mu, logvar)

                    features, cnn_code = image_encoder(img_256)
                    w_loss = words_loss(features, words_emb.transpose(1, 2), None, cap_lens, batch_size)
                    s_loss = sent_loss(cnn_code, sent_emb, None, batch_size)
                    damsm_loss = w_loss + s_loss

                    errG = g_gan_loss + args.gamma_damsm * damsm_loss + args.lambda_kl * kl

                scaler.scale(errG).backward()
                if args.clip_grad:
                    scaler.unscale_(optimizerG)
                    torch.nn.utils.clip_grad_norm_(netG.parameters(), args.clip_grad)
                scaler.step(optimizerG)
                scaler.update()
            else:
                img_64, img_128, img_256, mu, logvar = netG(noise, sent_emb, word_emb)
                fake_scales = [img_64, img_128, img_256]

                g_gan_loss = torch.tensor(0.0, device=device)
                for netD, fake_scale in zip(netsD, fake_scales):
                    logits = netD(fake_scale, sent_emb)
                    g_gan_loss = g_gan_loss + criterion(logits, torch.ones_like(logits))

                kl = KL_loss(mu, logvar)

                features, cnn_code = image_encoder(img_256)
                w_loss = words_loss(features, words_emb.transpose(1, 2), None, cap_lens, batch_size)
                s_loss = sent_loss(cnn_code, sent_emb, None, batch_size)
                damsm_loss = w_loss + s_loss

                errG = g_gan_loss + args.gamma_damsm * damsm_loss + args.lambda_kl * kl
                errG.backward()
                if args.clip_grad:
                    torch.nn.utils.clip_grad_norm_(netG.parameters(), args.clip_grad)
                optimizerG.step()

            global_step += 1
            logger.log(global_step, {
                "G_Loss": errG.item(),
                "G_GAN": g_gan_loss.item(),
                "L_Words": w_loss.item(),
                "L_Sent": s_loss.item(),
                "L_DAMSM": damsm_loss.item(),
                "L_KL": kl.item(),
                "Gamma_DAMSM": args.gamma_damsm,
                "Lambda_KL": args.lambda_kl,
            })

        print(f"\nEpoch {epoch + 1}/{args.epochs} complete.")

        if val_loader and (epoch + 1) % args.validate_interval == 0:
            print(f"Running validation for epoch {epoch + 1}...")
            val_D, val_G, val_damsm, val_kl = run_validation(
                netG, netsD, text_encoder, image_encoder, val_loader, device, args
            )
            val_loss = val_G
            print(f"Validation — D: {val_D:.4f} | G: {val_G:.4f} | DAMSM: {val_damsm:.4f} | KL: {val_kl:.4f}")

            schedulerG.step(val_loss)
            for sD in schedulersD:
                sD.step(val_loss)

            if val_loss < best_val_loss - args.min_delta:
                best_val_loss = val_loss
                early_stop_counter = 0
                save_checkpoint(args.checkpoint_dir, epoch + 1, netG, netsD, optimizerG, optimizersD,
                               best_val_loss=best_val_loss,
                               schedulerG=schedulerG.state_dict(),
                               schedulersD=[s.state_dict() for s in schedulersD])
                print(f"  New best validation loss: {best_val_loss:.4f}")
            else:
                early_stop_counter += 1
                print(f"  No improvement for {early_stop_counter}/{args.patience_early_stop} epochs")
                if early_stop_counter >= args.patience_early_stop:
                    print(f"Early stopping triggered after epoch {epoch + 1}")
                    break

        if (epoch + 1) % args.checkpoint_interval == 0:
            save_checkpoint(args.checkpoint_dir, epoch + 1, netG, netsD, optimizerG, optimizersD,
                           best_val_loss=best_val_loss,
                           schedulerG=schedulerG.state_dict(),
                           schedulersD=[s.state_dict() for s in schedulersD])

        save_checkpoint(args.checkpoint_dir, epoch + 1, netG, netsD, optimizerG, optimizersD,
                       best_val_loss=best_val_loss,
                       schedulerG=schedulerG.state_dict(),
                       schedulersD=[s.state_dict() for s in schedulersD])

    logger.save()
    print("Training complete.")


def parse_args():
    parser = argparse.ArgumentParser(description="AttenX training pipeline")
    parser.add_argument("--data-dir", type=str, default="./data")
    parser.add_argument("--image-size", type=int, default=256)
    parser.add_argument("--num-workers", type=int, default=4)
    parser.add_argument("--ngf", type=int, default=64)
    parser.add_argument("--ndf", type=int, default=64)
    parser.add_argument("--nef", type=int, default=512)
    parser.add_argument("--nhidden", type=int, default=256)
    parser.add_argument("--nembed", type=int, default=256)
    parser.add_argument("--vocab-size", type=int, default=10000)
    parser.add_argument("--epochs", type=int, default=100)
    parser.add_argument("--batch-size", type=int, default=4)
    parser.add_argument("--lr-g", type=float, default=1e-4)
    parser.add_argument("--lr-d", type=float, default=4e-4)
    parser.add_argument("--D-steps", type=int, default=1)
    parser.add_argument("--gamma-damsm", type=float, default=5.0)
    parser.add_argument("--lambda-kl", type=float, default=2.0)
    parser.add_argument("--clip-grad", type=float, default=None)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--use-amp", action="store_true")
    parser.add_argument("--lr-patience", type=int, default=5)
    parser.add_argument("--lr-factor", type=float, default=0.5)
    parser.add_argument("--validate-interval", type=int, default=1)
    parser.add_argument("--patience-early-stop", type=int, default=10)
    parser.add_argument("--min-delta", type=float, default=1e-4)
    parser.add_argument("--checkpoint-dir", type=str, default="./checkpoints")
    parser.add_argument("--checkpoint-interval", type=int, default=10)
    parser.add_argument("--log-dir", type=str, default="./logs/training")
    parser.add_argument("--resume", action="store_true")
    parser.add_argument("--damsm-text-path", type=str, default=os.getenv("DAMSM_TEXT_ENCODER_PATH"))
    parser.add_argument("--damsm-image-path", type=str, default=os.getenv("DAMSM_IMAGE_ENCODER_PATH"))
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()
    train(args)