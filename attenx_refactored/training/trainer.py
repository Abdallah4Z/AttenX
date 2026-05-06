import contextlib
import os
import random

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.cuda.amp import autocast, GradScaler
from torch.optim.lr_scheduler import ReduceLROnPlateau
from torch.utils.data import DataLoader

from attenx_refactored.data.dataset import TextImageDataset
from attenx_refactored.losses.damsm import words_loss, sent_loss, KL_loss
from attenx_refactored.models.discriminator import D_NET64, D_NET128, D_NET256
from attenx_refactored.models.encoders import RNN_ENCODER, CNN_ENCODER
from attenx_refactored.models.generator import G_NET
from attenx_refactored.training.validator import validate
from attenx_refactored.utils.checkpointing import save_checkpoint, load_checkpoint
from attenx_refactored.utils.collate import collate_fn
from attenx_refactored.utils.logging import LossLogger


class _AMPHelper:
    def __init__(self, scaler):
        self.scaler = scaler

    def autocast(self):
        if self.scaler is not None:
            return autocast()
        return contextlib.nullcontext()

    def backward(self, loss):
        if self.scaler is not None:
            self.scaler.scale(loss).backward()
        else:
            loss.backward()

    def step(self, optimizer):
        if self.scaler is not None:
            self.scaler.step(optimizer)
        else:
            optimizer.step()

    def update(self):
        if self.scaler is not None:
            self.scaler.update()


def _set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
        torch.backends.cudnn.benchmark = True


def _resolve_device(device_str):
    if device_str == "auto":
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")
    return torch.device(device_str)


def _load_pretrained_encoder(model, path, name, device):
    if not path or not os.path.isfile(path):
        return False
    state = torch.load(path, map_location=device)
    if isinstance(state, dict) and "state_dict" in state:
        state = state["state_dict"]
    model.load_state_dict(state, strict=False)
    print(f"Loaded pretrained {name} from {path}")
    return True


def _sort_captions(captions, cap_lens):
    cap_lens_sorted, sort_idx = torch.sort(cap_lens, descending=True)
    captions_sorted = captions[sort_idx]
    return captions_sorted, cap_lens_sorted


def _make_real_scales(images):
    real_64 = nn.functional.interpolate(images, size=(64, 64), mode="bilinear", align_corners=False)
    real_128 = nn.functional.interpolate(images, size=(128, 128), mode="bilinear", align_corners=False)
    return [real_64, real_128, images]


def _make_fake_scales(netG, noise, sent_emb, word_emb):
    f64, f128, f256, _, _ = netG(noise, sent_emb, word_emb)
    return [f64, f128, f256]


def _train_d_step(netsD, optimizersD, real_scales, fake_scales, sent_emb, criterion, amp):
    err_d_total = torch.tensor(0.0, device=sent_emb.device)
    for netD, optD, real_s, fake_s in zip(netsD, optimizersD, real_scales, fake_scales):
        optD.zero_grad()
        with amp.autocast():
            r_logits = netD(real_s, sent_emb)
            r_loss = criterion(r_logits, torch.ones_like(r_logits))
            f_logits = netD(fake_s, sent_emb)
            f_loss = criterion(f_logits, torch.zeros_like(f_logits))
            err_d = r_loss + f_loss
        amp.backward(err_d)
        amp.step(optD)
        err_d_total = err_d_total + err_d.detach()
    amp.update()
    return err_d_total


def _train_g_step(netG, netsD, optimizerG, noise, sent_emb, word_emb, cap_lens, image_encoder, criterion, config, amp):
    with amp.autocast():
        f64, f128, f256, mu, logvar = netG(noise, sent_emb, word_emb)
        fake_scales = [f64, f128, f256]

        g_gan = torch.tensor(0.0, device=sent_emb.device)
        for netD, fs in zip(netsD, fake_scales):
            logits = netD(fs, sent_emb)
            g_gan = g_gan + criterion(logits, torch.ones_like(logits))

        kl_loss = KL_loss(mu, logvar)
        features, cnn_code = image_encoder(f256)
        w_loss = words_loss(features, word_emb.transpose(1, 2), None, cap_lens, config.batch_size)
        s_loss = sent_loss(cnn_code, sent_emb, None, config.batch_size)
        damsm_loss = w_loss + s_loss
        err_g = g_gan + config.gamma_damsm * damsm_loss + config.lambda_kl * kl_loss

    amp.backward(err_g)
    amp.step(optimizerG)
    amp.update()

    return err_g, g_gan, w_loss, s_loss, damsm_loss, kl_loss


def train(config):
    _set_seed(config.seed)
    device = _resolve_device(config.device)
    print(f"Using device: {device}")

    scaler = GradScaler() if (config.use_amp and device.type == "cuda") else None
    amp = _AMPHelper(scaler)

    logger = LossLogger(config.log_dir)
    os.makedirs(config.checkpoint_dir, exist_ok=True)
    pin = device.type == "cuda"

    train_dataset = TextImageDataset(data_dir=config.data_dir, split="train", image_size=config.image_size)
    train_loader = DataLoader(
        train_dataset, batch_size=config.batch_size, shuffle=True,
        num_workers=config.num_workers, pin_memory=pin, drop_last=True,
        collate_fn=collate_fn,
    )
    print(f"Train: {len(train_dataset)} images, {len(train_loader)} batches")

    val_loader = None
    if config.validate_interval > 0:
        val_dataset = TextImageDataset(data_dir=config.data_dir, split="test", image_size=config.image_size)
        val_loader = DataLoader(
            val_dataset, batch_size=config.batch_size, shuffle=False,
            num_workers=config.num_workers, pin_memory=pin, drop_last=False,
            collate_fn=collate_fn,
        )
        print(f"Val: {len(val_dataset)} images, {len(val_loader)} batches")

    word_dim = config.nhidden * 2
    nz = 100

    netG = G_NET(
        ngf=config.ngf, nz=nz, nef=config.nef, word_dim=word_dim,
        attention_mode=config.attention_mode, num_heads=config.num_heads,
    ).to(device)

    netsD = [
        D_NET64(ndf=config.ndf, nef=config.nef, use_sn=config.use_sn).to(device),
        D_NET128(ndf=config.ndf, nef=config.nef, use_sn=config.use_sn).to(device),
        D_NET256(ndf=config.ndf, nef=config.nef, use_sn=config.use_sn).to(device),
    ]

    text_encoder = RNN_ENCODER(n_words=config.vocab_size, nhidden=config.nhidden, nembed=config.nembed).to(device)
    image_encoder = CNN_ENCODER(nef=config.nef).to(device)

    _load_pretrained_encoder(text_encoder, config.damsm_text_path, "text encoder", device)
    _load_pretrained_encoder(image_encoder, config.damsm_image_path, "image encoder", device)

    text_encoder.eval()
    image_encoder.eval()
    for p in text_encoder.parameters():
        p.requires_grad = False
    for p in image_encoder.parameters():
        p.requires_grad = False

    optimizerG = optim.Adam(netG.parameters(), lr=config.lr_g, betas=(0.5, 0.999))
    optimizersD = [optim.Adam(netD.parameters(), lr=config.lr_d, betas=(0.5, 0.999)) for netD in netsD]

    schedulerG = ReduceLROnPlateau(optimizerG, mode="min", factor=config.lr_factor, patience=config.lr_patience)
    schedulersD = [ReduceLROnPlateau(opt, mode="min", factor=config.lr_factor, patience=config.lr_patience) for opt in optimizersD]

    start_epoch = 0
    best_val_loss = float("inf")

    if config.resume:
        ckpt_path = os.path.join(config.checkpoint_dir, "checkpoint_latest.pth")
        if os.path.isfile(ckpt_path):
            start_epoch, best_val_loss = load_checkpoint(ckpt_path, netG, netsD, optimizerG, optimizersD, device)
            print(f"Resumed from epoch {start_epoch}")

    early_stop_counter = 0
    criterion = nn.BCELoss()
    global_step = start_epoch * len(train_loader)

    for epoch in range(start_epoch, config.epochs):
        netG.train()
        for netD in netsD:
            netD.train()

        for batch_idx, batch in enumerate(train_loader):
            real_imgs, captions, cap_lens, _, _ = batch
            real_imgs = real_imgs.to(device)
            captions = captions.to(device)

            captions, cap_lens_sorted = _sort_captions(captions, cap_lens)
            words_emb, sent_emb = text_encoder(captions, cap_lens_sorted, None)
            sent_emb_d = sent_emb.detach()
            word_emb_d = words_emb.detach()

            real_scales = _make_real_scales(real_imgs)

            for _ in range(config.D_steps):
                noise = torch.randn(config.batch_size, nz, 1, 1, device=device)
                with torch.no_grad():
                    fake_scales = _make_fake_scales(netG, noise, sent_emb_d, word_emb_d)
                _train_d_step(netsD, optimizersD, real_scales, fake_scales, sent_emb_d, criterion, amp)

            optimizerG.zero_grad()
            noise = torch.randn(config.batch_size, nz, 1, 1, device=device)
            err_g, g_gan, w_loss, s_loss, damsm_loss, kl_loss = _train_g_step(
                netG, netsD, optimizerG, noise, sent_emb, word_emb_d, cap_lens_sorted, image_encoder, criterion, config, amp,
            )

            global_step += 1
            logger.log(global_step, {
                "G_Loss": err_g.item(),
                "G_GAN": g_gan.item() if isinstance(g_gan, torch.Tensor) else g_gan,
                "L_Words": w_loss.item(),
                "L_Sent": s_loss.item(),
                "L_DAMSM": damsm_loss.item(),
                "L_KL": kl_loss.item(),
            })

        print(f"Epoch {epoch + 1}/{config.epochs} complete.")

        if val_loader and (epoch + 1) % config.validate_interval == 0:
            val_d, val_g, val_damsm, val_kl = validate(netG, netsD, text_encoder, image_encoder, val_loader, device, config)
            val_loss = val_g
            print(f"Val - D: {val_d:.4f}  G: {val_g:.4f}  DAMSM: {val_damsm:.4f}  KL: {val_kl:.4f}")

            schedulerG.step(val_loss)
            for sD in schedulersD:
                sD.step(val_loss)

            if val_loss < best_val_loss - config.min_delta:
                best_val_loss = val_loss
                early_stop_counter = 0
                ckpt = os.path.join(config.checkpoint_dir, "checkpoint_best.pth")
                save_checkpoint(ckpt, epoch + 1, netG, netsD, optimizerG, optimizersD, schedulerG, schedulersD, best_val_loss)
                print(f"  New best val loss: {best_val_loss:.4f}")
            else:
                early_stop_counter += 1
                print(f"  No improvement ({early_stop_counter}/{config.patience_early_stop})")
                if early_stop_counter >= config.patience_early_stop:
                    print(f"Early stopping at epoch {epoch + 1}")
                    break

        latest_ckpt = os.path.join(config.checkpoint_dir, "checkpoint_latest.pth")
        save_checkpoint(latest_ckpt, epoch + 1, netG, netsD, optimizerG, optimizersD, schedulerG, schedulersD, best_val_loss)

    logger.save()
    print("Training complete.")
