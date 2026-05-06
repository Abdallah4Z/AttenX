import os
import random

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.nn.utils.rnn import pad_sequence
from torch.optim.lr_scheduler import ReduceLROnPlateau
from torch.utils.data import DataLoader
from torch.cuda.amp import autocast, GradScaler

from attenx.data.dataset import TextImageDataset
from attenx.models.encoders import RNN_ENCODER, CNN_ENCODER
from attenx.models.generator import G_NET
from attenx.models.discriminator import D_NET64, D_NET128, D_NET256
from attenx.losses.damsm import words_loss, sent_loss, KL_loss
from attenx.training.validator import validate
from attenx.utils.checkpointing import save_checkpoint, load_checkpoint
from attenx.utils.logging import LossLogger


def _collate_fn(batch):
    images, captions, cap_lens, cls_ids, keys = zip(*batch)
    images = torch.stack(images, 0)
    cap_lens = torch.stack(cap_lens, 0).squeeze(-1)
    captions = pad_sequence(captions, batch_first=True, padding_value=0)
    return images, captions, cap_lens, cls_ids, keys


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


def train(config):
    _set_seed(config.seed)
    device = _resolve_device(config.device)
    print(f"Using device: {device}")

    scaler = None
    if config.use_amp and device.type == "cuda":
        scaler = GradScaler()

    logger = LossLogger(config.log_dir)
    os.makedirs(config.checkpoint_dir, exist_ok=True)
    pin = device.type == "cuda"

    train_dataset = TextImageDataset(
        data_dir=config.data_dir, split="train",
        image_size=config.image_size,
    )
    train_loader = DataLoader(
        train_dataset, batch_size=config.batch_size, shuffle=True,
        num_workers=config.num_workers, pin_memory=pin, drop_last=True,
        collate_fn=_collate_fn,
    )
    print(f"Train: {len(train_dataset)} images, {len(train_loader)} batches")

    val_loader = None
    if config.validate_interval > 0:
        val_dataset = TextImageDataset(
            data_dir=config.data_dir, split="test",
            image_size=config.image_size,
        )
        val_loader = DataLoader(
            val_dataset, batch_size=config.batch_size, shuffle=False,
            num_workers=config.num_workers, pin_memory=pin, drop_last=False,
            collate_fn=_collate_fn,
        )
        print(f"Val: {len(val_dataset)} images, {len(val_loader)} batches")

    word_dim = config.nhidden * 2
    nz = 100

    netG = G_NET(
        ngf=config.ngf, nz=nz, nef=config.nef,
        word_dim=word_dim,
        attention_mode=config.attention_mode,
        num_heads=config.num_heads,
    ).to(device)

    use_sn = config.attention_mode != "none"
    netsD = [
        D_NET64(ndf=config.ndf, nef=config.nef, use_sn=use_sn).to(device),
        D_NET128(ndf=config.ndf, nef=config.nef, use_sn=use_sn).to(device),
        D_NET256(ndf=config.ndf, nef=config.nef, use_sn=use_sn).to(device),
    ]

    text_encoder = RNN_ENCODER(
        n_words=config.vocab_size, nhidden=config.nhidden,
        nembed=config.nembed,
    ).to(device)
    image_encoder = CNN_ENCODER(nef=config.nef).to(device)

    _load_pretrained_encoder(text_encoder, config.damsm_text_path,
                             "text encoder", device)
    _load_pretrained_encoder(image_encoder, config.damsm_image_path,
                             "image encoder", device)

    text_encoder.eval()
    image_encoder.eval()
    for p in text_encoder.parameters():
        p.requires_grad = False
    for p in image_encoder.parameters():
        p.requires_grad = False

    optimizerG = optim.Adam(netG.parameters(), lr=config.lr_g,
                            betas=(0.5, 0.999))
    optimizersD = [
        optim.Adam(netD.parameters(), lr=config.lr_d, betas=(0.5, 0.999))
        for netD in netsD
    ]

    schedulerG = ReduceLROnPlateau(optimizerG, mode="min",
                                   factor=config.lr_factor,
                                   patience=config.lr_patience)
    schedulersD = [
        ReduceLROnPlateau(opt, mode="min", factor=config.lr_factor,
                          patience=config.lr_patience)
        for opt in optimizersD
    ]

    start_epoch = 0
    best_val_loss = float("inf")

    if config.resume:
        ckpt_path = os.path.join(config.checkpoint_dir, "checkpoint_latest.pth")
        if os.path.isfile(ckpt_path):
            start_epoch, best_val_loss = load_checkpoint(
                ckpt_path, netG, netsD, optimizerG, optimizersD, device,
            )
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

            cap_lens_sorted, sort_idx = torch.sort(cap_lens.squeeze(-1),
                                                   descending=True)
            captions = captions[sort_idx]

            words_emb, sent_emb = text_encoder(captions, cap_lens_sorted,
                                                None)
            sent_emb_d = sent_emb.detach()
            word_emb_d = words_emb.detach()

            real_64 = nn.functional.interpolate(
                real_imgs, size=(64, 64), mode="bilinear",
                align_corners=False,
            )
            real_128 = nn.functional.interpolate(
                real_imgs, size=(128, 128), mode="bilinear",
                align_corners=False,
            )
            real_scales = [real_64, real_128, real_imgs]

            for _ in range(config.D_steps):
                for netD, optD, real_s in zip(netsD, optimizersD, real_scales):
                    optD.zero_grad()

                    noise = torch.randn(config.batch_size, nz, 1, 1,
                                        device=device)
                    with torch.no_grad():
                        f64, f128, f256, _, _ = netG(noise, sent_emb_d,
                                                     word_emb_d)
                    fake_scales = [f64, f128, f256]

                    if scaler:
                        with autocast():
                            r_logits = netD(real_s, sent_emb_d)
                            r_loss = criterion(r_logits,
                                               torch.ones_like(r_logits))
                            f_logits = netD(fake_scales[netsD.index(netD)],
                                            sent_emb_d)
                            f_loss = criterion(f_logits,
                                               torch.zeros_like(f_logits))
                            err_d = r_loss + f_loss
                        scaler.scale(err_d).backward()
                        scaler.step(optD)
                        scaler.update()
                    else:
                        r_logits = netD(real_s, sent_emb_d)
                        r_loss = criterion(r_logits, torch.ones_like(r_logits))
                        f_logits = netD(fake_scales[netsD.index(netD)],
                                        sent_emb_d)
                        f_loss = criterion(f_logits, torch.zeros_like(f_logits))
                        err_d = r_loss + f_loss
                        err_d.backward()
                        optD.step()

            optimizerG.zero_grad()
            noise = torch.randn(config.batch_size, nz, 1, 1, device=device)

            if scaler:
                with autocast():
                    f64, f128, f256, mu, logvar = netG(noise, sent_emb,
                                                        word_emb_d)
                    fake_scales = [f64, f128, f256]
                    g_gan = torch.tensor(0.0, device=device)
                    for netD, fs in zip(netsD, fake_scales):
                        logits = netD(fs, sent_emb)
                        g_gan = g_gan + criterion(logits,
                                                  torch.ones_like(logits))
                    kl_loss = KL_loss(mu, logvar)
                    features, cnn_code = image_encoder(f256)
                    w_loss = words_loss(features, words_emb.transpose(1, 2),
                                        None, cap_lens_sorted,
                                        config.batch_size)
                    s_loss = sent_loss(cnn_code, sent_emb, None,
                                       config.batch_size)
                    damsm_loss = w_loss + s_loss
                    err_g = (g_gan + config.gamma_damsm * damsm_loss
                             + config.lambda_kl * kl_loss)

                scaler.scale(err_g).backward()
                scaler.step(optimizerG)
                scaler.update()
            else:
                f64, f128, f256, mu, logvar = netG(noise, sent_emb,
                                                    word_emb_d)
                fake_scales = [f64, f128, f256]
                g_gan = torch.tensor(0.0, device=device)
                for netD, fs in zip(netsD, fake_scales):
                    logits = netD(fs, sent_emb)
                    g_gan = g_gan + criterion(logits,
                                              torch.ones_like(logits))
                kl_loss = KL_loss(mu, logvar)
                features, cnn_code = image_encoder(f256)
                w_loss = words_loss(features, words_emb.transpose(1, 2),
                                    None, cap_lens_sorted, config.batch_size)
                s_loss = sent_loss(cnn_code, sent_emb, None, config.batch_size)
                damsm_loss = w_loss + s_loss
                err_g = (g_gan + config.gamma_damsm * damsm_loss
                         + config.lambda_kl * kl_loss)
                err_g.backward()
                optimizerG.step()

            global_step += 1
            logger.log(global_step, {
                "G_Loss": err_g.item(),
                "G_GAN": g_gan.item() if isinstance(g_gan, torch.Tensor)
                         else g_gan,
                "L_Words": w_loss.item(),
                "L_Sent": s_loss.item(),
                "L_DAMSM": damsm_loss.item(),
                "L_KL": kl_loss.item(),
            })

        print(f"Epoch {epoch + 1}/{config.epochs} complete.")

        if val_loader and (epoch + 1) % config.validate_interval == 0:
            val_d, val_g, val_damsm, val_kl = validate(
                netG, netsD, text_encoder, image_encoder, val_loader,
                device, config,
            )
            val_loss = val_g
            print(f"Val — D: {val_d:.4f}  G: {val_g:.4f}  "
                  f"DAMSM: {val_damsm:.4f}  KL: {val_kl:.4f}")

            schedulerG.step(val_loss)
            for sD in schedulersD:
                sD.step(val_loss)

            if val_loss < best_val_loss - config.min_delta:
                best_val_loss = val_loss
                early_stop_counter = 0
                ckpt = os.path.join(config.checkpoint_dir,
                                    "checkpoint_best.pth")
                save_checkpoint(ckpt, epoch + 1, netG, netsD,
                                optimizerG, optimizersD,
                                schedulerG, schedulersD, best_val_loss)
                print(f"  New best val loss: {best_val_loss:.4f}")
            else:
                early_stop_counter += 1
                print(f"  No improvement "
                      f"({early_stop_counter}/{config.patience_early_stop})")
                if early_stop_counter >= config.patience_early_stop:
                    print(f"Early stopping at epoch {epoch + 1}")
                    break

        latest_ckpt = os.path.join(config.checkpoint_dir,
                                   "checkpoint_latest.pth")
        save_checkpoint(latest_ckpt, epoch + 1, netG, netsD,
                        optimizerG, optimizersD,
                        schedulerG, schedulersD, best_val_loss)

    logger.save()
    print("Training complete.")
