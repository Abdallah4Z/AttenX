import torch
import torch.nn as nn

from attenx_refactored.losses.damsm import words_loss, sent_loss, KL_loss


@torch.no_grad()
def validate(netG, netsD, text_encoder, image_encoder, val_loader, device, config):
    netG.eval()
    for netD in netsD:
        netD.eval()

    criterion = nn.BCELoss()
    nz = 100

    total_d = 0.0
    total_g = 0.0
    total_damsm = 0.0
    total_kl = 0.0
    n_batches = 0

    for batch in val_loader:
        real_imgs, captions, cap_lens, _, _ = batch
        real_imgs = real_imgs.to(device)
        captions = captions.to(device)
        cap_lens = cap_lens.squeeze(-1)
        batch_size = real_imgs.size(0)

        cap_lens_sorted, sort_idx = torch.sort(cap_lens, descending=True)
        captions_sorted = captions[sort_idx]

        words_emb, sent_emb = text_encoder(captions_sorted, cap_lens_sorted, None)
        sent_emb_d = sent_emb.detach()
        word_emb_d = words_emb.detach()

        real_64 = nn.functional.interpolate(real_imgs, size=(64, 64), mode="bilinear", align_corners=False)
        real_128 = nn.functional.interpolate(real_imgs, size=(128, 128), mode="bilinear", align_corners=False)
        real_scales = [real_64, real_128, real_imgs]

        noise = torch.randn(batch_size, nz, 1, 1, device=device)
        img_64, img_128, img_256, mu, logvar = netG(noise, sent_emb_d, word_emb_d)
        fake_scales = [img_64, img_128, img_256]

        err_d = torch.tensor(0.0, device=device)
        for netD, real_s, fake_s in zip(netsD, real_scales, fake_scales):
            real_logits = netD(real_s, sent_emb_d)
            err_d_real = criterion(real_logits, torch.ones_like(real_logits))

            fake_logits = netD(fake_s.detach(), sent_emb_d)
            err_d_fake = criterion(fake_logits, torch.zeros_like(fake_logits))

            err_d = err_d + err_d_real + err_d_fake

        noise = torch.randn(batch_size, nz, 1, 1, device=device)
        img_64, img_128, img_256, mu, logvar = netG(noise, sent_emb, word_emb_d)
        fake_scales = [img_64, img_128, img_256]

        g_gan = torch.tensor(0.0, device=device)
        for netD, fake_s in zip(netsD, fake_scales):
            logits = netD(fake_s, sent_emb)
            g_gan = g_gan + criterion(logits, torch.ones_like(logits))

        kl = KL_loss(mu, logvar)
        features, cnn_code = image_encoder(img_256)
        w_loss = words_loss(features, words_emb.transpose(1, 2), None, cap_lens_sorted, batch_size)
        s_loss = sent_loss(cnn_code, sent_emb, None, batch_size)
        damsm_loss = w_loss + s_loss
        err_g = g_gan + config.gamma_damsm * damsm_loss + config.lambda_kl * kl

        total_d += err_d.item()
        total_g += err_g.item()
        total_damsm += damsm_loss.item()
        total_kl += kl.item()
        n_batches += 1

    netG.train()
    for netD in netsD:
        netD.train()

    n = max(n_batches, 1)
    return total_d / n, total_g / n, total_damsm / n, total_kl / n
