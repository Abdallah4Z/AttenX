import torch

from code.generator import G_NET
from code.model import D_NET64, D_NET128, D_NET256


def test_generator_output_shapes():
    batch = 2
    nz = 100
    nef = 512
    nhidden = 256
    word_dim = nhidden * 2
    net_g = G_NET(ngf=64, nz=nz, nef=nef, nhidden=nhidden, word_dim=word_dim)

    z = torch.randn(batch, nz, 1, 1)
    sent_emb = torch.randn(batch, nef)
    word_emb = torch.randn(batch, 18, word_dim)
    img_64, img_128, img_256, mu, logvar = net_g(z, sent_emb, word_emb)

    assert img_64.shape == (batch, 3, 64, 64)
    assert img_128.shape == (batch, 3, 128, 128)
    assert img_256.shape == (batch, 3, 256, 256)
    assert mu.shape == (batch, nz)
    assert logvar.shape == (batch, nz)


def test_discriminator_64_shape():
    net_d = D_NET64(ndf=64, nef=512)
    fake_imgs = torch.randn(2, 3, 64, 64)
    sent_emb = torch.randn(2, 512)
    logits = net_d(fake_imgs, sent_emb)
    assert logits.shape == (2,)


def test_discriminator_128_shape():
    net_d = D_NET128(ndf=64, nef=512)
    fake_imgs = torch.randn(2, 3, 128, 128)
    sent_emb = torch.randn(2, 512)
    logits = net_d(fake_imgs, sent_emb)
    assert logits.shape == (2,)


def test_discriminator_256_shape():
    net_d = D_NET256(ndf=64, nef=512)
    fake_imgs = torch.randn(2, 3, 256, 256)
    sent_emb = torch.randn(2, 512)
    logits = net_d(fake_imgs, sent_emb)
    assert logits.shape == (2,)