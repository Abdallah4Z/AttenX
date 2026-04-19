import torch

from code.generator import G_NET
from code.model import D_NET256


def test_generator_output_shape():
    net_g = G_NET()
    z = torch.randn(2, 100, 1, 1)
    out = net_g(z)
    assert out.shape == (2, 3, 256, 256)


def test_discriminator_output_shape():
    net_d = D_NET256(ndf=64, nef=512)
    fake_imgs = torch.randn(2, 3, 256, 256)
    sent_emb = torch.randn(2, 512)
    logits = net_d(fake_imgs, sent_emb)
    assert logits.shape == (2,)
