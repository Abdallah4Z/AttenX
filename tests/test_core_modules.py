import os
import pickle

import pytest
import torch
from PIL import Image

from code.datasets import TextImageDataset
from code.generator import G_NET
from code.model import D_NET64, D_NET128, D_NET256
from code.modules import SelfAttention, ConditioningAugmentation, CrossAttention, GenStage


def _build_minimal_dataset_root(tmp_path):
    data_dir = tmp_path / "data"
    cub_dir = data_dir / "CUB_200_2011"
    images_dir = cub_dir / "images" / "001.Class"
    train_dir = data_dir / "train"

    images_dir.mkdir(parents=True, exist_ok=True)
    train_dir.mkdir(parents=True, exist_ok=True)

    key = "001.Class/sample_0001"

    (cub_dir / "bounding_boxes.txt").write_text("1 0 0 10 10\n", encoding="utf-8")
    (cub_dir / "images.txt").write_text(f"1 {key}.jpg\n", encoding="utf-8")

    with open(train_dir / "filenames.pickle", "wb") as f:
        pickle.dump([key], f)

    with open(train_dir / "class_info.pickle", "wb") as f:
        pickle.dump([0], f)

    return data_dir, key


def test_self_attention_output_shape_and_grad_flow():
    torch.manual_seed(7)
    attn = SelfAttention(in_dim=64)
    x = torch.randn(2, 64, 16, 16, requires_grad=True)

    out = attn(x)
    assert out.shape == x.shape

    loss = out.mean()
    loss.backward()

    assert x.grad is not None
    assert attn.query_conv.weight.grad is not None
    assert attn.key_conv.weight.grad is not None
    assert attn.value_conv.weight.grad is not None


def test_self_attention_gamma_parameter_updates_with_optimizer_step():
    torch.manual_seed(11)
    attn = SelfAttention(in_dim=32)
    opt = torch.optim.SGD(attn.parameters(), lr=0.1)
    x = torch.randn(2, 32, 8, 8)

    gamma_before = attn.gamma.detach().clone()

    opt.zero_grad()
    out = attn(x)
    loss = -out.pow(2).mean()
    loss.backward()
    opt.step()

    gamma_after = attn.gamma.detach().clone()
    assert not torch.allclose(gamma_before, gamma_after)


def test_conditioning_augmentation():
    torch.manual_seed(42)
    ca = ConditioningAugmentation(emb_dim=512, nz=100)
    sent_emb = torch.randn(4, 512)
    c, mu, logvar = ca(sent_emb)
    assert c.shape == (4, 100)
    assert mu.shape == (4, 100)
    assert logvar.shape == (4, 100)


def test_cross_attention():
    torch.manual_seed(42)
    cross_attn = CrossAttention(h_dim=128, w_dim=512, attn_dim=64)
    h = torch.randn(2, 128, 16, 16)
    w = torch.randn(2, 18, 512)
    ctx = cross_attn(h, w)
    assert ctx.shape == (2, 64, 16, 16)


def test_gen_stage():
    torch.manual_seed(42)
    stage = GenStage(in_ch=256, out_ch=128, w_dim=512, attn_dim=64, use_self_attn=True)
    h = torch.randn(2, 256, 16, 16)
    w = torch.randn(2, 18, 512)
    out = stage(h, w)
    assert out.shape == (2, 128, 32, 32)


def test_generator_forward_with_text_conditioning():
    torch.manual_seed(42)
    batch_size = 2
    nz = 100
    nef = 512
    nhidden = 256
    word_dim = nhidden * 2

    netG = G_NET(ngf=64, nz=nz, nef=nef, nhidden=nhidden, word_dim=word_dim)
    z = torch.randn(batch_size, nz, 1, 1)
    sent_emb = torch.randn(batch_size, nef)
    word_emb = torch.randn(batch_size, 18, word_dim)

    img_64, img_128, img_256, mu, logvar = netG(z, sent_emb, word_emb)
    assert img_64.shape == (batch_size, 3, 64, 64)
    assert img_128.shape == (batch_size, 3, 128, 128)
    assert img_256.shape == (batch_size, 3, 256, 256)
    assert mu.shape == (batch_size, nz)
    assert logvar.shape == (batch_size, nz)


def test_discriminator_64():
    net_d = D_NET64(ndf=64, nef=512)
    fake_imgs = torch.randn(2, 3, 64, 64)
    sent_emb = torch.randn(2, 512)
    logits = net_d(fake_imgs, sent_emb)
    assert logits.shape == (2,)


def test_discriminator_128():
    net_d = D_NET128(ndf=64, nef=512)
    fake_imgs = torch.randn(2, 3, 128, 128)
    sent_emb = torch.randn(2, 512)
    logits = net_d(fake_imgs, sent_emb)
    assert logits.shape == (2,)


def test_discriminator_256():
    net_d = D_NET256(ndf=64, nef=512)
    fake_imgs = torch.randn(2, 3, 256, 256)
    sent_emb = torch.randn(2, 512)
    logits = net_d(fake_imgs, sent_emb)
    assert logits.shape == (2,)


def test_text_dataset_missing_image_raises_file_not_found(tmp_path):
    data_dir, _ = _build_minimal_dataset_root(tmp_path)

    captions = [["missing"] * 10]
    with open(data_dir / "train" / "captions.pickle", "wb") as f:
        pickle.dump(captions, f)

    ds = TextImageDataset(str(data_dir), split="train", transform=None)

    with pytest.raises((FileNotFoundError, RuntimeError)):
        _ = ds[0]


def test_spectral_norm_applied_to_all_discriminator_conv_layers():
    for DClass, res in [(D_NET64, 64), (D_NET128, 128), (D_NET256, 256)]:
        net_d = DClass(ndf=64, nef=512)
        conv_layers = [m for m in net_d.modules() if isinstance(m, torch.nn.Conv2d)]
        assert conv_layers, f"Expected Conv2d layers in D_NET{res}"
        without_sn = [m for m in conv_layers if not hasattr(m, "weight_u")]
        assert not without_sn, f"D_NET{res} has Conv2d layers missing spectral normalization"