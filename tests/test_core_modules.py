import os
import pickle

import pytest
import torch
from PIL import Image

from code.datasets import TextDataset
from code.generator import G_NET
from code.model import D_NET256
from code.modules import SelfAttention


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
    # Encourage non-identity behavior so gamma receives a useful gradient.
    loss = -out.pow(2).mean()
    loss.backward()
    opt.step()

    gamma_after = attn.gamma.detach().clone()
    assert not torch.allclose(gamma_before, gamma_after)


def test_generator_initialization_and_forward():
    net_g = G_NET(ngf=64)
    assert isinstance(net_g.attn_stage, SelfAttention)

    z = torch.randn(2, 100, 1, 1)
    out = net_g(z)
    assert out.shape == (2, 3, 256, 256)


def test_discriminator_initialization_and_forward():
    net_d = D_NET256(ndf=64, nef=512)
    assert net_d.get_logits.bcondition is True

    fake_imgs = torch.randn(2, 3, 256, 256)
    sent_emb = torch.randn(2, 512)
    logits = net_d(fake_imgs, sent_emb)
    assert logits.shape == (2,)


def test_text_dataset_missing_image_raises_file_not_found(tmp_path):
    data_dir, _ = _build_minimal_dataset_root(tmp_path)

    captions = [["missing"] * 10]
    with open(data_dir / "train" / "captions.pickle", "wb") as f:
        pickle.dump(captions, f)

    ds = TextDataset(str(data_dir), split="train", transform=None)

    with pytest.raises(FileNotFoundError):
        _ = ds[0]


def test_text_dataset_empty_captions_returns_empty_string(tmp_path):
    data_dir, key = _build_minimal_dataset_root(tmp_path)

    # Ensure the target image exists.
    img_path = data_dir / "CUB_200_2011" / "images" / f"{key}.jpg"
    os.makedirs(img_path.parent, exist_ok=True)
    Image.new("RGB", (32, 32), color=(255, 255, 255)).save(img_path)

    captions = [[""] * 10]
    with open(data_dir / "train" / "captions.pickle", "wb") as f:
        pickle.dump(captions, f)

    ds = TextDataset(str(data_dir), split="train", transform=None)
    _, caption, cls_id, sample_key = ds[0]

    assert caption == ""
    assert cls_id == 0
    assert sample_key == key


def test_spectral_norm_applied_to_all_discriminator_conv_layers():
    net_d = D_NET256(ndf=64, nef=512)

    conv_layers = [m for m in net_d.modules() if isinstance(m, torch.nn.Conv2d)]
    assert conv_layers, "Expected at least one Conv2d layer in discriminator"

    # spectral_norm wraps Conv2d with internal buffers such as weight_u.
    without_sn = [m for m in conv_layers if not hasattr(m, "weight_u")]
    assert not without_sn, "Some Conv2d layers are missing spectral normalization"
