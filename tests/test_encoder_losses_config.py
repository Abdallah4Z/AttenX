import types

import torch
import torch.nn as nn

from code import config
from code.encoder import CNN_ENCODER, RNN_ENCODER
from code.losses import KL_loss, cosine_similarity, sent_loss, words_loss


class _ChannelMap(nn.Module):
    def __init__(self, in_ch, out_ch, target_hw=None):
        super().__init__()
        self.conv = nn.Conv2d(in_ch, out_ch, kernel_size=1)
        self.target_hw = target_hw

    def forward(self, x):
        x = self.conv(x)
        if self.target_hw is not None:
            x = torch.nn.functional.interpolate(x, size=self.target_hw, mode="nearest")
        return x


def _build_fake_inception():
    # Minimal Inception-like object with required attributes for CNN_ENCODER.
    fake = types.SimpleNamespace()
    fake.Conv2d_1a_3x3 = _ChannelMap(3, 32)
    fake.Conv2d_2a_3x3 = _ChannelMap(32, 32)
    fake.Conv2d_2b_3x3 = _ChannelMap(32, 64)
    fake.Conv2d_3b_1x1 = _ChannelMap(64, 80)
    fake.Conv2d_4a_3x3 = _ChannelMap(80, 192)

    fake.Mixed_5b = _ChannelMap(192, 256)
    fake.Mixed_5c = _ChannelMap(256, 288)
    fake.Mixed_5d = _ChannelMap(288, 288)
    fake.Mixed_6a = _ChannelMap(288, 768)
    fake.Mixed_6b = _ChannelMap(768, 768)
    fake.Mixed_6c = _ChannelMap(768, 768)
    fake.Mixed_6d = _ChannelMap(768, 768)
    fake.Mixed_6e = _ChannelMap(768, 768)

    fake.Mixed_7a = _ChannelMap(768, 1280)
    fake.Mixed_7b = _ChannelMap(1280, 2048)
    # Match expected shape before avg_pool2d(kernel_size=8) in CNN_ENCODER.
    fake.Mixed_7c = _ChannelMap(2048, 2048, target_hw=(8, 8))

    # Expose parameters() like a real nn.Module model.
    modules = [
        fake.Conv2d_1a_3x3,
        fake.Conv2d_2a_3x3,
        fake.Conv2d_2b_3x3,
        fake.Conv2d_3b_1x1,
        fake.Conv2d_4a_3x3,
        fake.Mixed_5b,
        fake.Mixed_5c,
        fake.Mixed_5d,
        fake.Mixed_6a,
        fake.Mixed_6b,
        fake.Mixed_6c,
        fake.Mixed_6d,
        fake.Mixed_6e,
        fake.Mixed_7a,
        fake.Mixed_7b,
        fake.Mixed_7c,
    ]

    def _params():
        for m in modules:
            yield from m.parameters()

    fake.parameters = _params
    return fake


def test_config_constants_are_defined():
    assert config.DROPOUT == 0.5
    assert config.LEAKY_RELU_SLOPE == 0.2
    assert config.LATENT_DIM == 100
    assert config.IMAGE_SIZE == 256
    assert config.NDF == 64
    assert config.NEF == 512
    assert config.GF_DIM == 128
    assert config.VOCAB_SIZE == 12000
    assert config.WORD_DIM == 300
    assert config.EMBEDDING_NUM == 10
    assert config.SEQ_LEN == 18


def test_rnn_encoder_forward_shapes():
    torch.manual_seed(3)
    batch_size, seq_len = 4, 18
    enc = RNN_ENCODER(n_words=12000, nhidden=256, nembed=256, nlayers=1)

    captions = torch.randint(0, 12000, (batch_size, seq_len), dtype=torch.long)
    cap_lens = torch.tensor([18, 16, 12, 7], dtype=torch.long)
    hidden = (
        torch.zeros(2, batch_size, 256),
        torch.zeros(2, batch_size, 256),
    )

    words_emb, sent_emb = enc(captions, cap_lens, hidden)
    assert words_emb.shape == (batch_size, seq_len, 512)
    assert sent_emb.shape == (batch_size, 512)


def test_cnn_encoder_forward_with_fake_inception(monkeypatch):
    fake_model = _build_fake_inception()
    monkeypatch.setattr("code.encoder.models.inception_v3", lambda pretrained=True: fake_model)

    enc = CNN_ENCODER(nef=512)
    x = torch.randn(2, 3, 256, 256)
    features, cnn_code = enc(x)

    assert features.ndim == 4
    assert features.shape[0] == 2
    assert features.shape[1] == 512
    assert cnn_code.shape == (2, 512)


def test_losses_end_to_end():
    torch.manual_seed(5)
    batch_size, nef, seq_len = 3, 512, 18

    img_features = torch.randn(batch_size, nef, 17, 17)
    words_emb = torch.randn(batch_size, nef, seq_len)
    cap_lens = torch.tensor([18, 12, 5], dtype=torch.long)

    w = words_loss(img_features, words_emb, labels=None, cap_lens=cap_lens, batch_size=batch_size)
    assert torch.isfinite(w)

    cnn_code = torch.randn(batch_size, nef)
    sent_emb = torch.randn(batch_size, nef)
    s = sent_loss(cnn_code, sent_emb, labels=None, batch_size=batch_size)
    assert torch.isfinite(s)

    c1 = torch.tensor([[1.0, 0.0], [0.0, 1.0]])
    c2 = torch.tensor([[1.0, 0.0], [1.0, 0.0]])
    sim = cosine_similarity(c1, c2, dim=1)
    assert sim.shape == (2,)
    assert torch.all(sim <= 1.0 + 1e-6)

    mu = torch.zeros(4, 8)
    logvar = torch.zeros(4, 8)
    kld = KL_loss(mu, logvar)
    assert torch.isfinite(kld)
    assert kld >= 0
