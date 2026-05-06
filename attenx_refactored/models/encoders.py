"""
Text and image encoders for the DAMSM (Deep Attentional Multimodal
Similarity Model).

- RNN_ENCODER: Bidirectional LSTM that encodes word indices into
  word-level features and a global sentence vector.
- CNN_ENCODER: Pretrained Inception-v3 (frozen) that extracts local
  feature maps and global image codes.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision import models


class RNN_ENCODER(nn.Module):
    """
    Bidirectional LSTM text encoder (DAMSM).

    Embeds word indices, applies dropout, and runs through a
    bidirectional LSTM. Returns word-level features and a global
    sentence embedding (concatenated final hidden states).

    Args:
        n_words: Vocabulary size.
        nhidden: LSTM hidden dimension per direction.
        nembed: Word embedding dimension.
        nlayers: Number of LSTM layers.
    """

    def __init__(self, n_words, nhidden=256, nembed=256, nlayers=1):
        super().__init__()
        self.n_words = n_words
        self.nhidden = nhidden
        self.nembed = nembed

        self.word_embeddings = nn.Embedding(n_words, nembed)
        self.dropout = nn.Dropout(0.5)
        self.rnn = nn.LSTM(nembed, nhidden, nlayers,
                           dropout=0.5 if nlayers > 1 else 0,
                           bidirectional=True)

    def forward(self, captions, cap_lens, hidden=None):
        """
        Encode caption word indices into features.

        Args:
            captions: Padded word index tensor (B, seq_len).
            cap_lens: Length of each caption (B,).
            hidden: Optional initial hidden state for the LSTM.

        Returns:
            Tuple of (word_embeddings, sent_emb) where:
                word_embeddings: Word-level features (B, seq_len, nhidden*2).
                sent_emb: Global sentence embedding (B, nhidden*2).
        """
        embeddings = self.word_embeddings(captions)
        embeddings = self.dropout(embeddings)

        packed = nn.utils.rnn.pack_padded_sequence(
            embeddings, cap_lens.cpu(), batch_first=True, enforce_sorted=False,
        )
        output, hidden = self.rnn(packed, hidden)
        output, _ = nn.utils.rnn.pad_packed_sequence(output, batch_first=True)

        sent_emb = hidden[0].transpose(0, 1).contiguous()
        sent_emb = sent_emb.view(-1, self.nhidden * 2)

        return output, sent_emb


class CNN_ENCODER(nn.Module):
    """
    Inception-v3 based image encoder (DAMSM).

    Uses a pretrained Inception-v3 backbone (frozen) to extract
    local feature maps (768-dim) and global image codes (2048-dim).
    Projects both to the DAMSM embedding dimension (nef).

    Args:
        nef: Target embedding dimension for features and codes.
    """

    def __init__(self, nef=512):
        super().__init__()
        self.nef = nef

        model = models.inception_v3(pretrained=True)
        for param in model.parameters():
            param.requires_grad = False
        self._define_module(model)

    def _define_module(self, model):
        """
        Extract Inception-v3 submodules and add projection layers.

        Args:
            model: Pretrained Inception-v3 model.
        """
        self.Conv2d_1a_3x3 = model.Conv2d_1a_3x3
        self.Conv2d_2a_3x3 = model.Conv2d_2a_3x3
        self.Conv2d_2b_3x3 = model.Conv2d_2b_3x3
        self.Conv2d_3b_1x1 = model.Conv2d_3b_1x1
        self.Conv2d_4a_3x3 = model.Conv2d_4a_3x3
        self.Mixed_5b = model.Mixed_5b
        self.Mixed_5c = model.Mixed_5c
        self.Mixed_5d = model.Mixed_5d
        self.Mixed_6a = model.Mixed_6a
        self.Mixed_6b = model.Mixed_6b
        self.Mixed_6c = model.Mixed_6c
        self.Mixed_6d = model.Mixed_6d
        self.Mixed_6e = model.Mixed_6e
        self.Mixed_7a = model.Mixed_7a
        self.Mixed_7b = model.Mixed_7b
        self.Mixed_7c = model.Mixed_7c

        self.emb_features = nn.Conv2d(768, self.nef, 1, 1, 0, bias=False)
        self.emb_cnn_code = nn.Linear(2048, self.nef)

    def forward(self, x):
        """
        Extract image features and global code.

        Resizes input to 299x299 if needed, runs through Inception-v3,
        and returns local feature maps and a global image code.

        Args:
            x: Input image tensor (B, 3, H, W).

        Returns:
            Tuple of (features, cnn_code) where:
                features: Spatial feature maps (B, nef, H', W').
                cnn_code: Global image code (B, nef).
        """
        if x.shape[-1] != 299:
            x = F.interpolate(x, size=(299, 299), mode="bilinear", align_corners=False)

        x = self.Conv2d_1a_3x3(x)
        x = self.Conv2d_2a_3x3(x)
        x = self.Conv2d_2b_3x3(x)
        x = F.max_pool2d(x, 3, stride=2)
        x = self.Conv2d_3b_1x1(x)
        x = self.Conv2d_4a_3x3(x)
        x = F.max_pool2d(x, 3, stride=2)
        x = self.Mixed_5b(x)
        x = self.Mixed_5c(x)
        x = self.Mixed_5d(x)
        x = self.Mixed_6a(x)
        x = self.Mixed_6b(x)
        x = self.Mixed_6c(x)
        x = self.Mixed_6d(x)
        x = self.Mixed_6e(x)

        features = self.emb_features(x)

        x = self.Mixed_7a(x)
        x = self.Mixed_7b(x)
        x = self.Mixed_7c(x)
        x = F.avg_pool2d(x, kernel_size=8)
        x = x.view(x.size(0), -1)
        cnn_code = self.emb_cnn_code(x)

        return features, cnn_code
