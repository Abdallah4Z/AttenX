import torch
import torch.nn as nn
import torch.nn.parallel
from torchvision import models
import torch.nn.functional as F

class RNN_ENCODER(nn.Module):
    def __init__(self, n_words, nhidden=256, nembed=256, nlayers=1):
        super(RNN_ENCODER, self).__init__()
        self.n_words = n_words
        self.nhidden = nhidden
        self.nembed = nembed
        self.nlayers = nlayers
        self.word_embeddings = nn.Embedding(n_words, nembed)
        self.dropout = nn.Dropout(0.5)
        self.rnn = nn.LSTM(nembed, nhidden, nlayers, dropout=0.5, bidirectional=True)

    def forward(self, captions, cap_lens, hidden):
        # captions: (batch_size, seq_len)
        embeddings = self.word_embeddings(captions)
        embeddings = self.dropout(embeddings)
        
        # Pack padded sequence
        packed = nn.utils.rnn.pack_padded_sequence(embeddings, cap_lens, batch_first=True, enforce_sorted=False)
        output, hidden = self.rnn(packed, hidden)
        
        # Unpack
        output, _ = nn.utils.rnn.pad_packed_sequence(output, batch_first=True)
        # output: (batch_size, seq_len, nhidden * 2)
        
        # global sentence embedding
        sent_emb = hidden[0].transpose(0, 1).contiguous()
        sent_emb = sent_emb.view(-1, self.nhidden * 2)
        
        return output, sent_emb

class CNN_ENCODER(nn.Module):
    def __init__(self, nef):
        super(CNN_ENCODER, self).__init__()
        self.nef = nef
        model = models.inception_v3(pretrained=True)
        for param in model.parameters():
            param.requires_grad = False
            
        self.define_module(model)

    def define_module(self, model):
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
        
        self.emb_features = nn.Conv2d(768, self.nef, kernel_size=1, stride=1, padding=0, bias=False)
        self.emb_cnn_code = nn.Linear(2048, self.nef)

    def forward(self, x):
        # Initial layers
        x = self.Conv2d_1a_3x3(x)
        x = self.Conv2d_2a_3x3(x)
        x = self.Conv2d_2b_3x3(x)
        x = F.max_pool2d(x, kernel_size=3, stride=2)
        x = self.Conv2d_3b_1x1(x)
        x = self.Conv2d_4a_3x3(x)
        x = F.max_pool2d(x, kernel_size=3, stride=2)
        x = self.Mixed_5b(x)
        x = self.Mixed_5c(x)
        x = self.Mixed_5d(x)
        x = self.Mixed_6a(x)
        x = self.Mixed_6b(x)
        x = self.Mixed_6c(x)
        x = self.Mixed_6d(x)
        x = self.Mixed_6e(x)
        
        # Local features (from Mixed_6e)
        features = self.emb_features(x)
        
        # Global features
        x = self.Mixed_7a(x)
        x = self.Mixed_7b(x)
        x = self.Mixed_7c(x)
        x = F.avg_pool2d(x, kernel_size=8)
        x = x.view(x.size(0), -1)
        cnn_code = self.emb_cnn_code(x)
        
        return features, cnn_code
