import torch.nn as nn


def build_gan_criterion(device):
    return nn.BCELoss()
