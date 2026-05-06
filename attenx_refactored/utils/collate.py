import torch
from torch.nn.utils.rnn import pad_sequence


def collate_fn(batch):
    images, captions, cap_lens, cls_ids, keys = zip(*batch)
    images = torch.stack(images, 0)
    cap_lens = torch.stack(cap_lens, 0).squeeze(-1)
    captions = pad_sequence(captions, batch_first=True, padding_value=0)
    return images, captions, cap_lens, cls_ids, keys
