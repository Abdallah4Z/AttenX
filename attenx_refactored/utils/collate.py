"""
Shared collation function for DataLoader.

Provides a single collate_fn used across training, evaluation,
and DAMSM pretraining scripts to avoid code duplication.
"""

import torch
from torch.nn.utils.rnn import pad_sequence


def collate_fn(batch):
    """
    Collate a list of (image, caption, cap_len, cls_id, key) tuples into a batch.

    Images are stacked into a 4D tensor. Caption tensors are padded
    to equal length using pad_sequence. Caption lengths are stacked
    and squeezed to a 1D tensor.

    Args:
        batch: List of tuples from TextImageDataset __getitem__.

    Returns:
        Tuple of (images, captions, cap_lens, cls_ids, keys).
    """
    images, captions, cap_lens, cls_ids, keys = zip(*batch)
    images = torch.stack(images, 0)
    cap_lens = torch.stack(cap_lens, 0).squeeze(-1)
    captions = pad_sequence(captions, batch_first=True, padding_value=0)
    return images, captions, cap_lens, cls_ids, keys
