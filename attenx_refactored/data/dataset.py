"""
CUB-200-2011 text-image paired dataset.

Loads bird images cropped by bounding box and resized to the
target resolution. Each image is associated with 10 captions
(stored as pre-tokenized word indices), one of which is
randomly sampled per __getitem__ call.
"""

import os
import pickle

import numpy as np
import pandas as pd
import torch
from PIL import Image
from torch.utils.data import Dataset
from torchvision import transforms


class TextImageDataset(Dataset):
    """
    CUB-200-2011 dataset for text-to-image generation.

    Loads images cropped by bounding box and resized to `image_size`.
    Each image has 10 captions; one is randomly chosen per access.

    Args:
        data_dir: Root directory containing CUB_200_2011 and split dirs.
        split: 'train' or 'test'.
        image_size: Target resolution for images.
        transform: Optional custom transform (default: resize+normalize).
    """

    def __init__(self, data_dir, split="train", image_size=256, transform=None):
        super().__init__()
        self.data_dir = data_dir
        self.split = split
        self.image_size = image_size

        if transform is None:
            self.transform = transforms.Compose([
                transforms.Resize((image_size, image_size)),
                transforms.ToTensor(),
                transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)),
            ])
        else:
            self.transform = transform

        self.cub_dir = os.path.join(data_dir, "CUB_200_2011")
        self.split_dir = os.path.join(data_dir, split)

        self.filenames = self._load_pickle(self.split_dir, "filenames.pickle")
        self.captions = self._load_pickle(self.split_dir, "captions.pickle")
        self.class_id = self._load_pickle(self.split_dir, "class_info.pickle", encoding="latin1")
        self.bbox = self._load_bbox()
        self.embeddings_num = 10

    def _load_pickle(self, data_dir, filename, encoding=None):
        """
        Load a pickle file from the given directory.

        Args:
            data_dir: Directory containing the pickle file.
            filename: Name of the pickle file.
            encoding: Optional encoding (e.g., 'latin1' for legacy files).

        Returns:
            Unpickled object.

        Raises:
            FileNotFoundError: If the pickle file does not exist.
        """
        path = os.path.join(data_dir, filename)
        if not os.path.isfile(path):
            raise FileNotFoundError(f"Missing pickle: {path}")
        with open(path, "rb") as f:
            return pickle.load(f, encoding=encoding) if encoding else pickle.load(f)

    def _load_bbox(self):
        """
        Load bounding box annotations from CUB_200_2011.

        Returns:
            Dict mapping image filename (without .jpg) to [x, y, w, h].
        """
        bbox_path = os.path.join(self.cub_dir, "bounding_boxes.txt")
        img_path = os.path.join(self.cub_dir, "images.txt")

        bbox_df = pd.read_csv(bbox_path, sep=r"\s+", header=None).astype(int)
        img_df = pd.read_csv(img_path, sep=r"\s+", header=None)
        filenames = img_df[1].tolist()

        mapping = {}
        for i, fname in enumerate(filenames):
            mapping[fname[:-4]] = bbox_df.iloc[i][1:].tolist()
        return mapping

    def __getitem__(self, index):
        """
        Load an image-caption pair by index.

        Crops the image using its bounding box (with 0.75 padding),
        applies transforms, and randomly selects one of 10 captions.

        Args:
            index: Dataset index.

        Returns:
            Tuple of (image, caption, cap_len, class_id, filename_key).
        """
        if index >= len(self.filenames):
            raise IndexError(f"Index {index} out of bounds")

        key = self.filenames[index]
        cls_id = self.class_id[index]
        img_name = os.path.join(self.cub_dir, "images", key + ".jpg")
        bbox = self.bbox.get(key)

        img = Image.open(img_name).convert("RGB")
        width, height = img.size

        if bbox is not None:
            r = int(np.maximum(bbox[2], bbox[3]) * 0.75)
            cx = int((2 * bbox[0] + bbox[2]) / 2)
            cy = int((2 * bbox[1] + bbox[3]) / 2)
            y1 = max(0, cy - r)
            y2 = min(height, cy + r)
            x1 = max(0, cx - r)
            x2 = min(width, cx + r)
            img = img.crop([x1, y1, x2, y2])

        if self.transform:
            img = self.transform(img)

        caps = self.captions[index]
        sent_ix = np.random.randint(0, self.embeddings_num)
        caption = torch.LongTensor(caps[sent_ix])
        cap_len = torch.LongTensor([len(caps[sent_ix])])

        return img, caption, cap_len, cls_id, key

    def __len__(self):
        """
        Return the total number of samples in the dataset.
        """
        return len(self.filenames)
