import os
import pickle

import numpy as np
import pandas as pd
import torch
from PIL import Image
from torch.utils.data import Dataset
from torchvision import transforms


class TextImageDataset(Dataset):
    """CUB-200-2011 text-image paired dataset.

    Loads images cropped by bounding box and resized to
    `image_size`. Each image is accompanied by 10 captions
    (stored as word indices). One caption is randomly sampled
    per __getitem__ call.
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
        self.class_id = self._load_pickle(self.split_dir, "class_info.pickle",
                                          encoding="latin1")
        self.bbox = self._load_bbox()
        self.embeddings_num = 10

    def _load_pickle(self, data_dir, filename, encoding=None):
        path = os.path.join(data_dir, filename)
        if not os.path.isfile(path):
            raise FileNotFoundError(f"Missing pickle: {path}")
        with open(path, "rb") as f:
            return pickle.load(f, encoding=encoding) if encoding else pickle.load(f)

    def _load_bbox(self):
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
        return len(self.filenames)
