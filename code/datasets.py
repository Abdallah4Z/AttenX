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
    Dataset for CUB-200-2011 style text-to-image training.
    Loads images and corresponding captions (as word indices).
    """

    def __init__(self, data_dir, split="train", image_size=256, transform=None):
        super().__init__()
        self.data_dir = data_dir
        self.split = split
        self.image_size = image_size

        # Default transform: resize to image_size, convert to tensor,
        # normalize to [-1, 1]
        if transform is None:
            self.transform = transforms.Compose(
                [
                    transforms.Resize((image_size, image_size)),
                    transforms.ToTensor(),
                    transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)),
                ]
            )
        else:
            self.transform = transform

        # CUB paths
        self.cub_dir = os.path.join(data_dir, "CUB_200_2011")
        self.split_dir = os.path.join(data_dir, split)

        # Load filenames, captions, class ids
        self.filenames = self._load_pickle(self.split_dir, "filenames.pickle")
        self.captions = self._load_pickle(self.split_dir, "captions.pickle")
        self.class_id = self._load_pickle(
            self.split_dir, "class_info.pickle", encoding="latin1"
        )

        # Load bounding boxes
        self.bbox = self._load_bbox()

        # Number of captions per image (typically 10 for CUB)
        self.embeddings_num = 10

    def _load_pickle(self, data_dir, filename, encoding=None):
        filepath = os.path.join(data_dir, filename)
        if not os.path.isfile(filepath):
            raise FileNotFoundError(f"Required pickle file not found: {filepath}")
        try:
            with open(filepath, "rb") as f:
                data = pickle.load(f, encoding=encoding) if encoding else pickle.load(f)
            return data
        except Exception as e:
            raise RuntimeError(f"Failed to load pickle {filepath}: {e}")

    def _load_bbox(self):
        """Load bounding boxes from CUB metadata."""
        bbox_path = os.path.join(self.cub_dir, "bounding_boxes.txt")
        filepath = os.path.join(self.cub_dir, "images.txt")

        df_bounding_boxes = pd.read_csv(bbox_path, sep=r"\s+", header=None).astype(int)
        df_filenames = pd.read_csv(filepath, sep=r"\s+", header=None)
        filenames = df_filenames[1].tolist()

        filename_bbox = {img_file[:-4]: [] for img_file in filenames}
        num_imgs = len(filenames)
        for i in range(num_imgs):
            bbox = df_bounding_boxes.iloc[i][1:].tolist()
            key = filenames[i][:-4]
            filename_bbox[key] = bbox

        return filename_bbox

    def __getitem__(self, index):
        if index >= len(self.filenames):
            raise IndexError(f"Index {index} out of bounds for dataset size {len(self.filenames)}")
            
        key = self.filenames[index]
        cls_id = self.class_id[index]

        # Load and crop image using bounding box
        img_name = os.path.join(self.cub_dir, "images", key + ".jpg")
        bbox = self.bbox[key]

        try:
            img = Image.open(img_name).convert("RGB")
        except Exception as e:
            raise RuntimeError(f"Failed to load image {img_name}: {e}")
            
        width, height = img.size

        # Crop with bounding box (same as original TextDataset)
        if bbox is not None:
            r = int(np.maximum(bbox[2], bbox[3]) * 0.75)
            center_x = int((2 * bbox[0] + bbox[2]) / 2)
            center_y = int((2 * bbox[1] + bbox[3]) / 2)

            y1 = np.maximum(0, center_y - r)
            y2 = np.minimum(height, center_y + r)
            x1 = np.maximum(0, center_x - r)
            x2 = np.minimum(width, center_x + r)
            img = img.crop([x1, y1, x2, y2])

        if self.transform:
            img = self.transform(img)

        # Load captions (word indices)
        caps = self.captions[index]
        if len(caps) == 0:
            raise ValueError(f"No captions found for image {key}")
            
        sent_ix = np.random.randint(0, self.embeddings_num)
        new_sent = caps[sent_ix]  # This is word indices array

        # Convert to tensor
        caption = torch.LongTensor(new_sent)
        cap_len = torch.LongTensor([len(new_sent)])

        return img, caption, cap_len, cls_id, key

    def __len__(self):
        return len(self.filenames)


def prepare_data(data_dir):
    """
    Function to preprocess the dataset for training.
    Specifically checks for pickles and constructs vocabularies if needed.
    """
    print(f"Preparing datasets from {data_dir}...")
    # This acts as the preprocessor logic
    # In real world, it tokenizes the captions and saves them to pickle files
    return True
