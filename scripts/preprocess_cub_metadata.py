#!/usr/bin/env python3
"""
Preprocess CUB-200-2011 metadata from the AttnGAN 'birds/' layout
to the train/ and test/ splits expected by TextImageDataset.

Expected input layout (after download_cub.sh extracts birds_text.zip):
    data/
    ├── birds/
    │   ├── captions.pickle      # [train_caps_flat, test_caps_flat, ixtoword, wordtoix]
    │   ├── train/
    │   │   ├── filenames.pickle
    │   │   └── class_info.pickle
    │   └── test/
    │       ├── filenames.pickle
    │       └── class_info.pickle
    └── CUB_200_2011/            # raw images

Output layout (ready for training):
    data/
    ├── train/
    │   ├── filenames.pickle
    │   ├── captions.pickle      # list of 10-caption groups per image
    │   └── class_info.pickle
    └── test/
        ├── filenames.pickle
        ├── captions.pickle
        └── class_info.pickle
"""

import os
import pickle
import shutil
import sys


def preprocess_cub_metadata(data_dir="./data"):
    birds_dir = os.path.join(data_dir, "birds")
    if not os.path.isdir(birds_dir):
        raise FileNotFoundError(
            f"Expected 'birds/' directory not found under {data_dir}. "
            "Run data/download_cub.sh first."
        )

    captions_path = os.path.join(birds_dir, "captions.pickle")
    if not os.path.isfile(captions_path):
        raise FileNotFoundError(
            f"Required file not found: {captions_path}"
        )

    # Load raw captions (AttnGAN format)
    with open(captions_path, "rb") as f:
        raw_data = pickle.load(f, encoding="latin1")

    # raw_data is typically [train_captions, test_captions, ixtoword, wordtoix]
    if not isinstance(raw_data, (list, tuple)) or len(raw_data) < 2:
        raise ValueError(
            f"Unexpected captions.pickle format: {type(raw_data)} with length {len(raw_data)}"
        )

    train_caps_flat = raw_data[0]
    test_caps_flat = raw_data[1]

    # Load filenames
    train_filenames_path = os.path.join(birds_dir, "train", "filenames.pickle")
    test_filenames_path = os.path.join(birds_dir, "test", "filenames.pickle")

    for path in (train_filenames_path, test_filenames_path):
        if not os.path.isfile(path):
            raise FileNotFoundError(f"Required file not found: {path}")

    with open(train_filenames_path, "rb") as f:
        train_filenames = pickle.load(f, encoding="latin1")
    with open(test_filenames_path, "rb") as f:
        test_filenames = pickle.load(f, encoding="latin1")

    # Reshape flat caption lists into per-image groups
    def reshape_captions(flat_captions, num_images):
        total = len(flat_captions)
        if total % num_images != 0:
            raise ValueError(
                f"Cannot evenly divide {total} captions into {num_images} images "
                f"(remainder {total % num_images})"
            )
        per_image = total // num_images
        grouped = [
            flat_captions[i * per_image : (i + 1) * per_image]
            for i in range(num_images)
        ]
        return grouped

    train_captions_grouped = reshape_captions(train_caps_flat, len(train_filenames))
    test_captions_grouped = reshape_captions(test_caps_flat, len(test_filenames))

    # Ensure output directories exist
    train_out = os.path.join(data_dir, "train")
    test_out = os.path.join(data_dir, "test")
    os.makedirs(train_out, exist_ok=True)
    os.makedirs(test_out, exist_ok=True)

    # Save grouped captions
    with open(os.path.join(train_out, "captions.pickle"), "wb") as f:
        pickle.dump(train_captions_grouped, f)
    with open(os.path.join(test_out, "captions.pickle"), "wb") as f:
        pickle.dump(test_captions_grouped, f)

    # Copy filenames and class_info pickles
    for split in ("train", "test"):
        src_dir = os.path.join(birds_dir, split)
        dst_dir = os.path.join(data_dir, split)
        for filename in ("filenames.pickle", "class_info.pickle"):
            src = os.path.join(src_dir, filename)
            dst = os.path.join(dst_dir, filename)
            if os.path.isfile(src):
                shutil.copy2(src, dst)
            else:
                print(f"Warning: {src} not found, skipping.")

    print("CUB metadata preprocessed successfully!")
    print(f"  Train: {len(train_filenames)} images, {len(train_captions_grouped)} caption groups")
    print(f"  Test:  {len(test_filenames)} images, {len(test_captions_grouped)} caption groups")


if __name__ == "__main__":
    data_dir = sys.argv[1] if len(sys.argv) > 1 else "./data"
    preprocess_cub_metadata(data_dir)
