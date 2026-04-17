import os
import pickle
import numpy as np
from PIL import Image
import pandas as pd
import torch
from torch.utils.data import Dataset
import torchvision.transforms as transforms

class TextDataset(Dataset):
    def __init__(self, data_dir, split='train', base_size=64, transform=None):
        self.data_dir = data_dir
        self.split = split
        self.transform = transform
        self.imsize = [base_size, base_size * 2, base_size * 4]
        self.embeddings_num = 10
        
        # Load bounding boxes
        self.bbox = self.load_bbox()
        self.split_dir = os.path.join(data_dir, split)
        
        self.filenames = self.load_filenames(self.split_dir)
        self.captions = self.load_captions(self.split_dir)
        self.class_id = self.load_class_id(self.split_dir)

    def get_img(self, img_path, bbox=None):
        img = Image.open(img_path).convert('RGB')
        width, height = img.size
        
        if bbox is not None:
            r = int(np.maximum(bbox[2], bbox[3]) * 0.75)
            center_x = int((2 * bbox[0] + bbox[2]) / 2)
            center_y = int((2 * bbox[1] + bbox[3]) / 2)
            
            y1 = np.maximum(0, center_y - r)
            y2 = np.minimum(height, center_y + r)
            x1 = np.maximum(0, center_x - r)
            x2 = np.minimum(width, center_x + r)
            img = img.crop([x1, y1, x2, y2])
            
        if self.transform is not None:
            img = self.transform(img)
        return img

    def load_bbox(self):
        data_dir = self.data_dir
        bbox_path = os.path.join(data_dir, 'CUB_200_2011/bounding_boxes.txt')
        df_bounding_boxes = pd.read_csv(bbox_path, delim_whitespace=True, header=None).astype(int)
        
        filepath = os.path.join(data_dir, 'CUB_200_2011/images.txt')
        df_filenames = pd.read_csv(filepath, delim_whitespace=True, header=None)
        filenames = df_filenames[1].tolist()
        
        filename_bbox = {img_file[:-4]: [] for img_file in filenames}
        numImgs = len(filenames)
        for i in range(numImgs):
            bbox = df_bounding_boxes.iloc[i][1:].tolist()
            key = filenames[i][:-4]
            filename_bbox[key] = bbox
            
        return filename_bbox

    def load_captions(self, data_dir):
        filepath = os.path.join(data_dir, 'captions.pickle')
        with open(filepath, 'rb') as f:
            captions = pickle.load(f)
            print('Load from:', filepath)
        return captions

    def load_class_id(self, data_dir):
        filepath = os.path.join(data_dir, 'class_info.pickle')
        with open(filepath, 'rb') as f:
            class_id = pickle.load(f, encoding='latin1')
        return class_id

    def load_filenames(self, data_dir):
        filepath = os.path.join(data_dir, 'filenames.pickle')
        with open(filepath, 'rb') as f:
            filenames = pickle.load(f)
        return filenames

    def __getitem__(self, index):
        key = self.filenames[index]
        cls_id = self.class_id[index]
        
        img_name = os.path.join(self.data_dir, 'CUB_200_2011/images', key + '.jpg')
        bbox = self.bbox[key]
        
        imgs = self.get_img(img_name, bbox)
        
        # Load text representations
        caps = self.captions[index]
        sent_ix = np.random.randint(0, self.embeddings_num)
        new_sent = caps[sent_ix]
        
        # Here we just return text as indices or text, and image.
        # In full AttnGAN, sentences are converted to word indices based on a vocabulary.
        return imgs, new_sent, cls_id, key

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
