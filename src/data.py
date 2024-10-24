import os
import torch
from PIL import Image

from torch.utils.data import Dataset

class YoloDataset(Dataset):
    def __init__(self, images_dir, labels_dir, transform=None):
        self.images_dir = images_dir
        self.labels_dir = labels_dir
        self.transform = transform
        self.image_files = [f for f in os.listdir(images_dir) if f.endswith('.jpg')]

    def __len__(self):
        return len(self.image_files)

    def __getitem__(self, idx):
        img_path = os.path.join(self.images_dir, os.listdir(self.images_dir)[idx])
        img = Image.open(img_path).convert('RGB')

        label_path = os.path.join(self.labels_dir,os.listdir(self.images_dir)[idx].replace('.jpg', '.txt'))
        with open(label_path, 'r') as f:
            lines = f.readlines()
            class_ids = [int(line.split()[0]) for line in lines]

        if class_ids:
            label = class_ids[0]
        else:
            label = 0

        if self.transform:
            img = self.transform(img)

        return img, torch.tensor(label)
