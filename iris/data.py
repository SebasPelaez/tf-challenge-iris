import os
import torch
import pandas as pd
from torch.utils.data import Dataset
from torchvision.io import read_image
from typing import Tuple
import json

class LandMarkDataset(Dataset):
    def __init__(self, img_dir, annotations_file: pd.DataFrame, transform=None, target_transform=None) -> None:
        self.img_dir = img_dir
        self.annotations_file = annotations_file
        self.transform = transform
        self.target_transform = target_transform
    
    def __len__(self):
        return self.annotations_file.shape[0]
    
    def __getitem__(self, index) -> Tuple[torch.tensor, torch.tensor]:
        img_metadata_path = os.path.join(self.img_dir, self.annotations_file.iloc[index, 0])
        image = read_image(img_metadata_path + ".png")
        with open(img_metadata_path + ".json", "r") as file:
            label_metadata = json.load(file)
        label = label_metadata["labels"][0]

        if self.transform:
            image = self.transform(image)
        
        if self.target_transform:
            label = self.target_transform(label)

        return image[:3], label
