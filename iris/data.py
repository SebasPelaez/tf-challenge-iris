import os
import torch
import pandas as pd
import numpy as np
from torch.utils.data import Dataset
from torchvision.io import read_image
from torchvision.transforms import ToTensor
from typing import Tuple

from albumentations.core.composition import Compose as Acompose

from PIL import Image
import json

id_labels = {
    0: 'animales',
    1: 'bar',
    2: 'belleza/barbería/peluquería',
    3: 'café/restaurante',
    4: 'carnicería/fruver',
    5: 'deporte',
    6: 'electrodomésticos',
    7: 'electrónica/cómputo',
    8: 'farmacia',
    9: 'ferretería',
    10: 'hotel',
    11: 'licorera',
    12: 'muebles/tapicería',
    13: 'parqueadero',
    14: 'puesto móvil/toldito',
    15: 'ropa',
    16: 'supermercado',
    17: 'talleres carros/motos',
    18: 'tienda',
    19: 'zapatería'
}

label_id = {label: id for id, label in id_labels.items()}

class LandMarkDataset(Dataset):
    def __init__(
        self,
        img_dir,
        background_dir,
        annotations_file: pd.DataFrame,
        transform=None,
        target_transform=None,
    ) -> None:
        self.img_dir = img_dir
        self.background_dir = background_dir
        self.annotations_file = annotations_file
        self.transform = transform
        self.target_transform = target_transform

    def __len__(self):
        return self.annotations_file.shape[0]

    def __getitem__(self, index: int) -> Tuple[torch.tensor, torch.tensor]:
        """[summary]

        Args:
            index (int): Index

        Returns:
            Tuple[torch.tensor, torch.tensor]: [description]
        """
        img_metadata_path = os.path.join(
            self.img_dir, self.annotations_file.iloc[index, 0]
        )

        image = Image.open(img_metadata_path + ".png").convert("RGB")
        with open(img_metadata_path + ".json", "r") as file:
            label_metadata = json.load(file)

        
        is_facade = 'is_facade' in label_metadata.keys()
        is_facade = True

        if is_facade and np.random.rand() < 0.7:
            background_image_dir = list(self.background_dir.glob('*'))
            background_image_path = str(np.random.choice(background_image_dir, 1)[0])
            background_image = Image.open(background_image_path)
            image = self.paste_image_into_background(background=background_image, overlay=image)

        label_name = (
            label_metadata["labels"]
            if isinstance(label_metadata["labels"], str)
            else label_metadata["labels"][0]
        )
        label = label_id[label_name]

        if self.transform:
            image = (
                self.transform(image=np.array(image))["image"]
                if isinstance(self.transform, Acompose)
                else self.transform(image)
            )
        else:
            image = ToTensor()(image)

        if self.target_transform:
            label = self.target_transform(label)

        return image, label

    def paste_image_into_background(self, background, overlay):
    
        width, height = overlay.size
        
        left = width//8
        top = height//7 
        right = 7 * (width//8)
        bottom = 6 * (height//7)
        
        crop_overlay = overlay.crop((left, top, right, bottom))
        
        background = background.convert('RGBA')
        crop_overlay = crop_overlay.convert('RGBA')
        background.paste(crop_overlay,(left, top, right, bottom), mask = crop_overlay)
        
        return background.convert("RGB")
