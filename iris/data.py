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

labels = [
    "tienda",
    "parqueadero",
    "belleza/barbería/peluquería",
    "electrónica/cómputo",
    "café/restaurante",
    "electrodomésticos",
    "talleres carros/motos",
    "zapatería",
    "muebles/tapicería",
    "ferretería",
    "carnicería/fruver",
    "puesto móvil/toldito",
    "farmacia",
    "supermercado",
    "ropa",
    "deporte",
    "licorera",
    "hotel",
    "animales",
    "bar",
]


class LandMarkDataset(Dataset):
    def __init__(
        self,
        img_dir,
        annotations_file: pd.DataFrame,
        transform=None,
        target_transform=None,
    ) -> None:
        self.img_dir = img_dir
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

        label_name = (
            label_metadata["labels"]
            if isinstance(label_metadata["labels"], str)
            else label_metadata["labels"][0]
        )
        label = labels.index(label_name)

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
