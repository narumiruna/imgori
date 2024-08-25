import os
import random
from pathlib import Path
from typing import Callable

import torch
import torchvision.transforms.functional as F
from mlconfig import register
from torch.utils.data import DataLoader
from torch.utils.data import Dataset
from torchvision.transforms._presets import ImageClassification

from .utils import open_image

IMG_EXTENSIONS = (".jpg", ".jpeg", ".png", ".ppm", ".bmp", ".pgm", ".tif", ".tiff", ".webp")


class ColorJitterDataset(Dataset):
    def __init__(
        self,
        root: str | Path,
        transform: Callable | None = None,
        img_extensions: tuple[str] = IMG_EXTENSIONS,
    ) -> None:
        if isinstance(root, str):
            root = os.path.expanduser(root)
        self.root = root
        self.transform = transform
        self.img_extensions = img_extensions

        self.samples = []
        for path, _, fnames in os.walk(self.root):
            for fname in fnames:
                if fname.lower().endswith(self.img_extensions):
                    self.samples.append(os.path.join(path, fname))

    def __getitem__(self, index: int) -> dict[str, torch.Tensor | float]:
        sample = self.samples[index]
        img = open_image(sample)

        if self.transform is not None:
            img = self.transform(img)

        brightness = 1.0
        contrast = 1.0
        saturation = 1.0
        hue = 0.0
        match random.randint(0, 4):
            case 0:
                pass
            case 1:
                brightness = random.uniform(0.0, 2.0)
                img = F.adjust_brightness(img, brightness)
            case 2:
                contrast = random.uniform(0.0, 2.0)
                img = F.adjust_contrast(img, contrast)
            case 3:
                saturation = random.uniform(0.0, 2.0)
                img = F.adjust_saturation(img, saturation)
            case 4:
                hue = random.uniform(-0.5, 0.5)
                img = F.adjust_hue(img, hue)
            case _:
                raise ValueError("Invalid randint value")

        return {
            "image": img,
            "brightness": brightness,
            "contrast": contrast,
            "saturation": saturation,
            "hue": hue,
        }

    def __len__(self) -> int:
        return len(self.samples)


@register
class ColorJitterDataLoader(DataLoader):
    def __init__(
        self,
        root: str | Path,
        batch_size: int,
        crop_size: int = 224,
        resize_size: int = 224,
        **kwargs: dict,
    ) -> None:
        super().__init__(
            dataset=ColorJitterDataset(
                root, transform=ImageClassification(crop_size=crop_size, resize_size=resize_size)
            ),
            batch_size=batch_size,
            **kwargs,
        )
