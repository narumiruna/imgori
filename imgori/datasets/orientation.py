import random
from enum import Enum
from pathlib import Path
from typing import Callable

from loguru import logger
from mlconfig import register
from PIL import Image
from PIL import ImageOps
from torch.utils.data import DataLoader
from torch.utils.data import Dataset
from torchvision import transforms
from tqdm import tqdm

from ..typing import PILImage
from .utils import get_image_extensions
from .utils import read_image


class Orientation(int, Enum):
    NORMAL = 0
    FLIP = 1
    CLOCKWISE = 2
    COUNTERCLOCKWISE = 3
    MIRROR = 4
    MIRROR_FLIP = 5
    MIRROR_CLOCKWISE = 6
    MIRROR_COUNTERCLOCKWISE = 7


class RandomOrientationDataset(Dataset):
    def __init__(self, root: str, transform: Callable, cache: bool = True) -> None:
        self.root = Path(root)
        self.transform = transform
        self.cache = cache

        exts = get_image_extensions()
        self.images = [p for p in self.root.rglob("*") if p.suffix in exts]
        if cache:
            logger.info("cache images")
            self.images = [read_image(p) for p in tqdm(self.images)]

    def __len__(self) -> int:
        return len(self.images)

    def __getitem__(self, index: int) -> (PILImage, int):
        img = self.images[index]
        if not self.cache:
            img = read_image(img)

        ori = random.choice(list(Orientation))
        match ori:
            case Orientation.NORMAL:
                pass
            case Orientation.FLIP:
                img = ImageOps.flip(img)
            case Orientation.CLOCKWISE:
                img = img.rotate(90)
            case Orientation.COUNTERCLOCKWISE:
                img = img.rotate(270)
            case Orientation.MIRROR:
                img = ImageOps.mirror(img)
            case Orientation.MIRROR_FLIP:
                img = ImageOps.mirror(img)
                img = ImageOps.flip(img)
            case Orientation.MIRROR_CLOCKWISE:
                img = ImageOps.mirror(img)
                img = img.rotate(90)
            case Orientation.MIRROR_COUNTERCLOCKWISE:
                img = ImageOps.mirror(img)
                img = img.rotate(270)
            case _:
                raise ValueError(f"invalid orientation: {ori}")

        if self.transform is not None:
            img = self.transform(img)

        return img, int(ori)


@register
class RandomOrientationDataLoader(DataLoader):
    def __init__(
        self, root: str, train: bool, batch_size: int, image_size: int = 256, **kwargs
    ) -> None:
        normalize = transforms.Normalize(
            mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]
        )

        if train:
            transform = transforms.Compose(
                [
                    transforms.RandomResizedCrop(
                        image_size, interpolation=Image.BICUBIC
                    ),
                    transforms.RandomHorizontalFlip(),
                    transforms.ToTensor(),
                    normalize,
                ]
            )
        else:
            transform = transforms.Compose(
                [
                    transforms.Resize(image_size + 32, interpolation=Image.BICUBIC),
                    transforms.CenterCrop(image_size),
                    transforms.ToTensor(),
                    normalize,
                ]
            )

        dataset = RandomOrientationDataset(root, transform=transform)

        super(RandomOrientationDataLoader, self).__init__(
            dataset=dataset, batch_size=batch_size, shuffle=train, **kwargs
        )
