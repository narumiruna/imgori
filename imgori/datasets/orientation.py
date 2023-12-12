from enum import Enum
from pathlib import Path
from typing import Callable

from loguru import logger
from mlconfig import register
from PIL import ImageOps
from torch.utils.data import DataLoader
from torch.utils.data import Dataset
from torchvision.transforms._presets import ImageClassification
from tqdm import tqdm

from ..typing import PILImage
from .phase import Phase
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
    def __init__(
        self, root: str, phase: Phase, transform: Callable, cache: bool = True
    ) -> None:
        self.root = Path(root)
        self.phase = phase
        self.transform = transform
        self.cache = cache

        exts = get_image_extensions()
        self.images = [p for p in self.root.rglob(f"{phase}/*") if p.suffix in exts]
        if cache:
            logger.info("cache images")
            self.images = [read_image(p) for p in tqdm(self.images)]

    def __len__(self) -> int:
        return len(self.images) * len(Orientation)

    def __getitem__(self, index: int) -> (PILImage, int):
        ori_len = len(Orientation)

        img_index = index // ori_len
        ori_index = index % ori_len
        assert 0 <= img_index < len(self.images)
        assert 0 <= ori_index < ori_len

        img = self.images[img_index]
        if not self.cache:
            img = read_image(img)

        ori = Orientation(ori_index)
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
        self,
        root: str,
        phase: Phase,
        batch_size: int,
        crop_size: int = 224,
        resize_size: int = 256,
        **kwargs,
    ) -> None:
        super(RandomOrientationDataLoader, self).__init__(
            dataset=RandomOrientationDataset(
                root,
                phase=phase,
                transform=ImageClassification(
                    crop_size=crop_size, resize_size=resize_size
                ),
            ),
            batch_size=batch_size,
            shuffle=True,
            **kwargs,
        )
