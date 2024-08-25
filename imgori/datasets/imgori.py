from collections.abc import Callable
from pathlib import Path
from typing import Any

from loguru import logger
from mlconfig import register
from torch.utils.data import DataLoader
from torch.utils.data import Dataset
from torchvision.transforms._presets import ImageClassification
from tqdm import tqdm

from ..types import Orientation
from ..types import Phase
from .utils import get_image_extensions
from .utils import open_image


class ImgoriDataset(Dataset):
    def __init__(
        self,
        root: str,
        phase: Phase,
        transform: Callable | None = None,
        cache: bool = True,
    ) -> None:
        self.root = Path(root) / phase
        self.phase = phase
        self.transform = transform
        self.cache = cache

        exts = get_image_extensions()
        self.image_paths = [p for p in self.root.rglob("*") if p.suffix in exts]
        if cache:
            logger.info("cache images")
            self.images = [open_image(p) for p in tqdm(self.image_paths)]

    def __len__(self) -> int:
        return len(self.image_paths) * len(Orientation)

    def __getitem__(self, index: int) -> tuple[Any, int]:
        ori_len = len(Orientation)

        img_index = index // ori_len
        ori_index = index % ori_len
        assert 0 <= img_index < len(self.image_paths)
        assert 0 <= ori_index < ori_len

        img = self.images[img_index] if self.cache else open_image(self.image_paths[img_index])

        ori = Orientation(ori_index)
        img = ori.do(img)

        if self.transform is not None:
            img = self.transform(img)

        return img, int(ori)


@register
class ImgoriDataLoader(DataLoader):
    def __init__(
        self,
        root: str,
        phase: Phase,
        batch_size: int,
        crop_size: int = 224,
        resize_size: int = 224,
        **kwargs,
    ) -> None:
        super().__init__(
            dataset=ImgoriDataset(
                root,
                phase=phase,
                transform=ImageClassification(crop_size=crop_size, resize_size=resize_size),
            ),
            batch_size=batch_size,
            shuffle=True,
            **kwargs,
        )
