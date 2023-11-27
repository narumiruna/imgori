import random
from pathlib import Path
from typing import Callable
from typing import List

from PIL import Image
from PIL import ImageOps
from torch.utils.data import Dataset

from ..typing import PILImage


def get_image_extensions() -> List[str]:
    Image.init()
    return list(Image.EXTENSION.keys())


class RandomHorizontalFlipDataset(Dataset):
    def __init__(self, root: str, transform: Callable, p: float = 0.5) -> None:
        self.root = Path(root)
        self.transform = transform
        self.p = p

        if self.p < 0 or self.p > 1:
            raise ValueError("p must be in [0, 1]")

        exts = get_image_extensions()
        self.paths = [p for p in self.root.rglob("*") if p.suffix in exts]

    def __len__(self) -> int:
        return len(self.paths)

    def __getitem__(self, index: int) -> (PILImage, int):
        path = self.paths[index]
        img = Image.open(path).convert("RGB")

        if self.transform is not None:
            img = self.transform(img)

        if random.random() < self.p:
            return ImageOps.mirror(img), 1

        return img, 0
