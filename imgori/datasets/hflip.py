import random
from pathlib import Path
from typing import Callable
from typing import List

from mlconfig import register
from PIL import Image
from PIL import ImageOps
from torch.utils.data import DataLoader
from torch.utils.data import Dataset
from torchvision import transforms

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

        flip = random.random() < self.p
        if flip:
            img = ImageOps.mirror(img)

        if self.transform is not None:
            img = self.transform(img)

        return img, int(flip)


@register
class RandomHorizontalFlipDataLoader(DataLoader):
    def __init__(
        self, root: str, train: bool, batch_size: int, image_size: int = 256, **kwargs
    ):
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

        dataset = RandomHorizontalFlipDataset(
            root, train=train, transform=transform, download=True
        )

        super(RandomHorizontalFlipDataLoader, self).__init__(
            dataset=dataset, batch_size=batch_size, shuffle=train, **kwargs
        )
