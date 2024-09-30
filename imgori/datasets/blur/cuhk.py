from pathlib import Path
from typing import Any

import torch
from loguru import logger
from PIL import Image
from torch.utils.data import DataLoader
from torch.utils.data import Dataset
from torchvision import tv_tensors
from torchvision.transforms import v2


def make_dataset(root: str) -> tuple[Path, Path]:
    root = Path(root)

    image_dir: Path = root / "image"
    gt_dir: Path = root / "gt"

    samples = []
    for img_path in image_dir.rglob("*"):
        if img_path.suffix.lower() != ".jpg":
            logger.warning(f"Unsupported image format: {img_path}")
            continue

        gt_path = gt_dir / img_path.relative_to(image_dir).with_suffix(".png")

        if gt_path.exists():
            samples.append((img_path, gt_path))
        else:
            logger.warning(f"GT not found: {gt_path}")

    return samples


class CUHK(Dataset):
    def __init__(self, root: str, transform=None) -> None:
        """https://www.cse.cuhk.edu.hk/leojia/projects/dblurdetect/"""
        self.root = Path(root)
        self.transform = transform
        self.samples = make_dataset(root)

    def __getitem__(self, index: int) -> Any:
        img_path, gt_path = self.samples[index]

        img = tv_tensors.Image(Image.open(img_path).convert("RGB"))
        gt = tv_tensors.Mask(Image.open(gt_path).convert("L"))

        if self.transform is not None:
            img, gt = self.transform(img, gt)

        return img, gt

    def __len__(self) -> int:
        return len(self.samples)


class CUHKDataLoader(DataLoader):
    def __init__(self, root: str, **kwargs) -> None:
        transform = v2.Compose(
            [
                v2.ToImage(),
                v2.Resize(size=[224, 224]),
                v2.ToDtype(torch.float32, scale=True),
            ]
        )
        dataset = CUHK(root, transform=transform)
        super().__init__(
            dataset,
            # collate_fn=lambda batch: tuple(zip(*batch)),
            **kwargs,
        )
