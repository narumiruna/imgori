from PIL import Image

from ..types import PathLike
from ..types import PILImage


def get_image_extensions() -> list[str]:
    Image.init()
    return list(Image.EXTENSION.keys())


def open_image(f: PathLike) -> PILImage:
    with open(f, "rb") as f:
        img = Image.open(f)
        return img.convert("RGB")
