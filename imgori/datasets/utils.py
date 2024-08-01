
from PIL import Image

from ..types import PathLike
from ..types import PILImage


def get_image_extensions() -> list[str]:
    Image.init()
    return list(Image.EXTENSION.keys())


def read_image(f: PathLike) -> PILImage:
    return Image.open(f).convert("RGB")
