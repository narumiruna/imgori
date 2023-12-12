from typing import List

from PIL import Image

from ..typing import PathLike
from ..typing import PILImage


def get_image_extensions() -> List[str]:
    Image.init()
    return list(Image.EXTENSION.keys())


def read_image(f: PathLike) -> PILImage:
    return Image.open(f).convert("RGB")
