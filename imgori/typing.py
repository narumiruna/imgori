from enum import Enum
from pathlib import Path
from typing import Union

from PIL import Image
from PIL import ImageOps

PILImage = Image.Image
PathLike = Union[str, Path]


class Phase(str, Enum):
    TRAIN = "train"
    VALID = "valid"
    TEST = "test"


class Orientation(int, Enum):
    NORMAL = 0
    FLIP = 1
    CLOCKWISE = 2
    COUNTERCLOCKWISE = 3
    MIRROR = 4
    MIRROR_FLIP = 5
    MIRROR_CLOCKWISE = 6
    MIRROR_COUNTERCLOCKWISE = 7

    def do(self, img: PILImage) -> PILImage:
        match self:
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
                raise ValueError(f"invalid orientation: {self}")
        return img

    def undo(self, img: PILImage) -> PILImage:
        match self:
            case Orientation.NORMAL:
                pass
            case Orientation.FLIP:
                img = ImageOps.flip(img)
            case Orientation.CLOCKWISE:
                img = img.rotate(-90)
            case Orientation.COUNTERCLOCKWISE:
                img = img.rotate(-270)
            case Orientation.MIRROR:
                img = ImageOps.mirror(img)
            case Orientation.MIRROR_FLIP:
                img = ImageOps.flip(img)
                img = ImageOps.mirror(img)
            case Orientation.MIRROR_CLOCKWISE:
                img = img.rotate(-90)
                img = ImageOps.mirror(img)
            case Orientation.MIRROR_COUNTERCLOCKWISE:
                img = img.rotate(-270)
                img = ImageOps.mirror(img)
            case _:
                raise ValueError(f"invalid orientation: {self}")
        return img
