from __future__ import annotations

from enum import Enum
from pathlib import Path

from PIL import Image
from PIL import ImageOps

PILImage = Image.Image
PathLike = str | Path


class Phase(str, Enum):
    TRAIN = "TRAIN"
    VALID = "VALID"
    TEST = "TEST"


class Orientation(str, Enum):
    NORMAL = "NORMAL"
    FLIP = "FLIP"
    CLOCKWISE = "CLOCKWISE"
    COUNTERCLOCKWISE = "COUNTERCLOCKWISE"
    MIRROR = "MIRROR"
    MIRROR_FLIP = "MIRROR_FLIP"
    MIRROR_CLOCKWISE = "MIRROR_CLOCKWISE"
    MIRROR_COUNTERCLOCKWISE = "MIRROR_COUNTERCLOCKWISE"

    @staticmethod
    def from_int(v: int) -> Orientation:
        match v:
            case 0:
                return Orientation.NORMAL
            case 1:
                return Orientation.FLIP
            case 2:
                return Orientation.CLOCKWISE
            case 3:
                return Orientation.COUNTERCLOCKWISE
            case 4:
                return Orientation.MIRROR
            case 5:
                return Orientation.MIRROR_FLIP
            case 6:
                return Orientation.MIRROR_CLOCKWISE
            case 7:
                return Orientation.MIRROR_COUNTERCLOCKWISE
            case _:
                raise ValueError(f"invalid orientation: {v}")

    def do(self, img: PILImage) -> PILImage:
        match self:
            case Orientation.NORMAL:
                pass
            case Orientation.FLIP:
                img = ImageOps.flip(img)
            case Orientation.CLOCKWISE:
                img = img.rotate(90, expand=True)
            case Orientation.COUNTERCLOCKWISE:
                img = img.rotate(270, expand=True)
            case Orientation.MIRROR:
                img = ImageOps.mirror(img)
            case Orientation.MIRROR_FLIP:
                img = ImageOps.mirror(img)
                img = ImageOps.flip(img)
            case Orientation.MIRROR_CLOCKWISE:
                img = ImageOps.mirror(img)
                img = img.rotate(90, expand=True)
            case Orientation.MIRROR_COUNTERCLOCKWISE:
                img = ImageOps.mirror(img)
                img = img.rotate(270, expand=True)
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
                img = img.rotate(-90, expand=True)
            case Orientation.COUNTERCLOCKWISE:
                img = img.rotate(-270, expand=True)
            case Orientation.MIRROR:
                img = ImageOps.mirror(img)
            case Orientation.MIRROR_FLIP:
                img = ImageOps.flip(img)
                img = ImageOps.mirror(img)
            case Orientation.MIRROR_CLOCKWISE:
                img = img.rotate(-90, expand=True)
                img = ImageOps.mirror(img)
            case Orientation.MIRROR_COUNTERCLOCKWISE:
                img = img.rotate(-270, expand=True)
                img = ImageOps.mirror(img)
            case _:
                raise ValueError(f"invalid orientation: {self}")
        return img
