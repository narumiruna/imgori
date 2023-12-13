import io
import os

import numpy as np
import pytest
import requests
from PIL import Image
from PIL import ImageChops

from imgori.datasets.orientation import Orientation
from imgori.typing import PILImage


@pytest.fixture
def fern() -> PILImage:
    url = "https://frieren-anime.jp/wp-content/uploads/2023/08/chara2_face4.jpg"
    res = requests.get(url)
    img = Image.open(io.BytesIO(res.content)).convert("RGB")
    return img


def test_orientation_do_undo(fern: PILImage) -> None:
    for orientation in Orientation:
        img = orientation.do(fern)
        img = orientation.undo(img)
        diff = ImageChops.difference(img, fern)
        assert pytest.approx(np.array(diff).sum()) == 0
