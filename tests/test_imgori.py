import io

import pytest
import requests
from PIL import Image

from imgori import Imgori
from imgori.typing import Orientation
from imgori.typing import PILImage


@pytest.fixture
def img() -> PILImage:
    url = "https://upload.wikimedia.org/wikipedia/commons/f/f6/ROC_mibunsho.jpg"
    res = requests.get(url)
    img = Image.open(io.BytesIO(res.content)).convert("RGB")
    return img


def test_imgori(img: PILImage) -> None:
    m = Imgori()
    for ori in Orientation:
        res = m(ori.do(img))
        assert res == ori
