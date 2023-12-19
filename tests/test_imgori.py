import io

import pytest
import requests
from PIL import Image

from imgori import Imgori
from imgori.typing import Orientation
from imgori.typing import PILImage


@pytest.fixture
def img() -> PILImage:
    url = "https://www.ris.gov.tw/documents/data/apply-idCard/images/ddccc3f2-2aa9-4e92-9578-41d035af66ea.jpg"
    res = requests.get(url)
    img = Image.open(io.BytesIO(res.content)).convert("RGB")
    return img


def test_imgori(img: PILImage) -> None:
    m = Imgori()
    for ori in Orientation:
        res = m(ori.do(img))
        assert res == ori
