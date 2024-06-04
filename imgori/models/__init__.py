from pathlib import Path

from ..utils import download_url

DEFAULT_MODEL = Path(__file__).parent / "imgori_mobilenet_v3_small.pth"


if not DEFAULT_MODEL.exists():
    DEFAULT_MODEL = download_url(
        "https://github.com/narumiruna/imgori/releases/download/v0.2.7-mobilenet-v3/imgori_mobilenet_v3_small.pth"
    )
