from pathlib import Path

from ..utils import download_url

model_path = Path(__file__).parent / "imgori_mobilenet_v3_small.pth"


DEFAULT_MODEL = model_path.as_posix()
if not model_path.exists():
    DEFAULT_MODEL = download_url(
        "https://github.com/narumiruna/imgori/releases/download/v0.2.7-mobilenet-v3/imgori_mobilenet_v3_small.pth"
    )
