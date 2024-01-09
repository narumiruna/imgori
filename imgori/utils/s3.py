from pathlib import Path
from typing import Any
from urllib.parse import urlparse

import boto3
import torch
from loguru import logger
from torch.hub import get_dir
from torch.serialization import MAP_LOCATION


def load_state_dict_from_s3_url(
    url: str,
    model_dir: str | Path | None = None,
    map_location: MAP_LOCATION = None,
    filename: str | None = None,
    weights_only: bool = False,
) -> Any:
    if model_dir is None:
        hub_dir = Path(get_dir())
        model_dir = hub_dir / "checkpoints"

    model_dir = Path(model_dir)
    model_dir.mkdir(parents=True, exist_ok=True)

    if filename is None:
        parts = urlparse(url)
        filename = Path(parts.path).name

    cached_file = model_dir / filename
    if not cached_file.exists():
        logger.info('Downloading: "{}" to {}', url, cached_file)
        download_file_from_s3_url(url, cached_file)

    return torch.load(cached_file, map_location=map_location, weights_only=weights_only)


def download_file_from_s3_url(url: str, dst: str) -> None:
    Path(dst).parent.mkdir(parents=True, exist_ok=True)

    s3 = boto3.client("s3")
    parts = urlparse(url)
    s3.download_file(parts.netloc, Path(parts.path).name, dst)
