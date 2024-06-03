import os
from urllib.parse import urlparse

import boto3
from loguru import logger
from torch.hub import HASH_REGEX
from torch.hub import download_url_to_file
from torch.hub import get_dir


def download_url(
    url: str,
    filename: str | None = None,
    model_dir: str | None = None,
    progress: bool = True,
    check_hash: bool = False,
) -> str:
    """Download a file from a URL or AWS S3 bucket to a local directory.

    Args:
        url (str): HTTP, HTTPS, or S3 URL.
        filename (str, optional): Name of the file to save. If None, the filename is extracted from the URL.
        model_dir (str, optional): Directory to save the file. If None, the file is saved to the cache directory.
        progress (bool, optional): Show download progress. Defaults to True.
        check_hash (bool, optional): Check the hash of the downloaded file. Defaults to False.

    Returns:
        str: Path to the downloaded file.
    """
    if model_dir is None:
        hub_dir = get_dir()
        model_dir = os.path.join(hub_dir, "ditto")

    os.makedirs(model_dir, exist_ok=True)

    parts = urlparse(url)
    if filename is None:
        filename = os.path.basename(parts.path)

    cached_file = os.path.join(model_dir, filename)
    if os.path.exists(cached_file):
        return cached_file

    logger.info("Downloading {} to {}", url, cached_file)
    hash_prefix = None
    if check_hash:
        r = HASH_REGEX.search(str(filename))  # r is Optional[Match[str]]
        hash_prefix = r.group(1) if r else None

    schema = parts.scheme
    if schema == "s3":
        boto3.client("s3").download_file(parts.netloc, filename, cached_file)
    elif schema in ["http", "https"]:
        download_url_to_file(url, cached_file, hash_prefix, progress=progress)
    else:
        raise ValueError(f"Unsupported URL scheme: {url}")

    return cached_file
