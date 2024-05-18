import hashlib
import pathlib
import urllib.request
from typing import NamedTuple

from .config import get_logger
logger = get_logger(__name__)


def get_sha224_hash(url: str) -> str:
    m = hashlib.sha224()
    m.update(url.encode("utf-8"))
    return m.hexdigest()


class DownloadResult(NamedTuple):
    data: bytes
    # Helpful to store and transmit when uploading to cloud bucket.
    content_type: str


def download_podcast_file(url: str) -> DownloadResult:
    req = urllib.request.Request(
        url,
        data=None,
        # Set a user agent to avoid 403 response from some podcast audio servers.
        headers={
            "User-Agent": "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_9_3) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/35.0.1916.47 Safari/537.36"
        },
    )
    with urllib.request.urlopen(req) as response:
        return DownloadResult(
            data=response.read(),
            content_type=response.headers["content-type"],
        )


def sizeof_fmt(num, suffix="B") -> str:
    for unit in ["", "Ki", "Mi", "Gi", "Ti", "Pi", "Ei", "Zi"]:
        if abs(num) < 1024.0:
            return "%3.1f%s%s" % (num, unit, suffix)
        num /= 1024.0
    return "%.1f%s%s" % (num, "Yi", suffix)


def download_episode(
    url: str, destination: pathlib.Path, overwrite: bool = False
) -> None:
    logger.info(f"Downloading episode from {url} to {destination}.")
    if destination.exists():
        if overwrite:
            logger.info(
                f"Audio file exists at {destination} but overwrite option is specified."
            )
        else:
            logger.info(
                f"Audio file exists at {destination}, skipping download."
            )
            return

    podcast_download_result = download_podcast_file(url=url)
    humanized_bytes_str = sizeof_fmt(num=len(podcast_download_result.data))
    logger.info(f"Downloaded {humanized_bytes_str} episode from URL.")
    with open(destination, "wb") as f:
        f.write(podcast_download_result.data)
    logger.info(f"Stored audio episode at {destination}.")

