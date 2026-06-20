import pathlib

import requests
from requests.adapters import HTTPAdapter, Retry

from .config import get_logger
logger = get_logger(__name__)

# Set a user agent to avoid 403 response from some podcast audio servers.
_HEADERS = {
    "User-Agent": "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_9_3) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/35.0.1916.47 Safari/537.36"
}

# urllib3 enforces Content-Length, so a truncated body raises instead of caching corrupt
_session = requests.Session()
_adapter = HTTPAdapter(max_retries=Retry(
    total=4, backoff_factor=1, status_forcelist=(429, 500, 502, 503, 504)
))
_session.mount("http://", _adapter)
_session.mount("https://", _adapter)


def sizeof_fmt(num, suffix="B") -> str:
    for unit in ["", "Ki", "Mi", "Gi", "Ti", "Pi", "Ei", "Zi"]:
        if abs(num) < 1024.0:
            return "%3.1f%s%s" % (num, unit, suffix)
        num /= 1024.0
    return "%.1f%s%s" % (num, "Yi", suffix)


def download_episode(url: str, destination: pathlib.Path) -> None:
    if destination.exists():
        logger.info(f"Audio already cached at {destination}, skipping download.")
        return

    response = _session.get(url, headers=_HEADERS, timeout=60)
    response.raise_for_status()
    # rename is atomic, so the cache never holds a partial file
    tmp = destination.with_name(destination.name + ".partial")
    tmp.write_bytes(response.content)
    tmp.rename(destination)
    logger.info(f"Downloaded {sizeof_fmt(len(response.content))} from {url} to {destination}.")
