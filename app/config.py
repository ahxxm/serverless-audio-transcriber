import logging
import hashlib
import pathlib


def get_logger(name, level=logging.INFO):
    logger = logging.getLogger(name)
    handler = logging.StreamHandler()
    handler.setFormatter(
        logging.Formatter("%(levelname)s: %(asctime)s: %(name)s  %(message)s")
    )
    logger.addHandler(handler)
    logger.setLevel(level)
    return logger


class CONFIG:
    CACHE_DIR = "/cache"
    # Where downloaded podcasts are stored, by guid hash.
    # Mostly .mp3 files 50-100MiB.
    RAW_AUDIO_DIR = pathlib.Path(CACHE_DIR, "raw_audio")
    # Completed episode transcriptions. Stored as flat files with
    # files structured as '{url_guid_hash}-{model_slug}.json'.
    TRANSCRIPTIONS_DIR = pathlib.Path(CACHE_DIR, "transcript")
    # Location of modal checkpoint.
    MODEL_DIR = pathlib.Path(CACHE_DIR, "model")
    # Transcribe options
    DEFAULT_MODEL = "large-v3-turbo"
    DEFAULT_LANG = "en"
    # v3-turbo: 60 for L4/A10G, 80 for L40S didn't make it faster.
    # v3: 40 for L4/A10G
    BATCH_SIZE = 60


def get_sha224_hash(url: str) -> str:
    m = hashlib.sha224()
    m.update(url.encode("utf-8"))
    return m.hexdigest()

def get_paths(url: str):
    url_hash = get_sha224_hash(url)
    audio = CONFIG.RAW_AUDIO_DIR / url_hash
    transcript = CONFIG.TRANSCRIPTIONS_DIR / f"{url_hash}.txt"
    return audio, transcript, url_hash