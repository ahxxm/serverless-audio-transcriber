import logging
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
    DEFAULT_MODEL = "large-v3"
    DEFAULT_LANG = "en"
    COMPUTE_TYPE = "float16"
    BATCH_SIZE = 40  # A4500 can survive 40, yet T4 only 16?