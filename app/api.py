import time
from typing import NamedTuple

from fastapi import FastAPI
from pydantic import BaseModel

from . import config
from .main import (
    in_progress,
    process_episode,
)
from .podcast import get_sha224_hash

logger = config.get_logger(__name__)
web_app = FastAPI()

# A transcription taking > 10 minutes should be exceedingly rare.
MAX_JOB_AGE_SECS = 10 * 60


class InProgressJob(NamedTuple):
    call_id: str
    start_time: int


class TranscribeReq(BaseModel):
    url: str


@web_app.post("/api/transcribe")
async def transcribe_job(req: TranscribeReq):
    now = int(time.time())
    url = req.url
    hash = get_sha224_hash(url)
    transcript_file = config.CONFIG.TRANSCRIPTIONS_DIR / f"{hash}.txt"
    if transcript_file.exists():
        logger.info(f"Found existing transcript for episode {url}")
        return {"text": transcript_file.read_text(), "url_hash": hash, "url": url, "exec_time": 0.0}

    try:
        inprogress_job = in_progress[url]
        # NB: runtime type check is to handle present of old `str` values that didn't expire.
        if (
            isinstance(inprogress_job, InProgressJob)
            and (now - inprogress_job.start_time) < MAX_JOB_AGE_SECS
        ):
            existing_call_id = inprogress_job.call_id
            logger.info(
                f"Found existing, unexpired call ID {existing_call_id} for episode {url}"
            )
            return {"call_id": existing_call_id, "status": "in_progress", "hash": hash}
    except KeyError:
        pass

    in_progress[url] = InProgressJob(
        call_id=hash, start_time=now
    )
    call = process_episode.spawn(url)
    transcript = call.get()
    return {"text": transcript, "url_hash": hash, "url": url, "exec_time": time.time() - now}