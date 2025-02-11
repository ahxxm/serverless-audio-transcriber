import time
from typing import NamedTuple

from fastapi import FastAPI
from pydantic import BaseModel

from . import config
from .main import (
    in_progress,
    process_episode,
    volume,
)
from .podcast import download_episode

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
    audio_dest_path, transcript_file, url_hash = config.get_paths(url)
    if transcript_file.exists():
        logger.info(f"Found existing transcript for episode {url}")
        return {"text": transcript_file.read_text(), "url_hash": url_hash, "url": url, "exec_time": 0.0}
    
    audio_dest_path.parent.mkdir(exist_ok=True, parents=True)
    transcript_file.parent.mkdir(exist_ok=True, parents=True)
    download_episode(
        url=url,
        destination=audio_dest_path,
    )
    volume.commit()

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
            return {"call_id": existing_call_id, "status": "in_progress", "hash": url_hash}
    except KeyError:
        pass

    in_progress[url] = InProgressJob(
        call_id=url_hash, start_time=now
    )
    call = process_episode.spawn(url)
    transcript = call.get()
    return {"text": transcript, "url_hash": url_hash, "url": url, "exec_time": time.time() - now}