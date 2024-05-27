"""
whisper-pod-transcriber uses OpenAI's Whisper model(WhisperX implementation) 
to do speech-to-text transcription of podcasts efficiently.
"""

from modal import (
    App,
    Dict,
    Image,
    Volume,
    asgi_app,
    gpu,
)

from .podcast import get_sha224_hash, download_episode
from .config import get_logger, CONFIG


logger = get_logger(__name__)

# can share between contains, but needs commit/refresh. work better with ~50000 files.
volume = Volume.from_name(
    "dataset-cache-vol", create_if_missing=True
)


app_gpu = gpu.A10G()
app_image = Image.from_registry("ahxxm/base:whisperx-modal")
app = App(
    "whisperx-pod-transcriber",
    image=app_image,
)


in_progress = Dict.from_name(
    "transcribe-wip", create_if_missing=True
)


@app.function(
    image=app_image,
    volumes={CONFIG.CACHE_DIR: volume},
    timeout=900,
    gpu=app_gpu,
    cpu=8.0,
    memory=8192,
    container_idle_timeout=2,  # shutdown immediately
)
def process_episode(url: str) -> str:
    import whisperx
    CONFIG.RAW_AUDIO_DIR.mkdir(parents=True, exist_ok=True)
    CONFIG.TRANSCRIPTIONS_DIR.mkdir(parents=True, exist_ok=True)

    audio_url_hash = get_sha224_hash(url)
    audio_dest_path = CONFIG.RAW_AUDIO_DIR / audio_url_hash
    transcription_path = CONFIG.TRANSCRIPTIONS_DIR / f"{audio_url_hash}.txt"
    if transcription_path.exists():
        logger.info(
            f"Transcription already exists for '{url}' with ID {audio_url_hash}."
        )
        return

    in_progress[url] = True
    model = whisperx.load_model(
        CONFIG.DEFAULT_MODEL, 
        "cuda", 
        language=CONFIG.DEFAULT_LANG,
        compute_type=CONFIG.COMPUTE_TYPE, 
        # download_root=CONFIG.MODEL_DIR,  # just use the weights from docker image
    )
    download_episode(
        url=url,
        destination=audio_dest_path,
    )
    volume.commit()

    logger.info(
        f"Using the {CONFIG.DEFAULT_MODEL} model."
    )
    audio = whisperx.load_audio(audio_dest_path)
    result = model.transcribe(audio, batch_size=CONFIG.BATCH_SIZE)
    transcript = ''.join([val["text"] for val in result['segments']])
    with transcription_path.open("w") as f:
        f.write(transcript)
    volume.commit()
    logger.info(f"Finished processing '{url}' with ID {audio_url_hash}.")

    try:
        del in_progress[url]
    except KeyError:
        pass

    return transcript


@app.function(
    volumes={CONFIG.CACHE_DIR: volume},
)
@asgi_app()
def fastapi_app():
    from .api import web_app
    return web_app
