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

from .config import get_logger, get_paths, CONFIG


logger = get_logger(__name__)

# can share between contains, but needs commit/refresh. work better with ~50000 files.
volume = Volume.from_name(
    "dataset-cache-vol", create_if_missing=True
)

docker_images = {
    "large-v3": "ahxxm/base:whisperx-modal",
    "deepdml/faster-whisper-large-v3-turbo-ct2": "ahxxm/base:whisperx-turbo-modal"
}


app_gpu = gpu.A10G()
app_image = Image.from_registry(docker_images[CONFIG.DEFAULT_MODEL])
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
    audio_dest_path, transcription_path, _ = get_paths(url)
    if transcription_path.exists():
        logger.info(
            f"Transcription already exists for '{url}'"
        )
        return

    logger.info(f"Loading model {CONFIG.DEFAULT_MODEL}...")
    in_progress[url] = True
    model = whisperx.load_model(
        CONFIG.DEFAULT_MODEL, 
        "cuda", 
        language=CONFIG.DEFAULT_LANG,
        compute_type=CONFIG.COMPUTE_TYPE, 
        # download_root=CONFIG.MODEL_DIR,  # just use the weights from docker image
    )

    logger.info(
        f"Using the {CONFIG.DEFAULT_MODEL} model."
    )
    audio = whisperx.load_audio(audio_dest_path)
    result = model.transcribe(audio, batch_size=CONFIG.BATCH_SIZE)
    transcript = ''.join([val["text"] for val in result['segments']])
    with transcription_path.open("w") as f:
        f.write(transcript)
    volume.commit()
    logger.info(f"Finished processing '{url}'")

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
