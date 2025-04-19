"""
whisper-pod-transcriber uses OpenAI's Whisper model(faster-whisper implementation) 
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

app_gpu = "L40S"
app_image = (
    Image
    .micromamba(python_version="3.12")
    .pip_install("fastapi", "faster-whisper==1.1.1", extra_options="--no-cache-dir")
    # runtime: cudnn9, decode without ffmpeg
    .micromamba_install("cudnn>=9", channels=["anaconda"], gpu=app_gpu)
    # prevent converting to fp32 on CPU
    .run_commands(
        """python -c 'import faster_whisper; faster_whisper.WhisperModel("large-v3-turbo", compute_type="bfloat16")'""", 
        gpu=app_gpu
    )
)
app = App(
    "whisperx-pod-transcriber",
    image=app_image,
)
with app_image.imports():
    import faster_whisper


in_progress = Dict.from_name(
    "transcribe-wip", create_if_missing=True
)


@app.function(
    image=app_image,
    volumes={CONFIG.CACHE_DIR: volume},
    timeout=600,
    gpu=app_gpu,
    cpu=8.0,
    memory=32768,
    scaledown_window=2,  # shutdown immediately
)
def process_episode(url: str) -> str:
    audio_dest_path, transcription_path, _ = get_paths(url)
    logger.info(f"Loading model {CONFIG.DEFAULT_MODEL}...")
    in_progress[url] = True
    model = faster_whisper.WhisperModel(
        CONFIG.DEFAULT_MODEL,
        device="cuda",
        compute_type="bfloat16",
    )
    batched = faster_whisper.BatchedInferencePipeline(model=model)

    logger.info(
        f"Using the {CONFIG.DEFAULT_MODEL} model."
    )
    segments, _ = batched.transcribe(audio_dest_path, batch_size=CONFIG.BATCH_SIZE, language=CONFIG.DEFAULT_LANG)
    transcript = ''.join([segment.text for segment in segments])
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
