"""
Pod transcriber using nano_parakeet for speech-to-text.
"""

from modal import (
    App,
    Dict,
    Image,
    Volume,
    asgi_app,
)

from .config import get_logger, get_paths, CONFIG


logger = get_logger(__name__)

# can share between contains, but needs commit/refresh. work better with ~50000 files.
volume = Volume.from_name(
    "dataset-cache-vol", create_if_missing=True
)

# L40S OOM despite PYTORCH_ALLOC_CONF
app_gpu = "RTX-PRO-6000"
app_image = (
    Image
    .micromamba(python_version="3.13")
    .apt_install("ffmpeg")
    .env({"CONDA_OVERRIDE_CUDA": "12"})
    .micromamba_install("pytorch", channels=["conda-forge"])
    .pip_install("fastapi", "nano-parakeet", extra_options="--no-cache-dir")
    # micromamba 3.13 base image has stale CA certs, point ssl to certifi's bundle
    .env({
        "SSL_CERT_FILE": "/opt/conda/lib/python3.13/site-packages/certifi/cacert.pem",
        "HF_HOME": "/hf_cache",
        "PYTORCH_ALLOC_CONF": "expandable_segments:True",
    })
    .run_commands(
        "python -c 'from nano_parakeet import from_pretrained; from_pretrained()'",
        gpu=app_gpu,
    )
)
app = App(
    "whisperx-pod-transcriber",
    image=app_image,
)
with app_image.imports():
    from nano_parakeet import from_pretrained
    from .local_attention import enable_local_attention


in_progress = Dict.from_name(
    "transcribe-wip", create_if_missing=True
)


@app.function(
    image=app_image,
    volumes={CONFIG.CACHE_DIR: volume},
    timeout=600,
    gpu=app_gpu,
    cpu=8.0,
    memory=16384,
    scaledown_window=2,  # shutdown immediately
    enable_memory_snapshot=True,
    experimental_options={"enable_gpu_snapshot": True},
)
def process_episode(url: str) -> str:
    import os
    os.environ["PYTORCH_CUDA_ALLOC_CONF"] = \
        "expandable_segments:True,"\
        "roundup_power2_divisions:[32:256,64:128,256:64,>:32]"
    audio_dest_path, transcription_path, _ = get_paths(url)
    logger.info("Loading parakeet TDT model...")
    in_progress[url] = True
    model = from_pretrained()
    enable_local_attention(model)
    logger.info("Transcribing with nano_parakeet.")
    transcript = model.transcribe(str(audio_dest_path))
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
