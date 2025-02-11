# This file: for whisperx implementation on Modal
# cudnn<9 : https://github.com/m-bain/whisperX/issues/954
# in the latest whisperx release, torch comes with cu12 and Python needs to be >=3.9(22.04), leading to this base image.
FROM pytorch/pytorch:2.3.1-cuda12.1-cudnn8-runtime

COPY --from=mwader/static-ffmpeg:7.1 /ffmpeg /usr/bin/

RUN pip install --no-cache-di -U pip && \
  pip install --no-cache-dir fastapi whisperx==3.3.1

# download weights, turbo model is "deepdml/faster-whisper-large-v3-turbo-ct2"
RUN (touch sample.wav && whisperx sample.wav --model large-v3) || echo "ok"

# upgrade checkpoint but file reverted in runtime
# RUN sed -i '/hexdigest/,+4d' /usr/local/lib/python3.10/dist-packages/whisperx/vad.py && python -m pytorch_lightning.utilities.upgrade_checkpoint /root/.cache/torch/whisperx-vad-segmentation.bin
