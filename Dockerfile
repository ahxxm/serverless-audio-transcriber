FROM nvidia/cuda:11.8.0-cudnn8-runtime-ubuntu22.04

COPY --from=mwader/static-ffmpeg:7.0.1 /ffmpeg /usr/bin/

RUN apt update && \
  apt install --no-install-suggests --no-install-recommends -y python3 python3-pip git wget && \
  ln -s $(which python3) /usr/local/bin/python && \
  python -m pip install --no-cache-dir "numpy<2" torch==2.0 torchaudio==2.0.0 git+https://github.com/m-bain/whisperX.git && \
  apt clean && \
  rm -rf /var/lib/apt/lists/*

# download weights, turbo model is "deepdml/faster-whisper-large-v3-turbo-ct2"
RUN (touch sample.wav && whisperx sample.wav --model large-v3) || echo "ok"

# upgrade checkpoint but file reverted in runtime
# RUN sed -i '/hexdigest/,+4d' /usr/local/lib/python3.10/dist-packages/whisperx/vad.py && python -m pytorch_lightning.utilities.upgrade_checkpoint /root/.cache/torch/whisperx-vad-segmentation.bin
