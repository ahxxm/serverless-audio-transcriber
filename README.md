# Serverless Transcriber

This repo uses [WhisperX](https://github.com/m-bain/whisperX) to transcribe podcasts serverlessly and efficiently: 1 minute -> 60 minutes audio, using `large-v3`. Some codes are from the official Modal example.

The shared base image `ahxxm/base:whisperx-modal` is built from `Dockerfile`.

## Deploy

This can be hosted serverlessly on Modal or Runpod, the caller might reside in AWS Lambda or wherever it fits.

### Modal

`modal deploy app.main` to deploy your app to Modal, get url, and call API for synchronous result.

```python
import requests
url = "https://your--pod-name.modal.run"
json = {
	"url": "url-to-mp3"
}
result = requests.post(f"{url}/api/transcribe", json=json).json()
transcript = result["text"]
```

### Runpod

Simply start a serverless endpoint with container image `ahxxm/base:whisperx-cuda122`, then get API key and call.

```python
API_ENDPOINT = "https://api.runpod.ai/v2/****/runsync"
RUNPOD_KEY = ""
auth_header = {
    "Authorization": f"Bearer {RUNPOD_KEY}"
}
payload = {
    "input": {
        "audio": mp3_link,
        "language": "en",
        "batch_size": 40,
    }
}
rsp = requests.post(API_ENDPOINT, headers=auth_header, json=payload).json()
text = rsp["output"]["transcription"]
costMs = rsp["executionTime"]
delayMs = rsp["delayTime"]
```

## Development

To develop locally,

```shell
pip install modal
modal serve app.main
```

It reloads on code changes, press `Ctrl+C` to stop.

## Benchmarks and Caveats

Modal charges startup loading time(around 25s) as well, A10G is recommended.

I used [this episode](https://www.podtrac.com/pts/redirect.mp3/pdst.fm/e/chrt.fm/track/G481GD/traffic.megaphone.fm/ADV6859367463.mp3) for benchmarks, its length is `01:05:44`.

| Provider | CPU | Memory | Graphic Card | Batch Size | Charged GPU Seconds |
|----------|-----|--------|--------------|------------|---------------------|
| Runpod   | ?   | ?      | RTX A4500    | 40         | 74.43s              |
| Modal    | 8.0 | 8192MB | T4           | 16         | 179s                |
| Modal    | 8.0 | 8192MB | L4           | 40         | 116s                |
| Modal    | 8.0 | 8192MB | A10G         | 40         | 86s                 |