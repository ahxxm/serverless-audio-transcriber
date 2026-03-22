# Serverless Transcriber

This repo uses [nano-parakeet](https://github.com/andimarafioti/nano-parakeet) to transcribe podcasts serverlessly and efficiently: 20 seconds -> 60 minutes audio, using `parakeet-tdt-0.6b-v3`. Some codes are from the official Modal example.

## Deploy

This can be hosted serverlessly on Modal, the caller might reside in AWS Lambda or wherever it fits.

### Modal

`modal deploy -m app.main` to deploy your app to Modal, get url, and call API for synchronous result.

```python
import requests
url = "https://your--pod-name.modal.run"
json = {
	"url": "url-to-mp3"
}
result = requests.post(f"{url}/api/transcribe", json=json).json()
transcript = result["text"]
```

## Development

To develop locally,

```shell
uv sync
uv run modal serve app.main
```

It reloads on code changes, press `Ctrl+C` to stop.

## Benchmarks and Recommendations

Unlike Runpod, Modal charges for model loading time, incentivizing cache and smaller images: fewer dependencies, distilled/turbo model.

[This episode](https://www.podtrac.com/pts/redirect.mp3/pdst.fm/e/chrt.fm/track/G481GD/traffic.megaphone.fm/ADV6859367463.mp3) is used for benchmarks, its length is `01:05:44`.

Language is hardcoded to `en`, skipping detection overhead.

### (old) Faster-whisper + large-v3-turbo: L4/L40S

| Graphic Card | Startup(s) | Execution(s) |
|--------------|------------|--------------|
| L4           | 2.8        | 54           |
| A10G         | 2.2        | 47           |
| L40S         | 2.1        | 27           |

Batch sizes are all 60, increasing to 80 on L40S didn't make it faster.

`Time(s) ~= 0.38 * minutes + 2.5`

### (now) nano-parakeet + TDT-0.6b bundled: L40S

The setup needs a few calls to cache both CPU and GPU state, so that execution logs skip loading(10s) and jump into transcribing.

| Graphic Card | Startup(s) | Execution(s) |
|--------------|------------|--------------|
| L40S         | 7.8        | 12.1         |

(Faster-whipser version should also benefit from correctly implemented caching, until whisper-v4.)
