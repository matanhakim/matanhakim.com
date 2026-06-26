---
title: "Recipe: set up local, offline speech-to-text"
---

**How to use this file:** hand it to Claude Code (or any coding agent) from inside a
working directory and say something like *"follow this recipe to set up local
transcription for me."* The agent should work through the steps in order, inspecting
the machine before it picks a model. Everything here runs locally: no cloud service,
no API key, no per-minute billing.

---

## 1. Detect the machine

Before recommending anything, find out what you are working with:

- **Operating system** and architecture.
- **CPU**: model and number of physical / logical cores.
- **RAM**: total, in GB.
- **GPU**: is there one, and is it an **NVIDIA** card? Only NVIDIA GPUs accelerate the
  transcription stack below (via CUDA). Apple Silicon, AMD, and Intel integrated
  graphics do **not** help here, so those machines run on the CPU.

Report the findings back before continuing.

## 2. Pick a model for the hardware and the language

The engine is [`faster-whisper`](https://github.com/SYSTRAN/faster-whisper), a fast
CTranslate2 reimplementation of OpenAI's Whisper. Choose the model from this guide:

| Machine | Model | `compute_type` |
|---|---|---|
| 8 GB RAM, no NVIDIA GPU | `small` | `int8` |
| 16 GB RAM, no NVIDIA GPU | `large-v3-turbo` | `int8` |
| 32 GB RAM, no NVIDIA GPU | `large-v3-turbo` or `large-v3` | `int8` |
| NVIDIA GPU, 6 GB+ VRAM | `large-v3` | `float16` |

**Language matters more than people expect.** The `tiny` and `base` models are weak
outside English; for any non-English language use `small` at the very least, and
prefer `large-v3-turbo`. Before defaulting to a generic model, check whether a
**fine-tune for the target language** exists on Hugging Face, because a dedicated
fine-tune usually beats the generic model of the same size. For Hebrew, for example,
use [`ivrit-ai/whisper-large-v3-turbo-ct2`](https://huggingface.co/ivrit-ai/whisper-large-v3-turbo-ct2).

## 3. Install

```bash
# Python 3.9+ is assumed.
pip install faster-whisper

# ffmpeg is needed to decode audio. Install it with the system package manager:
#   macOS:    brew install ffmpeg
#   Debian:   sudo apt install ffmpeg
#   Windows:  winget install Gyan.FFmpeg   (or: choco install ffmpeg)
```

The model itself downloads automatically from Hugging Face the first time it runs and
is cached on disk after that, so the first run needs an internet connection but every
run afterwards is fully offline.

## 4. A minimal transcribe script

```python
from faster_whisper import WhisperModel

# Fill these in from steps 1-2.
MODEL = "ivrit-ai/whisper-large-v3-turbo-ct2"  # or "large-v3-turbo", "small", ...
COMPUTE = "int8"        # "int8" on CPU, "float16" on an NVIDIA GPU
DEVICE = "cpu"          # "cuda" only if an NVIDIA GPU is present
LANGUAGE = "he"         # ISO code of the spoken language, or None to auto-detect

model = WhisperModel(MODEL, device=DEVICE, compute_type=COMPUTE)

segments, info = model.transcribe(
    "recording.m4a",
    language=LANGUAGE,
    beam_size=8,          # raise for noisy phone audio
    vad_filter=True,      # skip long silences; speeds up real recordings
)

for s in segments:
    print(f"[{s.start:6.1f}s] {s.text.strip()}")
```

## 5. What to include in the setup

- **`vad_filter=True`** drops long silences before decoding. On real meetings and
  voice memos this is most of the speed win.
- **`compute_type`**: `int8` is the fastest option on a CPU. `float16` is for NVIDIA
  GPUs; if you ask for it on a CPU the library quietly falls back to `float32`, which
  is slower, so set `int8` explicitly on CPU machines.
- **Roughly what to expect on a CPU**: transcription runs at about real time, give or
  take. A 10-minute recording takes on the order of 10 minutes; an NVIDIA GPU is
  several times faster.
- Wrap the script so it takes the audio path as an argument and writes a timestamped
  `.md` or `.srt` file, and you have a reusable local transcription tool.
