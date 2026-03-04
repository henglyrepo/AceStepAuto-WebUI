# AceStepAuto Agent Guide

This repo is a small orchestrator that:

- generates a song package (caption + structured lyrics) using a local OpenAI-compatible LLM server
- calls the ACE-Step 1.5 REST API to generate audio
- downloads the resulting audio and writes a sidecar JSON with metadata

The ACE-Step engine is vendored as a git subfolder so it can run locally.

## Directory Layout

- `ACE-Step-1.5/` - upstream ACE-Step repo (runs the music generation API)
- `main.py` - orchestrator CLI (LLM -> ACE-Step -> audio download)
- `output/` - generated songs and metadata (created at runtime)

## Prerequisites

- Windows 10/11
- NVIDIA GPU recommended (example target: RTX 3060 Ti 8GB)
- Python 3.11 or 3.12 installed
- `uv` installed (https://astral.sh/uv)

## Setup ACE-Step (one-time)

From `C:\Users\HENGLY\Documents\Automation\AceStepAuto\ACE-Step-1.5`:

1) Install dependencies:

```bat
uv sync
```

2) Start the ACE-Step REST API server:

```bat
uv run acestep-api --host 127.0.0.1 --port 8001
```

Or use the upstream launcher:

```bat
start_api_server.bat
```

Verify:

- `http://127.0.0.1:8001/health`
- `http://127.0.0.1:8001/docs`

Notes:

- First run downloads models (large) and may take a while.
- If you see `uv` cache lock timeouts, ensure no other `uv` process is running.

## Run The Orchestrator

The orchestrator assumes:

- Local LLM server: `http://localhost:8317/v1`
- ACE-Step API: `http://127.0.0.1:8001`

Example:

```bat
python main.py --topic "first love" --style "k-pop dance pop" --lang en --duration 45 --audio-format mp3
```

Outputs:

- `output/<timestamp>_<title>.mp3` (or wav/flac)
- `output/<timestamp>_<title>.json` (caption, lyrics, ACE-Step params, returned metadata)

## Configuration

`main.py` accepts CLI flags and also reads these env vars:

- `LLM_BASE_URL` (default `http://localhost:8317/v1`)
- `LLM_MODEL` (default `gpt-5.2`)
- `ACESTEP_API_BASE` (default `http://127.0.0.1:8001`)

## Troubleshooting

- ACE-Step not reachable: start the API server and confirm `/health`.
- OOM on 8GB VRAM: use `--duration 30` and keep `--batch-size 1` (default).
- Slow first run: model downloads + CUDA compilation caches can take time.
