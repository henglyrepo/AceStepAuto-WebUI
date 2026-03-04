# AceStepAuto

Generate a song package (caption + structured lyrics) using a local OpenAI-compatible LLM server, then render audio with ACE-Step 1.5 via its REST API.

This repo includes:
- `main.py`: CLI orchestrator (LLM -> ACE-Step -> download audio + sidecar JSON)
- `webui.py`: Gradio WebUI wrapper (start/stop + logs + output history)

Local dependency:
- `ACE-Step-1.5/`: the ACE-Step engine folder must exist at the repo root (same level as `main.py` and `webui.py`).
  - This folder is intentionally gitignored (large). You provide it locally.

## Quick Start (Windows)

### 1) Start ACE-Step API server

Option A (recommended):

`ACE-Step-1.5\start_api_server.bat`

Verify:
- `http://127.0.0.1:8001/health`
- `http://127.0.0.1:8001/docs`

### 2) Configure your local LLM server

Defaults:
- Base URL: `http://localhost:8317/v1`
- Model: `gpt-5.2`

If your LLM server requires an API key (common), create a root `.env` file:

```text
LLM_BASE_URL=http://localhost:8317/v1
LLM_MODEL=gpt-5.2
LLM_API_KEY=replace_me

ACESTEP_API_BASE=http://127.0.0.1:8001
```

Notes:
- `.env` is ignored by git via `.gitignore`.
- `main.py` auto-loads `.env` at startup.
- The WebUI auto-loads `.env` too. You can also paste a key in the WebUI and click `Save Key to .env`.

### 3) Run the WebUI (recommended)

Use ACE-Step's embedded Python (bundles Gradio already):

`start_webui.bat`

Or:

`ACE-Step-1.5\python_embeded\python.exe -u webui.py`

Open:
- The URL printed in the console (preferred port 7865; auto-increments if busy)

WebUI features:
- Start/Stop song generation (runs `main.py`)
- Start/Stop ACE-Step server (best-effort process control)
- Live logs
- Recent output history picker
- Genre selection (mix up to 5)
- Idea tools: generate topic/style using the LLM
- Batch generation (sequential)

Batch re-suggest options:
- None: run the same topic/style for every batch item
- Topic from Style: each run after the first generates a new topic from the current style
- Style from Topic: each run after the first generates a new style from the current topic
- Both: each run after the first generates both topic and style

Environment variables:
- `WEBUI_PORT` (preferred `7865`; auto-picks next free port if busy)
- `WEBUI_HOST` (default `127.0.0.1`)
- `WEBUI_OPEN_BROWSER` (set `1` to auto-open browser on launch)

## Run the CLI

```bat
python main.py --topic "first love" --style "k-pop dance pop" --lang en --duration 30 --audio-format mp3 --batch-size 1
```

If your LLM server needs auth and you didn't set `.env`, pass the key:

```bat
python main.py ... --llm-api-key "your_key_here"
```

## Output Layout

Each run writes into a single folder:

- `output/<timestamp>_<title>/audio.<mp3|wav|flac>`
- `output/<timestamp>_<title>/meta.json`

## Troubleshooting

- ACE-Step unreachable
  - Start `ACE-Step-1.5\start_api_server.bat` and verify `/health`.

- LLM 401 Unauthorized / Missing API key
  - Set `LLM_API_KEY` in `.env` or pass `--llm-api-key`.

- Stop button behavior
  - WebUI Stop terminates the local `main.py` process (stops waiting/downloading).
  - ACE-Step server-side generation may continue unless ACE-Step adds a cancel endpoint.
