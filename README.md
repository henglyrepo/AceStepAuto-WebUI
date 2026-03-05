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

## Lyric Video (MP4)

You can render a standard horizontal (16:9, 1920x1080) lyric video from a generated run's `audio.*` + `meta.json`.

Features:
- Section-aware timing pauses (e.g. before Chorus/Bridge)
- Karaoke-like word highlighting (best-effort heuristic)
- Background motion + BPM pulse (visual reacts to tempo metadata when available)

Optional sync feature (recommended):
- Auto Sync derives real per-line timestamps from the singing audio using open-source ASR.
- This significantly improves on-screen lyric timing vs the default heuristic timing.

Requirements:
- FFmpeg

This repo supports a local, no-PATH setup:
- Put `ffmpeg.exe` at `tools/ffmpeg.exe`
- (Optional) also add `tools/ffprobe.exe`

Quick install (PowerShell):

```powershell
powershell -ExecutionPolicy Bypass -File tools\get_ffmpeg.ps1
```

Quick install (Python, cross-platform):

```bat
python tools\get_ffmpeg.py
```

Then:

```bat
python video_lyrics.py --run-dir "output\<timestamp>_<title>" --timing smart --preset hd16x9
```

### Cinematic Background Loop (ComfyUI)

If ComfyUI is running (default `http://127.0.0.1:8188`), you can generate a looping background clip per run:

```bat
ACE-Step-1.5\python_embeded\python.exe generate_bg_loop.py --run-dir "output\<timestamp>_<title>" --comfy-url http://127.0.0.1:8188 --ckpt sd_xl_turbo_1.0.safetensors --loop-seconds 8
```

This writes:
- `output/<run>/bg_keyframe_base.png` (and verse/chorus/bridge variants)
- `output/<run>/bg_loop_base.mp4` (and verse/chorus/bridge variants)
- `output/<run>/bg_loop.mp4` (compat copy of base)
- `output/<run>/visual_prompt.json`

Then render the lyric video; `video_lyrics.py` automatically uses `run_dir/bg_loop.mp4` when present:

```bat
ACE-Step-1.5\python_embeded\python.exe video_lyrics.py --run-dir "output\<timestamp>_<title>" --preset hd16x9
```

Advanced:
- Section backgrounds (crossfades when [Verse]/[Chorus]/[Bridge] tags are available):

```bat
ACE-Step-1.5\python_embeded\python.exe video_lyrics.py --run-dir "output\<run>" --bg-mode sections --bg-crossfade 0.6
```

- Audio visualization overlay:

```bat
ACE-Step-1.5\python_embeded\python.exe video_lyrics.py --run-dir "output\<run>" --viz spectrum --viz-opacity 0.18
```

- Background grading (more cinematic / less sharp):

```bat
ACE-Step-1.5\python_embeded\python.exe video_lyrics.py --run-dir "output\<run>" --bg-blur 2.8 --bg-grain 10 --bg-vignette 0.6 --bg-brightness -0.04 --bg-saturation 1.08 --bg-contrast 1.06 --plate-alpha 0.30
```

WebUI tip:
- Use the `BG Preset` dropdown (e.g. `Cinematic Soft`, `Warm Film`, `Dark Club`) to set these values quickly.

Auto recommendation:
- After `Generate BG Loop`, click `Apply Recommended Preset` to apply the LLM's suggested background grading preset.

### Auto Sync Lyrics (English-first)

Auto Sync uses:
- `faster-whisper` (Whisper ASR with word timestamps)
- optional `demucs` vocal isolation (improves alignment accuracy)

Install dependencies (using ACE-Step embedded Python):

```bat
ACE-Step-1.5\python_embeded\python.exe -m pip install -U faster-whisper demucs
```

Notes:
- First run downloads Whisper model weights.
- Demucs is optional but recommended (it isolates vocals and improves timing).

Run Auto Sync for a generated run folder:

```bat
ACE-Step-1.5\python_embeded\python.exe align_lyrics.py --run-dir "output\<timestamp>_<title>" --model small --demucs
```

This writes:
- `output/<run>/alignment_words.json`
- `output/<run>/alignment_lines.json`
- `output/<run>/alignment_report.json`

Then render the video normally; `video_lyrics.py` auto-detects and uses `alignment_lines.json`.

Presets:
- `hd16x9` -> 1920x1080
- `uhd4k16x9` -> 3840x2160
- `vertical1080` -> 1080x1920

Or use the WebUI button:
- `Make Lyric Video` (in the Results column)

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
