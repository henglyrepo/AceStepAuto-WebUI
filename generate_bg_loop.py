import argparse
import json
import os
import re
import subprocess
import sys
import tempfile
import time
import urllib.parse
import urllib.request
import uuid


def _read_json(path: str) -> dict:
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def _write_json(path: str, obj: object) -> None:
    with open(path, "w", encoding="utf-8") as f:
        json.dump(obj, f, ensure_ascii=True, indent=2)


def _http_get_json(url: str, timeout: float = 10.0) -> dict:
    req = urllib.request.Request(url, method="GET")
    with urllib.request.urlopen(req, timeout=timeout) as resp:
        raw = resp.read().decode("utf-8", errors="replace")
    return json.loads(raw)


def _http_post_json(url: str, payload: dict, timeout: float = 30.0) -> dict:
    body = json.dumps(payload).encode("utf-8")
    req = urllib.request.Request(url, data=body, method="POST")
    req.add_header("Content-Type", "application/json")
    with urllib.request.urlopen(req, timeout=timeout) as resp:
        raw = resp.read().decode("utf-8", errors="replace")
    return json.loads(raw)


def _find_ffmpeg() -> str:
    local = os.path.join(os.path.dirname(__file__), "tools", "ffmpeg.exe")
    if os.path.isfile(local):
        return local
    return "ffmpeg"


def _slugify(s: str, max_len: int = 60) -> str:
    s = (s or "").strip().lower()
    s = re.sub(r"[^a-z0-9]+", "_", s)
    s = re.sub(r"_+", "_", s).strip("_")
    return (s or "bg")[:max_len].strip("_")


def _extract_json_object(text: str) -> dict:
    text = (text or "").strip()
    try:
        obj = json.loads(text)
        if isinstance(obj, dict):
            return obj
    except Exception:
        pass
    dec = json.JSONDecoder()
    i = text.find("{")
    while i != -1:
        try:
            obj, _end = dec.raw_decode(text[i:])
            if isinstance(obj, dict):
                return obj
        except Exception:
            pass
        i = text.find("{", i + 1)
    raise ValueError("LLM did not return a JSON object")


def llm_make_visual_prompt(
    *,
    llm_base: str,
    llm_model: str,
    llm_api_key: str,
    meta: dict,
    loop_seconds: float,
) -> dict:
    """Generate a cinematic visual prompt pack from song meta."""

    base = (llm_base or "http://localhost:8317/v1").rstrip("/")
    url = base + "/chat/completions"

    title = str(meta.get("title") or "").strip()
    style = str(meta.get("style") or "").strip()
    caption = str(meta.get("caption") or "").strip()
    lyrics = str(meta.get("lyrics") or "").strip()
    bpm = None
    try:
        bpm = float(
            ((meta.get("acestep") or {}).get("format_input") or {}).get("bpm") or 0
        )
    except Exception:
        bpm = 0

    prompt = (
        "Return ONLY JSON. ASCII only. No markdown.\n"
        "Goal: create a CINEMATIC 16:9 lyric video background prompt for AI image generation.\n"
        "Constraints:\n"
        "- No readable text, no logos, no subtitles, no watermarks\n"
        "- Prefer environments and cinematic B-roll, no faces (or faces far away)\n"
        "- It should feel like a music video shot: lens, lighting, mood, atmosphere\n"
        "- Output keys: scene, palette, positive_prompt, negative_prompt, camera, motion_notes, recommended_bg_preset\n"
        "- recommended_bg_preset must be ONE of: Cinematic Soft, Warm Film, Dark Club, Crisp\n"
        "Song info:\n"
        f"Title: {title}\n"
        f"Style: {style}\n"
        f"BPM: {bpm}\n"
        f"Caption: {caption}\n"
        "Lyrics (may include sections):\n" + lyrics[:3500] + "\n"
        f"Loop seconds: {float(loop_seconds):.2f}\n"
    )

    headers = {}
    key = (
        (llm_api_key or "").strip()
        or os.environ.get("LLM_API_KEY")
        or os.environ.get("OPENAI_API_KEY")
    )
    if key:
        headers["Authorization"] = f"Bearer {key}"

    payload = {
        "model": llm_model or "gpt-5.2",
        "messages": [
            {
                "role": "system",
                "content": "You create cinematic visual prompts for AI image generation. Return ONLY JSON.",
            },
            {"role": "user", "content": prompt},
        ],
        "temperature": 0.85,
    }

    body = json.dumps(payload).encode("utf-8")
    req = urllib.request.Request(url, data=body, method="POST")
    req.add_header("Content-Type", "application/json")
    for k, v in headers.items():
        if v:
            req.add_header(k, v)
    with urllib.request.urlopen(req, timeout=60.0) as resp:
        raw = resp.read().decode("utf-8", errors="replace")
    resp_obj = json.loads(raw)
    content = resp_obj["choices"][0]["message"]["content"]
    out = _extract_json_object(content)
    # Ensure required keys exist.
    for k in ("positive_prompt", "negative_prompt", "scene"):
        if not str(out.get(k) or "").strip():
            raise ValueError(f"LLM output missing key: {k}")
    if not str(out.get("recommended_bg_preset") or "").strip():
        out["recommended_bg_preset"] = "Cinematic Soft"
    return out


def llm_make_visual_prompt_sections(
    *,
    llm_base: str,
    llm_model: str,
    llm_api_key: str,
    meta: dict,
    loop_seconds: float,
) -> dict:
    """Generate prompts for multiple cinematic scenes (verse/chorus/bridge).

    Returns JSON with keys:
    - base: {scene,palette,positive_prompt,negative_prompt,camera,motion_notes}
    - verse: {scene,positive_prompt}
    - chorus: {scene,positive_prompt}
    - bridge: {scene,positive_prompt}
    - negative_prompt: string (global)
    """

    base = (llm_base or "http://localhost:8317/v1").rstrip("/")
    url = base + "/chat/completions"

    title = str(meta.get("title") or "").strip()
    style = str(meta.get("style") or "").strip()
    caption = str(meta.get("caption") or "").strip()
    lyrics = str(meta.get("lyrics") or "").strip()

    prompt = (
        "Return ONLY JSON. ASCII only. No markdown.\n"
        "You generate CINEMATIC lyric video background prompts for 16:9 video loops.\n"
        "Constraints:\n"
        "- No readable text, no logos, no subtitles, no watermarks\n"
        "- Prefer environments / cinematic B-roll; avoid close-up faces\n"
        "- Keep the same art direction across all scenes (palette, lens, mood)\n"
        "- Provide 3 scene variants: verse, chorus, bridge (each distinct location/energy)\n"
        "Output JSON schema:\n"
        "{\n"
        '  "palette": ["..."],\n'
        '  "camera": "...",\n'
        '  "motion_notes": "...",\n'
        '  "negative_prompt": "...",\n'
        '  "recommended_bg_preset": "Cinematic Soft|Warm Film|Dark Club|Crisp",\n'
        '  "base": {"scene":"...", "positive_prompt":"..."},\n'
        '  "verse": {"scene":"...", "positive_prompt":"..."},\n'
        '  "chorus": {"scene":"...", "positive_prompt":"..."},\n'
        '  "bridge": {"scene":"...", "positive_prompt":"..."}\n'
        "}\n"
        "Song info:\n"
        f"Title: {title}\n"
        f"Style: {style}\n"
        f"Caption: {caption}\n"
        "Lyrics:\n" + lyrics[:3200] + "\n"
        f"Loop seconds: {float(loop_seconds):.2f}\n"
        "Prompt writing rules:\n"
        "- Start positive prompts with the scene subject/location\n"
        "- Include: lighting, weather/atmosphere, lens, film grain, depth of field\n"
        "- Avoid: words like 'text', 'subtitles', 'typography' in positive\n"
    )

    headers = {}
    key = (
        (llm_api_key or "").strip()
        or os.environ.get("LLM_API_KEY")
        or os.environ.get("OPENAI_API_KEY")
    )
    if key:
        headers["Authorization"] = f"Bearer {key}"

    payload = {
        "model": llm_model or "gpt-5.2",
        "messages": [
            {
                "role": "system",
                "content": "You create cinematic prompts. Return ONLY JSON.",
            },
            {"role": "user", "content": prompt},
        ],
        "temperature": 0.85,
    }

    body = json.dumps(payload).encode("utf-8")
    req = urllib.request.Request(url, data=body, method="POST")
    req.add_header("Content-Type", "application/json")
    for k, v in headers.items():
        if v:
            req.add_header(k, v)
    with urllib.request.urlopen(req, timeout=75.0) as resp:
        raw = resp.read().decode("utf-8", errors="replace")
    resp_obj = json.loads(raw)
    content = resp_obj["choices"][0]["message"]["content"]
    out = _extract_json_object(content)

    neg = str(out.get("negative_prompt") or "").strip()
    if not neg:
        raise ValueError("LLM output missing negative_prompt")
    if not str(out.get("recommended_bg_preset") or "").strip():
        out["recommended_bg_preset"] = "Cinematic Soft"
    for k in ("base", "verse", "chorus", "bridge"):
        obj = out.get(k) or {}
        if not isinstance(obj, dict):
            raise ValueError(f"LLM output key {k} is not an object")
        if not str(obj.get("positive_prompt") or "").strip():
            raise ValueError(f"LLM output missing {k}.positive_prompt")
        if not str(obj.get("scene") or "").strip():
            out[k]["scene"] = k
    return out


def comfy_render_keyframe(
    *,
    comfy_url: str,
    ckpt_name: str,
    positive: str,
    negative: str,
    width: int,
    height: int,
    steps: int,
    cfg: float,
    seed: int,
    sampler: str,
    scheduler: str,
    filename_prefix: str,
) -> tuple[str, dict]:
    """Render a single keyframe via ComfyUI and return local downloaded file path."""

    base = (comfy_url or "http://127.0.0.1:8188").rstrip("/")
    client_id = str(uuid.uuid4())

    # Standard text2img workflow using built-in nodes.
    workflow = {
        "1": {
            "class_type": "CheckpointLoaderSimple",
            "inputs": {"ckpt_name": ckpt_name},
        },
        "2": {
            "class_type": "CLIPTextEncode",
            "inputs": {"text": positive, "clip": ["1", 1]},
        },
        "3": {
            "class_type": "CLIPTextEncode",
            "inputs": {"text": negative, "clip": ["1", 1]},
        },
        "4": {
            "class_type": "EmptyLatentImage",
            "inputs": {"width": int(width), "height": int(height), "batch_size": 1},
        },
        "5": {
            "class_type": "KSampler",
            "inputs": {
                "model": ["1", 0],
                "seed": int(seed),
                "steps": int(steps),
                "cfg": float(cfg),
                "sampler_name": str(sampler),
                "scheduler": str(scheduler),
                "positive": ["2", 0],
                "negative": ["3", 0],
                "latent_image": ["4", 0],
                "denoise": 1.0,
            },
        },
        "6": {
            "class_type": "VAEDecode",
            "inputs": {"samples": ["5", 0], "vae": ["1", 2]},
        },
        "7": {
            "class_type": "SaveImage",
            "inputs": {"images": ["6", 0], "filename_prefix": filename_prefix},
        },
    }

    r = _http_post_json(base + "/prompt", {"prompt": workflow, "client_id": client_id})
    prompt_id = str(r.get("prompt_id") or "").strip()
    if not prompt_id:
        raise RuntimeError("ComfyUI did not return prompt_id")

    # Poll history until outputs appear.
    deadline = time.time() + 120.0
    hist = None
    while time.time() < deadline:
        try:
            h = _http_get_json(
                base + "/history/" + urllib.parse.quote(prompt_id), timeout=10.0
            )
            if isinstance(h, dict) and prompt_id in h:
                hist = h[prompt_id]
                outs = (hist or {}).get("outputs") or {}
                if outs:
                    break
        except Exception:
            pass
        time.sleep(0.75)
    if not hist:
        raise RuntimeError("Timed out waiting for ComfyUI history")

    outs = (hist.get("outputs") or {}).get("7") or {}
    images = outs.get("images") or []
    if not images:
        raise RuntimeError("ComfyUI returned no images")

    img0 = images[0]
    filename = str(img0.get("filename") or "")
    subfolder = str(img0.get("subfolder") or "")
    ftype = str(img0.get("type") or "output")
    if not filename:
        raise RuntimeError("ComfyUI image missing filename")

    qs = urllib.parse.urlencode(
        {"filename": filename, "subfolder": subfolder, "type": ftype}
    )
    view_url = base + "/view?" + qs
    with urllib.request.urlopen(view_url, timeout=30.0) as resp:
        data = resp.read()

    # Save to a temp file; caller moves it into run folder.
    tmp = tempfile.NamedTemporaryFile(
        prefix="bg_keyframe_", suffix=".png", delete=False
    )
    tmp.write(data)
    tmp.close()
    return tmp.name, hist


def _parse_lyrics_sections(lyrics: str) -> list[tuple[str, str]]:
    """Parse [Section] blocks from lyrics.

    Returns list of (section_name, block_text).
    """

    lines = (lyrics or "").replace("\r\n", "\n").replace("\r", "\n").split("\n")
    out: list[tuple[str, list[str]]] = []
    cur_name = ""
    cur: list[str] = []
    for raw in lines:
        s = (raw or "").strip()
        if not s:
            continue
        m = re.fullmatch(r"\[(.+?)\]", s)
        if m:
            if cur_name or cur:
                out.append((cur_name or "", cur))
            cur_name = m.group(1).strip()
            cur = []
            continue
        cur.append(s)
    if cur_name or cur:
        out.append((cur_name or "", cur))

    blocks: list[tuple[str, str]] = []
    for name, ls in out:
        txt = "\n".join(ls).strip()
        if txt:
            blocks.append((name or "", txt))
    return blocks


def _section_timeline_windows(
    lyrics: str, duration_s: float
) -> list[tuple[str, float, float]]:
    """Estimate section windows by distributing time across lyric lines.

    This is a heuristic (we don't have true section timestamps), but it works
    well enough to drive background crossfades.
    """

    blocks = _parse_lyrics_sections(lyrics)
    # Flatten into (section,line)
    flat: list[tuple[str, str]] = []
    for name, txt in blocks:
        for ln in txt.split("\n"):
            t = (ln or "").strip()
            if t and not re.fullmatch(r"\[.+?\]", t):
                flat.append((name, t))
    if not flat:
        return []

    usable = max(2.0, float(duration_s))
    per = usable / float(len(flat))
    per = max(0.7, min(3.0, per))

    t = 0.0
    windows: list[tuple[str, float, float]] = []
    cur = flat[0][0]
    st = 0.0
    for sec, _ln in flat:
        if sec != cur:
            windows.append((cur, st, t))
            cur = sec
            st = t
        t = min(usable, t + per)
    windows.append((cur, st, usable))
    # Clamp and remove empty.
    out: list[tuple[str, float, float]] = []
    for name, a, b in windows:
        a2 = max(0.0, min(usable, float(a)))
        b2 = max(0.0, min(usable, float(b)))
        if b2 > a2 + 0.25:
            out.append((name or "", a2, b2))
    return out


def make_loop_from_keyframe(
    *,
    keyframe_path: str,
    out_mp4: str,
    width: int,
    height: int,
    fps: int,
    loop_seconds: float,
) -> None:
    ffmpeg = _find_ffmpeg()
    out_mp4 = os.path.abspath(out_mp4)
    loop_seconds = float(loop_seconds)
    fps = int(fps)

    # Build a smooth motion loop (ping-pong) from a still image.
    half = max(1.5, loop_seconds / 2.0)

    with tempfile.TemporaryDirectory(prefix="bgloop_") as tmp:
        raw_mp4 = os.path.join(tmp, "raw.mp4")

        # Motion: gentle zoom + tiny drift. Then grade + grain.
        # Avoid force_original_aspect_ratio=cover for broad FFmpeg compatibility.
        vf = (
            f"scale={int(width * 1.08)}:{int(height * 1.08)},"
            f"crop={width}:{height},"
            f"zoompan=z='min(zoom+0.00055,1.06)':"
            f"x='iw/2-(iw/zoom/2)+sin(on/31)*8':"
            f"y='ih/2-(ih/zoom/2)+cos(on/29)*6':"
            f"d=999999:s={width}x{height}:fps={fps},"
            "gblur=sigma=3.0,"
            "eq=brightness=-0.03:saturation=1.10:contrast=1.06,"
            "noise=alls=10:allf=t+u,"
            "vignette=PI/5,"
            "format=yuv420p"
        )

        cmd1 = [
            ffmpeg,
            "-y",
            "-hide_banner",
            "-loglevel",
            "error",
            "-loop",
            "1",
            "-i",
            keyframe_path,
            "-vf",
            vf,
            "-t",
            f"{half:.3f}",
            "-r",
            str(fps),
            "-c:v",
            "libx264",
            "-pix_fmt",
            "yuv420p",
            "-preset",
            "veryfast",
            "-crf",
            "18",
            raw_mp4,
        ]
        p = subprocess.run(cmd1, capture_output=True, text=True)
        if p.returncode != 0:
            raise RuntimeError((p.stderr or p.stdout or "").strip() or "ffmpeg failed")

        # Ping-pong to create a seamless loop.
        fc = "[0:v]split=2[a][b];[b]reverse[r];[a][r]concat=n=2:v=1:a=0,format=yuv420p[v]"
        cmd2 = [
            ffmpeg,
            "-y",
            "-hide_banner",
            "-loglevel",
            "error",
            "-i",
            raw_mp4,
            "-filter_complex",
            fc,
            "-map",
            "[v]",
            "-t",
            f"{loop_seconds:.3f}",
            "-r",
            str(fps),
            "-c:v",
            "libx264",
            "-pix_fmt",
            "yuv420p",
            "-preset",
            "veryfast",
            "-crf",
            "18",
            out_mp4,
        ]
        p2 = subprocess.run(cmd2, capture_output=True, text=True)
        if p2.returncode != 0:
            raise RuntimeError(
                (p2.stderr or p2.stdout or "").strip() or "ffmpeg failed"
            )


def main() -> int:
    ap = argparse.ArgumentParser(
        description="Generate a cinematic looping background video"
    )
    ap.add_argument("--run-dir", required=True, help="Output run folder")
    ap.add_argument(
        "--comfy-url", default="http://127.0.0.1:8188", help="ComfyUI base URL"
    )
    ap.add_argument(
        "--ckpt", default="sd_xl_turbo_1.0.safetensors", help="Checkpoint name"
    )
    ap.add_argument(
        "--quality",
        default=os.environ.get("BG_QUALITY", "fast"),
        choices=["fast", "balanced", "high"],
        help="Keyframe render quality preset (affects resolution/steps).",
    )
    # Leave these as None so --quality can provide meaningful defaults.
    ap.add_argument("--width", type=int, default=None)
    ap.add_argument("--height", type=int, default=None)
    ap.add_argument("--steps", type=int, default=None)
    ap.add_argument("--cfg", type=float, default=None)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--sampler", default="euler")
    ap.add_argument("--scheduler", default="simple")
    ap.add_argument("--fps", type=int, default=30)
    ap.add_argument("--loop-seconds", type=float, default=8.0)
    ap.add_argument("--out", default="", help="Output mp4 path (optional)")
    ap.add_argument(
        "--llm-base", default=os.environ.get("LLM_BASE_URL", "http://localhost:8317/v1")
    )
    ap.add_argument("--llm-model", default=os.environ.get("LLM_MODEL", "gpt-5.2"))
    ap.add_argument(
        "--llm-api-key",
        default=os.environ.get("LLM_API_KEY") or os.environ.get("OPENAI_API_KEY") or "",
    )
    ap.add_argument(
        "--use-llm",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Use the local LLM server to generate a visual prompt pack",
    )
    ap.add_argument("--prompt", default="", help="Override positive prompt")
    ap.add_argument("--negative", default="", help="Override negative prompt")
    ap.add_argument(
        "--scenes",
        default="base,verse,chorus,bridge",
        help="Comma-separated list of scenes to generate (base,verse,chorus,bridge)",
    )
    args = ap.parse_args()

    run_dir = os.path.abspath(str(args.run_dir))
    meta_path = os.path.join(run_dir, "meta.json")
    if not os.path.isfile(meta_path):
        print(f"Error: meta.json not found: {meta_path}")
        return 2

    meta = _read_json(meta_path)
    loop_s = float(args.loop_seconds or 8.0)
    title = str(meta.get("title") or "").strip()
    prefix = "bg_" + _slugify(title)

    # Quality presets.
    q = str(args.quality or "fast").strip().lower()
    if q not in ("fast", "balanced", "high"):
        q = "fast"

    ckpt_l = str(args.ckpt or "").lower()
    turbo = "turbo" in ckpt_l

    # Defaults for SDXL Turbo (low steps/cfg) vs classic diffusion (more steps/cfg).
    if turbo:
        if q == "fast":
            q_width, q_height, q_steps, q_cfg = 768, 432, 6, 2.0
        elif q == "balanced":
            q_width, q_height, q_steps, q_cfg = 1024, 576, 8, 2.2
        else:
            q_width, q_height, q_steps, q_cfg = 1280, 720, 10, 2.4
    else:
        if q == "fast":
            q_width, q_height, q_steps, q_cfg = 768, 432, 18, 6.0
        elif q == "balanced":
            q_width, q_height, q_steps, q_cfg = 1024, 576, 26, 6.5
        else:
            q_width, q_height, q_steps, q_cfg = 1280, 720, 34, 7.0

    # Allow explicit CLI overrides.
    width = int(args.width if args.width is not None else q_width)
    height = int(args.height if args.height is not None else q_height)
    steps = int(args.steps if args.steps is not None else q_steps)
    cfg = float(args.cfg if args.cfg is not None else q_cfg)

    # Scenes to generate.
    want = [x.strip().lower() for x in str(args.scenes or "").split(",") if x.strip()]
    allowed = {"base", "verse", "chorus", "bridge"}
    scenes = [x for x in want if x in allowed]
    if not scenes:
        scenes = ["base", "verse", "chorus", "bridge"]

    # Build prompt pack (single or multi-scene).
    pack: dict
    if str(args.prompt or "").strip():
        neg0 = (
            str(args.negative or "").strip()
            or "text, watermark, logo, subtitles, readable text, UI, poster, typography"
        )
        pack = {
            "palette": [],
            "camera": "",
            "motion_notes": "",
            "negative_prompt": neg0,
            "recommended_bg_preset": "Cinematic Soft",
            "base": {
                "scene": "(override)",
                "positive_prompt": str(args.prompt).strip(),
            },
            "verse": {
                "scene": "(override)",
                "positive_prompt": str(args.prompt).strip(),
            },
            "chorus": {
                "scene": "(override)",
                "positive_prompt": str(args.prompt).strip(),
            },
            "bridge": {
                "scene": "(override)",
                "positive_prompt": str(args.prompt).strip(),
            },
        }
    elif not bool(args.use_llm):
        neg0 = "text, watermark, logo, subtitles, readable text, UI, faces, people, lowres, blurry, artifacts"
        base_pos = "cinematic night city street in rain, neon reflections, moody lighting, shallow depth of field, film grain, bokeh, 35mm, high detail"
        pack = {
            "palette": ["deep navy", "neon teal", "warm amber"],
            "camera": "35mm lens, shallow depth of field",
            "motion_notes": "slow dolly drift, handheld micro-movement",
            "negative_prompt": neg0,
            "recommended_bg_preset": "Cinematic Soft",
            "base": {
                "scene": "Night city establishing",
                "positive_prompt": base_pos + ", empty streets, no people",
            },
            "verse": {
                "scene": "Interior warm glow",
                "positive_prompt": "cinematic warm interior at night, window rain bokeh, practical lights, film grain, 35mm, shallow depth of field, no people",
            },
            "chorus": {
                "scene": "Neon street energy",
                "positive_prompt": base_pos
                + ", brighter neon signs out of focus, wet asphalt, faster light streaks, no people",
            },
            "bridge": {
                "scene": "Rooftop skyline",
                "positive_prompt": "cinematic rooftop skyline at night, distant city lights bokeh, moody haze, film grain, 50mm, no people",
            },
        }
    else:
        try:
            pack = llm_make_visual_prompt_sections(
                llm_base=str(args.llm_base),
                llm_model=str(args.llm_model),
                llm_api_key=str(args.llm_api_key),
                meta=meta,
                loop_seconds=loop_s,
            )
        except Exception as e:
            neg0 = "text, watermark, logo, subtitles, readable text, UI, faces, people, lowres, blurry, artifacts"
            pack = {
                "palette": ["deep navy", "neon teal", "warm amber"],
                "camera": "35mm lens, shallow depth of field",
                "motion_notes": "slow dolly drift",
                "negative_prompt": neg0,
                "recommended_bg_preset": "Cinematic Soft",
                "base": {
                    "scene": "cinematic background (LLM unavailable)",
                    "positive_prompt": "cinematic night city street in rain, neon reflections, moody lighting, shallow depth of field, film grain, bokeh, 35mm, high detail, no people",
                },
                "verse": {
                    "scene": "verse",
                    "positive_prompt": "cinematic moody streetlights in rain, shallow depth of field, film grain, no people",
                },
                "chorus": {
                    "scene": "chorus",
                    "positive_prompt": "cinematic neon reflections, wet asphalt, bokeh, film grain, no people",
                },
                "bridge": {
                    "scene": "bridge",
                    "positive_prompt": "cinematic skyline haze, distant bokeh, film grain, no people",
                },
                "llm_error": str(e),
            }

    # Persist prompt pack for the run.
    _write_json(os.path.join(run_dir, "visual_prompt.json"), pack)

    neg = (
        str(pack.get("negative_prompt") or "").strip()
        or "text, watermark, logo, subtitles, readable text, UI"
    )

    if int(args.seed) == 0:
        seed = int(uuid.uuid4().int % 2147483647)
    else:
        seed = int(args.seed)

    # Generate scene keyframes + loops.
    out_files: dict[str, str] = {}
    for scene in scenes:
        pos = str(((pack.get(scene) or {}).get("positive_prompt") or "")).strip()
        if not pos:
            pos = str(((pack.get("base") or {}).get("positive_prompt") or "")).strip()
        if not pos:
            continue

        scene_prefix = prefix + "_" + scene
        tmp_png, _hist = comfy_render_keyframe(
            comfy_url=str(args.comfy_url),
            ckpt_name=str(args.ckpt),
            positive=pos,
            negative=neg,
            width=int(width),
            height=int(height),
            steps=int(steps),
            cfg=float(cfg),
            seed=int(seed + (17 * (scenes.index(scene) + 1))),
            sampler=str(args.sampler),
            scheduler=str(args.scheduler),
            filename_prefix=scene_prefix,
        )

        keyframe_out = os.path.join(run_dir, f"bg_keyframe_{scene}.png")
        try:
            import shutil

            shutil.move(tmp_png, keyframe_out)
        except Exception:
            keyframe_out = tmp_png

        out_loop = os.path.join(run_dir, f"bg_loop_{scene}.mp4")
        make_loop_from_keyframe(
            keyframe_path=keyframe_out,
            out_mp4=out_loop,
            width=1920,
            height=1080,
            fps=int(args.fps),
            loop_seconds=float(loop_s),
        )
        out_files[scene] = out_loop

    # Compatibility: write bg_loop.mp4 as base if available.
    base_loop = out_files.get("base")
    if base_loop and os.path.isfile(base_loop):
        compat = os.path.join(run_dir, "bg_loop.mp4")
        try:
            import shutil

            shutil.copyfile(base_loop, compat)
        except Exception:
            pass

    if args.out:
        # If user requested an explicit output path, copy base loop there.
        # This keeps CLI behavior stable.
        target = os.path.abspath(str(args.out))
        src = (
            base_loop
            or out_files.get("verse")
            or out_files.get("chorus")
            or out_files.get("bridge")
        )
        if src and os.path.isfile(src):
            try:
                import shutil

                shutil.copyfile(src, target)
                print(f"Saved background loop: {target}")
            except Exception:
                pass

    # Print an easy-to-scan summary.
    if out_files:
        for k in sorted(out_files.keys()):
            print(f"Saved {k} loop: {out_files[k]}")
    else:
        print("Error: no loops generated")
        return 2
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
