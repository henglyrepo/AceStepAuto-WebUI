import json
import os
import queue
import re
import shlex
import subprocess
import sys
import threading
import time
import webbrowser
import urllib.error
import urllib.request
import urllib.parse


import gradio as gr


def _load_dotenv(path: str = ".env") -> None:
    """Load simple KEY=VALUE pairs into os.environ.

    This is intentionally dependency-free so it works with ACE-Step embedded Python
    environments that may restrict import paths.
    """

    try:
        with open(path, "r", encoding="utf-8") as f:
            for raw_line in f:
                line = raw_line.strip()
                if not line or line.startswith("#"):
                    continue
                if line.lower().startswith("export "):
                    line = line[7:].strip()
                if "=" not in line:
                    continue
                key, val = line.split("=", 1)
                key = key.strip()
                val = val.strip()
                if not key or key.startswith("#"):
                    continue
                if len(val) >= 2 and (
                    (val[0] == '"' and val[-1] == '"')
                    or (val[0] == "'" and val[-1] == "'")
                ):
                    val = val[1:-1]
                os.environ.setdefault(key, val)
    except FileNotFoundError:
        return


def _dotenv_upsert(path: str, key: str, value: str) -> None:
    """Upsert a KEY=VALUE entry in a .env file.

    - Preserves existing lines/comments as much as possible.
    - Replaces the first occurrence of `key=` (optionally prefixed by `export `).
    - Appends if not present.
    """

    key = (key or "").strip()
    if not key:
        raise ValueError("Missing key")

    value = "" if value is None else str(value)
    # Quote only if needed.
    if any(c in value for c in (" ", "#", "\t", '"')):
        value_out = '"' + value.replace('"', '\\"') + '"'
    else:
        value_out = value

    new_line = f"{key}={value_out}"

    try:
        with open(path, "r", encoding="utf-8") as f:
            lines = f.read().splitlines()
    except FileNotFoundError:
        lines = []

    updated = False
    out: list[str] = []
    for raw in lines:
        line = raw
        stripped = line.strip()
        if not stripped or stripped.startswith("#"):
            out.append(line)
            continue

        check = stripped
        if check.lower().startswith("export "):
            check = check[7:].lstrip()

        if not updated and check.startswith(key + "="):
            out.append(new_line)
            updated = True
        else:
            out.append(line)

    if not updated:
        if out and out[-1].strip() != "":
            out.append("")
        out.append(new_line)

    with open(path, "w", encoding="utf-8") as f:
        f.write("\n".join(out) + "\n")


def save_llm_api_key(llm_api_key: str) -> str:
    k = (llm_api_key or "").strip()
    if not k:
        return "No key provided."
    try:
        _dotenv_upsert(".env", "LLM_API_KEY", k)
        os.environ["LLM_API_KEY"] = k
        return "Saved LLM_API_KEY to .env"
    except Exception as e:
        return f"Failed to save: {e}"


WEBUI_HOST = os.environ.get("WEBUI_HOST", "127.0.0.1")
WEBUI_PORT = int(os.environ.get("WEBUI_PORT", "7865"))


GENRES_TOP20 = [
    "Pop",
    "Hip-Hop / Rap",
    "Trap",
    "R&B",
    "Afrobeats",
    "Amapiano",
    "K-pop",
    "Reggaeton",
    "Latin Pop",
    "EDM",
    "House",
    "Tech House",
    "Techno",
    "Drum & Bass",
    "UK Garage",
    "Jersey Club",
    "Phonk",
    "Lo-fi",
    "Indie Pop",
    "Alternative Rock",
]


def _pick_port(host: str, preferred: int) -> int:
    """Pick a free port, preferring `preferred` then a small range."""

    def _free(p: int) -> bool:
        s = socket.socket()
        try:
            s.settimeout(0.2)
            return s.connect_ex((host, p)) != 0
        finally:
            s.close()

    import socket

    if _free(preferred):
        return preferred
    for p in range(preferred + 1, preferred + 21):
        if _free(p):
            return p
    return preferred


def _http_get_json(
    url: str, timeout: float = 5.0, headers: dict[str, str] | None = None
) -> dict:
    req = urllib.request.Request(url, method="GET")
    for k, v in (headers or {}).items():
        if v:
            req.add_header(k, v)
    with urllib.request.urlopen(req, timeout=timeout) as resp:
        raw = resp.read().decode("utf-8", errors="replace")
    return json.loads(raw)


def _http_post_json(
    url: str,
    payload: dict,
    timeout: float = 10.0,
    headers: dict[str, str] | None = None,
) -> dict:
    body = json.dumps(payload).encode("utf-8")
    req = urllib.request.Request(url, data=body, method="POST")
    req.add_header("Content-Type", "application/json")
    for k, v in (headers or {}).items():
        if v:
            req.add_header(k, v)
    with urllib.request.urlopen(req, timeout=timeout) as resp:
        raw = resp.read().decode("utf-8", errors="replace")
    return json.loads(raw)


def _check_endpoints(
    *,
    llm_base: str,
    llm_api_key: str,
    acestep_api: str,
    comfy_url: str,
) -> str:
    lines: list[str] = []

    # ACE-Step health
    try:
        health = _http_get_json(acestep_api.rstrip("/") + "/health", timeout=5.0)
        status = (health.get("data") or {}).get("status") or "ok"
        lines.append(f"ACE-Step: OK ({status})")
    except Exception as e:
        lines.append(f"ACE-Step: NOT OK ({e})")

    # ComfyUI health (best-effort)
    comfy = (comfy_url or "http://127.0.0.1:8188").strip() or "http://127.0.0.1:8188"
    try:
        _http_get_json(comfy.rstrip("/") + "/system_stats", timeout=3.0)
        lines.append("ComfyUI: OK (/system_stats)")
    except Exception:
        host, port = _parse_http_host_port(comfy, 8188)
        if _tcp_is_open(host, port, timeout_s=0.5):
            lines.append("ComfyUI: REACHABLE")
        else:
            lines.append("ComfyUI: NOT OK")

    # LLM models
    try:
        headers = {}
        key = (
            (llm_api_key or "").strip()
            or os.environ.get("LLM_API_KEY")
            or os.environ.get("OPENAI_API_KEY")
        )
        if key:
            headers["Authorization"] = f"Bearer {key}"
        _http_get_json(llm_base.rstrip("/") + "/models", timeout=5.0, headers=headers)
        lines.append("LLM: OK (/models)")
    except urllib.error.HTTPError as e:
        if getattr(e, "code", None) == 401:
            lines.append("LLM: UNAUTHORIZED (missing/invalid API key)")
        else:
            lines.append(f"LLM: NOT OK (HTTP {getattr(e, 'code', '?')})")
    except Exception as e:
        lines.append(f"LLM: NOT OK ({e})")

    return "\n".join(lines)


def _extract_json_object(text: str) -> dict:
    text = (text or "").strip()
    # Try direct parse first.
    try:
        obj = json.loads(text)
        if isinstance(obj, dict):
            return obj
    except Exception:
        pass

    # Fall back to extracting the first JSON object from mixed text.
    # This handles common cases like code fences or preambles.
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


def _llm_key_from_inputs(llm_api_key: str) -> str:
    return (
        (llm_api_key or "").strip()
        or os.environ.get("LLM_API_KEY")
        or os.environ.get("OPENAI_API_KEY")
        or ""
    )


def llm_suggest_topic(
    style: str, genres: list[str], llm_base: str, llm_model: str, llm_api_key: str
) -> tuple[str, str]:
    genre_mix = ", ".join(genres or [])
    prompt = (
        "Return ONLY JSON with keys: topic. Do not wrap in markdown or code fences. "
        "Topic must be short (2-8 words), modern, and ASCII.\n"
        f"Style: {style.strip()}\n"
        f"Genres: {genre_mix}\n"
    )
    url = llm_base.rstrip("/") + "/chat/completions"
    headers: dict[str, str] = {}
    key = _llm_key_from_inputs(llm_api_key)
    if key:
        headers["Authorization"] = f"Bearer {key}"
    payload = {
        "model": llm_model,
        "messages": [
            {
                "role": "system",
                "content": "You generate music prompts. Return only JSON. No code fences.",
            },
            {"role": "user", "content": prompt},
        ],
        "temperature": 0.9,
    }
    try:
        resp = _http_post_json(url, payload, timeout=30.0, headers=headers)
        content = resp["choices"][0]["message"]["content"]
        obj = _extract_json_object(content)
        topic = (obj.get("topic") or "").strip()
        if not topic:
            return "", "LLM returned empty topic."
        return topic, "OK"
    except Exception as e:
        return "", f"LLM error: {e}"


def llm_suggest_style(
    topic: str, genres: list[str], llm_base: str, llm_model: str, llm_api_key: str
) -> tuple[str, str]:
    genre_mix = ", ".join(genres or [])
    prompt = (
        "Return ONLY JSON with keys: style. Do not wrap in markdown or code fences. "
        "Style must be a single short line (genre + production vibe + instruments + vocal vibe). ASCII only.\n"
        f"Topic: {topic.strip()}\n"
        f"Genres: {genre_mix}\n"
    )
    url = llm_base.rstrip("/") + "/chat/completions"
    headers: dict[str, str] = {}
    key = _llm_key_from_inputs(llm_api_key)
    if key:
        headers["Authorization"] = f"Bearer {key}"
    payload = {
        "model": llm_model,
        "messages": [
            {
                "role": "system",
                "content": "You generate music prompts. Return only JSON. No code fences.",
            },
            {"role": "user", "content": prompt},
        ],
        "temperature": 0.9,
    }
    try:
        resp = _http_post_json(url, payload, timeout=30.0, headers=headers)
        content = resp["choices"][0]["message"]["content"]
        obj = _extract_json_object(content)
        style = (obj.get("style") or "").strip()
        if not style:
            return "", "LLM returned empty style."
        return style, "OK"
    except Exception as e:
        return "", f"LLM error: {e}"


def _normalize_genre_name(s: str) -> str:
    import re

    s = (s or "").strip().lower()
    # Keep only alphanumerics for tolerant matching.
    return re.sub(r"[^a-z0-9]+", "", s)


def _coerce_genres(raw, allowed: list[str]) -> list[str]:
    if raw is None:
        return []
    if isinstance(raw, str):
        # Support comma-separated strings.
        raw_list = [x.strip() for x in raw.split(",") if x.strip()]
    elif isinstance(raw, list):
        raw_list = [str(x).strip() for x in raw if str(x).strip()]
    else:
        raw_list = [str(raw).strip()] if str(raw).strip() else []

    canon = {_normalize_genre_name(g): g for g in (allowed or [])}
    out: list[str] = []
    seen: set[str] = set()
    for item in raw_list:
        key = _normalize_genre_name(item)
        g = canon.get(key)
        if not g:
            continue
        if g in seen:
            continue
        out.append(g)
        seen.add(g)
    return out


def llm_suggest_style_with_genres(
    topic: str,
    llm_base: str,
    llm_model: str,
    llm_api_key: str,
    allowed_genres: list[str],
) -> tuple[str, list[str], str]:
    """Suggest style and (optional) relevant genres.

    Used by the WebUI when the user has not selected any genres.
    """

    genre_allowed = ", ".join(allowed_genres or [])
    prompt = (
        "Return ONLY JSON with keys: style, genres. Do not wrap in markdown or code fences.\n"
        "Rules:\n"
        "- style: one line (genre + production vibe + instruments + vocal vibe), ASCII only\n"
        "- genres: 1-3 items, must be picked from the Allowed genres list exactly\n"
        f"Topic: {topic.strip()}\n"
        f"Allowed genres: {genre_allowed}\n"
    )

    url = llm_base.rstrip("/") + "/chat/completions"
    headers: dict[str, str] = {}
    key = _llm_key_from_inputs(llm_api_key)
    if key:
        headers["Authorization"] = f"Bearer {key}"

    payload = {
        "model": llm_model,
        "messages": [
            {
                "role": "system",
                "content": "You generate music prompts. Return only JSON. No code fences.",
            },
            {"role": "user", "content": prompt},
        ],
        "temperature": 0.9,
    }

    try:
        resp = _http_post_json(url, payload, timeout=30.0, headers=headers)
        content = resp["choices"][0]["message"]["content"]
        obj = _extract_json_object(content)
        style = (obj.get("style") or "").strip()
        if not style:
            return "", [], "LLM returned empty style."
        genres = _coerce_genres(obj.get("genres"), allowed_genres)
        genres, _ = _enforce_max_genres(genres)
        return style, genres, "OK"
    except Exception as e:
        return "", [], f"LLM error: {e}"


def suggest_style_click(
    topic: str,
    genres: list[str] | None,
    llm_base: str,
    llm_model: str,
    llm_api_key: str,
) -> tuple[str, list[str], str]:
    """UI handler for 'Suggest Style From Topic'.

    If the user hasn't selected any genres yet, also suggest 1-3 relevant genres
    from the allowed list and auto-select them.
    """

    g = genres or []
    if not g:
        style_out, genres_out, status = llm_suggest_style_with_genres(
            topic=topic,
            llm_base=llm_base,
            llm_model=llm_model,
            llm_api_key=llm_api_key,
            allowed_genres=GENRES_TOP20,
        )
        return style_out, genres_out, status

    style_out, status = llm_suggest_style(
        topic=topic,
        genres=g,
        llm_base=llm_base,
        llm_model=llm_model,
        llm_api_key=llm_api_key,
    )
    return style_out, g, status


def llm_suggest_both(
    topic: str,
    style: str,
    genres: list[str],
    llm_base: str,
    llm_model: str,
    llm_api_key: str,
) -> tuple[str, str, str]:
    genre_mix = ", ".join(genres or [])
    prompt = (
        "Return ONLY JSON with keys: topic, style. ASCII only. Do not wrap in markdown or code fences.\n"
        "Rules:\n"
        "- topic: short (2-8 words)\n"
        "- style: one line (genre + production + instruments + vocal vibe)\n"
        f"Input topic (may be blank): {topic.strip()}\n"
        f"Input style (may be blank): {style.strip()}\n"
        f"Genres: {genre_mix}\n"
    )
    url = llm_base.rstrip("/") + "/chat/completions"
    headers: dict[str, str] = {}
    key = _llm_key_from_inputs(llm_api_key)
    if key:
        headers["Authorization"] = f"Bearer {key}"
    payload = {
        "model": llm_model,
        "messages": [
            {
                "role": "system",
                "content": "You generate music prompts. Return only JSON. No code fences.",
            },
            {"role": "user", "content": prompt},
        ],
        "temperature": 0.9,
    }
    try:
        resp = _http_post_json(url, payload, timeout=30.0, headers=headers)
        content = resp["choices"][0]["message"]["content"]
        obj = _extract_json_object(content)
        out_topic = (obj.get("topic") or topic or "").strip()
        out_style = (obj.get("style") or style or "").strip()
        return out_topic, out_style, "OK"
    except Exception as e:
        return topic, style, f"LLM error: {e}"


def llm_random_topic_style(
    genres: list[str] | None,
    llm_base: str,
    llm_model: str,
    llm_api_key: str,
    allowed_genres: list[str],
) -> tuple[str, str, list[str], str]:
    """Generate a random topic + style.

    - If genres are selected, do not change them; style should match.
    - If no genres selected, also suggest 1-3 genres from allowed list.
    """

    g_in = [x for x in (genres or []) if (x or "").strip()]
    genre_mix = ", ".join(g_in)
    allowed_mix = ", ".join(allowed_genres or [])

    if genre_mix:
        prompt = (
            "Return ONLY JSON with keys: topic, style. ASCII only. Do not wrap in markdown or code fences.\n"
            "Rules:\n"
            "- topic: short (2-8 words), modern, catchy\n"
            "- style: one line (genre + production vibe + instruments + vocal vibe)\n"
            "- Make the style strongly match the selected genres\n"
            "- Keep it feasible for a 30-60s generated song\n"
            f"Selected genres: {genre_mix}\n"
        )
    else:
        prompt = (
            "Return ONLY JSON with keys: topic, style, genres. ASCII only. Do not wrap in markdown or code fences.\n"
            "Rules:\n"
            "- topic: short (2-8 words), modern, catchy\n"
            "- style: one line (genre + production vibe + instruments + vocal vibe)\n"
            "- genres: 1-3 items, must be picked from the Allowed genres list exactly\n"
            "- Keep it feasible for a 30-60s generated song\n"
            f"Allowed genres: {allowed_mix}\n"
        )

    url = llm_base.rstrip("/") + "/chat/completions"
    headers: dict[str, str] = {}
    key = _llm_key_from_inputs(llm_api_key)
    if key:
        headers["Authorization"] = f"Bearer {key}"

    payload = {
        "model": llm_model,
        "messages": [
            {
                "role": "system",
                "content": "You generate music prompt ideas. Return only JSON. No code fences.",
            },
            {"role": "user", "content": prompt},
        ],
        "temperature": 1.0,
    }

    try:
        resp = _http_post_json(url, payload, timeout=30.0, headers=headers)
        content = resp["choices"][0]["message"]["content"]
        obj = _extract_json_object(content)
        out_topic = (obj.get("topic") or "").strip()
        out_style = (obj.get("style") or "").strip()
        if not out_topic or not out_style:
            return "", "", (g_in or []), "LLM returned empty topic/style."

        if g_in:
            g_out, _ = _enforce_max_genres(g_in)
            return out_topic, out_style, g_out, "OK"

        # No genres selected: try to coerce from allowed list.
        g_out = _coerce_genres(obj.get("genres"), allowed_genres)
        g_out, _ = _enforce_max_genres(g_out)
        if not g_out:
            # Leave blank if LLM didn't comply.
            return out_topic, out_style, [], "OK (no genres suggested)"
        return out_topic, out_style, g_out, "OK"
    except Exception as e:
        return "", "", (g_in or []), f"LLM error: {e}"


def random_idea_click(
    genres: list[str] | None,
    llm_base: str,
    llm_model: str,
    llm_api_key: str,
) -> tuple[str, str, list[str], str]:
    return llm_random_topic_style(
        genres=genres,
        llm_base=llm_base,
        llm_model=llm_model,
        llm_api_key=llm_api_key,
        allowed_genres=GENRES_TOP20,
    )


def _enforce_max_genres(genres: list[str]) -> tuple[list[str], str]:
    g = list(genres or [])
    if len(g) <= 5:
        return g, ""
    trimmed = g[:5]
    return trimmed, "Max 5 genres. Trimmed selection to first 5."


def _bg_preset_updates(preset: str):
    """Return gr.update() objects for background grading sliders."""

    p = (preset or "").strip().lower()
    if p in ("", "custom"):
        return (
            gr.update(),
            gr.update(),
            gr.update(),
            gr.update(),
            gr.update(),
            gr.update(),
            gr.update(),
        )

    presets = {
        # Soft, filmic, readable.
        "cinematic soft": {
            "blur": 2.8,
            "grain": 10.0,
            "vignette": 0.60,
            "brightness": -0.04,
            "saturation": 1.08,
            "contrast": 1.06,
            "plate": 0.30,
        },
        # Minimal processing.
        "crisp": {
            "blur": 1.4,
            "grain": 6.0,
            "vignette": 0.40,
            "brightness": -0.02,
            "saturation": 1.04,
            "contrast": 1.02,
            "plate": 0.26,
        },
        # Darker, heavier plate for club/EDM.
        "dark club": {
            "blur": 3.2,
            "grain": 12.0,
            "vignette": 0.70,
            "brightness": -0.07,
            "saturation": 1.10,
            "contrast": 1.10,
            "plate": 0.36,
        },
        # Warmer, slightly lifted mids.
        "warm film": {
            "blur": 2.6,
            "grain": 9.0,
            "vignette": 0.55,
            "brightness": -0.03,
            "saturation": 1.12,
            "contrast": 1.04,
            "plate": 0.28,
        },
    }

    cfg = presets.get(p)
    if not cfg:
        return (
            gr.update(),
            gr.update(),
            gr.update(),
            gr.update(),
            gr.update(),
            gr.update(),
            gr.update(),
        )

    return (
        gr.update(value=float(cfg["blur"])),
        gr.update(value=float(cfg["grain"])),
        gr.update(value=float(cfg["vignette"])),
        gr.update(value=float(cfg["brightness"])),
        gr.update(value=float(cfg["saturation"])),
        gr.update(value=float(cfg["contrast"])),
        gr.update(value=float(cfg["plate"])),
    )


def _normalize_bg_preset_name(s: str) -> str:
    x = (s or "").strip().lower()
    x = re.sub(r"\s+", " ", x)
    return x


def _recommend_bg_preset_from_meta(meta_path: str) -> str:
    """Heuristic preset when LLM prompt pack is missing.

    Uses the style/genre keywords from meta.json.
    """

    try:
        with open(meta_path, "r", encoding="utf-8") as f:
            meta = json.load(f)
    except Exception:
        return "Cinematic Soft"

    style = str(meta.get("style") or "").lower()
    caption = str(meta.get("caption") or "").lower()
    blob = style + "\n" + caption
    if any(
        k in blob
        for k in (
            "edm",
            "club",
            "dance",
            "techno",
            "house",
            "trance",
            "dnb",
            "drum",
            "bass",
        )
    ):
        return "Dark Club"
    if any(
        k in blob
        for k in (
            "lofi",
            "lo-fi",
            "acoustic",
            "ballad",
            "folk",
            "indie",
            "warm",
            "vintage",
        )
    ):
        return "Warm Film"
    if any(k in blob for k in ("rock", "metal", "punk")):
        return "Crisp"
    return "Cinematic Soft"


def apply_recommended_bg_preset(
    run_folder: str,
    visual_prompt_json: str,
) -> tuple[str, str]:
    """Pick preset from visual_prompt.json (LLM) or fallback to meta.json."""

    run_folder = (run_folder or "").strip()
    rec = ""
    # Prefer LLM recommendation.
    vp = (visual_prompt_json or "").strip()
    if vp and os.path.isfile(vp):
        try:
            with open(vp, "r", encoding="utf-8") as f:
                pack = json.load(f)
            rec = str(pack.get("recommended_bg_preset") or "").strip()
        except Exception:
            rec = ""

    if not rec and run_folder and os.path.isdir(run_folder):
        meta = os.path.join(run_folder, "meta.json")
        if os.path.isfile(meta):
            rec = _recommend_bg_preset_from_meta(meta)

    if not rec:
        rec = "Cinematic Soft"

    # Normalize to our dropdown values.
    m = {
        "cinematic soft": "Cinematic Soft",
        "warm film": "Warm Film",
        "dark club": "Dark Club",
        "crisp": "Crisp",
    }
    key = _normalize_bg_preset_name(rec)
    out = m.get(key)
    if not out:
        out = "Cinematic Soft"
    return (out, f"Applied BG preset: {out}")


def _combine_style(style: str, genres: list[str]) -> str:
    s = (style or "").strip()
    g = ", ".join([x for x in (genres or []) if (x or "").strip()])
    if g and s:
        return f"{g}; {s}"
    if g:
        return g
    return s


def _make_runner_state() -> dict:
    return {
        "proc": None,
        "queue": None,
        "reader": None,
        "log_lines": [],
        "audio_path": "",
        "meta_path": "",
        "out_dir": "",
        "task_id": "",
        "acestep_api": "",
        "run_started_at": None,
        "last_eta_poll_at": 0.0,
        "eta_seconds": None,
        "queue_position": None,
        "job_status": "",
    }


def _make_server_state() -> dict:
    return {
        "proc": None,
        "log_lines": [],
        "started_at": None,
        "acestep_dir": "",
    }


def _make_comfy_state() -> dict:
    return {
        "proc": None,
        "log_lines": [],
        "started_at": None,
        "bat_path": "",
    }


def _split_args(arg_string: str) -> list[str]:
    """Split a command-line arg string into argv safely."""

    s = (arg_string or "").strip()
    if not s:
        return []
    try:
        return shlex.split(s, posix=False)
    except Exception:
        return [x for x in s.split(" ") if x]


def _resolve_path(repo_root: str, p: str) -> str:
    """Resolve a user-provided path relative to repo root when needed."""
    p = (p or "").strip()
    if not p:
        return ""
    if os.path.isabs(p):
        return os.path.normpath(p)
    return os.path.normpath(os.path.abspath(os.path.join(repo_root, p)))


def _tcp_is_open(host: str, port: int, timeout_s: float = 0.6) -> bool:
    import socket

    s = None
    try:
        s = socket.socket()
        s.settimeout(timeout_s)
        return s.connect_ex((host, int(port))) == 0
    except Exception:
        return False
    finally:
        if s is not None:
            try:
                s.close()
            except Exception:
                pass


def _parse_http_host_port(url: str, default_port: int) -> tuple[str, int]:
    try:
        u = urllib.parse.urlparse((url or "").strip())
        host = u.hostname or "127.0.0.1"
        port = int(u.port or default_port)
        return host, port
    except Exception:
        return "127.0.0.1", int(default_port)


def _append_log(state: dict, line: str) -> None:
    if not line:
        return
    s = line.rstrip("\r\n")
    state["log_lines"].append(s)
    # Keep last N lines to avoid runaway memory.
    if len(state["log_lines"]) > 2000:
        state["log_lines"] = state["log_lines"][len(state["log_lines"]) - 2000 :]

    if s.startswith("Saved audio:"):
        state["audio_path"] = s.split(":", 1)[1].strip()
        state["out_dir"] = os.path.dirname(state["audio_path"])
    elif s.startswith("Saved meta"):
        state["meta_path"] = s.split(":", 1)[1].strip()
    elif s.startswith("Submitted task:"):
        state["task_id"] = s.split(":", 1)[1].strip()


def _fmt_seconds(secs: float | None) -> str:
    if secs is None:
        return ""
    try:
        secs_i = int(round(float(secs)))
    except Exception:
        return ""
    if secs_i < 0:
        secs_i = 0
    mm, ss = divmod(secs_i, 60)
    hh, mm = divmod(mm, 60)
    if hh:
        return f"{hh:02d}:{mm:02d}:{ss:02d}"
    return f"{mm:02d}:{ss:02d}"


def _poll_acestep_eta(state: dict) -> None:
    task_id = (state.get("task_id") or "").strip()
    api_base = (state.get("acestep_api") or "").strip()
    if not task_id or not api_base:
        return

    now = time.time()
    last = float(state.get("last_eta_poll_at") or 0.0)
    if now - last < 1.0:
        return
    state["last_eta_poll_at"] = now

    try:
        resp = _http_post_json(
            api_base.rstrip("/") + "/query_result",
            {"task_id_list": [task_id]},
            timeout=5.0,
        )
        items = resp.get("data") or []
        if not items:
            return
        item = items[0] if isinstance(items, list) else items

        # Support multiple response shapes.
        # Preferred fields from newer server: queue_position, eta_seconds, status.
        eta = item.get("eta_seconds") if isinstance(item, dict) else None
        qp = item.get("queue_position") if isinstance(item, dict) else None
        st = item.get("status") if isinstance(item, dict) else None

        state["eta_seconds"] = eta
        state["queue_position"] = qp
        state["job_status"] = (
            str(st) if st is not None else state.get("job_status") or ""
        )
    except Exception:
        return


def _build_run_status(state: dict, base: str) -> str:
    parts: list[str] = [base]

    started = state.get("run_started_at")
    if isinstance(started, (int, float)) and started > 0:
        parts.append(f"elapsed {_fmt_seconds(time.time() - float(started))}")

    jp = state.get("queue_position")
    if jp not in (None, ""):
        try:
            jp_i = int(jp)
            if jp_i > 0:
                parts.append(f"queue pos {jp_i}")
        except Exception:
            pass

    eta = state.get("eta_seconds")
    eta_str = _fmt_seconds(eta if isinstance(eta, (int, float)) else None)
    if eta_str:
        parts.append(f"eta {eta_str}")

    job_status = (state.get("job_status") or "").strip()
    if job_status and job_status not in ("0", "1", "2"):
        parts.append(f"status {job_status}")

    return ", ".join(parts)


def _log_text(state: dict) -> str:
    return "\n".join(state.get("log_lines") or [])


def _server_log_text(state: dict) -> str:
    return "\n".join(state.get("log_lines") or [])


def _append_server_log(state: dict, line: str) -> None:
    if not line:
        return
    s = line.rstrip("\r\n")
    state["log_lines"].append(s)
    if len(state["log_lines"]) > 800:
        state["log_lines"] = state["log_lines"][len(state["log_lines"]) - 800 :]


def _comfy_log_text(state: dict) -> str:
    return "\n".join(state.get("log_lines") or [])


def _append_comfy_log(state: dict, line: str) -> None:
    if not line:
        return
    s = line.rstrip("\r\n")
    state["log_lines"].append(s)
    if len(state["log_lines"]) > 800:
        state["log_lines"] = state["log_lines"][len(state["log_lines"]) - 800 :]


def _is_proc_running(p) -> bool:
    try:
        return p is not None and p.poll() is None
    except Exception:
        return False


def _taskkill_tree(pid: int) -> tuple[bool, str]:
    try:
        r = subprocess.run(
            ["taskkill", "/PID", str(pid), "/T", "/F"],
            capture_output=True,
            text=True,
        )
        if r.returncode == 0:
            return True, (r.stdout or "").strip() or "taskkill ok"
        msg = (r.stdout or "").strip() + "\n" + (r.stderr or "").strip()
        return False, msg.strip() or f"taskkill failed ({r.returncode})"
    except Exception as e:
        return False, f"taskkill error: {e}"


def _read_meta_title(meta_path: str) -> str:
    try:
        with open(meta_path, "r", encoding="utf-8") as f:
            obj = json.load(f)
        title = (obj.get("title") or "").strip()
        return title
    except Exception:
        return ""


def list_output_runs(outdir: str) -> list[dict]:
    runs: list[dict] = []
    if not outdir or not os.path.isdir(outdir):
        return runs

    for name in os.listdir(outdir):
        folder = os.path.join(outdir, name)
        if not os.path.isdir(folder):
            continue

        audio = ""
        meta = ""
        for fn in ("audio.mp3", "audio.wav", "audio.flac"):
            cand = os.path.join(folder, fn)
            if os.path.isfile(cand):
                audio = cand
                break
        cand_meta = os.path.join(folder, "meta.json")
        if os.path.isfile(cand_meta):
            meta = cand_meta

        if not audio and not meta:
            continue

        try:
            mtime = os.path.getmtime(folder)
        except Exception:
            mtime = 0.0

        title = _read_meta_title(meta) if meta else ""
        runs.append(
            {
                "label": f"{name}" + (f" - {title}" if title else ""),
                "folder": folder,
                "audio": audio,
                "meta": meta,
                "mtime": mtime,
            }
        )

    runs.sort(key=lambda x: x.get("mtime", 0.0), reverse=True)
    return runs[:50]


def refresh_history(outdir: str):
    runs = list_output_runs(outdir)
    choices = [r["label"] for r in runs]
    value = choices[0] if choices else None
    # Store full run list in a hidden JSON string for selection.
    return gr.Dropdown(choices=choices, value=value), json.dumps(
        runs, ensure_ascii=True
    )


def select_history_run(selected: str, runs_json: str):
    if not selected or not runs_json:
        return "", None, None
    try:
        runs = json.loads(runs_json)
    except Exception:
        return "", None, None
    for r in runs:
        if r.get("label") == selected:
            return r.get("folder") or "", r.get("audio") or None, r.get("meta") or None
    return "", None, None


def start_run(
    topic: str,
    style: str,
    genres: list[str] | None,
    batch_count: int,
    reroll_mode: str,
    reroll_on_first: bool,
    lang: str,
    duration: float,
    audio_format: str,
    batch_size: int,
    inference_steps: int,
    thinking: bool,
    skip_format: bool,
    outdir: str,
    llm_base: str,
    llm_model: str,
    llm_api_key: str,
    acestep_api: str,
    state: dict,
):
    # Generator to stream logs.
    proc = state.get("proc")
    if proc is not None and proc.poll() is None:
        yield (
            _log_text(state),
            "Already running.",
            state.get("out_dir") or "",
            state.get("audio_path") or None,
            state.get("meta_path") or None,
            state,
            (topic or "").strip(),
            _combine_style((style or "").strip(), genres or []),
        )
        return

    # Validate basics
    seed_topic = (topic or "").strip()
    seed_style = (style or "").strip()

    genres, trim_msg = _enforce_max_genres(genres or [])
    # Reroll mode normalization
    reroll_mode = (reroll_mode or "None").strip()
    if reroll_mode not in ("None", "Topic from Style", "Style from Topic", "Both"):
        reroll_mode = "None"

    try:
        duration_f = float(duration)
    except Exception:
        duration_f = 30.0
    if duration_f > 210:
        duration_f = 210.0

    try:
        total = int(batch_count) if batch_count else 1
    except Exception:
        total = 1
    if total < 1:
        total = 1
    if total > 20:
        total = 20

    repo_root = os.path.abspath(os.path.dirname(__file__))
    main_py = os.path.join(repo_root, "main.py")

    # Preserve log_lines across batch runs.
    state = _make_runner_state()
    state["acestep_api"] = acestep_api
    state["run_started_at"] = time.time()
    if trim_msg:
        _append_log(state, trim_msg)

    cur_topic = seed_topic
    cur_style = seed_style
    if total > 1 and reroll_mode != "None":
        _append_log(
            state, f"Batch reroll: {reroll_mode} (on first: {bool(reroll_on_first)})"
        )

    for run_idx in range(1, total + 1):
        prefix = f"Batch {run_idx}/{total}" if total > 1 else "Run"

        # Optionally re-suggest topic/style for this run.
        do_reroll = reroll_mode != "None" and (reroll_on_first or run_idx > 1)
        if do_reroll:
            if reroll_mode == "Topic from Style":
                # Use current style; if empty, fall back to seed_style.
                src_style = cur_style or seed_style
                if not src_style:
                    _append_log(
                        state, f"{prefix}: Error: Style is required to suggest a topic"
                    )
                    yield (
                        _log_text(state),
                        _build_run_status(state, f"{prefix}: Error"),
                        "",
                        None,
                        None,
                        state,
                        cur_topic,
                        cur_style,
                    )
                    break
                new_topic, msg = llm_suggest_topic(
                    style=src_style,
                    genres=genres,
                    llm_base=llm_base,
                    llm_model=llm_model,
                    llm_api_key=llm_api_key,
                )
                if msg != "OK":
                    _append_log(state, f"{prefix}: Suggest topic failed: {msg}")
                    break
                cur_topic = new_topic
                _append_log(state, f"{prefix}: Suggested topic: {cur_topic}")

            elif reroll_mode == "Style from Topic":
                src_topic = cur_topic or seed_topic
                if not src_topic:
                    _append_log(
                        state, f"{prefix}: Error: Topic is required to suggest a style"
                    )
                    yield (
                        _log_text(state),
                        _build_run_status(state, f"{prefix}: Error"),
                        "",
                        None,
                        None,
                        state,
                        cur_topic,
                        cur_style,
                    )
                    break
                new_style, msg = llm_suggest_style(
                    topic=src_topic,
                    genres=genres,
                    llm_base=llm_base,
                    llm_model=llm_model,
                    llm_api_key=llm_api_key,
                )
                if msg != "OK":
                    _append_log(state, f"{prefix}: Suggest style failed: {msg}")
                    break
                cur_style = new_style
                _append_log(state, f"{prefix}: Suggested style: {cur_style}")

            elif reroll_mode == "Both":
                t_out, s_out, msg = llm_suggest_both(
                    topic=cur_topic,
                    style=cur_style,
                    genres=genres,
                    llm_base=llm_base,
                    llm_model=llm_model,
                    llm_api_key=llm_api_key,
                )
                if msg != "OK":
                    _append_log(state, f"{prefix}: Suggest both failed: {msg}")
                    break
                cur_topic = (t_out or "").strip()
                cur_style = (s_out or "").strip()
                _append_log(state, f"{prefix}: Suggested topic: {cur_topic}")
                _append_log(state, f"{prefix}: Suggested style: {cur_style}")

        # Ensure we have enough to run.
        if not (cur_topic or "").strip():
            _append_log(state, f"{prefix}: Error: Topic is required")
            break
        if not (cur_style or "").strip() and not genres:
            _append_log(
                state,
                f"{prefix}: Error: Style is required (or select at least one genre)",
            )
            break

        combined_style = _combine_style(cur_style, genres)

        # Reset run-specific fields (keep log_lines)
        state["proc"] = None
        state["queue"] = None
        state["reader"] = None
        state["audio_path"] = ""
        state["meta_path"] = ""
        state["out_dir"] = ""
        state["task_id"] = ""
        state["eta_seconds"] = None
        state["queue_position"] = None
        state["job_status"] = ""
        state["last_eta_poll_at"] = 0.0
        state["run_started_at"] = time.time()

        _append_log(state, f"{prefix}: Starting...")
        yield (
            _log_text(state),
            _build_run_status(state, f"{prefix}: Starting..."),
            "",
            None,
            None,
            state,
            cur_topic,
            combined_style,
        )

        cmd = [
            sys.executable,
            main_py,
            "--topic",
            cur_topic,
            "--style",
            combined_style,
            "--lang",
            lang,
            "--duration",
            str(duration_f),
            "--audio-format",
            audio_format,
            "--batch-size",
            str(batch_size),
            "--inference-steps",
            str(inference_steps),
            "--outdir",
            outdir,
            "--llm-base",
            llm_base,
            "--llm-model",
            llm_model,
            "--acestep-api",
            acestep_api,
        ]
        cmd.append("--thinking" if thinking else "--no-thinking")
        if skip_format:
            cmd.append("--skip-format")

        env = os.environ.copy()
        key = _llm_key_from_inputs(llm_api_key)
        if key:
            env["LLM_API_KEY"] = key

        q: queue.Queue[str | None] = queue.Queue()

        def _reader_thread(p: subprocess.Popen):
            try:
                assert p.stdout is not None
                for line in p.stdout:
                    q.put(line)
            finally:
                q.put(None)

        try:
            p = subprocess.Popen(
                cmd,
                cwd=repo_root,
                env=env,
                stdout=subprocess.PIPE,
                stderr=subprocess.STDOUT,
                text=True,
                bufsize=1,
            )
        except Exception as e:
            _append_log(state, f"{prefix}: Failed to start: {e}")
            yield (
                _log_text(state),
                f"{prefix}: Failed to start.",
                "",
                None,
                None,
                state,
                cur_topic,
                combined_style,
            )
            return

        state["proc"] = p
        state["queue"] = q
        t = threading.Thread(target=_reader_thread, args=(p,), daemon=True)
        state["reader"] = t
        t.start()

        _append_log(state, f"{prefix}: Running...")
        yield (
            _log_text(state),
            _build_run_status(state, f"{prefix}: Running..."),
            "",
            None,
            None,
            state,
            cur_topic,
            combined_style,
        )

        done = False
        while not done:
            _poll_acestep_eta(state)

            got_any = False
            while True:
                try:
                    item = q.get_nowait()
                except queue.Empty:
                    break
                if item is None:
                    done = True
                    break
                got_any = True
                _append_log(state, item)

            if got_any or done:
                audio = state.get("audio_path") or None
                meta = state.get("meta_path") or None
                out_dir = state.get("out_dir") or ""
                yield (
                    _log_text(state),
                    _build_run_status(
                        state,
                        f"{prefix}: Running..." if not done else f"{prefix}: Finished.",
                    ),
                    out_dir,
                    audio,
                    meta,
                    state,
                    cur_topic,
                    combined_style,
                )

            if not done:
                time.sleep(0.15)

        code = p.poll()
        if code == 0:
            _append_log(state, f"{prefix}: Process exit: 0")
            status = f"{prefix}: Finished (success)."
        else:
            _append_log(state, f"{prefix}: Process exit: {code}")
            status = f"{prefix}: Finished (exit {code})."

        audio = state.get("audio_path") or None
        meta = state.get("meta_path") or None
        out_dir = state.get("out_dir") or ""
        yield (
            _log_text(state),
            _build_run_status(state, status),
            out_dir,
            audio,
            meta,
            state,
            cur_topic,
            combined_style,
        )

        if code != 0:
            # Stop the remaining batch on failure.
            break


def stop_run(state: dict):
    proc = state.get("proc")
    if proc is None or proc.poll() is not None:
        return (
            _log_text(state),
            "Not running.",
            state.get("out_dir") or "",
            state.get("audio_path") or None,
            state.get("meta_path") or None,
            state,
            "",
            "",
        )

    try:
        proc.terminate()
        try:
            proc.wait(timeout=5)
        except subprocess.TimeoutExpired:
            proc.kill()
    except Exception as e:
        _append_log(state, f"Stop error: {e}")

    _append_log(state, "Stopped.")
    return (
        _log_text(state),
        _build_run_status(state, "Stopped."),
        state.get("out_dir") or "",
        state.get("audio_path") or None,
        state.get("meta_path") or None,
        state,
        "",
        "",
    )


def open_output_folder(out_dir: str) -> str:
    if not out_dir:
        return "No output folder yet."
    if not os.path.isdir(out_dir):
        return f"Folder not found: {out_dir}"
    try:
        os.startfile(out_dir)  # type: ignore[attr-defined]
        return f"Opened: {out_dir}"
    except Exception as e:
        return f"Failed to open: {e}"


def open_run_folder_from_history(folder: str) -> str:
    if not folder:
        return "No run folder selected."
    if not os.path.isdir(folder):
        return f"Folder not found: {folder}"
    try:
        os.startfile(folder)  # type: ignore[attr-defined]
        return f"Opened: {folder}"
    except Exception as e:
        return f"Failed to open: {e}"


def _find_audio_in_run(folder: str) -> str:
    for fn in ("audio.mp3", "audio.wav", "audio.flac"):
        p = os.path.join(folder, fn)
        if os.path.isfile(p):
            return p
    return ""


def render_lyric_video(
    run_folder: str,
    outdir: str,
    video_preset: str,
    lyric_offset: float,
    auto_lead_in: bool,
    bg_video_path: str,
    bg_mode: str,
    bg_crossfade_s: float,
    viz_mode: str,
    viz_opacity: float,
    bg_blur: float,
    bg_grain: float,
    bg_vignette: float,
    bg_brightness: float,
    bg_saturation: float,
    bg_contrast: float,
    plate_alpha: float,
) -> tuple[str, str]:
    """Render lyric video for the selected run folder.

    Returns: (status, video_path)
    """

    run_folder = (run_folder or "").strip()
    if not run_folder:
        return "No run selected.", ""
    if not os.path.isdir(run_folder):
        return f"Folder not found: {run_folder}", ""

    meta = os.path.join(run_folder, "meta.json")
    if not os.path.isfile(meta):
        return f"Missing meta.json: {meta}", ""
    audio = _find_audio_in_run(run_folder)
    if not audio:
        return "Missing audio file in run folder.", ""

    # Use the standalone renderer.
    repo_root = os.path.abspath(os.path.dirname(__file__))
    script = os.path.join(repo_root, "video_lyrics.py")
    if not os.path.isfile(script):
        return f"Missing: {script}", ""

    preset = (video_preset or "hd16x9").strip() or "hd16x9"
    # Place output next to the run, so it stays with the audio/meta.
    out_path = os.path.join(run_folder, f"lyrics_video_{preset}.mp4")

    bgp = (bg_video_path or "").strip()
    if not bgp:
        cand = os.path.join(run_folder, "bg_loop.mp4")
        if os.path.isfile(cand):
            bgp = cand

    cmd = [
        sys.executable,
        script,
        "--run-dir",
        run_folder,
        "--out",
        out_path,
        "--preset",
        preset,
        "--fps",
        "30",
        "--timing",
        "smart",
        "--offset",
        str(float(lyric_offset or 0.0)),
        ("--auto-lead-in" if bool(auto_lead_in) else "--no-auto-lead-in"),
        "--font",
        "C:/Windows/Fonts/arial.ttf",
    ]

    if bgp:
        cmd += ["--bg-video", bgp]

    bmode = (bg_mode or "auto").strip() or "auto"
    cmd += ["--bg-mode", bmode]
    cmd += ["--bg-crossfade", str(float(bg_crossfade_s or 0.6))]
    cmd += ["--viz", (viz_mode or "off").strip() or "off"]
    cmd += ["--viz-opacity", str(float(viz_opacity or 0.18))]
    cmd += ["--bg-blur", str(float(bg_blur or 0.0))]
    cmd += ["--bg-grain", str(float(bg_grain or 0.0))]
    cmd += ["--bg-vignette", str(float(bg_vignette or 0.0))]
    cmd += ["--bg-brightness", str(float(bg_brightness or 0.0))]
    cmd += ["--bg-saturation", str(float(bg_saturation or 1.0))]
    cmd += ["--bg-contrast", str(float(bg_contrast or 1.0))]
    cmd += ["--plate-alpha", str(float(plate_alpha or 0.0))]

    try:
        r = subprocess.run(cmd, cwd=repo_root, capture_output=True, text=True)
        if r.returncode != 0:
            msg = (r.stdout or "").strip()
            err = (r.stderr or "").strip()
            detail = (err or msg or "unknown error").strip()
            return ("Video render failed: " + detail), ""
    except FileNotFoundError:
        return (
            "Video render failed: ffmpeg not found. Install FFmpeg (add to PATH) or add tools/ffmpeg.exe.",
            "",
        )
    except Exception as e:
        return f"Video render failed: {e}", ""

    if os.path.isfile(out_path):
        return "Video saved.", out_path
    return "Video render finished but file not found.", ""


def auto_sync_lyrics(
    run_folder: str,
    use_gpu: bool,
    use_demucs: bool,
    whisper_model: str,
    words_per_segment: int,
    max_segment_s: float,
    lead_s: float,
) -> tuple[str, str]:
    """Run align_lyrics.py to compute per-line timings."""

    run_folder = (run_folder or "").strip()
    if not run_folder:
        return "No run selected.", ""
    if not os.path.isdir(run_folder):
        return f"Folder not found: {run_folder}", ""

    meta = os.path.join(run_folder, "meta.json")
    if not os.path.isfile(meta):
        return f"Missing meta.json: {meta}", ""
    audio = _find_audio_in_run(run_folder)
    if not audio:
        return "Missing audio file in run folder.", ""

    repo_root = os.path.abspath(os.path.dirname(__file__))
    script = os.path.join(repo_root, "align_lyrics.py")
    if not os.path.isfile(script):
        return f"Missing: {script}", ""

    model = (whisper_model or "small").strip() or "small"
    if model not in ("tiny", "base", "small", "medium"):
        model = "small"

    # Use embedded python (sys.executable) so it shares deps with Gradio/ACE-Step.
    cmd = [
        sys.executable,
        script,
        "--run-dir",
        run_folder,
        "--language",
        "en",
        "--mode",
        "auto",
        "--use-gpu" if bool(use_gpu) else "--no-use-gpu",
        "--demucs" if bool(use_demucs) else "--no-demucs",
        "--model",
        model,
        "--words-per-segment",
        str(int(words_per_segment or 6)),
        "--max-segment-s",
        str(float(max_segment_s or 2.2)),
        "--lead",
        str(float(lead_s or 0.0)),
    ]

    try:
        r = subprocess.run(cmd, cwd=repo_root, capture_output=True, text=True)
        if r.returncode != 0:
            msg = (r.stdout or "").strip()
            err = (r.stderr or "").strip()
            detail = (err or msg or "unknown error").strip()
            return ("Auto Sync failed: " + detail), ""
        # Prefer report file if present.
        report_path = os.path.join(run_folder, "alignment_report.json")
        if os.path.isfile(report_path):
            try:
                with open(report_path, "r", encoding="utf-8") as f:
                    rep = json.load(f)
                avg = float(rep.get("avg_coverage") or 0.0)
                low = int(rep.get("low_confidence_lines") or 0)
                return (
                    f"Auto Sync complete. avg_coverage={avg:.2f}, low_lines={low}"
                ), report_path
            except Exception:
                pass
        out = (r.stdout or "").strip()
        return (out or "Auto Sync complete."), ""
    except Exception as e:
        return f"Auto Sync failed: {e}", ""


def generate_bg_loop(
    run_folder: str,
    comfy_url: str,
    ckpt_name: str,
    loop_seconds: float,
    quality: str,
    use_llm: bool,
    llm_base: str,
    llm_model: str,
    llm_api_key: str,
) -> tuple[str, str, str]:
    """Generate bg_loop.mp4 in the selected run folder."""

    run_folder = (run_folder or "").strip()
    if not run_folder:
        return "No run selected.", "", ""
    if not os.path.isdir(run_folder):
        return f"Folder not found: {run_folder}", "", ""

    meta = os.path.join(run_folder, "meta.json")
    if not os.path.isfile(meta):
        return f"Missing meta.json: {meta}", "", ""

    repo_root = os.path.abspath(os.path.dirname(__file__))
    script = os.path.join(repo_root, "generate_bg_loop.py")
    if not os.path.isfile(script):
        return f"Missing: {script}", "", ""

    ckpt = (ckpt_name or "").strip() or "sd_xl_turbo_1.0.safetensors"
    comfy = (comfy_url or "http://127.0.0.1:8188").strip() or "http://127.0.0.1:8188"
    loop_s = float(loop_seconds or 8.0)
    out_mp4 = os.path.join(run_folder, "bg_loop.mp4")

    cmd = [
        sys.executable,
        script,
        "--run-dir",
        run_folder,
        "--comfy-url",
        comfy,
        "--ckpt",
        ckpt,
        "--quality",
        str(quality or "fast"),
        "--loop-seconds",
        f"{loop_s:.3f}",
        "--out",
        out_mp4,
        "--llm-base",
        str(llm_base),
        "--llm-model",
        str(llm_model),
        "--llm-api-key",
        str(llm_api_key),
    ]
    # Generate all scene variants so bg_mode=sections works.
    cmd += ["--scenes", "base,verse,chorus,bridge"]
    cmd += ["--use-llm" if bool(use_llm) else "--no-use-llm"]

    try:
        r = subprocess.run(cmd, cwd=repo_root, capture_output=True, text=True)
        if r.returncode != 0:
            msg = (r.stdout or "").strip()
            err = (r.stderr or "").strip()
            detail = (err or msg or "unknown error").strip()
            return ("BG loop failed: " + detail), "", ""
    except Exception as e:
        return f"BG loop failed: {e}", "", ""

    keyframe = os.path.join(run_folder, "bg_keyframe.png")
    prompt_pack = os.path.join(run_folder, "visual_prompt.json")
    if os.path.isfile(out_mp4):
        # If LLM recommended a background preset, return it in status.
        rec = ""
        if os.path.isfile(prompt_pack):
            try:
                with open(prompt_pack, "r", encoding="utf-8") as f:
                    pack = json.load(f)
                rec = str(pack.get("recommended_bg_preset") or "").strip()
            except Exception:
                rec = ""
        msg = "BG loop saved."
        if rec:
            msg += f" Recommended preset: {rec}"
        return (msg, out_mp4, (prompt_pack if os.path.isfile(prompt_pack) else ""))
    return (
        "BG loop finished but file not found.",
        "",
        (prompt_pack if os.path.isfile(prompt_pack) else ""),
    )


def start_acestep_server(acestep_api: str, acestep_dir_input: str, server_state: dict):
    # If already healthy, no-op.
    try:
        _http_get_json(acestep_api.rstrip("/") + "/health", timeout=2.0)
        return _server_log_text(server_state), "ACE-Step already running.", server_state
    except Exception:
        pass

    proc = server_state.get("proc")
    if _is_proc_running(proc):
        return (
            _server_log_text(server_state),
            "ACE-Step launch already in progress.",
            server_state,
        )

    repo_root = os.path.abspath(os.path.dirname(__file__))
    acestep_dir = _resolve_path(repo_root, acestep_dir_input or "ACE-Step-1.5")
    bat = os.path.join(acestep_dir, "start_api_server.bat")
    if not os.path.isfile(bat):
        _append_server_log(server_state, f"Missing: {bat}")
        return (
            _server_log_text(server_state),
            "start_api_server.bat not found.",
            server_state,
        )

    _append_server_log(server_state, "Starting ACE-Step server...")
    try:
        # .bat requires cmd.exe on Windows.
        p = subprocess.Popen(
            ["cmd.exe", "/c", bat],
            cwd=acestep_dir,
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            text=True,
            bufsize=1,
        )
    except Exception as e:
        _append_server_log(server_state, f"Launch error: {e}")
        return _server_log_text(server_state), "Launch failed.", server_state

    server_state["proc"] = p
    server_state["started_at"] = time.time()
    server_state["acestep_dir"] = acestep_dir

    def _reader():
        try:
            assert p.stdout is not None
            for line in p.stdout:
                _append_server_log(server_state, line)
        except Exception as e:
            _append_server_log(server_state, f"Reader error: {e}")

    threading.Thread(target=_reader, daemon=True).start()
    return _server_log_text(server_state), "ACE-Step launch started.", server_state


def start_comfyui_server(
    comfy_url: str,
    comfy_bat_input: str,
    comfy_args: str,
    comfy_state: dict,
):
    # If already reachable, no-op.
    host, port = _parse_http_host_port(comfy_url or "http://127.0.0.1:8188", 8188)
    if _tcp_is_open(host, port, timeout_s=0.5):
        return _comfy_log_text(comfy_state), "ComfyUI already reachable.", comfy_state

    proc = comfy_state.get("proc")
    if _is_proc_running(proc):
        return (
            _comfy_log_text(comfy_state),
            "ComfyUI launch already in progress.",
            comfy_state,
        )

    repo_root = os.path.abspath(os.path.dirname(__file__))
    bat = _resolve_path(repo_root, comfy_bat_input)
    if not bat or not os.path.isfile(bat):
        _append_comfy_log(comfy_state, "Missing ComfyUI .bat path.")
        if bat:
            _append_comfy_log(comfy_state, f"Missing: {bat}")
        return (
            _comfy_log_text(comfy_state),
            "ComfyUI start .bat not found.",
            comfy_state,
        )

    comfy_dir = os.path.dirname(bat)
    host, port = _parse_http_host_port(comfy_url or "http://127.0.0.1:8188", 8188)
    extra = _split_args(comfy_args)

    _append_comfy_log(comfy_state, "Starting ComfyUI...")

    # If this looks like the portable build launcher, run python directly so we
    # can append memory-saving flags.
    py = os.path.join(comfy_dir, "python_embeded", "python.exe")
    main_py = os.path.join(comfy_dir, "ComfyUI", "main.py")
    try:
        if os.path.isfile(py) and os.path.isfile(main_py):
            cmd = [
                py,
                "-s",
                main_py,
                "--windows-standalone-build",
                "--disable-auto-launch",
                "--listen",
                str(host),
                "--port",
                str(int(port)),
            ]
            cmd += extra
            _append_comfy_log(comfy_state, "Command: " + " ".join(cmd))
            p = subprocess.Popen(
                cmd,
                cwd=comfy_dir,
                stdout=subprocess.PIPE,
                stderr=subprocess.STDOUT,
                text=True,
                bufsize=1,
            )
        else:
            _append_comfy_log(comfy_state, f"Command: {bat}")
            p = subprocess.Popen(
                ["cmd.exe", "/c", bat],
                cwd=comfy_dir,
                stdout=subprocess.PIPE,
                stderr=subprocess.STDOUT,
                text=True,
                bufsize=1,
            )
    except Exception as e:
        _append_comfy_log(comfy_state, f"Launch error: {e}")
        return _comfy_log_text(comfy_state), "Launch failed.", comfy_state

    comfy_state["proc"] = p
    comfy_state["started_at"] = time.time()
    comfy_state["bat_path"] = bat

    def _reader():
        try:
            assert p.stdout is not None
            for line in p.stdout:
                _append_comfy_log(comfy_state, line)
        except Exception as e:
            _append_comfy_log(comfy_state, f"Reader error: {e}")

    threading.Thread(target=_reader, daemon=True).start()
    return _comfy_log_text(comfy_state), "ComfyUI launch started.", comfy_state


def stop_comfyui_server(comfy_state: dict):
    proc = comfy_state.get("proc")
    if not _is_proc_running(proc):
        return (
            _comfy_log_text(comfy_state),
            "No managed ComfyUI process to stop.",
            comfy_state,
        )

    pid = getattr(proc, "pid", None)
    if not pid:
        return _comfy_log_text(comfy_state), "Missing PID.", comfy_state

    ok, msg = _taskkill_tree(int(pid))
    _append_comfy_log(comfy_state, f"Stop: {msg}")
    comfy_state["proc"] = None
    return (
        _comfy_log_text(comfy_state),
        ("Stopped." if ok else "Stop attempted (see log)."),
        comfy_state,
    )


def poll_managed_logs(
    enabled_sys: bool,
    enabled_gen: bool,
    acestep_api: str,
    comfy_url: str,
    server_state: dict,
    comfy_state: dict,
):
    """Periodic UI updater for managed server logs.

    Only shows live output for processes started via the WebUI buttons.
    """

    enabled = bool(enabled_sys) or bool(enabled_gen)
    if not enabled:
        return gr.update(), gr.update(), gr.update(), gr.update()

    # ACE-Step status
    s_proc = server_state.get("proc")
    s_running = _is_proc_running(s_proc)
    s_pid = getattr(s_proc, "pid", None) if s_running else None
    host, port = _parse_http_host_port(acestep_api or "http://127.0.0.1:8001", 8001)
    s_reach = _tcp_is_open(host, port, timeout_s=0.2)
    s_status = f"Managed: {'running' if s_running else 'stopped'}"
    if s_pid:
        s_status += f" (pid {s_pid})"
    s_status += f"; API: {'reachable' if s_reach else 'down'}"

    # ComfyUI status
    c_proc = comfy_state.get("proc")
    c_running = _is_proc_running(c_proc)
    c_pid = getattr(c_proc, "pid", None) if c_running else None
    chost, cport = _parse_http_host_port(comfy_url or "http://127.0.0.1:8188", 8188)
    c_reach = _tcp_is_open(chost, cport, timeout_s=0.2)
    c_status = f"Managed: {'running' if c_running else 'stopped'}"
    if c_pid:
        c_status += f" (pid {c_pid})"
    c_status += f"; URL: {'reachable' if c_reach else 'down'}"

    return (
        _server_log_text(server_state),
        s_status,
        _comfy_log_text(comfy_state),
        c_status,
    )


def stop_acestep_server(server_state: dict):
    proc = server_state.get("proc")
    if not _is_proc_running(proc):
        return (
            _server_log_text(server_state),
            "No managed ACE-Step process to stop.",
            server_state,
        )

    pid = getattr(proc, "pid", None)
    if not pid:
        return _server_log_text(server_state), "Missing PID.", server_state

    ok, msg = _taskkill_tree(int(pid))
    _append_server_log(server_state, f"Stop: {msg}")
    server_state["proc"] = None
    return (
        _server_log_text(server_state),
        ("Stopped." if ok else "Stop attempted (see log)."),
        server_state,
    )


def main() -> int:
    _load_dotenv(".env")

    default_llm_base = os.environ.get("LLM_BASE_URL", "http://localhost:8317/v1")
    default_llm_model = os.environ.get("LLM_MODEL", "gpt-5.2")
    default_acestep_api = os.environ.get("ACESTEP_API_BASE", "http://127.0.0.1:8001")
    default_acestep_dir = os.environ.get("ACESTEP_DIR", "ACE-Step-1.5")
    default_comfy_bat = os.environ.get("COMFYUI_START_BAT", "")
    default_comfy_args = os.environ.get(
        "COMFYUI_ARGS",
        "--cache-none --disable-all-custom-nodes --normalvram",
    )
    default_bg_quality = os.environ.get("BG_QUALITY", "fast")

    css = """
    .as-btn-row { gap: 10px !important; flex-wrap: wrap !important; }
    .as-btn-row button { min-height: 42px; }
    .as-card { border-radius: 14px; }
    """.strip()

    # No auto-tab switching; results are shown alongside settings.

    with gr.Blocks(title="AceStepAuto WebUI", css=css) as demo:
        gr.Markdown("# AceStepAuto WebUI")

        state = gr.State(_make_runner_state())
        server_state = gr.State(_make_server_state())
        comfy_state = gr.State(_make_comfy_state())

        with gr.Tabs(elem_classes=["as-card"]) as tabs:
            with gr.Tab("Generate", id="generate"):
                with gr.Row():
                    # 3-column workspace:
                    # - Left: idea tools
                    # - Middle: status + run settings
                    # - Right: results (run status/logs/files/history)
                    with gr.Column(scale=4, min_width=380):
                        with gr.Group(elem_classes=["as-card"]):
                            gr.Markdown("## Idea Tools")
                            topic = gr.Textbox(label="Topic", value="first love")
                            style = gr.Textbox(label="Style", value="k-pop dance pop")
                            genres = gr.CheckboxGroup(
                                label="Genres (pick up to 5)",
                                choices=GENRES_TOP20,
                                value=[],
                            )
                            genres_note = gr.Textbox(
                                label="Genres Note", value="", interactive=False
                            )

                            with gr.Row(elem_classes=["as-btn-row"]):
                                suggest_topic_btn = gr.Button(
                                    "Suggest Topic From Style", variant="secondary"
                                )
                                suggest_style_btn = gr.Button(
                                    "Suggest Style From Topic", variant="secondary"
                                )
                                random_idea_btn = gr.Button(
                                    "Random Topic + Style", variant="primary"
                                )
                            suggest_status = gr.Textbox(
                                label="Idea Tool Status", value="", interactive=False
                            )

                        with gr.Group(elem_classes=["as-card"]):
                            gr.Markdown("## Server Logs (managed)")
                            gr.Markdown(
                                "Live output is available only for servers started from the WebUI buttons."
                            )
                            live_logs_gen = gr.Checkbox(label="Live Logs", value=True)
                            srv_status_gen = gr.Textbox(
                                label="ACE-Step Control", value="", interactive=False
                            )
                            srv_log_gen = gr.Textbox(
                                label="ACE-Step Server Log (managed)",
                                value="",
                                lines=7,
                                interactive=False,
                            )
                            comfy_status_gen = gr.Textbox(
                                label="ComfyUI Control", value="", interactive=False
                            )
                            comfy_log_gen = gr.Textbox(
                                label="ComfyUI Server Log (managed)",
                                value="",
                                lines=7,
                                interactive=False,
                            )

                    with gr.Column(scale=3, min_width=340):
                        with gr.Group(elem_classes=["as-card"]):
                            gr.Markdown("## Status")
                            with gr.Row(elem_classes=["as-btn-row"]):
                                status_btn = gr.Button(
                                    "Refresh Status", variant="secondary"
                                )
                                status_box = gr.Textbox(
                                    label="Server Status",
                                    value="",
                                    lines=2,
                                    interactive=False,
                                )

                        with gr.Group(elem_classes=["as-card"]):
                            gr.Markdown("## Server Control")
                            with gr.Row(elem_classes=["as-btn-row"]):
                                gen_srv_start = gr.Button(
                                    "Start ACE-Step", variant="secondary"
                                )
                                gen_srv_stop = gr.Button(
                                    "Stop ACE-Step", variant="stop"
                                )
                                gen_comfy_start = gr.Button(
                                    "Start ComfyUI", variant="secondary"
                                )
                                gen_comfy_stop = gr.Button(
                                    "Stop ComfyUI", variant="stop"
                                )

                        with gr.Group(elem_classes=["as-card"]):
                            gr.Markdown("## Run Settings")
                            lang = gr.Textbox(label="Language", value="en")
                            with gr.Row():
                                duration = gr.Slider(
                                    label="Duration (sec)",
                                    minimum=10,
                                    maximum=210,
                                    step=1,
                                    value=30,
                                )
                                audio_format = gr.Dropdown(
                                    label="Audio Format",
                                    choices=["mp3", "wav", "flac"],
                                    value="mp3",
                                )

                            with gr.Row():
                                batch_size = gr.Slider(
                                    label="Batch Size",
                                    minimum=1,
                                    maximum=8,
                                    step=1,
                                    value=1,
                                )
                                inference_steps = gr.Slider(
                                    label="Inference Steps",
                                    minimum=1,
                                    maximum=50,
                                    step=1,
                                    value=8,
                                )

                            with gr.Row():
                                thinking = gr.Checkbox(label="Thinking", value=False)
                                skip_format = gr.Checkbox(
                                    label="Skip /format_input", value=False
                                )

                            batch_count = gr.Slider(
                                label="Batch Count (sequential)",
                                minimum=1,
                                maximum=20,
                                step=1,
                                value=1,
                            )

                            with gr.Row():
                                reroll_mode = gr.Dropdown(
                                    label="Batch Re-suggest",
                                    choices=[
                                        "None",
                                        "Topic from Style",
                                        "Style from Topic",
                                        "Both",
                                    ],
                                    value="None",
                                )
                                reroll_on_first = gr.Checkbox(
                                    label="Re-suggest on run #1", value=False
                                )

                            with gr.Row(elem_classes=["as-btn-row"]):
                                start_btn = gr.Button("Start", variant="primary")
                                stop_btn = gr.Button("Stop", variant="stop")

                    with gr.Column(scale=4, min_width=420):
                        with gr.Group(elem_classes=["as-card"]):
                            gr.Markdown("## Results")
                            run_status = gr.Textbox(
                                label="Run Status", value="Idle.", interactive=False
                            )
                            effective_topic = gr.Textbox(
                                label="Effective Topic (current run)",
                                value="",
                                interactive=False,
                            )
                            effective_style = gr.Textbox(
                                label="Effective Style (includes genres)",
                                value="",
                                interactive=False,
                            )

                            outdir = gr.Textbox(label="Output Folder", value="output")
                            out_dir_box = gr.Textbox(
                                label="Latest Output Directory",
                                value="",
                                interactive=False,
                            )
                            with gr.Row(elem_classes=["as-btn-row"]):
                                open_latest_btn = gr.Button(
                                    "Open Latest Output Folder", variant="secondary"
                                )
                                open_latest_result = gr.Textbox(
                                    label="Open Result",
                                    value="",
                                    interactive=False,
                                )

                            with gr.Group(elem_classes=["as-card"]):
                                gr.Markdown("## Video Builder")

                                with gr.Accordion(
                                    "1) Background (optional)", open=True
                                ):
                                    with gr.Row(elem_classes=["as-btn-row"]):
                                        comfy_url = gr.Textbox(
                                            label="ComfyUI URL",
                                            value="http://127.0.0.1:8188",
                                        )

                                    with gr.Row(elem_classes=["as-btn-row"]):
                                        ckpt_name = gr.Dropdown(
                                            label="Checkpoint",
                                            choices=[
                                                "v1-5-pruned-emaonly-fp16.safetensors",
                                                "sd_xl_turbo_1.0.safetensors",
                                                "flux1-dev-fp8.safetensors",
                                                "sd3.5_large_fp8_scaled.safetensors",
                                            ],
                                            value="v1-5-pruned-emaonly-fp16.safetensors",
                                        )
                                        bg_quality = gr.Dropdown(
                                            label="Quality",
                                            choices=["fast", "balanced", "high"],
                                            value=default_bg_quality
                                            if default_bg_quality
                                            in ("fast", "balanced", "high")
                                            else "fast",
                                        )
                                        bg_loop_seconds = gr.Slider(
                                            label="Loop Seconds",
                                            minimum=4.0,
                                            maximum=12.0,
                                            step=0.5,
                                            value=8.0,
                                        )
                                        bg_use_llm = gr.Checkbox(
                                            label="Use LLM", value=True
                                        )

                                    with gr.Row(elem_classes=["as-btn-row"]):
                                        bg_btn = gr.Button(
                                            "Generate BG Loop", variant="primary"
                                        )
                                        bg_status = gr.Textbox(
                                            label="BG Status",
                                            value="",
                                            interactive=False,
                                        )

                                    with gr.Row():
                                        bg_file = gr.File(label="Background Loop (mp4)")
                                        bg_prompt_file = gr.File(
                                            label="Visual Prompt (json)"
                                        )

                                    with gr.Accordion("Playback + Overlay", open=False):
                                        bg_video_path = gr.Textbox(
                                            label="BG Video Path Override",
                                            value="",
                                            placeholder="Leave blank to use run_dir/bg_loop.mp4",
                                        )

                                        with gr.Row(elem_classes=["as-btn-row"]):
                                            bg_mode = gr.Dropdown(
                                                label="Mode",
                                                choices=[
                                                    "auto",
                                                    "single",
                                                    "sections",
                                                ],
                                                value="auto",
                                            )
                                            bg_crossfade = gr.Slider(
                                                label="Crossfade (sec)",
                                                minimum=0.2,
                                                maximum=1.5,
                                                step=0.05,
                                                value=0.6,
                                            )
                                            viz_mode = gr.Dropdown(
                                                label="Audio Viz",
                                                choices=[
                                                    "off",
                                                    "spectrum",
                                                    "waveform",
                                                ],
                                                value="spectrum",
                                            )
                                            viz_opacity = gr.Slider(
                                                label="Viz Opacity",
                                                minimum=0.0,
                                                maximum=0.6,
                                                step=0.02,
                                                value=0.18,
                                            )

                                    with gr.Accordion("Look + Preset", open=False):
                                        with gr.Row(elem_classes=["as-btn-row"]):
                                            bg_preset = gr.Dropdown(
                                                label="Preset",
                                                choices=[
                                                    "Cinematic Soft",
                                                    "Warm Film",
                                                    "Dark Club",
                                                    "Crisp",
                                                    "Custom",
                                                ],
                                                value="Cinematic Soft",
                                            )
                                            apply_bg_preset_btn = gr.Button(
                                                "Apply Recommended",
                                                variant="secondary",
                                            )
                                            apply_bg_preset_status = gr.Textbox(
                                                label="Preset Status",
                                                value="",
                                                interactive=False,
                                            )

                                        with gr.Row(elem_classes=["as-btn-row"]):
                                            bg_blur = gr.Slider(
                                                label="Blur",
                                                minimum=0.0,
                                                maximum=10.0,
                                                step=0.2,
                                                value=2.8,
                                            )
                                            bg_grain = gr.Slider(
                                                label="Grain",
                                                minimum=0.0,
                                                maximum=25.0,
                                                step=0.5,
                                                value=10.0,
                                            )
                                            bg_vignette = gr.Slider(
                                                label="Vignette",
                                                minimum=0.0,
                                                maximum=1.0,
                                                step=0.05,
                                                value=0.6,
                                            )

                                        with gr.Row(elem_classes=["as-btn-row"]):
                                            bg_brightness = gr.Slider(
                                                label="Brightness",
                                                minimum=-0.25,
                                                maximum=0.15,
                                                step=0.01,
                                                value=-0.04,
                                            )
                                            bg_saturation = gr.Slider(
                                                label="Saturation",
                                                minimum=0.6,
                                                maximum=1.6,
                                                step=0.02,
                                                value=1.08,
                                            )
                                            bg_contrast = gr.Slider(
                                                label="Contrast",
                                                minimum=0.7,
                                                maximum=1.4,
                                                step=0.02,
                                                value=1.06,
                                            )
                                            plate_alpha = gr.Slider(
                                                label="Lyric Plate",
                                                minimum=0.0,
                                                maximum=0.6,
                                                step=0.02,
                                                value=0.30,
                                            )

                                with gr.Accordion(
                                    "2) Auto Sync Lyrics (recommended)", open=False
                                ):
                                    with gr.Row(elem_classes=["as-btn-row"]):
                                        sync_use_gpu = gr.Checkbox(
                                            label="Use GPU", value=True
                                        )
                                        sync_demucs = gr.Checkbox(
                                            label="Demucs (vocals)",
                                            value=True,
                                        )
                                        sync_model = gr.Dropdown(
                                            label="Whisper Model",
                                            choices=[
                                                "tiny",
                                                "base",
                                                "small",
                                                "medium",
                                            ],
                                            value="small",
                                        )

                                    with gr.Accordion("Sync Timing", open=False):
                                        with gr.Row(elem_classes=["as-btn-row"]):
                                            sync_words = gr.Slider(
                                                label="Words/Segment",
                                                minimum=3,
                                                maximum=10,
                                                step=1,
                                                value=6,
                                            )
                                            sync_max_seg = gr.Slider(
                                                label="Max Segment (sec)",
                                                minimum=0.8,
                                                maximum=4.0,
                                                step=0.1,
                                                value=2.2,
                                            )
                                            sync_lead = gr.Slider(
                                                label="Lead (sec)",
                                                minimum=-0.6,
                                                maximum=0.6,
                                                step=0.05,
                                                value=-0.15,
                                            )

                                    with gr.Row(elem_classes=["as-btn-row"]):
                                        sync_btn = gr.Button(
                                            "Auto Sync", variant="primary"
                                        )
                                        sync_status = gr.Textbox(
                                            label="Sync Status",
                                            value="",
                                            interactive=False,
                                        )
                                    sync_report_file = gr.File(
                                        label="Sync Report (json)"
                                    )

                            with gr.Row(elem_classes=["as-btn-row"]):
                                video_preset = gr.Dropdown(
                                    label="Video Preset",
                                    choices=[
                                        "hd16x9",
                                        "uhd4k16x9",
                                        "vertical1080",
                                    ],
                                    value="hd16x9",
                                )
                                lyric_offset = gr.Slider(
                                    label="Lyric Offset (sec)",
                                    minimum=-3.0,
                                    maximum=3.0,
                                    step=0.05,
                                    value=0.0,
                                )
                                auto_lead_in = gr.Checkbox(
                                    label="Auto Lead-in", value=True
                                )
                                make_video_btn = gr.Button(
                                    "Make Lyric Video", variant="primary"
                                )
                                open_btn = gr.Button(
                                    "Open Output Folder", variant="secondary"
                                )
                                open_result = gr.Textbox(
                                    label="Open Result", value="", interactive=False
                                )

                            video_status = gr.Textbox(
                                label="Video Status", value="", interactive=False
                            )
                            video_file = gr.File(label="Lyric Video (mp4)")

                        with gr.Group(elem_classes=["as-card"]):
                            gr.Markdown("## History")
                            with gr.Row(elem_classes=["as-btn-row"]):
                                hist_refresh_btn = gr.Button(
                                    "Refresh History", variant="secondary"
                                )
                                history = gr.Dropdown(
                                    label="Recent Runs", choices=[], value=None
                                )
                                open_run_btn = gr.Button(
                                    "Open Selected Run Folder", variant="secondary"
                                )

                    runs_json = gr.Textbox(label="_runs_json", value="", visible=False)

                    with gr.Group(elem_classes=["as-card"]):
                        gr.Markdown("## Logs")
                        log_box = gr.Textbox(
                            label="Logs", value="", lines=18, interactive=False
                        )

                    with gr.Group(elem_classes=["as-card"]):
                        gr.Markdown("## Files")
                        with gr.Row():
                            audio_file = gr.File(label="Audio")
                            meta_file = gr.File(label="Meta JSON")

            with gr.Tab("System & Server", id="system"):
                with gr.Group(elem_classes=["as-card"]):
                    gr.Markdown("## Connections")
                    llm_base = gr.Textbox(label="LLM Base URL", value=default_llm_base)
                    llm_model = gr.Textbox(label="LLM Model", value=default_llm_model)
                    llm_api_key = gr.Textbox(
                        label="LLM API Key (optional)",
                        value=os.environ.get("LLM_API_KEY")
                        or os.environ.get("OPENAI_API_KEY")
                        or "",
                        type="password",
                        placeholder="If blank, uses .env / environment",
                    )
                    with gr.Row(elem_classes=["as-btn-row"]):
                        save_key_btn = gr.Button(
                            "Save Key to .env", variant="secondary"
                        )
                    save_key_result = gr.Textbox(
                        label="Key Save Result", value="", interactive=False
                    )
                    acestep_api = gr.Textbox(
                        label="ACE-Step API Base", value=default_acestep_api
                    )

                with gr.Group(elem_classes=["as-card"]):
                    gr.Markdown("## Paths")
                    acestep_dir_in = gr.Textbox(
                        label="ACE-Step Folder",
                        value=default_acestep_dir,
                        placeholder="Example: ACE-Step-1.5 or D:\\path\\to\\ACE-Step-1.5",
                    )
                    comfy_bat_in = gr.Textbox(
                        label="ComfyUI Start .bat (optional)",
                        value=default_comfy_bat,
                        placeholder="Example: D:\\ComfyUI\\run_nvidia_gpu.bat",
                    )
                    comfy_args = gr.Textbox(
                        label="ComfyUI Args (advanced)",
                        value=default_comfy_args,
                        placeholder="Example: --cache-none --disable-all-custom-nodes --normalvram",
                    )

                with gr.Row():
                    with gr.Column(scale=1, min_width=420):
                        with gr.Group(elem_classes=["as-card"]):
                            gr.Markdown("## ACE-Step Server")
                            with gr.Row(elem_classes=["as-btn-row"]):
                                srv_start_btn = gr.Button(
                                    "Start ACE-Step Server", variant="primary"
                                )
                                srv_stop_btn = gr.Button(
                                    "Stop ACE-Step Server", variant="stop"
                                )
                                live_logs = gr.Checkbox(label="Live Logs", value=True)
                            srv_status = gr.Textbox(
                                label="ACE-Step Control", value="", interactive=False
                            )
                            srv_log = gr.Textbox(
                                label="ACE-Step Server Log (managed)",
                                value="",
                                lines=10,
                                interactive=False,
                            )

                    with gr.Column(scale=1, min_width=420):
                        with gr.Group(elem_classes=["as-card"]):
                            gr.Markdown("## ComfyUI Server")
                            with gr.Row(elem_classes=["as-btn-row"]):
                                comfy_start_btn = gr.Button(
                                    "Start ComfyUI", variant="primary"
                                )
                                comfy_stop_btn = gr.Button(
                                    "Stop ComfyUI", variant="stop"
                                )
                            comfy_status = gr.Textbox(
                                label="ComfyUI Control", value="", interactive=False
                            )
                            comfy_log = gr.Textbox(
                                label="ComfyUI Server Log (managed)",
                                value="",
                                lines=10,
                                interactive=False,
                            )

        status_btn.click(
            fn=lambda llm_base, llm_api_key, acestep_api, comfy_url: _check_endpoints(
                llm_base=llm_base,
                llm_api_key=llm_api_key,
                acestep_api=acestep_api,
                comfy_url=comfy_url,
            ),
            inputs=[llm_base, llm_api_key, acestep_api, comfy_url],
            outputs=[status_box],
        )

        save_key_btn.click(
            fn=save_llm_api_key,
            inputs=[llm_api_key],
            outputs=[save_key_result],
        )

        genres.change(
            fn=_enforce_max_genres,
            inputs=[genres],
            outputs=[genres, genres_note],
        )

        suggest_topic_btn.click(
            fn=lambda style, genres, llm_base, llm_model, llm_api_key: (
                llm_suggest_topic(
                    style=style,
                    genres=genres,
                    llm_base=llm_base,
                    llm_model=llm_model,
                    llm_api_key=llm_api_key,
                )
            ),
            inputs=[style, genres, llm_base, llm_model, llm_api_key],
            outputs=[topic, suggest_status],
        )

        suggest_style_btn.click(
            fn=suggest_style_click,
            inputs=[topic, genres, llm_base, llm_model, llm_api_key],
            outputs=[style, genres, suggest_status],
        )

        random_idea_btn.click(
            fn=random_idea_click,
            inputs=[genres, llm_base, llm_model, llm_api_key],
            outputs=[topic, style, genres, suggest_status],
        )

        srv_start_btn.click(
            fn=start_acestep_server,
            inputs=[acestep_api, acestep_dir_in, server_state],
            outputs=[srv_log, srv_status, server_state],
        )

        srv_stop_btn.click(
            fn=stop_acestep_server,
            inputs=[server_state],
            outputs=[srv_log, srv_status, server_state],
        )

        comfy_start_btn.click(
            fn=start_comfyui_server,
            inputs=[comfy_url, comfy_bat_in, comfy_args, comfy_state],
            outputs=[comfy_log, comfy_status, comfy_state],
        )

        comfy_stop_btn.click(
            fn=stop_comfyui_server,
            inputs=[comfy_state],
            outputs=[comfy_log, comfy_status, comfy_state],
        )

        # Generate tab shortcuts (mirror System & Server controls).
        gen_srv_start.click(
            fn=start_acestep_server,
            inputs=[acestep_api, acestep_dir_in, server_state],
            outputs=[srv_log_gen, srv_status_gen, server_state],
        )
        gen_srv_stop.click(
            fn=stop_acestep_server,
            inputs=[server_state],
            outputs=[srv_log_gen, srv_status_gen, server_state],
        )
        gen_comfy_start.click(
            fn=start_comfyui_server,
            inputs=[comfy_url, comfy_bat_in, comfy_args, comfy_state],
            outputs=[comfy_log_gen, comfy_status_gen, comfy_state],
        )
        gen_comfy_stop.click(
            fn=stop_comfyui_server,
            inputs=[comfy_state],
            outputs=[comfy_log_gen, comfy_status_gen, comfy_state],
        )

        # Live log updater (managed processes only).
        try:
            log_timer = gr.Timer(1.0)
            log_timer.tick(
                fn=poll_managed_logs,
                inputs=[
                    live_logs,
                    live_logs_gen,
                    acestep_api,
                    comfy_url,
                    server_state,
                    comfy_state,
                ],
                outputs=[srv_log, srv_status, comfy_log, comfy_status],
            )

            # Mirror the same managed logs onto the Generate tab.
            log_timer.tick(
                fn=poll_managed_logs,
                inputs=[
                    live_logs,
                    live_logs_gen,
                    acestep_api,
                    comfy_url,
                    server_state,
                    comfy_state,
                ],
                outputs=[srv_log_gen, srv_status_gen, comfy_log_gen, comfy_status_gen],
            )
        except Exception:
            # Older Gradio may not support Timer; logs will still update on button clicks.
            pass

        hist_refresh_btn.click(
            fn=refresh_history,
            inputs=[outdir],
            outputs=[history, runs_json],
        )

        open_latest_btn.click(
            fn=open_run_folder_from_history,
            inputs=[out_dir_box],
            outputs=[open_latest_result],
        )

        sync_btn.click(
            fn=auto_sync_lyrics,
            inputs=[
                out_dir_box,
                sync_use_gpu,
                sync_demucs,
                sync_model,
                sync_words,
                sync_max_seg,
                sync_lead,
            ],
            outputs=[sync_status, sync_report_file],
        )

        history.change(
            fn=select_history_run,
            inputs=[history, runs_json],
            outputs=[out_dir_box, audio_file, meta_file],
        )

        start_btn.click(
            fn=start_run,
            inputs=[
                topic,
                style,
                genres,
                batch_count,
                reroll_mode,
                reroll_on_first,
                lang,
                duration,
                audio_format,
                batch_size,
                inference_steps,
                thinking,
                skip_format,
                outdir,
                llm_base,
                llm_model,
                llm_api_key,
                acestep_api,
                state,
            ],
            outputs=[
                log_box,
                run_status,
                out_dir_box,
                audio_file,
                meta_file,
                state,
                effective_topic,
                effective_style,
            ],
        )

        stop_btn.click(
            fn=stop_run,
            inputs=[state],
            outputs=[
                log_box,
                run_status,
                out_dir_box,
                audio_file,
                meta_file,
                state,
                effective_topic,
                effective_style,
            ],
        )

        open_btn.click(
            fn=open_output_folder,
            inputs=[out_dir_box],
            outputs=[open_result],
        )

        open_run_btn.click(
            fn=open_run_folder_from_history,
            inputs=[out_dir_box],
            outputs=[open_result],
        )

        make_video_btn.click(
            fn=render_lyric_video,
            inputs=[
                out_dir_box,
                outdir,
                video_preset,
                lyric_offset,
                auto_lead_in,
                bg_video_path,
                bg_mode,
                bg_crossfade,
                viz_mode,
                viz_opacity,
                bg_blur,
                bg_grain,
                bg_vignette,
                bg_brightness,
                bg_saturation,
                bg_contrast,
                plate_alpha,
            ],
            outputs=[video_status, video_file],
        )

        bg_btn.click(
            fn=generate_bg_loop,
            inputs=[
                out_dir_box,
                comfy_url,
                ckpt_name,
                bg_loop_seconds,
                bg_quality,
                bg_use_llm,
                llm_base,
                llm_model,
                llm_api_key,
            ],
            outputs=[bg_status, bg_file, bg_prompt_file],
        )

        bg_preset.change(
            fn=_bg_preset_updates,
            inputs=[bg_preset],
            outputs=[
                bg_blur,
                bg_grain,
                bg_vignette,
                bg_brightness,
                bg_saturation,
                bg_contrast,
                plate_alpha,
            ],
        )

        apply_bg_preset_btn.click(
            fn=apply_recommended_bg_preset,
            inputs=[out_dir_box, bg_prompt_file],
            outputs=[bg_preset, apply_bg_preset_status],
        )

        demo.load(fn=refresh_history, inputs=[outdir], outputs=[history, runs_json])

    demo.queue(default_concurrency_limit=1)
    port = _pick_port(WEBUI_HOST, WEBUI_PORT)
    url = f"http://{WEBUI_HOST}:{port}"
    if str(os.environ.get("WEBUI_OPEN_BROWSER", "")).strip() in (
        "1",
        "true",
        "yes",
        "on",
    ):
        try:
            webbrowser.open(url)
        except Exception:
            pass
    demo.launch(server_name=WEBUI_HOST, server_port=port)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
