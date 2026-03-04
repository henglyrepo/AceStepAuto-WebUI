import json
import os
import queue
import subprocess
import sys
import threading
import time
import webbrowser
import urllib.error
import urllib.request

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


def _check_endpoints(*, llm_base: str, llm_api_key: str, acestep_api: str) -> str:
    lines: list[str] = []

    # ACE-Step health
    try:
        health = _http_get_json(acestep_api.rstrip("/") + "/health", timeout=5.0)
        status = (health.get("data") or {}).get("status") or "ok"
        lines.append(f"ACE-Step: OK ({status})")
    except Exception as e:
        lines.append(f"ACE-Step: NOT OK ({e})")

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
    try:
        obj = json.loads(text)
        if isinstance(obj, dict):
            return obj
    except Exception:
        pass

    import re

    m = re.search(r"\{.*\}", text, flags=re.DOTALL)
    if not m:
        raise ValueError("LLM did not return JSON")
    obj = json.loads(m.group(0))
    if not isinstance(obj, dict):
        raise ValueError("LLM JSON root is not an object")
    return obj


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
    key = _llm_key_from_inputs(llm_api_key)
    if not key:
        return "", "Missing LLM API key (set LLM_API_KEY in .env or paste it and Save)."

    genre_mix = ", ".join(genres or [])
    prompt = (
        "Return ONLY JSON with keys: topic. "
        "Topic must be short (2-8 words), modern, and ASCII.\n"
        f"Style: {style.strip()}\n"
        f"Genres: {genre_mix}\n"
    )
    url = llm_base.rstrip("/") + "/chat/completions"
    headers = {"Authorization": f"Bearer {key}"}
    payload = {
        "model": llm_model,
        "messages": [
            {
                "role": "system",
                "content": "You generate music prompts. Return only JSON.",
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
    key = _llm_key_from_inputs(llm_api_key)
    if not key:
        return "", "Missing LLM API key (set LLM_API_KEY in .env or paste it and Save)."

    genre_mix = ", ".join(genres or [])
    prompt = (
        "Return ONLY JSON with keys: style. "
        "Style must be a single short line (genre + production vibe + instruments + vocal vibe). ASCII only.\n"
        f"Topic: {topic.strip()}\n"
        f"Genres: {genre_mix}\n"
    )
    url = llm_base.rstrip("/") + "/chat/completions"
    headers = {"Authorization": f"Bearer {key}"}
    payload = {
        "model": llm_model,
        "messages": [
            {
                "role": "system",
                "content": "You generate music prompts. Return only JSON.",
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


def llm_suggest_both(
    topic: str,
    style: str,
    genres: list[str],
    llm_base: str,
    llm_model: str,
    llm_api_key: str,
) -> tuple[str, str, str]:
    key = _llm_key_from_inputs(llm_api_key)
    if not key:
        return (
            topic,
            style,
            "Missing LLM API key (set LLM_API_KEY in .env or paste it and Save).",
        )

    genre_mix = ", ".join(genres or [])
    prompt = (
        "Return ONLY JSON with keys: topic, style. ASCII only.\n"
        "Rules:\n"
        "- topic: short (2-8 words)\n"
        "- style: one line (genre + production + instruments + vocal vibe)\n"
        f"Input topic (may be blank): {topic.strip()}\n"
        f"Input style (may be blank): {style.strip()}\n"
        f"Genres: {genre_mix}\n"
    )
    url = llm_base.rstrip("/") + "/chat/completions"
    headers = {"Authorization": f"Bearer {key}"}
    payload = {
        "model": llm_model,
        "messages": [
            {
                "role": "system",
                "content": "You generate music prompts. Return only JSON.",
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


def _enforce_max_genres(genres: list[str]) -> tuple[list[str], str]:
    g = list(genres or [])
    if len(g) <= 5:
        return g, ""
    trimmed = g[:5]
    return trimmed, "Max 5 genres. Trimmed selection to first 5."


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
    genres: list[str],
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
        )
        return

    # Validate basics
    seed_topic = (topic or "").strip()
    seed_style = (style or "").strip()

    genres, trim_msg = _enforce_max_genres(genres)
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


def start_acestep_server(acestep_api: str, server_state: dict):
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
    acestep_dir = os.path.join(repo_root, "ACE-Step-1.5")
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
        p = subprocess.Popen(
            [bat],
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

    with gr.Blocks(title="AceStepAuto WebUI") as demo:
        gr.Markdown("# AceStepAuto WebUI")

        state = gr.State(_make_runner_state())
        server_state = gr.State(_make_server_state())

        with gr.Row():
            status_btn = gr.Button("Refresh Status", variant="secondary")
            status_box = gr.Textbox(
                label="Server Status", value="", lines=2, interactive=False
            )

        with gr.Row():
            srv_start_btn = gr.Button("Start ACE-Step Server", variant="primary")
            srv_stop_btn = gr.Button("Stop ACE-Step Server", variant="stop")
            srv_status = gr.Textbox(
                label="ACE-Step Control", value="", interactive=False
            )

        srv_log = gr.Textbox(
            label="ACE-Step Server Log (managed)", value="", lines=8, interactive=False
        )

        with gr.Row():
            with gr.Column(scale=2):
                topic = gr.Textbox(label="Topic", value="first love")
                style = gr.Textbox(label="Style", value="k-pop dance pop")
                genres = gr.CheckboxGroup(
                    label="Genres (pick up to 5)", choices=GENRES_TOP20, value=[]
                )
                genres_note = gr.Textbox(
                    label="Genres Note", value="", interactive=False
                )

                with gr.Row():
                    suggest_topic_btn = gr.Button(
                        "Suggest Topic From Style", variant="secondary"
                    )
                    suggest_style_btn = gr.Button(
                        "Suggest Style From Topic", variant="secondary"
                    )
                    suggest_both_btn = gr.Button("Suggest Both", variant="secondary")
                suggest_status = gr.Textbox(
                    label="Idea Tool Status", value="", interactive=False
                )

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
                        label="Batch Size", minimum=1, maximum=8, step=1, value=1
                    )
                    inference_steps = gr.Slider(
                        label="Inference Steps", minimum=1, maximum=50, step=1, value=8
                    )

                with gr.Row():
                    thinking = gr.Checkbox(label="Thinking", value=True)
                    skip_format = gr.Checkbox(label="Skip /format_input", value=False)

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

                outdir = gr.Textbox(label="Output Folder", value="output")

            with gr.Column(scale=2):
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
                save_key_btn = gr.Button("Save Key to .env", variant="secondary")
                save_key_result = gr.Textbox(
                    label="Key Save Result", value="", interactive=False
                )
                acestep_api = gr.Textbox(
                    label="ACE-Step API Base", value=default_acestep_api
                )

        with gr.Row():
            start_btn = gr.Button("Start", variant="primary")
            stop_btn = gr.Button("Stop", variant="stop")

        with gr.Row():
            run_status = gr.Textbox(
                label="Run Status", value="Idle.", interactive=False
            )
            open_btn = gr.Button("Open Output Folder", variant="secondary")
            open_result = gr.Textbox(label="Open Result", value="", interactive=False)

        out_dir_box = gr.Textbox(
            label="Latest Output Directory", value="", interactive=False
        )

        effective_topic = gr.Textbox(
            label="Effective Topic (current run)", value="", interactive=False
        )
        effective_style = gr.Textbox(
            label="Effective Style (includes genres)", value="", interactive=False
        )

        with gr.Row():
            hist_refresh_btn = gr.Button("Refresh History", variant="secondary")
            history = gr.Dropdown(label="Recent Runs", choices=[], value=None)

        runs_json = gr.Textbox(label="_runs_json", value="", visible=False)

        log_box = gr.Textbox(label="Logs", value="", lines=18, interactive=False)

        with gr.Row():
            audio_file = gr.File(label="Audio")
            meta_file = gr.File(label="Meta JSON")

        status_btn.click(
            fn=lambda llm_base, llm_api_key, acestep_api: _check_endpoints(
                llm_base=llm_base, llm_api_key=llm_api_key, acestep_api=acestep_api
            ),
            inputs=[llm_base, llm_api_key, acestep_api],
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
            fn=lambda topic, genres, llm_base, llm_model, llm_api_key: (
                llm_suggest_style(
                    topic=topic,
                    genres=genres,
                    llm_base=llm_base,
                    llm_model=llm_model,
                    llm_api_key=llm_api_key,
                )
            ),
            inputs=[topic, genres, llm_base, llm_model, llm_api_key],
            outputs=[style, suggest_status],
        )

        suggest_both_btn.click(
            fn=lambda topic, style, genres, llm_base, llm_model, llm_api_key: (
                llm_suggest_both(
                    topic=topic,
                    style=style,
                    genres=genres,
                    llm_base=llm_base,
                    llm_model=llm_model,
                    llm_api_key=llm_api_key,
                )
            ),
            inputs=[topic, style, genres, llm_base, llm_model, llm_api_key],
            outputs=[topic, style, suggest_status],
        )

        srv_start_btn.click(
            fn=start_acestep_server,
            inputs=[acestep_api, server_state],
            outputs=[srv_log, srv_status, server_state],
        )

        srv_stop_btn.click(
            fn=stop_acestep_server,
            inputs=[server_state],
            outputs=[srv_log, srv_status, server_state],
        )

        hist_refresh_btn.click(
            fn=refresh_history,
            inputs=[outdir],
            outputs=[history, runs_json],
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
