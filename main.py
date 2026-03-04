import argparse
import datetime as _dt
import json
import os
import re
import time
import urllib.request
import urllib.error


def _load_dotenv(path: str = ".env") -> None:
    """Load simple KEY=VALUE pairs into os.environ.

    - No dependency on python-dotenv.
    - Ignores blank lines and comments (# ...).
    - Does not override existing environment variables.
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
                # Strip surrounding quotes
                if len(val) >= 2 and (
                    (val[0] == '"' and val[-1] == '"')
                    or (val[0] == "'" and val[-1] == "'")
                ):
                    val = val[1:-1]
                os.environ.setdefault(key, val)
    except FileNotFoundError:
        return


def _now_stamp() -> str:
    return _dt.datetime.now().strftime("%Y%m%d-%H%M%S")


def _slugify(text: str, max_len: int = 60) -> str:
    text = (text or "").strip().lower()
    text = re.sub(r"[^a-z0-9]+", "_", text)
    text = re.sub(r"_+", "_", text).strip("_")
    if not text:
        text = "song"
    return text[:max_len].rstrip("_")


def _http_post_json(
    url: str, payload: dict, timeout: int = 120, headers: dict[str, str] | None = None
) -> dict:
    body = json.dumps(payload).encode("utf-8")
    req = urllib.request.Request(url, data=body, method="POST")
    req.add_header("Content-Type", "application/json")
    for k, v in (headers or {}).items():
        if v:
            req.add_header(k, v)
    try:
        with urllib.request.urlopen(req, timeout=timeout) as resp:
            data = resp.read().decode("utf-8", errors="replace")
    except urllib.error.HTTPError as e:
        err_body = ""
        try:
            err_body = e.read().decode("utf-8", errors="replace")
        except Exception:
            err_body = ""
        msg = f"HTTP {getattr(e, 'code', '?')} POST {url}"
        if err_body:
            msg += f": {err_body}"
        raise RuntimeError(msg) from e
    return json.loads(data)


def _http_get_bytes(url: str, timeout: int = 300) -> bytes:
    req = urllib.request.Request(url, method="GET")
    with urllib.request.urlopen(req, timeout=timeout) as resp:
        return resp.read()


def _extract_json_object(text: str) -> dict:
    text = (text or "").strip()
    try:
        obj = json.loads(text)
        if isinstance(obj, dict):
            return obj
    except Exception:
        pass

    # Fallback: extract first {...} block
    m = re.search(r"\{.*\}", text, flags=re.DOTALL)
    if not m:
        raise ValueError("LLM did not return JSON")
    obj = json.loads(m.group(0))
    if not isinstance(obj, dict):
        raise ValueError("LLM JSON root is not an object")
    return obj


def llm_generate_song_package(
    *,
    llm_base_url: str,
    llm_model: str,
    llm_api_key: str | None,
    topic: str,
    style: str,
    lang: str,
    duration: float,
) -> dict:
    url = llm_base_url.rstrip("/") + "/chat/completions"
    system_msg = (
        "You write song packages for music generation. "
        "Return ONLY valid JSON, no markdown, no extra text. "
        "All text should be ASCII."
    )
    user_msg = (
        "Create a complete song package as JSON with keys: title, caption, lyrics.\n"
        "- title: short, 2-6 words\n"
        "- caption: 1-2 lines describing genre/mood/instruments/vocals clearly\n"
        "- lyrics: structured with tags like [Verse 1], [Chorus], [Verse 2], [Bridge], [Outro]\n"
        "Constraints:\n"
        f"- topic: {topic}\n"
        f"- style: {style}\n"
        f"- language: {lang}\n"
        f"- target duration seconds: {duration}\n"
        "- Avoid profanity. Avoid named real artists.\n"
    )
    payload = {
        "model": llm_model,
        "messages": [
            {"role": "system", "content": system_msg},
            {"role": "user", "content": user_msg},
        ],
        "temperature": 0.9,
    }
    headers = {}
    if llm_api_key:
        headers["Authorization"] = f"Bearer {llm_api_key}"
    resp = _http_post_json(url, payload, timeout=120, headers=headers)
    content = resp["choices"][0]["message"]["content"]
    obj = _extract_json_object(content)

    title = (obj.get("title") or "").strip() or "Untitled"
    caption = (obj.get("caption") or "").strip()
    lyrics = (obj.get("lyrics") or "").strip()
    if not caption or not lyrics:
        raise ValueError("LLM response missing caption/lyrics")
    return {"title": title, "caption": caption, "lyrics": lyrics}


def acestep_health(api_base: str) -> bool:
    try:
        _http_get_bytes(api_base.rstrip("/") + "/health", timeout=10)
        return True
    except Exception:
        return False


def acestep_format_input(
    *, api_base: str, caption: str, lyrics: str, lang: str, duration: float | None
) -> dict:
    url = api_base.rstrip("/") + "/format_input"
    meta: dict[str, object] = {"language": lang}
    if duration and duration > 0:
        meta["duration"] = float(duration)
    payload = {
        "prompt": caption,
        "lyrics": lyrics,
        "param_obj": json.dumps(meta, ensure_ascii=True),
    }
    resp = _http_post_json(url, payload, timeout=120)
    return resp.get("data") or {}


def acestep_release_task(
    *,
    api_base: str,
    caption: str,
    lyrics: str,
    lang: str,
    duration: float,
    audio_format: str,
    inference_steps: int,
    batch_size: int,
    thinking: bool,
) -> str:
    url = api_base.rstrip("/") + "/release_task"
    payload = {
        "prompt": caption,
        "lyrics": lyrics,
        "vocal_language": lang,
        "audio_duration": float(duration),
        "audio_format": audio_format,
        "inference_steps": int(inference_steps),
        "batch_size": int(batch_size),
        "thinking": bool(thinking),
        "use_random_seed": True,
    }
    resp = _http_post_json(url, payload, timeout=120)
    data = resp.get("data") or {}
    task_id = data.get("task_id")
    if not task_id:
        raise RuntimeError(f"ACE-Step release_task missing task_id: {resp}")
    return task_id


def acestep_wait_result(
    *, api_base: str, task_id: str, poll_sec: float = 1.0, timeout_sec: int = 3600
) -> dict:
    url = api_base.rstrip("/") + "/query_result"
    start = time.time()
    while True:
        if time.time() - start > timeout_sec:
            raise TimeoutError(f"Timed out waiting for task {task_id}")
        resp = _http_post_json(url, {"task_id_list": [task_id]}, timeout=120)
        items = resp.get("data") or []
        if not items:
            time.sleep(poll_sec)
            continue
        item = items[0]
        status = item.get("status")
        if status == 0:
            time.sleep(poll_sec)
            continue
        if status == 2:
            raise RuntimeError(f"ACE-Step task failed: {item}")
        if status != 1:
            time.sleep(poll_sec)
            continue

        result_str = item.get("result") or ""
        result_list = json.loads(result_str)
        if not isinstance(result_list, list) or not result_list:
            raise RuntimeError(f"ACE-Step result list empty: {result_list!r}")
        return {"query_item": item, "result_list": result_list}


def _join_url(base: str, path_or_url: str) -> str:
    if path_or_url.startswith("http://") or path_or_url.startswith("https://"):
        return path_or_url
    return base.rstrip("/") + "/" + path_or_url.lstrip("/")


def main() -> int:
    _load_dotenv(".env")

    p = argparse.ArgumentParser(
        description="Generate lyrics then trigger ACE-Step to render audio"
    )
    p.add_argument("--topic", required=True)
    p.add_argument("--style", required=True)
    p.add_argument("--lang", default="en")
    p.add_argument("--duration", type=float, default=45)
    p.add_argument("--audio-format", default="mp3", choices=["mp3", "wav", "flac"])
    p.add_argument("--batch-size", type=int, default=1)
    p.add_argument("--inference-steps", type=int, default=8)
    p.add_argument("--thinking", action=argparse.BooleanOptionalAction, default=True)

    p.add_argument("--outdir", default="output")
    p.add_argument(
        "--llm-base", default=os.environ.get("LLM_BASE_URL", "http://localhost:8317/v1")
    )
    p.add_argument("--llm-model", default=os.environ.get("LLM_MODEL", "gpt-5.2"))
    p.add_argument(
        "--llm-api-key",
        default=os.environ.get("LLM_API_KEY") or os.environ.get("OPENAI_API_KEY"),
        help="Optional. Sent as Authorization: Bearer <key> to the LLM server.",
    )
    p.add_argument(
        "--acestep-api",
        default=os.environ.get("ACESTEP_API_BASE", "http://127.0.0.1:8001"),
    )
    p.add_argument(
        "--skip-format", action="store_true", help="Skip ACE-Step /format_input"
    )

    args = p.parse_args()

    if args.duration and args.duration > 210:
        print("Error: --duration must be <= 210 seconds")
        print(f"- Got: {args.duration}")
        return 2

    api_base = args.acestep_api
    if not acestep_health(api_base):
        print("ACE-Step API is not reachable.")
        print(f"- Expected: {api_base}")
        print("- Start it with: ACE-Step-1.5\\start_api_server.bat")
        return 2

    os.makedirs(args.outdir, exist_ok=True)

    song = llm_generate_song_package(
        llm_base_url=args.llm_base,
        llm_model=args.llm_model,
        llm_api_key=args.llm_api_key,
        topic=args.topic,
        style=args.style,
        lang=args.lang,
        duration=args.duration,
    )
    title = song["title"]
    caption = song["caption"]
    lyrics = song["lyrics"]

    format_data = {}
    if not args.skip_format:
        try:
            format_data = acestep_format_input(
                api_base=api_base,
                caption=caption,
                lyrics=lyrics,
                lang=args.lang,
                duration=args.duration,
            )
            caption = (format_data.get("caption") or caption).strip()
            lyrics = (format_data.get("lyrics") or lyrics).strip()
        except Exception as e:
            print(f"Warning: /format_input failed, continuing without it: {e}")

    task_id = acestep_release_task(
        api_base=api_base,
        caption=caption,
        lyrics=lyrics,
        lang=args.lang,
        duration=args.duration,
        audio_format=args.audio_format,
        inference_steps=args.inference_steps,
        batch_size=args.batch_size,
        thinking=args.thinking,
    )
    print(f"Submitted task: {task_id}")

    result = acestep_wait_result(api_base=api_base, task_id=task_id, poll_sec=1.0)
    out0 = result["result_list"][0]
    file_url = out0.get("file")
    if not file_url:
        raise RuntimeError(f"ACE-Step result missing file url: {out0}")
    download_url = _join_url(api_base, file_url)

    stamp = _now_stamp()
    safe_title = _slugify(title)
    song_dir = os.path.join(args.outdir, f"{stamp}_{safe_title}")
    os.makedirs(song_dir, exist_ok=True)
    audio_path = os.path.join(song_dir, f"audio.{args.audio_format}")
    meta_path = os.path.join(song_dir, "meta.json")

    audio_bytes = _http_get_bytes(download_url, timeout=600)
    with open(audio_path, "wb") as f:
        f.write(audio_bytes)

    sidecar = {
        "title": title,
        "topic": args.topic,
        "style": args.style,
        "lang": args.lang,
        "duration": args.duration,
        "caption": caption,
        "lyrics": lyrics,
        "llm": {"base_url": args.llm_base, "model": args.llm_model},
        "acestep": {
            "api_base": api_base,
            "task_id": task_id,
            "request": {
                "audio_format": args.audio_format,
                "inference_steps": args.inference_steps,
                "batch_size": args.batch_size,
                "thinking": args.thinking,
            },
            "format_input": format_data,
            "query_item": result["query_item"],
            "result": out0,
            "download_url": download_url,
        },
        "output": {"audio": audio_path, "meta": meta_path},
        "created_at": stamp,
    }
    with open(meta_path, "w", encoding="utf-8") as f:
        json.dump(sidecar, f, ensure_ascii=True, indent=2)

    print(f"Saved audio: {audio_path}")
    print(f"Saved meta : {meta_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
