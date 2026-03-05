import argparse
import json
import math
import os
import re
import shlex
import subprocess
import tempfile


def _read_meta(meta_path: str) -> dict:
    with open(meta_path, "r", encoding="utf-8") as f:
        return json.load(f)


def _infer_paths(run_dir: str) -> tuple[str, str]:
    run_dir = os.path.abspath(run_dir)
    meta = os.path.join(run_dir, "meta.json")
    if not os.path.isfile(meta):
        raise FileNotFoundError(f"meta.json not found: {meta}")

    audio = ""
    for fn in ("audio.mp3", "audio.wav", "audio.flac"):
        cand = os.path.join(run_dir, fn)
        if os.path.isfile(cand):
            audio = cand
            break
    if not audio:
        raise FileNotFoundError(f"audio file not found in: {run_dir}")
    return meta, audio


def _slugify(s: str, max_len: int = 60) -> str:
    s = (s or "").strip().lower()
    s = re.sub(r"[^a-z0-9]+", "_", s)
    s = re.sub(r"_+", "_", s).strip("_")
    return (s or "video")[:max_len].strip("_")


def _parse_lyrics(lyrics: str) -> list[dict]:
    """Convert lyrics text into a list of events.

    Each event is one of:
    - {"type": "section", "text": "Verse 1"}
    - {"type": "line", "text": "..."}
    """

    out: list[dict] = []
    lines = (lyrics or "").replace("\r\n", "\n").replace("\r", "\n").split("\n")
    for raw in lines:
        line = (raw or "").strip()
        if not line:
            continue
        m = re.fullmatch(r"\[(.+?)\]", line)
        if m:
            out.append({"type": "section", "text": m.group(1).strip()})
        else:
            out.append({"type": "line", "text": line})
    return out


def _load_alignment_lines(run_dir: str) -> list[dict] | None:
    """Load per-line timings produced by align_lyrics.py.

    Returns a list of items in the same shape as our internal timed list:
    - {"kind": "line", "text": ..., "start": ..., "end": ..., "words": [...] (optional)}

    If no alignment exists or it's invalid, returns None.
    """

    path = os.path.join(os.path.abspath(run_dir), "alignment_lines.json")
    if not os.path.isfile(path):
        return None
    try:
        with open(path, "r", encoding="utf-8") as f:
            data = json.load(f)
    except Exception:
        return None

    lines = data.get("lines")
    if not isinstance(lines, list) or not lines:
        return None

    out: list[dict] = []
    for ln in lines:
        try:
            txt = str((ln or {}).get("text") or "").strip()
            if not txt:
                continue
            st = float((ln or {}).get("start") or 0.0)
            en = float((ln or {}).get("end") or 0.0)
            if en <= st:
                continue
            item: dict = {"kind": "line", "text": txt, "start": st, "end": en}
            # Optional word timing for true karaoke.
            w = (ln or {}).get("words")
            if isinstance(w, list) and w:
                item["words"] = w
            sec = str((ln or {}).get("section") or "").strip()
            if sec:
                item["section"] = sec
            out.append(item)
        except Exception:
            continue

    if not out:
        return None
    out.sort(key=lambda x: float(x.get("start") or 0.0))
    return out


def _find_ffmpeg() -> str:
    # Allow a local vendored exe.
    local = os.path.join(os.path.dirname(__file__), "tools", "ffmpeg.exe")
    if os.path.isfile(local):
        return local
    return "ffmpeg"


def _assign_times(
    events: list[dict], duration: float, *, start_at: float = 0.0
) -> list[dict]:
    """Assign start/end times for each displayed item.

    Heuristic (simple + robust):
    - Only line events are timed.
    - Sections are shown briefly before the next line.
    """

    try:
        dur = float(duration)
    except Exception:
        dur = 30.0
    if dur <= 0:
        dur = 30.0

    start_at = max(0.0, float(start_at or 0.0))
    # Keep a tiny tail so last fade has time.
    tail = 0.8
    usable = max(2.0, dur - start_at - tail)

    line_idxs = [i for i, e in enumerate(events) if e.get("type") == "line"]
    if not line_idxs:
        return []

    per = usable / float(len(line_idxs))
    # Clamp to something readable.
    per = max(1.1, min(4.5, per))

    timed: list[dict] = []
    t = start_at
    for i, idx in enumerate(line_idxs):
        e = events[idx]
        # If preceding event is a section, create a short section overlay.
        if idx - 1 >= 0 and events[idx - 1].get("type") == "section":
            sec_text = str(events[idx - 1].get("text") or "").strip()
            if sec_text:
                timed.append(
                    {
                        "kind": "section",
                        "text": sec_text,
                        "start": max(0.0, t - 0.05),
                        "end": min(dur, t + 0.65),
                    }
                )

        start = t
        end = min(dur, start + per)
        timed.append({"kind": "line", "text": e["text"], "start": start, "end": end})
        t = end

    # Ensure we don't exceed duration.
    for item in timed:
        item["start"] = float(max(0.0, min(dur, item["start"])))
        item["end"] = float(max(0.0, min(dur, item["end"])))
        if item["end"] <= item["start"]:
            item["end"] = min(dur, item["start"] + 0.8)
    return timed


def _section_weight(name: str) -> float:
    n = (name or "").strip().lower()
    if not n:
        return 1.0
    if "chorus" in n or "hook" in n:
        return 1.25
    if "bridge" in n:
        return 1.15
    if "pre" in n and "chorus" in n:
        return 1.1
    if "verse" in n:
        return 1.0
    if "intro" in n:
        return 0.9
    if "outro" in n:
        return 0.9
    return 1.0


def _section_pause_seconds(name: str) -> float:
    n = (name or "").strip().lower()
    if not n:
        return 0.0
    if "chorus" in n or "hook" in n:
        return 0.35
    if "bridge" in n:
        return 0.28
    if "intro" in n:
        return 0.22
    if "outro" in n:
        return 0.22
    return 0.18


def _assign_times_smart(events: list[dict], duration: float) -> list[dict]:
    """Smarter timing based on section + line length.

    - Uses section weights (chorus tends to linger slightly longer).
    - Uses words/characters to give longer lines more time.
    """

    try:
        dur = float(duration)
    except Exception:
        dur = 30.0
    if dur <= 0:
        dur = 30.0

    tail = 0.8
    usable = max(2.0, dur - tail)

    # Flatten into lines with section context.
    cur_section = ""
    lines: list[dict] = []
    for e in events or []:
        if e.get("type") == "section":
            cur_section = str(e.get("text") or "").strip()
            continue
        if e.get("type") == "line":
            txt = str(e.get("text") or "").strip()
            if txt:
                lines.append({"text": txt, "section": cur_section})

    if not lines:
        return []

    # Build a scored sequence with explicit section pauses.
    seq: list[dict] = []
    last_section = None
    for ln in lines:
        sec_name = str(ln.get("section") or "").strip()
        if sec_name and sec_name != last_section:
            pause_s = _section_pause_seconds(sec_name)
            if pause_s > 0:
                seq.append({"kind": "pause", "section": sec_name, "base": pause_s})
            last_section = sec_name

        text = ln["text"]
        words = [w for w in re.split(r"\s+", text) if w]
        wcount = len(words)
        base = 0.85
        word_part = max(0.55, min(2.8, 0.18 * float(wcount)))
        sec_w = _section_weight(sec_name)
        seq.append(
            {
                "kind": "line",
                "text": text,
                "section": sec_name,
                "base": (base + word_part) * sec_w,
            }
        )

    total = sum(float(x.get("base") or 0.0) for x in seq) or 1.0
    scale = usable / total
    min_line = 1.1
    max_line = 5.2
    min_pause = 0.10
    max_pause = 0.55

    timed: list[dict] = []
    t = 0.0
    last_section = None
    for item in seq:
        kind = item.get("kind")
        sec_name = str(item.get("section") or "").strip()

        if kind == "pause":
            if sec_name and sec_name != last_section:
                timed.append(
                    {
                        "kind": "section",
                        "text": sec_name,
                        "start": max(0.0, t - 0.05),
                        "end": min(dur, t + 0.85),
                    }
                )
                last_section = sec_name
            span = max(
                min_pause,
                min(max_pause, float(item.get("base") or 0.0) * scale),
            )
            t = min(dur, t + span)
            continue

        if kind == "line":
            if sec_name and sec_name != last_section:
                timed.append(
                    {
                        "kind": "section",
                        "text": sec_name,
                        "start": max(0.0, t - 0.05),
                        "end": min(dur, t + 0.7),
                    }
                )
                last_section = sec_name

            start = t
            span = max(1.1, min(5.2, float(item.get("base") or 0.0) * scale))
            end = min(dur, start + span)
            timed.append(
                {
                    "kind": "line",
                    "text": str(item.get("text") or ""),
                    "start": start,
                    "end": end,
                }
            )
            t = end

    for item in timed:
        item["start"] = float(max(0.0, min(dur, item["start"])))
        item["end"] = float(max(0.0, min(dur, item["end"])))
        if item["end"] <= item["start"]:
            item["end"] = min(dur, item["start"] + 0.8)
    return timed


def _escape_drawtext(s: str) -> str:
    # Escape for ffmpeg drawtext.
    # https://ffmpeg.org/ffmpeg-filters.html#drawtext-1
    s = s.replace("\\", r"\\")
    s = s.replace(":", r"\:")
    s = s.replace("'", r"\'")
    s = s.replace("%", r"\%")
    s = s.replace("\n", " ")
    return s


def _find_ffprobe() -> str:
    local = os.path.join(os.path.dirname(__file__), "tools", "ffprobe.exe")
    if os.path.isfile(local):
        return local
    return "ffprobe"


def _probe_audio_duration(audio_path: str) -> float | None:
    ffprobe = _find_ffprobe()
    try:
        r = subprocess.run(
            [
                ffprobe,
                "-v",
                "error",
                "-show_entries",
                "format=duration",
                "-of",
                "default=noprint_wrappers=1:nokey=1",
                audio_path,
            ],
            capture_output=True,
            text=True,
        )
        if r.returncode != 0:
            return None
        s = (r.stdout or "").strip()
        if not s:
            return None
        val = float(s)
        if val <= 0:
            return None
        return val
    except Exception:
        return None


def _detect_silences(
    audio_path: str, *, noise_db: int = -30, min_silence_s: float = 0.25
) -> list[tuple[float, float]]:
    """Detect silence intervals using ffmpeg silencedetect.

    Returns list of (start, end) seconds.
    """

    ffmpeg = _find_ffmpeg()
    try:
        r = subprocess.run(
            [
                ffmpeg,
                "-hide_banner",
                "-loglevel",
                "info",
                "-i",
                audio_path,
                "-af",
                f"silencedetect=noise={int(noise_db)}dB:d={float(min_silence_s):.3f}",
                "-f",
                "null",
                "-",
            ],
            capture_output=True,
            text=True,
        )
    except Exception:
        return []

    txt = (r.stderr or "") + "\n" + (r.stdout or "")
    starts: list[float] = []
    ends: list[float] = []
    for line in txt.splitlines():
        line = line.strip()
        if "silence_start:" in line:
            try:
                starts.append(
                    float(line.split("silence_start:", 1)[1].strip().split()[0])
                )
            except Exception:
                pass
        elif "silence_end:" in line:
            try:
                ends.append(float(line.split("silence_end:", 1)[1].strip().split()[0]))
            except Exception:
                pass

    # Pair in order.
    out: list[tuple[float, float]] = []
    j = 0
    for s in starts:
        while j < len(ends) and ends[j] < s:
            j += 1
        if j < len(ends):
            e = ends[j]
            if e > s:
                out.append((max(0.0, s), max(0.0, e)))
            j += 1
    return out


def _auto_lead_in(audio_path: str, duration: float) -> float:
    """Estimate when vocals start based on initial silence."""

    sil = _detect_silences(audio_path)
    if sil:
        s0, e0 = sil[0]
        if s0 <= 0.05 and e0 >= 0.25:
            return min(float(duration), float(e0))
    # Default small lead-in.
    return min(float(duration), 0.35)


def _build_between_sum(
    windows: list[tuple[float, float]], *, max_terms: int = 10
) -> str:
    """Return an ffmpeg expression sum(between(t,s,e), ...)."""

    terms: list[str] = []
    for s, e in (windows or [])[:max_terms]:
        s2 = max(0.0, float(s))
        e2 = max(0.0, float(e))
        if e2 <= s2:
            continue
        terms.append(f"between(t,{s2:.3f},{e2:.3f})")
    if not terms:
        return "0"
    if len(terms) == 1:
        return terms[0]
    return "(" + "+".join(terms) + ")"


def _section_windows_from_timed(
    timed: list[dict], duration: float
) -> list[tuple[str, float, float]]:
    """Create (section_name, start, end) windows from timed items."""

    secs = [
        (
            str(it.get("text") or "").strip(),
            float(it.get("start") or 0.0),
        )
        for it in (timed or [])
        if it.get("kind") == "section" and str(it.get("text") or "").strip()
    ]
    if not secs:
        return []

    secs.sort(key=lambda x: x[1])
    out: list[tuple[str, float, float]] = []
    for i, (name, st) in enumerate(secs):
        end = float(duration)
        if i + 1 < len(secs):
            end = min(end, float(secs[i + 1][1]))
        out.append((name, max(0.0, st), max(0.0, end)))
    return out


def _wrap_for_ass(text: str, max_cols: int = 42, max_lines: int = 3) -> str:
    words = [w for w in re.split(r"\s+", (text or "").strip()) if w]
    if not words:
        return ""
    lines: list[str] = []
    cur = ""
    for w in words:
        # Break very long tokens (no spaces) so they don't run off-screen.
        if len(w) > max_cols:
            # Flush current line
            if cur:
                lines.append(cur)
                cur = ""
            chunk = w
            while len(chunk) > max_cols:
                lines.append(chunk[:max_cols])
                chunk = chunk[max_cols:]
            if chunk:
                lines.append(chunk)
            continue
        test = (cur + " " + w).strip() if cur else w
        if len(test) <= max_cols:
            cur = test
        else:
            if cur:
                lines.append(cur)
            cur = w
    if cur:
        lines.append(cur)
    if len(lines) > max_lines:
        head = lines[: max_lines - 1]
        tail = " ".join(lines[max_lines - 1 :])
        lines = head + [tail]
    return "\\N".join(lines)


def _ass_time(t: float) -> str:
    if t < 0:
        t = 0.0
    cs = int(round(t * 100.0))
    s, cc = divmod(cs, 100)
    m, ss = divmod(s, 60)
    h, mm = divmod(m, 60)
    return f"{h}:{mm:02d}:{ss:02d}.{cc:02d}"


def _k_tag_sequence(text: str, duration_s: float) -> str:
    words = [w for w in re.split(r"\s+", (text or "").strip()) if w]
    if not words:
        return ""
    dur_cs = max(1, int(round(float(duration_s) * 100.0)))
    weights = [max(1, len(re.sub(r"[^A-Za-z0-9]", "", w))) for w in words]
    total_w = sum(weights) or 1
    parts = []
    allocated = 0
    for i, (w, wt) in enumerate(zip(words, weights)):
        if i == len(words) - 1:
            k = max(1, dur_cs - allocated)
        else:
            k = max(1, int(round(dur_cs * (wt / total_w))))
            allocated += k
        parts.append(f"{{\\k{k}}}{w}")
    return " ".join(parts)


def _k_tag_sequence_from_words(words: list[dict], duration_s: float) -> str:
    """Build an ASS {\k..} sequence using true per-word timing.

    words: list of {word,start,end}. We clamp into duration_s and allocate centiseconds.
    """

    if not words:
        return ""
    dur_cs = max(1, int(round(float(duration_s) * 100.0)))
    # Normalize into relative times.
    w0 = float(words[0].get("start") or 0.0)
    w_last_end = float(words[-1].get("end") or w0)
    span = max(0.05, float(w_last_end - w0))

    parts: list[str] = []
    allocated = 0
    for i, w in enumerate(words):
        txt = str(w.get("word") or "").strip()
        if not txt:
            continue
        ws = float(w.get("start") or w0)
        we = float(w.get("end") or ws)
        rel_s = max(0.0, min(span, ws - w0))
        rel_e = max(0.0, min(span, we - w0))
        seg = max(0.02, rel_e - rel_s)
        if i == len(words) - 1:
            k = max(1, dur_cs - allocated)
        else:
            k = max(1, int(round(dur_cs * (seg / span))))
            allocated += k
        parts.append(f"{{\\k{k}}}{txt}")
    return " ".join(parts)


def _write_ass(
    *,
    timed: list[dict],
    out_path: str,
    width: int,
    height: int,
    font_name: str,
    base_font_px: int,
) -> None:
    header = """[Script Info]
ScriptType: v4.00+
PlayResX: {w}
PlayResY: {h}
WrapStyle: 2
ScaledBorderAndShadow: yes

[V4+ Styles]
Format: Name, Fontname, Fontsize, PrimaryColour, SecondaryColour, OutlineColour, BackColour, Bold, Italic, Underline, StrikeOut, ScaleX, ScaleY, Spacing, Angle, BorderStyle, Outline, Shadow, Alignment, MarginL, MarginR, MarginV, Encoding
Style: Line,{font},{fs},&H001CFFF9,&H80FFFFFF,&HAA000000,&H00000000,0,0,0,0,100,100,0,0,1,3,1,5,90,90,60,1
Style: Section,{font},{sfs},&H40FFFFFF,&H40FFFFFF,&HAA000000,&H00000000,0,0,0,0,100,100,0,0,1,3,1,8,90,90,40,1

[Events]
Format: Layer, Start, End, Style, Name, MarginL, MarginR, MarginV, Effect, Text
""".format(
        w=int(width),
        h=int(height),
        font=str(font_name),
        fs=int(base_font_px),
        sfs=int(max(18, int(base_font_px * 0.65))),
    )

    lines: list[str] = [header]
    # Layout adapts to aspect ratio.
    x_center = int(width // 2)
    if width >= height:
        # 16:9 - keep lyrics in lower third.
        y_line = int(height * 0.74)
        y_section = int(height * 0.12)
        max_cols = 56
    else:
        # 9:16 - center-ish for phone.
        y_line = int(height * 0.62)
        y_section = int(height * 0.18)
        max_cols = 44

    for item in timed:
        kind = item.get("kind")
        start = float(item.get("start") or 0.0)
        end = float(item.get("end") or 0.0)
        if end <= start:
            continue

        if kind == "section":
            text = str(item.get("text") or "").strip()
            if not text:
                continue
            txt = _wrap_for_ass(f"[{text}]", max_cols=34, max_lines=1)
            ass = f"{{\\an8\\pos({x_center},{y_section})\\fad(120,240)}}{txt}"
            lines.append(
                f"Dialogue: 1,{_ass_time(start)},{_ass_time(end)},Section,,0,0,0,,{ass}"
            )
            continue

        if kind == "line":
            raw = str(item.get("text") or "").strip()
            if not raw:
                continue
            wrapped = _wrap_for_ass(raw, max_cols=max_cols, max_lines=3)
            if "\\N" in wrapped:
                ass = f"{{\\an5\\pos({x_center},{y_line})\\fad(160,200)}}{wrapped}"
            else:
                # Prefer true word timing if provided by align_lyrics.py.
                kara = ""
                w = item.get("words")
                if isinstance(w, list) and w:
                    kara = _k_tag_sequence_from_words(w, max(0.1, end - start))
                if not kara:
                    kara = _k_tag_sequence(raw, max(0.1, end - start))
                ass = f"{{\\an5\\pos({x_center},{y_line})\\fad(160,200)}}{kara}"
            lines.append(
                f"Dialogue: 2,{_ass_time(start)},{_ass_time(end)},Line,,0,0,0,,{ass}"
            )

    with open(out_path, "w", encoding="utf-8") as f:
        f.write("\n".join(lines))


def _run(cmd: list[str]) -> None:
    p = subprocess.run(cmd, capture_output=True, text=True)
    if p.returncode != 0:
        msg = (p.stderr or p.stdout or "").strip()
        raise RuntimeError(msg or f"Command failed: {shlex.join(cmd)}")


def render(
    *,
    run_dir: str,
    out_path: str | None,
    width: int,
    height: int,
    fps: int,
    font: str,
    timing: str,
    offset: float,
    auto_lead_in: bool,
    bg_video: str | None = None,
    bg_mode: str = "auto",
    bg_crossfade_s: float = 0.6,
    viz: str = "off",
    viz_opacity: float = 0.18,
    bg_blur: float = 2.2,
    bg_grain: float = 8.0,
    bg_vignette: float = 0.52,
    bg_brightness: float = -0.02,
    bg_saturation: float = 1.06,
    bg_contrast: float = 1.0,
    plate_alpha: float = 0.28,
) -> str:
    meta_path, audio_path = _infer_paths(run_dir)
    meta = _read_meta(meta_path)
    title = str(meta.get("title") or "Lyric Video").strip() or "Lyric Video"
    lyrics = str(meta.get("lyrics") or "").strip()
    duration = float(meta.get("duration") or 0) or 0.0
    probed = _probe_audio_duration(audio_path)
    if probed:
        duration = probed
    if duration <= 0:
        duration = 30.0

    lead_in = 0.0
    if auto_lead_in:
        lead_in = _auto_lead_in(audio_path, duration)

    events = _parse_lyrics(lyrics)
    # If alignment exists (generated by align_lyrics.py), prefer it.
    aligned = _load_alignment_lines(run_dir)
    if aligned:
        timed = aligned
        # Add section overlays based on section transitions if available.
        enriched: list[dict] = []
        last_sec = None
        for it in timed:
            sec = str(it.get("section") or "").strip()
            if sec and sec != last_sec:
                st = float(it.get("start") or 0.0)
                enriched.append(
                    {
                        "kind": "section",
                        "text": sec,
                        "start": max(0.0, st - 0.05),
                        "end": min(duration, st + 0.7),
                    }
                )
                last_sec = sec
            enriched.append(it)
        timed = enriched
    else:
        mode = (timing or "smart").strip().lower()
        if mode in ("even", "equal"):
            timed = _assign_times(events, duration, start_at=lead_in)
        else:
            # smart timing already includes section pauses; start after lead-in
            timed = _assign_times_smart(events, duration)
            if lead_in > 0.01:
                for it in timed:
                    it["start"] = float(it.get("start", 0.0)) + float(lead_in)
                    it["end"] = float(it.get("end", 0.0)) + float(lead_in)
    if not timed:
        raise RuntimeError("No lyrics lines found in meta.json")

    off = float(offset or 0.0)
    if abs(off) > 0.0001:
        for it in timed:
            it["start"] = float(it.get("start", 0.0)) + off
            it["end"] = float(it.get("end", 0.0)) + off

    # Clamp into the audio duration.
    for it in timed:
        it["start"] = float(max(0.0, min(duration, float(it.get("start") or 0.0))))
        it["end"] = float(max(0.0, min(duration, float(it.get("end") or 0.0))))
        if it["end"] <= it["start"]:
            it["end"] = min(duration, it["start"] + 0.6)

    if not out_path:
        out_path = os.path.join(run_dir, f"lyrics_{_slugify(title)}.mp4")
    out_path = os.path.abspath(out_path)

    ffmpeg = _find_ffmpeg()
    if ffmpeg.lower() == "ffmpeg" and os.path.basename(ffmpeg) == "ffmpeg":
        # Likely not on PATH on Windows.
        # Provide a clear error so user knows what to install.
        try:
            subprocess.run([ffmpeg, "-version"], capture_output=True, text=True)
        except FileNotFoundError:
            raise RuntimeError(
                "ffmpeg not found. Install FFmpeg and add it to PATH, or place tools/ffmpeg.exe in this repo."
            )

    # Background: prefer user-provided loop video if present.
    # If bg_video is None, auto-detect bg_loop.mp4 in the run folder.
    if not bg_video:
        auto_bg = os.path.join(run_dir, "bg_loop.mp4")
        if os.path.isfile(auto_bg):
            bg_video = auto_bg

    # Multi-scene background support.
    # If bg_mode is sections, prefer bg_loop_verse/chorus/bridge.mp4.
    bg_mode2 = (bg_mode or "auto").strip().lower()
    if bg_mode2 not in ("auto", "single", "sections"):
        bg_mode2 = "auto"

    bg_scene_files: dict[str, str] = {}
    if bg_mode2 in ("auto", "sections"):
        for key in ("verse", "chorus", "bridge", "base"):
            p = os.path.join(run_dir, f"bg_loop_{key}.mp4")
            if os.path.isfile(p):
                bg_scene_files[key] = p

    bpm = None
    try:
        bpm = float(
            ((meta.get("acestep") or {}).get("format_input") or {}).get("bpm") or 0
        )
    except Exception:
        bpm = None
    if not bpm or bpm <= 0:
        bpm = 120.0

    omega = 2.0 * math.pi * (float(bpm) / 60.0)

    # Lyric readability plate (darken lower third a bit).
    # Implemented as a gradient overlay using drawbox (simple + robust).
    # We apply it regardless of background source.

    if bg_video:
        # Use filter_complex pipeline with bg video input.
        bg_path = os.path.abspath(bg_video)
        if not os.path.isfile(bg_path):
            raise RuntimeError(f"Background video not found: {bg_path}")

        # ASS subtitles overlay (karaoke-ish word highlight)
        font_name = os.path.splitext(os.path.basename(font))[0] or "Arial"
        base_font_px = max(38, int(round(height * 0.085)))
        with tempfile.TemporaryDirectory(prefix="lyrics_ass_") as tmp:
            ass_path = os.path.join(tmp, "lyrics.ass")
            _write_ass(
                timed=timed,
                out_path=ass_path,
                width=width,
                height=height,
                font_name=font_name,
                base_font_px=base_font_px,
            )
            ass_ff = ass_path.replace("\\", "/")

            # Slight pulse in background contrast (subtle).
            base_con = float(bg_contrast or 1.0)
            pulse = f"{base_con:.3f}+0.020*sin({omega:.6f}*t)"

            # Scale/crop background to target size, then add grade + grain + plate.
            # Plate: drawbox over the lyric region to boost readability.
            plate_y = int(round(height * 0.60))
            plate_h = int(round(height * 0.40))
            pa = max(0.0, min(0.85, float(plate_alpha or 0.28)))

            # Optional audio visualization derived from audio.
            viz2 = (viz or "off").strip().lower()
            if viz2 not in ("off", "spectrum", "waveform"):
                viz2 = "off"

            # Background graph.
            blur = max(0.0, min(12.0, float(bg_blur or 0.0)))
            grain = max(0.0, min(40.0, float(bg_grain or 0.0)))
            vig = max(0.0, min(1.2, float(bg_vignette or 0.52)))
            bright = max(-0.4, min(0.4, float(bg_brightness or 0.0)))
            sat = max(0.0, min(2.0, float(bg_saturation or 1.0)))

            base_bg_chain = (
                f"scale={width}:{height},"
                f"crop={width}:{height},"
                + (f"gblur=sigma={blur:.2f}," if blur > 0.001 else "")
                + f"eq=brightness={bright:.3f}:saturation={sat:.3f}:contrast={pulse},"
                + (f"noise=alls={grain:.1f}:allf=t+u," if grain > 0.01 else "")
                + (
                    f"vignette=PI/{(1.0 / max(0.05, min(1.0, vig))) * 6.0:.3f},"
                    if vig > 0.001
                    else ""
                )
                + f"drawbox=x=0:y={plate_y}:w={width}:h={plate_h}:color=black@{pa:.3f}:t=fill"
            )

            # Build filter_complex depending on whether we have section clips.
            filters: list[str] = []
            inputs: list[str] = []
            # Input 0 always base bg.
            inputs.append(bg_path)

            # Add extra bg inputs for sections if available.
            bg_mode_effective = "single"
            if bg_mode2 == "sections" or (
                bg_mode2 == "auto"
                and ("chorus" in bg_scene_files or "verse" in bg_scene_files)
            ):
                if os.path.isfile(bg_scene_files.get("verse", "")) and os.path.isfile(
                    bg_scene_files.get("chorus", "")
                ):
                    bg_mode_effective = "sections"
                    # Order: base, verse, chorus, bridge(optional)
                    # We'll map: [0]=base, [1]=verse, [2]=chorus, [3]=bridge if present.
                    base_p = str(bg_scene_files.get("base") or bg_path)
                    verse_p = str(bg_scene_files.get("verse") or "")
                    chorus_p = str(bg_scene_files.get("chorus") or "")
                    inputs = [base_p, verse_p, chorus_p]
                    bridge_p = str(bg_scene_files.get("bridge") or "")
                    if bridge_p:
                        inputs.append(bridge_p)

            # Prepare background streams.
            if bg_mode_effective == "single":
                filters.append(f"[0:v]{base_bg_chain}[bg0]")
                bg_out = "[bg0]"
            else:
                # base
                filters.append(f"[0:v]{base_bg_chain}[b0]")
                # verse
                filters.append(f"[1:v]{base_bg_chain}[b1]")
                # chorus
                filters.append(f"[2:v]{base_bg_chain}[b2]")
                bg_out = "[b0]"

                # Determine approx section windows from lyrics timing events.
                # We have section markers in `timed` if alignment included sections.
                # Build simple windows: use 'chorus' markers to switch to chorus clip.
                # Fallback: keep base.
                chorus_starts = [
                    float(it.get("start") or 0.0)
                    for it in timed
                    if it.get("kind") == "section"
                    and (
                        "chorus" in str(it.get("text") or "").lower()
                        or "hook" in str(it.get("text") or "").lower()
                    )
                ]
                verse_starts = [
                    float(it.get("start") or 0.0)
                    for it in timed
                    if it.get("kind") == "section"
                    and "verse" in str(it.get("text") or "").lower()
                ]
                bridge_starts = [
                    float(it.get("start") or 0.0)
                    for it in timed
                    if it.get("kind") == "section"
                    and "bridge" in str(it.get("text") or "").lower()
                ]

                xfd = max(0.15, min(1.2, float(bg_crossfade_s or 0.6)))

                # If we have no usable section markers, don't try to build a section graph.
                # (This is common when lyrics were auto-synced from sung words and have no [Verse]/[Chorus] tags.)
                has_bridge = bool(bridge_starts and len(inputs) >= 4)
                if not verse_starts and not chorus_starts and not has_bridge:
                    # Reset to single background to avoid unconnected streams.
                    inputs = [inputs[0]]
                    filters = [f"[0:v]{base_bg_chain}[bg0]"]
                    bg_out = "[bg0]"
                else:
                    # Start from base, then crossfade to verse/chorus/bridge at section markers.
                    cur = "[b0]"
                    stage = 0
                    if verse_starts:
                        t0 = max(0.0, verse_starts[0])
                        filters.append(
                            f"{cur}[b1]xfade=transition=fade:duration={xfd:.3f}:offset={t0:.3f}[x{stage}]"
                        )
                        cur = f"[x{stage}]"
                        stage += 1
                    if chorus_starts:
                        for t0 in chorus_starts:
                            t0 = max(0.0, float(t0))
                            filters.append(
                                f"{cur}[b2]xfade=transition=fade:duration={xfd:.3f}:offset={t0:.3f}[x{stage}]"
                            )
                            cur = f"[x{stage}]"
                            stage += 1
                    if bridge_starts and len(inputs) >= 4:
                        t0 = max(0.0, bridge_starts[0])
                        # Prepare bridge stream as [b3].
                        filters.append(f"[3:v]{base_bg_chain}[b3]")
                        filters.append(
                            f"{cur}[b3]xfade=transition=fade:duration={xfd:.3f}:offset={t0:.3f}[x{stage}]"
                        )
                        cur = f"[x{stage}]"

                    bg_out = cur

            # Visualization from audio.
            overlay_chain = ""
            if viz2 != "off":
                op = max(0.0, min(0.65, float(viz_opacity or 0.18)))
                # Use input audio (which will be input index len(inputs) after we add audio).
                # We'll add audio as the last input in the actual ffmpeg command.
                # Here we reference it as [a] by using an asplit.
                if viz2 == "spectrum":
                    # colorful, subtle spectrum strip
                    filters.append(
                        f"[a]showspectrum=s={width}x{int(height * 0.20)}:mode=separate:color=intensity:scale=lin:slide=scroll:fps={fps},format=rgba,colorchannelmixer=aa={op:.3f}[viz]"
                    )
                else:
                    filters.append(
                        f"[a]showwaves=s={width}x{int(height * 0.18)}:mode=line:rate={fps}:colors=White@0.9,format=rgba,colorchannelmixer=aa={op:.3f}[viz]"
                    )
                # Place viz at bottom, behind subtitles but above background.
                viz_y = int(round(height * 0.80))
                filters.append(f"{bg_out}[viz]overlay=x=0:y={viz_y}[bgviz]")
                bg_out = "[bgviz]"

            # Title + subtitles on top.
            filters.append(
                f"{bg_out}drawtext=fontfile='{_escape_drawtext(font)}':text='{_escape_drawtext(title)}':"
                "x=(w-text_w)/2:y=h*0.14:fontsize=h*0.075:fontcolor=white@0.92:"
                "shadowcolor=black@0.55:shadowx=2:shadowy=2:"
                "enable='between(t,0,1.6)':"
                "alpha='if(lt(t,0.2),0, if(lt(t,0.55),(t-0.2)/0.35, if(lt(t,1.25),1, max(0,(1.6-t)/0.35))))'"
                + f",subtitles='{_escape_drawtext(ass_ff)}'[v]"
            )

            vf = ";".join(filters)

            cmd = [
                ffmpeg,
                "-y",
                "-hide_banner",
                "-loglevel",
                "error",
            ]

            # Add bg inputs.
            for p in inputs:
                cmd += ["-stream_loop", "-1", "-i", os.path.abspath(str(p))]
            # Audio input as last.
            cmd += ["-i", audio_path]

            # Map audio as [a] for viz filters when needed.
            # We'll always create [a] as the audio stream reference.
            if viz2 != "off":
                vf = vf.replace("[a]", f"[{len(inputs)}:a]")

            cmd += [
                "-filter_complex",
                vf,
                "-map",
                "[v]",
                "-map",
                f"{len(inputs)}:a",
                "-shortest",
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
                "-c:a",
                "aac",
                "-b:a",
                "192k",
                "-t",
                f"{duration:.3f}",
                out_path,
            ]
            _run(cmd)
            return out_path

    # Background: generated + drift + pulse (reacts to BPM if present).
    w2 = int(round(width * 1.06))
    h2 = int(round(height * 1.06))
    bg = f"color=c=#0b0d12:s={w2}x{h2}:r={fps},format=rgb24"
    bg = bg + ",noise=alls=10:allf=t+u"

    # Chorus / hook windows for extra hit effects.
    sec_windows = _section_windows_from_timed(timed, duration)
    chorus_windows: list[tuple[float, float]] = []
    chorus_hits: list[tuple[float, float]] = []
    for name, st, en in sec_windows:
        n = (name or "").lower()
        if "chorus" in n or "hook" in n:
            chorus_windows.append((st, en))
            chorus_hits.append((st, min(duration, st + 0.20)))

    chorus_active = _build_between_sum(chorus_windows)
    chorus_flash = _build_between_sum(chorus_hits)

    # Pulse via eq filter (simpler expression support than lutrgb on Windows builds).
    pulse = f"1+0.030*sin({omega:.6f}*t)"
    # eq doesn't allow boolean expressions like between(t,...) inside parameters reliably.
    # Use it only for the smooth pulse; handle chorus flash/zoom with other filters.
    bg = bg + f",eq=brightness=0.010:saturation=1.10:contrast={pulse}"

    dx = max(2, int(round((w2 - width) / 2)))
    dy = max(2, int(round((h2 - height) / 2)))

    # Camera drift crop window (keep expressions simple for lavfi parsing)
    bg = bg + (
        f",crop=w={width}:h={height}:"
        + f"x={dx} + {dx}*0.35*sin(0.21*t) + {dx}*0.15*sin(0.53*t):"
        + f"y={dy} + {dy}*0.35*sin(0.17*t) + {dy}*0.15*sin(0.47*t)"
    )
    bg = bg + ",format=yuv420p"

    filters: list[str] = [bg]

    # Title card
    title_txt = _escape_drawtext(title)
    filters.append(
        "drawtext="
        + f"fontfile='{_escape_drawtext(font)}':"
        + f"text='{title_txt}':"
        + "x=(w-text_w)/2:y=h*0.14:"
        + "fontsize=h*0.075:fontcolor=white@0.92:"
        + "shadowcolor=black@0.55:shadowx=2:shadowy=2:"
        + "enable='between(t,0,1.6)':"
        + "alpha='if(lt(t,0.2),0, if(lt(t,0.55),(t-0.2)/0.35, if(lt(t,1.25),1, max(0,(1.6-t)/0.35))))'"
    )

    # ASS subtitles overlay (karaoke-ish word highlight)
    font_name = os.path.splitext(os.path.basename(font))[0] or "Arial"
    base_font_px = max(38, int(round(height * 0.085)))
    with tempfile.TemporaryDirectory(prefix="lyrics_ass_") as tmp:
        ass_path = os.path.join(tmp, "lyrics.ass")
        _write_ass(
            timed=timed,
            out_path=ass_path,
            width=width,
            height=height,
            font_name=font_name,
            base_font_px=base_font_px,
        )
        ass_ff = ass_path.replace("\\", "/")
        filters.append(f"subtitles='{_escape_drawtext(ass_ff)}'")

        vf = ",".join(filters)

        cmd = [
            ffmpeg,
            "-y",
            "-hide_banner",
            "-loglevel",
            "error",
            "-f",
            "lavfi",
            "-i",
            vf,
            "-i",
            audio_path,
            "-shortest",
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
            "-c:a",
            "aac",
            "-b:a",
            "192k",
            "-t",
            f"{duration:.3f}",
            out_path,
        ]

        _run(cmd)
        return out_path

    raise RuntimeError("Unexpected render flow")


def main() -> int:
    p = argparse.ArgumentParser(description="Render a simple lyric video (mp4)")
    p.add_argument("--run-dir", required=True, help="Output run folder")
    p.add_argument("--out", default="", help="Output mp4 path (optional)")
    p.add_argument(
        "--preset",
        default="hd16x9",
        choices=["hd16x9", "uhd4k16x9", "vertical1080"],
        help="Video size preset",
    )
    p.add_argument("--width", type=int, default=1920)
    p.add_argument("--height", type=int, default=1080)
    p.add_argument("--fps", type=int, default=30)
    p.add_argument(
        "--timing",
        default="smart",
        choices=["smart", "even"],
        help="Lyric timing mode",
    )
    p.add_argument(
        "--offset",
        type=float,
        default=0.0,
        help="Shift lyrics by seconds (positive = later)",
    )
    p.add_argument(
        "--auto-lead-in",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Auto-detect intro silence and start lyrics after it",
    )
    p.add_argument(
        "--font",
        default="C:/Windows/Fonts/arial.ttf",
        help="Path to a .ttf font file",
    )
    p.add_argument(
        "--bg-video",
        default="",
        help="Optional looping background video (mp4). If omitted, uses run_dir/bg_loop.mp4 when present.",
    )
    p.add_argument(
        "--bg-mode",
        default="auto",
        choices=["auto", "single", "sections"],
        help="Background mode (sections uses bg_loop_verse/chorus/bridge.mp4 when present)",
    )
    p.add_argument(
        "--bg-crossfade",
        type=float,
        default=0.6,
        help="Crossfade duration for section background switches (seconds)",
    )
    p.add_argument(
        "--viz",
        default="off",
        choices=["off", "spectrum", "waveform"],
        help="Optional audio visualization overlay",
    )
    p.add_argument(
        "--viz-opacity",
        type=float,
        default=0.18,
        help="Audio visualization opacity (0..1)",
    )
    p.add_argument("--bg-blur", type=float, default=2.2, help="Background blur sigma")
    p.add_argument(
        "--bg-grain",
        type=float,
        default=8.0,
        help="Background grain amount (0..40 typical)",
    )
    p.add_argument(
        "--bg-vignette",
        type=float,
        default=0.52,
        help="Background vignette strength (0..1)",
    )
    p.add_argument(
        "--bg-brightness",
        type=float,
        default=-0.02,
        help="Background brightness (-0.4..0.4)",
    )
    p.add_argument(
        "--bg-saturation",
        type=float,
        default=1.06,
        help="Background saturation (0..2)",
    )
    p.add_argument(
        "--bg-contrast",
        type=float,
        default=1.0,
        help="Background contrast base (0..2)",
    )
    p.add_argument(
        "--plate-alpha",
        type=float,
        default=0.28,
        help="Lyric readability plate alpha (0..1)",
    )
    args = p.parse_args()

    # Apply preset unless user explicitly overrides width/height.
    # (Argparse doesn't tell us if user set it, so we treat preset as the intended size.)
    preset = str(args.preset or "hd16x9")
    if preset == "uhd4k16x9":
        args.width, args.height = 3840, 2160
    elif preset == "vertical1080":
        args.width, args.height = 1080, 1920
    else:
        args.width, args.height = 1920, 1080

    try:
        out = render(
            run_dir=args.run_dir,
            out_path=(args.out or None),
            width=int(args.width),
            height=int(args.height),
            fps=int(args.fps),
            font=str(args.font),
            timing=str(args.timing),
            offset=float(args.offset),
            auto_lead_in=bool(args.auto_lead_in),
            bg_video=(str(args.bg_video).strip() or None),
            bg_mode=str(args.bg_mode),
            bg_crossfade_s=float(args.bg_crossfade),
            viz=str(args.viz),
            viz_opacity=float(args.viz_opacity),
            bg_blur=float(args.bg_blur),
            bg_grain=float(args.bg_grain),
            bg_vignette=float(args.bg_vignette),
            bg_brightness=float(args.bg_brightness),
            bg_saturation=float(args.bg_saturation),
            bg_contrast=float(args.bg_contrast),
            plate_alpha=float(args.plate_alpha),
        )
        print(f"Saved video: {out}")
        return 0
    except Exception as e:
        print(f"Error: {e}")
        return 2


if __name__ == "__main__":
    raise SystemExit(main())
