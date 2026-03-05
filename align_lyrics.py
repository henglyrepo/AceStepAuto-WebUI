import argparse
import json
import os
import re
import subprocess
import sys
import tempfile
from dataclasses import dataclass


def _read_json(path: str) -> dict:
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def _write_json(path: str, obj: object) -> None:
    with open(path, "w", encoding="utf-8") as f:
        json.dump(obj, f, ensure_ascii=True, indent=2)


def _infer_paths(run_dir: str) -> tuple[str, str]:
    run_dir = os.path.abspath(run_dir)
    meta = os.path.join(run_dir, "meta.json")
    if not os.path.isfile(meta):
        raise FileNotFoundError(f"meta.json not found: {meta}")
    audio = ""
    for fn in ("audio.mp3", "audio.wav", "audio.flac"):
        p = os.path.join(run_dir, fn)
        if os.path.isfile(p):
            audio = p
            break
    if not audio:
        raise FileNotFoundError(f"audio file not found in: {run_dir}")
    return meta, audio


def _find_ffmpeg_exe() -> str:
    local = os.path.join(os.path.dirname(__file__), "tools", "ffmpeg.exe")
    if os.path.isfile(local):
        return local
    return "ffmpeg"


def _ensure_ffmpeg_on_path() -> None:
    ffmpeg = _find_ffmpeg_exe()
    if os.path.basename(ffmpeg).lower() == "ffmpeg.exe" and os.path.isfile(ffmpeg):
        d = os.path.dirname(os.path.abspath(ffmpeg))
        os.environ["PATH"] = d + os.pathsep + os.environ.get("PATH", "")


def _norm_word(w: str) -> str:
    s = (w or "").strip().lower()
    # Keep alphanumerics only; drop punctuation.
    s = re.sub(r"[^a-z0-9]+", "", s)
    if s in ("im",):
        return "im"
    if s in (
        "dont",
        "doesnt",
        "didnt",
        "cant",
        "wont",
        "ive",
        "youre",
        "theyre",
        "were",
        "isnt",
    ):
        return s
    return s


def _clean_asr_word(w: str) -> str:
    # faster-whisper word strings may include leading spaces.
    return str(w or "").replace("\u00a0", " ").strip()


def _tokenize(text: str) -> list[str]:
    out: list[str] = []
    for raw in re.split(r"\s+", (text or "").strip()):
        n = _norm_word(raw)
        if n:
            out.append(n)
    return out


def _strip_speaker_prefix(text: str) -> str:
    """Remove common duet speaker tags like '(F) ' or '(M) ' for alignment.

    We keep the original text for display in the video, but aligning against
    these tags hurts matching because the ASR transcript won't contain them.
    """

    s = (text or "").strip()
    # Examples:
    #   (F) City on low...
    #   (M) Backseat rhythm...
    #   (DUET) ...
    s = re.sub(r"^\(([A-Za-z]{1,6})\)\s+", "", s)
    # Colon form: F: ... / M: ...
    s = re.sub(r"^([A-Za-z]{1,6})\:\s+", "", s)
    return s.strip()


def _pick_anchor(tokens: list[str]) -> str:
    """Choose an anchor token for candidate search.

    Using the first token is brittle for common words; prefer a longer token.
    """

    stop = {
        "a",
        "an",
        "and",
        "are",
        "as",
        "at",
        "be",
        "but",
        "by",
        "for",
        "from",
        "i",
        "im",
        "in",
        "is",
        "it",
        "me",
        "my",
        "no",
        "of",
        "on",
        "or",
        "so",
        "the",
        "to",
        "up",
        "we",
        "you",
        "your",
    }
    for t in tokens:
        if len(t) >= 5 and t not in stop:
            return t
    for t in tokens:
        if len(t) >= 3 and t not in stop:
            return t
    return tokens[0] if tokens else ""


def _parse_lyrics(lyrics: str) -> list[dict]:
    out: list[dict] = []
    lines = (lyrics or "").replace("\r\n", "\n").replace("\r", "\n").split("\n")
    cur_section = ""
    for raw in lines:
        line = (raw or "").strip()
        if not line:
            continue
        m = re.fullmatch(r"\[(.+?)\]", line)
        if m:
            cur_section = m.group(1).strip()
            continue
        out.append(
            {
                "section": cur_section,
                "text": line,
                "align_text": _strip_speaker_prefix(line),
            }
        )
    return out


def _pip_install_hint(*pkgs: str) -> str:
    joined = " ".join(pkgs)
    return (
        "Missing Python dependencies. Install with:\n"
        f'  "{sys.executable}" -m pip install -U {joined}\n'
        "Then re-run Auto Sync."
    )


def _maybe_run_demucs(
    *,
    audio_path: str,
    run_dir: str,
    device: str,
    model: str,
    enabled: bool,
) -> str:
    if not enabled:
        return audio_path
    try:
        import demucs  # noqa: F401
    except Exception:
        raise RuntimeError(_pip_install_hint("demucs"))

    # Demucs writes into a structured folder; use a temp dir and then pick vocals.wav.
    with tempfile.TemporaryDirectory(prefix="demucs_") as tmp:
        cmd = [
            sys.executable,
            "-m",
            "demucs.separate",
            "--two-stems=vocals",
            "-n",
            str(model or "htdemucs"),
            "-o",
            tmp,
        ]
        # device flag differs by demucs versions; best-effort.
        if device in ("cuda", "cpu"):
            cmd += ["-d", device]
        cmd += [audio_path]

        p = subprocess.run(cmd, capture_output=True, text=True)
        if p.returncode != 0:
            msg = (p.stderr or p.stdout or "").strip()
            raise RuntimeError("Demucs failed: " + (msg or "unknown error"))

        # Find vocals.wav (Demucs layout: <tmp>/separated/<model>/<basename>/vocals.wav)
        vocals = ""
        for root, _dirs, files in os.walk(tmp):
            for fn in files:
                if fn.lower() == "vocals.wav":
                    vocals = os.path.join(root, fn)
                    break
            if vocals:
                break
        if not vocals:
            raise RuntimeError("Demucs finished but vocals.wav not found")

        # Copy into run folder so it can be reused.
        out = os.path.join(run_dir, "vocals.wav")
        try:
            # Keep it simple: use shutil.copyfile to avoid bringing in external deps.
            import shutil

            shutil.copyfile(vocals, out)
        except Exception:
            # If copy fails, just use the temp path.
            out = vocals
        return out


def _transcribe_words(
    *,
    audio_path: str,
    device: str,
    model_size: str,
    compute_type: str,
    language: str,
) -> tuple[list[dict], dict]:
    try:
        from faster_whisper import WhisperModel
    except Exception:
        raise RuntimeError(_pip_install_hint("faster-whisper"))

    # faster-whisper uses ffmpeg for decoding; ensure vendored ffmpeg is discoverable.
    _ensure_ffmpeg_on_path()

    m = WhisperModel(
        model_size_or_path=str(model_size),
        device=str(device),
        compute_type=str(compute_type),
    )
    segments, info = m.transcribe(
        audio_path,
        language=str(language or "en"),
        beam_size=5,
        vad_filter=True,
        word_timestamps=True,
    )

    words: list[dict] = []
    for seg in segments:
        # seg.words items have: start, end, word, probability
        for w in seg.words or []:
            ww = _clean_asr_word(getattr(w, "word", "") or "")
            if not ww:
                continue
            words.append(
                {
                    "word": ww,
                    "norm": _norm_word(ww),
                    "start": float(getattr(w, "start", 0.0) or 0.0),
                    "end": float(getattr(w, "end", 0.0) or 0.0),
                    "prob": float(getattr(w, "probability", 0.0) or 0.0),
                }
            )

    meta = {
        "language": getattr(info, "language", None),
        "language_probability": float(
            getattr(info, "language_probability", 0.0) or 0.0
        ),
        "duration": float(getattr(info, "duration", 0.0) or 0.0),
    }
    return words, meta


def _join_words(words: list[dict]) -> str:
    parts: list[str] = []
    for w in words or []:
        t = _clean_asr_word(w.get("word") or "")
        if not t:
            continue
        parts.append(t)
    return " ".join(parts).strip()


@dataclass
class _Match:
    start_idx: int
    end_idx: int
    matched: list[tuple[int, int]]  # (lyric_token_index, transcript_word_index)
    score: float


def _align_line_to_transcript(
    *,
    line_tokens: list[str],
    transcript_norm: list[str],
    start_search: int,
    max_search: int,
    extra_window: int,
) -> _Match | None:
    if not line_tokens:
        return None

    n = len(transcript_norm)
    lo = max(0, int(start_search))
    hi = min(n, lo + int(max_search))
    if hi - lo <= 1:
        return None

    first = _pick_anchor(line_tokens)
    cand: list[int] = []
    for i in range(lo, hi):
        if transcript_norm[i] == first:
            cand.append(i)
            if len(cand) >= 80:
                break
    if not cand:
        # Fallback: scan with a stride for best local overlap (cheap).
        stride = 3
        for i in range(lo, hi, stride):
            cand.append(i)
            if len(cand) >= 80:
                break

    best: _Match | None = None
    for pos in cand:
        win_end = min(n, pos + len(line_tokens) + int(extra_window))
        window = transcript_norm[pos:win_end]
        if not window:
            continue

        # DP alignment: lyric tokens vs window tokens.
        # Score: match +2, mismatch -1, gaps -1.
        a = line_tokens
        b = window
        la = len(a)
        lb = len(b)
        dp = [[0] * (lb + 1) for _ in range(la + 1)]
        bt = [[0] * (lb + 1) for _ in range(la + 1)]
        # bt: 1=diag, 2=up (gap in b), 3=left (gap in a)
        for i in range(1, la + 1):
            dp[i][0] = dp[i - 1][0] - 1
            bt[i][0] = 2
        for j in range(1, lb + 1):
            dp[0][j] = dp[0][j - 1] - 1
            bt[0][j] = 3
        for i in range(1, la + 1):
            ai = a[i - 1]
            for j in range(1, lb + 1):
                bj = b[j - 1]
                s_diag = dp[i - 1][j - 1] + (2 if ai == bj else -1)
                s_up = dp[i - 1][j] - 1
                s_left = dp[i][j - 1] - 1
                if s_diag >= s_up and s_diag >= s_left:
                    dp[i][j] = s_diag
                    bt[i][j] = 1
                elif s_up >= s_left:
                    dp[i][j] = s_up
                    bt[i][j] = 2
                else:
                    dp[i][j] = s_left
                    bt[i][j] = 3

        # Backtrack.
        i, j = la, lb
        matched: list[tuple[int, int]] = []
        while i > 0 or j > 0:
            step = bt[i][j]
            if step == 1:
                if i > 0 and j > 0 and a[i - 1] == b[j - 1]:
                    matched.append((i - 1, pos + (j - 1)))
                i -= 1
                j -= 1
            elif step == 2:
                i -= 1
            else:
                j -= 1
        matched.reverse()
        if not matched:
            continue

        start_idx = matched[0][1]
        end_idx = matched[-1][1]
        score = float(dp[la][lb])
        m = _Match(
            start_idx=int(start_idx),
            end_idx=int(end_idx),
            matched=matched,
            score=score,
        )
        if not best or m.score > best.score:
            best = m

    return best


def _best_contiguous_cluster(
    *,
    words: list[dict],
    uniq_word_idxs: list[int],
    max_gap_s: float,
) -> list[int]:
    """Pick the best contiguous time cluster from matched word indices.

    When alignment produces a false late match (e.g., a common word much later),
    it can stretch a line across many seconds. We keep the largest cluster where
    the time gap between consecutive words is small.
    """

    if not uniq_word_idxs:
        return []
    idxs = sorted(int(i) for i in uniq_word_idxs)
    clusters: list[list[int]] = []
    cur: list[int] = [idxs[0]]
    for a, b in zip(idxs, idxs[1:]):
        ta = float(words[a].get("end") or words[a].get("start") or 0.0)
        tb = float(words[b].get("start") or 0.0)
        if (tb - ta) <= float(max_gap_s):
            cur.append(b)
        else:
            clusters.append(cur)
            cur = [b]
    clusters.append(cur)

    best = clusters[0]
    for c in clusters[1:]:
        if len(c) > len(best):
            best = c
        elif len(c) == len(best):
            bs = float(words[best[0]].get("start") or 0.0)
            be = float(words[best[-1]].get("end") or 0.0)
            cs = float(words[c[0]].get("start") or 0.0)
            ce = float(words[c[-1]].get("end") or 0.0)
            if (ce - cs) < (be - bs):
                best = c
    return best


def _min_line_coverage_from_match(
    matched: list[tuple[int, int]], line_len: int
) -> float:
    if not matched or line_len <= 0:
        return 0.0
    covered = len(set(int(i) for i, _j in matched))
    return float(covered / max(1, int(line_len)))


def _segment_asr_words(
    *,
    words: list[dict],
    words_per_segment: int,
    max_segment_s: float,
    min_prob: float,
    break_gap_s: float,
) -> list[dict]:
    """Create readable segments directly from the ASR words."""

    out: list[dict] = []
    buf: list[dict] = []

    def flush() -> None:
        nonlocal buf
        if not buf:
            return
        st = float(buf[0].get("start") or 0.0)
        en = float(buf[-1].get("end") or st)
        if en <= st:
            en = st + 0.6
        text = _join_words(buf)
        probs = [float(w.get("prob") or 0.0) for w in buf]
        conf = float(sum(probs) / max(1, len(probs)))
        out.append(
            {
                "section": "",
                "text": text,
                "start": st,
                "end": en,
                "coverage": 1.0,
                "confidence": conf,
                "words": [
                    {
                        "word": _clean_asr_word(w.get("word") or ""),
                        "start": float(w.get("start") or 0.0),
                        "end": float(w.get("end") or 0.0),
                        "prob": float(w.get("prob") or 0.0),
                    }
                    for w in buf
                ],
                "source": "sung",
            }
        )
        buf = []

    wps = max(3, int(words_per_segment or 6))
    max_s = max(0.8, float(max_segment_s or 2.2))
    last_end = None
    for w in words or []:
        if str(w.get("norm") or "") == "":
            continue
        if float(w.get("prob") or 0.0) < float(min_prob):
            continue
        st = float(w.get("start") or 0.0)
        en = float(w.get("end") or st)
        if last_end is not None and (st - float(last_end)) > float(break_gap_s):
            flush()
        buf.append(w)
        last_end = en
        span = float((buf[-1].get("end") or 0.0) - (buf[0].get("start") or 0.0))
        if len(buf) >= wps or span >= max_s:
            flush()
    flush()
    return out


def _estimate_line_duration_s(tokens: list[str]) -> float:
    w = len(tokens)
    # Conservative defaults for pop-ish vocals.
    return max(1.0, min(6.0, 0.32 * float(w) + 0.35))


def align_run(
    *,
    run_dir: str,
    device: str,
    use_gpu: bool,
    model_size: str,
    demucs: bool,
    demucs_model: str,
    compute_type: str,
    min_coverage: float,
    language: str,
    mode: str = "auto",
    words_per_segment: int = 6,
    max_segment_s: float = 2.2,
    lead_s: float = -0.15,
    auto_fallback_coverage: float = 0.25,
) -> dict:
    meta_path, audio_path = _infer_paths(run_dir)
    meta = _read_json(meta_path)
    lyrics = str(meta.get("lyrics") or "")
    title = str(meta.get("title") or "").strip()

    # Resolve device.
    dev = (device or "auto").strip().lower()
    if dev == "auto":
        try:
            import torch

            dev = "cuda" if bool(use_gpu) and torch.cuda.is_available() else "cpu"
        except Exception:
            dev = "cuda" if bool(use_gpu) else "cpu"
    if dev not in ("cuda", "cpu"):
        dev = "cuda" if bool(use_gpu) else "cpu"

    ct = (compute_type or "auto").strip().lower()
    if ct == "auto":
        ct = "float16" if dev == "cuda" else "int8"
    if dev == "cuda" and ct not in ("float16", "float32"):
        ct = "float16"
    if dev == "cpu" and ct not in ("int8", "int8_float16", "float32"):
        ct = "int8"

    # Optional vocal isolation.
    audio_for_asr = _maybe_run_demucs(
        audio_path=audio_path,
        run_dir=os.path.abspath(run_dir),
        device=dev,
        model=str(demucs_model or "htdemucs"),
        enabled=bool(demucs),
    )

    words, info = _transcribe_words(
        audio_path=audio_for_asr,
        device=dev,
        model_size=str(model_size or "small"),
        compute_type=str(ct),
        language=str(language or "en"),
    )

    duration = (
        float(info.get("duration") or 0.0) or float(meta.get("duration") or 0.0) or 0.0
    )
    if duration <= 0 and words:
        duration = float(max(w.get("end", 0.0) for w in words))
    if duration <= 0:
        duration = 30.0

    _write_json(
        os.path.join(run_dir, "alignment_words.json"), {"words": words, "info": info}
    )

    # Prepare transcript norms.
    transcript_norm = [str(w.get("norm") or "") for w in words]

    mode2 = (mode or "auto").strip().lower()
    if mode2 not in ("auto", "provided", "sung"):
        mode2 = "auto"

    # Sung words mode: generate segments directly from ASR transcript.
    if mode2 == "sung":
        segs = _segment_asr_words(
            words=words,
            words_per_segment=int(words_per_segment or 6),
            max_segment_s=float(max_segment_s or 2.2),
            min_prob=0.15,
            break_gap_s=0.95,
        )
        ls = float(lead_s or 0.0)
        for it in segs:
            it["start"] = float(it.get("start") or 0.0) + ls
            it["end"] = float(it.get("end") or 0.0) + ls
            it["start"] = float(max(0.0, min(duration, it["start"])))
            it["end"] = float(max(0.0, min(duration, it["end"])))
            if it["end"] <= it["start"]:
                it["end"] = min(duration, it["start"] + 0.6)

        report = {
            "title": title,
            "duration": float(duration),
            "avg_coverage": 1.0,
            "min_coverage_threshold": float(min_coverage),
            "low_confidence_lines": 0,
            "worst_lines": [],
            "notes": [
                "Using Sung Words mode: subtitles are generated from ASR transcript.",
                "This matches the singing even if meta.json lyrics differ.",
            ],
        }
        out = {
            "version": 2,
            "run_dir": os.path.abspath(run_dir),
            "audio": os.path.basename(audio_path),
            "audio_for_asr": os.path.basename(audio_for_asr),
            "duration": float(duration),
            "language": str(language or "en"),
            "mode": "sung",
            "segmentation": {
                "words_per_segment": int(words_per_segment or 6),
                "max_segment_s": float(max_segment_s or 2.2),
                "lead_s": float(lead_s or 0.0),
            },
            "model": {
                "name": str(model_size or "small"),
                "device": str(dev),
                "compute_type": str(ct),
                "demucs": bool(demucs),
                "demucs_model": str(demucs_model or "htdemucs"),
            },
            "lines": [{"index": i, **ln} for i, ln in enumerate(segs)],
            "report": report,
        }
        _write_json(os.path.join(run_dir, "alignment_lines.json"), out)
        _write_json(os.path.join(run_dir, "alignment_report.json"), report)
        return report

    lyric_lines = _parse_lyrics(lyrics)
    aligned_lines: list[dict] = []

    # Alignment parameters.
    max_search = 1200  # words
    extra_window = 10
    pointer = 0
    prev_end = 0.0

    max_gap_s = 1.1
    max_span_per_token = 0.65
    max_span_base = 0.9

    for idx, ln in enumerate(lyric_lines):
        raw = str(ln.get("text") or "").strip()
        sec = str(ln.get("section") or "").strip()
        align_text = str(ln.get("align_text") or raw)
        tokens_full = _tokenize(align_text)
        if not tokens_full:
            continue

        wps = max(3, int(words_per_segment or 6))
        chunks: list[list[str]] = [
            tokens_full[i : i + wps] for i in range(0, len(tokens_full), wps)
        ]
        if not chunks:
            continue

        for chunk_i, tokens in enumerate(chunks):
            if float(prev_end) >= float(duration) - 0.25:
                break
            if not tokens:
                continue

            # Find approximate pointer based on time.
            # Use previous segment end as an anchor.
            while pointer < len(words) and float(words[pointer].get("start") or 0.0) < (
                prev_end - 0.2
            ):
                pointer += 1

            m = _align_line_to_transcript(
                line_tokens=tokens,
                transcript_norm=transcript_norm,
                start_search=pointer,
                max_search=max_search,
                extra_window=extra_window,
            )

            line_start = None
            line_end = None
            matched_words: list[dict] = []
            coverage = 0.0
            conf = 0.0

            if m and m.matched:
                matched_word_idxs = [j for _i, j in m.matched]
                uniq0 = sorted(set(int(x) for x in matched_word_idxs))
                uniq = _best_contiguous_cluster(
                    words=words, uniq_word_idxs=uniq0, max_gap_s=max_gap_s
                )
                coverage = _min_line_coverage_from_match(m.matched, len(tokens))
                if uniq:
                    line_start = float(words[uniq[0]].get("start") or 0.0)
                    line_end = float(words[uniq[-1]].get("end") or 0.0)
                    max_span = float(max_span_base + max_span_per_token * len(tokens))
                    if (line_end - line_start) > max_span:
                        line_start = None
                        line_end = None
                        matched_words = []
                    # Reject single-word weak matches for multi-word chunks.
                    elif len(uniq) < 2 and len(tokens) >= 3:
                        line_start = None
                        line_end = None
                        matched_words = []
                    else:
                        matched_words = [
                            {
                                "word": _clean_asr_word(words[k].get("word") or ""),
                                "start": float(words[k].get("start") or 0.0),
                                "end": float(words[k].get("end") or 0.0),
                                "prob": float(words[k].get("prob") or 0.0),
                            }
                            for k in uniq
                        ]
                        probs = [float(words[k].get("prob") or 0.0) for k in uniq]
                        conf = float(sum(probs) / max(1, len(probs)))

                pointer = max(pointer, int(m.end_idx) + 1)

            # If low coverage or no match, fall back to monotonic timing.
            if line_start is None or line_end is None or coverage < float(min_coverage):
                line_start = float(prev_end)
                est = min(
                    _estimate_line_duration_s(tokens), float(max_segment_s or 2.2)
                )
                line_end = float(min(duration, line_start + max(0.7, float(est))))
                matched_words = []
                if coverage < float(min_coverage):
                    conf = float(conf or 0.0)

            # Apply lead (negative shows earlier).
            line_start = float(line_start) + float(lead_s or 0.0)
            line_end = float(line_end) + float(lead_s or 0.0)

            # Clamp and ensure ordering.
            line_start = float(max(0.0, min(duration, line_start)))
            line_end = float(max(0.0, min(duration, line_end)))
            if line_end <= line_start:
                line_end = float(min(duration, line_start + 0.8))

            # Display text: first chunk keeps original, subsequent chunks show just chunk words.
            disp = raw if chunk_i == 0 else (" ".join(tokens))

            aligned_lines.append(
                {
                    "index": int(len(aligned_lines)),
                    "section": sec,
                    "text": disp,
                    "start": float(line_start),
                    "end": float(line_end),
                    "coverage": float(coverage),
                    "confidence": float(conf),
                    "words": matched_words,
                    "source": "provided",
                }
            )
            prev_end = float(line_end)

        if float(prev_end) >= float(duration) - 0.25:
            break

    # Report.
    covs = [float(x.get("coverage") or 0.0) for x in aligned_lines]
    avg_cov = float(sum(covs) / max(1, len(covs))) if aligned_lines else 0.0
    low = [
        x
        for x in aligned_lines
        if float(x.get("coverage") or 0.0) < float(min_coverage)
    ]
    worst = sorted(aligned_lines, key=lambda x: float(x.get("coverage") or 0.0))[:5]
    report = {
        "title": title,
        "duration": float(duration),
        "avg_coverage": float(avg_cov),
        "min_coverage_threshold": float(min_coverage),
        "low_confidence_lines": int(len(low)),
        "worst_lines": [
            {
                "text": str(w.get("text") or ""),
                "section": str(w.get("section") or ""),
                "coverage": float(w.get("coverage") or 0.0),
                "confidence": float(w.get("confidence") or 0.0),
            }
            for w in worst
        ],
        "notes": [
            "Coverage measures how much of the provided lyric line was found in the sung audio.",
            "Low coverage can mean the vocalist sang different words (common in music generation).",
        ],
    }

    # Auto fallback: if provided lyrics don't match, use sung words segmentation.
    if mode2 == "auto" and float(avg_cov) < float(auto_fallback_coverage or 0.25):
        segs = _segment_asr_words(
            words=words,
            words_per_segment=int(words_per_segment or 6),
            max_segment_s=float(max_segment_s or 2.2),
            min_prob=0.15,
            break_gap_s=0.95,
        )
        ls = float(lead_s or 0.0)
        for it in segs:
            it["start"] = float(it.get("start") or 0.0) + ls
            it["end"] = float(it.get("end") or 0.0) + ls
            it["start"] = float(max(0.0, min(duration, it["start"])))
            it["end"] = float(max(0.0, min(duration, it["end"])))
            if it["end"] <= it["start"]:
                it["end"] = min(duration, it["start"] + 0.6)

        report2 = {
            **report,
            "avg_coverage": 1.0,
            "low_confidence_lines": 0,
            "worst_lines": [],
            "notes": report.get("notes", [])
            + [
                f"Auto fallback enabled: avg_coverage {float(avg_cov):.2f} < {float(auto_fallback_coverage or 0.25):.2f}.",
                "Using Sung Words mode for better sync.",
            ],
        }
        out = {
            "version": 2,
            "run_dir": os.path.abspath(run_dir),
            "audio": os.path.basename(audio_path),
            "audio_for_asr": os.path.basename(audio_for_asr),
            "duration": float(duration),
            "language": str(language or "en"),
            "mode": "auto->sung",
            "segmentation": {
                "words_per_segment": int(words_per_segment or 6),
                "max_segment_s": float(max_segment_s or 2.2),
                "lead_s": float(lead_s or 0.0),
                "auto_fallback_coverage": float(auto_fallback_coverage or 0.25),
            },
            "model": {
                "name": str(model_size or "small"),
                "device": str(dev),
                "compute_type": str(ct),
                "demucs": bool(demucs),
                "demucs_model": str(demucs_model or "htdemucs"),
            },
            "lines": [{"index": i, **ln} for i, ln in enumerate(segs)],
            "report": report2,
        }
        _write_json(os.path.join(run_dir, "alignment_lines.json"), out)
        _write_json(os.path.join(run_dir, "alignment_report.json"), report2)
        return report2

    out = {
        "version": 2,
        "run_dir": os.path.abspath(run_dir),
        "audio": os.path.basename(audio_path),
        "audio_for_asr": os.path.basename(audio_for_asr),
        "duration": float(duration),
        "language": str(language or "en"),
        "mode": "provided" if mode2 in ("auto", "provided") else str(mode2),
        "segmentation": {
            "words_per_segment": int(words_per_segment or 6),
            "max_segment_s": float(max_segment_s or 2.2),
            "lead_s": float(lead_s or 0.0),
            "auto_fallback_coverage": float(auto_fallback_coverage or 0.25),
        },
        "model": {
            "name": str(model_size or "small"),
            "device": str(dev),
            "compute_type": str(ct),
            "demucs": bool(demucs),
            "demucs_model": str(demucs_model or "htdemucs"),
        },
        "lines": aligned_lines,
        "report": report,
    }

    _write_json(os.path.join(run_dir, "alignment_lines.json"), out)
    _write_json(os.path.join(run_dir, "alignment_report.json"), report)
    return report


def main() -> int:
    p = argparse.ArgumentParser(
        description="Auto-sync lyrics to singing (English-first)"
    )
    p.add_argument("--run-dir", required=True, help="Output run folder")
    p.add_argument("--language", default="en", help="Language for ASR")
    p.add_argument(
        "--device", default="auto", choices=["auto", "cuda", "cpu"], help="ASR device"
    )
    p.add_argument(
        "--use-gpu",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Prefer GPU when available",
    )
    p.add_argument(
        "--model",
        default="small",
        choices=["tiny", "base", "small", "medium"],
        help="Whisper model size",
    )
    p.add_argument("--compute-type", default="auto", help="faster-whisper compute_type")
    p.add_argument(
        "--demucs",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Use Demucs vocal isolation",
    )
    p.add_argument("--demucs-model", default="htdemucs", help="Demucs model name")
    p.add_argument(
        "--min-coverage",
        type=float,
        default=0.45,
        help="Minimum coverage to accept alignment",
    )
    p.add_argument(
        "--mode",
        default="auto",
        choices=["auto", "provided", "sung"],
        help="Alignment mode (auto falls back to sung words when mismatch)",
    )
    p.add_argument(
        "--words-per-segment",
        type=int,
        default=6,
        help="Split captions into N-word segments (recommended 5-7)",
    )
    p.add_argument(
        "--max-segment-s",
        type=float,
        default=2.2,
        help="Maximum duration per segment (seconds)",
    )
    p.add_argument(
        "--lead",
        type=float,
        default=-0.15,
        help="Shift subtitles earlier/later (negative shows earlier)",
    )
    p.add_argument(
        "--auto-fallback-coverage",
        type=float,
        default=0.25,
        help="If avg coverage is below this, auto mode uses Sung Words",
    )
    args = p.parse_args()

    try:
        report = align_run(
            run_dir=str(args.run_dir),
            device=str(args.device),
            use_gpu=bool(args.use_gpu),
            model_size=str(args.model),
            demucs=bool(args.demucs),
            demucs_model=str(args.demucs_model),
            compute_type=str(args.compute_type),
            min_coverage=float(args.min_coverage),
            language=str(args.language),
            mode=str(args.mode),
            words_per_segment=int(args.words_per_segment),
            max_segment_s=float(args.max_segment_s),
            lead_s=float(args.lead),
            auto_fallback_coverage=float(args.auto_fallback_coverage),
        )
        # Print a short, UI-friendly summary.
        print(
            "Auto Sync complete: "
            + f"avg_coverage={float(report.get('avg_coverage') or 0.0):.2f}, "
            + f"low_lines={int(report.get('low_confidence_lines') or 0)}"
        )
        return 0
    except Exception as e:
        print(f"Error: {e}")
        return 2


if __name__ == "__main__":
    raise SystemExit(main())
