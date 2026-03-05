import argparse
import os
import shutil
import sys
import tempfile
import urllib.request
import zipfile


DEFAULT_URL = "https://www.gyan.dev/ffmpeg/builds/ffmpeg-release-essentials.zip"


def _download(url: str, dst: str) -> None:
    with urllib.request.urlopen(url, timeout=120) as r:
        data = r.read()
    with open(dst, "wb") as f:
        f.write(data)


def _find_in_tree(root: str, name: str) -> str:
    for dirpath, _dirnames, filenames in os.walk(root):
        for fn in filenames:
            if fn.lower() == name.lower():
                return os.path.join(dirpath, fn)
    return ""


def main() -> int:
    p = argparse.ArgumentParser(description="Download FFmpeg into tools/")
    p.add_argument("--url", default=DEFAULT_URL)
    args = p.parse_args()

    here = os.path.abspath(os.path.dirname(__file__))
    zip_path = os.path.join(here, "ffmpeg.zip")
    out_ffmpeg = os.path.join(here, "ffmpeg.exe")
    out_ffprobe = os.path.join(here, "ffprobe.exe")

    with tempfile.TemporaryDirectory(prefix="ffmpeg_dl_") as tmp:
        print(f"Downloading: {args.url}")
        _download(str(args.url), zip_path)
        print(f"Saved: {zip_path}")

        print("Extracting...")
        with zipfile.ZipFile(zip_path, "r") as z:
            z.extractall(tmp)

        ffmpeg_src = _find_in_tree(tmp, "ffmpeg.exe")
        if not ffmpeg_src:
            print("Error: ffmpeg.exe not found in archive")
            return 2
        ffprobe_src = _find_in_tree(tmp, "ffprobe.exe")

        shutil.copy2(ffmpeg_src, out_ffmpeg)
        print(f"Installed: {out_ffmpeg}")
        if ffprobe_src:
            shutil.copy2(ffprobe_src, out_ffprobe)
            print(f"Installed: {out_ffprobe}")

    try:
        os.remove(zip_path)
    except Exception:
        pass

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
