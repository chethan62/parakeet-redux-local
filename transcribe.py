#!/usr/bin/env python3
"""Parakeet Redux transcriptions (Photon runtime), local + CPU by default.

Usage:
  .venv/bin/python transcribe.py FILE [--device cpu] [--timestamps segment|word|none] [--rtf]
"""
import argparse
import subprocess
import sys
import time
from pathlib import Path

import moondream as md

MODEL = "moondream/parakeet-redux"


def duration(path: Path) -> float:
    out = subprocess.run(
        ["ffprobe", "-v", "error", "-show_entries", "format=duration", "-of", "csv=p=0", str(path)],
        capture_output=True, text=True, check=True,
    )
    return float(out.stdout.strip())


def to_wav16k(src: Path, dst: Path) -> None:
    """Photon wants clean 16 kHz mono; ffmpeg is the extractor for video/anything else."""
    subprocess.run(
        ["ffmpeg", "-y", "-v", "error", "-i", str(src), "-vn",
         "-acodec", "pcm_s16le", "-ar", "16000", "-ac", "1", str(dst)],
        check=True,
    )


def transcribe(path: Path, speech, timestamps: str = "none", rtf: bool = False):
    wav = path if path.suffix.lower() == ".wav" else path.with_suffix(".16k.wav")
    if wav != path:
        to_wav16k(path, wav)
    t0 = time.perf_counter()
    result = speech.transcribe(audio=str(wav), timestamps=timestamps)
    wall = time.perf_counter() - t0
    if rtf:
        dur = duration(wav)
        print(f"# {path.name}: {dur:.2f}s audio in {wall:.2f}s wall = {dur/wall:.2f}x realtime", file=sys.stderr)
    return result


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("files", type=Path, nargs="+")
    ap.add_argument("--device", default="cpu")
    ap.add_argument("--timestamps", default="none", choices=["none", "segment", "word"])
    ap.add_argument("--rtf", action="store_true")
    a = ap.parse_args()
    t0 = time.perf_counter()
    with md.photon(MODEL, device=a.device) as speech:
        print(f"# model load: {time.perf_counter() - t0:.1f}s", file=sys.stderr)
        for path in a.files:
            r = transcribe(path, speech, a.timestamps, a.rtf)
            print(f"=== {path.name} ===")
            if a.timestamps == "none":
                print(r["text"])
                continue
            for seg in r["segments"]:
                print(f"[{seg['start']:7.2f} -> {seg['end']:7.2f}] {seg['text']}")
                for w in seg.get("words", []):
                    print(f"    {w['start']:7.2f} {w['word']}")


if __name__ == "__main__":
    main()
