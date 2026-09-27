#!/usr/bin/env python3
# SPDX-License-Identifier: MIT
"""Fails if Parakeet Redux stops working or crawls. Run: .venv/bin/python check.py

Portable: fetches its audio fixture on first run (public dataset, ~210 KB), so a fresh clone can run it.
Two checks, because an ASR regression shows up in two different ways:

  1. RECORDED_BASELINE below - the transcript this model version produced on 2026-09-27. NOT an
     independent ground truth (the dataset ships no reference transcripts, and inventing one would
     make the test unfalsifiable), so it is a regression tripwire: it fires when the model, the
     runtime or the device path silently changes behaviour.
  2. Reference-free invariants - the 12x-looped clip must come back as 12 repetitions of the same
     sentence (catches truncation and dropped segments on long input) plus a speed floor.

Importing this module is side-effect free (bench.sh reuses fixture/long_clip).
"""
import difflib
import re
import subprocess
import sys
import time
import urllib.request
from pathlib import Path

import moondream as md

from transcribe import MODEL, device_facts, has_native_kernel, resolve_device, transcribe, duration

M = Path(__file__).parent / "models"
FIXTURE = ("1.flac", "https://huggingface.co/datasets/Narsil/asr_dummy/resolve/main/1.flac")
LOOPS = 12
RECORDED_BASELINE = (
    "He hoped there would be stew for dinner, turnips and carrots and bruised potatoes, and fat "
    "mutton pieces to be ladled out in thick, peppered, flour fattened sauce"
)

norm = lambda s: re.sub(r"[^a-z0-9 ]", "", s.lower()).strip()


def fixture() -> Path:
    name, url = FIXTURE
    p = M / name
    if not p.exists():
        M.mkdir(exist_ok=True)
        print(f"fetching {url} -> {p}", file=sys.stderr)
        urllib.request.urlretrieve(url, p)
    return p


def long_clip(src: Path) -> Path:
    """LOOPS x src as 16 kHz mono wav. Re-encoded, not -c copy, so the header duration is truthful."""
    p = M / f"long{LOOPS}.wav"
    if not p.exists():
        subprocess.run(
            ["ffmpeg", "-y", "-v", "error", "-stream_loop", str(LOOPS - 1), "-i", str(src),
             "-ar", "16000", "-ac", "1", str(p)],
            check=True,
        )
    return p


def main() -> None:
    flac = fixture()
    long_wav = long_clip(flac)

    device = resolve_device("auto")
    native = has_native_kernel(device)
    floor = 2.0 if native else 1.0
    print(f"device={device} native_kernel={native} {device_facts(device)}", file=sys.stderr)

    with md.photon(MODEL, device=device) as speech:
        got = transcribe(flac, speech)["text"].strip()
        ratio = difflib.SequenceMatcher(None, norm(got), norm(RECORDED_BASELINE)).ratio()
        assert ratio >= 0.95, f"transcript drifted from the recorded baseline ({ratio:.2f} similar):\n{got}"

        segs = transcribe(flac, speech, timestamps="word")["segments"]
        words = [w for s in segs for w in s.get("words", [])]
        assert len(words) >= len(RECORDED_BASELINE.split()), f"word timings missing: {len(words)}"
        assert all(w["end"] >= w["start"] for w in words), "non-monotonic word timings"
        assert segs[-1]["end"] <= duration(flac) + 0.5, "segment runs past the end of the audio"

        t0 = time.perf_counter()
        long_text = transcribe(long_wav, speech)["text"]
        rtf = duration(long_wav) / (time.perf_counter() - t0)
        reps = norm(long_text).count(norm(RECORDED_BASELINE))
        assert reps >= LOOPS - 1, f"only {reps}/{LOOPS} repetitions survived:\n{long_text[:200]}"
        assert rtf > floor, f"slower than {floor}x realtime on the {duration(long_wav):.0f}s clip: {rtf:.2f}x"

    print(f"ok: baseline {ratio:.2f} similar, {len(words)} word timings, {reps}/{LOOPS} reps, "
          f"{rtf:.2f}x realtime on {duration(long_wav):.0f}s ({device})")


if __name__ == "__main__":
    main()
