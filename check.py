#!/usr/bin/env python3
"""Fails if Parakeet Redux stops working or crawls. Run: .venv/bin/python check.py"""
import re, time, sys
from pathlib import Path
import moondream as md
from transcribe import MODEL, device_facts, resolve_device, transcribe, duration

M = Path(__file__).parent / "models"
TRUTH = "The quick brown fox jumps over the lazy dog. Subtitle synchronization depends on accurate word timestamps. Ship the smallest change that actually fixes the reported problem."

norm = lambda s: re.sub(r"[^a-z0-9 ]", "", s.lower()).split()

device = resolve_device("auto")   # the default user path, whatever this machine's best kernel is
print(f"device={device} {device_facts(device)}", file=sys.stderr)
with md.photon(MODEL, device=device) as speech:
    r = transcribe(M / "tts_truth.mp3", speech)
    got = norm(r["text"])
    assert got == norm(TRUTH), f"WER>0 on own TTS ground truth:\n{' '.join(got)}"

    segs = transcribe(M / "tts_truth.mp3", speech, timestamps="word")["segments"]
    words = [w for s in segs for w in s.get("words", [])]
    assert len(words) >= len(TRUTH.split()), f"word timestamps missing: {len(words)} words"
    assert all(w["end"] >= w["start"] for w in words), "non-monotonic word timings"
    assert segs[-1]["end"] <= duration(M / "tts_truth.16k.wav") + 0.5, "segment past end of audio"

    t0 = time.perf_counter()
    transcribe(M / "long.wav", speech)
    rtf = duration(M / "long.wav") / (time.perf_counter() - t0)
    assert rtf > 2.0, f"slower than 2x realtime on 125s clip: {rtf:.2f}x"

print(f"ok: exact TTS transcript, {len(words)} word timings, {rtf:.2f}x realtime on 125s")
