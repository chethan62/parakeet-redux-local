#!/usr/bin/env python3
# SPDX-License-Identifier: MIT
"""Does Redux survive background noise? Mixes white noise into the fixture at set SNRs and measures
word error rate against the CLEAN transcript.

That reference is the model's own clean output, so this measures degradation under noise, not absolute
accuracy (the original weights would be needed for that). 10.4 s = 32 words, so each word is ~3% of WER.
"""
import sys
import time
import wave
from pathlib import Path

import numpy as np
import moondream as md

from check import fixture, norm, RECORDED_BASELINE
from transcribe import MODEL, resolve_device, device_facts, to_wav16k, transcribe

SNRS = (10.0, 5.0, 0.0)


def wer(ref: str, hyp: str) -> float:
    """Word-level Levenshtein distance / len(ref). stdlib only, ~32 words so O(n^2) is irrelevant."""
    r, h = ref.split(), hyp.split()
    d = np.zeros((len(r) + 1, len(h) + 1), dtype=int)
    d[:, 0] = np.arange(len(r) + 1)
    d[0, :] = np.arange(len(h) + 1)
    for i in range(1, len(r) + 1):
        for j in range(1, len(h) + 1):
            d[i, j] = min(d[i - 1, j] + 1, d[i, j - 1] + 1, d[i - 1, j - 1] + (r[i - 1] != h[j - 1]))
    return d[-1, -1] / len(r)


def read_wav(path: Path) -> tuple[np.ndarray, int]:
    with wave.open(str(path)) as w:
        return np.frombuffer(w.readframes(w.getnframes()), dtype="<i2").astype(np.float32), w.getframerate()


def write_wav(path: Path, samples: np.ndarray, rate: int) -> None:
    with wave.open(str(path), "wb") as w:
        w.setnchannels(1)
        w.setsampwidth(2)
        w.setframerate(rate)
        w.writeframes(np.clip(samples, -32768, 32767).astype("<i2").tobytes())


def add_noise(src: Path, snr_db: float, out: Path) -> Path:
    """White noise scaled so its RMS is snr_db below the speech RMS (measured, not assumed)."""
    samples, rate = read_wav(src)
    noise = np.random.default_rng(0).standard_normal(samples.shape).astype(np.float32)  # fixed seed: repeatable
    speech_rms, noise_rms = float(np.sqrt((samples**2).mean())), float(np.sqrt((noise**2).mean()))
    write_wav(out, samples + noise * (speech_rms / noise_rms) * 10 ** (-snr_db / 20), rate)
    return out


def main() -> None:
    flac = fixture()
    clean_wav = flac.with_suffix(".16k.wav")
    to_wav16k(flac, clean_wav)

    device = resolve_device("auto")
    print(f"device={device} {device_facts(device)}", file=sys.stderr)
    with md.photon(MODEL, device=device) as speech:
        t0 = time.perf_counter()
        clean = transcribe(clean_wav, speech)["text"]
        clean_seconds = time.perf_counter() - t0
        ref = norm(clean)
        assert wer(ref, norm(RECORDED_BASELINE)) == 0.0, f"clean transcript drifted from the file baseline: {clean}"
        print(f"clean        WER  0.0%  ({clean_seconds:.1f}s)")
        worst = 0.0
        for snr in SNRS:
            wav = add_noise(clean_wav, snr, clean_wav.with_name(f"noisy_{int(snr)}db.wav"))
            got = norm(transcribe(wav, speech)["text"])
            worst = max(worst, wer(ref, got))
            print(f"noise {snr:4.0f} dB  WER {100 * wer(ref, got):5.1f}%  got: {got}")
        # Noise is this model's known weak spot, but it must degrade, not collapse: at 0 dB SNR the words
        # stay recognisable. A rewrite that loses the acoustic margin shows up here, not in check.py.
        assert worst < 0.25, f"noisy transcription degraded past 25% WER at 0 dB SNR: {100 * worst:.1f}%"


if __name__ == "__main__":
    main()
