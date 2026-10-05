#!/usr/bin/env python3
"""Transcribe one clip with Parakeet via sherpa-onnx — the *permitted* runtime.

Why this exists: `transcribe.py` runs Moondream's Parakeet Redux through `kestrel_kernels`,
whose licence grants nothing without a separate written agreement with M87 Labs ("if you have
not entered into such an Agreement, you have no license to use this software") and whose §2
forbids reverse engineering and unpacking the packed kernel collection. The *weights* were
CC-BY-4.0 and fine; the runtime was the problem. This script keeps the model family and swaps
the runtime for sherpa-onnx (Apache-2.0) using the k2-fsa int8 ONNX conversion.

Measured on this box (see ~/projects/parakeet-redux/bench_sherpa.py and the vlc-ai-subs
research notes): sherpa int8 reads ~7.45x realtime where Redux read 9.25-9.92x — roughly 25%
slower, which is the price of a licence that can actually be used.

Usage (the same shape as transcribe.py, so it can replace it in a command provider):
    <venv-with-sherpa>/bin/python transcribe_sherpa.py -q clip.wav
    <venv-with-sherpa>/bin/python transcribe_sherpa.py --self-test

The transcript goes to stdout: that is what Hermes' stt command provider reads.
"""
import argparse
import os
import sys
import time
import wave

SAMPLE_RATE = 16000
MODELS_ROOT = os.path.expanduser("~/.local/share/sherpa-onnx/models")
VARIANTS = {
    "v2": "sherpa-onnx-nemo-parakeet-tdt-0.6b-v2-int8",   # English
    "v3": "sherpa-onnx-nemo-parakeet-tdt-0.6b-v3-int8",   # 25 European languages
}
FILES = ("encoder.int8.onnx", "decoder.int8.onnx", "joiner.int8.onnx", "tokens.txt")

# ponytail: ONE pass, no chunking. Parakeet's encoder refuses very long input (measured
# between 5 and 10 minutes on the plugin's lane), and voice messages are seconds long, so a
# clip past the ceiling fails loudly with the reason instead of silently truncating. Add
# 30 s + 2 s-overlap chunking (as parakeet_runner.py does) only if long dictation is needed.
MAX_SECONDS = 120.0


def model_dir(variant: str) -> str:
    """Model directory for *variant*, or raise with an actionable message."""
    d = os.path.join(MODELS_ROOT, VARIANTS[variant])
    missing = [f for f in FILES if not os.path.isfile(os.path.join(d, f))]
    if missing:
        raise SystemExit(
            f"parakeet {variant} model incomplete at {d} (missing {', '.join(missing)}).\n"
            f"Install it with: bash install-parakeet-model.sh  (from the vlc-ai-subs repo)"
        )
    return d


def load_wav(path: str):
    """Read a 16 kHz mono wav into float32 samples in [-1, 1] (numpy array)."""
    with wave.open(path) as w:
        if w.getnchannels() != 1 or w.getframerate() != SAMPLE_RATE:
            raise SystemExit(
                f"{path}: expected mono {SAMPLE_RATE} Hz wav, got "
                f"{w.getnchannels()} channel(s) at {w.getframerate()} Hz. "
                f"Convert with: ffmpeg -i in -ac 1 -ar {SAMPLE_RATE} out.wav"
            )
        frames = w.readframes(w.getnframes())
    import numpy as np  # only needed on this path
    return np.frombuffer(frames, dtype=np.int16).astype("float32") / 32768.0


def transcribe(samples, d: str, threads: int) -> str:
    import sherpa_onnx
    rec = sherpa_onnx.OfflineRecognizer.from_transducer(
        tokens=os.path.join(d, "tokens.txt"),
        encoder=os.path.join(d, "encoder.int8.onnx"),
        decoder=os.path.join(d, "decoder.int8.onnx"),
        joiner=os.path.join(d, "joiner.int8.onnx"),
        # Both are required: the defaults raise "'vocab_size' does not exist in the metadata".
        model_type="nemo_transducer",
        modeling_unit="cjkchar",
        num_threads=threads,
    )
    stream = rec.create_stream()
    stream.accept_waveform(SAMPLE_RATE, samples)
    rec.decode_stream(stream)
    return stream.result.text.strip()


def self_test() -> int:
    """Cheap checks that fail if the model layout or the contract drifts."""
    for variant in VARIANTS:
        d = model_dir(variant)
        assert os.path.getsize(os.path.join(d, "encoder.int8.onnx")) > 1_000_000, d
    print(f"self-test ok: {len(VARIANTS)} model variants present under {MODELS_ROOT}")
    try:
        import sherpa_onnx  # noqa: F401
    except ImportError:
        print("note: sherpa_onnx is not importable with THIS interpreter — point the command "
              "provider at a venv that has it (the vlc-ai-subs install creates one).")
        return 0
    # One real decode: this is what fails if the model files or the sherpa API contract drift.
    # Deliberately not asserting on the text — silence can legitimately decode to nothing, so
    # an assertion on content here would be vacuous (it would pass however broken the call was).
    transcribe([0.0] * SAMPLE_RATE, model_dir("v3"), 2)
    print("self-test ok: sherpa_onnx importable, a decode round-trips")
    return 0


def main() -> int:
    ap = argparse.ArgumentParser(description="Parakeet (sherpa-onnx) transcriber")
    ap.add_argument("input", nargs="?", help="16 kHz mono wav")
    ap.add_argument("-q", "--quiet", action="store_true", help="print only the transcript")
    ap.add_argument("-m", "--model", choices=sorted(VARIANTS), default="v3")
    ap.add_argument("-t", "--threads", type=int, default=max(1, (os.cpu_count() or 4) // 2))
    ap.add_argument("--self-test", action="store_true")
    a = ap.parse_args()
    if a.self_test:
        return self_test()
    if not a.input:
        ap.error("give a wav, or --self-test")

    d = model_dir(a.model)
    samples = load_wav(a.input)
    dur = len(samples) / SAMPLE_RATE
    if dur > MAX_SECONDS:
        raise SystemExit(f"{a.input}: {dur:.1f}s is past this script's one-pass ceiling "
                         f"({MAX_SECONDS:.0f}s) — see the note in this file.")
    t0 = time.time()
    text = transcribe(samples, d, a.threads)
    if not a.quiet:
        print(f"# parakeet-{a.model} (sherpa-onnx int8) {dur:.2f}s audio, "
              f"{time.time() - t0:.2f}s, {a.threads} threads", file=sys.stderr)
    print(text)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
