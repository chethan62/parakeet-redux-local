#!/usr/bin/env python3
# SPDX-License-Identifier: MIT
"""How fast is vlc-ai-subs' production ASR path on this box?

Their parakeet_runner.py runs sherpa-onnx int8 Parakeet-TDT with provider="cpu" in 30 s chunks with
2 s of overlap (not Photon, not ggml). This mirrors that configuration exactly -- same model files,
same thread count, same chunking -- so its realtime factor is comparable with Photon's 13.98x on the
same 125 s clip. Run it with THEIR venv (sherpa_onnx 1.13.4 lives there):

  ~/.local/share/vlc-ai-subs/venv-whisperx/bin/python3 bench_sherpa.py

Read-only: loads their installed models, writes nothing.
"""
import os
import sys
import time
import wave

import numpy as np
import sherpa_onnx

ROOT = os.path.expanduser("~/.local/share/sherpa-onnx/models")
CLIP = os.path.expanduser("~/projects/parakeet-redux/models/long.wav")
# Optional first argument: any 16 kHz mono wav, so this can time the SAME file a
# different engine was measured on (the 125 s fixture is synthetic read speech
# repeated 12x and flatters every engine; real film audio is the honest clip).
if len(sys.argv) > 1:
    CLIP = sys.argv[1]
SR = 16000
CHUNK, OVERLAP = 30.0, 2.0
FILES = ("encoder.int8.onnx", "decoder.int8.onnx", "joiner.int8.onnx", "tokens.txt")


def pick():
    """English prefers v2 (their select_variant does the same), v3 is the multilingual one."""
    for tag in ("v2", "v3"):
        d = os.path.join(ROOT, f"sherpa-onnx-nemo-parakeet-tdt-0.6b-{tag}-int8")
        if all(os.path.isfile(os.path.join(d, f)) for f in FILES):
            return tag, d
    sys.exit("no sherpa-onnx parakeet model installed")


def load(path):
    with wave.open(path) as w:
        assert w.getframerate() == SR and w.getnchannels() == 1, "expected 16k mono"
        raw = w.readframes(w.getnframes())
    return np.frombuffer(raw, dtype="<i2").astype(np.float32) / 32768.0


def main():
    tag, d = pick()
    samples = load(CLIP)
    dur = len(samples) / SR
    threads = min(8, os.cpu_count() or 2)

    t = time.time()
    rec = sherpa_onnx.OfflineRecognizer.from_transducer(
        encoder=os.path.join(d, "encoder.int8.onnx"),
        decoder=os.path.join(d, "decoder.int8.onnx"),
        joiner=os.path.join(d, "joiner.int8.onnx"),
        tokens=os.path.join(d, "tokens.txt"),
        num_threads=threads, provider="cpu",
        model_type="nemo_transducer", modeling_unit="cjkchar",
    )
    load_s = time.time() - t

    n, ov = int(CHUNK * SR), int(OVERLAP * SR)
    tokens = 0
    t = time.time()
    for start in range(0, len(samples), n):
        stop = min(start + n, len(samples))
        st = rec.create_stream()
        st.accept_waveform(SR, samples[start:min(stop + ov, len(samples))])
        rec.decode_stream(st)
        tokens += len(st.result.tokens or [])
    chunked = time.time() - t

    t = time.time()
    st = rec.create_stream()
    st.accept_waveform(SR, samples)
    rec.decode_stream(st)
    single = time.time() - t

    print(f"model {tag}  load {load_s:.1f}s  clip {dur:.2f}s  threads {threads}  provider cpu (int8)")
    print(f"chunked {CHUNK:.0f}s+{OVERLAP:.0f}s overlap : {chunked:.2f}s = {dur / chunked:.2f}x realtime"
          f"  ({tokens} tokens)")
    print(f"single pass {dur:.0f}s          : {single:.2f}s = {dur / single:.2f}x realtime")
    print(f"text: {(st.result.text or '')[:150]!r}")


if __name__ == "__main__":
    main()
