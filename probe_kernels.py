#!/usr/bin/env python3
# SPDX-License-Identifier: MIT
"""What ternary kernel path is live on this CPU, and does anything make it faster."""
import time
from pathlib import Path
from kestrel_kernels import ternary as T
import moondream as md
from transcribe import MODEL, transcribe, duration

LONG = Path(__file__).parent / "models" / "long.wav"
print("ternary_gemm_ready :", T.ternary_gemm_ready())
print("ternary_gemm_isa   :", T.ternary_gemm_isa())
print("resident_form(cpu) :", T.resident_form("cpu"))
print("metal_gemm_ready   :", T.metal_gemm_ready())
print("default threads    :", T.worker_threads())

with md.photon(MODEL, device="cpu") as speech:

    def rtf(label):
        t0 = time.perf_counter()
        transcribe(LONG, speech)
        el = time.perf_counter() - t0
        print(f"  {label:24s} {el:6.2f}s wall  {duration(LONG)/el:5.2f}x realtime")

    print("timing 125s clip:")
    for label, threads in (("threads=default", None), ("threads=4", 4), ("threads=8", 8)):
        T.set_worker_threads(threads)
        rtf(label)
    T.set_worker_threads(None)
    live = T.ternary_gemm_isa()
    T.set_gemm_isa(live)
    rtf(f"isa={live} (forced)")
    T.set_gemm_isa(None)
