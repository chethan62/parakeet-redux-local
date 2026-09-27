#!/usr/bin/env python3
# SPDX-License-Identifier: MIT
"""GPU/CUDA reality check for the Parakeet models under Photon.

Needs a venv whose torch is the CUDA build - create a throwaway one (see the README's install recipe
without --torch-backend cpu) and run it there:

  .venv-cuda/bin/python gpu_test.py                        # redux on cuda
  .venv-cuda/bin/python gpu_test.py moondream/parakeet-ultra
"""
import sys, time
from pathlib import Path

import torch
import moondream as md
from kestrel_kernels import ternary as T

MODELS = Path(__file__).parent / "models"
files = [MODELS / "1.flac", MODELS / "long.wav"]
model = sys.argv[1] if len(sys.argv) > 1 else "moondream/parakeet-redux"
device = sys.argv[2] if len(sys.argv) > 2 else "cuda"

print(f"torch {torch.__version__} cuda {torch.version.cuda} | {torch.cuda.get_device_name(0)}")
print(f"resident_form('cuda') = {T.resident_form('cuda')}   ternary_gemm_isa = {T.ternary_gemm_isa()}")
torch.cuda.reset_peak_memory_stats()

t0 = time.perf_counter()
with md.photon(model, device=device) as speech:
    print(f"load: {time.perf_counter() - t0:.1f}s  {getattr(speech, '_active_gpu', '')}")
    for f in files:
        wav = f if f.suffix == ".wav" else f.with_suffix(".16k.wav")
        torch.cuda.synchronize()
        t1 = time.perf_counter()
        r = speech.transcribe(audio=str(wav))
        torch.cuda.synchronize()
        el = time.perf_counter() - t1
        dur = float(__import__("subprocess").run(
            ["ffprobe", "-v", "error", "-show_entries", "format=duration", "-of", "csv=p=0", str(wav)],
            capture_output=True, text=True).stdout)
        print(f"  {f.name:12s} {dur:7.2f}s audio {el:6.2f}s wall = {dur/el:5.2f}x realtime")
        print(f"    peak VRAM {torch.cuda.max_memory_allocated()/2**20:7.0f} MiB | {r['text'][:70]}")
