#!/usr/bin/env python3
# SPDX-License-Identifier: MIT
"""Parakeet Redux (Moondream/Photon) transcription — picks the device, not the marketing.

  .venv/bin/python transcribe.py --list-devices
  .venv/bin/python transcribe.py FILE... [--device auto|cpu|cuda|mps] [--timestamps segment|word|none] [--rtf]

`--device auto` (the default) prefers whichever device has a NATIVE ternary kernel — Metal (mps) on
Apple silicon, int8 avx2/avxvnni/avx512vnni (cpu) on x86, neon (cpu) on aarch64 — because CUDA has
none: there the codes are dequantized once (dense form) and four conformer kernels fall back to
plain PyTorch, which measured 2-2.4x SLOWER than this CPU. Force a device to override.
"""
import argparse
import subprocess
import sys
import time
from pathlib import Path

import moondream as md

MODEL = "moondream/parakeet-redux"
DEVICES = ("cpu", "cuda", "mps")


def _ternary():
    """The engine's public kernel API; absent/renamed -> device facts are simply unknown."""
    try:
        from kestrel_kernels import ternary

        return ternary
    except Exception:
        return None


def has_native_kernel(device: str) -> bool:
    """Only two backends ship a ternary kernel: Apple Metal (mps) and the CPU int8 paths.
    CUDA runs the dequantized dense form, so it is never a native-kernel device."""
    t = _ternary()
    if t is None:
        return False
    base = device.split(":")[0]
    try:
        if base == "mps":
            return t.metal_gemm_ready()
        if base == "cpu":
            return t.ternary_gemm_ready()
        return False
    except Exception:
        return False


def available() -> list[str]:
    import torch

    out = ["cpu"]
    if torch.backends.mps.is_available():
        out.append("mps")
    if torch.cuda.is_available():
        out.append("cuda")
    return out


def resolve_device(pref: str) -> str:
    """auto = first available device WITH a native kernel, else the last available one."""
    if pref != "auto":
        return pref
    avail = available()
    for d in ("mps", "cpu", "cuda"):  # native Apple GPU, native CPU, fallback GPU
        if d in avail and has_native_kernel(d):
            return d
    return avail[-1]


def device_facts(device: str) -> str:
    t = _ternary()
    if t is None:
        return ""
    base = device.split(":")[0]
    try:
        facts = f"resident_form={t.resident_form(base)}"
        if base == "cpu":
            facts += f" isa={t.ternary_gemm_isa()}"
        return facts
    except Exception:
        return ""


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
    ap.add_argument("files", type=Path, nargs="*")
    ap.add_argument("--device", default="auto", help="auto|cpu|cuda|mps (or cuda:N); default: auto")
    ap.add_argument("--timestamps", default="none", choices=["none", "segment", "word"])
    ap.add_argument("--rtf", action="store_true")
    ap.add_argument("-q", "--quiet", action="store_true",
                    help="print only the transcript text (no '=== file ===' header; status still goes to stderr)")
    ap.add_argument("--list-devices", action="store_true", help="devices, native-kernel status, the auto pick")
    a = ap.parse_args()

    if a.list_devices:
        avail = available()
        for d in ("mps", "cpu", "cuda"):
            print(f"  {d:4s} {'available' if d in avail else 'not present':12s} native_kernel={has_native_kernel(d)}")
        print(f"  -> auto picks: {resolve_device('auto')}")
        return
    if not a.files:
        ap.error("no input files (or use --list-devices)")
    if a.device.split(":")[0] not in ("auto",) + DEVICES:
        ap.error(f"--device {a.device!r} not one of {('auto',) + DEVICES} (Photon accepts cpu, cuda, mps)")

    device = resolve_device(a.device)
    base = device.split(":")[0]
    if base != "cpu" and base not in available():
        ap.error(f"--device {device}: not available on this machine (available: {available()}); see --list-devices")
    print(f"# device={device} (requested {a.device}) {device_facts(device)}", file=sys.stderr)
    if device.startswith("cuda"):
        print("# note: CUDA runs this model dense (no ternary kernel) and falls back to PyTorch for four "
              "conformer kernels - a CPU with a native int8 path is usually faster", file=sys.stderr)

    t0 = time.perf_counter()
    with md.photon(MODEL, device=device) as speech:
        print(f"# model load: {time.perf_counter() - t0:.1f}s", file=sys.stderr)
        for path in a.files:
            r = transcribe(path, speech, a.timestamps, a.rtf)
            if not a.quiet:
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
