# Parakeet Redux on a CPU without AVX-512 — what the 113× headline actually means

Independent measurements of [Moondream Parakeet Redux](https://huggingface.co/moondream/parakeet-redux)
(`moondream/parakeet-redux`, a 1.58-bit ternary re-export of
[NVIDIA parakeet-tdt-0.6b-v3](https://huggingface.co/nvidia/parakeet-tdt-0.6b-v3)) run through
Moondream's Photon runtime on this machine:

- **CPU**: Intel i5-10300H (Comet Lake, 8 threads, **AVX2 only — no AVX-512, no AVX-VNNI**), 4 GB GTX 1650
  (clock-capped at 300 MHz by the EC/BIOS), CachyOS, Python 3.12, `moondream 2.5.0`
- fixtures: 10.4 s / 125.2 s real speech (`Narsil/asr_dummy`) plus a KittentTS-synthesised sentence with a
  known transcript, so word-error-rate is checkable without a dataset.

## Results

One utterance at a time, same audio, same machine:

| engine / device | 10 s clip | 125 s clip | notes |
| --- | --- | --- | --- |
| **Parakeet Redux, CPU** | **5.24× realtime** | **5.63–6.03×** | +12 s one-time model load |
| whisper.cpp `ggml-small.en -t 8`, CPU | — | 3.22× realtime | 38.9 s wall, 13.5 s user → not CPU-bound |
| Parakeet Redux, CUDA (GTX 1650) | 1.95× | 2.48× | 1.7–2.1 GB VRAM |
| Parakeet Ultra, CUDA (GTX 1650) | 2.10× | 2.70× | 1.7–2.0 GB VRAM |

So on this laptop: **Redux is ~1.6× faster than CPU whisper.cpp-small and ~2–2.4× faster than the same
model on the GPU.** The published "113× real time on eight x86 CPU cores" is measured on an AMD EPYC 9575F
(Zen 5, AVX-512 VNNI) — a machine with the 512-bit int8 dot-product instructions this Comet Lake part does
not have. Quoting that number for a consumer laptop is off by ~19×.

## Why: the kernel path is hardware-gated, not a setting

Photon ships one CPU ternary GEMM with runtime ISA dispatch. On x86 the paths are AVX-512 VNNI, then 256-bit
AVX-VNNI, then AVX2; an Apple silicon machine gets the Metal path, and a CPU with none of them refuses to
load rather than silently going scalar. Observable via the package's public API and by forcing a path:

```
ternary_gemm_ready : True
ternary_gemm_isa   : avx2      # the live path on this CPU (auto-selected)
resident_form(cpu) : gemm8     # weights stay in the packed int8 execution form
metal_gemm_ready   : False

isa=auto  → avx2    1.99 s on the 10 s clip   5.24×   # same transcript
isa=scalar (forced) 10.45 s                   1.00×
125 s clip, worker threads default / 4 / 8:  5.63× / 5.69× / 5.92×   # i.e. noise
```

`set_gemm_isa()` cannot conjure a path the silicon lacks — AVX2 *is* this CPU's fast path, and it is already
5× the scalar reference. Thread and tile knobs are worth ~5%. Check your own CPU before believing any speed
number: `grep -o avx512_vnni /proc/cpuinfo` (or `avx_vnni` for the 256-bit variant) must be non-empty.

## On CUDA the ternary kernel is not used at all

`resident_form('cuda')` is `dense`: the packed codes are dequantized once into the activation dtype, so a GPU
run loses the 178 MB weight advantage entirely. Four encoder kernels also have no CUDA build for bfloat16 —
the first CUDA transcribe prints (runtime output, verbatim):

```
RuntimeWarning: kestrel-kernels: no native conformer LayerNorm variant for x=(1, 131, 1024) dtype=torch.bfloat16; using PyTorch fallback.
RuntimeWarning: kestrel-kernels: no native conformer residual LayerNorm variant for x=(1, 131, 1024) dtype=torch.bfloat16; using PyTorch fallback.
RuntimeWarning: kestrel-kernels: no native conformer attention variant for qkv=(1, 131, 3072) heads=8x128 dtype=torch.bfloat16; using PyTorch fallback.
RuntimeWarning: kestrel-kernels: no native conformer depthwise variant for x=(1, 131, 1024) taps=(1024, 9) dtype=torch.bfloat16; using PyTorch fallback.
```

The same transcribe on the CPU prints **zero** fallback warnings. So the warning count is a one-line health
check for which path you actually got:

```sh
.venv/bin/python transcribe.py audio.wav 2>&1 | grep -c "using PyTorch fallback"   # 0 = native CPU kernels
```

Accuracy is unchanged by device (Parakeet Ultra on CUDA reproduced the synthesised sentence exactly, as did
Redux on the CPU), so GPU means the same text, half the speed and ~2 GB of VRAM. Keep these models on the CPU.

## Accuracy notes

- Clean, close-mic speech: exact transcript on the synthesised ground-truth sentence; word and segment
  timestamps are monotonic and stay inside the audio (`check.py`).
- Noise is the model's weak point, per Moondream's own card (self-reported, not measured here): background
  noise average rises from 6.72 to 9.04 WER versus the original weights, and 0 dB FLEURS-German from 14.45 to
  19.18. Adopt for meetings, podcasts, voice notes; be careful with film audio that has music and effects.
- Redux is ahead of the original on the 25-language FLEURS average (10.56 vs 11.62) and on long-form
  TED-LIUM (2.51 vs 2.71), and behind it on the seven English sets (6.55 vs 6.26).

## Install gotcha: a 178 MB model in a 6 GB virtualenv

`pip install moondream` pulls `kestrel` → `torch` from PyPI, which is the CUDA build plus ~3.2 GB of
`nvidia-*` wheels. For CPU-only use:

```sh
uv venv -p 3.12 .venv
uv pip install --python .venv/bin/python "moondream>=2.4.1" numpy
uv pip install --python .venv/bin/python --torch-backend cpu --reinstall-package torch torch
uv pip uninstall --python .venv/bin/python nvidia-cublas nvidia-cudnn-cu13 nvidia-cufft nvidia-curand \
  nvidia-cusolver nvidia-cusparse nvidia-nccl-cu13 nvidia-nvjitlink nvidia-cuda-runtime nvidia-cuda-cupti \
  nvidia-cuda-nvrtc nvidia-cufile nvidia-cusparselt-cu13 nvidia-nvshmem-cu13 nvidia-cuda-nvcc triton
```

6.0 GB → **1.6 GB**, verified still transcribing. `device="cuda"` then fails loudly
("Photon needs PyTorch built with CUDA support") instead of silently falling back to the CPU.

## Files

| file | what it does |
| --- | --- |
| `transcribe.py` | transcribe any audio/video (ffmpeg → 16 kHz mono), `--timestamps segment|word`, `--rtf` |
| `check.py` | regression gate: exact transcript vs synthesised ground truth, one timing per word, monotonic timings, RTF > 2 on the 125 s clip |
| `bench.sh` | sequential CPU A/B against whisper.cpp on the same clip |
| `probe_kernels.py` | prints the live ISA / resident form and times forced paths |
| `gpu_test.py` | device=`cuda` reality check: speed, peak VRAM, fallback warnings |

```sh
cp <your audio> models/          # or point the scripts anywhere
.venv/bin/python transcribe.py meeting.m4a --timestamps segment
.venv/bin/python check.py        # needs models/{tts_truth.mp3,long.wav}; see bench.sh for the fixture recipe
```

## Licensing and attribution

| component | role | licence |
| --- | --- | --- |
| [Moondream Parakeet Redux / Ultra](https://huggingface.co/moondream/parakeet-redux) | the ternary and fp weights this repo runs | CC-BY-4.0 |
| [NVIDIA parakeet-tdt-0.6b-v3](https://huggingface.co/nvidia/parakeet-tdt-0.6b-v3) | the original model Redux re-exports | CC-BY-4.0 |
| Photon (`moondream`, [release post](https://moondream.ai/blog/introducing-parakeet-redux-and-ultra)) | the inference runtime | proprietary (M87 Labs) |
| `kestrel` / `kestrel-kernels` / `kestrel-native` | engine and CPU kernels Photon drives | `kestrel-kernels` ships a **proprietary** licence — "proprietary and confidential … you have no license to use this software" unless covered by a separate written agreement |
| [Narsil/asr_dummy](https://huggingface.co/datasets/Narsil/asr_dummy) | test audio with known transcripts | see dataset card |
| [47thtechcorner/RayCodes_Parakeet_Redux](https://github.com/47thtechcorner/RayCodes_Parakeet_Redux) | the walkthrough this started from | — |

**The weights are CC-BY-4.0; the engine is not open source.** "Free local Whisper alternative" describes the
price, not the licence: the only runtime that executes these ternary kernels ships under a proprietary
agreement that also forbids reverse engineering, deobfuscation or extracting the packed kernel collections.
This repository therefore reports **observable behaviour only** — timings, public API return values and
runtime warnings — and deliberately contains no engine internals, no kernel binaries and no source excerpts
from that package.

Nothing here is vendored: models and wheels are downloaded at install time, so there is no NOTICE/third-party
file to ship.

The scripts in this repository are MIT (see `LICENSE`).
