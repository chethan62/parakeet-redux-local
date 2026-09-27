# Parakeet Redux on a CPU without AVX-512 — what the 113× headline actually means

[![ci](https://github.com/chethan62/parakeet-redux-local/actions/workflows/ci.yml/badge.svg)](https://github.com/chethan62/parakeet-redux-local/actions/workflows/ci.yml)
[![license: MIT](https://img.shields.io/badge/license-MIT-blue.svg)](LICENSE)
[![python 3.12](https://img.shields.io/badge/python-3.12-blue.svg)](#requirements)
[![model: CC-BY-4.0](https://img.shields.io/badge/weights-CC--BY--4.0-blue.svg)](https://huggingface.co/moondream/parakeet-redux)

Independent measurements of [Moondream Parakeet Redux](https://huggingface.co/moondream/parakeet-redux)
(`moondream/parakeet-redux`, a 1.58-bit ternary re-export of
[NVIDIA parakeet-tdt-0.6b-v3](https://huggingface.co/nvidia/parakeet-tdt-0.6b-v3)) run through
Moondream's Photon runtime on this machine:

- **CPU**: Intel i5-10300H (Comet Lake, 8 threads, **AVX2 only — no AVX-512, no AVX-VNNI**), 4 GB GTX 1650
  (clock-capped at 300 MHz by the EC/BIOS), CachyOS, Python 3.12, `moondream 2.5.0`
- fixtures: 10.4 s of real speech (`Narsil/asr_dummy`, public) looped into a 125 s clip; a KittentTS-synthesised
  sentence with a known transcript was used during development, and `check.py` now fetches its own fixture.

## Requirements

- Python 3.12 (3.10+ works; the venv recipe below uses 3.12)
- `ffmpeg` on PATH — used to normalise any input to 16 kHz mono
- an engine-supported device: x86 CPU with **AVX2** at minimum (AVX-512 VNNI or AVX-VNNI for the fast path),
  aarch64 CPU (NEON), Apple silicon (`mps`), or CUDA
- ~1.6 GB disk for the venv, 178 MB for the weights, plus ~250 KB of fixtures the gate downloads

## Devices

`--device auto` (the default) picks whichever device actually has a **native ternary kernel** — Apple
Metal on `mps`, int8 AVX2 / AVX-VNNI / AVX-512-VNNI on `cpu`, NEON on aarch64 `cpu` — and only falls back to
CUDA when the machine has no native path at all. CUDA is never "native" for these models, so it is chosen
only if you ask for it. See for yourself:

```sh
$ .venv/bin/python transcribe.py --list-devices
  mps  not present  native_kernel=False
  cpu  available    native_kernel=True
  cuda not present  native_kernel=False
  -> auto picks: cpu
$ .venv/bin/python transcribe.py meeting.m4a --device cuda --rtf
# device=cuda (requested cuda) resident_form=dense
# note: CUDA runs this model dense (no ternary kernel) and falls back to PyTorch for four conformer kernels - a CPU with a native int8 path is usually faster
```

| device | native ternary kernel | measured here | notes |
| --- | --- | --- | --- |
| `cpu` | yes (int8) | **12.6–14.9×** | the default; needs AVX2 at minimum, AVX-512 VNNI is the 113×-class path |
| `cuda` | no — dense form | 1.95–2.70× (provisional) | 1.7–2.1 GB VRAM; four conformer kernels fall back to PyTorch; timed on a hot box, see Results |
| `mps` | yes (Metal) | not measured | **no Apple hardware here, so untested** — the engine reports the Metal path as the supported one; run `--list-devices` on the Mac to confirm `native_kernel=True` |

A device this build cannot reach fails loudly instead of silently running on the CPU: `--device mps` on
Linux exits with the missing-support error, and `--device cuda` under a CPU-only PyTorch tells you to
install a CUDA build.

## Results

One utterance at a time, same audio, same machine, **measured cold and quiet** — see the correction notice
right below, that qualifier is not decoration.

| engine / device | 10 s clip | 125 s clip | notes |
| --- | --- | --- | --- |
| **Parakeet Redux, CPU** | **14.90×** | **13.98×** | +4.3–5.1 s one-time model load; 12.71–13.69× across three gate runs |
| ggml `parakeet-cli` q8_0, Vulkan (GTX 1650) | — | 5.24× (23.94 s) | whisper.cpp's own Parakeet example, already installed here |
| whisper.cpp `ggml-small.en -t 8`, CPU | — | 4.64× (26.97 s) | the reference everyone means by "Whisper on CPU" |
| ggml `parakeet-cli` q8_0, CPU (`-ng`) | — | 4.46× (28.13 s) | same model, plain ggml int8 path |
| Parakeet Redux, CUDA (GTX 1650) | 1.95× | 2.48× | **provisional** — timed on a hot box; 1.7–2.1 GB VRAM |
| Parakeet Ultra, CUDA (GTX 1650) | 2.10× | 2.70× | **provisional**, same reason |

So on this laptop, comparing transcribe time only: **Redux is ~3× faster than CPU whisper.cpp-small**
(8.96 s vs 26.97 s on the same 125 s clip) and **~2.7–3× faster than the best ggml Parakeet path** on this
box, and it beats its own CUDA path by a wide margin — on GPU the model runs the dense form with four PyTorch
fallback kernels, where the ggml Vulkan build is at least self-consistent. Wall-to-wall including the
one-time model load the ratio is **1.6×** (16.84 s vs 27.15 s), which is the number to use for a single short
file and the wrong one to use for a batch. The published "113× real time on eight x86 CPU cores" is measured
on an AMD EPYC 9575F (Zen 5, AVX-512 VNNI) — this Comet Lake part has neither the 512-bit nor the 256-bit
int8 dot-product instruction, so quoting 113× for a consumer laptop is off by ~8×.

> **Correction — every earlier local number in this repo's history was void.** The figures that used to be
> here (5.24× on the 10 s clip, 5.63–6.03× on the 125 s clip, and the CUDA row) were taken on a box that had
> climbed to **94 °C** with concurrent load; this i5-10300H power-caps hard under sustained heat. The same
> regression gate reads **5.9× hot and 13.7× cool**, which is a 2.3× measurement error. Everything above was
> re-measured with `loadavg < 2` and the package at **55–58 °C**, and `bench_ggml.sh` now refuses to time a
> busy machine at all. If you re-run this, check `cut -d' ' -f1 /proc/loadavg` and the package temperature
> **before** believing any ratio. The one number that was never affected is CI's: a 4-core `ubuntu-latest`
> runner, which has no reason to be hot, has read a consistent **10.3–10.6×** throughout.

## Why: the kernel path is hardware-gated, not a setting

Photon ships one CPU ternary GEMM with runtime ISA dispatch. On x86 the paths are AVX-512 VNNI, then 256-bit
AVX-VNNI, then AVX2; an Apple silicon machine gets the Metal path, and a CPU with none of them refuses to
load rather than silently going scalar. Observable via the package's public API and by forcing a path:

```
ternary_gemm_ready : True
ternary_gemm_isa   : avx2      # the live path on this CPU (auto-selected)
resident_form(cpu) : gemm8     # weights stay in the packed int8 execution form
metal_gemm_ready   : False

125 s clip, worker threads default / 4 / 8:   11.49× / 13.31× / 13.10×
isa=avx2 (forced — it is already the selection): 12.58×
```

The first timing after a model load reads ~15 % slow (`11.49×` above, then three settings in a row at
13.1–13.3×), so that is warm-up, not a thread effect — one run per setting cannot distinguish thread counts
on this box. `set_gemm_isa()` cannot conjure a path the silicon lacks: AVX2 *is* this CPU's fast path and
forcing it changes nothing, because it is already the selection. Check your own CPU before believing any
speed number: `grep -o avx512_vnni /proc/cpuinfo` (or `avx_vnni` for the 256-bit variant) must be non-empty.

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
**Provisional magnitudes:** the CUDA table above was timed while an unrelated transcribe job held this box
at ~430 % CPU and 94 °C; the CPU side of the same comparison has since re-measured **2× faster** on a quiet
machine (12.6–13.7×, see Results), so read these as directions, not rates. The direction is structural
regardless — no ternary CUDA kernel exists — and is corroborated by the vendor's own engine notes.

## Accuracy notes

- Clean, close-mic speech: exact transcript on the synthesised ground-truth sentence; word and segment
  timestamps are monotonic and stay inside the audio (`check.py`).
- Noise is the model's weak point. Moondream's own card (self-reported): background-noise average rises
  from 6.72 to 9.04 WER versus the original weights, and 0 dB FLEURS-German from 14.45 to 19.18.
  **Measured here** (`noise_test.py`, white noise mixed into the 10.4 s fixture at a measured SNR, WER
  against the clean transcript): clean 0.0 %, 10 dB 0.0 %, 5 dB 7.1 %, 0 dB 10.7 % — and every error is a
  similar-sounding-word substitution (`turnips`→`turnets`, `bruised`→`brewed`, `fattened`→`satin`), with no
  dropped or invented content. Different fixture and method from the card's numbers, so read them as a
  direction, not a comparison. CI measures 10.7 % at both 5 dB and 0 dB against 7.1 %/10.7 % locally: the
  gap is one substituted word (`fattened`→`flattened`) and traces to the ffmpeg build that decodes the
  fixture, not to the model — which is why the gate asserts a 25 % ceiling rather than an exact figure.
  Adopt for meetings, podcasts, voice notes; expect this when a film has music and effects over dialogue.
- Redux is ahead of the original on the 25-language FLEURS average (10.56 vs 11.62) and on long-form
  TED-LIUM (2.51 vs 2.71), and behind it on the seven English sets (6.55 vs 6.26).

## What independent sources do and do not establish

Separate from everything Moondream reports about itself. Every link is a third party or the vendor's own
docs; where nothing exists, it says so instead of filling the gap.

- **The weights are real and portable.** Independent GGUF ports unpack the ternary codes exactly
  (`w = scale·(code−1)`) and reproduce Photon's transcripts: [cstr's GGUF](https://huggingface.co/cstr/parakeet-redux-GGUF),
  and a [transcribe.cpp conversion](https://huggingface.co/Nairod785/parakeet-ultra-gguf) reporting encoder
  max |Δ| 3.0e-4 against transformers. Those ports drop the 6 `vad_head.*` tensors, so the built-in VAD
  segmentation is Photon-only.
- **The 178 MB and the direction of the accuracy table are corroborated**, on Apple, by Fluid Inference's
  Core ML builds ([redux](https://huggingface.co/FluidInference/parakeet-redux-coreml),
  [ultra](https://huggingface.co/FluidInference/parakeet-ultra-coreml)): 183 MB encoder, and Redux ahead of v3
  on multilingual / behind on English. The same page shows Redux **slower** than v3 on the Neural Engine
  (83.9× vs 128.6× RTFx), so "compressed is faster" does not hold on that engine — and the Apple path here is
  untested for the opposite reason (no Mac to test on).
- **Two unrelated runtimes agree on this box.** whisper.cpp 1.9.2's `parakeet-cli` (already installed here, ggml
  + Vulkan) loads `ggml-org/parakeet-GGUF` q8_0 — 669 MB, 668,757,119 B, exactly the size the card advertises —
  and returns the same sentence as Photon on the 10.4 s fixture: different engine, different quantisation
  (ternary int8 vs q8_0), same words. Note the dialects are not interchangeable: a CrispASR-dialect GGUF of
  the same model is rejected with `failed to load Parakeet model`. CPU-vs-Vulkan timing for this pair is still
  pending a quiet machine (`bench_ggml.sh` measures it, and refuses to time a busy box — a contended run
  already produced one bogus halved number here).
- **The 113× has no third-party replication.** It is vendor-only, on an EPYC 9575F. The one outward test
  quoted in the launch post is a single X user on a Ryzen 9 9950X3D, and the vendor's own reply to it
  admits they never ran that test.
- **Nobody documents a CPU instruction-set requirement**, yet CPUs without the int8 path crash instead of
  falling back (`Illegal instruction (core dumped)` on a Jetson Nano in the vendor's tracker; a third-party
  runtime states the AVX2 floor for its own build). On this box AVX2 is the ceiling — `probe_kernels.py`.
- **The vendor's support matrix lists no Parakeet CPU row and no Redux/Ultra row at all** — support is
  stated for Parakeet v3 BF16 on NVIDIA, and Photon's docs floor GPUs at Ampere. This repo's CUDA numbers
  come from a Turing GTX 1650, outside that matrix, and it works.
- **On a CUDA-visible machine, asking for the CPU costs ~14 %** (vendor tracker
  [kestrel #259](https://github.com/m87-labs/kestrel/issues/259)): the fused CPU conformer kernels are only
  selected when CUDA is invisible to the process; the stated workaround is `CUDA_VISIBLE_DEVICES=`. Not
  reproduced here — the attempt timed on a contended box, so it stays a reported figure (`bench_ggml.sh`
  re-tries it and refuses to time anything while the machine is busy).

## Install gotcha: a 178 MB model in a 6 GB virtualenv

`pip install moondream` pulls `kestrel` → `torch` from PyPI, which is the CUDA build plus ~3.2 GB of
`nvidia-*` wheels. Ask uv for the CPU wheel at install time and none of it is fetched:

```sh
uv venv -p 3.12 .venv
uv pip install --python .venv/bin/python --torch-backend cpu "moondream>=2.4.1" numpy
```

Verified: torch `2.14.0+cpu`, **0 `nvidia-*` packages**, venv **1.6 GB** against 6.0 GB — CI re-checks all
three on every push. Already have the CUDA build in a venv? Reinstall the wheel and drop the dead ones:

```sh
uv pip install --python .venv/bin/python --torch-backend cpu --reinstall-package torch torch
uv pip uninstall --python .venv/bin/python nvidia-cublas nvidia-cudnn-cu13 nvidia-cufft nvidia-curand \
  nvidia-cusolver nvidia-cusparse nvidia-nccl-cu13 nvidia-nvjitlink nvidia-cuda-runtime nvidia-cuda-cupti \
  nvidia-cuda-nvrtc nvidia-cufile nvidia-cusparselt-cu13 nvidia-nvshmem-cu13 nvidia-cuda-nvcc triton
```

`device="cuda"` then fails loudly ("Photon needs PyTorch built with CUDA support") instead of silently
falling back to the CPU.

## Files

| file | what it does |
| --- | --- |
| `transcribe.py` | transcribe any audio/video (ffmpeg → 16 kHz mono), `--device auto\|cpu\|cuda\|mps`, `--timestamps segment\|word`, `--rtf`, `--list-devices` |
| `check.py` | portable regression gate — fetches its own fixture, checks the recorded baseline transcript, one timing per word, monotonic timings, 12/12 repetitions on the long clip, device-aware speed floor |
| `.github/workflows/ci.yml` | the same gate on a clean `ubuntu-latest` box with CPU-only torch |
| `noise_test.py` | noise degradation: mixes white noise at measured SNRs, WER vs the clean transcript, asserts the clean baseline and a 25% ceiling |
| `bench.sh` | sequential CPU A/B against whisper.cpp on the same clip |
| `probe_kernels.py` | prints the live ISA / resident form and times forced paths |
| `gpu_test.py` | device=`cuda` reality check: speed, peak VRAM, fallback warnings |
| `bench_ggml.sh` | cross-checks: the ggml `parakeet-cli` (already on this box) on ggml-org's GGUF, CPU vs Vulkan, plus the CUDA-hidden A/B — waits for a quiet machine before timing |
| `upstream-issue.md` | drafted (not posted) issue for the walkthrough repo this started from: its app omits `device=`, so any NVIDIA machine silently takes the slower CUDA path |

```sh
uv venv -p 3.12 .venv
# --torch-backend cpu from the start: a bare `pip install moondream` pulls the CUDA torch + ~3.2 GB of nvidia-* wheels
uv pip install --python .venv/bin/python --torch-backend cpu "moondream>=2.4.1" numpy

.venv/bin/python transcribe.py meeting.m4a --timestamps segment
.venv/bin/python transcribe.py --list-devices
.venv/bin/python check.py     # self-fetching gate: baseline transcript, word timings, 12 reps, speed floor
```

CI runs exactly that on a clean `ubuntu-latest` box (with the CPU-only torch step), so the install recipe in
this README is verified on every push rather than described.

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

The scripts in this repository are MIT — see [`LICENSE`](LICENSE), and `SPDX-License-Identifier: MIT` at the
top of each file. That licence covers this repository's own code only: it does not relicense the model
weights, the runtime or the kernels listed above, and nothing third-party (code, weights or audio) is
redistributed here — models and fixtures are downloaded at install time.
