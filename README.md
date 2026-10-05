# Parakeet on a CPU — what the 113× headline means, and the runtime you can actually use

[![ci](https://github.com/chethan62/parakeet-redux-local/actions/workflows/ci.yml/badge.svg)](https://github.com/chethan62/parakeet-redux-local/actions/workflows/ci.yml)
[![license: MIT](https://img.shields.io/badge/license-MIT-blue.svg)](LICENSE)
[![python 3.12](https://img.shields.io/badge/python-3.12-blue.svg)](#requirements)
[![model: CC-BY-4.0](https://img.shields.io/badge/weights-CC--BY--4.0-blue.svg)](https://huggingface.co/nvidia/parakeet-tdt-0.6b-v3)

This repository measures Parakeet TDT 0.6B on a consumer CPU. It began as a study of
[Moondream Parakeet Redux](https://huggingface.co/moondream/parakeet-redux) (a 1.58-bit ternary
re-export of [NVIDIA parakeet-tdt-0.6b-v3](https://huggingface.co/nvidia/parakeet-tdt-0.6b-v3))
running on Moondream's Photon runtime — and that runtime is gone from here, because its licence
grants nothing to anyone who has not signed something with a third party. See
[the runtime question](#the-runtime-question-what-was-removed-and-why).

What runs here today is `transcribe_sherpa.py`: the same model family, the same job, on
[sherpa-onnx](https://github.com/k2-fsa/sherpa-onnx) (Apache-2.0) with the k2-fsa int8 ONNX
conversion. It is ~25 % slower than the runtime that was removed, which is the price of a licence
that lets you run it at all.

- **CPU**: Intel i5-10300H (Comet Lake, 8 threads, **AVX2 only — no AVX-512, no AVX-VNNI**), 4 GB GTX 1650
  (clock-capped at 300 MHz by the EC/BIOS), CachyOS, Python 3.12
- fixtures: 10.4 s of real speech (`Narsil/asr_dummy`, public) and its 12× loop as a 125 s clip, in
  `models/` (16 kHz mono wav; kept out of git)

## Requirements

- Python 3.12 (3.10+ works; the recipe below uses 3.12)
- `ffmpeg` on PATH — used to normalise any input to 16 kHz mono
- any x86-64 or aarch64 CPU. **No GPU is needed and none helps**: on this model family the GPU path
  was measured 2–2.7× *slower* than the CPU, and sherpa-onnx is CPU ONNX Runtime anyway
- disk: ~641 MB for the Parakeet v3 int8 model (631 MB for v2), plus a small venv — sherpa-onnx and
  numpy, **no torch**

## The runtime question: what was removed, and why

Two of the artefacts this repo started with are the same model in different hands. The **weights**
were open: Moondream's ternary re-export is CC-BY-4.0, 178 MB, and NVIDIA's original is CC-BY-4.0
too. The **runtime that executes them** was not. `kestrel-kernels` (M87 Labs) is licensed *only*
under a separate written agreement with M87 Labs, and its licence says it plainly:

> "proprietary and confidential … if you have not entered into such an Agreement, you have no
> license to use this software"

§2 adds that you may not reverse engineer, decompile, deobfuscate or defeat the protection of the
packed kernel collection, nor unpack, extract or redistribute it — naming AI assistants among those
who must not. §4 notes that circumventing the container may break anti-circumvention law, and §6
terminates the licence on any breach.

An opt-in installer, an external venv, an honest banner and a build exclusion make a dependency like
that *defensible*; they do not make it *licensed*. Every user of the feature would be relying on a
permission that neither they nor this repo can grant, so the integration is not wrapped here — it is
**deleted**: the scripts that drove it, the venv that held it, the CI steps that installed it, and
the walkthrough that showed how. Two other projects that had shipped it (`vlc-ai-subs` and
`whisperer`) removed it for the same reason.

The measurements stay, because they are the answer to a question people still ask — *can't I just
run the open ggml build?* — and they were made on this box, cold and quiet, with the conditions
recorded. Everything in [Results](#results) and the two sections after it describes that removed
runtime. Nothing here installs, invokes or imports it, and `check.py` fails if that changes.

## Results

One utterance at a time, same audio, same machine, **measured cold and quiet** — see the correction
notice right below, that qualifier is not decoration.

| engine / device | 10 s clip | 125 s clip | notes |
| --- | --- | --- | --- |
| **sherpa-onnx int8 Parakeet v3, CPU** (what runs here) | — | **7.46×** (16.79 s) | 30 s chunks with 2 s overlap; `bench_sherpa.py`; **7.45× on a real 183.6 s film excerpt** |
| Parakeet Redux, CPU *(removed runtime)* | **14.90×** | **13.98×** | +4.3–5.1 s one-time model load; 12.71–13.69× across three gate runs |
| ggml `parakeet-cli` q8_0, Vulkan (GTX 1650) | — | 5.24× (23.94 s) | whisper.cpp's own Parakeet example, already installed here |
| whisper.cpp `ggml-small.en -t 8`, CPU | — | 4.64× (26.97 s) | the reference everyone means by "Whisper on CPU" |
| ggml `parakeet-cli` q8_0, CPU (`-ng`) | — | 4.46× (28.13 s) | same model, plain ggml int8 path |
| ggml `parakeet-cli` **q4_k** 415 MB, CPU | — | 6.21× (20.17 s) | same binary, smaller weights |
| ggml `parakeet-cli` **q4_0** 355 MB, CPU | — | 6.30× (19.88 s) | the smallest ggml quant of this model |
| Parakeet Redux, CUDA (GTX 1650) *(removed runtime)* | 1.95× | 2.48× | **provisional** — timed on a hot box; 1.7–2.1 GB VRAM |
| Parakeet Ultra, CUDA (GTX 1650) *(removed runtime)* | 2.10× | 2.70× | **provisional**, same reason |

The three ggml quants are the experiment that separates **weight size** from **kernel quality**: dropping the
same model from 669 MB (q8_0) to 355 MB (q4_0) — halving it — buys 4.87× → 6.30×, **+29 %**. Size is worth
something, and nowhere near the gap: the ternary build at 178 MB ran **13.98×**, 2.2× the fastest ggml quant, and
the trend across 355/415/669 MB explains at most 1.3× of that. So the lead was the packed ternary GEMM, not a
smaller file — which is also why "just run the open runtime with small weights" does not reach it on this CPU,
and why the permitted path here is the int8 ONNX conversion without the ternary kernels, at ~7.5×. (All six runs
were within one quiet window: load 1.93, 52 °C at the start, 75 °C at the end, so the whole spread sits inside a
single thermal drift; the ordering did not favour the removed runtime, which ran first.)

So on this laptop, comparing transcribe time only: the ternary runtime was ~3× faster than CPU
whisper.cpp-small (8.96 s vs 26.97 s on the same 125 s clip) and ~2.2–3× faster than the best ggml
Parakeet path on this box, and it beat its own CUDA path by a wide margin — on GPU the model ran the
dense form with four PyTorch fallback kernels, where the ggml Vulkan build is at least
self-consistent. Wall-to-wall including the one-time model load the ratio was **1.6×** (16.84 s vs
27.15 s), which is the number to use for a single short file and the wrong one to use for a batch.
The published "113× real time on eight x86 CPU cores" is measured on an AMD EPYC 9575F (Zen 5,
AVX-512 VNNI) — this Comet Lake part has neither the 512-bit nor the 256-bit int8 dot-product
instruction, so quoting 113× for a consumer laptop is off by ~8×.

> **Correction — every earlier local number in this repo's history was void.** The figures that used to
> be here (5.24× on the 10 s clip, 5.63–6.03× on the 125 s clip, and the CUDA row) were taken on a box that had
> climbed to **94 °C** with concurrent load; this i5-10300H power-caps hard under sustained heat. The same
> regression gate reads **5.9× hot and 13.7× cool**, which is a 2.3× measurement error. Everything above was
> re-measured with `loadavg < 2` and the package at **55–58 °C**, and `bench_ggml.sh` refuses to time a
> busy machine at all. If you re-run this, check `cut -d' ' -f1 /proc/loadavg` and the package temperature
> **before** believing any ratio. The one number that was never affected is CI's: a 4-core `ubuntu-latest`
> runner, which has no reason to be hot, read a consistent **10.3–10.6×** on the ternary gate while it existed.

## Why: the kernel path was hardware-gated, not a setting *(measured on the removed runtime)*

The removed engine shipped one CPU ternary GEMM with runtime ISA dispatch. On x86 the paths are AVX-512 VNNI, then
256-bit AVX-VNNI, then AVX2; an Apple silicon machine gets the Metal path, and a CPU with none of them refuses to
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
on this box. Forcing the ISA cannot conjure a path the silicon lacks: AVX2 *is* this CPU's fast path and
forcing it changed nothing, because it was already the selection. The harness that printed this
(`probe_kernels.py`) went with the runtime; its output above is what it printed, and git history has it.

## On CUDA the ternary kernel was not used at all *(removed runtime)*

`resident_form('cuda')` was `dense`: the packed codes are dequantized once into the activation dtype, so a GPU
run loses the 178 MB weight advantage entirely. Four encoder kernels also have no CUDA build for bfloat16 —
the first CUDA transcribe printed (runtime output, verbatim):

```
RuntimeWarning: kestrel-kernels: no native conformer LayerNorm variant for x=(1, 131, 1024) dtype=torch.bfloat16; using PyTorch fallback.
RuntimeWarning: kestrel-kernels: no native conformer residual LayerNorm variant for x=(1, 131, 1024) dtype=torch.bfloat16; using PyTorch fallback.
RuntimeWarning: kestrel-kernels: no native conformer attention variant for qkv=(1, 131, 3072) heads=8x128 dtype=torch.bfloat16; using PyTorch fallback.
RuntimeWarning: kestrel-kernels: no native conformer depthwise variant for x=(1, 131, 1024) taps=(1024, 9) dtype=torch.bfloat16; using PyTorch fallback.
```

The same transcribe on the CPU printed **zero** fallback warnings, so the warning count was a one-line
health check for which path you actually got. Accuracy was unchanged by device (the full-precision build
on CUDA reproduced the synthesised sentence exactly, as did the ternary build on the CPU), so the GPU
meant the same text, half the speed and ~2 GB of VRAM. **The conclusion carries over to the permitted
path**: keep this model family on the CPU.

**Provisional magnitudes:** the CUDA rows above were timed while an unrelated transcribe job held this box
at ~430 % CPU and 94 °C; the CPU side of the same comparison was later re-measured **2× faster** on a quiet
machine (12.6–13.7×, see Results), so read those as directions, not rates. The direction is structural
regardless — no ternary CUDA kernel exists — and is corroborated by the vendor's own engine notes.

## Accuracy notes

- **The permitted path is checked end to end, live**: `check.py` runs one real decode through
  sherpa-onnx on the 10.4 s `Narsil/asr_dummy` fixture and prints what came back —
  `'He hoped there would be stew for dinner, turnips and carrots'`. It asserts that text comes back
  non-empty (an assertion on the exact string would be brittle across fixture re-encodes), and prints
  the rate it achieved without asserting a floor, because a CI runner is not a benchmark box.
- Noise was the ternary model's weak point. Moondream's own card (self-reported): background-noise
  average rises from 6.72 to 9.04 WER versus the original weights, and 0 dB FLEURS-German from 14.45
  to 19.18. **Measured here** with white noise mixed into the 10.4 s fixture at a measured SNR, WER
  against the clean transcript: clean 0.0 %, 10 dB 0.0 %, 5 dB 7.1 %, 0 dB 10.7 % — every error a
  similar-sounding-word substitution (`turnips`→`turnets`, `bruised`→`brewed`, `fattened`→`satin`),
  no dropped or invented content. That local harness (`noise_test.py`) went with the runtime, so the
  four figures are a record, not something CI re-runs; the vendor's numbers above need no runtime to
  read. Different fixture and method from the card's numbers, so read them as a direction, not a
  comparison.
- The ternary build was ahead of the original on the 25-language FLEURS average (10.56 vs 11.62) and on
  long-form TED-LIUM (2.51 vs 2.71), and behind it on the seven English sets (6.55 vs 6.26).

## What independent sources do and do not establish

Separate from everything Moondream reports about itself. Every link is a third party or the vendor's own
docs; where nothing exists, it says so instead of filling the gap.

- **The ternary weights are real and portable.** Independent GGUF ports unpack the ternary codes exactly
  (`w = scale·(code−1)`) and reproduce the runtime's transcripts: [cstr's GGUF](https://huggingface.co/cstr/parakeet-redux-GGUF),
  and a [transcribe.cpp conversion](https://huggingface.co/Nairod785/parakeet-ultra-gguf) reporting encoder
  max |Δ| 3.0e-4 against transformers. Those ports drop the 6 `vad_head.*` tensors, so the built-in VAD
  segmentation was that runtime's own.
- **The 178 MB and the direction of the accuracy table are corroborated**, on Apple, by Fluid Inference's
  Core ML builds ([redux](https://huggingface.co/FluidInference/parakeet-redux-coreml),
  [ultra](https://huggingface.co/FluidInference/parakeet-ultra-coreml)): 183 MB encoder, and Redux ahead of v3
  on multilingual / behind on English. The same page shows Redux **slower** than v3 on the Neural Engine
  (83.9× vs 128.6× RTFx), so "compressed is faster" does not hold on that engine — and the Apple path here is
  untested for the opposite reason (no Mac to test on).
- **Two unrelated runtimes agree on this box.** whisper.cpp 1.9.2's `parakeet-cli` (ggml + Vulkan) loads
  `ggml-org/parakeet-GGUF` q8_0 — 669 MB, 668,757,119 B, exactly the size the card advertises — and returns
  the same sentence as the removed runtime on the 10.4 s fixture: different engine, different quantisation
  (ternary int8 vs q8_0), same words. Note the dialects are not interchangeable: a CrispASR-dialect GGUF of
  the same model is rejected with `failed to load Parakeet model`. `bench_ggml.sh` measures that binary's
  CPU path with the GPU hidden and visible, and refuses to time a busy box — a contended run already produced
  one bogus halved number here.
- **The 113× has no third-party replication.** It is vendor-only, on an EPYC 9575F. The one outward test
  quoted in the launch post is a single X user on a Ryzen 9 9950X3D, and the vendor's own reply to it
  admits they never ran that test.
- **Nobody documents a CPU instruction-set requirement**, yet CPUs without the int8 path crash instead of
  falling back (`Illegal instruction (core dumped)` on a Jetson Nano in the vendor's tracker; a third-party
  runtime states the AVX2 floor for its own build). On this box AVX2 was the ceiling for the ternary
  runtime — measured through its public API before it was removed.
- **The vendor's support matrix lists no Parakeet CPU row and no Redux/Ultra row at all** — support is
  stated for Parakeet v3 BF16 on NVIDIA, and that runtime's docs floor GPUs at Ampere. This repo's CUDA
  numbers come from a Turing GTX 1650, outside that matrix, and they were measured anyway.
- **On a CUDA-visible machine, asking that runtime for the CPU cost ~14 %** (vendor tracker
  [kestrel #259](https://github.com/m87-labs/kestrel/issues/259)): the fused CPU conformer kernels are only
  selected when CUDA is invisible to the process; the stated workaround is `CUDA_VISIBLE_DEVICES=`. Not
  reproduced here — the attempt timed on a contended box, so it stays a reported figure.

## Install (the runtime you can actually use)

```sh
uv venv -p 3.12 .venv
uv pip install --python .venv/bin/python sherpa-onnx numpy    # no torch, no nvidia-* wheels
bash ../vlc-ai-subs/install-parakeet-model.sh                 # the int8 ONNX model, ~641 MB
.venv/bin/python check.py                                    # the gate: guard, contract, live decode
```

```sh
# transcribe (the transcript is stdout, so a host can capture it)
ffmpeg -i meeting.m4a -ac 1 -ar 16000 meeting.wav
.venv/bin/python transcribe_sherpa.py -q meeting.wav

# measure it on your own audio, and confirm the box is quiet first
cut -d' ' -f1 /proc/loadavg; sensors | grep 'Package id 0'
.venv/bin/python bench_sherpa.py models/long12.wav
```

CI runs the install + gate on a clean `ubuntu-latest` runner, so the recipe above is verified on every
push rather than described. It also fails if the proprietary runtime reappears anywhere: not in the
dependency list, not in a source file.

There is deliberately no torch here. The removed runtime needed it — a 178 MB model behind a 1.6–1.7 GB
venv, plus ~3.2 GB of `nvidia-*` wheels if you let pip pick the default wheel — and all of that went with it.

## Files

| file | what it does |
| --- | --- |
| `transcribe_sherpa.py` | **the permitted runtime**: transcribe a 16 kHz mono wav through sherpa-onnx + the k2-fsa int8 ONNX Parakeet — `-q/--quiet` (transcript alone on stdout, status on stderr), `--self-test` for the model layout; one pass, no chunking (a clip past 120 s fails with the reason rather than truncating) |
| `check.py` | the gate: refuses to let the removed runtime back in (source scan + interpreter check), pins the `-q` host contract and the two failure paths (missing model names the installer, a clip past the ceiling is refused), and runs one live decode whenever the model is on disk |
| `bench_sherpa.py` | reproduces a production sherpa-onnx int8 configuration, on the 125 s fixture (7.46× chunked) **and on a real 183.6 s film excerpt (7.45×)** — takes any 16 kHz mono wav as its first argument |
| `setup-sherpa-stt.sh` | **one command** to point Hermes' dictation at the above: finds or builds a python with sherpa-onnx, checks the model, sets `stt.provider` + the command provider, then proves it through Hermes' own `transcribe_audio` and fails if that does not reach the provider |
| `bench_ggml.sh` | the open-runtime cross-check: ggml `parakeet-cli` on ggml-org's own GGUF, CPU vs Vulkan and with the GPU hidden — waits for a quiet machine before timing |
| `bench_quants.sh` | the size-vs-kernel experiment: the same ggml model at q8_0/q4_k/q4_0 through the same binary (answer: halving the file buys 29 %, so the gap was the kernel) |
| `.github/workflows/ci.yml` | the same gate on a clean `ubuntu-latest` box, with sherpa-onnx and no torch |
| `models/` | local fixtures (16 kHz mono wav), not tracked |

## Licensing and attribution

| component | role | licence |
| --- | --- | --- |
| [NVIDIA parakeet-tdt-0.6b-v3](https://huggingface.co/nvidia/parakeet-tdt-0.6b-v3) | the model this repo runs | CC-BY-4.0 |
| [Moondream Parakeet Redux / Ultra](https://huggingface.co/moondream/parakeet-redux) | the ternary and full-precision weights this repo measured | CC-BY-4.0 |
| [k2-fsa sherpa-onnx int8 ONNX conversion](https://github.com/k2-fsa/sherpa-onnx) | the model artefacts the permitted path loads | Apache-2.0 |
| [sherpa-onnx](https://github.com/k2-fsa/sherpa-onnx) | the runtime used here | Apache-2.0 |
| [Narsil/asr_dummy](https://huggingface.co/datasets/Narsil/asr_dummy) | test audio with known transcripts | see dataset card |
| [47thtechcorner/RayCodes_Parakeet_Redux](https://github.com/47thtechcorner/RayCodes_Parakeet_Redux) | the walkthrough this started from | — |

**The weights are CC-BY-4.0; the runtime that used them was not, and it is no longer here.** "Free
local Whisper alternative" describes the price, not the licence: the only runtime that executed these
ternary kernels shipped under a proprietary agreement that also forbids reverse engineering,
deobfuscation or extracting the packed kernel collections, and that grants no licence at all without a
separate written agreement — which is why neither this repo nor the two projects that had shipped it
still contains it. Its measurements are recorded above; its code, installer, venv, CI steps and build
guards are deleted, and `check.py` fails if any of that returns.

For a runtime you can actually use, `transcribe_sherpa.py` does the same job on sherpa-onnx
(Apache-2.0) with the k2-fsa int8 ONNX conversion of the same model — **7.45×** against the ternary
runtime's 9.25–9.92× on real film audio. Roughly 25 % slower; that difference is the price of a
licence that grants you something. Note that the ternary weights are **not** reusable outside their
own runtime, so the permitted path swaps the model conversion too, not just the engine.

Nothing here is vendored: models and wheels are downloaded at install time, so there is no NOTICE/third-party
file to ship.

The scripts in this repository are MIT — see [`LICENSE`](LICENSE), and `SPDX-License-Identifier: MIT` at the
top of each file. That licence covers this repository's own code only: it does not relicense the model
weights, either runtime or the kernels named above, and nothing third-party (code, weights or audio) is
redistributed here — models and fixtures are downloaded at install time.
