# `load_speech_model()` omits `device=`, so any NVIDIA machine silently takes the ~2–2.4× slower path

`app.py:36` calls `md.photon("moondream/parakeet-redux")` with no `device`. (Checked against `app.py` blob `527209b`, repo head `2ad1b71`.)

Moondream's model card documents the default: *"leave it out to take CUDA, then Apple silicon, then the CPU"* — and `moondream 2.5.0` does check `torch.cuda.is_available()` first. Photon ships ternary kernels only for the CPU int8 paths (AVX2 / AVX-VNNI / AVX-512 VNNI) and Apple Metal; on CUDA the weights are dequantized to the dense form and four conformer kernels fall back to plain PyTorch. So the GPU ends up **~2–2.4× slower than the CPU on the same audio**, with no warning.

Measured here (i5-10300H, AVX2 only, GTX 1650 clock-capped at 300 MHz, same 125 s clip):

| device | 125 s clip |
| --- | --- |
| `cpu` | 5.63–6.03× realtime |
| `cuda` | 2.48× realtime |
| whisper.cpp `ggml-small.en -t 8` (reference) | 3.22× realtime |

Fix — pass the device instead of letting Photon guess (needs `import torch`, which `app.py` does not have yet):

```python
device = "mps" if torch.backends.mps.is_available() else "cpu"
return md.photon("moondream/parakeet-redux", device=device)
```

`device="cuda"` still works for anyone who wants it. On a machine whose CPU lacks an int8 matrix-multiply path the engine refuses to load rather than going scalar, so if you want to support such hardware, pick the device by capability instead of by hand — that helper is ~10 lines in `transcribe.py` of the measurement repo linked below (`--device auto`).

Separately, the README's "113× real-time" is measured on an AMD EPYC 9575F (Zen 5, AVX-512 VNNI). A consumer laptop CPU with AVX2 only measures 5–6×: the kernel path is hardware-gated, so no setting reaches 113× on that class of machine.

Method, fixtures and a portable regression gate: https://github.com/chethan62/parakeet-redux-local
