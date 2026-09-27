#!/usr/bin/env bash
# SPDX-License-Identifier: MIT
# Sequential A/B: whisper.cpp vs Parakeet Redux on the SAME 125s clip, CPU only.
# ponytail: plain shell + /usr/bin/time, no bench framework.
set -euo pipefail
cd "$(dirname "$0")"
W=${W:-~/.local/share/whisper.cpp/models/ggml-small.en.bin}
W=$(eval echo "$W")
mkdir -p "$(dirname "$W")"
[ -f "$W" ] || curl -L -o "$W" "https://huggingface.co/ggerganov/whisper.cpp/resolve/main/$(basename "$W")"

# Same fixture recipe as the gate: 12x the public 10.4s flac as 16 kHz mono (re-encoded, so the
# header duration is truthful - -c copy would keep the first file's 10.4s and lie to the RTF maths).
CLIP=$(.venv/bin/python -c "from check import fixture, long_clip; print(long_clip(fixture()))")

echo "=== whisper.cpp $(basename "$W") (8 threads) ==="
/usr/bin/time -f "wall %e s  user %U s" \
  whisper-cli -m "$W" -f "$CLIP" -t 8 -nt 2>&1 | tail -3

echo "=== Parakeet Redux (CPU, default threads) ==="
/usr/bin/time -f "wall %e s  user %U s" \
  .venv/bin/python transcribe.py "$CLIP" --rtf 2>&1 | grep -E "device=|model load|realtime|wall"
