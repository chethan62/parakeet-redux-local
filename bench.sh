#!/usr/bin/env bash
# Sequential A/B: whisper.cpp vs Parakeet Redux on the SAME 125s clip, CPU only.
# ponytail: plain shell + /usr/bin/time, no bench framework.
set -euo pipefail
cd "$(dirname "$0")"
W=${W:-~/.local/share/whisper.cpp/models/ggml-small.en.bin}
W=$(eval echo "$W")
mkdir -p "$(dirname "$W")"
[ -f "$W" ] || curl -L -o "$W" "https://huggingface.co/ggerganov/whisper.cpp/resolve/main/$(basename "$W")"

echo "=== whisper.cpp $(basename "$W") (8 threads) ==="
/usr/bin/time -f "wall %e s  user %U s" \
  whisper-cli -m "$W" -f models/long.wav -t 8 -nt 2>&1 | tail -3

echo "=== Parakeet Redux (CPU, default threads) ==="
/usr/bin/time -f "wall %e s  user %U s" \
  .venv/bin/python transcribe.py models/long.wav 2>&1 | grep -E "model load|realtime|wall"
