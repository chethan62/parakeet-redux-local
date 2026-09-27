#!/usr/bin/env bash
# SPDX-License-Identifier: MIT
# Two independent cross-checks, run only when the box is quiet (a contended box halved our numbers once:
# an unrelated 430%-CPU transcribe job plus 94C package temperature).
#
#   1. ggml parakeet-cli (whisper.cpp 1.9.2, already installed) on ggml-org's own GGUF - CPU vs Vulkan.
#      This is the unverified step from the research pass: does that GGUF even load in 1.9.2?
#   2. Photon with CUDA hidden vs visible, to test kestrel issue #259 on this box (claims ~14%).
set -uo pipefail
CLI="$HOME/.local/share/whisper-cpp/parakeet-cli"
GGUF="$HOME/models/parakeet/ggml-parakeet-tdt-0.6b-v3-q8_0.bin"
URL="https://huggingface.co/ggml-org/parakeet-GGUF/resolve/main/ggml-parakeet-tdt-0.6b-v3-q8_0.bin"
REPO="$HOME/projects/parakeet-redux"
CLIP="$REPO/models/long.wav"

# 1. Fetch (network-bound, harmless while the box is busy).
if [ ! -s "$GGUF" ]; then
  mkdir -p "$(dirname "$GGUF")"
  echo "downloading q8_0 (~669 MB) ..."
  curl -L --fail --retry 3 -o "$GGUF" "$URL" || { echo "DOWNLOAD FAILED"; exit 1; }
fi
ls -l "$GGUF"

# 2. Wait for the box to go quiet (cap ~90 min), so the timings mean something.
for i in $(seq 1 90); do
  L=$(cut -d' ' -f1 /proc/loadavg)
  awk -v l="$L" 'BEGIN{exit !(l < 2.0)}' && { echo "box quiet (load $L) after ${i} min"; break; }
  [ $((i % 5)) -eq 0 ] && echo "waiting: load $L, temp $(sensors 2>/dev/null | awk '/Package id 0/{print $4}')"
  sleep 60
done
echo "peak temp before runs: $(sensors 2>/dev/null | awk '/Package id 0/{print $4}')"

echo "=== 1a. ggml CPU (-ng -t 8) ==="
/usr/bin/time -f "WALL %e s  MAXRSS %M KB" \
  "$CLI" -ng -t 8 -ps -otxt -of /tmp/pk_cpu -m "$GGUF" -f "$CLIP" 2>&1 | tail -6
echo "transcript head: $(head -c 160 /tmp/pk_cpu.txt 2>/dev/null)"
echo "segment lines: $(wc -l < /tmp/pk_cpu.txt 2>/dev/null)"

echo "=== 1b. ggml VULKAN (GTX 1650, -dev 0 -t 8) ==="
/usr/bin/time -f "WALL %e s  MAXRSS %M KB" \
  "$CLI" -t 8 -np -dev 0 -m "$GGUF" -f "$CLIP" 2>&1 | tail -6

echo "=== 2. Photon: does hiding CUDA matter on this box? (kestrel #259) ==="
cd "$REPO"
for tag in visible hidden visible2; do
  [ "$tag" = hidden ] && PRE="CUDA_VISIBLE_DEVICES=" || PRE=""
  printf '%-9s ' "$tag"
  env $PRE .venv/bin/python check.py 2>&1 | grep -oE "[0-9.]+x realtime on 125s" || echo "(gate failed)"
done
echo "temp after: $(sensors 2>/dev/null | awk '/Package id 0/{print $4}')"
