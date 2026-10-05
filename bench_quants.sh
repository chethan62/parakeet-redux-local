#!/usr/bin/env bash
# SPDX-License-Identifier: MIT
# Where does the removed runtime's ~3x lead come from - its ternary kernel, or just 178 MB of weights?
# Runs the same clip through the same ggml binary at three quantisation sizes:
#   q8_0 669 MB (measured: 4.46x)   q4_k 416 MB   q4_0 356 MB   — vs Photon ternary 178 MB = 13.98x
# If the small quants stay near 4.5x, the speed is the engine's avx2 kernel, not the weight size.
set -uo pipefail
CLI="$HOME/.local/share/whisper-cpp/parakeet-cli"
CLIP="$HOME/projects/parakeet-redux/models/long.wav"
D="$HOME/models/parakeet"

# Wait for the sherpa bench (pid 130891) to finish AND the box to be quiet: two timed jobs sharing
# one machine is how the earlier numbers in this repo got corrupted.
while kill -0 130891 2>/dev/null; do echo "waiting for the sherpa bench (pid 130891)"; sleep 30; done
for i in $(seq 1 60); do
  L=$(cut -d' ' -f1 /proc/loadavg)
  awk -v l="$L" 'BEGIN{exit !(l < 2.0)}' && { echo "quiet (load $L)"; break; }
  [ $((i % 5)) -eq 0 ] && echo "waiting: load $L temp $(sensors 2>/dev/null | awk '/Package id 0/{print $4}')"
  sleep 60
done
echo "temp before: $(sensors 2>/dev/null | awk '/Package id 0/{print $4}')  load: $(cat /proc/loadavg)"

for q in q4_k q4_0 q8_0; do
  F="$D/ggml-parakeet-tdt-0.6b-v3-$q.bin"
  [ -s "$F" ] || { echo "$q: missing"; continue; }
  printf '%-6s %4s MB  ' "$q" "$(( $(stat -c%s "$F") / 1000000 ))"
  OUT=$(/usr/bin/time -f "WALL %e" "$CLI" -ng -t 8 -np -otxt -of "/tmp/pk_$q" -m "$F" -f "$CLIP" 2>&1 | tail -1)
  W=$(echo "$OUT" | awk '{print $2}')
  awk -v w="$W" -v q="$q" 'BEGIN{printf "%s s = %.2fx realtime\n", w, 125.22/w}'
  head -c 90 "/tmp/pk_$q.txt" 2>/dev/null | tr -d '\n'; echo
done
echo "temp after: $(sensors 2>/dev/null | awk '/Package id 0/{print $4}')"
