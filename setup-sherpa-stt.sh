#!/usr/bin/env bash
#
# setup-sherpa-stt.sh — wire Hermes' dictation to transcribe_sherpa.py in ONE command.
#
#     bash setup-sherpa-stt.sh
#
# Why: the same job the repo used to do on the proprietary runtime, on a runtime whose licence
# you can actually use — that runtime (kestrel-kernels) is deleted from this repo entirely.
# Redux's weights are CC-BY-4.0 and fine; its runtime (kestrel-kernels, M87 Labs) grants no
# licence without a separate written agreement and forbids unpacking its packed kernels, and
# that bites on USE, not just redistribution. sherpa-onnx is Apache-2.0.
#
# What it does, in order, and stops at the first thing it cannot do safely:
#   1. finds a python with sherpa_onnx (the vlc-ai-subs install provides one), else makes its
#      own venv with uv — never installing into a venv it does not own;
#   2. checks the Parakeet int8 models are on disk and says exactly how to get them if not;
#   3. points Hermes at this directory's transcriber (stt.provider + the command provider);
#   4. proves it end to end through Hermes' own transcribe_audio, on 3 s of generated audio —
#      the assertion is `success: true`, which is what catches a mis-wired provider.
#
# Idempotent: run it again after a Hermes upgrade or a move.
set -euo pipefail

HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
TRANSCRIBER="$HERE/transcribe_sherpa.py"
MODELS="${SHERPA_MODELS_ROOT:-$HOME/.local/share/sherpa-onnx/models}"
VARIANT="sherpa-onnx-nemo-parakeet-tdt-0.6b-v3-int8"
PLUGIN_VENV="$HOME/.local/share/vlc-ai-subs/venv-whisperx/bin/python"
OWN_VENV="$HERE/.venv-sherpa"
PROVIDER="parakeet"

log()  { printf '  → %s\n' "$*"; }
ok()   { printf '  ✓ %s\n' "$*"; }
die()  { printf '  ✗ %s\n' "$*" >&2; exit 1; }

[ -f "$TRANSCRIBER" ] || die "transcribe_sherpa.py must sit next to this script ($HERE)"

# --- 1. a python that has sherpa_onnx --------------------------------------------------------
find_python() {
    local candidates=("$PLUGIN_VENV" "$OWN_VENV/bin/python" "$(command -v python3 || true)")
    local p
    for p in "${candidates[@]}"; do
        [ -x "$p" ] && "$p" -c "import sherpa_onnx" 2>/dev/null && { printf '%s' "$p"; return 0; }
    done
    return 1
}

if PY="$(find_python)"; then
    ok "sherpa_onnx already available: $PY"
else
    command -v uv >/dev/null || die "no python with sherpa_onnx, and uv is not installed (https://docs.astral.sh/uv/)"
    log "creating $OWN_VENV and installing sherpa-onnx + numpy…"
    uv venv --python 3.12 "$OWN_VENV" >/dev/null
    uv pip install --python "$OWN_VENV/bin/python" sherpa-onnx numpy >/dev/null
    PY="$OWN_VENV/bin/python"
    "$PY" -c "import sherpa_onnx" || die "sherpa-onnx install did not take (see $OWN_VENV)"
    ok "sherpa-onnx ready: $PY"
fi

# --- 2. the model ---------------------------------------------------------------------------
if [ ! -f "$MODELS/$VARIANT/encoder.int8.onnx" ]; then
    cat >&2 <<EOF
  ✗ Parakeet int8 model not found at $MODELS/$VARIANT

    Get it with the vlc-ai-subs installer, which knows the right conversion:
        bash install-parakeet-model.sh          # from a chethan62/vlc-ai-subs clone
    or point elsewhere with SHERPA_MODELS_ROOT=/path/to/models and re-run.
EOF
    exit 1
fi
ok "model present: $VARIANT"

# --- 3. Hermes config ------------------------------------------------------------------------
command -v hermes >/dev/null || die "the hermes CLI is not on PATH"
# The command provider is not in the CLI's known-key schema, hence --force; the reader does read it.
# `-o {output_path}` is load-bearing: on stdout alone a genuinely silent recording produces no
# output, and the provider reports "produced no stdout" — indistinguishable from a command that
# never ran. Writing the file (empty when silent) makes the difference visible.
hermes config set --force "stt.providers.$PROVIDER.command" \
    "$PY $TRANSCRIBER -q {input_path} -o {output_path}" >/dev/null
hermes config set "stt.provider" "$PROVIDER" >/dev/null
ok "Hermes: stt.provider=$PROVIDER -> $TRANSCRIBER"

# --- 4. prove it through Hermes' own dispatch --------------------------------------------------
# A real speech fixture is the honest probe. Silence is NOT usable here: Hermes' command provider
# requires *non-empty* output ("non-empty output file > non-empty stdout > RuntimeError"), so a
# silent clip reports as a failure even though the wiring is correct — that rule is Hermes', and
# it is right: with the file option alone, a broken command and a quiet recording look identical.
PROBE="$(find "$HERE/models" -name '*.wav' -printf '%s %p\n' 2>/dev/null | sort -n | head -1 | cut -d' ' -f2-)"
if [ -n "$PROBE" ]; then
    ok "probe fixture: $(basename "$PROBE")"
else
    die "no .wav fixture under $HERE/models to prove the wiring with — run 'bash check.py' once to fetch the fixtures"
fi
AGENT_PY="$(ls -d "$HOME"/.hermes/installs/*/environments/*/venv/bin/python3 2>/dev/null | head -1)"
[ -n "$AGENT_PY" ] || die "could not find the Hermes agent interpreter (looked under ~/.hermes/installs)"
RESULT="$(cd "$HOME/.hermes/hermes-agent" && PYTHONPATH="$PWD" "$AGENT_PY" - "$PROBE" <<'PY'
import json, sys
from tools.transcription_tools import transcribe_audio
r = transcribe_audio(sys.argv[1]) or {}
# The field is `transcript`; reading `.get("text")` yields None and mimics a broken provider.
print(json.dumps({"success": bool(r.get("success")), "provider": r.get("provider"),
                  "chars": len((r.get("transcript") or "").strip()),
                  "error": r.get("error")}))
PY
)"
echo "  → Hermes said: $RESULT"
case "$RESULT" in
    *'"success": true'*) ok "dictation is wired: Hermes reached $PROVIDER and returned a transcript" ;;
    *) die "Hermes did not reach the provider (result above) — check 'hermes config get stt'" ;;
esac
echo
echo "Done. Speak or send a voice note; transcripts come from sherpa-onnx (Apache-2.0)."
