#!/usr/bin/env python3
# SPDX-License-Identifier: MIT
"""Gate for the permitted runtime (sherpa-onnx).

Every check here runs without the 641 MB model, so CI and a fresh clone can run them;
the decode arm runs only where the model is on disk. Asserts, not a framework.

Was `check.py` for the removed Photon/kestrel runtime (baseline transcript, word timings,
12 reps, speed floor). That runtime is gone: `kestrel-kernels` is licensed only under a
separate written agreement and forbids reverse engineering / unpacking its packed kernels,
so it is not installed, not invoked and not imported here any more. The measurements it
produced are kept in the README as the reason nothing open reaches that speed.
"""
import importlib.util
import os
import subprocess
import sys
import tempfile
import time
import wave

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)

FAILED = []

# The licence guard runs BEFORE anything from this repo is imported. If a source file ever drives
# the proprietary runtime again, that import raises here and the gate would die with a traceback
# instead of naming the file — measured by appending `import moondream` to a script: the first
# version of this gate crashed at import rather than reporting the guard failure.
GUARD_SKIP = (".git", ".venv", ".venv-sherpa", "__pycache__")
GUARD_PATTERNS = ("import moondream", "import kestrel", "from kestrel", "pip install moondream")


def scan_sources():
    hits = []
    for root, dirs, files in os.walk(HERE):
        dirs[:] = [d for d in dirs if d not in GUARD_SKIP]
        for f in files:
            # this file *defines* the guard; its own pattern strings are not imports
            if f == "check.py" or not (f.endswith(".py") or f.endswith(".sh")):
                continue
            with open(os.path.join(root, f), encoding="utf-8", errors="replace") as fh:
                for i, line in enumerate(fh, 1):
                    if any(p in line for p in GUARD_PATTERNS):
                        hits.append(f"{os.path.relpath(os.path.join(root, f), HERE)}:{i}")
    return hits


GUARD_HITS = scan_sources()
try:
    import transcribe_sherpa as T  # noqa: E402
    IMPORT_ERROR = ""
except ImportError as e:
    T = None
    IMPORT_ERROR = f"{e} — the guard above names the file responsible"


def check(name, ok, detail=""):
    print(f"  {'✓' if ok else '✗'} {name}{(' — ' + detail) if detail else ''}")
    if not ok:
        FAILED.append(name)


def silence_wav(path, seconds):
    with wave.open(path, "w") as w:
        w.setnchannels(1)
        w.setsampwidth(2)
        w.setframerate(T.SAMPLE_RATE)
        w.writeframes(b"\x00\x00" * int(T.SAMPLE_RATE * seconds))
    return path


def run_main(argv):
    """Call transcribe_sherpa.main() with argv patched. Returns (exit, message)."""
    old = sys.argv
    sys.argv = ["transcribe_sherpa.py"] + argv
    try:
        T.main()
        return 0, ""
    except SystemExit as e:
        return e.code if isinstance(e.code, int) else 1, str(e)
    finally:
        sys.argv = old


def main() -> int:
    # 1. the licence guard: nothing in the tree may drive the proprietary runtime.
    check("no source imports the proprietary runtime", not GUARD_HITS, ", ".join(GUARD_HITS) or "clean")

    # 2. and it is not installed in the interpreter running this gate.
    installed = [m for m in ("moondream", "kestrel", "kestrel_kernels")
                 if importlib.util.find_spec(m) is not None]
    check("gate interpreter has no proprietary runtime", not installed, ", ".join(installed) or "clean")

    if T is None:  # a source file drove the runtime: say so, do not traceback
        check("transcribe_sherpa.py is importable", False, IMPORT_ERROR)
        print(f"\nFAILED: {', '.join(FAILED)}")
        return 1
    assert T is not None  # the guard above returned if it were not

    # 3. the host contract: -q/--quiet is what puts the transcript alone on stdout.
    out = subprocess.run([sys.executable, os.path.join(HERE, "transcribe_sherpa.py"), "--help"],
                         capture_output=True, text=True)
    check("--help exits 0 and documents -q/--quiet",
          out.returncode == 0 and "--quiet" in out.stdout, f"exit {out.returncode}")

    with tempfile.TemporaryDirectory() as tmp:
        wav = silence_wav(os.path.join(tmp, "3s.wav"), 3)

        # 4. a model that is not on disk fails with the actionable command, not a traceback.
        T.MODELS_ROOT = os.path.join(tmp, "empty")
        code, msg = run_main(["-q", wav])
        check("missing model names the installer",
              "install-parakeet-model.sh" in msg, msg.splitlines()[0] if msg else f"exit {code}")

        # 5. past the one-pass ceiling the clip is refused by name (never truncated silently).
        #    model_dir() only checks that the four files exist, so a stub model dir exercises the
        #    ceiling path without the 641 MB download — which keeps this check alive in CI.
        stub = os.path.join(tmp, "model", T.VARIANTS["v3"])
        os.makedirs(stub)
        for f in T.FILES:
            open(os.path.join(stub, f), "w").close()
        T.MODELS_ROOT = os.path.join(tmp, "model")
        long_wav = silence_wav(os.path.join(tmp, "121s.wav"), 121)
        try:
            code, msg = run_main(["-q", long_wav])
        except Exception as e:
            # With the ceiling guard removed this reaches a decode against the stub model and the
            # ONNX runtime raises. Report that as the failed check it is — the negative control for
            # this check was run by raising MAX_SECONDS, and it must not read as a crash.
            code, msg = 1, f"no ceiling guard: {type(e).__name__} reached a decode"
        check("clip past the ceiling is refused, not truncated",
              "ceiling" in msg, msg.splitlines()[0] if msg else f"exit {code}")

    # 6. the decode arm: only where the model is installed. Deliberately not asserting on the
    #    text of silence (it legitimately decodes to nothing) — this asserts the call round-trips
    #    on real speech, which is what fails if the model files or the sherpa API contract drift.
    fixtures = [os.path.join(HERE, "models", f) for f in ("1.16k.wav", "2.16k.wav")]
    fixtures = [f for f in fixtures if os.path.isfile(f)]
    if not fixtures:
        print("  – decode arm skipped: no fixture in models/ and no model on disk")
    else:
        T.MODELS_ROOT = os.path.expanduser("~/.local/share/sherpa-onnx/models")
        try:
            d = T.model_dir("v3")
        except SystemExit:
            print("  – decode arm skipped: parakeet v3 int8 model not installed")
        else:
            samples = T.load_wav(fixtures[0])
            t0 = time.time()
            text = T.transcribe(samples, d, 2)
            dt = time.time() - t0
            dur = len(samples) / T.SAMPLE_RATE
            check("real decode through sherpa-onnx returns text", bool(text.strip()),
                  f"{dur:.1f}s in {dt:.2f}s ({dur / dt:.1f}x) → {text[:60]!r}")

    print(f"\n{'FAILED: ' + ', '.join(FAILED) if FAILED else 'all checks passed'}")
    return 1 if FAILED else 0


if __name__ == "__main__":
    raise SystemExit(main())
