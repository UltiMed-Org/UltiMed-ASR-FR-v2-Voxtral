#!/usr/bin/env -S uv run --script
# /// script
# requires-python = ">=3.10"
# dependencies = []
# ///
"""Check that a `tts.py --batch` row's own knobs reach the request, and only its own.

Run it directly: `uv run voxtral/cloning/tests/check_batch_overrides.py`.

Like its siblings in this directory, this is not a test suite (this repo has none, per
CLAUDE.md) but a pin on one specific thing that can silently stop working.

The thing: a sweep over cfg_alpha or voices is written as one JSONL file where every row
carries its own value, and nothing in the wav tells you whether that value was honoured. A
per-row knob that quietly fell back to the global flag produces 160 files that look like a
result and are all the same experiment. That failure mode is not hypothetical here, it is
the house speciality: stock vllm-omni declares no `model_config` on its speech request, so
pydantic drops unknown fields and still answers 200 (see CLAUDE.md on the `perso` fork).

So this pins four things, against a fake server that records what it was actually POSTed
rather than against a mock of our own request builder:

1. a row's `voice` and `cfg_alpha` arrive, and cfg_alpha arrives *inside* `extra_params`,
   which is the routing tts.py deliberately chose because it also works on a stock server;
2. an override does not leak into the next row (the namespace is copied, not mutated);
3. `noise_seed` keeps its own key name on the wire (`voxtral_noise_seed`), since the flag
   name and the field name differ and an override bypassing build_speech_body would send
   the wrong one;
4. a misspelled knob (`cfg_alpa`) is refused, not ignored, because ignoring it is exactly
   the 160-identical-files outcome above.

No GPU and no container: the fake server returns a 44-byte silent WAV, which is all
_synthesize needs to call a row done.
"""
import json
import subprocess
import sys
import tempfile
import threading
import wave
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from io import BytesIO
from pathlib import Path

# Derived, never hardcoded: an absolute home path in a tracked file leaks the username.
REPO = Path(__file__).resolve().parents[3]
TTS = REPO / "tts.py"

received: list[dict] = []
lock = threading.Lock()


def silent_wav() -> bytes:
    """Return a valid one-frame 24 kHz mono WAV, so `wav_seconds` parses it."""
    buf = BytesIO()
    with wave.open(buf, "wb") as w:
        w.setnchannels(1)
        w.setsampwidth(2)
        w.setframerate(24000)
        w.writeframes(b"\x00\x00")
    return buf.getvalue()


class Handler(BaseHTTPRequestHandler):
    """Accept /v1/audio/speech, record the decoded body, answer with audio."""

    def do_POST(self) -> None:  # BaseHTTPRequestHandler dispatches on this spelling
        body = self.rfile.read(int(self.headers.get("Content-Length", 0)))
        with lock:
            received.append(json.loads(body))
        payload = silent_wav()
        self.send_response(200)
        self.send_header("Content-Type", "audio/wav")
        self.send_header("Content-Length", str(len(payload)))
        self.end_headers()
        self.wfile.write(payload)

    def log_message(self, *_args) -> None:
        """Silence the per-request stderr line; the checks below are the output."""


failures: list[str] = []


def check(label: str, ok: bool, detail: str = "") -> None:
    """Record one check, print it, and keep going so one run reports everything."""
    print(f"  {'ok  ' if ok else 'FAIL'}  {label}{'' if ok else f'  <- {detail}'}")
    if not ok:
        failures.append(label)


server = ThreadingHTTPServer(("127.0.0.1", 0), Handler)
port = server.server_address[1]
threading.Thread(target=server.serve_forever, daemon=True).start()

# One row per thing being pinned. Concurrency 1 so `received` is in file order, which is
# what lets check 2 below talk about "the next row".
rows = [
    {"input": "Un.", "output": "a.wav", "voice": "fr_male", "cfg_alpha": 2.5},
    {"input": "Deux.", "output": "b.wav"},
    {"input": "Trois.", "output": "c.wav", "noise_seed": 7},
    {"input": "Quatre.", "output": "d.wav", "cfg_alpa": 2.5},
]

with tempfile.TemporaryDirectory() as tmp:
    tmp = Path(tmp)
    jsonl = tmp / "rows.jsonl"
    jsonl.write_text("".join(json.dumps(r) + "\n" for r in rows))
    proc = subprocess.run(
        [sys.executable, str(TTS), "--batch", str(jsonl), "-o", str(tmp / "out"),
         "--url", f"http://127.0.0.1:{port}", "--concurrency", "1",
         "--voice", "fr_female", "--cfg-alpha", "1.2"],
        capture_output=True, text=True, check=False,
    )
    server.shutdown()
    print(f"== tts.py --batch exited {proc.returncode}")

    bad_row_refused = ("cfg_alpa" in proc.stderr and "row skipped" in proc.stderr
                       and not (tmp / "out" / "d.wav").exists())
    files = sorted(p.name for p in (tmp / "out").glob("*.wav"))

check("three good rows synthesized", files == ["a.wav", "b.wav", "c.wav"], str(files))
check("three bodies received", len(received) == 3, str(len(received)))

if len(received) == 3:
    first, second, third = received
    check("row 1 voice overrides the --voice flag",
          first.get("voice") == "fr_male", repr(first.get("voice")))
    check("row 1 cfg_alpha rides inside extra_params",
          (first.get("extra_params") or {}).get("cfg_alpha") == 2.5,
          json.dumps(first.get("extra_params")))
    check("row 2 falls back to the flags (no leak from row 1)",
          second.get("voice") == "fr_female"
          and (second.get("extra_params") or {}).get("cfg_alpha") == 1.2,
          f"voice={second.get('voice')!r} extra={json.dumps(second.get('extra_params'))}")
    check("row 3 noise_seed travels as voxtral_noise_seed",
          (third.get("extra_params") or {}).get("voxtral_noise_seed") == 7,
          json.dumps(third.get("extra_params")))

check("misspelled knob refused, not ignored", bad_row_refused,
      f"rc={proc.returncode} stderr tail={proc.stderr[-300:]!r}")
check("exit code non-zero when a row was refused", proc.returncode == 1,
      str(proc.returncode))

print(f"\n{len(failures)} failed" if failures else "\nall checks passed")
sys.exit(1 if failures else 0)
