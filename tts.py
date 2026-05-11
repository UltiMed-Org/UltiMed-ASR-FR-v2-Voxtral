#!/usr/bin/env -S uv run --script
# /// script
# requires-python = ">=3.10"
# dependencies = []
# ///
"""Query the crispasr-tts OpenAI-compatible server to synthesize speech.

The container exposes `POST /v1/audio/speech` on host port 8003 (see
docker-compose.yml). Run `tts.py --check` to confirm the server is alive
and the model loaded.

Refs:
  https://github.com/CrispStrobe/CrispASR/issues/86#issuecomment-4422827115
  https://huggingface.co/cstr/qwen3-tts-1.7b-voicedesign-GGUF

Generated with help from Claude Code.
"""
from __future__ import annotations

import argparse
import json
import sys
import urllib.error
import urllib.request
from pathlib import Path

DEFAULT_URL = "http://127.0.0.1:8003"
DEFAULT_INSTRUCTION = "warm calm middle-aged male narrator, neutral accent"


def _get(url: str, timeout: float = 10.0) -> tuple[int, bytes, str]:
    req = urllib.request.Request(url, method="GET")
    try:
        with urllib.request.urlopen(req, timeout=timeout) as r:
            return r.status, r.read(), r.headers.get("Content-Type", "")
    except urllib.error.HTTPError as e:
        return e.code, e.read(), e.headers.get("Content-Type", "")


def _post_json(url: str, body: dict, timeout: float) -> tuple[int, bytes, str]:
    data = json.dumps(body).encode("utf-8")
    req = urllib.request.Request(
        url, data=data, method="POST",
        headers={"Content-Type": "application/json"},
    )
    try:
        with urllib.request.urlopen(req, timeout=timeout) as r:
            return r.status, r.read(), r.headers.get("Content-Type", "")
    except urllib.error.HTTPError as e:
        return e.code, e.read(), e.headers.get("Content-Type", "")


def cmd_check(base_url: str) -> int:
    print(f"GET {base_url}/health", file=sys.stderr)
    try:
        code, body, _ = _get(f"{base_url}/health", timeout=5.0)
    except urllib.error.URLError as e:
        print(f"unreachable: {e}", file=sys.stderr)
        return 2
    print(f"  -> {code}: {body[:200].decode('utf-8', 'replace')}", file=sys.stderr)
    if code != 200:
        return 1

    # Smoke test: synthesize a tiny clip to confirm the model actually loads.
    print(f"POST {base_url}/v1/audio/speech (smoke test)", file=sys.stderr)
    code, body, ctype = _post_json(
        f"{base_url}/v1/audio/speech",
        {
            "input": "Test.",
            "instructions": DEFAULT_INSTRUCTION,
            "response_format": "wav",
        },
        timeout=120.0,
    )
    if code != 200:
        print(f"  -> {code}: {body[:500].decode('utf-8', 'replace')}", file=sys.stderr)
        return 1
    print(f"  -> {code}, {len(body)} bytes ({ctype}); model OK", file=sys.stderr)
    return 0


def cmd_synth(base_url: str, instruction: str, text: str, out: Path,
              response_format: str, speed: float, timeout: float) -> int:
    body = {
        "input": text,
        "instructions": instruction,
        "response_format": response_format,
        "speed": speed,
    }
    url = f"{base_url}/v1/audio/speech"
    print(f"POST {url}", file=sys.stderr)
    try:
        code, payload, ctype = _post_json(url, body, timeout=timeout)
    except urllib.error.URLError as e:
        print(f"unreachable: {e}", file=sys.stderr)
        return 2
    if code != 200:
        print(f"server returned {code}: {payload[:500].decode('utf-8', 'replace')}",
              file=sys.stderr)
        return 1
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_bytes(payload)
    print(f"  -> {code}, {len(payload)} bytes ({ctype})", file=sys.stderr)
    print(str(out.resolve()))
    return 0


def main() -> int:
    p = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    p.add_argument("--url", default=DEFAULT_URL,
                   help=f"Base URL of the crispasr-tts server (default: {DEFAULT_URL}).")
    p.add_argument("--check", action="store_true",
                   help="Probe /health and run a tiny synthesis to confirm the model is loaded.")
    p.add_argument("-i", "--instruction", default=DEFAULT_INSTRUCTION,
                   help=f"Natural-language voice description (default: {DEFAULT_INSTRUCTION!r}).")
    p.add_argument("-t", "--text", help="Text to read aloud.")
    p.add_argument("-o", "--output", default="tts_out.wav",
                   help="Output filename (default: tts_out.wav).")
    p.add_argument("--format", default="wav", choices=("wav", "pcm", "f32"),
                   help="Audio response format (default: wav).")
    p.add_argument("--speed", type=float, default=1.0,
                   help="Speech speed, 0.25 to 4.0 (default: 1.0).")
    p.add_argument("--timeout", type=float, default=300.0,
                   help="HTTP timeout in seconds (default: 300).")
    args = p.parse_args()

    base_url = args.url.rstrip("/")

    if args.check:
        return cmd_check(base_url)

    if not args.text:
        p.error("--text is required unless --check is given")

    return cmd_synth(
        base_url=base_url,
        instruction=args.instruction,
        text=args.text,
        out=Path(args.output),
        response_format=args.format,
        speed=args.speed,
        timeout=args.timeout,
    )


if __name__ == "__main__":
    sys.exit(main())
