#!/usr/bin/env -S uv run --script
# /// script
# requires-python = ">=3.10"
# dependencies = []
# ///
"""Drive the crispasr-tts docker container to synthesize speech.

Refs:
  https://huggingface.co/cstr/qwen3-tts-1.7b-voicedesign-GGUF
  https://github.com/CrispStrobe/CrispASR

Generated with help from Claude Code.
"""
from __future__ import annotations

import argparse
import subprocess
import sys
from pathlib import Path

CONTAINER = "crispasr-tts"
TALKER = "/models/qwen3-tts-12hz-1.7b-voicedesign-q8_0.gguf"
CODEC = "/models/qwen3-tts-tokenizer-12hz.gguf"
# /cache is bind-mounted from ./cache on the host
CONTAINER_CACHE = "/cache"
HOST_CACHE = Path(__file__).resolve().parent / "cache"


def main() -> int:
    p = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    p.add_argument("-i", "--instruction", required=True,
                   help="Natural-language voice description (e.g. 'warm calm middle-aged male narrator').")
    p.add_argument("-t", "--text", required=True, help="Text to read aloud.")
    p.add_argument("-o", "--output", default="tts_out.wav",
                   help="Output WAV filename (written under ./cache/). Default: tts_out.wav")
    p.add_argument("--container", default=CONTAINER, help=f"Container name (default: {CONTAINER}).")
    args = p.parse_args()

    out_name = Path(args.output).name
    in_container = f"{CONTAINER_CACHE}/{out_name}"
    host_path = HOST_CACHE / out_name

    cmd = [
        "sudo", "docker", "exec", args.container,
        "crispasr",
        "--backend", "qwen3-tts-1.7b-voicedesign",
        "-m", TALKER,
        "--codec-model", CODEC,
        "--instruct", args.instruction,
        "--tts", args.text,
        "--tts-output", in_container,
    ]

    print(f"$ {' '.join(cmd)}", file=sys.stderr)
    r = subprocess.run(cmd)
    if r.returncode != 0:
        print(f"crispasr exited with {r.returncode}", file=sys.stderr)
        return r.returncode

    if host_path.exists():
        print(str(host_path))
    else:
        print(f"warning: expected output at {host_path} but it was not found", file=sys.stderr)
        return 1
    return 0


if __name__ == "__main__":
    sys.exit(main())
