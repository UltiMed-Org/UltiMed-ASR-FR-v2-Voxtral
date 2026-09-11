#!/usr/bin/env -S uv run --script
# /// script
# requires-python = ">=3.10"
# dependencies = []
# ///
"""Query the crispasr (ASR) OpenAI-compatible server to transcribe audio.

The container exposes `POST /v1/audio/transcriptions` on host port 8002 (see
docker-compose.yml), a drop-in replacement for the OpenAI Whisper API. Run
`asr.py --check` to confirm the server is alive and the model loaded.

This is the transcription counterpart to tts.py: it stays deliberately general
so any CrispASR ASR backend (parakeet, whisper, voxtral, canary, firered, ...)
can be exercised the same way as new models land in a CrispASR release. Every
knob below is a form field the server documents (docs/server.md); anything left
unset is simply not sent, so the server keeps the default it was started with.

Auth: the crispasr ASR service in docker-compose.yml sets CRISPASR_API_KEYS, so
`/v1/audio/transcriptions` is protected (only `/health` is public). Pass the key
with --api-key or the CRISPASR_API_KEY env var; it is sent as
`Authorization: Bearer <key>`. If no key is given the request is made without
one (fine for a server started without CRISPASR_API_KEYS).

Refs:
  CrispASR docs/server.md (/v1/audio/transcriptions), docs/cli.md
  https://platform.openai.com/docs/api-reference/audio/createTranscription

Generated with help from Claude Code.
"""
from __future__ import annotations

import argparse
import json
import mimetypes
import os
import sys
import urllib.error
import urllib.request
from pathlib import Path

DEFAULT_URL = "http://127.0.0.1:8002"
# Bundled clip used by --check to prove the model actually transcribes, not just
# that /health answers. Any file under asr_audio_test_files/ works; jfk.mp3 is
# the canonical short English sample.
CHECK_CLIP = Path(__file__).resolve().parent / "asr_audio_test_files" / "jfk.mp3"

# response_format values the server accepts; the JSON-shaped ones are pretty
# printed on stdout, the rest (text/srt/vtt) are emitted verbatim.
JSON_FORMATS = ("json", "verbose_json", "diarized_json")
TEXT_FORMATS = ("text", "srt", "vtt")


def _headers(api_key: str | None) -> dict:
    return {"Authorization": f"Bearer {api_key}"} if api_key else {}


def _get(url: str, api_key: str | None, timeout: float = 10.0) -> tuple[int, bytes, str]:
    req = urllib.request.Request(url, method="GET", headers=_headers(api_key))
    try:
        with urllib.request.urlopen(req, timeout=timeout) as r:
            return r.status, r.read(), r.headers.get("Content-Type", "")
    except urllib.error.HTTPError as e:
        return e.code, e.read(), e.headers.get("Content-Type", "")


def _encode_multipart(fields: dict, file_path: Path) -> tuple[bytes, str]:
    """Build a multipart/form-data body (stdlib only) from string `fields` plus
    the audio `file`. Returns (body, content_type_header)."""
    boundary = "----crispasr-asr-" + os.urandom(16).hex()
    crlf = b"\r\n"
    parts: list[bytes] = []
    for name, value in fields.items():
        parts += [
            f"--{boundary}".encode(),
            f'Content-Disposition: form-data; name="{name}"'.encode(),
            b"",
            str(value).encode("utf-8"),
        ]
    mime = mimetypes.guess_type(file_path.name)[0] or "application/octet-stream"
    parts += [
        f"--{boundary}".encode(),
        f'Content-Disposition: form-data; name="file"; filename="{file_path.name}"'.encode(),
        f"Content-Type: {mime}".encode(),
        b"",
        file_path.read_bytes(),
    ]
    parts += [f"--{boundary}--".encode(), b""]
    body = crlf.join(parts)
    return body, f"multipart/form-data; boundary={boundary}"


def transcribe(
    base_url: str,
    audio: Path,
    fields: dict,
    api_key: str | None,
    timeout: float,
) -> tuple[int, bytes, str]:
    """POST `audio` + form `fields` to /v1/audio/transcriptions."""
    body, ctype = _encode_multipart(fields, audio)
    headers = _headers(api_key)
    headers["Content-Type"] = ctype
    req = urllib.request.Request(
        f"{base_url}/v1/audio/transcriptions", data=body, method="POST", headers=headers,
    )
    try:
        with urllib.request.urlopen(req, timeout=timeout) as r:
            return r.status, r.read(), r.headers.get("Content-Type", "")
    except urllib.error.HTTPError as e:
        return e.code, e.read(), e.headers.get("Content-Type", "")


def build_fields(args: argparse.Namespace) -> dict:
    """Collect the OpenAI/CrispASR form fields, sending only what was set so the
    server keeps its own startup default for every omitted knob."""
    optional = {
        # Sent like every other knob: only when asked for. Recent CrispASR builds
        # reject a transcription request that names no model with a 422, so this is
        # usually required in practice even though the client never assumes it.
        "model": args.model,
        "response_format": args.format,
        "language": args.language,
        "prompt": args.prompt,
        "temperature": args.temperature,
        "translate": _bool(args.translate),
        "diarize": _bool(args.diarize),
        "diarize_method": args.diarize_method,
        "hotwords": args.hotwords,
        "hotwords_boost": args.hotwords_boost,
        "no_timestamps": _bool(args.no_timestamps),
    }
    return {k: v for k, v in optional.items() if v is not None}


def _bool(v: bool | None) -> str | None:
    """argparse BooleanOptionalAction leaves None when unset; map True/False to
    the server's expected 'true'/'false' strings, and None to omitted."""
    return None if v is None else ("true" if v else "false")


def _emit(code: int, payload: bytes, ctype: str, out: Path | None) -> int:
    if code != 200:
        print(f"server returned {code}: {payload[:500].decode('utf-8', 'replace')}",
              file=sys.stderr)
        return 1
    if out is not None:
        out.parent.mkdir(parents=True, exist_ok=True)
        out.write_bytes(payload)
        print(f"  -> {len(payload)} bytes ({ctype}) written to {out.resolve()}", file=sys.stderr)
        return 0
    # stdout: pretty-print JSON responses, pass text/srt/vtt through untouched.
    text = payload.decode("utf-8", "replace")
    if "json" in ctype:
        try:
            print(json.dumps(json.loads(text), ensure_ascii=False, indent=2))
            return 0
        except json.JSONDecodeError:
            pass
    print(text)
    return 0


def cmd_check(base_url: str, api_key: str | None) -> int:
    print(f"GET {base_url}/health", file=sys.stderr)
    try:
        code, body, _ = _get(f"{base_url}/health", api_key, timeout=5.0)
    except urllib.error.URLError as e:
        print(f"unreachable: {e}", file=sys.stderr)
        return 2
    print(f"  -> {code}: {body[:200].decode('utf-8', 'replace')}", file=sys.stderr)
    if code != 200:
        return 1

    if not CHECK_CLIP.exists():
        print(f"  (skipping transcription smoke test: {CHECK_CLIP} not found)", file=sys.stderr)
        return 0
    print(f"POST {base_url}/v1/audio/transcriptions (smoke test: {CHECK_CLIP.name})", file=sys.stderr)
    code, payload, ctype = transcribe(
        base_url, CHECK_CLIP, {"response_format": "text"}, api_key, timeout=120.0,
    )
    if code != 200:
        print(f"  -> {code}: {payload[:500].decode('utf-8', 'replace')}", file=sys.stderr)
        return 1
    print(f"  -> {code}, {len(payload)} bytes ({ctype}); model OK", file=sys.stderr)
    print(f"  transcript: {payload[:200].decode('utf-8', 'replace').strip()!r}", file=sys.stderr)
    return 0


def main() -> int:
    p = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    p.add_argument("audio", nargs="?", help="Audio file to transcribe.")
    p.add_argument("--url", default=DEFAULT_URL,
                   help=f"Base URL of the crispasr ASR server (default: {DEFAULT_URL}).")
    p.add_argument("--api-key", default=os.environ.get("CRISPASR_API_KEY"),
                   help="Bearer key for the protected endpoint (default: $CRISPASR_API_KEY). "
                        "The ASR service sets CRISPASR_API_KEYS in docker-compose.yml.")
    p.add_argument("--check", action="store_true",
                   help="Probe /health and transcribe a tiny bundled clip to confirm the model loaded.")
    p.add_argument("-o", "--output", default=None,
                   help="Write the raw response to this file (default: print to stdout).")
    p.add_argument("--model", default=os.environ.get("CRISPASR_MODEL"),
                   help="Model name to transcribe with (env: CRISPASR_MODEL). Newer "
                        "servers return 422 when it is missing.")
    p.add_argument("--format", default="text", choices=JSON_FORMATS + TEXT_FORMATS,
                   help="response_format (default: text). diarized_json adds per-segment speakers.")
    p.add_argument("-l", "--language", default=None,
                   help="ISO-639-1 code (default: the server's -l setting).")
    p.add_argument("--prompt", default=None, help="Initial prompt / context to bias decoding.")
    p.add_argument("--temperature", type=float, default=None, help="Sampling temperature (server default 0.0).")
    p.add_argument("--timeout", type=float, default=600.0, help="HTTP timeout in seconds (default: 600).")

    diar = p.add_argument_group("diarization", "Speaker labelling (needs a diarize-capable setup).")
    diar.add_argument("--diarize", action=argparse.BooleanOptionalAction, default=None,
                      help="Enable speaker diarization (pairs well with --format diarized_json).")
    diar.add_argument("--diarize-method", default=None,
                      choices=("energy", "xcorr", "vad-turns", "pyannote", "sherpa"),
                      help="Diarization method (server default: energy). pyannote works on mono.")

    p.add_argument("--translate", action=argparse.BooleanOptionalAction, default=None,
                   help="Translate to English (backends with CAP_TRANSLATE, e.g. whisper/canary).")
    p.add_argument("--hotwords", default=None,
                   help="Comma-separated hotwords to bias decoding (e.g. 'CrispASR,GGUF,parakeet').")
    p.add_argument("--hotwords-boost", type=float, default=None,
                   help="Log-prob boost per hotword match (server default 2.0).")
    p.add_argument("--no-timestamps", action=argparse.BooleanOptionalAction, default=None,
                   help="Omit timestamps from the output.")
    args = p.parse_args()

    base_url = args.url.rstrip("/")

    if args.check:
        return cmd_check(base_url, args.api_key)

    if not args.audio:
        p.error("an audio file is required unless --check is given")
    audio = Path(args.audio)
    if not audio.exists():
        p.error(f"audio file not found: {audio}")

    print(f"POST {base_url}/v1/audio/transcriptions  ({audio.name}, {args.format})", file=sys.stderr)
    try:
        code, payload, ctype = transcribe(
            base_url, audio, build_fields(args), args.api_key, timeout=args.timeout,
        )
    except urllib.error.URLError as e:
        print(f"unreachable: {e}", file=sys.stderr)
        return 2
    return _emit(code, payload, ctype, Path(args.output) if args.output else None)


if __name__ == "__main__":
    sys.exit(main())
