#!/usr/bin/env -S uv run --script
# /// script
# requires-python = ">=3.10"
# dependencies = []
# ///
"""Query the crispasr-tts OpenAI-compatible server to synthesize speech.

The container exposes `POST /v1/audio/speech` on host port 8003 (see
docker-compose.yml). Run `tts.py --check` to confirm the server is alive
and the model loaded.

The deployed backend is HumeAI TADA 3B multilingual (French), so this
client exposes TADA's per-request talker-sampling knobs. As of CrispASR
#197 the talker samples by default (matching upstream `InferenceOptions`:
do_sample=on, temperature=0.6, top_p=0.9, top_k=0, repetition_penalty=1.1)
instead of pure greedy argmax, which fixed the looping / word-dropping /
garbled-liaison hallucinations on French. Those knobs, plus the duration
flow-matching `num_candidates`, are now tunable at query time without
restarting the container. See PR.md and CrispASR docs/tts.md
("Talker text sampling") + docs/server.md for the field contract.

Any knob left unset here is omitted from the request body so the server
falls back to the value it was started with (env vars / CLI flags); this
is the server's per-request contract and keeps requests from leaking
settings into each other.

Refs:
  https://github.com/CrispStrobe/CrispASR/issues/86#issuecomment-4422827115
  CrispASR docs/tts.md (TADA talker sampling), docs/server.md (/v1/audio/speech)
  https://huggingface.co/cstr/tada-tts-3b-ml-GGUF

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


def build_speech_body(
    *,
    text: str,
    response_format: str,
    speed: float,
    instruction: str | None = None,
    voice: str | None = None,
    seed: int | None = None,
    temperature: float | None = None,
    top_p: float | None = None,
    top_k: int | None = None,
    repetition_penalty: float | None = None,
    do_sample: bool | None = None,
    num_candidates: int | None = None,
    consent_attestation: str | None = None,
    spoken_disclaimer: bool | None = None,
) -> dict:
    """Assemble the JSON body for `POST /v1/audio/speech`.

    Only `input` / `response_format` / `speed` are always sent. Every other
    knob is included *only* when explicitly set (non-None); anything left
    None is dropped so the server applies the default it was started with
    (env vars such as `TADA_TEMPERATURE` / `TADA_NUM_CANDIDATES`, or CLI
    flags). Omitting rather than sending a sentinel is what keeps per-request
    settings from leaking into the shared, long-lived backend context.

    Parameters
    ----------
    text : str
        Text to synthesize (server field `input`).
    response_format : str
        `wav`, `pcm`, or `f32`.
    speed : float
        Tempo multiplier 0.25..4.0 (post-synth resample on the server).
    instruction : str or None
        Voice-direction prose. Only qwen3-tts VoiceDesign honours it; TADA
        silently ignores it, so it is opt-in here rather than always sent.
    voice : str or None
        Backend voice selector. For TADA this is a baked reference: a bare
        name resolving under the server's `--voice-dir`, or a path to a
        `tada-ref-*.gguf`. Omit to use the server's `--voice` default.
    seed : int or None
        RNG seed. With sampling on, the same seed + text is reproducible and
        different seeds change the wording.
    temperature : float or None
        Talker sampling temperature (TADA server default 0.6). 0 = greedy.
    top_p : float or None
        Talker nucleus-sampling cutoff (TADA server default 0.9).
    top_k : int or None
        Talker top-k cutoff, 0 = disabled (TADA server default 0).
    repetition_penalty : float or None
        Talker repetition penalty, 1.0 = none (TADA server default 1.1).
        Raising it measurably reduces repeated/looped words.
    do_sample : bool or None
        Enable talker sampling. False forces the old greedy argmax decode
        (loops / drops words on hard or non-English text). TADA default: on.
    num_candidates : int or None
        TADA per-token *duration* flow-matching candidates: higher = more
        robust timing, lower = faster (1 = single noise draw). Tunes timing
        only, not which words are spoken. Server default 4 (env
        `TADA_NUM_CANDIDATES`).
    consent_attestation : str or None
        Required by the server when `voice` ends in `.wav` (runtime voice
        cloning): a free-text speaker-consent statement, logged for audit.
    spoken_disclaimer : bool or None
        Set False to skip the audible AI-disclosure prefix on voice-cloned
        output. Machine-readable provenance (watermark + C2PA) is always
        applied regardless; the caller then owns disclosing AI use.

    Returns
    -------
    dict
        The request body, ready for `json.dumps`.
    """
    body: dict = {
        "input": text,
        "response_format": response_format,
        "speed": speed,
    }
    # Server field name -> value; entries that are None are not sent so the
    # server keeps its own startup default for that knob.
    optional = {
        "instructions": instruction,
        "voice": voice,
        "seed": seed,
        "temperature": temperature,
        "top_p": top_p,
        "top_k": top_k,
        "repetition_penalty": repetition_penalty,
        "do_sample": do_sample,
        "num_candidates": num_candidates,
        "consent_attestation": consent_attestation,
        "spoken_disclaimer": spoken_disclaimer,
    }
    body.update({k: v for k, v in optional.items() if v is not None})
    return body


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
    # No sampling knobs are sent, so this exercises exactly the server's
    # configured defaults (the path real requests take when run bare).
    print(f"POST {base_url}/v1/audio/speech (smoke test)", file=sys.stderr)
    code, body, ctype = _post_json(
        f"{base_url}/v1/audio/speech",
        build_speech_body(text="Test.", response_format="wav", speed=1.0),
        timeout=120.0,
    )
    if code != 200:
        print(f"  -> {code}: {body[:500].decode('utf-8', 'replace')}", file=sys.stderr)
        return 1
    print(f"  -> {code}, {len(body)} bytes ({ctype}); model OK", file=sys.stderr)
    return 0


def cmd_synth(base_url: str, body: dict, out: Path, timeout: float) -> int:
    """POST a prepared speech `body` and write the returned audio to `out`."""
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
    p.add_argument("-i", "--instruction", default=None,
                   help="Voice-direction prose (qwen3-tts VoiceDesign only; TADA ignores it). "
                        "Opt-in: not sent unless given.")
    p.add_argument("-t", "--text", help="Text to read aloud.")
    p.add_argument("-o", "--output", default="tts_out.wav",
                   help="Output filename (default: tts_out.wav).")
    p.add_argument("--format", default="wav", choices=("wav", "pcm", "f32"),
                   help="Audio response format (default: wav).")
    p.add_argument("--speed", type=float, default=1.0,
                   help="Speech speed, 0.25 to 4.0 (default: 1.0).")
    p.add_argument("--timeout", type=float, default=300.0,
                   help="HTTP timeout in seconds (default: 300).")
    p.add_argument("--seed", type=int, default=None,
                   help="Integer seed for reproducible synthesis (default: not sent). "
                        "With sampling on, different seeds change the wording.")

    voice = p.add_argument_group(
        "voice / cloning",
        "Select or clone the speaking voice. For TADA, --voice is a baked "
        "reference (a bare name under the server's --voice-dir, or a path to a "
        "tada-ref-*.gguf).")
    voice.add_argument("--voice", default=None,
                       help="Voice name or reference path (default: server's --voice).")
    voice.add_argument("--consent-attestation", default=None,
                       help="Speaker-consent statement, required by the server when "
                            "--voice ends in .wav (runtime voice cloning).")
    voice.add_argument("--spoken-disclaimer", action=argparse.BooleanOptionalAction,
                       default=None,
                       help="Audible AI-disclosure prefix on cloned output (server default: on). "
                            "--no-spoken-disclaimer skips it; provenance watermark + C2PA still apply.")

    # TADA talker text-sampling knobs (CrispASR #197). These steer *which
    # words* are spoken; leaving them unset uses the server's sampling
    # defaults that match upstream InferenceOptions.
    tada = p.add_argument_group(
        "TADA talker sampling",
        "Per-request talker knobs. Unset = server default (upstream InferenceOptions: "
        "do_sample on, temperature 0.6, top_p 0.9, top_k 0, repetition_penalty 1.1).")
    tada.add_argument("--temperature", type=float, default=None,
                      help="Talker sampling temperature (0 = greedy).")
    tada.add_argument("--top-p", type=float, default=None,
                      help="Talker nucleus-sampling cutoff.")
    tada.add_argument("--top-k", type=int, default=None,
                      help="Talker top-k cutoff (0 = disabled).")
    tada.add_argument("--repetition-penalty", type=float, default=None,
                      help="Talker repetition penalty (1.0 = none; raise to cut repeats/loops).")
    tada.add_argument("--do-sample", action=argparse.BooleanOptionalAction, default=None,
                      help="Enable talker sampling. --no-do-sample forces the old greedy "
                           "argmax decode (loops / drops words on hard text).")
    tada.add_argument("--num-candidates", type=int, default=None,
                      help="TADA per-token duration flow-matching candidates: higher = more "
                           "robust timing, lower = faster (1 = single noise draw). Server default 4.")
    args = p.parse_args()

    base_url = args.url.rstrip("/")

    if args.check:
        return cmd_check(base_url)

    if not args.text:
        p.error("--text is required unless --check is given")

    body = build_speech_body(
        text=args.text,
        response_format=args.format,
        speed=args.speed,
        instruction=args.instruction,
        voice=args.voice,
        seed=args.seed,
        temperature=args.temperature,
        top_p=args.top_p,
        top_k=args.top_k,
        repetition_penalty=args.repetition_penalty,
        do_sample=args.do_sample,
        num_candidates=args.num_candidates,
        consent_attestation=args.consent_attestation,
        spoken_disclaimer=args.spoken_disclaimer,
    )

    return cmd_synth(
        base_url=base_url,
        body=body,
        out=Path(args.output),
        timeout=args.timeout,
    )


if __name__ == "__main__":
    sys.exit(main())
