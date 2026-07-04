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
client exposes TADA's per-request knobs. As of CrispASR #197 the talker
samples by default (matching upstream `InferenceOptions`: do_sample=on,
temperature=0.6, top_p=0.9, top_k=0, repetition_penalty=1.1) instead of
pure greedy argmax, which fixed the looping / word-dropping /
garbled-liaison hallucinations on French. The same rebuild also made TADA
synthesize the whole utterance in one pass; it used to split on every
period and pad silence between sentences, which produced a trailing ~9 s
pause plus wind/hum on short inputs. Tunable at query time without a
restart: the talker sampler, the duration flow-matching `num_candidates`,
and the acoustic flow-matching knobs `num_steps` / `cfg_scale` /
`noise_temp`. See PR.md and CrispASR docs/tts.md ("Talker text sampling")
+ docs/server.md for the field contract.

Any knob left unset here is omitted from the request body so the server
falls back to the value it was started with (env vars / CLI flags); this
is the server's per-request contract and keeps requests from leaking
settings into each other.

Voice is per-request too (CrispASR #201): point `voice` at a different
`tada-ref-*.gguf` (an absolute path, or a cache/registry name that
auto-downloads) and TADA reloads it on the next call. Omitting `voice`,
or passing `"default"` / `"auto"`, keeps whatever ref is currently loaded
(no reload, no cross-request leak). A `.wav` handed straight to `voice` is
still rejected by TADA: bake a ref offline first (CLI `--make-ref`, or
convert-tada-ref-to-gguf.py from audio + transcript), then switch to it
live. On-the-fly ref generation from audio is not wired into the server
yet. As a convenience, `--voice-name NAME` bakes ./voices/<NAME> into a
ref on the host (delegating to utils/clone_voice.py, which owns the torch
converter + naming) if it is not already baked, then synthesizes with it
in one call, no restart.

Refs:
  https://github.com/CrispStrobe/CrispASR/issues/86#issuecomment-4422827115
  CrispASR docs/tts.md (TADA talker sampling), docs/server.md (/v1/audio/speech)
  https://huggingface.co/cstr/tada-tts-3b-ml-GGUF

Generated with help from Claude Code.
"""
from __future__ import annotations

import argparse
import json
import subprocess
import sys
import time
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


def _post_json(url: str, body: dict, timeout: float) -> tuple[int, bytes, str, float]:
    """POST `body` as JSON and return (status, payload, content-type, elapsed).

    `elapsed` is the wall-clock seconds of the backend round-trip: from the
    moment the request is sent until the full response body is read (so it
    covers synthesis time, not client-side arg parsing or file writing).
    Measured with `time.perf_counter` and reported on HTTP errors too, since a
    slow 4xx/5xx is still useful timing information.
    """
    data = json.dumps(body).encode("utf-8")
    req = urllib.request.Request(
        url, data=data, method="POST",
        headers={"Content-Type": "application/json"},
    )
    t0 = time.perf_counter()
    try:
        with urllib.request.urlopen(req, timeout=timeout) as r:
            return r.status, r.read(), r.headers.get("Content-Type", ""), time.perf_counter() - t0
    except urllib.error.HTTPError as e:
        return e.code, e.read(), e.headers.get("Content-Type", ""), time.perf_counter() - t0


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
    num_steps: int | None = None,
    cfg_scale: float | None = None,
    noise_temp: float | None = None,
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
        Backend voice selector, passed through verbatim. TADA honours this
        per request (CrispASR #201): a `tada-ref-*.gguf` given as an
        absolute path, or as a cache/registry name that auto-downloads,
        makes the backend reload that reference on the next call (it logs
        `switched voice -> ...`). Passing `"default"` / `"auto"`, or
        omitting it, keeps whatever ref is currently loaded (no reload, no
        cross-request leak). A `.wav` handed straight to `voice` is still
        rejected by TADA (bake a ref offline first). Also honoured by
        qwen3-tts base, vibevoice-1.5b, ...
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
        TADA per-token flow-matching candidates (server maps this to
        `num_acoustic_candidates`): draw N noise samples per token and keep
        the "best" by a reconstruction scorer. Server default 1 (matches
        upstream InferenceOptions). >1 is opt-in via `TADA_NUM_CANDIDATES`;
        on the C++/CPU path best-of-N often ranks a WORSE draw (it can
        mangle words, e.g. "four hours" -> "and forth"), so 1 is the
        recommended default.
    num_steps : int or None
        TADA *acoustic* flow-matching ODE steps (server field `num_steps`,
        mapping to `num_flow_matching_steps`, default 10). The main
        quick-vs-accurate lever: more steps = crisper acoustics but slower
        (4 vs 25 steps both ASR-verbatim, 25 audibly crisper). Env
        `TADA_NUM_FM_STEPS`.
    cfg_scale : float or None
        TADA acoustic classifier-free guidance (server field `cfg_scale`,
        mapping to `acoustic_cfg`, default 1.6). Env `TADA_ACOUSTIC_CFG`.
    noise_temp : float or None
        TADA acoustic noise temperature (server field `noise_temp`, default
        0.9). Env `TADA_NOISE_TEMP`.
    consent_attestation : str or None
        Required by the server when `voice` ends in `.wav` (runtime voice
        cloning): a free-text speaker-consent statement, logged for audit.
        Inert for TADA, which rejects `.wav` voices outright (bake a
        `tada-ref-*.gguf` instead); relevant only for backends that clone
        from a `.wav` at request time.
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
        "num_steps": num_steps,
        "cfg_scale": cfg_scale,
        "noise_temp": noise_temp,
        "consent_attestation": consent_attestation,
        "spoken_disclaimer": spoken_disclaimer,
    }
    body.update({k: v for k, v in optional.items() if v is not None})
    return body


def ensure_baked_voice(name: str, language: str | None, force: bool) -> str:
    """Bake ./voices/<name> into a tada-ref GGUF (if needed) and return the
    container-side voice path to send as the request `voice`.

    The heavy lifting is delegated to utils/clone_voice.py (the host
    "torch converter" path): it pins torch, wraps convert-tada-ref-to-gguf.py,
    transcodes the reference audio, writes tada-ref-<slug>.gguf into the /models
    mount, and owns the naming/caching/mount convention. We reuse it rather than
    reimplement any of that, so tts.py stays dependency-free. The baked ref is
    then usable per request thanks to TADA's live voice reload (CrispASR #201),
    with no container restart.

    Requires `uv` (already used to run this script) and, on the first bake for a
    voice, ffmpeg + the encoder deps/weights clone_voice.py pulls in. Baking is
    skipped when the gguf already exists unless `force` is set.
    """
    cloner = Path(__file__).resolve().parent / "utils" / "clone_voice.py"
    if not cloner.exists():
        raise SystemExit(f"--voice-name needs utils/clone_voice.py next to tts.py, not found at {cloner}")

    def _run(extra: list[str], capture: bool) -> str:
        cmd = ["uv", "run", "--script", str(cloner), "--name", name, *extra]
        try:
            if capture:
                r = subprocess.run(cmd, capture_output=True, text=True, check=True)
                return (r.stdout or "").strip()
            # Baking: stream clone_voice.py's chatter to OUR stderr so tts.py's
            # stdout stays reserved for the final output path.
            subprocess.run(cmd, stdout=sys.stderr, check=True)
            return ""
        except FileNotFoundError:
            raise SystemExit("`uv` not found on PATH (needed to run clone_voice.py)")
        except subprocess.CalledProcessError as e:
            detail = (((e.stderr or "") + (e.stdout or "")).strip() if capture else "")
            raise SystemExit(f"clone_voice.py failed for voice {name!r}: {detail or 'see output above'}")

    # Resolve the canonical /models/*.gguf path first (cheap: no encoder/torch).
    container_voice = _run(["--print-path"], capture=True).splitlines()[-1].strip()
    if not container_voice:
        raise SystemExit(f"could not resolve a voice path for {name!r} via clone_voice.py "
                         f"(is there a ./voices/{name}.<audio> + {name}.txt pair?)")
    # Bake only when missing (--skip-if-baked) unless the caller forces a rebuild.
    bake_extra = [] if language is None else ["--language", language]
    if not force:
        bake_extra.append("--skip-if-baked")
    _run(bake_extra, capture=False)
    return container_voice


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
    code, body, ctype, elapsed = _post_json(
        f"{base_url}/v1/audio/speech",
        build_speech_body(text="Test.", response_format="wav", speed=1.0),
        timeout=120.0,
    )
    if code != 200:
        print(f"  -> {code}: {body[:500].decode('utf-8', 'replace')}", file=sys.stderr)
        return 1
    print(f"  -> {code}, {len(body)} bytes ({ctype}); model OK, backend took {elapsed:.2f}s",
          file=sys.stderr)
    return 0


def cmd_synth(base_url: str, body: dict, out: Path, timeout: float) -> int:
    """POST a prepared speech `body` and write the returned audio to `out`."""
    url = f"{base_url}/v1/audio/speech"
    print(f"POST {url}", file=sys.stderr)
    try:
        code, payload, ctype, elapsed = _post_json(url, body, timeout=timeout)
    except urllib.error.URLError as e:
        print(f"unreachable: {e}", file=sys.stderr)
        return 2
    if code != 200:
        print(f"server returned {code} after {elapsed:.2f}s: "
              f"{payload[:500].decode('utf-8', 'replace')}",
              file=sys.stderr)
        return 1
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_bytes(payload)
    print(f"  -> {code}, {len(payload)} bytes ({ctype}), backend took {elapsed:.2f}s",
          file=sys.stderr)
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
        "Select the speaking voice. TADA honours --voice per request (CrispASR "
        "#201): pass a tada-ref-*.gguf path or a cache/registry name and the "
        "backend reloads it live; 'default'/'auto'/unset keeps the loaded ref. A "
        ".wav is still rejected by TADA (bake a ref offline first). Also honoured "
        "by qwen3-tts base, vibevoice-1.5b, ...")
    voice.add_argument("--voice", default=None,
                       help="Voice name or reference path (TADA reloads it live; "
                            "'default'/'auto'/unset keeps the server's current voice).")
    voice.add_argument("--voice-name", default=None,
                       help="Bake ./voices/<NAME> (its <NAME>.txt transcript) into a tada-ref "
                            "gguf via clone_voice.py if not already baked, then synthesize with "
                            "it live (no container restart; TADA reloads per request). Mutually "
                            "exclusive with --voice. With no --text, just bakes and prints the "
                            "resolved /models/*.gguf path.")
    voice.add_argument("--voice-language", default=None,
                       help="Language code for the encoder when baking --voice-name "
                            "(default: clone_voice.py's own default, fr).")
    voice.add_argument("--force-bake", action="store_true",
                       help="Re-bake the --voice-name reference even if its gguf already exists.")
    voice.add_argument("--consent-attestation", default=None,
                       help="Speaker-consent statement, required by the server when "
                            "--voice ends in .wav (request-time cloning). TADA rejects "
                            ".wav voices, so this is inert for TADA.")
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
                      help="TADA per-token flow-matching candidates (best-of-N noise draws). "
                           "Server default 1; >1 is opt-in and can pick a worse draw on CPU.")

    # TADA acoustic flow-matching knobs (CrispASR #197). These steer audio
    # *quality* (not which words), and like the sampler knobs fall back to the
    # server's startup value when unset.
    acoustic = p.add_argument_group(
        "TADA acoustics",
        "Per-request acoustic flow-matching knobs. Unset = server default "
        "(num_steps 10, cfg_scale 1.6, noise_temp 0.9).")
    acoustic.add_argument("--num-steps", type=int, default=None,
                          help="Acoustic flow-matching ODE steps: more = crisper but slower "
                               "(the main quick-vs-accurate lever). Server default 10.")
    acoustic.add_argument("--cfg-scale", type=float, default=None,
                          help="Acoustic classifier-free guidance weight. Server default 1.6.")
    acoustic.add_argument("--noise-temp", type=float, default=None,
                          help="Acoustic noise temperature. Server default 0.9.")
    args = p.parse_args()

    base_url = args.url.rstrip("/")

    if args.check:
        return cmd_check(base_url)

    if args.voice_name:
        if args.voice:
            p.error("--voice and --voice-name are mutually exclusive")
        args.voice = ensure_baked_voice(args.voice_name, args.voice_language, args.force_bake)
        if not args.text:
            # Bake-only: emit the resolved container voice path on stdout.
            print(args.voice)
            return 0

    if not args.text:
        p.error("--text is required unless --check or --voice-name is given")

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
        num_steps=args.num_steps,
        cfg_scale=args.cfg_scale,
        noise_temp=args.noise_temp,
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
