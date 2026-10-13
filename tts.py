#!/usr/bin/env -S uv run --script
# /// script
# requires-python = ">=3.10"
# dependencies = []
# ///
"""Query the OpenAI-compatible TTS server on :8003 to synthesize speech.

The container exposes `POST /v1/audio/speech` on host port 8003 (see
docker-compose.yml). Run `tts.py --check` to confirm the server is alive
and the model loaded. Both TTS services bind :8003 by design (only one up
at a time), and this single client covers whichever answers: `crispasr-tts`
(CrispASR, TADA backend) or `voxtral-tts` (vLLM + vllm-omni serving
mistralai/Voxtral-4B-TTS-2603; it absorbed the former
voxtral/query_test.py).

Against Voxtral, `input`, `voice`, `response_format`, `speed`,
`max_new_tokens` and `extra_params` are honoured per request. Pass `--voice`
explicitly with one of the 20 built-in embedding names in VOXTRAL_VOICES
(e.g. fr_male, neutral_female), and `--max-new-tokens` to raise the
audio-length cap (frames at 12.5 Hz, so seconds x 12.5; the deploy default 2048
truncates at 163.84 s). Leave the TADA knobs below unset: they are only sent
when explicitly set, and the vllm-omni request schema does not define them.

The quality/reproducibility knobs `--cfg-alpha`, `--euler-steps` and
`--noise-seed` ride `extra_params`, which is how upstream vllm-omni already
reads cfg_alpha. The other two are served by our patched vllm_omni (the
`voxtral-tts` image builds it from voxtral/vllm-omni, branch `perso`); against a
stock server they are merged and ignored rather than rejected, so sending them
is always safe, it just silently changes nothing. Note `--seed` is a different
thing and still does nothing on Voxtral: stage 0's sampler is handed logits with
a single finite entry, so its result is forced. The one random draw in that
pipeline is each frame's flow-matching noise, which `--noise-seed` pins per
request. See the Voxtral notes in CLAUDE.md. If vLLM rejects a request without a
model name, add `--model mistralai/Voxtral-4B-TTS-2603`.

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

Batch mode (--batch FILE.jsonl): synthesize every row of a JSONL file
instead of a single --text. Each row needs an `input` (or `text`) field;
the output name is the row's `output` field, else `<id>.<format>`, else a
zero-padded line number, written under --output (a DIRECTORY in this mode).
All the voice/speed/sampling flags apply to every row, and a row may
override any of them for itself by naming it as a key, e.g.
{"input": "Bonjour.", "output": "a.wav", "voice": "fr_male",
 "cfg_alpha": 2.5, "noise_seed": 7}. That is how a sweep over voices or
cfg_alpha runs as one invocation rather than one process per cell. An
unknown key is rejected rather than ignored. Existing files are
skipped so an interrupted run resumes on re-run (unless --overwrite); rows
go out --concurrency at a time (default 4), which a vLLM/Voxtral server
batches server-side for a big speedup (use 1 for the single-context TADA
backend, or when sweeping euler_steps, which is shared per decode step).

Refs:
  https://github.com/CrispStrobe/CrispASR/issues/86#issuecomment-4422827115
  CrispASR docs/tts.md (TADA talker sampling), docs/server.md (/v1/audio/speech)
  https://huggingface.co/cstr/tada-tts-3b-ml-GGUF

Generated with help from Claude Code.
"""
from __future__ import annotations

import argparse
import base64
import copy
import inspect
import io
import json
import subprocess
import sys
import time
import urllib.error
import urllib.request
import wave
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

DEFAULT_URL = "http://127.0.0.1:8003"

# Unicode spaces that French typography scatters through ordinary prose: the
# no-break space before ; : ! ? and inside guillemets, plus its thin and
# narrow variants. Voxtral TTS degenerates on them, measured rather than
# guessed: the PROBE run (voxtral/cloning, 8 sentences x 6 voices x 4 text
# variants at a pinned seed) took the affected sentences from 47.7 pct median
# WER and 16 of 48 unusable takes down to 10.5 pct and 3 of 48 by replacing
# these with an ordinary space and changing nothing else. Removing the
# guillemets instead changed nothing at all (57.6 pct, 15 of 48), so the
# space is the trigger and the quotes were only ever guilty by association.
# Full write-up: cache/cloning/FLEURS/RESULTS.md.
UNICODE_SPACES = "\u00a0\u2009\u202f\u2007\u2008\u205f\u3000"


def normalise_spaces(text: str) -> str:
    """Replace the exotic Unicode spaces with ordinary ones.

    Applied to every request by default because the failure it prevents is
    silent and expensive: the model starts the sentence correctly and then
    loops syllables until it runs out of tokens, which costs minutes of GPU
    time and produces an unusable file with a 200 response. Nothing else about
    the text is touched, so this cannot change wording, punctuation or
    pronunciation; the audible difference is that the take terminates.

    Parameters
    ----------
    text : str
        Text as the caller wrote it.

    Returns
    -------
    str
        The same text with every character in `UNICODE_SPACES` replaced by
        `" "`. Unchanged when it contains none, which is the common case.
    """
    return text.translate({ord(char): " " for char in UNICODE_SPACES})

# Built-in voice embeddings shipped with mistralai/Voxtral-4B-TTS-2603 (from
# voice_embedding/*.pt in the HF repo). Only relevant when the voxtral-tts
# service is the one on :8003; TADA voices are tada-ref-*.gguf paths instead.
# Kept as a reference list, not argparse choices, since --voice is
# backend-dependent free text.
# Voice `--check` retries with when the server refuses a voiceless request
# (vllm-omni does, TADA does not). Any preset works; this one is the voice
# the compose service is actually used with.
VOXTRAL_CHECK_VOICE = "fr_female"

VOXTRAL_VOICES = [
    "ar_male",
    "casual_female",
    "casual_male",
    "cheerful_female",
    "de_female",
    "de_male",
    "es_female",
    "es_male",
    "fr_female",
    "fr_male",
    "hi_female",
    "hi_male",
    "it_female",
    "it_male",
    "neutral_female",
    "neutral_male",
    "nl_female",
    "nl_male",
    "pt_female",
    "pt_male",
]

# Voxtral's audio-token rate: stage 0 (the AR LLM) emits exactly one frame per
# decode step at this rate, so tokens / 12.5 = seconds of audio. From the
# checkpoint's own params.json
# (multimodal.audio_model_args.audio_encoding_args.frame_rate).
VOXTRAL_FRAME_RATE_HZ = 12.5
# Hard server-side ceiling on the per-request `max_new_tokens` field
# (vllm-omni 0.22 serving_speech.py `_TTS_MAX_NEW_TOKENS_MAX`), i.e. 327.68 s.
VOXTRAL_MAX_NEW_TOKENS_CAP = 4096

# Knobs that only one of the two servers on :8003 can act on, keyed by argparse
# dest. Used by warn_on_mixed_backend_knobs() below.
#
# Why this needs a guard at all: this client is deliberately backend-agnostic and
# both servers accept an unknown per-request field WITHOUT complaining, so a knob
# aimed at the wrong one is silently dropped behind an HTTP 200. CrispASR ignores
# fields its active backend does not declare; vllm-omni is worse, because
# `OpenAICreateSpeechRequest` sets no pydantic `model_config` and v2 then defaults
# to extra='ignore', dropping every unknown top-level key. The near-collision
# --cfg-scale (TADA) vs --cfg-alpha (Voxtral) is the sharp edge: passing the wrong
# one produces byte-identical audio across a "sweep" and nothing says why.
TADA_ONLY_KNOBS = (
    "temperature", "top_p", "top_k", "repetition_penalty", "do_sample",
    "num_candidates", "num_steps", "cfg_scale", "noise_temp",
)
VOXTRAL_ONLY_KNOBS = ("cfg_alpha", "euler_steps", "noise_seed", "dump_codes", "max_new_tokens")


def warn_on_mixed_backend_knobs(args: argparse.Namespace) -> None:
    """Warn when one call mixes TADA-only and Voxtral-only per-request knobs.

    Only one TTS server can own :8003 at a time (crispasr-tts or voxtral-tts), so
    a request carrying both families always has one family thrown away. Which one
    depends on who is listening, and this client cannot tell from here, hence a
    warning rather than an error: the request is still sent, in full.

    Parameters
    ----------
    args : argparse.Namespace
        Parsed CLI arguments. Every knob inspected here defaults to ``None``, so a
        non-None value means the user set it explicitly.
    """
    tada = [d for d in TADA_ONLY_KNOBS if getattr(args, d, None) is not None]
    voxtral = [d for d in VOXTRAL_ONLY_KNOBS if getattr(args, d, None) is not None]
    if not tada or not voxtral:
        return

    def flags(dests: list[str]) -> str:
        """Render dests as CLI flags, with the verb that agrees with the count."""
        rendered = ", ".join("--" + d.replace("_", "-") for d in dests)
        return f"{rendered} {'apply' if len(dests) > 1 else 'applies'}"

    print(f"warning: {flags(tada)} only to the CrispASR TADA backend, while "
          f"{flags(voxtral)} only to the Voxtral (vLLM) backend. Only one of "
          "them owns :8003, so one group is being silently ignored by whichever "
          "server is up (both answer 200 for fields they do not know).",
          file=sys.stderr)
    if "cfg_scale" in tada and "cfg_alpha" in voxtral:
        print("warning: --cfg-scale (TADA) and --cfg-alpha (Voxtral) are different "
              "knobs on different servers, not aliases; pick the one matching the "
              "running backend.", file=sys.stderr)


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
    model: str | None = None,
    instruction: str | None = None,
    voice: str | None = None,
    ref_audio: str | None = None,
    seed: int | None = None,
    temperature: float | None = None,
    top_p: float | None = None,
    top_k: int | None = None,
    repetition_penalty: float | None = None,
    do_sample: bool | None = None,
    num_candidates: int | None = None,
    num_steps: int | None = None,
    cfg_scale: float | None = None,
    cfg_alpha: float | None = None,
    euler_steps: int | None = None,
    noise_seed: int | None = None,
    dump_codes: str | None = None,
    noise_temp: float | None = None,
    max_new_tokens: int | None = None,
    consent_attestation: str | None = None,
    spoken_disclaimer: bool | None = None,
    raw_spaces: bool = False,
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
    model : str or None
        OpenAI `model` field. CrispASR serves whatever backend it was
        started with and does not need it; vLLM validates it against the
        served model, so against voxtral-tts pass
        `mistralai/Voxtral-4B-TTS-2603` if a modelless request is rejected.
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
    ref_audio : str or None
        Voxtral voice cloning: a reference clip as a base64 data URI, an
        http(s) URL, or a file:// URI resolved inside the container. Mutually
        exclusive with `voice` server-side (the request needs exactly one of
        the two). Unlike TADA, this needs no baking step and no encoder on our
        side: vllm-omni hands the decoded waveform to the stage-0 LLM as
        multimodal input (`_build_voxtral_prompt` -> `multi_modal_data`), which
        is a different mechanism from the preset voices' pre-encoded prefixes.
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
    cfg_alpha : float or None
        Voxtral classifier-free guidance scale (this repo's server default is
        2.0; usable 1.2-2.0 on the shipped voices, 1.5-2.0 on the hybrids),
        sent nested inside `extra_params` rather than as a flat field. Unrelated
        to `cfg_scale` above: different backend, different server field,
        different default. Unset = the value the server booted with
        (`VOXTRAL_CFG_ALPHA` in docker-compose.yml).
    euler_steps : int or None
        Voxtral Euler ODE / NFE step count for the stage-0 flow-matching
        acoustic transformer, sent nested in `extra_params` as
        `voxtral_euler_steps`. The compute<->quality lever: more steps = steadier
        audio, cost scaling linearly with audio length. Honoured only by our
        patched vllm_omni, and only when every request sharing a decode step asks
        for the same value (it is a loop bound, not a per-row tensor), at the
        price of that request running eagerly instead of from a CUDA graph. Unset
        = the server's startup value (`VOXTRAL_EULER_STEPS`, else 7).
    noise_seed : int or None
        Voxtral per-request seed for the flow-matching noise, sent nested in
        `extra_params` as `voxtral_noise_seed`. This is the only thing that makes
        one specific take reproducible: same text plus same seed gives identical
        audio at any point in the container's life, no restart. Honoured by our
        patched vllm_omni; stock vllm-omni merges the key, finds no reader, and
        leaves the take unreproducible. Unrelated to `seed` above, which Voxtral
        cannot act on at all.
    dump_codes : str or None
        Voxtral: ask the server to append the codes stage 0 emits for THIS
        request to `<VOXTRAL_CODE_DUMP_DIR>/<tag>.i16`, sent nested in
        `extra_params` as `voxtral_dump_codes`. A data-collection tool, not a
        synthesis knob: the audio is unaffected. It exists because the open
        checkpoint ships no audio encoder, so the model generating its own codes
        is the only source of real audio/codes pairs to train one on. Needs our
        patched vllm_omni and the env var set on the service; stock vllm-omni
        merges the key, finds no reader, and writes nothing.
    noise_temp : float or None
        TADA acoustic noise temperature (server field `noise_temp`, default
        0.9). Env `TADA_NOISE_TEMP`.
    max_new_tokens : int or None
        Longest utterance THIS request may produce, in stage-0 decode steps.
        Voxtral emits one 12.5 Hz audio frame per step, so this is seconds x
        12.5 (the deploy default 2048 = 163.84 s, past which audio is cut off
        mid-sentence with finish_reason "length" and no error). vllm-omni
        rejects anything above 4096 (`_TTS_MAX_NEW_TOKENS_MAX`, = 327.68 s) and
        stage 0's `max_model_len` caps it further, since the prompt's text
        tokens share that context: to actually use a large value, raise
        `VOXTRAL_MAX_MODEL_LEN` in docker-compose.yml. Unset = the server's
        startup value (`VOXTRAL_MAX_TOKENS`).
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
    raw_spaces : bool
        Send `text` byte for byte, skipping `normalise_spaces`. Off by
        default, so ordinary callers get the fix without asking. Turn it on
        when the exact bytes are the point: a measurement that has to send
        the corpus verbatim (`voxtral/cloning/sweep.py` does), or a
        reproduction of the degeneration itself.

    Returns
    -------
    dict
        The request body, ready for `json.dumps`.
    """
    body: dict = {
        "input": text if raw_spaces else normalise_spaces(text),
        "response_format": response_format,
        "speed": speed,
    }
    # Server field name -> value; entries that are None are not sent so the
    # server keeps its own startup default for that knob.
    optional = {
        "model": model,
        "instructions": instruction,
        "voice": voice,
        "ref_audio": ref_audio,
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
        "max_new_tokens": max_new_tokens,
        "consent_attestation": consent_attestation,
        "spoken_disclaimer": spoken_disclaimer,
    }
    body.update({k: v for k, v in optional.items() if v is not None})
    # Voxtral takes its model-specific knobs through a nested `extra_params`
    # object instead of flat fields, which is why these three cannot join the map
    # above. Server side (vllm-omni 0.22) `serving_speech.py` merges that object
    # verbatim into stage 0's SamplingParams.extra_args, and the model then reads
    # one cfg_alpha per request from it (`voxtral_tts.py::_extract_cfg_alpha`,
    # falling back to 1.2). `voxtral_noise_seed` and `voxtral_euler_steps` ride the
    # same route and are read by our patched vllm_omni (voxtral/vllm-omni, branch
    # `perso`); a stock server merges them, finds no reader, and carries on, which
    # is why sending them is harmless rather than an error.
    #
    # Deliberately NOT sent as the flat `cfg_alpha` / `euler_steps` fields our fork
    # also accepts: extra_params is the one route that works against both servers,
    # and the fork lets extra_params win on conflict anyway. Still opt-in like
    # everything else here: the whole object is omitted when all three are unset,
    # so the server keeps its deploy-YAML defaults.
    extra_params = {
        "cfg_alpha": cfg_alpha,
        "voxtral_euler_steps": euler_steps,
        "voxtral_noise_seed": noise_seed,
        "voxtral_dump_codes": dump_codes,
    }
    extra_params = {k: v for k, v in extra_params.items() if v is not None}
    if extra_params:
        body["extra_params"] = extra_params
    return body


def resolve_ref_audio(value: str | None) -> str | None:
    """Turn `--ref-audio` into something the server accepts.

    A local path becomes an inline base64 data URI, which is the only form that
    works regardless of where the server runs: an http(s) URL needs the clip to
    be published somewhere, and a `file://` URI is resolved INSIDE the container,
    so it would silently miss a host path that is not mounted there. URLs and
    file:// URIs are passed through untouched for the cases where they are what
    the user wants.

    The MIME type is a hint only. mistral_common sniffs the actual container
    (soundfile/libsndfile), so ogg, wav, flac and mp3 all work whatever is
    claimed here.

    Parameters
    ----------
    value : str or None
        A local path, an http(s) URL, a file:// URI, or an already-built
        data URI. None passes through.

    Returns
    -------
    str or None
        The value to put in the request's `ref_audio` field.
    """
    if value is None:
        return None
    if value.startswith(("http://", "https://", "file://", "data:")):
        return value
    path = Path(value)
    if not path.is_file():
        raise SystemExit(f"--ref-audio: no such file: {path}")
    mime = {".ogg": "audio/ogg", ".wav": "audio/wav", ".mp3": "audio/mpeg",
            ".flac": "audio/flac", ".m4a": "audio/mp4"}.get(path.suffix.lower(), "audio/ogg")
    encoded = base64.b64encode(path.read_bytes()).decode("ascii")
    return f"data:{mime};base64,{encoded}"


def _body_from_args(text: str, args: argparse.Namespace) -> dict:
    """Build a /v1/audio/speech body from the parsed CLI args for `text`.
    Single and batch modes share this so their request shapes never drift."""
    return build_speech_body(
        text=text,
        response_format=args.format,
        speed=args.speed,
        model=args.model,
        instruction=args.instruction,
        voice=args.voice,
        ref_audio=args.ref_audio,
        seed=args.seed,
        temperature=args.temperature,
        top_p=args.top_p,
        top_k=args.top_k,
        repetition_penalty=args.repetition_penalty,
        do_sample=args.do_sample,
        num_candidates=args.num_candidates,
        num_steps=args.num_steps,
        cfg_scale=args.cfg_scale,
        cfg_alpha=args.cfg_alpha,
        euler_steps=args.euler_steps,
        noise_seed=args.noise_seed,
        dump_codes=args.dump_codes,
        noise_temp=args.noise_temp,
        max_new_tokens=args.max_new_tokens,
        raw_spaces=args.raw_spaces,
        consent_attestation=args.consent_attestation,
        spoken_disclaimer=args.spoken_disclaimer,
    )


# Keys a batch row may carry besides its text and its filename: every synthesis knob
# build_speech_body accepts, derived from its signature rather than listed, so a knob added
# there becomes per-row overridable with no second list to forget. `text` is the row's own
# input and `response_format` is excluded on purpose: the extension of the output file and
# the format asked for must not be able to disagree row by row.
ROW_OVERRIDES = frozenset(inspect.signature(build_speech_body).parameters) - {
    "text", "response_format",
}
# The row's own plumbing, not knobs. `text` is accepted as an alias for `input` because
# cmd_batch already reads either.
ROW_RESERVED = frozenset({"input", "text", "output", "id"})


def row_overrides(row: dict) -> tuple[dict, str | None]:
    """Split a batch row's per-request knobs out of its plumbing keys.

    Parameters
    ----------
    row
        One decoded JSONL object from `--batch`.

    Returns
    -------
    tuple of (dict, str or None)
        The overrides to apply for that row, and an error message if the row
        names a key that is not a knob (in which case the overrides are empty).

    An unknown key is an error rather than something ignored, because silently
    ignoring it is the exact failure this whole repo keeps tripping on: stock
    vllm-omni drops unknown request fields and still answers 200, so a sweep
    over a misspelled `cfg_alpa` runs entirely at the server default and the
    wavs look like a result. A rejected row costs one line of output instead.
    """
    overrides = {k: v for k, v in row.items() if k not in ROW_RESERVED}
    unknown = sorted(set(overrides) - ROW_OVERRIDES)
    if unknown:
        return {}, (f"unknown key(s) {', '.join(unknown)}; "
                    f"row keys must be one of {', '.join(sorted(ROW_RESERVED | ROW_OVERRIDES))}")
    return overrides, None


def args_with_overrides(args: argparse.Namespace, overrides: dict) -> argparse.Namespace:
    """Return a copy of `args` with a batch row's overrides applied.

    Overriding the namespace rather than patching the request body is what
    keeps one request shape in the program: the row's `cfg_alpha` then travels
    through build_speech_body exactly as the flag does, which matters because
    that function decides *where* a knob goes (Voxtral's cfg_alpha lands in
    `extra_params`, not as a flat field) and warns about bad values. A shallow
    copy is enough since every knob is a scalar or a string.
    """
    row_args = copy.copy(args)
    for key, value in overrides.items():
        setattr(row_args, key, value)
    return row_args


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


def cmd_check(base_url: str, voice: str | None = None) -> int:
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
    def smoke(with_voice: str | None) -> tuple[int, bytes, str, float]:
        print(f"POST {base_url}/v1/audio/speech (smoke test"
              + (f", voice={with_voice}" if with_voice else "") + ")", file=sys.stderr)
        return _post_json(
            f"{base_url}/v1/audio/speech",
            build_speech_body(text="Test.", response_format="wav", speed=1.0,
                              voice=with_voice),
            timeout=120.0,
        )

    code, body, ctype, elapsed = smoke(voice)
    if code == 400 and voice is None and b"'voice'" in body:
        # Backend-dependent, and the reason --check takes a voice at all: TADA
        # synthesizes with whatever reference it was started with, but vllm-omni
        # requires `voice` or `ref_audio` on EVERY request (serving_speech.py),
        # so a bare smoke call is a 400 no matter how healthy the server is.
        # Retry with a preset rather than reporting a live server as down.
        print(f"  -> 400 (no default voice on this backend); retrying with "
              f"{VOXTRAL_CHECK_VOICE}", file=sys.stderr)
        code, body, ctype, elapsed = smoke(VOXTRAL_CHECK_VOICE)
    if code != 200:
        print(f"  -> {code}: {body[:500].decode('utf-8', 'replace')}", file=sys.stderr)
        return 1
    print(f"  -> {code}, {len(body)} bytes ({ctype}); model OK, backend took {elapsed:.2f}s",
          file=sys.stderr)
    return 0


def wav_seconds(payload: bytes) -> float | None:
    """Best-effort clip duration in seconds. None for non-RIFF payloads (mp3,
    opus, flac, ...) and float32 wavs, which stdlib `wave` cannot parse; sticking
    to `wave` keeps this client dependency-free (no soundfile)."""
    if payload[:4] != b"RIFF":
        return None
    try:
        with wave.open(io.BytesIO(payload)) as w:
            rate = w.getframerate()
            return w.getnframes() / rate if rate else None
    except (wave.Error, EOFError):
        return None


def _synthesize(base_url: str, body: dict, out: Path,
                timeout: float) -> tuple[float, int, str, float | None]:
    """POST `body`, write the audio to `out` atomically, and return
    (backend elapsed seconds, byte count, content-type, clip seconds or None).

    Raises RuntimeError on a non-200 response and lets URLError propagate (an
    unreachable server should abort a batch, not fail every row). The bytes go
    to a `.tmp` sibling then get renamed, so a killed run never leaves a
    truncated file that a resume would skip as already done.
    """
    code, payload, ctype, elapsed = _post_json(
        f"{base_url}/v1/audio/speech", body, timeout=timeout)
    if code != 200:
        raise RuntimeError(f"server returned {code} after {elapsed:.2f}s: "
                           f"{payload[:500].decode('utf-8', 'replace')}")
    out.parent.mkdir(parents=True, exist_ok=True)
    tmp = out.with_suffix(out.suffix + ".tmp")
    tmp.write_bytes(payload)
    tmp.replace(out)
    return elapsed, len(payload), ctype, wav_seconds(payload)


def cmd_synth(base_url: str, body: dict, out: Path, timeout: float) -> int:
    """POST a prepared speech `body` and write the returned audio to `out`."""
    print(f"POST {base_url}/v1/audio/speech", file=sys.stderr)
    try:
        elapsed, nbytes, ctype, dur = _synthesize(base_url, body, out, timeout)
    except urllib.error.URLError as e:
        print(f"unreachable: {e}", file=sys.stderr)
        return 2
    except RuntimeError as e:
        print(str(e), file=sys.stderr)
        return 1
    wav_info = f", {dur:.2f}s" if dur is not None else ""
    print(f"  -> 200, {nbytes} bytes ({ctype}){wav_info}, backend took {elapsed:.2f}s",
          file=sys.stderr)
    print(str(out.resolve()))
    return 0


def _iter_progress(it, total: int):
    """Yield from `it` while printing an overwriting [n/total] counter to
    stderr: a stdlib stand-in for tqdm (this client stays dependency-free)."""
    for n, item in enumerate(it, 1):
        print(f"\r  [{n}/{total}]", end="", file=sys.stderr, flush=True)
        yield item
    print(file=sys.stderr)


def cmd_batch(base_url: str, jsonl_path: Path, args: argparse.Namespace) -> int:
    """Synthesize every row of a JSONL file into the --output directory.

    Row schema is general (not tied to any dataset): each row needs an `input`
    (or `text`) field; the output filename is the row's `output` field, else
    `<id>.<format>`, else a zero-padded line number. Existing files are skipped
    (resume-friendly) unless --overwrite. Rows are fanned out --concurrency at a
    time so a vLLM/Voxtral server can batch them server-side. Reuses the same
    _body_from_args + _synthesize path as single mode, so every voice/speed/
    sampling flag applies identically to each row.

    A row may also carry any synthesis knob as a key (`voice`, `cfg_alpha`,
    `euler_steps`, `noise_seed`, `speed`, ...), which overrides the flag of the
    same name for that row alone; see ROW_OVERRIDES. That is what makes a sweep
    one invocation instead of one process per cell: on Voxtral these knobs are
    genuinely per request (the model reads a cfg_alpha per row), so varying them
    inside one concurrent batch is sound. The exception is `euler_steps`, which
    is a loop bound shared by every request in a decode step and is honoured
    only when they all agree, so sweep that one with --concurrency 1.
    """
    out_dir = Path(args.output or "tts_batch")
    out_dir.mkdir(parents=True, exist_ok=True)

    todo: list[tuple[Path, str, dict]] = []
    planned: set[Path] = set()
    skipped_existing = bad_rows = 0
    with jsonl_path.open() as f:
        for lineno, line in enumerate(f, 1):
            if not line.strip():
                continue
            try:
                row = json.loads(line)
            except json.JSONDecodeError as e:
                print(f"{jsonl_path}:{lineno} invalid JSON ({e}), row skipped", file=sys.stderr)
                bad_rows += 1
                continue
            text = row.get("input") or row.get("text")
            if not text:
                print(f"{jsonl_path}:{lineno} no 'input'/'text' field, row skipped", file=sys.stderr)
                bad_rows += 1
                continue
            overrides, bad_key = row_overrides(row)
            if bad_key:
                print(f"{jsonl_path}:{lineno} {bad_key}, row skipped", file=sys.stderr)
                bad_rows += 1
                continue
            if row.get("output"):
                name = str(row["output"])
            elif "id" in row:
                name = f"{row['id']}.{args.format}"
            else:
                name = f"{lineno:06d}.{args.format}"
            out = out_dir / name
            if out in planned:
                print(f"{jsonl_path}:{lineno} duplicate target {out.name}, row skipped",
                      file=sys.stderr)
                bad_rows += 1
                continue
            planned.add(out)
            if out.exists() and not args.overwrite:
                skipped_existing += 1
                continue
            todo.append((out, text, overrides))

    print(f"{jsonl_path}: {len(planned)} rows planned, {skipped_existing} already done, "
          f"{len(todo)} to generate, {bad_rows} bad rows", file=sys.stderr)
    if not todo:
        return 1 if bad_rows else 0

    failed = generated = 0
    total_audio = 0.0
    aborted = False
    # Fan out: each _synthesize() is an independent request writing its own file,
    # so it is thread-safe. Concurrency lets vLLM/Voxtral batch requests together;
    # for the single-context TADA/crispasr backend pass --concurrency 1.
    pool = ThreadPoolExecutor(max_workers=max(1, args.concurrency))
    futures = {pool.submit(_synthesize, base_url,
                           _body_from_args(text, args_with_overrides(args, overrides)),
                           out, args.timeout): out
               for out, text, overrides in todo}
    try:
        for fut in _iter_progress(as_completed(futures), len(futures)):
            out = futures[fut]
            try:
                elapsed, _nbytes, _ctype, dur = fut.result()
            except urllib.error.URLError as e:
                print(f"\nserver unreachable ({e}), aborting; re-run to resume", file=sys.stderr)
                aborted = True
                break
            except RuntimeError as e:
                print(f"\n{out.name}: {e}", file=sys.stderr)
                failed += 1
                continue
            generated += 1
            if dur is not None:
                total_audio += dur
    finally:
        # On abort, drop queued-but-unstarted work so an unreachable server fails
        # fast; in-flight requests finish. On success, wait for the pool normally.
        pool.shutdown(wait=not aborted, cancel_futures=aborted)
    if aborted:
        failed += len(todo) - generated - failed

    audio_info = f", {total_audio / 60:.1f} min of audio" if total_audio else ""
    print(f"batch: {generated} generated, {failed} failed, {skipped_existing} skipped, "
          f"{bad_rows} bad rows{audio_info}", file=sys.stderr)
    return 1 if (failed or bad_rows) else 0


def main() -> int:
    p = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    p.add_argument("--url", default=DEFAULT_URL,
                   help=f"Base URL of the crispasr-tts server (default: {DEFAULT_URL}).")
    p.add_argument("--check", action="store_true",
                   help="Probe /health and run a tiny synthesis to confirm the model is loaded.")
    p.add_argument("-i", "--instruction", default=None,
                   help="Voice-direction prose (qwen3-tts VoiceDesign only; TADA ignores it). "
                        "Opt-in: not sent unless given.")
    p.add_argument("-t", "--text", help="Text to read aloud (single mode).")
    p.add_argument("--batch", default=None, metavar="FILE.jsonl",
                   help="Batch mode: synthesize every row of a JSONL file instead of "
                        "a single --text. Each row needs an 'input' (or 'text') field; "
                        "the output name comes from the row's 'output' field, else "
                        "'<id>.<format>', else a zero-padded line number. All the "
                        "voice/speed/sampling flags apply to every row, and a row can "
                        "override any one of them for itself by naming it as a key "
                        "('voice', 'cfg_alpha', 'noise_seed', ...), which is how a sweep "
                        "runs in one invocation.")
    p.add_argument("--overwrite", action="store_true",
                   help="Batch mode: regenerate rows whose output file already exists "
                        "(default: skip them, so a killed run resumes on re-run).")
    p.add_argument("--concurrency", type=int, default=4,
                   help="Batch mode: requests in flight at once (default: 4). A "
                        "vLLM/Voxtral server batches concurrent requests server-side "
                        "(big speedup); use 1 for the single-context TADA/crispasr backend.")
    p.add_argument("-o", "--output", default=None,
                   help="Single mode: output audio file (default: tts_out.wav). "
                        "Batch mode: output DIRECTORY (default: tts_batch/).")
    p.add_argument("--format", default="wav",
                   choices=("wav", "pcm", "f32", "flac", "mp3", "aac", "opus"),
                   help="Audio response format (default: wav). CrispASR accepts "
                        "wav/pcm/f32; Voxtral (vLLM) accepts wav/flac/mp3/aac/opus/pcm.")
    p.add_argument("--model", default=None,
                   help="OpenAI `model` field (default: not sent). CrispASR does not "
                        "need it; vLLM validates it, so against voxtral-tts pass "
                        "mistralai/Voxtral-4B-TTS-2603 if a modelless request is rejected.")
    p.add_argument("--speed", type=float, default=1.0,
                   help="Speech speed, 0.25 to 4.0 (default: 1.0).")
    p.add_argument("--timeout", type=float, default=300.0,
                   help="HTTP timeout in seconds (default: 300).")
    p.add_argument("--raw-spaces", action="store_true",
                   help="Send the text byte for byte instead of replacing "
                        "non-breaking and thin spaces with ordinary ones. The "
                        "replacement is on by default because Voxtral TTS "
                        "degenerates on those characters (measured: 47.7 -> "
                        "10.5 pct median WER on the sentences that carry them); "
                        "use this when the exact bytes are the point.")
    p.add_argument("--seed", type=int, default=None,
                   help="Integer seed for reproducible synthesis (default: not sent). "
                        "With sampling on, different seeds change the wording. NOTE: this "
                        "is a TADA/CrispASR knob; against voxtral-tts it lands in stage 0's "
                        "sampling params, which are forced by construction there, so it "
                        "does nothing: use --noise-seed instead (see the Voxtral seed "
                        "notes in CLAUDE.md).")

    voice = p.add_argument_group(
        "voice / cloning",
        "Select the speaking voice. TADA honours --voice per request (CrispASR "
        "#201): pass a tada-ref-*.gguf path or a cache/registry name and the "
        "backend reloads it live; 'default'/'auto'/unset keeps the loaded ref. A "
        ".wav is still rejected by TADA (bake a ref offline first). Also honoured "
        "by qwen3-tts base, vibevoice-1.5b, ... For voxtral-tts, pass one of its 20 "
        "built-in embedding names (VOXTRAL_VOICES in this file).")
    voice.add_argument("--voice", default=None,
                       help="Voice name or reference path (TADA reloads it live; "
                            "'default'/'auto'/unset keeps the server's current voice). "
                            "For Voxtral: a built-in embedding name, e.g. fr_male, "
                            "neutral_female.")
    voice.add_argument("--ref-audio", default=None,
                       help="VOXTRAL ONLY: clone the voice in this reference clip instead of "
                            "using a preset. Takes a local audio file (ogg/wav/mp3/flac, sent "
                            "inline as base64), an http(s) URL, or a file:// URI resolved inside "
                            "the container. Mutually exclusive with --voice. No baking and no "
                            "registration: the clip is handed to the model as multimodal input "
                            "per request, so nothing persists server-side.")
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
                          help="Acoustic classifier-free guidance weight. Server default 1.6. "
                               "TADA only: on a Voxtral server this is dropped silently, "
                               "use --cfg-alpha there.")
    acoustic.add_argument("--noise-temp", type=float, default=None,
                          help="Acoustic noise temperature. Server default 0.9.")

    # Voxtral (vllm-omni) knobs. --cfg-alpha, --euler-steps and --noise-seed are
    # the odd ones out: unlike every group above they do not map to flat server
    # fields but ride inside the request's `extra_params` object (see
    # build_speech_body), while --max-new-tokens is a plain field. Together they
    # are the whole per-request surface for Voxtral quality/length/
    # reproducibility. The last two need the voxtral-tts image built from our
    # patched vllm-omni clone (voxtral/vllm-omni, branch `perso`); a stock server
    # accepts and ignores them.
    voxtral = p.add_argument_group(
        "Voxtral (vLLM)",
        "Per-request knobs the voxtral-tts service honours. Unset = the value the "
        "server booted with (VOXTRAL_CFG_ALPHA / VOXTRAL_EULER_STEPS / "
        "VOXTRAL_MAX_TOKENS in docker-compose.yml; vllm-omni deploy defaults 1.2, "
        "7 and 2048).")
    voxtral.add_argument("--cfg-alpha", type=float, default=None,
                         help="Voxtral classifier-free guidance scale for THIS request "
                              "(higher = tighter text adherence; our server default is "
                              "2.0). Measured band: 1.2-2.0 for the shipped voices, "
                              "1.5-2.0 for the hybrids, 1.5 being the floor worth "
                              "retrying a bad take at. Below 1.2 takes stop terminating, "
                              "and 3.0+ distorts and risks a take that never finishes. "
                              "Distinct from TADA's --cfg-scale. 1.0 disables CFG and "
                              "yields garbled, off-text audio.")
    voxtral.add_argument("--euler-steps", type=int, default=None,
                         help="Euler ODE steps for THIS request's flow-matching acoustic "
                              "transformer (server default 7, or VOXTRAL_EULER_STEPS): the "
                              "compute-vs-quality lever, cost scaling with audio length. "
                              "Honoured only when concurrent requests agree on the value, "
                              "and it runs that request eagerly instead of from a CUDA "
                              "graph, so a sweep is best run one call at a time.")
    voxtral.add_argument("--noise-seed", type=int, default=None,
                         help="Seed THIS request's per-frame flow-matching noise, so the "
                              "same text plus the same value gives identical audio with no "
                              "restart. The only route to a reproducible Voxtral take; not "
                              "the same as --seed, which Voxtral cannot act on.")
    voxtral.add_argument("--dump-codes", type=str, default=None, metavar="TAG",
                         help="Ask the server to append the codes stage 0 emits for THIS "
                              "request to <VOXTRAL_CODE_DUMP_DIR>/TAG.i16 (raw int16, one "
                              "row of 37 per 12.5 Hz frame). Does not change the audio: it "
                              "is how audio/codes training pairs are collected, the open "
                              "checkpoint having shipped no encoder to produce them. Needs "
                              "the patched image and the env var set on the service.")
    voxtral.add_argument("--max-new-tokens", type=int, default=None,
                         help="Max audio frames for THIS request, i.e. seconds x 12.5 "
                              "(deploy default 2048 = 163.84 s, server hard cap 4096 = "
                              "327.68 s). Longer text is cut off mid-sentence. Also "
                              "bounded by stage 0's max_model_len, shared with the "
                              "prompt: see VOXTRAL_MAX_MODEL_LEN.")
    args = p.parse_args()

    warn_on_mixed_backend_knobs(args)

    # Mirrors the startup guardrail in voxtral/entrypoint.sh, but warns instead
    # of refusing: cfg_alpha=1.0 turns CFG off and the flow-matching decoder
    # then hallucinates phonetically-plausible-but-wrong audio. Per request that
    # is recoverable (the next call is unaffected) and sweeping the knob is what
    # this client exists for, so it is allowed through, loudly.
    if args.cfg_alpha is not None and abs(args.cfg_alpha - 1.0) < 1e-9:
        print("warning: --cfg-alpha 1.0 disables classifier-free guidance; expect "
              "garbled, off-text audio (voxtral/entrypoint.sh refuses this value at "
              "startup for that reason)", file=sys.stderr)

    # vllm-omni validates this field against _TTS_MAX_NEW_TOKENS_MIN/MAX (1..4096)
    # and answers 400, so catch both ends here rather than spend a round trip.
    # Above 4096 the only route to longer audio is raising VOXTRAL_MAX_TOKENS *and*
    # VOXTRAL_MAX_MODEL_LEN at startup, since 4096 is hardcoded server-side.
    if args.max_new_tokens is not None:
        if args.max_new_tokens < 1:
            p.error("--max-new-tokens must be at least 1")
        if args.max_new_tokens > VOXTRAL_MAX_NEW_TOKENS_CAP:
            p.error(
                f"--max-new-tokens cannot exceed {VOXTRAL_MAX_NEW_TOKENS_CAP} "
                f"({VOXTRAL_MAX_NEW_TOKENS_CAP / VOXTRAL_FRAME_RATE_HZ:.2f} s of audio); "
                "vllm-omni rejects more. For longer utterances raise VOXTRAL_MAX_TOKENS "
                "and VOXTRAL_MAX_MODEL_LEN in docker-compose.yml instead")

    base_url = args.url.rstrip("/")

    if args.check:
        return cmd_check(base_url, voice=args.voice)

    if args.ref_audio:
        # Server-side this is a hard either/or (`_validate_voxtral_tts_request`
        # wants exactly one), so catch it here rather than as a 400.
        if args.voice or args.voice_name:
            p.error("--ref-audio is mutually exclusive with --voice / --voice-name")
        args.ref_audio = resolve_ref_audio(args.ref_audio)

    if args.voice_name:
        if args.voice:
            p.error("--voice and --voice-name are mutually exclusive")
        args.voice = ensure_baked_voice(args.voice_name, args.voice_language, args.force_bake)
        if not args.text and not args.batch:
            # Bake-only: emit the resolved container voice path on stdout.
            print(args.voice)
            return 0

    if args.batch:
        return cmd_batch(base_url, Path(args.batch), args)

    if not args.text:
        p.error("--text is required unless --check, --batch or --voice-name is given")

    return cmd_synth(base_url, _body_from_args(args.text, args),
                     Path(args.output or "tts_out.wav"), args.timeout)


if __name__ == "__main__":
    sys.exit(main())
