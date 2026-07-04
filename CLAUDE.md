# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## What this repo is

This is a **deployment + testing harness** for [CrispASR](https://github.com/CrispStrobe/CrispASR),
an OpenAI-compatible ASR/TTS server. It is **not** the CrispASR source. The goal is to
run CrispASR via docker-compose and quickly test new ASR/TTS models as they ship in new
CrispASR releases, without touching client code.

Two moving parts:

1. **`docker-compose.yml`** runs two containers from the prebuilt image
   `ghcr.io/crispstrobe/crispasr:main-cuda`:
   - `crispasr` (ASR) on host **:8002** — active backend `parakeet` (auto-downloaded).
     Protected by `CRISPASR_API_KEYS`.
   - `crispasr-tts` (TTS) on host **:8003** — active backend `tada` (HumeAI TADA 3B
     multilingual, French). No API key.
   Both have `pull_policy: always` and a watchtower `autoupdate` label, so new releases
   roll in automatically. A third, optional service `voxtral-tts` (locally built, vLLM,
   not the CrispASR image) offers an alternative TTS on the same :8003; see "Voxtral TTS".
2. **Two stdlib-only client scripts at the root**, run via `uv` (PEP 723 inline deps,
   no venv): `tts.py` (→ :8003) and `asr.py` (→ :8002). Support tooling lives in `utils/`.

## Common commands

```bash
# Bring up / restart a service after editing compose:
sudo docker compose up -d crispasr        # ASR
sudo docker compose up -d crispasr-tts    # TTS
sudo docker compose logs -f crispasr-tts  # follow logs (backends log verbosely)

# Confirm a server is alive and the model actually loads (health + a real smoke call):
./tts.py --check
./asr.py --check --api-key "$KEY"         # ASR endpoint is key-protected (see below)

# TTS: synthesize
./tts.py -t "Bonjour." -o out.wav
./tts.py -t "Salut." --voice /models/tada-ref-alicia.gguf     # pick a baked voice, live
./tts.py -t "Salut." --voice-name alicia                       # bake ./voices/alicia then synth

# ASR: transcribe
./asr.py audio.mp3 --api-key "$KEY"
./asr.py meeting.ogg --format diarized_json --diarize --diarize-method pyannote --api-key "$KEY"

# Bake a TADA voice reference (torch encoder; first run pulls torch + weights):
uv run utils/clone_voice.py --name alicia
```

There is no build/lint/test suite here (the client scripts are the "tests"; the server is
a prebuilt image). Verify changes by running the script against a live server, or
`python -m py_compile <script>` for a quick syntax check.

## The core workflow: testing a new model

**You almost never edit the client scripts to try a new backend.** Both scripts are
backend-agnostic OpenAI clients. To swap a model, edit the service's `environment:` block
in `docker-compose.yml` and restart that service:

- `CRISPASR_BACKEND` — e.g. `parakeet`, `whisper`, `voxtral4b` (ASR); `tada`, `qwen3-tts-*`,
  `vibevoice`, `orpheus` (TTS). Registry keys are exact; see the long comments in
  `docker-compose.yml` (e.g. `tada` works but `tada-3b-ml` is not a registry key).
- `CRISPASR_AUTO_DOWNLOAD=1` fetches the model into `/cache`; `=0` uses the local file in
  `CRISPASR_MODEL` (a `/models/*.gguf` path).
- `CRISPASR_EXTRA_ARGS` — extra CLI flags (e.g. `--verbose --voice /models/tada-ref-x.gguf`,
  diarization flags). **Do not quote-wrap this value**: the entrypoint runs
  `eval "args+=($CRISPASR_EXTRA_ARGS)"`, so surrounding quotes collapse it into one arg and
  break startup.

Then `sudo docker compose up -d <service>`. Every backend registers both `/v1/audio/*`
routes; non-matching backends return 400 pointing at `POST /load`.

## Client architecture: the request-vs-startup contract

Both `tts.py` (`build_speech_body`) and `asr.py` (`build_fields`) follow one rule: **only
fields the user explicitly sets are sent**; anything left unset (`None`) is dropped from
the request so the server applies the default it was *started* with (env var / CLI flag).
This deliberately keeps per-request settings from leaking into the shared, long-lived
backend context. When adding a knob, keep it opt-in (default `None`) unless it must always
be sent (`input` / `response_format` / the audio file).

## Auth asymmetry (easy to trip on)

The **ASR** service sets `CRISPASR_API_KEYS`, so `/v1/audio/transcriptions` needs a Bearer
key (`asr.py --api-key` or `$CRISPASR_API_KEY`); only `/health` is public. The **TTS**
service has no key, so `tts.py` sends none.

## TADA voice cloning (TTS-specific)

Voice cloning currently means TADA. Flow: `tts.py --voice-name X` shells out to
`utils/clone_voice.py`, which bakes `./voices/X.<audio>` + `X.txt` into
`models_tts_ignore_backups/tada-ref-X.gguf` (mounted at `/models`) using the torch encoder
`utils/convert-tada-ref-to-gguf.py`. `clone_voice.py` owns the naming/mount convention;
`tts.py` never duplicates it (it calls `--print-path`). The voice is switchable per request
(CrispASR #201): `--voice tada-ref-x.gguf` reloads live; `default`/`auto`/omitted keeps the
loaded one. A raw `.wav` is **not** accepted by TADA — bake a ref first.

`utils/tada_reference_engine.py` runs the **upstream** HumeAI TADA model directly (bypassing
CrispASR's C++ port) to decide whether silent/garbled output is a CrispASR port bug vs a
bad model/reference/transcript. See `utils/README.md`.

## Voxtral TTS (alternative TTS via vLLM, not CrispASR)

`voxtral-tts` is a **separate service** for a different text->audio model,
`mistralai/Voxtral-4B-TTS-2603`, which is a **vLLM** model, NOT a CrispASR backend. (In
CrispASR, `voxtral`/`voxtral4b` are ASR, i.e. speech->text; the "multiple voxtral models"
in `docker-compose.yml` and `models_ignore_backups/` are all ASR.) So this service uses
none of the `ghcr.io/crispstrobe/crispasr` image or the `CRISPASR_*` env vars. It builds
its own image from `voxtral/Dockerfile` (base `vllm/vllm-openai` + the `vllm-omni`
extension, versions pinned to match `voxtral/install.sh`), and `voxtral/entrypoint.sh`
runs `vllm-omni serve` the same way `install.sh` does by hand.

- **Port clash by design:** it binds host **:8003**, the same port as `crispasr-tts`
  (TADA), so only one TTS can be up at a time. Do NOT `sudo docker compose up` with no
  args (that starts everything and the two collide on :8003); start it explicitly:
  `sudo docker compose up -d --build voxtral-tts`.
- **Client:** `voxtral/query_test.py` (PEP 723, run via `uv`), a small OpenAI TTS client
  hitting `:8003/v1/audio/speech`. The model is fixed; the **voice is per request** (20
  built-in voices: `fr_male`, `fr_female`, `neutral_female`, ...):
  `uv run voxtral/query_test.py --input "Bonjour." --voice neutral_female --output-path out.wav`
- **Per-request fields Voxtral actually honors:** `input`, `voice` (or `ref_audio` for
  cloning), `response_format`, and `speed` (post-processing playback rate, non-streaming
  only). It **ignores** `instructions`, `language`, `task_type` and `max_new_tokens`:
  those only take effect on the Qwen3-TTS / Fish paths in vllm-omni's
  `serving_speech.py` (`_build_tts_params`), whereas Voxtral builds its prompt from just
  `SpeechRequest(input, voice|ref_audio)`. So `query_test.py` exposes `--voice`, `--speed`
  and `--response-format` only. Sampling knobs (temperature/seed/top_p) are NOT per
  request either; they live in the startup YAML (see below).
- **Tuning env vars** (read by `entrypoint.sh`, surfaced commented in the service's
  `environment:`): `VLLM_GPU_MEM_UTIL` (shrink GPU memory when sharing the card),
  `HF_TOKEN` (if the HF repo is gated), `VLLM_EXTRA_ARGS` (extra `vllm-omni serve` flags).
  Weights (~9GB) download into the mounted `./cache` via `HF_HOME`.
- **Status:** ported faithfully from the hand-run `install.sh` but not yet verified
  end-to-end in Docker; the `vllm/vllm-openai:v0.18.1` base tag and the GPU-memory fit may
  need adjustment on the first `--build`.

## Layout notes

- **`CrispASR_source/`** (gitignored) — a vendored checkout of the CrispASR source, kept as
  the **API contract reference** and an optional build context. When unsure about a
  server field or flag, read `CrispASR_source/docs/server.md` (endpoints + every form/body
  field), `docs/tts.md`, `docs/cli.md` instead of guessing.
- **Host→container mounts:** `./models_ignore_backups/` → `/models` (ASR local models);
  `./models_tts_ignore_backups/` → `/models` (TTS baked refs); `./cache/` → `/cache`
  (auto-downloaded models land here). `voices/` is a symlink to the reference-clip source.
- Model/audio binaries are gitignored (`**/*.gguf`, `**/*.mp3`, `**/*.wav`, the `models_*`
  dirs, `cache`, `CrispASR_source`, `voices`), so the tracked repo is just compose + the
  scripts + `utils/` + `voxtral/`.
- `voxtral/` holds the Voxtral TTS experiment (serving Voxtral TTS via **vLLM**, not
  CrispASR; see "Voxtral TTS" above): `Dockerfile` + `entrypoint.sh` back the `voxtral-tts`
  compose service, `install.sh` is the original hand-run recipe, `query_test.py` is the
  client (it overlaps with `tts.py`), and `env.reference` is a captured `pip freeze`.
