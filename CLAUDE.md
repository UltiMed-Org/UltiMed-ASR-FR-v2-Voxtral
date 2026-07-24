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
   - `crispasr` (ASR) on host **:8002**, active backend `voxtral4b` (local f16 GGUF
     `voxtral-mini-4b-realtime.gguf`, `AUTO_DOWNLOAD=0`, transcribe-only). The compose
     comments mark ACTIVE vs ALT and keep `parakeet` / other voxtral variants as
     alternatives. Protected by `CRISPASR_API_KEYS`.
   - `crispasr-tts` (TTS) on host **:8003** — active backend switches often as models
     get tested (the compose comments mark ACTIVE vs ALT; currently
     `qwen3-tts-1.7b-voicedesign`, local GGUFs, French output). No API key.
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
its own image from `voxtral/Dockerfile`, a **thin layer on the official
`vllm/vllm-omni:v0.22.0-x86_64` image** (which already bundles the matching vLLM +
vllm-omni + deploy YAMLs); the layer just adds a passwd entry for the runtime uid and our
entrypoint. `voxtral/entrypoint.sh` copies the bundled deploy YAML, patches the tuning
knobs into it, and runs `vllm serve MODEL --omni --deploy-config <patched>`. (History:
0.18 built `FROM vllm/vllm-openai` + `pip install vllm-omni==0.18.0` + a hand-built
flash-attn wheel and ran `vllm-omni serve --stage-configs-path`; `voxtral/install.sh` is
that older hand-run recipe.)

- **Port clash by design:** it binds host **:8003**, the same port as `crispasr-tts`
  (TADA), so only one TTS can be up at a time. Do NOT `sudo docker compose up` with no
  args (that starts everything and the two collide on :8003); start it explicitly:
  `sudo docker compose up -d --build voxtral-tts`.
- **Client:** the root `tts.py`, same as for crispasr-tts (the former
  `voxtral/query_test.py` was folded into it). The **voice is per request** (20
  built-in voices, `VOXTRAL_VOICES` in tts.py: `fr_male`, `fr_female`,
  `neutral_female`, ...) and should be passed explicitly:
  `./tts.py -t "Bonjour." --voice neutral_female -o out.wav`. If vLLM rejects a
  request without a model name, add `--model mistralai/Voxtral-4B-TTS-2603`.
- **Per-request fields Voxtral actually honors:** `input`, `voice` (or `ref_audio` for
  cloning), `response_format`, and `speed` (post-processing playback rate, non-streaming
  only). It **ignores** `instructions`, `language`, `task_type` and `max_new_tokens`:
  those only take effect on the Qwen3-TTS / Fish paths in vllm-omni's
  `serving_speech.py` (`_build_tts_params`), whereas Voxtral builds its prompt from just
  `SpeechRequest(input, voice|ref_audio)`. So of tts.py's flags only `--voice`, `--speed`,
  `--format` (and `--model`) matter here; the TADA knobs must be left unset (they are
  opt-in, never sent unless given, and vllm-omni's schema does not define them). Sampling
  knobs (temperature/seed/top_p) are NOT per request either; they live in the startup
  YAML (see below).
- **Startup / tuning env vars** (read by `entrypoint.sh`, surfaced commented in the
  service's `environment:`; all editable without a rebuild, just `up -d`). Voxtral is a
  **two-stage** pipeline: stage 0 (the AR LLM, greedy at temp 0.0, `enforce_eager: false`
  so already compiled/CUDA-graph'd) then stage 1 (the acoustic decoder). At boot
  `entrypoint.sh` copies the bundled `deploy/voxtral_tts.yaml`, patches it with pyyaml
  (0.22 schema: a top-level `stages:` list with flat fields + nested
  `default_sampling_params`; stages are keyed by `stage_id`), and passes it via
  `vllm serve --omni --deploy-config`:
  - `VOXTRAL_ACOUSTIC_TEMP` -> stage_id 1 sampling temperature. Entrypoint default **0.7**
    (lowered from upstream 0.9 for steadier, less artifacty audio; raise toward 0.9 for
    more expressive/varied output). This is the main audio-quality lever.
  - `VOXTRAL_SEED` -> seed on both stages (fixed = reproducible; change for a different
    take). Not settable per request.
  - `VOXTRAL_MAX_NUM_SEQS` -> both stages' `max_num_seqs` (packaged **32**). Unset = 32.
    Note the acoustic stage is `codec-bs=1`, so per the maintainers audio throughput
    **saturates around client concurrency 4-8**; raising this past that buys little. (The
    0.18 `runtime.defaults.max_inflight` lever no longer exists in the 0.22 schema.)
  - `VOXTRAL_GPU_MEM_UTIL` -> stage_id 0's `gpu_memory_utilization`, patched into the YAML
    (packaged **0.8**; unset = 0.8). Stage 0 (0.8) + stage 1 (0.1) + each engine's CUDA
    context + any desktop GPU usage overflow a 24GB card shared with a display, and stage 1
    then dies at boot with *"Free memory on device cuda:0 ... less than desired GPU memory
    utilization"*. Lower stage 0 to ~**0.7** to make both stages fit (the AR LLM weights are
    ~8GB, so you only give up unused KV-cache headroom). The packaged 0.8/0.1 split is tuned
    for 80GB H100/H200 cards.
  - `VOXTRAL_CFG_ALPHA` -> stage_id 0 classifier-free-guidance scale, patched into that
    stage's `default_sampling_params.extra_args.cfg_alpha` (the same field vllm-omni reads
    per-request cfg_alpha from). Deploy default **1.2**. Nudge up for tighter text
    adherence. The entrypoint **refuses `1.0`**: disabling CFG makes the flow-matching
    decoder hallucinate phonetically-plausible-but-wrong audio (stops following the text).
  - `VOXTRAL_EULER_STEPS` -> the **Euler ODE / NFE step count** of the stage-1 flow-matching
    acoustic decoder (`n_decoding_steps`): the primary compute<->quality lever (more steps =
    steadier, higher-fidelity audio for more compute; CFG doubles each step's cost, so this
    is the pricey one). Unlike the knobs above it is **not** a deploy-YAML field, env var, or
    per-request param in vllm-omni: the value is read **only** from the model's `params.json`
    at `multimodal.audio_model_args.acoustic_transformer_args.n_decoding_steps`, and this
    model doesn't ship the key, so vllm-omni's mistral parser (`parsers/voxtral_tts.py`)
    **warns and falls back to 7**. So `entrypoint.sh` does exactly what that warning asks:
    when the var is set it `hf_hub_download`s just `params.json` (works on first boot before
    the ~9GB weights land; uses `HF_TOKEN` if gated), injects `n_decoding_steps`, and
    re-patches idempotently every boot (surviving HF restoring the file). Unset = the
    vllm-omni default **7** (raise toward 8-16 for quality, drop to 3-4 for speed/latency).
  - `HF_TOKEN` (if the HF repo is gated), `VLLM_EXTRA_ARGS` (extra `vllm serve` flags).
    Weights (~9GB) download into the mounted `./cache` via `HF_HOME`.
- **Status:** upgraded to the prebuilt `vllm/vllm-omni:v0.22.0` image and the 0.22 deploy
  schema, statically validated (the YAML patcher runs against the real v0.22.0 deploy
  config) but not yet run end-to-end in Docker; the first `--build` + smoke call is the
  real check. The prebuilt image ships no standalone flash-attn; the Dockerfile re-adds it
  from a prebuilt wheel (the torch2.10 cu13 one, ABI-compatible with the image's torch2.11
  and verified in-image, since no torch2.11 wheel exists yet) so stage 1 uses flash-attn
  instead of the SDPA fallback.

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
  compose service, `install.sh` is the original hand-run recipe, and `env.reference` is a
  captured `pip freeze`. The client is the root `tts.py` (the old `voxtral/query_test.py`
  was merged into it).
