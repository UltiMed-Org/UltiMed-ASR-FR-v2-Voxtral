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
   - `crispasr` (ASR), active backend `whisper` (`ggml-large-v3-turbo.bin`, pinned
     via `--hf-repo`, multilingual, `-fa` flash-attn, transcribe-only). The compose
     comments mark ACTIVE vs ALT and keep lighter whisper models (`base`/`small`),
     `parakeet` (quality: Q8_0 + MAES), and `voxtral4b` as alternatives. Protected by
     `CRISPASR_API_KEYS`. This service is **replicated** (`deploy.replicas: 4`, no
     `container_name`/host ports) and fronted by `crispasr-lb`, a small `caddy:2-alpine`
     load balancer that owns host **:8002** and `least_conn`-balances across the
     replicas (they underutilize the GPU per request, so replicas lift throughput;
     each is still mutex-serialized, one request at a time). Scale live with
     `docker compose up -d --scale crispasr=N crispasr-lb`; config in `./Caddyfile`.
   - `crispasr-lb` (`caddy:2-alpine`) on host **:8002**: the ASR load balancer above.
     Clients (`asr.py`, any OpenAI client) hit :8002 unchanged; it discovers the
     `crispasr` replicas via Docker DNS (`dynamic a`, so `--scale` needs no LB edit).
   - `crispasr-tts` (TTS) on host **:8003**: active backend switches often as models
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
sudo docker compose up -d crispasr crispasr-lb   # ASR (4 replicas) + its load balancer
sudo docker compose up -d --scale crispasr=6 crispasr-lb  # rescale ASR replicas live
sudo docker compose up -d crispasr-tts           # TTS
sudo docker compose logs -f crispasr-tts         # follow logs (backends log verbosely)

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

- `CRISPASR_BACKEND`: e.g. `parakeet`, `whisper`, `voxtral4b` (ASR); `tada`, `qwen3-tts-*`,
  `vibevoice`, `orpheus` (TTS). Registry keys are exact; see the long comments in
  `docker-compose.yml` (e.g. `tada` works but `tada-3b-ml` is not a registry key).
- `CRISPASR_AUTO_DOWNLOAD=1` fetches the model into `/cache`; `=0` uses the local file in
  `CRISPASR_MODEL` (a `/models/*.gguf` path).
- `CRISPASR_EXTRA_ARGS`: extra CLI flags (e.g. `--verbose --voice /models/tada-ref-x.gguf`,
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

**One exception, and it rewrites the text:** `tts.py` replaces the exotic Unicode spaces
(`\u00a0`, `\u2009`, `\u202f`, and siblings) with ordinary ones before sending `input`,
via `normalise_spaces`. This is on by default because Voxtral TTS degenerates on them:
it starts the sentence correctly then loops syllables until it hits the token ceiling,
burning minutes of GPU time and returning an unusable file with a 200. Measured in the
`PROBE` run (`cache/cloning/FLEURS/RESULTS.md`): on the sentences that carry one, median
WER 47.7 -> 10.5 pct and unusable takes 16 of 48 -> 3 of 48, with the guillemets shown to
be innocent. Nothing else about the text is touched. Pass `--raw-spaces` (or
`raw_spaces=True`) when the exact bytes matter; `voxtral/cloning/sweep.py` does, since a
benchmark must send its corpus verbatim.

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
loaded one. A raw `.wav` is **not** accepted by TADA, so bake a ref first.

`utils/tada_reference_engine.py` runs the **upstream** HumeAI TADA model directly (bypassing
CrispASR's C++ port) to decide whether silent/garbled output is a CrispASR port bug vs a
bad model/reference/transcript. See `utils/README.md`.

## Voxtral TTS (alternative TTS via vLLM, not CrispASR)

`voxtral-tts` is a **separate service** for a different text->audio model,
`mistralai/Voxtral-4B-TTS-2603`, which is a **vLLM** model, NOT a CrispASR backend. (In
CrispASR, `voxtral`/`voxtral4b` are ASR, i.e. speech->text; the "multiple voxtral models"
in `docker-compose.yml` and `models_ignore_backups/` are all ASR.) So this service uses
none of the `ghcr.io/crispstrobe/crispasr` image or the `CRISPASR_*` env vars. It builds
its own image from `voxtral/Dockerfile`, on top of the official
`vllm/vllm-omni:v0.22.0-x86_64` image (which already bundles the matching vLLM +
vllm-omni + deploy YAMLs). The layer adds a passwd entry for the runtime uid, flash-attn,
our entrypoint, and **reinstalls `vllm_omni` from a local fork** (see below).
`voxtral/entrypoint.sh` copies the bundled deploy YAML, patches the tuning knobs into it,
and runs `vllm serve MODEL --omni --deploy-config <patched>`. (History: 0.18 built
`FROM vllm/vllm-openai` + `pip install vllm-omni==0.18.0` + a hand-built flash-attn wheel
and ran `vllm-omni serve --stage-configs-path`; `voxtral/install.sh` is that older
hand-run recipe.)

- **The fork (`voxtral/vllm-omni`, branch `perso`) is load-bearing.** Upstream vllm-omni
  has no request-level route to Voxtral's quality levers, and hides that fact:
  `OpenAICreateSpeechRequest` declares no `model_config`, so pydantic v2's default
  `extra='ignore'` **drops unknown flat keys and still answers 200**. A client sending
  `{"cfg_alpha": 1.5, "seed": 7}` looks like it worked while every call runs at the
  server's startup defaults. The `perso` branch fixes that in two places: request routing
  (`serving_speech.py::_apply_voxtral_request_overrides`, plus real `cfg_alpha` /
  `euler_steps` / `acoustic_temperature` fields on the schema) and the model
  (`voxtral_tts.py::_make_frame_noise` / `_extract_euler_steps`). Patch 0005 adds a
  third fix, `_resolve_voice_embedding`: stock vllm_omni `hf_hub_download`s every voice
  named in `params.json` from the REMOTE repo, so a voice added locally (see
  `voxtral/cloning/`) 404s at model init and boot-loops the container. The fork resolves
  the cache first and skips a voice it cannot find instead of raising. Note
  `HF_HUB_OFFLINE=1` is NOT an alternative: vLLM cannot resolve a repo-id model offline. The clone is
  **gitignored**: `voxtral/patches/*.patch` is the tracked backup and
  `voxtral/patches/README.md` has the recreate recipe, which a fresh checkout must run
  **before its first `--build`**. Regenerate those patches whenever `perso` gains a commit.
  The boot log states which copy is serving (`entrypoint: vllm_omni at ... is PATCHED
  (perso fork)` vs `STOCK`); `./voxtral/test_seed_determinism.sh doctor` checks the whole
  chain inside the container.

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
- **Voice cloning via `ref_audio` does NOT work on the open weights.** The route looks
  complete and is not: `serving_speech.py::_build_voxtral_prompt` encodes the clip with
  mistral_common and passes it as `multi_modal_data`, the stage-0 model is
  `SupportsMultiModal` with a real processor, and then
  `voxtral_tts_audio_generation.py::embed_multimodal` calls
  `audio_tokenizer.encode_waveforms()`, which raises *"requires encoder weights which are
  not available in the open-source checkpoint"*. The multimodal audio path IS the codec
  encoder, not a separate understanding tower, so cloning hits exactly the missing piece.
  Worse, that raise happens inside `execute_model`, which is fatal in vLLM: **one clone
  request killed EngineCore and every concurrent request with it** (2026-09-09, 500 then
  connection resets until restart). Fork patch 0006 refuses `ref_audio` at the serving
  layer so it costs one 400 instead of an outage; drop that guard if a checkpoint ever
  ships the encoder, since everything else on the path is complete. `tts.py --ref-audio`
  exists and is what surfaced this; it stays for that eventual case. The **decoder** does
  ship in full, which is why decoder inversion (`PLAN_VOXTRALTTS_CLONING_TEST.md` Track B)
  is the remaining route to a cloned voice. `soxr` is in the Dockerfile because the
  cloning path needs it to resample; unrelated to the encoder problem.
- **Per-request fields Voxtral actually honors:** `input`, `voice` (or `ref_audio` for
  cloning), `response_format`, `speed` (post-processing playback rate, non-streaming
  only), `max_new_tokens` (see the length section below), and `extra_params`, which is
  how `cfg_alpha`, `voxtral_noise_seed`, `voxtral_euler_steps` and `voxtral_dump_codes`
  travel (`tts.py --cfg-alpha / --noise-seed / --euler-steps / --dump-codes`; all but
  the first need the fork). Our fork also
  accepts `cfg_alpha`, `euler_steps`, `acoustic_temperature` and `seed` as flat fields,
  but `tts.py` deliberately uses `extra_params` because that route also works against a
  stock server. It **ignores** `instructions`, `language` and `task_type`: those
  only take effect on the Qwen3-TTS / Fish paths in vllm-omni's `serving_speech.py`
  (`_build_tts_params`), whereas Voxtral builds its prompt from just
  `SpeechRequest(input, voice|ref_audio)`. `max_new_tokens` is the exception that looks
  like it should be in that ignored list but is not: it is applied *after* prompt
  building by a generic override gated on `_SAMPLING_MAX_TOKENS_TTS_MODEL_TYPES`, which
  includes `voxtral_tts`. So of tts.py's flags only `--voice`, `--speed`, `--cfg-alpha`,
  `--euler-steps`, `--noise-seed`, `--max-new-tokens`, `--format` (and `--model`) matter
  here; the TADA knobs must be left unset (they are opt-in, never sent unless given, and
  vllm-omni's schema does not define them). Sampling knobs (temperature/top_p) are NOT per
  request; they live in the startup YAML (below). `--seed` remains pointless whatever the
  server: it lands in `SamplingParams`, and stage 0's sampler is handed logits with a
  single finite entry, so its result is forced. Use `--noise-seed` for reproducibility.
- **`--cfg-alpha`, the one per-request quality lever:** `serving_speech.py` merges a
  request's `extra_params` dict into stage 0's `SamplingParams.extra_args`, and the
  model reads one cfg_alpha per request out of it
  (`model_executor/models/voxtral_tts/voxtral_tts.py::_extract_cfg_alpha`, falling back
  to 1.2). So `./tts.py -t "..." --voice fr_male --cfg-alpha 1.5` overrides
  `VOXTRAL_CFG_ALPHA` (which this repo sets to **2.0**, see below for the measured band)
  for that one call, no restart, and sweeps need no `up -d` between
  values. Being per request is also what lets a dataset run carry a **per-voice** value,
  or retry a bad take at 1.5 without touching the server. tts.py keeps it opt-in like every other knob (omitted when unset, server keeps
  its deploy default) and only **warns** on the known-bad `1.0` rather than refusing it
  the way `entrypoint.sh` does at startup: per request the damage stops at that call.
  Do not confuse it with TADA's `--cfg-scale` (different backend, field and default).
- **Audio length: the 163.84 s cap and how to move it.** Stage 0 emits exactly one audio
  frame per decode step at 12.5 Hz (the checkpoint's `params.json`,
  `multimodal.audio_model_args.audio_encoding_args.frame_rate`), so **tokens / 12.5 =
  seconds** and the deploy YAML's stage-0 `default_sampling_params.max_tokens: 2048`
  means one request maxes out at **163.84 s**. Overrun is silent: the audio is cut off
  mid-sentence with `finish_reason: "length"` and HTTP 200. This is **not** a model limit
  (the checkpoint says `max_seq_len 65536`, `max_position_embeddings 128000`). Two ways
  up: `./tts.py --max-new-tokens N` per request (server-capped at vllm-omni's
  `_TTS_MAX_NEW_TOKENS_MAX = 4096`, i.e. 327.68 s), or `VOXTRAL_MAX_TOKENS` at startup
  (uncapped). Either way the **real** ceiling is stage 0's `max_model_len` (packaged
  4096), because the prompt's text tokens and the generated frames share that one
  context, so a `max_tokens` past it is simply never reached: raise
  `VOXTRAL_MAX_MODEL_LEN` too, and keep it above `VOXTRAL_MAX_TOKENS` + the prompt
  (`entrypoint.sh` warns when it is not). Stage 1's own `max_tokens: 2048` is a red
  herring: `generator2tokenizer_async_chunk` feeds it 50-frame windows
  (`codec_left_context_frames` 25 + `codec_chunk_frames` 25), so it never sees the whole
  utterance and never bounds duration. Splitting long text per sentence client-side is
  usually still the better answer (steadier prosody, far lower time-to-first-audio, and a
  bad take costs seconds not minutes).
- **Startup / tuning env vars** (read by `entrypoint.sh`, surfaced commented in the
  service's `environment:`). Changing a knob's **value** needs no rebuild, just `up -d`
  (`restart` is not enough: it does not re-read `environment:`). Changing **`entrypoint.sh`
  itself does**, because `voxtral/Dockerfile` COPYs it into the image, so a knob added in
  this repo is silently ignored by an older image, with the var set and nothing happening.
  That already bit the retired `VOXTRAL_PER_REQUEST_SEED` once. To tell the two apart, read
  the `voxtral entrypoint: patched ... -> ...` line the script logs at boot: it lists every
  knob that image knows about and their effective values (and the `vllm_omni at ... is
  PATCHED/STOCK` line right after it says whether the fork is the one serving). Voxtral is a
  **two-stage** pipeline (topology frozen in `model_executor/models/voxtral_tts/pipeline.py`):
  stage 0 = `audio_generation`, the 4B AR LLM, text -> acoustic latents, greedy at temp 0.0,
  `enforce_eager: false` so already compiled/CUDA-graph'd; stage 1 = `audio_tokenizer`,
  latents -> 24 kHz waveform. Note stage 0 also owns the small (3-layer) flow-matching
  **acoustic transformer** that emits the 36 acoustic codebooks per frame, so the Euler-step
  knob below is a stage-0 cost, not a stage-1 one. At boot
  `entrypoint.sh` copies the bundled `deploy/voxtral_tts.yaml`, patches it with pyyaml
  (0.22 schema: a top-level `stages:` list with flat fields + nested
  `default_sampling_params`; stages are keyed by `stage_id`), and passes it via
  `vllm serve --omni --deploy-config`:
  - `VOXTRAL_ACOUSTIC_TEMP` -> stage_id 1 sampling temperature. Entrypoint default **0.7**
    (packaged 0.9). **Suspected no-op, do not tune blind:** stage 1 is `audio_tokenizer`, a
    deterministic codec decoder, and its model file
    (`models/voxtral_tts/voxtral_tts_audio_tokenizer.py`) contains no sampling path at all
    (no `temperature`, `logits`, `argmax` or `multinomial`); its `engine_output_type` is
    `audio`, so there are no logits to temper. Meanwhile stage 0's semantic head is a hard
    `argmax` (`voxtral_tts_audio_generation.py:605`) and its acoustic head is deterministic
    flow matching, so **the only stochasticity in the output is the fresh Gaussian the ODE
    starts from each frame** (`static_noise.normal_()` in
    `cuda_graph_acoustic_transformer_wrapper.py`), which draws from ambient CUDA RNG and is
    therefore not covered by `VOXTRAL_SEED` either. The fork makes the A/B cheap: send
    `acoustic_temperature` as a flat request field (no restart) with `--noise-seed` pinned,
    and see whether anything moves. The real quality levers appear to be
    `VOXTRAL_EULER_STEPS` and `VOXTRAL_CFG_ALPHA`.
  - `VOXTRAL_SEED` -> the vLLM **engine** seed (`ModelConfig.seed`) on both stages, *not* a
    sampling param. The distinction is the whole game, because **sampling params cannot
    affect Voxtral's audio at all**: stage 0 feeds vLLM's Sampler the output of
    `fake_logits_for_audio_tokens()`, a vocab-wide `-inf` row with exactly **one** finite
    entry (`voxtral_tts_audio_generation.py:1041`), so the winning token is forced whatever
    seed/temperature/top_p/top_k/repetition_penalty say, and stage 1 raises if asked for
    logits at all. The audio is produced *beside* that path in `make_omni_output`
    (`voxtral_tts.py:325-337`). So the **only** randomness in the whole pipeline is the
    Gaussian each frame's flow-matching ODE starts from (`torch.randn` at
    `voxtral_tts_audio_generation.py:527`, or `static_noise.normal_()` in the CUDA-graph
    path), drawn from the **global** torch RNG with no generator argument. vLLM seeds that
    global RNG from `ModelConfig.seed` and deliberately re-seeds it after warmup
    (`v1/worker/gpu_worker.py:712`, "so the random state is not affected by model
    initialization and profiling"), and the default is already **0**. Consequences:
    takes are **already deterministic per process** (request N after a fresh boot is
    reproducible if requests 1..N-1 were identical), `VOXTRAL_SEED` picks a *different*
    deterministic stream rather than switching determinism on, and reproducing one specific
    request needs a **restart** because every generated frame advances that RNG. Concurrency
    destroys it outright (interleaved draws). All three were **confirmed empirically** on
    2026-07-27 with tests A to C of `voxtral/test_seed_determinism.sh`: the request `seed`
    field changed nothing, both request positions reproduced across boots, and waveforms
    varied call to call inside one process. All of that still describes requests that carry
    **no** seed; the fork's per-request path below is what lifts the restart requirement,
    and `VOXTRAL_SEED` now only picks the stream for unseeded traffic.
  - **Per-request seeding (fork, no env var).** `tts.py --noise-seed N` sends
    `extra_params: {"voxtral_noise_seed": N}`, the same route `cfg_alpha` travels (a flat
    `seed` field works too on the fork, routed to the same key). The model then draws each
    frame's noise from a `torch.Generator` seeded `SplitMix64(seed, frame index)`, in
    `voxtral_tts.py::_make_frame_noise`, instead of from the global RNG. Same text plus
    same seed gives identical audio at any point in the container's life, **and it holds
    under concurrency**, because each row gets its own stream rather than a slice of one
    global sequence (the retired `sitecustomize.py` monkeypatch re-pinned that global RNG,
    which is why it could only promise reproducibility per batch composition). Requests
    without a seed keep upstream behaviour bit for bit. This mirrors what vllm-omni already
    does for Qwen3-TTS (`extra_args["qwen3_tts_request_seed"]`, read in
    `worker/gpu_model_runner.py`); Voxtral simply had no such hookup. One dependency
    remains: stage 0 must keep a non-empty `default_sampling_params.extra_args`, which is
    what makes vllm-omni pass per-request extras at all (`has_sampling_extra_args`), **so
    keep `VOXTRAL_CFG_ALPHA` set**. Test with `./voxtral/test_seed_determinism.sh D` (no
    restarts) and `E` for the cfg/steps half; diagnose with `... doctor` when either skips.
    Note there was **no** built-in alternative: `/v1/omni/sleep` + `/v1/omni/wakeup` only
    move weights (neither calls `set_random_seed`), and vLLM's `collective_rpc`, whose
    `compile_or_warm_up_model` *would* re-seed, has no HTTP route in 0.22.1.
  - `VLLM_BATCH_INVARIANT=1` (a vLLM env var, not read by `entrypoint.sh`; surfaced
    commented in the compose service) forces batch-invariant kernels so results stop
    depending on how requests happen to batch together. Needed for *bitwise* determinism on
    top of a pinned RNG, since reduction order otherwise shifts with batch shape. Costs
    throughput; only worth it when chasing a numerical diff.
  - `VOXTRAL_MAX_NUM_SEQS` -> both stages' `max_num_seqs` (packaged **32**). Unset = 32.
    Note the acoustic stage is `codec-bs=1`, so per the maintainers audio throughput
    **saturates around client concurrency 4-8**; raising this past that buys little. (The
    0.18 `runtime.defaults.max_inflight` lever no longer exists in the 0.22 schema.)
  - `VOXTRAL_OPT_LEVEL` -> both stages' `optimization_level` (vLLM's `-O`, default 2,
    passed through `StageDeployConfig.engine_extras`). Per vLLM's docs `-O3` is
    currently identical to `-O2`, so this is future-proofing (or `-O0`/`-O1` for faster
    boots). Compose runs 3. It does nothing on stage 1, which **cannot leave
    `enforce_eager: true`**: tried 2026-10-20, capture dies in
    `voxtral_tts.py::forward` on `(input_ids == 0).all()` ("operation not permitted
    when stream is capturing") and the container boot-loops. The whole audio_tokenizer
    forward is data-dependent Python (per-request splitting into lists), so making it
    capturable would mean a fork patch that graphs only the inner decoder at the fixed
    50-frame window shape. Note vLLM also logs that this model does not support
    `torch.compile`, so on stage 0 the `-O` level only governs CUDA graph mode.
  - `VOXTRAL_GPU_MEM_UTIL` -> stage_id 0's `gpu_memory_utilization`, patched into the YAML
    (packaged **0.8**; unset = 0.8). Stage 0 (0.8) + stage 1 (0.1) + each engine's CUDA
    context + any desktop GPU usage overflow a 24GB card shared with a display, and stage 1
    then dies at boot with *"Free memory on device cuda:0 ... less than desired GPU memory
    utilization"*. Lower stage 0 to ~**0.7** to make both stages fit (the AR LLM weights are
    ~8GB, so you only give up unused KV-cache headroom). The packaged 0.8/0.1 split is tuned
    for 80GB H100/H200 cards.
  - `VOXTRAL_CFG_ALPHA` -> stage_id 0 classifier-free-guidance scale, patched into that
    stage's `default_sampling_params.extra_args.cfg_alpha` (the same field vllm-omni reads
    per-request cfg_alpha from, so this only sets the **default**: `tts.py --cfg-alpha`
    overrides it per call, see above). Packaged default 1.2; **this repo runs 2.0**, and
    the entrypoint now injects 2.0 when the var is unset rather than leaving the field
    absent. The value is measured, not inherited: 16 French voices x 10 values from 0.1 to
    5.0 on one dense medical sentence with the noise seed pinned, listened to on
    2026-10-13 (`voxtral/cloning/cfg_sweep.sh`, PLAN stage A10). **2.0 is the starting
    point and 1.5 the floor worth retrying a bad take at**, which is why the default sits
    at the top of the usable band rather than the middle: there is room to go down and
    little to gain going up. Usable band **1.2-2.0 for the 20 shipped voices, 1.5-2.0 for
    the hybrids**, which need more guidance to hold their articulation. Asymmetric in both
    directions: below 1.2 it degrades much faster than the same distance above 2.0 (at 0.5
    and below, all 16 voices rambled to ~2.7x the sentence's natural length instead of
    stopping), and from 3.0 up distortion grows with the value while takes increasingly
    fail to terminate at all (above 3.0 the sweep's longest takes were exactly 327.68 s,
    i.e. they hit `VOXTRAL_MAX_TOKENS` and were cut off). Note A8's grid and A9's medical
    audition were both heard at **1.3**, the value compose carried from 2026-07-06.
    The entrypoint **refuses `1.0`**: disabling CFG makes the flow-matching
    decoder hallucinate phonetically-plausible-but-wrong audio (stops following the text).
    **Do not comment it out**: an empty stage-0 `extra_args` sets
    `has_sampling_extra_args=False` and vllm-omni then drops *every* per-request extra,
    the noise seed and euler steps included (the injected default now prevents that
    silently happening, but the explicit line documents the value).
  - `VOXTRAL_EULER_STEPS` -> the **Euler ODE / NFE step count** of the **stage-0**
    flow-matching acoustic transformer (`n_decoding_steps`): the primary compute<->quality
    lever (more steps = steadier, higher-fidelity audio for more compute; CFG batches a
    conditional + unconditional pass into each step, so it is 2x wide, and this is the pricey
    knob). The loop runs **once per 12.5 Hz frame**, starting from fresh noise each time with
    nothing carried between frames, so its per-frame quality effect is the same at second 1
    and second 160 while its **cost scales linearly with audio length**. It costs essentially
    **no extra VRAM**: the only `n_steps`-sized allocation is a `linspace` of `n_steps + 1`
    scalars, the ODE reuses one buffer under `no_grad`, and peak activation is set by the 2x
    CFG batching, not the step count (only the CUDA-graph capture pool grows, marginally).
    Upstream it is **not** a deploy-YAML field, env var, or
    per-request param in vllm-omni: the value is read **only** from the model's `params.json`
    at `multimodal.audio_model_args.acoustic_transformer_args.n_decoding_steps`, and this
    model doesn't ship the key, so vllm-omni's mistral parser (`parsers/voxtral_tts.py`)
    **warns and falls back to 7**. So `entrypoint.sh` does exactly what that warning asks:
    when the var is set it `hf_hub_download`s just `params.json` (works on first boot before
    the ~9GB weights land; uses `HF_TOKEN` if gated), injects `n_decoding_steps`, and
    re-patches idempotently every boot (surviving HF restoring the file). Unset = the
    vllm-omni default **7** (raise toward 8-16 for quality, drop to 3-4 for speed/latency).
    On the fork this is only the **default**: `tts.py --euler-steps N` overrides it per
    call, so a sweep needs no restart. Two constraints, both in `_extract_euler_steps`: it
    is a loop bound rather than a per-row tensor, so it is honoured only when every request
    sharing a decode step agrees (warning logged once otherwise), and a value differing
    from the captured one **forces the eager path**, since the CUDA graphs bake one fixed
    step count into their capture. Sweep one call at a time.
  - `VOXTRAL_MAX_TOKENS` -> stage_id 0's `default_sampling_params.max_tokens`, the
    per-request audio-length cap in 12.5 Hz frames (packaged **2048** = 163.84 s; unset =
    2048). See the length bullet above; `tts.py --max-new-tokens` overrides it per call up
    to 4096. **Compose runs 4096** (= 327.68 s), matching that per-request ceiling.
  - `VOXTRAL_MAX_MODEL_LEN` -> stage_id 0's `max_model_len` (packaged **4096**; unset =
    4096), shared by the prompt's text tokens and the generated frames, so it is the true
    length ceiling. Each token of context costs ~104 KiB of KV cache (2 x 26 layers x 8 kv
    heads x 128 head_dim x 2 bytes bf16), and the KV pool is fixed by
    `VOXTRAL_GPU_MEM_UTIL`, so raising this does not allocate more memory: it trades
    concurrency (how many long requests fit before preemption) for length. Note
    `VLLM_EXTRA_ARGS=--max-model-len ...` is **not** the lever, the per-stage YAML field
    is. **Compose runs 6144**: 4096 frames plus ~2048 tokens of prompt room, because
    leaving it at 4096 alongside `max_tokens=4096` would make the last frames unreachable
    (the entrypoint warns when `max_tokens >= max_model_len`).
  - `VOXTRAL_CODE_DUMP_DIR` -> where the **fork** appends the codes stage 0 emits for any
    request carrying `voxtral_dump_codes` (`./tts.py --dump-codes TAG` writes
    `<dir>/TAG.i16`: raw int16, one row of 37 per 12.5 Hz frame, appended so a cancelled
    request still leaves usable frames). Compose sets `/cache/cloning/dumps`. Unlike the
    knobs above this is **not** read by `entrypoint.sh` and does not appear in its boot
    line: the model file reads it directly, so an unpatched image ignores it silently.
    Nothing is written unless a request asks, and the audio is byte-identical either way.
    It exists because the open checkpoint ships **no audio encoder**, which makes stage 0
    the only thing that can produce real audio/codes pairs: it *is* the text ->
    on-manifold-codes map, and what it emits is the same table-space ids a voice prefix
    carries. That is the input to `voxtral/cloning/train_encoder.py` (plan stage B6),
    whose hard limit was that the 20 shipped voices are only 2823 frames covering 13% of
    the 8192 semantic codes.
  - `HF_TOKEN` (if the HF repo is gated), `VLLM_EXTRA_ARGS` (extra `vllm serve` flags).
    Weights (~9GB) download into the mounted `./cache` via `HF_HOME`.
- **Status:** upgraded to the prebuilt `vllm/vllm-omni:v0.22.0` image and the 0.22 deploy
  schema, then switched to building `vllm_omni` from the local `perso` fork. Statically
  validated only (the YAML patcher runs against the real v0.22.0 deploy config; the fork
  passes ruff and its unit tests are written but need the image to run, since the host has
  no matching vLLM), **not yet run end-to-end in Docker**: the first `--build` plus
  `./voxtral/test_seed_determinism.sh D` and `E` are the real check. The prebuilt image
  ships no standalone flash-attn; the Dockerfile re-adds it from a prebuilt wheel (the
  torch2.10 cu13 one, ABI-compatible with the image's torch2.11 and verified in-image,
  since no torch2.11 wheel exists yet) so stage 1 uses flash-attn instead of the SDPA
  fallback.

## Layout notes

- **`CrispASR_source/`** (gitignored): a vendored checkout of the CrispASR source, kept as
  the **API contract reference** and an optional build context. When unsure about a
  server field or flag, read `CrispASR_source/docs/server.md` (endpoints + every form/body
  field), `docs/tts.md`, `docs/cli.md` instead of guessing.
- **Host→container mounts:** `./models_ignore_backups/` → `/models` (ASR local models);
  `./models_tts_ignore_backups/` → `/models` (TTS baked refs); `./cache/` → `/cache`
  (auto-downloaded models land here). `voices/` is a symlink to the reference-clip source.
- Model/audio binaries are gitignored (`**/*.gguf`, `**/*.mp3`, `**/*.wav`, the `models_*`
  dirs, `cache`, `CrispASR_source`, `voices`, `voxtral/vllm-omni/`), so the tracked repo is
  just compose + the scripts + `Caddyfile` (the `crispasr-lb` ASR load-balancer config) +
  `utils/` + `voxtral/`.
- `voxtral/` holds the Voxtral TTS experiment (serving Voxtral TTS via **vLLM**, not
  CrispASR; see "Voxtral TTS" above): `Dockerfile` + `entrypoint.sh` back the `voxtral-tts`
  compose service, `install.sh` is the original hand-run recipe, and `env.reference` is a
  captured `pip freeze`. The client is the root `tts.py` (the old `voxtral/query_test.py`
  was merged into it). Also here: `test_seed_determinism.sh`, which owns the container
  lifecycle and probes what actually makes the output vary (modes
  `A|B|C|D|E|all|up|down|doctor`; `doctor` checks the per-request-knob chain inside the
  running container and is the first thing to run when a knob looks dead);
  `vllm-omni/`, the **gitignored** upstream clone the image is built from (branch `perso`);
  and `patches/`, which is now that fork's tracked backup (`*.patch` + a README with the
  recreate recipe) rather than a runtime mount. `.dockerignore` keeps the clone's git
  objects and the local `audios/` out of the build context.
