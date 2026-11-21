# UltiMed-ASR-FR-v2-Voxtral

> [!WARNING]
> **Release in progress.** UltiMed v2 is being released right now. As long as this notice is here, files, links and numbers in this repository are still being changed.

Part of [UltiMed-Org](https://github.com/UltiMed-Org), which lists every UltiMed repository, dataset and model.

The speech synthesis harness behind the audio of [UltiMed-ASR-FR-v2](https://huggingface.co/datasets/Olicorne/UltiMed-ASR-FR-v2). It is where the five v2 voices were built, blended, auditioned and served. It replaces [UltiMed-ASR-FR-v1-Voxtral](https://github.com/UltiMed-Org/UltiMed-ASR-FR-v1-Voxtral), the single-voice setup used for v1.

It is a working repository, published as is: documentation of what was actually run rather than a packaged tool.

## What is in it

- `docker-compose.yml`: the services. `voxtral-tts` serves `mistralai/Voxtral-4B-TTS-2603` with vLLM 0.22 and a patched vllm-omni. The `crispasr` services run [CrispASR](https://github.com/CrispStrobe/CrispASR) as the ASR judge used for quality control.
- `voxtral/`: the Voxtral image (`Dockerfile`, `entrypoint.sh`, which turns env vars into the deploy YAML), the vllm-omni fork as patches (`voxtral/patches/`, with the recreate recipe in its README), the seed determinism tests, and `voxtral/cloning/`, the voice building and audition scripts.
- `tts.py` and `asr.py`: stdlib-only OpenAI-compatible clients, run with `uv`.
- `utils/`: support tooling.
- `CLAUDE.md`: the detailed notes, including every Voxtral knob, what it really does, and how it was measured (CFG band, Euler steps, per-request seeding, the 163.84 s length cap, why `ref_audio` cloning fails on the open weights).

## Before you run it

- Set your own `CRISPASR_API_KEYS` in `docker-compose.yml` (the placeholder is `CHANGE_ME_set_your_own_key`).
- Recreate the vllm-omni fork from `voxtral/patches/` before the first `docker compose up -d --build voxtral-tts`.
- Paths written as `/media/$USER/4TO_DRIVE/...` are the author's drives: point them at yours.

Written with Claude Code.
