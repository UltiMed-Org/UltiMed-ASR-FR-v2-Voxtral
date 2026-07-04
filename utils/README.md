# utils/

Support scripts for the CrispASR ASR/TTS harness. They are kept out of the repo
root so `tts.py` / `asr.py` stay the obvious entrypoints; reach for these only
when baking a voice or debugging TADA. Each is a self-contained `uv run --script`
(PEP 723 inline deps), so run them directly, e.g. `uv run utils/clone_voice.py`.

| Script | What it does |
|---|---|
| `clone_voice.py` | Bake `./voices/<name>` (its `<name>.txt` transcript) into a `tada-ref-*.gguf` under `models_tts_ignore_backups/` (mounted at `/models`). Owns the tada-ref naming/mount convention; `tts.py --voice-name` shells out to it. Interactive menu, or `--name/--all/--print-path`. |
| `convert-tada-ref-to-gguf.py` | The torch encoder `clone_voice.py` wraps: turns a 24 kHz WAV + transcript into the gguf reference. A pinned copy of CrispASR's `models/convert-tada-ref-to-gguf.py`. |
| `tada_reference_engine.py` | Run HumeAI **TADA's upstream PyTorch engine** directly (bypassing CrispASR) to tell whether silent/garbled TTS is CrispASR's C++ port's fault or the model's. Diagnostic only. |

## Baking a voice (`clone_voice.py`)

Pair each `<name>.<audio>` under `./voices/` with a `<name>.txt` transcript, then:

```bash
uv run utils/clone_voice.py --name alicia          # bake one
uv run utils/clone_voice.py --all --skip-if-baked  # bake all missing
uv run utils/clone_voice.py --print-path --name alicia   # just print /models/*.gguf
```

The baked `tada-ref-<name>.gguf` lands in `models_tts_ignore_backups/`; use it
live with `../tts.py --voice /models/tada-ref-alicia.gguf` (TADA reloads per
request, CrispASR #201) or bake+synth in one shot with `../tts.py --voice-name alicia`.

Requires `uv` + ffmpeg/ffprobe. The first bake pulls torch + the HumeAI encoder
weights on demand (via `uv run --with`), so the menu itself stays instant.

## Debugging silent/garbled TTS (`tada_reference_engine.py`)

Runs the upstream model end to end with the same voice-cloning inputs and
inference knobs the server exposes, so you can A/B:

- Reference sounds fine, CrispASR is silent -> **CrispASR port bug** (file a repro).
- Reference is also silent -> **the model / reference clip / transcript** is at
  fault, not CrispASR.

```bash
./utils/tada_reference_engine.py \
  --ref-audio fr.wav \
  --ref-text "Le texte exact prononce dans fr.wav." \
  --text "Bonjour, ceci est un test de la voix clonee." \
  --language fr \
  --output tada_ref_out.wav
```

Defaults equal upstream `InferenceOptions` / the server defaults (including
`--num-acoustic-candidates 1`). See `./utils/tada_reference_engine.py --help` for
every knob (`--temperature`, `--top-p`, `--repetition-penalty`,
`--do-sample/--no-do-sample`, `--acoustic-cfg-scale`, `--noise-temperature`,
`--scorer`, ...).

Reading the output:

- **Prompt alignment** is printed before synthesis. Bunched/garbled/nonsensical
  tokens mean `--ref-text` does not match `--ref-audio` -> fix that first; a bad
  alignment is the most common upstream cause of bad audio.
- **`engine output text`** is what the model "read"; a mismatch vs `--text` is a
  text-head hallucination (a content bug, distinct from silent audio).
- **VERDICT** reports peak / RMS / dBFS and whether the clip is essentially
  silent. Exit code: `0` audible, `2` silent, `1` decode failed.

Notes: TADA uses the gated Llama-3.2 tokenizer (`huggingface-cli login`, or the
script falls back to the public unsloth mirror). CPU works but a 3B model takes
minutes per clip; pass `--device cuda` if available.

Generated with help from Claude Code.
