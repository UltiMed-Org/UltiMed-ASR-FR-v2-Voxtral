# tada_test

Run HumeAI **TADA's reference PyTorch engine** directly (bypassing CrispASR) to
tell whether silent/garbled TTS output is CrispASR's fault or the model's.

The crispasr-tts container runs CrispASR's C++/ggml re-implementation of TADA.
`generate.py` runs the *upstream* model end to end with the same voice-cloning
inputs and the same inference knobs, so you can A/B:

- Reference sounds fine, CrispASR is silent -> **CrispASR port bug** (file a repro).
- Reference is also silent -> **the model / reference clip / transcript** is at
  fault, not CrispASR.

## Usage

Runs via `uv` (PEP 723 inline deps; first run pulls torch + the TADA model). The
dependency pins match `../clone_voice.py` to avoid the torch/torchaudio ABI crash.

```bash
./generate.py \
  --ref-audio ../fr.wav \
  --ref-text "Le texte exact prononce dans fr.wav." \
  --text "Bonjour, ceci est un test de la voix clonee." \
  --language fr \
  --output out.wav
```

To match the container exactly, add `--num-acoustic-candidates 4` (the server's
`TADA_NUM_CANDIDATES`). All other defaults already equal upstream
`InferenceOptions` / the server defaults. See `./generate.py --help` for every
knob (`--temperature`, `--top-p`, `--repetition-penalty`, `--do-sample/--no-do-sample`,
`--acoustic-cfg-scale`, `--noise-temperature`, `--scorer`, ...).

## Reading the output

- **Prompt alignment** is printed before synthesis. Bunched/garbled/nonsensical
  tokens mean `--ref-text` does not match `--ref-audio` -> fix that first; a bad
  alignment is the most common upstream cause of bad audio.
- **`engine output text`** is what the model "read"; a mismatch vs `--text` is a
  text-head hallucination (a content bug, distinct from silent audio).
- **VERDICT** line reports peak / RMS / dBFS and whether the clip is essentially
  silent. Exit code: `0` audible, `2` silent, `1` decode failed.

Notes: TADA uses the gated Llama-3.2 tokenizer (`huggingface-cli login`, or the
script falls back to the public unsloth mirror). CPU works but a 3B model takes
minutes per clip; pass `--device cuda` if available.

Generated with help from Claude Code.
