# Voxtral TTS: a serving setup and a grid of derived voices

Voxtral TTS (`mistralai/Voxtral-4B-TTS-2603`) served with vLLM + vllm-omni, plus the 480 extra voices this repo derives from the 20 the checkpoint ships, benchmarked per language so you can pick one on evidence instead of by ear.

Written with Claude Code. Deep documentation lives in `CLAUDE.md` at the repo root; this file is the short version.

TODO: HuggingFace repo id, once the push target is decided.

## Serving

```bash
sudo docker compose up -d --build voxtral-tts     # binds host :8003
./tts.py -t "Bonjour." --voice fr_female -o out.wav
```

The image builds on `vllm/vllm-omni:v0.22.0-x86_64` and reinstalls `vllm_omni` from a local fork (`voxtral/patches/`, recreate recipe in its README). The fork is load-bearing: upstream silently drops unknown request fields and answers 200, so a client asking for a different guidance scale or noise seed gets the server's defaults while appearing to work. The fork routes them, which is what makes per-request `--cfg-alpha`, `--euler-steps` and `--noise-seed` real.

## Voices

A Voxtral voice is an `[N, 3072]` bfloat16 prefix: the sum of 37 codebook rows per frame at 12.5 Hz, 1 semantic code and 36 acoustic ones. Identity lives **entirely in the 36 acoustic codes**, measured rather than assumed (every derived voice sits at 0.84 to 0.98 ECAPA cosine from its acoustic donor, while the 20 shipped voices sit at most 0.58 from each other, with nothing in between).

That splits a voice in two and makes new ones free, with no encoder and no training:

- `h_<semantic>_<acoustic>` takes the semantic code column from one voice and the 36 acoustic levels from another. It **sounds like** the acoustic donor and **reads like** the semantic one. 380 of these, the full ordered grid over the 20 shipped voices.
- `x_<a>_<b>_...` blends whole prefixes with one dominant parent (3-, 4- and 5-way, 100 of them). These are the only construction that can land on a speaker the checkpoint does not ship, and the only one that can produce mush.

```bash
CLONING_POOL=GRID uv run voxtral/cloning/derive_voices.py pool   # mint all 480
sudo docker compose up -d voxtral-tts                            # model init registers them
./tts.py -t "Ciao." --voice h_frm_itm -o out.wav
```

**Read this before counting voices.** 380 hybrids are not 380 speakers. They are 20 speakers with 20 ways of reading each, so the grid buys per-language delivery quality, not acoustic diversity. If you want more distinct speakers, the sparse mixes are the only lever here, and the benchmark says how well they work.

## The benchmark

Every voice reads 30 FLEURS sentences in each of 8 languages (fr, de, es, it, nl, pt, ar, hi), transcribed with whisper-large-v3-turbo. Roughly 120,000 takes and 89 hours of GPU, resumable at any point: stop it, take the card back, start the same command later.

```bash
sudo -E ./voxtral/cloning/grid.sh          # run it, under tmux
./voxtral/cloning/grid.sh --list           # where each language got to
./voxtral/cloning/grid_report.sh           # reports, safe to run while it is still going
```

Three numbers per voice per language, because they answer different questions:

| metric | what it means | why it is separate |
|---|---|---|
| median WER / CER | what a typical sentence costs | a good median can hide a bad tail |
| junk rate | share of takes past 50 pct WER | what a generator has to catch and redo |
| runaway rate | share that degenerate into looping syllables until the token ceiling | measured from the waveform, so the ASR has no say in it |

All three are **first-attempt** rates. Retrying and reporting the survivor is best-of-N, which would score a voice that degenerates on a tenth of its takes the same as one that never does.

Outputs land in `cache/cloning/GRID_<lang>/` (per language: error tables, failure modes, an N x N speaker-similarity matrix, the speaker roster) and `cache/cloning/GRID/` (`cross.md` and `voices.csv`, the cross-language tables). The one to read first is the table in `cross.md` giving, for each shipped speaker and each language, the name that reads it best.

## One client-side gotcha

`tts.py` replaces non-breaking and thin spaces with ordinary ones before sending text, by default. Voxtral degenerates on them: it starts the sentence correctly, then loops syllables until the token cap, burning minutes of GPU and returning an unusable file with a 200. Measured on the sentences that carry one, median WER 47.7 -> 10.5 pct and unusable takes 16 of 48 -> 3 of 48. Pass `--raw-spaces` when the exact bytes matter.
