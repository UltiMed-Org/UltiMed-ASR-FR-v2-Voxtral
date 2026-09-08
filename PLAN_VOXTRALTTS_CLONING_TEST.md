# Voxtral TTS: adding voices (plan and running log)

**STATUS:** A1 host side done and accepted, container half pending (you must run it: sudo has no TTY for the agent)

**NEXT ACTION:** run the two commands in A1's Result to confirm `zz_test` works on the live server, then Stage A2

**Agent:** written with Claude Code. Any agent picking this up: read "How to use this file" first, do the next unchecked step only, then update STATUS / NEXT ACTION / the session log and commit.

## Goal

More **acoustic diversity** when generating an audio dataset with the `voxtral-tts` service. Not identity cloning of a specific person, not a product feature.

Consequences of that framing, which relax the problem a lot:

- Any new voice that sounds natural, distinct and stable is a win. It does not have to match the source clip closely.
- The LLM side stays untouched. `fr_female` output quality is already the bar, and nothing here changes generation, sampling, cfg or euler steps.
- A voice that sounds like "a French man in his fifties, slightly hoarse" is useful even if it sounds nothing like the person in the reference clip.
- Failure is cheap. A bad voice is deleted, no server change, no rebuild.

Non-goals: natural-language voice description (ruled out, the model has no text conditioning path for style), changing the acoustic decoder, retraining anything large, publishing weights.

## How to use this file

Sessions are days apart, so this file is the only memory. Rules:

1. Start by reading the STATUS and NEXT ACTION lines at the top, then the last row of the session log at the bottom.
2. Do **one stage** per session. Stages are sized to fit one sitting.
3. Every stage has an **acceptance test**. If it fails, write what happened in the Result field and stop. Do not improvise the next stage on top of a failed one.
4. End every session by updating: the stage checkbox, the stage Result field, the STATUS line, the NEXT ACTION line, and a new session log row. Then commit (one commit per session, message `plan: <stage> <outcome>`).
5. Keep code in `voxtral/cloning/`, artifacts in `cache/cloning/` (already gitignored, so `.pt` and intermediate files stay out of git).
6. Scripts follow the repo convention: standalone `uv` scripts with PEP 723 inline deps, no venv, no shared package.

Checkbox legend: `[ ]` todo, `[x]` done and accepted, `[!]` attempted and failed (see Result), `[-]` skipped deliberately.

## Environment notes

- GPU is a 3090 Ti, 24 GB. The `voxtral-tts` service takes `VOXTRAL_GPU_MEM_UTIL=0.7` of it, so **stop the service before any local torch work**: `sudo docker compose stop voxtral-tts`. Both at once will OOM.
- Ampere (sm_86) does bf16 natively. The decoder is small (roughly 100 M params), so all of Track B fits in a couple of GB. No flash-attn needed for offline work.
- Model files are already downloaded at `cache/huggingface/hub/models--mistralai--Voxtral-4B-TTS-2603/snapshots/<sha>/`. Referred to below as `$SNAP`.
- Source clips: `voxtral/audios/` currently holds test output, not references. **TODO:** decide which clips are the reference material (`voices/` is the TADA reference source and is probably the right pool). Pick 3 to 5 clean, single-speaker, 8 to 20 s clips before Stage B4.
- Reminder from `CLAUDE.md`: HF snapshot files are symlinks into `blobs/`, which are content-addressed and can be read-only. `voxtral/entrypoint.sh:246` already patches `params.json` in place via `os.path.realpath()` plus `os.chmod(0o644)`, and re-applies it idempotently at every boot. Follow that same pattern, and expect an HF refresh to undo edits.

## Background (so no session has to re-derive it)

A "voice" in Voxtral is **not** a speaker vector. It is a pre-encoded reference clip, 5 to 17 s of real audio, frozen as a prefix in the LLM's embedding space.

- `$SNAP/voice_embedding/<name>.pt` is a bfloat16 `[N, 3072]` tensor. N is frames at 12.5 Hz, so `fr_male` is 97 frames (7.76 s), `neutral_female` is 218 (17.44 s).
- `$SNAP/tekken.json` -> `audio.voice_num_audio_tokens` maps name to N. mistral_common emits exactly N placeholder tokens (id 24) for that voice.
- `$SNAP/params.json` -> `multimodal.audio_tokenizer_args.voice` is the name list that becomes `supported_speakers` server side.
- At request time the server sends only the **name**. `voxtral_tts.py:209-218` (`tts_preprocess`) looks up the tensor and scatters it into the id-24 positions.
- Each frame of that tensor is the **sum of 37 codebook lookups** in a shared `[9088, 3072]` table (`mm_audio_embeddings.audio_codebook_embeddings`): 1 semantic code (VQ, 8192 entries, 256 latent dims) plus 36 acoustic codes (FSQ, 21 levels, 36 latent dims). See `voxtral_tts_audio_tokenizer.py:968` (`encode_tokens`) and `:349` (`MistralAudioCodebook.encode`).
- The **encoder is missing** from the open weights (`voxtral_tts_audio_tokenizer.py:983` raises on `encode_waveforms`). The **decoder ships in full** (`audio_tokenizer.decoder_blocks.*`, 112 tensors, plus `output_proj`). That asymmetry is the whole reason for Track B.

## Track A: free diversity from the 20 shipped voices

No training, no decoder, no GPU beyond the running server. This may already satisfy the goal, so it runs first.

### [x] A1. Inventory and registration mechanics

Write `voxtral/cloning/voices_tool.py` with three subcommands:

- `list`: read `$SNAP/voice_embedding/*.pt`, print name, N, duration, dtype, plus the matching `tekken.json` entry, and flag any mismatch.
- `add --name X --from <file.pt>`: write the tensor, add `X` to `params.json`'s voice dict and to `tekken.json`'s `voice_num_audio_tokens`, both idempotent, both handling the symlink-into-blobs case (resolve realpath, chmod 0644, never write through a symlink you did not resolve).
- `remove --name X`: the inverse.

Acceptance: `list` prints 20 voices with N matching `tekken.json` exactly. Then `add` a copy of `fr_female` under the name `zz_test`, restart the service, and `./tts.py -t "Bonjour, ceci est un test." --voice zz_test -o /tmp/zz.wav` produces audio identical in character to `fr_female`.

Estimated: 1 to 2 h. Notes: the server reads voices once at model init, so every registration needs `sudo docker compose up -d voxtral-tts` (not `restart`). If `hf_hub_download` clobbers edits, set `HF_HUB_OFFLINE=1` in the service environment. **TODO:** if that is not enough, the cleaner fix is a local model dir via `VOXTRAL_TTS_MODEL` (`voxtral/entrypoint.sh:32`), but then the euler-steps patch at `:246` needs to handle a local path instead of a repo id.

**Result:** 2026-09-09, Claude Code. `voxtral/cloning/voices_tool.py` written (click + loguru, PEP 723, no torch: it only copies a file and patches two JSON dicts). Subcommands `paths`, `list`, `add`, `remove`, `restore`.

Host-side acceptance PASSED:

- `list` shows all 20 shipped voices with `.pt` frame counts matching `tekken.json` exactly, 0 problems. Frame counts are derived from the tensor's byte size rather than trusted, so drift cannot go unnoticed.
- `add --name zz_test --from-voice fr_female` registered a 97-frame copy, byte-identical to the source.
- `add` then `remove` of a throwaway voice round-tripped cleanly, leaving 21 voices.
- `remove --name fr_male` correctly refuses shipped voices (their `.pt` is still a `blobs/` symlink).
- `params.json` and `tekken.json` stayed valid, kept every other key, and `n_decoding_steps` is still 16, so entrypoint.sh's own patch survives this tool.
- Pristine copies of both configs saved to `cache/cloning/backup/`, restorable with `voices_tool.py restore`.

Container-side acceptance NOT run: `sudo` needs a password and the agent has no TTY. `zz_test` is left registered so you can finish it:

```
sudo docker compose up -d voxtral-tts     # NOT restart: voices load at model init
./tts.py -t "Bonjour, ceci est un test." --voice zz_test -o /tmp/zz.wav
```

Expect audio indistinguishable in character from `--voice fr_female`. If the server answers "Invalid speaker 'zz_test'", the snapshot edits did not reach the container (see the `HF_HUB_OFFLINE` / local-model-dir TODOs). Once confirmed, `./voxtral/cloning/voices_tool.py remove --name zz_test`.

Note for later sessions: the agent cannot run `sudo docker`, so every stage that needs the server has to end with commands for you to run.

### [ ] A2. Derived voices: truncation, cross-lingual, interpolation

Cheap manipulations of existing prefixes, each producing a new registered voice:

- **Truncate**: first k frames of a voice (k = 40, 60, 97). Shorter prefix, different prosody anchor.
- **Cross-lingual**: use `es_male`, `it_female`, `pt_male`, `nl_male` with French text. Accented French is legitimate diversity for a dataset, and costs nothing but a request.
- **Interpolate**: `alpha * A + (1 - alpha) * B` at alpha in {0.25, 0.5, 0.75}. Lengths differ, so truncate both to `min(N_A, N_B)` first.
- **Concatenate**: first half of A followed by second half of B.

For each, synthesize the same French sentence and keep a labelled wav. Build a contact sheet directory `cache/cloning/A2/` with one wav per variant plus a `notes.md` of subjective judgements (natural / robotic / unstable / distinct-from-parents).

Acceptance: at least **4 new voices** that are clearly distinct from the 20 originals and sound as natural as `fr_female`. If interpolation gives mush, truncation and cross-lingual alone probably still clear the bar.

Estimated: 2 to 3 h, mostly listening.

**Result:**

### [ ] A3. Decision point

If A2 produced enough usable voices for the dataset, **stop here** and mark Track B as skipped. Write down how many voices exist and how diverse they actually are. Track B is only worth it if A2 falls short or if the diversity is too obviously "the same 20 speakers, blended".

**Result:**

## Track B: new voices from real clips, by inverting the decoder

Only start after A3 says Track A is insufficient. Each stage is a gate for the next.

### [ ] B1. Recover the true codes of the shipped voices (CPU, no decoder)

Each `[N, 3072]` embedding is a sum of 37 rows from the known `[9088, 3072]` table, one row from each codebook's known offset range (offsets are `cumsum` of codebook sizes: 8194 for semantic, then 36 x 23 for acoustic, per `MultiVocabEmbeddings`). Recover the indices by coordinate descent: initialize by picking each codebook's nearest row to the residual, then repeatedly hold 36 fixed and re-solve the 37th exactly by nearest-row lookup on the residual.

Write `voxtral/cloning/recover_codes.py`, dump `cache/cloning/codes/<voice>.npy` of shape `[N, 37]`.

Acceptance: for all 20 voices, the reconstruction residual `||sum(rows) - embedding||` is at or near bf16 rounding noise, semantic codes land in `[0, 8193]`, acoustic codes in `[0, 22]`, and the recovered codes re-embed to the original tensor within tolerance.

Bonus output: the marginal statistics of real code streams (semantic histogram, transition structure, acoustic level distribution). Save them, they become the prior for B3 and B4.

Estimated: 2 to 3 h. Pure numpy or torch on CPU, seconds of compute.

**Result:**

### [ ] B2. Load the decoder standalone and hear the reference clips

Instantiate `VoxtralTTSAudioTokenizer` in plain torch, load only `audio_tokenizer.decoder_blocks.*` and `output_proj` from `consolidated.safetensors`, feed it the B1 codes, write wavs. The class only reads `vllm_config.model_config.hf_config`, so a small namespace shim is enough. Watch for weight-norm parameterisation, the causal conv padding, and bf16 autocast.

Write `voxtral/cloning/decoder.py` (loader plus `decode(codes) -> waveform`) and `voxtral/cloning/decode_voices.py`.

Acceptance: `cache/cloning/B2/fr_female.wav` is intelligible speech that plainly matches the voice the server produces under `--voice fr_female`. This is the single most informative artifact in the whole plan: it is Mistral's actual reference recording, reconstructed.

Estimated: 3 to 5 h. This is the most engineering-heavy stage. If it stalls on plumbing, that is normal, log where and stop.

**Result:**

### [ ] B3. The paired test: does analysis by synthesis land on real codes

The decisive experiment for the "off-distribution prefix" risk. B1 and B2 together give a paired example: a waveform **and** its ground-truth codes.

Take the B2 waveform for one voice, forget its codes, and re-derive them by optimizing through the frozen decoder: continuous latents `[N, 292]` as the only parameters, straight-through rounding for the 36 FSQ dims, softmax-over-negative-distances with annealed temperature for the semantic dim, loss = multi-resolution STFT plus mel against the target. Because the decoder is causal with a small sliding window, prefer a frame-by-frame sequential search (CELP style) over a joint optimization if the joint version gets stuck.

Write `voxtral/cloning/invert.py`.

Acceptance, in order of importance:

1. Reconstruction is perceptually close to the target.
2. Recovered acoustic codes are mostly within one or two levels of the truth.
3. Recovered semantic codes match the truth on a **large fraction of frames** (report the number, do not eyeball it).

If 1 and 2 pass but 3 fails badly, the off-distribution risk is real: continue only with the mitigation of adding a prior term from B1's statistics, and re-test once.

Estimated: 4 to 6 h across possibly two sessions. Split into "get it converging" and "measure against truth" if needed.

**Result:**

### [ ] B4. A real clip becomes a voice

Pick one clean French reference clip (8 to 20 s, single speaker, no music, no clipping, resampled to 24 kHz mono). Run `invert.py` on it, embed the codes with the B1 machinery, register with `voices_tool.py add`, restart, synthesize.

Acceptance: the new voice produces natural, stable French speech at roughly `fr_female` quality. Speaker resemblance is a **bonus, not a requirement**, per the goal.

Estimated: 2 to 3 h.

**Result:**

### [ ] B5. Scale to a voice set and generate

Repeat B4 for the remaining clips. Keep a table of voice name, source clip, duration, subjective quality, and whether it is dataset-approved. Then run the actual dataset generation with the approved set.

Acceptance: N usable voices with enough acoustic spread for the dataset, and a note on which ones are unstable so they can be excluded.

Estimated: 1 h per voice once the pipeline works.

**Result:**

## Kill criteria and fallbacks

Stop Track B and fall back if:

- **B2 fails** (decoder cannot be loaded or produces noise): the entire inversion route is dead. Fall back to Track A plus a wider search for a different TTS engine for voice variety.
- **B3 fails criterion 3 even with the prior term**: inversion produces codes the LLM has probably never seen. Do not spend time on B4. The remaining option is the donor-encoder route (Qwen3-TTS ships a Mimi encoder at exactly 24 kHz and 12.5 Hz in `vllm_omni/model_executor/models/qwen3_tts/tokenizer_12hz/`, frame-aligned with Voxtral, so an adapter has no resampling to learn), which is a much bigger project and needs its own plan.
- **B4 gives audio that is natural but always the same voice regardless of the clip**: the prefix is being ignored or washed out. Log it and stop, it means acoustic identity is not carried the way this plan assumes.

Prior estimate before starting, for later calibration: roughly one in three that Track B yields good quality end to end, with almost all of the uncertainty concentrated in B3. Track A is high confidence but limited ceiling.

## Open questions and TODOs

- **TODO:** choose the reference clip pool (`voices/` vs new recordings) before B4.
- **TODO:** confirm whether editing the HF snapshot is stable across restarts, or whether `HF_HUB_OFFLINE=1` or a local model dir via `VOXTRAL_TTS_MODEL` is needed. If the local dir wins, `voxtral/entrypoint.sh:246` needs to stop assuming a repo id.
- **TODO:** decide whether new voices live in the HF cache (fragile, re-applied at boot) or in a materialised local model dir (clean, needs the entrypoint change).
- Open: is there an upper bound on N for a registered voice, given the prefix shares stage 0's `max_model_len` (compose runs 6144) with the generated frames? A 20 s prefix is 250 tokens, so this is not tight, but confirm before using very long references.
- Open: does `VOXTRAL_CFG_ALPHA` interact with prefix adherence? If a new voice drifts, try nudging cfg before blaming the prefix.

## Session log

Newest at the bottom. One row per working session.

| date | stage | what happened | next |
|---|---|---|---|
| 2026-09-08 | plan | Plan written (Claude Code). Established that voices are pre-encoded audio prefixes, the encoder is missing from the open weights, the decoder is not. | A1 |
| 2026-09-09 | A1 | voices_tool.py written and verified host side: 20 voices consistent, add/remove/restore round trip clean, configs intact. Container check left to the user (no sudo TTY for the agent). | finish A1 on the server, then A2 |
