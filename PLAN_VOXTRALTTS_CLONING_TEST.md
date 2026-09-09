# Voxtral TTS: adding voices (plan and running log)

**STATUS:** A2 accepted (6 keepers). A3 found that Voxtral clones from a raw clip natively via `ref_audio`, which is a different mechanism from the missing encoder. Needs one rebuild (soxr) to test.

**NEXT ACTION:** run `./voxtral/cloning/step.sh a3`: rebuilds with soxr, then clones `olivier 2.ogg`, `olivier.ogg`, `alicia2.ogg`, `alicia.ogg`. Listen and compare against the source clips.

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
7. **The agent cannot run `sudo docker`** (no password, no TTY), so anything server-side goes into `voxtral/cloning/step.sh`. The agent updates `stage_<x>()` and the `CURRENT_STAGE` variable, the user runs `./voxtral/cloning/step.sh`, and the run is teed to `cache/cloning/logs/<stage>-<timestamp>.log`. The agent reads that log next session instead of asking for pasted output.

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

Estimated: 1 to 2 h. Notes: the server reads voices once at model init, so every registration needs `sudo docker compose up -d voxtral-tts` (not `restart`). (Superseded, kept for the record: this line originally guessed `HF_HUB_OFFLINE=1` as the fix for `hf_hub_download` clobbering the edits. It is not, see the Result addendum below.)

**Result:** 2026-09-09, Claude Code. `voxtral/cloning/voices_tool.py` written (click + loguru, PEP 723, no torch: it only copies a file and patches two JSON dicts). Subcommands `paths`, `list`, `add`, `remove`, `restore`.

Host-side acceptance PASSED:

- `list` shows all 20 shipped voices with `.pt` frame counts matching `tekken.json` exactly, 0 problems. Frame counts are derived from the tensor's byte size rather than trusted, so drift cannot go unnoticed.
- `add --name zz_test --from-voice fr_female` registered a 97-frame copy, byte-identical to the source.
- `add` then `remove` of a throwaway voice round-tripped cleanly, leaving 21 voices.
- `remove --name fr_male` correctly refuses shipped voices (their `.pt` is still a `blobs/` symlink).
- `params.json` and `tekken.json` stayed valid, kept every other key, and `n_decoding_steps` is still 16, so entrypoint.sh's own patch survives this tool.
- Pristine copies of both configs saved to `cache/cloning/backup/`, restorable with `voices_tool.py restore`.

Container-side acceptance NOT run yet: `sudo` needs a password and the agent has no TTY. `zz_test` is left registered. Run:

```
./voxtral/cloning/step.sh
```

It handles the :8003 clash with `crispasr-tts`, brings up `voxtral-tts` with `up -d` (not `restart`), waits for `tts.py --check` to pass, prints the container's own "Available voice embeddings" line, then synthesizes the same sentence twice with `--noise-seed 7`, once with `fr_female` and once with `zz_test`.

Acceptance is sharper than "sounds similar": with the seed pinned the fork's per-frame noise is reproducible and the rest of the pipeline is deterministic, so the two wavs should be **byte identical**. `cmp` decides it. Output lands in `cache/cloning/out/`, the whole run in `cache/cloning/logs/a1-*.log`, which the agent reads next session. Then clean up with `./voxtral/cloning/voices_tool.py remove --name zz_test`.

**Result addendum, same day, two failed boots before the working fix:**

Registering a local voice **bricked the server** rather than being politely ignored. Root cause at `voxtral_tts.py:134`: stock vllm_omni loads voice embeddings with `hf_hub_download(repo_id, f"voice_embedding/{sid}.pt")` for **every** name in `params.json`. Since the model is named by repo id, that goes to the REMOTE repo, so `zz_test.pt` (which exists only on disk) 404s, `StageEngineCoreProc` exits 1, and `restart: unless-stopped` turns it into a silent boot loop that looks exactly like a slow first boot. That cost the first 30-minute run.

**First fix was wrong.** `HF_HUB_OFFLINE=1` in the service environment makes those lookups resolve from `/cache`, but it also breaks vLLM's own config resolution for a repo-id model: `Invalid repository ID or local directory specified: 'mistralai/Voxtral-4B-TTS-2603'`, then `Could not determine model_type` and `Could not detect config format for no config file found`. Note the asymmetry that made this misleading: the entrypoint's own `hf_hub_download(model, "params.json")` works fine offline, so the flag looks harmless right up until vLLM starts. Reverted, with a comment in `docker-compose.yml` so nobody re-adds it.

**Actual fix: fork patch 0005**, `_resolve_voice_embedding()` in the `perso` branch. It tries `local_files_only=True` first, falls back to a real download, and returns `None` (caller skips the voice, logging a warning) instead of raising. So locally-added voices work, shipped voices still download on a cold cache, and a typo degrades to one missing voice rather than a dead server. This reaches the container **only through a rebuild**, since the Dockerfile installs `vllm_omni` from the local clone, hence `step.sh` now uses `up -d --build`.

Guards added so this cannot silently repeat: `step.sh` refuses to start when the fork source lacks `_resolve_voice_embedding`, and `voices_tool.py` warns on `add` for the same reason.

**Container-side acceptance PASSED**, 2026-09-09, Claude Code, third run:

- The container's own log lists `zz_test` among `Available voice embeddings`, so the snapshot edit reached model init through patch 0005.
- `fr_female` and `zz_test` both synthesized "Bonjour, ceci est un test de voix." at `--noise-seed 7`: 115244 bytes each, **byte identical** under `cmp`. The registration path is faithful, and per-request noise seeding is confirmed reproducible across voices as a bonus.
- Boot to healthy took **1147 s** (~19 min) including the image rebuild, so the 1800 s budget is right and 900 s was never going to work.

One client bug found and fixed on the way: `tts.py --check` sent no `voice`, which TADA tolerates (it uses whatever reference it was started with) but vllm-omni refuses with a 400 on every request. So `--check` reported a perfectly healthy Voxtral server as down, and `step.sh` sat in its wait loop watching 400s go by for 1147 s. `cmd_check` now passes `--voice` when given and, on that specific 400, retries once with `VOXTRAL_CHECK_VOICE`. Also fixed a false "no Available voice embeddings line" warning caused by `pipefail` plus `grep -m1` (SIGPIPE on the upstream `docker compose logs`).

`zz_test` removed after the test.
### [ ] A2. Derived voices: truncation, cross-lingual, interpolation

Cheap manipulations of existing prefixes, each producing a new registered voice:

- **Truncate**: first k frames of a voice (k = 40, 60, 97). Shorter prefix, different prosody anchor.
- **Cross-lingual**: use `es_male`, `it_female`, `pt_male`, `nl_male` with French text. Accented French is legitimate diversity for a dataset, and costs nothing but a request.
- **Interpolate**: `alpha * A + (1 - alpha) * B` at alpha in {0.25, 0.5, 0.75}. Lengths differ, so truncate both to `min(N_A, N_B)` first.
- **Concatenate**: first half of A followed by second half of B.

For each, synthesize the same French sentence and keep a labelled wav. Build a contact sheet directory `cache/cloning/A2/` with one wav per variant plus a `notes.md` of subjective judgements (natural / robotic / unstable / distinct-from-parents).

Acceptance: at least **4 new voices** that are clearly distinct from the 20 originals and sound as natural as `fr_female`. If interpolation gives mush, truncation and cross-lingual alone probably still clear the bar.

Estimated: 2 to 3 h, mostly listening.

**Result:** 2026-09-09, Claude Code, host side. `voxtral/cloning/derive_voices.py` written (click + loguru + CPU-only torch via a pinned PEP 723 index, so ~200 MB instead of ~2.5 GB of CUDA and no contention with the running server). It does tensor surgery only and shells out to `voices_tool.py add` for registration, so the three-files-must-agree logic is not duplicated. Ops: `head`, `tail`, `mix` (alpha blend, both cut to the shorter), `cat`. Math runs in float32 and is cast back to bfloat16 on save, so a derived file is format-identical to a shipped one.

8 variants built and registered, all `d_*` so `clean` can remove exactly what was created. `voices_tool.py list` reports 28 voices, 0 problems:

| voice | recipe | frames |
|---|---|---|
| d_frf_head40 | fr_female[:40] | 40 |
| d_frf_head60 | fr_female[:60] | 60 |
| d_frf_tail60 | fr_female[-60:] | 60 |
| d_frf_itf_50 | 0.5 fr_female + 0.5 it_female | 97 |
| d_frf_esf_50 | 0.5 fr_female + 0.5 es_female | 97 |
| d_frf_frm_50 | 0.5 fr_female + 0.5 fr_male | 97 |
| d_frf_frm_25 | 0.75 fr_female + 0.25 fr_male | 97 |
| d_frf_itf_cat | fr_female[:48] + it_female[48:] | 172 |

Cross-lingual needs no registration at all (shipped voices, French text), so `stage_a2()` just adds `es_male`, `it_female`, `pt_male`, `nl_male` to the sweep. 13 takes total, same text and same `--noise-seed 7` throughout so every difference is attributable to the prefix, plus `fr_female` as the anchor the acceptance bar is set against. Container side pending: one boot, then listening.

**Sweep done**, 2026-09-09, Claude Code, `cache/cloning/logs/a2-20260909-212024.log`. All 13 takes returned 200, and `cmp` confirms all 13 are pairwise distinct, so no variant silently collapsed onto a parent. `wav_stats.py` (added this session, stdlib-only triage: duration, autocorrelation F0, rms) says:

| voice | sec | F0 Hz | rms |
|---|---|---|---|
| fr_female (anchor) | 7.04 | 188 | 1321 |
| d_frf_esf_50 | 6.40 | 207 | 412 |
| d_frf_frm_25 | 6.96 | 188 | 890 |
| d_frf_frm_50 | 8.24 | 150 | 946 |
| d_frf_head40 | 6.32 | 207 | 1623 |
| d_frf_head60 | 7.28 | 188 | 1546 |
| d_frf_itf_50 | 8.16 | 214 | 559 |
| d_frf_itf_cat | 8.72 | 194 | 1177 |
| d_frf_tail60 | 7.68 | 190 | 1111 |
| es_male | 9.76 | 118 | 1540 |
| it_female | 8.72 | 194 | 1240 |
| nl_male | 9.68 | 182 | 1601 |
| pt_male | 8.88 | 128 | 1777 |

Reading, to be confirmed by ear:

- **Blending moves the voice, it does not just average it.** `d_frf_frm_50` lands at 150 Hz, between its 188 Hz and 150-ish parents, a register no shipped voice occupies. If it sounds natural, interpolation is a real axis and A3 should expand it (more pairs, more alphas), not drop it.
- **The two cross-language blends look like the predicted mush**: rms 412 and 559 against the anchor's 1321, i.e. a third of the level, which is what a decoder hedging on an off-manifold prefix produces. Same-language `frm` blends sit at 890 to 946, much closer.
- **Truncation is safe**: all three at or above the anchor's level. `head40` also shifts pitch up to 207 Hz, so shortening is not neutral, it is its own axis.
- **Cross-lingual is the cheapest diversity of all**: 118 and 128 Hz, 25 to 40 percent slower, zero registration, zero risk.

Note the seed makes takes comparable but not neutral: all 13 share `--noise-seed 7`, so a single unlucky draw affects them all equally. Worth a second sweep at another seed before rejecting a variant on one bad take.
### [x] A3. Decision point

If A2 produced enough usable voices for the dataset, **stop here** and mark Track B as skipped. Write down how many voices exist and how diverse they actually are. Track B is only worth it if A2 falls short or if the diversity is too obviously "the same 20 speakers, blended".

**Result:** 2026-09-09, Claude Code. Two separate findings.

**1. A2 cleared its bar.** The user's listening table (`cache/cloning/A2/notes.md`) marks 6 keepers: `d_frf_head40`, `d_frf_head60`, `d_frf_tail60`, `d_frf_itf_50`, `d_frf_itf_cat`, plus `fr_female` itself. Judgements worth carrying forward:

- Truncation is reliable and mild. `tail60` was called indistinguishable from the parent, the two `head` cuts only slightly different, so the axis works but its range is narrow.
- `d_frf_itf_50` and `d_frf_itf_cat` were both heard as genuinely different voices, the splice with "a bit of an accent". Cross-language blending is therefore NOT uniformly mush.
- The failure mode is a **seam**, not a smear: `d_frf_esf_50` switches voice mid-utterance ("before 'le vieux chêne' it uses a different voice than after") and `d_frf_frm_50` garbles "bonjour" before settling into a good male voice. Both are the frame-by-frame average being coherent in places and not others, which matches the low rms flagged in triage. So a blend is worth keeping only if it holds ONE identity across the whole utterance.
- `d_frf_frm_25` was indistinguishable from the parent: at 25 percent the partner barely registers.
- Cross-lingual works but costs intelligibility: `nl_male` was rejected for a strong accent, `it_female` and `pt_male` kept with reservations, and `es_male` "fails during 75 and after", i.e. on the number, which is where a non-French speaker prefix hurts most.

**2. The bigger finding: Voxtral already clones from a raw clip, per request.** Reading `serving_speech.py::_build_voxtral_prompt` (and `_validate_voxtral_tts_request`, and the upstream `examples/.../voxtral_tts/end2end.py`), a request may carry `ref_audio` instead of `voice`. mistral_common's `encode_speech_request` decodes the clip and vllm-omni passes the waveform to stage 0 as `multi_modal_data: {"audio": [...]}`. The stage-0 model is `SupportsMultiModal` with an `audio_arrays` field and a real `VoxtralTTSMultiModalProcessor`, so this is a first-class shipped feature, not a stub.

**This does not contradict the "the encoder is missing" background above, it sidesteps it.** The absent codec encoder is what would be needed to bake a new *preset* voice (a `[N, 3072]` pre-encoded prefix on disk). Cloning by `ref_audio` uses the AR LLM's own multimodal audio input instead, a completely different tower, and that one ships with the checkpoint.

Verified live against the running server on the first attempt, with `tts.py --ref-audio` added this session: the request reached the model and failed on a missing dependency, not on the mechanism. `400: "soxr is not installed"`, mistral_common's resampler, absent from the base image because preset voices never resample. One `pip install soxr` line in `voxtral/Dockerfile` and a rebuild.

**Decision: Track B is on hold, not started.** Decoder inversion was designed to work around exactly the problem `ref_audio` does not have. It becomes worth doing only if the clones sound bad, or if a *preset* (a persistent, registerable voice with no per-request clip) turns out to be needed. Track A's 6 derived voices stay useful regardless: they are free diversity that costs no reference audio at all.

Stage A4 below now carries the actual goal: clone `olivier 2.ogg` and `alicia2.ogg`.

### [ ] A4. Clone the target voices (the project's actual acceptance test)

The user's own success criterion: `voices/olivier 2.ogg` and `voices/alicia2.ogg` must come out sounding like those people. Longer takes of the same two speakers exist (`voices/olivier.ogg`, `voices/alicia.ogg`), so the sweep does both lengths per speaker to see whether more reference audio clones better.

`step.sh a3` runs it: rebuild (soxr), boot, one anchor take with `fr_female` to prove the server is fine independently of cloning, then four clones at one fixed text and `--noise-seed 7`. Output in `cache/cloning/A3/wav/`.

Acceptance: both short clips produce a recognisable imitation of their speaker. Failure modes to watch for and what each would mean:

- Right speaker, wrong prosody or a wobbly identity: usable, tune with a longer or cleaner reference clip.
- Generic voice unrelated to the clip: the audio reached the model but carried no identity, so check the clip's sample rate and length against what `encode_speech_request` produced.
- Another `soxr` 400: the image did not actually rebuild (`up -d --build --force-recreate` is what `start_voxtral` does, check the log).

Open question once it works: cloning is per request and stateless, so a dataset run pays the reference clip on **every** call. Measure whether that costs meaningful latency, and if it does, revisit whether a preset is worth Track B after all.

**Result:**

## Track B: new voices from real clips, by inverting the decoder

**ON HOLD since 2026-09-09, see A3.** `ref_audio` clones without the missing encoder, so this whole track is a fallback now rather than the plan. Start it only if A4's clones are unusable, or if a persistent preset voice turns out to be required.

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
| 2026-09-09 | A1 | voices_tool.py written and verified host side: 20 voices consistent, add/remove/restore round trip clean, configs intact. Added step.sh so all server-side work is one command the user runs, logged for the agent to read back. | user runs step.sh, then A2 |
| 2026-09-09 | A1 | Two failed container boots. A locally-added voice bricks stock vllm_omni (remote hf_hub_download per voice, 404, boot loop); HF_HUB_OFFLINE=1 fixed that but broke vLLM's repo-id config resolution, so it was reverted. Real fix is fork patch 0005 (local-first voice lookup, None instead of raise). step.sh now builds and preflights the fork. | re-run step.sh a1 |
| 2026-09-09 | A1 | **A1 accepted.** zz_test loaded by the container and produced audio byte identical to fr_female at seed 7. Fixed tts.py --check (sent no voice, so it read a healthy Voxtral server as down) and a pipefail false warning in step.sh. zz_test removed. | A2 |
| 2026-09-09 | A2 | derive_voices.py written; 8 derived voices (truncations, blends, splice) built and registered, 28 voices consistent. step.sh stage_a2 sweeps them plus 4 cross-lingual at one fixed seed. | user runs step.sh a2, then listens |
| 2026-09-09 | A2 | Sweep run: 13 takes, all 200, all distinct. wav_stats.py added for triage. Blends shift pitch for real (frm_50 at 150 Hz); cross-language blends show the low-rms mush signature; truncation and cross-lingual look safe. | user listens, then A3 |
| 2026-09-09 | A3 | A2 accepted, 6 keepers, with the seam-vs-smear failure mode understood. Found that Voxtral clones from a raw clip per request via ref_audio (LLM multimodal input, not the missing codec encoder); verified live, blocked only by soxr missing from the image. Track B put on hold. | user runs step.sh a3 |
