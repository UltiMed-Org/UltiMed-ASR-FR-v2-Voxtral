# Voxtral TTS: adding voices (plan and running log)

**STATUS:** B7 ran and the encoder route stalled at 4.8 pct held-out semantic top-1 (23.6 pct top-16) even with 93x more data, but B8 found why the whole track was stuck, and it was never the data. **The search destroys its own starting point.** Handed a seed with 100 pct correct semantic codes and acoustic codes off by at most one grid step, the optimizer drove semantic accuracy down to 43.75 pct; with the semantic half frozen (`--lr-semantic 0`) the same seed recovers **100 pct semantic and 100 pct acoustic, bit exact**. The acoustic half is not a search problem at all: it falls out of the audio by plain descent once the semantic codes are held still, and degrades gracefully (89.6 pct exact) even when the frozen semantic codes are only 25 pct right. Everything hard about this problem is one discrete choice: which of the 8192 semantic rows per frame.

**NEXT ACTION:** B8's continuation. Build the alternating solver: freeze semantic, descend acoustic, then re-score the 8192 semantic candidates per frame against the solved acoustics (`invert.py brute` already does the scoring for one frame), and repeat. B3 measured that the true semantic code is a unique argmin by 400x **when the acoustics are right**, and B8 shows an acoustic solve gets them right, so the two halves now feed each other instead of fighting. The encoder is demoted to an optional shortlist and needs no acoustic head at all.

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

**2. A claim I got wrong, corrected the same day by A4.** I read `serving_speech.py::_build_voxtral_prompt` and `_validate_voxtral_tts_request` and concluded that `ref_audio` cloning worked without the missing encoder, on the reasoning that `multi_modal_data: {"audio": [...]}` meant the AR LLM's own audio-understanding tower. That reasoning stopped one frame too early. The real chain is `embed_multimodal` (`voxtral_tts_audio_generation.py:1022`) -> `audio_tokenizer.encode_waveforms` (`voxtral_tts_audio_tokenizer.py:985`), which raises *"requires encoder weights which are not available in the open-source checkpoint"*. **The multimodal audio input IS the codec encoder.** Cloning hits exactly the piece that is missing, and the Background section was right all along.

Two things from A3 survive and matter. First, the request path is complete right up to the encoder call, so a checkpoint shipping encoder weights would clone with no code change. Second, `encode_tokens` (same file, just above `encode_waveforms`) turns codes into the `[N, 3072]` prefix by summing the 37 codebook lookups: that is B1's premise confirmed in shipped code, so recovering codes is sufficient and no encoder is needed for the second half of Track B.

**Decision: Track B is the route.** There is no shortcut past the missing encoder, so cloning `olivier 2.ogg` and `alicia2.ogg` means decoder inversion. Track A's 6 derived voices stay useful meanwhile, but they are blends of other people's voices, not the two targets.

Stage A4 below carried the actual goal, and is where this fell over.

### [!] A4. Clone the target voices via ref_audio: FAILED, mechanism unavailable

The user's own success criterion: `voices/olivier 2.ogg` and `voices/alicia2.ogg` must come out sounding like those people. Longer takes of the same two speakers exist (`voices/olivier.ogg`, `voices/alicia.ogg`), so the sweep does both lengths per speaker to see whether more reference audio clones better.

`step.sh a3` runs it: rebuild (soxr), boot, one anchor take with `fr_female` to prove the server is fine independently of cloning, then four clones at one fixed text and `--noise-seed 7`. Output in `cache/cloning/A3/wav/`.

Acceptance: both short clips produce a recognisable imitation of their speaker. Failure modes to watch for and what each would mean:

- Right speaker, wrong prosody or a wobbly identity: usable, tune with a longer or cleaner reference clip.
- Generic voice unrelated to the clip: the audio reached the model but carried no identity, so check the clip's sample rate and length against what `encode_speech_request` produced.
- Another `soxr` 400: the image did not actually rebuild (`up -d --build --force-recreate` is what `start_voxtral` does, check the log).

**Result:** 2026-09-09, Claude Code. Failed, and structurally, not as a bug we can fix.

Run 1 (`cache/cloning/logs/a3-20260909-222246.log`): `soxr` fixed, so the request reached the model. The anchor take with `fr_female` succeeded, then the first clone returned `500 EngineCore encountered an issue` and every request after it got `Broken pipe` / `Connection reset`: the engine process was dead.

Run 2 (`diag`) recovered the traceback:

```
embed_multimodal (voxtral_tts.py:286)
  -> embed_multimodal (voxtral_tts_audio_generation.py:1022)
    -> audio_tokenizer.encode_waveforms (voxtral_tts_audio_tokenizer.py:985)
RuntimeError: encode_waveforms requires encoder weights which are not
available in the open-source checkpoint.
```

Two lessons:

1. **`ref_audio` is not a way around the missing encoder, it is the way INTO it.** See the A3 correction above.
2. **One clone request killed the whole engine.** The raise happens inside `execute_model`, which vLLM treats as fatal, so it took down concurrent traffic too, from an unauthenticated request. Fork patch 0006 refuses `ref_audio` in `_validate_voxtral_tts_request`, turning an outage into one 400 that explains itself. Remove that guard if a checkpoint ever ships the encoder: everything else on the path already works.

`tts.py --ref-audio` and the `soxr` layer are kept. They cost nothing, they are correct, and they are exactly what a working checkpoint would need.

## Track B: new voices from real clips, by inverting the decoder

**THE route, since 2026-09-09, see A4.** `ref_audio` cloning turned out to call the missing encoder, so there is no shortcut. B1 is next and needs no server: CPU tensor work against the shipped decoder and codebooks.

Only start after A3 says Track A is insufficient. Each stage is a gate for the next.

### [x] B1. Recover the true codes of the shipped voices (CPU, no decoder)

Each `[N, 3072]` embedding is a sum of 37 rows from the known `[9088, 3072]` table, one row from each codebook's known offset range (offsets are `cumsum` of codebook sizes: 8194 for semantic, then 36 x 23 for acoustic, per `MultiVocabEmbeddings`). Recover the indices by coordinate descent: initialize by picking each codebook's nearest row to the residual, then repeatedly hold 36 fixed and re-solve the 37th exactly by nearest-row lookup on the residual.

Write `voxtral/cloning/recover_codes.py`, dump `cache/cloning/codes/<voice>.npy` of shape `[N, 37]`.

Acceptance: for all 20 voices, the reconstruction residual `||sum(rows) - embedding||` is at or near bf16 rounding noise, semantic codes land in `[0, 8193]`, acoustic codes in `[0, 22]`, and the recovered codes re-embed to the original tensor within tolerance.

Bonus output: the marginal statistics of real code streams (semantic histogram, transition structure, acoustic level distribution). Save them, they become the prior for B3 and B4.

Estimated: 2 to 3 h. Pure numpy or torch on CPU, seconds of compute.

**Result:** 2026-09-09, Claude Code. `voxtral/cloning/recover_codes.py` (`run` + `verify`). **Passed**, with one documented anomaly.

**Layout confirmed against the source, not guessed.** `MultiVocabEmbeddings` + `MultimodalAudioModelArgs.get_codebook_sizes` give sizes `[8194] + [23] * 36` (each codebook gets +2 special ids, `AudioSpecialTokens` has exactly two members), summing to 9022 real rows, and `nn.Embedding` allocates up to the next multiple of 128, which is the table's 9088. The plan's guess was right and the padding explains the rest. Sizes are re-derived from params.json at runtime rather than hardcoded.

**Results:** 18 of the 20 shipped voices recover **bit for bit**: re-embedding the codes and rounding to bfloat16 reproduces the stored tensor exactly, every element of every frame. Derived voices behave exactly as their construction predicts: the four truncations and the splice are bit-exact too (they are slices of real frames), while all four blends fail hard (0 to 1 percent of frames). **A blend of two code sums is not itself a code sum**, which is the A2 seam-vs-smear story confirmed arithmetically: the user heard `esf_50` switch voice mid-utterance, and here it is the only prefix that no set of codes can explain at all (0.26 percent).

That gives Track B a free on-manifold detector: any candidate prefix can be checked for "is this a thing the codebooks can actually represent" before spending a second of GPU on it.

**Two anomalies, both recorded rather than smoothed over:**

1. `cheerful_female` and `neutral_male` reach a residual of 0.0126 mean (relative ~9e-3, within bf16 rounding, so they pass the acceptance criterion as written) but **0 percent of frames are bit-exact**, against 0.0075 and 100 percent for the other 18. The gap is systematic, not a few stuck frames: every frame is slightly wrong by a similar amount, and the residual is not a constant offset (its per-frame mean vector is small next to its spread). 120 restarts and 25 sweeps did not move it. Something about those two prefixes differs; unresolved.
2. **Excluding the special-token rows makes recovery worse, not better.** Real audio codes are shifted past `[EMPTY_AUDIO]` / `[END_AUDIO]`, so forbidding ids 0 and 1 looks principled. Measured: it makes the worst frame 150x worse. Real voices genuinely contain frames built from those rows. The flag survives as `--no-special` purely to reproduce that measurement.

**Method note for B3/B4.** Plain block coordinate descent leaves ~5 percent of frames in a local minimum (95 percent bit-exact). Frames are independent, so the fix is to keep the solved ones and restart only the stragglers from random codes: 24 restarts take 18 of 20 voices to 100 percent. Expect the same shape of problem when inverting the decoder, and budget for restarts there too.

Outputs: `cache/cloning/codes/<voice>.npy` (`[N, 37]` int16) plus `stats.json` with the semantic-code usage and acoustic level histogram, the prior B3/B4 asked for.

### [x] B2. Load the decoder standalone and hear the reference clips

Instantiate `VoxtralTTSAudioTokenizer` in plain torch, load only `audio_tokenizer.decoder_blocks.*` and `output_proj` from `consolidated.safetensors`, feed it the B1 codes, write wavs. The class only reads `vllm_config.model_config.hf_config`, so a small namespace shim is enough. Watch for weight-norm parameterisation, the causal conv padding, and bf16 autocast.

Write `voxtral/cloning/decoder.py` (loader plus `decode(codes) -> waveform`) and `voxtral/cloning/decode_voices.py`.

Acceptance: `cache/cloning/B2/fr_female.wav` is intelligible speech that plainly matches the voice the server produces under `--voice fr_female`. This is the single most informative artifact in the whole plan: it is Mistral's actual reference recording, reconstructed.

Estimated: 3 to 5 h. This is the most engineering-heavy stage. If it stalls on plumbing, that is normal, log where and stop.

**Result: passed, and more cleanly than hoped.** All 20 shipped voices decode to fluent, intelligible speech in the language their name claims. `cache/cloning/B2/fr_female.wav` is 7.68 s of Madame Bovary read by a French woman ("Ce discours s'adressait plus particulierement a Emma, qui ne s'en etonna pas..."), transcribed back by the ASR service at :8002 to prove it rather than by ear. Same F0 as the server's own synthesis under `--voice fr_female` (181.8 Hz vs 187.5 Hz). So a voice prefix is literally a reference recording, and the whole encode side is now the only missing piece.

Notes worth carrying forward:

- The class instantiates outside vLLM through a meta-path finder that fabricates stub modules for the `vllm`, `mistral_common`, `transformers` and `aenum` roots plus every `vllm_omni.*` outside the voxtral_tts package. Two traps: import torch **before** installing the stubs (torch.library calls `inspect.getsourcefile` on whatever it finds), and do **not** stub `flash_attn` or `apex`, which the model imports under `try/except ImportError` and replaces with real fallbacks when absent. Stubbing them makes the import succeed and the stub then reaches `forward`.
- `load_state_dict(strict=False)` legitimately misses 115 tensors: the encoder half (`input_proj.`, `encoder_blocks.`) and `audio_token_embedding.`, which only `encode_tokens` reads. None is on the decode path.
- The B1 on-manifold test reproduces perfectly at the audio level. Every real voice has exactly one special-token frame, its last (the end-of-audio marker). The pure cut/concat derived voices have zero or one and decode to clean excerpts of the same recording (`d_frf_head60` is the first 4.8 s of the fr_female sentence, `d_frf_tail60` the tail). The 50/50 blends have 96 or 97 of 97 frames off-manifold and decode to 0.08 s of nothing. `d_frf_frm_25` loses 14 interior frames and decodes to French with words missing, which is exactly the "seam" A2 heard.
- The two voices B1 could not invert bit-exactly, `cheerful_female` and `neutral_male`, still decode to perfectly intelligible English. Their residual is below perceptual relevance, so it stops being a blocker for B3/B4. Still unexplained, still not worth chasing.
- Unrelated find: the ASR service now rejects a request without a `model` field, and `asr.py` has no `--model` flag, so `./asr.py file.wav` fails with 422. The transcriptions above went through curl with `model=deepdml/faster-whisper-large-v3-turbo-ct2`. Worth a separate fix.

### [!] B3. The paired test: does analysis by synthesis land on real codes

The decisive experiment for the "off-distribution prefix" risk. B1 and B2 together give a paired example: a waveform **and** its ground-truth codes.

Take the B2 waveform for one voice, forget its codes, and re-derive them by optimizing through the frozen decoder: continuous latents `[N, 292]` as the only parameters, straight-through rounding for the 36 FSQ dims, softmax-over-negative-distances with annealed temperature for the semantic dim, loss = multi-resolution STFT plus mel against the target. Because the decoder is causal with a small sliding window, prefer a frame-by-frame sequential search (CELP style) over a joint optimization if the joint version gets stuck.

Write `voxtral/cloning/invert.py`.

Acceptance, in order of importance:

1. Reconstruction is perceptually close to the target.
2. Recovered acoustic codes are mostly within one or two levels of the truth.
3. Recovered semantic codes match the truth on a **large fraction of frames** (report the number, do not eyeball it).

If 1 and 2 pass but 3 fails badly, the off-distribution risk is real: continue only with the mitigation of adding a prior term from B1's statistics, and re-test once.

Estimated: 4 to 6 h across possibly two sessions. Split into "get it converging" and "measure against truth" if needed.

**Result: the codes are identifiable, the optimizer cannot find them, and the reason is now pinned down exactly.** Written as `voxtral/cloning/invert.py` with four subcommands: `run` (gradient analysis by synthesis), `solve` (causal exhaustive search), `brute` (score all 8192 semantic codes for one frame) and `sensitivity` (how much does the audio depend on each code group). All numbers below are fr_female, the first 32 frames, on CPU.

Against the acceptance criteria, in the order the plan set:

1. Reconstruction perceptually close to target: **no**. Best loss 2.2 against 0.0 at the truth, which is worse than randomising a whole frame's 36 acoustic codes (1.65).
2. Acoustic codes within one or two levels: **partially**. 70.9 percent within 1 and 81.4 percent within 2, against 13.8 percent and roughly 27 percent by chance. Real signal, not "mostly".
3. Semantic codes on a large fraction of frames: **no**. 0 to 3 percent, i.e. chance (0.012 percent).

So this is not the branch the plan anticipated (1 and 2 passing, 3 failing). What the diagnostics found instead:

- **The waveform determines the codes exactly.** `brute` scores all 8192 semantic codes for one frame with everything else at the truth: the true code is the unique global minimum at loss 5.3e-5, the runner-up is at 0.021 and the median at 0.078. A factor of 400. True at frame 7 with seven frames of history, and equally true at frame 0 with none. Identifiability is not the problem.
- **The objective is a needle, not a slope.** 8192 sharp isolated minima in a 256-dimensional space is not something a continuous relaxation can descend into, which is why every gradient variant lands at chance on the semantic half.
- **The two halves of a frame are mutually blocking, and this is the actual wall.** Move only the probed frame's 36 acoustic dims from the truth to the mid level a search would start from, and the true semantic code falls from rank 1 to **rank 241 of 8192**, with the best impostor at half its loss (3.11 against 6.30). You need the acoustics to identify the semantic code and the semantic code to fit the acoustics. The causal frame-by-frame solver confirms it end to end: alternating the two, it got 0 of the first 4 frames right.
- **The semantic code is 14x less audible than the acoustic ones.** `sensitivity`: a random semantic substitution in one frame moves the loss by 0.118, randomising that frame's 36 acoustic codes moves it by 1.65. And this despite the semantic half being 256 of the 292 dims and carrying 47x the magnitude (per-frame norm 145 against 3.1). The decoder projects most of it away.
- **Two implementation findings worth keeping.** The plan's annealed softmax over negative distances is strictly worse than a VQ-VAE straight-through estimator: at a soft temperature the backward mixture describes nothing the forward pass did, and measured, it took a run holding 65 percent of the semantic codes down to 0 percent. The acoustic half wants a small rate (its grid step is 0.1 in [-1, 1]); the semantic half, despite its codebook rows sitting ~107 apart, wants a small one too. A first reading of that geometry said the opposite and the GPU follow-up below disproves it: a large semantic rate walks a good starting point straight out of its basin. The straight-through phase is off by default now: snapping the converged continuous solution is strictly better, and a control started at the truth plus 5 percent per-dimension noise recovers 100 percent of the semantic codes and 99.9 percent of the acoustic within 1, which is what proves the machinery and the gradient path are correct.

**Where that leaves Track B.** A prior term from B1's marginal statistics, the mitigation the plan named, does not address a deadlock between two halves of the same frame. Two things would, in rough order of promise:

1. **Fit the acoustics per candidate before ranking.** The rank-241 result says the true code loses only because its acoustics were not fitted; the rank-1 result says it wins by 400x once they are. So shortlist semantic candidates, fit each one's 36 acoustic dims properly, then re-rank. That is roughly 8192 small optimizations per frame, which is hours per frame on CPU and ordinary on the GPU this machine has. No learned prior needed.
2. **Do not recover the semantic codes at all: supply them.** They are the content stream and are largely speaker independent, which is why the acoustic half is the one that carries timbre. With the semantic sequence fixed, the per-frame problem is 36 dims on a 21-level grid, and the control above already recovers that at 94 percent within 1 by gradient alone. Where a plausible semantic sequence for the reference text comes from is the open question, and stage 0 of the server generates exactly such sequences.

Stopping here rather than burning more CPU: the measurement the stage existed to make is made, and both continuations want the GPU.

**GPU follow-up, same day.** Ran continuation 1 on the RTX 3090 Ti. `decoder.py` gained a `device` argument and now drops `encoder_blocks`, `input_proj` and `audio_token_embedding` after loading: they hold no checkpoint weights and 709 MB of the 1.3 GB parameter budget, which matters because the ASR replicas hold 19+ GB of the card and leave about 2.4 GB free. That cap on batch size is the reason everything below ran at chunk 16-64 rather than 512.

- **One default was wrong and it mattered.** Setting the semantic learning rate from the codebook geometry (rows ~107 apart, so a large rate) is exactly backwards: it walks a good initialisation out of its basin. At 1.0 a run seeded 5 percent from the truth recovers 12.5 percent of semantic codes; at 0.05 the same run recovers 100 percent.
- **The basin is far wider than the CPU session suggested, once that is fixed.** Seeded at the truth plus per-dimension Gaussian noise, semantic recovery is 100 percent at 0.3 sigma, 75 percent at 0.6, 50 percent at 1.0, 12.5 percent at 2.0 and 0 at 4.0. Acoustic recovery is 100 percent within one level out to 0.3 sigma.
- **But every cold start still fails, and now for a reason with a shape.** A random codebook row start gives 0 percent even though a row sits ~107 from the truth while 1.0 sigma of noise sits ~146 away, i.e. *further*, and recovers half the codes. A valid code is the bottom of its own basin, so descent has no reason to leave it; isotropic noise is not. Starting from the codebook's centre instead, which is no code at all and ~76 from any row, also gives 0 percent, at every noise level tried.
- **Continuation 1 fails on the fit, not on the ranking.** Judging 8192 candidates needs each one's acoustics fitted, and nothing fits them: against a floor of 0.0048 (true code, true acoustics), Adam reaches 1.92 at 400 steps and 1.88 at 3000, and exact coordinate descent over the 21 levels of each of the 36 dims reaches only 5.0, worse, because the dims turn out to be strongly coupled. At that distance from the floor the comparison between candidates is noise, and indeed an impostor beats the truth 1.51 to 1.62.
- **Parallel restarts do not rescue it either.** 256 independent continuous descents on one frame, 600 steps each, best loss 1.35 against the same 0.0048 floor, and the wrong code.

**What this actually establishes.** Decoder inversion works whenever it starts within about one sigma of the answer and never otherwise. Generic optimization cannot supply that starting point. What can is an approximate audio -> codes map, which is precisely the encoder the open checkpoint does not ship, so Track B as originally framed (invert the decoder *instead of* having an encoder) does not stand.

**The way through, and it is a good one: train the missing encoder.** The decoder generates unlimited labelled data for free. Decode code sequences to audio, keep the pair, and fit a small audio -> codes network on it; nothing has to be scraped or licensed and the labels are exact by construction. Off-distribution risk is controllable too, because the training codes can come from the server's own stage 0, which emits exactly the distribution the LLM later has to be prefixed with. Such a network does not even need to be good: getting within one sigma is enough, and the inversion machinery already built here finishes the job exactly. That is the next stage, it needs the GPU free rather than at 2.4 GB, and it replaces continuation 1.

### [!] B6. Train the missing encoder on decoder-generated pairs

`voxtral/cloning/train_encoder.py`. B3 left one job: an approximate audio -> codes map to seed the inversion with. The decoder is a deterministic function we hold in full, so pairs are free (choose codes, decode them, keep the pair) and the labels are *chosen rather than discovered*, which makes this ordinary supervised learning rather than the needle hunt B3 failed at. A per-frame conv encoder: six strided layers collapsing each frame's 1920 samples (1920 = 4x4x4x5x3x2), four dilation-free context layers so a frame can be read with its neighbours (the decoder had a 16-frame attention span when it produced the audio), then an 8192-way softmax for the semantic code and 36 21-way softmaxes for the acoustic levels. One voice held out.

Estimated: 3 to 4 h. Actual: 4 h, four dead ends, one hard limit.

**Result: the encoder trains, and the data to train it on does not exist yet.** Four things were measured, in this order, each one killing the previous plan.

- **Regress the decoder's continuous 292-dim input: 0.54 sigma from the truth on the held-out voice, and 0 pct of codes recovered.** Comfortably inside the band where B3 says descent recovers 75 to 100 pct, and useless anyway, because a squared-error fit predicts the mean of everything plausible and the mean of several codes is not a code. It lands between rows, which is exactly where B3 found descent helpless (a start at the codebook's centre also recovers zero from every distance tried). Hence classification heads: their answer is always a real row.
- **Uniform random codes teach nothing at all.** The obvious way to exploit a free decoder is to sample its whole domain. Logging accuracy on the *training* batch settled what the flat curve meant: **0.2 pct semantic top-1 on the batch just trained on**. Not a transfer failure, no learning. It is not capacity either: the same network fits 32 fixed frames to 100 pct on both heads in 3000 steps. Uniform codes put 37 independent dimensions of noise in every frame and inverting dense noise generalises about as well as inverting a hash. Real speech codes lie on a thin manifold where the mapping is smooth, and that manifold is the only place the encoder is ever asked to work.
- **A codebook-geometry semantic head is worse than a free one**, 4 pct top-16 after 1200 steps against 30 pct after 200. Emitting a 256-dim point and scoring it by distance to each real row should help, since neighbouring rows sound alike. It does not, because standardised rows sit ~22 apart, so the distance difference between two candidates maps to logits of order one, the softmax stays near uniform and so does the gradient. A free layer sets its own scale from the first step.
- **Train on the real frames and it memorises them, and this is the hard limit.** With jittered real runs as the data (real frames whose acoustic levels move a step or two, which stay speech-like and are unlimited), training loss collapses 12 -> 2.6 and train semantic top-1 reaches **94 pct**, while the held-out voice stays at **2 pct top-1, 14 pct top-16**. The reason is countable: the 20 shipped voices are **2823 frames, about 4 minutes of speech, and they contain only 1071 distinct semantic codes, 13 pct of the 8192**. A map whose output space the training set covers an eighth of cannot be learned, only memorised. (Jitter cannot fix it, because the semantic code is a choice among 8192 unordered rows, so "nudge it" is not defined: any change is a jump to unrelated audio.)

**What this establishes.** The pipeline is right and the data is the whole problem. It is also fixable, and cheaply, because the thing that emits real on-manifold code sequences is the server itself: stage 0 is an AR LLM that produces exactly these codes from arbitrary text in any of the 20 voices, and one 163 s request is 2048 frames, i.e. **most of a shipped voice per call**. A hundred calls is 70x the entire current training set, on distribution by construction rather than by argument. Getting it out needs a few lines in the fork (dump stage 0's codes alongside the audio in `make_omni_output`), the container back up, and an hour of generation.

**TODO:** the fork patch to dump stage 0 codes does not exist yet.

### [!] B7. Mint real code sequences from stage 0, then retrain B6

B6's wall was data, and the generator was the server all along: stage 0 is an AR LLM whose whole job is text -> on-manifold codes in any of the 20 voices, and what it emits is the same table-space ids a voice prefix carries. Fork patch **0007** (`voxtral_tts.py::_dump_frame_codes`) makes it append those codes to `<VOXTRAL_CODE_DUMP_DIR>/<tag>.i16` for any request carrying `voxtral_dump_codes`, riding the same per-row `sampling_extra_args` route as the noise seed, so it is per request and stays correct under concurrency. Nothing is written unless a request asks and the audio is byte-identical either way. `tts.py --dump-codes TAG` is the client end; compose sets the dir to `/cache/cloning/dumps`.

`voxtral/cloning/mint_codes.py` drives it: `corpus` fetches random Wikipedia article intros in the language each voice actually speaks (random topics on purpose, since semantic-code coverage follows phonetic variety and an encyclopedia wanders further than a novel), `mint` sends them across all 19 non-holdout voices and collects the dumps into `cache/cloning/codes/m_<voice>_<n>.npy`, which `train_encoder.py` already reads. The audio is deliberately not kept: the trainer decodes the codes locally to get exactly the audio they stand for. `load_pool` now treats the holdout as a voice rather than a filename, so minted utterances of it cannot quietly make the honest test dishonest.

Written and statically checked; **not yet run against a live server**, since `sudo docker` needs a password this agent does not have. The build is required, not optional: `voxtral/Dockerfile` COPYs the fork in, so an image built before patch 0007 accepts `voxtral_dump_codes` and writes nothing.

Steps: (1) `sudo docker compose up -d --build voxtral-tts`; (2) `./voxtral/cloning/mint_codes.py mint --minutes 240`; (3) retrain and compare held-out top-1/top-16 against B6's 2 pct / 14 pct; (4) if the encoder ranks the truth usefully, `invert.py run --init encoder` on `fr_female` and read the recovery.

Acceptance: held-out semantic top-16 well clear of B6's 14 pct, and `--init encoder` recovering a materially larger fraction of codes than any cold start (which is 0 pct). Note the target may be lower than it looks: B3 found the true semantic code is a unique, sharp argmin **once the acoustics are right**, so a good acoustic prediction plus a per-frame search over the 8192 rows is a second route that does not need the semantic head to win outright.

Estimated: 1 h of generation plus 2 to 3 h of training.

**Result: the pipeline works, the encoder still does not generalise, and that turned out not to matter (see B8).**

Everything server-side ran clean. Patch 0007 dumps were verified two independent ways: frame count against wav duration (52 frames vs 50 s of audio), and a full round trip through the local decoder (correlation 0.999976, relative RMS error 0.69 pct, the residual being the server's flash-attn/bf16 against the local SDPA/fp32). Generation produced **689 utterances, 0 failed, 259,585 frames**. The training pool went from 20 voices / 2823 frames / 13.1 pct semantic coverage to **709 files / 262,408 frames / 5.8 h of speech / 46.4 pct coverage** (3798 of 8192 codes). The held-out set stopped being a 96-frame prefix holding 64 distinct codes and became **40 minted `fr_female` utterances, 9364 frames**.

The retrain (40000 steps, batch 64, 32-frame windows, about 45 min on the 24 GB card) beat B6 and then flatlined:

| | B6 (2823 frames) | B7 (262,408 frames) |
|---|---|---|
| held-out semantic top-1 | 2 pct | **4.8 pct** |
| held-out semantic top-16 | 14 pct | **23.6 pct** |
| held-out acoustic exact | - | 16.5 pct |
| held-out acoustic within-1 | - | 43.9 pct |
| training-batch semantic top-1 | 94 pct | 96 to 99 pct |

Held-out top-1 did not move after step 3200 while the training batch climbed to 99 pct, so this is memorisation, not a data shortage that another 93x would fix. 93x the data bought 9 points of top-16.

`--init encoder` was run against the retrained checkpoint for the record, on 32 frames of the held-out `fr_female`, both ways:

```
encoder seed: semantic top1 3.1%  top16 18.8%  median rank 111  acoustic exact 16.2% within1 41.4%
semantic free:   semantic exact 0.00%  (the search deletes even a weak seed)
semantic frozen: semantic exact 3.12%  acoustic within1 92.97%  (up from 41.4%)
```

The keeper is **median rank 111**: the true code sits in the top 111 of 8192 half the time, a 74x pruning of the discrete search, even though the encoder is useless as a predictor. That is the job it should be given, and it needs no acoustic head to do it.

Acceptance therefore **fails as written** (top-16 is clear of 14 pct, but `--init encoder` cannot recover materially more than a cold start from a 4.8 pct seed). The note in the acceptance criterion turned out to be the important sentence: the second route, per-frame search over the 8192 rows with the acoustics right, is the one that works. B8.

### [x] B8. The search was miscoupled: freeze the semantic half

Not a planned stage. It came out of asking why the B7 encoder's acoustic head could not fit even its own training batch (about 50 pct within-1 on frames it had just seen), while the semantic head memorised them at 99 pct.

First hypothesis, that the decoder is many-to-one in the 36 FSQ dims so those codes are unrecoverable by construction: **measured and killed.** `invert.py sensitivity --per-dim` perturbs one code row by one step and reports how far the waveform moves relative to its own RMS. No dead dimensions: the weakest single-step effect is 1.28 pct of RMS, the median 2.24 pct, against 0.83 pct for replacing a semantic code with an unrelated row. Every acoustic dimension is audible, so inversion is well posed.

Second, `invert.py run --init degraded` was added, which corrupts the truth to a chosen accuracy instead of asking a real encoder for a guess (semantic errors uniform over the codebook, since a wrong row in an unordered table is an unrelated row; acoustic errors as near misses on the ordered grid). Sweeping the seed quality, all at 200 warmup steps on 16 frames of `fr_female`:

| seed semantic | seed acoustic | search result (semantic exact) |
|---|---|---|
| 100 pct | exact | 100 pct |
| 100 pct | within-1 | 43.75 pct |
| 87.5 pct | within-1 | 31.25 pct |
| 68.8 pct | within-1 | 25.0 pct |
| 25.0 pct | within-1 | 18.75 pct |
| truth + 0.1 sigma noise | - | 31.25 pct |

The second row is the finding. A seed whose semantic codes were **entirely correct**, with acoustic codes off by at most one grid step, came out at 43.75 pct: the search converted acoustic error into semantic error. It sheds the most from the best seed (a 25 pct seed loses 6 points, a 100 pct seed loses 56).

Then the same seed with `--lr-semantic 0`:

| | semantic free | semantic frozen |
|---|---|---|
| semantic exact | 43.75 pct | **100 pct** |
| acoustic exact | not recovered | **100 pct** (mean abs error 0.0) |

Same seed, same steps, everything else identical. With the 256-dim semantic embeddings held still, the 36 acoustic dims descended from 35.4 pct exact to bit-exact on every dim of every frame. With them free, the gradient spends itself dragging correct semantic codes off their codebook rows, and once one crosses into a neighbouring cell nothing pulls it back.

The halves separate cleanly, which is what makes the alternating scheme viable:

| frozen semantic accuracy | acoustic recovered exactly |
|---|---|
| 100 pct | 100.0 pct |
| 87.5 pct | 99.65 pct |
| 25 pct | 89.6 pct |

Consequences: B3's "identifiable but not findable" was measuring an optimizer that unfreezes the one half that must be held still, so that conclusion is superseded rather than merely refined. The encoder needs **no acoustic head**, so more than half of what B6 and B7 trained was never needed. And the remaining problem is discrete, not continuous.

Not yet done, and the reason this stage is `[x]` for the diagnosis but not for the fix: the alternating solver itself is unbuilt, and the open question is whether re-scoring the 8192 semantic candidates against *solved* acoustics ranks the truth first (B3 measured rank 1 against *true* acoustics and rank 241 against a search's starting values, so the interesting case sits between two known numbers).

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
| 2026-09-09 | A4 | ref_audio cloning FAILED: embed_multimodal calls encode_waveforms, whose weights are absent, and the raise killed EngineCore. Fork patch 0006 refuses it with a 400 instead. My A3 claim that cloning sidesteps the encoder was wrong, corrected in place. Track B is the route. | B1 |
| 2026-09-09 | B1 | Codes recovered: 18/20 shipped voices bit-exact, 2 at bf16 noise but never bit-exact (unresolved). Blends provably unrepresentable, giving a free on-manifold test. Special-token rows are used by real voices, excluding them hurts. | B2 |
| 2026-09-10 | B2 | Codec decoder runs standalone on CPU via an import shim. All 20 voices decode to fluent speech in their advertised language, ASR-verified; fr_female is Madame Bovary at 181.8 Hz. B1's on-manifold test reproduces exactly at the audio level. The 2 non-bit-exact voices still sound fine. | B3 |
| 2026-09-10 | B3 | Codes are identifiable (true semantic code is the unique argmin by 400x, with or without history) but not findable: the semantic and acoustic halves of a frame block each other, rank 1 -> rank 241 when one frame's acoustics move to a search's starting value. Gradient AbS at chance on semantic, 70.9 pct within 1 on acoustic. Two GPU-shaped continuations proposed. | user picks a continuation |
| 2026-09-10 | B3 GPU | Fixed a wrong semantic learning rate (1.0 -> 0.05), which widened the measured basin to 100 pct at 0.3 sigma and 50 pct at 1.0 sigma. Cold starts still fail everywhere: random codebook row, codebook centre, 256 restarts, shortlist-then-fit (no fitter gets within 400x of the floor). Conclusion: inversion needs an approximate encoder to seed it, and one can be distilled from the decoder. | train that encoder |
| 2026-09-10 | B6 | Encoder trains, data does not exist. Four dead ends: regression to 0.54 sigma recovers 0 pct (the mean of codes is not a code); uniform random codes give 0.2 pct on the training batch itself (inverting dense noise generalises like inverting a hash); a codebook-geometry head is worse than a free one (flat logits, flat gradient); real data gives 94 pct train and 2 pct held out because 20 voices are 2823 frames covering 13 pct of the 8192 semantic codes. | mint real codes from stage 0 |
| 2026-09-10 | B7 built | Fork patch 0007 dumps stage-0 codes per request (tag in extra_params, same route as the noise seed, correct under concurrency); tts.py --dump-codes, VOXTRAL_CODE_DUMP_DIR in compose, mint_codes.py to drive it, unit tests for the dump. Not yet run: needs `sudo docker compose up -d --build voxtral-tts`, which needs a password. | generate and retrain |
| 2026-09-10 | B7 run | Minted 689 utterances / 259,585 frames from stage 0 (0 failed), taking the pool from 2823 frames at 13.1 pct semantic coverage to 262,408 at 46.4 pct, with a real 40-utterance held-out set. Retrain beat B6 (top-16 14 -> 23.6 pct, top-1 2 -> 4.8 pct) then flatlined from step 3200 while the training batch hit 99 pct. 93x the data bought 9 points. | why the acoustic head cannot fit |
| 2026-09-10 | B8 | **The search was destroying its seed.** Killed the many-to-one-decoder hypothesis by measurement (no dead dims, weakest single-step effect 1.28 pct of RMS). Added --init degraded and found a 100 pct-correct semantic seed comes out at 43.75 pct; frozen (--lr-semantic 0) the same seed recovers 100 pct semantic and 100 pct acoustic bit-exact. The acoustic half is free once semantic is held still, even at 25 pct frozen accuracy (89.6 pct exact). B3's conclusion is superseded; the encoder needs no acoustic head. | build the alternating solver |
