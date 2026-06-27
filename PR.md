# TADA TTS: talker decodes greedily with no repetition penalty (diverges from the HumeAI reference)

## What I need
Make the TADA talker decode like the HumeAI reference: repetition penalty + temperature + top-k/top-p, instead of pure greedy argmax. Plus expose the flow-matching knobs that are currently frozen at init. Defaults should equal upstream `InferenceOptions`.

## Why
With `tada-3b-ml`, `-l fr`, and a cloned French voice, output regularly hallucinates: wrong/repeated words, looping, mangled liaisons and punctuation. `TADA_NUM_CANDIDATES` does not help (even at 1000), because it only tunes the duration flow-matching head, not the content. The real cause is in the talker decoder, and right now none of it is tunable at runtime.

## The mistakes (current `main`)
1. The talker decodes **purely greedily**: `src/tada_tts.cpp:2625` is an unconditional `argmax_logits(...)` (its own comment says "Next token prediction (greedy)."). Greedy AR TTS with no repetition control loops and hallucinates, worst on lower-confidence non-English text.
2. **No repetition penalty, no top-p, no top-k** anywhere in the generation loop.
3. The `temperature` knob is **dead code**: the field is declared (`src/tada_tts.h:30`), defaulted to `0.0` (`src/tada_tts.cpp:1727`), and has a public setter `tada_set_temperature()` (`src/tada_tts.cpp:1955`), but it is never read by the token-selection step. So the one knob that looks available is a no-op.
4. This diverges from the reference. `tada/modules/tada.py` `InferenceOptions` defaults: `text_do_sample=True, text_temperature=0.6, text_top_k=0, text_top_p=0.9, text_repetition_penalty=1.1`. CrispASR effectively runs `do_sample=False` with no penalty. The acoustic/FM defaults (`acoustic_cfg=1.6`, `noise_temp=0.9`, `num_acoustic_candidates`) DO match, so the gap is specifically the text decoder.
5. Several model knobs are frozen at init with no setter/env (`src/tada_tts.h:32-37`): `acoustic_cfg`, `noise_temp`, `num_fm_steps`, `max_tokens`. And `duration_cfg` (1.0) and `cfg_schedule` (cosine) are hardcoded literals in `src/tada_tts.cpp` (not even parameters). Cannot experiment without recompiling.

## Proposed fix
- **Primary:** at the token-selection step (`src/tada_tts.cpp:2625`), apply repetition penalty over the already-generated ids, then temperature + top-k/top-p, then sample; fall back to argmax when `do_sample` is false or `temperature == 0`. Wire the existing `temperature` field. Add `text_top_p`, `text_top_k`, `text_repetition_penalty`, `text_do_sample` to `tada_context_params` with the upstream defaults above.
- **Plumbing:** expose via env vars + C ABI setters + server `/v1/audio/speech` params, mirroring the existing `TADA_NUM_CANDIDATES` / `tada_set_num_candidates()` pattern (`src/tada_tts.h:64`). Suggested env: `TADA_TEMPERATURE`, `TADA_TOP_P`, `TADA_TOP_K`, `TADA_REPETITION_PENALTY`, `TADA_DO_SAMPLE`.
- **Secondary (easy, same PR or follow-up):** add setters/env for the frozen knobs `acoustic_cfg`, `noise_temp`, `num_fm_steps`, `max_tokens`, and make `duration_cfg` / `cfg_schedule` parameters.

## Acceptance check
Same French text + cloned ref, new defaults vs current `main`: looping/hallucination gone; `--seed` now changes wording (proves sampling is live); raising `TADA_REPETITION_PENALTY` measurably reduces repeats.

---
Investigated and written with help from Claude Code.
