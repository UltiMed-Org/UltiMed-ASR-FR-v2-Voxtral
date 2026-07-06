#!/usr/bin/env bash
# Serve mistralai/Voxtral-4B-TTS-2603 via vllm-omni (0.22.x) on 0.0.0.0 so the
# host can reach it. In 0.22 the model is registered, so `vllm serve MODEL --omni`
# auto-loads the bundled deploy YAML (vllm_omni/deploy/voxtral_tts.yaml); we copy
# that YAML, patch a few knobs, and hand it back with --deploy-config.
#
# Voxtral is a TWO-STAGE pipeline (see the deploy YAML, top-level `stages:` list):
#   stage_id 0 = AR LLM, text -> audio tokens. Compiled (enforce_eager: false),
#                greedy (temperature 0.0), gpu_memory_utilization ~0.8.
#   stage_id 1 = acoustic decoder, tokens -> 24 kHz wave. Its temperature
#                (0.9 upstream) is the main audio-quality lever.
# Voxtral's sampling params are NOT settable per HTTP request (only voice/speed/
# response_format are), so we expose them here as startup knobs by editing the
# copied YAML at boot. No rebuild needed to change them, just `up -d`.
#
# NOTE: the 0.22 schema differs from 0.18 (`stages` with flat fields, not
# `stage_args`+`engine_args`+`runtime`). The old `runtime.defaults.max_inflight`
# lever is gone; per the maintainers the acoustic stage is `codec-bs=1`, so audio
# throughput saturates around client concurrency 4-8 regardless.
set -euo pipefail

MODEL="${VOXTRAL_TTS_MODEL:-mistralai/Voxtral-4B-TTS-2603}"
PORT="${VOXTRAL_TTS_PORT:-8003}"

TUNED_YAML=/tmp/voxtral_tts_tuned.yaml

# Locate the bundled deploy YAML AND patch a copy of it in ONE python process, so
# there is no `$(python3 ...)` capture to pollute. (Capturing the path separately
# broke: `import vllm_omni` prints a monkey-patch WARNING to stdout, which got
# concatenated onto the path.) Check both the 0.22 path (deploy/) and the legacy
# 0.18 path so this survives a version bump. Only the knobs an env var asks for
# are touched:
#   VOXTRAL_ACOUSTIC_TEMP -> stage_id 1 sampling temperature. Default 0.7 (lowered
#                            from upstream 0.9 for steadier, less artifacty audio;
#                            raise toward 0.9 for more expressive/varied output).
#   VOXTRAL_SEED          -> seed on BOTH stages (fixed = reproducible; change it
#                            for a different rendition of the same text+voice).
#   VOXTRAL_MAX_NUM_SEQS  -> both stages' max_num_seqs (packaged 32). The acoustic
#                            stage is codec-bs=1, so raising this past the ~4-8
#                            concurrency saturation point buys little.
#   VOXTRAL_CFG_ALPHA     -> stage_id 0 classifier-free guidance scale (default
#                            1.2). Nudge up for tighter text adherence; NEVER 1.0
#                            (disables CFG -> garbled/off-text audio; refused).
# (VOXTRAL_EULER_STEPS is handled separately below: it patches the model's
#  params.json, not this deploy YAML.)
VOXTRAL_ACOUSTIC_TEMP="${VOXTRAL_ACOUSTIC_TEMP:-0.7}" \
python3 - "$TUNED_YAML" <<'PY'
import os, sys, vllm_omni, yaml

dst = sys.argv[1]
base = os.path.dirname(vllm_omni.__file__)
src = next((p for p in (
    os.path.join(base, "deploy/voxtral_tts.yaml"),
    os.path.join(base, "model_executor/stage_configs/voxtral_tts.yaml"),
) if os.path.exists(p)), None)
if src is None:
    sys.exit("voxtral entrypoint: could not find voxtral_tts.yaml under %s" % base)

with open(src) as f:
    cfg = yaml.safe_load(f)

temp = os.environ.get("VOXTRAL_ACOUSTIC_TEMP")  # always set (default injected above)
seed = os.environ.get("VOXTRAL_SEED")
max_num_seqs = os.environ.get("VOXTRAL_MAX_NUM_SEQS")
cfg_alpha = os.environ.get("VOXTRAL_CFG_ALPHA")  # stage 0 classifier-free guidance
# Guardrail: cfg_alpha=1.0 disables CFG, which makes the flow-matching decoder
# hallucinate phonetically-plausible-but-wrong audio (it stops following the
# text). The paper/default is 1.2; refuse the known-broken value outright.
if cfg_alpha is not None and abs(float(cfg_alpha) - 1.0) < 1e-9:
    sys.exit(
        "voxtral entrypoint: VOXTRAL_CFG_ALPHA=1.0 disables classifier-free "
        "guidance and produces garbled/off-text audio. Use ~1.2 (default); "
        "raise slightly for tighter text adherence."
    )

for stage in cfg.get("stages", []):
    sampling = stage.setdefault("default_sampling_params", {})
    if seed:
        sampling["seed"] = int(seed)
    if max_num_seqs:
        stage["max_num_seqs"] = int(max_num_seqs)
    # stage_id 1 = acoustic decoder; its temperature is the audio-quality lever.
    if stage.get("stage_id") == 1 and temp:
        sampling["temperature"] = float(temp)
    # stage_id 0 = AR LLM; cfg_alpha lives in ITS sampling extra_args (that is
    # where vllm-omni's voxtral model reads per-request cfg_alpha from too).
    if stage.get("stage_id") == 0 and cfg_alpha:
        sampling.setdefault("extra_args", {})["cfg_alpha"] = float(cfg_alpha)

with open(dst, "w") as f:
    yaml.safe_dump(cfg, f, sort_keys=False)

sys.stderr.write(
    "voxtral entrypoint: patched %s -> %s "
    "(acoustic_temp=%s seed=%s max_num_seqs=%s cfg_alpha=%s)\n"
    % (src, dst, temp, seed or "default", max_num_seqs or "default",
       cfg_alpha or "default")
)
PY

# VOXTRAL_EULER_STEPS -> n_decoding_steps: the number of Euler ODE steps the
# stage-1 flow-matching acoustic decoder runs per frame. THIS is the compute<->
# quality lever the user asked for: more steps = steadier/higher-fidelity audio
# for more compute, fewer = faster/rougher. Unlike temp/seed/cfg_alpha it is NOT
# a deploy-YAML field, env var, or per-request knob in vllm-omni. vllm-omni reads
# it ONLY from the model's params.json at
#   multimodal.audio_model_args.acoustic_transformer_args.n_decoding_steps
# and this model does not ship the key, so vllm-omni's mistral parser WARNS and
# falls back to 7 (parsers/voxtral_tts.py). So, exactly as that warning tells you
# to, we inject the key into params.json. We pre-fetch just params.json with
# hf_hub_download (tiny; uses HF_TOKEN if the repo is gated) so this also works on
# the very first boot before the ~9GB weights download, and re-patch every boot
# (idempotent) so it survives HF restoring the original file. Leave the var unset
# to keep vllm-omni's default of 7.
if [ -n "${VOXTRAL_EULER_STEPS:-}" ]; then
  python3 - "$MODEL" "$VOXTRAL_EULER_STEPS" <<'PY'
import sys, os, json
from huggingface_hub import hf_hub_download

model, raw = sys.argv[1], sys.argv[2]
try:
    steps = int(raw)
except ValueError:
    sys.exit("voxtral entrypoint: VOXTRAL_EULER_STEPS must be an integer, got %r" % raw)
if steps < 1:
    sys.exit("voxtral entrypoint: VOXTRAL_EULER_STEPS must be >= 1, got %d" % steps)

path = os.path.realpath(hf_hub_download(model, "params.json"))
with open(path) as f:
    cfg = json.load(f)
ata = (cfg.setdefault("multimodal", {})
          .setdefault("audio_model_args", {})
          .setdefault("acoustic_transformer_args", {}))
old = ata.get("n_decoding_steps")
ata["n_decoding_steps"] = steps
os.chmod(path, 0o644)  # HF cache blobs can be read-only; make writable
with open(path, "w") as f:
    json.dump(cfg, f)
sys.stderr.write(
    "voxtral entrypoint: set acoustic_transformer_args.n_decoding_steps "
    "%s -> %d in %s\n" % (old if old is not None else "unset(default 7)", steps, path)
)
PY
fi

# 0.22 serve form: `vllm serve MODEL --omni` (the vllm-omni plugin registers the
# omni pipeline into vllm). --deploy-config feeds our patched YAML. VLLM_EXTRA_ARGS
# is an opt-in escape hatch, left unquoted so an empty value expands to nothing.
exec vllm serve "$MODEL" \
  --omni \
  --deploy-config "$TUNED_YAML" \
  --host 0.0.0.0 \
  --port "$PORT" \
  ${VLLM_EXTRA_ARGS:-}
