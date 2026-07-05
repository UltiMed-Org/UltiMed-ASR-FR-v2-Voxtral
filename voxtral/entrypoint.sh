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

# Locate the bundled deploy YAML inside vllm_omni and copy it so we can tune it
# without mutating the read-only site-packages copy. Check both the 0.22 path
# (deploy/) and the legacy 0.18 path so this survives a version bump either way.
STAGE_YAML="$(python3 - <<'PY'
import os, vllm_omni
base = os.path.dirname(vllm_omni.__file__)
for rel in ("deploy/voxtral_tts.yaml", "model_executor/stage_configs/voxtral_tts.yaml"):
    p = os.path.join(base, rel)
    if os.path.exists(p):
        print(p)
        break
PY
)"
if [ -z "$STAGE_YAML" ]; then
  echo "voxtral entrypoint: could not find voxtral_tts.yaml in vllm_omni" >&2
  exit 1
fi
TUNED_YAML=/tmp/voxtral_tts_tuned.yaml

# Patch precisely with pyyaml (ships with vllm-omni). Only the knobs an env var
# asks for are touched:
#   VOXTRAL_ACOUSTIC_TEMP -> stage_id 1 sampling temperature. Default 0.7 (lowered
#                            from upstream 0.9 for steadier, less artifacty audio;
#                            raise toward 0.9 for more expressive/varied output).
#   VOXTRAL_SEED          -> seed on BOTH stages (fixed = reproducible; change it
#                            for a different rendition of the same text+voice).
#   VOXTRAL_MAX_NUM_SEQS  -> both stages' max_num_seqs (packaged 32). The acoustic
#                            stage is codec-bs=1, so raising this past the ~4-8
#                            concurrency saturation point buys little.
VOXTRAL_ACOUSTIC_TEMP="${VOXTRAL_ACOUSTIC_TEMP:-0.7}" \
python3 - "$STAGE_YAML" "$TUNED_YAML" <<'PY'
import os, sys, yaml

src, dst = sys.argv[1], sys.argv[2]
with open(src) as f:
    cfg = yaml.safe_load(f)

temp = os.environ.get("VOXTRAL_ACOUSTIC_TEMP")  # always set (default injected above)
seed = os.environ.get("VOXTRAL_SEED")
max_num_seqs = os.environ.get("VOXTRAL_MAX_NUM_SEQS")

for stage in cfg.get("stages", []):
    sampling = stage.setdefault("default_sampling_params", {})
    if seed:
        sampling["seed"] = int(seed)
    if max_num_seqs:
        stage["max_num_seqs"] = int(max_num_seqs)
    # stage_id 1 = acoustic decoder; its temperature is the audio-quality lever.
    if stage.get("stage_id") == 1 and temp:
        sampling["temperature"] = float(temp)

with open(dst, "w") as f:
    yaml.safe_dump(cfg, f, sort_keys=False)

sys.stderr.write(
    "voxtral entrypoint: patched deploy config -> %s "
    "(acoustic_temp=%s seed=%s max_num_seqs=%s)\n"
    % (dst, temp, seed or "default", max_num_seqs or "default")
)
PY

# 0.22 serve form: `vllm serve MODEL --omni` (the vllm-omni plugin registers the
# omni pipeline into vllm). --deploy-config feeds our patched YAML. VLLM_EXTRA_ARGS
# is an opt-in escape hatch, left unquoted so an empty value expands to nothing.
exec vllm serve "$MODEL" \
  --omni \
  --deploy-config "$TUNED_YAML" \
  --host 0.0.0.0 \
  --port "$PORT" \
  ${VLLM_EXTRA_ARGS:-}
