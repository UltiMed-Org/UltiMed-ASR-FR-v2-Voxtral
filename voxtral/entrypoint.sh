#!/usr/bin/env bash
# Mirrors the serve step of voxtral/install.sh, but for the container: copies the
# packaged voxtral_tts stage config, patches a few knobs per stage, then serves
# mistralai/Voxtral-4B-TTS-2603 on 0.0.0.0 so the host can reach it.
#
# Voxtral is a TWO-STAGE pipeline (see the copied YAML):
#   stage "audio_generation" (stage 0) = AR LLM, text -> audio tokens. It is the
#     VRAM hog (gpu_memory_utilization ~0.8) and runs greedily (temperature 0.0).
#   stage "audio_tokenizer"  (stage 1) = acoustic decoder, tokens -> 24 kHz wave.
#     Its temperature (default 0.9 upstream) is the main audio-quality lever.
# Voxtral's sampling params are NOT settable per HTTP request (only voice/speed/
# response_format are), so we expose them here as startup knobs by editing the
# copied YAML at boot. No rebuild needed to change them, just `up -d`.
set -euo pipefail

MODEL="${VOXTRAL_TTS_MODEL:-mistralai/Voxtral-4B-TTS-2603}"
PORT="${VOXTRAL_TTS_PORT:-8003}"

# Locate the stage config shipped inside vllm_omni and copy it so we can tune it
# without mutating the read-only site-packages copy.
STAGE_YAML="$(python3 -c 'import vllm_omni, os; print(os.path.join(os.path.dirname(vllm_omni.__file__), "model_executor/stage_configs/voxtral_tts.yaml"))')"
TUNED_YAML=/tmp/voxtral_tts_tuned.yaml

# Patch precisely per stage with pyyaml (ships with vllm-omni, which loads these
# same YAMLs). Only the knobs an env var asks for are touched:
#   VLLM_GPU_MEM_UTIL     -> stage-0 gpu_memory_utilization ONLY (leave stage 1's
#                            small 0.1 reservation alone; a blunt global replace
#                            would wrongly inflate it).
#   VOXTRAL_ACOUSTIC_TEMP -> stage-1 sampling temperature. Default 0.7 (lowered
#                            from upstream 0.9 for steadier, less artifacty audio;
#                            raise toward 0.9 for more expressive/varied output).
#   VOXTRAL_SEED          -> seed on BOTH stages (fixed = reproducible; change it
#                            for a different rendition of the same text+voice).
VOXTRAL_ACOUSTIC_TEMP="${VOXTRAL_ACOUSTIC_TEMP:-0.7}" \
python3 - "$STAGE_YAML" "$TUNED_YAML" <<'PY'
import os, sys, yaml

src, dst = sys.argv[1], sys.argv[2]
with open(src) as f:
    cfg = yaml.safe_load(f)

gpu = os.environ.get("VLLM_GPU_MEM_UTIL")
temp = os.environ.get("VOXTRAL_ACOUSTIC_TEMP")  # always set (default injected above)
seed = os.environ.get("VOXTRAL_SEED")

for stage in cfg.get("stage_args", []):
    engine_args = stage.setdefault("engine_args", {})
    model_stage = engine_args.get("model_stage")
    sampling = stage.setdefault("default_sampling_params", {})

    if model_stage == "audio_generation" and gpu:
        engine_args["gpu_memory_utilization"] = float(gpu)
    if model_stage == "audio_tokenizer" and temp:
        sampling["temperature"] = float(temp)
    if seed:
        sampling["seed"] = int(seed)

with open(dst, "w") as f:
    yaml.safe_dump(cfg, f, sort_keys=False)

sys.stderr.write(
    "voxtral entrypoint: patched stage config -> %s (gpu=%s acoustic_temp=%s seed=%s)\n"
    % (dst, gpu or "default", temp, seed or "default")
)
PY

# Eager mode (default ON, matching install.sh): --enforce-eager skips CUDA-graph
# capture, so it uses less VRAM and starts faster but is a bit slower in steady
# state. Set VOXTRAL_ENFORCE_EAGER=0 to drop the flag and let the per-stage YAML
# decide (stage 0 audio_generation captures graphs -> faster generation, at the
# cost of more VRAM + a first-boot capture delay; stage 1 stays eager).
EAGER_FLAG="--enforce-eager"
case "${VOXTRAL_ENFORCE_EAGER:-1}" in
  0 | false | False | no | off) EAGER_FLAG="" ;;
esac

# VLLM_EXTRA_ARGS is an opt-in escape hatch for further flags. Both it and
# EAGER_FLAG are intentionally unquoted so an empty value expands to nothing.
exec vllm-omni serve "$MODEL" \
  --omni \
  --stage-configs-path "$TUNED_YAML" \
  --host 0.0.0.0 \
  --port "$PORT" \
  ${EAGER_FLAG} \
  ${VLLM_EXTRA_ARGS:-}
