#!/usr/bin/env bash
# Mirrors the serve step of voxtral_test/install.sh, but for the container:
# copies the packaged voxtral_tts stage config, optionally tunes GPU memory, then
# serves mistralai/Voxtral-4B-TTS-2603 on 0.0.0.0 so the host can reach it.
set -euo pipefail

MODEL="${VOXTRAL_TTS_MODEL:-mistralai/Voxtral-4B-TTS-2603}"
PORT="${VOXTRAL_TTS_PORT:-8003}"

# Locate the stage config shipped inside vllm_omni and copy it so we can tune it
# without mutating the read-only site-packages copy.
STAGE_YAML="$(python3 -c 'import vllm_omni, os; print(os.path.join(os.path.dirname(vllm_omni.__file__), "model_executor/stage_configs/voxtral_tts.yaml"))')"
TUNED_YAML=/tmp/voxtral_tts_tuned.yaml
cp "$STAGE_YAML" "$TUNED_YAML"

# Optional: shrink the GPU memory footprint when sharing the card with the
# crispasr / crispasr-tts containers (install.sh tuned this down to 0.35).
if [ -n "${VLLM_GPU_MEM_UTIL:-}" ]; then
  sed -i "s/gpu_memory_utilization: .*/gpu_memory_utilization: ${VLLM_GPU_MEM_UTIL}/g" "$TUNED_YAML"
fi

# --enforce-eager matches install.sh (skips CUDA-graph capture; lower VRAM, a bit
# slower). VLLM_EXTRA_ARGS is an opt-in escape hatch for further flags.
exec vllm-omni serve "$MODEL" \
  --omni \
  --stage-configs-path "$TUNED_YAML" \
  --host 0.0.0.0 \
  --port "$PORT" \
  --enforce-eager \
  ${VLLM_EXTRA_ARGS:-}
