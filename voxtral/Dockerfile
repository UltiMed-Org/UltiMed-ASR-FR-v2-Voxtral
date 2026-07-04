# Voxtral text->audio (TTS) served via vLLM's "omni" stack.
#
# This is NOT CrispASR: mistralai/Voxtral-4B-TTS-2603 is a vLLM model, so it runs
# on the official vLLM image plus the vllm-omni extension. Versions are pinned to
# match voxtral/install.sh (the recipe that was validated by hand).
#
# NOTE: the base tag must match the pinned vllm version below. If Docker Hub does
# not have vllm/vllm-openai:v0.18.1, bump both to the nearest published tag.
FROM vllm/vllm-openai:v0.18.1

# vllm-omni adds the TTS serving stages (voxtral_tts.yaml). It ships in no
# prebuilt image, so install the exact version install.sh used.
RUN pip install --no-cache-dir vllm-omni==0.18.0

# The entrypoint copies the packaged voxtral_tts stage config (so gpu memory can
# be tuned via VLLM_GPU_MEM_UTIL) and then runs `vllm-omni serve`, exactly like
# install.sh does locally.
COPY entrypoint.sh /usr/local/bin/voxtral-tts-entrypoint.sh
RUN chmod +x /usr/local/bin/voxtral-tts-entrypoint.sh
ENTRYPOINT ["/usr/local/bin/voxtral-tts-entrypoint.sh"]
