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

# Without flash-attn the stage-1 audio_tokenizer (acoustic decoder) logs
# "flash_attn is not installed. Falling back to PyTorch SDPA ..." and runs its
# attention on the slower SDPA path. Build it against the torch already in the
# base image (--no-build-isolation); if no matching prebuilt wheel exists this
# compiles from source and the build gets slow. The version is pinned so the
# wheel matches the base image's torch/CUDA; bump it if the base tag changes.
# TODO: confirm 2.7.4 has a prebuilt wheel for vllm/vllm-openai:v0.18.1's torch;
# if the build stalls compiling from source, pin to whatever wheel matches.
RUN pip install --no-cache-dir --no-build-isolation flash-attn==2.7.4.post1

# compose runs this container as a non-root uid (default 1000) so ./cache stays
# host-owned, but the base image has no /etc/passwd entry for that uid. vLLM
# startup calls getpass.getuser() -> pwd.getpwuid(), which then dies with
# "getpwuid(): uid not found: 1000". Bake a matching user so the lookup resolves.
# APP_UID/APP_GID are fed from CRISPASR_UID/GID by docker-compose build.args, so
# this tracks whatever uid compose actually runs as.
ARG APP_UID=1000
ARG APP_GID=1000
RUN groupadd -g "${APP_GID}" voxtral 2>/dev/null || true \
 && useradd -u "${APP_UID}" -g "${APP_GID}" -M -d /cache -s /usr/sbin/nologin voxtral 2>/dev/null || true

# The entrypoint copies the packaged voxtral_tts stage config (so gpu memory can
# be tuned via VLLM_GPU_MEM_UTIL) and then runs `vllm-omni serve`, exactly like
# install.sh does locally.
COPY entrypoint.sh /usr/local/bin/voxtral-tts-entrypoint.sh
RUN chmod +x /usr/local/bin/voxtral-tts-entrypoint.sh
ENTRYPOINT ["/usr/local/bin/voxtral-tts-entrypoint.sh"]
