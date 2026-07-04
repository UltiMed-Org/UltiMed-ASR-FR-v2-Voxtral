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
