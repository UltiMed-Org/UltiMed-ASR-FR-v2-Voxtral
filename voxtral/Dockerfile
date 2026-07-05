# Voxtral text->audio (TTS) served via vLLM's "omni" stack.
#
# This is NOT CrispASR: mistralai/Voxtral-4B-TTS-2603 is a vLLM model. As of
# vllm-omni 0.22 there is an OFFICIAL prebuilt image (vllm/vllm-omni) that
# already bundles the matching vLLM + vllm-omni + the deploy YAMLs, so we build
# a thin layer on top of it instead of the old base-image + `pip install
# vllm-omni` + hand-built flash-attn wheel. -x86_64 pins the arch (there are also
# -aarch64 and multi-arch tags); bump the tag to move vllm-omni versions.
FROM vllm/vllm-omni:v0.22.0-x86_64

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

# The entrypoint copies the bundled voxtral_tts deploy config, patches the tuning
# knobs into it, and runs `vllm serve MODEL --omni --deploy-config <patched>`.
# NOTE: the prebuilt image does NOT ship the standalone flash-attn package, so the
# stage-1 acoustic decoder may still log "flash_attn is not installed. Falling
# back to PyTorch SDPA". To restore it, add a prebuilt-wheel install matching this
# image's torch/CUDA/Python/ABI (read them from the image; see git history for the
# 0.18 wheel line). Left out here to keep the upgrade minimal.
COPY entrypoint.sh /usr/local/bin/voxtral-tts-entrypoint.sh
RUN chmod +x /usr/local/bin/voxtral-tts-entrypoint.sh
ENTRYPOINT ["/usr/local/bin/voxtral-tts-entrypoint.sh"]
