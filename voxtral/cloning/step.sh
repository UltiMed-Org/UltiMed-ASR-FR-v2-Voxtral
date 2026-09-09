#!/usr/bin/env bash
# Run the container-side half of the current PLAN_VOXTRALTTS_CLONING_TEST.md stage.
#
# Why this exists: the agent (Claude Code) cannot run `sudo docker` (sudo wants a
# password and the agent has no TTY), and every stage that touches the server
# needs it. So each session the agent updates the stage function below, and you
# just run:
#
#     ./voxtral/cloning/step.sh
#
# It asks for your sudo password once, does everything, and tees the whole run to
# cache/cloning/logs/<stage>-<timestamp>.log so the agent can read the result on
# its own next session instead of asking you to paste terminal output.
#
# Usage:
#     ./voxtral/cloning/step.sh            # run CURRENT_STAGE
#     ./voxtral/cloning/step.sh a1         # run a specific stage
#     ./voxtral/cloning/step.sh diag       # post-mortem after a failed boot (starts nothing)
#     ./voxtral/cloning/step.sh logs       # tail the voxtral-tts container logs
#     ./voxtral/cloning/step.sh down       # stop voxtral-tts (frees the GPU)
#
# Safe to re-run: every stage is idempotent, and nothing here deletes audio or
# model files.

set -euo pipefail

# The stage the agent last prepared. Updated as part of the session protocol.
CURRENT_STAGE="a2"

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
cd "$REPO_ROOT"

LOG_DIR="cache/cloning/logs"
OUT_DIR="cache/cloning/out"
mkdir -p "$LOG_DIR" "$OUT_DIR"

STAGE="${1:-$CURRENT_STAGE}"
LOG_FILE="$LOG_DIR/${STAGE}-$(date +%Y%m%d-%H%M%S).log"

# Everything from here on is captured. 2>&1 so the agent sees failures too.
exec > >(tee "$LOG_FILE") 2>&1

say() { printf '\n\033[1;36m== %s\033[0m\n' "$*"; }
ok() { printf '\033[1;32mOK\033[0m   %s\n' "$*"; }
warn() { printf '\033[1;33mWARN\033[0m %s\n' "$*"; }
fail() { printf '\033[1;31mFAIL\033[0m %s\n' "$*"; }

# ---------------------------------------------------------------- helpers ----

require_sudo() {
    # Prompt once up front rather than mid-run, so a long wait does not stall on
    # a password prompt hidden behind the tee.
    say "sudo access"
    sudo -v
    ok "sudo ready"
}

check_port_clash() {
    # crispasr-tts and voxtral-tts both bind host :8003 by design (see CLAUDE.md),
    # so only one TTS can be up. Starting voxtral-tts while the other holds the
    # port fails with a confusing bind error, hence this explicit check.
    if sudo docker compose ps --status running --services 2>/dev/null | grep -qx "crispasr-tts"; then
        warn "crispasr-tts is running and owns :8003; voxtral-tts cannot start alongside it"
        read -r -p "Stop crispasr-tts now? [y/N] " answer
        if [[ "$answer" =~ ^[Yy]$ ]]; then
            sudo docker compose stop crispasr-tts
            ok "crispasr-tts stopped"
        else
            fail "cannot continue with :8003 taken"
            exit 1
        fi
    fi
}

check_fork_has_local_voice_lookup() {
    # Preflight for the 2026-09-09 boot loop: stock vllm_omni hf_hub_downloads
    # every voice named in params.json, so a locally-added one 404s at model
    # init and the container restarts forever. The perso fork resolves the cache
    # first (patch 0005). Check the source the image is built from, since a
    # stale image is exactly the failure this is meant to catch.
    if ! grep -q "_resolve_voice_embedding" \
        voxtral/vllm-omni/vllm_omni/model_executor/models/voxtral_tts/voxtral_tts.py 2>/dev/null; then
        fail "voxtral/vllm-omni is missing the local-first voice lookup (patch 0005)"
        warn "locally-added voices will 404 at model init and boot-loop the container"
        warn "re-apply voxtral/patches/*.patch, see voxtral/patches/README.md"
        exit 1
    fi
    ok "fork has the local-first voice lookup"
}

start_voxtral() {
    # `up -d` and not `restart`: the container only re-reads environment: and
    # re-loads the voice embeddings at model init.
    # --build because the Dockerfile installs vllm_omni FROM the local clone and
    # COPYs entrypoint.sh: without it, a fork or entrypoint change is silently
    # served by the old image. Layers are cached, so it is nearly free when
    # nothing changed.
    say "starting voxtral-tts (building if the fork or entrypoint changed)"
    sudo docker compose up -d --build voxtral-tts
    ok "container up, waiting for the model to load (9 GB of weights, this takes minutes)"
}

wait_for_health() {
    # tts.py --check hits /health and does a real smoke synthesis, so a pass here
    # means the model actually loaded rather than just the process being alive.
    #
    # Two lessons from the 2026-09-09 run, both encoded here:
    #   - 900s was not enough. That run timed out one second after the container
    #     logged "Stage 0 engine launch started", i.e. it gave up exactly when the
    #     interesting part began. First boot pays for plugin init, model-info
    #     caching and loading ~9 GB of weights. The user reports boots taking
#     well over 900s, so the budget is 1800s (30 min).
    #   - A dead container looked identical to a slow one: the loop polled a
    #     closed port for the full deadline while nothing was running. It now
    #     checks container state every round and bails out immediately.
    local deadline=$((SECONDS + 1800))
    local last_report=0
    while ((SECONDS < deadline)); do
        if ./tts.py --check >/dev/null 2>&1; then
            ok "server healthy after ${SECONDS}s"
            return 0
        fi

        # Exited container: stop waiting, the logs are the whole story.
        if ! sudo docker compose ps --status running --services 2>/dev/null | grep -qx "voxtral-tts"; then
            fail "voxtral-tts is no longer running (exited after ${SECONDS}s of waiting)"
            dump_container_state
            return 1
        fi

        # A heartbeat every 60s with the container's own last line, so a human
        # watching can tell "loading weights" from "wedged".
        if ((SECONDS - last_report >= 60)); then
            last_report=$SECONDS
            printf '\n  %4ds  %s\n' "$SECONDS" "$(sudo docker compose logs --tail 1 voxtral-tts 2>/dev/null | tail -1 | cut -c1-160)"
        fi
        sleep 10
        printf '.'
    done
    fail "server did not become healthy within 1800s"
    dump_container_state
    return 1
}

dump_container_state() {
    # Everything needed to diagnose a failed boot, in one place, inside the log
    # file the agent reads. 400 lines because the interesting failure (CUDA OOM,
    # a missing weight, a YAML rejection) is usually a traceback, not one line.
    say "container status"
    sudo docker compose ps -a voxtral-tts || true
    say "exit code and OOM flag"
    sudo docker inspect --format 'exit={{.State.ExitCode}} oomkilled={{.State.OOMKilled}} error={{.State.Error}}' \
        "$(sudo docker compose ps -a -q voxtral-tts 2>/dev/null | head -1)" 2>/dev/null || warn "container not found"
    say "GPU"
    nvidia-smi --query-gpu=name,memory.used,memory.total --format=csv || true
    say "last 400 container log lines"
    sudo docker compose logs --tail 400 voxtral-tts || true
}

show_voice_boot_log() {
    # voxtral_tts.py:141 logs the voice list it actually loaded. This is the
    # direct proof that a snapshot edit reached the container, as opposed to the
    # server merely accepting the name.
    say "voices the container loaded"
    # Captured into a variable rather than piped into an `if`: `grep -m1` exits
    # on the first match, `docker compose logs` then dies of SIGPIPE, and
    # `set -o pipefail` reports that 141 as the pipeline status. The line got
    # printed and warned about at the same time on the 2026-09-09 run.
    local line
    line="$(sudo docker compose logs voxtral-tts 2>/dev/null | grep -m1 "Available voice embeddings" || true)"
    if [[ -n "$line" ]]; then
        echo "$line"
    else
        warn "no 'Available voice embeddings' line found in the logs"
    fi
}

# ------------------------------------------------------------------ stages ----

stage_a1() {
    # Acceptance: a registered copy of fr_female must behave exactly like
    # fr_female. With --noise-seed pinned the fork's per-frame noise is
    # reproducible, and everything else in the pipeline is deterministic, so the
    # two wavs should come out BYTE IDENTICAL. That is a far crisper test than
    # "sounds similar", and any difference means the copy or the registration is
    # not faithful.
    local text="Bonjour, ceci est un test de voix."
    local seed=7

    check_fork_has_local_voice_lookup
    check_port_clash
    start_voxtral
    wait_for_health
    show_voice_boot_log

    say "synthesizing with the shipped voice (fr_female)"
    ./tts.py -t "$text" --voice fr_female --noise-seed "$seed" -o "$OUT_DIR/a1_fr_female.wav"

    say "synthesizing with the registered copy (zz_test)"
    if ! ./tts.py -t "$text" --voice zz_test --noise-seed "$seed" -o "$OUT_DIR/a1_zz_test.wav"; then
        fail "zz_test was rejected by the server"
        warn "the container is up, so this is a lookup problem, not the boot-loop bug"
        return 1
    fi

    say "comparing"
    ls -l "$OUT_DIR"/a1_*.wav
    if cmp -s "$OUT_DIR/a1_fr_female.wav" "$OUT_DIR/a1_zz_test.wav"; then
        ok "byte identical: registration path fully verified, A1 acceptance PASSED"
    else
        warn "audio differs; not necessarily broken (listen to both), but it means"
        warn "something in the path is not bit-reproducible. Sizes above, files in $OUT_DIR"
    fi

    say "listen to both, then clean up with:"
    echo "    ./voxtral/cloning/voices_tool.py remove --name zz_test"
}

stage_diag() {
    # Read-only post-mortem: starts nothing, changes nothing. Run this after a
    # failed boot, BEFORE any `up -d`, because `docker compose logs` can only
    # reach a container that still exists. Recreating it destroys the evidence.
    dump_container_state
    say "image and compose config"
    sudo docker images voxtral-tts-local --format '{{.Repository}}:{{.Tag}} {{.Size}} created {{.CreatedSince}}' || true
    say "disk"
    df -h . | tail -2
    say "done. Nothing was started or stopped."
}

stage_a2() {
    # Derived voices. The 8 d_* variants are already built and registered on the
    # host by derive_voices.py, so this side is one boot (model init reads the
    # voice list exactly once) followed by a synthesis sweep.
    #
    # Every take uses the SAME text and the SAME noise seed, so any difference
    # between two wavs comes from the voice prefix alone. fr_female is included
    # as the anchor to compare against: it is the voice the dataset work already
    # likes, and "as natural as fr_female" is the A2 acceptance bar.
    local text="Bonjour, je m'appelle Camille et je vous propose aujourd'hui une petite histoire. Le vieux chêne du jardin a soixante-quinze ans."
    local seed=7
    local wav_dir="cache/cloning/A2/wav"
    mkdir -p "$wav_dir"

    # Cross-lingual costs nothing and needs no registration: these are shipped
    # voices, just driven with French text. Accented French is legitimate
    # diversity for a dataset.
    local voices=(
        fr_female
        d_frf_head40 d_frf_head60 d_frf_tail60
        d_frf_itf_50 d_frf_esf_50 d_frf_frm_50 d_frf_frm_25
        d_frf_itf_cat
        es_male it_female pt_male nl_male
    )

    check_fork_has_local_voice_lookup
    check_port_clash
    start_voxtral
    wait_for_health
    show_voice_boot_log

    say "synthesizing ${#voices[@]} takes (same text, same seed $seed)"
    local failed=()
    for v in "${voices[@]}"; do
        printf '\n--- %s\n' "$v"
        if ! ./tts.py -t "$text" --voice "$v" --noise-seed "$seed" -o "$wav_dir/$v.wav"; then
            failed+=("$v")
            fail "$v was rejected"
        fi
    done

    if ((${#failed[@]})); then
        fail "${#failed[@]} voice(s) rejected: ${failed[*]}"
        warn "a d_* rejection means the boot did not pick up the registration"
    fi

    say "results"
    ls -l "$wav_dir"
    # A blank notes file so the listening pass has somewhere to land. Written
    # only once: re-running the stage must not wipe judgements already made.
    local notes="cache/cloning/A2/notes.md"
    if [[ ! -f "$notes" ]]; then
        {
            echo "# A2 listening notes"
            echo
            echo "Same text, same noise seed, one wav per voice in wav/."
            echo "Judge each against fr_female: natural / robotic / unstable, and"
            echo "distinct-from-parents or not. Acceptance: 4+ keepers."
            echo
            echo "| voice | natural? | distinct? | keep | note |"
            echo "|---|---|---|---|---|"
            for v in "${voices[@]}"; do echo "| $v | | | | |"; done
        } > "$notes"
        ok "listening template written to $notes"
    else
        ok "notes already exist at $notes, left untouched"
    fi

    say "listen through $wav_dir, fill in $notes, then tell the agent"
    echo "    to drop the losers:  ./voxtral/cloning/derive_voices.py clean"
}

# ------------------------------------------------------------------- main ----

case "$STAGE" in
    logs)
        exec sudo docker compose logs -f voxtral-tts
        ;;
    down)
        require_sudo
        sudo docker compose stop voxtral-tts
        ok "voxtral-tts stopped, GPU freed"
        ;;
    diag | a1 | a2)
        require_sudo
        say "stage $STAGE (log: $LOG_FILE)"
        "stage_$STAGE"
        say "done. Log saved to $LOG_FILE"
        echo "The agent reads that file next session; no need to paste anything."
        ;;
    *)
        fail "unknown stage '$STAGE' (expected: diag, a1, a2, logs, down)"
        exit 1
        ;;
esac
