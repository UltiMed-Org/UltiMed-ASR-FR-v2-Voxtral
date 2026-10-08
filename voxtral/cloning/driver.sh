#!/usr/bin/env bash
# Unattended driver for a voice sweep, on one GPU.
#
# Serves two runs, selected with RUN (see below):
#
#   RUN=A6      screen 420 candidates on 12 short sentences, to find the survivors
#   RUN=FLEURS  benchmark the survivors on the whole French FLEURS test split
#
# `fleurs.sh` is the one-command entry point for the second; this script is the engine
# under both, because the work is identical and only the run directory differs.
#
# Alternates two GPU passes and one offline one until the sweep converges:
#
#   bring voxtral-tts up (ASR down)  -> sweep.py tts     synthesize pending takes
#   bring crispasr up (voxtral down) -> sweep.py stt     transcribe and score them
#   no container needed              -> sweep.py retry   re-queue the takes that came
#                                                        out degenerate, at a new seed
#
# The retry pass is why the loop is a loop and not two passes. Voxtral degenerates on a
# small fraction of takes (it starts the sentence, then loops nonsense syllables until the
# token cap), and since the only stochasticity in the model is the noise the decoder
# starts from, another seed usually fixes it. `retry` spots those from the waveform alone
# and bumps the row's attempt number, which makes the tts pass pick it up again. It stops
# when every broken take is either repaired or out of attempts.
#
# The alternation exists because one 24 GB card cannot hold both stacks: Voxtral
# takes about 20 GB at VOXTRAL_GPU_MEM_UTIL=0.7 and whisper wants ~3 GB per replica.
# Passes are therefore chunky on purpose (a whole depth level per pass, not a row
# at a time), because every switch costs a model load of a couple of minutes.
#
# Run ONCE under sudo so it can flip containers without a password per command; the
# Python passes are dropped back to the normal user with runuser, so every wav, the
# ledger and the log stay user-owned. Run it under tmux (or nohup) so a disconnect
# does not take the sweep with it:
#
#   sudo -E ./voxtral/cloning/driver.sh
#   sudo -E RUN=FLEURS DEPTH=0 ONLY_FILE=cache/cloning/A6/keepers.txt ./voxtral/cloning/driver.sh
#
# To get the GPU back at any moment: ./voxtral/cloning/stop_all.sh (from another
# shell, no sudo needed for the flag part). It writes a STOP flag that the current
# pass honours at the next row boundary, then takes the containers down. Nothing is
# lost: work is recorded in the ledger as each take completes.
#
# Environment:
#   RUN              which run directory under cache/cloning/ to work in (default A6).
#                    Exported to the Python passes as CLONING_RUN.
#   POOL             which directory under cache/cloning/ holds the pool.json naming the
#                    voices to sweep (default A6). Exported as CLONING_POOL. A run reads
#                    a pool, it does not own one, so two runs can share a pool and one
#                    pool can be re-measured by several runs; what must NOT happen is a
#                    bigger pool appearing under a finished run's feet, which is why this
#                    is a knob and not the run directory.
#   DEPTH            sentences per voice to reach this run (default 2), 0 = every
#                    sentence in the run. Raising it and re-running deepens the sweep;
#                    rows already done are skipped.
#   TARGET_USER      user to run the Python passes as (default: $SUDO_USER)
#   INCLUDE_BASES    1 (default) sweeps the 20 shipped voices too, as controls
#   TTS_CONCURRENCY  parallel speech requests (default 4; throughput saturates at 4-8)
#   STT_CONCURRENCY  parallel transcriptions (default 6, one per whisper replica)
#   STT_LANGUAGE     ISO language hint given to the ASR (default fr). It MUST match the
#                    language of this run's sentences: whisper will happily transcribe
#                    Italian audio under a French hint and score it as catastrophic WER,
#                    which reads exactly like a degenerate take and is not one.
#   STT_SCALE        whisper replicas to run (default 6)
#   EULER_STEPS      ODE steps per frame, 0 = server default (quality/speed lever)
#   MAX_TOKENS       per-take token budget at 12.5 Hz, so 1125 (the default) is 90 s and
#                    0 means the server's own 4096 = 327.7 s. A runaway take is thrown
#                    away whatever its length, so this only decides how much GPU and
#                    disk it wastes on the way: measured on GRID_fr_fr, capping at 90 s
#                    truncates 3.5 pct of takes and saves 22.8 pct of both.
#   MIN_FREE_GB      refuse to start a pass with less than this free on the stage
#                    filesystem (default 25). sweep.py parks at its own 20 GB floor
#                    mid-pass; this is the earlier, friendlier refusal.
#   MAX_ATTEMPTS     attempts allowed per take including the first (default 5). 1 turns
#                    retrying off entirely. A7 measured what the budget buys: of the
#                    takes that started as a runaway, 63 pct were rescued by the first
#                    retry and 12 pct more by the second, so a WER is recovered for
#                    roughly three in four. The tail is why this is 5 and not 3, and
#                    the junk gate below is what keeps that tail affordable.
#   JUNK_THRESHOLD   abandon a voice once this fraction of its first attempts is junk
#                    (default 0.25). 1.0 disables the gate.
#   JUNK_AFTER       first-attempt takes a voice needs before it can be judged
#                    (default 12, i.e. 12 of a 30-sentence language).
#   CLONING_GRID_RUNS  space-separated run names this sweep belongs to, set by grid.sh.
#                    Only effect: sweep.py draws a second progress bar covering every
#                    language. Unset (a standalone run) draws the usual single bar.
#   RETRY_ON         comma-separated retry triggers (default "runaway"). Adding "wer"
#                    turns the benchmark into best-of-N: see `sweep.py retry --help`
#                    before you do, and read the results with `stats --attempt first`.
#   PREPARE          sweep.py subcommand (with args) that mints this run's sentences.json
#                    when it is missing, e.g. "fleurs". Empty (the default) means the list
#                    has to exist already, which is A6's case: its 12 screen sentences
#                    were picked by hand with `sweep.py sentences` and are not re-rolled.
#   ONLY_FILE        restrict the sweep to the voices named in this file, one per line.
#                    Screening is wide and shallow; choosing between survivors is narrow
#                    and deep, e.g. ONLY_FILE=cache/cloning/A6/contenders.txt DEPTH=12
#   HEALTH_TIMEOUT   seconds to wait for a stack to become healthy (default 900)
#   COMPOSE          path to docker-compose.yml (default: the repo's own)
#
# The container switching below is deliberately a 20-line function rather than a
# general tool. NOTE for whoever reads this next: the NeMo training repo has a
# fuller `switch_server.sh` that does the same job against this very compose file.
# This is a second implementation of that idea, kept local so this repo has no path
# into a sibling checkout. If the two ever need to agree on anything more than
# service names, delete one.
set -euo pipefail

HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "${HERE}/../.." && pwd)"
COMPOSE="${COMPOSE:-${REPO_ROOT}/docker-compose.yml}"
RUN="${RUN:-A6}"
STAGE_DIR="${REPO_ROOT}/cache/cloning/${RUN}"
STOP_FLAG="${STAGE_DIR}/STOP"

DEPTH="${DEPTH:-2}"
POOL="${POOL:-A6}"
INCLUDE_BASES="${INCLUDE_BASES:-1}"
TTS_CONCURRENCY="${TTS_CONCURRENCY:-4}"
STT_CONCURRENCY="${STT_CONCURRENCY:-6}"
STT_LANGUAGE="${STT_LANGUAGE:-fr}"
STT_SCALE="${STT_SCALE:-6}"
EULER_STEPS="${EULER_STEPS:-0}"
MAX_TOKENS="${MAX_TOKENS:-1125}"
MIN_FREE_GB="${MIN_FREE_GB:-25}"
MAX_ATTEMPTS="${MAX_ATTEMPTS:-5}"
JUNK_THRESHOLD="${JUNK_THRESHOLD:-0.25}"
JUNK_AFTER="${JUNK_AFTER:-12}"
RETRY_ON="${RETRY_ON:-runaway}"
HEALTH_TIMEOUT="${HEALTH_TIMEOUT:-900}"
# SCORE=0 produces audio and stops: no stt pass, no swap back to the ASR container, no
# retry loop. For a listening session that is the whole job, and it halves the wall clock
# because the second 9 GB model load is the expensive part, not the whisper decode.
# What it gives up is the automatic junk net: a runaway is `seconds >= factor * words /
# rate` and the tts pass records only `bytes` and `request_seconds`, never `seconds`, so
# nothing detects a degenerate take without the ASR. With SCORE=0 a looped take reaches
# your ears instead of being regenerated, which for a voice audition is informative
# anyway. Keep SCORE=1 whenever the WER numbers are the point.
SCORE="${SCORE:-1}"

if [[ "$(id -u)" -ne 0 ]]; then
    echo "driver: needs root to switch containers. Run: sudo -E $0" >&2
    exit 1
fi
TARGET_USER="${TARGET_USER:-${SUDO_USER:-}}"
if [[ -z "${TARGET_USER}" || "${TARGET_USER}" == root ]]; then
    echo "driver: set TARGET_USER to the user who owns the repo (files must not end up root-owned)" >&2
    exit 1
fi
# Derived, never hardcoded: a literal /home/<name> in a tracked file would leak the
# username into git.
TARGET_HOME="$(getent passwd "${TARGET_USER}" | cut -d: -f6)"
UV_BIN="${TARGET_HOME}/.local/bin/uv"
if [[ ! -x "${UV_BIN}" ]]; then
    UV_BIN="$(runuser -u "${TARGET_USER}" -- bash -lc 'command -v uv' || true)"
fi
[[ -x "${UV_BIN}" ]] || { echo "driver: cannot find uv for ${TARGET_USER}" >&2; exit 1; }

mkdir -p "${STAGE_DIR}/logs"
LOG="${STAGE_DIR}/logs/driver-$(date '+%Y%m%d-%H%M%S').log"
touch "${LOG}"
chown "${TARGET_USER}" "${LOG}" "${STAGE_DIR}/logs" 2>/dev/null || true
# Everything from here on is teed, so the terminal and the log show the same thing and
# a disconnected run is still readable afterwards.
exec > >(tee -a "${LOG}") 2>&1

# The ASR endpoint is key-protected and the key lives in the compose file, which is the
# single source of truth for it (no separate .env to drift).
CRISPASR_API_KEY="$(sed -n 's/.*CRISPASR_API_KEYS=\([^, ]*\).*/\1/p' "${COMPOSE}" | head -1)"
[[ -n "${CRISPASR_API_KEY}" ]] || { echo "driver: no CRISPASR_API_KEYS found in ${COMPOSE}" >&2; exit 1; }

DC=(docker compose -f "${COMPOSE}")

if [[ -e "${STOP_FLAG}" ]]; then
    echo "driver: clearing a stale STOP flag from a previous run"
    rm -f "${STOP_FLAG}"
fi

wait_healthy() {
    # $@ = container ids. Polls compose healthchecks so a pass never starts against a
    # server that is still loading 9 GB of weights.
    local deadline=$(( SECONDS + HEALTH_TIMEOUT )) id status
    for id in "$@"; do
        while :; do
            status="$(docker inspect -f '{{.State.Health.Status}}' "${id}" 2>/dev/null || echo missing)"
            [[ "${status}" == healthy ]] && break
            if (( SECONDS > deadline )); then
                echo "driver: container ${id} still '${status}' after ${HEALTH_TIMEOUT}s, giving up" >&2
                exit 69
            fi
            sleep 5
        done
    done
}

switch() {
    # $1 = tts|stt. Frees the GPU held by the other stack, then blocks until this one
    # is healthy. crispasr is replicated and has no container_name, so its health is
    # polled per replica id rather than by name.
    local want="$1"
    case "${want}" in
        tts)
            echo "--- $(date '+%F %T') switching to voxtral-tts"
            "${DC[@]}" stop crispasr crispasr-lb crispasr-tts >/dev/null 2>&1 || true
            "${DC[@]}" up -d voxtral-tts
            wait_healthy "$("${DC[@]}" ps -q voxtral-tts)"
            ;;
        stt)
            echo "--- $(date '+%F %T') switching to crispasr (x${STT_SCALE})"
            "${DC[@]}" stop voxtral-tts crispasr-tts >/dev/null 2>&1 || true
            "${DC[@]}" up -d --scale "crispasr=${STT_SCALE}" crispasr crispasr-lb
            # shellcheck disable=SC2046 # word splitting is what turns the ids into args
            wait_healthy $("${DC[@]}" ps -q crispasr)
            ;;
    esac
}

run_pass() {
    # $1 = tts|stt. Returns sweep.py's exit code (0 work done, 10 converged, 11 nothing
    # to do, 20 stopped). `nice -n 40` keeps an unattended multi-hour job from making an
    # interactive session stutter; nice takes an increment, so an oversized step just
    # pins it at the floor wherever sudo started us.
    local mode="$1" rc=0
    local common=(--sentences "${DEPTH}")
    [[ -n "${ONLY_FILE:-}" ]] && common+=(--only-file "${ONLY_FILE}")
    [[ "${INCLUDE_BASES}" == 1 ]] || common+=(--no-include-bases)
    case "${mode}" in
        tts)   common+=(--concurrency "${TTS_CONCURRENCY}" --euler-steps "${EULER_STEPS}"
                        --max-tokens "${MAX_TOKENS}"
                        --junk-threshold "${JUNK_THRESHOLD}"
                        --junk-after "${JUNK_AFTER}") ;;
        stt)   common+=(--concurrency "${STT_CONCURRENCY}" --language "${STT_LANGUAGE}") ;;
        retry) common+=(--max-attempts "${MAX_ATTEMPTS}" --on "${RETRY_ON}") ;;
    esac
    runuser -u "${TARGET_USER}" -- env HOME="${TARGET_HOME}" \
        PATH="${TARGET_HOME}/.local/bin:/usr/local/bin:/usr/bin:/bin" \
        CRISPASR_API_KEY="${CRISPASR_API_KEY}" \
        CLONING_RUN="${RUN}" CLONING_POOL="${POOL}" CLONING_GRID_RUNS="${CLONING_GRID_RUNS:-}" \
        nice -n 40 "${UV_BIN}" run "${HERE}/sweep.py" "${mode}" "${common[@]}" || rc=$?
    return "${rc}"
}

# Fail before the GPU is warmed rather than hours in. A language is about 15 GB of wav and
# ENOSPC mid-pass is the worst shape of failure available here: a truncated wav on disk, no
# ledger record for the takes that did land, and every other writer on the filesystem
# breaking at the same instant. sweep.py has its own floor for the case where the space
# runs out during a pass; this one just means nobody waits for that to happen.
mkdir -p "${STAGE_DIR}"
free_gb="$(df -BG --output=avail "${STAGE_DIR}" | tail -1 | tr -dc '0-9')"
if (( free_gb < MIN_FREE_GB )); then
    echo "driver: only ${free_gb} GB free on ${STAGE_DIR}, need ${MIN_FREE_GB}. Free space," \
         "move the run, or lower MIN_FREE_GB if you know better." >&2
    exit 1
fi
echo "=== $(date '+%F %T') ${free_gb} GB free on the stage filesystem"

DEPTH_LABEL="${DEPTH}"
[[ "${DEPTH}" -le 0 ]] && DEPTH_LABEL="all"
echo "=== $(date '+%F %T') ${RUN} driver starting: depth ${DEPTH_LABEL}, log ${LOG}"
echo "=== stop it any time with  RUN=${RUN} ${HERE}/stop_all.sh"

# A run cannot start without a sentence list, and for a fetched corpus there is no reason
# to make that a separate manual step. Never overwritten: sentence ids are half of the
# ledger key, so re-rolling them mid-run would orphan every take already recorded.
if [[ ! -f "${STAGE_DIR}/sentences.json" ]]; then
    [[ -n "${PREPARE:-}" ]] || {
        echo "driver: no ${STAGE_DIR}/sentences.json and no PREPARE command to mint one" >&2
        exit 1
    }
    echo "--- $(date '+%F %T') preparing the sentence list: sweep.py ${PREPARE}"
    # shellcheck disable=SC2086 # PREPARE carries its own arguments on purpose
    runuser -u "${TARGET_USER}" -- env HOME="${TARGET_HOME}" CLONING_RUN="${RUN}" CLONING_POOL="${POOL}" CLONING_GRID_RUNS="${CLONING_GRID_RUNS:-}" \
        "${UV_BIN}" run "${HERE}/sweep.py" ${PREPARE}
fi

while :; do
    switch tts
    rc=0; run_pass tts || rc=$?
    case "${rc}" in
        0|11) ;;
        20) echo "=== $(date '+%F %T') stopped during the tts pass, leaving the ledger as it is"; break ;;
        *)  echo "!!! tts pass exited ${rc}; aborting" >&2; exit "${rc}" ;;
    esac
    tts_dry=$(( rc == 11 ? 1 : 0 ))

    if (( ! SCORE )); then
        # Break rather than fall through: with no stt pass `stt_dry` stays 0, so the
        # "both passes ran dry" test below could never fire and the loop would spin,
        # re-running a tts pass that has nothing to do until someone killed it.
        echo "=== $(date '+%F %T') SCORE=0: audio only, skipping the stt pass and its swap"
        break
    fi

    switch stt
    rc=0; run_pass stt || rc=$?
    scored_all=0
    case "${rc}" in
        0|11) ;;
        10) scored_all=1 ;;
        20) echo "=== $(date '+%F %T') stopped during the stt pass, leaving the ledger as it is"; break ;;
        *)  echo "!!! stt pass exited ${rc}; aborting" >&2; exit "${rc}" ;;
    esac
    stt_dry=$(( rc == 11 ? 1 : 0 ))

    # Offline, no container switch: it only reads the ledger. Exit 0 means it queued work
    # for the next tts pass, 11 means every broken take is repaired or out of attempts.
    retry_rc=0; run_pass retry || retry_rc=$?
    case "${retry_rc}" in
        0)  echo "--- $(date '+%F %T') retries queued, going round again"; continue ;;
        11) ;;
        *)  echo "!!! retry pass exited ${retry_rc}; aborting" >&2; exit "${retry_rc}" ;;
    esac

    if (( scored_all )); then
        echo "=== $(date '+%F %T') converged at depth ${DEPTH_LABEL}: every take scored, "\
             "nothing left to retry."
        break
    fi

    # Both GPU passes finding nothing to do without the stt pass reporting convergence
    # means rows are stuck (a voice that 500s on every attempt, say). Stop rather than swap
    # two 9 GB model loads back and forth forever.
    if (( tts_dry && stt_dry )); then
        echo "=== $(date '+%F %T') both passes ran dry without converging; leftovers need a look" >&2
        break
    fi
done

# The report is offline, so it costs nothing to leave one behind for whoever reads the
# log in the morning.
for summary in report stats; do
    runuser -u "${TARGET_USER}" -- env HOME="${TARGET_HOME}" CLONING_RUN="${RUN}" CLONING_POOL="${POOL}" CLONING_GRID_RUNS="${CLONING_GRID_RUNS:-}" \
        "${UV_BIN}" run "${HERE}/sweep.py" "${summary}" || true
done
echo "=== $(date '+%F %T') driver done. Containers are still up; ${HERE}/stop_all.sh frees the GPU."
