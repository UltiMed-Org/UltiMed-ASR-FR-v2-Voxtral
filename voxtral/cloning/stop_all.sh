#!/usr/bin/env bash
# Give the GPU back: ask a running A6 pass to park, then take every model container down.
#
#   ./voxtral/cloning/stop_all.sh          # park, then stop the containers
#   ./voxtral/cloning/stop_all.sh --park   # only park the sweep, leave containers up
#   WAIT=120 ./voxtral/cloning/stop_all.sh # wait longer for the pass to finish its chunk
#   RUN=FLEURS ./voxtral/cloning/stop_all.sh   # park the FLEURS benchmark, not the A6 screen
#
# Order matters. The STOP flag goes first so `sweep.py` finishes the requests already in
# flight and records them, instead of losing them to a container that vanished mid-call.
# Then the containers stop. Nothing is ever lost beyond the rows in flight: the ledger is
# appended and fsynced as each take completes, and `driver.sh` picks up where it stopped.
#
# Stopping (not `down`) is deliberate: `down` would also remove the network and any
# anonymous volumes, and the next run would pay for that for no reason.
set -euo pipefail

HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "${HERE}/../.." && pwd)"
COMPOSE="${COMPOSE:-${REPO_ROOT}/docker-compose.yml}"
# A STOP flag is per run, so a benchmark and a screen can be parked independently. RUN
# defaults to A6 to match driver.sh; pass RUN=FLEURS to stop a benchmark instead.
RUN="${RUN:-A6}"
STOP_FLAG="${REPO_ROOT}/cache/cloning/${RUN}/STOP"
WAIT="${WAIT:-90}"
PARK_ONLY=0
[[ "${1:-}" == --park ]] && PARK_ONLY=1

mkdir -p "$(dirname "${STOP_FLAG}")"
date '+stopped by stop_all.sh at %F %T' > "${STOP_FLAG}"
echo "stop_all: STOP flag written to ${STOP_FLAG}"

# Wait for the pass to park. Filesystem-free detection is not possible here (the pass is
# a normal process), so poll for it by name and fall through when the wait runs out: the
# containers still come down, and at worst the rows in flight are simply redone later.
waited=0
while pgrep -f 'sweep.py (tts|stt)' >/dev/null 2>&1; do
    if (( waited >= WAIT )); then
        echo "stop_all: a pass is still running after ${WAIT}s; stopping the containers anyway" >&2
        break
    fi
    (( waited % 15 == 0 )) && echo "stop_all: waiting for the current pass to park (${waited}s)"
    sleep 5
    waited=$(( waited + 5 ))
done
(( waited < WAIT )) && echo "stop_all: no sweep pass running"

if (( PARK_ONLY )); then
    echo "stop_all: --park given, leaving the containers up"
    exit 0
fi

# sudo only from here on, and only for docker. Asking for it this late means --park needs
# no password at all.
SUDO=()
[[ "$(id -u)" -ne 0 ]] && SUDO=(sudo)
"${SUDO[@]}" docker compose -f "${COMPOSE}" stop voxtral-tts crispasr crispasr-lb crispasr-tts || true

echo "stop_all: containers stopped. GPU now:"
nvidia-smi --query-gpu=memory.used,memory.total --format=csv,noheader || true
echo "stop_all: resume with  sudo -E ${HERE}/driver.sh   (delete ${STOP_FLAG} first, or the driver clears it)"
