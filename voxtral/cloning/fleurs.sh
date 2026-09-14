#!/usr/bin/env bash
# Benchmark the voices on the whole French FLEURS test split. One command, resumable.
#
#   sudo -E ./voxtral/cloning/fleurs.sh
#
# Every voice reads all 332 sentences of google/fleurs fr_fr test (8520 words), each take
# is transcribed by whisper-large-v3-turbo, and the per-sentence WER and CER end up in
# cache/cloning/FLEURS/: ledger.jsonl (the record), per_sentence.csv (every take, flat)
# and stats.md (per voice: min, p25, median, mean, p75, max, stdev for both rates).
#
# Run it under tmux or nohup. It takes hours: about 29 takes a minute to synthesize and
# about 70 a minute to transcribe, so roughly 11 h for 40 voices, 5 h for the 20 keepers
# alone. Stop it whenever the GPU is needed:
#
#   RUN=FLEURS ./voxtral/cloning/stop_all.sh
#
# and start it again with the same command. Nothing is repeated: a take with a wav is
# never re-synthesized and a take with a transcript is never re-transcribed, so a rerun of
# a finished benchmark costs seconds and re-running it after adding voices costs only the
# new rows. That ledger is the whole caching story, and it is the only one available:
# vLLM prefix caching cannot be turned on here because a named voice reaches the model out
# of band while its prefix occupies a run of identical placeholder token ids, so caching
# by token id would hand one voice's KV cache to another. See sweep.py's module docstring.
#
# This is a thin wrapper: driver.sh is the engine, and it does the same work for the A6
# screen. Only the run directory, the sentence source and the voice list differ.
#
# Environment (everything driver.sh reads still applies, notably TTS_CONCURRENCY,
# STT_CONCURRENCY, STT_SCALE and EULER_STEPS):
#   WITH_BASES   1 (default) benchmarks the keepers *and* the 20 shipped voices, so each
#                keeper's number can be read next to its own donor's. 0 = keepers only.
#   ONLY_FILE    an explicit voice list, one name per line, overriding WITH_BASES.
#   LIMIT        benchmark only the first N sentences of the split (0 = all 332). The
#                split is shuffled with a fixed seed first, so a prefix is a fair sample
#                rather than a handful of topics, and enlarging it later reuses every
#                take already recorded.
set -euo pipefail

HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "${HERE}/../.." && pwd)"
WITH_BASES="${WITH_BASES:-1}"
LIMIT="${LIMIT:-0}"

# The voice list comes from A6's keeper selection rather than being restated here: which
# voices are worth benchmarking is a conclusion of the screen, not a constant.
if [[ -z "${ONLY_FILE:-}" ]]; then
    if [[ "${WITH_BASES}" == 1 ]]; then
        ONLY_FILE="${REPO_ROOT}/cache/cloning/A6/keepers_and_bases.txt"
    else
        ONLY_FILE="${REPO_ROOT}/cache/cloning/A6/keepers.txt"
    fi
fi
if [[ ! -f "${ONLY_FILE}" ]]; then
    echo "fleurs: no voice list at ${ONLY_FILE}." >&2
    echo "fleurs: generate one with  uv run ${HERE}/voice_distance.py keepers --contenders 3" >&2
    exit 1
fi
echo "fleurs: benchmarking $(grep -cve '^\s*$' "${ONLY_FILE}") voices from ${ONLY_FILE}"

export RUN="${RUN:-FLEURS}"
export ONLY_FILE
export PREPARE="${PREPARE:-fleurs --limit ${LIMIT}}"
# 0 means "every sentence in the run", which is what a benchmark wants; the A6 screen uses
# a small depth instead so it can widen before it deepens.
export DEPTH="${DEPTH:-0}"
# The shipped voices have to stay in the candidate list for ONLY_FILE to be able to name
# them: driver.sh passes --no-include-bases when this is 0, and the filter is an
# intersection, so a base dropped here can never be benchmarked as a control.
export INCLUDE_BASES=1

exec "${HERE}/driver.sh"
