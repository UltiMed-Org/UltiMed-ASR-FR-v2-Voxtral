#!/usr/bin/env bash
# Turn the FLEURS ledger into one report: error-rate tables, then the speaker matrix.
#
#   ./voxtral/cloning/fleurs_report.sh
#
# Offline apart from the speaker embeddings, so it can be run against a benchmark that is
# still going: everything is computed from what has been scored so far, and voices are
# compared only on the sentences all of them have read. Re-run it as often as you like,
# it overwrites its own output and nothing else.
#
# Produces, in cache/cloning/FLEURS/:
#   stats.md          per voice: min, p25, median, mean, p75, max, stdev of WER and CER
#   per_sentence.csv  one row per (voice, sentence), both rates, duration, F0, transcript
#   failures.md       what broke: runaway takes, the hardest sentences, error rate by
#                     text feature, and runaways per voice
#   similarity.md     the N x N speaker-embedding cosine matrix, plus what it implies
#   distinct.md       how many DIFFERENT speakers those voices actually are, and which
#                     name to use for each
#   retries.md        whether a second seed fixed the degenerate takes, and where the
#                     ones it could not fix pile up (skipped if nothing was retried)
#   RESULTS.md        the three above assembled, with the written analysis last
#
# The analysis is prose and lives in a tracked file (fleurs_analysis.md) rather than
# being regenerated, so re-running this never destroys it. Everything else here is
# derived from the ledger and safe to throw away.
#
# It is spliced in *after* the tables, deliberately: the conclusion is only worth
# anything if the numbers it rests on have already been read, and the speaker matrix in
# particular changes how the error rates should be read (two voices that score well but
# sound the same buy no acoustic diversity at all).
#
# Environment:
#   DEVICE    where to run the ECAPA speaker embedding (default: cuda, falling back to
#             cpu on failure, which is what happens when the model containers are still
#             up and holding the card).
#   VOICES    voice list to restrict the report to (default: the benchmark's own list,
#             A6's keepers plus the shipped voices as controls).
#   MAX_TAKES takes averaged per voice for the embedding (default 16; more is steadier
#             and slower, and 4 is enough to separate speakers).
#   SPEAKER   cosine similarity at which two names count as one speaker (default 0.70).
#             Not the 0.50 the metric's operating point suggests: on this pool the pair
#             similarities are empty between 0.58 and 0.84, so any threshold in that band
#             gives the same answer, and 0.70 sits in the middle of the gap. 0.50 instead
#             chains genuinely different shipped voices together through borderline pairs.
set -euo pipefail

HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "${HERE}/../.." && pwd)"
STAGE_DIR="${REPO_ROOT}/cache/cloning/FLEURS"
VOICES="${VOICES:-${REPO_ROOT}/cache/cloning/A6/keepers_and_bases.txt}"
MAX_TAKES="${MAX_TAKES:-16}"
SPEAKER="${SPEAKER:-0.70}"
DEVICE="${DEVICE:-cuda}"

[[ -f "${STAGE_DIR}/ledger.jsonl" ]] || { echo "report: no ledger at ${STAGE_DIR}" >&2; exit 1; }
[[ -f "${VOICES}" ]] || { echo "report: no voice list at ${VOICES}" >&2; exit 1; }

export CLONING_RUN=FLEURS
# Every step is a PEP 723 script in this directory, run through uv so none of them
# needs a venv. Named rather than repeated so CLONING_RUN cannot be forgotten on one.
run() { local script="$1"; shift; uv run "${HERE}/${script}" "$@"; }

echo "=== error-rate tables"
run sweep.py stats --only-file "${VOICES}" --out "${STAGE_DIR}/stats.md" \
    --csv-out "${STAGE_DIR}/per_sentence.csv"

echo "=== failure modes"
run sweep.py diagnose --only-file "${VOICES}" --out "${STAGE_DIR}/failures.md"

echo "=== speaker embeddings (${MAX_TAKES} takes per voice, ${DEVICE})"
# The benchmark leaves its containers up, so the card is usually still full when this
# runs. Falling back rather than failing keeps the report a single command either way.
if ! run voice_distance.py embed --device "${DEVICE}" --max-takes "${MAX_TAKES}" --all; then
    echo "--- ${DEVICE} failed, retrying on cpu"
    run voice_distance.py embed --device cpu --max-takes "${MAX_TAKES}" --all
fi

echo "=== speaker similarity matrix"
run voice_distance.py matrix --only-file "${VOICES}" --out "${STAGE_DIR}/similarity.md"

echo "=== retries"
# Only meaningful once the retry rounds have run, and saying so is better than failing:
# the first pass over a fresh benchmark has nothing to report here.
run sweep.py retry-report || echo "--- nothing retried yet, skipping"

echo "=== distinct speakers (threshold ${SPEAKER})"
run voice_distance.py select --threshold "${SPEAKER}" --out "${STAGE_DIR}/distinct.md"

echo "=== assembling ${STAGE_DIR}/RESULTS.md"
# Each part is a standalone document with its own H1. Demoting every heading one level on
# the way in keeps RESULTS.md a single document with one title, without the parts having
# to know they are being assembled (they are all readable on their own too).
splice() { sed 's/^\(#\{1,5\}\) /#\1 /' "$1"; }
{
    echo "# French FLEURS benchmark: $(grep -cve '^[[:space:]]*$' "${VOICES}") voices"
    echo
    echo "Generated by \`voxtral/cloning/fleurs_report.sh\` (Claude Code) on $(date '+%F %T')."
    echo "Raw per-sentence data: \`per_sentence.csv\`. Raw record: \`ledger.jsonl\`."
    echo
    splice "${STAGE_DIR}/stats.md"
    echo
    splice "${STAGE_DIR}/failures.md"
    echo
    splice "${STAGE_DIR}/similarity.md"
    echo
    splice "${STAGE_DIR}/distinct.md"
    if [[ -f "${STAGE_DIR}/retries.md" ]]; then
        echo
        splice "${STAGE_DIR}/retries.md"
    fi
    if [[ -f "${HERE}/fleurs_analysis.md" ]]; then
        echo
        splice "${HERE}/fleurs_analysis.md"
    fi
} > "${STAGE_DIR}/RESULTS.md"

echo "=== done: ${STAGE_DIR}/RESULTS.md"
