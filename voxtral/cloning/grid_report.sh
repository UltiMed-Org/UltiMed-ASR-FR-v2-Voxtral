#!/usr/bin/env bash
# Turn the grid's eight language ledgers into reports: one per language, then one across.
#
#   ./voxtral/cloning/grid_report.sh
#
# Offline apart from the speaker embeddings, so it can be run while the grid is still
# going: every table is computed from what has been scored so far, and voices are compared
# only on the sentences all of them have read. Re-run it as often as you like, it
# overwrites its own output and nothing else. A language that has not started yet is
# skipped with a line rather than failing the run.
#
# Produces, per language, in cache/cloning/GRID_<lang>/:
#   stats.md, summary.json, per_sentence.csv, failures.md, similarity.md, distinct.md,
#   retries.md, RESULTS.md      (exactly what fleurs_report.sh produces, which is why
#                                this calls it rather than restating any of it)
#
# and once, in cache/cloning/GRID/:
#   cross.md     which name to call per speaker per language, which semantic donor reads
#                which language, whether the sparse-mix tiers bought anything, and the
#                voices that hold up everywhere
#   voices.csv   one row per (voice, language): the flat table everything else is a view
#                of, and the one to ship
#
# On the similarity matrices: there is one per language because the pool was asked to
# read eight different corpora, but they are expected to agree. Identity is a property of
# the voice prefix and the text does not touch it, so eight near-identical matrices is the
# CORRECT result and not a wasted computation: it is the evidence that a voice is the same
# speaker in Hindi as in Dutch, which is exactly what someone shipping a multilingual
# dataset needs to know and cannot assume.
#
# Environment:
#   LANGS      space-separated `fleurs_code:iso_hint` pairs, same default as grid.sh.
#   POOL       pool directory naming the voices (default GRID).
#   SPEAKER    cosine at which two names count as one speaker (default 0.70).
#   DEVICE     where to run the ECAPA embedding (default cuda, falls back to cpu).
#   MAX_TAKES  takes averaged per voice for the embedding (default 16).
set -euo pipefail

HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "${HERE}/../.." && pwd)"

LANGS="${LANGS:-fr_fr:fr de_de:de es_419:es it_it:it nl_nl:nl pt_br:pt ar_eg:ar hi_in:hi}"
POOL="${POOL:-GRID}"
SPEAKER="${SPEAKER:-0.70}"
DEVICE="${DEVICE:-cuda}"
MAX_TAKES="${MAX_TAKES:-16}"

reported=0
for pair in ${LANGS}; do
    lang="${pair%%:*}"
    run="GRID_${lang}"
    if [[ ! -f "${REPO_ROOT}/cache/cloning/${run}/ledger.jsonl" ]]; then
        echo "--- ${lang}: nothing swept yet, skipping"
        continue
    fi
    echo
    echo "=== ${lang}"
    # VOICES= (empty) means the whole pool: the grid swept all of it deliberately, and a
    # keeper list would hide exactly the rows these reports exist to show. ANALYSIS and
    # PROBE are emptied because both belong to the French A7 run and splicing French
    # prose into the Hindi report would be worse than having no prose at all.
    RUN="${run}" \
    POOL="${POOL}" \
    VOICES="" \
    ANALYSIS="" \
    PROBE="" \
    TITLE="Hybrid grid on FLEURS ${lang}" \
    SPEAKER="${SPEAKER}" \
    DEVICE="${DEVICE}" \
    MAX_TAKES="${MAX_TAKES}" \
        "${HERE}/fleurs_report.sh"
    reported=$((reported + 1))
done

if (( reported == 0 )); then
    echo "grid_report: no language has been swept yet; run ./voxtral/cloning/grid.sh first" >&2
    exit 1
fi

echo
echo "=== across ${reported} language(s)"
CLONING_POOL="${POOL}" uv run "${HERE}/grid_report.py" cross --threshold "${SPEAKER}"

echo
echo "=== done: ${REPO_ROOT}/cache/cloning/GRID/cross.md"
