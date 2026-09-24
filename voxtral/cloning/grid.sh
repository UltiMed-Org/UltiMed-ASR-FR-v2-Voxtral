#!/usr/bin/env bash
# Benchmark the whole hybrid grid on FLEURS, in every language the shipped voices speak.
#
#   sudo -E ./voxtral/cloning/grid.sh
#
# 480 candidates (380 code hybrids + 100 sparse mixes) plus the 20 shipped voices, each
# reading 30 FLEURS sentences in each of 8 languages. That is 120,000 takes and roughly
# 89 hours of GPU at the rate A7 measured (12,948 takes in 9.6 h). Run it under tmux and
# expect it to take days. Nothing about that is a problem as long as it is resumable, and
# it is: every take is recorded in its language's ledger as it completes, so stopping and
# restarting costs only the take in flight.
#
#   ./voxtral/cloning/grid.sh --list     print the plan and the state of each language
#   RUN=GRID ./voxtral/cloning/stop_all.sh   ... no: see "Stopping" below
#
# Why one run directory per language
# ----------------------------------
# A run owns a sentence list, and FLEURS ships a different one per language, so eight
# languages are eight runs (`cache/cloning/GRID_fr_fr/`, `GRID_de_de/`, ...). They share
# one POOL, because the voices are the same 500 in all eight; that split is exactly what
# CLONING_POOL exists for. Sharing a run directory instead would mean one ledger keyed by
# (voice, sid) with sentence ids colliding across languages, which is not a thing that
# can be fixed later.
#
# Stopping
# --------
# The STOP flag is per run, so stopping the grid means stopping the language currently
# in flight: `RUN=GRID_<lang> ./voxtral/cloning/stop_all.sh`, or read the language from
# the log. This script then notices the flag and stops rather than starting the next
# language, so one stop_all ends the whole grid and a re-launch picks up where it left.
#
# Arabic and Hindi
# ----------------
# whisper-large-v3-turbo scores all eight languages, which is why it is still the ASR
# here rather than the parakeet ONNX server: parakeet-tdt-0.6b-v3 covers 25 European
# languages and neither Arabic nor Hindi. If you later want the ONNX numbers for the six
# European languages, nothing here needs to change: point STT_URL at that server, since
# it speaks the same OpenAI transcription route sweep.py already uses.
#
# Environment (everything driver.sh takes, plus):
#   LANGS      space-separated `fleurs_code:iso_hint` pairs. Default is the 8 languages
#              with a native shipped voice. The FLEURS code selects the corpus, the ISO
#              hint is what the ASR is told it is hearing, and they are NOT the same
#              string (es_419 is scored as `es`), which is why both are listed.
#   LIMIT      sentences per language (default 30). Raising it later re-uses every take
#              already recorded, because `sweep.py fleurs` shuffles at a fixed seed and
#              takes a prefix, so a deeper run is a superset rather than a new draw.
#   POOL       pool directory (default GRID, minted by `derive_voices.py pool`).
#   SKIP_PREPARE  1 to refuse to fetch any corpus, failing instead on a missing one.
set -euo pipefail

HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "${HERE}/../.." && pwd)"

# FLEURS directory code : ISO-639-1 hint for the ASR. One pair per shipped voice family.
# ar and hi have no European-ASR coverage; see the header.
LANGS="${LANGS:-fr_fr:fr de_de:de es_419:es it_it:it nl_nl:nl pt_br:pt ar_eg:ar hi_in:hi}"
LIMIT="${LIMIT:-30}"

# Every run name the sweep is part of, so a single language's sweep.py can draw a second
# progress bar for the whole grid instead of only its own language. Derived from LANGS so
# the two can never disagree; a language dropped from LANGS disappears from the bar too.
CLONING_GRID_RUNS=""
for _pair in ${LANGS}; do CLONING_GRID_RUNS="${CLONING_GRID_RUNS} GRID_${_pair%%:*}"; done
export CLONING_GRID_RUNS="${CLONING_GRID_RUNS# }"
POOL="${POOL:-GRID}"

POOL_MANIFEST="${REPO_ROOT}/cache/cloning/${POOL}/pool.json"

if [[ "${1:-}" == "--list" ]]; then
    printf '%-10s %-6s %-9s %s\n' language hint takes state
    for pair in ${LANGS}; do
        lang="${pair%%:*}"; hint="${pair##*:}"
        ledger="${REPO_ROOT}/cache/cloning/GRID_${lang}/ledger.jsonl"
        if [[ -f "${ledger}" ]]; then
            takes="$(grep -c '"wav"' "${ledger}" || true)"
            state="started"
        else
            takes=0; state="not started"
        fi
        printf '%-10s %-6s %-9s %s\n' "${lang}" "${hint}" "${takes}" "${state}"
    done
    exit 0
fi

if [[ ! -f "${POOL_MANIFEST}" ]]; then
    echo "grid: no pool at ${POOL_MANIFEST}." >&2
    echo "grid: mint one first, as your normal user (it needs no GPU and no container):" >&2
    echo "grid:   CLONING_POOL=${POOL} uv run ${HERE}/derive_voices.py pool" >&2
    echo "grid:   sudo docker compose up -d --build voxtral-tts   # model init reads them once" >&2
    exit 1
fi

# Counted from the manifest rather than asserted, so the log says what was actually swept
# if the pool was minted with a different --mix or --limit than the header assumes.
candidates="$(python -c 'import json,sys; print(len(json.load(open(sys.argv[1]))))' "${POOL_MANIFEST}")"
languages="$(printf '%s\n' ${LANGS} | grep -c .)"
echo "=== $(date '+%F %T') grid: ${candidates} candidates + shipped voices"
echo "=== ${languages} languages x ${LIMIT} sentences, pool ${POOL}"

for pair in ${LANGS}; do
    lang="${pair%%:*}"
    hint="${pair##*:}"
    run="GRID_${lang}"
    stop_flag="${REPO_ROOT}/cache/cloning/${run}/STOP"

    # A flag left by a previous stop would otherwise make the driver return instantly and
    # the loop march through the remaining languages doing nothing, which reads in the log
    # exactly like a finished grid. Clear it deliberately at the start of a language and
    # treat its reappearance as the stop signal.
    rm -f "${stop_flag}"

    echo
    echo "=== $(date '+%F %T') === ${lang} (ASR hint ${hint}) ==="
    RUN="${run}" \
    POOL="${POOL}" \
    DEPTH=0 \
    INCLUDE_BASES=1 \
    STT_LANGUAGE="${hint}" \
    PREPARE="fleurs --lang ${lang} --limit ${LIMIT}" \
        "${HERE}/driver.sh"

    if [[ -f "${stop_flag}" ]]; then
        echo "=== $(date '+%F %T') stopped during ${lang}; leaving the rest of the grid for later"
        echo "=== resume with the same command: finished languages cost seconds to re-check"
        exit 0
    fi
done

echo
echo "=== $(date '+%F %T') grid complete across ${languages} languages"
echo "=== build the report with:  ./voxtral/cloning/grid_report.sh"
