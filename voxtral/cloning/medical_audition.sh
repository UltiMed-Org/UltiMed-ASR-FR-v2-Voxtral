#!/usr/bin/env bash
# Speak ten hard medical sentences with each of the 16 French candidates, score them, and
# leave a listening tree in the drop box. Written with Claude Code.
#
#   ./voxtral/cloning/medical_audition.sh
#
# It answers one question: do the four marginal voices join the twelve cleared ones for
# UltiMed v2? The twelve were cleared on FLEURS sentences, which is the wrong text for
# that decision, because v2 speaks drug brand names, acronym strings and suture gauges. A
# voice that reads encyclopedia prose cleanly can still derail on "Epidyolex solution
# buvable" or "Monocryl 4-0", so the corpus here is the ten highest-CER rows of the
# finished v1 run (see corpus_medical_fr.txt).
#
# 160 takes, about 40 minutes of audio, so roughly 10 minutes of GPU plus two container
# swaps. Resumable: every take lands in the ledger as it completes, and the sentence list
# is minted once, so re-running after an interruption costs only the take in flight.
#
# Do NOT run the whole script under sudo. Only the container swap needs root, and that is
# the one line that asks for it; running the rest as root would leave root-owned wavs in
# cache/ and in the drop box. You will be prompted for your password at step 1. No API key
# is needed either: driver.sh reads CRISPASR_API_KEYS straight out of docker-compose.yml.
#
# Environment:
#   DEST      where the listening tree goes (default below: the external drive, since the
#             repo filesystem is the one driver.sh needs 25 GB of headroom on)
#   RUN       stage directory name under cache/cloning (default MED16)
#   VOICES    the candidate list (default voices_fr_16.txt)
#   SENTENCES how many corpus sentences per voice (default 10, i.e. all of them)
#   Anything driver.sh reads also passes through, e.g. STT_SCALE or EULER_STEPS.
set -euo pipefail

HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "${HERE}/../.." && pwd)"
cd "${REPO_ROOT}"

RUN="${RUN:-MED16}"
POOL="${POOL:-GRID}"
VOICES="${VOICES:-${HERE}/voices_fr_16.txt}"
CORPUS="${CORPUS:-${HERE}/corpus_medical_fr.txt}"
SENTENCES="${SENTENCES:-10}"
# Derived, never hardcoded: a literal /media/<name> in a tracked file leaks the username.
DEST="${DEST:-/media/$(id -un)/4TO_DRIVE/voxtral_grid/audition_medical}"
STAGE_DIR="${REPO_ROOT}/cache/cloning/${RUN}"

# The run owns only a sentence list and a ledger; the 480 minted candidates live in the
# GRID pool, which is the run/pool split driver.sh documents. Exported for the two python
# calls below (driver.sh gets them as sudo assignments instead, since sudo resets the env).
export CLONING_RUN="${RUN}" CLONING_POOL="${POOL}"

echo "== ${RUN}: $(grep -vc '^#\|^$' "${VOICES}") voices x ${SENTENCES} medical sentences -> ${DEST}"

# 1. Synthesize and score. driver.sh owns the container swaps, the health waits, the retry
# budget, the junk gate and the disk-space floor, so nothing here reimplements any of it;
# it also drops privileges to TARGET_USER for every python call, which is why the files it
# writes stay ours. PREPARE mints the sentence list on the first run only (driver.sh skips
# it once sentences.json exists, so a resume cannot re-roll the corpus mid-run).
#
# --max-words 40 is not decoration: the default is 18 and seven of the ten sentences are
# longer than that, so the default would silently audition the three shortest.
#
# INCLUDE_BASES is deliberately left at its default 1. It is not "also add the 20 shipped
# voices": `restrict(load_voices(include_bases=...), only_file)` intersects, so with 0 the
# shipped fr_female and fr_male would be absent from the candidate set and ONLY_FILE would
# quietly narrow 16 voices to 14, losing the two references the comparison rests on.
sudo RUN="${RUN}" POOL="${POOL}" DEPTH="${SENTENCES}" ONLY_FILE="${VOICES}" \
     STT_LANGUAGE=fr TARGET_USER="$(id -un)" \
     PREPARE="sentences --corpus ${CORPUS} --count ${SENTENCES} --min-words 8 --max-words 40" \
     "${HERE}/driver.sh"

# 2. What the numbers say, before any listening. A voice that is junk on this corpus is
# junk on the text v2 is made of, whatever it sounded like on FLEURS. This also writes the
# per_sentence.csv that step 4 reads to pick each voice's worst takes, so it is a
# dependency, not a report: skipping it makes audition.py fall back to arbitrary sentences.
uv run "${HERE}/sweep.py" stats --only-file "${VOICES}"

# 3. audition.py wants cluster/role/voice rows rather than bare names. The roster groups by
# identity, which is what puts the two de_female candidates (h_itf_def and h_frm_def) in
# one directory pair instead of nine apart, since a hybrid is heard as its second name.
ROSTER="${STAGE_DIR}/med_roster.txt"
uv run "${HERE}/voice_distance.py" roster "${VOICES}" --out "${ROSTER}"

# 4. The listening tree.
uv run "${HERE}/audition.py" --dest "${DEST}" --stage-dir "${STAGE_DIR}" \
    --sentences "${SENTENCES}" --voice-list "${ROSTER}"

cat <<SUMMARY

== done. Listen here:
   ${DEST}
   00_meme_phrase/ is one sentence read by all 16, for fast A/B.
   The four voices actually on trial: h_frf_him, h_frf_ptm and h_frm_chf (the
   Canadian-sounding one) are the last three clusters, 13 to 15. The fourth,
   h_frm_def, is NOT last: the roster sorts by identity, so it sits in cluster 7
   right beside h_itf_def, which is the point (both are de_female, and that pair
   is the one choice where you are picking a variant rather than a voice).
   Numbers: ${STAGE_DIR}/stats.md
SUMMARY
