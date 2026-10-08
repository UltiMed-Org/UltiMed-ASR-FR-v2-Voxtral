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
# 192 takes, about 45 minutes of audio. Ten sentences are ~12 s and the last two are ~30 s
# (63 and 64 words at the grid's measured 2.15 words/s), because stability over length is a
# separate failure from mispronunciation: Voxtral degenerates by looping syllables, and a
# voice can read three short sentences perfectly and lose the plot halfway through a long
# one. The long pair is sized so its runaway threshold (3 x words / rate = 89 s) still fits
# under the 90 s token cap; see corpus_medical_fr.txt.
#
# By default it only SYNTHESIZES: no whisper pass, no swap back to the ASR container,
# because judging these voices is your ears' job and the second 9 GB model load costs more
# wall clock than the decode it would run. It still auto-repairs degenerate takes, because
# the runaway test is duration-based and the tts pass records each take's duration itself.
# What SCORE=1 adds is WER, i.e. whether the right words came out, which is a different
# question from whether the take held together.
#
# Resumable: every take lands in the ledger as it completes, and the sentence list is
# minted once, so re-running after an interruption costs only the take in flight.
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
#   SENTENCES how many corpus sentences per voice (default 12, i.e. all of them; lowering
#             it cuts from the END of the corpus, which is where the two long ones are)
#   SCORE     1 to add the whisper pass and the numbers (default 0, audio only)
#   Anything driver.sh reads also passes through, e.g. EULER_STEPS.
set -euo pipefail

HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "${HERE}/../.." && pwd)"
cd "${REPO_ROOT}"

RUN="${RUN:-MED16}"
POOL="${POOL:-GRID}"
VOICES="${VOICES:-${HERE}/voices_fr_16.txt}"
CORPUS="${CORPUS:-${HERE}/corpus_medical_fr.txt}"
SENTENCES="${SENTENCES:-12}"
SCORE="${SCORE:-0}"
# Derived, never hardcoded: a literal /media/<name> in a tracked file leaks the username.
DEST="${DEST:-/media/$(id -un)/4TO_DRIVE/voxtral_grid/audition_medical}"
STAGE_DIR="${REPO_ROOT}/cache/cloning/${RUN}"

# The run owns only a sentence list and a ledger; the 480 minted candidates live in the
# GRID pool, which is the run/pool split driver.sh documents. Exported for the two python
# calls below (driver.sh gets them as sudo assignments instead, since sudo resets the env).
export CLONING_RUN="${RUN}" CLONING_POOL="${POOL}"

echo "== ${RUN}: $(grep -vc '^#\|^$' "${VOICES}") voices x ${SENTENCES} medical sentences" \
     "(10 short, 2 of ~30s) -> ${DEST}"

# 1. Synthesize. driver.sh owns the container swap, the health wait and the disk-space
# floor, so nothing here reimplements any of it; it also drops privileges to TARGET_USER
# for every python call, which is why the files it writes stay ours. PREPARE mints the
# sentence list on the first run only (driver.sh skips it once sentences.json exists, so a
# resume cannot re-roll the corpus mid-run).
#
# --curated is mandatory for this corpus, not a preference. `sentences` defaults to
# screening prose out of a Wikipedia dump, and its filters are right for that and exactly
# backwards here: it rejects any line containing a digit, so it keeps 2 of these 10 and
# the 2 it keeps are the only ones with no drug name, dose or acronym in them.
#
# INCLUDE_BASES is deliberately left at its default 1. It is not "also add the 20 shipped
# voices": `restrict(load_voices(include_bases=...), only_file)` intersects, so with 0 the
# shipped fr_female and fr_male would be absent from the candidate set and ONLY_FILE would
# quietly narrow 16 voices to 14, losing the two references the comparison rests on.
sudo RUN="${RUN}" POOL="${POOL}" DEPTH="${SENTENCES}" ONLY_FILE="${VOICES}" \
     SCORE="${SCORE}" STT_LANGUAGE=fr TARGET_USER="$(id -un)" \
     PREPARE="sentences --curated --corpus ${CORPUS} --count ${SENTENCES}" \
     "${HERE}/driver.sh"

# 2. The numbers, if there are any. driver.sh already runs `stats` itself when it scores,
# so this is only here to restrict the tables to our 16 voices.
if (( SCORE )); then
    uv run "${HERE}/sweep.py" stats --only-file "${VOICES}"
fi

# 3. audition.py wants cluster/role/voice rows rather than bare names. The roster groups by
# identity, which is what puts the two de_female candidates (h_itf_def and h_frm_def) in
# one directory pair instead of nine apart, since a hybrid is heard as its second name.
ROSTER="${STAGE_DIR}/med_roster.txt"
uv run "${HERE}/voice_distance.py" roster "${VOICES}" --out "${ROSTER}"

# 4. The listening tree.
# --sentences 0 means "every sentence in the list, in order". The default instead PICKS
# the clearest few, which is right for a 300-line FLEURS slice and wrong for a corpus of
# ten lines chosen by hand: its 12-26 word window alone would drop the 4 longest of ours,
# and its clarity ranking needs a WER per voice, which an unscored run has none of.
uv run "${HERE}/audition.py" --dest "${DEST}" --stage-dir "${STAGE_DIR}" \
    --sentences 0 --voice-list "${ROSTER}"

cat <<SUMMARY

== done. Listen here:
   ${DEST}
   00_meme_phrase/ is one sentence read by all 16, for fast A/B.
   The four voices actually on trial: h_frf_him, h_frf_ptm and h_frm_chf (the
   Canadian-sounding one) are the last three clusters, 13 to 15. The fourth,
   h_frm_def, is NOT last: the roster sorts by identity, so it sits in cluster 7
   right beside h_itf_def, which is the point (both are de_female, and that pair
   is the one choice where you are picking a variant rather than a voice).
   Each voice's directory holds all twelve sentences, in corpus order: s00 to
   s09 are dictionary, PARHAF, drug names then acronyms, and s10/s11 are the two
   ~30s ones. Listen to those two first: they are the stability test, and a voice
   that holds for 30 seconds of dense dosage text will hold for v2.
   Runaway takes were already regenerated automatically (duration-based, no ASR
   needed), so a take that still loops means the voice ran out of attempts.
SUMMARY
