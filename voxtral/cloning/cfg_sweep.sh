#!/usr/bin/env bash
# Read one medical sentence with every French candidate at ten values of cfg_alpha, and
# leave all 160 takes in one flat directory for listening. Written with Claude Code.
#
#   ./voxtral/cloning/cfg_sweep.sh
#
# It answers plan stage A10: does cfg_alpha change how many voices are valid? Every voice
# verdict so far (A5, A8, A9) was taken at the one deploy default 1.2, because that is how
# the grid was generated, so "valid" has only ever been measured for the pair
# (voice, cfg_alpha) and then reported as a property of the voice. cfg_alpha is the
# classifier-free-guidance scale of stage 0's flow-matching acoustic transformer, i.e. how
# hard the decoder is pushed to follow the text, which is the same axis A9's six rejects
# failed on: they lost articulation on dense dosage text, not timbre. So a reject may hold
# at 2.5, and a keeper may turn out to have no margin at all.
#
# No container swap and no restart: cfg_alpha is a per-request knob (CLAUDE.md, the
# `perso` fork's `_extract_cfg_alpha`), so the whole sweep is one `tts.py --batch` against
# the already-running voxtral-tts server. It needs the TTS container up on :8003:
#   sudo docker compose up -d --build voxtral-tts
#
# The seed is the part that is easy to get wrong. Without --noise-seed every take also
# draws a fresh Gaussian for each frame's ODE, so two takes of one (voice, cfg) pair differ
# from each other and any difference you hear across cfg values is confounded by that
# noise. Pinning it makes cfg_alpha the only thing that varies. Step 2 below verifies the
# pin actually holds on this image, because per-request seeding is a fork feature and a
# stock image ignores it silently.
#
# Environment:
#   DEST        flat output directory (default: the Nextcloud drop box, below)
#   VOICES      candidate list (default voices_fr_16.txt, i.e. A9's 16, rejects included,
#              since recovering a reject is the whole question)
#   CORPUS      corpus file (default corpus_medical_fr.txt)
#   SENTENCE    1-based index into that corpus, counting non-comment lines (default 4)
#   CFGS        the cfg_alpha grid (default below)
#   SEED        noise seed pinned on every take (default 20261013)
#   CONCURRENCY requests in flight (default 4; vllm-omni saturates around 4-8)
#   URL         server base URL (default http://127.0.0.1:8003)
set -euo pipefail

HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "${HERE}/../.." && pwd)"
cd "${REPO_ROOT}"

VOICES="${VOICES:-${HERE}/voices_fr_16.txt}"
CORPUS="${CORPUS:-${HERE}/corpus_medical_fr.txt}"
# Sentence 4 is the Synacthene line: a brand name, a dose with its unit, a route and two
# ordinals in 24 words, about 11 s. It is one of the ten highest-CER rows of the finished
# v1 run, so it is text that already broke voices, and it is short enough that 160 takes
# stay listenable (about 30 minutes of audio).
SENTENCE="${SENTENCE:-4}"
# Ten values, since the ask was ten samples per voice over 0.1 to 5.0. Not a linear ramp:
# the interesting range is narrow, so this is dense between 1.0 and 3.0 and coarse above.
# 1.0 is deliberately in the grid as the known-bad anchor (CFG off, the decoder emits
# phonetically plausible but wrong audio, which is why entrypoint.sh refuses that value at
# startup) so the ear has a floor to compare the rest against. 1.2 is the deploy default,
# i.e. the value every earlier verdict was taken at.
CFGS="${CFGS:-0.1 0.5 1.0 1.2 1.5 2.0 2.5 3.0 4.0 5.0}"
SEED="${SEED:-20261013}"
CONCURRENCY="${CONCURRENCY:-4}"
URL="${URL:-http://127.0.0.1:8003}"
# Derived, never hardcoded: a literal /media/<name> in a tracked file leaks the username.
DEST="${DEST:-/media/$(id -un)/4TO_DRIVE/exports/cfg_sweep}"
ROWS="${REPO_ROOT}/cache/cloning/CFGSWEEP/rows.jsonl"

# 1. The server has to be the TTS one. :8003 is shared with crispasr-tts (TADA), which
# would answer /health just as happily and then ignore cfg_alpha entirely, so check the
# model name rather than just liveness.
mkdir -p "$(dirname "${ROWS}")" "${DEST}"
if ! curl -sf -m 10 "${URL}/v1/models" -o "${ROWS}.models"; then
    echo "no server on ${URL}. Start it with:" >&2
    echo "  sudo docker compose up -d --build voxtral-tts" >&2
    exit 1
fi
if ! grep -q 'Voxtral-4B-TTS' "${ROWS}.models"; then
    echo "${URL} is up but is not serving Voxtral TTS (cfg_alpha would be ignored):" >&2
    cut -c1-200 "${ROWS}.models" >&2
    exit 1
fi
rm -f "${ROWS}.models"

TEXT="$(python - "${CORPUS}" "${SENTENCE}" <<'PY'
import sys
from pathlib import Path
lines = [ln.strip() for ln in Path(sys.argv[1]).read_text().splitlines()]
lines = [ln for ln in lines if ln and not ln.startswith("#")]
idx = int(sys.argv[2])
if not 1 <= idx <= len(lines):
    raise SystemExit(f"SENTENCE={idx} out of range, corpus has {len(lines)} sentences")
print(lines[idx - 1])
PY
)"
N_VOICES="$(grep -vc '^#\|^$' "${VOICES}")"
N_CFGS="$(wc -w <<<"${CFGS}")"
echo "== cfg sweep: ${N_VOICES} voices x ${N_CFGS} values of cfg_alpha = $((N_VOICES * N_CFGS)) takes"
echo "   sentence ${SENTENCE}: ${TEXT}"
echo "   -> ${DEST}"

# 2. Does the seed pin hold on this image? Two takes of identical (text, voice, cfg, seed)
# must be byte-identical. If they are not, this image is serving stock vllm_omni, which
# declares no route for voxtral_noise_seed and drops it while still answering 200 (the
# failure mode CLAUDE.md documents for the whole fork). The sweep is still worth running in
# that case, because upstream does read cfg_alpha, but every take then carries its own
# fresh ODE noise on top of the cfg difference, so only large effects survive. Warn rather
# than abort: the audio is the deliverable, the pin is a quality of the comparison.
PF="$(mktemp -d)"
trap 'rm -rf "${PF}"' EXIT
for take in a b; do
    uv run ./tts.py --url "${URL}" -t "${TEXT}" --voice fr_female --cfg-alpha 1.2 \
        --noise-seed "${SEED}" -o "${PF}/pf_${take}.wav" >/dev/null
done
if cmp -s "${PF}/pf_a.wav" "${PF}/pf_b.wav"; then
    echo "   seed check: ok, two identical requests gave identical audio (fork is serving)"
else
    cat >&2 <<'WARN'
   seed check: FAILED. Two identical seeded requests returned different audio, so
   voxtral_noise_seed is being ignored (stock vllm_omni, not the perso fork). cfg_alpha
   still applies, but each take also varies by its own ODE noise, so treat small
   differences between neighbouring cfg values as noise. To fix:
     sudo docker compose up -d --build voxtral-tts   # after voxtral/patches/README.md
WARN
fi

# 3. One JSONL, one invocation. Per-row knobs are a tts.py batch feature (see its --batch
# help): each row carries its own voice and cfg_alpha, so this runs as one concurrent
# request stream the server can batch, and it resumes by skipping files that already exist.
# Varying cfg_alpha inside a batch is sound (the model reads one per request); euler_steps
# would not be, which is why nothing here touches it.
#
# Filename <voice>__cfg<D.DD>.wav: two decimals and a single leading digit, so a plain name
# sort in any file manager groups the flat directory by voice with cfg ascending inside
# each voice, which is the order the question is asked in ("does this voice ever work?").
python - "${VOICES}" "${TEXT}" "${SEED}" "${ROWS}" ${CFGS} <<'PY'
import json
import sys
from pathlib import Path

voices_file, text, seed, rows_path, *cfgs = sys.argv[1:]
voices = [ln.strip() for ln in Path(voices_file).read_text().splitlines()]
voices = [v for v in voices if v and not v.startswith("#")]
with Path(rows_path).open("w") as f:
    for voice in voices:
        for cfg in (float(c) for c in cfgs):
            f.write(json.dumps({
                "input": text,
                "output": f"{voice}__cfg{cfg:.2f}.wav",
                "voice": voice,
                "cfg_alpha": cfg,
                "noise_seed": int(seed),
            }, ensure_ascii=False) + "\n")
print(f"{len(voices) * len(cfgs)} rows -> {rows_path}", file=sys.stderr)
PY

uv run ./tts.py --url "${URL}" --batch "${ROWS}" -o "${DEST}" \
    --concurrency "${CONCURRENCY}"

# 4. A note next to the audio, because a flat directory of 160 wavs does not say what it is
# and the answer is wanted weeks from now.
cat > "${DEST}/00_NOTES.txt" <<NOTES
cfg_alpha sweep, $(date +%Y-%m-%d). Plan stage A10.

Question: does cfg_alpha change how many voices are valid? Every earlier verdict
(the 10 go / 6 out of the medical audition) was taken at cfg_alpha 1.2 only.

${N_VOICES} voices x ${N_CFGS} values = $((N_VOICES * N_CFGS)) takes of ONE sentence:
  ${TEXT}

Files are <voice>__cfg<value>.wav, so sorting by name groups them by voice with
cfg_alpha ascending. Noise seed pinned at ${SEED}, so cfg_alpha is the only
difference between a voice's ten takes.

cfg_alpha grid: ${CFGS}
  1.00 is the known-bad anchor: it turns CFG off and the decoder stops following
       the text (plausible-sounding but wrong audio). Expect it to be broken.
  1.20 is the deploy default, i.e. what every earlier audition heard.

The six voices rejected on medical text, which is what this sweep is really
about: h_frm_itf, h_frf_hif, h_ptm_nlm, h_ptf_esf, h_frf_him, h_frm_def.
The ten in production: fr_female, fr_male, h_itf_arm, h_frf_esm, h_frm_nlf,
h_itf_def, h_frf_itm, h_frm_ptf, h_frf_ptm, h_frm_chf.

Generated by voxtral/cloning/cfg_sweep.sh (Claude Code).
NOTES

cat <<SUMMARY

== done. Listen here:
   ${DEST}
   Sort by name: each voice's ten takes sit together, cfg_alpha ascending.
   00_NOTES.txt repeats which voices were rejected and why this exists.
   Two things to listen for, not one: whether a rejected voice becomes usable at
   some value (the roster would then be short by that voice), and whether a
   production voice is only usable near 1.2 (the roster would then be fragile).
   If the answer is yes either way, the fix is free: cfg_alpha is per request, so
   v2 can send a different value per voice.
SUMMARY
