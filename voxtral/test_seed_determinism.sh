#!/usr/bin/env bash
# Test what actually makes Voxtral TTS output vary, and whether it can be pinned.
#
# Background (see the Voxtral seed notes in CLAUDE.md): stage 0 hands vLLM's
# sampler logits with exactly one finite entry, so sampling params (seed,
# temperature, top_p, top_k, repetition_penalty) cannot change the output. The
# only random draw in the pipeline is the Gaussian each frame's flow-matching ODE
# starts from, taken from the GLOBAL torch RNG, which vLLM seeds from
# ModelConfig.seed (default 0) and re-seeds after warmup. So the prediction is:
#   - the request `seed` field does nothing,
#   - takes are deterministic per process, from the first request onward,
#   - repeated calls inside one process differ, because every generated frame
#     advances that one global stream,
#   - duration usually survives that variation (the semantic head is a
#     high-margin argmax) but can occasionally jump.
# This script tries to falsify all four.
#
# THE CONTROL THAT MATTERS: a comparison is only meaningful between requests at
# the same RNG position, which in practice means "first request after a restart".
# So this script restarts the service between takes and NEVER sends a warmup
# call. In particular it does not use `tts.py --check`, which would synthesize
# and thereby advance the RNG before the measurement. Readiness is polled on
# /health only.
#
# Usage:
#   ./voxtral/test_seed_determinism.sh [A|B|C|all]     (default: all)
#
#     A  Does the request `seed` field change anything?  (2 restarts)
#     B  Is output reproducible per process?             (2 restarts)
#     C  Does duration ever move within one process?     (1 restart, N calls)
#
# Every restart reloads ~9GB of weights, so `all` takes a while. Env overrides:
#   TEXT VOICE MODEL BASE SERVICE DOCKER TTS_CMD READY_TIMEOUT REPEATS OUTDIR

set -euo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$REPO_ROOT"

TEXT="${TEXT:-Le patient présente une créatinine à 120 micromoles par litre, contrôle prévu dans six semaines.}"
VOICE="${VOICE:-fr_male}"
BASE="${BASE:-http://127.0.0.1:8003}"
SERVICE="${SERVICE:-voxtral-tts}"
DOCKER="${DOCKER:-sudo docker}"
TTS_CMD="${TTS_CMD:-./tts.py}"
READY_TIMEOUT="${READY_TIMEOUT:-600}"
REPEATS="${REPEATS:-6}"
OUTDIR="${OUTDIR:-$(mktemp -d "${TMPDIR:-/tmp}/voxtral-seed-XXXXXX")}"

MODEL="${MODEL:-}"   # auto-detected from /v1/models when empty

WHICH="${1:-all}"
case "$WHICH" in
  A|B|C|all) ;;
  -h|--help) sed -n '2,40p' "${BASH_SOURCE[0]}"; exit 0 ;;
  *) echo "unknown test '$WHICH' (want A, B, C or all)" >&2; exit 2 ;;
esac

# ---------------------------------------------------------------- helpers

say()  { printf '%s\n' "$*"; }
head1() { printf '\n=== %s ===\n' "$*"; }
verdict() { printf '  >> %-4s %s\n' "$1" "$2"; }

wait_ready() {
  local deadline=$((SECONDS + READY_TIMEOUT))
  printf '  waiting for %s/health ' "$BASE"
  while (( SECONDS < deadline )); do
    if curl -fsS -o /dev/null --max-time 3 "$BASE/health" 2>/dev/null; then
      printf ' ready\n'
      return 0
    fi
    printf '.'
    sleep 3
  done
  printf '\n'
  say "  TIMEOUT after ${READY_TIMEOUT}s. Check: $DOCKER compose logs --tail 50 $SERVICE"
  return 1
}

# Restart = fresh process = the RNG is re-seeded to ModelConfig.seed. This is the
# only way to put two requests at the same RNG position.
restart_server() {
  say "  restarting $SERVICE (fresh process, RNG back to its seed)"
  $DOCKER compose restart "$SERVICE" >/dev/null
  wait_ready
}

detect_model() {
  [[ -n "$MODEL" ]] && return 0
  MODEL="$(curl -fsS --max-time 5 "$BASE/v1/models" 2>/dev/null \
    | python -c 'import json,sys; print(json.load(sys.stdin)["data"][0]["id"])' 2>/dev/null || true)"
  if [[ -z "$MODEL" ]]; then
    say "  could not read $BASE/v1/models; continuing without --model"
    return 0
  fi
  say "  model: $MODEL"
  if [[ "${MODEL,,}" != *voxtral* ]]; then
    say ""
    say "  REFUSING: :8003 is serving '$MODEL', not Voxtral. Both TTS services bind"
    say "  this port by design, so you are probably pointed at crispasr-tts (TADA)."
    say "  Start the right one:  $DOCKER compose up -d --build $SERVICE"
    exit 1
  fi
}

# One synthesis. $1 = output basename, $2... = extra tts.py flags.
synth() {
  local name="$1"; shift
  local out="$OUTDIR/$name.wav"
  local -a cmd=("$TTS_CMD" -t "$TEXT" --voice "$VOICE" --format wav -o "$out")
  [[ -n "$MODEL" ]] && cmd+=(--model "$MODEL")
  cmd+=("$@")
  # Diagnostics go to stderr: this function's stdout is captured by the caller,
  # so anything printed there would end up in the filename instead of on screen.
  if ! "${cmd[@]}" >"$OUTDIR/$name.log" 2>&1; then
    say "  synthesis FAILED for '$name'; see $OUTDIR/$name.log" >&2
    tail -5 "$OUTDIR/$name.log" | sed 's/^/    /' >&2
    exit 1
  fi
  printf '%s' "$out"
}

# sha256 (first 12 chars) of a file.
hash_of() { sha256sum "$1" | cut -c1-12; }

# Duration in seconds from the WAV header, and the implied 12.5 Hz frame count:
# frames are what stage 0 actually decided, so a duration change IS a frame change.
dur_of() {
  python - "$1" <<'PY'
import sys, wave
with wave.open(sys.argv[1]) as w:
    secs = w.getnframes() / float(w.getframerate())
print(f"{secs:8.3f}s  ~{round(secs * 12.5):5d} frames")
PY
}

# Report one take: label, hash, duration.
report() {
  local label="$1" f="$2"
  printf '  %-22s %s  %s\n' "$label" "$(hash_of "$f")" "$(dur_of "$f")"
}

# ---------------------------------------------------------------- tests

test_A() {
  head1 "TEST A: does the request \`seed\` field change anything?"
  say "  Two first-after-boot requests, identical except --seed. Same RNG position,"
  say "  so any difference is the seed's doing."

  restart_server
  local a1; a1="$(synth A_seed1 --seed 1)"
  report "seed=1  (1st after boot)" "$a1"

  restart_server
  local a2; a2="$(synth A_seed999 --seed 999)"
  report "seed=999 (1st after boot)" "$a2"

  if cmp -s "$a1" "$a2"; then
    verdict PASS "byte-identical: the request seed is INERT, as the code predicts."
    say "     (stage 0's logits have one finite entry, so the sampler's choice is forced)"
  else
    verdict FAIL "outputs differ, so the seed DOES reach something."
    say "     This contradicts fake_logits_for_audio_tokens(); re-read the sampling path"
    say "     before trusting the CLAUDE.md notes. Compare: $a1 vs $a2"
  fi
}

test_B() {
  head1 "TEST B: is output reproducible per process?"
  say "  Two calls per boot, across two boots. Position 1 vs position 1, and"
  say "  position 2 vs position 2, should each match if the global RNG is the only"
  say "  source of variation and is pinned."

  restart_server
  local b1 b2; b1="$(synth B_boot1_call1)"; b2="$(synth B_boot1_call2)"
  report "boot 1, call 1" "$b1"
  report "boot 1, call 2" "$b2"

  restart_server
  local b3 b4; b3="$(synth B_boot2_call1)"; b4="$(synth B_boot2_call2)"
  report "boot 2, call 1" "$b3"
  report "boot 2, call 2" "$b4"

  if cmp -s "$b1" "$b2"; then
    verdict INFO "the two calls within a boot are identical."
    say "     Unexpected: the RNG should have advanced. Either the noise is not"
    say "     actually consumed per frame, or something re-seeds between requests."
  else
    verdict PASS "calls differ within one process, as predicted (RNG advances)."
  fi

  if cmp -s "$b1" "$b3" && cmp -s "$b2" "$b4"; then
    verdict PASS "both positions reproduce across boots: fully deterministic per process."
    say "     So a specific take is reproducible by restarting and replaying the same"
    say "     request sequence. VOXTRAL_SEED picks a different such stream."
  elif cmp -s "$b1" "$b3"; then
    verdict INFO "position 1 reproduces but position 2 does not."
    say "     The first request is pinned; later ones drift. Suspect batching or"
    say "     kernel nondeterminism. Try VLLM_BATCH_INVARIANT=1 in the compose env."
  else
    verdict FAIL "not even the first request reproduces across boots."
    say "     The RNG is not the only source. Try VLLM_BATCH_INVARIANT=1, and confirm"
    say "     nothing else hit the server between restart and measurement."
  fi
}

test_C() {
  head1 "TEST C: does duration move within one process? ($REPEATS calls)"
  say "  Duration is set by when the semantic argmax emits end_audio. The noise"
  say "  feeds back into that (acoustic embeddings are summed into the next input),"
  say "  so duration should be mostly stable with occasional jumps."

  restart_server
  local -a hashes=() durs=()
  local i f
  for (( i = 1; i <= REPEATS; i++ )); do
    f="$(synth "C_call$i")"
    hashes+=("$(hash_of "$f")")
    durs+=("$(python -c 'import sys,wave;w=wave.open(sys.argv[1]);print("%.3f"%(w.getnframes()/float(w.getframerate())))' "$f")")
    printf '  call %-3d %s  %ss\n' "$i" "${hashes[-1]}" "${durs[-1]}"
  done

  local uniq_h uniq_d
  uniq_h="$(printf '%s\n' "${hashes[@]}" | sort -u | wc -l)"
  uniq_d="$(printf '%s\n' "${durs[@]}"   | sort -u | wc -l)"
  say "  distinct waveforms: $uniq_h/$REPEATS   distinct durations: $uniq_d/$REPEATS"

  if (( uniq_h == 1 )); then
    verdict INFO "every call identical: no per-frame noise consumption visible."
  elif (( uniq_d == 1 )); then
    verdict PASS "waveforms vary, duration does not: the argmax margin held for all"
    say "     $REPEATS calls. Raise REPEATS to hunt for the occasional jump."
  else
    verdict PASS "duration moved across calls with no seed change involved."
    say "     This is the prediction: what shifts duration is the noise draw, not the"
    say "     seed value. Durations seen: $(printf '%s ' "${durs[@]}")"
  fi
}

# ---------------------------------------------------------------- main

say "Voxtral TTS determinism probe"
say "  service : $SERVICE at $BASE"
say "  text    : ${TEXT:0:60}..."
say "  voice   : $VOICE"
say "  outputs : $OUTDIR"
say ""
say "  NOTE: no warmup call is sent, ever. A synthesis before the measurement"
say "  would advance the RNG and invalidate every comparison below."

if ! curl -fsS -o /dev/null --max-time 5 "$BASE/health" 2>/dev/null; then
  say ""
  say "  $SERVICE is not up yet. Start it first:"
  say "    $DOCKER compose up -d --build $SERVICE"
  exit 1
fi
detect_model

case "$WHICH" in
  A)   test_A ;;
  B)   test_B ;;
  C)   test_C ;;
  all) test_A; test_B; test_C ;;
esac

say ""
say "Artifacts kept in $OUTDIR (wavs + per-call tts.py logs)."
