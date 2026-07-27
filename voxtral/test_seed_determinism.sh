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
# Tests A to C try to falsify all four.
#
# THE CONTROL THAT MATTERS: a comparison is only meaningful between requests at
# the same RNG position, which in practice means "first request after a restart".
# So tests A to C restart the service between takes and NEVER send a warmup
# call. In particular they do not use `tts.py --check`, which would synthesize
# and thereby advance the RNG before the measurement. Readiness is polled on
# /health only.
#
# Test D is the odd one out: it checks the optional monkeypatch that lifts that
# whole restriction (VOXTRAL_PER_REQUEST_SEED, voxtral/patches/sitecustomize.py),
# so it needs no restart and deliberately sends traffic between its measurements.
# It skips itself when the patch is not loaded.
#
# It owns the container lifecycle: brings voxtral-tts up if it is down, stops
# crispasr-tts first if that one is holding :8003 (both TTS services bind that
# port by design), and puts everything back the way it found it on exit. Every
# docker call goes through sudo, and sudo is primed up front and kept warm, so a
# password prompt cannot land in the middle of a run.

set -euo pipefail

# Durations come out of Python as "0.500", so bash printf must read them with a
# dot. Under a comma locale (LC_NUMERIC=fr_FR here) `printf %8.3f 0.500` fails
# with "invalid number" and prints 0,000, which silently zeroed the whole
# duration column. Verdicts were unaffected (they compare secs_of output, which
# never goes through printf), but the numbers on screen were wrong.
export LC_NUMERIC=C

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$REPO_ROOT"

TEXT="${TEXT:-Le patient présente une créatinine à 120 micromoles par litre, contrôle prévu dans six semaines.}"
VOICE="${VOICE:-fr_male}"
BASE="${BASE:-http://127.0.0.1:8003}"
SERVICE="${SERVICE:-voxtral-tts}"
RIVAL="${RIVAL:-crispasr-tts}"          # the other TTS service on :8003
DOCKER="${DOCKER:-sudo docker}"         # no docker group here, so sudo always
TTS_CMD="${TTS_CMD:-./tts.py}"
READY_TIMEOUT="${READY_TIMEOUT:-1800}"  # first boot may download ~9GB of weights
REPEATS="${REPEATS:-6}"
KEEP_UP="${KEEP_UP:-0}"                 # 1 = skip teardown, leave the service up
OUTDIR="${OUTDIR:-$(mktemp -d "${TMPDIR:-/tmp}/voxtral-seed-XXXXXX")}"

MODEL="${MODEL:-}"   # auto-detected from /v1/models when empty

# Word-split TTS_CMD so multi-word launchers work too (e.g. TTS_CMD="uv run tts.py").
read -ra TTS_ARGV <<< "$TTS_CMD"

# Lifecycle bookkeeping, so cleanup can restore the state we found.
STARTED_BY_US=0
RIVAL_WAS_UP=0
SUDO_KEEPALIVE_PID=""

usage() {
  cat <<'EOF'
Usage: ./voxtral/test_seed_determinism.sh [A|B|C|D|all|up|down|doctor]

  A     Does the request `seed` field change anything?   (2 restarts)
  B     Is output reproducible per process?              (2 restarts)
  C     Does duration ever move within one process?      (1 restart, N calls)
  D     Does --noise-seed reproduce one request?         (no restart)
  all   A then B then C then D (default)
  up    Just bring voxtral-tts up and leave it running
  down  Just stop voxtral-tts
  doctor Check why per-request seeding is not active (no synthesis)

Every restart reloads ~9GB of weights, so `all` takes a while. The service is
started if needed and restored to its original state on exit (KEEP_UP=1 to keep
it running instead).

Env overrides:
  TEXT VOICE MODEL BASE SERVICE RIVAL DOCKER TTS_CMD READY_TIMEOUT REPEATS
  KEEP_UP OUTDIR
EOF
}

WHICH="${1:-all}"
case "$WHICH" in
  A|B|C|D|all|up|down|doctor) ;;
  -h|--help|help) usage; exit 0 ;;
  *) echo "unknown mode '$WHICH'" >&2; usage >&2; exit 2 ;;
esac

# ---------------------------------------------------------------- helpers

say()   { printf '%s\n' "$*"; }
head1() { printf '\n=== %s ===\n' "$*"; }
verdict() { printf '  >> %-4s %s\n' "$1" "$2"; }

# Ask for the sudo password once, at the start, then keep the timestamp alive.
# `all` does five restarts and can easily outlive sudo's default 15-minute
# timeout; a prompt appearing mid-sequence would stall the run unattended.
prime_sudo() {
  [[ "$DOCKER" == sudo\ * ]] || return 0
  say "  priming sudo (no docker group here, so every compose call needs it)"
  if ! sudo -v; then
    say "  sudo failed, cannot drive compose" >&2
    exit 1
  fi
  ( while true; do sleep 60; sudo -n true 2>/dev/null || exit 0; done ) &
  SUDO_KEEPALIVE_PID=$!
}

cleanup() {
  local rc=$?
  set +e
  if [[ -n "$SUDO_KEEPALIVE_PID" ]]; then
    kill "$SUDO_KEEPALIVE_PID" 2>/dev/null
    wait "$SUDO_KEEPALIVE_PID" 2>/dev/null
  fi
  # Succeeds only if empty, so up/down modes leave no litter but real runs keep
  # their wavs and logs.
  rmdir "$OUTDIR" 2>/dev/null
  if (( STARTED_BY_US )) || (( RIVAL_WAS_UP )); then
    if [[ "$KEEP_UP" == 1 ]]; then
      say ""
      say "KEEP_UP=1, leaving containers as they are."
      (( RIVAL_WAS_UP )) && say "  note: $RIVAL is still stopped; restore it with"
      (( RIVAL_WAS_UP )) && say "        $DOCKER compose up -d $RIVAL"
    else
      say ""
      say "restoring the container state we found:"
      if (( STARTED_BY_US )); then
        say "  stopping $SERVICE (we started it)"
        $DOCKER compose stop "$SERVICE" >/dev/null 2>&1
      fi
      if (( RIVAL_WAS_UP )); then
        say "  bringing $RIVAL back up (we stopped it for the port)"
        $DOCKER compose up -d "$RIVAL" >/dev/null 2>&1
      fi
    fi
  fi
  exit "$rc"
}

is_running() {
  [[ -n "$($DOCKER compose ps -q --status running "$1" 2>/dev/null)" ]]
}

wait_ready() {
  local start=$SECONDS elapsed
  printf '  waiting for %s/health ' "$BASE"
  while (( SECONDS - start < READY_TIMEOUT )); do
    if curl -fsS -o /dev/null --max-time 3 "$BASE/health" 2>/dev/null; then
      printf ' ready in %ds\n' $((SECONDS - start))
      return 0
    fi
    elapsed=$((SECONDS - start))
    # A dot every 3s, the elapsed seconds every 30s, so a long weight download
    # looks like progress rather than a hang.
    if (( elapsed > 0 && elapsed % 30 < 3 )); then printf '[%ds]' "$elapsed"; else printf '.'; fi
    sleep 3
  done
  printf '\n'
  say "  TIMEOUT after ${READY_TIMEOUT}s. Check:  $DOCKER compose logs --tail 50 $SERVICE" >&2
  return 1
}

# Bring the service up if needed, evicting the rival TTS service from :8003
# first. Records what it changed so cleanup can undo it.
ensure_up() {
  if is_running "$SERVICE"; then
    say "  $SERVICE is already running (leaving it up afterwards)"
  else
    if is_running "$RIVAL"; then
      say "  $RIVAL holds :8003 (both TTS services bind it by design); stopping it"
      $DOCKER compose stop "$RIVAL" >/dev/null
      RIVAL_WAS_UP=1
    fi
    say "  starting $SERVICE (first run also builds the image)"
    $DOCKER compose up -d --build "$SERVICE" >/dev/null
    STARTED_BY_US=1
  fi
  wait_ready
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
    say "  REFUSING: :8003 is serving '$MODEL', not Voxtral. Something other than" >&2
    say "  the $SERVICE container is answering on that port." >&2
    exit 1
  fi
}

# One synthesis. $1 = output basename, $2... = extra tts.py flags.
synth() {
  local name="$1"; shift
  local out="$OUTDIR/$name.wav"
  # --url from BASE, so overriding BASE moves the synthesis too and not just the
  # health poll. Without this they can silently diverge and the test measures a
  # different server than the one it restarts.
  local -a cmd=("${TTS_ARGV[@]}" --url "$BASE" -t "$TEXT" --voice "$VOICE" --format wav -o "$out")
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

hash_of() { sha256sum "$1" | cut -c1-12; }

secs_of() {
  python -c 'import sys,wave;w=wave.open(sys.argv[1]);print("%.3f"%(w.getnframes()/float(w.getframerate())))' "$1"
}

# Duration plus the implied 12.5 Hz frame count: frames are what stage 0 actually
# decided, so a duration change IS a frame count change.
dur_of() {
  local s; s="$(secs_of "$1")"
  printf '%8.3fs  ~%5d frames' "$s" "$(python -c "print(round($s * 12.5))")"
}

report() {
  local label="$1" f="$2"
  printf '  %-26s %s  %s\n' "$label" "$(hash_of "$f")" "$(dur_of "$f")"
}

# ---------------------------------------------------------------- tests

test_A() {
  head1 "TEST A: does the request \`seed\` field change anything?"
  say "  Two first-after-boot requests, identical except --seed. Same RNG position,"
  say "  so any difference is the seed's doing."

  restart_server
  local a1; a1="$(synth A_seed1 --seed 1)"
  report "seed=1   (1st after boot)" "$a1"

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
    durs+=("$(secs_of "$f")")
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

# Reading the log, not the env, on purpose: the env var only says what was asked
# for, while this line proves the patch really loaded inside the stage-0 engine
# process (the one that owns the RNG).
patch_active() {
  $DOCKER compose logs "$SERVICE" 2>/dev/null | grep -q 'voxtral-seed-patch\] active'
}

test_D() {
  head1 "TEST D: does --noise-seed reproduce one request, with no restart?"
  say "  This one tests our own monkeypatch rather than stock behaviour: it needs"
  say "  VOXTRAL_PER_REQUEST_SEED=1 on the service, which injects"
  say "  voxtral/patches/sitecustomize.py into the stage-0 engine process."

  if ! patch_active; then
    verdict SKIP "patch not loaded: no '[voxtral-seed-patch] active' line in the log."
    say "     Setting the env var is not enough on its own, so here is what the running"
    say "     container actually has:"
    $DOCKER compose exec -T "$SERVICE" env 2>/dev/null \
      | grep -E '^(VOXTRAL_PER_REQUEST_SEED|PYTHONPATH)=' | sed 's/^/       /' \
      || say "       neither VOXTRAL_PER_REQUEST_SEED nor PYTHONPATH is set in there"
    say "     Both are needed: the var is the toggle, PYTHONPATH=/opt/voxtral-patches is"
    say "     what actually imports the patch. Both come from docker-compose.yml, and"
    say "     env edits need a recreate, which \`restart\` does not do:"
    say "       $DOCKER compose up -d $SERVICE && ./voxtral/test_seed_determinism.sh D"
    say "     If they are both set and this still skips, the patch raised on import:"
    say "       $DOCKER compose logs $SERVICE | grep voxtral-seed-patch"
    return 0
  fi

  # No restart anywhere here, which is the entire claim. The unseeded call in the
  # middle deliberately advances the global RNG, so a pass means the seed pins the
  # take irrespective of RNG position, not merely that nothing moved.
  local d1 dx d2 d3
  d1="$(synth D_seed7_first --noise-seed 7)"; report "seed 7, call 1" "$d1"
  dx="$(synth D_unseeded)";                   report "no seed (RNG moves)" "$dx"
  d2="$(synth D_seed7_again --noise-seed 7)"; report "seed 7, call 2" "$d2"
  d3="$(synth D_seed8 --noise-seed 8)";       report "seed 8" "$d3"

  if cmp -s "$d1" "$d2"; then
    verdict PASS "same seed, same audio, across intervening traffic and with no restart."
  else
    verdict FAIL "the same seed gave different audio."
    say "     Check the log for '[voxtral-seed-patch] WARNING' lines: 'shared a step'"
    say "     means requests overlapped (serialize them), 'no sampling_extra_args' means"
    say "     stage 0 lost its extra_args (keep VOXTRAL_CFG_ALPHA set). $d1 vs $d2"
  fi

  if cmp -s "$d1" "$d3"; then
    verdict FAIL "seeds 7 and 8 gave identical audio, so the value is being ignored."
  else
    verdict PASS "a different seed gives a different take, so the value picks the stream."
  fi
}

# ---------------------------------------------------------------- doctor

# Check every link in the seeding chain INSIDE the running container, in
# dependency order, so the first failing step is the cause and not a symptom.
# Written because "test D skips" was indistinguishable from three different
# breakages, and none of them can be seen from the host.
doctor() {
  head1 "DOCTOR: is per-request seeding actually wired up?"
  if ! is_running "$SERVICE"; then
    say "  $SERVICE is NOT running, so nothing below can be inspected. That is also"
    say "  what a crash loop looks like from here, and 'start it' would be useless"
    say "  advice in that case, so here is the state and the tail of the log:"
    say ""
    $DOCKER compose ps "$SERVICE" 2>&1 | sed 's/^/     /' || true
    say ""
    $DOCKER compose logs --tail 25 "$SERVICE" 2>&1 | sed 's/^/     /' || true
    say ""
    say "  If it never started:  $DOCKER compose up -d $SERVICE"
    say "  If it died at boot, the tail above says why. A likely candidate after a"
    say "  length change is stage 0 failing to fit: lower VOXTRAL_MAX_MODEL_LEN or"
    say "  VOXTRAL_GPU_MEM_UTIL in docker-compose.yml."
    return 1
  fi

  say ""
  say "1. env inside the container (both lines must be there)"
  $DOCKER compose exec -T "$SERVICE" env 2>/dev/null \
    | grep -E '^(VOXTRAL_PER_REQUEST_SEED|PYTHONPATH)=' | sed 's/^/     /' \
    || { say "     NEITHER is set, yet docker-compose.yml has both: this container predates"
         say "     that edit. Recreate it (\`restart\` does NOT re-read environment:):"
         say "       $DOCKER compose up -d $SERVICE"; }

  say ""
  say "2. the patch file as the container sees it (must be readable by the runtime uid)"
  $DOCKER compose exec -T "$SERVICE" ls -l /opt/voxtral-patches/ 2>&1 | sed 's/^/     /' \
    || say "     could not list /opt/voxtral-patches (mount missing?)"

  say ""
  say "3. does python in the container pick it up as sitecustomize?"
  $DOCKER compose exec -T "$SERVICE" python3 -c '
import sys
try:
    import sitecustomize
except ModuleNotFoundError:
    print("NO: sitecustomize not importable, so PYTHONPATH does not reach python")
    raise SystemExit(1)
print("yes:", getattr(sitecustomize, "__file__", "?"))
print("import hook installed:", any(type(f).__name__ == "_Finder" for f in sys.meta_path))
' 2>&1 | sed 's/^/     /' || say "     ^ this step failed, see above"

  say ""
  say "4. does importing the model module apply the patch? (pulls in torch, ~30s)"
  local log4="$OUTDIR/doctor_import.log"
  if $DOCKER compose exec -T "$SERVICE" python3 -c '
import vllm_omni.model_executor.models.voxtral_tts.voxtral_tts as m
print("PATCHED:", getattr(m.VoxtralTTSForConditionalGeneration.make_omni_output,
                          "_voxtral_seed_patched", False))
' >"$log4" 2>&1; then
    grep -E 'voxtral-seed-patch|PATCHED:' "$log4" | sed 's/^/     /' || tail -3 "$log4" | sed 's/^/     /'
  else
    say "     the import itself failed; last lines:"
    tail -5 "$log4" | sed 's/^/     /'
  fi

  say ""
  say "5. what the running server logged (seeding, and the effective YAML knobs)"
  $DOCKER compose logs "$SERVICE" 2>/dev/null \
    | grep -E 'voxtral-seed-patch|entrypoint: patched|entrypoint: per-request' \
    | tail -8 | sed 's/^/     /' \
    || say "     nothing: no [voxtral-seed-patch] lines and no 'entrypoint: patched' line"

  say ""
  say "  How to read step 5:"
  say "    'active'           -> the patch IS on; a failing test D is then a behaviour"
  say "                          problem (concurrency, or extra_args lost), not wiring."
  say "    'armed' but no     -> this file was imported but the model module never came"
  say "    'active'              through the hook. Paste steps 3-5 and I can dig."
  say "    neither            -> PYTHONPATH never reached the server process. Step 1 says"
  say "                          whether the container even has it."
  say "    'entrypoint: patched ... max_tokens=N' is the audio-length answer: no"
  say "    max_tokens= field there means that image predates the knob (up -d --build)."
}

# ---------------------------------------------------------------- main

say "Voxtral TTS determinism probe"
say "  mode    : $WHICH"
say "  service : $SERVICE at $BASE"
[[ "$WHICH" == up || "$WHICH" == down ]] || say "  outputs : $OUTDIR"

prime_sudo
trap cleanup EXIT INT TERM

if [[ "$WHICH" == down ]]; then
  say ""
  if is_running "$SERVICE"; then
    say "stopping $SERVICE"
    $DOCKER compose stop "$SERVICE" >/dev/null
    say "done. To put the other TTS back on :8003:  $DOCKER compose up -d $RIVAL"
  else
    say "$SERVICE is not running, nothing to do"
  fi
  exit 0
fi

if [[ "$WHICH" == doctor ]]; then
  KEEP_UP=1
  doctor
  exit $?
fi

say "  text    : ${TEXT:0:60}..."
say "  voice   : $VOICE"
say ""
say "  NOTE: no warmup call is sent, ever. A synthesis before the measurement"
say "  would advance the RNG and invalidate every comparison below."
say ""
ensure_up
detect_model

if [[ "$WHICH" == up ]]; then
  KEEP_UP=1
  say ""
  say "$SERVICE is up and ready at $BASE. Left running (mode 'up')."
  say "Stop it with:  ./voxtral/test_seed_determinism.sh down"
  exit 0
fi

case "$WHICH" in
  A)   test_A ;;
  D)   test_D ;;
  B)   test_B ;;
  C)   test_C ;;
  all) test_A; test_B; test_C; test_D ;;
esac

say ""
say "Artifacts kept in $OUTDIR (wavs + per-call tts.py logs)."
