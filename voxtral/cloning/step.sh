#!/usr/bin/env bash
# Run the container-side half of the current PLAN_VOXTRALTTS_CLONING_TEST.md stage.
#
# Why this exists: the agent (Claude Code) cannot run `sudo docker` (sudo wants a
# password and the agent has no TTY), and every stage that touches the server
# needs it. So each session the agent updates the stage function below, and you
# just run:
#
#     ./voxtral/cloning/step.sh
#
# It asks for your sudo password once, does everything, and tees the whole run to
# cache/cloning/logs/<stage>-<timestamp>.log so the agent can read the result on
# its own next session instead of asking you to paste terminal output.
#
# Usage:
#     ./voxtral/cloning/step.sh            # run CURRENT_STAGE
#     ./voxtral/cloning/step.sh a1         # run a specific stage
#     ./voxtral/cloning/step.sh b7         # mint training codes (the current stage)
#     ./voxtral/cloning/step.sh diag       # post-mortem after a failed boot (starts nothing)
#     ./voxtral/cloning/step.sh logs       # tail the voxtral-tts container logs
#     ./voxtral/cloning/step.sh down       # stop voxtral-tts (frees the GPU)
#
# Safe to re-run: every stage is idempotent, and nothing here deletes audio or
# model files.

set -euo pipefail

# The stage the agent last prepared. Updated as part of the session protocol.
CURRENT_STAGE="a5"

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
cd "$REPO_ROOT"

LOG_DIR="cache/cloning/logs"
OUT_DIR="cache/cloning/out"
mkdir -p "$LOG_DIR" "$OUT_DIR"

STAGE="${1:-$CURRENT_STAGE}"
LOG_FILE="$LOG_DIR/${STAGE}-$(date +%Y%m%d-%H%M%S).log"

# Everything from here on is captured. 2>&1 so the agent sees failures too.
exec > >(tee "$LOG_FILE") 2>&1

say() { printf '\n\033[1;36m== %s\033[0m\n' "$*"; }
ok() { printf '\033[1;32mOK\033[0m   %s\n' "$*"; }
warn() { printf '\033[1;33mWARN\033[0m %s\n' "$*"; }
fail() { printf '\033[1;31mFAIL\033[0m %s\n' "$*"; }

# ---------------------------------------------------------------- helpers ----

require_sudo() {
    # Prompt once up front rather than mid-run, so a long wait does not stall on
    # a password prompt hidden behind the tee.
    say "sudo access"
    sudo -v

    # Then KEEP it alive. Prompting up front is not enough on its own: sudo's
    # credential timestamp expires after ~15 minutes, and B7 spends 45 of them
    # generating before it needs sudo again to stop the container. The second
    # prompt then lands with its output swallowed by the `exec > >(tee ...)`
    # above, so the pane just sits there with no prompt, looking wedged, holding
    # the GPU. That happened on the 2026-09-10 run. Refresh every 60s instead,
    # and stop when this script does.
    while true; do
        sudo -n true 2>/dev/null || true
        sleep 60
        kill -0 "$$" 2>/dev/null || exit
    done &
    SUDO_KEEPALIVE=$!
    trap 'kill "$SUDO_KEEPALIVE" 2>/dev/null || true' EXIT
    ok "sudo ready (refreshed every 60s for the life of this run)"
}

check_port_clash() {
    # crispasr-tts and voxtral-tts both bind host :8003 by design (see CLAUDE.md),
    # so only one TTS can be up. Starting voxtral-tts while the other holds the
    # port fails with a confusing bind error, hence this explicit check.
    if sudo docker compose ps --status running --services 2>/dev/null | grep -qx "crispasr-tts"; then
        warn "crispasr-tts is running and owns :8003; voxtral-tts cannot start alongside it"
        read -r -p "Stop crispasr-tts now? [y/N] " answer
        if [[ "$answer" =~ ^[Yy]$ ]]; then
            sudo docker compose stop crispasr-tts
            ok "crispasr-tts stopped"
        else
            fail "cannot continue with :8003 taken"
            exit 1
        fi
    fi
}

check_fork_has_local_voice_lookup() {
    # Preflight for the 2026-09-09 boot loop: stock vllm_omni hf_hub_downloads
    # every voice named in params.json, so a locally-added one 404s at model
    # init and the container restarts forever. The perso fork resolves the cache
    # first (patch 0005). Check the source the image is built from, since a
    # stale image is exactly the failure this is meant to catch.
    if ! grep -q "_resolve_voice_embedding" \
        voxtral/vllm-omni/vllm_omni/model_executor/models/voxtral_tts/voxtral_tts.py 2>/dev/null; then
        fail "voxtral/vllm-omni is missing the local-first voice lookup (patch 0005)"
        warn "locally-added voices will 404 at model init and boot-loop the container"
        warn "re-apply voxtral/patches/*.patch, see voxtral/patches/README.md"
        exit 1
    fi
    ok "fork has the local-first voice lookup"
}

check_fork_can_dump_codes() {
    # Preflight for B7. Patch 0007 adds the per-request stage-0 code dump, and an
    # image built before it fails in the worst possible way: vllm-omni merges any
    # unknown extra_params key without complaint, so every request returns 200
    # with audio and writes no codes at all. An hour of generation would look
    # perfect and produce nothing. Check the source the image is built from.
    if ! grep -q "_dump_frame_codes" \
        voxtral/vllm-omni/vllm_omni/model_executor/models/voxtral_tts/voxtral_tts.py 2>/dev/null; then
        fail "voxtral/vllm-omni is missing the stage-0 code dump (patch 0007)"
        warn "requests would return 200 and silently write nothing"
        warn "re-apply voxtral/patches/*.patch, see voxtral/patches/README.md"
        exit 1
    fi
    ok "fork has the stage-0 code dump"
}

start_voxtral() {
    # `up -d` and not `restart`: the container only re-reads environment: and
    # re-loads the voice embeddings at model init.
    #
    # --build because the Dockerfile installs vllm_omni FROM the local clone and
    # COPYs entrypoint.sh: without it, a fork or entrypoint change is silently
    # served by the old image. Layers are cached, so it is nearly free when
    # nothing changed.
    #
    # --force-recreate because a newly registered voice is a change to files in
    # the MOUNTED cache, not to the compose config or the image. Plain `up -d`
    # compares config and image only, finds them identical, prints "Running" and
    # leaves the old process alive, which still has the old voice list from its
    # own model init. Every stage here registers voices, so the recreate is
    # never wasted, and a stale container would fail the stage in a way that
    # looks exactly like a broken registration.
    say "starting voxtral-tts (rebuild if needed, always recreate so voices reload)"
    sudo docker compose up -d --build --force-recreate voxtral-tts
    ok "container up, waiting for the model to load (9 GB of weights, this takes minutes)"
}

wait_for_health() {
    # tts.py --check hits /health and does a real smoke synthesis, so a pass here
    # means the model actually loaded rather than just the process being alive.
    #
    # Two lessons from the 2026-09-09 run, both encoded here:
    #   - 900s was not enough. That run timed out one second after the container
    #     logged "Stage 0 engine launch started", i.e. it gave up exactly when the
    #     interesting part began. First boot pays for plugin init, model-info
    #     caching and loading ~9 GB of weights. The user reports boots taking
#     well over 900s, so the budget is 1800s (30 min).
    #   - A dead container looked identical to a slow one: the loop polled a
    #     closed port for the full deadline while nothing was running. It now
    #     checks container state every round and bails out immediately.
    local deadline=$((SECONDS + 1800))
    local last_report=0
    while ((SECONDS < deadline)); do
        if ./tts.py --check >/dev/null 2>&1; then
            ok "server healthy after ${SECONDS}s"
            return 0
        fi

        # Exited container: stop waiting, the logs are the whole story.
        if ! sudo docker compose ps --status running --services 2>/dev/null | grep -qx "voxtral-tts"; then
            fail "voxtral-tts is no longer running (exited after ${SECONDS}s of waiting)"
            dump_container_state
            return 1
        fi

        # A heartbeat every 60s with the container's own last line, so a human
        # watching can tell "loading weights" from "wedged".
        if ((SECONDS - last_report >= 60)); then
            last_report=$SECONDS
            printf '\n  %4ds  %s\n' "$SECONDS" "$(sudo docker compose logs --tail 1 voxtral-tts 2>/dev/null | tail -1 | cut -c1-160)"
        fi
        sleep 10
        printf '.'
    done
    fail "server did not become healthy within 1800s"
    dump_container_state
    return 1
}

dump_container_state() {
    # Everything needed to diagnose a failed boot, in one place, inside the log
    # file the agent reads. 400 lines because the interesting failure (CUDA OOM,
    # a missing weight, a YAML rejection) is usually a traceback, not one line.
    say "container status"
    sudo docker compose ps -a voxtral-tts || true
    say "exit code and OOM flag"
    sudo docker inspect --format 'exit={{.State.ExitCode}} oomkilled={{.State.OOMKilled}} error={{.State.Error}}' \
        "$(sudo docker compose ps -a -q voxtral-tts 2>/dev/null | head -1)" 2>/dev/null || warn "container not found"
    say "GPU"
    nvidia-smi --query-gpu=name,memory.used,memory.total --format=csv || true
    say "last 400 container log lines"
    sudo docker compose logs --tail 400 voxtral-tts || true
}

show_voice_boot_log() {
    # voxtral_tts.py:141 logs the voice list it actually loaded. This is the
    # direct proof that a snapshot edit reached the container, as opposed to the
    # server merely accepting the name.
    say "voices the container loaded"
    # Captured into a variable rather than piped into an `if`: `grep -m1` exits
    # on the first match, `docker compose logs` then dies of SIGPIPE, and
    # `set -o pipefail` reports that 141 as the pipeline status. The line got
    # printed and warned about at the same time on the 2026-09-09 run.
    local line
    line="$(sudo docker compose logs voxtral-tts 2>/dev/null | grep -m1 "Available voice embeddings" || true)"
    if [[ -n "$line" ]]; then
        echo "$line"
    else
        warn "no 'Available voice embeddings' line found in the logs"
    fi
}

# ------------------------------------------------------------------ stages ----

stage_a1() {
    # Acceptance: a registered copy of fr_female must behave exactly like
    # fr_female. With --noise-seed pinned the fork's per-frame noise is
    # reproducible, and everything else in the pipeline is deterministic, so the
    # two wavs should come out BYTE IDENTICAL. That is a far crisper test than
    # "sounds similar", and any difference means the copy or the registration is
    # not faithful.
    local text="Bonjour, ceci est un test de voix."
    local seed=7

    check_fork_has_local_voice_lookup
    check_port_clash
    start_voxtral
    wait_for_health
    show_voice_boot_log

    say "synthesizing with the shipped voice (fr_female)"
    ./tts.py -t "$text" --voice fr_female --noise-seed "$seed" -o "$OUT_DIR/a1_fr_female.wav"

    say "synthesizing with the registered copy (zz_test)"
    if ! ./tts.py -t "$text" --voice zz_test --noise-seed "$seed" -o "$OUT_DIR/a1_zz_test.wav"; then
        fail "zz_test was rejected by the server"
        warn "the container is up, so this is a lookup problem, not the boot-loop bug"
        return 1
    fi

    say "comparing"
    ls -l "$OUT_DIR"/a1_*.wav
    if cmp -s "$OUT_DIR/a1_fr_female.wav" "$OUT_DIR/a1_zz_test.wav"; then
        ok "byte identical: registration path fully verified, A1 acceptance PASSED"
    else
        warn "audio differs; not necessarily broken (listen to both), but it means"
        warn "something in the path is not bit-reproducible. Sizes above, files in $OUT_DIR"
    fi

    say "listen to both, then clean up with:"
    echo "    ./voxtral/cloning/voices_tool.py remove --name zz_test"
}

stage_a3() {
    # Voice cloning from a real clip, via Voxtral's own ref_audio path.
    #
    # This is NOT the derived-voice machinery: nothing is registered, no .pt is
    # written, params.json is untouched. The clip rides along with each request
    # and vllm-omni hands the decoded waveform to the stage-0 LLM as multimodal
    # input, so the missing codec encoder (the thing Track B was designed to work
    # around) is not on this path at all.
    #
    # The rebuild is required once: cloning needs soxr, mistral_common's
    # resampler, which the base image does not ship because preset voices never
    # resample anything. Without it the request 400s with "`soxr` is not
    # installed" and the server stays perfectly healthy, which is why this failed
    # against a running server rather than at boot.
    local text="Bonjour, je vous propose aujourd'hui une petite histoire. Le vieux chêne du jardin a soixante-quinze ans."
    local seed=7
    local wav_dir="cache/cloning/A3/wav"
    mkdir -p "$wav_dir"

    # Both lengths of both speakers: the short clips are the target, the long
    # ones test whether more reference audio clones better. Same text and seed
    # throughout, so clip length is the only variable within a speaker.
    local -a clips=(
        "olivier2:voices/olivier 2.ogg"
        "olivier_long:voices/olivier.ogg"
        "alicia2:voices/alicia2.ogg"
        "alicia_long:voices/alicia.ogg"
    )

    check_fork_has_local_voice_lookup
    check_port_clash
    start_voxtral
    wait_for_health

    say "anchor take with a preset voice (proves the server is fine if cloning is not)"
    ./tts.py -t "$text" --voice fr_female --noise-seed "$seed" -o "$wav_dir/_anchor_fr_female.wav"

    local failed=()
    for entry in "${clips[@]}"; do
        local label="${entry%%:*}"
        local clip="${entry#*:}"
        printf '\n--- %s  (%s)\n' "$label" "$clip"
        if [[ ! -f "$clip" ]]; then
            warn "missing clip: $clip"
            failed+=("$label")
            continue
        fi
        if ! ./tts.py -t "$text" --ref-audio "$clip" --noise-seed "$seed" \
             -o "$wav_dir/$label.wav"; then
            failed+=("$label")
            fail "$label was rejected"
        fi
    done

    if ((${#failed[@]})); then
        fail "${#failed[@]} clone(s) failed: ${failed[*]}"
        warn "a 'soxr is not installed' 400 means the image did not actually rebuild"
        # A request that kills EngineCore leaves its traceback ONLY in the
        # container log, and the agent cannot run docker. Dumping it here is the
        # difference between diagnosing this next session and asking for another
        # run. Costs nothing on the success path, which never reaches this.
        warn "capturing the container state so the traceback lands in this log"
        dump_container_state
    fi

    say "results"
    ls -l "$wav_dir"
    ./voxtral/cloning/wav_stats.py "$wav_dir" --ref _anchor_fr_female || true

    say "listen: does <name>.wav sound like the person in its source clip?"
    echo "    source clips are in voices/, takes in $wav_dir"
    echo "    if yes, Track B (decoder inversion) is unnecessary and A3 closes the project's core goal"
}

stage_diag() {
    # Read-only post-mortem: starts nothing, changes nothing. Run this after a
    # failed boot, BEFORE any `up -d`, because `docker compose logs` can only
    # reach a container that still exists. Recreating it destroys the evidence.
    dump_container_state
    say "image and compose config"
    sudo docker images voxtral-tts-local --format '{{.Repository}}:{{.Tag}} {{.Size}} created {{.CreatedSince}}' || true
    say "disk"
    df -h . | tail -2
    say "done. Nothing was started or stopped."
}

stage_a2() {
    # Derived voices. The 8 d_* variants are already built and registered on the
    # host by derive_voices.py, so this side is one boot (model init reads the
    # voice list exactly once) followed by a synthesis sweep.
    #
    # Every take uses the SAME text and the SAME noise seed, so any difference
    # between two wavs comes from the voice prefix alone. fr_female is included
    # as the anchor to compare against: it is the voice the dataset work already
    # likes, and "as natural as fr_female" is the A2 acceptance bar.
    local text="Bonjour, je m'appelle Camille et je vous propose aujourd'hui une petite histoire. Le vieux chêne du jardin a soixante-quinze ans."
    local seed=7
    local wav_dir="cache/cloning/A2/wav"
    mkdir -p "$wav_dir"

    # Cross-lingual costs nothing and needs no registration: these are shipped
    # voices, just driven with French text. Accented French is legitimate
    # diversity for a dataset.
    local voices=(
        fr_female
        d_frf_head40 d_frf_head60 d_frf_tail60
        d_frf_itf_50 d_frf_esf_50 d_frf_frm_50 d_frf_frm_25
        d_frf_itf_cat
        es_male it_female pt_male nl_male
    )

    check_fork_has_local_voice_lookup
    check_port_clash
    start_voxtral
    wait_for_health
    show_voice_boot_log

    say "synthesizing ${#voices[@]} takes (same text, same seed $seed)"
    local failed=()
    for v in "${voices[@]}"; do
        printf '\n--- %s\n' "$v"
        if ! ./tts.py -t "$text" --voice "$v" --noise-seed "$seed" -o "$wav_dir/$v.wav"; then
            failed+=("$v")
            fail "$v was rejected"
        fi
    done

    if ((${#failed[@]})); then
        fail "${#failed[@]} voice(s) rejected: ${failed[*]}"
        warn "a d_* rejection means the boot did not pick up the registration"
    fi

    say "results"
    ls -l "$wav_dir"
    # A blank notes file so the listening pass has somewhere to land. Written
    # only once: re-running the stage must not wipe judgements already made.
    local notes="cache/cloning/A2/notes.md"
    if [[ ! -f "$notes" ]]; then
        {
            echo "# A2 listening notes"
            echo
            echo "Same text, same noise seed, one wav per voice in wav/."
            echo "Judge each against fr_female: natural / robotic / unstable, and"
            echo "distinct-from-parents or not. Acceptance: 4+ keepers."
            echo
            echo "| voice | natural? | distinct? | keep | note |"
            echo "|---|---|---|---|---|"
            for v in "${voices[@]}"; do echo "| $v | | | | |"; done
        } > "$notes"
        ok "listening template written to $notes"
    else
        ok "notes already exist at $notes, left untouched"
    fi

    say "listen through $wav_dir, fill in $notes, then tell the agent"
    echo "    to drop the losers:  ./voxtral/cloning/derive_voices.py clean"
}

stage_a5() {
    # How many usable, DISTINCT French voices do the 20 shipped ones yield?
    #
    # Cost: GPU, exclusively, in two phases that never overlap. Phase 1 boots
    # voxtral-tts (~17 GB, first boot up to 30 min, warm boot ~2 min) and
    # synthesizes 40 takes, roughly 8 min. Phase 2 stops it and boots the ASR
    # stack instead, so the two never contend for the card, then transcribes
    # those 40 takes, roughly 3 min. Total 15 to 45 min depending on boot.
    # Disk: negligible, ~15 MB of wav. Network: none (both images are local).
    #
    # The sweep is the 20 shipped voices only. Derived voices are A6's business:
    # there is no point blending voices before knowing which bases survive
    # French, and A2 only ever auditioned 13 takes, 5 of the 20 bases never
    # among them.
    #
    # Two texts, because A2 measured that accented prefixes stay fluent on easy
    # material and break on numbers and liaisons. Scoring only the easy one
    # would flatter half the sweep. Neither text contains digits: the reference
    # is compared word by word, and "75" against "soixante-quinze" would count
    # as an error the listener would not hear.
    local hard="Le vieux chene du jardin a soixante-quinze ans, et les enfants s'y retrouvent chaque ete depuis vingt ans."
    local easy="Bonjour, je m'appelle Camille et je vous propose aujourd'hui une petite histoire."
    local seed=7
    local root="cache/cloning/A5"
    mkdir -p "$root/hard" "$root/easy"

    local voices=(
        ar_male casual_female casual_male cheerful_female
        de_female de_male es_female es_male
        fr_female fr_male hi_female hi_male
        it_female it_male neutral_female neutral_male
        nl_female nl_male pt_female pt_male
    )

    check_fork_has_local_voice_lookup
    check_port_clash
    start_voxtral
    wait_for_health || return 1
    show_voice_boot_log

    # Same text, same seed for every take, so the only thing that varies between
    # two wavs is the voice prefix.
    say "phase 1: synthesizing ${#voices[@]} voices x 2 texts (seed $seed)"
    local failed=()
    local v
    for v in "${voices[@]}"; do
        printf '\n--- %s\n' "$v"
        ./tts.py -t "$hard" --voice "$v" --noise-seed "$seed" -o "$root/hard/$v.wav" || failed+=("$v/hard")
        ./tts.py -t "$easy" --voice "$v" --noise-seed "$seed" -o "$root/easy/$v.wav" || failed+=("$v/easy")
    done
    ((${#failed[@]})) && fail "${#failed[@]} take(s) rejected: ${failed[*]}"

    say "phase 2: freeing the GPU for the ASR stack"
    sudo docker compose stop voxtral-tts
    ok "voxtral-tts stopped"

    # The ASR service is key-protected and, on recent builds, refuses a request
    # that names no model. Both are passed through to asr.py by voice_survey.
    if [[ -z "${CRISPASR_API_KEY:-}" ]]; then
        warn "CRISPASR_API_KEY is unset, so scoring will fail; export it and re-run"
        warn "the wavs are already in $root, so a re-run only repeats phase 2"
    fi
    sudo docker compose up -d crispasr crispasr-lb
    local deadline=$((SECONDS + 600))
    while ((SECONDS < deadline)); do
        ./asr.py --check --api-key "${CRISPASR_API_KEY:-}" >/dev/null 2>&1 && break
        sleep 5
        printf '.'
    done
    ./asr.py --check --api-key "${CRISPASR_API_KEY:-}" >/dev/null 2>&1 \
        && ok "ASR healthy after ${SECONDS}s" \
        || { fail "ASR did not come up; wavs are in $root, re-run after fixing it"; return 1; }

    say "phase 3: scoring"
    ./voxtral/cloning/voice_survey.py --wav-dir "$root/hard" --text "$hard" --out "$root/survey_hard.md"
    ./voxtral/cloning/voice_survey.py --wav-dir "$root/easy" --text "$easy" --out "$root/survey_easy.md"

    say "results"
    echo "    listen:  $root/hard/  and  $root/easy/"
    echo "    ranked:  $root/survey_hard.md  (start at the top, it is sorted by WER)"
    echo "    the 'pitch neighbours' list at the bottom is the same-voice check"
    echo
    echo "Fill in natural? / distinct? / keep, then tell the agent the counts."
    echo "A6 (blends of the survivors) is only worth building once that number exists."
}

stage_b7() {
    # Mint real code sequences to train the missing audio encoder on.
    #
    # Why: B6 established that the encoder trains fine and there is nothing to
    # train it on. The 20 shipped voices are 2823 frames, about four minutes,
    # holding 1071 distinct semantic codes, 13 percent of the 8192. A network
    # cannot learn a map whose output space it has seen an eighth of, and it does
    # not: 94 percent on the training frames, 2 percent on a held-out voice.
    # Sampling the code space uniformly instead is free and useless (0.2 percent
    # on the training batch itself), because uniform codes are 37 dimensions of
    # noise per frame.
    #
    # The generator we need is the server. Stage 0 is an AR LLM whose whole job is
    # text -> on-manifold codes, and patch 0007 makes it write out what it emits.
    # So this stage is just: build, prove the dump works, generate, collect.
    #
    # Cost, so nothing here is a surprise:
    #   - GPU: yes, for the generation, and exclusively. The container takes ~17 GB
    #     (VOXTRAL_GPU_MEM_UTIL 0.7 of a 24 GB card) and the encoder training needs
    #     the GPU too, so the two cannot overlap. This stage STOPS the container
    #     when it is done, which is what frees the card for the agent to train.
    #   - Time: build a few minutes (layers are cached, only the fork's install
    #     layer rebuilds), boot several minutes (9 GB of weights, already in
    #     ./cache), generation is the long pole. It runs 4 requests at a time and
    #     prints seconds-per-call every 20, so the true rate is visible early;
    #     budget an hour for --minutes 240 and read the first report to refine it.
    #   - Disk: negligible, and deliberately so. The audio is NOT kept: the trainer
    #     decodes the codes locally to get exactly the audio they stand for, so
    #     saving the server's copy would only mean storing it twice. Codes are 74
    #     bytes a frame, so 240 minutes of speech is about 13 MB of dumps plus
    #     ~53 MB of npy. The image rebuild is the only real disk cost.
    #   - Network: the text corpus is already fetched (cache/cloning/corpus).
    local minutes="${B7_MINUTES:-240}"
    local smoke_text="Bonjour, ceci est un test de collecte de codes."
    local smoke_tag="b7_smoke_$(date +%Y%m%d%H%M%S)"
    local dump_dir="cache/cloning/dumps"

    check_fork_can_dump_codes
    check_fork_has_local_voice_lookup

    say "corpus"
    if [[ ! -d cache/cloning/corpus ]] || [[ -z "$(ls -A cache/cloning/corpus 2>/dev/null)" ]]; then
        fail "no corpus in cache/cloning/corpus"
        warn "run: ./voxtral/cloning/mint_codes.py corpus --articles 500 --delay 5"
        exit 1
    fi
    ls -l cache/cloning/corpus

    mkdir -p "$dump_dir"
    check_port_clash
    start_voxtral
    wait_for_health

    # The whole run hinges on the dump working, and the failure mode is silent, so
    # spend one short call proving it before spending an hour on the assumption.
    say "smoke test: does a request actually write codes?"
    ./tts.py -t "$smoke_text" --voice fr_male --noise-seed 7 \
        --dump-codes "$smoke_tag" -o "$OUT_DIR/b7_smoke.wav"

    local dump="$dump_dir/$smoke_tag.i16"
    if [[ ! -s "$dump" ]]; then
        fail "no codes written to $dump"
        warn "the request succeeded, so this is the stale-image failure: the running"
        warn "container predates patch 0007, or VOXTRAL_CODE_DUMP_DIR is not set on it"
        say "what the container thinks its environment is"
        sudo docker compose exec voxtral-tts printenv VOXTRAL_CODE_DUMP_DIR || \
            warn "VOXTRAL_CODE_DUMP_DIR is unset inside the container"
        sudo docker compose logs --tail 40 voxtral-tts | grep -i "PATCHED\|STOCK" || \
            warn "no PATCHED/STOCK line in the recent logs"
        exit 1
    fi

    # 74 bytes a frame (37 int16). Frames should match the wav's own duration at
    # 12.5 Hz, which catches a dump that is written but misaligned or truncated.
    local bytes frames expected wav_bytes
    bytes=$(stat -c%s "$dump")
    wav_bytes=$(stat -c%s "$OUT_DIR/b7_smoke.wav")
    frames=$((bytes / 74))
    expected=$(( (wav_bytes - 44) / 3840 ))
    printf '  %s bytes, %s frames; wav implies %s frames\n' "$bytes" "$frames" "$expected"
    if (( bytes % 74 != 0 )); then
        fail "dump is not a whole number of frames, the row width is wrong"
        exit 1
    fi
    if (( frames < expected - 2 || frames > expected + 2 )); then
        warn "frame count and audio duration disagree by more than 2 frames"
        warn "generation will still run, but check the collected codes before training"
    else
        ok "codes written and aligned with the audio, the dump path works"
    fi
    rm -f "$dump"

    # The check above proves a dump appeared with the right shape. It does NOT prove
    # the codes in it are the ones that made that audio: a dump written from the
    # wrong tensor, off by a frame, or in the wrong code space would pass it and then
    # poison every hour of collection that follows, silently, because wrong codes are
    # still valid codes. This decodes one dump locally and compares waveforms.
    say "round trip: are the dumped codes really the codes behind the audio?"
    if ! ./voxtral/cloning/verify_dump.py --voice fr_male; then
        fail "the dump does not reproduce the server's audio"
        warn "collecting with it would produce valid-looking, wrong training data"
        exit 1
    fi

    say "generating ${minutes} minutes of speech across 19 voices (holding out fr_female)"
    say "this is the long part; watch the seconds-per-call line to judge the rate"
    ./voxtral/cloning/mint_codes.py mint --minutes "$minutes"

    say "what was collected"
    ls -1 cache/cloning/codes | wc -l | xargs printf '  %s npy files in cache/cloning/codes\n'
    du -sh cache/cloning/codes "$dump_dir"

    # The GPU is the point of stopping here: the encoder training needs the whole
    # card, and nothing else in this stage does.
    say "stopping voxtral-tts to free the GPU for training"
    sudo docker compose stop voxtral-tts
    ok "GPU free"

    say "B7 generation done. Hand back to the agent to train:"
    echo "    ./voxtral/cloning/train_encoder.py --steps 30000 --batch 64 --frames 32"
    echo "    ./voxtral/cloning/invert.py run --voice fr_female --init encoder"
}

# ------------------------------------------------------------------- main ----

case "$STAGE" in
    logs)
        exec sudo docker compose logs -f voxtral-tts
        ;;
    down)
        require_sudo
        sudo docker compose stop voxtral-tts
        ok "voxtral-tts stopped, GPU freed"
        ;;
    diag | a1 | a2 | a3 | a5 | b7)
        require_sudo
        say "stage $STAGE (log: $LOG_FILE)"
        "stage_$STAGE"
        say "done. Log saved to $LOG_FILE"
        echo "The agent reads that file next session; no need to paste anything."
        ;;
    *)
        fail "unknown stage '$STAGE' (expected: diag, a1, a2, a3, a5, b7, logs, down)"
        exit 1
        ;;
esac
