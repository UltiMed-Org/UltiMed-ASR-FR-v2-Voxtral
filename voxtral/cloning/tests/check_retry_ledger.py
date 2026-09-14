#!/usr/bin/env -S uv run --script
# /// script
# requires-python = ">=3.10"
# dependencies = ["click", "loguru", "tqdm"]
# ///
"""Check the attempt semantics of the ledger, which are too subtle to eyeball.

Run it directly: `uv run voxtral/cloning/tests/check_retry_ledger.py`.

This repo has no test suite (the client scripts are the tests, per CLAUDE.md), and this
file is not an attempt to start one. It exists because retries changed the meaning of the
ledger, and one specific mistake there is both easy to make and invisible afterwards: if a
new attempt MERGED into the old row instead of replacing it, the previous attempt's WER
would sit next to the new attempt's wav, both passes would believe a freshly synthesized
take was already scored, and the benchmark would silently report stale numbers for every
retried take. Nothing downstream would complain.

It works on a throwaway run directory (cache/cloning/ZZTEST) and removes it afterwards, so
it touches no real data and needs no server or GPU.
"""
import json, os, shutil, sys
from pathlib import Path
# Derived, never hardcoded: an absolute home path in a tracked file leaks the username.
REPO = Path(__file__).resolve().parents[3]
run = REPO / "cache" / "cloning" / "ZZTEST"
shutil.rmtree(run, ignore_errors=True)
run.mkdir(parents=True)
os.environ["CLONING_RUN"] = "ZZTEST"
sys.path.insert(0, str(REPO / "voxtral" / "cloning"))
import sweep

(run / "sentences.json").write_text(json.dumps([{"sid": "f1", "text": "Bonjour le monde ici."}]))
led = run / "ledger.jsonl"

def w(**kw):
    with led.open("a") as h: h.write(json.dumps(kw) + "\n")

# attempt 0: synthesized then scored, came out degenerate
w(voice="v", sid="f1", attempt=0, wav="cache/cloning/ZZTEST/wav/v/f1.wav", request_seconds=3.0)
w(voice="v", sid="f1", attempt=0, transcript="bla bla bla bla", wer=2.5, cer=1.8, seconds=40.0)
state = sweep.read_ledger()
assert state[("v", "f1")]["wer"] == 2.5 and state[("v", "f1")]["attempt"] == 0
print("1. merge within an attempt          OK")

# retry schedules attempt 1: no wav, no wer, and it must REPLACE not merge
w(voice="v", sid="f1", attempt=1, retry_reason="runaway")
state = sweep.read_ledger()
row = state[("v", "f1")]
assert row["attempt"] == 1, row
assert "wer" not in row, "the old score leaked into the new attempt"
assert "wav" not in row, "the old wav leaked into the new attempt"
print("2. higher attempt replaces the row  OK")

# so both passes see it as pending again
assert not (row.get("wav") and sweep.wav_for("v", "f1", row.get("attempt", 0)).is_file())
print("3. row is pending for tts again     OK")

# attempt 1 succeeds
w(voice="v", sid="f1", attempt=1, wav="cache/cloning/ZZTEST/wav/v/f1.a1.wav", request_seconds=3.1)
w(voice="v", sid="f1", attempt=1, transcript="Bonjour le monde ici.", wer=0.0, cer=0.0, seconds=2.0)
hist = sweep.read_attempts()[("v", "f1")]
assert [h["attempt"] for h in hist] == [0, 1]
assert [h["wer"] for h in hist] == [2.5, 0.0]
print("4. history keeps both attempts      OK")

# paths are distinct and attempt 0 keeps its legacy name
assert sweep.wav_for("v", "f1", 0).name == "f1.wav"
assert sweep.wav_for("v", "f1", 1).name == "f1.a1.wav"
print("5. attempt 0 keeps its legacy path  OK")

# seeds differ per attempt, and agree across voices for one sentence
s0 = sweep.sentence_seed({"sid": "f1"}) + 0 * sweep.ATTEMPT_SEED_STRIDE
s1 = sweep.sentence_seed({"sid": "f1"}) + 1 * sweep.ATTEMPT_SEED_STRIDE
assert s0 != s1 and s1 - s0 == sweep.ATTEMPT_SEED_STRIDE
print("6. attempt seeds differ, stride ok  OK")

# a record with no attempt field is attempt 0 (every pre-retry row in FLEURS)
w(voice="w", sid="f1", wav="x.wav")
w(voice="w", sid="f1", wer=0.1, seconds=2.0)
assert sweep.read_ledger()[("w", "f1")].get("attempt", 0) == 0
assert len(sweep.read_attempts()[("w", "f1")]) == 1
print("7. legacy rows are attempt 0        OK")

# runaway detection: 40 s for 4 words at 2 words/s should be 20x
st = sweep.stretch_of(seconds=40.0, words=4, rate=2.0)
assert abs(st - 20.0) < 1e-9, st
assert sweep.stretch_of(seconds=2.0, words=4, rate=2.0) == 1.0
print("8. stretch_of arithmetic            OK")
shutil.rmtree(run, ignore_errors=True)
print("\nall checks passed")
