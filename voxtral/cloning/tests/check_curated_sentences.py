#!/usr/bin/env -S uv run --script
# /// script
# requires-python = ">=3.10"
# dependencies = ["click", "loguru"]
# ///
"""Check that a hand-written corpus survives `sweep.py sentences` intact.

Run it directly: `uv run voxtral/cloning/tests/check_curated_sentences.py`.

Like its siblings in this directory, this is not a test suite (this repo has none, per
CLAUDE.md) but a pin on one specific way two parts of the sweep disagreed.

`sentences` filters prose out of a Wikipedia dump, and every filter it applies is correct
for that job: digits are rejected because the ASR writes "75" where the input says
"soixante-quinze", interior capitals are capped because a mistranscribed proper noun adds
WER variance for every voice alike, and lines are split on sentence boundaries. Pointed at
a corpus written by hand to be HARD, all of that inverts: corpus_medical_fr.txt is the ten
highest-CER rows of a finished run, and the screen keeps 2 of the 10, the only two
carrying no drug name, dose or acronym. It did not fail, it reported "only 2 usable
sentences ... wanted 10", which reads like a corpus problem rather than a mode mismatch.

So the pin is in two halves: curated mode must keep all ten lines byte for byte and in
file order, and screen mode must still reject them, because a screen that stopped
filtering would quietly change what every earlier grid run measured.
"""
import json
import os
import shutil
import subprocess
import sys
from pathlib import Path

# Derived, never hardcoded: an absolute home path in a tracked file leaks the username.
REPO = Path(__file__).resolve().parents[3]
SWEEP = REPO / "voxtral" / "cloning" / "sweep.py"
CORPUS = REPO / "voxtral" / "cloning" / "corpus_medical_fr.txt"

run = "ZZCURATED"
stage = REPO / "cache" / "cloning" / run
shutil.rmtree(stage, ignore_errors=True)


def mint(*flags: str) -> subprocess.CompletedProcess:
    """Run `sweep.py sentences` against the medical corpus in a throwaway run directory.

    Parameters
    ----------
    *flags : str
        Extra flags, e.g. "--curated".

    Returns
    -------
    subprocess.CompletedProcess
        Not checked: half of this file is about the failure path.
    """
    return subprocess.run(
        ["uv", "run", str(SWEEP), "sentences", "--corpus", str(CORPUS),
         "--count", str(len(expected)), *flags],
        cwd=REPO, capture_output=True, text=True,
        # CLONING_POOL is irrelevant here (no pool is read) but CLONING_RUN decides where
        # sentences.json lands, so it must point somewhere disposable.
        env={**os.environ, "CLONING_RUN": run},
    )


expected = [line.strip() for line in CORPUS.read_text(encoding="utf-8").splitlines()
            if line.strip() and not line.lstrip().startswith("#")]
assert len(expected) == 12, f"corpus changed shape: {len(expected)} lines, expected 12"

# The last two are the ~30 s stability pair, and their length is load-bearing in both
# directions: under ~60 words they stop testing stability over length, and over 64 their
# runaway threshold (3 x words / 2.15 words per second) passes the sweep's 90 s token cap,
# at which point a degenerate take is truncated and measures as healthy.
long_words = [len(line.split()) for line in expected[-2:]]
assert all(60 <= count <= 64 for count in long_words), \
    f"the long pair is {long_words} words; outside 60-64 the junk test stops working"

# 1. Curated mode keeps every line, verbatim and in order. Verbatim matters beyond
# tidiness: the sentence text IS the WER reference, so a corpus line silently reshaped
# between file and sentences.json would be scored against audio of something else.
done = mint("--curated")
assert done.returncode == 0, f"curated mode failed:\n{done.stderr}"
rows = json.loads((stage / "sentences.json").read_text(encoding="utf-8"))
assert [row["text"] for row in rows] == expected, "curated mode altered or reordered lines"
assert [row["sid"] for row in rows] == [f"s{i:02d}" for i in range(len(expected))], rows
# File order is the contract that puts the long pair last, so --sentences N trims the
# short ones first and the stability test is the thing you keep, not the thing you lose.
assert len(rows[-1]["text"].split()) >= 60, "the long pair did not land last"
print(f"1. curated: all {len(rows)} lines verbatim, in file order, long pair last OK")

# 2. The sid scheme is the join key for tts, stt, stats and audition.py, so both modes must
# produce it identically. One shared writer is why; this notices if that is ever forked.
assert all(row["sid"] == f"s{index:02d}" for index, row in enumerate(rows)), rows
print(f"2. sids are s00..s{len(rows) - 1:02d}, the join key later passes use OK")

# 3. Screen mode must still reject this corpus, with the count it actually keeps. If this
# ever passes, the prose filters went missing and every grid run's numbers changed meaning.
shutil.rmtree(stage, ignore_errors=True)
done = mint()
assert done.returncode != 0, "screen mode accepted a digit-heavy medical corpus"
assert "only 2 usable" in done.stderr + done.stdout, \
    f"screen mode failed differently than expected:\n{done.stderr}"
print(f"3. screen: still rejects it, keeping 2 of {len(expected)}         OK")

shutil.rmtree(stage, ignore_errors=True)
if (REPO / "cache" / "cloning" / "ZZSENT").exists():
    shutil.rmtree(REPO / "cache" / "cloning" / "ZZSENT", ignore_errors=True)
print("\nall checks passed")
sys.exit(0)
