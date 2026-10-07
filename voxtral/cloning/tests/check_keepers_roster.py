#!/usr/bin/env -S uv run --script
# /// script
# requires-python = ">=3.10"
# dependencies = ["click", "loguru"]
# ///
"""Check that the keepers roster audition.py reads is the one voice_distance.py writes.

Run it directly: `uv run voxtral/cloning/tests/check_keepers_roster.py`.

This repo has no test suite (the client scripts are the tests, per CLAUDE.md), and this
file is not an attempt to start one. It exists for the same reason check_retry_ledger.py
does: two tools in `voxtral/cloning/` had to agree on a file format and silently did not.
`voice_distance.py keepers` wrote one bare voice name per line, while audition.py's
`read_voice_list` splits every row on tabs into (cluster, role, voice). Pointing
`--voice-list` at keepers.txt therefore died with "not enough values to unpack (expected
3, got 1)" AFTER the keepers run had already reported success, which reads like a broken
roster rather than two files disagreeing about what a roster is.

It works in a throwaway directory and needs no server, GPU or sweep data, but when a real
roster exists for `CLONING_RUN` it holds that one to the same contract.
"""
import collections
import json
import os
import shutil
import sys
from pathlib import Path

# Derived, never hardcoded: an absolute home path in a tracked file leaks the username.
REPO = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(REPO / "voxtral" / "cloning"))
import audition

tmp = REPO / "cache" / "cloning" / "ZZROSTER"
shutil.rmtree(tmp, ignore_errors=True)
tmp.mkdir(parents=True)

# 1. The shape voice_distance.py writes: a comment header, then one cluster per identity
# with the shipped donor ahead of the voice kept for it, and a lone row where the keeper is
# its own donor (fr_female below, which no hybrid beat).
roster = tmp / "keepers_roster.txt"
roster.write_text(
    "# Keepers of ZZROSTER, for: audition.py --voice-list x\n"
    "# cluster<TAB>role<TAB>voice; the shipped donor precedes the voice kept for it.\n"
    "1\tshipped\tfr_female\n"
    "2\tshipped\tde_female\n"
    "2\thybrid\th_def_ptm\n"
    "3\tshipped\tnl_male\n"
    "3\tmixn\tx_hif_nlm_esf\n"
)
rows = audition.read_voice_list(roster)
assert rows == [(1, "shipped", "fr_female"), (2, "shipped", "de_female"),
                (2, "hybrid", "h_def_ptm"), (3, "shipped", "nl_male"),
                (3, "mixn", "x_hif_nlm_esf")], rows
print("1. roster parses, comments skipped  OK")

# 2. Same-cluster rows stay adjacent and clusters ascend. audition.py documents that it
# relies on file order to keep a keeper next to the speaker it stands in for, so a roster
# that interleaved clusters would scatter the comparison without erroring.
runs = []
for cluster, _, _ in rows:
    if not runs or runs[-1] != cluster:
        runs.append(cluster)
assert len(runs) == len(set(runs)), f"a cluster is split across the file: {runs}"
assert runs == sorted(runs), runs
print("2. clusters contiguous and ordered  OK")

# 3. The regression itself: the bare-name list must NOT parse as a roster. keepers.txt is
# the file --voice-list was pointed at before the fix, and this is the exception it raised.
plain = tmp / "keepers.txt"
plain.write_text("fr_female\nh_def_ptm\nx_hif_nlm_esf\n")
try:
    audition.read_voice_list(plain)
except ValueError as exc:
    assert "unpack" in str(exc), exc
    print("3. keepers.txt still rejected       OK")
else:
    raise AssertionError("keepers.txt parsed as a roster: the two formats have merged, so "
                         "whichever file --voice-list gets, the mismatch is now silent")
finally:
    shutil.rmtree(tmp, ignore_errors=True)

# 4. When a real run has a roster, hold it to the same contract: every keeper listed, no
# voice twice, cluster numbering dense and ascending.
run = os.environ.get("CLONING_RUN")
real = REPO / "cache" / "cloning" / (run or "_unset") / "keepers_roster.txt"
if run and real.is_file():
    rows = audition.read_voice_list(real)
    names = [voice for _, _, voice in rows]
    clusters = [cluster for cluster, _, _ in rows]
    assert rows, f"{real} has no rows"
    assert len(names) == len(set(names)), "a voice appears twice in the roster"
    assert clusters == sorted(clusters), "clusters out of order"
    assert set(clusters) == set(range(1, max(clusters) + 1)), "cluster numbering has holes"
    listed = (real.parent / "keepers.txt").read_text().split()
    assert set(listed) <= set(names), set(listed) - set(names)
    print(f"4. {run}: {len(rows)} rows, {max(clusters)} clusters, "
          f"all {len(listed)} keepers present OK")
else:
    print("4. no real roster for CLONING_RUN   SKIP")

# 5. First attempts are selected by VALUE, not by the presence of the `attempt` key. Two
# ledger conventions coexist in the cache: A7's sweep tagged only retries, so its first
# attempts carry no key, while the sweep that grew the retry budget stamps `attempt: 0` on
# every row. `"attempt" in record` matched only the first and rejected every grid take.
led = tmp_ledger = REPO / "cache" / "cloning" / "ZZROSTER2"
shutil.rmtree(led, ignore_errors=True)
led.mkdir(parents=True)
(led / "ledger.jsonl").write_text(
    # A7 convention: first attempt has no `attempt` key.
    '{"voice": "a", "sid": "f1", "wav": "cache/x/a1.wav"}\n'
    '{"voice": "a", "sid": "f1", "attempt": 1, "wav": "cache/x/a1b.wav"}\n'
    # Grid convention: first attempt is an explicit 0.
    '{"voice": "b", "sid": "f1", "attempt": 0, "wav": "cache/x/b1.wav"}\n'
    '{"voice": "b", "sid": "f1", "attempt": 2, "wav": "cache/x/b1c.wav"}\n'
    # A scored row with no wav must not shadow the take it scores.
    '{"voice": "b", "sid": "f1", "attempt": 0, "wer": 0.1}\n'
)
first = audition.read_first_attempts(led / "ledger.jsonl")
assert first[("a", "f1")].name == "a1.wav", first[("a", "f1")]
assert first[("b", "f1")].name == "b1.wav", first[("b", "f1")]
assert len(first) == 2, first
shutil.rmtree(led, ignore_errors=True)
print("5. first attempt by value, both conventions OK")

# 6. The alternates roster is read by the same parser and really excludes what was already
# auditioned. Two defaults were wrong on the first cut and both failed silently, by offering
# a voice that had in fact been heard rather than by raising: it defaulted `--heard` to
# keepers.txt (which omits the 20 shipped voices the first tree also contained, so every
# shipped voice came back as "unheard"), and it offered sparse mixes, the one kind whose WER
# gate does not predict how it is received. An over-offering roster wastes a listening
# session, which is the scarce resource here, so pin both.
import subprocess

run = os.environ.get("CLONING_RUN")
stage = REPO / "cache" / "cloning" / (run or "_unset")
if run and (stage / "keepers_and_bases.txt").is_file() and (stage / "embeddings.npz").is_file():
    dest = REPO / "cache" / "cloning" / "ZZALT.txt"
    subprocess.run(
        ["uv", "run", str(REPO / "voxtral" / "cloning" / "voice_distance.py"), "alternates",
         "--per-identity", "2", "--out", str(dest)],
        cwd=REPO, check=True, capture_output=True,
    )
    rows = audition.read_voice_list(dest)
    alts = [voice for _, _, voice in rows]
    heard = set((stage / "keepers_and_bases.txt").read_text().split())
    pool = {entry["name"]: entry for entry in json.loads(
        (stage.parent / os.environ.get("CLONING_POOL", "GRID") / "pool.json").read_text())}
    assert alts, "alternates wrote no rows"
    assert len(alts) == len(set(alts)), "a voice is offered twice"
    assert not (set(alts) & heard), f"already auditioned: {sorted(set(alts) & heard)}"
    assert all(pool[voice]["kind"] == "hybrid" for voice in alts), \
        f"not a hybrid: {[v for v in alts if pool[v]['kind'] != 'hybrid']}"
    counts = collections.Counter(cluster for cluster, _, _ in rows)
    assert max(counts.values()) <= 2, f"--per-identity 2 exceeded: {counts}"
    dest.unlink()
    print(f"6. {run}: {len(alts)} alternates, none heard, all hybrids OK")
else:
    print("6. no alternates inputs for CLONING_RUN SKIP")


print("\nall checks passed")
