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

# 7. `voice_distance.py roster` turns a bare voice list into the roster format, which is the
# converter that closes the format gap check 3 pins open. Two bugs in it both failed by
# WARNING rather than by raising, which is the dangerous shape here: the roster is still
# written, so the audition proceeds on a file whose warning scrolled past. It warned that
# the shipped fr_female/fr_male were "not in pool" (pool.json holds only the minted
# candidates, so their absence is correct, not a typo), and clustering must group by
# identity so a hybrid lands next to the other candidate for the same speaker: a hybrid is
# heard as its SECOND name, so h_itf_def and h_frm_def are two takes on de_female and are
# useless to compare nine directories apart.
if run and (REPO / "cache" / "cloning" / os.environ.get("CLONING_POOL", "GRID") / "pool.json").is_file():
    src = REPO / "cache" / "cloning" / "ZZLIST.txt"
    dest = REPO / "cache" / "cloning" / "ZZLIST.roster.txt"
    src.write_text("# a comment, must be ignored\nfr_female\nh_itf_def\nh_frf_hif\nh_frm_def\n")
    done = subprocess.run(
        ["uv", "run", str(REPO / "voxtral" / "cloning" / "voice_distance.py"), "roster",
         str(src), "--out", str(dest)],
        cwd=REPO, check=True, capture_output=True, text=True,
    )
    rows = audition.read_voice_list(dest)
    names = [voice for _, _, voice in rows]
    assert names == ["fr_female", "h_itf_def", "h_frm_def", "h_frf_hif"], names
    # The two de_female candidates share a cluster; fr_female and h_frf_hif do not join it.
    clusters = {voice: cluster for cluster, _, voice in rows}
    assert clusters["h_itf_def"] == clusters["h_frm_def"], rows
    assert len({clusters["fr_female"], clusters["h_frf_hif"], clusters["h_itf_def"]}) == 3, rows
    assert "fr_female" not in done.stderr, \
        f"shipped voice reported as a typo:\n{done.stderr}"
    src.unlink()
    dest.unlink()
    print("7. roster: identities grouped, shipped voices not flagged OK")
else:
    print("7. no pool for CLONING_POOL              SKIP")

# 8. `--sentences 0` takes the run's sentence list as given, and an unscored run still
# builds a tree. Both halves were broken for a listening session on a hand-written corpus:
# pick_sentences exists to find EASY sentences in a 300-line FLEURS slice (clarity rank,
# 12-26 word window, no guillemets), so on ten lines chosen to be hard it dropped the ones
# that mattered; and with no stt pass there is no per_sentence.csv, which made read_wer
# raise on open and write_index raise a KeyError AFTER the wavs were already copied, the
# worst time to fail. Built from silent wavs so this needs no GPU, server or sweep data.
import json as _json
import struct
import wave

stage = REPO / "cache" / "cloning" / "ZZSENT0"
shutil.rmtree(stage, ignore_errors=True)
(stage / "wav").mkdir(parents=True)
# 10 and 27 words: one under MIN_WORDS, one over MAX_WORDS, so both are invisible to the
# picker. If a future change routes --sentences 0 back through it, they vanish and this
# fails.
texts = {"s00": " ".join(["mot"] * 10) + ".", "s01": " ".join(["mot"] * 27) + "."}
(stage / "sentences.json").write_text(_json.dumps(
    [{"sid": sid, "text": text} for sid, text in texts.items()]), encoding="utf-8")
ledger = []
for voice in ["fr_female", "h_itf_def"]:
    (stage / "wav" / voice).mkdir(parents=True, exist_ok=True)
    for sid in texts:
        wav = stage / "wav" / voice / f"{sid}.wav"
        with wave.open(str(wav), "wb") as handle:
            handle.setnchannels(1)
            handle.setsampwidth(2)
            handle.setframerate(24000)
            handle.writeframes(struct.pack("<h", 0) * 2400)
        ledger.append({"voice": voice, "sid": sid, "attempt": 0,
                       "wav": str(wav.relative_to(REPO)), "bytes": wav.stat().st_size})
(stage / "ledger.jsonl").write_text("".join(_json.dumps(row) + "\n" for row in ledger))
(stage / "roster.txt").write_text("1\tshipped\tfr_female\n2\thybrid\th_itf_def\n")
assert not (stage / "per_sentence.csv").exists(), "the unscored case is the one under test"

tree = REPO / "cache" / "cloning" / "ZZSENT0_tree"
shutil.rmtree(tree, ignore_errors=True)
done = subprocess.run(
    ["uv", "run", str(REPO / "voxtral" / "cloning" / "audition.py"), "--dest", str(tree),
     "--stage-dir", str(stage), "--sentences", "0", "--voice-list", str(stage / "roster.txt")],
    cwd=REPO, capture_output=True, text=True,
)
assert done.returncode == 0, f"audition.py failed on an unscored run:\n{done.stderr}"
wavs = sorted(path.name for path in tree.rglob("*.wav"))
# 2 voices x 2 sentences, plus one side-by-side copy per voice.
assert len(wavs) == 6, wavs
assert (tree / "00_meme_phrase").is_dir(), "no side-by-side directory"
readme = (tree / "README.md").read_text(encoding="utf-8")
assert "SCORE=1" in readme, "the index does not say why the WER column is empty"
assert "| - | - |" in readme, f"missing WER should render as a dash:\n{readme}"
shutil.rmtree(stage, ignore_errors=True)
shutil.rmtree(tree, ignore_errors=True)
print("8. --sentences 0 on an unscored run: tree built  OK")


print("\nall checks passed")
