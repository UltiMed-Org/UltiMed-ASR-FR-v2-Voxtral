#!/usr/bin/env -S uv run --script
# /// script
# requires-python = ">=3.10"
# dependencies = ["click", "loguru"]
# ///
"""Screen a large pool of Voxtral voices for French, resumably (stage A6).

A5 auditioned 20 voices by hand. A6 generates about 420 candidates, which is past
what anyone can listen to, so the ears have to be spent on survivors only. This
script is the unattended half of that: it drives synthesis and scoring over a
(voice, sentence) grid, writing every completed unit of work to an append-only
ledger so that a run killed at any moment resumes where it stopped.

Two GPU passes, never at the same time, because one 24 GB card cannot hold the
Voxtral stack and the whisper stack together:

    tts   for every pending row, synthesize one wav        (voxtral-tts up)
    stt   for every un-scored wav, transcribe and score it (crispasr up)

`driver.sh` alternates them and owns the container switching. Two offline commands
need no GPU and no server: `sentences` picks the screen texts, `report` ranks what
has been scored so far and can be run mid-sweep.

Exit codes, matching the driver's contract:

    0   work was done, call again
    10  converged: every row of the grid has a wav and a score
    11  this pass has nothing to do, but the other one does
    20  stopped early because the STOP flag appeared

Row order is sentence-major on purpose: all voices read sentence 0, then all read
sentence 1. An interrupted run therefore leaves *every* candidate measured at a
lower depth, rather than half the candidates measured deeply and half not at all.
Depth is a parameter (`--sentences`), so deepening a finished sweep is just the
same command with a larger number: the rows already in the ledger are skipped.

Design notes worth keeping in mind before editing:

- Intelligibility is a GATE, never a ranking. The lowest-WER voice is the blandest
  one, so sorting by WER and keeping the top N would select against the diversity
  this whole track exists to produce. See `report`.
- The noise seed is derived from the sentence alone, not from the voice, so that
  two candidates reading the same sentence differ only by their prefix. That is
  what makes takes comparable across voices (and what a speaker-embedding distance
  needs to mean anything).
"""

from __future__ import annotations

import json
import os
import random
import re
import statistics
import sys
import time
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import click
from loguru import logger

REPO_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO_ROOT))
import tts  # noqa: E402 (needs REPO_ROOT on the path first)

sys.path.insert(0, str(Path(__file__).resolve().parent))
from voice_survey import transcribe, word_error_rate  # noqa: E402
from wav_stats import median_f0, read_wav  # noqa: E402

STAGE_DIR = REPO_ROOT / "cache" / "cloning" / "A6"
CORPUS = REPO_ROOT / "cache" / "cloning" / "corpus" / "fr.txt"
# Touching this file asks a running pass to park at the next row boundary. stop_all.sh
# writes it before taking the containers down, so an interrupted sweep never loses more
# than the rows in flight.
STOP_FLAG = STAGE_DIR / "STOP"
# Anything above this word error rate is garbled rather than merely accented. A5
# calibrated it: the worst of the 20 shipped voices scored 26 pct on a hard sentence,
# so a candidate above 35 pct is worse than every voice that actually ships.
WER_REJECT = 0.35
# Letters a French voice is expected to read, plus the punctuation that survives
# `mint_codes.clean`. A sentence carrying anything else (Portuguese tildes, Catalan
# grave accents, transliterated names) is skipped rather than scored.
FRENCH_ONLY = re.compile(r"[A-Za-zÀÂÄÇÉÈÊËÎÏÔÖÙÛÜŸÆŒàâäçéèêëîïôöùûüÿæœ' \-,;:!?.()«»\u2019\u202f\u00a0]+")


def stop_requested() -> bool:
    """Whether the STOP flag is present.

    Returns
    -------
    bool
        True when a pass should park.
    """
    return STOP_FLAG.exists()


def load_sentences() -> list[dict]:
    """Read the screen texts.

    Returns
    -------
    list of dict
        `{"sid": str, "text": str}` in a fixed order.
    """
    path = STAGE_DIR / "sentences.json"
    if not path.is_file():
        raise SystemExit(f"no sentences yet ({path}); run `sweep.py sentences` first")
    return json.loads(path.read_text())


def load_voices(*, include_bases: bool) -> list[str]:
    """Read the candidate pool, optionally with the shipped voices as controls.

    The controls are what make the numbers interpretable: a candidate scoring worse
    than the worst shipped voice is broken, and without them the reject threshold
    would be my guess rather than a measurement.

    Parameters
    ----------
    include_bases : bool
        Add every base voice named as a parent in the pool manifest.

    Returns
    -------
    list of str
        Voice names, candidates first.
    """
    path = STAGE_DIR / "pool.json"
    if not path.is_file():
        raise SystemExit(f"no pool yet ({path}); run `derive_voices.py pool` first")
    pool = json.loads(path.read_text())
    names = [entry["name"] for entry in pool]
    if include_bases:
        bases = sorted({parent for entry in pool for parent in entry["parents"]})
        names += [name for name in bases if name not in set(names)]
    return names


def ledger_path() -> Path:
    """Where the append-only record of finished work lives.

    Returns
    -------
    Path
        The ledger file (may not exist yet).
    """
    return STAGE_DIR / "ledger.jsonl"


def read_ledger() -> dict[tuple[str, str], dict]:
    """Replay the ledger into one state dict per row.

    Append-only with last-write-wins per key, so a row written twice (a retry after a
    crash, say) collapses to its latest state, and a half-written final line from a
    `kill -9` is skipped instead of aborting the run.

    Returns
    -------
    dict
        `{(voice, sid): {...}}`.
    """
    state: dict[tuple[str, str], dict] = {}
    path = ledger_path()
    if not path.is_file():
        return state
    for line in path.read_text(errors="replace").splitlines():
        line = line.strip()
        if not line:
            continue
        try:
            record = json.loads(line)
        except json.JSONDecodeError:
            logger.warning("skipping a truncated ledger line (interrupted write?)")
            continue
        key = (record["voice"], record["sid"])
        state.setdefault(key, {}).update(record)
    return state


def append_ledger(record: dict) -> None:
    """Append one record, flushed, so a kill loses at most the row in flight.

    Parameters
    ----------
    record : dict
        Must carry `voice` and `sid`.
    """
    with ledger_path().open("a") as handle:
        handle.write(json.dumps(record, ensure_ascii=False) + "\n")
        handle.flush()
        os.fsync(handle.fileno())


def grid(*, voices: list[str], sentences: list[dict], depth: int) -> list[tuple[str, dict]]:
    """Build the work list in sentence-major order.

    Parameters
    ----------
    voices : list of str
        Voices to sweep.
    sentences : list of dict
        Screen texts.
    depth : int
        How many sentences per voice.

    Returns
    -------
    list of tuple
        `(voice, sentence)` pairs.
    """
    return [(voice, sentence) for sentence in sentences[:depth] for voice in voices]


def wav_for(voice: str, sid: str) -> Path:
    """Where one take lives.

    Parameters
    ----------
    voice : str
        Voice name.
    sid : str
        Sentence id.

    Returns
    -------
    Path
        Wav path, inside a per-voice directory so a voice can be deleted as a unit.
    """
    return STAGE_DIR / "wav" / voice / f"{sid}.wav"


@click.group()
def cli() -> None:
    """Screen a pool of voices for French (stage A6)."""
    logger.remove()
    logger.add(sys.stderr, format="{time:HH:mm:ss} <level>{level: <7}</level> {message}", level="INFO")


@cli.command("sentences")
@click.option("--count", default=12, show_default=True, help="How many screen sentences to keep.")
@click.option("--min-words", default=8, show_default=True, help="Shortest sentence to accept.")
@click.option("--max-words", default=18, show_default=True, help="Longest sentence to accept.")
@click.option("--seed", default=0, show_default=True, help="Seed for the sentence draw.")
@click.option("--corpus", default=str(CORPUS), show_default=True, help="Source text, one paragraph per line.")
def cmd_sentences(count: int, min_words: int, max_words: int, seed: int, corpus: str) -> None:
    """Pick the screen texts out of the French corpus B7 already downloaded.

    Not FLEURS, and the reason is worth recording: FLEURS' value is its *human*
    transcripts, and here the reference is the text we hand the model, so gold labels
    add nothing. Meanwhile the HuggingFace rows API refuses to serve google/fleurs
    (its parquet row groups exceed the scan limit), so using it would mean downloading
    gigabytes of audio we would throw away. The Wikipedia corpus in cache/cloning/corpus
    is local, already cleaned by mint_codes.py, and real French prose.

    Sentences with digits are rejected: the ASR writes "75" where the input says
    "soixante-quinze", and while `voice_survey.fold_numbers` repairs the common cases,
    the cheapest fix is to not create the problem. Dates, decimals and ranges are
    exactly where that repair is least reliable.
    """
    source = Path(corpus)
    if not source.is_file():
        raise SystemExit(f"no corpus at {source}; run `mint_codes.py corpus --language fr`")

    candidates: list[str] = []
    for paragraph in source.read_text().splitlines():
        for piece in re.split(r"(?<=[.!?])\s+", paragraph.strip()):
            piece = piece.strip()
            words = piece.split()
            if not min_words <= len(words) <= max_words:
                continue
            if any(character.isdigit() for character in piece):
                continue
            # A sentence that opens mid-clause makes the model guess at a prosody the
            # text does not support, and those takes drift. Require a capital start and
            # terminal punctuation so every screen text is a whole sentence.
            if not piece[0].isupper() or piece[-1] not in ".!?":
                continue
            # Proper nouns are the other WER trap, and Wikipedia is full of them. A
            # candidate is not rejected for mispronouncing "Mastacembelidae" or
            # "Bastianelli", but the ASR will mistranscribe them for EVERY voice, which
            # adds variance and can push an otherwise clean take over the gate. Allow at
            # most one interior capitalised word, and no letters outside French.
            interior_capitals = sum(1 for word in words[1:] if word[:1].isupper())
            if interior_capitals > 1:
                continue
            if not FRENCH_ONLY.fullmatch(piece):
                continue
            candidates.append(piece)

    unique = sorted(set(candidates))
    if len(unique) < count:
        raise SystemExit(f"only {len(unique)} usable sentences in {source}, wanted {count}")
    random.Random(seed).shuffle(unique)
    chosen = unique[:count]

    STAGE_DIR.mkdir(parents=True, exist_ok=True)
    sentences = [{"sid": f"s{index:02d}", "text": text} for index, text in enumerate(chosen)]
    (STAGE_DIR / "sentences.json").write_text(json.dumps(sentences, indent=2, ensure_ascii=False))
    for entry in sentences:
        logger.info(f"{entry['sid']}  {entry['text']}")
    logger.success(f"{len(sentences)} screen sentences written to {STAGE_DIR / 'sentences.json'}")


@cli.command("tts")
@click.option("--sentences", "depth", default=2, show_default=True, help="Sentences per voice this pass.")
@click.option("--url", default=tts.DEFAULT_URL, show_default=True, help="voxtral-tts base URL.")
@click.option("--concurrency", default=4, show_default=True,
              help="Parallel requests (audio throughput saturates around 4 to 8).")
@click.option("--timeout", default=300.0, show_default=True, help="Seconds per request.")
@click.option("--noise-seed", default=7, show_default=True,
              help="Base noise seed; sentence index is added so takes stay comparable across voices.")
@click.option("--euler-steps", default=0, show_default=True, help="Override the ODE step count (0 = server default).")
@click.option("--include-bases/--no-include-bases", default=True, show_default=True,
              help="Sweep the shipped voices too, as controls.")
@click.option("--limit", default=0, show_default=True, help="Stop after N rows this pass (0 = all pending).")
def cmd_tts(depth: int, url: str, concurrency: int, timeout: float, noise_seed: int,
            euler_steps: int, include_bases: bool, limit: int) -> None:
    """Synthesize every pending row of the grid."""
    sentences = load_sentences()
    voices = load_voices(include_bases=include_bases)
    state = read_ledger()
    rows = grid(voices=voices, sentences=sentences, depth=depth)

    # A row counts as done when the ledger says so AND the wav is still on disk, so
    # deleting a take is enough to ask for it again.
    pending = [(voice, sentence) for voice, sentence in rows
               if not (state.get((voice, sentence["sid"]), {}).get("wav")
                       and wav_for(voice, sentence["sid"]).is_file())]
    logger.info(f"{len(rows)} rows in the grid ({len(voices)} voices x {min(depth, len(sentences))} "
                f"sentences), {len(pending)} to synthesize")
    if not pending:
        logger.info("nothing to synthesize")
        sys.exit(11)
    if limit:
        pending = pending[:limit]

    endpoint = f"{url.rstrip('/')}/v1/audio/speech"
    counters = {"ok": 0, "failed": 0, "seconds": 0.0}
    started = time.monotonic()

    def run(job: tuple[str, dict]) -> None:
        voice, sentence = job
        sid = sentence["sid"]
        # Same seed for every voice reading this sentence: the take then differs only by
        # the prefix, which is what makes cross-voice comparison (and a later speaker
        # embedding distance) mean anything.
        body = tts.build_speech_body(text=sentence["text"], response_format="wav", speed=1.0,
                                     voice=voice, noise_seed=noise_seed + int(sid[1:]),
                                     euler_steps=euler_steps or None)
        try:
            status, payload, _, elapsed = tts._post_json(endpoint, body, timeout)
        except OSError as error:
            counters["failed"] += 1
            logger.warning(f"{voice}/{sid}: {error}")
            return
        if status != 200:
            counters["failed"] += 1
            logger.warning(f"{voice}/{sid}: HTTP {status} in {elapsed:.1f}s: {payload[:160]!r}")
            append_ledger({"voice": voice, "sid": sid, "tts_error": f"HTTP {status}"})
            return
        destination = wav_for(voice, sid)
        destination.parent.mkdir(parents=True, exist_ok=True)
        destination.write_bytes(payload)
        counters["ok"] += 1
        counters["seconds"] += elapsed
        append_ledger({"voice": voice, "sid": sid, "wav": str(destination.relative_to(REPO_ROOT)),
                       "bytes": len(payload), "request_seconds": round(elapsed, 2)})
        if counters["ok"] % 25 == 0:
            rate = counters["ok"] / max(time.monotonic() - started, 1e-9)
            left = (len(pending) - counters["ok"]) / max(rate, 1e-9)
            logger.info(f"{counters['ok']}/{len(pending)} synthesized, {counters['failed']} failed, "
                        f"{counters['seconds'] / counters['ok']:.1f}s per call, "
                        f"{rate * 60:.0f}/min, about {left / 60:.0f} min left")

    # Chunked submission rather than one map over everything: it is the only way a
    # thread pool can honour the STOP flag promptly without cancelling in-flight work.
    stopped = False
    with ThreadPoolExecutor(max_workers=concurrency) as pool:
        for start in range(0, len(pending), concurrency * 4):
            if stop_requested():
                stopped = True
                break
            list(pool.map(run, pending[start:start + concurrency * 4]))

    logger.info(f"tts pass: {counters['ok']} ok, {counters['failed']} failed, "
                f"{(time.monotonic() - started) / 60:.1f} min")
    if stopped:
        logger.warning("STOP flag seen, parking")
        sys.exit(20)


@cli.command("stt")
@click.option("--sentences", "depth", default=2, show_default=True, help="Sentences per voice this pass.")
@click.option("--url", default="http://localhost:8002", show_default=True, help="ASR base URL.")
@click.option("--api-key", default=None, envvar="CRISPASR_API_KEY", help="ASR API key (env: CRISPASR_API_KEY).")
@click.option("--model", default=None, envvar="CRISPASR_MODEL", help="ASR model name (env: CRISPASR_MODEL).")
@click.option("--language", default="fr", show_default=True, help="Language hint.")
@click.option("--concurrency", default=6, show_default=True, help="Parallel transcriptions.")
@click.option("--include-bases/--no-include-bases", default=True, show_default=True,
              help="Score the shipped voices too, as controls.")
@click.option("--limit", default=0, show_default=True, help="Stop after N rows this pass (0 = all pending).")
def cmd_stt(depth: int, url: str, api_key: str | None, model: str | None, language: str,
            concurrency: int, include_bases: bool, limit: int) -> None:
    """Transcribe and score every synthesized row that has no score yet."""
    sentences = load_sentences()
    voices = load_voices(include_bases=include_bases)
    state = read_ledger()
    by_sid = {sentence["sid"]: sentence["text"] for sentence in sentences}

    pending, missing = [], 0
    for voice, sentence in grid(voices=voices, sentences=sentences, depth=depth):
        sid = sentence["sid"]
        row = state.get((voice, sid), {})
        if not (row.get("wav") and wav_for(voice, sid).is_file()):
            missing += 1
            continue
        if "wer" not in row:
            pending.append((voice, sid))
    logger.info(f"{len(pending)} takes to score, {missing} not synthesized yet")
    if not pending:
        if missing:
            logger.info("nothing to score, the tts pass owns the rest")
            sys.exit(11)
        logger.success("converged: every row has a wav and a score")
        sys.exit(10)
    if limit:
        pending = pending[:limit]

    counters = {"ok": 0, "failed": 0, "rejected": 0}
    started = time.monotonic()

    def run(job: tuple[str, str]) -> None:
        voice, sid = job
        wav = wav_for(voice, sid)
        samples, rate = read_wav(wav)
        heard = transcribe(wav, url=url, api_key=api_key, model=model, language=language)
        if not heard:
            # An empty transcript is either a server error or genuinely silent audio, and
            # scoring it as 100 pct would permanently condemn the voice on a transport
            # failure. Record the attempt without a score so a later pass retries it.
            counters["failed"] += 1
            append_ledger({"voice": voice, "sid": sid, "stt_error": "empty transcript"})
            return
        wer = word_error_rate(by_sid[sid], heard)
        counters["ok"] += 1
        counters["rejected"] += wer > WER_REJECT
        append_ledger({"voice": voice, "sid": sid, "transcript": heard, "wer": round(wer, 4),
                       "f0": round(median_f0(samples, rate), 1),
                       "seconds": round(len(samples) / rate, 2)})
        if counters["ok"] % 50 == 0:
            rate_per_min = counters["ok"] / max(time.monotonic() - started, 1e-9) * 60
            logger.info(f"{counters['ok']}/{len(pending)} scored, {counters['rejected']} over "
                        f"{100 * WER_REJECT:.0f} pct WER, {counters['failed']} failed, "
                        f"{rate_per_min:.0f}/min")

    stopped = False
    with ThreadPoolExecutor(max_workers=concurrency) as pool:
        for start in range(0, len(pending), concurrency * 8):
            if stop_requested():
                stopped = True
                break
            list(pool.map(run, pending[start:start + concurrency * 8]))

    logger.info(f"stt pass: {counters['ok']} scored ({counters['rejected']} rejected), "
                f"{counters['failed']} failed, {(time.monotonic() - started) / 60:.1f} min")
    if stopped:
        logger.warning("STOP flag seen, parking")
        sys.exit(20)


@cli.command("report")
@click.option("--out", default=None, help="Where to write the markdown (default: <stage>/report.md).")
@click.option("--include-bases/--no-include-bases", default=True, show_default=True,
              help="Include the shipped voices as controls.")
def cmd_report(out: str | None, include_bases: bool) -> None:
    """Summarise the ledger: who passes the gate, grouped by kind, with the controls.

    Runs offline and mid-sweep, so progress can be checked without touching the GPU.
    Deliberately NOT a ranking by WER: see the module docstring. Distinctness is the
    axis that decides what to keep, and `voice_distance.py` owns it.
    """
    pool = {entry["name"]: entry for entry in json.loads((STAGE_DIR / "pool.json").read_text())}
    state = read_ledger()
    voices = load_voices(include_bases=include_bases)

    rows = []
    for voice in voices:
        scored = [record for (name, _), record in state.items() if name == voice and "wer" in record]
        if not scored:
            continue
        wers = sorted(record["wer"] for record in scored)
        # statistics.median and not wers[len(wers) // 2]: at an even number of takes the
        # latter is the UPPER middle, so at the default depth of 2 it silently reports the
        # worse of the two takes as the voice's typical score.
        entry = pool.get(voice, {"kind": "base", "parents": []})
        rows.append({
            "voice": voice,
            "kind": entry["kind"],
            "parents": "+".join(entry["parents"]),
            "takes": len(scored),
            # The median is the headline and the worst take is shown next to it: a voice
            # that reads one sentence cleanly and garbles another is not usable, and a
            # mean would hide that.
            "median_wer": statistics.median(wers),
            "worst_wer": wers[-1],
            "f0": sum(record.get("f0", 0.0) for record in scored) / len(scored),
            "seconds": sum(record.get("seconds", 0.0) for record in scored) / len(scored),
        })

    passed = [row for row in rows if row["median_wer"] <= WER_REJECT]
    by_kind: dict[str, list] = {}
    for row in rows:
        by_kind.setdefault(row["kind"], []).append(row)

    lines = ["# A6 screening report", "",
             f"{len(rows)} voices scored, {len(passed)} pass the gate "
             f"(median WER <= {100 * WER_REJECT:.0f} pct).", ""]
    for kind, group in sorted(by_kind.items()):
        kept = [row for row in group if row["median_wer"] <= WER_REJECT]
        lines.append(f"- **{kind}**: {len(kept)} of {len(group)} pass")
    lines += ["", "Intelligibility is a gate, not a ranking: the lowest-WER voice is the",
              "blandest one. Pick among the passers by distance, not by this table.", "",
              "| voice | kind | parents | takes | median WER | worst WER | F0 (Hz) | s |",
              "|---|---|---|---|---|---|---|---|"]
    for row in sorted(rows, key=lambda item: (item["kind"], item["median_wer"])):
        lines.append(f"| {row['voice']} | {row['kind']} | {row['parents']} | {row['takes']} | "
                     f"{100 * row['median_wer']:.0f}% | {100 * row['worst_wer']:.0f}% | "
                     f"{row['f0']:.0f} | {row['seconds']:.1f} |")

    destination = Path(out) if out else STAGE_DIR / "report.md"
    destination.write_text("\n".join(lines) + "\n")
    logger.success(f"{len(passed)} of {len(rows)} voices pass, wrote {destination}")


if __name__ == "__main__":
    cli()
