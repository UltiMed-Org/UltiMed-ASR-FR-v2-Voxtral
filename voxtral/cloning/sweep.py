#!/usr/bin/env -S uv run --script
# /// script
# requires-python = ">=3.10"
# dependencies = ["click", "loguru", "tqdm"]
# ///
"""Drive Voxtral voices over a (voice, sentence) grid and score them, resumably.

A5 auditioned 20 voices by hand. A6 generates about 420 candidates, which is past
what anyone can listen to, so the ears have to be spent on survivors only. This
script is the unattended half of that: it drives synthesis and scoring over a
(voice, sentence) grid, writing every completed unit of work to an append-only
ledger so that a run killed at any moment resumes where it stopped.

The same machinery serves two different questions, so it is parameterised by a RUN
(the `CLONING_RUN` environment variable, default "A6") that names a directory under
`cache/cloning/` holding that run's sentences, wavs, ledger and reports:

    A6      screen: 420 candidates x 12 hand-filtered Wikipedia lines, `sentences`
            picks the texts, `report` applies a pass/fail intelligibility gate.
    FLEURS  benchmark: the survivors x the whole French FLEURS test split, `fleurs`
            fetches the texts, `stats` reports each voice's WER and CER distribution.

Screening asks "is this voice broken", benchmarking asks "how well does it read
French", and they want opposite things from their corpus: a screen filters hard
sentences out to keep the measurement about voices, a benchmark must not filter at
all or its number stops meaning what its name says.

Two GPU passes, never at the same time, because one 24 GB card cannot hold the
Voxtral stack and the whisper stack together:

    tts   for every pending row, synthesize one wav        (voxtral-tts up)
    stt   for every un-scored wav, transcribe and score it (crispasr up)

`driver.sh` alternates them and owns the container switching. The other commands are
offline, need no GPU and no server, and can all be run mid-sweep: `sentences` and
`fleurs` pick a run's texts, `report` applies the screening gate, `stats` reports the
WER and CER distributions plus a per-sentence CSV.

On caching, because it is the first thing anyone asks when a benchmark takes hours:
the ledger *is* the cache, and it is the only one that matters. A take that already
has a wav is never re-synthesized and a take that already has a transcript is never
re-transcribed, so rerunning a finished benchmark costs seconds and adding voices or
sentences costs only the new rows. There is deliberately no server-side cache: vLLM
prefix caching is off for stage 0 and must stay off, because a named voice reaches
the model as out-of-band `additional_information` while its prefix occupies a run of
identical placeholder token ids, so two different voices reading the same sentence
produce byte-identical `prompt_token_ids`. Turning prefix caching on would serve one
voice's KV cache to another and silently return the wrong speaker. It would also buy
almost nothing: the prefix is a couple of hundred tokens against several seconds of
autoregressive decoding per take.

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

import csv
import json
import os
import random
import re
import statistics
import sys
import time
import urllib.request
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import click
from loguru import logger
from tqdm import tqdm

REPO_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO_ROOT))
import tts  # noqa: E402 (needs REPO_ROOT on the path first)

sys.path.insert(0, str(Path(__file__).resolve().parent))
from voice_survey import character_error_rate, transcribe, word_error_rate  # noqa: E402
from wav_stats import median_f0, read_wav  # noqa: E402

# A run is one (voice set, sentence set) pair with its own sentences, wavs, ledger and
# reports: "A6" is the 420-candidate screen on 12 Wikipedia sentences, "FLEURS" is the
# full French FLEURS benchmark over the survivors. Selected by environment variable
# rather than by a --run flag so `driver.sh` sets it once for both passes and every
# module that imports STAGE_DIR from here (voice_distance.py) follows automatically.
RUN = os.environ.get("CLONING_RUN", "A6")
STAGE_DIR = REPO_ROOT / "cache" / "cloning" / RUN
# The candidate pool is a property of the voices, not of a run: derive_voices.py mints it
# once into the A6 directory, and every later run reads that same file rather than
# carrying its own copy out of sync.
POOL_PATH = REPO_ROOT / "cache" / "cloning" / "A6" / "pool.json"
FLEURS_TSV = "https://huggingface.co/datasets/google/fleurs/resolve/main/data/{lang}/{split}.tsv"
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


def restrict(voices: list[str], only_file: str | None) -> list[str]:
    """Narrow a voice list to the names in a file, preserving order.

    The point is cheap depth. Screening 440 voices on two sentences answers "which are
    broken"; choosing between the survivors needs more takes each, and running the whole
    pool deeper wastes hours on candidates already eliminated. A file of finalists keeps
    the same ledger, the same grid and the same resume logic.

    Parameters
    ----------
    voices : list of str
        Candidate names.
    only_file : str, optional
        Path to a newline-separated list of names (blank lines and # comments ignored).

    Returns
    -------
    list of str
        The intersection, in the original order.
    """
    if not only_file:
        return voices
    wanted = {line.strip() for line in Path(only_file).read_text().splitlines()
              if line.strip() and not line.startswith("#")}
    missing = wanted - set(voices)
    if missing:
        logger.warning(f"{len(missing)} name(s) in {only_file} are not in the pool, e.g. {sorted(missing)[:3]}")
    return [voice for voice in voices if voice in wanted]


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
    path = POOL_PATH
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


def at_depth(sentences: list[dict], depth: int) -> list[dict]:
    """Cut the sentence list down to the depth this pass is asked for.

    Parameters
    ----------
    sentences : list of dict
        The run's full sentence list, in its stored order.
    depth : int
        Sentences per voice. 0 or less means every sentence, which is what a benchmark
        run wants ("read all of FLEURS") as opposed to a screen ("2 sentences each,
        deepen later").

    Returns
    -------
    list of dict
        A prefix of `sentences`.
    """
    return sentences if depth <= 0 else sentences[:depth]


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
    return [(voice, sentence) for sentence in at_depth(sentences, depth) for voice in voices]


def run_pool(*, jobs: list, work, concurrency: int, chunk: int, desc: str) -> bool:
    """Run `work` over `jobs` in a thread pool, with a progress bar and STOP handling.

    Chunked submission rather than one map over everything is the only way a thread pool
    can honour the STOP flag promptly without cancelling work already in flight. Both GPU
    passes go through here so the stop semantics exist in exactly one place, and so the
    two bars look the same to whoever is watching the log.

    Parameters
    ----------
    jobs : list
        Work items, handed to `work` one at a time.
    work : callable
        Called with a single job. Recording results and counting failures is its own
        business; an exception here kills the pass, which is intended.
    concurrency : int
        Worker threads.
    chunk : int
        Jobs submitted between two STOP checks. Larger keeps the pool busy, smaller parks
        sooner: a few multiples of the concurrency is the useful range.
    desc : str
        Progress bar label.

    Returns
    -------
    bool
        True when the pass parked on the STOP flag before finishing every job.
    """
    stopped = False
    # `pool.map` yields in submission order, so the bar can lag a completed job or two
    # behind reality. The total and the ETA are still right, which is what it is for.
    with tqdm(total=len(jobs), desc=desc, unit="take", smoothing=0.02) as bar:
        with ThreadPoolExecutor(max_workers=concurrency) as pool:
            for start in range(0, len(jobs), chunk):
                if stop_requested():
                    stopped = True
                    break
                for _ in pool.map(work, jobs[start:start + chunk]):
                    bar.update(1)
    return stopped


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
    """Screen and benchmark Voxtral voices for French (stages A6 and FLEURS)."""
    logger.remove()
    # Written through tqdm.write, not straight to stderr: a bare write from a worker
    # thread would be printed on top of the progress bar and leave it duplicated.
    logger.add(lambda message: tqdm.write(message, end=""),
               format="{time:HH:mm:ss} <level>{level: <7}</level> {message}", level="INFO",
               colorize=True)


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


@cli.command("fleurs")
@click.option("--lang", default="fr_fr", show_default=True, help="FLEURS language directory.")
@click.option("--split", default="test", show_default=True, help="FLEURS split: test, dev or train.")
@click.option("--limit", default=0, show_default=True, help="Keep only the first N sentences (0 = all).")
@click.option("--seed", default=0, show_default=True, help="Seed for the sentence shuffle.")
def cmd_fleurs(lang: str, split: str, limit: int, seed: int) -> None:
    """Write this run's sentence list from FLEURS, for a benchmark rather than a screen.

    `sentences` mints a dozen hand-filtered Wikipedia lines to answer "is this voice
    broken". This command takes the whole of a FLEURS split instead, to answer "how well
    does this voice read French", on material that has nothing to do with how the voices
    were built and that everyone else also reports numbers on.

    The TSV is the entire dataset text (about 50 KB) and is the only part of FLEURS
    needed here: the reference audio is irrelevant, the benchmark scores synthesized
    speech against the reference *text*. It is cached in the run directory, so reruns and
    a machine with no network both work.

    Nothing here filters the sentences. A screen may reject a hard line to keep the
    measurement about voices; a benchmark may not, or the number stops meaning what its
    name says. The known rough edge is that `voice_survey.fold_numbers` spells French
    integers only up to 999, so a reference "1800" is matched literally against whatever
    the ASR wrote. It affects the same 70 of 332 sentences for every voice, so it moves
    the absolute rate slightly and the ranking not at all.

    Parameters
    ----------
    lang : str
        FLEURS language directory, e.g. "fr_fr".
    split : str
        Which split to read: "test", "dev" or "train".
    limit : int
        Keep only the first N sentences after the shuffle. 0 keeps every one.
    seed : int
        Shuffle seed. The shuffle matters because FLEURS ids are grouped by source
        article: reading them in id order would make an interrupted run a handful of
        topics, while a shuffled prefix is a fair sample of the whole split.
    """
    STAGE_DIR.mkdir(parents=True, exist_ok=True)
    cached = STAGE_DIR / f"fleurs-{lang}-{split}.tsv"
    if not cached.is_file():
        url = FLEURS_TSV.format(lang=lang, split=split)
        logger.info(f"fetching {url}")
        with urllib.request.urlopen(url, timeout=120) as response:  # noqa: S310 (fixed host)
            cached.write_bytes(response.read())
        logger.info(f"cached {cached.stat().st_size / 1024:.0f} KB at {cached}")

    # FLEURS ships one row per *recording* and most sentences were read by up to three
    # speakers, so the file holds about twice as many rows as texts. The benchmark reads
    # each text once, hence the dedupe on the sentence id in column 0.
    texts: dict[int, str] = {}
    with cached.open(encoding="utf-8", newline="") as handle:
        for row in csv.reader(handle, delimiter="\t"):
            if len(row) < 3 or not row[0].isdigit():
                continue
            # Column 2 is `raw_transcription`, not column 3's normalised `transcription`:
            # it keeps capitalisation, punctuation and digits, which is what a TTS model
            # needs in order to phrase a sentence at all. Scoring is unaffected because
            # `voice_survey.normalise` strips all three from both sides first.
            texts.setdefault(int(row[0]), row[2].strip())

    # Ids are the stable part, so a sid survives re-running this command, changing the
    # limit, or shuffling with another seed. Rows already in the ledger stay valid.
    sentences = [{"sid": f"f{key}", "text": texts[key]} for key in sorted(texts)]
    random.Random(seed).shuffle(sentences)
    if limit:
        sentences = sentences[:limit]

    destination = STAGE_DIR / "sentences.json"
    destination.write_text(json.dumps(sentences, indent=2, ensure_ascii=False) + "\n")
    words = sum(len(entry["text"].split()) for entry in sentences)
    logger.success(f"{len(sentences)} sentences ({words} words, "
                   f"{words / max(len(sentences), 1):.0f} per sentence) to {destination}")


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
@click.option("--only-file", default=None, help="Restrict to the voices named in this file (one per line).")
def cmd_tts(depth: int, url: str, concurrency: int, timeout: float, noise_seed: int,
            euler_steps: int, include_bases: bool, limit: int, only_file: str | None) -> None:
    """Synthesize every pending row of the grid."""
    sentences = load_sentences()
    voices = restrict(load_voices(include_bases=include_bases), only_file)
    state = read_ledger()
    rows = grid(voices=voices, sentences=sentences, depth=depth)

    def synthesized(voice: str, sid: str) -> bool:
        """Whether this row still needs audio.

        A row counts as done when the ledger says so AND the wav is still on disk, so
        deleting a take is enough to ask for it again. An already-scored row is the
        exception: once a take has a WER the audio has served its purpose, and deleting
        wavs to reclaim space (a full FLEURS run is about 7 GB) must not trigger hours of
        resynthesis for work whose result is already on record.
        """
        row = state.get((voice, sid), {})
        if "wer" in row:
            return True
        return bool(row.get("wav")) and wav_for(voice, sid).is_file()

    pending = [(voice, sentence) for voice, sentence in rows
               if not synthesized(voice, sentence["sid"])]
    logger.info(f"{len(rows)} rows in the grid ({len(voices)} voices x "
                f"{len(at_depth(sentences, depth))} sentences), {len(pending)} to synthesize")
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

    stopped = run_pool(jobs=pending, work=run, concurrency=concurrency,
                       chunk=concurrency * 4, desc=f"tts {RUN}")
    per_call = counters["seconds"] / max(counters["ok"], 1)
    logger.info(f"tts pass: {counters['ok']} ok, {counters['failed']} failed, "
                f"{per_call:.1f}s per call, {(time.monotonic() - started) / 60:.1f} min")
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
@click.option("--only-file", default=None, help="Restrict to the voices named in this file (one per line).")
def cmd_stt(depth: int, url: str, api_key: str | None, model: str | None, language: str,
            concurrency: int, include_bases: bool, limit: int, only_file: str | None) -> None:
    """Transcribe and score every synthesized row that has no score yet."""
    sentences = load_sentences()
    voices = restrict(load_voices(include_bases=include_bases), only_file)
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
        # CER is stored next to WER rather than recomputed by the report, because the
        # transcript that produced it is the one thing a rerun cannot reproduce exactly.
        cer = character_error_rate(by_sid[sid], heard)
        counters["ok"] += 1
        counters["rejected"] += wer > WER_REJECT
        append_ledger({"voice": voice, "sid": sid, "transcript": heard, "wer": round(wer, 4),
                       "cer": round(cer, 4),
                       "f0": round(median_f0(samples, rate), 1),
                       "seconds": round(len(samples) / rate, 2)})

    stopped = run_pool(jobs=pending, work=run, concurrency=concurrency,
                       chunk=concurrency * 8, desc=f"stt {RUN}")
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
    pool = {entry["name"]: entry for entry in json.loads(POOL_PATH.read_text())}
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


def summarise(values: list[float]) -> dict[str, float]:
    """Spread of one voice's per-sentence error rates.

    Parameters
    ----------
    values : list of float
        Error rates, one per sentence, in any order.

    Returns
    -------
    dict
        Count, min, p25, median, mean, p75, max and sample standard deviation.

    Notes
    -----
    The quartiles use `method="inclusive"`, the definition that places p25 and p75 at
    interpolated positions inside the sample rather than extrapolating past its ends.
    With a single take there is nothing to spread, so every statistic collapses to that
    value and the stdev is reported as 0 rather than raising.
    """
    ordered = sorted(values)
    if len(ordered) > 1:
        quarters = statistics.quantiles(ordered, n=4, method="inclusive")
        spread = statistics.stdev(ordered)
    else:
        quarters, spread = [ordered[0]] * 3, 0.0
    return {"n": len(ordered), "min": ordered[0], "p25": quarters[0],
            "median": statistics.median(ordered), "mean": statistics.fmean(ordered),
            "p75": quarters[2], "max": ordered[-1], "stdev": spread}


@cli.command("stats")
@click.option("--out", default=None, help="Where to write the markdown (default: <stage>/stats.md).")
@click.option("--csv-out", default=None,
              help="Where to write the per-sentence rows (default: <stage>/per_sentence.csv).")
@click.option("--include-bases/--no-include-bases", default=True, show_default=True,
              help="Include the shipped voices as controls.")
@click.option("--only-file", default=None, help="Restrict to the voices named in a file.")
def cmd_stats(out: str | None, csv_out: str | None, include_bases: bool,
              only_file: str | None) -> None:
    """Per-voice WER and CER distributions over everything scored so far.

    Offline: reads the ledger, touches no GPU and no server, so it can be run against a
    benchmark that is still hours from finishing. Voices are compared on their *shared*
    sentences only, because a voice asked 300 questions and one asked 30 are not
    otherwise comparable, and a partial run always has both.

    Unlike `report`, which is a pass/fail gate, this one is a ranking: on a fixed, neutral
    corpus the whole distribution of a voice's error rate is the thing of interest, and
    the tail (p75, max) says more about whether a voice can be trusted to generate a
    dataset than the median does. A voice that reads 90 pct of FLEURS perfectly and
    detonates on the rest still ships broken audio.
    """
    sentences = load_sentences()
    voices = restrict(load_voices(include_bases=include_bases), only_file)
    wanted = set(voices)
    by_sid = {entry["sid"]: entry["text"] for entry in sentences}
    state = read_ledger()

    scored: dict[str, dict[str, dict]] = {}
    for (voice, sid), record in state.items():
        if voice not in wanted or sid not in by_sid or "wer" not in record:
            continue
        scored.setdefault(voice, {})[sid] = record
    if not scored:
        raise SystemExit(f"nothing scored yet in {STAGE_DIR}")

    # A take scored before CER existed has a WER and no CER. Deriving the missing rate
    # needs nothing but the transcript already on record, so it is repaired here rather
    # than by replaying a GPU pass, and written back so the next run of this command is
    # free. This is also why it lives in the offline report and not in `stt`: it must work
    # for a take whose wav has since been deleted.
    backfilled = 0
    for voice, takes in scored.items():
        for sid, record in takes.items():
            if "cer" in record or not record.get("transcript"):
                continue
            record["cer"] = round(character_error_rate(by_sid[sid], record["transcript"]), 4)
            append_ledger({"voice": voice, "sid": sid, "cer": record["cer"]})
            backfilled += 1
    if backfilled:
        logger.info(f"{backfilled} take(s) got a CER back-filled from their stored transcript")

    # The common set is what makes the table a comparison rather than a list. It is
    # recomputed here rather than assumed, because a benchmark is normally read while it
    # is still running and the voices are never exactly level.
    common = set.intersection(*(set(rows) for rows in scored.values()))
    ragged = {voice: len(rows) for voice, rows in scored.items() if len(rows) != len(common)}

    csv_path = Path(csv_out) if csv_out else STAGE_DIR / "per_sentence.csv"
    with csv_path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.writer(handle)
        writer.writerow(["voice", "sid", "wer", "cer", "seconds", "f0", "words",
                         "reference", "transcript"])
        for voice in sorted(scored):
            for sid in sorted(scored[voice]):
                record = scored[voice][sid]
                writer.writerow([voice, sid, record.get("wer", ""), record.get("cer", ""),
                                 record.get("seconds", ""), record.get("f0", ""),
                                 len(by_sid[sid].split()), by_sid[sid],
                                 record.get("transcript", "")])

    rows = []
    for voice, takes in scored.items():
        shared = [takes[sid] for sid in common]
        cers = [record["cer"] for record in shared if "cer" in record]
        rows.append({"voice": voice,
                     "wer": summarise([record["wer"] for record in shared]),
                     "cer": summarise(cers) if cers else None,
                     "seconds": sum(record.get("seconds", 0.0) for record in shared)})
    rows.sort(key=lambda row: row["wer"]["median"])
    # Surfaced because it is the one artifact that could make a slow voice look broken
    # rather than merely slow: whisper works in 30 s windows, FLEURS runs to 65 words, and
    # the slowest keeper reads at 1 word a second, which is a 66 s take. Measured before
    # trusting it: a 48.5 s wav (12 A6 takes concatenated, 136 words) sent as one request
    # came back complete, last sentence included, at 4.9 pct WER and 1.4 pct CER, so the
    # backend chunks long audio instead of truncating at the window. Kept in the report
    # anyway, because that guarantee belongs to the ASR backend and not to this script.
    longest = max(((record.get("seconds", 0.0), voice, sid)
                   for voice, takes in scored.items() for sid, record in takes.items()),
                  default=(0.0, "", ""))

    header = "| voice | takes | min | p25 | median | mean | p75 | max | stdev |"
    rule = "|---|---|---|---|---|---|---|---|---|"

    def table(metric: str) -> list[str]:
        lines = [header, rule]
        for row in rows:
            stats = row[metric]
            if stats is None:
                lines.append(f"| {row['voice']} | - | - | - | - | - | - | - | - |")
                continue
            lines.append(f"| {row['voice']} | {stats['n']} | " + " | ".join(
                f"{100 * stats[key]:.1f}%" for key in
                ("min", "p25", "median", "mean", "p75", "max", "stdev")) + " |")
        return lines

    audio_hours = sum(row["seconds"] for row in rows) / 3600
    lines = [f"# {RUN}: WER and CER per voice", "",
             f"{len(rows)} voices, compared on the {len(common)} sentences every one of them "
             f"has read ({len(by_sid)} in the run, {audio_hours:.1f} h of audio scored).",
             "",
             "All figures are percentages. WER is Levenshtein over words, CER over "
             "characters, both after the same normalisation (case, accents, punctuation "
             "and spelled-out numbers folded on both sides). A large WER with a small CER "
             "means mangled word endings; the two moving together means the voice read "
             "different words. Rows are ordered by median WER, and the CER table keeps "
             "that order so the two can be read side by side.", "",
             f"Longest take: {longest[0]:.1f} s (`{longest[1]}` on `{longest[2]}`)."
             + (" Past whisper's 30 s window; the backend was measured to chunk rather than"
                " truncate, but a take this long is still worth one listen."
                if longest[0] > 30 else ""), ""]
    if ragged:
        lines += [f"Note: {len(ragged)} voice(s) have a different number of takes than the "
                  f"common set and were trimmed to it for this table, e.g. "
                  + ", ".join(f"`{voice}` ({count})" for voice, count in
                              sorted(ragged.items())[:4]) + ".", ""]
    lines += ["## Word error rate", ""] + table("wer")
    lines += ["", "## Character error rate", ""] + table("cer")
    lines += ["", f"Per-sentence detail, one row per (voice, sentence): `{csv_path}`.", ""]

    destination = Path(out) if out else STAGE_DIR / "stats.md"
    destination.write_text("\n".join(lines) + "\n")
    logger.success(f"{len(rows)} voices over {len(common)} shared sentences, "
                   f"wrote {destination} and {csv_path}")


if __name__ == "__main__":
    cli()
