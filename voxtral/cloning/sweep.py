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
# once and every later run reads that same file rather than carrying its own copy out of
# sync. Which pool, though, IS a choice, so `CLONING_POOL` names the directory holding it
# (default A6, the 420-candidate screen pool that A6 and A7 were both measured against).
# Without this knob a second, larger pool minted in place would silently hand every
# finished run a set of voices it never benchmarked, and those runs would report
# themselves unconverged forever.
POOL = os.environ.get("CLONING_POOL", "A6")
POOL_PATH = REPO_ROOT / "cache" / "cloning" / POOL / "pool.json"
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


def load_pool() -> dict[str, dict]:
    """The derived-voice recipes, keyed by voice name.

    The pool is written once by `derive_voices.py` and read by every command that needs
    to know how a voice was built, including the ones in `voice_distance.py`. It lives
    under the A6 run rather than the current one because a later run (a benchmark, say)
    reuses A6's voices without re-deriving them, so `CLONING_RUN` must not move it.

    Returns
    -------
    dict
        `{name: {"name", "kind", "parents", ...}}`, empty if no pool has been built yet.
        Callers that cannot work without it should say so themselves; the common case
        here is a report that simply treats every voice as a shipped base.
    """
    if not POOL_PATH.is_file():
        return {}
    return {entry["name"]: entry for entry in json.loads(POOL_PATH.read_text())}


def ledger_path() -> Path:
    """Where the append-only record of finished work lives.

    Returns
    -------
    Path
        The ledger file (may not exist yet).
    """
    return STAGE_DIR / "ledger.jsonl"


def iter_ledger():
    """Yield every well-formed record in the ledger, oldest first.

    A half-written final line from a `kill -9` is skipped rather than aborting the run,
    which is the whole reason the ledger is append-only JSON lines and not a database.

    Yields
    ------
    dict
        One record as it was written.
    """
    path = ledger_path()
    if not path.is_file():
        return
    for line in path.read_text(errors="replace").splitlines():
        line = line.strip()
        if not line:
            continue
        try:
            yield json.loads(line)
        except json.JSONDecodeError:
            logger.warning("skipping a truncated ledger line (interrupted write?)")


def read_ledger() -> dict[tuple[str, str], dict]:
    """Replay the ledger into the current state of each row.

    Append-only with last-write-wins per key, so the several records a row accumulates
    (a wav, then a transcript and a score) collapse into one.

    Attempts are the exception to the merge. When a take is re-synthesized at a new noise
    seed, its new record carries a higher `attempt` and REPLACES the row instead of
    merging into it. Merging would leave the previous attempt's WER sitting next to the
    new attempt's wav, and the passes would then believe a freshly synthesized take was
    already scored. A record without an `attempt` is attempt 0, which is what every row
    written before retries existed is.

    Returns
    -------
    dict
        `{(voice, sid): {...}}`, holding the latest attempt only. Use `read_attempts`
        when the earlier ones matter, which they do for any honest report.
    """
    state: dict[tuple[str, str], dict] = {}
    for record in iter_ledger():
        key = (record["voice"], record["sid"])
        current = state.get(key)
        if current is not None and record.get("attempt", 0) > current.get("attempt", 0):
            state[key] = dict(record)
        else:
            state.setdefault(key, {}).update(record)
    return state


def read_attempts() -> dict[tuple[str, str], list[dict]]:
    """Replay the ledger into every attempt of each row, in order.

    The benchmark needs this and `read_ledger` is not enough for it: once a take can be
    retried, reporting only the surviving attempt turns the measurement into best-of-N and
    quietly flatters every voice. Keeping the whole history lets a report say explicitly
    which attempt it is quoting.

    Returns
    -------
    dict
        `{(voice, sid): [attempt 0, attempt 1, ...]}`, each entry merged the same way
        `read_ledger` merges a single attempt.
    """
    rows: dict[tuple[str, str], dict[int, dict]] = {}
    for record in iter_ledger():
        key = (record["voice"], record["sid"])
        rows.setdefault(key, {}).setdefault(record.get("attempt", 0), {}).update(record)
    return {key: [attempts[index] for index in sorted(attempts)] for key, attempts in rows.items()}


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


def sentence_seed(sentence: dict) -> int:
    """Noise seed offset for one sentence.

    Normally derived from the sentence id, so every voice reading a given sentence draws
    the same acoustic noise and two takes differ only by the prefix that generated them.

    A sentence may pin an explicit `seed` instead, which is what the `probe` variants
    need: they hold the seed still while the *text* changes, and the id cannot express
    that because each variant needs its own id to have its own ledger row.

    Parameters
    ----------
    sentence : dict
        Entry from the run's sentence list.

    Returns
    -------
    int
        Offset added to the pass's base noise seed.
    """
    if "seed" in sentence:
        return int(sentence["seed"])
    return int(sentence["sid"][1:])


def wav_for(voice: str, sid: str, attempt: int = 0) -> Path:
    """Where one take lives.

    Parameters
    ----------
    voice : str
        Voice name.
    sid : str
        Sentence id.
    attempt : int, optional
        Which attempt. Attempt 0 keeps the plain `<sid>.wav` name, so the thousands of
        takes that exist from before retries were added stay exactly where they are, and
        later attempts land beside it as `<sid>.a1.wav`. The earlier audio is never
        overwritten: a retry that came out worse has to remain listenable, or there is no
        way to check the retry rule itself.

    Returns
    -------
    Path
        Wav path, inside a per-voice directory so a voice can be deleted as a unit.
    """
    suffix = ".wav" if attempt == 0 else f".a{attempt}.wav"
    return STAGE_DIR / "wav" / voice / f"{sid}{suffix}"


# Offset between the noise seeds of consecutive attempts. Large and odd so that attempt 1
# of one sentence cannot collide with attempt 0 of another: the per-sentence offsets are
# small integers, and a collision would silently make two rows the same audio.
ATTEMPT_SEED_STRIDE = 7919


def stretch_of(*, seconds: float, words: int, rate: float) -> float:
    """How much longer a take ran than the voice's own speaking rate predicts.

    The degeneration signal, and deliberately independent of the ASR: a take that runs
    three times longer than it should has stopped reading the sentence, and that is
    visible in the waveform without asking a second model anything. A5 measured a 3x
    spread in speaking rate across the shipped voices, so the comparison has to be against
    the voice's own rate rather than a fixed seconds-per-word.

    Parameters
    ----------
    seconds : float
        Duration of the take.
    words : int
        Words in the reference sentence.
    rate : float
        That voice's median words per second.

    Returns
    -------
    float
        1.0 when the take is exactly as long as expected, higher when it overran.
    """
    if words <= 0 or rate <= 0 or seconds <= 0:
        return 1.0
    return seconds / (words / rate)


def declared_identity(voice: str, pool: dict) -> str:
    """Which shipped speaker a candidate is supposed to sound like, by construction.

    This is the *claim*, read off the recipe, not a measurement: the similarity matrix
    exists to check it. A6 established that identity travels in the 36 acoustic codes, so
    a hybrid `h_<semantic>_<acoustic>` should sound like its acoustic donor (the last
    parent) and not at all like its semantic one. A sparse `x_` mix blends whole frames
    with a dominant first component rather than splitting semantic from acoustic, so its
    claim is that first parent. A shipped voice is its own identity.

    Parameters
    ----------
    voice : str
        Voice name.
    pool : dict
        `{name: entry}` from the candidate manifest.

    Returns
    -------
    str
        Name of the shipped voice this one is expected to resemble.
    """
    entry = pool.get(voice)
    if entry is None or not entry.get("parents"):
        return voice
    return entry["parents"][-1] if entry["kind"] == "hybrid" else entry["parents"][0]


def speaker_components(*, names: list[str], similarity: "np.ndarray",
                       index_of: dict[str, int], threshold: float) -> dict[str, list[str]]:
    """Group voices into speakers: connected components of the "at or above threshold" graph.

    Components rather than neighbourhoods, because cosine similarity is not transitive.
    `de_female` sits at 0.58 from `pt_female`, which sits at 0.97 from `h_frm_ptf`, so the
    neighbourhood of one kept voice overlaps another's and the same name ends up listed
    under two speakers. Components cannot do that. Merging a borderline chain is the
    conservative error here: it under-counts diversity rather than shipping two names for
    one speaker.

    Parameters
    ----------
    names : list of str
        Voices to cluster.
    similarity : numpy.ndarray
        Square cosine-similarity matrix.
    index_of : dict
        Voice name to its row in `similarity`.
    threshold : float
        At or above this cosine, two voices are the same speaker.

    Returns
    -------
    dict
        `{root_voice: [members]}`, one entry per speaker. The root is an arbitrary member
        and carries no meaning beyond identifying the cluster.
    """
    parent = {name: name for name in names}

    def find(name: str) -> str:
        """Root of this voice's cluster, with path compression."""
        while parent[name] != name:
            parent[name] = parent[parent[name]]
            name = parent[name]
        return name

    for position, a in enumerate(names):
        for b in names[position + 1:]:
            if similarity[index_of[a], index_of[b]] >= threshold:
                parent[find(a)] = find(b)

    components: dict[str, list[str]] = {}
    for name in names:
        components.setdefault(find(name), []).append(name)
    return components


RUNAWAY_FACTOR = 3.0


ATTEMPT_CHOICES = ("first", "last", "best")


def pick_attempt(attempts: list[dict], *, which: str) -> dict | None:
    """Choose which attempt of one take a report is about.

    Every table in this project has to name its convention, because once takes can be
    retried the same ledger supports three different honest answers and mixing them in one
    document is how a benchmark starts lying. `first` is the unaided voice, `best` is what
    a dataset generator that regenerates its own bad takes would end up containing, `last`
    is what is on disk to listen to.

    Parameters
    ----------
    attempts : list of dict
        Every attempt of one take, oldest first, as `read_attempts` returns them.
    which : str
        One of `ATTEMPT_CHOICES`.

    Returns
    -------
    dict or None
        The chosen attempt, or None when none of them was ever scored.
    """
    scored = [record for record in attempts if "wer" in record]
    if not scored:
        return None
    if which == "first":
        return scored[0]
    if which == "last":
        return scored[-1]
    return min(scored, key=lambda record: record["wer"])


def voice_profiles(*, only: set[str] | None = None,
                   runaway_factor: float = RUNAWAY_FACTOR,
                   which: str = "first") -> dict[str, dict]:
    """One robustness row per voice, straight from the ledger.

    Collapses every scored take into the four numbers that decide whether a voice can be
    used to generate a dataset: what it does on a typical sentence (median WER), what it
    costs on average once the disasters are included (mean WER), how often it degenerates
    (runaway share), and how fast it speaks. The last is not a quality measure, it is a
    diversity axis: two voices that are the same speaker still differ if one reads at 1.6
    and the other at 3.0 words per second.

    Lives here rather than in a command because `voice_distance.py` has to rank voices by
    robustness when it picks one representative per speaker cluster, and a second
    implementation of "runaway" would eventually disagree with `diagnose`.

    Parameters
    ----------
    only : set of str, optional
        Restrict to these voices. Default is every voice in the ledger.
    runaway_factor : float
        Duration multiple of the voice's own median speaking rate that counts as
        degenerate. Waveform-based, so it does not depend on the ASR being right.
    which : str
        Which attempt of a retried take to summarise, from `ATTEMPT_CHOICES`. Defaults to
        `first`, the unaided voice, so a roster built on this cannot quietly become a
        best-of-N ranking once the retry loop has run.

    Returns
    -------
    dict
        `{voice: {"takes", "median", "mean", "runaway", "rate"}}`, rates as fractions.
    """
    by_sid = {entry["sid"]: entry["text"] for entry in load_sentences()}
    takes = []
    for (voice, sid), attempts in read_attempts().items():
        if (only is not None and voice not in only) or sid not in by_sid:
            continue
        record = pick_attempt(attempts, which=which)
        if record is not None and record.get("seconds"):
            takes.append(dict(record, voice=voice, sid=sid))
    if not takes:
        return {}

    def words_of(sid: str) -> int:
        """Word count of a sentence."""
        return len(by_sid[sid].split())

    rates = voice_rates(takes, words_of=words_of)
    per: dict[str, list] = {}
    for take in takes:
        take["stretch"] = stretch_of(seconds=take["seconds"], words=words_of(take["sid"]),
                                     rate=rates[take["voice"]])
        per.setdefault(take["voice"], []).append(take)

    profiles = {}
    for voice, group in per.items():
        wers = [take["wer"] for take in group]
        profiles[voice] = {
            "takes": len(group),
            "median": statistics.median(wers),
            "mean": statistics.fmean(wers),
            "runaway": sum(1 for t in group if t["stretch"] >= runaway_factor) / len(group),
            "rate": rates[voice],
        }
    return profiles


def voice_rates(takes: list[dict], *, words_of) -> dict[str, float]:
    """Each voice's median speaking rate, in words per second.

    Parameters
    ----------
    takes : list of dict
        Scored records carrying `voice`, `sid` and `seconds`.
    words_of : callable
        Maps a sid to the word count of its sentence.

    Returns
    -------
    dict
        `{voice: words per second}`.
    """
    per: dict[str, list[float]] = {}
    for take in takes:
        if take.get("seconds"):
            per.setdefault(take["voice"], []).append(words_of(take["sid"]) / take["seconds"])
    return {voice: statistics.median(values) for voice, values in per.items()}


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

        `retry` schedules a re-synthesis by appending a record that bumps `attempt` and
        carries neither a wav nor a WER, so a retried row fails both tests here and comes
        back to this pass without any special case.
        """
        row = state.get((voice, sid), {})
        if "wer" in row:
            return True
        return bool(row.get("wav")) and wav_for(voice, sid, row.get("attempt", 0)).is_file()

    pending = [(voice, sentence, state.get((voice, sentence["sid"]), {}).get("attempt", 0))
               for voice, sentence in rows if not synthesized(voice, sentence["sid"])]
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

    def run(job: tuple[str, dict, int]) -> None:
        voice, sentence, attempt = job
        sid = sentence["sid"]
        # Same seed for every voice reading this sentence: the take then differs only by
        # the prefix, which is what makes cross-voice comparison (and a later speaker
        # embedding distance) mean anything. A retry moves the seed by a fixed stride, so
        # attempt N is reproducible rather than merely different, and every voice retrying
        # the same sentence still lands on the same seed as the others.
        seed = noise_seed + sentence_seed(sentence) + attempt * ATTEMPT_SEED_STRIDE
        # raw_spaces: the sweep is the instrument, so it sends the corpus byte for byte.
        # tts.py normalises the exotic Unicode spaces for ordinary callers (the PROBE run
        # is what established that it should), and letting it do so here would silently
        # rewrite the very text a benchmark is measuring, starting with the probe's own
        # `raw` variant.
        body = tts.build_speech_body(text=sentence["text"], response_format="wav", speed=1.0,
                                     voice=voice, noise_seed=seed, raw_spaces=True,
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
            append_ledger({"voice": voice, "sid": sid, "attempt": attempt,
                           "tts_error": f"HTTP {status}"})
            return
        destination = wav_for(voice, sid, attempt)
        destination.parent.mkdir(parents=True, exist_ok=True)
        destination.write_bytes(payload)
        counters["ok"] += 1
        counters["seconds"] += elapsed
        append_ledger({"voice": voice, "sid": sid, "attempt": attempt, "noise_seed": seed,
                       "wav": str(destination.relative_to(REPO_ROOT)),
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
        if not (row.get("wav") and wav_for(voice, sid, row.get("attempt", 0)).is_file()):
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
        attempt = state.get((voice, sid), {}).get("attempt", 0)
        wav = wav_for(voice, sid, attempt)
        samples, rate = read_wav(wav)
        heard = transcribe(wav, url=url, api_key=api_key, model=model, language=language)
        if not heard:
            # An empty transcript is either a server error or genuinely silent audio, and
            # scoring it as 100 pct would permanently condemn the voice on a transport
            # failure. Record the attempt without a score so a later pass retries it.
            counters["failed"] += 1
            append_ledger({"voice": voice, "sid": sid, "attempt": attempt,
                           "stt_error": "empty transcript"})
            return
        wer = word_error_rate(by_sid[sid], heard)
        # CER is stored next to WER rather than recomputed by the report, because the
        # transcript that produced it is the one thing a rerun cannot reproduce exactly.
        cer = character_error_rate(by_sid[sid], heard)
        counters["ok"] += 1
        counters["rejected"] += wer > WER_REJECT
        append_ledger({"voice": voice, "sid": sid, "attempt": attempt,
                       "transcript": heard, "wer": round(wer, 4),
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


@cli.command("retry")
@click.option("--sentences", "depth", default=2, show_default=True, help="Sentences per voice.")
@click.option("--max-attempts", default=3, show_default=True,
              help="Total attempts allowed per take, including the first.")
@click.option("--runaway-factor", default=RUNAWAY_FACTOR, show_default=True,
              help="Duration multiple of a voice's own speaking rate that counts as degenerate.")
@click.option("--on", "triggers", default="runaway", show_default=True,
              help="Comma-separated retry triggers: runaway, ceiling, empty, wer.")
@click.option("--wer-over", default=1.0, show_default=True,
              help="With the 'wer' trigger, retry above this rate. Read the warning below.")
@click.option("--include-bases/--no-include-bases", default=True, show_default=True,
              help="Consider the shipped voices too.")
@click.option("--only-file", default=None, help="Restrict to the voices named in a file.")
@click.option("--dry-run", is_flag=True, help="Report what would be retried, schedule nothing.")
def cmd_retry(depth: int, max_attempts: int, runaway_factor: float, triggers: str,
              wer_over: float, include_bases: bool, only_file: str | None, dry_run: bool) -> None:
    """Schedule a re-synthesis, at a different noise seed, of takes that came out broken.

    Voxtral degenerates on a small fraction of takes: it starts the sentence correctly and
    then loops nonsense syllables until the token cap. The audio is genuinely unusable, and
    since the only stochasticity in the model is the noise the flow-matching decoder starts
    from, another seed usually produces a clean take of the same sentence.

    This is offline and costs nothing: it appends a record bumping the row's `attempt`,
    which makes the tts pass pick it up again, and the driver then loops tts, stt and retry
    until no row is both broken and under budget. Earlier attempts are kept, wav and all.

    **What this must not be used for.** Retrying until a number looks good is best-of-N,
    and best-of-N is not a measurement: a voice that degenerates on a tenth of its takes
    would score like one that never does. That is why the default trigger is `runaway`
    and not `wer`. A runaway is identified from the waveform alone, without the ASR, and
    means the generation broke; retrying it repairs a defect. A high WER usually means the
    sentence was hard, which is a fact about the voice worth keeping. The `wer` trigger
    exists because dataset generation genuinely does want best-of-N (nobody ships audio
    they know is wrong), but it is opt-in, every attempt stays in the ledger, and
    `stats --attempt first` reports the honest number whatever this did.

    Exit codes match the driver contract: 0 when retries were scheduled, 11 when there is
    nothing to retry.

    Parameters
    ----------
    depth : int
        Sentences per voice, matching the passes.
    max_attempts : int
        Budget per take. 1 disables retrying entirely.
    runaway_factor : float
        Overrun multiple that counts as degeneration.
    triggers : str
        Which failures to retry: `runaway` (degenerate duration), `ceiling` (pinned at the
        token cap), `empty` (the ASR returned nothing), `wer` (see the warning above).
    wer_over : float
        Threshold for the `wer` trigger. The default of 1.0 is "the transcript is longer
        than the reference", which is a broken take rather than a hard sentence.
    include_bases : bool
        Whether the shipped voices are considered.
    only_file : str or None
        Restrict to the voices named in this file.
    dry_run : bool
        Report and schedule nothing.
    """
    wanted_triggers = {name.strip() for name in triggers.split(",") if name.strip()}
    unknown = wanted_triggers - {"runaway", "ceiling", "empty", "wer"}
    if unknown:
        raise SystemExit(f"unknown retry trigger(s): {sorted(unknown)}")

    sentences = load_sentences()
    voices = restrict(load_voices(include_bases=include_bases), only_file)
    wanted = set(voices)
    # Only rows inside the grid this run is working on, so raising the depth later cannot
    # be preempted by retries of sentences nobody has asked for yet.
    by_sid = {entry["sid"]: entry["text"] for entry in at_depth(sentences, depth)}

    def words_of(sid: str) -> int:
        """Word count of a sentence, as both the rate and the stretch measure it."""
        return len(by_sid[sid].split())

    state = read_ledger()

    scored = [dict(record, voice=voice, sid=sid) for (voice, sid), record in state.items()
              if voice in wanted and sid in by_sid and ("wer" in record or "stt_error" in record)]
    if not scored:
        logger.info("nothing scored yet, so nothing to judge")
        sys.exit(11)
    rates = voice_rates([take for take in scored if take.get("seconds")], words_of=words_of)

    broken: list[tuple[str, str, int, str]] = []
    exhausted = 0
    for take in scored:
        attempt = take.get("attempt", 0)
        reasons = []
        seconds = take.get("seconds") or 0.0
        if "runaway" in wanted_triggers and seconds and take["voice"] in rates:
            if stretch_of(seconds=seconds, words=words_of(take["sid"]),
                          rate=rates[take["voice"]]) >= runaway_factor:
                reasons.append("runaway")
        # 4096 tokens at 12.5 Hz: generation that never terminated and was cut off.
        if "ceiling" in wanted_triggers and seconds >= 327.0:
            reasons.append("ceiling")
        if "empty" in wanted_triggers and take.get("stt_error"):
            reasons.append("empty")
        if "wer" in wanted_triggers and take.get("wer", 0.0) > wer_over:
            reasons.append("wer")
        if not reasons:
            continue
        if attempt + 1 >= max_attempts:
            exhausted += 1
            continue
        broken.append((take["voice"], take["sid"], attempt + 1, "+".join(reasons)))

    logger.info(f"{len(scored)} scored takes, {len(broken)} to retry, "
                f"{exhausted} already out of attempts (budget {max_attempts})")
    if not broken:
        logger.info("nothing to retry")
        sys.exit(11)
    by_reason: dict[str, int] = {}
    for _, _, _, reason in broken:
        by_reason[reason] = by_reason.get(reason, 0) + 1
    for reason, count in sorted(by_reason.items(), key=lambda item: -item[1]):
        logger.info(f"  {count} x {reason}")
    if dry_run:
        for voice, sid, attempt, reason in broken[:20]:
            logger.info(f"  would retry {voice}/{sid} as attempt {attempt} ({reason})")
        sys.exit(11)

    for voice, sid, attempt, reason in broken:
        # Carries no wav and no wer on purpose: `read_ledger` replaces the row on a higher
        # attempt, so this is what makes the take pending again for both passes.
        append_ledger({"voice": voice, "sid": sid, "attempt": attempt, "retry_reason": reason})
    logger.success(f"scheduled {len(broken)} retry/retries; the tts pass will pick them up")


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
    pool = load_pool()
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
@click.option("--attempt", "which", default="first",
              type=click.Choice(ATTEMPT_CHOICES), show_default=True,
              help="Which attempt of a retried take to report. Read the docstring.")
def cmd_stats(out: str | None, csv_out: str | None, include_bases: bool,
              only_file: str | None, which: str) -> None:
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

    `--attempt` decides what a retried take contributes, and the three answers are three
    different questions:

    - `first` (the default) is what the model does unaided, which is the only honest
      benchmark number. Retrying and then reporting the survivor is best-of-N, and
      best-of-N would score a voice that degenerates on a tenth of its takes the same as
      one that never does.
    - `best` is what a dataset built with this pipeline would actually contain, since a
      generator that spots a degenerate take and regenerates it would keep the good one.
      Quote it as a production expectation, never as a property of the voice.
    - `last` is what is currently on disk, which is what you get if you go and listen.

    The gap between `first` and `best` is itself the useful number: it says how much the
    retry loop is carrying.
    """
    sentences = load_sentences()
    voices = restrict(load_voices(include_bases=include_bases), only_file)
    wanted = set(voices)
    by_sid = {entry["sid"]: entry["text"] for entry in sentences}
    history = read_attempts()

    scored: dict[str, dict[str, dict]] = {}
    retried = 0
    for (voice, sid), attempts in history.items():
        if voice not in wanted or sid not in by_sid:
            continue
        record = pick_attempt(attempts, which=which)
        if record is None:
            continue
        retried += sum(1 for entry in attempts if "wer" in entry) > 1
        scored.setdefault(voice, {})[sid] = dict(record)
    if not scored:
        raise SystemExit(f"nothing scored yet in {STAGE_DIR}")
    if retried:
        logger.info(f"{retried} take(s) have more than one scored attempt; "
                    f"reporting the '{which}' one")

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
            append_ledger({"voice": voice, "sid": sid, "attempt": record.get("attempt", 0),
                           "cer": record["cer"]})
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
        wers = [record["wer"] for record in shared]
        rows.append({"voice": voice,
                     "wer": summarise(wers),
                     "cer": summarise(cers) if cers else None,
                     # Yield, which is what a generator actually budgets against: quantiles
                     # say how bad a typical take is, these say how many takes survive a
                     # threshold. The two answer different questions and a median of 6 pct
                     # can hide either 30 pct or 60 pct perfect takes.
                     "perfect": sum(1 for wer in wers if wer == 0) / len(wers),
                     "usable": sum(1 for wer in wers if wer <= 0.10) / len(wers),
                     "junk": sum(1 for wer in wers if wer > 0.50) / len(wers),
                     "seconds": sum(record.get("seconds", 0.0) for record in shared)})
    # Degeneration and pace come from `voice_profiles` rather than being recomputed here,
    # so "runaway" means one thing across stats.md, failures.md and the roster. It reads
    # the same ledger and is keyed by voice, so the merge is a lookup.
    profiles = voice_profiles(only=wanted, which=which)
    for row in rows:
        profile = profiles.get(row["voice"], {})
        row["runaway"] = profile.get("runaway")
        row["words_per_second"] = profile.get("rate")
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
             f"Retried takes are reported at their **{which}** attempt"
             + (f", and {retried} take(s) have more than one." if retried
                else "; no take needed a retry.") + "",
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
    # Relative to the repo, never absolute: this markdown gets pasted into issues and
    # shared, and an absolute path there leaks the author's home directory.
    try:
        shown = csv_path.relative_to(REPO_ROOT)
    except ValueError:
        shown = csv_path
    lines += ["", "## Yield: how many takes you keep", "",
              "The tables above say how wrong a take is; this says how many takes are "
              "worth keeping, which is what a dataset run budgets against. `perfect` is "
              "takes the ASR transcribed exactly, `usable` is at or under 10 pct WER, "
              "`junk` is past 50 pct and has to be caught and regenerated. Ordered by "
              "`perfect`, and note this reorders the voices against the WER tables: a "
              "voice can have a good median and still land fewer clean takes.", "",
              "| voice | takes | perfect | usable (<= 10%) | junk (> 50%) |",
              "|---|---|---|---|---|"]
    for row in sorted(rows, key=lambda row: -row["perfect"]):
        lines.append(f"| {row['voice']} | {row['wer']['n']} | {100 * row['perfect']:.1f}% | "
                     f"{100 * row['usable']:.1f}% | {100 * row['junk']:.1f}% |")
    lines += ["", f"Per-sentence detail, one row per (voice, sentence): `{shown}`.", ""]

    destination = Path(out) if out else STAGE_DIR / "stats.md"
    destination.write_text("\n".join(lines) + "\n")

    # The same rows as a machine-readable summary, beside the markdown. This exists so a
    # report spanning SEVERAL runs (grid_report.py, which compares one voice across eight
    # languages) can read what this command already computed instead of re-deriving the
    # common-sentence set, the CER back-fill and the yield buckets. A second
    # implementation of those would eventually disagree with this one, and the whole
    # point of the cross-language table is that its cells are comparable.
    summary_path = destination.with_name("summary.json")
    summary_path.write_text(json.dumps({
        "run": RUN,
        "attempt": which,
        "shared_sentences": len(common),
        "sentences_in_run": len(by_sid),
        "audio_hours": round(audio_hours, 2),
        "voices": rows,
    }, indent=2))

    logger.success(f"{len(rows)} voices over {len(common)} shared sentences, "
                   f"wrote {destination}, {summary_path} and {csv_path}")


def text_features(text: str) -> dict:
    """Properties of a sentence that might explain why voices struggle with it.

    Chosen before looking at the results, from what FLEURS actually contains that the A6
    screen deliberately filtered out: the screen required a capital start, a terminal
    stop, no digits, at most one interior capital and French letters only, so every one
    of these is a category the voices have never been measured on.

    Parameters
    ----------
    text : str
        The reference sentence.

    Returns
    -------
    dict
        `words`, `quoted` (French guillemets), `multi` (more than one sentence),
        `digits`, `parens`, `nbsp` (any of the exotic Unicode spaces French typography
        puts before `?!:;»`, which the model was measured to degenerate on).

    The `nbsp` test is exactly the set `tts.normalise_spaces` replaces, so a sentence
    flagged here is a sentence the client's default now fixes. Detecting a character the
    fix does not touch, or missing one it does, would make every table below disagree
    with what a caller actually gets.
    """
    return {"words": len(text.split()),
            "quoted": "«" in text or "»" in text,
            "multi": bool(re.search(r"[.!?]\s+\S", text)),
            "digits": any(character.isdigit() for character in text),
            "parens": "(" in text,
            "nbsp": any(character in text for character in tts.UNICODE_SPACES)}


@cli.command("diagnose")
@click.option("--out", default=None, help="Where to write the markdown (default: <stage>/failures.md).")
@click.option("--only-file", default=None, help="Restrict to the voices named in a file.")
@click.option("--include-bases/--no-include-bases", default=True, show_default=True,
              help="Include the shipped voices as controls.")
@click.option("--runaway-factor", default=RUNAWAY_FACTOR, show_default=True,
              help="Duration multiple of a voice's own speaking rate that counts as a runaway.")
@click.option("--worst", default=25, show_default=True, help="How many hard sentences to list.")
@click.option("--attempt", "which", type=click.Choice(ATTEMPT_CHOICES), default="first",
              show_default=True, help="Which attempt of a retried take to report.")
def cmd_diagnose(out: str | None, only_file: str | None, include_bases: bool,
                 runaway_factor: float, worst: int, which: str) -> None:
    """Separate broken voices from broken sentences, and name the failure modes.

    A per-voice mean is useless on its own when the failures are catastrophic and rare: a
    voice whose mean WER is 98 pct and whose median is 4 pct is not a bad voice, it is a
    good voice that detonated twice. What matters is whether the detonations follow the
    voice or the sentence, because only the first is a reason to drop a voice and only the
    second is a reason to fix the text before generating a dataset with it.

    Two signals, deliberately independent. Word error rate comes from the ASR and can be
    wrong for its own reasons. Runaway duration does not: a take is flagged when it runs
    more than `--runaway-factor` times longer than the voice's *own* median speaking rate
    predicts for that word count, which detects degeneration without asking a second model
    anything. A take pinned at the token ceiling is the extreme case of the same thing,
    and is reported separately because it means the audio is also truncated mid-word.
    """
    sentences = load_sentences()
    voices = restrict(load_voices(include_bases=include_bases), only_file)
    wanted = set(voices)
    by_sid = {entry["sid"]: entry["text"] for entry in sentences}
    features = {sid: text_features(text) for sid, text in by_sid.items()}
    state = read_ledger()

    takes, retried = [], 0
    for (voice, sid), attempts in read_attempts().items():
        if voice not in wanted or sid not in by_sid:
            continue
        retried += sum(1 for entry in attempts if "wer" in entry) > 1
        record = pick_attempt(attempts, which=which)
        if record is not None and record.get("seconds"):
            takes.append(dict(record, voice=voice, sid=sid))
    if not takes:
        raise SystemExit(f"nothing scored yet in {STAGE_DIR}")

    # Each voice is judged against its own speaking rate, because A5 measured a 3x spread
    # across the shipped voices: a fixed seconds-per-word threshold would flag every slow
    # voice as degenerate and miss a fast one that doubled. Shared with `retry`, which has
    # to agree with this table about what "broken" means or the two tell different stories.
    def words_of(sid: str) -> int:
        """Word count of a sentence."""
        return len(by_sid[sid].split())

    rates = voice_rates(takes, words_of=words_of)
    for take in takes:
        take["stretch"] = stretch_of(seconds=take["seconds"], words=words_of(take["sid"]),
                                     rate=rates[take["voice"]])
    runaways = [take for take in takes if take["stretch"] >= runaway_factor]
    # 4096 tokens at 12.5 Hz. A take at the cap did not finish, it was cut off.
    ceiling = [take for take in takes if take["seconds"] >= 327.0]

    lines = [f"# {RUN}: failure modes", "",
             f"{len(takes)} scored takes over {len({t['sid'] for t in takes})} sentences "
             f"and {len({t['voice'] for t in takes})} voices, each quoted at its "
             f"**{which}** attempt"
             + (f" ({retried} take(s) were retried, so this is the unaided number unless "
                "you asked for another)." if retried else "; no take needed a retry."), "",
             f"- **{len(runaways)} runaway takes** ({100 * len(runaways) / len(takes):.1f} pct), "
             f"meaning the audio ran at least {runaway_factor:g}x longer than that voice's own "
             "median speaking rate predicts. This is measured from the waveform, not from the "
             "transcript, so it does not depend on the ASR being right.",
             f"- **{len(ceiling)} takes pinned at the 327.7 s token ceiling**, which is "
             "generation that never terminated and was cut off mid-word.", ""]

    # Per-sentence first: if the hard sentences are the same ones for everybody, the
    # problem is the corpus and no amount of choosing between voices fixes it.
    per_sid: dict[str, list] = {}
    for take in takes:
        per_sid.setdefault(take["sid"], []).append(take)
    hardest = sorted(per_sid.items(),
                     key=lambda item: -statistics.median([t["wer"] for t in item[1]]))[:worst]
    lines += [f"### The {len(hardest)} hardest sentences", "",
              "Median WER is across voices, so it measures the *sentence*: a high median "
              "means every voice struggled, which is a property of the text.", "",
              "| sid | voices | median WER | max WER | runaways | words | «» | multi | digits | text |",
              "|---|---|---|---|---|---|---|---|---|---|"]
    for sid, group in hardest:
        wers = [take["wer"] for take in group]
        feature = features[sid]
        lines.append(
            f"| `{sid}` | {len(group)} | {100 * statistics.median(wers):.0f}% | "
            f"{100 * max(wers):.0f}% | {sum(1 for t in group if t['stretch'] >= runaway_factor)} | "
            f"{feature['words']} | {'yes' if feature['quoted'] else ''} | "
            f"{'yes' if feature['multi'] else ''} | {'yes' if feature['digits'] else ''} | "
            + by_sid[sid][:70].replace("|", "/") + "… |")

    # Then the same takes grouped by text feature, which turns the table above from an
    # anecdote into a rate.
    lines += ["", "### Error rate by text feature", "",
              "Each row is every take whose sentence has that property. `runaway` is the "
              "share of those takes that degenerated. A feature that raises the median "
              "raises it for all voices at once and is a corpus property; a feature that "
              "raises only the mean and the runaway rate is a rare catastrophic trigger, "
              "which is the more dangerous kind for dataset generation.", "",
              "| group | sentences | takes | median WER | mean WER | runaway |",
              "|---|---|---|---|---|---|"]

    def bucket(label: str, chosen: list) -> None:
        if not chosen:
            return
        wers = [take["wer"] for take in chosen]
        lines.append(f"| {label} | {len({t['sid'] for t in chosen})} | {len(chosen)} | "
                     f"{100 * statistics.median(wers):.1f}% | {100 * statistics.fmean(wers):.1f}% | "
                     f"{100 * sum(1 for t in chosen if t['stretch'] >= runaway_factor) / len(chosen):.1f}% |")

    bucket("all takes", takes)
    for name, key in (("guillemets «»", "quoted"), ("several sentences", "multi"),
                      ("digits", "digits"), ("parentheses", "parens"), ("nbsp / thin space", "nbsp")):
        bucket(f"with {name}", [t for t in takes if features[t["sid"]][key]])
        bucket(f"without {name}", [t for t in takes if not features[t["sid"]][key]])
    for low, high in ((0, 20), (20, 30), (30, 40), (40, 999)):
        chosen = [t for t in takes if low <= features[t["sid"]]["words"] < high]
        bucket(f"{low} to {high if high < 999 else '65'} words", chosen)

    # How a voice was BUILT, which is the lever anyone has: if one construction fails more
    # than another on the same sentences, that is a recipe to change, not a voice to drop.
    pool = load_pool()

    def construction(voice: str) -> tuple[str, str]:
        """How this voice was made: its kind, and the donor that lent it articulation.

        A6 built hybrids as `h_<semantic>_<acoustic>`, taking the semantic code column from
        one voice and the 36 acoustic columns from another, so the semantic donor is what
        decides whether the result can pronounce the language. Grouping by it is the only
        way to see whether the recipe, rather than the speaker, is what breaks.
        """
        entry = pool.get(voice)
        if entry is None or not entry.get("parents"):
            return ("base", "-")
        return (entry["kind"], entry["parents"][0])

    lines += ["", "### Failure by construction", "",
              "Same takes, grouped by how the voice was built rather than by which voice "
              "it is. `semantic donor` is the voice a hybrid took its semantic code column "
              "from, which is what lends it the ability to pronounce French at all.", "",
              "| group | voices | takes | median WER | mean WER | p95 WER | runaway |",
              "|---|---|---|---|---|---|---|"]

    def group_row(label: str, chosen: list) -> None:
        if not chosen:
            return
        wers = sorted(take["wer"] for take in chosen)
        p95 = wers[min(len(wers) - 1, int(0.95 * len(wers)))]
        lines.append(f"| {label} | {len({t['voice'] for t in chosen})} | {len(chosen)} | "
                     f"{100 * statistics.median(wers):.1f}% | {100 * statistics.fmean(wers):.1f}% | "
                     f"{100 * p95:.1f}% | "
                     f"{100 * sum(1 for t in chosen if t['stretch'] >= runaway_factor) / len(chosen):.1f}% |")

    for kind in sorted({construction(take["voice"])[0] for take in takes}):
        chosen = [t for t in takes if construction(t["voice"])[0] == kind]
        group_row(f"kind: {kind}", chosen)
        # The same group with its degenerate takes removed. If a kind's mean collapses back
        # onto the others here, it is not a worse reader, it is a reader that occasionally
        # detonates, and a retry loop fixes that where a different recipe would not.
        group_row(f"kind: {kind}, runaways removed",
                  [t for t in chosen if t["stretch"] < runaway_factor])
    for donor in sorted({construction(take["voice"])[1] for take in takes} - {"-"}):
        group_row(f"semantic donor: {donor}",
                  [t for t in takes if construction(t["voice"])[1] == donor])

    per_voice: dict[str, list] = {}
    for take in takes:
        per_voice.setdefault(take["voice"], []).append(take)

    # The question the whole A6 pool exists to answer: is a derived voice better than just
    # using the voice it borrowed its identity from? A hybrid keeps the acoustic donor's
    # speaker and swaps in another voice's semantic column, so the donor is the thing it
    # has to beat. If it does not, the pool is 19 more names for 19 voices already shipped.
    def identity_donor(voice: str) -> str | None:
        """The voice a derived voice took its speaker identity from.

        For a hybrid that is the acoustic donor (`parents[-1]`, the second half of
        `h_<semantic>_<acoustic>`), because A6 measured identity to live entirely in the
        36 acoustic codes. For a sparse mix it is the dominant component, which is stored
        first. Returns None for a shipped voice, which has nothing to be compared against.
        """
        entry = pool.get(voice)
        if entry is None or not entry.get("parents"):
            return None
        return entry["parents"][-1] if entry["kind"] == "hybrid" else entry["parents"][0]

    profiles = voice_profiles(only=wanted, runaway_factor=runaway_factor)

    def profile(voice: str) -> dict[str, float]:
        """Shared robustness row for one voice, so this table and `select` agree."""
        return profiles[voice]

    pairs = [(voice, identity_donor(voice)) for voice in sorted(per_voice)]
    pairs = [(voice, donor) for voice, donor in pairs if donor in per_voice]
    if pairs:
        better_median = sum(1 for v, d in pairs if profile(v)["median"] < profile(d)["median"])
        better_mean = sum(1 for v, d in pairs if profile(v)["mean"] < profile(d)["mean"])
        worse_tail = sum(1 for v, d in pairs if profile(v)["runaway"] > profile(d)["runaway"])
        lines += ["", "### Derived voice against the voice it copied", "",
                  "Every derived voice next to the donor it took its speaker identity from "
                  "(the acoustic parent of a hybrid, the dominant component of a mix), on "
                  "the same sentences. `x / y` is derived / donor. A derived voice only "
                  "earns its place if it beats this row, because the donor is already "
                  "available and costs nothing.", "",
                  f"Of {len(pairs)} pairs: **{better_median} beat their donor on the median "
                  f"sentence**, {better_mean} on the mean, and **{worse_tail} degenerate more "
                  "often than their donor does**.", "",
                  "| derived | donor | median WER | mean WER | runaway | words/s |",
                  "|---|---|---|---|---|---|"]
        for voice, donor in sorted(pairs, key=lambda pair: profile(pair[0])["median"]):
            here, there = profile(voice), profile(donor)
            lines.append(
                f"| `{voice}` | `{donor}` | {100 * here['median']:.1f}% / {100 * there['median']:.1f}% | "
                f"{100 * here['mean']:.1f}% / {100 * there['mean']:.1f}% | "
                f"{100 * here['runaway']:.1f}% / {100 * there['runaway']:.1f}% | "
                f"{here['rate']:.2f} / {there['rate']:.2f} |")

    # And finally per voice, so a voice that really is fragile is still visible.
    lines += ["", "### Runaways per voice", "",
              "A voice at the top of this table degenerates more often than the others on "
              "the same sentences, which is a reason to drop it. A voice near the bottom "
              "whose mean WER is nonetheless high was tripped by the text, not by itself.", "",
              "| voice | takes | runaways | ceiling | median WER | mean WER | words/s |",
              "|---|---|---|---|---|---|---|"]
    for voice, group in sorted(per_voice.items(),
                               key=lambda item: -sum(1 for t in item[1] if t["stretch"] >= runaway_factor)):
        wers = [take["wer"] for take in group]
        lines.append(f"| `{voice}` | {len(group)} | "
                     f"{sum(1 for t in group if t['stretch'] >= runaway_factor)} | "
                     f"{sum(1 for t in group if t['seconds'] >= 327.0)} | "
                     f"{100 * statistics.median(wers):.1f}% | {100 * statistics.fmean(wers):.1f}% | "
                     f"{rates[voice]:.2f} |")

    destination = Path(out) if out else STAGE_DIR / "failures.md"
    destination.write_text("\n".join(lines) + "\n")
    logger.success(f"{len(runaways)} runaway(s) and {len(ceiling)} at the ceiling "
                   f"in {len(takes)} takes, wrote {destination}")


# Non-breaking space, thin space, narrow no-break space. French typography puts one of
# these before ?!:;» and inside « », so they are invisible in every editor and travel
# with the guillemets, which is exactly why the two have to be separated by experiment.
def _plain_spaces(text: str) -> str:
    """Turn the exotic Unicode spaces into ordinary ones, leaving the guillemets.

    Delegates to `tts.normalise_spaces` rather than repeating the character list: that
    function is what ordinary callers now get by default, so the probe must test exactly
    it. A second copy here would eventually test a set of characters the client does not
    actually replace.
    """
    return tts.normalise_spaces(text)


def _no_guillemets(text: str) -> str:
    """Delete the guillemets and nothing else.

    Deliberately not `\\s*[«»]\\s*`: swallowing the neighbouring whitespace would remove
    the typographic space at the same time, and then this variant would be testing both
    changes at once instead of one.
    """
    return text.replace("«", "").replace("»", "")


# A 2x2, not a ladder: `nbsp` keeps the guillemets and fixes the spaces, `quotes` keeps
# the spaces and removes the guillemets, so whichever of the two moves the numbers is the
# culprit and `clean` bounds what is achievable. Keys appear verbatim in the report.
PROBE_VARIANTS = {
    "raw": lambda text: text,
    "nbsp": _plain_spaces,
    "quotes": _no_guillemets,
    "clean": lambda text: re.sub(r" +", " ", _plain_spaces(_no_guillemets(text))).strip(),
}


@cli.command("retry-report")
@click.option("--out", default=None, help="Where to write the markdown (default: <stage>/retries.md).")
@click.option("--runaway-factor", default=RUNAWAY_FACTOR, show_default=True,
              help="Duration multiple of a voice's own speaking rate that counts as a runaway.")
def cmd_retry_report(out: str | None, runaway_factor: float) -> None:
    """Did retrying with a different seed actually fix the degenerate takes.

    This is the question the retry loop was built to answer and the one thing the loop
    cannot be trusted to answer about itself: a loop that retries until the take looks
    fine will always report that takes end up looking fine. So the report is written from
    the whole attempt history, counts how many takes ran out of budget still broken, and
    keeps the first-attempt population number next to the best-attempt one so the cost of
    quoting the flattering figure is visible.

    A take converging on attempt 2 also means something specific about the model: the
    degeneration is a property of the sampled noise, not of the (voice, sentence) pair,
    and generating a dataset only needs a retry loop rather than a different voice list.
    """
    by_sid = {entry["sid"]: entry["text"] for entry in load_sentences()}

    def words_of(sid: str) -> int:
        """Word count of a sentence."""
        return len(by_sid[sid].split())

    # Rates come from the surviving attempt of every row, which is what `diagnose` uses, so
    # the two agree on what "runaway" means. A median over 332 takes does not move for the
    # 1 pct that degenerate.
    settled = [dict(record, voice=voice, sid=sid)
               for (voice, sid), record in read_ledger().items()
               if sid in by_sid and record.get("seconds")]
    rates = voice_rates(settled, words_of=words_of)

    def broken(take: dict, sid: str, voice: str) -> bool:
        """Whether one attempt degenerated, from its waveform alone."""
        if not take.get("seconds"):
            return False
        return stretch_of(seconds=take["seconds"], words=words_of(sid),
                          rate=rates[voice]) >= runaway_factor

    history = {key: attempts for key, attempts in read_attempts().items()
               if key[1] in by_sid and len(attempts) > 1}
    if not history:
        raise SystemExit(f"no take in {STAGE_DIR} has been retried yet")

    # Per take: where it started, where it ended, and on which attempt it stopped breaking.
    fixed_on: dict[int, int] = {}
    unfixed, gains = [], []
    for (voice, sid), attempts in history.items():
        landed = None
        for index, take in enumerate(attempts):
            if not broken(take, sid, voice):
                landed = index
                break
        if landed is None:
            unfixed.append((voice, sid, attempts))
        else:
            fixed_on[landed] = fixed_on.get(landed, 0) + 1
            if "wer" in attempts[0] and "wer" in attempts[landed]:
                gains.append((attempts[0]["wer"], attempts[landed]["wer"]))

    lines = [f"# {RUN}: did the retries converge", "",
             f"{len(history)} takes were retried at least once. A take is counted as fixed "
             f"on the first attempt whose audio is under {runaway_factor:g}x the voice's own "
             "speaking rate, measured from the waveform, so the ASR has no say in it.", "",
             f"- **{sum(fixed_on.values())} of {len(history)} converged**, "
             f"{len(unfixed)} ran out of attempts still degenerate.", ""]
    if gains:
        before = statistics.median([pair[0] for pair in gains])
        after = statistics.median([pair[1] for pair in gains])
        lines += [f"- Median WER of a retried take: **{100 * before:.1f}% -> {100 * after:.1f}%** "
                  "between its first attempt and the one that converged.", ""]
    lines += ["| converged on | takes |", "|---|---|"]
    for attempt in sorted(fixed_on):
        label = "first attempt (never broken)" if attempt == 0 else f"attempt {attempt + 1}"
        lines.append(f"| {label} | {fixed_on[attempt]} |")
    lines.append(f"| never, budget exhausted | {len(unfixed)} |")

    # Which voices cost the retries, and whether any of them is hopeless rather than unlucky.
    per_voice: dict[str, list[int]] = {}
    for (voice, sid), attempts in history.items():
        ok = any(not broken(take, sid, voice) for take in attempts)
        per_voice.setdefault(voice, [0, 0])
        per_voice[voice][0] += 1
        per_voice[voice][1] += 0 if ok else 1
    lines += ["", "## Per voice", "",
              "`retried` is how many of its takes needed a second seed, `still broken` how "
              "many never came good. A voice with a high second number is failing on the "
              "sentence, not on the seed, and no amount of retrying will save it.", "",
              "| voice | retried | still broken |", "|---|---|---|"]
    for voice, (retried, still) in sorted(per_voice.items(), key=lambda item: (-item[1][1], -item[1][0])):
        lines.append(f"| `{voice}` | {retried} | {still} |")

    # Grouped by sentence, because that is the test of whether a retry loop is the right
    # tool at all: failures spread thinly across many sentences are bad luck and more
    # attempts would clear them, failures piled on a few sentences are the text, and the
    # fix is to change the text rather than to keep paying for seeds.
    if unfixed:
        stuck: dict[str, int] = {}
        for voice, sid, _ in unfixed:
            stuck[sid] = stuck.get(sid, 0) + 1
        features = {sid: text_features(by_sid[sid]) for sid in stuck}
        everything = {sid: text_features(text) for sid, text in by_sid.items()}
        lines += ["", "## Where the unfixable takes pile up", "",
                  f"The {len(unfixed)} takes that never converged land on only "
                  f"**{len(stuck)} sentences**. Comparing what those sentences contain "
                  "against the corpus they were drawn from says whether the retry budget "
                  "was too small or the text is the trigger: a feature present in nearly "
                  "every stuck sentence but a minority of the corpus is not luck.", "",
                  "| feature | stuck sentences | whole corpus |", "|---|---|---|"]
        for label, key in (("guillemets «»", "quoted"), ("nbsp / thin space", "nbsp"),
                           ("digits", "digits"), ("parentheses", "parens"),
                           ("several sentences", "multi")):
            here = sum(1 for sid in stuck if features[sid][key])
            there = sum(1 for feature in everything.values() if feature[key])
            lines.append(f"| {label} | {here}/{len(stuck)} ({100 * here / len(stuck):.0f}%) | "
                         f"{there}/{len(everything)} ({100 * there / len(everything):.0f}%) |")
        lines += ["",
                  "| sid | voices stuck | «» | nbsp | digits | words | text |",
                  "|---|---|---|---|---|---|---|"]
        for sid, count in sorted(stuck.items(), key=lambda item: -item[1]):
            feature = features[sid]
            lines.append(f"| `{sid}` | {count} | {'yes' if feature['quoted'] else ''} | "
                         f"{'yes' if feature['nbsp'] else ''} | "
                         f"{'yes' if feature['digits'] else ''} | {feature['words']} | "
                         + by_sid[sid][:60].replace("|", "/") + "… |")

    if unfixed:
        lines += ["", "## The takes that never converged", "",
                  "Same sentence, same voice, a different noise seed each time. These are "
                  "the rows to look at if the degeneration turns out to be a property of "
                  "the text rather than of the sampling.", "",
                  "| voice | sid | attempts | text |", "|---|---|---|---|"]
        for voice, sid, attempts in sorted(unfixed)[:40]:
            lines.append(f"| `{voice}` | `{sid}` | {len(attempts)} | "
                         + by_sid[sid][:60].replace("|", "/") + "… |")

    destination = Path(out) if out else STAGE_DIR / "retries.md"
    destination.write_text("\n".join(lines) + "\n")
    logger.success(f"{len(history)} retried takes, {len(unfixed)} still broken, wrote {destination}")


@cli.command("probe")
@click.option("--from-run", default="FLEURS", show_default=True,
              help="Run whose hardest sentences are re-read as variants.")
@click.option("--worst", default=8, show_default=True, help="How many hard sentences to probe.")
@click.option("--require", default="nbsp", show_default=True,
              help="Only probe sentences with this text feature (empty = any).")
@click.option("--voices", default=6, show_default=True,
              help="How many voices to probe with: the most fragile ones, plus one control.")
def cmd_probe(from_run: str, worst: int, require: str, voices: int) -> None:
    """Mint a tiny run that re-reads the hardest sentences with the text cleaned up.

    `diagnose` can only say that guillemets and non-breaking spaces travel with the
    catastrophes. It cannot say which one causes them, because French typography puts a
    non-breaking space inside every pair of guillemets, so the two features are almost the
    same set of sentences. The only way to separate them is to change one at a time and
    re-synthesize, which is what this mints the sentence list for.

    The default filter is the non-breaking space rather than the guillemets, on the
    evidence of `retry-report`: all 15 sentences whose takes were still degenerate after
    three different seeds carry one, against 18 pct of the corpus, while only 11 of the 15
    carry guillemets. Filtering on it also puts both kinds of sentence in the sample, since
    every guillemet pair contains one, which is what lets the 2x2 separate them.

    Each source sentence becomes four rows sharing one pinned noise seed, so the audio
    differs by the text and by nothing else: `raw` as FLEURS ships it, `nbsp` with the
    non-breaking and thin spaces turned into ordinary ones, `quotes` with the guillemets
    removed, and `clean` with both. Running it is the normal driver over a small run:

        sudo -E RUN=PROBE DEPTH=0 PREPARE=probe \\
            ONLY_FILE=cache/cloning/PROBE/voices.txt ./voxtral/cloning/driver.sh

    then `sweep.py probe-report`. It costs two container flips and a few minutes, which is
    the whole reason it is a separate run rather than more rows in the benchmark.

    Parameters
    ----------
    from_run : str
        Run directory to take the sentences and the scores from.
    worst : int
        How many of the hardest sentences to include.
    require : str
        A `text_features` key the sentence must have, so the probe spends its rows on the
        suspected trigger rather than on whatever happened to score badly. Empty for any.
    voices : int
        Size of the voice list written to `voices.txt`: the most fragile voices of the
        source run, plus one control that never degenerated.
    """
    source = REPO_ROOT / "cache" / "cloning" / from_run
    sentences = json.loads((source / "sentences.json").read_text())
    by_sid = {entry["sid"]: entry["text"] for entry in sentences}

    # Ranked by how badly the *sentence* does across voices, which is the only ranking that
    # points at the text rather than at one unlucky voice.
    per_sid: dict[str, list[float]] = {}
    # Degenerate takes per voice, counted the one way that needs no rate model: a take at
    # the token ceiling generated until it ran out of tokens, whatever its speaking rate.
    # This is what picks the probe's voices, so the probe spends its rows on the voices
    # that actually break rather than on a hand-picked list.
    ceiling: dict[str, int] = {}
    for line in (source / "ledger.jsonl").read_text(errors="replace").splitlines():
        try:
            record = json.loads(line)
        except json.JSONDecodeError:
            continue
        if "wer" in record and record["sid"] in by_sid:
            per_sid.setdefault(record["sid"], []).append(record["wer"])
        if record.get("seconds", 0) >= 327.0:
            ceiling[record["voice"]] = ceiling.get(record["voice"], 0) + 1
    if not per_sid:
        raise SystemExit(f"no scores in {source}/ledger.jsonl")
    ranked = sorted(per_sid.items(), key=lambda item: -statistics.median(item[1]))
    if require:
        ranked = [item for item in ranked if text_features(by_sid[item[0]])[require]]
    chosen = ranked[:worst]
    if not chosen:
        raise SystemExit(f"no sentences in {from_run} have the '{require}' feature")

    out = []
    for index, (sid, _) in enumerate(chosen):
        text = by_sid[sid]
        for offset, (variant, transform) in enumerate(PROBE_VARIANTS.items()):
            out.append({"sid": f"p{index * len(PROBE_VARIANTS) + offset}",
                        "text": transform(text),
                        # Shared across the four variants of one source sentence, so the
                        # only thing that moves between them is the text.
                        "seed": index,
                        "variant": variant,
                        "origin": sid})
    STAGE_DIR.mkdir(parents=True, exist_ok=True)
    (STAGE_DIR / "sentences.json").write_text(json.dumps(out, indent=2, ensure_ascii=False) + "\n")
    logger.success(f"{len(chosen)} sentences x {len(PROBE_VARIANTS)} variants = {len(out)} rows "
                   f"to {STAGE_DIR / 'sentences.json'}")
    # The voice list is written here rather than left to the caller, because the driver
    # command in the docstring names it and a probe run that dies on a missing file is a
    # wasted container flip. The fragile voices come first (they are the ones that can show
    # a text fix working) and a control that never breaks comes last, so a variant that
    # still fails on the control is the corpus, not the recipe.
    fragile = sorted(ceiling, key=lambda name: -ceiling[name])[:max(voices - 1, 1)]
    control = "fr_female" if "fr_female" not in fragile else "fr_male"
    chosen_voices = fragile + [control]
    (STAGE_DIR / "voices.txt").write_text("\n".join(chosen_voices) + "\n")
    logger.success(f"probing with {', '.join(chosen_voices)} "
                   f"-> {STAGE_DIR / 'voices.txt'}")


@cli.command("probe-report")
@click.option("--out", default=None, help="Where to write the markdown (default: <stage>/probe.md).")
def cmd_probe_report(out: str | None) -> None:
    """Compare the probe variants: which text change actually fixes the failures.

    Reads the same ledger everything else does. The comparison is within a source
    sentence, across variants, so it is immune to some sentences simply being harder.
    """
    sentences = load_sentences()
    if not any("variant" in entry for entry in sentences):
        raise SystemExit(f"{STAGE_DIR}/sentences.json is not a probe list; run `probe` first")
    meta = {entry["sid"]: entry for entry in sentences}
    state = read_ledger()

    rows = [dict(record, **meta[sid]) for (voice, sid), record in state.items()
            if sid in meta and "wer" in record]
    if not rows:
        raise SystemExit(f"nothing scored yet in {STAGE_DIR}")

    per_variant: dict[str, list] = {}
    for row in rows:
        per_variant.setdefault(row["variant"], []).append(row)
    lines = [f"# {RUN}: does cleaning the text fix the failures", "",
             "Every row below is the same sentences read by the same voices at the same "
             "noise seed, with one text change applied. `raw` is FLEURS as shipped, `nbsp` "
             "replaces non-breaking and thin spaces with ordinary ones, `quotes` removes "
             "the guillemets, `clean` does both. A variant that drops the mean and the "
             "runaway rate is the fix; a variant that changes nothing exonerates that "
             "feature. `unusable` counts takes at or past 100 pct WER, which have lost at "
             "least as many words as the sentence contains.", "",
             "| variant | takes | median WER | mean WER | max WER | unusable | median s |",
             "|---|---|---|---|---|---|---|"]
    for variant in PROBE_VARIANTS:
        group = per_variant.get(variant, [])
        if not group:
            continue
        wers = [row["wer"] for row in group]
        seconds = [row.get("seconds", 0.0) for row in group]
        lines.append(f"| `{variant}` | {len(group)} | {100 * statistics.median(wers):.1f}% | "
                     f"{100 * statistics.fmean(wers):.1f}% | {100 * max(wers):.0f}% | "
                     f"{sum(1 for wer in wers if wer >= 1.0)} | "
                     f"{statistics.median(seconds):.1f} |")

    # The verdict, computed rather than eyeballed: a 2x2 is only worth running if the
    # reading of it is mechanical. Each single-feature variant is scored against `raw` on
    # the two numbers that matter for dataset generation (what a typical take costs, and
    # how often a take is unusable), and `clean` says whether the two features interact.
    def summary(variant: str) -> tuple[float, float, int]:
        """Median WER, mean WER and count of unusable takes for one variant."""
        group = per_variant.get(variant, [])
        wers = [row["wer"] for row in group]
        return (statistics.median(wers), statistics.fmean(wers),
                sum(1 for wer in wers if wer >= 1.0))

    if {"raw", "nbsp", "quotes"} <= set(per_variant):
        base = summary("raw")
        lines += ["", "### Verdict", "",
                  "Each single-feature variant against `raw`, on the same sentences, voices "
                  "and seed. A feature is implicated only if removing it alone moves both "
                  "numbers.", "",
                  "| removed | median WER | unusable takes (WER >= 100 pct) | implicated |",
                  "|---|---|---|---|"]
        for variant, label in (("nbsp", "nbsp / thin space"), ("quotes", "guillemets «»"),
                               ("clean", "both")):
            if variant not in per_variant:
                continue
            here = summary(variant)
            guilty = here[0] < base[0] and here[2] < base[2]
            lines.append(
                f"| {label} | {100 * base[0]:.1f}% -> {100 * here[0]:.1f}% | "
                f"{base[2]} -> {here[2]} of {len(per_variant[variant])} | "
                f"{'**yes**' if guilty else 'no'} |")

    lines += ["", "### Per source sentence", "",
              "| origin | voice | " + " | ".join(f"`{name}`" for name in PROBE_VARIANTS) + " |",
              "|---|---|" + "---|" * len(PROBE_VARIANTS)]
    cells: dict[tuple[str, str], dict[str, float]] = {}
    for (voice, sid), record in state.items():
        if sid in meta and "wer" in record:
            cells.setdefault((meta[sid]["origin"], voice), {})[meta[sid]["variant"]] = record["wer"]
    for (origin, voice), values in sorted(cells.items()):
        lines.append(f"| `{origin}` | `{voice}` | " + " | ".join(
            f"{100 * values[name]:.0f}%" if name in values else "-" for name in PROBE_VARIANTS) + " |")

    destination = Path(out) if out else STAGE_DIR / "probe.md"
    destination.write_text("\n".join(lines) + "\n")
    logger.success(f"{len(rows)} probe takes over {len(per_variant)} variants, wrote {destination}")


if __name__ == "__main__":
    cli()
