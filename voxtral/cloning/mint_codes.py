#!/usr/bin/env -S uv run --script
# /// script
# requires-python = ">=3.10"
# dependencies = ["click", "loguru", "numpy"]
# ///
"""Mint real audio-code sequences by having the server generate them (stage B6 data).

Why this exists
---------------
`train_encoder.py` needs pairs of (audio, codes) to learn the encoder the open
checkpoint does not ship. Codes are the scarce half. B6 measured the limit exactly:
the 20 shipped voice prefixes are 2823 frames, about four minutes of speech, holding
1071 distinct semantic codes, **13 percent of the 8192**. A network cannot learn a map
whose output space it has seen an eighth of, and it does not: 94 percent on the
training frames, 2 percent on a held-out voice. Sampling the code space uniformly
instead is free and worthless (0.2 percent on the training batch itself), because
uniform codes are 37 dimensions of independent noise per frame and inverting dense
noise generalises about as well as inverting a hash.

The way out is that the server already contains the generator we need. Stage 0 is an
AR LLM whose entire job is text -> on-manifold codes, in any of the 20 voices, and the
codes it emits are the same table-space ids a voice prefix carries. Fork patch 0007
makes it write them out on request (`tts.py --dump-codes`), so this script is just the
driver: fetch varied text, send it, collect what stage 0 emitted. One 40 s utterance is
about 500 frames, so a few hundred calls is two orders of magnitude more code than the
shipped voices carry, on distribution by construction rather than by argument.

Note the *audio* is not collected at all. `train_encoder.py` decodes the codes locally
to make its input, which is exactly the audio those codes stand for; asking the server
to send back a copy would only mean storing a second one.

Text
----
Random Wikipedia article intros, in the language each voice actually speaks. Random
topics are the point: coverage of the 8192 semantic codes follows phonetic variety, and
an encyclopedia wanders across names, places and jargon far more widely than any novel.
(Text is CC BY-SA. It is used here to drive a local generator, and neither it nor the
audio is redistributed.)

Usage
-----
    ./mint_codes.py corpus --articles 400          # fetch text for every voice language
    ./mint_codes.py mint --minutes 240             # generate, needs voxtral-tts up
    ./mint_codes.py collect                        # dumps -> codes/*.npy (mint does it too)
"""

from __future__ import annotations

import json
import re
import sys
import time
import urllib.error
import urllib.parse
import urllib.request
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import click
import numpy as np
from loguru import logger

REPO_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO_ROOT))

import tts  # noqa: E402  (needs REPO_ROOT on the path first)

CORPUS_DIR = REPO_ROOT / "cache" / "cloning" / "corpus"
# Host side of the container's VOXTRAL_CODE_DUMP_DIR (compose mounts ./cache at /cache).
DUMP_DIR = REPO_ROOT / "cache" / "cloning" / "dumps"
CODES_DIR = REPO_ROOT / "cache" / "cloning" / "codes"

# Wikipedia asks API clients to identify themselves. Deliberately no contact
# address: this is a local data-collection run, not a service worth mailing about.
USER_AGENT = "CrispASR-cloning/1.0 (local Voxtral TTS encoder training)"
# Seconds between article batches, and the base of the retry backoff.
BASE_DELAY = 1.0

# 1 semantic + 36 acoustic, the width of one stage-0 frame.
N_CODEBOOKS = 37
# Voices whose name is not a language code. B2 heard all 20 and these three speak
# English; the other 17 announce their language in the prefix of their name.
DEFAULT_LANGUAGE = "en"


def voice_language(voice: str) -> str:
    """Language code for a built-in voice name.

    Parameters
    ----------
    voice : str
        A name from `tts.VOXTRAL_VOICES`, e.g. `fr_male` or `cheerful_female`.

    Returns
    -------
    str
        Wikipedia language code.
    """
    prefix = voice.split("_")[0]
    return prefix if len(prefix) == 2 else DEFAULT_LANGUAGE


def fetch_articles(language: str, articles: int) -> list[str]:
    """Fetch random article intros from one Wikipedia.

    Parameters
    ----------
    language : str
        Wikipedia language code.
    articles : int
        Roughly how many to fetch; the API caps a batch at 20.

    Returns
    -------
    list[str]
        Plain-text extracts, empty ones dropped.

    Raises
    ------
    OSError
        If a batch keeps failing after the retries below.
    """
    out: list[str] = []
    while len(out) < articles:
        query = urllib.parse.urlencode({
            "action": "query", "generator": "random", "grnnamespace": 0,
            "grnlimit": 20, "prop": "extracts", "explaintext": 1,
            "exintro": 1, "exlimit": "max", "format": "json",
        })
        url = f"https://{language}.wikipedia.org/w/api.php?{query}"
        request = urllib.request.Request(url, headers={"User-Agent": USER_AGENT})
        # Wikipedia rate-limits bursts with 429, which is expected here rather than
        # exceptional: hundreds of articles is dozens of batches back to back. Back
        # off and retry instead of losing the language.
        for attempt in range(6):
            try:
                with urllib.request.urlopen(request, timeout=30) as response:
                    pages = json.load(response).get("query", {}).get("pages", {})
                break
            except urllib.error.HTTPError as error:
                if error.code not in (429, 500, 502, 503, 504) or attempt == 5:
                    raise
                delay = BASE_DELAY * 2 ** attempt
                logger.debug(f"{language}: HTTP {error.code}, retrying in {delay:.1f}s")
                time.sleep(delay)
        else:  # pragma: no cover - the loop always breaks or raises
            break
        # Courtesy pause between batches, which is also what keeps the retries above
        # from being needed on every one.
        time.sleep(BASE_DELAY)
        batch = [page.get("extract", "").strip() for page in pages.values()]
        out.extend(text for text in batch if text)
        if not batch:
            break
    return out[:articles]


def clean(text: str) -> str:
    """Strip what a TTS reads badly: headings, references, runs of whitespace.

    Parameters
    ----------
    text : str
        A raw Wikipedia extract.

    Returns
    -------
    str
        One paragraph of plain prose.
    """
    text = re.sub(r"==+[^=]*==+", " ", text)
    # Bracketed asides in an encyclopedia are pronunciation guides, dates and
    # reference marks, all of which a TTS either mangles or reads as noise.
    text = re.sub(r"\[[^\]]*\]", " ", text)
    text = re.sub(r"\([^)]*\)", " ", text)
    return re.sub(r"\s+", " ", text).strip()


def utterances(texts: list[str], max_chars: int) -> list[str]:
    """Cut cleaned text into utterance-sized pieces at sentence boundaries.

    Sentence boundaries matter because a piece that starts mid-clause makes the
    model guess at a prosody the text does not support, and those takes are the
    ones that drift.

    Parameters
    ----------
    texts : list[str]
        Cleaned extracts.
    max_chars : int
        Upper bound per utterance. At roughly 15 characters a second of speech,
        600 is about 40 s, i.e. some 500 frames, well under the length cap.

    Returns
    -------
    list[str]
        Utterances, each at least a full sentence.
    """
    out: list[str] = []
    for text in texts:
        current = ""
        for sentence in re.split(r"(?<=[.!?؟।])\s+", text):
            if not sentence.strip():
                continue
            if current and len(current) + len(sentence) + 1 > max_chars:
                out.append(current)
                current = sentence
            else:
                current = f"{current} {sentence}".strip()
        if len(current) > 40:
            out.append(current)
    return out


def collect_dumps(dump_dir: Path, codes_dir: Path) -> tuple[int, int]:
    """Turn raw `.i16` dumps into the `[T, 37]` npy files the trainer reads.

    A dump whose length is not a whole number of frames is truncated rather than
    rejected: the writer appends frame by frame, so a torn tail means the request
    was cancelled mid-frame and everything before it is still valid.

    Parameters
    ----------
    dump_dir : Path
        Where the server appended `<tag>.i16`.
    codes_dir : Path
        Where `train_encoder.py` looks for `*.npy`.

    Returns
    -------
    tuple[int, int]
        Files written and frames collected.
    """
    codes_dir.mkdir(parents=True, exist_ok=True)
    files = frames = 0
    for path in sorted(dump_dir.glob("*.i16")):
        raw = np.fromfile(path, dtype="<i2")
        usable = raw.size - raw.size % N_CODEBOOKS
        if usable == 0:
            logger.warning(f"{path.name}: {raw.size} values, less than one frame, skipped")
            continue
        if usable != raw.size:
            logger.warning(f"{path.name}: {raw.size - usable} trailing values dropped, torn frame")
        codes = raw[:usable].reshape(-1, N_CODEBOOKS).astype(np.int64)
        np.save(codes_dir / f"{path.stem}.npy", codes)
        files += 1
        frames += codes.shape[0]
    return files, frames


@click.group()
def cli() -> None:
    """Mint real code sequences from the server's own stage 0."""


@cli.command()
@click.option("--articles", default=400, show_default=True, help="Articles per language.")
@click.option("--language", "languages", multiple=True,
              help="Language codes to fetch. Default: every language the 20 voices speak.")
@click.option("--out-dir", default=str(CORPUS_DIR), show_default=True, help="Where to write the text.")
def corpus(articles: int, languages: tuple[str, ...], out_dir: str) -> None:
    """Fetch random Wikipedia article intros as generation material."""
    wanted = list(languages) or sorted({voice_language(voice) for voice in tts.VOXTRAL_VOICES})
    destination = Path(out_dir)
    destination.mkdir(parents=True, exist_ok=True)
    for language in wanted:
        try:
            texts = [clean(text) for text in fetch_articles(language, articles)]
        except OSError as error:
            logger.warning(f"{language}: fetch failed ({error}), skipped")
            continue
        texts = [text for text in texts if len(text) > 80]
        path = destination / f"{language}.txt"
        path.write_text("\n".join(texts), encoding="utf-8")
        logger.success(f"{language}: {len(texts)} articles, {sum(map(len, texts))} chars -> {path}")


@cli.command()
@click.option("--minutes", default=240.0, show_default=True, help="Target minutes of speech to generate.")
@click.option("--url", default=tts.DEFAULT_URL, show_default=True, help="voxtral-tts base URL.")
@click.option("--max-chars", default=600, show_default=True, help="Characters per utterance (~40 s).")
@click.option("--concurrency", default=4, show_default=True,
              help="Requests in flight. The acoustic stage is codec-bs=1, so throughput "
                   "saturates around 4 to 8.")
@click.option("--holdout", default="fr_female", show_default=True,
              help="Voice never generated with, so the trainer's held-out test stays honest.")
@click.option("--timeout", default=600.0, show_default=True, help="Seconds per request.")
@click.option("--corpus-dir", default=str(CORPUS_DIR), show_default=True, help="Where the text is.")
@click.option("--dump-dir", default=str(DUMP_DIR), show_default=True,
              help="Host side of the container's VOXTRAL_CODE_DUMP_DIR.")
@click.option("--codes-dir", default=str(CODES_DIR), show_default=True, help="Where to collect the codes.")
def mint(minutes: float, url: str, max_chars: int, concurrency: int, holdout: str, timeout: float,
         corpus_dir: str, dump_dir: str, codes_dir: str) -> None:
    """Generate speech across every voice and keep the codes stage 0 emitted."""
    voices = [voice for voice in tts.VOXTRAL_VOICES if voice != holdout]
    by_language: dict[str, list[str]] = {}
    for path in sorted(Path(corpus_dir).glob("*.txt")):
        by_language[path.stem] = utterances(path.read_text(encoding="utf-8").split("\n"), max_chars)
    if not by_language:
        raise click.ClickException(f"no corpus in {corpus_dir}, run `mint_codes.py corpus` first")

    # Roughly 15 characters of text a second of speech, which sets how many
    # utterances are needed for the requested duration.
    target_chars = minutes * 60 * 15
    jobs: list[tuple[str, str, str]] = []
    # One cursor per language rather than per voice, so the two voices of a language
    # read different text instead of both starting at the top of the same file.
    cursors: dict[str, int] = {}
    fallback = next(iter(by_language))
    total = 0.0
    index = 0
    while total < target_chars:
        voice = voices[index % len(voices)]
        language = voice_language(voice) if by_language.get(voice_language(voice)) else fallback
        pool = by_language[language]
        text = pool[cursors.get(language, 0) % len(pool)]
        cursors[language] = cursors.get(language, 0) + 1
        jobs.append((f"m_{voice}_{index:05d}", voice, text))
        total += len(text)
        index += 1
    logger.info(f"{len(jobs)} utterances, about {total / 15 / 60:.0f} min of speech, "
                f"{len(voices)} voices, holding out {holdout}")

    destination = Path(dump_dir)
    destination.mkdir(parents=True, exist_ok=True)
    stale = [job[0] for job in jobs if (destination / f"{job[0]}.i16").exists()]
    if stale:
        # The server appends, so reusing a tag would glue two takes into one
        # sequence with a discontinuity in the middle.
        raise click.ClickException(f"{len(stale)} dump files already exist (e.g. {stale[0]}.i16); "
                                   "move or delete them, tags must be fresh")

    endpoint = f"{url.rstrip('/')}/v1/audio/speech"
    done = {"ok": 0, "failed": 0, "seconds": 0.0}

    def run(job: tuple[str, str, str]) -> None:
        tag, voice, text = job
        body = tts.build_speech_body(text=text, response_format="wav", speed=1.0,
                                     voice=voice, dump_codes=tag)
        try:
            status, payload, _, elapsed = tts._post_json(endpoint, body, timeout)
        except OSError as error:
            # A dropped connection or a timeout is one lost utterance out of
            # hundreds, not a reason to abandon the run and lose the rest.
            done["failed"] += 1
            logger.warning(f"{tag}: {error}")
            return
        if status != 200:
            done["failed"] += 1
            logger.warning(f"{tag}: HTTP {status} in {elapsed:.1f}s: {payload[:200]!r}")
            return
        done["ok"] += 1
        done["seconds"] += elapsed
        if done["ok"] % 20 == 0:
            logger.info(f"{done['ok']}/{len(jobs)} done, {done['failed']} failed, "
                        f"{done['seconds'] / max(done['ok'], 1):.1f}s per call")

    with ThreadPoolExecutor(max_workers=concurrency) as pool_executor:
        list(pool_executor.map(run, jobs))
    logger.info(f"generation finished: {done['ok']} ok, {done['failed']} failed")

    files, frames = collect_dumps(destination, Path(codes_dir))
    logger.success(f"collected {files} files, {frames} frames "
                   f"({frames / 12.5 / 60:.1f} min of speech) into {codes_dir}")


@cli.command()
@click.option("--dump-dir", default=str(DUMP_DIR), show_default=True, help="Where the .i16 dumps are.")
@click.option("--codes-dir", default=str(CODES_DIR), show_default=True, help="Where to write the npy.")
def collect(dump_dir: str, codes_dir: str) -> None:
    """Convert raw dumps to the npy files the trainer reads (mint does this too)."""
    files, frames = collect_dumps(Path(dump_dir), Path(codes_dir))
    logger.success(f"collected {files} files, {frames} frames ({frames / 12.5 / 60:.1f} min)")


if __name__ == "__main__":
    cli()
