#!/usr/bin/env -S uv run --script
# /// script
# requires-python = ">=3.10"
# dependencies = ["click", "loguru"]
# ///
"""Rank a sweep of voice takes by intelligibility and distinctness (stage A5).

The question A5 exists to answer is how many *usable, distinct* French voices the 20
shipped ones can be made to yield. Two separate properties, and only one of them is a
matter of taste:

- Intelligible. Measurable without ears: transcribe the take and compare against the
  text it was asked to read. A voice that garbles "soixante-quinze" is rejected before
  anyone listens to it. A2 measured that accented prefixes break on exactly that sort of
  material, so the sweep text is chosen to provoke it.
- Distinct. Also measurable, imperfectly: median F0 separates most speakers, and takes
  that sit on top of each other in pitch are worth listening to as a pair to check
  whether they are the same voice wearing a different name.

What is left for the listener is naturalness and identity, which no metric here settles.
The point is to spend those ears on the survivors instead of on the whole sweep.

Deliberately stdlib-only apart from click and loguru: it shells out to `asr.py` rather
than reimplementing the transcription contract, and imports `wav_stats` rather than
recomputing pitch. Neither belongs in two places.
"""

from __future__ import annotations

import json
import re
import subprocess
import sys
import unicodedata
from pathlib import Path

import click
from loguru import logger

sys.path.insert(0, str(Path(__file__).resolve().parent))

from wav_stats import median_f0, read_wav  # noqa: E402 - sys.path must be set first

REPO_ROOT = Path(__file__).resolve().parents[2]
# Anything above this word error rate is garbled rather than merely accented. A2 heard
# es_male mangle numbers while staying recognisably speech, which lands around here.
WER_REJECT = 0.30


def french_numeral(n: int) -> str:
    """Spell an integer in French, for 0 to 999.

    Exists because the ASR writes numbers as digits ("75") while a voice reading the
    sweep text says them as words ("soixante-quinze"). Without this, every take would be
    charged several word errors for a number it pronounced perfectly, which is the
    opposite of what the number-heavy text is there to measure.

    Parameters
    ----------
    n : int
        The value to spell, 0 to 999.

    Returns
    -------
    str
        French spelling, hyphenated in the usual way.
    """
    units = ["zero", "un", "deux", "trois", "quatre", "cinq", "six", "sept", "huit", "neuf",
             "dix", "onze", "douze", "treize", "quatorze", "quinze", "seize", "dix-sept",
             "dix-huit", "dix-neuf"]
    tens = {20: "vingt", 30: "trente", 40: "quarante", 50: "cinquante", 60: "soixante"}
    if n < 20:
        return units[n]
    if n < 100:
        if n >= 80:
            # 80 to 99 is "quatre-vingt" plus 0 to 19, and bare 80 takes an s.
            rest = n - 80
            return "quatre-vingts" if rest == 0 else f"quatre-vingt-{units[rest]}"
        if n >= 70:
            # 70 to 79 is "soixante" plus 10 to 19, not a tens word of its own, and 71
            # keeps the "et" that 72 onwards drop.
            return "soixante et onze" if n == 71 else f"soixante-{units[n - 60]}"
        base, rest = (n // 10) * 10, n % 10
        if rest == 0:
            return tens[base]
        # 21, 31 ... take "et un"; the rest hyphenate.
        return f"{tens[base]} et un" if rest == 1 else f"{tens[base]}-{units[rest]}"
    hundreds, rest = divmod(n, 100)
    head = "cent" if hundreds == 1 else f"{units[hundreds]}-cents" if rest == 0 else f"{units[hundreds]}-cent"
    return head if rest == 0 else f"{head} {french_numeral(rest)}"


# Written form -> digits, longest first so "soixante quinze" is matched before "soixante".
_NUMBER_WORDS = sorted(((tuple(re.sub(r"[^a-z]", " ", french_numeral(n)).split()), str(n))
                        for n in range(0, 1000)),
                       key=lambda pair: len(pair[0]), reverse=True)


def fold_numbers(words: list[str]) -> list[str]:
    """Replace spelled-out French numbers with their digits.

    Applied to both sides of the comparison, so it does not matter which one spells them
    out; what matters is that both end up in the same form.

    Parameters
    ----------
    words : list[str]
        Tokenised, lowercased, unaccented words.

    Returns
    -------
    list[str]
        The same words with any number phrase collapsed to a single digit token.
    """
    out, i = [], 0
    while i < len(words):
        for form, digits in _NUMBER_WORDS:
            if tuple(words[i:i + len(form)]) == form:
                out.append(digits)
                i += len(form)
                break
        else:
            # "et" only ever joins a number phrase ("vingt et un"), which the loop above
            # matches whole, so a surviving "et" is an ordinary conjunction. Keep it.
            out.append(words[i])
            i += 1
    return out


def normalise(text: str) -> list[str]:
    """Reduce a transcript to comparable words.

    Case, punctuation and accents all differ between what was asked for and what an ASR
    returns without either being an error, so they are removed. Spelled-out numbers are
    folded to digits on both sides: the reference avoids digits on purpose, but the ASR
    writes them anyway ("75" for "soixante-quinze"), and charging a take for that would
    invert the very thing the number-heavy text is there to measure.

    Parameters
    ----------
    text : str
        Raw text or transcript.

    Returns
    -------
    list[str]
        Lowercased, unaccented, punctuation-free words.
    """
    folded = unicodedata.normalize("NFKD", text.lower())
    folded = "".join(c for c in folded if not unicodedata.combining(c))
    return fold_numbers(re.sub(r"[^a-z0-9\s]", " ", folded).split())


def word_error_rate(reference: str, hypothesis: str) -> float:
    """Levenshtein distance over words, divided by the reference length.

    Parameters
    ----------
    reference : str
        The text the voice was asked to read.
    hypothesis : str
        What the ASR heard.

    Returns
    -------
    float
        0.0 for a perfect match, 1.0 for nothing recognisable. Can exceed 1.0 when the
        transcript invents words, which is itself a useful signal.
    """
    ref, hyp = normalise(reference), normalise(hypothesis)
    if not ref:
        return float("nan")
    # Single-row Levenshtein: the sweep is dozens of short takes, so clarity beats speed.
    previous = list(range(len(hyp) + 1))
    for i, r in enumerate(ref, start=1):
        current = [i]
        for j, h in enumerate(hyp, start=1):
            current.append(min(previous[j] + 1,          # deletion
                               current[j - 1] + 1,       # insertion
                               previous[j - 1] + (r != h)))
        previous = current
    return previous[-1] / len(ref)


def transcribe(wav: Path, *, url: str, api_key: str | None, model: str | None,
               language: str) -> str:
    """Transcribe one take by shelling out to the repo's own ASR client.

    Parameters
    ----------
    wav : Path
        The take to transcribe.
    url : str
        ASR endpoint, passed through to `asr.py`.
    api_key : str, optional
        API key; the ASR service is key-protected.
    model : str, optional
        Model name. Newer servers answer 422 without one.
    language : str
        Language hint, `fr` for this sweep.

    Returns
    -------
    str
        The transcript, or an empty string when the call failed (which scores as a total
        miss rather than crashing the sweep half way through a long run).
    """
    command = [str(REPO_ROOT / "asr.py"), str(wav), "--url", url, "--language", language]
    if api_key:
        command += ["--api-key", api_key]
    if model:
        command += ["--model", model]
    done = subprocess.run(command, capture_output=True, text=True)
    if done.returncode != 0:
        logger.warning(f"{wav.name}: asr failed ({done.stderr.strip().splitlines()[-1:]})")
        return ""
    return done.stdout.strip()


@click.command()
@click.option("--wav-dir", type=click.Path(exists=True, file_okay=False, path_type=Path),
              required=True, help="Directory of takes, one wav per voice.")
@click.option("--text", "reference", required=True, help="The text every take was asked to read.")
@click.option("--url", default="http://localhost:8002/v1/audio/transcriptions", show_default=True,
              help="ASR endpoint.")
@click.option("--api-key", default=None, envvar="CRISPASR_API_KEY",
              help="ASR API key (env: CRISPASR_API_KEY).")
@click.option("--model", default=None, envvar="CRISPASR_MODEL",
              help="ASR model name (env: CRISPASR_MODEL).")
@click.option("--language", default="fr", show_default=True, help="Language hint for the ASR.")
@click.option("--out", type=click.Path(path_type=Path), default=None,
              help="Where to write the markdown table (default: <wav-dir>/../survey.md).")
def main(*, wav_dir: Path, reference: str, url: str, api_key: str | None, model: str | None,
         language: str, out: Path | None) -> None:
    """Score every take in WAV_DIR and rank them for a listening pass."""
    takes = sorted(wav_dir.glob("*.wav"))
    if not takes:
        raise click.ClickException(f"no wavs in {wav_dir}")
    logger.info(f"scoring {len(takes)} takes against {len(normalise(reference))} reference words")

    rows = []
    for wav in takes:
        samples, rate = read_wav(wav)
        heard = transcribe(wav, url=url, api_key=api_key, model=model, language=language)
        rows.append({"voice": wav.stem,
                     "wer": word_error_rate(reference, heard),
                     "f0": median_f0(samples, rate),
                     "seconds": len(samples) / rate,
                     "heard": heard})
        logger.info(f"{wav.stem:22} wer {100 * rows[-1]['wer']:5.1f}%  f0 {rows[-1]['f0']:6.1f} Hz")

    rows.sort(key=lambda r: (r["wer"], r["f0"]))
    keep = [r for r in rows if r["wer"] <= WER_REJECT]
    logger.info(f"{len(keep)} of {len(rows)} takes are intelligible (wer <= {100 * WER_REJECT:.0f}%)")

    # Pitch neighbours among the survivors: two takes this close are worth hearing back
    # to back, because the cheap way to end up with 20 voices that are really 8 is to
    # never check. Not proof of identity, just the shortlist for the ears.
    by_pitch = sorted((r for r in keep if r["f0"] == r["f0"]), key=lambda r: r["f0"])
    neighbours = [(a["voice"], b["voice"], b["f0"] - a["f0"])
                  for a, b in zip(by_pitch, by_pitch[1:]) if b["f0"] - a["f0"] < 10.0]

    destination = out or wav_dir.parent / "survey.md"
    lines = ["# A5 voice survey", "",
             f"{len(rows)} takes, {len(keep)} intelligible at WER <= {100 * WER_REJECT:.0f}%.",
             "Sorted by WER, so the ones worth your ears are at the top.", "",
             "Fill in the last three columns while listening. `distinct?` means distinct",
             "from every take above it, not merely from fr_female.", "",
             "| voice | WER | F0 (Hz) | s | natural? | distinct? | keep |", "|---|---|---|---|---|---|---|"]
    for r in rows:
        f0 = "-" if r["f0"] != r["f0"] else f"{r['f0']:.0f}"
        lines.append(f"| {r['voice']} | {100 * r['wer']:.0f}% | {f0} | {r['seconds']:.1f} | | | |")
    if neighbours:
        lines += ["", "## Pitch neighbours (listen back to back, same voice?)", ""]
        lines += [f"- {a} and {b}, {gap:.1f} Hz apart" for a, b, gap in neighbours]
    lines += ["", "## Transcripts of the rejected takes", ""]
    lines += [f"- **{r['voice']}** (WER {100 * r['wer']:.0f}%): {r['heard'] or '(nothing)'}"
              for r in rows if r["wer"] > WER_REJECT] or ["- none"]
    destination.write_text("\n".join(lines) + "\n")
    (destination.with_suffix(".json")).write_text(json.dumps(rows, indent=2, ensure_ascii=False))
    logger.success(f"wrote {destination}")


if __name__ == "__main__":
    main()
