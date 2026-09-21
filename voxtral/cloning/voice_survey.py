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
- Fast or slow. Words per second, which turned out to spread 3x across the 20 bases
  and is both a diversity axis in its own right and a naturalness warning: the takes
  under 2 words/s are the English-styled voices drawling through French.

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
from collections.abc import Sequence
from pathlib import Path

import click
from loguru import logger

sys.path.insert(0, str(Path(__file__).resolve().parent))

from wav_stats import median_f0, read_wav  # noqa: E402 - sys.path must be set first

REPO_ROOT = Path(__file__).resolve().parents[2]
# Anything above this word error rate is garbled rather than merely accented. A2 heard
# es_male mangle numbers while staying recognisably speech, which lands around here.
ASR_ROUTE = "/v1/audio/transcriptions"
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

    French-only, and deliberately left that way even now that the sweep reads eight
    languages. The table is applied to reference and hypothesis alike, so on a non-French
    corpus it can only ever fold both sides identically (Spanish "un libro" becomes
    "1 libro" on both), which changes no error rate. It simply does no useful work there:
    an Italian take that reads "sei" where the text says "6" is still charged for it. The
    cost of that is a slightly pessimistic WER on number-heavy sentences outside French,
    and the fix is a per-language table rather than anything structural here.

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


# German is the one Latin language whose accent folding NFKD cannot do: U+00DF has no
# decomposition, so it would be stripped as punctuation and turn "Straße" into two tokens
# ("stra", "e"), charging every take for a letter the ASR spelled correctly. Mapped to the
# conventional "ss" instead, on both sides, so the two agree either way it was written.
_LETTER_SUBSTITUTIONS = {"\u00df": "ss"}


def normalise(text: str) -> list[str]:
    """Reduce a transcript to comparable words, in any script.

    Case, punctuation and accents all differ between what was asked for and what an ASR
    returns without either being an error, so they are removed. Spelled-out numbers are
    folded to digits on both sides: the reference avoids digits on purpose, but the ASR
    writes them anyway ("75" for "soixante-quinze"), and charging a take for that would
    invert the very thing the number-heavy text is there to measure.

    **Accents are folded on Latin text only, and that restriction is load-bearing.** In
    Latin script a combining mark is a diacritic an ASR may or may not emit; in Devanagari
    the same Unicode category holds the vowel signs, which are not decoration but part of
    the syllable, and in Arabic it holds harakat. Folding them globally deleted every
    Hindi and Arabic word, and the ASCII allowlist that followed deleted whatever was
    left, so both scripts normalised to the empty list and every take scored NaN. What
    survives here is therefore "letters, marks and digits of any script", with diacritics
    dropped only where the base character was ASCII.

    Parameters
    ----------
    text : str
        Raw text or transcript.

    Returns
    -------
    list[str]
        Lowercased, punctuation-free words, unaccented where the script is Latin.
    """
    lowered = "".join(_LETTER_SUBSTITUTIONS.get(character, character)
                      for character in text.lower())
    kept: list[str] = []
    latin_base = False
    for character in unicodedata.normalize("NFKD", lowered):
        if unicodedata.combining(character):
            # A diacritic on an ASCII base is decoration an ASR may drop; anywhere else it
            # carries the sound, so it stays.
            if not latin_base:
                kept.append(character)
            continue
        latin_base = character.isascii()
        # Letters (L*), marks (M*) and numbers (N*) are content in every script; the rest
        # is punctuation, symbols and separators, and becomes a word break.
        kept.append(character if unicodedata.category(character)[0] in "LMN" else " ")
    return fold_numbers("".join(kept).split())


def _levenshtein(reference: Sequence, hypothesis: Sequence) -> int:
    """Edit distance between two sequences, counted in their own elements.

    Shared by the word-level and character-level error rates so the two can never
    drift apart: WER and CER differ only in what they tokenise into, never in how the
    alignment is computed.

    Parameters
    ----------
    reference : Sequence
        Tokens of the text the voice was asked to read.
    hypothesis : Sequence
        Tokens of what the ASR heard.

    Returns
    -------
    int
        Number of substitutions, insertions and deletions.
    """
    # Single-row Levenshtein. Pure Python is fast enough even for CER on the longest
    # FLEURS sentence (65 words, ~400 characters, so ~160k inner steps or ~30 ms), which
    # is two orders of magnitude below the ASR call that produced the transcript.
    previous = list(range(len(hypothesis) + 1))
    for i, r in enumerate(reference, start=1):
        current = [i]
        for j, h in enumerate(hypothesis, start=1):
            current.append(min(previous[j] + 1,          # deletion
                               current[j - 1] + 1,       # insertion
                               previous[j - 1] + (r != h)))
        previous = current
    return previous[-1]


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
    return _levenshtein(ref, hyp) / len(ref)


def character_error_rate(reference: str, hypothesis: str) -> float:
    """Levenshtein distance over characters, divided by the reference length.

    CER is WER's finer-grained companion and the two are worth reading together. A take
    that says "soixante-quize" loses a whole word under WER but one character under CER,
    so a big gap between the two rates means mangled endings, while WER and CER moving
    together means the voice read genuinely different words.

    Normalisation is deliberately the same one WER uses (case, accents, punctuation and
    spelled-out numbers folded), with the words rejoined by single spaces. Comparing the
    raw strings instead would charge every take for the ASR's punctuation and
    capitalisation habits, which is not the property being measured.

    Parameters
    ----------
    reference : str
        The text the voice was asked to read.
    hypothesis : str
        What the ASR heard.

    Returns
    -------
    float
        0.0 for a perfect match. Can exceed 1.0 when the transcript is longer than the
        reference.
    """
    ref, hyp = " ".join(normalise(reference)), " ".join(normalise(hypothesis))
    if not ref:
        return float("nan")
    return _levenshtein(ref, hyp) / len(ref)


def transcribe(wav: Path, *, url: str, api_key: str | None, model: str | None,
               language: str) -> str:
    """Transcribe one take by shelling out to the repo's own ASR client.

    Parameters
    ----------
    wav : Path
        The take to transcribe.
    url : str
        ASR *base* URL, passed through to `asr.py`, which appends the
        `/v1/audio/transcriptions` route itself.
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
    # asr.py takes a BASE url and builds `<base>/v1/audio/transcriptions` itself, so
    # handing it the full route produces a doubled path and a 404 on every take. The
    # route is stripped here rather than only fixed in the default, so an explicit
    # --url copied from a curl command still works.
    base = url.rstrip("/")
    if base.endswith(ASR_ROUTE):
        base = base[: -len(ASR_ROUTE)].rstrip("/")
    command = [str(REPO_ROOT / "asr.py"), str(wav), "--url", base, "--language", language]
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
@click.option("--url", default="http://localhost:8002", show_default=True,
              help="ASR base URL (the /v1/audio/transcriptions route is appended).")
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

    # Speaking rate is the third measurable axis, and the sweep showed it spreads 3x
    # across the 20 bases (4.5 words/s down to 1.5). It doubles as a naturalness flag:
    # the slowest takes are the English-styled voices drawling their way through French.
    n_words = len(normalise(reference))

    rows = []
    for wav in takes:
        samples, rate = read_wav(wav)
        heard = transcribe(wav, url=url, api_key=api_key, model=model, language=language)
        rows.append({"voice": wav.stem,
                     "wer": word_error_rate(reference, heard),
                     "f0": median_f0(samples, rate),
                     "seconds": len(samples) / rate,
                     "words_per_second": n_words / (len(samples) / rate),
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
             "| voice | WER | F0 (Hz) | s | w/s | natural? | distinct? | keep |",
             "|---|---|---|---|---|---|---|---|"]
    for r in rows:
        f0 = "-" if r["f0"] != r["f0"] else f"{r['f0']:.0f}"
        lines.append(f"| {r['voice']} | {100 * r['wer']:.0f}% | {f0} | {r['seconds']:.1f} | "
                     f"{r['words_per_second']:.1f} | | | |")
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
