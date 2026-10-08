#!/usr/bin/env -S uv run --script
# /// script
# requires-python = ">=3.10"
# dependencies = ["click", "loguru"]
# ///
"""Copy a listenable subset of the FLEURS benchmark out of the cache (stage A7).

Why this exists
---------------
The whole A7 verdict is built on ECAPA cosine distance and Whisper WER, and neither of
those has ever heard anything. Both can agree a voice is fine while it is nasal, muffled
or simply unpleasant, and a dataset built from a voice like that is wasted GPU time. So
before any of the 29 kept voices is used to generate at scale, a human has to listen.

What makes the copy comparable
------------------------------
Every voice in the sweep read the SAME 332 FLEURS sentences at the SAME noise seed, so
picking one sid gives 29 takes that differ ONLY by the voice prefix. This script keeps
that property: it picks sentences that exist for every kept voice, and hands the same
set to each one. Anything else (each voice with its own best take) would flatter whichever
voice got the easy sentence.

The sentences are picked clean on purpose: no guillemets and no non-breaking or thin
space. A7's PROBE proved the exotic space is what drives the model into degeneration
(median WER 47.7 pct -> 10.5 pct with nothing else changed), and `tts.normalise_spaces`
now strips it client side, so a take containing one is no longer representative of what
the pipeline produces. Judging timbre on a degenerate take would be judging the bug.

Layout produced
---------------
    <dest>/README.md                 what to listen for, the sentences, the WER per voice
    <dest>/00_meme_phrase/           one sentence, all 29 voices, for A/B by ear
    <dest>/<cluster>_<role>_<voice>/ that voice reading every picked sentence

Voices sharing a cluster number are the same speaker under two names, so they sort
next to each other: hearing `05_shipped_nl_male` against `05_hybrid_h_frf_nlm` back to
back is the only way to judge whether the hybrid's lower WER costs anything audible.
"""

from __future__ import annotations

import csv
import json
import shutil
from pathlib import Path

import click
from loguru import logger

REPO_ROOT = Path(__file__).resolve().parents[2]
STAGE_DIR = REPO_ROOT / "cache" / "cloning" / "FLEURS"
VOICE_LIST = Path(__file__).resolve().parent / "kept_voices.txt"

# The sentence picker rejects any text carrying one of these. Guillemets are innocent
# (PROBE cleared them) but they travel with the exotic spaces in this corpus, and the
# point of the audition set is to hear the voice rather than the corpus.
BANNED_CHARACTERS = "      　«»"

# Word-count window for a picked sentence. Under 12 words there is not enough audio to
# judge a voice, over 26 the take starts costing more than a listener will sit through.
MIN_WORDS = 12
MAX_WORDS = 26


def read_voice_list(path: Path) -> list[tuple[int, str, str]]:
    """Read the kept-voice roster.

    Parameters
    ----------
    path
        File with `cluster<TAB>role<TAB>voice` rows and `#` comments.

    Returns
    -------
    list of (cluster, role, voice)
        In file order, which is cluster order, so same-speaker rows stay adjacent.
    """
    rows: list[tuple[int, str, str]] = []
    for line in path.read_text(encoding="utf-8").splitlines():
        line = line.strip()
        if not line or line.startswith("#"):
            continue
        cluster, role, voice = line.split("\t")
        rows.append((int(cluster), role, voice))
    return rows


def read_first_attempts(path: Path) -> dict[tuple[str, str], Path]:
    """Map every (voice, sid) to the wav of its FIRST attempt.

    Retries were appended to the same ledger with an `attempt` field, and a retry is by
    construction a take that degenerated. The unaided first attempt is the honest thing
    to audition: it is what a dataset run gets without a QC loop.

    Parameters
    ----------
    path
        `ledger.jsonl` from the sweep.

    Returns
    -------
    dict
        (voice, sid) -> absolute wav path.
    """
    found: dict[tuple[str, str], Path] = {}
    for line in path.read_text(encoding="utf-8").splitlines():
        record = json.loads(line)
        # A missing `attempt` and `attempt: 0` both mean first attempt, and the two
        # conventions coexist in the cache. A7's sweep tagged only retries, so its first
        # attempts carry no key at all (FLEURS: 12,948 of them); the sweep that grew the
        # retry budget stamps every row, so the grid's first attempts are `attempt: 0`
        # (GRID_fr_fr: 14,645). Testing `"attempt" in record` matched A7 and silently
        # rejected every single grid take, which surfaced as "no takes in the ledger" for
        # all 37 voices at once. Compare the value, never the key's presence.
        if "wav" not in record or record.get("attempt", 0) != 0:
            continue
        found[(record["voice"], record["sid"])] = REPO_ROOT / record["wav"]
    return found


def read_wer(path: Path) -> dict[tuple[str, str], float]:
    """Map every (voice, sid) to its first-attempt WER, for the index table."""
    scored: dict[tuple[str, str], float] = {}
    with path.open(encoding="utf-8") as handle:
        for row in csv.DictReader(handle):
            scored[(row["voice"], row["sid"])] = float(row["wer"])
    return scored


def pick_sentences(
    *,
    sentences: dict[str, str],
    voices: list[str],
    wer: dict[tuple[str, str], float],
    available: dict[tuple[str, str], Path],
    count: int,
) -> list[str]:
    """Choose sentences every kept voice reads cleanly, spread over the length range.

    Ranking by median WER alone returns near-duplicates in length and register, so the
    candidates are first ordered by how badly the WORST kept voice did (a sentence no
    voice trips on is the one that isolates timbre), then sampled evenly across the
    word-count ordering so the set spans short to long.

    Parameters
    ----------
    sentences
        sid -> text, the whole FLEURS slice.
    voices
        The kept voices, all of which must have a take.
    wer
        (voice, sid) -> first-attempt WER.
    available
        (voice, sid) -> wav path.
    count
        How many sentences to return.

    Returns
    -------
    list of str
        sids, ordered short sentence first.
    """
    candidates: list[tuple[float, int, str]] = []
    for sid, text in sentences.items():
        if any(character in text for character in BANNED_CHARACTERS):
            continue
        words = len(text.split())
        if not MIN_WORDS <= words <= MAX_WORDS:
            continue
        if any((voice, sid) not in available for voice in voices):
            continue
        scores = [wer.get((voice, sid)) for voice in voices]
        if any(score is None for score in scores):
            continue
        candidates.append((max(scores), words, sid))

    # Keep a generous shortlist of easy sentences, then spread the final pick across it
    # by length rather than taking the top `count`, which would return one register.
    candidates.sort()
    shortlist = candidates[: max(count * 4, count)]
    shortlist.sort(key=lambda row: row[1])
    if len(shortlist) <= count:
        return [row[2] for row in shortlist]
    step = (len(shortlist) - 1) / (count - 1) if count > 1 else 0
    return [shortlist[round(index * step)][2] for index in range(count)]


def write_index(
    *,
    dest: Path,
    roster: list[tuple[int, str, str]],
    picked: list[str],
    sentences: dict[str, str],
    wer: dict[tuple[str, str], float],
) -> None:
    """Write the README that tells the listener what they are hearing and what to judge."""
    lines: list[str] = []
    lines.append("# Les 29 voix retenues, a ecouter")
    lines.append("")
    lines.append(
        "Genere par Claude Code depuis `cache/cloning/FLEURS` (banc A7, Voxtral TTS via vLLM). "
        "Chaque voix lit EXACTEMENT les memes phrases avec la meme graine de bruit, "
        "donc deux fichiers portant le meme numero de phrase ne different que par la voix."
    )
    lines.append("")
    lines.append(
        "Les 39 voix du banc se reduisent a **20 locuteurs** (cosinus ECAPA >= 0.7). "
        "Le numero en tete de chaque dossier est le locuteur: `05_shipped_nl_male` et "
        "`05_hybrid_h_frf_nlm` sont la MEME personne, l'hybride lisant avec l'articulation "
        "d'une autre voix. C'est la seule paire qui merite une ecoute comparee: "
        "si l'hybride ne sonne pas mieux, sa meilleure WER ne vaut pas le risque de derapage."
    )
    lines.append("")
    lines.append("## Quoi ecouter")
    lines.append("")
    lines.append(
        "1. `00_meme_phrase/` contient la phrase 1 dite par les 29 voix, c'est le A/B rapide."
    )
    lines.append(
        "2. Ensuite un dossier par voix, pour entendre si le timbre tient sur plusieurs phrases."
    )
    lines.append(
        "3. Les prises sont des PREMIERES tentatives, sans reprise ni selection: c'est ce que "
        "produit une generation sans controle qualite."
    )
    lines.append(
        "4. Les phrases ont ete choisies sans espace insecable ni guillemets, parce que "
        "l'espace fine est ce qui fait deraper le modele (PROBE: WER mediane 47.7 pct -> 10.5 pct) "
        "et que `tts.normalise_spaces` la supprime desormais cote client."
    )
    lines.append("")
    lines.append("## Les phrases")
    lines.append("")
    lines.append("| # | sid | mots | texte |")
    lines.append("|---|---|---|---|")
    for index, sid in enumerate(picked, start=1):
        text = sentences[sid]
        lines.append(f"| {index} | `{sid}` | {len(text.split())} | {text} |")
    lines.append("")
    lines.append("## WER de chaque voix sur ces phrases")
    lines.append("")
    if wer:
        lines.append(
            "WER Whisper de la premiere tentative, en pourcent. Sert a savoir si une prise "
            "deplaisante est un probleme de voix ou juste une prise ratee."
        )
    else:
        # A run synthesized without the stt pass has no WER at all, which is a normal state
        # for a listening session. Say so in the file rather than leaving an empty table or,
        # as this did before, raising a KeyError after the tree was already written.
        lines.append(
            "Cette serie n'a pas ete transcrite (SCORE=0) : aucun WER disponible. "
            "Relancer avec SCORE=1 pour les chiffres."
        )
    lines.append("")
    header = "| locuteur | role | voix | " + " | ".join(str(i) for i in range(1, len(picked) + 1)) + " |"
    lines.append(header)
    lines.append("|---" * (3 + len(picked)) + "|")
    for cluster, role, voice in roster:
        cells = ["-" if (voice, sid) not in wer else f"{100 * wer[(voice, sid)]:.0f}"
                 for sid in picked]
        lines.append(f"| {cluster} | {role} | `{voice}` | " + " | ".join(cells) + " |")
    lines.append("")
    (dest / "README.md").write_text("\n".join(lines) + "\n", encoding="utf-8")


@click.command()
@click.option(
    "--dest",
    required=True,
    type=click.Path(file_okay=False, path_type=Path),
    help="Directory to create the audition tree in.",
)
@click.option(
    "--sentences",
    "sentence_count",
    default=8,
    show_default=True,
    help="How many sentences each voice reads, picked for clarity (0 = every sentence in "
         "the run's list, in order, unfiltered).",
)
@click.option(
    "--voice-list",
    default=str(VOICE_LIST),
    show_default=True,
    type=click.Path(dir_okay=False, path_type=Path),
    help="Roster of kept voices (cluster/role/voice, tab separated).",
)
@click.option(
    "--stage-dir",
    default=str(STAGE_DIR),
    show_default=True,
    type=click.Path(file_okay=False, path_type=Path),
    help="Sweep output directory holding ledger.jsonl, sentences.json and the wavs.",
)
def main(*, dest: Path, sentence_count: int, voice_list: Path, stage_dir: Path) -> None:
    """Build a listening copy of the kept voices under DEST."""
    roster = read_voice_list(voice_list)
    voices = [voice for _, _, voice in roster]
    logger.info(f"roster: {len(voices)} voices across {len({c for c, _, _ in roster})} speakers")

    sentences = {row["sid"]: row["text"] for row in json.loads((stage_dir / "sentences.json").read_text(encoding="utf-8"))}
    available = read_first_attempts(stage_dir / "ledger.jsonl")
    # A run that was never scored has no per_sentence.csv, which is a legitimate state
    # rather than a missing step: `--sentences 0` needs no WER, since it picks nothing.
    scored = stage_dir / "per_sentence.csv"
    wer = read_wer(scored) if scored.is_file() else {}

    missing = [voice for voice in voices if not any(key[0] == voice for key in available)]
    if missing:
        raise click.ClickException(f"no takes in the ledger for: {', '.join(missing)}")

    if sentence_count == 0:
        # Hand-picked corpus: the selection already happened, outside this tool. Every
        # filter in pick_sentences would now work against the intent, since they exist to
        # find EASY sentences in a big FLEURS slice (clarity ranking, a 12-26 word window,
        # no guillemets) whereas a curated list is usually hard on purpose. On
        # corpus_medical_fr.txt the window alone would drop the 4 longest of 10, and the
        # clarity rank needs a WER for every voice, which an unscored run does not have.
        picked = list(sentences)
        logger.info(f"--sentences 0: taking all {len(picked)} sentences as listed")
    else:
        picked = pick_sentences(
            sentences=sentences, voices=voices, wer=wer, available=available, count=sentence_count
        )
        if len(picked) < sentence_count:
            logger.warning(f"only {len(picked)} sentences satisfied every kept voice")
    for index, sid in enumerate(picked, start=1):
        logger.info(f"sentence {index} {sid} ({len(sentences[sid].split())} words)")

    dest.mkdir(parents=True, exist_ok=True)
    side_by_side = dest / "00_meme_phrase"
    side_by_side.mkdir(exist_ok=True)

    copied = 0
    for cluster, role, voice in roster:
        # Zero padded so the shell and the file manager both sort speakers numerically,
        # which is what keeps a hybrid next to the shipped voice it shares a cluster with.
        voice_dir = dest / f"{cluster:02d}_{role}_{voice}"
        voice_dir.mkdir(exist_ok=True)
        for index, sid in enumerate(picked, start=1):
            source = available[(voice, sid)]
            shutil.copyfile(source, voice_dir / f"{index:02d}_{sid}_{voice}.wav")
            copied += 1
            if index == 1:
                # The same first take also lands in the A/B directory, named by speaker so
                # the 29 files there sort into the cluster pairs rather than alphabetically.
                shutil.copyfile(source, side_by_side / f"{cluster:02d}_{role}_{voice}.wav")
                copied += 1
        logger.debug(f"{voice}: {len(picked)} takes -> {voice_dir.name}")

    write_index(dest=dest, roster=roster, picked=picked, sentences=sentences, wer=wer)
    total_bytes = sum(path.stat().st_size for path in dest.rglob("*.wav"))
    logger.success(f"{copied} wavs ({total_bytes / 1e6:.0f} MB) under {dest}")


if __name__ == "__main__":
    main()
