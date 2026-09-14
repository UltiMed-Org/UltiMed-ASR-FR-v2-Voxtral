#!/usr/bin/env -S uv run --script
# /// script
# requires-python = ">=3.10"
# dependencies = ["click", "loguru", "torch", "torchaudio", "numpy", "speechbrain"]
# ///
"""Measure how many DIFFERENT speakers a pool of takes actually contains (stage A6).

Why a model and not a statistic
-------------------------------
A5 tried median F0 and it failed: the same voice moved 30 Hz between two sentences,
so F0 ranks prosody, not speakers. What is needed is a function trained to answer
"same person or not", which is exactly what a speaker verification embedding is.
This uses ECAPA-TDNN trained on VoxCeleb (speechbrain/spkrec-ecapa-voxceleb), where
the cosine similarity between two utterances has a published operating point rather
than a threshold someone made up.

The sweep makes this comparison fair in a way a random corpus would not: every voice
reads the SAME sentences with the SAME noise seed, so two takes differ only by the
prefix that generated them. Text and prosody are held still, and the embedding sees
speaker identity.

What it produces
----------------
1. `embed`   one L2-normalised embedding per voice, averaged over its takes.
2. `select`  a greedy max-min pick: start from an anchor, then repeatedly add the
             voice whose similarity to everything already kept is lowest. This is the
             opposite of ranking by intelligibility, which would select the blandest
             voices; it deliberately walks outward from the anchor.
3. `report`  the kept set with the nearest neighbour of each, plus the pairs that are
             too close to be different people, plus a validation section.

The validation section exists because a metric nobody checked is not evidence. Two
checks run automatically: the 20 shipped voices are 20 different recordings and should
sit far apart, and a hybrid should sit CLOSE to its acoustic donor if speaker identity
really lives in the acoustic codes (the sweep's F0 correlation of +0.97 says it does).
If those two come out wrong, nothing below them is worth reading.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import click
import numpy as np
import torch
import torchaudio
from loguru import logger

REPO_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(Path(__file__).resolve().parent))
from sweep import STAGE_DIR, WER_REJECT, read_ledger  # noqa: E402
from wav_stats import read_wav  # noqa: E402

# VoxCeleb-trained ECAPA-TDNN. Ungated, about 80 MB, and the standard baseline for
# this exact question, which matters more than squeezing out the last point of EER.
MODEL_ID = "speechbrain/spkrec-ecapa-voxceleb"
MODEL_RATE = 16000
# Cosine similarity above which two takes are treated as the same speaker. VoxCeleb
# systems of this family sit near 0.25 at equal error rate; 0.5 is deliberately well
# above that, because the cost of wrongly KEEPING two similar voices (a dataset with a
# duplicate) is lower than wrongly discarding a genuinely new one.
SAME_SPEAKER = 0.50
EMBEDDINGS = STAGE_DIR / "embeddings.npz"


def load_classifier(device: str):
    """Load ECAPA once.

    Parameters
    ----------
    device : str
        "cuda" or "cpu".

    Returns
    -------
    speechbrain.inference.speaker.EncoderClassifier
        The loaded model.
    """
    from speechbrain.inference.speaker import EncoderClassifier

    return EncoderClassifier.from_hparams(
        source=MODEL_ID,
        savedir=str(REPO_ROOT / "cache" / "cloning" / "ecapa"),
        run_opts={"device": device},
    )


def embed_wav(classifier, path: Path, device: str) -> np.ndarray:
    """Embed one take.

    Parameters
    ----------
    classifier : EncoderClassifier
        Loaded ECAPA.
    path : Path
        Wav file, any sample rate (resampled to 16 kHz for the model).
    device : str
        Where to run.

    Returns
    -------
    numpy.ndarray
        `[192]` float32, L2 normalised.
    """
    # wav_stats.read_wav and not torchaudio.load: recent torchaudio routes every load
    # through TorchCodec and raises without it, and these files are plain 16-bit PCM that
    # the standard library already reads in the one place this repo keeps that logic.
    samples, rate = read_wav(path)
    waveform = torch.tensor(samples, dtype=torch.float32).unsqueeze(0) / 32768.0
    if rate != MODEL_RATE:
        waveform = torchaudio.functional.resample(waveform, rate, MODEL_RATE)
    if waveform.shape[0] > 1:
        waveform = waveform.mean(dim=0, keepdim=True)
    with torch.no_grad():
        vector = classifier.encode_batch(waveform.to(device)).squeeze().float().cpu().numpy()
    return vector / (np.linalg.norm(vector) + 1e-9)


def load_embeddings() -> tuple[list[str], np.ndarray]:
    """Read the cached per-voice embeddings.

    Returns
    -------
    tuple
        (voice names, `[V, 192]` matrix, rows L2 normalised).
    """
    if not EMBEDDINGS.is_file():
        raise SystemExit(f"no embeddings yet ({EMBEDDINGS}); run `voice_distance.py embed` first")
    data = np.load(EMBEDDINGS, allow_pickle=False)
    return list(data["voices"]), data["vectors"]


def per_sentence_wer() -> dict[str, dict[str, float]]:
    """Every voice's word error rate, kept per sentence rather than collapsed.

    Returns
    -------
    dict
        `{voice: {sid: wer}}`.
    """
    out: dict[str, dict[str, float]] = {}
    for (voice, sid), record in read_ledger().items():
        if "wer" in record:
            out.setdefault(voice, {})[sid] = record["wer"]
    return out


def adjusted_scores(scores: dict[str, dict[str, float]], *, voices: list[str]) -> dict[str, float]:
    """Score voices against each sentence's own difficulty, not against zero.

    The 12 screen sentences are not equally hard: the median voice scores 0 pct on
    "Le castillan est la seule langue officielle" and 33 pct on the one containing
    "Idiothele" and "mygalomorphes", and even fr_female, the reference voice this whole
    project is built around, loses a third of that sentence. Comparing raw word error
    rates therefore mostly compares which sentences a voice happened to be given.

    Subtracting the per-sentence median turns the number into "how much worse than the
    typical voice on this exact sentence", so 0 means typical and negative means better.
    The mean is used rather than the median because a voice that destroys one sentence in
    twelve is not usable for dataset generation, and a median hides exactly that.

    Parameters
    ----------
    scores : dict
        `{voice: {sid: wer}}`.
    voices : list of str
        The comparison set whose medians define "typical" (the deeply measured ones).

    Returns
    -------
    dict
        `{voice: mean excess WER}` for every voice in `scores`.
    """
    import statistics

    baseline: dict[str, float] = {}
    for sid in {sid for voice in voices for sid in scores.get(voice, {})}:
        values = [scores[voice][sid] for voice in voices if sid in scores.get(voice, {})]
        if values:
            baseline[sid] = statistics.median(values)
    return {voice: statistics.mean([wer - baseline.get(sid, 0.0) for sid, wer in sentences.items()])
            for voice, sentences in scores.items() if sentences}


def passing_voices() -> dict[str, float]:
    """Voices whose median WER clears the gate, with that median.

    Distinctness is only asked about voices that can read the language: a broken take
    is easy to make and always far from everything else, so skipping this filter would
    fill the selection with mush.

    Returns
    -------
    dict
        `{voice: median WER}`.
    """
    import statistics

    per: dict[str, list[float]] = {}
    for (voice, _), record in read_ledger().items():
        if "wer" in record:
            per.setdefault(voice, []).append(record["wer"])
    return {voice: statistics.median(wers) for voice, wers in per.items()
            if statistics.median(wers) <= WER_REJECT}


@click.group()
def cli() -> None:
    """Speaker-embedding distance over the A6 takes."""
    logger.remove()
    logger.add(sys.stderr, format="{time:HH:mm:ss} <level>{level: <7}</level> {message}", level="INFO")


@cli.command("embed")
@click.option("--device", default="cuda" if torch.cuda.is_available() else "cpu", show_default=True,
              help="Where to run ECAPA.")
@click.option("--max-takes", default=4, show_default=True,
              help="Takes to average per voice (more is steadier, and slower).")
@click.option("--only-passing/--all", default=False, show_default=True,
              help="Embed only the voices that clear the WER gate.")
def cmd_embed(device: str, max_takes: int, only_passing: bool) -> None:
    """Embed every voice that has takes on disk, averaging over its takes."""
    state = read_ledger()
    wanted = set(passing_voices()) if only_passing else None

    by_voice: dict[str, list[Path]] = {}
    for (voice, _), record in sorted(state.items()):
        if wanted is not None and voice not in wanted:
            continue
        path = REPO_ROOT / record["wav"] if record.get("wav") else None
        if path and path.is_file():
            by_voice.setdefault(voice, []).append(path)
    if not by_voice:
        raise SystemExit("no takes on disk; run the sweep first")

    logger.info(f"embedding {len(by_voice)} voices on {device} "
                f"({min(max_takes, max(len(v) for v in by_voice.values()))} takes each)")
    classifier = load_classifier(device)

    names, vectors = [], []
    for index, (voice, paths) in enumerate(sorted(by_voice.items()), start=1):
        # Averaging over takes then renormalising is the standard way to turn several
        # utterances into one speaker vector: it cancels the per-sentence variation that
        # made F0 useless here, and keeps what is constant across them.
        stacked = np.stack([embed_wav(classifier, path, device) for path in paths[:max_takes]])
        mean = stacked.mean(axis=0)
        names.append(voice)
        vectors.append(mean / (np.linalg.norm(mean) + 1e-9))
        if index % 50 == 0:
            logger.info(f"{index}/{len(by_voice)} voices embedded")

    EMBEDDINGS.parent.mkdir(parents=True, exist_ok=True)
    np.savez(EMBEDDINGS, voices=np.array(names), vectors=np.stack(vectors).astype(np.float32))
    logger.success(f"wrote {len(names)} embeddings to {EMBEDDINGS}")


@cli.command("select")
@click.option("--anchor", default="fr_female", show_default=True,
              help="Voice to start from; the dataset work already uses this one.")
@click.option("--threshold", default=SAME_SPEAKER, show_default=True,
              help="Cosine similarity at which two voices count as the same speaker.")
@click.option("--out", default=None, help="Where to write the markdown (default: <stage>/distinct.md).")
def cmd_select(anchor: str, threshold: float, out: str | None) -> None:
    """Greedily pick a maximally spread set of intelligible voices, and report it."""
    names, vectors = load_embeddings()
    index_of = {name: i for i, name in enumerate(names)}
    gate = passing_voices()
    pool = {entry["name"]: entry for entry in json.loads((STAGE_DIR / "pool.json").read_text())}

    eligible = [name for name in names if name in gate]
    if anchor not in index_of:
        raise SystemExit(f"anchor {anchor} has no embedding")
    logger.info(f"{len(eligible)} voices clear the WER gate and have embeddings")

    similarity = vectors @ vectors.T

    # Greedy max-min (farthest point traversal). Each step adds the eligible voice whose
    # closest already-kept neighbour is as far away as possible, so the set spreads out
    # instead of clustering, and the ORDER is itself the answer: everything before the
    # threshold crossing is a distinct voice, everything after is a near duplicate.
    kept = [anchor]
    order = []
    remaining = [name for name in eligible if name != anchor]
    while remaining:
        kept_rows = [index_of[name] for name in kept]
        best, best_similarity = None, 2.0
        for name in remaining:
            worst = max(similarity[index_of[name], row] for row in kept_rows)
            if worst < best_similarity:
                best, best_similarity = name, worst
        order.append((best, best_similarity))
        kept.append(best)
        remaining.remove(best)

    distinct = [anchor] + [name for name, sim in order if sim < threshold]
    logger.info(f"{len(distinct)} voices stay below {threshold} cosine similarity to each other")

    # Validation, before any of the numbers above are taken seriously.
    bases = [name for name in names if name not in pool]
    base_rows = [index_of[name] for name in bases]
    base_pairs = [similarity[a, b] for i, a in enumerate(base_rows) for b in base_rows[i + 1:]]
    donor_pairs, semantic_pairs = [], []
    for name, entry in pool.items():
        if entry["kind"] != "hybrid" or name not in index_of:
            continue
        semantic, acoustic = entry["parents"]
        if acoustic in index_of:
            donor_pairs.append(similarity[index_of[name], index_of[acoustic]])
        if semantic in index_of:
            semantic_pairs.append(similarity[index_of[name], index_of[semantic]])

    lines = ["# A6 distinctness", "",
             f"{len(eligible)} voices clear the WER gate (median <= {100 * WER_REJECT:.0f} pct) "
             f"and were embedded with ECAPA-TDNN.",
             f"**{len(distinct)} of them are different speakers** at cosine similarity "
             f"< {threshold}, walking outward from `{anchor}`.", "",
             "## Validation", "",
             f"- The 20 shipped voices against each other: mean similarity "
             f"{np.mean(base_pairs):.2f}, max {np.max(base_pairs):.2f}. These are 20 genuinely "
             "different recordings, so this is what 'different speaker' looks like on this metric.",
             f"- A hybrid against its ACOUSTIC donor: mean {np.mean(donor_pairs):.2f}.",
             f"- A hybrid against its SEMANTIC donor: mean {np.mean(semantic_pairs):.2f}.",
             "", "The last two are the test of where identity lives. The sweep's pitch correlation "
             "(+0.97 with the acoustic donor, -0.05 with the semantic one) predicts the acoustic "
             "number should be much the higher of the two.", "",
             "## The selection, in the order it was picked", "",
             "Each row's similarity is to the closest voice ABOVE it, so the list stays usable "
             "if you cut it anywhere: take the first N rows and you have the N most spread-out "
             "voices available.", "",
             "| # | voice | kind | parents | WER | similarity to nearest kept |",
             "|---|---|---|---|---|---|"]
    rows = [(anchor, 0.0)] + order
    for position, (name, sim) in enumerate(rows, start=1):
        entry = pool.get(name, {"kind": "base", "parents": []})
        marker = "" if sim < threshold else " (duplicate)"
        lines.append(f"| {position} | {name} | {entry['kind']} | {'+'.join(entry['parents'])} | "
                     f"{100 * gate[name]:.0f}% | {sim:.2f}{marker} |")

    destination = Path(out) if out else STAGE_DIR / "distinct.md"
    destination.write_text("\n".join(lines) + "\n")
    logger.success(f"wrote {destination}")


@cli.command("shortlist")
@click.option("--kind", default=None, help="Only this kind of candidate (base, hybrid, mixn).")
@click.option("--count", default=10, show_default=True, help="How many takes to shortlist.")
@click.option("--sid", default="s00", show_default=True, help="Which sentence to play.")
@click.option("--max-wer", default=WER_REJECT, show_default=True, help="Reject anything worse than this.")
@click.option("--out", default=None, help="Where to write the markdown (default: <stage>/shortlist.md).")
def cmd_shortlist(kind: str | None, count: int, sid: str, max_wer: float, out: str | None) -> None:
    """Write a listening list of the candidates furthest from every shipped voice.

    This is the question the embedding cannot answer on its own. Distance from every
    base means one of two things, and they are opposite: either a genuinely new speaker,
    or audio degraded enough that the model cannot place it. The WER gate rules out the
    worst of the second case but not all of it, so the last step is ears.
    """
    names, vectors = load_embeddings()
    index_of = {name: i for i, name in enumerate(names)}
    gate = passing_voices()
    pool = {entry["name"]: entry for entry in json.loads((STAGE_DIR / "pool.json").read_text())}
    bases = [name for name in names if name not in pool]
    similarity = vectors @ vectors.T

    rows = []
    for name, entry in pool.items():
        if name not in index_of or name not in gate or gate[name] > max_wer:
            continue
        if kind and entry["kind"] != kind:
            continue
        nearest, nearest_name = max((similarity[index_of[name], index_of[base]], base) for base in bases)
        rows.append((nearest, name, nearest_name, entry))
    rows.sort()

    lines = [f"# A6 listening shortlist{f' ({kind})' if kind else ''}", "",
             f"The {min(count, len(rows))} candidates that sit furthest from EVERY shipped voice, "
             f"out of {len(rows)} that clear {100 * max_wer:.0f} pct WER.", "",
             "For reference, two different shipped voices sit at most 0.53 apart on this metric, "
             "so anything below that is, by the measure, a speaker who does not exist in the "
             "checkpoint. Your ear decides whether it is a person or a smear.", "",
             "| play | voice | parents | WER | nearest shipped voice |", "|---|---|---|---|---|"]
    picked = rows[:count]
    for nearest, name, nearest_name, entry in picked:
        lines.append(f"| `cache/cloning/A6/wav/{name}/{sid}.wav` | {name} | "
                     f"{'+'.join(entry['parents'])} | {100 * gate[name]:.0f}% | "
                     f"{nearest_name} at {nearest:.2f} |")
    lines += ["", "Bulk play, in order:", "", "```",
              "for v in " + " ".join(name for _, name, _, _ in picked) + "; do",
              f'  echo "$v"; ffplay -autoexit -nodisp "cache/cloning/A6/wav/$v/{sid}.wav" 2>/dev/null',
              "done", "```"]

    destination = Path(out) if out else STAGE_DIR / "shortlist.md"
    destination.write_text("\n".join(lines) + "\n")
    logger.success(f"{len(picked)} takes shortlisted, wrote {destination}")


@cli.command("keepers")
@click.option("--max-wer", default=0.15, show_default=True,
              help="Clarity a representative must reach to be kept.")
@click.option("--out", default=None, help="Where to write the markdown (default: <stage>/keepers.md).")
@click.option("--contenders", default=0, show_default=True,
              help="Also write the top N per identity to contenders.txt, for a deeper re-sweep.")
def cmd_keepers(max_wer: float, out: str | None, contenders: int) -> None:
    """Pick one usable voice per speaker identity: the production set.

    Every candidate is assigned to the shipped voice it sounds most like, which by the
    validation above is its acoustic donor, and the clearest member of each group wins.
    That is the whole practical yield of A6: identities that could not read French get
    represented by a hybrid that can, and identities that were already fine keep their
    own voice unless a variant reads better.
    """
    names, vectors = load_embeddings()
    index_of = {name: i for i, name in enumerate(names)}
    gate = passing_voices()
    pool = {entry["name"]: entry for entry in json.loads((STAGE_DIR / "pool.json").read_text())}
    bases = [name for name in names if name not in pool]
    similarity = vectors @ vectors.T

    scores = per_sentence_wer()
    # "Typical" is defined by the deepest tier available, which after a contenders
    # re-sweep is the 60 finalists and before one is everybody.
    deepest = max(len(sentences) for sentences in scores.values())
    reference = [voice for voice, sentences in scores.items() if len(sentences) == deepest]
    adjusted = adjusted_scores(scores, voices=reference)

    groups: dict[str, list[tuple[float, str]]] = {base: [] for base in bases}
    for name in names:
        if name not in gate or gate[name] > max_wer:
            continue
        identity = max(bases, key=lambda base: similarity[index_of[name], index_of[base]])
        groups[identity].append((adjusted[name], name))

    # Only the deepest-measured members of a group compete. Without this a candidate
    # measured on 2 sentences beats one measured on 12 by never having been asked the
    # hard ones, which is how the depth-2 pass produced picks that did not survive
    # contact with more data.
    for identity, members in groups.items():
        deep = [(score, name) for score, name in members if len(scores[name]) == deepest]
        if deep:
            groups[identity] = deep

    lines = ["# A6 keepers: one voice per speaker identity", "",
             "Each shipped voice is a speaker. A6 did not add speakers (a hybrid sits at 0.86 "
             "cosine similarity to its acoustic donor), it added a way to make a speaker read "
             "French. This table takes the clearest representative of each identity.", "",
             f"Representative must score at most {100 * max_wer:.0f} pct median WER, and is chosen",
             "by mean excess WER against each sentence's own difficulty (0 = as good as the",
             f"typical voice on that sentence, negative = better). Deepest tier: {deepest} takes.", "",
             "| identity | keep | kind | own WER | kept WER | excess | takes |",
             "|---|---|---|---|---|---|---|"]
    keepers = []
    for base in sorted(groups):
        if not groups[base]:
            lines.append(f"| {base} | (none clears {100 * max_wer:.0f} pct) | | "
                         f"{100 * gate[base]:.0f}% | |" if base in gate else
                         f"| {base} | (none clears {100 * max_wer:.0f} pct) | | fails gate | |")
            continue
        best_excess, best = min(groups[base])
        keepers.append(best)
        kind = pool.get(best, {"kind": "base"})["kind"]
        own = f"{100 * gate[base]:.0f}%" if base in gate else "fails gate"
        lines.append(f"| {base} | {best} | {kind} | {own} | {100 * gate[best]:.0f}% | "
                     f"{100 * best_excess:+.1f} | {len(scores[best])} |")

    lines += ["", f"**{len(keepers)} of {len(bases)} identities have a usable French voice.**", "",
              "Use them straight from tts.py:", "", "```",
              './tts.py -t "Bonjour." --voice ' + (keepers[0] if keepers else "fr_female") + " -o out.wav",
              "```", "", "The full set:", "", "```", " ".join(keepers), "```"]

    destination = Path(out) if out else STAGE_DIR / "keepers.md"
    destination.write_text("\n".join(lines) + "\n")

    # The same set as a plain list, because every consumer downstream wants names one per
    # line rather than a markdown table: `sweep.py --only-file`, `driver.sh ONLY_FILE`,
    # and `fleurs.sh`, which benchmarks exactly these voices against FLEURS.
    keepers_path = STAGE_DIR / "keepers.txt"
    keepers_path.write_text("\n".join(keepers) + "\n")

    # Keepers plus every shipped voice, deduplicated because some keepers *are* shipped
    # voices. A benchmark wants this one: a keeper's number only means something next to
    # the number its own donor scores on the same sentence.
    both = list(dict.fromkeys(keepers + sorted(bases)))
    (STAGE_DIR / "keepers_and_bases.txt").write_text("\n".join(both) + "\n")
    logger.success(f"{len(keepers)} of {len(bases)} identities usable, wrote {destination}, "
                   f"{keepers_path} and keepers_and_bases.txt ({len(both)} voices)")

    if contenders:
        # Picking the best of ~19 variants on two sentences each is selection on noise:
        # a 0 pct score means two clean sentences, not a clean voice. This list is the
        # finalists per identity, to be re-swept deeper before anything is called final.
        shortlist = [name for base in sorted(groups)
                     for _, name in sorted(groups[base])[:contenders]]
        path = STAGE_DIR / "contenders.txt"
        path.write_text("\n".join(shortlist) + "\n")
        logger.info(f"{len(shortlist)} contenders written to {path}: re-sweep them with "
                    f"ONLY_FILE={path} DEPTH=12 sudo -E ./voxtral/cloning/driver.sh")


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


def seriate(names: list[str], similarity: np.ndarray) -> list[int]:
    """Order voices so that similar ones sit next to each other.

    A 39x39 table in alphabetical order hides its own structure: the blocks are there but
    scattered. This walks a greedy nearest-neighbour chain from the most peripheral voice
    (lowest mean similarity to everything else), which pulls families onto the diagonal
    and makes the matrix readable as a picture rather than as 1521 numbers.

    It is a presentation order and nothing more. No claim is made that the chain is an
    optimal ordering, and none is needed: every number in the table is the raw cosine.

    Parameters
    ----------
    names : list of str
        Voice names, indexing `similarity`.
    similarity : numpy.ndarray
        `[V, V]` cosine similarities.

    Returns
    -------
    list of int
        Indices into `names`, in display order.
    """
    remaining = set(range(len(names)))
    # Start peripheral rather than central, so the chain runs from the most unusual voice
    # inward instead of starting in the middle of the largest cluster and splitting it.
    means = (similarity.sum(axis=1) - np.diag(similarity)) / max(len(names) - 1, 1)
    order = [int(np.argmin(means))]
    remaining.discard(order[0])
    while remaining:
        last = order[-1]
        nxt = max(remaining, key=lambda index: similarity[last, index])
        order.append(nxt)
        remaining.discard(nxt)
    return order


@cli.command("matrix")
@click.option("--out", default=None, help="Where to write the markdown (default: <stage>/similarity.md).")
@click.option("--threshold", default=SAME_SPEAKER, show_default=True,
              help="Cosine above which two voices are treated as the same speaker.")
@click.option("--only-file", default=None, help="Restrict to the voices named in a file, one per line.")
def cmd_matrix(out: str | None, threshold: float, only_file: str | None) -> None:
    """Write the full N x N speaker-similarity matrix, plus what it implies.

    The matrix is the direct answer to "how many acoustically different voices do I
    actually have", which is the question a dataset-diversity exercise asks. WER says a
    voice is usable; this says whether it is a *different* voice from the one next to it,
    and the two have to be read together: a set of twenty voices that are all the same
    speaker diversifies nothing, and a set of twenty distinct speakers half of whom garble
    French is a dataset with garbled audio in it.
    """
    names, vectors = load_embeddings()
    if only_file:
        wanted = {line.strip() for line in Path(only_file).read_text().splitlines()
                  if line.strip() and not line.startswith("#")}
        keep = [index for index, name in enumerate(names) if name in wanted]
        missing = wanted - set(names)
        if missing:
            logger.warning(f"{len(missing)} name(s) in {only_file} have no embedding, "
                           f"e.g. {sorted(missing)[:3]}")
        names = [names[index] for index in keep]
        vectors = vectors[keep]
    if len(names) < 2:
        raise SystemExit("need at least two embedded voices for a matrix")

    pool_path = REPO_ROOT / "cache" / "cloning" / "A6" / "pool.json"
    pool = {entry["name"]: entry for entry in json.loads(pool_path.read_text())} \
        if pool_path.is_file() else {}

    # Rows are L2 normalised by `embed`, so the Gram matrix is exactly the cosine.
    similarity = vectors @ vectors.T
    order = seriate(names, similarity)
    offdiag = similarity[np.triu_indices(len(names), k=1)]

    # Numbered columns: 39 voice names across a header is unreadable in any renderer, and
    # the legend below the table costs one lookup while the matrix stays a picture.
    labels = {index: position for position, index in enumerate(order, start=1)}
    lines = [f"## Speaker similarity, {len(names)} x {len(names)}", "",
             "Cosine similarity between ECAPA-TDNN (VoxCeleb) speaker embeddings, each "
             "voice averaged over its own takes. Every voice read the same sentences with "
             "the same noise seed, so two takes differ only by the prefix that generated "
             "them: text and prosody are held still and the embedding sees speaker "
             "identity. 1.00 is the same speaker, and this family of systems sits near "
             f"0.25 at equal error rate, so anything above {threshold:.2f} is two names for "
             "one voice.", "",
             "Columns are numbered to keep the table readable; the legend follows. Rows "
             "are ordered by a greedy nearest-neighbour chain starting from the most "
             "peripheral voice, so families fall on the diagonal. Leading zeros are "
             "dropped and the diagonal is blank.", "",
             "| # | voice | " + " | ".join(str(position) for position in range(1, len(order) + 1)) + " |",
             "|---|---|" + "---|" * len(order)]
    for index in order:
        cells = []
        for other in order:
            if other == index:
                cells.append("")
                continue
            value = float(similarity[index, other])
            cells.append(f"{value:.2f}".lstrip("0") if value >= 0 else f"{value:.2f}")
        lines.append(f"| {labels[index]} | `{names[index]}` | " + " | ".join(cells) + " |")

    lines += ["", "### Legend and nearest neighbour", "",
              "`claimed` is the shipped speaker each voice is supposed to resemble, read "
              "off its recipe (a hybrid `h_<semantic>_<acoustic>` should sound like its "
              "acoustic donor, a sparse `x_` mix like its dominant component). `nearest` "
              "is what the embedding actually says. The two agreeing is the check that "
              "the metric and the recipe are both working.", "",
              "| # | voice | claimed | nearest | cosine | matches |",
              "|---|---|---|---|---|---|"]
    agree = 0
    for index in order:
        row = similarity[index].copy()
        row[index] = -2.0
        best = int(np.argmax(row))
        claimed = declared_identity(names[index], pool)
        # A base is its own claim, so "matches" is only meaningful for derived voices.
        derived = claimed != names[index]
        ok = derived and names[best] == claimed
        agree += ok
        lines.append(f"| {labels[index]} | `{names[index]}` | "
                     f"{'`' + claimed + '`' if derived else '(shipped)'} | `{names[best]}` | "
                     f"{row[best]:.2f} | {'yes' if ok else ('no' if derived else '-')} |")
    derived_count = sum(1 for name in names if declared_identity(name, pool) != name)

    pairs = [(float(similarity[a, b]), names[a], names[b])
             for a in range(len(names)) for b in range(a + 1, len(names))
             if similarity[a, b] >= threshold]
    pairs.sort(reverse=True)
    lines += ["", "### What the matrix says", "",
              f"- Off-diagonal similarity: median {np.median(offdiag):.2f}, "
              f"p25 {np.percentile(offdiag, 25):.2f}, p75 {np.percentile(offdiag, 75):.2f}, "
              f"max {offdiag.max():.2f}, min {offdiag.min():.2f}.",
              f"- {len(pairs)} of {len(offdiag)} pairs sit at or above {threshold:.2f}, "
              f"i.e. are one speaker under two names.",
              f"- {agree} of {derived_count} derived voices have their claimed donor as "
              f"their nearest neighbour."]
    if pairs:
        lines += ["", "Pairs at or above the threshold, closest first:", "",
                  "| cosine | a | b |", "|---|---|---|"]
        lines += [f"| {value:.2f} | `{a}` | `{b}` |" for value, a, b in pairs]

    destination = Path(out) if out else STAGE_DIR / "similarity.md"
    destination.write_text("\n".join(lines) + "\n")
    logger.success(f"{len(names)} voices, {len(pairs)} pair(s) above {threshold:.2f}, "
                   f"wrote {destination}")


if __name__ == "__main__":
    cli()
