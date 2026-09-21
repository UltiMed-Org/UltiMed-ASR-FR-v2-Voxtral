#!/usr/bin/env -S uv run --script
# /// script
# requires-python = ">=3.10"
# dependencies = ["click", "loguru", "numpy", "tqdm"]
# ///
"""Compare every voice in the hybrid grid ACROSS languages (stage A8).

What this answers that a per-language report cannot
---------------------------------------------------
`fleurs_report.sh` produces one report per language: which voices read Italian well,
how they cluster, what broke. Eight of those do not answer the question the grid was
built for, which is what a user of this setup actually has to decide:

    "I want voice X. I am generating Portuguese. What name do I call?"

A7 established that speaker identity lives entirely in the 36 acoustic codes, so a
hybrid `h_<semantic>_<acoustic>` sounds like its acoustic donor and reads like its
semantic one. That splits the decision in two: the acoustic donor picks WHO, and the
semantic donor picks HOW WELL THEY READ THIS LANGUAGE. The grid exists so the second
half can be looked up per language instead of guessed, and this script is the lookup.

So the central output is one row per shipped speaker, one column per language, and in
each cell the name that reads that language best while still sounding like that speaker.

What it deliberately does not claim
-----------------------------------
More names is not more voices. The grid is 380 hybrids over 20 acoustic donors, so
every per-language similarity matrix will show the same ~20 clusters, and the cross
tables here rank delivery rather than identity. The only construction that can move off
those 20 is the sparse mix, which is why the mix tiers get their own section and are
measured by distance from the nearest shipped voice rather than by error rate.

Inputs
------
`cache/cloning/GRID_<lang>/summary.json`, written by `sweep.py stats`, one per language.
Nothing is recomputed from the ledgers: the common-sentence set, the CER back-fill and
the yield buckets are all decided there, and a second implementation of them here would
eventually disagree with the per-language reports this sits on top of.

    ./grid_report.py cross                      # the tables, into cache/cloning/GRID/
    ./grid_report.py cross --gate-junk 0.05     # stricter about degeneration
"""

from __future__ import annotations

import csv
import json
import sys
from pathlib import Path

import click
import numpy as np
from loguru import logger

sys.path.insert(0, str(Path(__file__).resolve().parent))

from sweep import (REPO_ROOT, declared_identity, load_pool,  # noqa: E402
                   speaker_components)

CLONING_DIR = REPO_ROOT / "cache" / "cloning"

# A voice is reported as usable in a language when it degenerates rarely enough that a
# generator does not spend more time regenerating than generating. Both are first-attempt
# rates, so they describe the voice rather than a retry loop carrying it.
GATE_RUNAWAY = 0.02
GATE_JUNK = 0.10


def language_runs(*, prefix: str = "GRID_") -> list[str]:
    """Languages that have a summary to read, in directory order.

    Returns
    -------
    list of str
        FLEURS language codes, e.g. `["ar_eg", "de_de", ...]`. A language whose sweep has
        not reached `sweep.py stats` yet is simply absent, which is what makes this
        runnable against a grid that is still going.
    """
    found = []
    for path in sorted(CLONING_DIR.glob(f"{prefix}*")):
        if (path / "summary.json").is_file():
            found.append(path.name[len(prefix):])
    return found


def read_summaries(languages: list[str], *, prefix: str = "GRID_") -> dict[str, dict]:
    """Load each language's `summary.json`, keyed by language then by voice."""
    summaries = {}
    for language in languages:
        data = json.loads((CLONING_DIR / f"{prefix}{language}" / "summary.json").read_text())
        data["by_voice"] = {row["voice"]: row for row in data["voices"]}
        summaries[language] = data
    return summaries


def passes(row: dict, *, gate_runaway: float, gate_junk: float) -> bool:
    """Whether one voice is usable in one language.

    A missing runaway figure counts as failing rather than passing: it means the voice
    was never profiled, and a gate that defaults to "fine" is not a gate.
    """
    runaway = row.get("runaway")
    return runaway is not None and runaway <= gate_runaway and row["junk"] <= gate_junk


def best_name(*, candidates: list[str], summary: dict,
              gate_runaway: float, gate_junk: float) -> tuple[str, dict] | None:
    """Pick the name to call for one speaker in one language.

    Ranked by median WER among the candidates that clear the degeneration gate, because
    the median is what a typical sentence costs and the gate is what keeps a voice with a
    beautiful median and a 15 pct disaster rate from winning. When nothing clears the
    gate the best of the failures is returned anyway, since reporting "no name" for a
    speaker a user asked for is less useful than reporting the least bad one and saying
    it failed.

    Returns
    -------
    tuple of (str, dict) or None
        Name and its summary row, or None when none of the candidates was measured.
    """
    rows = [(name, summary["by_voice"][name]) for name in candidates
            if name in summary["by_voice"]]
    if not rows:
        return None
    clean = [pair for pair in rows
             if passes(pair[1], gate_runaway=gate_runaway, gate_junk=gate_junk)]
    return min(clean or rows, key=lambda pair: pair[1]["wer"]["median"])


def cluster_counts(languages: list[str], *, threshold: float,
                   prefix: str = "GRID_") -> dict[str, int]:
    """How many distinct speakers each language's takes actually contain.

    Reads each run's cached ECAPA embeddings and clusters them with the same connected
    components `voice_distance.py select` uses. These numbers are expected to agree
    across languages to within a voice or two, because identity is a property of the
    prefix and not of the text being read; a language that disagrees badly is a signal
    that its takes are degenerate rather than that its voices are different.
    """
    counts = {}
    for language in languages:
        path = CLONING_DIR / f"{prefix}{language}" / "embeddings.npz"
        if not path.is_file():
            continue
        stored = np.load(path, allow_pickle=False)
        names = [str(name) for name in stored["voices"]]
        vectors = stored["vectors"]
        similarity = vectors @ vectors.T
        index_of = {name: position for position, name in enumerate(names)}
        counts[language] = len(speaker_components(names=names, similarity=similarity,
                                                  index_of=index_of, threshold=threshold))
    return counts


def percent(value: float | None) -> str:
    """Format a rate as a percentage, or `-` when it was never measured."""
    return "-" if value is None else f"{100 * value:.1f}%"


@click.group(context_settings={"help_option_names": ["-h", "--help"]})
def cli() -> None:
    """Cross-language reports over the hybrid grid."""


@cli.command("cross")
@click.option("--out-dir", default=None,
              help="Where to write cross.md and voices.csv (default cache/cloning/GRID).")
@click.option("--prefix", default="GRID_", show_default=True,
              help="Run-directory prefix the per-language sweeps used.")
@click.option("--gate-runaway", default=GATE_RUNAWAY, show_default=True,
              help="Maximum first-attempt runaway rate for a voice to count as usable.")
@click.option("--gate-junk", default=GATE_JUNK, show_default=True,
              help="Maximum share of takes past 50 pct WER for a voice to count as usable.")
@click.option("--threshold", default=0.70, show_default=True,
              help="Cosine at which two voices count as one speaker.")
def cmd_cross(out_dir: str | None, prefix: str, gate_runaway: float,
              gate_junk: float, threshold: float) -> None:
    """Build the cross-language tables from every per-language summary that exists."""
    languages = language_runs(prefix=prefix)
    if not languages:
        raise SystemExit(f"no {prefix}*/summary.json under {CLONING_DIR}; "
                         "run grid.sh and then grid_report.sh")
    summaries = read_summaries(languages, prefix=prefix)
    logger.info(f"{len(languages)} languages: {', '.join(languages)}")

    pool = load_pool()
    measured = sorted({voice for summary in summaries.values() for voice in summary["by_voice"]})
    identity = {voice: declared_identity(voice, pool) for voice in measured}
    speakers = sorted(set(identity.values()))
    logger.info(f"{len(measured)} voices over {len(speakers)} acoustic identities")

    destination = Path(out_dir) if out_dir else CLONING_DIR / "GRID"
    destination.mkdir(parents=True, exist_ok=True)

    # The flat table first, because it is the artifact anything downstream should read.
    # The markdown below is a view of it, not a separate computation.
    csv_path = destination / "voices.csv"
    with csv_path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.writer(handle)
        writer.writerow(["voice", "language", "kind", "acoustic_identity", "semantic_donor",
                         "takes", "median_wer", "mean_wer", "median_cer", "perfect",
                         "usable", "junk", "runaway", "words_per_second", "passes_gate"])
        for voice in measured:
            entry = pool.get(voice, {})
            semantic = entry["parents"][0] if entry.get("kind") == "hybrid" else ""
            for language in languages:
                row = summaries[language]["by_voice"].get(voice)
                if row is None:
                    continue
                writer.writerow([
                    voice, language, entry.get("kind", "base"), identity[voice], semantic,
                    row["wer"]["n"], round(row["wer"]["median"], 4),
                    round(row["wer"]["mean"], 4),
                    round(row["cer"]["median"], 4) if row.get("cer") else "",
                    round(row["perfect"], 4), round(row["usable"], 4), round(row["junk"], 4),
                    "" if row.get("runaway") is None else round(row["runaway"], 4),
                    "" if row.get("words_per_second") is None else round(row["words_per_second"], 3),
                    int(passes(row, gate_runaway=gate_runaway, gate_junk=gate_junk)),
                ])

    clusters = cluster_counts(languages, threshold=threshold, prefix=prefix)

    lines: list[str] = []
    lines += [f"# The hybrid grid across {len(languages)} languages", "",
              "Generated by `voxtral/cloning/grid_report.py` (Claude Code). Every number "
              "comes from the per-language `summary.json` that `sweep.py stats` wrote, so "
              "a cell here and the same cell in that language's `stats.md` cannot "
              "disagree. Retried takes are reported at their **first** attempt: that is "
              "the unaided voice, which is the only honest way to compare voices against "
              "each other.", "",
              f"A voice counts as **usable** in a language when it degenerates on at most "
              f"{100 * gate_runaway:.0f} pct of takes and lands at most "
              f"{100 * gate_junk:.0f} pct of them past 50 pct WER. Both are first-attempt "
              "rates, so they describe the voice and not a retry loop propping it up.", ""]

    lines += ["## What was measured", "",
              "| language | voices | sentences | takes | audio | usable voices | speakers |",
              "|---|---|---|---|---|---|---|"]
    for language in languages:
        summary = summaries[language]
        rows = summary["voices"]
        ok = sum(1 for row in rows if passes(row, gate_runaway=gate_runaway, gate_junk=gate_junk))
        takes = sum(row["wer"]["n"] for row in rows)
        lines.append(f"| `{language}` | {len(rows)} | {summary['shared_sentences']} | "
                     f"{takes} | {summary['audio_hours']:.1f} h | {ok} | "
                     f"{clusters.get(language, '-')} |")
    lines += ["",
              "`speakers` is connected components of the ECAPA similarity graph at cosine "
              f">= {threshold}. These should agree across languages to within a voice or "
              "two: identity is a property of the voice prefix, not of the text being "
              "read. A language that disagrees badly is telling you its takes degenerated, "
              "not that its voices are different people.", ""]

    # The central table. One row per speaker, one column per language, and in the cell the
    # name to call plus what it costs. This is the whole point of building the grid.
    lines += ["## Which name to call, per speaker and language", "",
              "Each row is one of the shipped acoustic identities: every name in that row "
              "sounds like that speaker, because identity travels in the 36 acoustic codes "
              "and a hybrid inherits them whole. The cell gives the name that reads that "
              "language best among them, with its median WER. A cell marked `!` did not "
              "clear the degeneration gate, so it is the least bad option rather than a "
              "recommendation.", "",
              "| speaker | " + " | ".join(f"`{language}`" for language in languages) + " |",
              "|---" * (1 + len(languages)) + "|"]
    for speaker in speakers:
        candidates = [voice for voice in measured if identity[voice] == speaker]
        cells = []
        for language in languages:
            pick = best_name(candidates=candidates, summary=summaries[language],
                             gate_runaway=gate_runaway, gate_junk=gate_junk)
            if pick is None:
                cells.append("-")
                continue
            name, row = pick
            flag = "" if passes(row, gate_runaway=gate_runaway, gate_junk=gate_junk) else " !"
            cells.append(f"`{name}` {100 * row['wer']['median']:.1f}%{flag}")
        lines.append(f"| `{speaker}` | " + " | ".join(cells) + " |")
    lines.append("")

    # The recipe behind that table: if the semantic donor is what decides how well a
    # language reads, it should be visible as whole rows being good rather than scattered
    # cells, and the winning donor per language should be a native reader of it.
    lines += ["## Which semantic donor reads which language", "",
              "Median over every hybrid built on that semantic donor, whatever its "
              "acoustic donor. A donor that reads a language cleanly lends its "
              "articulation to all 20 identities at once, which is the recipe A6 found "
              "and this table is the test of it at scale. Lower is better; the best donor "
              "in each column is the one to build new voices from.", "",
              "| semantic donor | " + " | ".join(f"`{language}`" for language in languages) + " |",
              "|---" * (1 + len(languages)) + "|"]
    donors = sorted({pool[voice]["parents"][0] for voice in measured
                     if pool.get(voice, {}).get("kind") == "hybrid"})
    for donor in donors:
        family = [voice for voice in measured
                  if pool.get(voice, {}).get("kind") == "hybrid"
                  and pool[voice]["parents"][0] == donor]
        cells = []
        for language in languages:
            medians = [summaries[language]["by_voice"][name]["wer"]["median"]
                       for name in family if name in summaries[language]["by_voice"]]
            cells.append(f"{100 * float(np.median(medians)):.1f}%" if medians else "-")
        lines.append(f"| `{donor}` | " + " | ".join(cells) + " |")
    lines.append("")

    # Sparse mixes are the only construction that can produce a speaker the checkpoint
    # does not ship, so they are judged on whether they moved, not on how well they read.
    by_arity: dict[int, list[str]] = {}
    for voice in measured:
        entry = pool.get(voice, {})
        if entry.get("kind") == "mixn":
            by_arity.setdefault(len(entry["parents"]), []).append(voice)
    if by_arity:
        lines += ["## Sparse mixes: does a longer tail of parents buy anything", "",
                  "Hybrids cannot invent a speaker, they re-read an existing one. Mixes "
                  "can, by landing between the shipped prefixes, and the open question "
                  "from A6 is whether more parents means more distance or only more "
                  "blandness. Judged on degeneration and error rate here; whether they "
                  "actually moved is the similarity matrix's answer, per language.", "",
                  "| parents | mixes | median WER | median junk | median runaway | usable |",
                  "|---|---|---|---|---|---|"]
        for arity in sorted(by_arity):
            family = by_arity[arity]
            wers, junks, runaways, ok, total = [], [], [], 0, 0
            for language in languages:
                for name in family:
                    row = summaries[language]["by_voice"].get(name)
                    if row is None:
                        continue
                    total += 1
                    wers.append(row["wer"]["median"])
                    junks.append(row["junk"])
                    if row.get("runaway") is not None:
                        runaways.append(row["runaway"])
                    ok += passes(row, gate_runaway=gate_runaway, gate_junk=gate_junk)
            if not wers:
                continue
            lines.append(f"| {arity}-way | {len(family)} | "
                         f"{percent(float(np.median(wers)))} | "
                         f"{percent(float(np.median(junks)))} | "
                         f"{percent(float(np.median(runaways)) if runaways else None)} | "
                         f"{100 * ok / total:.0f}% of {total} |")
        lines += ["", "`usable` is over (mix, language) pairs, not over mixes, so a tier "
                  "that works in French and detonates in Hindi reads as half usable "
                  "rather than as fine.", ""]

    # A voice that works everywhere is worth more than a voice that wins one column, and
    # nothing above would surface it: every table so far is read one language at a time.
    lines += ["## Voices that hold up in every language", "",
              "Ranked by how many languages they clear the gate in, then by their worst "
              "median across those languages. A voice near the top is one a multilingual "
              "dataset can standardise on instead of switching names per language.", "",
              "| voice | identity | languages passed | worst median WER | mean junk |",
              "|---|---|---|---|---|"]
    ranked = []
    for voice in measured:
        rows = [summaries[language]["by_voice"][voice] for language in languages
                if voice in summaries[language]["by_voice"]]
        if not rows:
            continue
        passed = [row for row in rows
                  if passes(row, gate_runaway=gate_runaway, gate_junk=gate_junk)]
        ranked.append((len(passed), -max(row["wer"]["median"] for row in rows), voice, rows))
    ranked.sort(reverse=True)
    for passed, negative_worst, voice, rows in ranked[:30]:
        lines.append(f"| `{voice}` | `{identity[voice]}` | {passed}/{len(languages)} | "
                     f"{percent(-negative_worst)} | "
                     f"{percent(float(np.mean([row['junk'] for row in rows])))} |")
    lines += ["", f"Full table, one row per (voice, language): "
              f"`{csv_path.relative_to(REPO_ROOT)}`.", ""]

    report = destination / "cross.md"
    report.write_text("\n".join(lines) + "\n", encoding="utf-8")
    logger.success(f"wrote {report} and {csv_path}")


if __name__ == "__main__":
    cli()
