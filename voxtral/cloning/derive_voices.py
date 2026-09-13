#!/usr/bin/env -S uv run --script
# /// script
# requires-python = ">=3.10"
# dependencies = ["click", "loguru", "torch", "numpy", "safetensors"]
#
# [tool.uv.sources]
# torch = { index = "pytorch-cpu" }
#
# [[tool.uv.index]]
# name = "pytorch-cpu"
# url = "https://download.pytorch.org/whl/cpu"
# explicit = true
# ///
"""Build derived Voxtral voices out of the 20 shipped ones, by tensor surgery.

Stage A2 of PLAN_VOXTRALTTS_CLONING_TEST.md. A voice is a `[N, 3072]` bfloat16
prefix in the LLM's embedding space (see voices_tool.py's docstring), so new
voices can be made without any encoder at all, just by cutting, blending and
splicing existing prefixes. That is the whole point of Track A: acoustic
diversity for free, no training, no missing encoder weights.

Nothing here is guaranteed to sound good. An interpolation lands off the
manifold the real prefixes live on (they are sums of codebook rows, this is not),
so mush is a plausible outcome and A2 exists to find out. Truncation is the safe
bet, cross-lingual costs nothing at all (it needs no new voice: just call an
existing one with French text), and interpolation is the speculative one.

Why this is a separate script from voices_tool.py: torch. Registering a voice is
a file copy plus two JSON edits and must stay dependency-free, whereas this one
pays for a torch wheel. It deliberately does NOT patch params.json / tekken.json
itself: it writes .pt files and then shells out to `voices_tool.py add`, so the
three-files-must-agree logic lives in exactly one place.

The torch here is pinned to the CPU wheel on purpose. It is ~200 MB instead of
~2.5 GB of CUDA, the work is slicing and averaging a few hundred KB of tensors,
and the GPU is usually busy serving voxtral-tts while this runs.

Derived voices are all named `d_*`, which makes them trivially separable from
the 20 shipped ones and lets `clean` remove exactly what this script created.

Usage
-----
    ./derive_voices.py build              # write .pt files and register them
    ./derive_voices.py build --dry-run    # print the recipe, touch nothing
    ./derive_voices.py clean              # unregister every d_* voice
    # then: sudo docker compose up -d voxtral-tts   (model init reads them once)
"""

from __future__ import annotations

import json
import random
import shutil
import subprocess
import sys
from pathlib import Path

import click
import numpy as np
import torch
from loguru import logger

sys.path.insert(0, str(Path(__file__).resolve().parent))
from recover_codes import codebook_layout, load_embedding_table  # noqa: E402

REPO_ROOT = Path(__file__).resolve().parents[2]
VOICES_TOOL = Path(__file__).resolve().parent / "voices_tool.py"
OUT_DIR = REPO_ROOT / "cache" / "cloning" / "A2" / "pt"
# Stage A6's pool lives apart from A2's eight hand-picked variants so that
# `clean` and the sweep can address one without touching the other.
POOL_DIR = REPO_ROOT / "cache" / "cloning" / "A6" / "pt"
# B1's output: the exact discrete codes behind each shipped prefix, [N, 37] int16.
CODES_DIR = REPO_ROOT / "cache" / "cloning" / "codes"

# The recipe. One entry per derived voice, edit freely between sessions.
#
# ("name", "op", (args...)) where op is:
#   head  (src, k)          first k frames
#   tail  (src, k)          last k frames
#   mix   (a, b, alpha)     alpha * a + (1 - alpha) * b, both cut to the shorter
#   cat   (a, b, split)     first `split` frames of a, then b from `split` on
#
# Chosen around fr_female (97 frames), the voice the dataset work already uses
# and likes, so every variant is a controlled deviation from a known-good anchor
# rather than a random pair. Partners are picked for contrast: it_female and
# es_female are other female voices in nearby languages, fr_male is the same
# language and recording style but the opposite gender, which is the blend most
# likely to expose whether averaging produces a voice or a smear.
RECIPE: list[tuple[str, str, tuple]] = [
    ("d_frf_head40", "head", ("fr_female", 40)),
    ("d_frf_head60", "head", ("fr_female", 60)),
    ("d_frf_tail60", "tail", ("fr_female", 60)),
    ("d_frf_itf_50", "mix", ("fr_female", "it_female", 0.5)),
    ("d_frf_esf_50", "mix", ("fr_female", "es_female", 0.5)),
    ("d_frf_frm_50", "mix", ("fr_female", "fr_male", 0.5)),
    ("d_frf_frm_25", "mix", ("fr_female", "fr_male", 0.75)),
    ("d_frf_itf_cat", "cat", ("fr_female", "it_female", 48)),
]


def snapshot_dir() -> Path:
    """Ask voices_tool where the checkpoint snapshot is, rather than re-deriving it.

    Returns
    -------
    Path
        The snapshot directory holding params.json, tekken.json, voice_embedding/.
    """
    out = subprocess.run([str(VOICES_TOOL), "paths"], capture_output=True, text=True,
                         check=True).stdout
    # `paths` prints "<key><whitespace><path>". Matching on the KEY and not on
    # the line is deliberate: every path contains "snapshots/", so a substring
    # test would happily return voice_embedding/ or params.json.
    for line in out.splitlines():
        parts = line.split()
        if len(parts) == 2 and parts[0] == "snapshot":
            candidate = Path(parts[1])
            if candidate.is_dir():
                return candidate
    raise SystemExit(f"could not parse a snapshot path out of `voices_tool.py paths`:\n{out}")


def load_voice(snapshot: Path, name: str) -> torch.Tensor:
    """Load a voice prefix as a float32 `[N, 3072]` tensor.

    float32 and not the stored bfloat16 because averaging in bfloat16 (8 bits of
    mantissa) would quantize every intermediate step. The result is cast back
    before saving, so the file format is unchanged.

    Parameters
    ----------
    snapshot : Path
        Checkpoint snapshot directory.
    name : str
        Registered voice name.

    Returns
    -------
    torch.Tensor
        `[N, hidden]` float32.
    """
    path = snapshot / "voice_embedding" / f"{name}.pt"
    if not path.is_file():
        raise SystemExit(f"no such voice: {name} ({path})")
    obj = torch.load(path, map_location="cpu", weights_only=True)
    if isinstance(obj, dict):
        # Defensive: the shipped files are bare tensors, but a dict with one
        # tensor value would be the obvious alternative packaging.
        tensors = [v for v in obj.values() if isinstance(v, torch.Tensor)]
        if len(tensors) != 1:
            raise SystemExit(f"{path} holds a dict with {len(tensors)} tensors, cannot pick one")
        obj = tensors[0]
    if obj.ndim != 2:
        raise SystemExit(f"{path} is {tuple(obj.shape)}, expected 2 dims [N, hidden]")
    return obj.float()


def short_name(voice: str) -> str:
    """Abbreviate a shipped voice name to three characters.

    Derived names have to carry both parents, and `h_cheerful_female__neutral_male`
    is unreadable in a table of hundreds of rows. Two letters of the first part plus
    the gender letter is collision-free over the 20 shipped voices (frf, frm, nef,
    nem, caf, cam, chf, ...), and deterministic, so a name can be read back to its
    parents by eye.

    Parameters
    ----------
    voice : str
        A registered voice name, `<language-or-style>_<gender>`.

    Returns
    -------
    str
        Three-character abbreviation.
    """
    parts = voice.split("_")
    return (parts[0][:2] + parts[-1][:1]).lower()


def load_codes(name: str) -> np.ndarray:
    """Load B1's recovered codes for one voice.

    Parameters
    ----------
    name : str
        Voice name.

    Returns
    -------
    numpy.ndarray
        `[N, 37]` int64: one semantic code plus 36 acoustic levels per frame.
    """
    path = CODES_DIR / f"{name}.npy"
    if not path.is_file():
        raise SystemExit(f"no recovered codes for {name} ({path}); run recover_codes.py run first")
    return np.load(path).astype(np.int64)


def prefix_from_codes(codes: np.ndarray, table: torch.Tensor, offsets: list[int]) -> torch.Tensor:
    """Rebuild a prefix from a code tuple, the way the model's `encode_tokens` does.

    This is the whole reason code-level recombination is different in kind from
    interpolating prefixes: a prefix is the SUM of 37 codebook rows, so any tuple of
    valid code indices lands exactly on the manifold real voices live on, while the
    average of two prefixes lands between codebook rows and can come out as mush.

    Parameters
    ----------
    codes : numpy.ndarray
        `[N, 37]` integer codes.
    table : torch.Tensor
        `[9088, 3072]` shared codebook embedding table.
    offsets : list of int
        Where each codebook starts inside that table.

    Returns
    -------
    torch.Tensor
        `[N, 3072]` float32 prefix.
    """
    rows = torch.as_tensor(codes, dtype=torch.long)
    return sum(table[offsets[g] + rows[:, g]] for g in range(len(offsets)))


def hybrid_codes(semantic: str, acoustic: str) -> np.ndarray:
    """Take the semantic code column from one voice and the 36 acoustic ones from another.

    Both are truncated to the shorter voice, the same alignment compromise `mix` makes,
    and for the same reason: frames are positions in someone's recording, not aligned
    units. What differs is that the result is still a valid code tuple.

    Recovery quality is deliberately ignored here. B1 recovered 18 of 20 voices bit
    exactly and left a small residual on the other two, but a residual only means the
    tuple is an imperfect reconstruction of the ORIGINAL prefix, which does not matter
    when the goal is a new voice rather than a faithful copy.

    Parameters
    ----------
    semantic : str
        Voice donating the semantic code (column 0).
    acoustic : str
        Voice donating the 36 acoustic levels (columns 1 to 36).

    Returns
    -------
    numpy.ndarray
        `[min(N_a, N_b), 37]` integer codes.
    """
    a, b = load_codes(semantic), load_codes(acoustic)
    n = min(len(a), len(b))
    out = b[:n].copy()
    out[:, 0] = a[:n, 0]
    return out


def apply_op(op: str, args: tuple, voices: dict[str, torch.Tensor]) -> torch.Tensor:
    """Produce one derived prefix.

    Parameters
    ----------
    op : str
        One of "head", "tail", "mix", "cat".
    args : tuple
        Operands, see RECIPE.
    voices : dict of str to torch.Tensor
        Already-loaded source prefixes, keyed by voice name.

    Returns
    -------
    torch.Tensor
        `[M, hidden]` float32.
    """
    if op == "head":
        src, k = args
        return voices[src][:k].clone()
    if op == "tail":
        src, k = args
        return voices[src][-k:].clone()
    if op == "mix":
        a, b, alpha = args
        # Truncating both to the shorter length is the only alignment available:
        # frames are positions in someone's recording, not aligned units, so
        # frame i of A and frame i of B are unrelated sounds. That is precisely
        # why this variant may fail, and why it is worth one experiment.
        n = min(len(voices[a]), len(voices[b]))
        return alpha * voices[a][:n] + (1.0 - alpha) * voices[b][:n]
    if op == "cat":
        a, b, split = args
        return torch.cat([voices[a][:split], voices[b][split:]], dim=0)
    if op == "mixn":
        # Sparse N-way blend: args is ((name, weight), ...). Weights are NOT normalised
        # to sum to 1 here, the recipe is expected to do that, because the interesting
        # setting is one dominant voice plus traces (0.6 / 0.3 / 0.1) rather than a flat
        # average. A flat average of N voices shrinks the distance from the global
        # centroid by roughly sqrt(N), which is the regression-to-the-mean that made B6's
        # regression head useless, so it produces a blander voice, not a new one.
        n = min(len(voices[name]) for name, _ in args)
        out = torch.zeros_like(voices[args[0][0]][:n])
        for name, weight in args:
            out = out + weight * voices[name][:n]
        return out
    raise SystemExit(f"unknown op '{op}'")


@click.group()
def cli() -> None:
    """Derive new Voxtral voices from the shipped ones (stage A2)."""


@cli.command("build")
@click.option("--dry-run", is_flag=True, help="Print what would be built, write and register nothing.")
@click.option("--force", is_flag=True, default=True, help="Overwrite an existing registration (on by default: the recipe is meant to be re-run).")
def cmd_build(dry_run: bool, force: bool) -> None:
    """Build every RECIPE entry and register it as a voice."""
    snapshot = snapshot_dir()
    logger.info(f"snapshot: {snapshot}")

    needed = {n for _, _, a in RECIPE for n in a if isinstance(n, str)}
    voices = {} if dry_run else {n: load_voice(snapshot, n) for n in sorted(needed)}
    if not dry_run:
        for name, tensor in voices.items():
            logger.debug(f"source {name}: {tuple(tensor.shape)}")

    OUT_DIR.mkdir(parents=True, exist_ok=True)
    for name, op, args in RECIPE:
        if dry_run:
            logger.info(f"would build {name} = {op}{args}")
            continue
        derived = apply_op(op, args, voices)
        # Back to the stored dtype so the file is indistinguishable in format
        # from a shipped one, which is what read_embedding_frames() assumes when
        # it infers N from the raw byte size.
        derived = derived.to(torch.bfloat16).contiguous()
        pt_path = OUT_DIR / f"{name}.pt"
        torch.save(derived, pt_path)
        logger.info(f"{name}: {op}{args} -> {tuple(derived.shape)} ({len(derived) / 12.5:.2f} s)")

        cmd = [str(VOICES_TOOL), "add", "--name", name, "--from-file", str(pt_path)]
        if force:
            cmd.append("--force")
        subprocess.run(cmd, check=True)

    if dry_run:
        logger.info(f"{len(RECIPE)} variants in the recipe, nothing written")
        return
    logger.success(f"{len(RECIPE)} derived voices registered")
    logger.info("apply with: sudo docker compose up -d voxtral-tts  (model init reads them once)")


@cli.command("pool")
@click.option("--grid/--no-grid", default=True, show_default=True,
              help="Every ordered pair of bases as a semantic/acoustic hybrid (380 voices).")
@click.option("--mixes", default=40, show_default=True,
              help="Sparse three-way embedding mixes to add (0 disables).")
@click.option("--weights", default="0.6,0.3,0.1", show_default=True,
              help="Comma-separated weights for a sparse mix, dominant first.")
@click.option("--seed", default=0, show_default=True, help="Seed for the mix triples.")
@click.option("--limit", default=0, show_default=True,
              help="Keep only the first N candidates (0 = all). For a quick smoke run.")
@click.option("--dry-run", is_flag=True, help="Print the pool, write and register nothing.")
@click.option("--register/--no-register", default=True, show_default=True,
              help="Register the pool with voices_tool after writing it.")
def cmd_pool(grid: bool, mixes: int, weights: str, seed: int, limit: int,
             dry_run: bool, register: bool) -> None:
    """Build stage A6's candidate pool: code-level hybrids plus sparse mixes.

    Two generators, deliberately different in kind so the sweep can tell them apart:

    - **Hybrids** (`h_<semantic>_<acoustic>`) take the semantic code from one voice and
      the 36 acoustic levels from another. Valid code tuples by construction, so they
      cannot produce the between-codebook-rows mush A2 heard. The full ordered grid also
      answers a question nobody has answered: which half of a frame carries speaker
      identity. If `h_frf_*` all sound like fr_female, it is the semantic code.
    - **Sparse mixes** (`x_<a>_<b>_<c>`) are embedding-space weighted averages with one
      dominant voice. These are the ones I expect to disappoint, for the reason in
      `apply_op`, and they are cheap enough to be worth disproving.
    """
    snapshot = snapshot_dir()
    logger.info(f"snapshot: {snapshot}")

    # A base is a registered voice that B1 recovered codes for and that this script did
    # not itself create. Reading it off disk rather than hardcoding 20 names means a
    # checkpoint that ships more voices is picked up without an edit.
    registered = {line.split()[0] for line in subprocess.run(
        [str(VOICES_TOOL), "list"], capture_output=True, text=True, check=True).stdout.splitlines()
        if line and not line.startswith(("voice", "\n")) and len(line.split()) > 2}
    bases = sorted(n for n in registered
                   if (CODES_DIR / f"{n}.npy").is_file() and not n.startswith(("d_", "h_", "x_", "m_")))
    if len(bases) < 2:
        raise SystemExit(f"need at least 2 base voices with recovered codes, found {bases}")
    logger.info(f"{len(bases)} base voices: {', '.join(bases)}")

    shorts = {n: short_name(n) for n in bases}
    if len(set(shorts.values())) != len(shorts):
        raise SystemExit(f"abbreviations collide: {sorted(shorts.values())}")

    # (name, kind, parents) triples. Built before any tensor is touched so --dry-run
    # costs nothing and the count is known up front.
    plan: list[tuple[str, str, list[str]]] = []
    if grid:
        for sem in bases:
            for ac in bases:
                if sem != ac:
                    plan.append((f"h_{shorts[sem]}_{shorts[ac]}", "hybrid", [sem, ac]))
    if mixes:
        weight_list = [float(w) for w in weights.split(",")]
        rng = random.Random(seed)
        seen = set()
        attempts = 0
        while len(seen) < mixes and attempts < mixes * 50:
            attempts += 1
            picked = tuple(rng.sample(bases, len(weight_list)))
            if picked in seen:
                continue
            seen.add(picked)
            plan.append(("x_" + "_".join(shorts[n] for n in picked), "mixn", list(picked)))
    if limit:
        plan = plan[:limit]
    logger.info(f"pool: {sum(1 for _, k, _ in plan if k == 'hybrid')} hybrids, "
                f"{sum(1 for _, k, _ in plan if k == 'mixn')} mixes, {len(plan)} total")

    if dry_run:
        for name, kind, parents in plan[:10]:
            logger.info(f"would build {name} = {kind}{tuple(parents)}")
        logger.info(f"... {len(plan)} candidates, nothing written")
        return

    POOL_DIR.mkdir(parents=True, exist_ok=True)
    # One read of the 3 GB safetensors for the whole pool, and one load per base voice.
    sizes, offsets = codebook_layout(snapshot)
    table = load_embedding_table(snapshot) if any(k == "hybrid" for _, k, _ in plan) else None
    voices = {n: load_voice(snapshot, n) for n in bases} if any(k == "mixn" for _, k, _ in plan) else {}
    weight_list = [float(w) for w in weights.split(",")]

    manifest = []
    for name, kind, parents in plan:
        if kind == "hybrid":
            codes = hybrid_codes(parents[0], parents[1])
            derived = prefix_from_codes(codes, table, offsets)
        else:
            derived = apply_op("mixn", tuple(zip(parents, weight_list)), voices)
        derived = derived.to(torch.bfloat16).contiguous()
        torch.save(derived, POOL_DIR / f"{name}.pt")
        manifest.append({"name": name, "kind": kind, "parents": parents,
                         "frames": len(derived), "seconds": len(derived) / 12.5})

    manifest_path = POOL_DIR.parent / "pool.json"
    manifest_path.write_text(json.dumps(manifest, indent=2))
    logger.success(f"{len(manifest)} candidates written to {POOL_DIR}, manifest in {manifest_path}")

    if register:
        subprocess.run([str(VOICES_TOOL), "add", "--from-dir", str(POOL_DIR), "--force"], check=True)
        logger.info("apply with: sudo docker compose up -d voxtral-tts  (model init reads them once)")


@cli.command("clean")
@click.option("--yes", is_flag=True, help="Do not ask.")
@click.option("--prefix", "prefixes", multiple=True, default=("d_", "h_", "x_"), show_default=True,
              help="Name prefixes to remove (d_ = A2 variants, h_ = A6 hybrids, x_ = A6 mixes).")
def cmd_clean(yes: bool, prefixes: tuple[str, ...]) -> None:
    """Unregister every derived voice and delete the .pt files behind it."""
    snapshot = snapshot_dir()
    derived = sorted(p.stem for pre in prefixes
                     for p in (snapshot / "voice_embedding").glob(f"{pre}*.pt"))
    if not derived:
        logger.info("no d_* voices registered")
        return
    logger.info(f"about to unregister: {', '.join(derived)}")
    if not yes and not click.confirm("proceed?", default=False):
        sys.exit(1)
    for name in derived:
        subprocess.run([str(VOICES_TOOL), "remove", "--name", name], check=True)
    for directory in (OUT_DIR, POOL_DIR):
        if directory.is_dir():
            shutil.rmtree(directory)
    logger.success(f"{len(derived)} derived voices removed")


if __name__ == "__main__":
    cli()
