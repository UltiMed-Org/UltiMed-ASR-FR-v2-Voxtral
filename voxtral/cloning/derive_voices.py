#!/usr/bin/env -S uv run --script
# /// script
# requires-python = ">=3.10"
# dependencies = ["click", "loguru", "torch"]
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

import shutil
import subprocess
import sys
from pathlib import Path

import click
import torch
from loguru import logger

REPO_ROOT = Path(__file__).resolve().parents[2]
VOICES_TOOL = Path(__file__).resolve().parent / "voices_tool.py"
OUT_DIR = REPO_ROOT / "cache" / "cloning" / "A2" / "pt"

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


@cli.command("clean")
@click.option("--yes", is_flag=True, help="Do not ask.")
def cmd_clean(yes: bool) -> None:
    """Unregister every d_* voice and delete the derived .pt files."""
    snapshot = snapshot_dir()
    derived = sorted(p.stem for p in (snapshot / "voice_embedding").glob("d_*.pt"))
    if not derived:
        logger.info("no d_* voices registered")
        return
    logger.info(f"about to unregister: {', '.join(derived)}")
    if not yes and not click.confirm("proceed?", default=False):
        sys.exit(1)
    for name in derived:
        subprocess.run([str(VOICES_TOOL), "remove", "--name", name], check=True)
    if OUT_DIR.is_dir():
        shutil.rmtree(OUT_DIR)
    logger.success(f"{len(derived)} derived voices removed")


if __name__ == "__main__":
    cli()
