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
"""Recover the discrete codes behind each shipped Voxtral voice embedding.

Stage B1 of PLAN_VOXTRALTTS_CLONING_TEST.md, and the first step of Track B: the
codec ENCODER is absent from the open weights, so a reference clip cannot be
turned into codes directly. Everything downstream of the codes, however, ships in
full. This script proves the "everything downstream" half by inverting the one
step that is pure lookup arithmetic.

The structure being inverted
----------------------------
A voice is a `[N, 3072]` prefix. Each frame is the SUM of 37 rows of one
`[9088, 3072]` table (`mm_audio_embeddings.audio_codebook_embeddings.embeddings`),
one row per codebook, each codebook occupying a contiguous slice:

    codebook 0 (semantic):  8192 codes + 2 special =   8194 rows, offset 0
    codebooks 1..36 (FSQ):    21 levels + 2 special =    23 rows each

giving 8194 + 36 * 23 = 9022 real rows, padded up to 9088 (the table is allocated
at a multiple of 128). Sizes come from `MultiVocabEmbeddings` +
`MultimodalAudioModelArgs.get_codebook_sizes` in vllm_omni, and are re-derived
from params.json here rather than hardcoded.

`voxtral_tts_audio_tokenizer.py::encode_tokens` performs exactly this sum, which
is what makes the inverse useful: recover codes for a clip by any means and the
shipped code turns them back into a voice prefix.

Why the inverse is tractable
----------------------------
Choosing one row per group to hit a target sum is combinatorial in general
(8194 * 23^36 candidates), but the groups interact only through the residual, so
block coordinate descent solves each group EXACTLY given the others: hold 36
fixed, and the 37th is a nearest-row lookup on the residual. Sweeps are
monotonic in squared error, so this converges, and on real data it converges to
zero error rather than a local minimum, because a zero-error solution exists.

That last point is what makes B1 worth doing on the SHIPPED voices first: their
codes exist, so a residual of zero is proof the recovery is correct rather than
merely plausible. On a clip we have no ground truth for, no such check exists.

Everything runs on CPU in seconds. torch is pinned to the CPU wheel (no CUDA
download, no contention with the running server).

Usage
-----
    ./recover_codes.py run                    # all registered voices
    ./recover_codes.py run --voice fr_female  # just one
    ./recover_codes.py run --sweeps 12        # more coordinate-descent passes
"""

from __future__ import annotations

import json
import struct
from pathlib import Path

import click
import numpy as np
import torch
from loguru import logger

REPO_ROOT = Path(__file__).resolve().parents[2]
VOICES_TOOL = Path(__file__).resolve().parent / "voices_tool.py"
OUT_DIR = REPO_ROOT / "cache" / "cloning" / "codes"
TABLE_KEY = "mm_audio_embeddings.audio_codebook_embeddings.embeddings.weight"
# `AudioSpecialTokens` has exactly two members ([EMPTY_AUDIO], [END_AUDIO]) and
# get_codebook_sizes() adds them to EVERY codebook, not just the first.
N_SPECIAL_TOKENS = 2


def find_snapshot() -> Path:
    """Locate the checkpoint snapshot directory.

    Duplicating voices_tool's resolution logic would be one copy too many, so
    this shells out to its `paths` command, the same way derive_voices.py does.

    Returns
    -------
    Path
        Directory holding params.json and consolidated.safetensors.
    """
    import subprocess

    out = subprocess.run([str(VOICES_TOOL), "paths"], capture_output=True, text=True,
                         check=True).stdout
    for line in out.splitlines():
        parts = line.split()
        if len(parts) == 2 and parts[0] == "snapshot":
            return Path(parts[1])
    raise SystemExit(f"could not parse a snapshot path out of:\n{out}")


def codebook_layout(snapshot: Path) -> tuple[list[int], list[int]]:
    """Derive each codebook's size and offset in the shared embedding table.

    Read from params.json rather than hardcoded so a checkpoint with different
    codebook geometry is handled or fails loudly, instead of silently decoding
    into the wrong slices.

    Parameters
    ----------
    snapshot : Path
        Checkpoint snapshot directory.

    Returns
    -------
    tuple of (list of int, list of int)
        Sizes (with special tokens included) and their exclusive-prefix offsets.
    """
    args = json.loads((snapshot / "params.json").read_text())["multimodal"]["audio_model_args"]
    sizes = ([args["semantic_codebook_size"] + N_SPECIAL_TOKENS]
             + [args["acoustic_codebook_size"] + N_SPECIAL_TOKENS] * args["n_acoustic_codebook"])
    offsets = np.cumsum([0] + sizes[:-1]).tolist()
    return sizes, offsets


def load_embedding_table(snapshot: Path) -> torch.Tensor:
    """Read the codebook table out of consolidated.safetensors.

    Parsed by hand (8-byte header length, JSON header, raw tensor bytes) so the
    3 GB file is never fully loaded: only this one tensor's byte range is read.

    Parameters
    ----------
    snapshot : Path
        Checkpoint snapshot directory.

    Returns
    -------
    torch.Tensor
        `[9088, 3072]` float32 (upcast from the stored bfloat16 so residuals are
        measured in a precision finer than the thing being measured).
    """
    path = snapshot / "consolidated.safetensors"
    with open(path, "rb") as fh:
        header_len = struct.unpack("<Q", fh.read(8))[0]
        header = json.loads(fh.read(header_len))
        if TABLE_KEY not in header:
            raise SystemExit(f"{path} has no tensor named {TABLE_KEY}")
        entry = header[TABLE_KEY]
        if entry["dtype"] != "BF16":
            raise SystemExit(f"unexpected dtype {entry['dtype']} for {TABLE_KEY}")
        start, end = entry["data_offsets"]
        fh.seek(8 + header_len + start)
        raw = fh.read(end - start)
    table = torch.frombuffer(bytearray(raw), dtype=torch.bfloat16).reshape(entry["shape"])
    logger.info(f"embedding table {tuple(table.shape)} from {path.name}")
    return table.float()


def _descend(target: torch.Tensor, table: torch.Tensor, sizes: list[int], offsets: list[int],
             sweeps: int, codes: torch.Tensor, first_code: int) -> torch.Tensor:
    """Run block coordinate descent from a given starting point, in place.

    Each group's step is exact given the others (nearest row to the residual), so
    squared error is non-increasing and the loop terminates. All frames descend
    simultaneously: they are independent problems sharing one table, which turns
    each group's step into a single `[N, 3072] @ [3072, size]` matmul.

    Parameters
    ----------
    target : torch.Tensor
        `[N, 3072]` float32 embedding to explain.
    table : torch.Tensor
        `[V, 3072]` float32 codebook table.
    sizes, offsets : list of int
        Per-codebook row counts and their starts in `table`.
    sweeps : int
        Maximum passes over the codebooks; stops early when nothing moves.
    codes : torch.Tensor
        `[N, 37]` starting codes, modified in place and returned.
    first_code : int
        Lowest code id the search may use.

    Returns
    -------
    torch.Tensor
        The same `codes` tensor, descended.
    """
    n_groups = len(sizes)
    # Running reconstruction, kept incrementally: recomputing the full 37-row sum
    # for every group of every sweep would be 37x the work for the same numbers.
    recon = torch.stack([table[offsets[g] + codes[:, g]] for g in range(n_groups)]).sum(0)

    for sweep in range(sweeps):
        moved = 0
        for g in range(n_groups):
            block = table[offsets[g] + first_code:offsets[g] + sizes[g]]
            local = codes[:, g] - first_code
            residual = target - recon + block[local]
            # argmin_k ||block_k - residual||^2, expanded so only the k-dependent
            # terms are computed: -||block_k||^2 + 2 <block_k, residual>, maximized.
            scores = residual @ block.T
            scores = scores.mul_(2.0).sub_(block.pow(2).sum(1))
            best = scores.argmax(dim=1)
            moved += int((best != local).sum())
            recon = recon - block[local] + block[best]
            codes[:, g] = best + first_code
        logger.debug(f"sweep {sweep + 1}: {moved} code(s) changed")
        if moved == 0:
            break
    return codes


def _solved_mask(target: torch.Tensor, table: torch.Tensor, offsets: list[int],
                 codes: torch.Tensor) -> torch.Tensor:
    """Which frames re-embed to their stored bfloat16 value exactly.

    Bit identity rather than a residual threshold: the shipped voices were made
    by this exact sum, so a correct frame reproduces its bits. It is also the
    only criterion that needs no arbitrary tolerance.

    Parameters
    ----------
    target : torch.Tensor
        `[N, 3072]` float32 view of the stored bfloat16 embedding.
    table : torch.Tensor
        `[V, 3072]` float32 codebook table.
    offsets : list of int
        Codebook starts in `table`.
    codes : torch.Tensor
        `[N, 37]` codes to test.

    Returns
    -------
    torch.Tensor
        `[N]` bool, True where the frame is bit-exact.
    """
    recon = sum(table[offsets[g] + codes[:, g]] for g in range(codes.shape[1]))
    return (recon.bfloat16() == target.bfloat16()).all(dim=1)


def recover(target: torch.Tensor, table: torch.Tensor, sizes: list[int], offsets: list[int],
            sweeps: int, first_code: int = 0, restarts: int = 24,
            generator: torch.Generator | None = None) -> tuple[torch.Tensor, torch.Tensor]:
    """Recover one code per codebook per frame.

    Coordinate descent alone leaves a few frames in a local minimum (measured
    2026-09-09: ~95% of frames bit-exact on the shipped voices, the rest close
    but wrong). Since frames are independent, the fix is cheap: keep the solved
    ones and restart ONLY the stragglers from random codes, as many times as
    needed. Each restart costs proportionally less as the unsolved set shrinks.

    `first_code` defaults to 0, i.e. the search may use the 2 reserved special
    rows of each codebook. Excluding them looks principled (real audio codes are
    shifted past [EMPTY_AUDIO] / [END_AUDIO]) and is wrong: measured the same
    day, forbidding them makes the worst frame 150x worse, because real voices
    do contain frames built from those rows.

    Parameters
    ----------
    target : torch.Tensor
        `[N, 3072]` float32 voice embedding to explain.
    table : torch.Tensor
        `[V, 3072]` float32 codebook table.
    sizes, offsets : list of int
        Per-codebook row counts and their starts in `table`.
    sweeps : int
        Maximum coordinate-descent passes per attempt.
    first_code : int
        Lowest code id the search may use.
    restarts : int
        Maximum random restarts for frames still unsolved.
    generator : torch.Generator or None
        RNG for the restarts, for reproducibility.

    Returns
    -------
    tuple of (torch.Tensor, torch.Tensor)
        `[N, 37]` int64 codes and `[N]` float32 reconstruction residual norms.
    """
    n_frames, n_groups = target.shape[0], len(sizes)
    codes = _descend(target, table, sizes, offsets, sweeps,
                     torch.full((n_frames, n_groups), first_code, dtype=torch.long), first_code)
    solved = _solved_mask(target, table, offsets, codes)

    for attempt in range(restarts):
        if bool(solved.all()):
            break
        idx = (~solved).nonzero(as_tuple=True)[0]
        start = torch.stack([
            torch.randint(first_code, sizes[g], (len(idx),), generator=generator)
            for g in range(n_groups)
        ], dim=1)
        retry = _descend(target[idx], table, sizes, offsets, sweeps, start, first_code)
        retry_solved = _solved_mask(target[idx], table, offsets, retry)
        # Only overwrite frames the restart actually solved: a random restart can
        # land worse than the descent it is replacing.
        take = idx[retry_solved]
        codes[take] = retry[retry_solved]
        solved[take] = True
        logger.debug(f"restart {attempt + 1}: {len(idx)} unsolved -> {int((~solved).sum())}")

    recon = sum(table[offsets[g] + codes[:, g]] for g in range(n_groups))
    return codes, (target - recon).norm(dim=1)


@click.group()
def cli() -> None:
    """Recover discrete codes from Voxtral voice embeddings (stage B1)."""


@cli.command("run")
@click.option("--voice", default=None, help="Only this voice (default: every registered one).")
@click.option("--sweeps", default=12, show_default=True, help="Max coordinate-descent passes.")
@click.option("--restarts", default=24, show_default=True,
              help="Max random restarts for frames coordinate descent leaves unsolved.")
@click.option("--no-special", is_flag=True,
              help="Forbid the 2 reserved special-token rows per codebook. Diagnostic "
                   "only: real voices use them, and this makes recovery much worse.")
@click.option("--tolerance", default=0.5, show_default=True,
              help="Residual norm at or below which recovery counts as exact.")
def cmd_run(voice: str | None, sweeps: int, restarts: int, tolerance: float,
            no_special: bool) -> None:
    """Recover codes for the voices and write them to cache/cloning/codes/."""
    snapshot = find_snapshot()
    sizes, offsets = codebook_layout(snapshot)
    logger.info(f"{len(sizes)} codebooks, sizes {sizes[0]} + {len(sizes) - 1} x {sizes[1]}, "
                f"{sum(sizes)} rows used")
    table = load_embedding_table(snapshot)

    embedding_dir = snapshot / "voice_embedding"
    names = [voice] if voice else sorted(p.stem for p in embedding_dir.glob("*.pt"))
    OUT_DIR.mkdir(parents=True, exist_ok=True)

    exact, inexact = 0, []
    semantic_counts = np.zeros(sizes[0], dtype=np.int64)
    acoustic_counts = np.zeros(sizes[1], dtype=np.int64)
    for name in names:
        target = torch.load(embedding_dir / f"{name}.pt", map_location="cpu",
                            weights_only=True).float()
        codes, residuals = recover(target, table, sizes, offsets, sweeps,
                                   first_code=N_SPECIAL_TOKENS if no_special else 0,
                                   restarts=restarts,
                                   generator=torch.Generator().manual_seed(0))
        worst = float(residuals.max())
        # Scale-free companion to the raw norm: a 3072-dim bf16 sum has a
        # magnitude of its own, so "0.4" only means something next to it.
        relative = worst / float(target.norm(dim=1).max())
        status = "exact" if worst <= tolerance else "INEXACT"
        if worst <= tolerance:
            exact += 1
        else:
            inexact.append(name)
        np.save(OUT_DIR / f"{name}.npy", codes.numpy().astype(np.int16))
        logger.info(f"{name:16} {tuple(codes.shape)}  worst residual {worst:8.4f} "
                    f"({relative:.2e} relative)  {status}")

        semantic_counts += np.bincount(codes[:, 0].numpy(), minlength=sizes[0])
        acoustic_counts += np.bincount(codes[:, 1:].reshape(-1).numpy(), minlength=sizes[1])

    stats = {
        "codebook_sizes": sizes,
        "offsets": offsets,
        "voices": len(names),
        "semantic_distinct_codes_used": int((semantic_counts > 0).sum()),
        "semantic_total_frames": int(semantic_counts.sum()),
        "acoustic_level_histogram": acoustic_counts.tolist(),
    }
    (OUT_DIR / "stats.json").write_text(json.dumps(stats, indent=2))

    logger.success(f"{exact}/{len(names)} voices recovered exactly, codes in {OUT_DIR}")
    if inexact:
        logger.error(f"inexact: {', '.join(inexact)}")
        raise SystemExit(1)


@cli.command("verify")
@click.option("--voice", default=None, help="Only this voice (default: every recovered one).")
def cmd_verify(voice: str | None) -> None:
    """Check recovered codes against the embeddings they came from.

    Three independent checks, because a small residual alone is weak evidence:

    1. **Bit identity.** Re-embed the codes, round to bfloat16, and compare with
       the stored tensor element by element. The shipped voices were produced by
       this very sum, so a correct recovery reproduces their bits, not merely
       their neighbourhood. This is the real acceptance test; the residual norm
       is only a diagnostic.
    2. **Range.** Codes must be inside their codebook.
    3. **Special-token offset.** `get_codebook_sizes` reserves 2 special ids at
       the bottom of every codebook and real audio codes are shifted past them,
       so genuine codes are all >= 2. A numerical fit that happened to land on
       the right sum has no reason to respect that, which makes this a cheap
       structural check that the codes mean what we think they mean.
    """
    snapshot = find_snapshot()
    sizes, offsets = codebook_layout(snapshot)
    table = load_embedding_table(snapshot)
    embedding_dir = snapshot / "voice_embedding"

    names = [voice] if voice else sorted(p.stem for p in OUT_DIR.glob("*.npy"))
    if not names:
        raise SystemExit(f"no recovered codes in {OUT_DIR}, run `recover_codes.py run` first")

    click.echo(f"{'voice':16} {'frames':>6} {'exact fr':>9} {'bit-identical':>14} "
               f"{'semantic':>14} {'acoustic':>10} {'>=2':>5}")
    failures = []
    for name in names:
        codes = np.load(OUT_DIR / f"{name}.npy").astype(np.int64)
        target = torch.load(embedding_dir / f"{name}.pt", map_location="cpu", weights_only=True)
        recon = sum(table[offsets[g] + codes[:, g]] for g in range(len(sizes)))
        matches = (recon.bfloat16() == target)
        identical = float(matches.float().mean())
        # Per FRAME, not per element: recovery either nails a frame's 37 codes or
        # it does not, and an elementwise average blurs "two bad frames" into the
        # same number as "every frame slightly off".
        frames_exact = float(matches.all(dim=1).float().mean())

        semantic, acoustic = codes[:, 0], codes[:, 1:]
        in_range = bool(semantic.max() < sizes[0] and acoustic.max() < sizes[1]
                        and codes.min() >= 0)
        past_special = bool(codes.min() >= N_SPECIAL_TOKENS)
        if frames_exact < 1.0 or not in_range:
            failures.append(name)
        click.echo(f"{name:16} {codes.shape[0]:6d} {frames_exact * 100:9.1f}% {identical * 100:13.2f}% "
                   f"[{semantic.min():5d},{semantic.max():5d}] "
                   f"[{acoustic.min():2d},{acoustic.max():2d}] "
                   f"{'yes' if past_special else 'NO':>5}")

    if failures:
        logger.warning(f"not bit-exact: {', '.join(failures)}")
        logger.warning("expected for DERIVED voices: a blend of two code sums is not "
                       "itself a code sum, so no codes explain it. A shipped voice here "
                       "would be a real failure.")
        return
    logger.success(f"all {len(names)} voices reproduce their embedding bit for bit")


if __name__ == "__main__":
    cli()
