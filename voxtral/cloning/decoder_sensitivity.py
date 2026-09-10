#!/usr/bin/env -S uv run --script
# /// script
# requires-python = ">=3.10"
# dependencies = ["click", "loguru", "torch", "numpy", "einops", "regex"]
# ///
"""Is the decoder injective enough that codes can be read back out of audio? (stage B7 diagnostic)

Motivation: during the B7 retrain the semantic head memorised its training batch
(97% top-1) while the acoustic head could not (about 50% within-1 on the same
frames it had just seen). The obvious explanation would be that the decoder is
many-to-one in the 36 FSQ dimensions, i.e. that different acoustic codes produce
near-identical audio. If that were true those codes would be unrecoverable by any
encoder, no matter how much data it saw, and the whole "predict the codes" framing
would be wrong: the objective would have to become "codes that decode to the right
audio" instead. It would also explain B3's finding that the 36 acoustic dims are
"strongly coupled" and that exact coordinate descent over them made the loss worse.

So this measures it rather than assuming it. It perturbs one code in one frame,
decodes, and reports how far the waveform moved relative to its own RMS, over the
changed frame plus the two that follow it (the decoder has receptive field beyond
the frame it is emitting, so a same-frame-only window would understate the effect).

A dimension whose single-step perturbation moves the audio by a negligible fraction
of RMS is a dimension no encoder can learn. A dimension that moves the audio
audibly is one where failure to predict it is a learning problem, not an ill-posed
one.

Result on 2026-09-10 (Claude Code, fr_female minted clip, frame 12): no dead
dimensions. The smallest single-step effect over all 36 acoustic dims was 1.28% of
RMS and the median was 2.24%, versus 0.83% for swapping the semantic code for its
arbitrary neighbour. The hypothesis is dead: the decoder is sensitive to every
acoustic dimension, inversion is well posed, and the acoustic head's failure to fit
is a learning problem.
"""

from __future__ import annotations

import sys
from pathlib import Path

import click
import numpy as np
import torch
from loguru import logger

sys.path.insert(0, str(Path(__file__).resolve().parent))

from decoder import (  # noqa: E402 - sys.path must be set first
    codes_from_table_space,
    find_snapshot,
    load_decoder,
)

SAMPLES_PER_FRAME = 1920
N_LEVELS = 21
SEMANTIC_SIZE = 8192
TAIL_FRAMES = 2


def rms(x: torch.Tensor) -> float:
    """Root mean square of a waveform.

    Parameters
    ----------
    x
        Any real tensor.

    Returns
    -------
    float
        The RMS as a plain float.
    """
    return x.pow(2).mean().sqrt().item()


def perturbed_distance(
    *,
    model,
    codes: torch.Tensor,
    base: torch.Tensor,
    scale: float,
    frame: int,
    dim: int,
    delta: int,
) -> float | None:
    """Decode one perturbed code sequence and report how far the audio moved.

    Parameters
    ----------
    model
        The loaded decoder.
    codes
        Quantizer-space codes, shape ``[37, T]``.
    base
        The unperturbed waveform, flat.
    scale
        RMS of ``base``, used to make the result a dimensionless fraction.
    frame
        Which frame to perturb.
    dim
        Row of ``codes``: 0 is the semantic code, 1..36 are the FSQ scalars.
    delta
        How many steps to move the code by.

    Returns
    -------
    float or None
        Relative RMS of the waveform difference over the changed frame plus
        ``TAIL_FRAMES`` following ones, or None when clamping at the edge of the
        FSQ range made the perturbation a no-op (nothing to measure).
    """
    moved = codes.clone()
    if dim == 0:
        # The semantic codebook has no ordering, so "+1" is just an arbitrary other
        # row. It is included as a scale reference, not as a comparable step size.
        moved[0, frame] = (moved[0, frame] + delta) % SEMANTIC_SIZE
    else:
        moved[dim, frame] = int(np.clip(moved[dim, frame].item() + delta, 0, N_LEVELS - 1))
        if moved[dim, frame] == codes[dim, frame]:
            return None
    with torch.no_grad():
        after = model.decode(moved[None]).reshape(-1)
    window = slice(frame * SAMPLES_PER_FRAME, (frame + 1 + TAIL_FRAMES) * SAMPLES_PER_FRAME)
    return rms((after - base)[window]) / scale


@click.command()
@click.option("--codes", "codes_path", type=click.Path(exists=True, path_type=Path),
              required=True, help="A .npy of table-space ids, shape [37, T].")
@click.option("--frame", default=12, show_default=True, help="Which frame to perturb.")
@click.option("--frames", default=24, show_default=True,
              help="How many frames of the clip to decode (keeps the sweep quick).")
@click.option("--device", default="cpu", show_default=True, help="Torch device.")
def main(*, codes_path: Path, frame: int, frames: int, device: str) -> None:
    """Sweep every code dimension and report how much the audio moves."""
    model = load_decoder(find_snapshot(), device=device)
    codes = codes_from_table_space(np.load(codes_path))[:, :frames]
    if frame + 1 + TAIL_FRAMES > codes.shape[1]:
        raise click.ClickException(f"--frame {frame} leaves no tail inside {codes.shape[1]} frames")

    with torch.no_grad():
        base = model.decode(codes[None]).reshape(-1)
    scale = rms(base)
    logger.info(f"reference rms {scale:.4f}; effect measured over frame {frame} plus {TAIL_FRAMES} following")

    kwargs = dict(model=model, codes=codes, base=base, scale=scale, frame=frame)
    semantic = perturbed_distance(**kwargs, dim=0, delta=1)
    logger.info(f"semantic code +1 (an unrelated codebook row): {100 * semantic:6.2f}% of rms")

    for delta in (1, 10):
        measured = [(perturbed_distance(**kwargs, dim=d, delta=delta), d) for d in range(1, 37)]
        measured = sorted((v, d) for v, d in measured if v is not None)
        values = np.array([v for v, _ in measured])
        logger.info(
            f"acoustic +{delta} step(s), {len(values)} dims: median {100 * np.median(values):.2f}%"
            f"  min {100 * values.min():.3f}% (dim {measured[0][1]})"
            f"  max {100 * values.max():.2f}% (dim {measured[-1][1]})"
        )
        for threshold in (0.01, 0.001):
            dead = int((values < threshold).sum())
            logger.info(f"  dims moving the audio by less than {100 * threshold}% of rms: {dead} of {len(values)}")


if __name__ == "__main__":
    main()
