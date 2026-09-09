#!/usr/bin/env -S uv run --script
# /// script
# requires-python = ">=3.10"
# dependencies = ["click", "loguru", "torch", "numpy", "einops", "regex"]
#
# [tool.uv.sources]
# torch = { index = "pytorch-cpu" }
#
# [[tool.uv.index]]
# name = "pytorch-cpu"
# url = "https://download.pytorch.org/whl/cpu"
# explicit = true
# ///
"""Decode the recovered voice codes back into audio, on CPU (stage B2).

Chains B1 and the standalone decoder: `recover_codes.py` turned each shipped
voice prefix into `[N, 37]` codes, and this turns those codes into a wav. The
output is Mistral's own reference recording for that voice, reconstructed from
the checkpoint, which is the single most informative artifact of the plan: it
shows what a voice prefix actually *is*, and it proves the decoder runs correctly
outside the server before Track B starts relying on it.

Usage
-----
    ./decode_voices.py                      # every recovered voice
    ./decode_voices.py --voice fr_female    # one
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
    write_wav,
)

REPO_ROOT = Path(__file__).resolve().parents[2]
CODES_DIR = REPO_ROOT / "cache" / "cloning" / "codes"
OUT_DIR = REPO_ROOT / "cache" / "cloning" / "B2"


@click.command()
@click.option("--voice", default=None, help="Only this voice (default: every recovered one).")
@click.option("--out-dir", default=str(OUT_DIR), show_default=True, help="Where to write wavs.")
def main(voice: str | None, out_dir: str) -> None:
    """Decode recovered codes to wavs."""
    snapshot = find_snapshot()
    model = load_decoder(snapshot)
    sample_rate = int(model.sampling_rate)
    logger.info(f"decoder ready, {sample_rate} Hz, frame rate {model.frame_rate} Hz")

    names = [voice] if voice else sorted(p.stem for p in CODES_DIR.glob("*.npy"))
    if not names:
        raise SystemExit(f"no codes in {CODES_DIR}, run recover_codes.py first")

    destination = Path(out_dir)
    for name in names:
        raw = np.load(CODES_DIR / f"{name}.npy")
        codes = codes_from_table_space(raw)
        dropped = raw.shape[0] - codes.shape[1]
        if codes.shape[1] == 0:
            # Every frame carried a special-token id, i.e. B1 could not express this
            # prefix as real audio codes at all. That is the expected outcome for the
            # interpolated derived voices (see B1: blends live off the code manifold),
            # and there is nothing to decode.
            logger.warning(f"{name:16} skipped, all {raw.shape[0]} frames are off-manifold")
            continue
        if dropped > 1:
            # One trailing end-of-audio frame is normal. More than that means interior
            # frames were dropped, so the wav is a splice of the representable parts.
            logger.warning(f"{name:16} {dropped} frames dropped, output is a splice")
        with torch.no_grad():
            waveform = model.decode(codes[None])          # [1, 1, samples]
        seconds = write_wav(destination / f"{name}.wav", waveform[0, 0], sample_rate)
        logger.info(f"{name:16} {codes.shape[1]:4d} frames -> {seconds:6.2f} s")

    logger.success(f"{len(names)} wav(s) in {destination}")


if __name__ == "__main__":
    main()
