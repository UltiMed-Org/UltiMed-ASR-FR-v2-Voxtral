#!/usr/bin/env -S uv run --script
# /// script
# requires-python = ">=3.10"
# dependencies = ["click", "loguru", "torch", "numpy", "einops", "regex"]
# ///
"""Prove the server's code dump really is the codes that made the server's audio.

Why this exists
---------------
`step.sh`'s B7 smoke test checks that a dump appeared and holds a whole number of
frames matching the audio's duration. That is necessary and nowhere near sufficient:
a dump written from the wrong tensor, off by a frame, or in the wrong code space
would pass it and then quietly poison every hour of training data collected
afterwards. Nothing downstream would complain, because wrong codes are still *valid*
codes.

This closes it. One request, keeping both halves, then the codes are decoded locally
with the shipped codec decoder and the waveform compared against the wav the server
returned. 37 codes drive every sample of their frame, so the two cannot agree by
accident: a correlation near 1 means the dump is exactly what produced that audio.
The residual is numerics, not content (the server runs flash-attn in bf16, this runs
SDPA in fp32), which is why the threshold is a correlation rather than equality.

Run it after any rebuild of the voxtral-tts image, and before trusting a long
collection run.

Usage
-----
    ./verify_dump.py                      # needs voxtral-tts up
    ./verify_dump.py --voice de_female --keep
"""

from __future__ import annotations

import sys
import wave
from pathlib import Path

import click
import numpy as np
import torch
from loguru import logger

REPO_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO_ROOT))
sys.path.insert(0, str(Path(__file__).resolve().parent))

import tts  # noqa: E402  (needs REPO_ROOT on the path first)
from decoder import codes_from_table_space, find_snapshot, load_decoder  # noqa: E402

DUMP_DIR = REPO_ROOT / "cache" / "cloning" / "dumps"
N_CODEBOOKS = 37
# Below this the dump is not the audio's codes. Well clear of the ~0.9999 the
# server/local numerics difference actually leaves.
MATCH_THRESHOLD = 0.99


def read_wav(path: Path) -> tuple[np.ndarray, int]:
    """Read a 16-bit mono wav as floats in [-1, 1].

    Parameters
    ----------
    path : Path
        The wav written by the server.

    Returns
    -------
    tuple[np.ndarray, int]
        Samples and sample rate.
    """
    with wave.open(str(path)) as handle:
        raw = handle.readframes(handle.getnframes())
        return np.frombuffer(raw, dtype="<i2").astype(np.float32) / 32768.0, handle.getframerate()


@click.command()
@click.option("--text", default="Le vieux chêne du jardin a soixante-quinze ans.", show_default=True,
              help="What to synthesize.")
@click.option("--voice", default="fr_male", show_default=True, help="Which built-in voice.")
@click.option("--url", default=tts.DEFAULT_URL, show_default=True, help="voxtral-tts base URL.")
@click.option("--device", default="cpu", show_default=True,
              help="Where to run the local decoder. Keep it on cpu when the server owns the GPU.")
@click.option("--keep", is_flag=True, help="Leave the dump and wav behind instead of cleaning up.")
@click.option("--dump-dir", default=str(DUMP_DIR), show_default=True,
              help="Host side of the container's VOXTRAL_CODE_DUMP_DIR.")
def main(text: str, voice: str, url: str, device: str, keep: bool, dump_dir: str) -> None:
    """Round-trip one request: server audio versus locally decoded dump."""
    import time

    tag = f"verify_{int(time.time())}"
    wav_path = REPO_ROOT / "cache" / "cloning" / "out" / f"{tag}.wav"
    wav_path.parent.mkdir(parents=True, exist_ok=True)

    body = tts.build_speech_body(text=text, response_format="wav", speed=1.0,
                                 voice=voice, noise_seed=7, dump_codes=tag)
    status, payload, _, elapsed = tts._post_json(f"{url.rstrip('/')}/v1/audio/speech", body, 300.0)
    if status != 200:
        raise click.ClickException(f"HTTP {status}: {payload[:200]!r}")
    wav_path.write_bytes(payload)
    logger.info(f"server answered in {elapsed:.1f}s, {len(payload)} bytes")

    dump = Path(dump_dir) / f"{tag}.i16"
    if not dump.exists() or dump.stat().st_size == 0:
        raise click.ClickException(
            f"no codes at {dump}. The request succeeded, so either the running image predates "
            "the code-dump patch or VOXTRAL_CODE_DUMP_DIR is not set on the service")

    raw = np.fromfile(dump, dtype="<i2")
    table = raw[: raw.size - raw.size % N_CODEBOOKS].reshape(-1, N_CODEBOOKS).astype(np.int64)
    codes = codes_from_table_space(table)
    logger.info(f"dump: {table.shape[0]} frames, {codes.shape[1]} after dropping special-token frames")

    model = load_decoder(find_snapshot(), device=device)
    with torch.no_grad():
        mine = model.decode(codes[None]).reshape(-1).cpu().numpy()
    theirs, rate = read_wav(wav_path)

    # Lengths differ by the special-token frames the decoder drops, so compare the
    # overlap rather than padding one of them with silence.
    n = min(len(mine), len(theirs))
    a, b = mine[:n], theirs[:n]
    logger.info(f"local {len(mine)} samples, server {len(theirs)} at {rate} Hz, comparing {n}")
    logger.info(f"rms local {np.sqrt((a ** 2).mean()):.4f}, server {np.sqrt((b ** 2).mean()):.4f}")
    correlation = float(np.corrcoef(a, b)[0, 1])
    error = float(np.sqrt(((a - b) ** 2).mean()) / (np.sqrt((b ** 2).mean()) + 1e-9))
    logger.info(f"correlation {correlation:.6f}, relative rms error {100 * error:.2f}%")

    if not keep:
        dump.unlink(missing_ok=True)
        wav_path.unlink(missing_ok=True)

    if correlation < MATCH_THRESHOLD:
        raise click.ClickException(
            f"correlation {correlation:.4f} is below {MATCH_THRESHOLD}: the dump is NOT the codes "
            "that produced this audio, so any data collected with it is unusable")
    logger.success(f"dump verified: it is exactly the codes behind the server's audio "
                   f"(correlation {correlation:.6f})")


if __name__ == "__main__":
    main()
