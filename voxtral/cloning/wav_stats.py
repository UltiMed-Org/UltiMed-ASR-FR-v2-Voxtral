#!/usr/bin/env -S uv run --script
# /// script
# requires-python = ">=3.10"
# dependencies = ["click", "loguru"]
# ///
"""Triage a directory of TTS takes before listening to them.

Stage A2 onwards of PLAN_VOXTRALTTS_CLONING_TEST.md. When a sweep produces a
dozen wavs of the same sentence, three numbers say which ones are worth the
first listen and which are probably degenerate:

- **seconds**: same text everywhere, so duration is a speaking-rate proxy.
- **F0**: median pitch. Separates registers, and catches a blend that simply
  collapsed onto one of its parents.
- **rms**: level. A derived voice that comes out much quieter than its parent is
  the signature of a smeared, off-manifold prefix (the decoder hedging), which
  is exactly the failure mode interpolation is expected to have.

Deliberately stdlib-only (`wave` + `array`), no numpy and no torch: this runs on
every sweep, and it would be absurd to pay a wheel for three scalars. The F0
estimator is autocorrelation on a 4x-downsampled signal, which is crude and can
land an octave off on breathy takes. It is a triage aid, never a verdict: the
acceptance test for a voice is listening to it.

Usage
-----
    ./wav_stats.py cache/cloning/A2/wav
    ./wav_stats.py cache/cloning/A2/wav --ref fr_female --markdown
"""

from __future__ import annotations

import array
import io
import statistics
import wave
from pathlib import Path

import click
from loguru import logger

# 70..350 Hz: wide enough for a low male voice and a high female one, narrow
# enough that halving/doubling errors stay rare.
F0_MIN_HZ = 70.0
F0_MAX_HZ = 350.0
DOWNSAMPLE = 4
WINDOW = 512
# Below this windowed energy the frame is silence; below this normalized
# autocorrelation peak the frame is unvoiced (a fricative, a breath).
SILENCE_ENERGY = 1e6
VOICING_THRESHOLD = 0.3

_WIDTH_TO_TYPECODE = {1: "b", 2: "h", 4: "i"}


def wav_seconds(source: Path | bytes) -> float:
    """Duration of a PCM wav, from its header alone.

    `read_wav` decodes every sample, which is far too much work when the only question is
    how long a take ran; the sweep asks that of every take it synthesizes, in flight, to
    decide whether a voice is degenerating. Reading the header answers it in microseconds,
    and accepting raw bytes means the freshly received response never has to be read back
    off the disk.

    Parameters
    ----------
    source : Path or bytes
        A PCM wav, either on disk or still in memory.

    Returns
    -------
    float
        Seconds of audio, 0.0 when the header cannot be parsed (a truncated or empty
        response, which the caller treats as a failed take rather than a crash).
    """
    handle = io.BytesIO(source) if isinstance(source, (bytes, bytearray)) else str(source)
    try:
        with wave.open(handle) as reader:
            rate = reader.getframerate()
            return reader.getnframes() / rate if rate else 0.0
    except (wave.Error, EOFError):
        return 0.0


def read_wav(path: Path) -> tuple[list[int], int]:
    """Read a PCM wav into a list of samples.

    Parameters
    ----------
    path : Path
        A PCM wav (the format tts.py writes with `--format wav`).

    Returns
    -------
    tuple of (list of int, int)
        Samples (first channel only) and the sample rate.
    """
    with wave.open(str(path)) as w:
        n_frames, rate, width, channels = (
            w.getnframes(), w.getframerate(), w.getsampwidth(), w.getnchannels())
        raw = w.readframes(n_frames)
    if width not in _WIDTH_TO_TYPECODE:
        raise click.ClickException(f"{path.name}: {width * 8}-bit samples are not supported")
    samples = array.array(_WIDTH_TO_TYPECODE[width])
    samples.frombytes(raw)
    values = list(samples)
    if channels > 1:
        values = values[::channels]
    return values, rate


def median_f0(samples: list[int], rate: int) -> float:
    """Median fundamental frequency in Hz, or nan when nothing is voiced.

    Parameters
    ----------
    samples : list of int
        PCM samples.
    rate : int
        Sample rate of `samples`.

    Returns
    -------
    float
        Median F0 over voiced windows, nan if there are none.
    """
    # Downsampling by averaging is both a cheap anti-alias and a 16x cut in the
    # autocorrelation cost, which is what makes a pure-Python estimator viable.
    decimated = [sum(samples[i:i + DOWNSAMPLE]) / DOWNSAMPLE
                 for i in range(0, len(samples) - DOWNSAMPLE, DOWNSAMPLE)]
    rate_d = rate / DOWNSAMPLE
    lag_min, lag_max = int(rate_d / F0_MAX_HZ), int(rate_d / F0_MIN_HZ)

    estimates: list[float] = []
    for start in range(0, len(decimated) - WINDOW, WINDOW):
        window = decimated[start:start + WINDOW]
        energy = sum(x * x for x in window)
        if energy < SILENCE_ENERGY:
            continue
        best_score, best_lag = 0.0, 0
        for lag in range(lag_min, min(lag_max, WINDOW - 1)):
            score = sum(window[i] * window[i + lag] for i in range(WINDOW - lag)) / (WINDOW - lag)
            if score > best_score:
                best_score, best_lag = score, lag
        if best_lag and best_score / (energy / WINDOW) > VOICING_THRESHOLD:
            estimates.append(rate_d / best_lag)
    return statistics.median(estimates) if estimates else float("nan")


@click.command()
@click.argument("directory", type=click.Path(exists=True, file_okay=False, path_type=Path))
@click.option("--ref", default=None, help="Voice to sort first and compare the others against.")
@click.option("--markdown", is_flag=True, help="Emit a markdown table (for notes.md).")
def main(directory: Path, ref: str | None, markdown: bool) -> None:
    """Print seconds / F0 / rms for every wav in DIRECTORY."""
    paths = sorted(directory.glob("*.wav"), key=lambda p: (p.stem != ref, p.stem))
    if not paths:
        raise click.ClickException(f"no wavs in {directory}")

    rows: list[tuple[str, float, float, float]] = []
    for path in paths:
        samples, rate = read_wav(path)
        rms = (sum(x * x for x in samples) / len(samples)) ** 0.5
        rows.append((path.stem, len(samples) / rate, median_f0(samples, rate), rms))
        logger.debug(f"{path.stem}: {len(samples)} samples @ {rate} Hz")

    if markdown:
        click.echo("| voice | sec | F0 Hz | rms |")
        click.echo("|---|---|---|---|")
        for name, seconds, f0, rms in rows:
            click.echo(f"| {name} | {seconds:.2f} | {f0:.0f} | {rms:.0f} |")
        return
    click.echo(f"{'voice':18} {'sec':>6} {'F0 Hz':>7} {'rms':>7}")
    for name, seconds, f0, rms in rows:
        click.echo(f"{name:18} {seconds:6.2f} {f0:7.1f} {rms:7.0f}")


if __name__ == "__main__":
    main()
