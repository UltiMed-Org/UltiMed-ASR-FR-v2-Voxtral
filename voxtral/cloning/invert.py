#!/usr/bin/env -S uv run --script
# /// script
# requires-python = ">=3.10"
# dependencies = ["click", "loguru", "torch", "numpy", "einops", "regex"]
# ///
#
# Unlike the other scripts here this one does NOT pin torch to the CPU index: the
# exhaustive search is only practical on a GPU, so it takes the default wheel, which
# on Linux is the CUDA build. It still runs on CPU (`--device cpu`), just slowly.
"""Analysis by synthesis: recover codes from a waveform through the frozen decoder (stage B3).

This is the decisive experiment for the "off-distribution prefix" risk. The open
checkpoint ships no encoder, so the only way from a waveform to a voice prefix is
to optimize codes until the decoder reproduces the waveform. That gives audio that
*sounds* right by construction, but says nothing about whether the codes are the
ones the real encoder would have emitted, and it is those codes the LLM was
trained to be prefixed with.

B1 and B2 together give the paired example needed to settle it: `recover_codes.py`
produced ground-truth codes for every shipped voice, and `decode_voices.py` turned
them into the exact waveform they decode to. So here the target waveform is one the
decoder provably *can* reach, a zero-loss solution is known to exist, and the only
question measured is whether the optimizer lands on the true codes or on a different
set that merely sounds the same.

What is optimized
-----------------
The decoder's input is `[B, 292, T]` and is fully determined by the codes:

- dims 0..255 are one row of the `[8192, 256]` semantic codebook, so the semantic
  choice is categorical. It is parameterized as a free vector in that 256-dim space
  snapped to its nearest codebook row, with a VQ-VAE style straight-through estimator
  (forward uses the row, backward is the identity) and a commitment term pulling the
  vector onto the row it chose. An annealed softmax over negative distances was tried
  first, as the plan specified, and is strictly worse: at a soft temperature the
  backward mixture bears no relation to the row the forward pass actually used, so the
  gradient is wrong exactly when the search is furthest from the answer. Measured, it
  took a run that had 65 percent of the semantic codes right down to 0 percent.
- dims 256..291 are 36 FSQ scalars on a 21-level grid in [-1, 1], recovered with the
  same straight-through rounding the codec's own `AcousticCodebook` uses.

Because the forward pass rounds, the reconstruction reported is always the true
hard-code reconstruction, never a soft relaxation that would flatter the result.

Usage
-----
    ./invert.py run --voice fr_female --frames 32 --warmup 2000 --steps 1000
    ./invert.py sensitivity --voice fr_female --frames 32
"""

from __future__ import annotations

import json
import math
import sys
from pathlib import Path

import click
import numpy as np
import torch
import torch.nn.functional as F
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
OUT_DIR = REPO_ROOT / "cache" / "cloning" / "B3"

# Multi-resolution STFT scales. Three window sizes so the loss sees both the fine
# harmonic structure (2048) and the transients (512) that a single resolution smears.
STFT_SCALES = ((512, 128), (1024, 256), (2048, 512))
N_MELS = 80


def hz_to_mel(hz: torch.Tensor | float) -> torch.Tensor | float:
    """HTK mel scale."""
    return 2595.0 * math.log10(1.0 + hz / 700.0) if isinstance(hz, float) else 2595.0 * torch.log10(1.0 + hz / 700.0)


def mel_filterbank(n_fft: int, n_mels: int, sample_rate: int) -> torch.Tensor:
    """Build a triangular mel filterbank.

    Hand-rolled rather than pulled from torchaudio/librosa to keep this script's
    dependency set to what the decoder itself already needs.

    Parameters
    ----------
    n_fft : int
        FFT size; the filterbank has `n_fft // 2 + 1` input bins.
    n_mels : int
        Number of triangular filters.
    sample_rate : int
        Audio sample rate in Hz.

    Returns
    -------
    torch.Tensor
        `[n_mels, n_fft // 2 + 1]` filterbank, rows summing to a triangle.
    """
    n_bins = n_fft // 2 + 1
    mel_max = hz_to_mel(sample_rate / 2.0)
    mel_points = torch.linspace(0.0, mel_max, n_mels + 2)
    hz_points = 700.0 * (10.0 ** (mel_points / 2595.0) - 1.0)
    bin_points = hz_points * (n_fft / sample_rate)

    bins = torch.arange(n_bins, dtype=torch.float32)[None, :]
    left, center, right = bin_points[:-2, None], bin_points[1:-1, None], bin_points[2:, None]
    up = (bins - left) / (center - left).clamp(min=1e-6)
    down = (right - bins) / (right - center).clamp(min=1e-6)
    return torch.minimum(up, down).clamp(min=0.0)


def spectral_loss(pred: torch.Tensor, target: torch.Tensor, sample_rate: int,
                  mel_banks: dict[int, torch.Tensor]) -> torch.Tensor:
    """Multi-resolution STFT loss plus a mel term.

    Parameters
    ----------
    pred : torch.Tensor
        Waveform, `[samples]` or a batch `[B, samples]`. The batched form exists so
        the exhaustive search can score thousands of candidates in one decoder pass.
    target : torch.Tensor
        `[samples]` reference waveform.
    sample_rate : int
        Audio sample rate, for the mel filterbank.
    mel_banks : dict[int, torch.Tensor]
        Cached filterbanks keyed by FFT size, so they are not rebuilt every step.

    Returns
    -------
    torch.Tensor
        Scalar loss, or `[B]` when `pred` is batched.
    """
    batched = pred.dim() == 2
    batch = pred if batched else pred[None]
    total = batch.new_zeros(batch.shape[0])
    for n_fft, hop in STFT_SCALES:
        # The progressive schedule evaluates prefixes as short as one 1920-sample
        # frame, which the coarsest window cannot resolve.
        if n_fft > batch.shape[-1]:
            continue
        window = torch.hann_window(n_fft, device=pred.device)
        kwargs = dict(n_fft=n_fft, hop_length=hop, win_length=n_fft, window=window, return_complex=True)
        mag_p = torch.stft(batch, **kwargs).abs()                    # [B, F, frames]
        mag_t = torch.stft(target[None], **kwargs).abs()              # [1, F, frames]
        # Spectral convergence plus log-magnitude, the usual pairing: the first is
        # scale-sensitive and dominated by loud bins, the second weights quiet bins.
        total = total + (mag_p - mag_t).flatten(1).norm(dim=1) / mag_t.norm().clamp(min=1e-8)
        total = total + (torch.log(mag_p + 1e-5) - torch.log(mag_t + 1e-5)).abs().flatten(1).mean(dim=1)
        # Mel term: the perceptual weighting, and what actually pins down timbre.
        bank = mel_banks[n_fft]
        mel_p = torch.einsum("mf,bft->bmt", bank, mag_p)
        mel_t = torch.einsum("mf,bft->bmt", bank, mag_t)
        total = total + (torch.log(mel_p + 1e-5) - torch.log(mel_t + 1e-5)).abs().flatten(1).mean(dim=1)
    return total if batched else total[0]


def build_targets(model: torch.nn.Module, voice: str, frames: int) -> tuple[torch.Tensor, torch.Tensor]:
    """Load a voice's ground-truth codes and the waveform they decode to.

    Parameters
    ----------
    model : torch.nn.Module
        The loaded audio tokenizer.
    voice : str
        Voice name, matching a `.npy` in the B1 codes directory.
    frames : int
        How many leading frames to use. The decoder is causal with a 16-frame
        sliding window, so a short excerpt is representative of the full clip and
        keeps a CPU optimization tractable.

    Returns
    -------
    tuple[torch.Tensor, torch.Tensor]
        `[K, frames]` ground-truth codes and the `[samples]` target waveform.
    """
    device = next(model.parameters()).device
    truth = codes_from_table_space(np.load(CODES_DIR / f"{voice}.npy"))[:, :frames].to(device)
    with torch.no_grad():
        target = model.decode(truth[None])[0, 0]
    return truth, target


def quantize_emb(model: torch.nn.Module, emb: torch.Tensor) -> torch.Tensor:
    """Snap a continuous decoder input to the nearest codes.

    This is exactly what the codec's own `quantizer.encode` does, minus the tanh the
    acoustic side applies to raw encoder output: here the values are already the
    decoder's own scale, i.e. already in [-1, 1].

    Parameters
    ----------
    model : torch.nn.Module
        The audio tokenizer, for its codebooks.
    emb : torch.Tensor
        `[T, 292]` continuous decoder input.

    Returns
    -------
    torch.Tensor
        `[K, T]` integer codes.
    """
    table = model.quantizer.semantic_codebook.embedding.float()
    n_levels = model.quantizer.acoustic_codebook.n_levels
    semantic_dim = model.quantizer.semantic_dim
    sem_codes = torch.cdist(emb[:, :semantic_dim], table).argmin(dim=-1)
    aco_codes = ((emb[:, semantic_dim:].clamp(-1.0, 1.0) + 1.0) / 2.0 * (n_levels - 1)).round().long()
    return torch.cat([sem_codes[None], aco_codes.transpose(0, 1)], dim=0)


def invert(model: torch.nn.Module, target: torch.Tensor, frames: int, *, steps: int, warmup: int,
           lr_semantic: float, lr_acoustic: float, commit_weight: float, wave_weight: float,
           seed: int, stage_steps: int = 0,
           init: torch.Tensor | None = None) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Optimize codes so the frozen decoder reproduces `target`.

    Parameters
    ----------
    model : torch.nn.Module
        The audio tokenizer, used only through `_forward_decoder`; its weights stay frozen.
    target : torch.Tensor
        Target waveform.
    frames : int
        Number of 12.5 Hz frames to solve for.
    steps : int
        Adam steps of the discrete (straight-through) phase.
    stage_steps : int
        Adam steps per prefix length in the progressive schedule that precedes
        everything else. 0 disables it and starts straight at full length.
    warmup : int
        Adam steps of a *continuous* phase, in which the decoder input is a
        free `[T, 292]` tensor with no quantization at all. That problem is smooth and
        has a known exact solution (the true decoder input), so it converges far better
        than the discrete one, and it doubles as the cleanest identifiability
        measurement available: how close the unconstrained inverse lands to the truth
        bounds what any discrete search can hope for.
    lr_semantic, lr_acoustic : float
        Adam learning rates for the semantic and acoustic halves of the decoder input.
        They differ by two orders of magnitude on purpose; see the comment where the
        parameters are built.
    commit_weight : float
        Weight of the commitment term pulling each frame's semantic vector onto the
        codebook row it selected. Without it the vector drifts into empty space where
        the straight-through gradient stops describing the forward pass.
    wave_weight : float
        Weight of an L1 term directly on the waveform. Legitimate here because the
        decoder is deterministic and the target is an exact decoder output, and it
        pins down phase, which magnitude spectrograms discard.
    seed : int
        Seed for the initial random codebook rows.
    init : torch.Tensor, optional
        `[T, 292]` starting point for the continuous phase. Used to start near the
        truth on purpose, which separates "the optimizer cannot descend" from "the
        optimizer cannot find the right basin from far away".

    Returns
    -------
    tuple[torch.Tensor, torch.Tensor, torch.Tensor]
        `[K, frames]` recovered codes, the reconstruction they decode to, and the
        `[K, frames]` codes the continuous phase alone snapped to (equal to the first
        when `warmup` is 0).
    """
    generator = torch.Generator().manual_seed(seed)
    table = model.quantizer.semantic_codebook.embedding.float()          # [8192, 256]
    n_levels = model.quantizer.acoustic_codebook.n_levels
    acoustic_dim = model.quantizer.acoustic_codebook.dim
    sample_rate = int(model.sampling_rate)
    mel_banks = {n_fft: mel_filterbank(n_fft, N_MELS, sample_rate).to(target.device)
                 for n_fft, _ in STFT_SCALES}

    # Start from random real codebook rows rather than from a mean or from zero: the
    # mean of the codebook is not itself a code, so it starts every frame off-manifold
    # and identical, which is the worst possible place to anneal from.
    start_rows = torch.randint(table.shape[0], (frames,), generator=generator).to(table.device)

    for param in model.parameters():
        param.requires_grad_(False)

    # Phase 1, continuous. No quantization anywhere, so there are no straight-through
    # plateaus and the landscape is the smooth one that provably passes through the
    # true decoder input.
    if init is None:
        init = torch.cat([table[start_rows].clone(),
                          torch.zeros(frames, acoustic_dim, device=table.device)], dim=-1)
    # Split into two parameters so each can carry its own learning rate. This is not
    # cosmetic: the two halves live at wildly different scales. Per frame the semantic
    # vector has norm ~145 and its codebook's rows sit ~107 apart, so changing one code
    # means travelling ~6.7 per dimension, while the acoustic dims span [-1, 1] on a
    # 0.1 grid. Adam's step is roughly the learning rate in parameter units, so a single
    # rate is necessarily ~100x too small for one half and several times too large for
    # the other, and the semantic codes then never move off their initialisation.
    sem_free = torch.nn.Parameter(init[:, :table.shape[1]].clone())
    aco_free = torch.nn.Parameter(init[:, table.shape[1]:].clone())
    optimizer = torch.optim.Adam([
        {"params": [sem_free], "lr": lr_semantic},
        {"params": [aco_free], "lr": lr_acoustic},
    ])

    def descend(prefix: int, n_steps: int, label: str) -> float:
        """Run `n_steps` continuous updates against the first `prefix` frames."""
        loss = torch.zeros(())
        for step in range(n_steps):
            emb = torch.cat([sem_free[:prefix], aco_free[:prefix]], dim=-1).transpose(0, 1)[None]
            waveform = model._forward_decoder(emb)[0, 0]
            piece = target[: waveform.numel()]
            loss = spectral_loss(waveform, piece, sample_rate, mel_banks)
            if wave_weight:
                loss = loss + wave_weight * (waveform - piece).abs().mean() / piece.abs().mean()
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            if step % 25 == 0 or step == n_steps - 1:
                logger.debug(f"{label} {step:4d}  loss {loss.item():8.4f}")
        return loss.item()

    # Progressive schedule. The decoder is causal, so the audio of frames 0..k does not
    # depend on anything after k: solving a growing prefix turns one 292*T-dimensional
    # global search into T small ones, each warm-started by the previous. This is the
    # frame-by-frame fallback the plan anticipated, in the cheapest form that still lets
    # earlier frames keep refining.
    for prefix in range(1, frames + 1) if stage_steps else ():
        descend(prefix, stage_steps, f"stage {prefix:3d}/{frames} step")

    descend(frames, warmup, "warmup")

    emb_free = torch.cat([sem_free, aco_free], dim=-1).detach()
    with torch.no_grad():
        warm_codes = quantize_emb(model, emb_free)

    # Phase 2, discrete. Seeded from the continuous solution, so the straight-through
    # search starts inside the right basin instead of at a random codebook row.
    sem_v = torch.nn.Parameter(emb_free[:, :table.shape[1]].clone())
    aco_u = torch.nn.Parameter(torch.atanh(emb_free[:, table.shape[1]:].clamp(-0.999, 0.999)))
    optimizer = torch.optim.Adam([
        {"params": [sem_v], "lr": lr_semantic},
        {"params": [aco_u], "lr": lr_acoustic},
    ])

    waveform = target
    for step in range(steps):
        nearest = table[torch.cdist(sem_v, table).argmin(dim=-1)]        # [frames, 256]
        sem_emb = sem_v + (nearest - sem_v).detach()                     # straight-through

        scaled = (torch.tanh(aco_u) + 1.0) / 2.0 * (n_levels - 1)
        quantized = scaled + (scaled.round() - scaled).detach()          # straight-through
        aco_emb = quantized * 2.0 / (n_levels - 1) - 1.0

        emb = torch.cat([sem_emb, aco_emb], dim=-1).transpose(0, 1)[None]
        waveform = model._forward_decoder(emb)[0, 0]

        loss = spectral_loss(waveform, target, sample_rate, mel_banks)
        if wave_weight:
            loss = loss + wave_weight * (waveform - target).abs().mean() / target.abs().mean()
        if commit_weight:
            loss = loss + commit_weight * F.mse_loss(sem_v, nearest.detach())

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        if step % 25 == 0 or step == steps - 1:
            logger.debug(f"step {step:4d}  loss {loss.item():8.4f}")

    with torch.no_grad():
        distances = torch.cdist(sem_v, table).pow(2)
        sem_codes = distances.argmin(dim=-1)
        aco_codes = ((torch.tanh(aco_u) + 1.0) / 2.0 * (n_levels - 1)).round().long()
        codes = torch.cat([sem_codes[None], aco_codes.transpose(0, 1)], dim=0)
    return codes, waveform.detach(), warm_codes


def score(codes: torch.Tensor, truth: torch.Tensor, n_levels: int) -> dict[str, float]:
    """Compare recovered codes against the ground truth.

    Parameters
    ----------
    codes, truth : torch.Tensor
        `[K, T]` code tensors, semantic on row 0 and acoustic on the rest.
    n_levels : int
        Acoustic levels, used for the random-chance baselines.

    Returns
    -------
    dict[str, float]
        Match rates and acoustic level errors, plus what chance alone would give.
    """
    delta = (codes[1:].float() - truth[1:].float()).abs()
    return {
        "semantic_exact_pct": (codes[0] == truth[0]).float().mean().item() * 100,
        "semantic_chance_pct": 100.0 / 8192,
        "acoustic_exact_pct": (delta == 0).float().mean().item() * 100,
        "acoustic_within1_pct": (delta <= 1).float().mean().item() * 100,
        "acoustic_within2_pct": (delta <= 2).float().mean().item() * 100,
        "acoustic_mean_abs_error": delta.mean().item(),
        # Chance for "within 1" is the average count of levels within 1 of a uniform
        # draw, i.e. (3 * n_levels - 2) / n_levels ** 2 in units of probability.
        "acoustic_within1_chance_pct": (3 * n_levels - 2) / n_levels**2 * 100,
    }


@click.group()
def cli() -> None:
    """Invert waveforms back to codes, and probe whether that is even possible."""


@cli.command()
@click.option("--voice", default="fr_female", show_default=True, help="Voice to invert.")
@click.option("--frames", default=32, show_default=True, help="Leading frames to solve for (12.5 Hz).")
@click.option("--steps", default=0, show_default=True,
              help="Adam steps of the discrete straight-through phase. Off by default: it "
                   "measured strictly worse than snapping the continuous solution.")
@click.option("--warmup", default=800, show_default=True, help="Adam steps of the continuous phase.")
@click.option("--stage-steps", default=80, show_default=True,
              help="Adam steps per prefix length in the progressive schedule (0 disables).")
@click.option("--lr-semantic", default=1.0, show_default=True,
              help="Adam rate for the semantic half; its codebook rows sit ~107 apart.")
@click.option("--lr-acoustic", default=0.01, show_default=True,
              help="Adam rate for the acoustic half; its grid step is 0.1 in [-1, 1].")
@click.option("--commit-weight", default=1.0, show_default=True,
              help="Weight of the semantic commitment term (0 disables).")
@click.option("--wave-weight", default=1.0, show_default=True, help="Weight of the waveform L1 term (0 disables).")
@click.option("--seed", default=0, show_default=True, help="Seed for the initial codebook rows.")
@click.option("--init", "init_mode", type=click.Choice(["random", "truth", "centre"]),
              default="random", show_default=True,
              help="Start from random codebook rows, from the truth plus noise, or from the "
                   "codebook's centre plus noise.")
@click.option("--init-noise", default=0.1, show_default=True,
              help="Relative noise added to the truth when --init truth.")
@click.option("--device", default="cpu", show_default=True, help="Where to run.")
@click.option("--out-dir", default=str(OUT_DIR), show_default=True, help="Where to write wavs and metrics.")
def run(voice: str, frames: int, steps: int, warmup: int, stage_steps: int, lr_semantic: float,
        lr_acoustic: float, commit_weight: float, wave_weight: float, seed: int, init_mode: str,
        init_noise: float, device: str, out_dir: str) -> None:
    """Recover codes from a waveform and score them against the truth."""
    model = load_decoder(find_snapshot(), device=device)
    sample_rate = int(model.sampling_rate)
    truth, target = build_targets(model, voice, frames)
    logger.info(f"{voice}: {frames} frames, target {target.numel() / sample_rate:.2f} s")

    init = None
    if init_mode == "centre":
        # A random codebook row is a valid code, which means it is the bottom of its own
        # basin and descent has no reason to leave it. The codebook's centre is not a
        # code at all, so a start there is downhill towards *some* row, and it happens to
        # sit closer to any given row (~76 in L2) than two rows sit to each other (~107).
        table = model.quantizer.semantic_codebook.embedding.float()
        noise = torch.randn(frames, table.shape[1], generator=torch.Generator().manual_seed(seed))
        centre = table.mean(dim=0)[None] + init_noise * table.std(dim=0)[None] * noise.to(table.device)
        init = torch.cat([centre, torch.zeros(frames, model.quantizer.acoustic_codebook.dim,
                                              device=table.device)], dim=-1)
    if init_mode == "truth":
        # Deliberately cheating, as a control: if the optimizer cannot descend back to
        # zero from a small perturbation of the true answer, no starting point helps and
        # the problem is the gradient path, not the search radius.
        with torch.no_grad():
            exact = model.quantizer.decode(truth[None])[0].transpose(0, 1)
        noise = torch.randn(exact.shape, generator=torch.Generator().manual_seed(seed)).to(exact.device)
        # Per-dim scaling: a single global std is dominated by the 256 semantic dims and
        # would swamp the 36 acoustic ones, whose grid spacing is only 0.1 in [-1, 1].
        init = exact + init_noise * exact.std(dim=0, keepdim=True) * noise
    codes, waveform, warm_codes = invert(model, target, frames, steps=steps, warmup=warmup,
                                         lr_semantic=lr_semantic, lr_acoustic=lr_acoustic,
                                         commit_weight=commit_weight, stage_steps=stage_steps,
                                         wave_weight=wave_weight, seed=seed, init=init)
    n_levels = model.quantizer.acoustic_codebook.n_levels
    codes, waveform, target = codes.cpu(), waveform.cpu(), target.cpu()
    truth, warm_codes = truth.cpu(), warm_codes.cpu()
    metrics = score(codes, truth, n_levels)
    # Kept separately because the two phases answer different questions: the continuous
    # one bounds what is identifiable at all, the discrete one says what a search that
    # respects the code grid actually reaches.
    metrics["continuous_phase"] = score(warm_codes, truth, n_levels)

    destination = Path(out_dir)
    write_wav(destination / f"{voice}_target.wav", target, sample_rate)
    write_wav(destination / f"{voice}_recon.wav", waveform, sample_rate)
    np.save(destination / f"{voice}_codes.npy", codes.numpy().astype(np.int16))
    (destination / f"{voice}_metrics.json").write_text(json.dumps(metrics, indent=2))

    warm = metrics["continuous_phase"]
    logger.info(f"continuous phase: semantic exact {warm['semantic_exact_pct']:6.2f}%  "
                f"acoustic within1 {warm['acoustic_within1_pct']:6.2f}%")
    logger.info(f"semantic exact {metrics['semantic_exact_pct']:6.2f}% "
                f"(chance {metrics['semantic_chance_pct']:.3f}%)")
    logger.info(f"acoustic exact {metrics['acoustic_exact_pct']:6.2f}%  "
                f"within1 {metrics['acoustic_within1_pct']:6.2f}% "
                f"(chance {metrics['acoustic_within1_chance_pct']:.1f}%)  "
                f"within2 {metrics['acoustic_within2_pct']:6.2f}%  "
                f"mean|d| {metrics['acoustic_mean_abs_error']:.2f}")
    logger.success(f"wrote {destination}")


def frame_context(model: torch.nn.Module, codes: torch.Tensor, target: torch.Tensor, frame: int, *,
                  window: int, mel_banks: dict[int, torch.Tensor], samples_per_frame: int):
    """Build the scoring closure for one frame of a causal solve.

    Everything before `frame` is frozen at whatever `codes` holds, everything after is
    irrelevant because the decoder is causal. The closure therefore varies only the last
    frame and judges it on the last one or two frames of audio, which is where the only
    difference between candidates can appear.

    Parameters
    ----------
    model : torch.nn.Module
        The audio tokenizer.
    codes : torch.Tensor
        `[K, T]` codes solved so far; only columns up to `frame` are read.
    target : torch.Tensor
        Target waveform.
    frame : int
        Index of the frame being solved.
    window : int
        Frames of prefix to decode, 0 for all of it.
    mel_banks : dict[int, torch.Tensor]
        Cached mel filterbanks.
    samples_per_frame : int
        Decoder output samples per 12.5 Hz frame.

    Returns
    -------
    callable
        `evaluate(sem_rows, acoustic) -> [B]` losses, differentiable in `acoustic`.
    """
    sample_rate = int(model.sampling_rate)
    low = max(0, frame + 1 - window) if window else 0
    span = min(2, frame + 1 - low) * samples_per_frame
    piece = target[(frame + 1) * samples_per_frame - span:(frame + 1) * samples_per_frame]
    with torch.no_grad():
        prefix = model.quantizer.decode(codes[None, :, low:frame + 1])[0]         # [292, L]
    head = prefix[None, :, :-1]

    def evaluate(sem_rows: torch.Tensor, acoustic: torch.Tensor) -> torch.Tensor:
        last = torch.cat([sem_rows, acoustic], dim=-1)[:, :, None]
        emb = torch.cat([head.expand(sem_rows.shape[0], -1, -1), last], dim=-1)
        waveforms = model._forward_decoder(emb)[:, 0]
        return spectral_loss(waveforms[:, -span:], piece, sample_rate, mel_banks)

    return evaluate


def sweep_acoustics(evaluate, rows: torch.Tensor, levels: torch.Tensor, n_levels: int, *,
                    sweeps: int) -> tuple[torch.Tensor, torch.Tensor]:
    """Fit each candidate's acoustic dims by exact coordinate descent.

    Gradient descent on these 36 dims plateaus about 400x above the achievable floor:
    the straight-through estimator's forward pass is piecewise constant and its identity
    backward path walks into local minima it cannot leave. Each dimension has only 21
    values though, so the exact minimum along one coordinate is a batch of 21 forward
    passes with no autograd at all, which is both cheaper in memory and free of local
    minima along that axis. Sweeping the 36 in turn converges in a handful of rounds.

    Parameters
    ----------
    evaluate : callable
        Closure from `frame_context`.
    rows : torch.Tensor
        `[B, 256]` semantic codebook rows, one per candidate.
    levels : torch.Tensor
        `[B, 36]` starting integer levels.
    n_levels : int
        Values per acoustic dimension.
    sweeps : int
        Passes over the 36 dimensions.

    Returns
    -------
    tuple[torch.Tensor, torch.Tensor]
        `[B]` losses and `[B, 36]` integer levels.
    """
    batch, acoustic_dim = levels.shape
    grid = torch.arange(n_levels, device=levels.device)

    def to_values(x: torch.Tensor) -> torch.Tensor:
        return x.float() * 2.0 / (n_levels - 1) - 1.0

    wide_rows = rows[:, None, :].expand(batch, n_levels, rows.shape[1]).reshape(-1, rows.shape[1])
    with torch.no_grad():
        for _ in range(sweeps):
            for dim in range(acoustic_dim):
                candidates = levels[:, None, :].repeat(1, n_levels, 1)
                candidates[:, :, dim] = grid
                losses = evaluate(wide_rows, to_values(candidates.reshape(-1, acoustic_dim)))
                levels[:, dim] = losses.view(batch, n_levels).argmin(dim=1)
        return evaluate(rows, to_values(levels)), levels


def fit_acoustics(evaluate, rows: torch.Tensor, carried: torch.Tensor, n_levels: int, *,
                  steps: int, lr: float) -> tuple[torch.Tensor, torch.Tensor]:
    """Fit each candidate's 36 acoustic dims, then score it on the rounded result.

    Rounding lives in the forward pass (the codec's own straight-through trick), so the
    loss returned is the one the real code grid produces and candidates are compared on
    equal terms, never on a continuous relaxation that would flatter some of them.

    Parameters
    ----------
    evaluate : callable
        Closure from `frame_context`.
    rows : torch.Tensor
        `[B, 256]` semantic codebook rows, one per candidate.
    carried : torch.Tensor
        `[36]` starting acoustic values, in the decoder's [-1, 1] scale.
    n_levels : int
        Acoustic levels per dimension.
    steps : int
        Adam steps.
    lr : float
        Adam rate.

    Returns
    -------
    tuple[torch.Tensor, torch.Tensor]
        `[B]` hard losses and `[B, 36]` integer levels.
    """
    def to_values(levels: torch.Tensor) -> torch.Tensor:
        return levels.float() * 2.0 / (n_levels - 1) - 1.0

    free = torch.nn.Parameter(torch.atanh(carried.clamp(-0.999, 0.999))[None].repeat(rows.shape[0], 1))
    optimizer = torch.optim.Adam([free], lr=lr)
    for _ in range(steps):
        scaled = (torch.tanh(free) + 1.0) / 2.0 * (n_levels - 1)
        quantized = scaled + (scaled.round() - scaled).detach()
        loss = evaluate(rows, to_values(quantized)).sum()
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
    with torch.no_grad():
        levels = ((torch.tanh(free) + 1.0) / 2.0 * (n_levels - 1)).round()
        return evaluate(rows, to_values(levels)), levels.long()


def fit_frame_restarts(evaluate, table: torch.Tensor, n_levels: int, acoustic_dim: int, *,
                       restarts: int, steps: int, lr_semantic: float, lr_acoustic: float,
                       chunk: int, generator: torch.Generator) -> tuple[float, int, torch.Tensor]:
    """Solve one frame by many parallel continuous restarts.

    Everything measured in B3 points here. Continuous descent on a frame's 292 dims
    converges perfectly when it starts near the answer (a control seeded at the truth
    plus 5 percent noise recovers 100 percent of the semantic codes), and fails from a
    cold start only because the basin is narrow. The shortlist route fails for a
    different reason: judging 8192 candidates requires fitting each one's acoustics, and
    neither Adam nor exact coordinate descent gets that fit within 400x of the floor.

    So instead of one descent from one place, run hundreds from hundreds of places at
    once. Each restart is 292 numbers and a single frame of audio, which is small enough
    that the batch dimension is free on a GPU, and the causal decoder means this has to
    be solved for one frame only before moving on.

    Parameters
    ----------
    evaluate : callable
        Closure from `frame_context`, differentiable in both of its arguments.
    table : torch.Tensor
        `[8192, 256]` semantic codebook, also the pool the restarts are drawn from.
    n_levels : int
        Values per acoustic dimension.
    acoustic_dim : int
        Number of acoustic dimensions.
    restarts : int
        Independent descents run in parallel.
    steps : int
        Adam steps, shared by all restarts.
    lr_semantic, lr_acoustic : float
        Adam rates for the two halves, two orders of magnitude apart by necessity.
    chunk : int
        Restarts per decoder pass, bounded by GPU memory.
    generator : torch.Generator
        Source of the starting rows.

    Returns
    -------
    tuple[float, int, torch.Tensor]
        Best loss, its semantic code, and its `[36]` acoustic levels.
    """
    device = table.device
    best = (float("inf"), None, None)
    for start in range(0, restarts, chunk):
        size = min(chunk, restarts - start)
        rows = torch.randint(table.shape[0], (size,), generator=generator).to(device)
        sem = torch.nn.Parameter(table[rows].clone())
        aco = torch.nn.Parameter(torch.zeros(size, acoustic_dim, device=device))
        optimizer = torch.optim.Adam([
            {"params": [sem], "lr": lr_semantic},
            {"params": [aco], "lr": lr_acoustic},
        ])
        for _ in range(steps):
            # VQ-VAE straight-through on both halves: the forward pass always uses real
            # codes, so the loss being minimised is the one the code grid can deliver.
            nearest = table[torch.cdist(sem, table).argmin(dim=-1)]
            sem_ste = sem + (nearest - sem).detach()
            scaled = (torch.tanh(aco) + 1.0) / 2.0 * (n_levels - 1)
            aco_ste = (scaled + (scaled.round() - scaled).detach()) * 2.0 / (n_levels - 1) - 1.0
            loss = evaluate(sem_ste, aco_ste).sum()
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
        with torch.no_grad():
            codes = torch.cdist(sem, table).argmin(dim=-1)
            levels = ((torch.tanh(aco) + 1.0) / 2.0 * (n_levels - 1)).round()
            losses = evaluate(table[codes], levels * 2.0 / (n_levels - 1) - 1.0)
            local = losses.argmin()
            if losses[local].item() < best[0]:
                best = (losses[local].item(), codes[local].item(), levels[local].long())
    return best


def solve_causally(model: torch.nn.Module, target: torch.Tensor, frames: int, *, shortlist: int,
                   fit_steps: int, chunk: int, window: int, lr: float, restarts: int = 0,
                   lr_semantic: float = 1.0,
                   truth: torch.Tensor | None = None) -> torch.Tensor:
    """Recover codes frame by frame, fitting each candidate's acoustics before ranking.

    This is the algorithm B3's measurements point at. Three facts shape it:

    1. The decoder is causal, so the audio of frames 0..k does not depend on anything
       after k. Each frame can be finished before the next starts and never revisited.
    2. Given a frame's 36 acoustic dims, its semantic code is the unique global minimum
       of the audio loss by a factor of ~400. Exhaustive over 8192 rows, no search needed.
    3. Given nothing, it is not: holding the acoustics at a neutral value drops the true
       code to rank 241 of 8192. So candidates must be judged *after* their acoustics are
       fitted, not before, which is the mistake the first solver made.

    Fitting 8192 candidates per frame is the expensive part, so it happens in two passes:
    one cheap gradient-free sweep over all 8192 with the acoustics carried over from the
    previous frame, used only to shortlist, then a real fit of the shortlist. The
    shortlist has to be generous precisely because pass one is known to misrank the truth
    by several hundred places.

    Parameters
    ----------
    model : torch.nn.Module
        The audio tokenizer, used through `_forward_decoder` only.
    target : torch.Tensor
        Target waveform, on the same device as the model.
    frames : int
        Number of frames to solve.
    shortlist : int
        Semantic candidates kept from the cheap sweep and actually fitted.
    fit_steps : int
        Adam steps spent fitting each candidate's 36 acoustic dims.
    chunk : int
        Candidates per decoder pass. The binding constraint is GPU memory during the
        fitting pass, where activations are kept for the backward.
    window : int
        Frames of prefix to decode when scoring, 0 for the whole prefix. A window keeps
        the cost per frame constant instead of linear in k; it truncates the causal
        convolution history, so keep it comfortably above the decoder's 16-frame
        attention span.
    lr : float
        Adam rate for the acoustic dims, whose grid step is 0.1 in [-1, 1].
    restarts : int
        When non-zero, solve each frame by that many parallel continuous restarts
        instead of by shortlist-then-fit. This is the strategy that works; the
        shortlist path is kept because it is what the measurements were made with.
    lr_semantic : float
        Adam rate for the semantic half under the restart strategy.
    truth : torch.Tensor, optional
        Ground-truth codes, used for logging only: it reports where the true code landed
        in the cheap sweep, the one number that says whether the shortlist is the cap.

    Returns
    -------
    torch.Tensor
        `[K, frames]` recovered codes.
    """
    device = target.device
    sample_rate = int(model.sampling_rate)
    mel_banks = {n_fft: mel_filterbank(n_fft, N_MELS, sample_rate).to(device)
                 for n_fft, _ in STFT_SCALES}
    table = model.quantizer.semantic_codebook.embedding.float().to(device)
    n_levels = model.quantizer.acoustic_codebook.n_levels
    acoustic_dim = model.quantizer.acoustic_codebook.dim
    samples_per_frame = target.numel() // frames

    codes = torch.zeros(1 + acoustic_dim, frames, dtype=torch.long, device=device)
    codes[1:] = (n_levels - 1) // 2
    # Running acoustic estimate, carried into the next frame as its starting point:
    # adjacent frames of one voice have similar texture, so this is a far better guess
    # than the neutral middle and it lifts the cheap sweep's ranking with it.
    carried = torch.zeros(acoustic_dim, device=device)

    generator = torch.Generator().manual_seed(0)
    for frame in range(frames):
        evaluate = frame_context(model, codes, target, frame, window=window,
                                 mel_banks=mel_banks, samples_per_frame=samples_per_frame)

        if restarts:
            best = fit_frame_restarts(evaluate, table, n_levels, acoustic_dim,
                                      restarts=restarts, steps=fit_steps,
                                      lr_semantic=lr_semantic, lr_acoustic=lr,
                                      chunk=chunk, generator=generator)
            codes[0, frame] = best[1]
            codes[1:, frame] = best[2]
            carried = best[2].float() * 2.0 / (n_levels - 1) - 1.0
            extra = ""
            if truth is not None:
                extra = f", truth {truth[0, frame].item()}"
            logger.info(f"frame {frame + 1}/{frames}: semantic {best[1]}, "
                        f"loss {best[0]:.5f}{extra}")
            continue

        sweep = []
        with torch.no_grad():
            for start in range(0, table.shape[0], chunk):
                rows = table[start:start + chunk]
                sweep.append(evaluate(rows, carried[None].expand(rows.shape[0], -1)))
        order = torch.cat(sweep).argsort()
        candidates = order[:shortlist]
        if truth is not None:
            rank = (order == truth[0, frame]).nonzero().item() + 1
            logger.debug(f"frame {frame + 1}/{frames}: true code ranks {rank} in the cheap sweep")

        best = (float("inf"), None, None)
        for start in range(0, candidates.numel(), chunk):
            picked = candidates[start:start + chunk]
            losses, levels = fit_acoustics(evaluate, table[picked], carried, n_levels,
                                           steps=fit_steps, lr=lr)
            local = losses.argmin()
            if losses[local].item() < best[0]:
                best = (losses[local].item(), picked[local].item(), levels[local])

        codes[0, frame] = best[1]
        codes[1:, frame] = best[2]
        carried = best[2].float() * 2.0 / (n_levels - 1) - 1.0
        logger.info(f"frame {frame + 1}/{frames}: semantic {best[1]}, loss {best[0]:.5f}")
    return codes


@cli.command()
@click.option("--voice", default="fr_female", show_default=True, help="Voice to invert.")
@click.option("--frames", default=8, show_default=True, help="Clip length in frames.")
@click.option("--shortlist", default=512, show_default=True,
              help="Semantic candidates kept from the cheap sweep and actually fitted.")
@click.option("--fit-steps", default=60, show_default=True, help="Adam steps per candidate's acoustics.")
@click.option("--chunk", default=128, show_default=True, help="Candidates per decoder pass.")
@click.option("--window", default=16, show_default=True, help="Prefix frames to decode (0 = all).")
@click.option("--lr", default=0.05, show_default=True, help="Adam rate for the acoustic dims.")
@click.option("--restarts", default=512, show_default=True,
              help="Parallel continuous restarts per frame; 0 uses the shortlist strategy.")
@click.option("--lr-semantic", default=1.0, show_default=True, help="Adam rate for the semantic half.")
@click.option("--device", default="cuda", show_default=True, help="Where to run; cpu is far slower.")
@click.option("--out-dir", default=str(OUT_DIR), show_default=True, help="Where to write wavs and metrics.")
def solve(voice: str, frames: int, shortlist: int, fit_steps: int, chunk: int, window: int, lr: float,
          restarts: int, lr_semantic: float, device: str, out_dir: str) -> None:
    """Recover codes by causal exhaustive search, and score them against the truth."""
    model = load_decoder(find_snapshot(), device=device)
    sample_rate = int(model.sampling_rate)
    truth, target = build_targets(model, voice, frames)
    codes = solve_causally(model, target, frames, shortlist=shortlist, fit_steps=fit_steps,
                           chunk=chunk, window=window, lr=lr, restarts=restarts,
                           lr_semantic=lr_semantic, truth=truth)
    codes, truth = codes.cpu(), truth.cpu()
    metrics = score(codes, truth, model.quantizer.acoustic_codebook.n_levels)

    destination = Path(out_dir)
    with torch.no_grad():
        waveform = model.decode(codes[None].to(device))[0, 0].cpu()
    write_wav(destination / f"{voice}_solve.wav", waveform, sample_rate)
    np.save(destination / f"{voice}_solve_codes.npy", codes.numpy().astype(np.int16))
    (destination / f"{voice}_solve_metrics.json").write_text(json.dumps(metrics, indent=2))
    logger.info(f"semantic exact {metrics['semantic_exact_pct']:6.2f}%  "
                f"acoustic exact {metrics['acoustic_exact_pct']:6.2f}%  "
                f"within1 {metrics['acoustic_within1_pct']:6.2f}%  "
                f"mean|d| {metrics['acoustic_mean_abs_error']:.2f}")
    logger.success(f"wrote {destination}")


@cli.command()
@click.option("--voice", default="fr_female", show_default=True, help="Voice to probe.")
@click.option("--frames", default=8, show_default=True, help="Clip length in frames.")
@click.option("--frame", "frame_index", default=0, show_default=True, help="Frame to solve.")
@click.option("--codes", "code_list", required=True,
              help="Comma-separated semantic codes to fit and compare.")
@click.option("--fit-steps", default=800, show_default=True, help="Adam steps per candidate.")
@click.option("--lr", default=0.05, show_default=True, help="Adam rate for the acoustic dims.")
@click.option("--sweeps", default=0, show_default=True,
              help="Coordinate-descent sweeps instead of Adam, when non-zero.")
@click.option("--window", default=16, show_default=True, help="Prefix frames to decode (0 = all).")
@click.option("--device", default="cuda", show_default=True, help="Where to run.")
def duel(voice: str, frames: int, frame_index: int, code_list: str, fit_steps: int, lr: float,
         sweeps: int, window: int, device: str) -> None:
    """Fit the acoustics of a handful of named semantic codes and compare the results.

    The cheap way to test the premise the solver rests on: that a candidate only loses
    to the truth once its own acoustics have been fitted properly. Give it the true code
    and whatever impostor beat it, spend a serious step budget on both, and see which
    ends up lower. If the impostor still wins, the premise is wrong and no shortlist
    size or compute budget rescues the approach.
    """
    model = load_decoder(find_snapshot(), device=device)
    truth, target = build_targets(model, voice, frames)
    mel_banks = {n_fft: mel_filterbank(n_fft, N_MELS, int(model.sampling_rate)).to(device)
                 for n_fft, _ in STFT_SCALES}
    n_levels = model.quantizer.acoustic_codebook.n_levels
    table = model.quantizer.semantic_codebook.embedding.float().to(device)

    # Frames before the probed one are held at the truth, so the only unknown is this
    # frame: the question is about one frame's two halves, not about error accumulation.
    codes = truth.clone()
    codes[1:, frame_index] = (n_levels - 1) // 2
    evaluate = frame_context(model, codes, target, frame_index, window=window,
                             mel_banks=mel_banks,
                             samples_per_frame=target.numel() // frames)

    # The floor: the true code with its true acoustics. Any candidate's fitted loss has
    # to be read against this, because a fit that stops far above it has not converged
    # and the comparison between candidates is then meaningless.
    true_code = truth[0, frame_index]
    with torch.no_grad():
        floor = evaluate(table[true_code][None],
                         (truth[1:, frame_index].float() * 2.0 / (n_levels - 1) - 1.0)[None])
    logger.info(f"floor, true code with true acoustics: {floor.item():.6f}")

    wanted = [int(c) for c in code_list.split(",")]
    picked = torch.tensor(wanted, device=device)
    acoustic_dim = model.quantizer.acoustic_codebook.dim
    if sweeps:
        start = torch.full((picked.numel(), acoustic_dim), (n_levels - 1) // 2,
                           dtype=torch.long, device=device)
        losses, levels = sweep_acoustics(evaluate, table[picked], start, n_levels, sweeps=sweeps)
    else:
        carried = torch.zeros(acoustic_dim, device=device)
        losses, levels = fit_acoustics(evaluate, table[picked], carried, n_levels,
                                       steps=fit_steps, lr=lr)
    true_levels = truth[1:, frame_index]
    logger.info(f"frame {frame_index}, true semantic code {true_code.item()}")
    for code, loss, level in zip(wanted, losses.tolist(), levels):
        delta = (level - true_levels).abs().float()
        marker = " <- truth" if code == true_code.item() else ""
        logger.info(f"code {code:5d}  loss {loss:10.6f}  acoustic within1 of truth "
                    f"{(delta <= 1).float().mean() * 100:5.1f}%{marker}")


@cli.command()
@click.option("--voice", default="fr_female", show_default=True, help="Voice to probe.")
@click.option("--frames", default=8, show_default=True, help="Clip length in frames.")
@click.option("--frame", "frame_index", default=None, type=int,
              help="Which frame's semantic code to search (default: the last).")
@click.option("--chunk", default=256, show_default=True, help="Candidates per decoder pass.")
@click.option("--acoustic", "acoustic_mode", type=click.Choice(["truth", "mid"]), default="truth",
              show_default=True,
              help="Hold the probed frame's acoustic dims at the truth, or at the mid level "
                   "a search would start from.")
def brute(voice: str, frames: int, frame_index: int | None, chunk: int, acoustic_mode: str) -> None:
    """Score every one of the 8192 semantic codes for one frame, with the rest at truth.

    This is the identifiability question stripped of any optimizer: if the true semantic
    code is not the argmin of the audio loss when everything else is already correct,
    then no search over waveforms can recover it, and analysis by synthesis can only ever
    produce a prefix that sounds right while carrying different codes. If it *is* the
    argmin, the failure is search difficulty and more compute or a better initialisation
    is the answer.
    """
    model = load_decoder(find_snapshot())
    sample_rate = int(model.sampling_rate)
    mel_banks = {n_fft: mel_filterbank(n_fft, N_MELS, sample_rate) for n_fft, _ in STFT_SCALES}
    truth, target = build_targets(model, voice, frames)
    index = frames - 1 if frame_index is None else frame_index


    with torch.no_grad():
        exact = model.quantizer.decode(truth[None])[0]                # [292, T]
        table = model.quantizer.semantic_codebook.embedding.float()
        if acoustic_mode == "mid":
            # What a search actually faces: the frame's acoustic dims are still unknown
            # when its semantic code is chosen. If the true code stops winning here, no
            # alternation between the two halves can bootstrap itself.
            exact[table.shape[1]:, index] = 0.0
        losses = []
        for start in range(0, table.shape[0], chunk):
            rows = table[start:start + chunk]
            candidates = exact[None].repeat(rows.shape[0], 1, 1)      # [B, 292, T]
            candidates[:, : rows.shape[1], index] = rows
            waveforms = model._forward_decoder(candidates)[:, 0]
            losses.append(spectral_loss(waveforms, target, sample_rate, mel_banks))
            logger.debug(f"scored {start + rows.shape[0]}/{table.shape[0]}")
        losses = torch.cat(losses)

    true_code = truth[0, index].item()
    order = losses.argsort()
    rank = (order == true_code).nonzero().item()
    logger.info(f"frame {index} of {frames}, true semantic code {true_code}")
    logger.info(f"loss at the truth {losses[true_code]:.6f}, best {losses[order[0]]:.6f} "
                f"(code {order[0].item()}), median {losses.median():.6f}")
    logger.info(f"rank of the true code: {rank + 1} of {table.shape[0]}")
    logger.info(f"top 5: {[(c.item(), round(losses[c].item(), 5)) for c in order[:5]]}")


@cli.command()
@click.option("--voice", default="fr_female", show_default=True, help="Voice to probe.")
@click.option("--frames", default=32, show_default=True, help="Leading frames to probe.")
@click.option("--trials", default=20, show_default=True, help="Random perturbations per group.")
@click.option("--seed", default=0, show_default=True, help="Seed for the perturbations.")
def sensitivity(voice: str, frames: int, trials: int, seed: int) -> None:
    """Ask whether the audio even depends on the codes, group by group.

    Inverting a waveform can only recover a code the waveform actually reflects. This
    perturbs the ground-truth codes one group at a time and reports how far the audio
    moves, in the same units the optimizer minimizes. If flipping a semantic code barely
    changes the audio, no amount of optimization will recover it, and a low semantic
    match rate in `run` is a property of the codec rather than of the search.
    """
    generator = torch.Generator().manual_seed(seed)
    model = load_decoder(find_snapshot())
    sample_rate = int(model.sampling_rate)
    mel_banks = {n_fft: mel_filterbank(n_fft, N_MELS, sample_rate) for n_fft, _ in STFT_SCALES}
    truth, target = build_targets(model, voice, frames)
    n_levels = model.quantizer.acoustic_codebook.n_levels

    def distance(codes: torch.Tensor) -> float:
        with torch.no_grad():
            waveform = model.decode(codes[None])[0, 0]
        return spectral_loss(waveform, target, sample_rate, mel_banks).item()

    logger.info(f"loss at the ground truth: {distance(truth):.4f} (zero by construction)")
    for label, row_slice, span in (
        ("semantic, 1 frame", slice(0, 1), 8192),
        ("acoustic +-1, all 36 dims, 1 frame", slice(1, None), 1),
        ("acoustic random, all 36 dims, 1 frame", slice(1, None), n_levels),
    ):
        losses = []
        for _ in range(trials):
            frame = torch.randint(frames, (1,), generator=generator).item()
            perturbed = truth.clone()
            block = perturbed[row_slice, frame]
            if span == 1:
                step = torch.randint(2, block.shape, generator=generator) * 2 - 1
                perturbed[row_slice, frame] = (block + step).clamp(0, n_levels - 1)
            else:
                perturbed[row_slice, frame] = torch.randint(span, block.shape, generator=generator)
            losses.append(distance(perturbed))
        losses_t = torch.tensor(losses)
        logger.info(f"{label:38} loss {losses_t.mean():7.4f} +- {losses_t.std():6.4f} "
                    f"(min {losses_t.min():7.4f})")


if __name__ == "__main__":
    cli()
