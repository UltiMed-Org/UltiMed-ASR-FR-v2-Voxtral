#!/usr/bin/env -S uv run --script
# /// script
# requires-python = ">=3.10"
# dependencies = ["click", "loguru", "torch", "numpy", "einops", "regex"]
# ///
"""Train the encoder the open checkpoint does not ship, on pairs the decoder makes (stage B6).

Why this exists
---------------
B3 established two things that together dictate this stage. Inverting the decoder
recovers codes exactly when the search starts within about one sigma of the answer
(100 percent of semantic codes at 0.3 sigma, 50 percent at 1.0), and no cold start
ever gets there: not a random codebook row, not the codebook centre, not 256
parallel restarts, not fitting each of 8192 candidates before ranking them. So the
missing piece is not a better search, it is an approximate audio -> codes map to
seed one with. That map is exactly the encoder Mistral withheld.

It can simply be trained. The decoder is a deterministic function we hold in full,
so pairs cost nothing: choose codes, decode them, and keep the pair. Crucially the
labels are *chosen rather than discovered*, which is what makes this ordinary
supervised learning instead of the needle hunt B3 failed at. There is no audio-domain
loss and no search anywhere in this file.

What it predicts
----------------
The codes, as classification: one 8192-way softmax for the semantic code and 36
21-way softmaxes for the acoustic levels.

The semantic head is a plain 8192-way linear layer. Replacing it with the codebook's own
geometry (emit a 256-dim point, logits = minus its squared distance to each real row) is
an appealing idea, since neighbouring rows do sound alike, and it was tried and is
**worse**: 4 percent top-16 after 1200 steps against the free layer's 30 percent after
200. The reason is flatness. Standardised rows sit about 22 apart, so the squared
distances that separate two candidates differ by an amount that maps to logits of order
one, the softmax stays near uniform and the gradient stays tiny, whereas a free layer
sets its own scale from the first step.

Regressing the decoder's continuous 292-dim input instead is the obvious first idea and
it was tried first. It trains well, reaching 0.54 sigma from the truth on a held-out
voice, comfortably inside the band where B3 says descent recovers 75 to 100 percent of
codes. It then recovers **zero**. The reason is that a squared-error fit predicts the
mean of everything plausible, and the mean of several codes is not a code: it lands in
the empty space between them, which is precisely where B3 found descent to be helpless
(a start at the codebook's centre also recovers zero, from every distance tried). A
classifier cannot make that mistake, because its answer is always one of the 8192 rows,
and rows are exactly the points descent can work from.

Training data
-------------
Ground-truth codes from B1 are the only *realistic* material and there are 2920 frames
of them, which is nowhere near enough: trained on those alone the network drives its
training loss down and still scores about 10 percent top-16 on a held-out voice, i.e.
it memorises the frames it was shown. But realism is not required here. The decoder is
the same deterministic function at every point of the code space, so codes can simply
be drawn anywhere and the pair is still exact.

Uniformly random codes are the obvious way to exploit that and they are **useless**:
the network reaches 0.2 percent semantic top-1 on the very batch it just trained on,
i.e. it does not learn the training task at all, let alone transfer it. That is not a
capacity problem (the same network fits 32 fixed frames to 100 percent on both heads).
It is that uniform codes put 37 independent dimensions of noise into every frame, and
inverting dense noise is close to inverting a hash: nothing generalises from one draw
to the next. Real speech codes lie on a thin manifold where the mapping is smooth, and
that is the only place the encoder is ever asked to work.

So clips are drawn near the real material rather than anywhere: contiguous runs from
one voice, which carry real temporal structure; frames resampled from the whole pool,
which keep the real marginal statistics; and jittered runs, real frames whose acoustic
levels are nudged by a few steps, which stay speech-like while giving unlimited
distinct labels and are the answer to the 2920-frame shortage. One voice is held out
entirely so the acceptance test is honest.

Usage
-----
    ./train_encoder.py --steps 4000
    ./train_encoder.py --steps 4000 --holdout fr_female
"""

from __future__ import annotations

import sys
from pathlib import Path

import click
import numpy as np
import torch
import torch.nn as nn
from loguru import logger

sys.path.insert(0, str(Path(__file__).resolve().parent))
from decoder import codes_from_table_space, find_snapshot, load_decoder  # noqa: E402

REPO_ROOT = Path(__file__).resolve().parents[2]
CODES_DIR = REPO_ROOT / "cache" / "cloning" / "codes"
OUT_DIR = REPO_ROOT / "cache" / "cloning" / "B6"

# 1920 = 4 * 4 * 4 * 5 * 3 * 2, the decoder's samples per 12.5 Hz frame. The encoder
# walks that back down in the same number of stages.
STRIDES = (4, 4, 4, 5, 3, 2)
SAMPLES_PER_FRAME = 1920
CHANNELS = (32, 64, 128, 256, 384, 512)
CONTEXT_LAYERS = 4


class FrameEncoder(nn.Module):
    """Waveform to per-frame code logits, `[B, samples]` -> semantic and acoustic logits.

    Two halves. The first collapses each frame's 1920 samples to a vector, with the
    frames processed independently so the reduction cannot smear across a frame
    boundary. The second mixes across frames, because a frame's codes are not
    identifiable from its own 80 ms alone: the decoder had 16 frames of attention span
    when it produced that audio, so the encoder is given a comparable view. Nothing here
    is causal, unlike the decoder, since this network only ever analyses complete audio.
    """

    def __init__(self, semantic_codebook: torch.Tensor, acoustic_dim: int = 36,
                 n_levels: int = 21) -> None:
        super().__init__()
        layers: list[nn.Module] = []
        in_channels = 1
        for stride, out_channels in zip(STRIDES, CHANNELS):
            # kernel 2*stride gives neighbouring windows a half-window of overlap, so
            # detail straddling a boundary is not lost. The padding rounds up, which for
            # an odd stride yields one sample too many rather than one too few; `forward`
            # trims back to the exact length the stride implies.
            layers += [
                nn.Conv1d(in_channels, out_channels, kernel_size=2 * stride,
                          stride=stride, padding=(stride + 1) // 2),
                nn.GroupNorm(8, out_channels),
                nn.GELU(),
            ]
            in_channels = out_channels
        self.within_frame = nn.ModuleList(layers)

        context: list[nn.Module] = []
        for _ in range(CONTEXT_LAYERS):
            context += [
                nn.Conv1d(in_channels, in_channels, kernel_size=5, padding=2),
                nn.GroupNorm(8, in_channels),
                nn.GELU(),
            ]
        self.across_frames = nn.Sequential(*context)
        self.acoustic_dim = acoustic_dim
        self.n_levels = n_levels
        self.semantic_head = nn.Conv1d(in_channels, semantic_codebook.shape[0], kernel_size=1)
        self.acoustic_head = nn.Conv1d(in_channels, acoustic_dim * n_levels, kernel_size=1)

    def forward(self, waveform: torch.Tensor, frames: int) -> torch.Tensor:
        """Encode a waveform.

        Parameters
        ----------
        waveform : torch.Tensor
            `[B, frames * 1920]` audio.
        frames : int
            Number of 12.5 Hz frames in the input.

        Returns
        -------
        tuple[torch.Tensor, torch.Tensor]
            `[B, frames, 8192]` semantic logits and `[B, frames, 36, 21]` acoustic logits.
        """
        batch = waveform.shape[0]
        x = waveform.reshape(batch * frames, 1, -1)
        length = x.shape[-1]
        for layer in self.within_frame:
            x = layer(x)
            if isinstance(layer, nn.Conv1d):
                # The padded convolution can overshoot by one; trim to the exact length
                # the stride implies so the 1920 -> 1 reduction stays on the nose.
                length //= layer.stride[0]
                x = x[..., :length]
        x = x.reshape(batch, frames, -1).transpose(1, 2)
        x = self.across_frames(x)
        semantic = self.semantic_head(x).transpose(1, 2)
        acoustic = self.acoustic_head(x).transpose(1, 2)
        return semantic, acoustic.reshape(batch, frames, self.acoustic_dim, self.n_levels)


def load_pool(holdout: str) -> tuple[list[torch.Tensor], torch.Tensor]:
    """Load the ground-truth codes B1 recovered.

    Parameters
    ----------
    holdout : str
        Voice kept out of training entirely.

    Returns
    -------
    tuple[list[torch.Tensor], torch.Tensor]
        Per-voice `[K, T]` code tensors for training, and every training frame
        concatenated as `[K, N]` for the resampling draw.
    """
    voices = []
    for path in sorted(CODES_DIR.glob("*.npy")):
        # The derived voices from A2 are blends and cuts of the real ones; the blends
        # are off the code manifold entirely (B1), so they are not training material.
        if path.stem.startswith("d_") or path.stem == holdout:
            continue
        voices.append(codes_from_table_space(np.load(path)))
    return voices, torch.cat(voices, dim=1)


def sample_codes(voices: list[torch.Tensor], pool: torch.Tensor, batch: int, frames: int,
                 contiguous: float, random: float, jitter: float, jitter_levels: int,
                 semantic_size: int, n_levels: int,
                 generator: torch.Generator) -> torch.Tensor:
    """Draw a batch of code sequences.

    Parameters
    ----------
    voices : list[torch.Tensor]
        Per-voice code tensors, for the contiguous draw.
    pool : torch.Tensor
        `[K, N]` all training frames, for the resampled draw.
    batch : int
        Sequences to draw.
    frames : int
        Length of each.
    contiguous : float
        Fraction drawn as a real contiguous run.
    random : float
        Fraction drawn uniformly from the whole code space. Kept because it is the clean
        way to sample the decoder's whole domain, but measured to teach the network
        nothing, so it defaults to zero: see the module docstring.
    jitter : float
        Fraction drawn as a contiguous run whose acoustic levels are then nudged. Whatever
        is left over after the three is resampled frame-by-frame from the real pool.
    jitter_levels : int
        How far a jittered acoustic level may move, in grid steps.
    semantic_size : int
        Semantic codebook size, the range of the uniform semantic draw.
    n_levels : int
        Acoustic levels per dimension, the range of the uniform acoustic draw.
    generator : torch.Generator
        Randomness, on the pool's device.

    Returns
    -------
    torch.Tensor
        `[batch, K, frames]` codes.
    """
    out = []
    for _ in range(batch):
        # One roll picks the draw kind, so the three fractions partition the batch.
        roll = torch.rand((), generator=generator, device=pool.device)
        if roll < random:
            # The decoder is defined on the whole code space, not just the frames the 20
            # shipped voices happen to visit, so uniform codes are as valid a training
            # pair as a real one and there is an unlimited supply of them.
            semantic = torch.randint(semantic_size, (1, frames), generator=generator,
                                     device=pool.device)
            acoustic = torch.randint(n_levels, (pool.shape[0] - 1, frames), generator=generator,
                                     device=pool.device)
            out.append(torch.cat([semantic, acoustic], dim=0))
            continue
        if roll < random + jitter:
            # A real run, then acoustic levels nudged a step or two. The semantic code is
            # left alone: it selects one of 8192 rows with no ordering, so "nearby" is not
            # defined for it and any change is a jump to unrelated audio.
            start = int(torch.randint(len(voices), (), generator=generator,
                                      device=pool.device))
            voice = voices[start]
            offset = int(torch.randint(max(1, voice.shape[1] - frames), (),
                                       generator=generator, device=pool.device))
            clip = voice[:, offset:offset + frames].clone()
            if clip.shape[1] < frames:
                clip = pool[:, torch.randint(pool.shape[1], (frames,), generator=generator,
                                             device=pool.device)]
            nudge = torch.randint(-jitter_levels, jitter_levels + 1, clip[1:].shape,
                                  generator=generator, device=pool.device)
            clip[1:] = (clip[1:] + nudge).clamp(0, n_levels - 1)
            out.append(clip)
            continue
        if roll < random + jitter + contiguous:
            voice = voices[int(torch.randint(len(voices), (), generator=generator,
                                             device=pool.device))]
            start = int(torch.randint(max(1, voice.shape[1] - frames), (), generator=generator,
                                      device=pool.device))
            piece = voice[:, start:start + frames]
            if piece.shape[1] < frames:                       # short voice, wrap around
                piece = torch.cat([piece, voice[:, :frames - piece.shape[1]]], dim=1)
        else:
            picks = torch.randint(pool.shape[1], (frames,), generator=generator,
                                  device=pool.device)
            piece = pool[:, picks]
        out.append(piece)
    return torch.stack(out)


def measure(model: torch.nn.Module, decoder: torch.nn.Module, codes: torch.Tensor,
            top_k: int = 16) -> dict[str, float]:
    """Score a prediction against the truth it was generated from.

    Returns
    -------
    dict[str, float]
        Top-1 and top-k semantic accuracy and acoustic accuracy. Top-k matters as much
        as top-1: the inversion only needs a shortlist of real codes to start descents
        from, so a frame whose truth is second is nearly as good as one where it is
        first, and a frame where it is nowhere is lost.
    """
    with torch.no_grad():
        waveform = decoder.decode(codes)[0, 0]
        semantic, acoustic = model(waveform[None], codes.shape[-1])
        truth_semantic, truth_acoustic = codes[0, 0], codes[0, 1:]
        ranked = semantic[0].topk(top_k, dim=-1).indices                  # [T, k]
        levels = acoustic[0].argmax(dim=-1).transpose(0, 1)               # [36, T]
        return {
            "semantic_top1_pct": (ranked[:, 0] == truth_semantic).float().mean().item() * 100,
            "semantic_topk_pct": (ranked == truth_semantic[:, None]).any(dim=-1).float().mean().item() * 100,
            "acoustic_exact_pct": (levels == truth_acoustic).float().mean().item() * 100,
            "acoustic_within1_pct": ((levels - truth_acoustic).abs() <= 1).float().mean().item() * 100,
        }


@click.command()
@click.option("--steps", default=4000, show_default=True, help="Adam steps.")
@click.option("--batch", default=16, show_default=True, help="Clips per step.")
@click.option("--frames", default=32, show_default=True, help="Frames per clip.")
@click.option("--buffer", default=4096, show_default=True, help="Clips held in the replay buffer.")
@click.option("--refresh", default=8, show_default=True, help="Clips decoded fresh each step.")
@click.option("--lr", default=3e-4, show_default=True, help="Adam learning rate.")
@click.option("--contiguous", default=0.2, show_default=True,
              help="Fraction of clips drawn as real contiguous runs.")
@click.option("--random", "random_fraction", default=0.0, show_default=True,
              help="Fraction of clips drawn uniformly from the whole code space (measured useless).")
@click.option("--jitter", default=0.5, show_default=True,
              help="Fraction of clips drawn as real runs with nudged acoustic levels.")
@click.option("--jitter-levels", default=2, show_default=True,
              help="How far --jitter may move an acoustic level, in grid steps.")
@click.option("--holdout", default="fr_female", show_default=True, help="Voice excluded from training.")
@click.option("--device", default="cuda", show_default=True, help="Where to run.")
@click.option("--seed", default=0, show_default=True, help="Seed for the data draw.")
@click.option("--out-dir", default=str(OUT_DIR), show_default=True, help="Where to write the weights.")
def main(steps: int, batch: int, frames: int, buffer: int, refresh: int, lr: float,
         contiguous: float, random_fraction: float, jitter: float, jitter_levels: int,
         holdout: str, device: str, seed: int, out_dir: str) -> None:
    """Train the audio -> decoder-input encoder on decoder-generated pairs."""
    decoder = load_decoder(find_snapshot(), device=device)
    for param in decoder.parameters():
        param.requires_grad_(False)

    voices, pool = load_pool(holdout)
    voices = [v.to(device) for v in voices]
    pool = pool.to(device)
    logger.info(f"{len(voices)} training voices, {pool.shape[1]} frames, holding out {holdout}")

    n_levels = decoder.quantizer.acoustic_codebook.n_levels
    semantic_codebook = decoder.quantizer.semantic_codebook.embedding
    semantic_size = semantic_codebook.shape[0]
    model = FrameEncoder(semantic_codebook=semantic_codebook,
                         acoustic_dim=decoder.quantizer.acoustic_codebook.dim,
                         n_levels=n_levels).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    schedule = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=steps)
    generator = torch.Generator(device=device).manual_seed(seed)
    held = codes_from_table_space(np.load(CODES_DIR / f"{holdout}.npy")).to(device)[None]

    # Decoding a fresh batch every step makes the *decoder* the cost of training (0.57 s a
    # step at batch 64, where the encoder's own forward and backward are a fraction of
    # that), which is absurd when the data is free in principle. So clips live in a replay
    # buffer and only `refresh` of them are decoded per step: each clip is then seen about
    # `batch * buffer / (refresh * buffer) = batch / refresh` times over its life, which
    # is far from the repetition that made the real-frames-only run memorise, while the
    # decoder does a fraction of the work.
    buffer_waveform = torch.empty(buffer, frames * SAMPLES_PER_FRAME, dtype=torch.float16,
                                  device=device)
    buffer_codes = torch.empty(buffer, pool.shape[0], frames, dtype=torch.long, device=device)

    def refill(slots: torch.Tensor) -> None:
        """Decode fresh clips into the given buffer slots."""
        fresh = sample_codes(voices, pool, len(slots), frames, contiguous, random_fraction,
                             jitter, jitter_levels, semantic_size, n_levels, generator)
        with torch.no_grad():
            buffer_waveform[slots] = decoder.decode(fresh)[:, 0].half()
        buffer_codes[slots] = fresh

    logger.info(f"filling the {buffer}-clip replay buffer")
    for start in range(0, buffer, batch):
        refill(torch.arange(start, min(start + batch, buffer), device=device))

    for step in range(steps):
        # The refreshed slots march through the buffer rather than being drawn at random,
        # so every clip has the same lifetime and none can linger.
        first = (step * refresh) % buffer
        refill((torch.arange(refresh, device=device) + first) % buffer)
        drawn = torch.randint(buffer, (batch,), generator=generator, device=device)
        waveform, codes = buffer_waveform[drawn].float(), buffer_codes[drawn]
        semantic, acoustic = model(waveform, frames)
        # Cross-entropy on both halves. The acoustic one is a plain 21-way choice per
        # dimension rather than a regression for the same reason as the semantic one:
        # the answer has to be a grid point, not the average of two.
        loss = nn.functional.cross_entropy(semantic.reshape(-1, semantic.shape[-1]),
                                           codes[:, 0].reshape(-1))
        loss = loss + nn.functional.cross_entropy(acoustic.reshape(-1, n_levels),
                                                  codes[:, 1:].transpose(1, 2).reshape(-1))

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        schedule.step()

        if step % 200 == 0 or step == steps - 1:
            model.eval()
            scores = measure(model, decoder, held)
            model.train()
            # The same two accuracies on the batch just trained on. Held-out numbers alone
            # cannot say whether a flat curve means the network is learning nothing or is
            # learning the mostly-random training clips and failing to transfer to speech.
            with torch.no_grad():
                train_top1 = (semantic.argmax(dim=-1) == codes[:, 0]).float().mean().item()
                train_within1 = ((acoustic.argmax(dim=-1) - codes[:, 1:].transpose(1, 2)
                                  ).abs() <= 1).float().mean().item()
            logger.info(f"step {step:5d}  loss {loss.item():7.4f}  train: semantic top1 "
                        f"{100 * train_top1:5.1f}% acoustic within1 {100 * train_within1:5.1f}%")
            logger.info(f"step {step:5d}  loss {loss.item():7.4f}  {holdout}: semantic "
                        f"top1 {scores['semantic_top1_pct']:5.1f}% top16 {scores['semantic_topk_pct']:5.1f}%  "
                        f"acoustic exact {scores['acoustic_exact_pct']:5.1f}% "
                        f"within1 {scores['acoustic_within1_pct']:5.1f}%")

    destination = Path(out_dir)
    destination.mkdir(parents=True, exist_ok=True)
    torch.save({"state_dict": model.state_dict(), "holdout": holdout}, destination / "encoder.pt")
    logger.success(f"wrote {destination / 'encoder.pt'}")


if __name__ == "__main__":
    main()
