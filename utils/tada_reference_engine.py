#!/usr/bin/env -S uv run --script
# /// script
# requires-python = ">=3.10"
# dependencies = [
#   "click",
#   "loguru",
#   "numpy",
#   "soundfile",
#   "torch==2.7.1",
#   "torchaudio==2.7.1",
#   "torchvision==0.22.1",
#   "transformers>=4.57.1,<5",
#   "hume-tada @ git+https://github.com/HumeAI/tada.git",
# ]
# ///
"""Run HumeAI TADA's *reference* PyTorch engine directly to synthesize one clip.

Why this exists
---------------
The crispasr-tts container runs CrispASR's C++/ggml *re-implementation* of TADA.
When that produces silence (or garbage) it is ambiguous whether the fault is in
CrispASR's port or in the TADA model / our reference + transcript. This script
runs the **upstream** `HumeAI/tada` model end to end on the host, bypassing
CrispASR entirely, with the same voice-cloning inputs and the same inference
knobs the server exposes. So:

  * Reference produces good audio, CrispASR produces silence
        -> the bug is in CrispASR's port (file an issue with this repro).
  * Reference *also* produces silence
        -> the model / reference clip / transcript is the problem, not CrispASR.

It is a deliberate sibling to clone_voice.py (next to it in utils/): that script
bakes a `tada-ref-*.gguf` for the server via the encoder only; this one additionally
runs `TadaForCausalLM.generate(...)` (the part CrispASR re-implements) so we can
A/B the actual synthesis. The heavy torch/torchaudio/torchvision pins are copied
verbatim from clone_voice.py, where they were proven to avoid the
"undefined symbol: torch_library_impl" ABI crash (torchaudio/torchvision must
match torch exactly).

Voice cloning
-------------
TADA clones zero-shot from a short reference clip: encode `--ref-audio` together
with its exact `--ref-text` transcript into a prompt, then generate `--text` in
that voice. For non-English (e.g. French) the transcript is REQUIRED, the
encoder's built-in ASR is English-only, so without it the alignment (and the
audio) degrades. Pick the aligner with `--language`.

The script always prints the prompt token<->audio alignment before generating
(`EncoderOutput.print_alignment`): a wrong alignment, bunched/garbled tokens, is
the most common upstream cause of bad output and is worth ruling out first. After
generation it reports peak / RMS / dBFS and an explicit silence verdict so the
"is it silent?" question is answered numerically, not by ear.

Usage
-----
    ./utils/tada_reference_engine.py \
        --ref-audio fr.wav \
        --ref-text "Le texte exact prononce dans fr.wav." \
        --text "Bonjour, ceci est un test de la voix clonee." \
        --language fr \
        --output tada_ref_out.wav

Defaults mirror upstream `InferenceOptions` (the same values CrispASR's server
uses), including `--num-acoustic-candidates`, which defaults to 1 both here and
on the server now (env `TADA_NUM_CANDIDATES`, default 1 since the drop of the
old =4 override). Raise it on both sides to A/B best-of-N.

Requires: `uv` on PATH. TADA is built on the gated Llama 3.2 tokenizer; either
accept the license + `huggingface-cli login`, or rely on the unsloth-mirror
fallback below. The encoder downloads `HumeAI/tada-codec` on first run.

Generated with help from Claude Code.
"""
from __future__ import annotations

import sys

import click
from loguru import logger


def _install_gated_tokenizer_fallback() -> None:
    """Redirect the gated Llama-3.2 tokenizer to the public unsloth mirror.

    TADA's encoder/model load the Llama-3.2 tokenizer, which is gated on the Hub.
    Without HF auth that raises a 401 / "gated" error. This mirrors the same
    fallback in convert-tada-ref-to-gguf.py (next to it in utils/, kept as a small
    local copy because that file is a standalone script, not an importable module): on a
    gated failure, retry against `unsloth/Llama-3.2-1B`, whose tokenizer is
    identical across the 1B/3B family. NOTE: this duplicates ~10 lines from the
    converter on purpose; if you touch one, touch the other.
    """
    from transformers import AutoTokenizer

    orig = AutoTokenizer.from_pretrained.__func__

    @classmethod
    def patched(cls, name, *args, **kwargs):  # noqa: ANN001
        try:
            return orig(cls, name, *args, **kwargs)
        except Exception as exc:  # noqa: BLE001
            if "gated" in str(exc).lower() or "401" in str(exc):
                logger.warning(
                    "tokenizer {!r} is gated; falling back to unsloth/Llama-3.2-1B",
                    name,
                )
                return orig(cls, "unsloth/Llama-3.2-1B", *args, **kwargs)
            raise

    AutoTokenizer.from_pretrained = patched


def _resolve_dtype(name: str, device: str):
    """Map a --dtype string to a torch dtype, honouring 'auto'."""
    import torch

    if name == "auto":
        # bf16 on GPU (README default, halves the 3B model to ~9 GB); fp32 on CPU
        # because bf16 matmul on CPU is slow and not all ops are covered.
        return torch.bfloat16 if device.startswith("cuda") else torch.float32
    return {"float32": torch.float32, "bfloat16": torch.bfloat16, "float16": torch.float16}[name]


def _load_ref_audio(path: str, device: str):
    """Load a reference clip as a mono (1, T) float tensor at its native rate.

    The TADA encoder resamples to 24 kHz internally, so we only need to hand it
    the native sample rate. We downmix to mono because the encoder iterates over
    the first (channel) dim when inferring per-item lengths; a stereo tensor
    would be misread as a batch of 2.
    """
    import torch

    try:
        import torchaudio

        wav, sr = torchaudio.load(path)
    except Exception as exc:  # noqa: BLE001  (fall back to soundfile for odd codecs)
        logger.debug("torchaudio.load failed ({}); trying soundfile", exc)
        import numpy as np
        import soundfile as sf

        data, sr = sf.read(path, dtype="float32", always_2d=True)  # (T, C)
        wav = torch.from_numpy(np.ascontiguousarray(data.T))  # (C, T)
    if wav.shape[0] > 1:
        wav = wav.mean(dim=0, keepdim=True)
    return wav.to(device), int(sr)


def _report_audio(wav) -> bool:
    """Log duration / peak / RMS / dBFS and return True if essentially silent."""
    import torch

    flat = wav.detach().float().cpu().flatten()
    n = flat.numel()
    dur = n / 24000.0
    if n == 0:
        logger.error("decoded audio is EMPTY (0 samples) -> nothing was generated")
        return True
    peak = flat.abs().max().item()
    rms = flat.pow(2).mean().sqrt().item()
    dbfs = 20.0 * torch.log10(torch.tensor(max(peak, 1e-12))).item()
    logger.info(
        "audio: {} samples, {:.2f}s @ 24 kHz | peak={:.5f} ({:.1f} dBFS) rms={:.6f}",
        n, dur, peak, dbfs, rms,
    )
    # Thresholds: a real utterance peaks near full scale and has RMS well above
    # 1e-3. Peak < 1e-3 (~ -60 dBFS) is inaudible -> treat as silent.
    silent = peak < 1e-3 or rms < 1e-4
    if silent:
        logger.error("VERDICT: essentially SILENT (peak {:.5f}, rms {:.6f})", peak, rms)
    else:
        logger.success("VERDICT: audible output (peak {:.5f}, rms {:.6f})", peak, rms)
    return silent


# Option groups are ordered: setup -> reference (cloning) -> generation -> sampler.
@click.command(context_settings={"max_content_width": 100})
# --- model / runtime setup ---
@click.option("--model", default="HumeAI/tada-3b-ml", show_default=True,
              help="TADA model repo or local path (3B-ml is the multilingual one).")
@click.option("--codec-repo", default="HumeAI/tada-codec", show_default=True,
              help="Encoder/codec repo (provides the voice-prompt encoder).")
@click.option("--language", default="fr", show_default=True,
              help="Encoder aligner language (ar/ch/de/es/fr/it/ja/pl/pt; empty = English).")
@click.option("--device", default="", help="Torch device (default: cuda if available else cpu).")
@click.option("--dtype", default="auto", show_default=True,
              type=click.Choice(["auto", "float32", "bfloat16", "float16"]),
              help="Model compute dtype (auto = bf16 on GPU, fp32 on CPU).")
@click.option("--seed", type=int, default=None,
              help="Torch RNG seed for reproducible sampling (default: nondeterministic).")
# --- reference clip (voice cloning) ---
@click.option("--ref-audio", type=click.Path(exists=True, dir_okay=False),
              help="Reference voice clip to clone (<=15 s recommended). Required unless "
                   "--load-prompt is given.")
@click.option("--ref-text", default=None,
              help="EXACT transcript of --ref-audio. Required for non-English (the encoder's "
                   "ASR is English-only); strongly recommended always.")
@click.option("--prompt-cache", type=click.Path(dir_okay=False), default=None,
              help="Encode once, reuse: load the prompt from this .pt if it exists, else "
                   "encode --ref-audio and save it here (skips the encoder on later runs).")
@click.option("--load-prompt", type=click.Path(exists=True, dir_okay=False), default=None,
              help="Load a previously saved prompt .pt and skip the encoder entirely.")
# --- generation ---
@click.option("--text", required=True, help="Target text to synthesize in the cloned voice.")
@click.option("--output", "-o", default="tada_ref_out.wav", show_default=True,
              help="Output WAV path (24 kHz mono).")
@click.option("--num-transition-steps", type=int, default=5, show_default=True,
              help="Prompt->generation transition frames trimmed from the prompt tail.")
@click.option("--num-extra-steps", type=int, default=0, show_default=True,
              help="Free-running continuation steps beyond --text (0 = pure TTS).")
@click.option("--normalize-text/--no-normalize-text", default=True, show_default=True,
              help="Apply upstream text normalization to --text.")
@click.option("--system-prompt", default=None, help="Optional system-prompt prose.")
@click.option("--user-turn-prompt", default=None, help="Optional user-turn-prompt prose.")
@click.option("--verbose", is_flag=True, help="Print the decoded prompt from the engine.")
# --- InferenceOptions: text-token sampler (defaults = upstream InferenceOptions) ---
@click.option("--do-sample/--no-do-sample", default=True, show_default=True,
              help="Sample the text head (--no-do-sample = greedy argmax).")
@click.option("--temperature", type=float, default=0.6, show_default=True)
@click.option("--top-k", type=int, default=0, show_default=True, help="0 = disabled.")
@click.option("--top-p", type=float, default=0.9, show_default=True)
@click.option("--repetition-penalty", type=float, default=1.1, show_default=True)
# --- InferenceOptions: acoustic / duration flow-matching head ---
@click.option("--acoustic-cfg-scale", type=float, default=1.6, show_default=True)
@click.option("--duration-cfg-scale", type=float, default=1.0, show_default=True)
@click.option("--cfg-schedule", type=click.Choice(["constant", "linear", "cosine"]),
              default="cosine", show_default=True)
@click.option("--noise-temperature", type=float, default=0.9, show_default=True)
@click.option("--num-flow-matching-steps", type=int, default=10, show_default=True)
@click.option("--time-schedule", type=click.Choice(["uniform", "cosine", "logsnr"]),
              default="logsnr", show_default=True)
@click.option("--num-acoustic-candidates", type=int, default=1, show_default=True,
              help="Per-frame best-of-N duration/acoustic candidates. Upstream default 1; "
                   "the crispasr-tts server also defaults to 1 now (TADA_NUM_CANDIDATES). "
                   "Raise it on both sides to A/B best-of-N.")
@click.option("--scorer", type=click.Choice(["likelihood", "spkr_verification", "duration_median"]),
              default="likelihood", show_default=True,
              help="Candidate ranking when --num-acoustic-candidates > 1.")
def main(
    model, codec_repo, language, device, dtype, seed,
    ref_audio, ref_text, prompt_cache, load_prompt,
    text, output, num_transition_steps, num_extra_steps, normalize_text,
    system_prompt, user_turn_prompt, verbose,
    do_sample, temperature, top_k, top_p, repetition_penalty,
    acoustic_cfg_scale, duration_cfg_scale, cfg_schedule, noise_temperature,
    num_flow_matching_steps, time_schedule, num_acoustic_candidates, scorer,
):
    """Synthesize one clip with the upstream TADA reference engine (voice-cloned)."""
    logger.remove()
    logger.add(sys.stderr, format="<level>{level: <8}</level> | {message}", colorize=True)

    _install_gated_tokenizer_fallback()
    import os

    import torch
    import torchaudio
    from tada.modules.encoder import Encoder, EncoderOutput
    from tada.modules.tada import InferenceOptions, TadaForCausalLM

    device = device or ("cuda" if torch.cuda.is_available() else "cpu")
    torch_dtype = _resolve_dtype(dtype, device)
    if device == "cpu":
        logger.warning("running on CPU; a 3B model will take minutes per clip")
    if seed is not None:
        torch.manual_seed(seed)
        if device.startswith("cuda"):
            torch.cuda.manual_seed_all(seed)
        logger.info("seed = {}", seed)

    logger.info("loading model {} ({}, {})", model, device, torch_dtype)
    tada = TadaForCausalLM.from_pretrained(model, torch_dtype=torch_dtype).to(device).eval()

    # --- build the voice-cloning prompt (encode ref clip, or reuse a cached one) ---
    if load_prompt:
        logger.info("loading cached prompt: {}", load_prompt)
        prompt = EncoderOutput.load(load_prompt, device=device)
    elif prompt_cache and os.path.exists(prompt_cache):
        logger.info("loading cached prompt: {}", prompt_cache)
        prompt = EncoderOutput.load(prompt_cache, device=device)
    else:
        if not ref_audio:
            raise click.UsageError("--ref-audio is required (or pass --load-prompt).")
        if not ref_text:
            # Non-fatal but the single biggest cause of garbled/silent non-English output.
            logger.warning("--ref-text not given; non-English alignment will be poor")
        logger.info("loading encoder {} (language={})", codec_repo, language or "en")
        encoder = Encoder.from_pretrained(
            codec_repo, subfolder="encoder", language=(language or None)
        ).to(device).eval()

        wav, sr = _load_ref_audio(ref_audio, device)
        logger.info("ref audio: {} ({:.2f}s @ {} Hz)", ref_audio, wav.shape[-1] / sr, sr)
        with torch.no_grad():
            prompt = encoder(wav, text=[ref_text or ""], sample_rate=sr)
        if prompt_cache:
            prompt.save(prompt_cache)
            logger.info("saved prompt cache -> {}", prompt_cache)

    # Alignment diagnostic: ruling out a bad reference alignment is step one when
    # output is silent/garbled (bunched or nonsensical tokens => fix --ref-text).
    logger.info("prompt alignment (verify the transcript matches the audio):")
    try:
        prompt.print_alignment(tada.tokenizer)
    except Exception as exc:  # noqa: BLE001
        logger.warning("could not print alignment: {}", exc)

    opts = InferenceOptions(
        text_do_sample=do_sample,
        text_temperature=temperature,
        text_top_k=top_k,
        text_top_p=top_p,
        text_repetition_penalty=repetition_penalty,
        acoustic_cfg_scale=acoustic_cfg_scale,
        duration_cfg_scale=duration_cfg_scale,
        cfg_schedule=cfg_schedule,
        noise_temperature=noise_temperature,
        num_flow_matching_steps=num_flow_matching_steps,
        time_schedule=time_schedule,
        num_acoustic_candidates=num_acoustic_candidates,
        scorer=scorer,
    )
    logger.info("generating {} char(s): {!r}", len(text), text)
    out = tada.generate(
        prompt=prompt,
        text=text,
        num_transition_steps=num_transition_steps,
        num_extra_steps=num_extra_steps,
        system_prompt=system_prompt,
        user_turn_prompt=user_turn_prompt,
        inference_options=opts,
        normalize_text=normalize_text,
        verbose=verbose,
    )

    # What the engine actually "read" back: a mismatch vs --text means the text
    # head hallucinated (a content bug, distinct from a silent-audio bug).
    if out.output_str:
        logger.info("engine output text: {!r}", out.output_str[0])

    wav = out.audio[0]
    if wav is None:
        logger.error("decode FAILED for this clip (generate returned None audio)")
        raise SystemExit(1)
    wav = wav.detach().float().cpu()
    if wav.ndim == 1:
        wav = wav.unsqueeze(0)

    if os.path.dirname(output):
        os.makedirs(os.path.dirname(output), exist_ok=True)
    torchaudio.save(output, wav, 24000)
    logger.info("wrote {}", os.path.abspath(output))

    silent = _report_audio(wav)
    # Exit non-zero on silence so this doubles as a CI/automation check.
    raise SystemExit(2 if silent else 0)


if __name__ == "__main__":
    main()
