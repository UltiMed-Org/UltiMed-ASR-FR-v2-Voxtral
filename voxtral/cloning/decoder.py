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
"""Run Voxtral's audio codec DECODER standalone, on CPU, without vLLM.

Stage B2 of PLAN_VOXTRALTTS_CLONING_TEST.md. The encoder is missing from the open
weights but the decoder ships in full, so codes -> waveform is runnable outside
the server. That is the tool Track B needs: analysis by synthesis means calling
this thousands of times, and doing that through a container and an HTTP API would
be unusable.

Why a shim instead of a rewrite
-------------------------------
`VoxtralTTSAudioTokenizer` lives inside vllm_omni and its module imports vLLM,
mistral_common and transformers, none of which are installable on a host that
just wants to run a 100M-parameter decoder on CPU. Re-implementing the
architecture here would duplicate ~700 lines of load-bearing code that upstream
will keep changing: weight-norm parameterisation, causal conv padding, ALiBi,
sliding-window attention, the FSQ level maths. Every one of those is a place for
a silent numerical divergence.

So we load the REAL source files and stub only what they import:

- a meta-path finder that fabricates empty modules for `vllm.*`,
  `mistral_common.*`, `transformers.*` and every
  `vllm_omni.*` EXCEPT the voxtral_tts model files themselves;
- stub attributes are classes, so they work as base classes, as decorators and
  as annotations, which is all the imported modules do with them;
- torch, numpy and einops are imported for real BEFORE the stubs go in
  (`torch.library` inspects the source file of what it registers and dies on a
  stub if it is imported afterwards).

Nothing in the decode path touches a stub: they exist only to get past import
time. The classes that run are byte for byte the ones the server runs.

The code-space offset that bites
---------------------------------
Two different code spaces are in play and they differ by 2:

- The **quantizer** (`decode` here) uses raw codes: semantic in [0, 8192),
  acoustic levels in [0, 21).
- The **embedding table** (stage B1's `recover_codes.py`) uses those codes
  shifted past the 2 reserved special ids, which is what `encode_waveforms` does
  explicitly (`audio_code + len(AudioSpecialTokens.all_special_tokens())`).

So codes recovered in B1 must be shifted DOWN by 2 before decoding, and the
trailing end-of-audio frame (semantic id 1, acoustic ids 0, i.e. -1 and -2 after
shifting) must be dropped rather than decoded. `codes_from_table_space` does both.

Usage
-----
    from decoder import load_decoder, codes_from_table_space
    dec = load_decoder()
    wav = dec.decode(codes_from_table_space(codes)[None])   # [1, 1, samples]
"""

from __future__ import annotations

import importlib.machinery
import importlib.util
import json
import logging
import struct
import sys
import types
import wave
from pathlib import Path

# Real dependencies first. Installing the stubs before torch is imported breaks
# torch.library's registration (it calls inspect.getsourcefile on a stub).
import numpy as np
import torch
from loguru import logger

REPO_ROOT = Path(__file__).resolve().parents[2]
VLLM_OMNI_SRC = REPO_ROOT / "voxtral" / "vllm-omni"
MODEL_SRC = VLLM_OMNI_SRC / "vllm_omni" / "model_executor" / "models" / "voxtral_tts"
# The real modules we want executed; everything else under vllm_omni is stubbed.
KEEP_PREFIX = "vllm_omni.model_executor.models.voxtral_tts"
# NOT stubbed on purpose: `flash_attn` and `apex`. The model file imports both
# under `try/except ImportError` and picks a real fallback (torch SDPA,
# `torch.nn.RMSNorm`) when they are missing. Stubbing them makes the import
# succeed, so the model happily calls a stub as if it were a kernel and blows up
# deep inside `forward` instead.
STUB_ROOTS = ("vllm", "mistral_common", "transformers", "aenum")
# See the module docstring: quantizer codes are the embedding-table ids minus this.
SPECIAL_TOKEN_OFFSET = 2
AUDIO_TOKENIZER_PREFIX = "audio_tokenizer."

# Weights the open checkpoint does not ship, so `load_state_dict(strict=False)` is
# expected to report them. None of them is on the decode path, which is why decoding
# still works: the encoder half (`input_proj.`, `encoder_blocks.`) is waveform -> codes,
# and `audio_token_embedding.` is only read by `encode_tokens` (codes -> LLM-space
# embeddings, the forward direction of what B1 inverts); the LLM ships its own copy of
# that table as `mm_audio_embeddings.audio_codebook_embeddings.embeddings.weight`.
# Anything else going missing means the loader is wrong, so it is warned about loudly.
NOT_IN_OPEN_WEIGHTS = ("input_proj.", "encoder_blocks.", "audio_token_embedding.")


class _StubMeta(type):
    """Metaclass making stub attributes usable as classes, calls and decorators."""

    def __getattr__(cls, name: str) -> type:
        return _StubMeta(name, (object,), {"__module__": "stub"})

    def __call__(cls, *args, **kwargs):
        # `@something(...)` returns this class, which is then called with the
        # decorated object: hand it straight back so decorators are transparent.
        if len(args) == 1 and not kwargs and (isinstance(args[0], type) or callable(args[0])):
            return args[0]
        return cls


def _stub_module(name: str) -> types.ModuleType:
    """Build an empty module whose every attribute is a fresh stub class."""
    module = types.ModuleType(name)
    module.__path__ = []      # a real list: the import machinery iterates it
    module.__all__ = []
    module.__getattr__ = lambda attr: _StubMeta(attr, (object,), {"__module__": name})
    return module


class _StubLoader:
    def __init__(self, name: str) -> None:
        self.name = name

    def create_module(self, spec):  # noqa: D102 - importlib protocol
        return _stub_module(spec.name)

    def exec_module(self, module) -> None:  # noqa: D102 - importlib protocol
        pass


class _StubFinder:
    """Meta-path finder that fabricates the heavy imports the model files make."""

    def find_spec(self, name: str, path=None, target=None):  # noqa: D102
        root = name.split(".")[0]
        if root in STUB_ROOTS or (root == "vllm_omni" and not name.startswith(KEEP_PREFIX)):
            return importlib.machinery.ModuleSpec(name, _StubLoader(name), is_package=True)
        return None


def _install_shims() -> None:
    """Make `vllm_omni.model_executor.models.voxtral_tts.*` importable alone."""
    if any(isinstance(f, _StubFinder) for f in sys.meta_path):
        return
    sys.meta_path.append(_StubFinder())
    # vllm.logger is the one stub with real behaviour: the model files call
    # init_logger(__name__) at module scope and then log through it.
    vllm_logger = _stub_module("vllm.logger")
    vllm_logger.init_logger = logging.getLogger
    sys.modules["vllm.logger"] = vllm_logger
    # Parent packages registered by hand so vllm_omni/__init__.py (which imports
    # aenum and the whole vLLM stack) never runs.
    for package in ["vllm_omni", "vllm_omni.model_executor", "vllm_omni.model_executor.models",
                    KEEP_PREFIX]:
        module = types.ModuleType(package)
        module.__path__ = []
        sys.modules[package] = module


def _load_model_module(name: str):
    """Execute one voxtral_tts source file under its real dotted name."""
    full_name = f"{KEEP_PREFIX}.{name}"
    if full_name in sys.modules:
        return sys.modules[full_name]
    spec = importlib.util.spec_from_file_location(full_name, MODEL_SRC / f"{name}.py")
    module = importlib.util.module_from_spec(spec)
    sys.modules[full_name] = module
    spec.loader.exec_module(module)
    return module


def import_audio_tokenizer():
    """Import the real `voxtral_tts_audio_tokenizer` module.

    Returns
    -------
    module
        The upstream module, with `VoxtralTTSAudioTokenizer` and friends.
    """
    _install_shims()
    _load_model_module("voxtral_tts_audio_generation")
    return _load_model_module("voxtral_tts_audio_tokenizer")


def find_snapshot() -> Path:
    """Locate the checkpoint snapshot via voices_tool, the one owner of that logic."""
    import subprocess

    out = subprocess.run([str(Path(__file__).resolve().parent / "voices_tool.py"), "paths"],
                         capture_output=True, text=True, check=True).stdout
    for line in out.splitlines():
        parts = line.split()
        if len(parts) == 2 and parts[0] == "snapshot":
            return Path(parts[1])
    raise SystemExit(f"could not parse a snapshot path out of:\n{out}")


def _config_shim(snapshot: Path):
    """Build the minimal object the tokenizer reads its config from.

    It touches `vllm_config.model_config.hf_config` only, for `.audio_config`
    (a dict) and `.text_config.hidden_size`. The mapping from params.json is
    copied from `transformers_utils/parsers/voxtral_tts.py`, which is 3 keys.

    Parameters
    ----------
    snapshot : Path
        Checkpoint snapshot directory.

    Returns
    -------
    types.SimpleNamespace
        Something shaped like a VllmConfig for our purposes.
    """
    params = json.loads((snapshot / "params.json").read_text())
    multimodal = params["multimodal"]
    audio_config = {
        "sampling_rate": multimodal["audio_model_args"]["audio_encoding_args"]["sampling_rate"],
        "codec_args": multimodal["audio_tokenizer_args"],
        "audio_model_args": multimodal["audio_model_args"],
        "speaker_id": multimodal["audio_tokenizer_args"].get("voice", {}),
    }
    hf_config = types.SimpleNamespace(
        audio_config=audio_config,
        text_config=types.SimpleNamespace(hidden_size=params["dim"]),
    )
    return types.SimpleNamespace(model_config=types.SimpleNamespace(hf_config=hf_config))


def read_audio_tokenizer_weights(snapshot: Path) -> dict[str, torch.Tensor]:
    """Read every `audio_tokenizer.*` tensor out of consolidated.safetensors.

    Parsed by hand so only these tensors are read, not the 4B language model
    sharing the file.

    Parameters
    ----------
    snapshot : Path
        Checkpoint snapshot directory.

    Returns
    -------
    dict of str to torch.Tensor
        State dict with the `audio_tokenizer.` prefix stripped.
    """
    dtypes = {"BF16": torch.bfloat16, "F16": torch.float16, "F32": torch.float32}
    path = snapshot / "consolidated.safetensors"
    weights: dict[str, torch.Tensor] = {}
    with open(path, "rb") as handle:
        header_len = struct.unpack("<Q", handle.read(8))[0]
        header = json.loads(handle.read(header_len))
        base = 8 + header_len
        for key, entry in header.items():
            if not key.startswith(AUDIO_TOKENIZER_PREFIX):
                continue
            if entry["dtype"] not in dtypes:
                raise SystemExit(f"unsupported dtype {entry['dtype']} for {key}")
            start, end = entry["data_offsets"]
            handle.seek(base + start)
            raw = handle.read(end - start)
            tensor = torch.frombuffer(bytearray(raw), dtype=dtypes[entry["dtype"]])
            weights[key[len(AUDIO_TOKENIZER_PREFIX):]] = tensor.reshape(entry["shape"])
    return weights


def load_decoder(snapshot: Path | None = None, dtype: torch.dtype = torch.float32):
    """Instantiate the audio tokenizer and load the decoder half of its weights.

    The encoder submodules are built by the constructor and left at their random
    initialisation: the checkpoint simply has no weights for them. That is
    harmless as long as only `decode` is called, and `missing` in the log makes
    it visible rather than silent.

    Parameters
    ----------
    snapshot : Path or None
        Checkpoint snapshot directory; located automatically when None.
    dtype : torch.dtype
        Compute dtype. float32 on CPU: bfloat16 matmuls are slower there and the
        decoder is small enough that precision is free.

    Returns
    -------
    torch.nn.Module
        The tokenizer, in eval mode, on CPU.
    """
    snapshot = snapshot or find_snapshot()
    module = import_audio_tokenizer()
    model = module.VoxtralTTSAudioTokenizer(vllm_config=_config_shim(snapshot), prefix="")

    weights = read_audio_tokenizer_weights(snapshot)
    result = model.load_state_dict(weights, strict=False)
    expected_missing = [k for k in result.missing_keys if k.startswith(NOT_IN_OPEN_WEIGHTS)]
    other_missing = [k for k in result.missing_keys if k not in expected_missing]
    logger.info(f"loaded {len(weights)} tensors; {len(expected_missing)} tensors absent "
                f"(expected: the open weights ship no encoder)")
    if other_missing:
        logger.warning(f"{len(other_missing)} NON-encoder tensors missing, decode may be wrong: "
                       f"{other_missing[:5]}")
    if result.unexpected_keys:
        logger.warning(f"{len(result.unexpected_keys)} unexpected tensors: "
                       f"{result.unexpected_keys[:5]}")
    return model.to(dtype).eval()


def codes_from_table_space(codes: np.ndarray | torch.Tensor) -> torch.Tensor:
    """Convert B1's embedding-table code ids into quantizer code ids.

    Subtracts the 2 reserved special ids and drops any trailing frame that is not
    a real audio frame (the end-of-audio marker every voice ends with, built from
    those special ids).

    Parameters
    ----------
    codes : numpy.ndarray or torch.Tensor
        `[N, 37]` ids in embedding-table space, as written by recover_codes.py.

    Returns
    -------
    torch.Tensor
        `[37, T]` int64 quantizer codes, ready for `decode`.
    """
    tensor = torch.as_tensor(np.asarray(codes), dtype=torch.long)
    shifted = tensor - SPECIAL_TOKEN_OFFSET
    keep = (shifted >= 0).all(dim=1)
    dropped = int((~keep).sum())
    if dropped:
        logger.debug(f"dropping {dropped} non-audio frame(s) (end-of-audio markers)")
    return shifted[keep].T.contiguous()


def write_wav(path: Path, waveform: torch.Tensor, sample_rate: int) -> float:
    """Write a mono waveform to a 16-bit PCM wav.

    Parameters
    ----------
    path : Path
        Destination.
    waveform : torch.Tensor
        1-D float samples in roughly [-1, 1].
    sample_rate : int
        Sample rate in Hz.

    Returns
    -------
    float
        Duration in seconds.
    """
    samples = waveform.detach().float().clamp(-1.0, 1.0).numpy()
    pcm = (samples * 32767.0).astype("<i2")
    path.parent.mkdir(parents=True, exist_ok=True)
    with wave.open(str(path), "wb") as handle:
        handle.setnchannels(1)
        handle.setsampwidth(2)
        handle.setframerate(sample_rate)
        handle.writeframes(pcm.tobytes())
    return len(samples) / sample_rate
