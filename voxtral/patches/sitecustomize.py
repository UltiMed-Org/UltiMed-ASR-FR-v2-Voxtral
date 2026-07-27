"""Per-request noise seeding for Voxtral TTS: reproducible takes with no restart.

Written with Claude Code.

The problem
-----------
Voxtral's audio has exactly one source of randomness: the Gaussian that each
12.5 Hz frame's flow-matching ODE starts from, drawn either as
`torch.randn(B, n_acoustic_codebook)` in `voxtral_tts_audio_generation.py`
(`AcousticTransformer.decode_one_frame`) or as `static_noise.normal_()` in
`cuda_graph_acoustic_transformer_wrapper.py`. Neither call passes a
`torch.Generator`, so both take from the *global* torch RNG. vLLM seeds that
once, at the end of `Worker.compile_or_warm_up_model()`, and never again, so a
take is only reproducible as "request N after a fresh boot, with requests
1..N-1 identical". Nothing in vllm-omni resets it later: `/v1/omni/sleep` and
`/v1/omni/wakeup` only move weights, and `collective_rpc` is not exposed over
HTTP. Hence: reproducing one request means restarting the container.

The request `seed` field cannot fix this. Stage 0 hands vLLM's sampler the
output of `fake_logits_for_audio_tokens()`, a vocab-wide -inf row with a single
finite entry, so every sampling param is arithmetically inert there.

What this patch does
--------------------
It re-pins the global RNG immediately before each frame's noise draw, from a
seed the *request* carries, so the draw depends only on (that seed, which frame
of that request) and not on how much audio the process generated earlier. Same
text plus same seed then gives byte-identical audio at any point in the
container's life.

The hook is `VoxtralTTSForConditionalGeneration.make_omni_output`, the one place
that dispatches to both noise-draw paths (CUDA-graph and eager) and the only
place with the per-request `sampling_extra_args` in hand. Clients pass the seed
as `extra_params: {"voxtral_noise_seed": N}` on `POST /v1/audio/speech`
(`tts.py --noise-seed N`), which vllm-omni merges verbatim into stage 0's
`SamplingParams.extra_args`, the same route per-request `cfg_alpha` already
travels.

This mirrors what vllm-omni does for Qwen3-TTS, which reads
`extra_args["qwen3_tts_request_seed"]` in `vllm_omni/worker/gpu_model_runner.py`
and keeps a per-request `torch.Generator`. Voxtral simply has no such hookup.

Limits, in order of how likely they are to bite
-----------------------------------------------
* **Concurrency weakens it.** One global RNG cannot feed several independent
  streams, and the batch's noise is drawn as one (B, C) tensor, so a row's
  values depend on the whole batch. Reproducibility therefore holds per batch
  composition, which in practice means: send reproducible work one request at a
  time. vllm-omni hits the same wall and answers it by serializing seeded
  batches row by row; that is not reachable from here. A one-time warning is
  logged when a seeded request shares a step with another request.
* **Preemption.** A request preempted and recomputed keeps advancing its frame
  counter, so its stream shifts. Unlikely at concurrency 1.
* **Bitwise-ness is still kernel-dependent.** Identical noise plus identical
  batch shapes gives identical audio; if you need it across *different* batch
  shapes, add `VLLM_BATCH_INVARIANT=1` (slower).
* Unseeded requests are unaffected: with no `voxtral_noise_seed` anywhere in the
  batch the RNG is left completely alone, so default behaviour is preserved.

Wiring
------
`voxtral/entrypoint.sh` prepends this directory to `PYTHONPATH` when
`VOXTRAL_PER_REQUEST_SEED` is truthy; CPython then imports this file in every
Python process, including the `StageEngineCoreProc_stage0` child that vLLM
spawns and that actually owns the model and its RNG (a patch applied in the API
server process would reach a different interpreter and do nothing). The patch
itself is applied lazily, when the model module is first imported, so ordinary
`python -c` helpers in the entrypoint do not drag in torch.
"""

import os
import sys

_ENV_FLAG = "VOXTRAL_PER_REQUEST_SEED"
_TARGET_MODULE = "vllm_omni.model_executor.models.voxtral_tts.voxtral_tts"
_TARGET_CLASS = "VoxtralTTSForConditionalGeneration"
# Request-supplied seed, and our own per-request frame counter kept beside it.
_SEED_KEY = "voxtral_noise_seed"
_STEP_KEY = "_voxtral_noise_step"
_MASK64 = (1 << 64) - 1


def _log(message):
    sys.stderr.write("[voxtral-seed-patch] %s\n" % message)
    sys.stderr.flush()


def _chain_other_sitecustomize():
    """Run any sitecustomize the base image ships, which this file shadows.

    Only one `sitecustomize` module can win an import, and PYTHONPATH puts ours
    first. Silently dropping the image's own startup hook could break the stack
    in ways that look nothing like a seeding bug, so find the next one along
    sys.path and execute it here.
    """
    ours = os.path.dirname(os.path.abspath(__file__))
    for entry in sys.path:
        try:
            candidate = os.path.join(os.path.abspath(entry or "."), "sitecustomize.py")
        except (TypeError, ValueError):
            continue
        if os.path.dirname(candidate) == ours or not os.path.isfile(candidate):
            continue
        try:
            with open(candidate, "rb") as handle:
                code = compile(handle.read(), candidate, "exec")
            exec(code, {"__name__": "sitecustomize_chained", "__file__": candidate})
            _log("chained to pre-existing %s" % candidate)
        except Exception as exc:  # never let this break interpreter startup
            _log("WARNING could not chain %s: %r" % (candidate, exc))
        return


def _mix64(state):
    """SplitMix64 finalizer. Python's hash() is salted per process, so it cannot
    be used here: the same seed has to give the same stream in every worker."""
    state = (state + 0x9E3779B97F4A7C15) & _MASK64
    state = ((state ^ (state >> 30)) * 0xBF58476D1CE4E5B9) & _MASK64
    state = ((state ^ (state >> 27)) * 0x94D049BB133111EB) & _MASK64
    return state ^ (state >> 31)


def _apply_patch(module):
    import torch

    cls = getattr(module, _TARGET_CLASS, None)
    if cls is None:
        _log("WARNING %s has no %s, seeding NOT active" % (_TARGET_MODULE, _TARGET_CLASS))
        return
    original = getattr(cls, "make_omni_output", None)
    if original is None:
        _log("WARNING %s has no make_omni_output, seeding NOT active" % _TARGET_CLASS)
        return
    if getattr(original, "_voxtral_seed_patched", False):
        return

    state = {"warned_batch": False, "warned_missing": False}

    def pin_rng(rows):
        """Re-seed the global torch RNG from the batch's (seed, frame) pairs.

        Returns True when the RNG was touched. Rows are `SamplingParams.extra_args`
        dicts in `input_batch.req_ids` order, i.e. the same order and the same
        objects the model's own `_extract_cfg_alpha` reads cfg_alpha from, which
        is why the frame counter can live inside them and survive across steps.
        """
        if not rows:
            if not state["warned_missing"]:
                state["warned_missing"] = True
                _log(
                    "no sampling_extra_args in this step; if that persists, stage 0's "
                    "default_sampling_params.extra_args is empty and vllm-omni set "
                    "has_sampling_extra_args=False, which drops per-request extras "
                    "(cfg_alpha included). Keep VOXTRAL_CFG_ALPHA set."
                )
            return False
        specs = []
        seeded = 0
        for row in rows:
            seed = row.get(_SEED_KEY) if isinstance(row, dict) else None
            if seed is None:
                # Folded in as a placeholder so a row's position still counts.
                specs.append((0, 0))
                continue
            seeded += 1
            step = int(row.get(_STEP_KEY, 0))
            row[_STEP_KEY] = step + 1
            specs.append((int(seed) & _MASK64, step & _MASK64))
        if not seeded:
            return False
        if len(rows) > 1 and not state["warned_batch"]:
            state["warned_batch"] = True
            _log(
                "WARNING a seeded request shared a step with %d other request(s). One "
                "global RNG cannot give each row its own stream, so takes are only "
                "reproducible for an identical batch composition: send reproducible "
                "work one request at a time." % (len(rows) - 1)
            )
        mixed = 0
        for seed, step in specs:
            mixed = _mix64(mixed ^ seed)
            mixed = _mix64(mixed ^ step)
        torch.manual_seed(mixed)
        return True

    def make_omni_output(self, *args, **kwargs):
        # Stage 1 (audio_tokenizer) shares this class but draws no noise.
        if getattr(self, "model_stage", None) == "audio_generation":
            try:
                pin_rng(kwargs.get("sampling_extra_args"))
            except Exception as exc:
                # Fail open: an unseeded take beats a dead TTS server.
                _log("WARNING seeding skipped for this frame: %r" % exc)
        return original(self, *args, **kwargs)

    make_omni_output._voxtral_seed_patched = True
    make_omni_output._voxtral_seed_original = original
    cls.make_omni_output = make_omni_output
    _log(
        "active in pid %d: requests carrying extra_params.%s are now reproducible "
        "without a restart" % (os.getpid(), _SEED_KEY)
    )


def _install_lazy_hook():
    """Patch the model class right after its module is first imported.

    Importing it here instead would pull torch and vLLM into every Python
    process at startup, including the entrypoint's small pyyaml/hf helpers.
    """
    import importlib.util

    class _PatchingLoader:
        def __init__(self, inner):
            self._inner = inner

        def create_module(self, spec):
            return self._inner.create_module(spec)

        def exec_module(self, module):
            self._inner.exec_module(module)
            try:
                _apply_patch(module)
            except Exception as exc:
                _log("WARNING patch failed, seeding NOT active: %r" % exc)

        def __getattr__(self, name):
            return getattr(self._inner, name)

    class _Finder:
        def find_spec(self, fullname, path=None, target=None):
            if fullname != _TARGET_MODULE:
                return None
            # Step aside so the real finders resolve the module, then wrap the
            # loader they picked rather than reimplementing module lookup.
            sys.meta_path.remove(self)
            try:
                spec = importlib.util.find_spec(fullname)
            except Exception as exc:
                _log("WARNING could not resolve %s: %r" % (fullname, exc))
                return None
            finally:
                sys.meta_path.insert(0, self)
            if spec is None or spec.loader is None:
                return None
            spec.loader = _PatchingLoader(spec.loader)
            return spec

    sys.meta_path.insert(0, _Finder())


_chain_other_sitecustomize()

if os.environ.get(_ENV_FLAG, "").strip().lower() in ("1", "true", "yes", "on"):
    try:
        _install_lazy_hook()
    except Exception as exc:
        _log("WARNING could not install import hook, seeding NOT active: %r" % exc)
