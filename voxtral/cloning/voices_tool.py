#!/usr/bin/env -S uv run --script
# /// script
# requires-python = ">=3.10"
# dependencies = ["click", "loguru"]
# ///
"""Register, list and unregister Voxtral TTS voices in the local HF snapshot.

Stage A1 of PLAN_VOXTRALTTS_CLONING_TEST.md. A Voxtral "voice" is not a speaker
vector: it is a pre-encoded reference clip frozen as a prefix in the LLM's
embedding space, and it is spread over THREE files that must agree, otherwise
the server either refuses the name or crashes when it scatters the prefix:

  1. voice_embedding/<name>.pt  a bfloat16 [N, 3072] tensor (N frames @ 12.5 Hz)
  2. params.json                multimodal.audio_tokenizer_args.voice, a
                                {name: index} dict. Becomes the server's
                                `supported_speakers` set, so a name missing here
                                is rejected with "Invalid speaker".
  3. tekken.json                audio.voice_num_audio_tokens, a {name: N} dict.
                                mistral_common emits exactly N placeholder
                                tokens (id 24) for the voice, and the model
                                scatters the [N, 3072] tensor into them. A wrong
                                N here is a shape mismatch at request time.

This tool keeps those three in sync. It is stdlib + click + loguru only: it
never imports torch, because installing ~2.5 GB of CUDA wheels to copy a file
and patch two JSON dicts is absurd. Tensor MATH (truncation, interpolation,
stage A2) belongs in a separate script that may pay for torch.

HF cache gotcha, the reason for `_materialize`
----------------------------------------------
Files inside a snapshot directory are symlinks into `blobs/`, which is
content-addressed and often read-only. Writing THROUGH the symlink corrupts a
blob that other revisions may share, and makes hf_hub_download's integrity check
disagree with what is on disk. So every file this tool edits is first
materialized: the symlink is replaced by a real copy of its content, then the
copy is edited. The blob stays pristine, and hf_hub_download is happy because it
only checks that the snapshot path exists.

Note voxtral/entrypoint.sh:246 takes the other route for its n_decoding_steps
patch (it writes to os.path.realpath, i.e. into the blob). Both work; this one is
the less destructive of the two, and it survives that entrypoint patch either way
since they touch different keys of params.json.

Edits do NOT survive an HF re-download of the repo. Before trusting a registered
voice long-term, either set HF_HUB_OFFLINE=1 on the service or move to a local
model dir via VOXTRAL_TTS_MODEL (see the TODOs in the plan).

Usage
-----
    ./voices_tool.py list
    ./voices_tool.py add --name zz_test --from-voice fr_female
    ./voices_tool.py add --name alicia --from-file cache/cloning/alicia.pt
    ./voices_tool.py remove --name zz_test
    ./voices_tool.py restore            # put params.json / tekken.json back

The server reads voice embeddings once at model init, so any change needs
`sudo docker compose up -d voxtral-tts` (NOT `restart`, which does not re-read
the environment either).
"""

from __future__ import annotations

import json
import os
import shutil
import sys
import zipfile
from pathlib import Path

import click
from loguru import logger

# Repo root, resolved from this file so the tool works from any cwd.
REPO_ROOT = Path(__file__).resolve().parents[2]

# Where the plan says intermediate artifacts live (already gitignored via cache/).
BACKUP_DIR = REPO_ROOT / "cache" / "cloning" / "backup"

HF_REPO_DIRNAME = "models--mistralai--Voxtral-4B-TTS-2603"

# torch storage class name -> bytes per element. Only what a voice embedding can
# plausibly be; anything else means the file is not what we think it is.
_ITEMSIZE_BY_STORAGE = {
    b"BFloat16Storage": 2,
    b"HalfStorage": 2,
    b"FloatStorage": 4,
}


def find_snapshot() -> Path:
    """Locate the Voxtral checkpoint snapshot directory.

    Honours HF_HOME (the service sets it to /cache/huggingface inside the
    container) and falls back to the repo's ./cache/huggingface, which is what
    the host actually has. When several snapshots are present, refs/main wins.

    Returns
    -------
    Path
        Directory holding params.json, tekken.json and voice_embedding/.

    Raises
    ------
    SystemExit
        If no snapshot can be found, since every command needs one.
    """
    hf_home = Path(os.environ.get("HF_HOME", REPO_ROOT / "cache" / "huggingface"))
    repo_dir = hf_home / "hub" / HF_REPO_DIRNAME
    if not repo_dir.is_dir():
        raise SystemExit(f"Voxtral checkpoint not found under {repo_dir}. Is the model downloaded?")

    ref_main = repo_dir / "refs" / "main"
    if ref_main.is_file():
        snapshot = repo_dir / "snapshots" / ref_main.read_text().strip()
        if snapshot.is_dir():
            return snapshot

    snapshots = sorted((repo_dir / "snapshots").glob("*"))
    if not snapshots:
        raise SystemExit(f"No snapshots under {repo_dir}/snapshots")
    if len(snapshots) > 1:
        logger.warning(f"{len(snapshots)} snapshots present, refs/main missing, using {snapshots[-1].name}")
    return snapshots[-1]


def read_embedding_frames(pt_path: Path, hidden_dim: int) -> int:
    """Return N (the frame count) of a voice embedding, without importing torch.

    A torch .pt is a zip archive: the raw tensor bytes sit in one member
    (``*/data/0``) and the pickle naming its dtype/shape in another
    (``*/data.pkl``). Rather than parse the pickle's shape tuple, which is
    fiddly, this derives N from the byte count: the tensor is always
    [N, hidden_dim], so N = bytes / (hidden_dim * itemsize). The exact-division
    check below is what makes that safe: a file with a different shape or an
    unexpected dtype fails loudly instead of reporting a wrong N (which would
    only show up much later as a shape mismatch inside the server).

    Parameters
    ----------
    pt_path : Path
        The voice embedding file.
    hidden_dim : int
        Model hidden size (3072 for this checkpoint), read from params.json.

    Returns
    -------
    int
        Number of 12.5 Hz frames in the prefix.
    """
    with zipfile.ZipFile(os.path.realpath(pt_path)) as archive:
        pickle_name = next(n for n in archive.namelist() if n.endswith("data.pkl"))
        storage_name = next(n for n in archive.namelist() if n.endswith("data/0"))
        pickle_bytes = archive.read(pickle_name)
        storage_bytes = archive.getinfo(storage_name).file_size

    itemsize = next((size for tag, size in _ITEMSIZE_BY_STORAGE.items() if tag in pickle_bytes), None)
    if itemsize is None:
        raise SystemExit(f"{pt_path.name}: unrecognised storage dtype (expected bfloat16/half/float)")

    row_bytes = hidden_dim * itemsize
    if storage_bytes % row_bytes != 0:
        raise SystemExit(
            f"{pt_path.name}: {storage_bytes} bytes is not a whole number of "
            f"[*, {hidden_dim}] rows at {itemsize} bytes/element; wrong shape or dtype"
        )
    return storage_bytes // row_bytes


def _materialize(path: Path) -> Path:
    """Replace an HF symlink with a real file so it can be edited safely.

    See the module docstring: editing through the symlink would write into the
    shared, content-addressed blob. No-op when the path is already a real file.

    Parameters
    ----------
    path : Path
        File inside the snapshot directory.

    Returns
    -------
    Path
        The same path, now guaranteed to be a regular writable file.
    """
    if path.is_symlink():
        content = Path(os.path.realpath(path)).read_bytes()
        path.unlink()
        path.write_bytes(content)
        logger.debug(f"materialized {path.name} (was a symlink into blobs/)")
    path.chmod(0o644)  # HF blobs, and copies of them, can land read-only
    return path


def _backup_once(path: Path) -> None:
    """Snapshot a config file the first time it is about to be edited.

    Gives `restore` something to put back if a voice registration goes wrong,
    without depending on re-downloading ~9 GB of weights.

    Parameters
    ----------
    path : Path
        File about to be modified.
    """
    BACKUP_DIR.mkdir(parents=True, exist_ok=True)
    backup = BACKUP_DIR / path.name
    if not backup.exists():
        shutil.copyfile(os.path.realpath(path), backup)
        logger.info(f"backed up pristine {path.name} -> {backup}")


def _load_json(path: Path) -> dict:
    """Read a JSON config from the snapshot.

    Parameters
    ----------
    path : Path
        params.json or tekken.json.

    Returns
    -------
    dict
        Parsed content.
    """
    with open(os.path.realpath(path)) as handle:
        return json.load(handle)


def _save_json(path: Path, data: dict) -> None:
    """Write a JSON config back, materializing the symlink first.

    Parameters
    ----------
    path : Path
        params.json or tekken.json.
    data : dict
        Full document to write.
    """
    _backup_once(path)
    _materialize(path)
    with open(path, "w") as handle:
        json.dump(data, handle)


def warn_if_server_will_brick() -> None:
    """Warn when the served vllm_omni would 404 on a locally-added voice.

    Learned the hard way on 2026-09-09: stock `voxtral_tts.py` loads voice
    embeddings by calling `hf_hub_download` for EVERY name in params.json,
    hitting the REMOTE repo. A voice that exists only on disk 404s, model init
    dies, and `restart: unless-stopped` turns that into a silent boot loop.

    The perso fork's patch 0005 resolves the local cache first, which is what
    makes locally-added voices usable at all. (HF_HUB_OFFLINE=1 looks like an
    alternative fix and is not: vLLM cannot resolve a repo-id model offline.)

    This greps the fork source, so it catches an unpatched clone but not a
    stale image built before the patch: `step.sh` builds with --build for that.
    """
    forked = REPO_ROOT / "voxtral/vllm-omni/vllm_omni/model_executor/models/voxtral_tts/voxtral_tts.py"
    if forked.is_file() and "_resolve_voice_embedding" in forked.read_text():
        return
    logger.warning(
        "voxtral/vllm-omni lacks the local-first voice lookup (patch 0005). Without "
        "it the server tries to DOWNLOAD this voice at model init, 404s, and boot-loops. "
        "Re-apply voxtral/patches/*.patch and rebuild before starting."
    )


def _voice_dicts(snapshot: Path) -> tuple[dict, dict, dict, dict]:
    """Load both configs and return them with their voice sub-dicts.

    Parameters
    ----------
    snapshot : Path
        Snapshot directory.

    Returns
    -------
    tuple of dict
        (params document, params voice {name: index} dict,
         tekken document, tekken {name: N} dict).
    """
    params = _load_json(snapshot / "params.json")
    tekken = _load_json(snapshot / "tekken.json")
    # Both keys are created by the checkpoint itself; setdefault only guards
    # against a hand-edited file, it should never fire on a pristine snapshot.
    params_voices = params["multimodal"]["audio_tokenizer_args"].setdefault("voice", {})
    tekken_voices = tekken["audio"].setdefault("voice_num_audio_tokens", {})
    return params, params_voices, tekken, tekken_voices


@click.group(context_settings={"help_option_names": ["-h", "--help"]})
def cli() -> None:
    """Manage the voice prefixes of the local Voxtral TTS checkpoint."""
    logger.remove()
    logger.add(sys.stderr, format="<level>{level: <8}</level> {message}", level="INFO")


@cli.command("paths")
def cmd_paths() -> None:
    """Print the resolved snapshot paths (useful for the other stage scripts)."""
    snapshot = find_snapshot()
    click.echo(f"snapshot        {snapshot}")
    click.echo(f"voice_embedding {snapshot / 'voice_embedding'}")
    click.echo(f"params.json     {snapshot / 'params.json'}")
    click.echo(f"tekken.json     {snapshot / 'tekken.json'}")
    click.echo(f"backups         {BACKUP_DIR}")


@cli.command("list")
def cmd_list() -> None:
    """List every registered voice and check the three files agree.

    Any row flagged MISMATCH or MISSING will misbehave at request time, so this
    is the acceptance test for a freshly added voice as much as an inventory.
    """
    snapshot = find_snapshot()
    params, params_voices, _tekken, tekken_voices = _voice_dicts(snapshot)
    hidden_dim = int(params["dim"])
    frame_rate = float(params["multimodal"]["audio_model_args"]["audio_encoding_args"]["frame_rate"])

    embedding_dir = snapshot / "voice_embedding"
    names = sorted(set(params_voices) | set(tekken_voices) | {p.stem for p in embedding_dir.glob("*.pt")})

    click.echo(f"{'voice':<20} {'idx':>4} {'N(.pt)':>7} {'N(tekken)':>10} {'seconds':>8}  status")
    problems = 0
    for name in names:
        pt_path = embedding_dir / f"{name}.pt"
        pt_frames = read_embedding_frames(pt_path, hidden_dim) if pt_path.exists() else None
        tekken_frames = tekken_voices.get(name)
        index = params_voices.get(name)

        issues = []
        if pt_frames is None:
            issues.append("no .pt")
        if tekken_frames is None:
            issues.append("not in tekken.json")
        if index is None:
            issues.append("not in params.json")
        if pt_frames is not None and tekken_frames is not None and pt_frames != tekken_frames:
            issues.append("MISMATCH .pt vs tekken.json")

        seconds = f"{pt_frames / frame_rate:.2f}" if pt_frames else "-"
        status = "ok" if not issues else "; ".join(issues)
        problems += bool(issues)
        click.echo(
            f"{name:<20} {index if index is not None else '-':>4} "
            f"{pt_frames if pt_frames is not None else '-':>7} "
            f"{tekken_frames if tekken_frames is not None else '-':>10} {seconds:>8}  {status}"
        )

    click.echo(f"\n{len(names)} voice(s), {problems} with problems, hidden_dim={hidden_dim}, {frame_rate} Hz")
    if problems:
        raise SystemExit(1)


def _install_one(*, name: str, source: Path, embedding_dir: Path, hidden_dim: int,
                 params_voices: dict, tekken_voices: dict, force: bool) -> int:
    """Copy one embedding into the snapshot and enter it in both voice dicts.

    Factored out of `add` so that the bulk path (--from-dir) registers voices by
    exactly the same rules as the single one, and so that a bulk run rewrites the
    two JSON files once instead of once per voice. The caller owns loading and
    saving those files.

    Parameters
    ----------
    name : str
        Registration name; lowercased here because the server lowercases.
    source : Path
        The .pt to install.
    embedding_dir : Path
        Snapshot's voice_embedding/ directory.
    hidden_dim : int
        params.json's `dim`, needed to infer N from the file size.
    params_voices : dict
        params.json's {name: index} dict, mutated in place.
    tekken_voices : dict
        tekken.json's {name: N} dict, mutated in place.
    force : bool
        Overwrite an existing registration instead of refusing.

    Returns
    -------
    int
        The frame count N that was registered.
    """
    name = name.lower()
    target = embedding_dir / f"{name}.pt"
    if (target.exists() or name in params_voices or name in tekken_voices) and not force:
        raise SystemExit(f"voice '{name}' already exists; pass --force to overwrite")
    if not source.is_file():
        raise SystemExit(f"source embedding not found: {source}")

    # The frame count N is read from the embedding itself rather than taken on
    # trust, because tekken.json's N is what drives the number of placeholder
    # tokens: deriving it from the tensor is the only way the two cannot drift.
    frames = read_embedding_frames(source, hidden_dim)

    # Copy rather than symlink: a symlink into blobs/ would be silently replaced
    # or removed by a future `huggingface-cli delete-cache` / re-download.
    if target.is_symlink():
        target.unlink()
    shutil.copyfile(source, target)
    target.chmod(0o644)

    # Index values are only identity within the dict, so the next free integer
    # is fine; keeping them contiguous just makes the file readable.
    params_voices[name] = params_voices.get(name, max(params_voices.values(), default=-1) + 1)
    tekken_voices[name] = frames
    return frames


@cli.command("add")
@click.option("--name", default=None, help="Name to register the voice under (lowercase; the server lowercases).")
@click.option("--from-voice", default=None, help="Copy an already registered voice's embedding.")
@click.option("--from-file", default=None, type=click.Path(exists=True, dir_okay=False), help="Install this .pt file.")
@click.option("--from-dir", default=None, type=click.Path(exists=True, file_okay=False),
              help="Register every .pt in this directory, named after its stem (bulk mode).")
@click.option("--force", is_flag=True, help="Overwrite an existing registration.")
def cmd_add(name: str | None, from_voice: str | None, from_file: str | None,
            from_dir: str | None, force: bool) -> None:
    """Install one or many voice embeddings and register them in both config files.

    Bulk mode exists for the A6 candidate pool, which is hundreds of voices: one
    process, one rewrite of params.json and tekken.json, instead of one subprocess
    and two JSON rewrites per voice.
    """
    sources = [bool(from_voice), bool(from_file), bool(from_dir)]
    if sum(sources) != 1:
        raise SystemExit("give exactly one of --from-voice, --from-file or --from-dir")
    if from_dir and name:
        raise SystemExit("--from-dir names each voice after its file stem, so --name makes no sense")
    if not from_dir and not name:
        raise SystemExit("--name is required unless registering in bulk with --from-dir")

    snapshot = find_snapshot()
    params, params_voices, tekken, tekken_voices = _voice_dicts(snapshot)
    hidden_dim = int(params["dim"])
    embedding_dir = snapshot / "voice_embedding"
    frame_rate = float(params["multimodal"]["audio_model_args"]["audio_encoding_args"]["frame_rate"])

    if from_dir:
        files = sorted(Path(from_dir).glob("*.pt"))
        if not files:
            raise SystemExit(f"no .pt files in {from_dir}")
        total = 0
        for path in files:
            total += _install_one(name=path.stem, source=path.resolve(), embedding_dir=embedding_dir,
                                 hidden_dim=hidden_dim, params_voices=params_voices,
                                 tekken_voices=tekken_voices, force=force)
        _save_json(snapshot / "params.json", params)
        _save_json(snapshot / "tekken.json", tekken)
        logger.success(f"registered {len(files)} voices from {from_dir} "
                       f"({total / frame_rate:.1f} s of prefix in total)")
    else:
        source = (Path(os.path.realpath(embedding_dir / f"{from_voice}.pt")) if from_voice
                  else Path(from_file).resolve())
        frames = _install_one(name=name, source=source, embedding_dir=embedding_dir,
                              hidden_dim=hidden_dim, params_voices=params_voices,
                              tekken_voices=tekken_voices, force=force)
        _save_json(snapshot / "params.json", params)
        _save_json(snapshot / "tekken.json", tekken)
        # Report the voice name, not source.name: for --from-voice the path was
        # resolved through the HF symlink, so source.name is an opaque blob hash.
        origin = f"voice '{from_voice}'" if from_voice else source.name
        logger.success(f"registered '{name.lower()}': {frames} frames "
                       f"({frames / frame_rate:.2f} s) from {origin}")

    warn_if_server_will_brick()
    logger.info("apply with: sudo docker compose up -d voxtral-tts   (restart is NOT enough)")


@cli.command("remove")
@click.option("--name", required=True, help="Voice to unregister.")
@click.option("--keep-file", is_flag=True, help="Unregister but leave the .pt on disk.")
def cmd_remove(name: str, keep_file: bool) -> None:
    """Unregister a voice and delete its embedding.

    Refuses to touch anything it did not plausibly add: a voice whose embedding
    is still a symlink into blobs/ is one that shipped with the checkpoint.
    """
    snapshot = find_snapshot()
    params, params_voices, tekken, tekken_voices = _voice_dicts(snapshot)

    name = name.lower()
    pt_path = snapshot / "voice_embedding" / f"{name}.pt"
    if pt_path.is_symlink():
        raise SystemExit(f"'{name}' is a shipped voice (still a blobs/ symlink); refusing to remove it")
    if name not in params_voices and name not in tekken_voices and not pt_path.exists():
        raise SystemExit(f"voice '{name}' is not registered")

    params_voices.pop(name, None)
    tekken_voices.pop(name, None)
    _save_json(snapshot / "params.json", params)
    _save_json(snapshot / "tekken.json", tekken)

    if pt_path.exists() and not keep_file:
        pt_path.unlink()

    logger.success(f"unregistered '{name}'")
    logger.info("apply with: sudo docker compose up -d voxtral-tts")


@cli.command("restore")
def cmd_restore() -> None:
    """Put the pristine params.json / tekken.json back from the first backup.

    Only the two config files: added .pt embeddings are left alone, so they can
    be re-registered without recomputing anything.
    """
    snapshot = find_snapshot()
    restored = []
    for filename in ("params.json", "tekken.json"):
        backup = BACKUP_DIR / filename
        if not backup.exists():
            continue
        target = snapshot / filename
        _materialize(target)
        shutil.copyfile(backup, target)
        restored.append(filename)

    if not restored:
        raise SystemExit(f"no backups in {BACKUP_DIR}; nothing to restore")
    logger.success(f"restored {', '.join(restored)} from {BACKUP_DIR}")
    logger.warning("this also undoes entrypoint.sh's n_decoding_steps patch; it re-applies at next boot")


if __name__ == "__main__":
    cli()
