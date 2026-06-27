#!/usr/bin/env -S uv run --script
# /// script
# requires-python = ">=3.10"
# dependencies = []
# ///
"""Interactive TADA voice-cloning helper for the crispasr-tts (French) server.

Pairs each <name>.<audio> file under ./voices/ with its <name>.txt transcript,
transcodes the audio to 24 kHz mono WAV, and bakes a TADA reference GGUF using
the upstream encoder. The baked tada-ref-<name>.gguf lands in
./models_tts_ignore_backups/ (mounted at /models in the container), ready to use
as  --voice /models/tada-ref-<name>.gguf  in docker-compose.yml.

It reuses CrispASR's models/convert-tada-ref-to-gguf.py (which wraps
HumeAI/tada); that script is auto-downloaded next to this one if missing. The
heavy encoder deps (torch, transformers, the tada package) are installed on
demand by `uv run --with ...` only when baking, so the menu stays instant.

Requires: uv and ffmpeg/ffprobe on PATH. The encoder downloads HumeAI/tada-codec
weights on first bake; run `huggingface-cli login` first if that repo is gated.

Refs:
  https://github.com/HumeAI/tada
  https://github.com/CrispStrobe/CrispASR/blob/main/docs/tts.md

Generated with help from Claude Code.
"""
from __future__ import annotations

import argparse
import re
import shutil
import subprocess
import sys
import tempfile
import urllib.request
from pathlib import Path

REPO = Path(__file__).resolve().parent
VOICES_DIR = REPO / "voices"
OUT_DIR = REPO / "models_tts_ignore_backups"
CONVERTER = REPO / "convert-tada-ref-to-gguf.py"
CONVERTER_URL = (
    "https://raw.githubusercontent.com/CrispStrobe/CrispASR/main/"
    "models/convert-tada-ref-to-gguf.py"
)
COMPOSE = REPO / "docker-compose.yml"

AUDIO_EXTS = (".ogg", ".wav", ".mp3", ".flac", ".m4a", ".opus", ".webm")
DEFAULT_LANG = "fr"
REC_MAX_SEC = 15.0  # upstream recommends a reference clip <= 15 s

# Encoder deps, injected per-bake via `uv run --with` (kept out of this script's
# own deps so the menu does not wait on a multi-GB torch install).
# hume-tada pins torch>=2.7,<2.8 but leaves torchaudio/torchvision unpinned;
# torchaudio and torchvision MUST match torch's version exactly or you get
# "undefined symbol: torch_library_impl" at import. Pin the matched 2.7.1 set.
# transformers is capped <5 (hume-tada's constraint) to avoid resolver thrash.
UV_WITH = [
    "torch==2.7.1",
    "torchaudio==2.7.1",
    "torchvision==0.22.1",
    "transformers>=4.57.1,<5",
    "gguf",
    "numpy",
    "soundfile",  # load_audio fallback if torchaudio.load ever balks
    "git+https://github.com/HumeAI/tada.git",
]


def die(msg: str, code: int = 1) -> "None":
    print(f"error: {msg}", file=sys.stderr)
    raise SystemExit(code)


def slugify(name: str) -> str:
    return re.sub(r"[^a-z0-9]+", "-", name.strip().lower()).strip("-")


def ensure_converter() -> None:
    if CONVERTER.exists():
        return
    print(f"Fetching converter -> {CONVERTER.name}", file=sys.stderr)
    try:
        with urllib.request.urlopen(CONVERTER_URL, timeout=30) as r:
            CONVERTER.write_bytes(r.read())
    except Exception as e:  # noqa: BLE001
        die(f"could not download {CONVERTER_URL}: {e}")


def ffprobe_duration(path: Path) -> float | None:
    try:
        out = subprocess.run(
            ["ffprobe", "-v", "error", "-show_entries", "format=duration",
             "-of", "default=nw=1:nk=1", str(path)],
            capture_output=True, text=True, check=True,
        ).stdout.strip()
        return float(out)
    except Exception:  # noqa: BLE001
        return None


class Voice:
    def __init__(self, name: str, audio: Path, txt: Path):
        self.name = name
        self.audio = audio
        self.txt = txt
        self.duration = ffprobe_duration(audio)

    @property
    def out_path(self) -> Path:
        return OUT_DIR / f"tada-ref-{slugify(self.name)}.gguf"

    @property
    def container_voice(self) -> str:
        return f"/models/{self.out_path.name}"

    def transcript(self) -> str:
        return self.txt.read_text(encoding="utf-8").strip()


def discover() -> list[Voice]:
    if not VOICES_DIR.exists():
        die(f"{VOICES_DIR} not found (expected a ./voices dir or symlink)")
    voices: list[Voice] = []
    for audio in sorted(VOICES_DIR.iterdir()):
        if audio.suffix.lower() not in AUDIO_EXTS:
            continue
        txt = audio.with_suffix(".txt")
        if txt.exists():
            voices.append(Voice(audio.stem, audio, txt))
    return voices


def show(voices: list[Voice]) -> None:
    print(f"\nVoices in {VOICES_DIR}:\n")
    for i, v in enumerate(voices, 1):
        dur = f"{v.duration:5.1f}s" if v.duration is not None else "  ?  "
        flag = "  (>15s, not recommended)" if (v.duration or 0) > REC_MAX_SEC else "  (ok)"
        done = " [baked]" if v.out_path.exists() else ""
        text = v.transcript()
        preview = (text[:60] + "...") if len(text) > 60 else text
        print(f"  {i}. {v.name:14s} {dur} {len(text):4d} chars{flag}{done}")
        print(f"       {preview!r}")
    print()


def parse_selection(raw: str, n: int) -> list[int]:
    raw = raw.strip().lower()
    if raw in ("a", "all"):
        return list(range(n))
    idx: list[int] = []
    for part in re.split(r"[,\s]+", raw):
        if not part:
            continue
        if not part.isdigit() or not (1 <= int(part) <= n):
            die(f"invalid selection: {part!r}")
        idx.append(int(part) - 1)
    return idx


def bake(v: Voice, language: str, device: str) -> bool:
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    transcript = v.transcript()
    if not transcript:
        print(f"  ! {v.name}: empty transcript, skipping", file=sys.stderr)
        return False

    with tempfile.TemporaryDirectory() as td:
        wav = Path(td) / "ref.wav"
        print(f"\n=== {v.name} ===")
        print(f"  transcode {v.audio.name} -> 24 kHz mono wav")
        try:
            subprocess.run(
                ["ffmpeg", "-hide_banner", "-loglevel", "error", "-y",
                 "-i", str(v.audio), "-ac", "1", "-ar", "24000",
                 "-c:a", "pcm_s16le", str(wav)],
                check=True,
            )
        except (subprocess.CalledProcessError, FileNotFoundError) as e:
            print(f"  ! ffmpeg failed: {e}", file=sys.stderr)
            return False

        cmd = ["uv", "run", "--no-project"]
        for w in UV_WITH:
            cmd += ["--with", w]
        cmd += [
            "python", str(CONVERTER),
            "--audio", str(wav),
            "--transcript", transcript,
            "--output", str(v.out_path),
            "--device", device,
        ]
        if language:
            cmd += ["--language", language]
        print(f"  baking -> {v.out_path}")
        try:
            subprocess.run(cmd, check=True)
        except subprocess.CalledProcessError as e:
            print(f"  ! converter failed (exit {e.returncode})", file=sys.stderr)
            return False
        except FileNotFoundError:
            die("`uv` not found on PATH (needed to run the encoder)")
    print(f"  OK -> {v.out_path}  (use as --voice {v.container_voice})")
    return True


def set_active(v: Voice) -> bool:
    if not COMPOSE.exists():
        print(f"  {COMPOSE} not found; skipping compose update", file=sys.stderr)
        return False
    text = COMPOSE.read_text()
    new, n = re.subn(r"(--voice\s+)/models/\S+?\.gguf", r"\g<1>" + v.container_voice, text)
    if n == 0:
        print("  could not find a `--voice /models/*.gguf` entry; edit compose manually",
              file=sys.stderr)
        return False
    COMPOSE.write_text(new)
    print(f"  docker-compose.yml: set --voice {v.container_voice} ({n} occurrence(s))")
    return True


def ask(prompt: str) -> str:
    if not sys.stdin.isatty():
        die("no TTY for interactive input; pass --name/--all (and --yes) instead")
    try:
        return input(prompt)
    except EOFError:
        return ""


def main() -> int:
    p = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    p.add_argument("--name", action="append", default=[],
                   help="Bake this voice by name (repeatable); skips the menu.")
    p.add_argument("--all", action="store_true", help="Bake every voice; skips the menu.")
    p.add_argument("--language", default=DEFAULT_LANG,
                   help=f"Language code for the encoder (default: {DEFAULT_LANG}).")
    p.add_argument("--device", default="cpu", help="Torch device for the encoder (default: cpu).")
    p.add_argument("--set-active", action="store_true",
                   help="After a single bake, point docker-compose.yml --voice at it.")
    p.add_argument("--yes", action="store_true", help="Assume yes to prompts (non-interactive).")
    args = p.parse_args()

    if not shutil.which("ffmpeg") or not shutil.which("ffprobe"):
        die("ffmpeg/ffprobe required (sudo apt install ffmpeg)")
    ensure_converter()

    voices = discover()
    if not voices:
        die(f"no <name>.<audio> + <name>.txt pairs found in {VOICES_DIR}")
    by_name = {v.name: v for v in voices}

    if args.all:
        selected = voices
    elif args.name:
        selected = []
        for nm in args.name:
            if nm not in by_name:
                die(f"unknown voice {nm!r}; available: {', '.join(by_name)}")
            selected.append(by_name[nm])
    else:
        show(voices)
        raw = ask("Select number(s) (comma-separated), 'a' for all, 'q' to quit: ")
        if raw.strip().lower() in ("q", "quit", ""):
            print("nothing to do.")
            return 0
        selected = [voices[i] for i in parse_selection(raw, len(voices))]

    baked = [v for v in selected if bake(v, args.language, args.device)]
    if not baked:
        die("no references were baked")

    print(f"\nBaked {len(baked)} reference(s):")
    for v in baked:
        print(f"  {v.out_path}  ->  --voice {v.container_voice}")

    # Offer to wire a single baked voice straight into the compose service.
    if len(baked) == 1:
        v = baked[0]
        do_it = args.set_active or args.yes
        if not do_it and sys.stdin.isatty():
            do_it = ask(f"\nSet {v.name!r} as the active TTS voice in docker-compose.yml? [y/N] ")\
                .strip().lower() in ("y", "yes")
        if do_it and set_active(v):
            print("\nRestart the service to apply:")
            print("  sudo docker compose up -d crispasr-tts")
            print(f"  ./tts.py -t \"Bonjour, ceci est ma voix clonee.\" -o {slugify(v.name)}.wav")
            return 0

    print("\nTo use one, set in docker-compose.yml (crispasr-tts):")
    print(f"  - CRISPASR_EXTRA_ARGS=--verbose --voice {baked[0].container_voice}")
    print("then: sudo docker compose up -d crispasr-tts")
    return 0


if __name__ == "__main__":
    sys.exit(main())
