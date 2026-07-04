# https://huggingface.co/mistralai/Voxtral-4B-TTS-2603
# /// script
# requires-python = ">=3.10"
# dependencies = [
#     "httpx",
#     "soundfile",
# ]
# ///
import argparse
import io
import time

import httpx
import soundfile as sf

BASE_URL = "http://localhost:8003/v1"

# Built-in voice embeddings shipped with mistralai/Voxtral-4B-TTS-2603
# (from voice_embedding/*.pt in the HF repo).
AVAILABLE_VOICES = [
    "ar_male",
    "casual_female",
    "casual_male",
    "cheerful_female",
    "de_female",
    "de_male",
    "es_female",
    "es_male",
    "fr_female",
    "fr_male",
    "hi_female",
    "hi_male",
    "it_female",
    "it_male",
    "neutral_female",
    "neutral_male",
    "nl_female",
    "nl_male",
    "pt_female",
    "pt_male",
]


# Per-request response_format containers accepted by /v1/audio/speech.
RESPONSE_FORMATS = ["wav", "flac", "mp3", "aac", "opus", "pcm"]


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", default="Paris is a beautiful city!")
    parser.add_argument(
        "--voice",
        default="casual_male",
        choices=AVAILABLE_VOICES,
        help="Built-in voice embedding name. Choices: " + ", ".join(AVAILABLE_VOICES),
    )
    # NOTE on Voxtral vs the other vllm-omni TTS backends: Voxtral builds its prompt
    # from just (input, voice) or (input, ref_audio). The extra OpenAI/vllm-omni
    # request fields "instructions", "language" and "max_new_tokens" are only honored
    # by the Qwen3-TTS / Fish paths (serving_speech.py _build_tts_params); Voxtral
    # silently ignores them, so no flags are exposed for them here. What Voxtral does
    # honor per request: voice, response_format and speed (speed is post-processing).
    parser.add_argument(
        "--speed",
        type=float,
        default=None,
        help="Playback speed 0.25-4.0 (post-processing, 1.0 = unchanged). Only sent if set.",
    )
    parser.add_argument(
        "--response-format",
        default="wav",
        choices=RESPONSE_FORMATS,
        help="Audio container the server returns.",
    )
    parser.add_argument("--output-path", default="output.wav")
    args = parser.parse_args()

    payload = {
        "input": args.input,
        "model": "mistralai/Voxtral-4B-TTS-2603",
        "response_format": args.response_format,
        "voice": args.voice,
    }
    # Opt-in: only send speed when the user asked for it, so the server keeps its
    # own default otherwise.
    if args.speed is not None:
        payload["speed"] = args.speed

    # Backend round-trip time: request sent until the full response body is
    # read (httpx.post is non-streaming, so the body is fully read here). This
    # is synthesis time only, not decoding or file writing below. Printed
    # before raise_for_status so a slow HTTP error still reports its timing.
    t0 = time.perf_counter()
    response = httpx.post(f"{BASE_URL}/audio/speech", json=payload, timeout=120.0)
    print(f"Backend took {time.perf_counter() - t0:.2f}s (HTTP {response.status_code})")
    response.raise_for_status()

    if args.response_format == "wav":
        # Decode to report what we got, then re-write as WAV.
        audio_array, sr = sf.read(io.BytesIO(response.content), dtype="float32")
        print(f"Got audio: {len(audio_array)} samples at {sr} Hz")
        sf.write(args.output_path, audio_array, sr, format="WAV")
    else:
        # Non-WAV containers (mp3/opus/...) may not be decodable by soundfile, so
        # write the server's bytes straight through.
        with open(args.output_path, "wb") as f:
            f.write(response.content)
        print(f"Wrote {len(response.content)} bytes ({args.response_format})")
    print(f"Saved to {args.output_path}")


if __name__ == "__main__":
    main()
