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


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", default="Paris is a beautiful city!")
    parser.add_argument(
        "--voice",
        default="casual_male",
        choices=AVAILABLE_VOICES,
        help="Built-in voice embedding name. Choices: " + ", ".join(AVAILABLE_VOICES),
    )
    parser.add_argument("--output-path", default="output.wav")
    args = parser.parse_args()

    payload = {
        "input": args.input,
        "model": "mistralai/Voxtral-4B-TTS-2603",
        "response_format": "wav",
        "voice": args.voice,
    }

    response = httpx.post(f"{BASE_URL}/audio/speech", json=payload, timeout=120.0)
    response.raise_for_status()

    audio_array, sr = sf.read(io.BytesIO(response.content), dtype="float32")
    print(f"Got audio: {len(audio_array)} samples at {sr} Hz")

    sf.write(args.output_path, audio_array, sr, format="WAV")
    print(f"Saved to {args.output_path}")


if __name__ == "__main__":
    main()
