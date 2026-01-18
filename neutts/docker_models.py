import argparse
import os
import sys
from pathlib import Path

import torch
from dotenv import load_dotenv

from neutts import NeuTTS

# Load environment variables
load_dotenv()


def preload_models(
    backbone_repo: str,
    codec_repo: str = "neuphonic/neucodec",
    voice: str = "dave",
) -> bool:
    """Pre-download and cache NeuTTS models and validate voice files."""

    try:

        # NeuTTS object
        tts = NeuTTS(
            backbone_repo=backbone_repo,
            backbone_device="cpu",
            codec_repo=codec_repo,
            codec_device="cpu",
            sample_rate=16_000,
        )

        # Voice files
        voice_dir = Path("voices")
        ref_text_path = voice_dir / f"{voice}.txt"
        ref_codes_path = voice_dir / f"{voice}.pt"

        # Load voice
        ref_text = open(ref_text_path, "r").read().strip()
        ref_codes = torch.load(ref_codes_path)

        # Generate some audio (should trigger model download))
        test_gen = tts.infer_stream("This is a simple test.", ref_codes, ref_text)
        _ = list(test_gen)

        # Done
        return True

    except Exception:
        return False


def main():
    """Main entry point for the pre-loading script."""
    parser = argparse.ArgumentParser(
        description="Pre-load NeuTTS models for Docker build",
    )

    parser.add_argument(
        "--model",
        default=os.getenv("NEUPHONIC_MODEL", "neuphonic/neutts-nano-q4-gguf"),
        help="HuggingFace model repo (default: from NEUPHONIC_MODEL env or neutts-nano-q4-gguf)",
    )

    parser.add_argument(
        "--codec",
        default="neuphonic/neucodec",
        help="HuggingFace codec repo (default: neuphonic/neucodec)",
    )

    parser.add_argument(
        "--voice",
        default=os.getenv("NEUPOHNIC_VOICE", "dave"),
        help="Voice name to validate (default: from NEUPOHNIC_VOICE env or 'dave')",
    )

    args = parser.parse_args()

    # Run the pre-loading
    success = preload_models(
        backbone_repo=args.model,
        codec_repo=args.codec,
        voice=args.voice,
    )

    # Exit with appropriate code
    sys.exit(0 if success else 1)


if __name__ == "__main__":
    main()
