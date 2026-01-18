"""
Neuphonic TTS HTTP Proxy Server

This module provides a FastAPI-based HTTP server that wraps the Neuphonic TTS model.
It handles model initialization, voice loading, and provides REST endpoints for
text-to-speech generation with streaming audio responses.

Key Features:
    - Automatic model initialization and pre-warming on startup
    - Voice loading from environment configuration
    - Streaming audio generation for low-latency responses
    - Both POST and GET endpoints for flexibility
    - Automatic sample rate conversion (TTS outputs 24kHz, converts to requested rate)
    - Two output modes: streaming PCM chunks or complete WAV files
    - 16-bit PCM audio output

Environment Variables:
    NEUPHONIC_MODEL: Model repo to use (default: "neuphonic/neutts-nano-q4-gguf")
    NEUPOHNIC_VOICE: Voice name to use (default: "dave")
    LOG_LEVEL: Logging level (default: DEBUG)

Endpoints:
    POST /generate: Generate TTS audio from JSON payload
    GET /generate: Generate TTS audio from query parameters

Request Parameters:
    text: The text to convert to speech (required)
    sample_rate: Target sample rate in Hz (default: 16000)
    stream: If true, return streaming PCM chunks; if false, return complete WAV (default: true)

Example Usage:
    # Start the server
    python proxy.py

    # POST request with streaming (default)
    curl -X POST http://localhost:8080/generate \\
        -H "Content-Type: application/json" \\
        -d '{"text": "Hello world", "sample_rate": 16000, "stream": true}' \\
        --output audio.pcm

    # POST request with complete WAV file
    curl -X POST http://localhost:8080/generate \\
        -H "Content-Type: application/json" \\
        -d '{"text": "Hello world", "sample_rate": 16000, "stream": false}' \\
        --output audio.wav

    # GET request
    curl "http://localhost:8080/generate?text=Hello%20world&sample_rate=16000&stream=false" \\
        --output audio.wav
"""

import io
import logging
import os
from pathlib import Path
from typing import Optional

import numpy as np
import torch
import uvicorn
from dotenv import load_dotenv
from fastapi import FastAPI
from fastapi.responses import StreamingResponse
from pydantic import BaseModel
from scipy import signal

from neutts import NeuTTS

# Load environment variables from .env file
load_dotenv()

# Logging configuration
# Custom format includes timestamp with milliseconds, log level, and message
LOG_FORMAT = "%(asctime)s.%(msecs)03d %(levelname)-8s %(message)s"
LOG_DATEFMT = "%Y-%m-%d %H:%M:%S"

# Configure the root logger with custom format
logging.basicConfig(
    level=os.getenv("LOG_LEVEL", logging.INFO),
    format=LOG_FORMAT,
    datefmt=LOG_DATEFMT,
)
logger = logging.getLogger(__name__)

# Initialize FastAPI application
app = FastAPI(title="Neuphonic TTS Proxy")

# Global state: TTS model instance and voice configuration
tts_instance: Optional[NeuTTS] = None  # The TTS model instance
ref_codes: Optional[torch.Tensor] = None  # Voice reference codes (embeddings)
ref_text: Optional[str] = None  # Voice reference text


class GenerateRequest(BaseModel):
    """Request model for the POST /generate endpoint.

    Attributes:
        text: The text to convert to speech
        sample_rate: Target sample rate for output audio (default: 16000 Hz)
        stream: If True, stream audio chunks. If False, return complete WAV file (default: True)
    """

    text: str
    sample_rate: int = 16000
    stream: bool = True


@app.post("/generate")
async def generate_endpoint(request: GenerateRequest):
    """Generate TTS audio from text with optional streaming and sample rate conversion.

    This endpoint accepts a JSON payload with text and returns audio either as
    streaming chunks or as a complete WAV file. The TTS model generates audio at
    24kHz which is resampled to the requested sample rate (default 16kHz).

    Args:
        request: GenerateRequest object containing:
            - text: The text to synthesize
            - sample_rate: Target sample rate (default: 16000)
            - stream: Whether to stream chunks or return complete WAV (default: True)

    Returns:
        StreamingResponse: Audio data as streaming chunks or complete WAV file

    Raises:
        500: If the TTS model is not properly initialized

    Example:
        POST /generate
        {"text": "Hello, this is a test", "sample_rate": 16000, "stream": true}
    """
    global tts_instance, ref_codes, ref_text

    # Validate that the TTS model and voice are loaded
    if not tts_instance or ref_codes is None or ref_text is None:
        return {"error": "TTS not initialized"}, 500

    logger.info(
        f"Generating audio for text: {request.text} "
        f"(sample_rate={request.sample_rate}, stream={request.stream})"
    )

    # TTS model output is always 24kHz
    TTS_SAMPLE_RATE = 24000

    def resample_audio(
        audio: np.ndarray, original_rate: int, target_rate: int
    ) -> np.ndarray:
        """Resample audio from original_rate to target_rate.

        Optimized for Raspberry Pi: Only handles 24kHz -> 16kHz conversion
        using efficient polyphase filtering (resample_poly).

        Args:
            audio: Input audio as numpy array
            original_rate: Original sample rate in Hz (must be 24000)
            target_rate: Target sample rate in Hz (must be 16000)

        Returns:
            Resampled audio as numpy array
        """
        if original_rate == target_rate:
            return audio

        # Only support 24kHz -> 16kHz conversion (3:2 ratio)
        assert (
            original_rate == 24000 and target_rate == 16000
        ), f"Only 24kHz->16kHz conversion supported, got {original_rate}->{target_rate}"

        # Use polyphase resampling for efficiency on Raspberry Pi
        # Converts 24kHz to 16kHz (multiply by 2/3)
        resampled = signal.resample_poly(audio, 2, 3)
        return resampled

    if request.stream:
        # Streaming mode: yield audio chunks as they are generated
        async def audio_stream():
            """Generate and yield audio chunks as they are produced.

            The TTS model generates audio at 24kHz in chunks. Each chunk is
            resampled to the target sample rate and converted to 16-bit PCM.

            Yields:
                bytes: Raw PCM audio data (16-bit signed integers)
            """
            # Get the streaming generator from the TTS model
            gen = tts_instance.infer_stream(
                request.text,
                ref_codes,
                ref_text,
            )

            # Stream each chunk as it's generated
            for chunk in gen:
                # Resample from 24kHz to target sample rate
                resampled = resample_audio(chunk, TTS_SAMPLE_RATE, request.sample_rate)

                # Convert numpy float array to 16-bit PCM audio bytes
                # chunk is normalized float (-1.0 to 1.0), scale to int16 range
                audio_bytes = (resampled * 32767).astype(np.int16).tobytes()
                yield audio_bytes

        # Return streaming response with appropriate headers
        return StreamingResponse(
            audio_stream(),
            media_type="audio/pcm",
            headers={
                "Content-Disposition": "attachment; filename=generated.pcm",
                "X-Sample-Rate": str(request.sample_rate),
            },
        )
    else:
        # Non-streaming mode: generate complete audio and return as WAV file
        import wave

        # Generate all audio chunks
        gen = tts_instance.infer_stream(
            request.text,
            ref_codes,
            ref_text,
        )

        # Collect all chunks
        chunks = []
        for chunk in gen:
            chunks.append(chunk)

        # Concatenate all chunks
        full_audio = np.concatenate(chunks)

        # Resample to target sample rate
        resampled = resample_audio(full_audio, TTS_SAMPLE_RATE, request.sample_rate)

        # Convert to 16-bit PCM
        audio_int16 = (resampled * 32767).astype(np.int16)

        # Create WAV file in memory
        wav_buffer = io.BytesIO()
        with wave.open(wav_buffer, "wb") as wav_file:
            wav_file.setnchannels(1)  # Mono
            wav_file.setsampwidth(2)  # 16-bit
            wav_file.setframerate(request.sample_rate)
            wav_file.writeframes(audio_int16.tobytes())

        # Seek to beginning of buffer
        wav_buffer.seek(0)

        # Return complete WAV file
        return StreamingResponse(
            wav_buffer,
            media_type="audio/wav",
            headers={"Content-Disposition": "attachment; filename=generated.wav"},
        )


@app.get("/generate")
async def generate_get_endpoint(
    text: str, sample_rate: int = 16000, stream: bool = True
):
    """Generate TTS audio from text using GET method.

    This is a convenience endpoint that accepts parameters as query strings
    instead of JSON. It delegates to the POST endpoint internally.

    Args:
        text: The text to convert to speech (query parameter)
        sample_rate: Target sample rate in Hz (default: 16000)
        stream: Whether to stream chunks (true) or return complete WAV (false)

    Returns:
        StreamingResponse: Audio data as streaming chunks or complete WAV file

    Examples:
        GET /generate?text=Hello%20world
        GET /generate?text=Hello&sample_rate=24000&stream=false
    """
    return await generate_endpoint(
        GenerateRequest(text=text, sample_rate=sample_rate, stream=stream)
    )


if __name__ == "__main__":
    """Main entry point for the TTS proxy server.

    This section handles:
    1. Uvicorn logging configuration
    2. TTS model initialization
    3. Voice loading from environment
    4. Model pre-warming
    5. Server startup
    """

    # Configure uvicorn to use our custom logging format
    log_config = uvicorn.config.LOGGING_CONFIG
    for formatter in log_config["formatters"].values():
        formatter["fmt"] = LOG_FORMAT
        formatter["datefmt"] = LOG_DATEFMT

    # Initialize the NeuTTS model from environment variable
    # Uses quantized nano model for faster inference on CPU
    MODEL_REPO = os.getenv("NEUPHONIC_MODEL", "neuphonic/neutts-nano-q4-gguf")
    logger.info(f"Initializing NeuTTS model: {MODEL_REPO}")
    tts_instance = NeuTTS(
        backbone_repo=MODEL_REPO,  # Model from environment or default
        backbone_device="cpu",  # Run on CPU
        codec_repo="neuphonic/neucodec",  # Audio codec model
        codec_device="cpu",
        sample_rate=16_000,  # 16kHz output
    )

    # Load the voice configuration from environment variable
    # Voice files must exist in voices/{VOICE}.txt and voices/{VOICE}.pt
    VOICE = os.getenv("NEUPOHNIC_VOICE", "dave")
    logger.info(f"Loading voice: {VOICE}")

    # Voice files
    voice_dir = Path("voices")
    ref_text_path = voice_dir / f"{VOICE}.txt"
    ref_codes_path = voice_dir / f"{VOICE}.pt"

    # Load voice reference text and embeddings
    ref_text = open(ref_text_path, "r").read().strip()
    ref_codes = torch.load(ref_codes_path)

    logger.info(f"Voice loaded successfully: {VOICE}")

    # Pre-warm the model by generating a sample
    # This ensures the first real request doesn't have cold-start latency
    logger.info("Pre-warming the model...")
    warmup_gen = tts_instance.infer_stream(
        "Warming up the text to speech model.", ref_codes, ref_text
    )
    _ = list(warmup_gen)
    logger.info("Model pre-warmed and ready")

    # Start the FastAPI server
    # Listens on all interfaces (0.0.0.0) on port 8080
    uvicorn.run(app, host="0.0.0.0", port=8080, log_config=log_config)
