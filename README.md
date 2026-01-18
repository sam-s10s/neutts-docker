# NeuTTS Proxy Server

A FastAPI-based HTTP proxy server for the Neuphonic TTS model, optimized for Raspberry Pi and low-powered ARM devices.

## Features

- 🚀 Streaming and non-streaming audio generation
- 🔄 Automatic sample rate conversion (24kHz → 16kHz)
- 📦 Docker support with model pre-loading
- 💨 Optimized for Raspberry Pi / ARM CPUs
- 🎙️ Configurable voices via environment variables
- 📝 Both POST and GET endpoints

## Quick Start

### Local Development

1. Install dependencies:

```bash
cd neutts
uv sync
```

2. Configure environment (`.env`):

```bash
NEUPHONIC_MODEL=neuphonic/neutts-nano-q4-gguf
NEUPOHNIC_VOICE=dave
LOG_LEVEL=INFO
```

3. Run the server:

```bash
cd neutts
uv run proxy.py
```

### Docker Build

The Docker build includes model pre-loading to embed models in the image:

```bash
# Build the image (models are downloaded during build)
docker build -t neutts:latest .

# Run the container
docker run -p 8080:8080 neutts:latest
```

The `preload_models.py` script runs during Docker build to:

- Download and cache TTS models from HuggingFace
- Validate voice files are present
- Perform test inference
- Ensure the container works without internet access

## API Endpoints

### POST /generate

Generate TTS audio from JSON payload.

**Request:**

```bash
curl -X POST http://localhost:8080/generate \
  -H "Content-Type: application/json" \
  -d '{
    "text": "Hello world",
    "sample_rate": 16000,
    "stream": false
  }' \
  --output audio.wav
```

**Parameters:**

- `text` (string, required): Text to synthesize
- `sample_rate` (int, optional): Target sample rate in Hz (default: 16000)
- `stream` (bool, optional): If true, return streaming PCM chunks; if false, return complete WAV (default: true)

### GET /generate

Generate TTS audio from query parameters (convenience endpoint).

**Request:**

```bash
# Streaming PCM chunks
curl "http://localhost:8080/generate?text=Hello%20world&sample_rate=16000&stream=true" \
  --output audio.pcm

# Complete WAV file
curl "http://localhost:8080/generate?text=Hello%20world&stream=false" \
  --output audio.wav
```

## Model Pre-loading

The `docker_models.py` script is designed to run during Docker build to cache models.

### Usage

```bash
# Change to neutts
cd neutts

# Use environment variables from .env
uv run docker_models.py

# Specify custom model
uv run docker_models.py --model neuphonic/neutts-nano-q4-gguf

# Skip voice validation (if voices mounted at runtime)
uv run docker_models.py --no-validate-voice
```

## Environment Variables

| Variable          | Description                                | Default                         |
| ----------------- | ------------------------------------------ | ------------------------------- |
| `NEUPHONIC_MODEL` | HuggingFace model repo                     | `neuphonic/neutts-nano-q4-gguf` |
| `NEUPOHNIC_VOICE` | Voice name (files must exist in `voices/`) | `dave`                          |
| `LOG_LEVEL`       | Logging level                              | `DEBUG`                         |

## Voice Files

Voice files must be present in the `voices/` directory:

```
voices/
  ├── dave.txt       # Reference text
  └── dave.pt        # Pre-encoded voice embeddings
```

To add a new voice:

1. Create `voices/{name}.txt` with reference text
2. Create `voices/{name}.pt` with encoded voice embeddings
3. Set `NEUPOHNIC_VOICE={name}` in `.env`

## Development

### Testing

Test the server locally:

```bash
# Start server
cd neutts
uv run proxy.py

# Test streaming mode
curl -X POST http://localhost:8080/generate \
  -H "Content-Type: application/json" \
  -d '{"text": "Testing streaming", "stream": true}' \
  --output test.pcm

# Test WAV mode
curl -X POST http://localhost:8080/generate \
  -H "Content-Type: application/json" \
  -d '{"text": "Testing WAV", "stream": false}' \
  --output test.wav

# Play the WAV file (macOS)
afplay test.wav
```

## License

See project license for details.
