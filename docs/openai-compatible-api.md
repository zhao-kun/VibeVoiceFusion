# OpenAI-Compatible TTS API Design Document

## Overview

This document describes the design of an OpenAI-compatible Text-to-Speech API endpoint for VibeVoice, allowing existing OpenAI TTS clients and SDKs to use VibeVoice as a drop-in replacement.

## API Comparison: OpenAI TTS vs VibeVoice

### Architecture Difference

```
OpenAI TTS (Synchronous):
  Client → POST /v1/audio/speech (JSON) → Binary audio response

VibeVoice Quick Generate (Asynchronous):
  Client → POST /api/v1/quick-generate (multipart) → JSON {request_id}
  Client → GET  /api/v1/quick-generate/{id} (poll) → JSON {status}
  Client → GET  /api/v1/quick-generate/{id}/download → Binary audio

VibeVoice OpenAI-Compat (Synchronous wrapper):
  Client → POST /v1/audio/speech (JSON) → [blocks until done] → Binary audio
```

### Parameter Comparison

| OpenAI Parameter | Type | VibeVoice Equivalent | Notes |
|-----------------|------|---------------------|-------|
| `model` | string (required) | `model_dtype` | `tts-1`→bf16, `tts-1-hd`→float8 |
| `input` | string (required) | `text` | Max 4096 chars in OpenAI, no limit in VibeVoice |
| `voice` | string (required) | Preset voice name | OpenAI has fixed set; VibeVoice uses preset voices |
| `response_format` | string | N/A (wav only) | Conversion via ffmpeg for other formats |
| `speed` | number | N/A | Not supported by VibeVoice engine |
| `instructions` | string | N/A | GPT-4o-mini-tts only, not applicable |
| N/A | | `seeds` | VibeVoice-specific: random seed |
| N/A | | `cfg_scale` | VibeVoice-specific: classifier-free guidance |
| N/A | | `batch_size` | VibeVoice-specific: multi-generation |
| N/A | | `offloading` | VibeVoice-specific: VRAM optimization |
| N/A | | `voice_file` (upload) | VibeVoice supports custom voice upload |

### Voice Mapping

OpenAI provides fixed voice names (`alloy`, `echo`, `coral`, etc.). VibeVoice uses **preset voices** with names like `Alice`, `Bowen`, etc.

**Strategy**: Map the `voice` parameter to VibeVoice preset voice names (case-insensitive).

```
voice: "Alice"  →  Finds preset voice named "Alice"  →  Uses its .wav file
voice: "Bowen"  →  Finds preset voice named "Bowen"  →  Uses its .wav file
```

### Model Mapping

| OpenAI-compat Name | VibeVoice model_dtype | Description |
|-------------------|----------------------|-------------|
| `vibevoice-7b` | `bf16` | Standard quality (recommended) |
| `vibevoice-7b-hd` | `float8_e4m3fn` | Higher precision |
| `tts-1` | `bf16` | OpenAI alias for compatibility |
| `tts-1-hd` | `float8_e4m3fn` | OpenAI alias for compatibility |

> **Fallback behavior**: If an unrecognized model name is provided, it silently falls back to `vibevoice-7b` (bf16) instead of returning an error. A warning is logged server-side.

### Response Format

| Format | OpenAI | VibeVoice Compat | Notes |
|--------|--------|-----------------|-------|
| `mp3` | Default | Supported (via ffmpeg) | Requires ffmpeg |
| `wav` | Supported | Native output | No conversion needed |
| `flac` | Supported | Supported (via ffmpeg) | Requires ffmpeg |
| `opus` | Supported | Supported (via ffmpeg) | Encoded with libopus in OGG container |
| `aac` | Supported | Supported (via ffmpeg) | Requires ffmpeg |
| `pcm` | Supported | Supported (via ffmpeg) | 16-bit signed LE, 24kHz mono |

### Error Format

OpenAI error format:
```json
{
  "error": {
    "message": "Description of what went wrong",
    "type": "invalid_request_error",
    "code": "specific_error_code"
  }
}
```

### Key Behavioral Differences

1. **Latency**: OpenAI TTS responds in ~1-3 seconds. VibeVoice generation takes 10-60+ seconds depending on text length and hardware. Clients should expect longer response times.

2. **Concurrency**: OpenAI handles concurrent requests. VibeVoice has a single-thread GPU task queue — only one generation runs at a time. Concurrent requests get 503.

3. **Streaming**: OpenAI supports chunked transfer encoding for real-time streaming. VibeVoice does not support streaming — the full audio is returned after generation completes.

4. **Voice cloning**: VibeVoice's unique advantage is voice cloning from audio samples. The OpenAI-compat API uses preset voices, but the native Quick Generate API supports custom voice upload.

5. **Multi-speaker**: VibeVoice supports multi-speaker dialogue in a single request. The OpenAI-compat API only exposes single-speaker narration mode.

## Authentication

The API supports optional Bearer token authentication, controlled by the `OPENAI_COMPAT_API_KEY` environment variable.

### Setting Up Authentication

```bash
# Set the API key (any string you choose)
export OPENAI_COMPAT_API_KEY="sk-your-secret-key-here"

# Start the server
python backend/run.py
```

### Behavior

| `OPENAI_COMPAT_API_KEY` env var | Request has valid Bearer token | Result |
|--------------------------------|-------------------------------|--------|
| **Not set** | N/A | Access allowed (open access, warning logged) |
| Set | Yes (`Authorization: Bearer <key>`) | Access allowed |
| Set | No or invalid token | `401 Unauthorized` |

When the env var is not set, **all requests are allowed without authentication** and a warning is logged on each request. This is suitable for local development but **not recommended for production or public-facing deployments**.

### Token Format

The API accepts the standard OpenAI `Authorization` header format:

```
Authorization: Bearer sk-your-secret-key-here
```

Both `Bearer <key>` and raw `<key>` formats are accepted, but `Bearer` prefix is recommended for SDK compatibility.

## Endpoints

### `POST /v1/audio/speech` — Generate Speech

The primary TTS endpoint. Accepts a JSON request body and returns binary audio data synchronously. The server blocks until generation is complete (up to 300 seconds).

### `POST /v1/audio/transcriptions` — Transcribe Audio

Speech-to-text backed by VibeVoice-ASR. Accepts `multipart/form-data` (same as OpenAI) and blocks until the transcription finishes (up to 1800 seconds). It uses the same GPU task queue, API-key check and `503`-when-busy behaviour as `/v1/audio/speech`. Each request is also stored in the standalone transcription history (`/api/v1/transcriptions/history`), so results can be reviewed in the Quick Transcribe page.

**Parameter mapping:**

| OpenAI parameter | VibeVoice behaviour |
|------------------|---------------------|
| `file` | Required. WAV, MP3, M4A, FLAC, WEBM |
| `model` | Required. Mapped to `model_dtype` (table below); unknown names fall back to `bf16` |
| `prompt` | Passed to the model as `context_info` (hotwords, names, topic) |
| `response_format` | `json` (default), `text`, `verbose_json`. `srt` / `vtt` return `400 unsupported_format` |
| `temperature` | 0-1, default 0 (greedy decoding) |
| `language` | Not used by the model; echoed back in `verbose_json` |
| `timestamp_granularities[]`, `stream`, `include[]` | Ignored |

**Model mapping:**

| Model | model_dtype |
|-------|-------------|
| `vibevoice-asr` | `bf16` |
| `vibevoice-asr-fp8` | `float8_e4m3fn` |
| `whisper-1` | `bf16` (OpenAI alias) |

**`verbose_json` differences from OpenAI:**
- Segments carry `id`, `start`, `end`, `text`, plus a VibeVoice `speaker` field; OpenAI's token-level fields (`tokens`, `avg_logprob`, `compression_ratio`, `no_speech_prob`, ...) are not returned.
- `language` is the request's `language` value, or `"unknown"`. VibeVoice-ASR does not report a detected language, so SDKs that expect an ISO language name may need to tolerate this.

See [APIs.md](APIs.md#2-create-transcription) for full request/response examples.

### `GET /v1/models` — List Available Models

Returns a list of available models in OpenAI-compatible format.

**Response:**
```json
{
  "object": "list",
  "data": [
    { "id": "tts-1", "object": "model", "created": 0, "owned_by": "vibevoice" },
    { "id": "tts-1-hd", "object": "model", "created": 0, "owned_by": "vibevoice" },
    { "id": "vibevoice-7b", "object": "model", "created": 0, "owned_by": "vibevoice" },
    { "id": "vibevoice-7b-hd", "object": "model", "created": 0, "owned_by": "vibevoice" },
    { "id": "vibevoice-asr", "object": "model", "created": 0, "owned_by": "vibevoice" },
    { "id": "vibevoice-asr-fp8", "object": "model", "created": 0, "owned_by": "vibevoice" },
    { "id": "whisper-1", "object": "model", "created": 0, "owned_by": "vibevoice" }
  ]
}
```

This endpoint does not require authentication.

## Configuration

### Environment Variables

| Variable | Default | Description |
|----------|---------|-------------|
| `OPENAI_COMPAT_API_KEY` | *(not set)* | API key for Bearer token auth. When unset, all requests are allowed without authentication. |

### Server-Side Constants

These are defined in `backend/services/openai_compat_service.py`:

| Constant | Value | Description |
|----------|-------|-------------|
| `DEFAULT_TIMEOUT` | `300` seconds | Maximum time the server will block waiting for generation to complete. Returns `504 Gateway Timeout` if exceeded. |
| `TRANSCRIPTION_TIMEOUT` | `1800` seconds | Same, for `/v1/audio/transcriptions` (long recordings take longer to decode). |
| `POLL_INTERVAL` | `0.5` seconds | Internal polling frequency when checking generation status. |
| Max input length | `4096` characters | Maximum allowed length for the `input` text field. |

### Client-Side Timeout Recommendations

Since the server blocks synchronously for the entire generation duration, **clients must set appropriate timeouts**:

```python
# Python SDK — set a longer timeout
import httpx
client = OpenAI(
    base_url="http://localhost:9527/v1",
    api_key="sk-your-key",
    timeout=httpx.Timeout(360.0, connect=10.0),  # 6 min read timeout
)
```

```javascript
// Node.js SDK — set a longer timeout
const client = new OpenAI({
    baseURL: "http://localhost:9527/v1",
    apiKey: "sk-your-key",
    timeout: 360000,  // 6 minutes in milliseconds
});
```

```bash
# curl — set a longer timeout
curl --max-time 360 http://localhost:9527/v1/audio/speech \
  -H "Authorization: Bearer sk-your-key" \
  -H "Content-Type: application/json" \
  -d '{"model": "vibevoice-7b", "input": "Hello", "voice": "Alice"}' \
  --output speech.wav
```

## HTTP Status Codes

| Code | Meaning | When |
|------|---------|------|
| `200` | Success | Audio generated and returned as binary data |
| `400` | Bad Request | Missing required parameters, unsupported format, invalid JSON, input too long |
| `401` | Unauthorized | `OPENAI_COMPAT_API_KEY` is set but request has no/invalid Bearer token |
| `500` | Internal Server Error | Generation failed, voice file not found, audio conversion failed |
| `503` | Service Unavailable | GPU task queue is busy with another generation request |
| `504` | Gateway Timeout | Generation did not complete within the 300-second timeout |

### Error Codes Reference

| HTTP Status | `error.code` | Description |
|-------------|-------------|-------------|
| 400 | `missing_model` | `model` parameter not provided |
| 400 | `missing_input` | `input` parameter not provided |
| 400 | `missing_voice` | `voice` parameter not provided |
| 400 | `input_too_long` | Input text exceeds 4096 characters |
| 400 | `unsupported_format` | Requested format not in (wav, mp3, flac, opus, aac, pcm) |
| 401 | `invalid_api_key` | Bearer token missing or does not match |
| 500 | `voice_not_found` | Preset voice name not recognized |

## VibeVoice Extension Parameters

> **Note**: The following extension parameters are defined in the design but **not yet implemented** in the current API. They are reserved for future use. Currently, `seeds` defaults to `42` and `offloading` follows the server's global configuration.

```json
{
  "model": "vibevoice-7b",
  "input": "Hello world",
  "voice": "Alice",
  "response_format": "wav",

  "seeds": 42,
  "cfg_scale": 1.3,
  "offloading": "balanced"
}
```

## Usage Examples

### curl

```bash
# Without authentication (when OPENAI_COMPAT_API_KEY is not set)
curl http://localhost:9527/v1/audio/speech \
  -H "Content-Type: application/json" \
  -d '{
    "model": "vibevoice-7b",
    "input": "Hello, this is VibeVoice speaking.",
    "voice": "Alice"
  }' \
  --max-time 360 \
  --output speech.wav

# With authentication
curl http://localhost:9527/v1/audio/speech \
  -H "Authorization: Bearer sk-your-key" \
  -H "Content-Type: application/json" \
  -d '{
    "model": "vibevoice-7b",
    "input": "Hello, this is VibeVoice speaking.",
    "voice": "Alice",
    "response_format": "mp3"
  }' \
  --max-time 360 \
  --output speech.mp3

# With opus format
curl http://localhost:9527/v1/audio/speech \
  -H "Authorization: Bearer sk-your-key" \
  -H "Content-Type: application/json" \
  -d '{
    "model": "vibevoice-7b",
    "input": "Hello, this is VibeVoice speaking.",
    "voice": "Alice",
    "response_format": "opus"
  }' \
  --max-time 360 \
  --output speech.ogg

# List available models
curl http://localhost:9527/v1/models
```

### OpenAI Python SDK

```python
import httpx
from openai import OpenAI

client = OpenAI(
    base_url="http://localhost:9527/v1",
    api_key="sk-your-key",  # Or any string if auth is disabled
    timeout=httpx.Timeout(360.0, connect=10.0),
)

response = client.audio.speech.create(
    model="vibevoice-7b",
    voice="Alice",
    input="Hello, this is VibeVoice speaking.",
    response_format="wav",
)

response.stream_to_file("output.wav")
```

### OpenAI Node.js SDK

```javascript
import OpenAI from "openai";
import fs from "fs";

const client = new OpenAI({
    baseURL: "http://localhost:9527/v1",
    apiKey: "sk-your-key",  // Or any string if auth is disabled
    timeout: 360000,        // 6 minutes
});

const response = await client.audio.speech.create({
    model: "vibevoice-7b",
    voice: "Alice",
    input: "Hello, this is VibeVoice speaking.",
});

const buffer = Buffer.from(await response.arrayBuffer());
await fs.promises.writeFile("output.wav", buffer);
```

## Limitations

- **No streaming support** — full audio is returned after generation completes; no chunked transfer encoding
- **Single concurrent request** — GPU task queue allows only one generation at a time; additional requests get `503`
- **Higher latency** — generation takes ~10-60+ seconds vs OpenAI's ~1-3 seconds; clients must set appropriate timeouts
- **All OpenAI output formats supported** — `wav` (native), `mp3`, `flac`, `opus`, `aac`, `pcm` (all non-wav formats require ffmpeg)
- **`speed` parameter ignored** — accepted for compatibility but has no effect on generation
- **`instructions` parameter not supported** — this is an OpenAI GPT-4o-mini-tts-only feature
- **No billing/usage tracking** — no token counting or usage metering
- **Extension parameters not yet wired** — `seeds`, `cfg_scale`, `offloading` in request body are currently ignored (reserved for future implementation)
- **Transcriptions: no `srt` / `vtt` output, no streaming, no language detection** — see the transcriptions section above
- **Non-wav format conversion requires ffmpeg** — if `ffmpeg` is not installed, requesting `mp3`, `flac`, `opus`, `aac`, or `pcm` formats will return `500`
