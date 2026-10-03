"""
OpenAI-Compatible TTS and transcription API endpoints

Implements POST /v1/audio/speech and POST /v1/audio/transcriptions for drop-in compatibility with OpenAI clients.
This uses a separate blueprint registered at /v1 (not /api/v1) to match the OpenAI URL scheme.
"""
from flask import Blueprint, request, jsonify, send_file, current_app, Response

from backend.api.quick_generate import ALLOWED_EXTENSIONS, _allowed_file
from backend.models.transcription import segment_seconds, transcript_plain_text
from backend.services.openai_compat_service import (
    OpenAICompatService, MODEL_MAPPING, FORMAT_MIME_TYPES,
    ASR_MODEL_MAPPING, TRANSCRIPTION_RESPONSE_FORMATS,
)
from util.logger import get_logger

logger = get_logger(__name__)

# Separate blueprint for OpenAI-compatible routes (mounted at /v1)
openai_bp = Blueprint('openai_compat', __name__)


def _openai_error(message: str, error_type: str = "invalid_request_error",
                  code: str = None, status: int = 400) -> tuple:
    """Return an OpenAI-style error response."""
    body = {
        "error": {
            "message": message,
            "type": error_type,
        }
    }
    if code:
        body["error"]["code"] = code
    return jsonify(body), status


def _get_service() -> OpenAICompatService:
    """Get OpenAICompatService instance from app config."""
    return OpenAICompatService(
        workspace_dir=current_app.config['WORKSPACE_DIR'],
        preset_dir=current_app.config['PRESET_VOICE_DIR'],
        fake_model=current_app.config.get('FAKE_MODEL', False),
    )


@openai_bp.route('/audio/speech', methods=['POST'])
def create_speech():
    """
    OpenAI-compatible TTS endpoint.

    Request (application/json):
        {
            "model": "vibevoice-7b",         // Required
            "input": "Hello world",           // Required, max 4096 chars
            "voice": "Alice",                 // Required, preset voice name
            "response_format": "wav",         // Optional, default: wav (supports: wav, mp3, flac, opus, aac, pcm)
            "speed": 1.0                      // Optional, accepted but ignored
        }

    Response:
        Binary audio data with appropriate Content-Type header.
    """
    service = _get_service()

    # --- Authentication ---
    auth_header = request.headers.get('Authorization')
    if not service.validate_api_key(auth_header):
        return _openai_error(
            "Invalid API key provided.",
            error_type="authentication_error",
            code="invalid_api_key",
            status=401,
        )

    # --- Parse JSON body ---
    if not request.is_json:
        return _openai_error("Request body must be JSON (Content-Type: application/json).")

    data = request.get_json(silent=True)
    if not data:
        return _openai_error("Invalid JSON in request body.")

    # --- Validate required fields ---
    model = data.get('model')
    if not model:
        return _openai_error("Missing required parameter: 'model'.", code="missing_model")

    input_text = data.get('input')
    if not input_text:
        return _openai_error("Missing required parameter: 'input'.", code="missing_input")

    if len(input_text) > 4096:
        return _openai_error(
            f"Input text is too long ({len(input_text)} chars). Maximum is 4096 characters.",
            code="input_too_long",
        )

    voice = data.get('voice')
    if not voice:
        return _openai_error("Missing required parameter: 'voice'.", code="missing_voice")

    # --- Validate optional fields ---
    response_format = data.get('response_format', 'wav')
    if response_format not in FORMAT_MIME_TYPES:
        supported = ', '.join(sorted(FORMAT_MIME_TYPES.keys()))
        return _openai_error(
            f"Unsupported response_format '{response_format}'. Supported formats: {supported}",
            code="unsupported_format",
        )

    # speed is accepted but ignored (not supported by engine)
    # data.get('speed', 1.0)

    # --- Resolve model (fallback to bf16 if unknown) ---
    model_dtype, err = service.resolve_model(model)
    if err:
        logger.warning(f"Unknown model '{model}', falling back to bf16")
        model_dtype = 'bf16'

    # --- Resolve voice ---
    voice_filename, err = service.resolve_voice(voice)
    if err:
        return _openai_error(err, code="voice_not_found")

    # --- Generate speech ---
    try:
        audio_path, err, status_code = service.generate_speech(
            text=input_text,
            voice_filename=voice_filename,
            model_dtype=model_dtype,
            response_format=response_format,
        )
    except Exception as e:
        logger.error(f"Unexpected error in speech generation: {e}", exc_info=True)
        return _openai_error(
            "An internal error occurred during speech generation.",
            error_type="server_error",
            status=500,
        )

    if err:
        error_type = "server_error" if status_code >= 500 else "invalid_request_error"
        return _openai_error(err, error_type=error_type, status=status_code)

    # --- Return audio ---
    mime_type = FORMAT_MIME_TYPES[response_format]
    # Map format to download extension (opus→ogg, pcm→raw)
    ext_map = {'opus': 'ogg', 'pcm': 'raw'}
    ext = ext_map.get(response_format, response_format)
    return send_file(
        str(audio_path),
        mimetype=mime_type,
        as_attachment=False,
        download_name=f"speech.{ext}",
    )


@openai_bp.route('/audio/transcriptions', methods=['POST'])
def create_transcription():
    """
    OpenAI-compatible transcription endpoint.

    Request (multipart/form-data):
        file             // Required, audio file
        model            // Required, e.g. "vibevoice-asr" or "whisper-1"
        prompt           // Optional, passed to the model as context info (hotwords, names, topic)
        response_format  // Optional, default: json (supports: json, text, verbose_json)
        temperature      // Optional, 0-1, default: 0 (greedy)
        language         // Optional, accepted but not used; echoed in verbose_json

    Response:
        json: {"text": "..."}; text: plain text; verbose_json: text, duration and speaker-labelled segments.
    """
    service = _get_service()

    auth_header = request.headers.get('Authorization')
    if not service.validate_api_key(auth_header):
        return _openai_error(
            "Invalid API key provided.",
            error_type="authentication_error",
            code="invalid_api_key",
            status=401,
        )

    audio = request.files.get('file')
    if audio is None or not audio.filename:
        return _openai_error("Missing required parameter: 'file'.", code="missing_file")
    if not _allowed_file(audio.filename):
        supported = ', '.join(sorted(ALLOWED_EXTENSIONS))
        return _openai_error(
            f"Unsupported file type '{audio.filename}'. Supported formats: {supported}",
            code="unsupported_file_type",
        )

    model = request.form.get('model')
    if not model:
        return _openai_error("Missing required parameter: 'model'.", code="missing_model")

    response_format = request.form.get('response_format', 'json')
    if response_format not in TRANSCRIPTION_RESPONSE_FORMATS:
        supported = ', '.join(TRANSCRIPTION_RESPONSE_FORMATS)
        return _openai_error(
            f"Unsupported response_format '{response_format}'. Supported formats: {supported}",
            code="unsupported_format",
        )

    try:
        temperature = float(request.form.get('temperature', 0) or 0)
    except ValueError:
        return _openai_error("Invalid 'temperature': must be a number.", code="invalid_temperature")
    if temperature < 0 or temperature > 1:
        return _openai_error("Invalid 'temperature': must be between 0 and 1.", code="invalid_temperature")

    model_dtype, err = service.resolve_asr_model(model)
    if err:
        logger.warning(f"Unknown transcription model '{model}', falling back to bf16")
        model_dtype = 'bf16'

    prompt = (request.form.get('prompt') or '').strip() or None
    language = request.form.get('language')

    try:
        transcription, err, status_code = service.transcribe_audio(
            file_data=audio.read(),
            filename=audio.filename,
            model_dtype=model_dtype,
            prompt=prompt,
            temperature=temperature,
        )
    except Exception as e:
        logger.error(f"Unexpected error in transcription: {e}", exc_info=True)
        return _openai_error(
            "An internal error occurred during transcription.",
            error_type="server_error",
            status=500,
        )

    if err:
        error_type = "server_error" if status_code >= 500 else "invalid_request_error"
        return _openai_error(err, error_type=error_type, status=status_code)

    text = transcript_plain_text(transcription.segments, transcription.raw_text)
    if response_format == 'text':
        return Response(text, mimetype='text/plain; charset=utf-8')
    if response_format == 'json':
        return jsonify({"text": text})

    segments = []
    for index, seg in enumerate(transcription.segments):
        segments.append({
            "id": index,
            "start": segment_seconds(seg.get('start_time')),
            "end": segment_seconds(seg.get('end_time')),
            "speaker": seg.get('speaker_id'),
            "text": str(seg.get('text', '')),
        })
    return jsonify({
        "task": "transcribe",
        "language": language or "unknown",
        "duration": transcription.audio_duration,
        "text": text,
        "segments": segments,
    })


@openai_bp.route('/models', methods=['GET'])
def list_models():
    """
    List available models (OpenAI-compatible format).

    Returns a list of model objects matching OpenAI's /v1/models response format.
    """
    models = []
    seen = set()
    for model_name in sorted(list(MODEL_MAPPING.keys()) + list(ASR_MODEL_MAPPING.keys())):
        if model_name not in seen:
            seen.add(model_name)
            models.append({
                "id": model_name,
                "object": "model",
                "created": 0,
                "owned_by": "vibevoice",
            })

    return jsonify({
        "object": "list",
        "data": models,
    })
