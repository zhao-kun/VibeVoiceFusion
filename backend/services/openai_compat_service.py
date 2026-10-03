"""
OpenAI-Compatible TTS and transcription Service

Provides synchronous wrappers over VibeVoice's async quick generation and ASR engines,
implementing the OpenAI API contracts POST /v1/audio/speech and POST /v1/audio/transcriptions.
"""
import os
import shutil
import subprocess
import time
from pathlib import Path
from typing import Optional, Tuple

from backend.services.preset_voice_service import PresetVoiceService
from backend.services.quick_generate_service import QuickGenerateService
from backend.services.transcription_service import TranscriptionService
from backend.models.transcription import Transcription
from backend.task_manager.task import gm
from backend.inference.quick_generate_inference import QuickGenerateInferenceBase
from config.configuration_vibevoice import InferencePhase
from util.logger import get_logger

logger = get_logger(__name__)

# Model name → model_dtype mapping
MODEL_MAPPING = {
    'vibevoice-7b': 'bf16',
    'vibevoice-7b-hd': 'float8_e4m3fn',
    'tts-1': 'bf16',
    'tts-1-hd': 'float8_e4m3fn',
}

# Transcription model name → model_dtype mapping
ASR_MODEL_MAPPING = {
    'vibevoice-asr': 'bf16',
    'vibevoice-asr-fp8': 'float8_e4m3fn',
    'whisper-1': 'bf16',
}

TRANSCRIPTION_RESPONSE_FORMATS = ('json', 'text', 'verbose_json')

# Supported output formats and their MIME types
# Matches OpenAI's supported formats: mp3, opus, aac, flac, wav, pcm
FORMAT_MIME_TYPES = {
    'wav': 'audio/wav',
    'mp3': 'audio/mpeg',
    'flac': 'audio/flac',
    'opus': 'audio/opus',
    'aac': 'audio/aac',
    'pcm': 'audio/pcm',
}

# Polling and timeout configuration
POLL_INTERVAL = 0.5  # seconds
DEFAULT_TIMEOUT = 300  # seconds
# Long recordings (the model accepts up to ~60 minutes) decode far slower than speech generation
TRANSCRIPTION_TIMEOUT = 1800  # seconds


class OpenAICompatService:
    """Service for OpenAI-compatible TTS API"""

    def __init__(self, workspace_dir: Path, preset_dir: Path, fake_model: bool = False):
        self.workspace_dir = workspace_dir
        self.fake_model = fake_model
        self.quick_generate_service = QuickGenerateService(
            workspace_dir=workspace_dir,
            fake_model=fake_model,
        )
        self.preset_voice_service = PresetVoiceService(preset_dir=preset_dir)

    def validate_api_key(self, auth_header: Optional[str]) -> bool:
        """
        Validate API key if OPENAI_COMPAT_API_KEY env var is set.

        Returns True if auth is valid or not required.
        """
        expected_key = os.environ.get('OPENAI_COMPAT_API_KEY')
        if not expected_key:
            logger.warning("OPENAI_COMPAT_API_KEY not set, allowing unauthenticated access to OpenAI-compatible API")
            return True  # No key configured, open access

        if not auth_header:
            return False

        # Accept "Bearer <key>" format
        if auth_header.startswith('Bearer '):
            provided_key = auth_header[7:]
        else:
            provided_key = auth_header

        return provided_key == expected_key

    def resolve_voice(self, voice_name: str) -> Tuple[Optional[str], Optional[str]]:
        """
        Resolve voice name to preset voice file path.

        Args:
            voice_name: Voice name (case-insensitive match against preset names)

        Returns:
            Tuple of (preset_filename, error_message).
            On success: (filename, None). On failure: (None, error_message).
        """
        presets, _ = self.preset_voice_service.list_presets()
        available_names = [p.name for p in presets]

        # Case-insensitive match
        for preset in presets:
            if preset.name.lower() == voice_name.lower():
                return preset.filename, None

        available_str = ', '.join(sorted(set(available_names))) if available_names else 'none'
        return None, f"Voice '{voice_name}' not found. Available voices: {available_str}"

    def resolve_model(self, model_name: str) -> Tuple[Optional[str], Optional[str]]:
        """
        Resolve OpenAI-compat model name to VibeVoice model_dtype.

        Returns:
            Tuple of (model_dtype, error_message).
        """
        model_dtype = MODEL_MAPPING.get(model_name.lower())
        if model_dtype:
            return model_dtype, None

        available = ', '.join(sorted(MODEL_MAPPING.keys()))
        return None, f"Model '{model_name}' not found. Available models: {available}"

    def resolve_asr_model(self, model_name: str) -> Tuple[Optional[str], Optional[str]]:
        """
        Resolve OpenAI-compat transcription model name to VibeVoice-ASR model_dtype.

        Returns:
            Tuple of (model_dtype, error_message).
        """
        model_dtype = ASR_MODEL_MAPPING.get(model_name.lower())
        if model_dtype:
            return model_dtype, None

        available = ', '.join(sorted(ASR_MODEL_MAPPING.keys()))
        return None, f"Model '{model_name}' not found. Available models: {available}"

    def transcribe_audio(self, file_data: bytes, filename: str, model_dtype: str,
                         prompt: Optional[str] = None,
                         temperature: float = 0.0,
                         timeout: int = TRANSCRIPTION_TIMEOUT) -> Tuple[Optional[Transcription], Optional[str], Optional[int]]:
        """
        Transcribe audio synchronously by submitting an ASR task and waiting for completion.

        The record stays in the standalone transcription history, like speech requests stay in quick generate.

        Returns:
            Tuple of (transcription, error_message, http_status_code).
            On success: (transcription, None, None). On failure: (None, error_msg, status_code).
        """
        if gm.has_task():
            return None, "Server is busy processing another request. Please retry later.", 503

        transcription_service = TranscriptionService.for_workspace(self.workspace_dir, fake_model=self.fake_model)
        audio_file = transcription_service.save_audio_file(file_data, filename)
        transcription = transcription_service.start_transcription(
            audio_file=audio_file,
            original_filename=filename,
            context_info=prompt,
            model_dtype=model_dtype,
            temperature=temperature,
            seeds=42,
        )
        if not transcription:
            return None, "Server is busy processing another request. Please retry later.", 503

        request_id = transcription.request_id
        start_time = time.time()
        while time.time() - start_time < timeout:
            time.sleep(POLL_INTERVAL)

            current = transcription_service.get_transcription(request_id)
            if not current:
                return None, "Transcription task disappeared unexpectedly.", 500
            if current.status == InferencePhase.COMPLETED:
                return current, None, None
            if current.status == InferencePhase.FAILED:
                return None, current.error_message or "Transcription failed.", 500

        return None, "Transcription timed out. The server may be under heavy load.", 504

    def generate_speech(self, text: str, voice_filename: str, model_dtype: str,
                        response_format: str = 'wav',
                        timeout: int = DEFAULT_TIMEOUT) -> Tuple[Optional[Path], Optional[str], Optional[int]]:
        """
        Generate speech synchronously by submitting a task and waiting for completion.

        Args:
            text: Text to speak
            voice_filename: Preset voice filename (e.g., "en-Alice_woman.wav")
            model_dtype: Model dtype (e.g., "bf16")
            response_format: Output format (wav, mp3, flac)
            timeout: Maximum wait time in seconds

        Returns:
            Tuple of (audio_path, error_message, http_status_code).
            On success: (path, None, None). On failure: (None, error_msg, status_code).
        """
        # Check if task queue is busy
        if gm.has_task():
            return None, "Server is busy processing another request. Please retry later.", 503

        # Get preset voice file path
        preset_path = self.preset_voice_service.get_preset_path(voice_filename)
        if not preset_path:
            return None, f"Voice file not found: {voice_filename}", 500

        # Copy preset voice to quick generate voices dir and save
        voice_data = preset_path.read_bytes()
        saved_filename = self.quick_generate_service.save_voice_file(voice_data, voice_filename)

        # Start generation
        quick_gen = self.quick_generate_service.start_generation(
            voice_files=[saved_filename],
            text=text,
            seeds=42,
            batch_size=1,
            model_dtype=model_dtype,
        )

        if not quick_gen:
            return None, "Server is busy processing another request. Please retry later.", 503

        request_id = quick_gen.request_id

        # Poll until completion or timeout
        start_time = time.time()
        while time.time() - start_time < timeout:
            time.sleep(POLL_INTERVAL)

            gen = self.quick_generate_service.get_generation(request_id)
            if not gen:
                return None, "Generation task disappeared unexpectedly.", 500

            if gen.status == InferencePhase.COMPLETED:
                # Get audio path
                audio_path = self.quick_generate_service.get_audio_path(request_id, item_index=0)
                if not audio_path:
                    return None, "Generation completed but audio file not found.", 500

                # Convert format if needed
                if response_format != 'wav':
                    converted_path = self._convert_audio(audio_path, response_format)
                    if converted_path:
                        return converted_path, None, None
                    else:
                        return None, f"Failed to convert audio to {response_format} format.", 500

                return audio_path, None, None

            if gen.status == InferencePhase.FAILED:
                error_msg = gen.error_message or "Generation failed."
                return None, error_msg, 500

        # Timeout
        return None, "Generation timed out. The server may be under heavy load.", 504

    def _convert_audio(self, source_path: Path, target_format: str) -> Optional[Path]:
        """
        Convert WAV audio to target format using ffmpeg.

        Returns the path to the converted file, or None on failure.
        """
        # Map format to file extension (opus uses .ogg container, pcm uses .raw)
        ext_map = {'opus': 'ogg', 'pcm': 'raw'}
        ext = ext_map.get(target_format, target_format)
        target_path = source_path.with_suffix(f'.{ext}')

        try:
            cmd = ['ffmpeg', '-y', '-i', str(source_path)]

            if target_format == 'mp3':
                cmd.extend(['-codec:a', 'libmp3lame', '-qscale:a', '2'])
            elif target_format == 'flac':
                cmd.extend(['-codec:a', 'flac'])
            elif target_format == 'opus':
                cmd.extend(['-codec:a', 'libopus', '-b:a', '96k'])
            elif target_format == 'aac':
                cmd.extend(['-codec:a', 'aac', '-b:a', '128k'])
            elif target_format == 'pcm':
                # PCM: 16-bit signed little-endian, 24kHz mono (OpenAI convention)
                cmd.extend(['-f', 's16le', '-acodec', 'pcm_s16le', '-ar', '24000', '-ac', '1'])

            cmd.append(str(target_path))

            result = subprocess.run(cmd, capture_output=True, text=True, timeout=60)
            if result.returncode != 0:
                logger.error(f"ffmpeg conversion failed: {result.stderr}")
                return None

            return target_path

        except subprocess.TimeoutExpired:
            logger.error("Audio conversion timed out")
            return None
        except FileNotFoundError:
            logger.error("ffmpeg not found for audio conversion")
            return None
        except Exception as e:
            logger.error(f"Audio conversion error: {e}")
            return None

    def get_available_voices(self) -> list:
        """Get list of available preset voice names for error messages."""
        presets, _ = self.preset_voice_service.list_presets()
        return sorted(set(p.name for p in presets))
