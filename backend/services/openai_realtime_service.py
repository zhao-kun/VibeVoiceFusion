"""
OpenAI Realtime transcription protocol over a live VibeVoice-ASR session

Accepts both the beta (`transcription_session.update`, pcm16) and the GA (`session.update` with
session.type "transcription", audio/pcm 24 kHz) client dialects. Only 24 kHz mono PCM16 input is supported.
"""
import base64
import binascii
import json
import time
from typing import Any, Dict, List, Optional
from uuid import uuid4

from backend.inference.streaming_asr_inference import StreamingASRSession, STATUS_EVENT
from backend.services.openai_compat_service import ASR_MODEL_MAPPING
from backend.services.transcription_service import TranscriptionService
from util.logger import get_logger

logger = get_logger(__name__)

DIALECT_BETA = 'beta'
DIALECT_GA = 'ga'
DEFAULT_REALTIME_MODEL = 'vibevoice-asr'
FINISH_EVENT = 'vibevoice.session.finish'


def error_event(message: str, error_type: str = 'invalid_request_error', code: Optional[str] = None,
                client_event_id: Optional[str] = None) -> Dict[str, Any]:
    error = {'type': error_type, 'message': message}
    if code:
        error['code'] = code
    if client_event_id:
        error['event_id'] = client_event_id
    return {'type': 'error', 'error': error}


class RealtimeTranscriptionHandler:
    """Translates client events into a live transcription and engine output into server events.

    The engine is started lazily by the first session update or audio append, so the client can still
    choose the model first; until then nothing holds the GPU slot.
    """

    def __init__(self, transcription_service: TranscriptionService, limits: Dict[str, float],
                 max_session_seconds: float, offloading_config: Optional[Dict[str, Any]] = None):
        self.transcription_service = transcription_service
        self.limits = limits
        self.max_session_seconds = max_session_seconds
        self.offloading_config = offloading_config
        self.session_id = f"sess_{uuid4().hex[:24]}"
        self.dialect = DIALECT_GA
        self.model = DEFAULT_REALTIME_MODEL
        self.language: Optional[str] = None
        self.prompt: Optional[str] = None
        self.session: Optional[StreamingASRSession] = None
        self.request_id: Optional[str] = None
        self.finishing = False
        self.done = False
        self.created_at = time.time()

    def _session_object(self) -> Dict[str, Any]:
        transcription = {'model': self.model, 'language': self.language, 'prompt': self.prompt or ''}
        if self.dialect == DIALECT_BETA:
            return {
                'id': self.session_id,
                'object': 'realtime.transcription_session',
                'input_audio_format': 'pcm16',
                'input_audio_transcription': transcription,
                'turn_detection': None,
                'input_audio_noise_reduction': None,
                'include': [],
                'modalities': ['text'],
            }
        return {
            'type': 'transcription',
            'id': self.session_id,
            'object': 'realtime.transcription_session',
            'audio': {'input': {
                'format': {'type': 'audio/pcm', 'rate': 24000},
                'transcription': transcription,
                'turn_detection': None,
                'noise_reduction': None,
            }},
            'include': [],
        }

    def created_event(self) -> Dict[str, Any]:
        return {'type': 'session.created', 'session': self._session_object()}

    def handle(self, raw: Any) -> List[Dict[str, Any]]:
        """Process one client message and return the events to send immediately"""
        if not isinstance(raw, str):
            return [error_event("Binary messages are not supported; send JSON events.", code='invalid_message')]
        try:
            message = json.loads(raw)
        except json.JSONDecodeError:
            return [error_event("Invalid JSON in client event.", code='invalid_json')]
        if not isinstance(message, dict):
            return [error_event("Client event must be a JSON object.", code='invalid_message')]

        event_type = message.get('type')
        client_event_id = message.get('event_id')
        if event_type == 'transcription_session.update':
            return self._update_beta(message.get('session') or {}, client_event_id)
        if event_type == 'session.update':
            return self._update_ga(message.get('session') or {}, client_event_id)
        if event_type == 'input_audio_buffer.append':
            return self._append(message.get('audio'), client_event_id)
        if event_type == 'input_audio_buffer.commit':
            if self.session:
                self.session.request_flush()
            return []
        if event_type == 'input_audio_buffer.clear':
            if self.session:
                self.session.request_clear()
            return [{'type': 'input_audio_buffer.cleared'}]
        if event_type == FINISH_EVENT:
            return self.finish('client_request')
        return [error_event(f"Unsupported client event type '{event_type}'.", code='unknown_event',
                            client_event_id=client_event_id)]

    def _apply_transcription(self, transcription: Dict[str, Any]) -> None:
        if not isinstance(transcription, dict):
            return
        if transcription.get('model'):
            self.model = str(transcription['model'])
        if 'language' in transcription:
            self.language = transcription.get('language')
        if 'prompt' in transcription:
            self.prompt = (str(transcription.get('prompt') or '')).strip() or None
            if self.session:
                self.session.context_info = self.prompt

    def _update_beta(self, config: Dict[str, Any], client_event_id: Optional[str]) -> List[Dict[str, Any]]:
        audio_format = config.get('input_audio_format')
        if audio_format not in (None, 'pcm16'):
            return [error_event(f"Unsupported input_audio_format '{audio_format}'. Only 'pcm16' (24 kHz mono) "
                                f"is supported.", code='unsupported_audio_format', client_event_id=client_event_id)]
        self.dialect = DIALECT_BETA
        self._apply_transcription(config.get('input_audio_transcription'))
        errors = self._start()
        if errors:
            return errors
        return [{'type': 'transcription_session.updated', 'session': self._session_object()}]

    def _update_ga(self, config: Dict[str, Any], client_event_id: Optional[str]) -> List[Dict[str, Any]]:
        session_type = config.get('type', 'transcription')
        if session_type != 'transcription':
            return [error_event(f"Unsupported session type '{session_type}'. Only 'transcription' sessions are "
                                f"supported.", code='unsupported_session_type', client_event_id=client_event_id)]
        audio_input = (config.get('audio') or {}).get('input') or {}
        audio_format = audio_input.get('format')
        if audio_format is not None:
            if not isinstance(audio_format, dict) or audio_format.get('type') != 'audio/pcm' \
                    or audio_format.get('rate', 24000) != 24000:
                return [error_event("Unsupported audio format. Only {'type': 'audio/pcm', 'rate': 24000} is "
                                    "supported.", code='unsupported_audio_format', client_event_id=client_event_id)]
        self.dialect = DIALECT_GA
        self._apply_transcription(audio_input.get('transcription'))
        errors = self._start()
        if errors:
            return errors
        return [{'type': 'session.updated', 'session': self._session_object()}]

    def _append(self, audio: Any, client_event_id: Optional[str]) -> List[Dict[str, Any]]:
        if self.finishing:
            return []
        if not isinstance(audio, str) or not audio:
            return [error_event("Missing required parameter: 'audio'.", code='missing_audio',
                                client_event_id=client_event_id)]
        try:
            pcm = base64.b64decode(audio, validate=True)
        except (binascii.Error, ValueError):
            return [error_event("'audio' must be base64-encoded PCM16.", code='invalid_audio',
                                client_event_id=client_event_id)]
        events = self._start()
        if self.done:
            return events
        self.session.append_pcm16(pcm)
        if self.session.total_seconds() >= self.max_session_seconds:
            events += self.finish('max_duration')
        return events

    def _start(self) -> List[Dict[str, Any]]:
        if self.session is not None or self.done:
            return []
        model_dtype = ASR_MODEL_MAPPING.get(self.model.lower())
        if model_dtype is None:
            logger.warning(f"Unknown realtime transcription model '{self.model}', falling back to bf16")
            model_dtype = 'bf16'
        session = StreamingASRSession(context_info=self.prompt)
        transcription = self.transcription_service.start_live_transcription(
            session=session,
            limits=self.limits,
            model_dtype=model_dtype,
            offloading_config=self.offloading_config,
        )
        if transcription is None:
            self.done = True
            return [error_event("Server is busy processing another request. Please retry later.",
                                error_type='server_error', code='server_busy')]
        self.session = session
        self.request_id = transcription.request_id
        logger.info(f"Realtime session {self.session_id} started live transcription {self.request_id}")
        return []

    def finish(self, reason: str) -> List[Dict[str, Any]]:
        """Stop accepting audio; the engine transcribes what is left, saves the record and then closes the events"""
        if self.finishing:
            return []
        self.finishing = True
        if self.session is None:
            self.done = True
            return [{'type': STATUS_EVENT, 'status': 'completed', 'request_id': None, 'reason': reason}]
        self.session.request_finish()
        return [{'type': STATUS_EVENT, 'status': 'stopping', 'request_id': self.request_id, 'reason': reason}]

    def disconnect(self) -> None:
        """The client went away; whatever was received is still transcribed and saved"""
        if self.session is not None:
            self.session.request_finish()
        self.finishing = True

    def drain(self) -> List[Dict[str, Any]]:
        """Engine events ready to send"""
        if self.session is None:
            return []
        events, done = self.session.drain_events()
        if done:
            self.done = True
        return events
