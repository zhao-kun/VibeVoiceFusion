"""
OpenAI-compatible Realtime transcription endpoint (WebSocket at /v1/realtime)

Streams 24 kHz PCM16 audio in, transcription events out. Browsers cannot set the Authorization header,
so the API key is also accepted as the `openai-insecure-api-key.<key>` subprotocol, as OpenAI does.
"""
import json
import time
from uuid import uuid4

from flask import current_app, request
from flask_sock import Sock
from simple_websocket import ConnectionClosed

from backend.api.openai_compat import openai_bp, _get_service
from backend.inference.asr_inference import OFFLOAD_PRESET_GPU_LAYERS
from backend.services.openai_realtime_service import RealtimeTranscriptionHandler, error_event
from backend.services.project_service import ProjectService
from backend.services.transcription_service import TranscriptionService
from backend.task_manager.task import gm
from util.logger import get_logger

logger = get_logger(__name__)

sock = Sock()

API_KEY_SUBPROTOCOL_PREFIX = 'openai-insecure-api-key.'
RECEIVE_POLL_SECONDS = 0.05
CLOSE_POLICY_VIOLATION = 1008
CLOSE_TRY_AGAIN_LATER = 1013


def _realtime_auth_header():
    auth_header = request.headers.get('Authorization')
    if auth_header:
        return auth_header
    for protocol in request.headers.get('Sec-WebSocket-Protocol', '').split(','):
        protocol = protocol.strip()
        if protocol.startswith(API_KEY_SUBPROTOCOL_PREFIX):
            return f"Bearer {protocol[len(API_KEY_SUBPROTOCOL_PREFIX):]}"
    return None


def _send(ws, event):
    event.setdefault('event_id', f"event_{uuid4().hex[:24]}")
    ws.send(json.dumps(event, ensure_ascii=False))


def _reject(ws, event, code):
    _send(ws, event)
    ws.close(reason=code, message=event['error']['message'][:120])


def _transcription_service(project_id):
    """Standalone storage, or the project's storage when project_id is given; None if the project is unknown"""
    workspace_dir = current_app.config['WORKSPACE_DIR']
    fake_model = current_app.config.get('FAKE_MODEL', False)
    if not project_id:
        return TranscriptionService.for_workspace(workspace_dir, fake_model=fake_model)
    project_service = ProjectService(workspace_dir, current_app.config['PROJECTS_META_FILE'])
    project_path = project_service.get_project_path(project_id)
    if not project_path:
        return None
    return TranscriptionService.for_project(project_path, project_id, fake_model=fake_model)


@sock.route('/realtime', bp=openai_bp)
def realtime(ws):
    """
    OpenAI Realtime transcription session.

    Query parameters (VibeVoice extensions, optional):
        project_id   // Save into this project's transcription history instead of the standalone one
        offloading   // Layer offloading preset: balanced, aggressive or extreme
    """
    if not _get_service().validate_api_key(_realtime_auth_header()):
        _reject(ws, error_event("Invalid API key provided.", error_type='authentication_error',
                                code='invalid_api_key'), CLOSE_POLICY_VIOLATION)
        return

    project_id = request.args.get('project_id') or None
    service = _transcription_service(project_id)
    if service is None:
        _reject(ws, error_event(f"Project '{project_id}' not found.", code='project_not_found'),
                CLOSE_POLICY_VIOLATION)
        return

    offloading_config = None
    offloading = request.args.get('offloading')
    if offloading:
        if offloading not in OFFLOAD_PRESET_GPU_LAYERS:
            supported = ', '.join(OFFLOAD_PRESET_GPU_LAYERS)
            _reject(ws, error_event(f"Unsupported offloading preset '{offloading}'. Supported presets: {supported}",
                                    code='invalid_offloading'), CLOSE_POLICY_VIOLATION)
            return
        offloading_config = {'enabled': True, 'mode': 'preset', 'preset': offloading}

    if gm.has_task():
        _reject(ws, error_event("Server is busy processing another request. Please retry later.",
                                error_type='server_error', code='server_busy'), CLOSE_TRY_AGAIN_LATER)
        return

    config = current_app.config
    handler = RealtimeTranscriptionHandler(
        transcription_service=service,
        limits={
            'max_window_seconds': config['STREAMING_ASR_MAX_WINDOW_SECONDS'],
            'silence_commit_seconds': config['STREAMING_ASR_SILENCE_COMMIT_SECONDS'],
            'silence_rms': config['STREAMING_ASR_SILENCE_RMS'],
        },
        max_session_seconds=config['STREAMING_ASR_MAX_SESSION_SECONDS'],
        offloading_config=offloading_config,
    )
    idle_timeout = config['STREAMING_ASR_IDLE_TIMEOUT_SECONDS']
    last_message_at = time.time()

    try:
        _send(ws, handler.created_event())
        while True:
            for event in handler.drain():
                _send(ws, event)
            if handler.done:
                break

            message = ws.receive(timeout=RECEIVE_POLL_SECONDS)
            if message is None:
                if not handler.finishing and time.time() - last_message_at > idle_timeout:
                    for event in handler.finish('idle_timeout'):
                        _send(ws, event)
                continue

            last_message_at = time.time()
            for event in handler.handle(message):
                _send(ws, event)
    except ConnectionClosed:
        logger.info(f"Realtime client of session {handler.session_id} disconnected")
        handler.disconnect()
        return

    ws.close()
