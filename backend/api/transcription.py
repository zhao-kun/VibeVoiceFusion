"""
Transcription (ASR) API endpoints

Standalone transcriptions live under /transcriptions, project transcriptions under
/projects/<project_id>/transcriptions. Both share the same handlers and the single GPU task queue.
"""
import json
import mimetypes
import random
from typing import Optional, Tuple
from urllib.parse import quote

from flask import request, jsonify, current_app, send_file, Response

from backend.api import api_bp
from backend.api.quick_generate import ALLOWED_EXTENSIONS, _allowed_file, _validate_offloading_config
from backend.models.transcription import transcript_readable_text
from backend.services.project_service import ProjectService
from backend.services.transcription_service import TranscriptionService
from backend.i18n import t
from config.configuration_vibevoice import InferencePhase
from util.logger import get_logger

logger = get_logger(__name__)

MODEL_DTYPES = ('bf16', 'float8_e4m3fn')
MAX_NEW_TOKENS_LIMIT = 32768
MAX_CONTEXT_INFO_LENGTH = 2000
DOWNLOAD_FORMATS = ('json', 'txt')


class _ParamError(ValueError):
    pass


def _standalone_service() -> TranscriptionService:
    return TranscriptionService.for_workspace(
        current_app.config['WORKSPACE_DIR'],
        fake_model=current_app.config.get('FAKE_MODEL', False),
    )


def _project_service(project_id: str) -> Tuple[Optional[TranscriptionService], Optional[tuple]]:
    """Return (service, None) or (None, 404 response) when the project does not exist"""
    project_service = ProjectService(workspace_dir=current_app.config['WORKSPACE_DIR'],
                                     meta_file_name=current_app.config['PROJECTS_META_FILE'])
    project_path = project_service.get_project_path(project_id)
    if not project_path:
        return None, (jsonify({
            'error': t('errors.not_found'),
            'message': t('errors.project_not_found')
        }), 404)
    return TranscriptionService.for_project(
        project_path, project_id, fake_model=current_app.config.get('FAKE_MODEL', False)
    ), None


def _internal_error(action: str, e: Exception):
    logger.error(f"Error {action}: {e}", exc_info=True)
    return jsonify({
        'error': t('errors.internal_error'),
        'message': str(e)
    }), 500


def _not_found(message_key: str = 'errors.transcription_not_found'):
    return jsonify({
        'error': t('errors.not_found'),
        'message': t(message_key)
    }), 404


def _number_param(name: str, cast, default, minimum, maximum):
    raw = request.form.get(name)
    if raw is None or raw.strip() == '':
        return default
    try:
        value = cast(raw)
    except ValueError:
        raise _ParamError(t('validation.invalid_format', field=name))
    if value < minimum or value > maximum:
        raise _ParamError(t('validation.out_of_range', field=name, min=minimum, max=maximum))
    return value


def _parse_start_params() -> dict:
    """Parse and validate the optional transcription parameters of a POST form"""
    model_dtype = request.form.get('model_dtype', 'bf16')
    if model_dtype not in MODEL_DTYPES:
        raise _ParamError(t('validation.invalid_format', field='model_dtype'))

    context_info = request.form.get('context_info', '').strip() or None
    if context_info and len(context_info) > MAX_CONTEXT_INFO_LENGTH:
        raise _ParamError(t('errors.transcription_context_too_long', max=MAX_CONTEXT_INFO_LENGTH))

    seeds = request.form.get('seeds')
    try:
        seeds = int(seeds) if seeds else random.randint(0, 2**32 - 1)
    except ValueError:
        raise _ParamError(t('validation.invalid_format', field='seeds'))

    offloading = None
    offloading_str = request.form.get('offloading', '')
    if offloading_str:
        try:
            offloading = _validate_offloading_config(json.loads(offloading_str))
        except (json.JSONDecodeError, ValueError) as e:
            raise _ParamError(str(e))

    return {
        'context_info': context_info,
        'model_dtype': model_dtype,
        'max_new_tokens': _number_param('max_new_tokens', int, 8192, 1, MAX_NEW_TOKENS_LIMIT),
        'temperature': _number_param('temperature', float, 0.0, 0.0, 2.0),
        'top_p': _number_param('top_p', float, 1.0, 0.01, 1.0),
        'repetition_penalty': _number_param('repetition_penalty', float, 1.0, 0.5, 2.0),
        'seeds': seeds,
        'offloading_config': offloading,
    }


def _start(service: TranscriptionService):
    audio = request.files.get('audio_file')
    if audio is None or not audio.filename:
        return jsonify({
            'error': t('errors.bad_request'),
            'message': t('errors.transcription_audio_required')
        }), 400
    if not _allowed_file(audio.filename):
        return jsonify({
            'error': t('errors.bad_request'),
            'message': t('errors.invalid_file_type', formats=', '.join(sorted(ALLOWED_EXTENSIONS)))
        }), 400

    try:
        params = _parse_start_params()
    except _ParamError as e:
        return jsonify({
            'error': t('errors.validation_error'),
            'message': str(e)
        }), 400

    audio_filename = service.save_audio_file(audio.read(), audio.filename)
    transcription = service.start_transcription(
        audio_file=audio_filename,
        original_filename=audio.filename,
        **params,
    )
    if not transcription:
        return jsonify({
            'error': t('errors.conflict'),
            'message': t('errors.task_manager_busy')
        }), 409

    return jsonify({
        'message': t('success.transcription_started'),
        'request_id': transcription.request_id,
        'project_id': transcription.project_id,
        'status': transcription.status
    }), 200


def _get(service: TranscriptionService, request_id: str):
    transcription = service.get_transcription(request_id)
    if not transcription:
        return _not_found()
    return jsonify(transcription.to_dict()), 200


def _current(service: TranscriptionService):
    transcription = service.get_current_transcription()
    if transcription:
        return jsonify({
            'message': 'Current transcription retrieved successfully',
            'transcription': transcription.to_dict()
        }), 200
    return jsonify({
        'message': 'No active transcription task',
        'transcription': None
    }), 200


def _history(service: TranscriptionService):
    try:
        limit = int(request.args.get('limit', '20'))
        offset = int(request.args.get('offset', '0'))
    except ValueError:
        limit, offset = 20, 0
    return jsonify(service.list_history(limit=max(1, limit), offset=max(0, offset))), 200


def _delete(service: TranscriptionService, request_id: str):
    if service.is_running(request_id):
        return jsonify({
            'error': t('errors.conflict'),
            'message': t('errors.transcription_running')
        }), 409
    if not service.delete_transcription(request_id):
        return _not_found()
    return jsonify({
        'message': t('success.transcription_deleted'),
        'request_id': request_id
    }), 200


def _batch_delete(service: TranscriptionService):
    data = request.get_json(silent=True) or {}
    request_ids = data.get('request_ids')
    if not request_ids or not isinstance(request_ids, list):
        return jsonify({
            'error': t('errors.bad_request'),
            'message': t('errors.validation_error')
        }), 400
    result = service.delete_transcriptions_batch(request_ids)
    return jsonify({'message': t('success.transcriptions_deleted'), **result}), 200


def _audio(service: TranscriptionService, request_id: str):
    audio_path = service.get_audio_path(request_id)
    if not audio_path:
        return _not_found('errors.transcription_audio_not_found')
    mimetype = mimetypes.guess_type(audio_path.name)[0] or 'application/octet-stream'
    return send_file(str(audio_path), mimetype=mimetype)


def _download(service: TranscriptionService, request_id: str):
    output_format = request.args.get('format', 'json').lower()
    if output_format not in DOWNLOAD_FORMATS:
        return jsonify({
            'error': t('errors.bad_request'),
            'message': t('validation.invalid_format', field='format')
        }), 400

    transcription = service.get_transcription(request_id)
    if not transcription:
        return _not_found()
    if transcription.status != InferencePhase.COMPLETED:
        return _not_found('errors.transcription_not_completed')

    base_name = transcription.original_filename.rsplit('.', 1)[0] or request_id
    if output_format == 'txt':
        body = transcript_readable_text(transcription.segments, transcription.raw_text)
        mimetype = 'text/plain; charset=utf-8'
    else:
        body = json.dumps({
            'request_id': transcription.request_id,
            'file': transcription.original_filename,
            'audio_duration': transcription.audio_duration,
            'context_info': transcription.context_info,
            'raw_text': transcription.raw_text,
            'segments': transcription.segments,
        }, ensure_ascii=False, indent=2)
        mimetype = 'application/json; charset=utf-8'

    response = Response(body, mimetype=mimetype)
    if request.args.get('download', 'true').lower() == 'true':
        response.headers['Content-Disposition'] = (
            f"attachment; filename*=UTF-8''{quote(f'{base_name}_transcript.{output_format}')}"
        )
    return response


# ============ Standalone transcriptions ============

@api_bp.route('/transcriptions', methods=['POST'])
def start_transcription():
    """
    Start a standalone transcription (multipart/form-data).

    Form Data:
        - audio_file: Audio file (required)
        - context_info, model_dtype, max_new_tokens, temperature, top_p,
          repetition_penalty, seeds, offloading (JSON string): optional
    """
    try:
        return _start(_standalone_service())
    except Exception as e:
        return _internal_error('starting transcription', e)


@api_bp.route('/transcriptions/current', methods=['GET'])
def get_current_transcription():
    """Get the running standalone transcription (200 with null when none)"""
    try:
        return _current(_standalone_service())
    except Exception as e:
        return _internal_error('getting current transcription', e)


@api_bp.route('/transcriptions/history', methods=['GET'])
def list_transcription_history():
    """List standalone transcriptions (query: limit, offset)"""
    try:
        return _history(_standalone_service())
    except Exception as e:
        return _internal_error('listing transcription history', e)


@api_bp.route('/transcriptions/batch-delete', methods=['POST'])
def batch_delete_transcriptions():
    """Delete several standalone transcriptions (body: {"request_ids": [...]})"""
    try:
        return _batch_delete(_standalone_service())
    except Exception as e:
        return _internal_error('batch deleting transcriptions', e)


@api_bp.route('/transcriptions/<request_id>', methods=['GET'])
def get_transcription(request_id: str):
    """Get a standalone transcription with live progress"""
    try:
        return _get(_standalone_service(), request_id)
    except Exception as e:
        return _internal_error('getting transcription', e)


@api_bp.route('/transcriptions/<request_id>', methods=['DELETE'])
def delete_transcription(request_id: str):
    """Delete a standalone transcription and its audio"""
    try:
        return _delete(_standalone_service(), request_id)
    except Exception as e:
        return _internal_error('deleting transcription', e)


@api_bp.route('/transcriptions/<request_id>/audio', methods=['GET'])
def get_transcription_audio(request_id: str):
    """Serve the source audio of a standalone transcription"""
    try:
        return _audio(_standalone_service(), request_id)
    except Exception as e:
        return _internal_error('serving transcription audio', e)


@api_bp.route('/transcriptions/<request_id>/download', methods=['GET'])
def download_transcription(request_id: str):
    """Download a completed standalone transcript (query: format=json|txt, download=true|false)"""
    try:
        return _download(_standalone_service(), request_id)
    except Exception as e:
        return _internal_error('downloading transcription', e)


# ============ Project transcriptions ============

@api_bp.route('/projects/<project_id>/transcriptions', methods=['POST'])
def start_project_transcription(project_id: str):
    """Start a transcription inside a project (same form fields as /transcriptions)"""
    try:
        service, error = _project_service(project_id)
        return error or _start(service)
    except Exception as e:
        return _internal_error('starting project transcription', e)


@api_bp.route('/projects/<project_id>/transcriptions', methods=['GET'])
def list_project_transcriptions(project_id: str):
    """List a project's transcriptions (query: limit, offset)"""
    try:
        service, error = _project_service(project_id)
        return error or _history(service)
    except Exception as e:
        return _internal_error('listing project transcriptions', e)


@api_bp.route('/projects/<project_id>/transcriptions/current', methods=['GET'])
def get_current_project_transcription(project_id: str):
    """Get the running transcription if it belongs to this project (200 with null otherwise)"""
    try:
        service, error = _project_service(project_id)
        return error or _current(service)
    except Exception as e:
        return _internal_error('getting current project transcription', e)


@api_bp.route('/projects/<project_id>/transcriptions/batch-delete', methods=['POST'])
def batch_delete_project_transcriptions(project_id: str):
    """Delete several project transcriptions (body: {"request_ids": [...]})"""
    try:
        service, error = _project_service(project_id)
        return error or _batch_delete(service)
    except Exception as e:
        return _internal_error('batch deleting project transcriptions', e)


@api_bp.route('/projects/<project_id>/transcriptions/<request_id>', methods=['GET'])
def get_project_transcription(project_id: str, request_id: str):
    """Get a project transcription with live progress"""
    try:
        service, error = _project_service(project_id)
        return error or _get(service, request_id)
    except Exception as e:
        return _internal_error('getting project transcription', e)


@api_bp.route('/projects/<project_id>/transcriptions/<request_id>', methods=['DELETE'])
def delete_project_transcription(project_id: str, request_id: str):
    """Delete a project transcription and its audio"""
    try:
        service, error = _project_service(project_id)
        return error or _delete(service, request_id)
    except Exception as e:
        return _internal_error('deleting project transcription', e)


@api_bp.route('/projects/<project_id>/transcriptions/<request_id>/audio', methods=['GET'])
def get_project_transcription_audio(project_id: str, request_id: str):
    """Serve the source audio of a project transcription"""
    try:
        service, error = _project_service(project_id)
        return error or _audio(service, request_id)
    except Exception as e:
        return _internal_error('serving project transcription audio', e)


@api_bp.route('/projects/<project_id>/transcriptions/<request_id>/download', methods=['GET'])
def download_project_transcription(project_id: str, request_id: str):
    """Download a completed project transcript (query: format=json|txt, download=true|false)"""
    try:
        service, error = _project_service(project_id)
        return error or _download(service, request_id)
    except Exception as e:
        return _internal_error('downloading project transcription', e)
