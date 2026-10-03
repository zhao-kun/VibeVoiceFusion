"""
Unified Tasks API endpoints
Provides a single endpoint to check all running tasks (inference, training, and quick generation)
"""
from flask import jsonify, current_app
from backend.api import api_bp
from backend.inference.inference import InferenceBase
from backend.inference.quick_generate_inference import QuickGenerateInferenceBase
from backend.inference.asr_inference import ASRInferenceBase
from backend.training.engine import BaseTrainingEngine
from backend.task_manager.task import gm, Task
from backend.services.project_service import ProjectService
from backend.services.speaker_service import SpeakerService
from backend.services.dialog_session_service import DialogSessionService
from backend.i18n import t
from util.logger import get_logger

logger = get_logger(__name__)


def _enrich_generation_with_session_name(generation, dialog_service):
    """
    Enrich generation dict with session_name field.
    """
    gen_dict = generation.to_dict()

    try:
        session = dialog_service.get_session(generation.session_id)
        if session:
            gen_dict['session_name'] = session.name
        else:
            gen_dict['session_name'] = t('session.deleted')
    except Exception as e:
        logger.warning(f"Failed to get session name for session_id {generation.session_id}: {e}")
        gen_dict['session_name'] = t('session.deleted')

    return gen_dict


@api_bp.route('/tasks/current', methods=['GET'])
def get_current_task():
    """
    Get the current running task (either inference or training)

    Returns:
        200: JSON response with task information
        {
            "message": "...",
            "task": {
                "type": "inference" | "training" | "quick_generation" | "transcription" | null,
                "project_id": "..." | null,
                "data": { ... task-specific data ... } | null
            }
        }
    """
    task: Task = gm.get_current_task()

    if not task:
        return jsonify({
            'message': 'No active task at the moment',
            'task': {
                'type': None,
                'project_id': None,
                'data': None
            }
        }), 200

    unwrapped = task.unwrap()

    # Check if it's an inference task
    if isinstance(unwrapped, InferenceBase):
        inference: InferenceBase = unwrapped
        generation = inference.get_generation()
        project_id = generation.project_id

        # Try to enrich with session name
        gen_dict = None
        try:
            if project_id:
                project_service = ProjectService(
                    workspace_dir=current_app.config['WORKSPACE_DIR'],
                    meta_file_name=current_app.config['PROJECTS_META_FILE']
                )
                project_path = project_service.get_project_path(project_id)

                if project_path:
                    speaker_service = SpeakerService(project_path / 'voices')
                    dialog_service = DialogSessionService(
                        project_path / 'scripts',
                        speaker_service=speaker_service
                    )
                    gen_dict = _enrich_generation_with_session_name(generation, dialog_service)
                else:
                    gen_dict = generation.to_dict()
                    gen_dict['session_name'] = t('session.deleted')
            else:
                gen_dict = generation.to_dict()
                gen_dict['session_name'] = t('session.unknown')
        except Exception as e:
            logger.warning(f"Failed to enrich generation with session name: {e}")
            gen_dict = generation.to_dict()
            gen_dict['session_name'] = t('session.unknown')

        return jsonify({
            'message': 'Current inference task retrieved successfully',
            'task': {
                'type': 'inference',
                'project_id': project_id,
                'data': gen_dict
            }
        }), 200

    # Check if it's a training task
    if isinstance(unwrapped, BaseTrainingEngine):
        training_engine: BaseTrainingEngine = unwrapped
        state = training_engine.state
        project_id = state.project_id

        state_dict = state.to_dict()
        state_dict['all_lora_files'] = state.get_all_lora_files()

        return jsonify({
            'message': 'Current training task retrieved successfully',
            'task': {
                'type': 'training',
                'project_id': project_id,
                'data': state_dict
            }
        }), 200

    # Check if it's a quick generation task
    if isinstance(unwrapped, QuickGenerateInferenceBase):
        quick_gen_inference: QuickGenerateInferenceBase = unwrapped
        quick_gen = quick_gen_inference.get_quick_generate()

        return jsonify({
            'message': 'Current quick generation task retrieved successfully',
            'task': {
                'type': 'quick_generation',
                'project_id': None,  # Quick generation has no project
                'data': quick_gen.to_dict()
            }
        }), 200

    if isinstance(unwrapped, ASRInferenceBase):
        transcription = unwrapped.get_transcription()

        return jsonify({
            'message': 'Current transcription task retrieved successfully',
            'task': {
                'type': 'transcription',
                'project_id': transcription.project_id,
                'data': transcription.to_dict()
            }
        }), 200

    # Unknown task type
    return jsonify({
        'message': 'Unknown task type running',
        'task': {
            'type': 'unknown',
            'project_id': None,
            'data': None
        }
    }), 200
