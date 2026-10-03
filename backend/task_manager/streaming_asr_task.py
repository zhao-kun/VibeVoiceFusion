"""
Streaming ASR Task - holds the GPU slot for the lifetime of a live transcription session
"""
from backend.task_manager.asr_task import ASRTask
from backend.task_manager.task import FAILURE_TYPE_GENERAL
from backend.inference.streaming_asr_inference import STATUS_EVENT
from util.logger import get_logger

logger = get_logger(__name__)


class StreamingASRTask(ASRTask):
    """ASR task whose outcome is also reported to the connected WebSocket client"""

    def task_failure(self, error_msg: str, failure_type: str = FAILURE_TYPE_GENERAL):
        super().task_failure(error_msg, failure_type)
        self.inference.session.emit({
            'type': 'error',
            'error': {'type': 'server_error', 'code': failure_type, 'message': error_msg},
        })

    def _task_finalize(self):
        # Captured first because the parent releases the inference object
        session = self.inference.session
        transcription = self.inference.get_transcription()
        super()._task_finalize()
        event = {'type': STATUS_EVENT, 'status': transcription.status, 'request_id': transcription.request_id}
        if transcription.error_message:
            event['error'] = transcription.error_message
        session.emit(event)
        session.close_events()
