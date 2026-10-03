"""
ASR Task - wraps transcription inference for task manager
"""
from pathlib import Path
from typing import Any, Dict, List

from backend.task_manager.task import FAILURE_TYPE_GENERAL, Task
from backend.inference.asr_inference import ASRInferenceBase
from utils.file_handler import FileHandler
from util.logger import get_logger

logger = get_logger(__name__)


class ASRTask(Task):
    """Task wrapper for ASR inference"""

    def __init__(self, inference: ASRInferenceBase, file_handler: FileHandler, history_file_path: str):
        super().__init__(task_id=inference.request_id)
        self.inference = inference
        self.file_handler = file_handler
        self.history_file_path = history_file_path

    @classmethod
    def from_inference(cls, inference: ASRInferenceBase, file_handler: FileHandler,
                       history_file_path: str) -> 'ASRTask':
        return cls(inference, file_handler, history_file_path)

    def run(self):
        logger.info(f"Transcription task {self.inference.request_id} is created, now running")
        self.inference.run_inference()

    def task_failure(self, error_msg: str, failure_type: str = FAILURE_TYPE_GENERAL):
        logger.error(f"Transcription task {self.inference.request_id} failed: {error_msg}")
        self.inference.failure(error_msg, failure_type)

    def task_success(self, message: str):
        logger.info(f"Transcription task completed successfully: {message}")
        self.inference.success(message)

    def unwrap(self) -> ASRInferenceBase:
        return self.inference

    def task_appended(self, message: str):
        # The service adds the history record once the task is accepted
        logger.info(f"Transcription task {self.task_id} appended to queue")

    def _task_finalize(self):
        """Update history with final transcription info"""
        try:
            self._update_history(self.inference.generation_info())
            self.inference.finalize()
            del self.inference
        except Exception as e:
            logger.warning(f"Failed to finalize transcription task {self.task_id}: {e}")

    def _load_history(self) -> List[Dict[str, Any]]:
        try:
            data = self.file_handler.read_json(self.history_file_path)
            if isinstance(data, list):
                return data
            return []
        except FileNotFoundError:
            return []
        except Exception as e:
            raise RuntimeError(f"Failed to load transcription history: {e}")

    def _save_history(self, history: List[Dict[str, Any]]) -> None:
        try:
            self.file_handler.write_json(Path(self.history_file_path), history)
        except Exception as e:
            raise RuntimeError(f"Failed to save transcription history: {e}")

    def _update_history(self, record: Dict[str, Any]) -> None:
        history = self._load_history()
        for i, item in enumerate(history):
            if item.get('request_id') == record.get('request_id'):
                history[i] = record
                break
        self._save_history(history)
