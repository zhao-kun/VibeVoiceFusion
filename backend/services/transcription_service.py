"""
Transcription Service - ASR transcription for standalone use and inside projects
"""
from pathlib import Path
from typing import List, Dict, Any, Optional
from uuid import uuid4

from utils.file_handler import FileHandler
from backend.models.transcription import Transcription
from backend.task_manager.task import gm
from backend.task_manager.asr_task import ASRTask
from backend.inference.asr_inference import ASRInferenceBase
from util.logger import get_logger

logger = get_logger(__name__)

STANDALONE_DIR_NAME = '_transcriptions'
PROJECT_DIR_NAME = 'transcriptions'


class TranscriptionService:
    """Service for ASR transcriptions stored under one base directory.

    Standalone transcriptions live in workspace/_transcriptions, project ones in {project}/transcriptions.
    """

    HISTORY_FILE = 'history.json'
    AUDIO_DIR = 'audio'

    def __init__(self, base_dir: Path, project_id: Optional[str] = None, fake_model: bool = False):
        self.base_dir = Path(base_dir)
        self.project_id = project_id
        self.audio_dir = self.base_dir / self.AUDIO_DIR
        self.history_file = self.base_dir / self.HISTORY_FILE
        self.file_handler = FileHandler()
        self.fake_model = fake_model

        self._ensure_directories()

    @classmethod
    def for_workspace(cls, workspace_dir: Path, fake_model: bool = False) -> 'TranscriptionService':
        return cls(Path(workspace_dir) / STANDALONE_DIR_NAME, project_id=None, fake_model=fake_model)

    @classmethod
    def for_project(cls, project_path: Path, project_id: str, fake_model: bool = False) -> 'TranscriptionService':
        return cls(Path(project_path) / PROJECT_DIR_NAME, project_id=project_id, fake_model=fake_model)

    def _ensure_directories(self):
        self.file_handler.ensure_directory(self.base_dir)
        self.file_handler.ensure_directory(self.audio_dir)
        if not self.history_file.exists():
            self._save_history([])

    def _load_history(self) -> List[Dict[str, Any]]:
        try:
            data = self.file_handler.read_json(self.history_file)
            if isinstance(data, list):
                return data
            return []
        except FileNotFoundError:
            return []
        except Exception as e:
            logger.error(f"Failed to load transcription history: {e}")
            return []

    def _save_history(self, history: List[Dict[str, Any]]) -> None:
        try:
            self.file_handler.write_json(self.history_file, history)
        except Exception as e:
            raise RuntimeError(f"Failed to save transcription history: {e}")

    def _current_transcription(self) -> Optional[Transcription]:
        """The running transcription if it belongs to this storage root"""
        task = gm.get_current_task()
        if not task:
            return None
        inference = task.unwrap()
        if not isinstance(inference, ASRInferenceBase):
            return None
        current = inference.get_transcription()
        if current.project_id != self.project_id:
            return None
        return current

    def save_audio_file(self, file_data: bytes, original_filename: str) -> str:
        """Save an uploaded audio file and return its stored (UUID-based) filename"""
        ext = Path(original_filename).suffix.lower() or '.wav'
        audio_filename = f"{uuid4().hex}{ext}"
        with open(self.audio_dir / audio_filename, 'wb') as f:
            f.write(file_data)
        return audio_filename

    def start_transcription(self, audio_file: str, original_filename: str,
                            context_info: Optional[str] = None,
                            model_dtype: str = "bf16",
                            max_new_tokens: int = 8192,
                            temperature: float = 0.0,
                            top_p: float = 1.0,
                            repetition_penalty: float = 1.0,
                            seeds: int = 42,
                            offloading_config: Optional[Dict[str, Any]] = None) -> Optional[Transcription]:
        """
        Start a transcription task.

        Returns:
            Transcription object if started, None if task manager is busy
        """
        transcription = Transcription.create(
            request_id=uuid4().hex,
            audio_file=audio_file,
            original_filename=original_filename,
            project_id=self.project_id,
            context_info=context_info,
            model_dtype=model_dtype,
            max_new_tokens=max_new_tokens,
            temperature=temperature,
            top_p=top_p,
            repetition_penalty=repetition_penalty,
            seeds=seeds,
            offloading=offloading_config,
        )

        inference = ASRInferenceBase.create(
            transcription=transcription,
            audio_path=str(self.audio_dir / audio_file),
            offload_config=offloading_config,
            fake=self.fake_model,
        )
        task = ASRTask.from_inference(
            inference=inference,
            file_handler=self.file_handler,
            history_file_path=str(self.history_file),
        )

        # Recorded before queueing so a task that finishes immediately still finds its record to update
        history = self._load_history()
        history.insert(0, transcription.to_dict())
        self._save_history(history)

        if gm.add_task(task):
            return transcription

        self._save_history([h for h in self._load_history() if h.get('request_id') != transcription.request_id])
        self._delete_audio_if_unused(audio_file, [])
        return None

    def get_transcription(self, request_id: str) -> Optional[Transcription]:
        """Get a transcription, preferring the live state of the running task"""
        current = self._current_transcription()
        if current and current.request_id == request_id:
            return current

        for item in self._load_history():
            if item.get('request_id') == request_id:
                return Transcription.from_dict(item)
        return None

    def get_current_transcription(self) -> Optional[Transcription]:
        return self._current_transcription()

    def list_history(self, limit: int = 20, offset: int = 0) -> Dict[str, Any]:
        """List transcription summaries with pagination"""
        history = self._load_history()
        current = self._current_transcription()

        transcriptions = []
        for item in history[offset:offset + limit]:
            record = Transcription.from_dict(item)
            if current and current.request_id == record.request_id:
                record = current
            transcriptions.append({
                'request_id': record.request_id,
                'project_id': record.project_id,
                'status': record.status,
                'original_filename': record.original_filename,
                'audio_duration': record.audio_duration,
                'text_preview': record.text_preview,
                'segment_count': len(record.segments),
                'model_dtype': record.model_dtype,
                'created_at': record.created_at,
                'completed_at': record.completed_at,
            })

        return {
            'transcriptions': transcriptions,
            'count': len(transcriptions),
            'total': len(history),
        }

    def is_running(self, request_id: str) -> bool:
        current = self._current_transcription()
        return bool(current and current.request_id == request_id)

    def _delete_audio_if_unused(self, audio_file: str, remaining: List[Dict[str, Any]]) -> None:
        if not audio_file or any(h.get('audio_file') == audio_file for h in remaining):
            return
        audio_path = self.audio_dir / audio_file
        if audio_path.exists():
            audio_path.unlink()

    def delete_transcription(self, request_id: str) -> bool:
        """Delete a transcription record and its audio. Returns False if not found."""
        history = self._load_history()
        record = next((h for h in history if h.get('request_id') == request_id), None)
        if not record:
            return False

        remaining = [h for h in history if h.get('request_id') != request_id]
        self._save_history(remaining)
        self._delete_audio_if_unused(record.get('audio_file'), remaining)
        return True

    def delete_transcriptions_batch(self, request_ids: List[str]) -> Dict[str, Any]:
        deleted_ids, failed_ids = [], []
        for request_id in request_ids:
            if self.is_running(request_id):
                failed_ids.append(request_id)
                continue
            if self.delete_transcription(request_id):
                deleted_ids.append(request_id)
            else:
                failed_ids.append(request_id)
        return {
            'deleted_count': len(deleted_ids),
            'failed_count': len(failed_ids),
            'deleted_ids': deleted_ids,
            'failed_ids': failed_ids,
        }

    def get_audio_path(self, request_id: str) -> Optional[Path]:
        """Path to the source audio of a transcription"""
        transcription = self.get_transcription(request_id)
        if not transcription:
            return None
        audio_path = self.audio_dir / transcription.audio_file
        return audio_path if audio_path.exists() else None
