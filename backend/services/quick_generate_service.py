"""
Quick Generate Service - handles voice generation without project setup
"""
import json
import os
import shutil
from datetime import datetime
from pathlib import Path
from typing import List, Dict, Any, Optional
from uuid import uuid4

from utils.file_handler import FileHandler
from backend.models.quick_generate import (
    QuickGenerate, QuickGenerateDetails, QuickGenerateItem,
    detect_mode, parse_dialogue_speakers
)
from backend.task_manager.task import gm
from backend.task_manager.quick_generate_task import QuickGenerateTask
from backend.inference.quick_generate_inference import QuickGenerateInferenceBase
from config.configuration_vibevoice import InferencePhase
from util.logger import get_logger

logger = get_logger(__name__)


class QuickGenerateService:
    """Service for handling quick voice generation without project setup"""

    HISTORY_FILE = 'history.json'
    VOICES_DIR = 'voices'
    OUTPUTS_DIR = 'outputs'

    def __init__(self, workspace_dir: Path, fake_model: bool = False):
        """
        Initialize quick generate service.

        Args:
            workspace_dir: Path to workspace directory
            fake_model: Use fake inference engine for testing
        """
        self.workspace_dir = Path(workspace_dir)
        self.quick_generate_dir = self.workspace_dir / '_quick-generate'
        self.voices_dir = self.quick_generate_dir / self.VOICES_DIR
        self.outputs_dir = self.quick_generate_dir / self.OUTPUTS_DIR
        self.history_file = self.quick_generate_dir / self.HISTORY_FILE
        self.file_handler = FileHandler()
        self.fake_model = fake_model

        # Ensure directories exist
        self._ensure_directories()

    def _ensure_directories(self):
        """Ensure all required directories exist"""
        self.file_handler.ensure_directory(self.quick_generate_dir)
        self.file_handler.ensure_directory(self.voices_dir)
        self.file_handler.ensure_directory(self.outputs_dir)

        # Initialize history file if it doesn't exist
        if not self.history_file.exists():
            self._save_history([])

    def _load_history(self) -> List[Dict[str, Any]]:
        """Load generation history from JSON file"""
        try:
            data = self.file_handler.read_json(self.history_file)
            if isinstance(data, list):
                return data
            return []
        except FileNotFoundError:
            return []
        except Exception as e:
            logger.error(f"Failed to load quick generate history: {e}")
            return []

    def _save_history(self, history: List[Dict[str, Any]]) -> None:
        """Atomically save history to JSON file"""
        try:
            self.file_handler.write_json(self.history_file, history)
        except Exception as e:
            raise RuntimeError(f"Failed to save quick generate history: {e}")

    def _add_to_history(self, quick_gen: QuickGenerate) -> None:
        """Add a quick generate record to history"""
        history = self._load_history()
        history.insert(0, quick_gen.to_dict())  # Add to beginning
        self._save_history(history)

    def _update_history(self, request_id: str, updates: Dict[str, Any]) -> None:
        """Update a specific record in history"""
        history = self._load_history()
        for item in history:
            if item.get('request_id') == request_id:
                item.update(updates)
                item['updated_at'] = datetime.utcnow().isoformat()
                break
        self._save_history(history)

    def save_voice_file(self, file_data: bytes, original_filename: str) -> str:
        """
        Save uploaded voice file to voices directory.

        Args:
            file_data: Binary file data
            original_filename: Original filename for extension

        Returns:
            Saved filename (UUID-based)
        """
        # Get file extension
        ext = Path(original_filename).suffix.lower()
        if not ext:
            ext = '.wav'

        # Generate unique filename
        voice_filename = f"{uuid4().hex}{ext}"
        voice_path = self.voices_dir / voice_filename

        # Save file
        with open(voice_path, 'wb') as f:
            f.write(file_data)

        return voice_filename

    def save_preset_voice_file(self, preset_path: Path) -> str:
        """
        Copy a preset voice into the voices directory once and reuse that copy on later requests.

        Returns:
            Saved filename (stable per preset)
        """
        voice_filename = f"preset-{preset_path.name}"
        voice_path = self.voices_dir / voice_filename
        if not voice_path.exists() or preset_path.stat().st_mtime > voice_path.stat().st_mtime:
            tmp_path = voice_path.with_name(f"{voice_filename}.{uuid4().hex}.tmp")
            shutil.copyfile(preset_path, tmp_path)
            os.replace(tmp_path, voice_path)
        return voice_filename

    def start_generation(self, voice_files: List[str], text: str,
                        seeds: int = 42,
                        batch_size: int = 1,
                        cfg_scale: float = 1.3,
                        model_dtype: str = "bf16",
                        attn_implementation: str = "sdpa",
                        offloading_config: Optional[Dict[str, Any]] = None) -> Optional[QuickGenerate]:
        """
        Start a quick generation task.

        Args:
            voice_files: List of filenames of uploaded voices in voices directory (1-4 files)
            text: Text to generate
            seeds: Random seed
            batch_size: Number of generations (1-20)
            cfg_scale: CFG scale
            model_dtype: Model dtype
            attn_implementation: Attention implementation
            offloading_config: Offloading configuration

        Returns:
            QuickGenerate object if started, None if task manager is busy
        """
        request_id = uuid4().hex

        # Create output directory for this request
        request_output_dir = self.outputs_dir / request_id
        self.file_handler.ensure_directory(request_output_dir)

        # Create quick generate object
        quick_gen = QuickGenerate.create(
            request_id=request_id,
            voice_files=voice_files,
            text=text,
            seeds=seeds,
            batch_size=batch_size,
            cfg_scale=cfg_scale,
            model_dtype=model_dtype,
            attn_implementation=attn_implementation,
            offloading=offloading_config
        )

        # Store offloading config in details
        if offloading_config:
            quick_gen.details.offloading_config = offloading_config

        # Create inference engine with voice paths
        voice_paths = [str(self.voices_dir / vf) for vf in voice_files]
        inference = QuickGenerateInferenceBase.create(
            quick_gen=quick_gen,
            voice_paths=voice_paths,
            output_dir=str(request_output_dir),
            offload_config=offloading_config,
            fake=self.fake_model
        )

        # Create task
        task = QuickGenerateTask.from_inference(
            inference=inference,
            file_handler=self.file_handler,
            history_file_path=str(self.history_file)
        )

        # Try to add task
        if gm.add_task(task):
            # Add to history
            self._add_to_history(quick_gen)
            return quick_gen
        else:
            # Clean up output directory
            if request_output_dir.exists():
                shutil.rmtree(request_output_dir)
            return None

    def get_generation(self, request_id: str) -> Optional[QuickGenerate]:
        """
        Get a specific generation by request ID.

        First checks if it's the current running task, then checks history.

        Args:
            request_id: Generation request ID

        Returns:
            QuickGenerate object or None if not found
        """
        # Check if it's the current running task
        task = gm.get_current_task()
        if task:
            inference = task.unwrap()
            if isinstance(inference, QuickGenerateInferenceBase):
                current_gen = inference.get_quick_generate()
                if current_gen and current_gen.request_id == request_id:
                    return current_gen

        # Check history
        history = self._load_history()
        for item in history:
            if item.get('request_id') == request_id:
                return QuickGenerate.from_dict(item)

        return None

    def list_history(self, limit: int = 20, offset: int = 0) -> Dict[str, Any]:
        """
        List generation history with pagination.

        Args:
            limit: Maximum items to return
            offset: Number of items to skip

        Returns:
            Dictionary with generations list, count, and total
        """
        history = self._load_history()
        total = len(history)

        # Apply pagination
        paginated = history[offset:offset + limit]

        # Convert to QuickGenerate objects (without full text for preview)
        generations = []
        for item in paginated:
            gen = QuickGenerate.from_dict(item)
            generations.append({
                'request_id': gen.request_id,
                'status': gen.status,  # Already a string
                'detected_mode': gen.detected_mode,
                'text_preview': gen.text_preview,
                'batch_size': gen.batch_size,
                'created_at': gen.created_at,
                'completed_at': gen.completed_at
            })

        return {
            'generations': generations,
            'count': len(generations),
            'total': total
        }

    def delete_generation(self, request_id: str) -> bool:
        """
        Delete a generation and its associated files.

        Args:
            request_id: Generation request ID

        Returns:
            True if deleted, False if not found
        """
        history = self._load_history()

        # Find the generation
        gen_data = next((g for g in history if g.get('request_id') == request_id), None)
        if not gen_data:
            return False

        # Delete output directory
        output_dir = self.outputs_dir / request_id
        if output_dir.exists():
            shutil.rmtree(output_dir)

        # Delete voice files if they exist and are unique to this generation
        # Support both old (voice_file) and new (voice_files) format
        voice_files = gen_data.get('voice_files', [])
        if not voice_files and gen_data.get('voice_file'):
            voice_files = [gen_data.get('voice_file')]

        for voice_file in voice_files:
            if voice_file:
                voice_path = self.voices_dir / voice_file
                # Check if any other generation uses this voice file
                other_uses = any(
                    (voice_file in g.get('voice_files', []) or g.get('voice_file') == voice_file)
                    and g.get('request_id') != request_id
                    for g in history
                )
                if not other_uses and voice_path.exists():
                    voice_path.unlink()

        # Remove from history
        updated_history = [g for g in history if g.get('request_id') != request_id]
        self._save_history(updated_history)

        return True

    def get_audio_path(self, request_id: str, item_index: int = 0) -> Optional[Path]:
        """
        Get the path to a generated audio file.

        Args:
            request_id: Generation request ID
            item_index: Index for multi-generation (0 for single)

        Returns:
            Path to audio file or None if not found
        """
        gen = self.get_generation(request_id)
        if not gen:
            return None

        # Check if generation is completed
        if gen.status != InferencePhase.COMPLETED:
            return None

        # Get output directory
        output_dir = self.outputs_dir / request_id

        # For multi-generation, check details
        if gen.is_multi_generation and gen.details and gen.details.generation_items:
            if item_index < 0 or item_index >= len(gen.details.generation_items):
                return None
            item = gen.details.generation_items[item_index]
            audio_path = item.audio_path if isinstance(item, QuickGenerateItem) else item.get('audio_path')
            if audio_path:
                full_path = Path(audio_path)
                if full_path.exists():
                    return full_path
                # Try relative path
                full_path = output_dir / Path(audio_path).name
                if full_path.exists():
                    return full_path

        # For single generation or fallback
        audio_file = output_dir / f"{request_id}_0.wav"
        if audio_file.exists():
            return audio_file

        return None

    def get_voice_path(self, voice_filename: str) -> Optional[Path]:
        """
        Get the path to a voice file.

        Args:
            voice_filename: Voice filename

        Returns:
            Path to voice file or None if not found
        """
        voice_path = self.voices_dir / voice_filename
        if voice_path.exists():
            return voice_path
        return None

    def cleanup_old_data(self, voice_days: int = 7, output_days: int = 30, history_days: int = 90):
        """
        Clean up old quick generate data based on retention policy.

        Args:
            voice_days: Delete voice files older than this
            output_days: Delete output audio older than this
            history_days: Delete history entries older than this
        """
        now = datetime.utcnow()
        history = self._load_history()
        updated_history = []

        for gen_data in history:
            created_at_str = gen_data.get('created_at', '')
            try:
                created_at = datetime.fromisoformat(created_at_str.replace('Z', '+00:00'))
            except (ValueError, TypeError):
                # Keep entries we can't parse
                updated_history.append(gen_data)
                continue

            age_days = (now - created_at.replace(tzinfo=None)).days

            # Delete old outputs
            if age_days > output_days:
                request_id = gen_data.get('request_id')
                if request_id:
                    output_dir = self.outputs_dir / request_id
                    if output_dir.exists():
                        shutil.rmtree(output_dir)
                        logger.info(f"Cleaned up old output: {request_id}")

            # Delete old voice files (only if no recent generations use them)
            if age_days > voice_days:
                voice_file = gen_data.get('voice_file')
                if voice_file:
                    # Check if any recent generation uses this voice
                    recent_uses = any(
                        g.get('voice_file') == voice_file
                        for g in history
                        if g.get('request_id') != gen_data.get('request_id')
                    )
                    if not recent_uses:
                        voice_path = self.voices_dir / voice_file
                        if voice_path.exists():
                            voice_path.unlink()
                            logger.info(f"Cleaned up old voice file: {voice_file}")

            # Keep or delete history entry
            if age_days <= history_days:
                updated_history.append(gen_data)
            else:
                logger.info(f"Removed old history entry: {gen_data.get('request_id')}")

        # Save updated history
        if len(updated_history) != len(history):
            self._save_history(updated_history)
