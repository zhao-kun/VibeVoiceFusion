"""
Transcription (ASR) data models
"""
from dataclasses import dataclass, asdict, field
from datetime import datetime
from typing import Optional, Dict, Any, List

from config.configuration_vibevoice import InferencePhase


@dataclass
class TranscriptionDetails:
    """Timing and token statistics of one transcription"""
    preprocessing_duration: Optional[float] = None
    model_load_duration: Optional[float] = None
    encode_time: Optional[float] = None
    prefill_time: Optional[float] = None
    decode_time: Optional[float] = None
    prompt_tokens: Optional[int] = None
    speech_tokens: Optional[int] = None
    offloading_config: Optional[Dict[str, Any]] = field(default_factory=dict)


@dataclass
class Transcription:
    """Transcription metadata model, shared by standalone and project-scoped transcriptions"""
    request_id: str
    audio_file: str  # Stored filename in the audio directory
    original_filename: str
    status: str  # InferencePhase constant
    model_dtype: str
    max_new_tokens: int
    temperature: float
    top_p: float
    repetition_penalty: float
    seeds: int
    created_at: str
    updated_at: str
    project_id: Optional[str] = None
    context_info: Optional[str] = None
    offloading: Optional[Dict[str, Any]] = None
    audio_duration: Optional[float] = None
    generated_tokens: int = 0
    reach_max_new_tokens: bool = False
    raw_text: Optional[str] = None
    segments: List[Dict[str, Any]] = field(default_factory=list)
    text_preview: Optional[str] = None
    percentage: Optional[float] = None
    details: Optional[TranscriptionDetails] = None
    error_message: Optional[str] = None
    completed_at: Optional[str] = None

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> 'Transcription':
        data = dict(data)
        if isinstance(data.get('details'), dict):
            data['details'] = TranscriptionDetails(**data['details'])
        return cls(**data)

    @classmethod
    def create(cls, request_id: str, audio_file: str, original_filename: str,
               project_id: Optional[str] = None,
               context_info: Optional[str] = None,
               model_dtype: str = "bf16",
               max_new_tokens: int = 8192,
               temperature: float = 0.0,
               top_p: float = 1.0,
               repetition_penalty: float = 1.0,
               seeds: int = 42,
               offloading: Optional[Dict[str, Any]] = None) -> 'Transcription':
        """Create a new pending transcription request"""
        now = datetime.utcnow().isoformat()
        return cls(
            request_id=request_id,
            audio_file=audio_file,
            original_filename=original_filename,
            status=InferencePhase.PENDING,
            model_dtype=model_dtype,
            max_new_tokens=max_new_tokens,
            temperature=temperature,
            top_p=top_p,
            repetition_penalty=repetition_penalty,
            seeds=seeds,
            created_at=now,
            updated_at=now,
            project_id=project_id,
            context_info=context_info,
            offloading=offloading,
            details=TranscriptionDetails(offloading_config=offloading or {}),
        )


def segment_seconds(value: Any) -> Optional[float]:
    """Parse a segment timestamp, which the model may emit as a number or a numeric string."""
    if value is None:
        return None
    try:
        return float(value)
    except (TypeError, ValueError):
        return None


def transcript_plain_text(segments: List[Dict[str, Any]], raw_text: Optional[str] = None) -> str:
    """Join segment texts; falls back to the raw model output when it could not be parsed into segments."""
    texts = [str(seg.get('text', '')).strip() for seg in segments if str(seg.get('text', '')).strip()]
    if texts:
        return '\n'.join(texts)
    return (raw_text or '').strip()


def _format_timestamp(seconds: Optional[float]) -> str:
    if seconds is None:
        return '?'
    minutes, secs = divmod(seconds, 60)
    hours, minutes = divmod(int(minutes), 60)
    if hours:
        return f"{hours:d}:{minutes:02d}:{secs:05.2f}"
    return f"{minutes:02d}:{secs:05.2f}"


def transcript_readable_text(segments: List[Dict[str, Any]], raw_text: Optional[str] = None) -> str:
    """One line per segment with time range and speaker, used for the .txt download."""
    if not segments:
        return (raw_text or '').strip()
    lines = []
    for seg in segments:
        start = _format_timestamp(segment_seconds(seg.get('start_time')))
        end = _format_timestamp(segment_seconds(seg.get('end_time')))
        speaker = seg.get('speaker_id')
        prefix = f"[{start} - {end}]"
        if speaker is not None and speaker != '':
            prefix += f" Speaker {speaker}:"
        lines.append(f"{prefix} {str(seg.get('text', '')).strip()}")
    return '\n'.join(lines)
