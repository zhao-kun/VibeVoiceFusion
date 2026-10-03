"""
Streaming ASR - live transcription by rolling re-transcription of a growing audio buffer

VibeVoice-ASR is an offline model, so each pass re-transcribes the audio after the last commit point.
Segments that stay identical across two consecutive passes are committed and the window moves forward.
"""
import json
import os
import queue
import tempfile
import threading
import time
import wave
from datetime import datetime
from typing import Any, Callable, Dict, List, Optional, Tuple

import numpy as np
import torch

from backend.inference.asr_inference import (
    ASRInferenceBase, ASRInferenceEngine, FakeASRInferenceEngine, FakeASRProcessor, TranscriptionVisitor,
    _build_offload_config,
)
from backend.models.transcription import Transcription, segment_seconds, transcript_plain_text
from util.logger import get_logger

logger = get_logger(__name__)

SAMPLE_RATE = 24000
BYTES_PER_SAMPLE = 2
FRAME_SECONDS = 0.02
MIN_WINDOW_SECONDS = 1.0
MIN_STEP_SECONDS = 0.5
STREAMING_MAX_NEW_TOKENS = 2048

STATUS_EVENT = 'vibevoice.session.status'
HYPOTHESIS_EVENT = 'vibevoice.transcription.hypothesis'

_EVENTS_DONE = object()


def _normalize_text(text: Any) -> str:
    return ' '.join(str(text or '').split())


def _is_word_char(ch: str) -> bool:
    return ch.isascii() and ch.isalnum()


def stable_prefix(previous: str, current: str) -> str:
    """Common prefix of two hypotheses, cut back to a word boundary for space-separated scripts."""
    n = 0
    limit = min(len(previous), len(current))
    while n < limit and previous[n] == current[n]:
        n += 1
    longer = current if len(current) >= len(previous) else previous
    while 0 < n < len(longer) and _is_word_char(longer[n - 1]) and _is_word_char(longer[n]):
        n -= 1
    return current[:n]


def analyze_energy(audio: np.ndarray, silence_rms: float, silence_seconds: float) -> Dict[str, bool]:
    """Whether a window contains speech at all, and whether it ends with enough silence to close an utterance"""
    frame = int(SAMPLE_RATE * FRAME_SECONDS)
    count = len(audio) // frame
    if count == 0:
        return {'has_speech': False, 'trailing_silence': False}
    frames = audio[:count * frame].reshape(count, frame)
    loud = np.sqrt(np.mean(frames.astype(np.float64) ** 2, axis=1)) >= silence_rms
    tail = max(1, int(round(silence_seconds / FRAME_SECONDS)))
    return {
        'has_speech': bool(loud.any()),
        'trailing_silence': count >= tail and not bool(loud[-tail:].any()),
    }


class StreamingASRSession:
    """Audio and events shared between the WebSocket thread (producer) and the GPU task thread (consumer)"""

    def __init__(self, context_info: Optional[str] = None):
        self._lock = threading.Lock()
        self._pcm = bytearray()
        self._audio_event = threading.Event()
        self._finish = threading.Event()
        self._flush = False
        self._clear_to: Optional[int] = None
        self.context_info = context_info
        self.events: 'queue.Queue' = queue.Queue()

    def append_pcm16(self, data: bytes) -> None:
        with self._lock:
            self._pcm.extend(data[:len(data) - len(data) % BYTES_PER_SAMPLE])
        self._audio_event.set()

    def total_samples(self) -> int:
        with self._lock:
            return len(self._pcm) // BYTES_PER_SAMPLE

    def total_seconds(self) -> float:
        return self.total_samples() / SAMPLE_RATE

    def read(self, start: int, end: int) -> np.ndarray:
        with self._lock:
            chunk = bytes(self._pcm[start * BYTES_PER_SAMPLE:end * BYTES_PER_SAMPLE])
        return np.frombuffer(chunk, dtype='<i2').astype(np.float32) / 32768.0

    def pcm_bytes(self) -> bytes:
        with self._lock:
            return bytes(self._pcm)

    def wait_for_audio(self, timeout: float) -> None:
        self._audio_event.wait(timeout)
        self._audio_event.clear()

    def request_flush(self) -> None:
        with self._lock:
            self._flush = True
        self._audio_event.set()

    def take_flush(self) -> bool:
        with self._lock:
            flush, self._flush = self._flush, False
            return flush

    def request_clear(self) -> None:
        with self._lock:
            self._clear_to = len(self._pcm) // BYTES_PER_SAMPLE
        self._audio_event.set()

    def take_clear(self) -> Optional[int]:
        with self._lock:
            clear_to, self._clear_to = self._clear_to, None
            return clear_to

    def request_finish(self) -> None:
        self._finish.set()
        self._audio_event.set()

    def is_finishing(self) -> bool:
        return self._finish.is_set()

    def emit(self, event: Dict[str, Any]) -> None:
        self.events.put(event)

    def close_events(self) -> None:
        self.events.put(_EVENTS_DONE)

    def drain_events(self) -> Tuple[List[Dict[str, Any]], bool]:
        """Events produced so far, and whether the engine has finished producing"""
        events, done = [], False
        while True:
            try:
                event = self.events.get_nowait()
            except queue.Empty:
                return events, done
            if event is _EVENTS_DONE:
                done = True
            else:
                events.append(event)


class RollingTranscriber:
    """Turns successive window hypotheses into committed segments and OpenAI transcription item events.

    One item is one committed segment: it gets `input_audio_buffer.committed` when first created, append-only
    `.delta` events with the prefix stable across passes, then `.completed` with the authoritative text.
    """

    def __init__(self, emit: Callable[[Dict[str, Any]], None], max_window_seconds: float):
        self.emit = emit
        self.max_window_samples = int(max_window_seconds * SAMPLE_RATE)
        self.commit_sample = 0
        self.committed: List[Dict[str, Any]] = []
        self._previous: List[str] = []
        self._tentative: List[Dict[str, Any]] = []
        self._hypothesis_text = ''
        self._item_id: Optional[str] = None
        self._last_item_id: Optional[str] = None
        self._emitted_delta = ''
        self._item_counter = 0

    def _ensure_item(self) -> str:
        if self._item_id is None:
            self._item_counter += 1
            self._item_id = f"item_{self._item_counter:06d}"
            self.emit({
                'type': 'input_audio_buffer.committed',
                'item_id': self._item_id,
                'previous_item_id': self._last_item_id,
            })
        return self._item_id

    def _complete_item(self, segment: Dict[str, Any]) -> None:
        item_id = self._ensure_item()
        self.committed.append(segment)
        self.emit({
            'type': 'conversation.item.input_audio_transcription.completed',
            'item_id': item_id,
            'content_index': 0,
            'transcript': segment['text'],
            'segment': segment,
        })
        self._last_item_id = item_id
        self._item_id = None
        self._emitted_delta = ''

    def _set_hypothesis(self, tentative: List[Dict[str, Any]]) -> None:
        text = ' '.join(seg['text'] for seg in tentative)
        if text == self._hypothesis_text:
            return
        self._hypothesis_text = text
        self.emit({
            'type': HYPOTHESIS_EVENT,
            'item_id': self._item_id,
            'text': text,
            'segments': tentative,
            'committed_until': round(self.commit_sample / SAMPLE_RATE, 2),
        })

    @staticmethod
    def _absolute_segments(segments: List[Dict[str, Any]], offset_seconds: float) -> List[Dict[str, Any]]:
        result = []
        for seg in segments:
            text = _normalize_text(seg.get('text'))
            if not text:
                continue
            start = segment_seconds(seg.get('start_time'))
            end = segment_seconds(seg.get('end_time'))
            result.append({
                'start_time': round(offset_seconds + start, 2) if start is not None else None,
                'end_time': round(offset_seconds + end, 2) if end is not None else None,
                'speaker_id': seg.get('speaker_id'),
                'text': text,
            })
        return result

    def _boundary_sample(self, segments: List[Dict[str, Any]], commit_count: int, window_end: int) -> int:
        """Audio position the next window starts from: the start of the first uncommitted segment."""
        candidates = []
        if commit_count < len(segments) and segments[commit_count]['start_time'] is not None:
            candidates.append(segments[commit_count]['start_time'])
        if commit_count > 0 and segments[commit_count - 1]['end_time'] is not None:
            candidates.append(segments[commit_count - 1]['end_time'])
        for seconds in candidates:
            sample = int(round(seconds * SAMPLE_RATE))
            if self.commit_sample < sample <= window_end:
                return sample
        # Unusable timestamps: dropping the tail is safer than transcribing committed audio twice
        return window_end

    def apply_pass(self, segments: List[Dict[str, Any]], window_start: int, window_end: int,
                   trailing_silence: bool, flush: bool) -> None:
        """Consume the hypothesis of one pass over audio[window_start:window_end] (relative timestamps)."""
        segs = self._absolute_segments(segments, window_start / SAMPLE_RATE)
        texts = [seg['text'] for seg in segs]

        if flush or trailing_silence:
            commit_count = len(segs)
        else:
            commit_count = 0
            while (commit_count < len(segs) - 1 and commit_count < len(self._previous)
                   and texts[commit_count] == self._previous[commit_count]):
                commit_count += 1
            if commit_count == 0 and window_end - window_start >= self.max_window_samples:
                commit_count = max(len(segs) - 1, 1) if segs else 0

        previous_for_delta = self._previous[commit_count] if commit_count < len(self._previous) else None
        for seg in segs[:commit_count]:
            self._complete_item(seg)

        if commit_count == len(segs):
            self.commit_sample = window_end
        elif commit_count > 0:
            self.commit_sample = self._boundary_sample(segs, commit_count, window_end)

        tentative = segs[commit_count:]
        self._tentative = tentative
        self._previous = texts[commit_count:]

        if tentative and previous_for_delta is not None:
            stable = stable_prefix(previous_for_delta, tentative[0]['text'])
            if len(stable) > len(self._emitted_delta) and stable.startswith(self._emitted_delta):
                item_id = self._ensure_item()
                self.emit({
                    'type': 'conversation.item.input_audio_transcription.delta',
                    'item_id': item_id,
                    'content_index': 0,
                    'delta': stable[len(self._emitted_delta):],
                })
                self._emitted_delta = stable
        self._set_hypothesis(tentative)

    def skip_silence(self, window_end: int) -> None:
        """A window with no speech carries nothing to transcribe"""
        self.commit_sample = window_end
        self._previous = []
        self._tentative = []
        self._set_hypothesis([])

    def commit_hypothesis(self, window_end: int) -> None:
        """Commit the last hypothesis as is, for a flush with too little new audio to re-transcribe"""
        for seg in self._tentative:
            self._complete_item(seg)
        self.commit_sample = max(self.commit_sample, window_end)
        self._previous = []
        self._tentative = []
        self._set_hypothesis([])

    def clear(self, to_sample: int) -> None:
        """Drop the uncommitted audio and hypothesis (input_audio_buffer.clear)"""
        self.commit_sample = max(self.commit_sample, to_sample)
        self._previous = []
        self._tentative = []
        self._item_id = None
        self._emitted_delta = ''
        self._set_hypothesis([])


class StreamingTranscriptionVisitor(TranscriptionVisitor):
    """Records live progress: committed segments and accumulated decoding statistics"""

    def visit_window_transcribed(self, outputs) -> None:
        transcription = self._transcription
        transcription.generated_tokens += len(outputs.generated_ids)
        details = transcription.details
        details.encode_time = (details.encode_time or 0.0) + (outputs.encode_time or 0.0)
        details.prefill_time = (details.prefill_time or 0.0) + (outputs.prefill_time or 0.0)
        details.decode_time = (details.decode_time or 0.0) + (outputs.decode_time or 0.0)
        details.speech_tokens = outputs.speech_tokens
        self._touch()

    def visit_progress(self, committed: List[Dict[str, Any]], audio_duration: float) -> None:
        transcription = self._transcription
        transcription.segments = list(committed)
        transcription.audio_duration = audio_duration
        plain_text = transcript_plain_text(committed)
        transcription.text_preview = plain_text[:100] + "..." if len(plain_text) > 100 else plain_text
        self._touch()


class StreamingASRMixin:
    """Replaces the single offline pass of an ASR engine with the rolling re-transcription loop"""

    session: StreamingASRSession
    visitor: StreamingTranscriptionVisitor

    def _init_streaming(self, session: StreamingASRSession, limits: Dict[str, float]):
        self.session = session
        self.visitor = StreamingTranscriptionVisitor(self._transcription)
        self.max_window_seconds = limits.get('max_window_seconds', 30.0)
        self.silence_commit_seconds = limits.get('silence_commit_seconds', 1.0)
        self.silence_rms = limits.get('silence_rms', 0.008)

    def _emit_status(self, status: str, **fields) -> None:
        self.session.emit({'type': STATUS_EVENT, 'status': status, 'request_id': self.request_id, **fields})

    def _transcribe_window(self, processor, model, audio: np.ndarray):
        inputs = processor(audio, context_info=self.session.context_info)
        outputs = model.generate(
            input_ids=inputs["input_ids"],
            acoustic_input_mask=inputs["acoustic_input_mask"],
            speech_tensors=inputs["speech_tensors"],
            speech_masks=inputs["speech_masks"],
            max_new_tokens=self.max_new_tokens,
            temperature=self.temperature,
            top_p=self.top_p,
            repetition_penalty=self.repetition_penalty,
            eos_token_id=processor.tokenizer.eos_token_id,
            show_progress_bar=False,
        )
        raw_text = processor.decode(outputs.generated_ids, skip_special_tokens=True)
        segments = processor.post_process_transcription(raw_text)
        self.visitor.visit_window_transcribed(outputs)
        del inputs
        return segments

    def run_inference(self):
        from util.rand_init import get_generator

        session = self.session
        self.visitor.visit_preprocessing(datetime.now().timestamp())
        self._emit_status('loading_model')

        processor = self._load_processor()
        load_dtype = torch.float8_e4m3fn if self.model_dtype == "float8_e4m3fn" else torch.bfloat16
        load_start = time.time()
        model = self._load_model(dtype=load_dtype)
        self.visitor.visit_model_loaded(time.time() - load_start)

        get_generator(self.seeds, force_set=True)
        self.visitor.visit_inference_start()
        self._emit_status('listening')
        logger.info(f"Live transcription {self.request_id} is listening")

        transcriber = RollingTranscriber(session.emit, self.max_window_seconds)
        min_window = int(MIN_WINDOW_SECONDS * SAMPLE_RATE)
        min_step = int(MIN_STEP_SECONDS * SAMPLE_RATE)
        last_end = 0
        passes = 0

        while True:
            # Read before the snapshot so no audio can arrive after the final pass
            finishing = session.is_finishing()
            clear_to = session.take_clear()
            if clear_to is not None:
                transcriber.clear(clear_to)
            flush = session.take_flush() or finishing
            start, end = transcriber.commit_sample, session.total_samples()

            if not flush and (end - start < min_window or end - last_end < min_step):
                session.wait_for_audio(0.1)
                continue

            if finishing:
                self._emit_status('finalizing')

            if end - start < min_window:
                transcriber.commit_hypothesis(end)
            else:
                audio = session.read(start, end)
                energy = analyze_energy(audio, self.silence_rms, self.silence_commit_seconds)
                if not energy['has_speech']:
                    transcriber.skip_silence(end)
                else:
                    segments = self._transcribe_window(processor, model, audio)
                    passes += 1
                    transcriber.apply_pass(segments, start, end, energy['trailing_silence'], flush)
            last_end = end
            self.visitor.visit_progress(transcriber.committed, end / SAMPLE_RATE)

            if finishing:
                break

        self._save_audio()
        logger.info(f"Live transcription {self.request_id} finished: {end / SAMPLE_RATE:.1f}s audio, "
                    f"{passes} passes, {len(transcriber.committed)} segments")
        del processor

    def _save_audio(self) -> None:
        """Write the whole session as 24 kHz mono 16-bit WAV, atomically so a crash never leaves a partial file"""
        directory = os.path.dirname(self.audio_path)
        fd, tmp_path = tempfile.mkstemp(dir=directory, suffix='.tmp')
        try:
            with os.fdopen(fd, 'wb') as f, wave.open(f, 'wb') as wav:
                wav.setnchannels(1)
                wav.setsampwidth(BYTES_PER_SAMPLE)
                wav.setframerate(SAMPLE_RATE)
                wav.writeframes(self.session.pcm_bytes())
            os.replace(tmp_path, self.audio_path)
        except Exception:
            if os.path.exists(tmp_path):
                os.unlink(tmp_path)
            raise


class StreamingASRInferenceEngine(StreamingASRMixin, ASRInferenceEngine):
    """Live transcription with the real VibeVoice-ASR model"""

    def __init__(self, transcription: Transcription, audio_path: str, session: StreamingASRSession,
                 limits: Dict[str, float], offload_config=None):
        super().__init__(transcription, audio_path, offload_config=offload_config)
        self._init_streaming(session, limits)


FAKE_STREAMING_SENTENCES = [
    "This is a fake live transcription.",
    "Words appear while the audio keeps growing.",
    "Nothing here comes from the real model.",
]
FAKE_SEGMENT_SECONDS = 3.0


class FakeStreamingASRProcessor(FakeASRProcessor):
    """Fake processor whose transcript grows with the window, so stabilization and commits can be exercised"""

    def __init__(self):
        self._duration = 0.0

    def __call__(self, audio, context_info: Optional[str] = None) -> Dict[str, Any]:
        if isinstance(audio, str):
            return super().__call__(audio, context_info)
        self._duration = len(audio) / SAMPLE_RATE
        return {
            "input_ids": torch.zeros(1, 120, dtype=torch.long),
            "acoustic_input_mask": torch.zeros(1, 120, dtype=torch.bool),
            "speech_tensors": torch.zeros(1, 1),
            "speech_masks": torch.ones(1, 1, dtype=torch.bool),
            "audio_duration": self._duration,
        }

    def decode(self, ids, skip_special_tokens: bool = True) -> str:
        items = []
        index = 0
        while index * FAKE_SEGMENT_SECONDS < self._duration:
            start = index * FAKE_SEGMENT_SECONDS
            end = min(start + FAKE_SEGMENT_SECONDS, self._duration)
            words = FAKE_STREAMING_SENTENCES[index % len(FAKE_STREAMING_SENTENCES)].split()
            count = max(1, int(len(words) * (end - start) / FAKE_SEGMENT_SECONDS))
            items.append({"Start time": round(start, 2), "End time": round(end, 2),
                          "Speaker ID": 0, "Content": ' '.join(words[:count])})
            index += 1
        return json.dumps(items)


class FakeStreamingASRInferenceEngine(StreamingASRMixin, FakeASRInferenceEngine):
    """Fake live transcription engine for development and tests"""

    def __init__(self, transcription: Transcription, audio_path: str, session: StreamingASRSession,
                 limits: Dict[str, float], offload_config=None):
        super().__init__(transcription, audio_path, offload_config=offload_config)
        self._init_streaming(session, limits)

    def _load_processor(self):
        return FakeStreamingASRProcessor()


def create_streaming_engine(transcription: Transcription, audio_path: str, session: StreamingASRSession,
                            limits: Dict[str, float], offload_config: Optional[Dict[str, Any]] = None,
                            fake: bool = False) -> ASRInferenceBase:
    """Create a real or fake live transcription engine"""
    if fake:
        return FakeStreamingASRInferenceEngine(transcription, audio_path, session, limits,
                                               offload_config=offload_config)
    return StreamingASRInferenceEngine(transcription, audio_path, session, limits,
                                       offload_config=_build_offload_config(offload_config))
