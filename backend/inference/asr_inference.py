"""
ASR Inference Engine - transcribes audio with VibeVoice-ASR
"""
import copy
import gc
import json
import random
import time
from abc import ABC, abstractmethod
from datetime import datetime
from pathlib import Path
from typing import Dict, Any, Optional, List

import torch
from flask import current_app

from backend.models.transcription import Transcription, transcript_plain_text
from config.configuration_vibevoice import InferencePhase
from util.logger import get_logger

logger = get_logger(__name__)

PROGRESS_UPDATE_INTERVAL_TOKENS = 8


# Import these only when needed to avoid loading the model code at API import time
def _get_model_classes():
    from vibevoice.modular.modeling_vibevoice_asr_inference import VibeVoiceASRForConditionalInference
    from vibevoice.modular.custom_offloading_utils import OffloadConfig
    from vibevoice.processor.vibevoice_asr_processor import VibeVoiceASRProcessor
    return VibeVoiceASRForConditionalInference, OffloadConfig, VibeVoiceASRProcessor


# Same presets as speech generation; the ASR decoder also has 28 layers
OFFLOAD_PRESET_GPU_LAYERS = {
    "balanced": 12,
    "aggressive": 8,
    "extreme": 4,
}


def _build_offload_config(offload_config: Optional[Dict[str, Any]]):
    if not offload_config or not offload_config.get('enabled', False):
        return None

    _, OffloadConfig, _ = _get_model_classes()
    mode = offload_config.get('mode', 'preset')
    if mode == 'manual':
        num_gpu_layers = offload_config.get('num_gpu_layers', 20)
    else:
        preset = offload_config.get('preset', 'balanced')
        if preset not in OFFLOAD_PRESET_GPU_LAYERS:
            logger.warning(f"Unknown preset '{preset}', using 'balanced'")
            preset = 'balanced'
        num_gpu_layers = OFFLOAD_PRESET_GPU_LAYERS[preset]

    return OffloadConfig(
        enabled=True,
        num_layers_on_gpu=num_gpu_layers,
        pin_memory=True,
        prefetch_next_layer=True,
        profile=True,
    )


class TranscriptionVisitor:
    """Visitor that records ASR progress on the Transcription object"""

    def __init__(self, transcription: Transcription):
        self._transcription = transcription
        self.preprocess_begin = None

    def _touch(self):
        self._transcription.updated_at = datetime.utcnow().isoformat()

    def visit_preprocessing(self, timestamp: float = None):
        self._transcription.status = InferencePhase.PREPROCESSING
        self.preprocess_begin = timestamp
        self._touch()

    def visit_audio_loaded(self, audio_duration: float, prompt_tokens: int):
        self._transcription.audio_duration = audio_duration
        self._transcription.details.prompt_tokens = prompt_tokens
        if self.preprocess_begin:
            self._transcription.details.preprocessing_duration = datetime.now().timestamp() - self.preprocess_begin
        self._touch()

    def visit_model_loaded(self, load_duration: float):
        self._transcription.details.model_load_duration = load_duration
        self._touch()

    def visit_inference_start(self):
        self._transcription.status = InferencePhase.INFERENCING
        self._transcription.generated_tokens = 0
        self._touch()

    def visit_token(self, token_id: int):
        self._transcription.generated_tokens += 1
        if self._transcription.generated_tokens % PROGRESS_UPDATE_INTERVAL_TOKENS == 0:
            self._touch()

    def visit_transcribed(self, raw_text: str, segments: List[Dict[str, Any]], outputs):
        self._transcription.raw_text = raw_text
        self._transcription.segments = segments
        self._transcription.generated_tokens = len(outputs.generated_ids)
        self._transcription.reach_max_new_tokens = outputs.reach_max_new_tokens
        self._transcription.details.encode_time = outputs.encode_time
        self._transcription.details.prefill_time = outputs.prefill_time
        self._transcription.details.decode_time = outputs.decode_time
        self._transcription.details.speech_tokens = outputs.speech_tokens
        plain_text = transcript_plain_text(segments, raw_text)
        self._transcription.text_preview = plain_text[:100] + "..." if len(plain_text) > 100 else plain_text
        self._touch()

    def visit_completed(self, message: str = None):
        self._transcription.status = InferencePhase.COMPLETED
        self._transcription.percentage = 100.0
        if not self._transcription.completed_at:
            self._transcription.completed_at = datetime.utcnow().isoformat()
        self._touch()

    def visit_failed(self, message: str, failure_type: str = None):
        self._transcription.status = InferencePhase.FAILED
        self._transcription.error_message = message
        self._touch()


class ASRInferenceBase(ABC):
    """Base class for ASR inference"""

    def __init__(self, transcription: Transcription, audio_path: str, offload_config=None):
        self._transcription = transcription
        self.visitor = TranscriptionVisitor(transcription)
        self.audio_path = audio_path
        self.offload_config = offload_config
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.model = None

        self.request_id = transcription.request_id
        self.model_dtype = transcription.model_dtype
        self.context_info = transcription.context_info
        self.max_new_tokens = transcription.max_new_tokens
        self.temperature = transcription.temperature
        self.top_p = transcription.top_p
        self.repetition_penalty = transcription.repetition_penalty
        self.seeds = transcription.seeds

    @staticmethod
    def create(transcription: Transcription, audio_path: str,
               offload_config: Optional[Dict[str, Any]] = None,
               fake: bool = False) -> 'ASRInferenceBase':
        """Create a real or fake ASR inference engine"""
        if fake:
            # The fake engine must not import the model code, which needs the GPU stack
            return FakeASRInferenceEngine(transcription, audio_path, offload_config=offload_config)
        return ASRInferenceEngine(transcription, audio_path, offload_config=_build_offload_config(offload_config))

    def get_transcription(self) -> Transcription:
        return copy.deepcopy(self._transcription)

    def generation_info(self) -> Dict[str, Any]:
        return self.get_transcription().to_dict()

    def failure(self, message: str, failure_type: str = None):
        self.visitor.visit_failed(message, failure_type)

    def success(self, message: str = None):
        self.visitor.visit_completed(message)

    @abstractmethod
    def _load_processor(self):
        pass

    @abstractmethod
    def _load_model(self, dtype: torch.dtype):
        pass

    def run_inference(self):
        """Run preprocessing, model loading and transcription"""
        from util.rand_init import get_generator

        self.visitor.visit_preprocessing(datetime.now().timestamp())

        processor = self._load_processor()
        inputs = processor(self.audio_path, context_info=self.context_info)
        self.visitor.visit_audio_loaded(inputs['audio_duration'], int(inputs['input_ids'].shape[1]))
        logger.info(f"Transcribing {self.audio_path} ({inputs['audio_duration']:.2f}s, "
                    f"{inputs['input_ids'].shape[1]} prompt tokens)")

        load_dtype = torch.float8_e4m3fn if self.model_dtype == "float8_e4m3fn" else torch.bfloat16
        load_start = time.time()
        model = self._load_model(dtype=load_dtype)
        self.visitor.visit_model_loaded(time.time() - load_start)

        get_generator(self.seeds, force_set=True)
        self.visitor.visit_inference_start()
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
            token_callback=self.visitor.visit_token,
            show_progress_bar=False,
        )

        raw_text = processor.decode(outputs.generated_ids, skip_special_tokens=True)
        segments = processor.post_process_transcription(raw_text)
        self.visitor.visit_transcribed(raw_text, segments, outputs)
        self.visitor.visit_completed()
        logger.info(f"Transcription {self.request_id} produced {len(segments)} segments, "
                    f"{len(outputs.generated_ids)} tokens")

        del inputs
        del outputs
        del processor

    def finalize(self):
        """Clean up GPU memory after inference."""
        if self.model is not None:
            if getattr(self.model, 'offloader', None) is not None:
                try:
                    self.model.offloader.cleanup()
                    self.model.offloader = None
                except Exception as e:
                    logger.warning(f"Failed to cleanup offloader: {e}")
            try:
                self.model.to('cpu')
            except Exception:
                pass
            del self.model
            self.model = None

        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.synchronize()
            torch.cuda.empty_cache()

        logger.info("GPU memory cleanup completed")


class ASRInferenceEngine(ASRInferenceBase):
    """Real VibeVoice-ASR inference engine"""

    def __init__(self, transcription: Transcription, audio_path: str, offload_config=None):
        super().__init__(transcription, audio_path, offload_config=offload_config)
        self.model_dir = Path(current_app.config['ASR_MODEL_PATH'])

    def _load_processor(self):
        _, _, VibeVoiceASRProcessor = _get_model_classes()
        model_dir = str(self.model_dir) if self.model_dir.is_dir() else None
        return VibeVoiceASRProcessor.from_pretrained(model_dir)

    def _resolve_checkpoint(self, dtype: torch.dtype):
        """Prefer a converted single-file checkpoint; otherwise load the HF directory and cast on load."""
        suffix = 'float8_e4m3fn' if dtype == torch.float8_e4m3fn else 'bf16'
        converted = self.model_dir / f"vibevoice_asr_{suffix}.safetensors"
        if converted.exists():
            return converted, None
        if (self.model_dir / 'model.safetensors.index.json').exists() or (self.model_dir / 'model.safetensors').exists():
            return self.model_dir, dtype
        raise FileNotFoundError(
            f"No VibeVoice-ASR checkpoint found in {self.model_dir}: expected {converted.name}, "
            f"model.safetensors.index.json or model.safetensors"
        )

    def _load_model(self, dtype: torch.dtype):
        from config.configuration_vibevoice import DEFAULT_ASR_CONFIG, VibeVoiceASRConfig
        VibeVoiceASRForConditionalInference, _, _ = _get_model_classes()

        config_path = self.model_dir / 'config.json'
        if config_path.exists():
            with open(config_path, 'r') as f:
                config_dict = json.load(f)
        else:
            logger.info(f"{config_path} not found, using default ASR configuration")
            config_dict = DEFAULT_ASR_CONFIG
        config = VibeVoiceASRConfig.from_dict(config_dict, torch_dtype=torch.bfloat16)

        if self.offload_config and self.offload_config.enabled:
            logger.info(f"Layer offloading enabled: {self.offload_config.num_layers_on_gpu} layers on GPU")
        else:
            logger.info("Layer offloading disabled")

        checkpoint, cast_dtype = self._resolve_checkpoint(dtype)
        logger.info(f"Loading VibeVoice-ASR from {checkpoint} (cast dtype: {cast_dtype})")
        model = VibeVoiceASRForConditionalInference.from_pretrain(
            str(Path(checkpoint).resolve()),
            config,
            device=self.device,
            offload_config=self.offload_config,
            dtype=cast_dtype,
        )
        model.eval()
        self.model = model
        return model


FAKE_TRANSCRIPT = [
    {"Start time": 0.0, "End time": 2.5, "Speaker ID": 0, "Content": "This is a fake transcription."},
    {"Start time": 2.5, "End time": 5.0, "Speaker ID": 1, "Content": "It is produced without loading the model."},
]


class _FakeOutputs:
    def __init__(self, generated_ids: List[int], encode_time: float, decode_time: float):
        self.generated_ids = generated_ids
        self.reach_max_new_tokens = False
        self.stopped_externally = False
        self.speech_tokens = 38
        self.encode_time = encode_time
        self.prefill_time = 0.01
        self.decode_time = decode_time


class _FakeTokenizer:
    eos_token_id = 0


class FakeASRProcessor:
    """Processor stand-in for development without the tokenizer and audio stack"""

    tokenizer = _FakeTokenizer()

    def __call__(self, audio_path: str, context_info: Optional[str] = None) -> Dict[str, Any]:
        if not Path(audio_path).exists():
            raise FileNotFoundError(f"Audio file not found: {audio_path}")
        return {
            "input_ids": torch.zeros(1, 120, dtype=torch.long),
            "acoustic_input_mask": torch.zeros(1, 120, dtype=torch.bool),
            "speech_tensors": torch.zeros(1, 24000 * 5),
            "speech_masks": torch.ones(1, 38, dtype=torch.bool),
            "audio_duration": 5.0,
        }

    def decode(self, ids, skip_special_tokens: bool = True) -> str:
        return json.dumps(FAKE_TRANSCRIPT)

    def post_process_transcription(self, text: str) -> List[Dict[str, Any]]:
        mapping = {"Start time": "start_time", "End time": "end_time", "Speaker ID": "speaker_id", "Content": "text"}
        return [{mapping[k]: v for k, v in item.items() if k in mapping} for item in json.loads(text)]


class FakeASRModel:
    """Fake model emitting tokens slowly so the live progress path can be exercised"""

    step_delay = 0.1

    def generate(self, token_callback=None, max_new_tokens: int = 8192, **kwargs) -> _FakeOutputs:
        start = time.time()
        steps = min(max_new_tokens, random.randint(20, 60))
        generated = []
        for i in range(steps):
            time.sleep(self.step_delay)
            generated.append(i + 1)
            if token_callback is not None:
                token_callback(i + 1)
        return _FakeOutputs(generated, encode_time=0.05, decode_time=time.time() - start)


class FakeASRInferenceEngine(ASRInferenceBase):
    """Fake ASR engine for development and tests"""

    def _load_processor(self):
        return FakeASRProcessor()

    def _load_model(self, dtype: torch.dtype):
        if self.offload_config and self.offload_config.get('enabled'):
            logger.info(f"[FAKE] Layer offloading enabled: {self.offload_config}")
        else:
            logger.info("[FAKE] Layer offloading disabled")
        return FakeASRModel()
