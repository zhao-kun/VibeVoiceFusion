import base64
from typing import Union
import time
import torch

from abc import ABC, abstractmethod
from flask import current_app
from pathlib import Path
from vibevoice.modular.modeling_vibevoice_inference import VibeVoiceForConditionalInference, VibeVoiceGenerationOutput
from vibevoice.modular.custom_offloading_utils import OffloadConfig
from vibevoice.processor.vibevoice_processor import VibeVoiceProcessor
from transformers.utils import logging
from config.configuration_vibevoice import DEFAULT_CONFIG, VibeVoiceConfig, InferencePhase
from backend.models.generation import Generation, UpdateStatusCallable
from backend.services.speaker_service import SpeakerService
from backend.services.dialog_session_service import DialogSessionService
from util.rand_init import get_generator
from typing import Dict, Any, Optional

logging.set_verbosity_info()
logger = logging.get_logger(__name__)


# Preset mapping for offloading configurations
OFFLOAD_PRESETS = {
    "balanced": OffloadConfig(
        enabled=True,
        num_layers_on_gpu=12,  # 12 GPU + 16 CPU
        pin_memory=True,
        prefetch_next_layer=True,
        profile=True,  # Enable profiling for metrics collection
    ),
    "aggressive": OffloadConfig(
        enabled=True,
        num_layers_on_gpu=8,  # 8 GPU + 20 CPU
        pin_memory=True,
        prefetch_next_layer=True,
        profile=True,  # Enable profiling for metrics collection
    ),
    "extreme": OffloadConfig(
        enabled=True,
        num_layers_on_gpu=4,  # 4 GPU + 24 CPU
        pin_memory=True,
        prefetch_next_layer=True,
        profile=True,  # Enable profiling for metrics collection
    ),
}

class FakeModel:
    def __init__(self):
        pass

    def generate(self, **kwargs) -> Union[torch.LongTensor, VibeVoiceGenerationOutput]:
        for i in range(100):
            time.sleep(0.5)  # Simulate some processing time
            kwargs.get("status_update", lambda phase, **kwargs: None)(InferencePhase.INFERENCING, current=i + 1, total_step=100)

        return torch.randn(1, 16000 * 5)  # Simulate 5 seconds of audio at 16kHz

class InferenceBase(ABC):
    def __init__(self, generation: Generation, speaker_service: SpeakerService,
                 dialog_service: DialogSessionService, meta_file_path: str):
        self.generation = generation
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.speaker_service = speaker_service
        self.dialog_service = dialog_service
        self.model_file = current_app.config['MODEL_PATH']
        self.meta_file_path = meta_file_path

    @staticmethod
    def create(generation: Generation, speaker_service: SpeakerService,
               dialog_service: DialogSessionService, meta_file_path: str,
               offload_config: Optional[Dict[str, Any]] = None,
               fake: bool = False) -> 'InferenceBase':
        """
        Create inference engine instance.

        Args:
            generation: Generation object
            speaker_service: Speaker service
            dialog_service: Dialog service
            meta_file_path: Path to metadata
            fake: Use fake inference engine for testing
            offload_config: Offloading configuration dict (optional)
                {
                    'enabled': True/False,
                    'mode': 'preset' or 'manual',
                    'preset': 'balanced'/'aggressive'/'extreme',  # for preset mode
                    'num_gpu_layers': int  # for manual mode
                }

        Returns:
            InferenceBase instance
        """
        # Parse offload config dict to create OffloadConfig object (for both real and fake engines)
        offload_config_obj = None
        if offload_config and offload_config.get('enabled', False):
            mode = offload_config.get('mode', 'preset')

            if mode == 'preset':
                preset = offload_config.get('preset', 'balanced')
                offload_config_obj = OFFLOAD_PRESETS.get(preset)
                if not offload_config_obj:
                    logger.warning(f"Unknown preset '{preset}', using 'balanced'")
                    offload_config_obj = OFFLOAD_PRESETS['balanced']

            elif mode == 'manual':
                num_gpu_layers = offload_config.get('num_gpu_layers', 20)
                offload_config_obj = OffloadConfig(
                    enabled=True,
                    num_layers_on_gpu=num_gpu_layers,
                    pin_memory=True,
                    prefetch_next_layer=True,
                    profile=True,  # Enable profiling for metrics collection
                )

        if fake:
            return FakeInferenceEngine(
                generation, speaker_service, dialog_service, meta_file_path,
                offload_config=offload_config_obj
            )

        return InferenceEngine(
            generation, speaker_service, dialog_service, meta_file_path,
            offload_config=offload_config_obj
        )

    @abstractmethod
    def _load_model(self, dtype: torch.dtype, config: str = None):
        pass

    @abstractmethod
    def _save_audio(self, outputs: Union[torch.LongTensor, VibeVoiceGenerationOutput],
                    processor: VibeVoiceProcessor, update_status: UpdateStatusCallable,
                    generation_time: float, input_tokens: int, **kwargs) -> None:
        pass

    def run_inference(self, status_update: UpdateStatusCallable = lambda phase, *args, **kwargs: None):

        get_generator(self.generation.seeds)

        txt_content, scripts, unique_speaker_names, max_speaker_id = self.dialog_service.parse_session_txt_script(self.generation.session_id)
        if not scripts or not unique_speaker_names:
            raise RuntimeError("No scripts, speaker_numbers found for the specified dialog session.")

        voice_names = []
        for i in range(max_speaker_id):
            voice_names.append(f"Speaker {i+1}")
        
        voice_sample = self.speaker_service.get_speakers_filepath(voice_names)

        logger.info(f"Max speaker ID: {max_speaker_id}, unique speakers: {unique_speaker_names}, all voice samples: {voice_sample}")
        # Proceed with inference using txt_content, scripts, and speaker_numbers
        # ...
        full_script = '\n'.join(scripts)
        full_script = full_script.replace("’", "'")

        status_update(InferencePhase.PREPROCESSING, scripts=scripts,
                      unique_speaker_names=unique_speaker_names,
                      voice_sample=voice_sample,
                      max_speaker_id=max_speaker_id)

        processor = VibeVoiceProcessor.from_pretrained(None)
        inputs = processor(text=[full_script],
                           voice_samples=[voice_sample],
                           padding=True,
                           return_tensors="pt",
                           return_attention_mask=True)

        for k, v in inputs.items():
            if torch.is_tensor(v):
                inputs[k] = v.to(self.device)

        load_dtype = torch.bfloat16
        if self.generation.model_dtype == "float8_e4m3fn":
            load_dtype = torch.float8_e4m3fn
        logger.info(f"Loading model with dtype: {load_dtype}, "
                    f"attn_implementation: {self.generation.attn_implementation}")

        # only for testing purpose, load a fake model
        model = self._load_model(dtype=load_dtype)

        start_time = time.time()

        outputs = model.generate(**inputs,
                                 max_new_tokens=None,
                                 cfg_scale=self.generation.cfg_scale,
                                 tokenizer=processor.tokenizer,
                                 generation_config={'do_sample': False},
                                 verbose=False,
                                 status_update=status_update)

        generation_time = time.time() - start_time
        self._save_audio(outputs, processor, status_update, generation_time,
                         inputs['input_ids'].shape[1],
                         unique_speaker_names=unique_speaker_names,
                         number_of_segments=len(scripts))


class InferenceEngine(InferenceBase):
    def __init__(self, generation, speaker_service, dialog_service, meta_file_path: str,
                 offload_config: Optional[OffloadConfig] = None):
        """
        Initialize inference engine with optional layer offloading.

        Args:
            generation: Generation object with parameters
            speaker_service: Service for managing speakers
            dialog_service: Service for managing dialogs
            meta_file_path: Path to metadata file
            offload_config: OffloadConfig object or None (default: no offloading)
        """
        super().__init__(generation, speaker_service, dialog_service, meta_file_path)

        # Offloading configuration
        self.offload_config = offload_config

    def _load_model(self, dtype: torch.dtype, config: str = None):
        config_dict = {}
        if config:
            with open(config, 'r') as f:
                import json
                config_dict = json.load(f)
        else:
            # Use default configuration
            print(f"Using default configuration: {DEFAULT_CONFIG}")
            config_dict = DEFAULT_CONFIG

        config = VibeVoiceConfig.from_dict(config_dict,
                                           torch_dtype=dtype,
                                           device_map="cuda",
                                           attn_implementation=self.generation.attn_implementation)

        # Use offload config if provided
        if self.offload_config and self.offload_config.enabled:
            logger.info(f"Layer offloading enabled: {self.offload_config.num_layers_on_gpu} layers on GPU")
        else:
            logger.info("Layer offloading disabled")

        # Load model with device-specific logic
        model_file = Path(self.model_file) / Path(f"vibevoice7b_{'bf16' if dtype == torch.bfloat16 else 'float8_e4m3fn'}.safetensors")
        model = VibeVoiceForConditionalInference.from_pretrain(
            str(model_file.resolve()),
            config,
            device=self.device,
            offload_config=self.offload_config
        )

        model.eval()
        self.model = model  # Store model for later access (e.g., metrics collection)
        return model

    def _collect_offloading_metrics(self, generation_time: float) -> Optional[Dict[str, Any]]:
        """
        Collect offloading statistics after generation.

        Args:
            generation_time: Total generation time in seconds

        Returns:
            Dictionary with offloading metrics or None if offloading not enabled
        """
        if not hasattr(self, 'model') or self.model.offloader is None:
            return None

        stats = self.model.offloader.get_stats()

        # Calculate VRAM savings estimate (~310MB per layer in Float8)
        vram_saved_gb = (stats['cpu_layers'] * 0.31)

        # Calculate overhead percentage
        transfer_overhead_sec = stats['total_transfer_time_ms'] / 1000
        overhead_percentage = (transfer_overhead_sec / generation_time * 100) if generation_time > 0 else 0

        return {
            "enabled": True,
            "gpu_layers": stats['gpu_layers'],
            "cpu_layers": stats['cpu_layers'],
            "transfer_overhead_ms": stats['total_transfer_time_ms'],
            "avg_layer_transfer_ms": stats['avg_layer_transfer_time_ms'],
            "overhead_percentage": overhead_percentage,
            "time_breakdown": {
                "pure_computation_ms": stats.get('total_compute_time_ms', 0),
                "cpu_to_gpu_transfer_ms": stats.get('total_pre_transfer_time_ms', 0),
                "gpu_to_cpu_release_ms": stats.get('total_post_transfer_time_ms', 0),
            },
            "theoretical_async_savings_ms": stats.get('theoretical_savings_with_async_ms', 0),
            "vram_saved_gb": round(vram_saved_gb, 2),
        }

    def _save_audio(self, outputs: Union[torch.LongTensor, VibeVoiceGenerationOutput],
                    processor: VibeVoiceProcessor, status_update: UpdateStatusCallable,
                    generation_time: float, input_tokens: int, **kwargs) -> None:
        if outputs.speech_outputs is None or len(outputs.speech_outputs) == 0:
            raise RuntimeError("No audio output generated.")

        sample_rate = 24000
        audio_samples = outputs.speech_outputs[0].shape[-1] \
            if len(outputs.speech_outputs[0].shape) > 0 else len(outputs.speech_outputs[0])
        audio_duration = audio_samples / sample_rate
        rtf = generation_time / audio_duration if audio_duration > 0 else float('inf')

        output_tokens = outputs.sequences.shape[1]  # Total tokens (input + generated)
        generated_tokens = output_tokens - input_tokens

        # Collect offloading metrics if enabled
        offloading_metrics = self._collect_offloading_metrics(generation_time)
        if offloading_metrics:
            self.generation.details['offloading_metrics'] = offloading_metrics
            logger.info(
                f"Offloading metrics collected: {offloading_metrics['gpu_layers']} GPU layers, "
                f"{offloading_metrics['overhead_percentage']:.1f}% overhead"
            )

        # Generate output filename and set it in the generation object
        output_filename = f"{self.generation.request_id}.wav"
        self.generation.output_filename = output_filename

        # Save output (processor handles device internally)
        output_audio_path = Path(self.generation.project_dir) / output_filename
        processor.save_audio(outputs.speech_outputs[0], output_path=output_audio_path)

        status_update(InferencePhase.SAVING_AUDIO,
                      output_audio_path=str(output_audio_path),
                      generation_time=generation_time,
                      prefilling_tokens=input_tokens,
                      total_tokens=output_tokens,
                      generated_tokens=generated_tokens,
                      audio_duration_seconds=audio_duration,
                      real_time_factor=rtf,
                      **kwargs)
        logger.info(f"Saving generated audio to {output_audio_path}")
        return

class FakeInferenceEngine(InferenceBase):
    def __init__(self, generation, speaker_service, dialog_service, meta_file_path: str,
                 offload_config: Optional[OffloadConfig] = None):
        """
        Initialize fake inference engine with optional layer offloading.

        Args:
            generation: Generation object with parameters
            speaker_service: Service for managing speakers
            dialog_service: Service for managing dialogs
            meta_file_path: Path to metadata file
            offload_config: OffloadConfig object or None (default: no offloading)
        """
        super().__init__(generation, speaker_service, dialog_service, meta_file_path)

        # Offloading configuration
        self.offload_config = offload_config

    def _load_model(self, dtype: torch.dtype, config: str = None):
        # Log offloading config for fake engine
        if self.offload_config and self.offload_config.enabled:
            logger.info(f"[FAKE] Layer offloading enabled: {self.offload_config.num_layers_on_gpu} layers on GPU")
        else:
            logger.info("[FAKE] Layer offloading disabled")
        return FakeModel()

    def _generate_fake_offloading_metrics(self, generation_time: float, num_tokens: int = 100) -> Optional[Dict[str, Any]]:
        """
        Generate realistic fake offloading metrics.

        Args:
            generation_time: Total generation time in seconds
            num_tokens: Number of generated tokens (ignored, calculated from generation_time)

        Returns:
            Dictionary with fake offloading metrics or None if offloading not enabled
        """
        if not self.offload_config or not self.offload_config.enabled:
            return None

        total_layers = 28
        gpu_layers = self.offload_config.num_layers_on_gpu
        cpu_layers = total_layers - gpu_layers

        # Realistic transfer times based on real measurements (from CLAUDE.md Phase 2 metrics)
        # CPU→GPU: ~37.8ms per layer, GPU→CPU: ~35.8ms per layer
        avg_cpu_to_gpu_ms = 37.8
        avg_gpu_to_cpu_ms = 35.8

        # Calculate realistic overhead percentages based on GPU layer count
        # These match the expected overhead from CLAUDE.md
        if gpu_layers >= 12:  # Balanced
            target_overhead = 0.75  # 75% overhead
        elif gpu_layers >= 8:   # Aggressive
            target_overhead = 0.82  # 82% overhead
        else:                    # Extreme (4 layers)
            target_overhead = 0.87  # 87% overhead

        # Calculate transfer time from target overhead
        generation_time_ms = generation_time * 1000
        total_transfer_ms = generation_time_ms * target_overhead

        # Compute time is the remaining time
        total_compute_ms = generation_time_ms * (1 - target_overhead)

        # Distribute transfer time across CPU→GPU and GPU→CPU (roughly equal)
        total_cpu_to_gpu_ms = total_transfer_ms * (avg_cpu_to_gpu_ms / (avg_cpu_to_gpu_ms + avg_gpu_to_cpu_ms))
        total_gpu_to_cpu_ms = total_transfer_ms * (avg_gpu_to_cpu_ms / (avg_cpu_to_gpu_ms + avg_gpu_to_cpu_ms))

        # Calculate theoretical async savings (could hide ~90% of GPU→CPU transfers)
        theoretical_savings_ms = total_gpu_to_cpu_ms * 0.9

        # VRAM savings (~310MB per layer for Float8)
        vram_saved_gb = cpu_layers * 0.31

        # Average layer transfer time (CPU→GPU + GPU→CPU) / 2
        avg_layer_transfer_ms = (avg_cpu_to_gpu_ms + avg_gpu_to_cpu_ms) / 2

        return {
            "enabled": True,
            "gpu_layers": gpu_layers,
            "cpu_layers": cpu_layers,
            "transfer_overhead_ms": round(total_transfer_ms, 2),
            "avg_layer_transfer_ms": round(avg_layer_transfer_ms, 2),
            "overhead_percentage": round(target_overhead * 100, 2),
            "time_breakdown": {
                "pure_computation_ms": round(total_compute_ms, 2),
                "cpu_to_gpu_transfer_ms": round(total_cpu_to_gpu_ms, 2),
                "gpu_to_cpu_release_ms": round(total_gpu_to_cpu_ms, 2),
            },
            "theoretical_async_savings_ms": round(theoretical_savings_ms, 2),
            "vram_saved_gb": round(vram_saved_gb, 2),
        }

    def _save_audio(self, outputs: Union[torch.LongTensor, VibeVoiceGenerationOutput],
                    processor: VibeVoiceProcessor, status_update: UpdateStatusCallable,
                    generation_time: float, input_tokens: int, **kwargs) -> None:
        base64_wav_audio = "UklGRiUAAABXQVZFZm10IBAAAAABAAEAQB8AAEAfAAABAAgAZGF0YQEAAACA"  # Fake short audio for test
        audio_data = base64.b64decode(base64_wav_audio)

        # Fake token counts
        fake_generated_tokens = 100
        fake_total_tokens = input_tokens + fake_generated_tokens

        # Generate fake offloading metrics if enabled
        offloading_metrics = self._generate_fake_offloading_metrics(generation_time, fake_generated_tokens)
        if offloading_metrics:
            self.generation.details['offloading_metrics'] = offloading_metrics
            logger.info(
                f"[FAKE] Offloading metrics generated: {offloading_metrics['gpu_layers']} GPU layers, "
                f"{offloading_metrics['overhead_percentage']:.1f}% overhead"
            )

        # Generate output filename and set it in the generation object
        output_filename = f"{self.generation.request_id}.wav"
        self.generation.output_filename = output_filename

        output_audio_path = Path(self.generation.project_dir) / output_filename
        with open(output_audio_path, 'wb') as f:
            f.write(audio_data)
        status_update(InferencePhase.SAVING_AUDIO,
                      output_audio_path=str(output_audio_path),
                      prefilling_tokens=input_tokens,
                      generation_time=generation_time,
                      total_tokens=fake_total_tokens,
                      generated_tokens=fake_generated_tokens,
                      audio_duration_seconds=5.0,               # fake value
                      real_time_factor=1.0,                     # fake value
                      **kwargs
                      )
        logger.info(f"Saving generated audio to {output_audio_path}")
        return
