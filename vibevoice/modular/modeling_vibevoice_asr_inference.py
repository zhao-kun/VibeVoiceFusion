import os
import time
from dataclasses import dataclass, field
from typing import Callable, List, Optional, Union

import torch
import torch.nn as nn
from tqdm import tqdm

from accelerate import init_empty_weights
from transformers.cache_utils import DynamicCache
from transformers.generation.logits_process import (
    LogitsProcessorList,
    RepetitionPenaltyLogitsProcessor,
    TemperatureLogitsWarper,
    TopKLogitsWarper,
    TopPLogitsWarper,
)

from config.configuration_vibevoice import QwenConfig, VibeVoiceASRConfig
from vibevoice.modular.modeling_vibevoice import SpeechConnector
from vibevoice.modular.modular_vibevoice_qwen import QwenModel
from vibevoice.modular.modular_vibevoice_tokenizer import (
    VibeVoiceAcousticTokenizerModel,
    VibeVoiceSemanticTokenizerModel,
    VibeVoiceTokenizerEncoderOutput,
    VibeVoiceTokenizerStreamingCache,
)
from util.float8_scale import AutoCast
from util.logger import get_logger

logger = get_logger(__name__)

ASR_SAMPLE_RATE = 24000
# ASR never decodes audio, so the acoustic decoder is not built and its checkpoint weights are dropped.
IGNORED_CHECKPOINT_PREFIXES = ("model.acoustic_tokenizer.decoder.",)


def _resolve_dtype(dtype) -> torch.dtype:
    if dtype is None:
        return torch.float32
    if isinstance(dtype, str):
        return getattr(torch, dtype)
    return dtype


@dataclass
class VibeVoiceASROutput:
    """Result of one ASR generation (batch size 1)."""
    sequences: torch.LongTensor
    generated_ids: List[int] = field(default_factory=list)
    reach_max_new_tokens: bool = False
    stopped_externally: bool = False
    prompt_tokens: int = 0
    speech_tokens: int = 0
    encode_time: float = 0.0
    prefill_time: float = 0.0
    decode_time: float = 0.0


class VibeVoiceASRModel(nn.Module):
    """Speech encoders + connectors + Qwen decoder of VibeVoice-ASR."""

    def __init__(self, config: VibeVoiceASRConfig):
        super().__init__()
        dtype = _resolve_dtype(getattr(config, "torch_dtype", None))
        self.dtype = dtype

        lm_config = QwenConfig.from_config(config.decoder_config)
        self.language_model = QwenModel(lm_config, dtype=dtype)

        self.acoustic_tokenizer = VibeVoiceAcousticTokenizerModel(config.acoustic_tokenizer_config, dtype=dtype).to(dtype)
        self.acoustic_tokenizer.decoder = None
        self.semantic_tokenizer = VibeVoiceSemanticTokenizerModel(config.semantic_tokenizer_config, dtype=dtype).to(dtype)

        self.acoustic_connector = SpeechConnector(config.acoustic_vae_dim, lm_config.hidden_size, dtype=dtype).to(dtype)
        self.semantic_connector = SpeechConnector(config.semantic_vae_dim, lm_config.hidden_size, dtype=dtype).to(dtype)


class VibeVoiceASRForConditionalInference(nn.Module):
    """VibeVoice-ASR inference: encode speech, splice into the prompt, decode text autoregressively."""

    def __init__(self, config: VibeVoiceASRConfig):
        super().__init__()
        self.config = config
        self.model = VibeVoiceASRModel(config)
        self.lm_head = AutoCast.Linear(config.decoder_config.hidden_size,
                                       config.decoder_config.vocab_size,
                                       bias=False,
                                       dtype=self.model.dtype)
        self.dtype = self.model.dtype
        self.device = torch.device("cuda")
        self.offloader = None

    def __del__(self):
        if getattr(self, "offloader", None) is not None:
            try:
                self.offloader.cleanup()
            except Exception:
                pass

    @property
    def compute_dtype(self) -> torch.dtype:
        """Activation dtype; FP8 weights compute in bf16 via AutoCast."""
        weight_dtype = self.lm_head.weight.dtype
        return torch.bfloat16 if weight_dtype == torch.float8_e4m3fn else weight_dtype

    @torch.no_grad()
    def encode_speech(
        self,
        speech_tensors: torch.FloatTensor,
        speech_masks: Optional[torch.BoolTensor] = None,
        streaming_segment_duration: float = 60.0,
    ) -> torch.Tensor:
        """Encode raw 24 kHz audio into LM-space speech embeddings (port of upstream `encode_speech`)."""
        speech_tensors = speech_tensors.to(self.compute_dtype)
        if speech_tensors.ndim == 1:
            speech_tensors = speech_tensors.unsqueeze(0)

        batch_size, total_samples = speech_tensors.shape
        segment_samples = int(streaming_segment_duration * ASR_SAMPLE_RATE)
        acoustic_tokenizer = self.model.acoustic_tokenizer
        semantic_tokenizer = self.model.semantic_tokenizer

        if total_samples <= segment_samples:
            encoder_output = acoustic_tokenizer.encode(speech_tensors.unsqueeze(1))
            audio_tokens = encoder_output.sample(dist_type=acoustic_tokenizer.std_dist_type)[0]
            acoustic_features = self.model.acoustic_connector(audio_tokens)

            semantic_tokens = semantic_tokenizer.encode(speech_tensors.unsqueeze(1)).mean
            semantic_features = self.model.semantic_connector(semantic_tokens)
        else:
            # Segmenting long audio keeps conv intermediates small; the streaming cache makes it seamless.
            acoustic_cache = VibeVoiceTokenizerStreamingCache()
            semantic_cache = VibeVoiceTokenizerStreamingCache()
            acoustic_means = []
            semantic_means = []
            sample_indices = torch.arange(batch_size, device=speech_tensors.device)

            segments = [(start, min(start + segment_samples, total_samples))
                        for start in range(0, total_samples, segment_samples)]
            for seg_idx, (start, end) in enumerate(segments):
                chunk = speech_tensors[:, start:end].contiguous()
                is_final = seg_idx == len(segments) - 1

                acoustic_means.append(acoustic_tokenizer.encode(
                    chunk.unsqueeze(1), cache=acoustic_cache, sample_indices=sample_indices,
                    use_cache=True, is_final_chunk=is_final,
                ).mean)
                semantic_means.append(semantic_tokenizer.encode(
                    chunk.unsqueeze(1), cache=semantic_cache, sample_indices=sample_indices,
                    use_cache=True, is_final_chunk=is_final,
                ).mean)

            # Sample once over the whole sequence, as upstream does, so the noise draw matches.
            acoustic_output = VibeVoiceTokenizerEncoderOutput(
                mean=torch.cat(acoustic_means, dim=1).contiguous(), std=acoustic_tokenizer.fix_std
            )
            audio_tokens = acoustic_output.sample(dist_type=acoustic_tokenizer.std_dist_type)[0]
            acoustic_features = self.model.acoustic_connector(audio_tokens)

            semantic_tokens = torch.cat(semantic_means, dim=1).contiguous()
            semantic_features = self.model.semantic_connector(semantic_tokens)

        if speech_masks is not None:
            return acoustic_features[speech_masks] + semantic_features[speech_masks]
        return acoustic_features + semantic_features

    def _lm_step(self, inputs_embeds: torch.Tensor, cache: DynamicCache) -> torch.Tensor:
        outputs = self.model.language_model(inputs_embeds=inputs_embeds, past_key_values=cache, use_cache=True)
        # Only the last position is needed; full-prompt logits for long audio would take ~16 GB.
        return self.lm_head(outputs.last_hidden_state[:, -1:, :])[:, -1, :].float()

    @staticmethod
    def build_logits_processor(temperature: float, top_k: int, top_p: float,
                               repetition_penalty: float) -> LogitsProcessorList:
        """Mirror the HF `generate` processor order so sampling matches upstream."""
        processors = LogitsProcessorList()
        if repetition_penalty is not None and repetition_penalty != 1.0:
            processors.append(RepetitionPenaltyLogitsProcessor(penalty=repetition_penalty))
        if temperature > 0:
            if temperature != 1.0:
                processors.append(TemperatureLogitsWarper(temperature))
            if top_k is not None and top_k != 0:
                processors.append(TopKLogitsWarper(top_k=top_k, min_tokens_to_keep=1))
            if top_p is not None and top_p < 1.0:
                processors.append(TopPLogitsWarper(top_p=top_p, min_tokens_to_keep=1))
        return processors

    @torch.no_grad()
    def generate(
        self,
        input_ids: torch.LongTensor,
        acoustic_input_mask: torch.BoolTensor,
        speech_tensors: torch.FloatTensor,
        speech_masks: Optional[torch.BoolTensor] = None,
        max_new_tokens: int = 8192,
        temperature: float = 0.0,
        top_k: int = 50,
        top_p: float = 1.0,
        repetition_penalty: float = 1.0,
        eos_token_id: Optional[Union[int, List[int]]] = None,
        stop_check_fn: Optional[Callable[[], bool]] = None,
        token_callback: Optional[Callable[[int], None]] = None,
        show_progress_bar: bool = True,
        **kwargs,
    ) -> VibeVoiceASROutput:
        """Transcribe one prompt (batch size 1). Greedy when temperature == 0, otherwise sampling."""
        device = self.device
        input_ids = input_ids.to(device)
        assert input_ids.shape[0] == 1, "Currently only supports batch size == 1"
        acoustic_input_mask = acoustic_input_mask.to(device).bool()
        if speech_masks is not None:
            speech_masks = speech_masks.to(device).bool()

        if eos_token_id is None:
            eos_token_id = self.config.decoder_config.eos_token_id
        eos_ids = set(eos_token_id if isinstance(eos_token_id, (list, tuple)) else [eos_token_id])

        prompt_len = input_ids.shape[1]
        context_left = self.config.decoder_config.max_position_embeddings - prompt_len
        if context_left <= 0:
            raise ValueError(f"Prompt length {prompt_len} exceeds the model context "
                             f"({self.config.decoder_config.max_position_embeddings})")
        max_new_tokens = min(max_new_tokens, context_left)

        start = time.time()
        inputs_embeds = self.model.language_model.embed_tokens(input_ids)
        speech_features = self.encode_speech(speech_tensors.to(device), speech_masks)
        num_speech_positions = int(acoustic_input_mask.sum().item())
        if speech_features.shape[0] != num_speech_positions:
            raise ValueError(f"Speech encoder produced {speech_features.shape[0]} embeddings, "
                             f"but the prompt has {num_speech_positions} speech positions")
        inputs_embeds[acoustic_input_mask] = speech_features.to(inputs_embeds.dtype)
        encode_time = time.time() - start

        start = time.time()
        cache = DynamicCache()
        logits = self._lm_step(inputs_embeds, cache)
        prefill_time = time.time() - start

        logits_processor = self.build_logits_processor(temperature, top_k, top_p, repetition_penalty)
        do_sample = temperature > 0
        sequences = input_ids
        generated_ids: List[int] = []
        reach_max_new_tokens = False
        stopped_externally = False
        progress_bar = tqdm(total=max_new_tokens, leave=False, desc="ASR decoding") if show_progress_bar else None

        start = time.time()
        for step in range(max_new_tokens):
            if stop_check_fn is not None and stop_check_fn():
                stopped_externally = True
                break

            scores = logits_processor(sequences, logits)
            if do_sample:
                probs = nn.functional.softmax(scores, dim=-1)
                next_token = torch.multinomial(probs, num_samples=1).squeeze(1)
            else:
                next_token = torch.argmax(scores, dim=-1)

            sequences = torch.cat([sequences, next_token[:, None]], dim=-1)
            token_id = int(next_token.item())
            generated_ids.append(token_id)
            if progress_bar is not None:
                progress_bar.update(1)
            if token_callback is not None:
                token_callback(token_id)

            if token_id in eos_ids:
                break
            if step == max_new_tokens - 1:
                reach_max_new_tokens = True
                break

            logits = self._lm_step(self.model.language_model.embed_tokens(next_token[:, None]), cache)
        decode_time = time.time() - start

        if progress_bar is not None:
            progress_bar.close()

        return VibeVoiceASROutput(
            sequences=sequences,
            generated_ids=generated_ids,
            reach_max_new_tokens=reach_max_new_tokens,
            stopped_externally=stopped_externally,
            prompt_tokens=prompt_len,
            speech_tokens=num_speech_positions,
            encode_time=encode_time,
            prefill_time=prefill_time,
            decode_time=decode_time,
        )

    @classmethod
    def from_pretrain(cls, model_path: str, config: VibeVoiceASRConfig, device="cuda", offload_config=None,
                      dtype: Optional[torch.dtype] = None):
        """Load from an HF sharded directory (model.safetensors.index.json) or a single safetensors file."""
        from util.safetensors_util import MultipleSafetensorLoader, MemoryEfficientSafeOpen
        from vibevoice.modular.custom_offloading_utils import LayerOffloader

        with init_empty_weights():
            model = cls(config)

        state_dict = {}
        index_file = os.path.join(model_path, "model.safetensors.index.json")
        if os.path.isdir(model_path) and os.path.exists(index_file):
            print(f"Begin to load model from model path {model_path}")
            state_dict = MultipleSafetensorLoader(index_file).load_dict()
        else:
            model_file = os.path.join(model_path, "model.safetensors") if os.path.isdir(model_path) else model_path
            print(f"Begin to load model from mono model file {model_file}")
            with MemoryEfficientSafeOpen(model_file) as safe:
                for key in safe.keys():
                    state_dict[key] = safe.get_tensor(key)

        ignored = [k for k in state_dict if k.startswith(IGNORED_CHECKPOINT_PREFIXES)]
        for key in ignored:
            del state_dict[key]
        if ignored:
            print(f"Ignored {len(ignored)} acoustic decoder weights (not used by ASR)")

        # Cast weights rather than calling model.to(dtype), which would also downcast the fp32 RoPE inv_freq buffer.
        if dtype is not None:
            for key in list(state_dict.keys()):
                state_dict[key] = state_dict[key].to(dtype)

        missing, unexpected = model.load_state_dict(state_dict, strict=False, assign=True)
        if missing or unexpected:
            raise RuntimeError(f"Checkpoint mismatch. missing={missing[:10]} unexpected={unexpected[:10]}")
        del state_dict
        print("Model loaded")

        model.dtype = model.lm_head.weight.dtype

        model.device = torch.device(device)

        if offload_config is not None and offload_config.enabled:
            print(f"Setting up layer offloading: {offload_config.num_layers_on_gpu} LM layers on GPU")
            language_model = model.model.language_model
            if language_model.embed_tokens.weight.dtype == torch.float8_e4m3fn:
                print("Converting embedding layer from Float8 to BF16...")
                language_model.embed_tokens.cpu()
                language_model.embed_tokens.weight.data = language_model.embed_tokens.weight.data.to(torch.bfloat16)

            for name, module in model.model.named_children():
                if name != "language_model":
                    module.to(device)
            language_model.embed_tokens.to(device)
            language_model.norm.to(device)
            language_model.rotary_emb.to(device)
            model.lm_head.to(device)

            model.offloader = LayerOffloader(
                language_model=language_model,
                config=offload_config,
                device=torch.device(device),
                logger=logger
            )
            print(f"Layer offloading enabled: {len(model.offloader.offloaded_layers)} LM layers offloaded")
        else:
            model.to(device)
            print(f"Model moved to {device} (no offloading)")

        return model


__all__ = [
    "VibeVoiceASROutput",
    "VibeVoiceASRModel",
    "VibeVoiceASRForConditionalInference",
]
