import copy
import os
from typing import Callable, Dict, Optional, Union

import torch
import torch.nn as nn
from tqdm import tqdm

from accelerate import init_empty_weights
from transformers.cache_utils import Cache, DynamicCache
from transformers.modeling_outputs import BaseModelOutputWithPast

from config.configuration_vibevoice import QwenConfig, VibeVoiceStreamingConfig
from vibevoice.modular.modeling_vibevoice import SpeechConnector
from vibevoice.modular.modeling_vibevoice_inference import VibeVoiceGenerationOutput
from vibevoice.modular.modular_vibevoice_diffusion_head import VibeVoiceDiffusionHead
from vibevoice.modular.modular_vibevoice_qwen import QwenModel
from vibevoice.modular.modular_vibevoice_tokenizer import (
    VibeVoiceAcousticTokenizerModel,
    VibeVoiceTokenizerStreamingCache,
)
from vibevoice.modular.streamer import AsyncAudioStreamer, AudioStreamer
from vibevoice.schedule.dpm_solver import DPMSolverMultistepScheduler
from util.float8_scale import AutoCast
from util.logger import get_logger
from util.rand_init import get_generator

logger = get_logger(__name__)

TTS_TEXT_WINDOW_SIZE = 5
TTS_SPEECH_WINDOW_SIZE = 6
NEGATIVE_TEXT_TOKEN = "<|image_pad|>"


def _resolve_dtype(dtype) -> torch.dtype:
    if dtype is None:
        return torch.float32
    if isinstance(dtype, str):
        return getattr(torch, dtype)
    return dtype


class StreamingQwenModel(QwenModel):
    """QwenModel that supports multi-token forwards on top of a non-empty KV cache."""

    @torch.no_grad()
    def forward(
        self,
        input_ids: Optional[torch.LongTensor] = None,
        attention_mask: Optional[torch.Tensor] = None,
        position_ids: Optional[torch.LongTensor] = None,
        past_key_values: Optional[Cache] = None,
        inputs_embeds: Optional[torch.FloatTensor] = None,
        use_cache: Optional[bool] = None,
        cache_position: Optional[torch.LongTensor] = None,
        **kwargs,
    ) -> BaseModelOutputWithPast:
        use_cache = use_cache if use_cache is not None else self.config.use_cache

        if (input_ids is None) ^ (inputs_embeds is not None):
            raise ValueError("You must specify exactly one of input_ids or inputs_embeds")

        if inputs_embeds is None:
            inputs_embeds = self.embed_tokens(input_ids)

        if use_cache and past_key_values is None:
            past_key_values = DynamicCache()

        past_seen_tokens = past_key_values.get_seq_length() if past_key_values is not None else 0
        q_len = inputs_embeds.shape[1]
        if cache_position is None:
            cache_position = torch.arange(past_seen_tokens, past_seen_tokens + q_len, device=inputs_embeds.device)

        if position_ids is None:
            position_ids = cache_position.unsqueeze(0)

        # SDPA's is_causal aligns the mask top-left, which is wrong when q_len > 1 and the
        # cache is non-empty (text windows are fed on top of the voice prompt cache).
        causal_mask = None
        if q_len > 1:
            kv_positions = torch.arange(past_seen_tokens + q_len, device=inputs_embeds.device)
            causal_mask = (kv_positions[None, :] <= cache_position[:, None])[None, None, :, :]

        hidden_states = inputs_embeds
        position_embeddings = self.rotary_emb(hidden_states, position_ids)

        for decoder_layer in self.layers[: self.config.num_hidden_layers]:
            layer_outputs = decoder_layer(
                hidden_states,
                attention_mask=causal_mask,
                position_ids=position_ids,
                past_key_value=past_key_values,
                use_cache=use_cache,
                cache_position=cache_position,
                position_embeddings=position_embeddings,
            )
            hidden_states = layer_outputs[0]

        hidden_states = self.norm(hidden_states)

        return BaseModelOutputWithPast(
            last_hidden_state=hidden_states,
            past_key_values=past_key_values if use_cache else None,
        )


class BinaryClassifier(nn.Module):
    def __init__(self, hidden_size: int):
        super().__init__()
        self.fc1 = AutoCast.Linear(hidden_size, hidden_size)
        self.fc2 = AutoCast.Linear(hidden_size, 1)

    def forward(self, x):
        x = torch.relu(self.fc1(x))
        return self.fc2(x)


class VibeVoiceStreamingModel(nn.Module):
    def __init__(self, config: VibeVoiceStreamingConfig):
        super().__init__()
        dtype = _resolve_dtype(getattr(config, "torch_dtype", None))
        self.dtype = dtype

        # The base 24-layer Qwen is split: the lower layers encode text only, the upper
        # `tts_backbone_num_hidden_layers` layers form the TTS backbone.
        lm_config = QwenConfig.from_config(config.decoder_config)
        lm_config.num_hidden_layers = lm_config.num_hidden_layers - config.tts_backbone_num_hidden_layers
        self.language_model = StreamingQwenModel(lm_config, dtype=dtype)
        self.language_model.norm = nn.Identity()

        tts_lm_config = copy.deepcopy(lm_config)
        tts_lm_config.num_hidden_layers = config.tts_backbone_num_hidden_layers
        self.tts_language_model = StreamingQwenModel(tts_lm_config, dtype=dtype)

        self.tts_input_types = AutoCast.Embedding(2, lm_config.hidden_size).to(dtype)

        self.acoustic_tokenizer = VibeVoiceAcousticTokenizerModel(config.acoustic_tokenizer_config, dtype=dtype).to(dtype)
        # The realtime checkpoint ships only the decoder; prompts come pre-encoded as KV caches.
        self.acoustic_tokenizer.encoder = None
        self.acoustic_connector = SpeechConnector(config.acoustic_vae_dim, lm_config.hidden_size, dtype=dtype).to(dtype)

        self.register_buffer('speech_scaling_factor', torch.tensor(float('nan')))
        self.register_buffer('speech_bias_factor', torch.tensor(float('nan')))

        self.prediction_head = VibeVoiceDiffusionHead(config.diffusion_head_config, dtype=dtype).to(dtype)

        self.noise_scheduler = DPMSolverMultistepScheduler(
            num_train_timesteps=config.diffusion_head_config.ddpm_num_steps,
            beta_schedule=config.diffusion_head_config.ddpm_beta_schedule,
            prediction_type=config.diffusion_head_config.prediction_type
        )


class VibeVoiceStreamingForConditionalInference(nn.Module):
    """Inference wrapper for the VibeVoice Realtime (streaming) model."""
    config_class = VibeVoiceStreamingConfig

    def __init__(self, config: VibeVoiceStreamingConfig):
        super().__init__()
        self.config = config
        self.model = VibeVoiceStreamingModel(config)
        self.tts_eos_classifier = BinaryClassifier(config.decoder_config.hidden_size)

        self.ddpm_inference_steps = config.diffusion_head_config.ddpm_num_inference_steps
        self.dtype = self.model.dtype
        self.device = torch.device("cuda")
        self.offloader = None

    def __del__(self):
        if getattr(self, "offloader", None) is not None:
            try:
                self.offloader.cleanup()
            except Exception:
                pass

    def set_ddpm_inference_steps(self, num_steps=None):
        self.ddpm_inference_steps = num_steps or self.config.diffusion_head_config.ddpm_num_inference_steps

    def forward_lm(self, input_ids: torch.LongTensor, past_key_values: Cache) -> BaseModelOutputWithPast:
        """Run new text tokens through the lower (text-only) LM."""
        inputs_embeds = self.model.language_model.embed_tokens(input_ids)
        return self.model.language_model(inputs_embeds=inputs_embeds, past_key_values=past_key_values, use_cache=True)

    def forward_tts_lm(self, hidden_states: torch.Tensor, past_key_values: Cache, is_text: bool) -> BaseModelOutputWithPast:
        """Run text hidden states (is_text=True) or acoustic embeddings through the TTS LM."""
        type_ids = torch.full(hidden_states.shape[:2], 1 if is_text else 0, dtype=torch.long, device=hidden_states.device)
        inputs_embeds = hidden_states + self.model.tts_input_types(type_ids).to(hidden_states.dtype)
        return self.model.tts_language_model(inputs_embeds=inputs_embeds, past_key_values=past_key_values, use_cache=True)

    def _ensure_prediction_head_on_gpu(self):
        if self.offloader and self.offloader.config.offload_prediction_head:
            if next(self.model.prediction_head.parameters()).device.type == 'cpu':
                self.model.prediction_head.to(self.device)

    def _move_prediction_head_to_cpu(self):
        if self.offloader and self.offloader.config.offload_prediction_head:
            if next(self.model.prediction_head.parameters()).device.type != 'cpu':
                self.model.prediction_head.cpu()
                if torch.cuda.is_available():
                    torch.cuda.empty_cache()

    @torch.no_grad()
    def sample_speech_tokens(self, condition, neg_condition, cfg_scale=1.5):
        self._ensure_prediction_head_on_gpu()
        try:
            self.model.noise_scheduler.set_timesteps(self.ddpm_inference_steps)
            head_param = next(self.model.prediction_head.parameters())
            condition = torch.cat([condition, neg_condition], dim=0).to(head_param.device)

            speech = torch.randn(condition.shape[0], self.config.acoustic_vae_dim, generator=get_generator()).to(condition)

            for t in self.model.noise_scheduler.timesteps:
                half = speech[: len(speech) // 2]
                combined = torch.cat([half, half], dim=0)
                eps = self.model.prediction_head(combined, t.repeat(combined.shape[0]).to(combined), condition=condition)
                cond_eps, uncond_eps = torch.split(eps, len(eps) // 2, dim=0)
                half_eps = uncond_eps + cfg_scale * (cond_eps - uncond_eps)
                eps = torch.cat([half_eps, half_eps], dim=0)
                speech = self.model.noise_scheduler.step(eps, t, speech).prev_sample

            return speech[: len(speech) // 2]
        finally:
            self._move_prediction_head_to_cpu()

    @torch.no_grad()
    def generate(
        self,
        tts_text_ids: torch.LongTensor,
        all_prefilled_outputs: Dict[str, BaseModelOutputWithPast],
        cfg_scale: float = 1.5,
        audio_streamer: Optional[Union[AudioStreamer, AsyncAudioStreamer]] = None,
        stop_check_fn: Optional[Callable[[], bool]] = None,
        max_new_tokens: Optional[int] = None,
        return_speech: bool = True,
        verbose: bool = False,
        show_progress_bar: bool = True,
        tts_lm_input_ids: Optional[torch.LongTensor] = None,
        **kwargs,
    ) -> VibeVoiceGenerationOutput:
        """Interleave 5-token text windows with 6-step speech sampling; batch size 1 only.

        `all_prefilled_outputs` holds the voice prompt caches (`lm`, `tts_lm`, `neg_tts_lm`);
        they are mutated in place, so pass a deep copy when reusing a preset.
        """
        device = self.device
        tts_text_ids = tts_text_ids.to(device)
        assert tts_text_ids.shape[0] == 1, "Currently only supports batch size == 1"

        lm_outputs = all_prefilled_outputs["lm"]
        tts_lm_outputs = all_prefilled_outputs["tts_lm"]
        neg_tts_lm_outputs = all_prefilled_outputs["neg_tts_lm"]
        lm_cache = lm_outputs.past_key_values
        tts_lm_cache = tts_lm_outputs.past_key_values
        neg_tts_lm_cache = neg_tts_lm_outputs.past_key_values

        prompt_len = tts_lm_cache.get_seq_length()
        if tts_lm_input_ids is not None:
            prompt_len = tts_lm_input_ids.shape[-1]
        if max_new_tokens is None:
            max_new_tokens = self.config.decoder_config.max_position_embeddings - prompt_len
        max_length = prompt_len + max_new_tokens

        if tts_lm_input_ids is None:
            tts_lm_input_ids = torch.zeros((1, prompt_len), dtype=torch.long)
        sequences = [tts_lm_input_ids.to(device)]
        seq_len = prompt_len
        acoustic_cache = VibeVoiceTokenizerStreamingCache()
        audio_chunks = []
        reach_max_step = False
        finished = False
        diffusion_indices = torch.LongTensor([0])
        window_index = 0
        generated_speech_tokens = 0
        prefilled_text_tokens = 0

        progress_bar = tqdm(total=max_length, initial=seq_len, leave=False) if show_progress_bar else None

        def _update_progress(n):
            if progress_bar is not None:
                progress_bar.update(n)
                progress_bar.set_description(
                    f"Prefilled {prefilled_text_tokens} text tokens, generated {generated_speech_tokens} speech tokens, "
                    f"current step ({seq_len} / {max_length})"
                )

        while not finished:
            if stop_check_fn is not None and stop_check_fn():
                if verbose:
                    print(f"Generation stopped externally at step {seq_len + 1}")
                break

            text_window = tts_text_ids[:, window_index * TTS_TEXT_WINDOW_SIZE:(window_index + 1) * TTS_TEXT_WINDOW_SIZE]
            window_index += 1

            if text_window.shape[1] > 0:
                seq_len += text_window.shape[1]
                sequences.append(text_window)
                if seq_len > max_length:
                    reach_max_step = True
                    break
                prefilled_text_tokens += text_window.shape[1]
                _update_progress(text_window.shape[1])

                lm_outputs = self.forward_lm(text_window, lm_cache)
                tts_lm_outputs = self.forward_tts_lm(lm_outputs.last_hidden_state, tts_lm_cache, is_text=True)

            for _ in range(TTS_SPEECH_WINDOW_SIZE):
                positive_condition = tts_lm_outputs.last_hidden_state[diffusion_indices, -1, :]
                negative_condition = neg_tts_lm_outputs.last_hidden_state[diffusion_indices, -1, :]

                speech_latent = self.sample_speech_tokens(positive_condition, negative_condition, cfg_scale=cfg_scale).unsqueeze(1)

                scaled_latent = speech_latent / self.model.speech_scaling_factor.to(speech_latent.device) \
                    - self.model.speech_bias_factor.to(speech_latent.device)
                audio_chunk = self.model.acoustic_tokenizer.decode(
                    scaled_latent.to(device),
                    cache=acoustic_cache,
                    sample_indices=diffusion_indices.to(device),
                    use_cache=True,
                    debug=False,
                )
                audio_chunks.append(audio_chunk[0])
                if audio_streamer is not None:
                    audio_streamer.put(audio_chunk, diffusion_indices)

                acoustic_embed = self.model.acoustic_connector(speech_latent)
                seq_len += 1
                sequences.append(torch.ones((1, 1), dtype=torch.long, device=device))
                if seq_len > max_length:
                    break
                generated_speech_tokens += 1
                _update_progress(1)

                tts_lm_outputs = self.forward_tts_lm(acoustic_embed, tts_lm_cache, is_text=False)
                neg_tts_lm_outputs = self.forward_tts_lm(acoustic_embed, neg_tts_lm_cache, is_text=False)

                eos_prob = torch.sigmoid(self.tts_eos_classifier(tts_lm_outputs.last_hidden_state[diffusion_indices, -1, :]))
                if eos_prob[0].item() > 0.5:
                    finished = True
                    if audio_streamer is not None:
                        audio_streamer.end(diffusion_indices)
                    break

            if not finished and seq_len > max_length:
                reach_max_step = True
                break

        if progress_bar is not None:
            progress_bar.close()
        if audio_streamer is not None:
            audio_streamer.end()
        if reach_max_step:
            print(f"Reached maximum generation length {max_length}, stopped it.")

        final_audio = torch.cat(audio_chunks, dim=-1) if audio_chunks else None
        return VibeVoiceGenerationOutput(
            sequences=torch.cat(sequences, dim=-1),
            speech_outputs=[final_audio] if return_speech else None,
            reach_max_step_sample=torch.tensor([reach_max_step], device=device),
        )

    @classmethod
    def from_pretrain(cls, model_path: str, config: VibeVoiceStreamingConfig, device="cuda", offload_config=None,
                      dtype: Optional[torch.dtype] = None):
        """Load from a safetensors file or a directory with model.safetensors[.index.json]."""
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

        missing, unexpected = model.load_state_dict(state_dict, strict=False, assign=True)
        if missing or unexpected:
            raise RuntimeError(f"Checkpoint mismatch. missing={missing[:10]} unexpected={unexpected[:10]}")
        print("Model loaded")

        if dtype is not None:
            model.to(dtype=dtype)
            model.dtype = dtype

        model.device = torch.device(device)

        if offload_config is not None and offload_config.enabled:
            print(f"Setting up layer offloading: {offload_config.num_layers_on_gpu} TTS layers on GPU")
            embed = model.model.language_model.embed_tokens
            if embed.weight.dtype == torch.float8_e4m3fn:
                print("Converting embedding layer from Float8 to BF16...")
                embed.cpu()
                embed.weight.data = embed.weight.data.to(torch.bfloat16)

            for name, module in model.model.named_children():
                if name not in ("tts_language_model", "prediction_head", "noise_scheduler"):
                    module.to(device)
            model.model.speech_scaling_factor = model.model.speech_scaling_factor.to(device)
            model.model.speech_bias_factor = model.model.speech_bias_factor.to(device)
            model.tts_eos_classifier.to(device)

            tts_lm = model.model.tts_language_model
            tts_lm.norm.to(device)
            tts_lm.rotary_emb.to(device)

            if offload_config.offload_prediction_head:
                model.model.prediction_head.cpu()
                print("Prediction head offloaded to CPU (will transfer on-demand)")
            else:
                model.model.prediction_head.to(device)

            model.offloader = LayerOffloader(
                language_model=tts_lm,
                config=offload_config,
                device=torch.device(device),
                logger=logger
            )
            print(f"Layer offloading enabled: {len(model.offloader.offloaded_layers)} TTS layers offloaded")
        else:
            model.to(device)
            print(f"Model moved to {device} (no offloading)")

        return model


__all__ = [
    "StreamingQwenModel",
    "VibeVoiceStreamingModel",
    "VibeVoiceStreamingForConditionalInference",
]
