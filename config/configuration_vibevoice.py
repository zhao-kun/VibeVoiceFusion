""" VibeVoice_AcousticTokenizer model configuration"""
from enum import Enum
from typing import Dict, List, Optional

from util.logger import get_logger
from dataclasses import dataclass


logger = get_logger(__name__)

class InferencePhase:
    PENDING: str = 'pending'
    PREPROCESSING: str = 'preprocessing'
    INFERENCING: str = 'inferencing'
    SAVING_AUDIO: str = 'saving_audio'
    FAILED: str = 'failed'
    COMPLETED: str = 'completed'

@dataclass
class QwenConfig:
    """
    Configuration class for QWen model.
    """
    model_type = "qwen2"

    def __init__(
        self,
        attention_dropout: float = 0.0,
        hidden_act: str = "silu",
        hidden_size: int = 3584,
        initializer_range: float = 0.02,
        intermediate_size: int = 18944,
        max_position_embeddings: int = 32768,
        max_window_layers: int = 28,
        num_attention_heads: int = 28,
        num_hidden_layers: int = 28,
        num_key_value_heads: int = 4,
        rms_norm_eps: float = 1e-06,
        rope_scaling: Optional[dict] = None,
        rope_theta: float = 1000000.0,
        sliding_window: Optional[int] = None,
        torch_dtype: str = "bfloat16",
        use_cache: bool = True,
        use_mrope: bool = False,
        use_sliding_window: bool = False,
        vocab_size: int = 152064,
        bos_token_id: int = 151643,
        eos_token_id: int = 151643,
        pad_token_id: Optional[int] = None,
        **kwargs
    ):
        self.attention_dropout = attention_dropout
        self.hidden_act = hidden_act
        self.hidden_size = hidden_size
        self.initializer_range = initializer_range
        self.intermediate_size = intermediate_size
        self.max_position_embeddings = max_position_embeddings
        self.max_window_layers = max_window_layers
        self.num_attention_heads = num_attention_heads
        self.num_hidden_layers = num_hidden_layers
        self.num_key_value_heads = num_key_value_heads
        self.rms_norm_eps = rms_norm_eps
        self.rope_scaling = rope_scaling
        self.rope_theta = rope_theta
        self.sliding_window = sliding_window
        self.torch_dtype = torch_dtype
        self.use_cache = use_cache
        self.use_mrope = use_mrope
        self.use_sliding_window = use_sliding_window
        self.vocab_size = vocab_size
        self.bos_token_id = bos_token_id
        self.eos_token_id = eos_token_id
        self.pad_token_id = pad_token_id
        self.use_cache = use_cache

    @classmethod
    def from_config(cls, config):
        return cls(**config.__dict__)

class VibeVoiceAcousticTokenizerConfig:
    model_type = "vibevoice_acoustic_tokenizer"

    def __init__(
        self,
        channels: int = 1,
        corpus_normalize: float = 0.0,
        causal: bool = True,
        vae_dim: int = 64,
        fix_std: float = 0.5,
        std_dist_type: str = 'gaussian',
        # common
        mixer_layer: str = 'depthwise_conv',
        conv_norm: str = 'none',
        pad_mode: str = 'constant',
        disable_last_norm: bool = True,
        layernorm: str = 'RMSNorm',
        layernorm_eps: float = 1e-5,
        layernorm_elementwise_affine: bool = True,
        conv_bias: bool = True,
        layer_scale_init_value: float = 1e-6,
        weight_init_value: float = 1e-2,
        # encoder specific
        encoder_n_filters: int = 32,
        encoder_ratios: Optional[List[int]] = [8, 5, 5, 4, 2, 2],
        encoder_depths: str = "3-3-3-3-3-3-8",
        # decoder specific
        decoder_n_filters: int = 32,
        decoder_ratios: Optional[List[int]] = None,  # if None, same as encoder
        decoder_depths: Optional[str] = None,
        **kwargs
    ):
        self.channels = channels
        self.corpus_normalize = corpus_normalize
        self.causal = causal
        self.vae_dim = vae_dim
        self.fix_std = fix_std
        self.std_dist_type = std_dist_type

        # common parameters
        self.conv_norm = conv_norm
        self.pad_mode = pad_mode
        self.layernorm_eps = layernorm_eps
        self.disable_last_norm = disable_last_norm
        self.layernorm = layernorm
        self.layernorm_elementwise_affine = layernorm_elementwise_affine
        self.conv_bias = conv_bias
        self.layer_scale_init_value = layer_scale_init_value
        self.weight_init_value = weight_init_value
        self.mixer_layer = mixer_layer

        # encoder specific parameters
        self.encoder_n_filters = encoder_n_filters
        self.encoder_ratios = encoder_ratios
        self.encoder_depths = encoder_depths

        # decoder specific parameters
        self.decoder_ratios = decoder_ratios if decoder_ratios is not None else encoder_ratios
        self.decoder_n_filters = decoder_n_filters
        self.decoder_depths = decoder_depths


class VibeVoiceSemanticTokenizerConfig:
    model_type = "vibevoice_semantic_tokenizer"

    def __init__(
        self,
        channels: int = 1,
        corpus_normalize: float = 0.0,
        causal: bool = True,
        vae_dim: int = 64,
        fix_std: float = 0,
        std_dist_type: str = 'none',
        # common
        mixer_layer: str = 'depthwise_conv',
        conv_norm: str = 'none',
        pad_mode: str = 'constant',
        disable_last_norm: bool = True,
        layernorm: str = 'RMSNorm',
        layernorm_eps: float = 1e-5,
        layernorm_elementwise_affine: bool = True,
        conv_bias: bool = True,
        layer_scale_init_value: float = 1e-6,
        weight_init_value: float = 1e-2,
        # encoder specific
        encoder_n_filters: int = 32,
        encoder_ratios: Optional[List[int]] = [8, 5, 5, 4, 2, 2],
        encoder_depths: str = "3-3-3-3-3-3-8",
        **kwargs
    ):
        self.channels = channels
        self.corpus_normalize = corpus_normalize
        self.causal = causal
        self.vae_dim = vae_dim
        self.fix_std = fix_std
        self.std_dist_type = std_dist_type

        # common parameters
        self.conv_norm = conv_norm
        self.pad_mode = pad_mode
        self.layernorm_eps = layernorm_eps
        self.disable_last_norm = disable_last_norm
        self.layernorm = layernorm
        self.layernorm_elementwise_affine = layernorm_elementwise_affine
        self.conv_bias = conv_bias
        self.layer_scale_init_value = layer_scale_init_value
        self.weight_init_value = weight_init_value
        self.mixer_layer = mixer_layer

        # encoder specific parameters
        self.encoder_n_filters = encoder_n_filters
        self.encoder_ratios = encoder_ratios
        self.encoder_depths = encoder_depths


class VibeVoiceDiffusionHeadConfig:
    model_type = "vibevoice_diffusion_head"

    def __init__(
        self,
        hidden_size=768,
        head_layers=4,
        head_ffn_ratio=3.0,
        rms_norm_eps=1e-5,
        latent_size=64,
        speech_vae_dim=None,
        prediction_type="v_prediction",
        diffusion_type="ddpm",
        ddpm_num_steps=1000,
        ddpm_num_inference_steps=20,
        ddpm_beta_schedule="cosine",
        ddpm_batch_mul=4,
        **kwargs
    ):
        self.hidden_size = hidden_size
        self.head_layers = head_layers
        self.head_ffn_ratio = head_ffn_ratio
        self.rms_norm_eps = rms_norm_eps
        self.latent_size = latent_size
        self.speech_vae_dim = speech_vae_dim
        self.prediction_type = prediction_type
        self.diffusion_type = diffusion_type
        self.ddpm_num_steps = ddpm_num_steps
        self.ddpm_num_inference_steps = ddpm_num_inference_steps
        self.ddpm_beta_schedule = ddpm_beta_schedule
        self.ddpm_batch_mul = ddpm_batch_mul

class VibeVoiceConfig:
    model_type = "vibevoice"
    is_composition = True
    is_encoder_decoder = False
    sub_configs = {
        "acoustic_tokenizer_config": VibeVoiceAcousticTokenizerConfig,
        "semantic_tokenizer_config": VibeVoiceSemanticTokenizerConfig,
        "decoder_config": QwenConfig,
        "diffusion_head_config": VibeVoiceDiffusionHeadConfig,
    }
    # keys_to_ignore_at_inference = ["past_key_values"]
    # Default tensor parallel plan for base model `Qwen2`
    base_model_tp_plan = {
        "layers.*.self_attn.q_proj": "colwise",
        "layers.*.self_attn.k_proj": "colwise",
        "layers.*.self_attn.v_proj": "colwise",
        "layers.*.self_attn.o_proj": "rowwise",
        "layers.*.mlp.gate_proj": "colwise",
        "layers.*.mlp.up_proj": "colwise",
        "layers.*.mlp.down_proj": "rowwise",
    }

    def __init__(
        self,
        acoustic_tokenizer_config=None,
        semantic_tokenizer_config=None,
        decoder_config=None,
        diffusion_head_config=None,
        **kwargs
    ):

        # kwargs["_attn_implementation"] = "flash_attention_2"
        kwargs["_attn_implementation_autoset"] = False

        if acoustic_tokenizer_config is None:
            self.acoustic_tokenizer_config = self.sub_configs["acoustic_tokenizer_config"]()
        elif isinstance(acoustic_tokenizer_config, dict):
            acoustic_tokenizer_config["model_type"] = "vibevoice_acoustic_tokenizer"
            self.acoustic_tokenizer_config = self.sub_configs["acoustic_tokenizer_config"](**acoustic_tokenizer_config)
        elif isinstance(acoustic_tokenizer_config, VibeVoiceAcousticTokenizerConfig):
            # If an instance of the config class is provided
            self.acoustic_tokenizer_config = acoustic_tokenizer_config

        if semantic_tokenizer_config is None:
            self.semantic_tokenizer_config = self.sub_configs["semantic_tokenizer_config"]()
        elif isinstance(semantic_tokenizer_config, dict):
            semantic_tokenizer_config["model_type"] = "vibevoice_semantic_tokenizer"
            self.semantic_tokenizer_config = self.sub_configs["semantic_tokenizer_config"](**semantic_tokenizer_config)
        elif isinstance(semantic_tokenizer_config, VibeVoiceSemanticTokenizerConfig):
            # If an instance of the config class is provided
            self.semantic_tokenizer_config = semantic_tokenizer_config

        if decoder_config is None:
            self.decoder_config = self.sub_configs["decoder_config"]()
        elif isinstance(decoder_config, dict):
            # If a dictionary is provided, instantiate the config class with it
            # self.decoder_config = self.sub_configs["decoder_config"](**decoder_config)
            if decoder_config.get("model_type", '') == "qwen2":
                self.decoder_config = QwenConfig(**decoder_config)
            else:
                raise ValueError(f"Unsupported decoder model type: {decoder_config.get('model_type', '')}")
        elif isinstance(decoder_config, (QwenConfig,)):
            # If an instance of the config class is provided
            self.decoder_config = decoder_config

        if diffusion_head_config is None:
            self.diffusion_head_config = self.sub_configs["diffusion_head_config"]()
        elif isinstance(diffusion_head_config, dict):
            diffusion_head_config["model_type"] = "vibevoice_diffusion_head"
            self.diffusion_head_config = self.sub_configs["diffusion_head_config"](**diffusion_head_config)
        elif isinstance(diffusion_head_config, VibeVoiceDiffusionHeadConfig):
            # If an instance of the config class is provided
            self.diffusion_head_config = diffusion_head_config

        # other parameters
        self.acoustic_vae_dim = getattr(self.acoustic_tokenizer_config, 'vae_dim', 64)
        self.semantic_vae_dim = getattr(self.semantic_tokenizer_config, 'vae_dim', 128)

        for key, value in kwargs.items():
            setattr(self, key, value)

    @classmethod
    def from_dict(cls, config_dict: Dict, **kwargs):
        """
        Create a VibeVoiceConfig instance from a dictionary.
        """
        # Extract sub-configs from the main config dict
        acoustic_tokenizer_config = config_dict.get("acoustic_tokenizer_config", None)
        semantic_tokenizer_config = config_dict.get("semantic_tokenizer_config", None)
        decoder_config = config_dict.get("decoder_config", None)
        diffusion_head_config = config_dict.get("diffusion_head_config", None)

        # Remove sub-configs from the main dict to avoid duplication
        main_config = {k: v for k, v in config_dict.items() if k not in [
            "acoustic_tokenizer_config",
            "semantic_tokenizer_config",
            "decoder_config",
            "diffusion_head_config"
        ]}

        main_config.update(kwargs)

        return cls(
            acoustic_tokenizer_config=acoustic_tokenizer_config,
            semantic_tokenizer_config=semantic_tokenizer_config,
            decoder_config=decoder_config,
            diffusion_head_config=diffusion_head_config,
            **main_config
        )

class VibeVoiceStreamingConfig:
    """Configuration for the VibeVoice Realtime (streaming) model."""
    model_type = "vibevoice_streaming"
    is_composition = True
    is_encoder_decoder = False
    sub_configs = {
        "acoustic_tokenizer_config": VibeVoiceAcousticTokenizerConfig,
        "decoder_config": QwenConfig,
        "diffusion_head_config": VibeVoiceDiffusionHeadConfig,
    }

    def __init__(
        self,
        acoustic_tokenizer_config=None,
        decoder_config=None,
        diffusion_head_config=None,
        tts_backbone_num_hidden_layers: int = 20,
        **kwargs
    ):
        kwargs["_attn_implementation_autoset"] = False

        if acoustic_tokenizer_config is None:
            self.acoustic_tokenizer_config = VibeVoiceAcousticTokenizerConfig()
        elif isinstance(acoustic_tokenizer_config, dict):
            acoustic_tokenizer_config["model_type"] = "vibevoice_acoustic_tokenizer"
            self.acoustic_tokenizer_config = VibeVoiceAcousticTokenizerConfig(**acoustic_tokenizer_config)
        elif isinstance(acoustic_tokenizer_config, VibeVoiceAcousticTokenizerConfig):
            self.acoustic_tokenizer_config = acoustic_tokenizer_config

        if decoder_config is None:
            self.decoder_config = QwenConfig()
        elif isinstance(decoder_config, dict):
            if decoder_config.get("model_type", '') == "qwen2":
                self.decoder_config = QwenConfig(**decoder_config)
            else:
                raise ValueError(f"Unsupported decoder model type: {decoder_config.get('model_type', '')}")
        elif isinstance(decoder_config, QwenConfig):
            self.decoder_config = decoder_config

        if diffusion_head_config is None:
            self.diffusion_head_config = VibeVoiceDiffusionHeadConfig()
        elif isinstance(diffusion_head_config, dict):
            diffusion_head_config["model_type"] = "vibevoice_diffusion_head"
            self.diffusion_head_config = VibeVoiceDiffusionHeadConfig(**diffusion_head_config)
        elif isinstance(diffusion_head_config, VibeVoiceDiffusionHeadConfig):
            self.diffusion_head_config = diffusion_head_config

        self.acoustic_vae_dim = getattr(self.acoustic_tokenizer_config, 'vae_dim', 64)
        self.tts_backbone_num_hidden_layers = tts_backbone_num_hidden_layers

        for key, value in kwargs.items():
            setattr(self, key, value)

    @classmethod
    def from_dict(cls, config_dict: Dict, **kwargs):
        """Create a VibeVoiceStreamingConfig from a config.json dictionary."""
        sub_keys = ["acoustic_tokenizer_config", "decoder_config", "diffusion_head_config"]
        main_config = {k: v for k, v in config_dict.items() if k not in sub_keys}
        main_config.update(kwargs)

        return cls(
            acoustic_tokenizer_config=config_dict.get("acoustic_tokenizer_config", None),
            decoder_config=config_dict.get("decoder_config", None),
            diffusion_head_config=config_dict.get("diffusion_head_config", None),
            **main_config
        )

_default_config_json = """
{
  "acostic_vae_dim": 64,
  "acoustic_tokenizer_config": {
    "causal": true,
    "channels": 1,
    "conv_bias": true,
    "conv_norm": "none",
    "corpus_normalize": 0.0,
    "decoder_depths": null,
    "decoder_n_filters": 32,
    "decoder_ratios": [
      8,
      5,
      5,
      4,
      2,
      2
    ],
    "disable_last_norm": true,
    "encoder_depths": "3-3-3-3-3-3-8",
    "encoder_n_filters": 32,
    "encoder_ratios": [
      8,
      5,
      5,
      4,
      2,
      2
    ],
    "fix_std": 0.5,
    "layer_scale_init_value": 1e-06,
    "layernorm": "RMSNorm",
    "layernorm_elementwise_affine": true,
    "layernorm_eps": 1e-05,
    "mixer_layer": "depthwise_conv",
    "model_type": "vibevoice_acoustic_tokenizer",
    "pad_mode": "constant",
    "std_dist_type": "gaussian",
    "vae_dim": 64,
    "weight_init_value": 0.01
  },
  "architectures": [
    "VibeVoiceForConditionalGeneration"
  ],
  "decoder_config": {
    "attention_dropout": 0.0,
    "hidden_act": "silu",
    "hidden_size": 3584,
    "initializer_range": 0.02,
    "intermediate_size": 18944,
    "max_position_embeddings": 32768,
    "max_window_layers": 28,
    "model_type": "qwen2",
    "num_attention_heads": 28,
    "num_hidden_layers": 28,
    "num_key_value_heads": 4,
    "rms_norm_eps": 1e-06,
    "rope_scaling": null,
    "rope_theta": 1000000.0,
    "sliding_window": null,
    "torch_dtype": "bfloat16",
    "use_cache": true,
    "use_mrope": false,
    "use_sliding_window": false,
    "vocab_size": 152064
  },
  "diffusion_head_config": {
    "ddpm_batch_mul": 4,
    "ddpm_beta_schedule": "cosine",
    "ddpm_num_inference_steps": 20,
    "ddpm_num_steps": 1000,
    "diffusion_type": "ddpm",
    "head_ffn_ratio": 3.0,
    "head_layers": 4,
    "hidden_size": 3584,
    "latent_size": 64,
    "model_type": "vibevoice_diffusion_head",
    "prediction_type": "v_prediction",
    "rms_norm_eps": 1e-05,
    "speech_vae_dim": 64
  },
  "model_type": "vibevoice",
  "semantic_tokenizer_config": {
    "causal": true,
    "channels": 1,
    "conv_bias": true,
    "conv_norm": "none",
    "corpus_normalize": 0.0,
    "disable_last_norm": true,
    "encoder_depths": "3-3-3-3-3-3-8",
    "encoder_n_filters": 32,
    "encoder_ratios": [
      8,
      5,
      5,
      4,
      2,
      2
    ],
    "fix_std": 0,
    "layer_scale_init_value": 1e-06,
    "layernorm": "RMSNorm",
    "layernorm_elementwise_affine": true,
    "layernorm_eps": 1e-05,
    "mixer_layer": "depthwise_conv",
    "model_type": "vibevoice_semantic_tokenizer",
    "pad_mode": "constant",
    "std_dist_type": "none",
    "vae_dim": 128,
    "weight_init_value": 0.01
  },
  "semantic_vae_dim": 128,
  "tie_word_embeddings": false,
  "torch_dtype": "bfloat16",
  "transformers_version": "4.51.3"
}
"""

import json # noqa F401

DEFAULT_CONFIG = json.loads(_default_config_json)


__all__ = [
    "VibeVoiceAcousticTokenizerConfig",
    "VibeVoiceSemanticTokenizerConfig",
    "VibeVoiceDiffusionHeadConfig",
    "VibeVoiceConfig",
    "VibeVoiceStreamingConfig",
]
