import json
import os
from typing import Any, Dict, List, Optional, Union

import numpy as np
import torch
from transformers.tokenization_utils_base import BatchEncoding

from util import vibevoice_root_dir
from util.logger import get_logger
from vibevoice.modular.modular_vibevoice_text_tokenizer import VibeVoiceTextTokenizerFast
from vibevoice.processor.vibevoice_tokenizer_processor import VibeVoiceTokenizerProcessor

logger = get_logger(__name__)


class VibeVoiceStreamingProcessor:
    """Text processor for the VibeVoice Realtime model; voices come from cached prompt presets."""

    def __init__(self, tokenizer=None, audio_processor=None, speech_tok_compress_ratio=3200, db_normalize=True, **kwargs):
        self.tokenizer = tokenizer
        self.audio_processor = audio_processor
        self.speech_tok_compress_ratio = speech_tok_compress_ratio
        self.db_normalize = db_normalize

    @classmethod
    def from_pretrained(cls, pretrained_model_name_or_path: Optional[str] = None, **kwargs):
        config = {}
        if pretrained_model_name_or_path is not None:
            config_path = os.path.join(pretrained_model_name_or_path, "preprocessor_config.json")
            if os.path.exists(config_path):
                with open(config_path, "r") as f:
                    config = json.load(f)

        # Qwen2.5 models share one tokenizer, so the bundled Qwen2.5-7B tokenizer serves the 0.5B model too.
        tokenizer = VibeVoiceTextTokenizerFast.from_pretrained(
            os.path.join(vibevoice_root_dir, "tokenizer"),
            local_files_only=True,
            **kwargs
        )

        audio_config = config.get("audio_processor", {})
        audio_processor = VibeVoiceTokenizerProcessor(
            sampling_rate=audio_config.get("sampling_rate", 24000),
            normalize_audio=audio_config.get("normalize_audio", True),
            target_dB_FS=audio_config.get("target_dB_FS", -25),
            eps=audio_config.get("eps", 1e-6),
        )

        return cls(
            tokenizer=tokenizer,
            audio_processor=audio_processor,
            speech_tok_compress_ratio=config.get("speech_tok_compress_ratio", 3200),
            db_normalize=config.get("db_normalize", True),
        )

    def process_input_with_cached_prompt(
        self,
        text: str,
        cached_prompt: Dict[str, Any],
        return_tensors: Optional[str] = "pt",
        **kwargs,
    ) -> BatchEncoding:
        """Tokenize one script against a cached voice prompt (single example only)."""
        script_tokens = self.tokenizer.encode(text.strip() + "\n", add_special_tokens=False)
        input_id_length = cached_prompt["lm"]["last_hidden_state"].size(1)
        tts_lm_input_id_length = cached_prompt["tts_lm"]["last_hidden_state"].size(1)

        # Placeholder ids: the prompt itself lives in the KV cache, only lengths matter.
        encoding = {
            "input_ids": [[self.tokenizer.pad_id] * input_id_length],
            "attention_mask": [[1] * input_id_length],
            "tts_lm_input_ids": [[self.tokenizer.pad_id] * tts_lm_input_id_length],
            "tts_lm_attention_mask": [[1] * tts_lm_input_id_length],
            "tts_text_ids": [script_tokens],
            "speech_input_mask": [[False] * tts_lm_input_id_length],
        }

        if return_tensors == "pt":
            encoding = {
                k: torch.tensor(v, dtype=torch.bool if k == "speech_input_mask" else torch.long)
                for k, v in encoding.items()
            }
        encoding["speech_tensors"] = None
        encoding["speech_masks"] = None
        return BatchEncoding(encoding)

    def save_audio(self, audio: Union[torch.Tensor, np.ndarray, List[Union[torch.Tensor, np.ndarray]]],
                   output_path: str = "output.wav",
                   sampling_rate: Optional[int] = None,
                   normalize: bool = False,
                   batch_prefix: str = "audio_") -> str:
        return self.audio_processor.save_audio(audio, output_path=output_path, sampling_rate=sampling_rate,
                                               normalize=normalize, batch_prefix=batch_prefix)


__all__ = [
    "VibeVoiceStreamingProcessor",
]
