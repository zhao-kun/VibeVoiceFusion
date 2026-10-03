import json
import math
import os
from typing import Any, Dict, List, Optional, Union

import numpy as np
import torch
from transformers.tokenization_utils_base import BatchEncoding

from util import vibevoice_root_dir
from util.logger import get_logger
from vibevoice.modular.modular_vibevoice_text_tokenizer import VibeVoiceASRTextTokenizerFast
from vibevoice.processor.vibevoice_tokenizer_processor import AudioNormalizer

logger = get_logger(__name__)

# Prompt strings are reproduced verbatim from upstream: the model was trained on them.
SYSTEM_PROMPT = "You are a helpful assistant that transcribes audio input into text output in JSON format."
TRANSCRIPTION_KEYS = ["Start time", "End time", "Speaker ID", "Content"]
OUTPUT_KEY_MAPPING = {
    "Start time": "start_time",
    "Start": "start_time",
    "End time": "end_time",
    "End": "end_time",
    "Speaker ID": "speaker_id",
    "Speaker": "speaker_id",
    "Content": "text",
}


class VibeVoiceASRProcessor:
    """Builds the VibeVoice-ASR prompt from audio (batch size 1) and parses the JSON transcript."""

    def __init__(self, tokenizer=None, speech_tok_compress_ratio=3200, target_sample_rate=24000,
                 normalize_audio=True, target_dB_FS=-25, eps=1e-6, **kwargs):
        self.tokenizer = tokenizer
        self.speech_tok_compress_ratio = speech_tok_compress_ratio
        self.target_sample_rate = target_sample_rate
        self.normalize_audio = normalize_audio
        self.audio_normalizer = AudioNormalizer(target_dB_FS=target_dB_FS, eps=eps) if normalize_audio else None

        self.speech_start_id = tokenizer.speech_start_id
        self.speech_end_id = tokenizer.speech_end_id
        self.speech_pad_id = tokenizer.speech_pad_id
        self.pad_id = tokenizer.pad_id

    @classmethod
    def from_pretrained(cls, pretrained_model_name_or_path: Optional[str] = None, **kwargs):
        config = {}
        if pretrained_model_name_or_path is not None:
            config_path = os.path.join(pretrained_model_name_or_path, "preprocessor_config.json")
            if os.path.exists(config_path):
                with open(config_path, "r") as f:
                    config = json.load(f)

        # Qwen2.5 models share one tokenizer, so the bundled Qwen2.5-7B tokenizer is used offline.
        tokenizer = VibeVoiceASRTextTokenizerFast.from_pretrained(
            os.path.join(vibevoice_root_dir, "tokenizer"),
            local_files_only=True,
            **kwargs
        )

        return cls(
            tokenizer=tokenizer,
            speech_tok_compress_ratio=config.get("speech_tok_compress_ratio", 3200),
            target_sample_rate=config.get("target_sample_rate", 24000),
            normalize_audio=config.get("normalize_audio", True),
            target_dB_FS=config.get("target_dB_FS", -25),
            eps=config.get("eps", 1e-6),
        )

    def load_audio(self, audio: Union[str, np.ndarray, torch.Tensor]) -> np.ndarray:
        """Return mono float32 audio at the target sample rate (arrays are assumed to be at that rate)."""
        if isinstance(audio, str):
            import librosa
            audio_array, _ = librosa.load(audio, sr=self.target_sample_rate, mono=True)
        elif isinstance(audio, torch.Tensor):
            audio_array = audio.detach().cpu().float().numpy()
        else:
            audio_array = np.asarray(audio, dtype=np.float32)
        if audio_array.ndim > 1:
            audio_array = audio_array.squeeze()
        return audio_array.astype(np.float32)

    def build_prompt_ids(self, num_speech_tokens: int, audio_duration: float,
                         context_info: Optional[str] = None) -> List[int]:
        """System turn + user turn with speech placeholders.

        Upstream never appends the assistant header (its add_generation_prompt flag is unused), so neither do we.
        """
        system_text = self.tokenizer.apply_chat_template([{"role": "system", "content": SYSTEM_PROMPT}], tokenize=False)
        system_tokens = self.tokenizer.encode(system_text)

        if context_info and context_info.strip():
            user_suffix = (f"This is a {audio_duration:.2f} seconds audio, with extra info: {context_info.strip()}"
                           f"\n\nPlease transcribe it with these keys: " + ", ".join(TRANSCRIPTION_KEYS))
        else:
            user_suffix = (f"This is a {audio_duration:.2f} seconds audio, please transcribe it with these keys: "
                           + ", ".join(TRANSCRIPTION_KEYS))

        sp_start = self.tokenizer.convert_ids_to_tokens(self.speech_start_id)
        sp_pad = self.tokenizer.convert_ids_to_tokens(self.speech_pad_id)
        sp_end = self.tokenizer.convert_ids_to_tokens(self.speech_end_id)
        user_text = sp_start + sp_pad * num_speech_tokens + sp_end + "\n" + user_suffix

        user_tokens = self.tokenizer.apply_chat_template([{"role": "user", "content": user_text}], tokenize=True)
        return system_tokens + user_tokens

    def __call__(self, audio: Union[str, np.ndarray, torch.Tensor], context_info: Optional[str] = None,
                 return_tensors: Optional[str] = "pt", **kwargs) -> BatchEncoding:
        """Process one audio input into model inputs."""
        audio_array = self.load_audio(audio)
        if self.audio_normalizer is not None:
            audio_array = self.audio_normalizer(audio_array)

        audio_duration = len(audio_array) / self.target_sample_rate
        num_speech_tokens = math.ceil(len(audio_array) / self.speech_tok_compress_ratio)
        input_ids = self.build_prompt_ids(num_speech_tokens, audio_duration, context_info)
        acoustic_input_mask = [token == self.speech_pad_id for token in input_ids]

        encoding = {
            "input_ids": [input_ids],
            "attention_mask": [[1] * len(input_ids)],
            "acoustic_input_mask": [acoustic_input_mask],
            "speech_tensors": audio_array[None, :],
            "speech_masks": np.ones((1, num_speech_tokens), dtype=bool),
        }
        if return_tensors == "pt":
            encoding = {
                "input_ids": torch.tensor(encoding["input_ids"], dtype=torch.long),
                "attention_mask": torch.tensor(encoding["attention_mask"], dtype=torch.long),
                "acoustic_input_mask": torch.tensor(encoding["acoustic_input_mask"], dtype=torch.bool),
                "speech_tensors": torch.from_numpy(encoding["speech_tensors"]),
                "speech_masks": torch.from_numpy(encoding["speech_masks"]),
            }
        encoding = BatchEncoding(encoding)
        encoding["audio_duration"] = audio_duration
        return encoding

    def decode(self, *args, **kwargs):
        return self.tokenizer.decode(*args, **kwargs)

    def batch_decode(self, *args, **kwargs):
        return self.tokenizer.batch_decode(*args, **kwargs)

    def post_process_transcription(self, text: str) -> List[Dict[str, Any]]:
        """Extract transcript segments from generated text (port of upstream parsing)."""
        try:
            if "```json" in text:
                json_start = text.find("```json") + 7
                json_end = text.find("```", json_start)
                json_str = text[json_start:json_end].strip()
            else:
                json_start = text.find("[")
                if json_start == -1:
                    json_start = text.find("{")
                if json_start != -1:
                    bracket_count = 0
                    json_end = json_start
                    for i in range(json_start, len(text)):
                        if text[i] in "[{":
                            bracket_count += 1
                        elif text[i] in "]}":
                            bracket_count -= 1
                            if bracket_count == 0:
                                json_end = i + 1
                                break
                    json_str = text[json_start:json_end]
                else:
                    json_str = text

            result = json.loads(json_str)
            if isinstance(result, dict):
                result = [result]

            cleaned_result = []
            for item in result:
                if isinstance(item, dict):
                    cleaned_item = {mapped: item[key] for key, mapped in OUTPUT_KEY_MAPPING.items() if key in item}
                    if cleaned_item:
                        cleaned_result.append(cleaned_item)
            return cleaned_result
        except json.JSONDecodeError as e:
            logger.warning(f"Failed to parse JSON from transcription: {e}")
            return []
        except Exception as e:
            logger.warning(f"Error post-processing transcription: {e}")
            return []


__all__ = ["VibeVoiceASRProcessor"]
