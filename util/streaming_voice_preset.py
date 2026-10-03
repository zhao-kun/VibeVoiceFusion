"""Voice presets for the VibeVoice Realtime model, stored as safetensors.

Upstream ships presets as pickled `.pt` files holding transformers `DynamicCache` objects,
which ties loading to a transformers version and requires `weights_only=False`. They are
converted once into plain tensors and rebuilt at runtime through the stable `DynamicCache.update` API.
"""
import json
from typing import Dict, Optional

import torch
from safetensors import safe_open
from safetensors.torch import save_file
from transformers.cache_utils import DynamicCache
from transformers.modeling_outputs import BaseModelOutputWithPast

PRESET_GROUPS = ("lm", "tts_lm", "neg_lm", "neg_tts_lm")
PRESET_FORMAT = "vibevoice_streaming_voice_preset"
PRESET_VERSION = "1"


def _cache_layers(cache):
    if hasattr(cache, "key_cache") and hasattr(cache, "value_cache"):
        return list(zip(cache.key_cache, cache.value_cache))
    if hasattr(cache, "layers"):
        return [(layer.keys, layer.values) for layer in cache.layers]
    raise TypeError(f"Unsupported cache type: {type(cache)}")


def prefilled_outputs_to_tensors(prefilled_outputs: Dict) -> Dict[str, torch.Tensor]:
    tensors = {}
    for group in PRESET_GROUPS:
        output = prefilled_outputs[group]
        tensors[f"{group}.last_hidden_state"] = output["last_hidden_state"].detach().cpu().contiguous()
        for idx, (key, value) in enumerate(_cache_layers(output["past_key_values"])):
            tensors[f"{group}.layers.{idx}.key"] = key.detach().cpu().contiguous()
            tensors[f"{group}.layers.{idx}.value"] = value.detach().cpu().contiguous()
    return tensors


def convert_pt_voice_preset(pt_path: str, output_path: str):
    """Convert an upstream `.pt` voice preset to safetensors. Only use on trusted files (unpickles)."""
    prefilled_outputs = torch.load(pt_path, map_location="cpu", weights_only=False)
    tensors = prefilled_outputs_to_tensors(prefilled_outputs)
    num_layers = {
        group: sum(1 for k in tensors if k.startswith(f"{group}.layers.") and k.endswith(".key"))
        for group in PRESET_GROUPS
    }
    metadata = {"format": PRESET_FORMAT, "version": PRESET_VERSION, "num_layers": json.dumps(num_layers)}
    save_file(tensors, output_path, metadata=metadata)


def load_voice_preset(path: str, device="cpu", dtype: Optional[torch.dtype] = None) -> Dict[str, BaseModelOutputWithPast]:
    """Load a converted preset into the `all_prefilled_outputs` structure expected by generate()."""
    with safe_open(path, framework="pt", device="cpu") as f:
        metadata = f.metadata() or {}
        if metadata.get("format") != PRESET_FORMAT:
            raise ValueError(f"{path} is not a VibeVoice streaming voice preset")
        num_layers = json.loads(metadata["num_layers"])

        def _get(name):
            tensor = f.get_tensor(name)
            if dtype is not None:
                tensor = tensor.to(dtype)
            return tensor.to(device)

        prefilled_outputs = {}
        for group in PRESET_GROUPS:
            cache = DynamicCache()
            for idx in range(num_layers[group]):
                cache.update(_get(f"{group}.layers.{idx}.key"), _get(f"{group}.layers.{idx}.value"), idx)
            prefilled_outputs[group] = BaseModelOutputWithPast(
                last_hidden_state=_get(f"{group}.last_hidden_state"),
                past_key_values=cache,
            )
    return prefilled_outputs
