"""
Process-wide model cache that keeps the last used model resident on the GPU between tasks.

Only one model is kept because the TTS and ASR models do not fit on a consumer GPU together,
and the task manager runs a single task at a time.
"""
import dataclasses
import gc
import threading
import time
from typing import Any, Callable, Dict, Hashable, Optional

import torch

from util.logger import get_logger

logger = get_logger(__name__)


def offload_cache_key(offload_config: Any) -> Optional[tuple]:
    """Hashable form of an OffloadConfig (or the raw offloading dict the fake engines keep)"""
    if not offload_config:
        return None
    if isinstance(offload_config, dict):
        return tuple(sorted(offload_config.items())) if offload_config.get('enabled') else None
    return dataclasses.astuple(offload_config) if offload_config.enabled else None


def tts_model_cache_key(kind: str, model_dir: Optional[str], dtype: torch.dtype, attn_implementation: str,
                        offload_config: Any, lora_model_path: Optional[str] = None,
                        lora_weight: Optional[float] = None) -> tuple:
    """Key shared by project and quick generation, so both reuse the same resident TTS model"""
    # LoRA weights are merged into the base weights at load time, so each LoRA/weight pair is its own model
    lora = (lora_model_path, lora_weight) if lora_model_path else None
    return (kind, model_dir, str(dtype), attn_implementation, offload_cache_key(offload_config), lora)


def _unload(model: Any) -> None:
    offloader = getattr(model, 'offloader', None)
    if offloader is not None:
        try:
            offloader.cleanup()
            model.offloader = None
        except Exception as e:
            logger.warning(f"Failed to cleanup offloader: {e}")
    if isinstance(model, torch.nn.Module):
        try:
            model.to('cpu')
        except Exception:
            pass  # Model might already be partially on CPU


class ModelCache:
    """Single-slot model cache plus a cache of CPU-side processors"""

    def __init__(self, idle_seconds: float = 0):
        self.idle_seconds = idle_seconds
        self._lock = threading.RLock()
        self._key: Optional[Hashable] = None
        self._model: Any = None
        self._last_used = 0.0
        self._processors: Dict[Hashable, Any] = {}

    def get_model(self, key: Hashable, loader: Callable[[], Any]) -> Any:
        """Return the cached model for key, evicting and loading when the key differs"""
        with self._lock:
            if self._model is not None and self._key == key:
                logger.info(f"Reusing cached model {key[0]}")
                self._last_used = time.time()
                offloader = getattr(self._model, 'offloader', None)
                if offloader is not None:
                    offloader.reset_stats()
                return self._model

            self.evict(reason=f"loading {key[0]}")
            model = loader()
            self._key, self._model = key, model
            self._last_used = time.time()
            return model

    def get_processor(self, key: Hashable, loader: Callable[[], Any]) -> Any:
        with self._lock:
            if key not in self._processors:
                self._processors[key] = loader()
            return self._processors[key]

    def touch(self) -> None:
        with self._lock:
            if self._model is not None:
                self._last_used = time.time()

    def has_model(self) -> bool:
        return self._model is not None

    def evict(self, reason: str = "") -> None:
        """Unload the cached model and return its VRAM to the driver"""
        with self._lock:
            if self._model is None:
                return
            key, model = self._key, self._model
            self._key, self._model = None, None
            logger.info(f"Unloading cached model {key[0]} ({reason})")
            _unload(model)
            del model
            gc.collect()
            if torch.cuda.is_available():
                torch.cuda.synchronize()
                torch.cuda.empty_cache()

    def evict_if_idle(self) -> None:
        with self._lock:
            if self.idle_seconds > 0 and self._model is not None \
                    and time.time() - self._last_used >= self.idle_seconds:
                self.evict(reason=f"idle for {self.idle_seconds:g}s")


model_cache = ModelCache()
