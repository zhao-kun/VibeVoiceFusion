# Model Residency & Task Manager Design

## Problem Statement

Every GPU task (OpenAI-compatible TTS/ASR, Quick Generate, project generation, transcription, live transcription) loaded the full model from disk at the start of the task and unloaded it at the end:

```text
request ─▶ task starts ─▶ from_pretrain() disk → GPU ─▶ generate ─▶ finalize(): offloader.cleanup()
                                                                               model.to('cpu'), del
                                                                               gc.collect(), empty_cache()
```

Two back-to-back `/v1/audio/speech` calls with identical settings therefore paid the full model load twice. The OpenAI-compatible API is meant for repeated programmatic calls, so this load dominated short requests.

An investigation of the same code path found several smaller costs and bugs:

| # | Issue | Where |
|---|-------|-------|
| 1 | Model loaded and unloaded on every task | All four inference engines |
| 2 | Processor rebuilt on every task | Same engines |
| 3 | Task loop slept 0.5 s per iteration before picking up a new task | `Manager.task_run_loop` |
| 4 | OpenAI service polled task status with 0.5 s sleeps | `OpenAICompatService` |
| 5 | Preset voice WAV copied to a new UUID file on every `/v1/audio/speech` call | `OpenAICompatService.generate_speech` |
| 6 | Offloader timing/profiling lists grow without bound (harmless when the model was discarded per task, a leak once it is kept) | `LayerOffloader` |
| 7 | `except torch.cuda.OutOfMemoryError` came after `except Exception`, so OOM was never reported as OOM | `Manager.task_run_loop` |
| 8 | Exceptions from `_task_finalize()` or `task_failure()` escaped and killed the task-loop thread | `Manager.task_run_loop` |
| 9 | `add_task` check-then-set was not atomic; two concurrent requests could both be accepted and the first silently dropped | `Manager.add_task` |

Items 3 and 4 add up to about 1 s per request. Items 1 and 2 were not measured; the load time depends on disk speed, dtype and GPU.

## Design Decisions

| Question | Decision |
|----------|----------|
| How many models stay resident | **One** process-wide slot shared by every engine. TTS and ASR do not fit on a consumer GPU together, and the task manager runs one task at a time anyway |
| When to reload | Only when the **cache key** changes (model type, dtype, attention implementation, offload config, LoRA path + weight) |
| When to unload | Key change, training start, any task failure, optional idle timeout |
| Idle timeout | `MODEL_CACHE_IDLE_SECONDS`, default `0` = never unload for idleness |
| Processors | Cached separately (CPU only, cheap to keep) |
| Waiting for completion | `threading.Event` / `threading.Condition` instead of sleeps |
| Preset voices | One stable copy per preset (`preset-<name>.wav`) instead of a new UUID copy per request |
| Concurrency | Still one task at a time; concurrent requests still get `503` |

## Architecture

```text
                      ┌──────────────────────────────────────────────┐
 HTTP / WebSocket ───▶│ services (openai_compat, quick_generate,     │
                      │ voice_generation, transcription, training)   │
                      └───────────────┬──────────────────────────────┘
                                      │ gm.add_task(task)   (atomic)
                                      ▼
                      ┌──────────────────────────────────────────────┐
                      │ Manager (task_manager/task.py)               │
                      │  task_run_loop ── _wakeup Event ─────────────┼─▶ wakes on add_task
                      │  _run_task ────── _task_done Condition ──────┼─▶ wakes wait_for_task
                      └───────────────┬──────────────────────────────┘
                                      │ task.run()
                                      ▼
                      ┌──────────────────────────────────────────────┐
                      │ Inference engines                            │
                      │  model_cache.get_model(key, loader)          │
                      │  model_cache.get_processor(key, loader)      │
                      │  finalize(): self.model = None (no unload)   │
                      └───────────────┬──────────────────────────────┘
                                      ▼
                      ┌──────────────────────────────────────────────┐
                      │ ModelCache (inference/model_cache.py)        │
                      │  one (key, model) slot + processor dict      │
                      │  evict() / evict_if_idle() / touch()         │
                      └──────────────────────────────────────────────┘
```

### ModelCache (`backend/inference/model_cache.py`)

A module-level singleton, `model_cache`, guarded by an `RLock`.

| Method | Behaviour |
|--------|-----------|
| `get_model(key, loader)` | Same key as the resident model → return it, refresh `_last_used`, call `offloader.reset_stats()`. Different key → `evict()` the old model, call `loader()`, store the result |
| `get_processor(key, loader)` | Dict cache; processors are never evicted |
| `touch()` | Refresh `_last_used` (called when a task finishes, so idle time counts from the end of the last task) |
| `evict(reason)` | Clear the slot, `offloader.cleanup()`, `model.to('cpu')`, `gc.collect()`, `cuda.synchronize()` + `empty_cache()`. This is the cleanup the engines used to do in `finalize()` |
| `evict_if_idle()` | Evict when `idle_seconds > 0` and the model has been idle at least that long |
| `has_model()` | Whether a model is resident |

The old slot is evicted **before** the new model loads, so two models never coexist in VRAM.

### Cache keys

The key must include everything that changes the weights or layout of the loaded model:

| Engine | Key |
|--------|-----|
| Project generation (`InferenceEngine`) | `('vibevoice-tts', model_file, dtype, attn_implementation, offload, (lora_path, lora_weight) or None)` |
| Quick generate / OpenAI TTS (`QuickGenerateInferenceEngine`) | `('vibevoice-tts', model_file, dtype, attn_implementation, offload, None)` |
| ASR + live ASR (`ASRInferenceEngine`) | `('vibevoice-asr', model_dir, dtype, offload)` |
| Fake engines | Same shape with kind `fake-vibevoice-tts` / `fake-vibevoice-asr` |

- `tts_model_cache_key()` builds both TTS keys. Project generation without LoRA and Quick Generate therefore produce **the same key** and reuse the same resident model.
- LoRA weights are merged into the base weights by `merge_lora_weights` at load time, so each `(path, weight)` pair is a different model.
- `offload_cache_key()` turns an `OffloadConfig` dataclass (or the raw dict the fake engines keep) into a hashable tuple. Disabled offloading maps to `None`, so `enabled=False` configs with different leftover fields still match.
- Per-request generation settings (seed, cfg_scale, batch size, text, voice prompt) are passed to `generate()` on every call and are **not** part of the key.

### Engine changes

All four engines follow the same pattern:

```python
model = model_cache.get_model(self._model_cache_key(load_dtype),
                              lambda: self._load_model(dtype=load_dtype))
self.model = model
processor = model_cache.get_processor(('vibevoice-tts',), lambda: VibeVoiceProcessor.from_pretrained(None))
...
def finalize(self):
    self.model = None          # the cache owns the model now
```

- `_load_model()` no longer assigns `self.model`; it only builds and returns the model.
- Each engine implements the new abstract `_model_cache_key(dtype)`.
- ASR and live ASR share `ASRInferenceBase._acquire_model()`, which still calls `visitor.visit_model_loaded(...)`. On a cache hit the recorded load time is close to zero.
- `self.model` is still set during the task, so `_collect_offloading_metrics()` keeps working.

### Offloader stats (`vibevoice/modular/custom_offloading_utils.py`)

New `LayerOffloader.reset_stats()` clears `transfer_times`, `forward_times`, `total_transfers`, `profile_data`, `token_count` and the lazily created `layer_compute_times` / `forward_start_times`. `get_model()` calls it on every cache hit, so the offloading metrics stored with each generation cover that generation only, and the lists no longer grow across requests.

## Eviction Rules

| Trigger | Where | Reason |
|---------|-------|--------|
| Request needs a different key | `ModelCache.get_model` | Only one model fits |
| Training job starts | `TrainingTask.run()` | Training runs in-process and needs the whole GPU |
| Task raises any exception | `Manager._run_task` | A task that died mid-forward can leave offloaded layers on the wrong device; the next task starts from a clean load |
| CUDA OOM | `Manager._run_task` | Free memory before reporting |
| Idle for `MODEL_CACHE_IDLE_SECONDS` | `Manager.task_run_loop` (checked at most every `TASK_IDLE_WAIT_SECONDS` = 1 s while idle) | Return VRAM to other processes |

## Task Manager (`backend/task_manager/task.py`)

### Event-driven loop

```python
def task_run_loop(self):
    while True:
        self._wakeup.clear()          # cleared *before* reading self.task
        task = self.task
        if task is None:
            self._evict_model(idle=True)
            self._wakeup.wait(TASK_IDLE_WAIT_SECONDS)
            continue
        self._run_task(task)
```

- `add_task()` sets `_wakeup`, so a new task starts immediately instead of after up to 0.5 s.
- Clearing the event before reading `self.task` avoids a lost wakeup: a task added after the check sets the event again and `wait()` returns at once.
- The 1 s timeout only bounds how often the idle timeout is checked.

### Task execution

```python
def _run_task(self, task):
    try:
        task.run()
        task.task_success(...)
    except torch.cuda.OutOfMemoryError:   # before Exception: OOM subclasses RuntimeError
        evict; report FAILURE_TYPE_OOM
    except Exception:
        evict; report FAILURE_TYPE_GENERAL
    finally:
        task.task_finalize()              # existing safe wrapper, logs and swallows errors
        model_cache.touch()
        with self._task_done:
            self.task = None
            self._task_done.notify_all()
```

- The loop works on the local `task` it read, not on `self.task`, so it only ever clears the task it actually ran.
- `_report_failure()` wraps `task.task_failure()`, so an error while recording a failure is logged instead of killing the loop.

### Waiting for a task

```python
def wait_for_task(self, task_id, timeout) -> bool:
    """Block until task_id is no longer running; returns False on timeout."""
```

It returns `True` as soon as the slot no longer holds `task_id`, which includes a task that was never running.

### Atomic `add_task`

```python
with self._task_done:
    if self.task is not None:
        return False
    task.task_appended(...)
    self.task = task
self._wakeup.set()
```

- Flask serves requests on multiple threads, so the check and the set must happen under one lock.
- `task_appended()` runs only for the accepted task. For project generation this is where the history entry is written; in the old race the overwritten task had already written its entry and then never ran, so it stayed `pending`.
- `task_appended()` runs before `self.task` is assigned, so the loop can never start a task before it has been recorded.

## OpenAI-Compatible Service (`backend/services/openai_compat_service.py`)

### Completion wait

Both `generate_speech` and `transcribe_audio` replace `time.sleep(POLL_INTERVAL)` with:

```python
finished = gm.wait_for_task(request_id, POLL_INTERVAL)
...status checks (completed → return, failed → 500)...
if finished:
    return None, "... task ended without a result.", 500
```

- The response is sent as soon as the task finishes instead of on the next 0.5 s tick.
- If the task has left the slot but its record is neither completed nor failed, the request returns `500` immediately. Before, it would spin until the 300 s / 1800 s timeout.

### Preset voice reuse

`QuickGenerateService.save_preset_voice_file(preset_path)` replaces the per-request `read_bytes()` + `save_voice_file()` (new UUID file each time):

- Target: `workspace/_quick-generate/voices/preset-<preset filename>`.
- Copied only when missing or older than the preset (`st_mtime`).
- Written atomically: copy to `<name>.<uuid>.tmp`, then `os.replace`.
- Deleting a Quick Generate history entry removes a voice file only when no other entry references it; if the shared copy is removed, the next request recreates it.

## Configuration

| Variable | Default | Description |
|----------|---------|-------------|
| `MODEL_CACHE_IDLE_SECONDS` | `0` | Unload the resident model after it has been idle this many seconds. `0` keeps it loaded until another model is needed |

The variable is read in `backend/config.py` (`Config.MODEL_CACHE_IDLE_SECONDS`) and applied in `create_app()` (`model_cache.idle_seconds = ...`).

## Behaviour Changes

| Area | Before | After |
|------|--------|-------|
| VRAM while idle | Freed after every task | Held by the last model until a different model, training, a failure or the idle timeout |
| Repeated requests with the same settings | Full model load each time | Model reused, log shows `Reusing cached model vibevoice-tts` |
| Switching TTS ↔ ASR, dtype, offload preset or LoRA | Load | Unload previous + load (same cost as before) |
| Task pickup latency | Up to 0.5 s | Immediate |
| OpenAI response after completion | Up to 0.5 s | Immediate |
| CUDA OOM failure | Reported as a general failure with the raw CUDA message | Reported as `out_of_memory` with "CUDA Out of Memory Error"; the existing OOM banners (`GenerationHistory`, `CurrentGeneration`, `TrainingHistory`) and training `is_oom_failure` now show up |
| Exception in finalize / failure reporting | Task-loop thread died; every later request got `503` until restart | Logged; the loop continues |
| Two simultaneous requests | Both could be accepted; the first could be lost in `pending` | Exactly one accepted, the other gets `503` |
| Project generation lost in that race | History entry written, then stuck in `pending` forever | Cannot happen; only the accepted task writes an entry |
| Preset voice files | One new UUID copy per `/v1/audio/speech` call | One `preset-<name>.wav` per preset |

No HTTP endpoint, request or response format changed.

## Files Changed

| File | Change |
|------|--------|
| `backend/inference/model_cache.py` (new) | `ModelCache`, `model_cache` singleton, `offload_cache_key()`, `tts_model_cache_key()` |
| `backend/inference/inference.py` | Model/processor from cache, `_model_cache_key()` incl. LoRA, `finalize()` only drops the reference |
| `backend/inference/quick_generate_inference.py` | Same, without LoRA |
| `backend/inference/asr_inference.py` | Same, plus `_acquire_model()`; processor cache keyed by model dir |
| `backend/inference/streaming_asr_inference.py` | Uses `_acquire_model()` |
| `vibevoice/modular/custom_offloading_utils.py` | `LayerOffloader.reset_stats()` |
| `backend/task_manager/task.py` | Event-driven loop, `_run_task`, `_report_failure`, `wait_for_task`, atomic `add_task`, OOM branch order, safe finalize |
| `backend/task_manager/training_task.py` | Evict before training |
| `backend/config.py`, `backend/app.py` | `MODEL_CACHE_IDLE_SECONDS` |
| `backend/services/openai_compat_service.py` | `wait_for_task`, preset reuse |
| `backend/services/quick_generate_service.py` | `save_preset_voice_file()` |
| `docs/openai-compatible-api.md` | Env var + Model Residency section |
| `tests/test_model_cache.py` (new) | Tests listed below |

## Testing

`tests/test_model_cache.py` uses fake models and fake tasks only; no real model is loaded.

| Area | Tests |
|------|-------|
| Cache | Same key reuses the model and resets stats; new key evicts and cleans up the old model; idle eviction; `idle_seconds=0` never evicts; processor cache; key equality for offload configs, LoRA and dtype |
| Offloader | `reset_stats()` clears counters and lazily created attributes |
| Manager | `wait_for_task` wakes on completion, times out, returns at once for unknown IDs; failure evicts, success keeps the model; idle timeout evicts between tasks; OOM reports `out_of_memory`; general errors report `general`; errors in finalize / failure hooks do not stop the loop; 16 concurrent `add_task` calls accept exactly one |
| Training | The cached model is gone before `train()` runs |
| OpenAI API | Three `/v1/audio/transcriptions` calls load the model once; switching to `vibevoice-asr-fp8` loads again |
| Preset voice | Copied once, refreshed when the preset is newer, one file per preset |

```bash
PYTHONPATH=.:backend pytest tests
# 142 passed, 1 skipped
```

### GPU validation (pending)

Not yet run on real hardware:

1. Two identical `/v1/audio/speech` requests: the second should log `Reusing cached model vibevoice-tts` with no load step, and return faster.
2. The same with offloading enabled (Balanced/Aggressive/Extreme): output quality and speed of the second request should match the first.
3. Alternate TTS and ASR requests: each switch should log `Unloading cached model ...` and `nvidia-smi` should never show both models at once.
4. Start training while a model is resident: training should start with the cache empty.
5. Set `MODEL_CACHE_IDLE_SECONDS=30` and wait: the model should unload about 30 s after the last task.
6. Force an OOM (e.g. large batch without offloading): the UI should show the OOM banner.

## Open Questions / Limitations

- **Offloader state on reuse (unverified).** The tests only cover reuse with fakes. It is not yet confirmed on a GPU that the offloader's hooks leave every layer on its expected device after a completed run, so that the next run starts from the same state as a fresh load. Item 2 above checks this.
- **Hidden per-run model state (unverified).** No per-request setting is applied to the model at load time (checked: `_load_model` only uses dtype, attention implementation, offload config and LoRA). It has not been verified on a GPU whether `generate()` leaves any internal state (e.g. caches) on the model between calls.
- **VRAM is held while idle** by design. Set `MODEL_CACHE_IDLE_SECONDS` when other processes share the GPU.
- **One task at a time** is unchanged; concurrent requests still receive `503`.
