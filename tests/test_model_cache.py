"""Model residency tests: models stay loaded across tasks and are unloaded only when they must be."""
import io
import os
import threading
import time

import pytest
import torch

from backend.app import create_app
from backend.inference.asr_inference import FakeASRInferenceEngine, FakeASRModel
from backend.inference.model_cache import ModelCache, model_cache, offload_cache_key, tts_model_cache_key
from backend.services.quick_generate_service import QuickGenerateService
from backend.task_manager.task import Manager, Task, gm
from backend.task_manager.training_task import TrainingTask
from vibevoice.modular.custom_offloading_utils import LayerOffloader, OffloadConfig


class _FakeOffloader:
    def __init__(self):
        self.resets = 0
        self.cleaned = False

    def reset_stats(self):
        self.resets += 1

    def cleanup(self):
        self.cleaned = True


class _FakeModel:
    def __init__(self):
        self.offloader = _FakeOffloader()


class _CountingLoader:
    def __init__(self):
        self.calls = 0
        self.models = []

    def __call__(self):
        self.calls += 1
        self.models.append(_FakeModel())
        return self.models[-1]


class _ScriptedTask(Task):
    def __init__(self, task_id, duration=0.0, error=None, failure_error=None, finalize_error=None):
        super().__init__(task_id=task_id)
        self.duration = duration
        self.error = error
        self.failure_error = failure_error
        self.finalize_error = finalize_error
        self.failed = None
        self.failure_type = None
        self.appended = 0

    def run(self):
        time.sleep(self.duration)
        if self.error:
            raise self.error

    def task_failure(self, error_msg, failure_type='general'):
        self.failed = error_msg
        self.failure_type = failure_type
        if self.failure_error:
            raise self.failure_error

    def task_success(self, message):
        pass

    def task_appended(self, message):
        self.appended += 1

    def unwrap(self):
        return None

    def _task_finalize(self):
        if self.finalize_error:
            raise self.finalize_error


@pytest.fixture(autouse=True)
def clean_cache():
    model_cache.evict(reason="test setup")
    model_cache.idle_seconds = 0
    yield
    model_cache.evict(reason="test teardown")
    model_cache.idle_seconds = 0


def _manager():
    manager = Manager()
    threading.Thread(target=manager.task_run_loop, daemon=True).start()
    return manager


def _wait_until(predicate, timeout=5):
    deadline = time.time() + timeout
    while not predicate() and time.time() < deadline:
        time.sleep(0.01)
    return predicate()


class TestModelCache:
    def test_same_key_reuses_model(self):
        cache, loader = ModelCache(), _CountingLoader()
        first = cache.get_model(('tts', 'bf16'), loader)
        second = cache.get_model(('tts', 'bf16'), loader)
        assert first is second
        assert loader.calls == 1
        assert first.offloader.resets == 1

    def test_new_key_evicts_previous_model(self):
        cache, loader = ModelCache(), _CountingLoader()
        tts = cache.get_model(('tts', 'bf16'), loader)
        tts_offloader = tts.offloader
        asr = cache.get_model(('asr', 'bf16'), loader)
        assert asr is not tts
        assert loader.calls == 2
        assert tts_offloader.cleaned and tts.offloader is None
        assert not asr.offloader.cleaned

    def test_idle_eviction(self):
        cache, loader = ModelCache(idle_seconds=0.05), _CountingLoader()
        offloader = cache.get_model(('tts',), loader).offloader
        cache.evict_if_idle()
        assert cache.has_model()
        time.sleep(0.06)
        cache.evict_if_idle()
        assert not cache.has_model()
        assert offloader.cleaned

    def test_zero_idle_seconds_keeps_model(self):
        cache = ModelCache(idle_seconds=0)
        cache.get_model(('tts',), _CountingLoader())
        cache._last_used -= 3600
        cache.evict_if_idle()
        assert cache.has_model()

    def test_processor_cached(self):
        cache, loader = ModelCache(), _CountingLoader()
        assert cache.get_processor(('tts',), loader) is cache.get_processor(('tts',), loader)
        assert loader.calls == 1

    def test_keys(self):
        balanced = OffloadConfig(enabled=True, num_layers_on_gpu=12)
        assert offload_cache_key(None) is None
        assert offload_cache_key(OffloadConfig(enabled=False)) is None
        assert offload_cache_key(balanced) == offload_cache_key(OffloadConfig(enabled=True, num_layers_on_gpu=12))
        assert offload_cache_key(balanced) != offload_cache_key(OffloadConfig(enabled=True, num_layers_on_gpu=8))
        assert offload_cache_key({'enabled': False, 'mode': 'preset'}) is None

        base = tts_model_cache_key('vibevoice-tts', 'm', torch.bfloat16, 'sdpa', None)
        assert base == tts_model_cache_key('vibevoice-tts', 'm', torch.bfloat16, 'sdpa', None, None, 1.0)
        assert base != tts_model_cache_key('vibevoice-tts', 'm', torch.bfloat16, 'sdpa', None, 'lora', 1.0)
        assert tts_model_cache_key('vibevoice-tts', 'm', torch.bfloat16, 'sdpa', None, 'lora', 0.5) != \
            tts_model_cache_key('vibevoice-tts', 'm', torch.bfloat16, 'sdpa', None, 'lora', 1.0)
        assert base != tts_model_cache_key('vibevoice-tts', 'm', torch.float8_e4m3fn, 'sdpa', None)


class TestLayerOffloaderStats:
    def test_reset_stats(self):
        offloader = LayerOffloader(torch.nn.Module(), OffloadConfig(enabled=False), torch.device('cpu'))
        offloader.transfer_times.append(0.1)
        offloader.total_transfers = 3
        offloader.profile_data['layer_0_pre_transfer'].append(1.0)
        offloader.layer_compute_times = {0: [1.0]}

        offloader.reset_stats()

        stats = offloader.get_stats()
        assert stats['total_transfers'] == 0
        assert stats['total_transfer_time_ms'] == 0
        assert stats['total_compute_time_ms'] == 0
        assert not hasattr(offloader, 'layer_compute_times')


class TestManager:
    def test_wait_for_task_wakes_on_completion(self):
        manager = _manager()
        assert manager.add_task(_ScriptedTask('t1', duration=0.2))
        start = time.time()
        assert manager.wait_for_task('t1', timeout=5)
        assert time.time() - start < 1.0
        assert not manager.has_task()

    def test_wait_for_task_times_out(self):
        manager = _manager()
        assert manager.add_task(_ScriptedTask('t2', duration=0.5))
        assert not manager.wait_for_task('t2', timeout=0.05)
        assert manager.wait_for_task('t2', timeout=5)

    def test_wait_for_unknown_task_returns_immediately(self):
        assert _manager().wait_for_task('nope', timeout=5)

    def test_task_failure_evicts_cached_model(self):
        offloader = model_cache.get_model(('tts',), _CountingLoader()).offloader
        manager = _manager()
        task = _ScriptedTask('t3', error=RuntimeError('boom'))
        assert manager.add_task(task)
        assert manager.wait_for_task('t3', timeout=5)
        assert task.failed == 'boom'
        assert not model_cache.has_model()
        assert offloader.cleaned

    def test_oom_reports_oom_failure_type(self):
        offloader = model_cache.get_model(('tts',), _CountingLoader()).offloader
        manager = _manager()
        task = _ScriptedTask('oom', error=torch.cuda.OutOfMemoryError('CUDA out of memory'))
        assert manager.add_task(task)
        assert manager.wait_for_task('oom', timeout=5)
        assert task.failure_type == 'out_of_memory'
        assert task.failed == 'CUDA Out of Memory Error'
        assert offloader.cleaned

    def test_general_failure_type(self):
        manager = _manager()
        task = _ScriptedTask('gen', error=ValueError('bad'))
        assert manager.add_task(task)
        assert manager.wait_for_task('gen', timeout=5)
        assert task.failure_type == 'general'

    @pytest.mark.parametrize('kwargs', [
        {'finalize_error': RuntimeError('finalize broke')},
        {'error': RuntimeError('boom'), 'failure_error': RuntimeError('failure hook broke')},
    ])
    def test_hook_errors_do_not_kill_the_loop(self, kwargs):
        manager = _manager()
        assert manager.add_task(_ScriptedTask('broken', **kwargs))
        assert manager.wait_for_task('broken', timeout=5)
        follow_up = _ScriptedTask('next')
        assert manager.add_task(follow_up)
        assert manager.wait_for_task('next', timeout=5)
        assert follow_up.failed is None

    def test_concurrent_add_task_accepts_one(self):
        manager = Manager()
        tasks = [_ScriptedTask(f'c{i}') for i in range(16)]
        barrier = threading.Barrier(len(tasks))
        results = []

        def add(task):
            barrier.wait()
            results.append(manager.add_task(task))

        threads = [threading.Thread(target=add, args=(t,)) for t in tasks]
        for t in threads:
            t.start()
        for t in threads:
            t.join()
        assert results.count(True) == 1
        assert sum(t.appended for t in tasks) == 1
        assert manager.get_current_task().appended == 1

    def test_success_keeps_cached_model(self):
        model_cache.get_model(('tts',), _CountingLoader())
        manager = _manager()
        assert manager.add_task(_ScriptedTask('t4'))
        assert manager.wait_for_task('t4', timeout=5)
        assert model_cache.has_model()

    def test_idle_timeout_evicts_between_tasks(self):
        model_cache.idle_seconds = 0.1
        model_cache.get_model(('tts',), _CountingLoader())
        _manager()
        assert _wait_until(lambda: not model_cache.has_model(), timeout=5)

    def test_training_evicts_cached_model(self):
        model_cache.get_model(('tts',), _CountingLoader())
        seen = {}

        class _Engine:
            def train(self):
                seen['model_loaded'] = model_cache.has_model()

        TrainingTask('train', _Engine()).run()
        assert seen['model_loaded'] is False


@pytest.fixture
def client(tmp_path, monkeypatch):
    monkeypatch.setattr(FakeASRModel, 'step_delay', 0.0)
    monkeypatch.delenv('OPENAI_COMPAT_API_KEY', raising=False)
    app = create_app('testing')
    app.config['WORKSPACE_DIR'] = tmp_path
    app.config['FAKE_MODEL'] = True
    assert _wait_until(lambda: not gm.has_task())
    with app.test_client() as c:
        yield c
    assert _wait_until(lambda: not gm.has_task())


class TestOpenAIResidency:
    def test_transcriptions_load_model_once(self, client, monkeypatch):
        loads = []
        original = FakeASRInferenceEngine._load_model

        def counting_load(self, dtype):
            loads.append(dtype)
            return original(self, dtype)

        monkeypatch.setattr(FakeASRInferenceEngine, '_load_model', counting_load)

        for _ in range(3):
            resp = client.post('/v1/audio/transcriptions', data={
                'file': (io.BytesIO(b'RIFF0000WAVEfake'), 'a.wav'),
                'model': 'vibevoice-asr',
                'response_format': 'json',
            }, content_type='multipart/form-data')
            assert resp.status_code == 200, resp.get_json()
            assert _wait_until(lambda: not gm.has_task())

        assert loads == [torch.bfloat16]

        resp = client.post('/v1/audio/transcriptions', data={
            'file': (io.BytesIO(b'RIFF0000WAVEfake'), 'a.wav'),
            'model': 'vibevoice-asr-fp8',
        }, content_type='multipart/form-data')
        assert resp.status_code == 200
        assert loads == [torch.bfloat16, torch.float8_e4m3fn]


class TestPresetVoiceReuse:
    def test_preset_copied_once(self, tmp_path):
        service = QuickGenerateService(workspace_dir=tmp_path / 'ws', fake_model=True)
        preset = tmp_path / 'en-Alice_woman.wav'
        preset.write_bytes(b'v1')

        first = service.save_preset_voice_file(preset)
        copy = service.voices_dir / first
        copied_at = copy.stat().st_mtime_ns
        assert service.save_preset_voice_file(preset) == first
        assert copy.stat().st_mtime_ns == copied_at
        assert copy.read_bytes() == b'v1'

        preset.write_bytes(b'v2-updated')
        later = copy.stat().st_mtime + 10
        os.utime(preset, (later, later))
        assert service.save_preset_voice_file(preset) == first
        assert copy.read_bytes() == b'v2-updated'
        assert len(list(service.voices_dir.iterdir())) == 1
