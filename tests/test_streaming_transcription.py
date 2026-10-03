"""Live (streaming) transcription tests over the OpenAI Realtime WebSocket using the fake engine; the real model is never loaded."""
import base64
import json
import threading
import time
import wave

import numpy as np
import pytest
from simple_websocket import Client, ConnectionClosed
from werkzeug.serving import make_server

from backend.app import create_app
from backend.inference.asr_inference import FakeASRModel
from backend.inference.streaming_asr_inference import (
    SAMPLE_RATE, HYPOTHESIS_EVENT, STATUS_EVENT, RollingTranscriber, analyze_energy, stable_prefix,
)
from backend.models.transcription import Transcription
from backend.services.openai_realtime_service import RealtimeTranscriptionHandler
from backend.task_manager.task import gm

from test_transcription_backend import _BlockingTask, _wait_idle


def _seg(start, end, text):
    return {'start_time': start, 'end_time': end, 'speaker_id': 0, 'text': text}


def _noise_pcm(seconds, amplitude=0.1, seed=0):
    rng = np.random.default_rng(seed)
    samples = rng.standard_normal(int(SAMPLE_RATE * seconds)) * amplitude
    return (samples * 32767).clip(-32768, 32767).astype('<i2').tobytes()


class TestStablePrefix:
    def test_cuts_back_to_word_boundary(self):
        assert stable_prefix("hello wor", "hello world") == "hello "
        assert stable_prefix("hello world", "hello world again") == "hello world"

    def test_no_common_prefix(self):
        assert stable_prefix("abc", "xyz") == ""

    def test_cjk_is_cut_per_character(self):
        assert stable_prefix("今天天气", "今天天气很好") == "今天天气"
        assert stable_prefix("今天天气不", "今天天气很好") == "今天天气"


class TestAnalyzeEnergy:
    def test_silence(self):
        result = analyze_energy(np.zeros(SAMPLE_RATE * 2, dtype=np.float32), 0.008, 1.0)
        assert result == {'has_speech': False, 'trailing_silence': True}

    def test_speech_then_silence(self):
        audio = np.concatenate([np.full(SAMPLE_RATE, 0.1, dtype=np.float32),
                                np.zeros(int(SAMPLE_RATE * 1.2), dtype=np.float32)])
        assert analyze_energy(audio, 0.008, 1.0) == {'has_speech': True, 'trailing_silence': True}

    def test_speech_until_end(self):
        audio = np.full(SAMPLE_RATE * 2, 0.1, dtype=np.float32)
        assert analyze_energy(audio, 0.008, 1.0) == {'has_speech': True, 'trailing_silence': False}


class TestRollingTranscriber:
    def _make(self, max_window=30.0):
        events = []
        return RollingTranscriber(events.append, max_window), events

    @staticmethod
    def _types(events):
        return [e['type'] for e in events]

    def test_commits_segments_stable_across_two_passes(self):
        t, events = self._make()
        t.apply_pass([_seg(0, 3, "first one"), _seg(3, 4, "second")], 0, 4 * SAMPLE_RATE, False, False)
        assert t.committed == []
        assert events[-1]['type'] == HYPOTHESIS_EVENT and events[-1]['text'] == "first one second"

        t.apply_pass([_seg(0, 3, "first one"), _seg(3, 5, "second part")], 0, 5 * SAMPLE_RATE, False, False)
        assert [s['text'] for s in t.committed] == ["first one"]
        assert t.commit_sample == 3 * SAMPLE_RATE
        completed = [e for e in events if e['type'].endswith('.completed')]
        assert completed[0]['transcript'] == "first one"
        delta = [e for e in events if e['type'].endswith('.delta')]
        assert delta[-1]['delta'] == "second"

    def test_last_segment_is_never_committed_without_silence(self):
        t, _ = self._make()
        for end in (2, 3, 4):
            t.apply_pass([_seg(0, end, "same text")], 0, end * SAMPLE_RATE, False, False)
        assert t.committed == []
        assert t.commit_sample == 0

    def test_relative_timestamps_become_absolute(self):
        t, _ = self._make()
        t.commit_sample = 10 * SAMPLE_RATE
        t.apply_pass([_seg(0, 2, "later")], 10 * SAMPLE_RATE, 12 * SAMPLE_RATE, True, False)
        assert t.committed == [_seg(10.0, 12.0, "later")]

    def test_trailing_silence_commits_everything(self):
        t, events = self._make()
        t.apply_pass([_seg(0, 1, "a"), _seg(1, 2, "b")], 0, 3 * SAMPLE_RATE, True, False)
        assert [s['text'] for s in t.committed] == ["a", "b"]
        assert t.commit_sample == 3 * SAMPLE_RATE
        assert self._types(events) == [
            'input_audio_buffer.committed', 'conversation.item.input_audio_transcription.completed',
            'input_audio_buffer.committed', 'conversation.item.input_audio_transcription.completed',
        ]
        assert events[2]['previous_item_id'] == events[0]['item_id']

    def test_window_cap_forces_commit(self):
        t, _ = self._make(max_window=5.0)
        t.apply_pass([_seg(0, 3, "x"), _seg(3, 6, "y")], 0, 6 * SAMPLE_RATE, False, False)
        assert [s['text'] for s in t.committed] == ["x"]
        assert t.commit_sample == 3 * SAMPLE_RATE

    def test_window_cap_with_single_segment_cuts_hard(self):
        t, _ = self._make(max_window=5.0)
        t.apply_pass([_seg(0, 6, "one long run")], 0, 6 * SAMPLE_RATE, False, False)
        assert [s['text'] for s in t.committed] == ["one long run"]
        assert t.commit_sample == 6 * SAMPLE_RATE

    def test_deltas_are_append_only(self):
        t, events = self._make()
        t.apply_pass([_seg(0, 1, "hello")], 0, SAMPLE_RATE, False, False)
        t.apply_pass([_seg(0, 2, "hello there")], 0, 2 * SAMPLE_RATE, False, False)
        t.apply_pass([_seg(0, 3, "hello there friend")], 0, 3 * SAMPLE_RATE, False, False)
        t.apply_pass([_seg(0, 4, "hello their friend")], 0, 4 * SAMPLE_RATE, False, False)
        deltas = [e['delta'] for e in events if e['type'].endswith('.delta')]
        assert deltas == ["hello", " there"]

    def test_clear_drops_hypothesis(self):
        t, events = self._make()
        t.apply_pass([_seg(0, 2, "to be dropped")], 0, 2 * SAMPLE_RATE, False, False)
        t.clear(2 * SAMPLE_RATE)
        assert t.commit_sample == 2 * SAMPLE_RATE
        assert events[-1]['type'] == HYPOTHESIS_EVENT and events[-1]['text'] == ""
        t.commit_hypothesis(2 * SAMPLE_RATE)
        assert t.committed == []

    def test_commit_hypothesis_keeps_last_pass(self):
        t, _ = self._make()
        t.apply_pass([_seg(0, 2, "pending words")], 0, 2 * SAMPLE_RATE, False, False)
        t.commit_hypothesis(int(2.2 * SAMPLE_RATE))
        assert [s['text'] for s in t.committed] == ["pending words"]


class _FakeService:
    def __init__(self, busy=False):
        self.busy = busy
        self.started = []

    def start_live_transcription(self, session, limits, model_dtype='bf16', seeds=42, offloading_config=None):
        if self.busy:
            return None
        self.started.append({'model_dtype': model_dtype, 'context_info': session.context_info})
        return Transcription.create(request_id='live1', audio_file='a.wav', original_filename='live.wav',
                                    source='live')


def _handler(service):
    return RealtimeTranscriptionHandler(service, limits={}, max_session_seconds=10)


class TestRealtimeHandler:
    def test_ga_session_update(self):
        service = _FakeService()
        handler = _handler(service)
        events = handler.handle(json.dumps({'type': 'session.update', 'session': {
            'type': 'transcription',
            'audio': {'input': {'format': {'type': 'audio/pcm', 'rate': 24000},
                                'transcription': {'model': 'vibevoice-asr-fp8', 'prompt': 'names: Kun'}}},
        }}))
        assert [e['type'] for e in events] == ['session.updated']
        assert events[0]['session']['audio']['input']['transcription']['model'] == 'vibevoice-asr-fp8'
        assert service.started == [{'model_dtype': 'float8_e4m3fn', 'context_info': 'names: Kun'}]

    def test_beta_session_update(self):
        service = _FakeService()
        handler = _handler(service)
        events = handler.handle(json.dumps({'type': 'transcription_session.update', 'session': {
            'input_audio_format': 'pcm16', 'input_audio_transcription': {'model': 'whisper-1', 'language': 'en'},
        }}))
        assert [e['type'] for e in events] == ['transcription_session.updated']
        assert events[0]['session']['input_audio_transcription']['language'] == 'en'
        assert service.started[0]['model_dtype'] == 'bf16'

    def test_rejects_unsupported_formats(self):
        handler = _handler(_FakeService())
        beta = handler.handle(json.dumps({'type': 'transcription_session.update',
                                          'session': {'input_audio_format': 'g711_ulaw'}}))
        ga = handler.handle(json.dumps({'type': 'session.update', 'session': {
            'type': 'transcription', 'audio': {'input': {'format': {'type': 'audio/pcm', 'rate': 16000}}}}}))
        realtime = handler.handle(json.dumps({'type': 'session.update', 'session': {'type': 'realtime'}}))
        assert [e['error']['code'] for e in beta + ga + realtime] == [
            'unsupported_audio_format', 'unsupported_audio_format', 'unsupported_session_type']
        assert handler.session is None

    def test_bad_messages(self):
        handler = _handler(_FakeService())
        assert handler.handle('not json')[0]['error']['code'] == 'invalid_json'
        assert handler.handle(b'\x00')[0]['error']['code'] == 'invalid_message'
        assert handler.handle(json.dumps({'type': 'nope', 'event_id': 'c1'}))[0]['error'] == {
            'type': 'invalid_request_error', 'message': "Unsupported client event type 'nope'.",
            'code': 'unknown_event', 'event_id': 'c1'}
        bad_audio = handler.handle(json.dumps({'type': 'input_audio_buffer.append', 'audio': '%%%'}))
        assert bad_audio[0]['error']['code'] == 'invalid_audio'

    def test_append_starts_with_defaults_and_enforces_max_duration(self):
        service = _FakeService()
        handler = _handler(service)
        audio = base64.b64encode(_noise_pcm(11)).decode()
        events = handler.handle(json.dumps({'type': 'input_audio_buffer.append', 'audio': audio}))
        assert service.started[0]['model_dtype'] == 'bf16'
        assert events[-1]['type'] == STATUS_EVENT and events[-1]['reason'] == 'max_duration'
        assert handler.session.is_finishing()

    def test_busy(self):
        handler = _handler(_FakeService(busy=True))
        events = handler.handle(json.dumps({'type': 'session.update', 'session': {'type': 'transcription'}}))
        assert events[0]['error']['code'] == 'server_busy'
        assert handler.done

    def test_finish_before_start(self):
        handler = _handler(_FakeService())
        events = handler.handle(json.dumps({'type': 'vibevoice.session.finish'}))
        assert events[0]['status'] == 'completed' and events[0]['request_id'] is None
        assert handler.done


@pytest.fixture
def live_server(tmp_path, monkeypatch):
    monkeypatch.setattr(FakeASRModel, 'step_delay', 0.0)
    monkeypatch.delenv('OPENAI_COMPAT_API_KEY', raising=False)
    app = create_app('testing')
    app.config['WORKSPACE_DIR'] = tmp_path
    app.config['FAKE_MODEL'] = True
    _wait_idle()
    server = make_server('127.0.0.1', 0, app, threaded=True)
    thread = threading.Thread(target=server.serve_forever, daemon=True)
    thread.start()
    yield app, f"ws://127.0.0.1:{server.server_port}/v1/realtime"
    server.shutdown()
    _wait_idle()


def _connect(url, subprotocols=('realtime',)):
    return Client.connect(url, subprotocols=list(subprotocols))


def _receive_all(ws, timeout=15):
    """Events until the server closes the connection"""
    events = []
    deadline = time.time() + timeout
    try:
        while time.time() < deadline:
            message = ws.receive(timeout=1)
            if message is not None:
                events.append(json.loads(message))
    except ConnectionClosed:
        return events
    raise AssertionError("server did not close the connection")


def _stream(ws, pcm, chunk_seconds=0.1, pause=0.01):
    step = int(SAMPLE_RATE * chunk_seconds) * 2
    for i in range(0, len(pcm), step):
        ws.send(json.dumps({'type': 'input_audio_buffer.append', 'audio': base64.b64encode(pcm[i:i + step]).decode()}))
        time.sleep(pause)


def _history(app, project_dir=None):
    base = (project_dir or app.config['WORKSPACE_DIR'] / '_transcriptions')
    return json.loads((base / 'history.json').read_text())


class TestRealtimeWebSocket:
    def test_full_session_ga(self, live_server):
        app, url = live_server
        ws = _connect(url)
        assert ws.subprotocol == 'realtime'
        created = json.loads(ws.receive(timeout=5))
        assert created['type'] == 'session.created' and created['event_id']

        ws.send(json.dumps({'type': 'session.update', 'session': {'type': 'transcription'}}))
        _stream(ws, _noise_pcm(8))
        ws.send(json.dumps({'type': 'vibevoice.session.finish'}))
        events = _receive_all(ws)
        types = [e['type'] for e in events]

        assert types[0] == 'session.updated'
        assert 'conversation.item.input_audio_transcription.completed' in types
        statuses = [e['status'] for e in events if e['type'] == STATUS_EVENT]
        assert statuses[:2] == ['loading_model', 'listening']
        assert statuses[-1] == 'completed'
        request_id = events[-1]['request_id']

        completed = [e for e in events if e['type'].endswith('.completed')]
        record = _history(app)[0]
        assert record['request_id'] == request_id
        assert record['status'] == 'completed' and record['source'] == 'live'
        assert record['audio_duration'] == pytest.approx(8.0)
        assert [s['text'] for s in record['segments']] == [e['transcript'] for e in completed]
        audio_path = app.config['WORKSPACE_DIR'] / '_transcriptions' / 'audio' / record['audio_file']
        with wave.open(str(audio_path)) as wav:
            assert wav.getframerate() == SAMPLE_RATE and wav.getnframes() == SAMPLE_RATE * 8

        with app.test_client() as client:
            listed = client.get('/api/v1/transcriptions/history').get_json()
            assert listed['transcriptions'][0]['source'] == 'live'
            assert client.get(f'/api/v1/transcriptions/{request_id}/audio').status_code == 200

    def test_beta_dialect_and_disconnect_still_saves(self, live_server):
        app, url = live_server
        ws = _connect(url)
        ws.receive(timeout=5)
        ws.send(json.dumps({'type': 'transcription_session.update',
                            'session': {'input_audio_format': 'pcm16'}}))
        assert json.loads(ws.receive(timeout=5))['type'] == 'transcription_session.updated'
        _stream(ws, _noise_pcm(2))
        ws.close()
        _wait_idle()
        record = _history(app)[0]
        assert record['status'] == 'completed' and record['source'] == 'live'
        assert record['audio_duration'] == pytest.approx(2.0)

    def test_current_task_reports_live_transcription(self, live_server):
        app, url = live_server
        ws = _connect(url)
        ws.receive(timeout=5)
        ws.send(json.dumps({'type': 'session.update', 'session': {'type': 'transcription'}}))
        _stream(ws, _noise_pcm(1))
        with app.test_client() as client:
            task = client.get('/api/v1/tasks/current').get_json()['task']
            assert task['type'] == 'transcription'
            assert task['data']['source'] == 'live'
        ws.send(json.dumps({'type': 'vibevoice.session.finish'}))
        _receive_all(ws)

    def test_project_scope(self, live_server):
        app, url = live_server
        with app.test_client() as client:
            project = client.post('/api/v1/projects', json={'name': 'Live'}).get_json()
        project_id = project['id']
        ws = _connect(f"{url}?project_id={project_id}")
        ws.receive(timeout=5)
        ws.send(json.dumps({'type': 'session.update', 'session': {'type': 'transcription'}}))
        _stream(ws, _noise_pcm(1.5))
        ws.send(json.dumps({'type': 'vibevoice.session.finish'}))
        _receive_all(ws)
        with app.test_client() as client:
            listed = client.get(f'/api/v1/projects/{project_id}/transcriptions').get_json()
            assert listed['transcriptions'][0]['source'] == 'live'
            assert listed['transcriptions'][0]['project_id'] == project_id

    def test_unknown_project(self, live_server):
        _, url = live_server
        ws = _connect(f"{url}?project_id=missing")
        events = _receive_all(ws)
        assert events[0]['error']['code'] == 'project_not_found'

    def test_busy(self, live_server):
        _, url = live_server
        blocker = _BlockingTask()
        assert gm.add_task(blocker)
        try:
            events = _receive_all(_connect(url))
            assert events[0]['error']['code'] == 'server_busy'
        finally:
            blocker.release = True

    def test_api_key(self, live_server, monkeypatch):
        _, url = live_server
        monkeypatch.setenv('OPENAI_COMPAT_API_KEY', 'secret')
        events = _receive_all(_connect(url))
        assert events[0]['error']['code'] == 'invalid_api_key'

        ws = _connect(url, subprotocols=('realtime', 'openai-insecure-api-key.secret'))
        assert json.loads(ws.receive(timeout=5))['type'] == 'session.created'
        ws.send(json.dumps({'type': 'vibevoice.session.finish'}))
        _receive_all(ws)
