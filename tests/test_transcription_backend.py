"""Backend transcription (ASR) integration tests using the fake engine; the real model is never loaded."""
import io
import json
import time

import pytest

from backend.app import create_app
from backend.inference.asr_inference import ASRInferenceBase, FakeASRModel
from backend.models.transcription import (
    Transcription, segment_seconds, transcript_plain_text, transcript_readable_text,
)
from backend.task_manager.task import Task, gm


class _BlockingTask(Task):
    """Occupies the single GPU slot so busy behaviour can be tested"""

    def __init__(self):
        super().__init__(task_id='blocking')
        self.release = False

    def run(self):
        while not self.release:
            time.sleep(0.01)

    def task_failure(self, error_msg, failure_type='general'):
        pass

    def task_success(self, message):
        pass

    def task_appended(self, message):
        pass

    def unwrap(self):
        return None

    def _task_finalize(self):
        pass


def _wait_idle(timeout=10):
    deadline = time.time() + timeout
    while gm.has_task() and time.time() < deadline:
        time.sleep(0.02)
    assert not gm.has_task(), "task manager did not become idle"


@pytest.fixture
def client(tmp_path, monkeypatch):
    monkeypatch.setattr(FakeASRModel, 'step_delay', 0.0)
    monkeypatch.delenv('OPENAI_COMPAT_API_KEY', raising=False)
    app = create_app('testing')
    app.config['WORKSPACE_DIR'] = tmp_path
    app.config['FAKE_MODEL'] = True
    _wait_idle()
    with app.test_client() as c:
        yield c
    _wait_idle()


def _audio_upload(name='sample.wav'):
    return (io.BytesIO(b'RIFF0000WAVEfake'), name)


def _wait_completed(client, url, timeout=10):
    deadline = time.time() + timeout
    while time.time() < deadline:
        data = client.get(url).get_json()
        if data['status'] in ('completed', 'failed'):
            return data
        time.sleep(0.05)
    raise AssertionError(f"{url} did not finish")


class TestTranscriptionModel:
    def test_round_trip(self):
        record = Transcription.create(request_id='r1', audio_file='a.wav', original_filename='x.wav',
                                      project_id='p', offloading={'enabled': True, 'mode': 'preset'})
        record.segments = [{'start_time': 0.0, 'end_time': 1.0, 'speaker_id': 0, 'text': 'hi'}]
        restored = Transcription.from_dict(json.loads(json.dumps(record.to_dict())))
        assert restored == record
        assert restored.status == 'pending'
        assert restored.details.offloading_config == {'enabled': True, 'mode': 'preset'}

    def test_text_helpers(self):
        segments = [
            {'start_time': '0.00', 'end_time': 1.5, 'speaker_id': 0, 'text': ' Hello '},
            {'start_time': 61.25, 'end_time': 3725, 'speaker_id': 1, 'text': 'World'},
        ]
        assert segment_seconds('1.5') == 1.5
        assert segment_seconds('n/a') is None
        assert transcript_plain_text(segments) == 'Hello\nWorld'
        assert transcript_plain_text([], raw_text=' raw ') == 'raw'
        assert transcript_readable_text(segments).splitlines() == [
            '[00:00.00 - 00:01.50] Speaker 0: Hello',
            '[01:01.25 - 1:02:05.00] Speaker 1: World',
        ]


class TestFakeEngine:
    def test_run_inference_updates_transcription(self, tmp_path, monkeypatch):
        monkeypatch.setattr(FakeASRModel, 'step_delay', 0.0)
        audio = tmp_path / 'a.wav'
        audio.write_bytes(b'data')
        record = Transcription.create(request_id='r1', audio_file='a.wav', original_filename='a.wav')
        engine = ASRInferenceBase.create(record, str(audio), fake=True)

        engine.run_inference()
        result = engine.get_transcription()

        assert result.status == 'completed'
        assert result.audio_duration == 5.0
        assert result.generated_tokens > 0
        assert [s['speaker_id'] for s in result.segments] == [0, 1]
        assert result.text_preview.startswith('This is a fake transcription.')
        assert result.details.prompt_tokens == 120

    def test_missing_audio_raises(self, tmp_path):
        record = Transcription.create(request_id='r1', audio_file='a.wav', original_filename='a.wav')
        engine = ASRInferenceBase.create(record, str(tmp_path / 'missing.wav'), fake=True)
        with pytest.raises(FileNotFoundError):
            engine.run_inference()


class TestStandaloneAPI:
    def test_full_lifecycle(self, client, tmp_path):
        resp = client.post('/api/v1/transcriptions', data={
            'audio_file': _audio_upload('meeting.wav'),
            'context_info': 'VibeVoice, Qwen',
            'max_new_tokens': '512',
        }, content_type='multipart/form-data')
        assert resp.status_code == 200, resp.get_json()
        request_id = resp.get_json()['request_id']
        assert resp.get_json()['project_id'] is None

        data = _wait_completed(client, f'/api/v1/transcriptions/{request_id}')
        assert data['status'] == 'completed', data
        assert data['context_info'] == 'VibeVoice, Qwen'
        assert data['max_new_tokens'] == 512
        assert len(data['segments']) == 2
        _wait_idle()

        history = client.get('/api/v1/transcriptions/history').get_json()
        assert history['total'] == 1
        assert history['transcriptions'][0]['segment_count'] == 2
        assert history['transcriptions'][0]['status'] == 'completed'
        assert (tmp_path / '_transcriptions' / 'history.json').exists()

        assert client.get('/api/v1/transcriptions/current').get_json()['transcription'] is None

        txt = client.get(f'/api/v1/transcriptions/{request_id}/download?format=txt')
        assert txt.status_code == 200
        assert 'Speaker 1: It is produced without loading the model.' in txt.get_data(as_text=True)
        assert 'meeting_transcript.txt' in txt.headers['Content-Disposition']
        as_json = client.get(f'/api/v1/transcriptions/{request_id}/download?format=json').get_json()
        assert as_json['file'] == 'meeting.wav'
        assert client.get(f'/api/v1/transcriptions/{request_id}/download?format=srt').status_code == 400

        audio = client.get(f'/api/v1/transcriptions/{request_id}/audio')
        assert audio.status_code == 200 and audio.data == b'RIFF0000WAVEfake'

        assert client.delete(f'/api/v1/transcriptions/{request_id}').status_code == 200
        assert client.get(f'/api/v1/transcriptions/{request_id}').status_code == 404
        assert list((tmp_path / '_transcriptions' / 'audio').iterdir()) == []

    def test_validation_errors(self, client):
        assert client.post('/api/v1/transcriptions', data={},
                           content_type='multipart/form-data').status_code == 400
        assert client.post('/api/v1/transcriptions', data={'audio_file': _audio_upload('a.txt')},
                           content_type='multipart/form-data').status_code == 400
        for field, value in [('max_new_tokens', '0'), ('temperature', 'hot'), ('model_dtype', 'fp4'),
                             ('offloading', '{"enabled": true, "mode": "manual", "num_gpu_layers": 99}')]:
            resp = client.post('/api/v1/transcriptions', data={'audio_file': _audio_upload(), field: value},
                               content_type='multipart/form-data')
            assert resp.status_code == 400, field
        assert not gm.has_task()

    def test_busy_returns_conflict_and_cleans_up(self, client, tmp_path):
        blocker = _BlockingTask()
        assert gm.add_task(blocker)
        try:
            resp = client.post('/api/v1/transcriptions', data={'audio_file': _audio_upload()},
                               content_type='multipart/form-data')
            assert resp.status_code == 409
            assert client.get('/api/v1/transcriptions/history').get_json()['total'] == 0
            assert list((tmp_path / '_transcriptions' / 'audio').iterdir()) == []
        finally:
            blocker.release = True

    def test_running_task_is_reported(self, client, monkeypatch):
        monkeypatch.setattr(FakeASRModel, 'step_delay', 0.05)
        request_id = client.post('/api/v1/transcriptions', data={'audio_file': _audio_upload()},
                                 content_type='multipart/form-data').get_json()['request_id']

        task = client.get('/api/v1/tasks/current').get_json()['task']
        assert task['type'] == 'transcription'
        assert task['project_id'] is None
        assert task['data']['request_id'] == request_id
        assert client.get('/api/v1/transcriptions/current').get_json()['transcription']['request_id'] == request_id
        assert client.delete(f'/api/v1/transcriptions/{request_id}').status_code == 409

        _wait_completed(client, f'/api/v1/transcriptions/{request_id}')


class TestProjectAPI:
    def test_project_scoped_lifecycle(self, client, tmp_path):
        project = client.post('/api/v1/projects', json={'name': 'ASR Project'}).get_json()
        project_id = project['id']
        base = f'/api/v1/projects/{project_id}/transcriptions'

        resp = client.post(base, data={'audio_file': _audio_upload()}, content_type='multipart/form-data')
        assert resp.status_code == 200, resp.get_json()
        request_id = resp.get_json()['request_id']
        assert resp.get_json()['project_id'] == project_id

        data = _wait_completed(client, f'{base}/{request_id}')
        assert data['status'] == 'completed' and data['project_id'] == project_id
        _wait_idle()

        assert client.get(base).get_json()['total'] == 1
        assert (tmp_path / project_id / 'transcriptions' / 'history.json').exists()
        # Project and standalone histories are separate storage roots
        assert client.get('/api/v1/transcriptions/history').get_json()['total'] == 0
        assert client.get(f'/api/v1/transcriptions/{request_id}').status_code == 404

        resp = client.post(f'{base}/batch-delete', json={'request_ids': [request_id, 'missing']})
        assert resp.get_json()['deleted_ids'] == [request_id]
        assert resp.get_json()['failed_ids'] == ['missing']

    def test_unknown_project(self, client):
        resp = client.post('/api/v1/projects/nope/transcriptions', data={'audio_file': _audio_upload()},
                           content_type='multipart/form-data')
        assert resp.status_code == 404
        assert client.get('/api/v1/projects/nope/transcriptions').status_code == 404


class TestOpenAITranscription:
    def _post(self, client, **form):
        data = {'file': _audio_upload(), 'model': 'whisper-1'}
        data.update(form)
        return client.post('/v1/audio/transcriptions', data=data, content_type='multipart/form-data')

    def test_response_formats(self, client):
        resp = self._post(client)
        assert resp.status_code == 200, resp.get_json()
        assert resp.get_json() == {
            'text': 'This is a fake transcription.\nIt is produced without loading the model.'
        }
        _wait_idle()

        resp = self._post(client, response_format='text', prompt='names: Alice')
        assert resp.mimetype == 'text/plain'
        assert resp.get_data(as_text=True).startswith('This is a fake transcription.')
        _wait_idle()

        verbose = self._post(client, response_format='verbose_json', language='en').get_json()
        assert verbose['task'] == 'transcribe' and verbose['language'] == 'en'
        assert verbose['duration'] == 5.0
        assert verbose['segments'][1] == {'id': 1, 'start': 2.5, 'end': 5.0, 'speaker': 1,
                                          'text': 'It is produced without loading the model.'}
        _wait_idle()

        history = client.get('/api/v1/transcriptions/history').get_json()
        assert history['total'] == 3

    def test_request_errors(self, client, monkeypatch):
        assert self._post(client, response_format='srt').status_code == 400
        assert self._post(client, temperature='2').status_code == 400
        assert client.post('/v1/audio/transcriptions', data={'model': 'whisper-1'},
                           content_type='multipart/form-data').status_code == 400

        monkeypatch.setenv('OPENAI_COMPAT_API_KEY', 'secret')
        assert self._post(client).status_code == 401

    def test_busy_returns_503(self, client):
        blocker = _BlockingTask()
        assert gm.add_task(blocker)
        try:
            resp = self._post(client)
            assert resp.status_code == 503
            assert resp.get_json()['error']['type'] == 'server_error'
        finally:
            blocker.release = True

    def test_models_lists_asr_models(self, client):
        ids = [m['id'] for m in client.get('/v1/models').get_json()['data']]
        assert {'vibevoice-asr', 'whisper-1', 'tts-1', 'vibevoice-7b'} <= set(ids)
