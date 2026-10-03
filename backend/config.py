"""
Configuration management for VibeVoice backend
"""
import os
from pathlib import Path
from dataclasses import dataclass


@dataclass
class Config:
    """Base configuration"""

    # Flask
    SECRET_KEY = os.environ.get('SECRET_KEY') or 'dev-secret-key-change-in-production'

    # CORS
    CORS_ORIGINS = os.environ.get('CORS_ORIGINS', 'http://localhost:3000,http://localhost:3001').split(',')

    # Workspace settings
    WORKSPACE_DIR = Path(os.environ.get('WORKSPACE_DIR', './workspace')).resolve()
    PROJECTS_META_FILE = 'projects.json'

    # Upload settings
    MAX_CONTENT_LENGTH = int(os.environ.get('MAX_CONTENT_LENGTH', 1024 * 1024 * 1024))  # 500MB max file size
    UPLOAD_FOLDER = Path(os.environ.get('UPLOAD_FOLDER', './uploads'))
    ALLOWED_AUDIO_EXTENSIONS = {'.wav', '.mp3', '.m4a', '.flac'}

    # Model settings
    MODEL_PATH = os.environ.get('MODEL_PATH', './models/vibevoice')
    MODEL_DEVICE = os.environ.get('MODEL_DEVICE', 'cuda')  # or 'cpu'
    ASR_MODEL_PATH = os.environ.get('ASR_MODEL_PATH', './models/VibeVoice-ASR')

    # Generation settings
    MAX_GENERATION_TIME = int(os.environ.get('MAX_GENERATION_TIME', 300))  # 5 minutes

    # Preset voice settings
    PRESET_VOICE_DIR = Path(os.environ.get('PRESET_VOICE_DIR', os.path.join(os.path.dirname(__file__), 'preset_voice'))).resolve()

    # API rate limiting
    RATELIMIT_ENABLED = os.environ.get('RATELIMIT_ENABLED', 'true').lower() == 'true'
    RATELIMIT_DEFAULT = os.environ.get('RATELIMIT_DEFAULT', '100 per hour')

    FAKE_MODEL = os.environ.get('FAKE_MODEL', 'false').lower() == 'true'

    # Live (streaming) transcription over the OpenAI Realtime WebSocket at /v1/realtime
    STREAMING_ASR_MAX_SESSION_SECONDS = int(os.environ.get('STREAMING_ASR_MAX_SESSION_SECONDS', 1800))
    STREAMING_ASR_IDLE_TIMEOUT_SECONDS = int(os.environ.get('STREAMING_ASR_IDLE_TIMEOUT_SECONDS', 60))
    STREAMING_ASR_MAX_WINDOW_SECONDS = float(os.environ.get('STREAMING_ASR_MAX_WINDOW_SECONDS', 30))
    STREAMING_ASR_SILENCE_COMMIT_SECONDS = float(os.environ.get('STREAMING_ASR_SILENCE_COMMIT_SECONDS', 1.0))
    STREAMING_ASR_SILENCE_RMS = float(os.environ.get('STREAMING_ASR_SILENCE_RMS', 0.008))
    # Browsers fail the handshake unless the server echoes one of the requested subprotocols
    SOCK_SERVER_OPTIONS = {'subprotocols': ['realtime'], 'ping_interval': 25}

@dataclass
class DevelopmentConfig(Config):
    """Development configuration"""
    FLASK_DEBUG = True
    FLASK_TESTING = False


@dataclass
class ProductionConfig(Config):
    """Production configuration"""
    FLASK_DEBUG = False
    FLASK_TESTING = False


@dataclass
class TestingConfig(Config):
    """Testing configuration"""
    FLASK_DEBUG = True
    FLASK_TESTING = True


# Configuration dictionary
config = {
    'development': DevelopmentConfig,
    'production': ProductionConfig,
    'testing': TestingConfig,
    'default': DevelopmentConfig
}


def get_config(env=None):
    """Get configuration based on environment"""
    if env is None:
        env = os.environ.get('FLASK_ENV', 'development')
    return config.get(env, config['default'])
