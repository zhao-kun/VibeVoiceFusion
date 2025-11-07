"""
Configuration management for VibeVoice backend
"""
import os
from pathlib import Path


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
    MAX_CONTENT_LENGTH = 100 * 1024 * 1024  # 100MB max file size
    UPLOAD_FOLDER = Path(os.environ.get('UPLOAD_FOLDER', './uploads'))
    ALLOWED_AUDIO_EXTENSIONS = {'.wav', '.mp3', '.m4a', '.flac'}

    # Model settings
    MODEL_PATH = os.environ.get('MODEL_PATH', './models/vibevoice')
    MODEL_DEVICE = os.environ.get('MODEL_DEVICE', 'cuda')  # or 'cpu'

    # Generation settings
    MAX_GENERATION_TIME = int(os.environ.get('MAX_GENERATION_TIME', 300))  # 5 minutes

    # API rate limiting
    RATELIMIT_ENABLED = os.environ.get('RATELIMIT_ENABLED', 'true').lower() == 'true'
    RATELIMIT_DEFAULT = os.environ.get('RATELIMIT_DEFAULT', '100 per hour')

    FAKE_MODEL = os.environ.get('FAKE_MODEL', 'false').lower() == 'true'


class DevelopmentConfig(Config):
    """Development configuration"""
    DEBUG = True
    TESTING = False


class ProductionConfig(Config):
    """Production configuration"""
    DEBUG = False
    TESTING = False


class TestingConfig(Config):
    """Testing configuration"""
    DEBUG = True
    TESTING = True


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
