"""
Device detection and platform-specific configuration.

Usage:
    from device_config import config
    db = relationalDB(config.DB_PATH)
    etl = contentETL(config.MEDIA_DIR, db=db)

Detects Jetson vs Mac automatically and loads the correct .env file
(.env.jetson or .env.mac) before falling back to .env.

All path constants are resolved here — no other file should hard-code
platform-specific paths.
"""

import os
import platform
from pathlib import Path
from dotenv import load_dotenv


def detect_device() -> str:
    """
    Detect the current device.
    Returns: 'jetson', 'mac', or 'linux'
    """
    # Check for Jetson first (most specific)
    try:
        if os.path.exists('/proc/device-tree/model'):
            with open('/proc/device-tree/model', 'r') as f:
                model = f.read().strip()
                if 'jetson' in model.lower():
                    return 'jetson'
    except Exception:
        pass

    # Tegra release file is another Jetson indicator
    if os.path.exists('/etc/nv_tegra_release'):
        return 'jetson'

    if platform.system() == 'Darwin':
        return 'mac'

    return 'linux'


# ── Defaults per platform ────────────────────────────────────────────

_DEFAULTS = {
    'jetson': {
        'DB_PATH':       '/mnt/nvme/db/industry_signals.db',
        'DB_BACKEND':    'postgres',
        'MEDIA_DIR':     '/mnt/nvme/media/',
        'VECTOR_PATH':   '/mnt/nvme/vectors/',
        'LLM_MODEL':     os.getenv('LLM_MODEL', 'llama3'),
        'LLM_URL':       os.getenv('LLM_URL', 'http://localhost:11434'),
        'LLM_PROVIDER':  'ollama',
    },
    'mac': {
        'DB_PATH':       'Database/industry_signals.db',
        'DB_BACKEND':    'duckdb',
        'MEDIA_DIR':     'media/',
        'VECTOR_PATH':   'Vectors/',
        'LLM_MODEL':     'gemini-2.5-flash',
        'LLM_URL':       None,
        'LLM_PROVIDER':  'gemini',
    },
    'linux': {
        'DB_PATH':       'Database/industry_signals.db',
        'DB_BACKEND':    'duckdb',
        'MEDIA_DIR':     'media/',
        'VECTOR_PATH':   'Vectors/',
        'LLM_MODEL':     'gemini-2.5-flash',
        'LLM_URL':       None,
        'LLM_PROVIDER':  'gemini',
    },
}


class DeviceConfig:
    """Immutable configuration resolved at import time."""

    def __init__(self):
        self.DEVICE = detect_device()
        self._load_env()
        defaults = _DEFAULTS[self.DEVICE]

        # Each value: env var override → platform default
        self.DB_PATH      = os.getenv('DB_PATH',      os.getenv('JETSON_DB_PATH', defaults['DB_PATH']))
        self.DB_BACKEND   = os.getenv('DB_BACKEND',    defaults['DB_BACKEND'])
        self.MEDIA_DIR    = os.getenv('MEDIA_DIR',     defaults['MEDIA_DIR'])
        self.VECTOR_PATH  = os.getenv('VECTOR_PATH',   os.getenv('JETSON_VECTOR_PATH',
                                                        os.getenv('VECTOR_DB_PATH', defaults['VECTOR_PATH'])))
        self.LLM_MODEL    = os.getenv('LLM_MODEL',     os.getenv('JETSON_LLM', defaults['LLM_MODEL']))
        self.LLM_URL      = os.getenv('LLM_URL',       os.getenv('JETSON_LLM_URL', defaults['LLM_URL']))
        self.LLM_PROVIDER = os.getenv('LLM_PROVIDER',  defaults['LLM_PROVIDER'])

        # Derived paths
        self.CORPUS_VECTOR_PATH = os.path.join(self.VECTOR_PATH.rstrip('/'), 'corpus_vectors')
        self.SIGNAL_VECTOR_PATH = os.path.join(self.VECTOR_PATH.rstrip('/'), 'signal_vectors')

        self._print_banner()

    def _load_env(self):
        """Load platform-specific .env file, then fall back to .env."""
        project_root = Path(__file__).parent
        platform_env = project_root / f'.env.{self.DEVICE}'
        generic_env  = project_root / '.env'

        if platform_env.exists():
            load_dotenv(platform_env, override=True)
            print(f"Loaded env: {platform_env.name}")
        elif generic_env.exists():
            load_dotenv(generic_env, override=True)
            print(f"Loaded env: .env")
        else:
            print("Warning: no .env file found")

    def _print_banner(self):
        print(f"┌─ Device Config ─────────────────────────")
        print(f"│ Device:   {self.DEVICE}")
        print(f"│ DB:       {self.DB_BACKEND} → {self.DB_PATH}")
        print(f"│ Media:    {self.MEDIA_DIR}")
        print(f"│ Vectors:  {self.VECTOR_PATH}")
        print(f"│ LLM:      {self.LLM_PROVIDER}/{self.LLM_MODEL}")
        print(f"└──────────────────────────────────────────")

    @property
    def is_jetson(self) -> bool:
        return self.DEVICE == 'jetson'

    @property
    def is_mac(self) -> bool:
        return self.DEVICE == 'mac'

    def as_dict(self) -> dict:
        return {k: v for k, v in self.__dict__.items() if k.isupper()}


# Singleton — resolved once at first import
config = DeviceConfig()
