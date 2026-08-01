"""Centralized deployment configuration."""
from __future__ import annotations

import os
from dataclasses import dataclass
from functools import lru_cache
from pathlib import Path

from dotenv import load_dotenv


PROJECT_ROOT = Path(__file__).resolve().parent.parent
load_dotenv(PROJECT_ROOT / ".env")


def _parse_bool(name: str, default: bool = False) -> bool:
    value = os.getenv(name)
    if value is None:
        return default
    return value.strip().lower() in {"1", "true", "yes", "on"}


def _parse_csv(name: str) -> tuple[str, ...]:
    value = os.getenv(name, "")
    return tuple(part.strip() for part in value.split(",") if part.strip())


@dataclass(frozen=True)
class Settings:
    environment: str
    base_url: str
    use_production_models: bool
    skip_startup_model_load: bool
    allowed_origins: tuple[str, ...]
    allowed_hosts: tuple[str, ...]

    @property
    def is_production(self) -> bool:
        return self.environment == "production"


def _build_settings() -> Settings:
    settings = Settings(
        environment=os.getenv("ENVIRONMENT", "development").strip().lower() or "development",
        base_url=os.getenv("BASE_URL", "http://localhost:8000").rstrip("/"),
        use_production_models=_parse_bool("USE_PRODUCTION_MODELS", default=True),
        skip_startup_model_load=_parse_bool("SKIP_STARTUP_MODEL_LOAD"),
        allowed_origins=_parse_csv("ALLOWED_ORIGINS"),
        allowed_hosts=_parse_csv("ALLOWED_HOSTS"),
    )

    return settings


@lru_cache(maxsize=1)
def get_settings() -> Settings:
    """Return cached settings parsed from the environment."""
    return _build_settings()


def clear_settings_cache() -> None:
    """Clear cached settings for tests."""
    get_settings.cache_clear()
