import importlib
import os
import sys

import pytest
from fastapi.testclient import TestClient


@pytest.fixture()
def client():
    env = {
        "ENVIRONMENT": "development",
        "USE_PRODUCTION_MODELS": "true",
        "SKIP_STARTUP_MODEL_LOAD": "true",
        "BASE_URL": "http://testserver",
    }
    previous = {key: os.environ.get(key) for key in env}
    os.environ.update(env)

    try:
        for module_name in ["deploy.config", "deploy.security", "deploy.main"]:
            if module_name in sys.modules:
                importlib.reload(sys.modules[module_name])
            else:
                importlib.import_module(module_name)

        from deploy.config import clear_settings_cache
        from deploy.main import app

        clear_settings_cache()
        with TestClient(app) as test_client:
            yield test_client
    finally:
        for key, value in previous.items():
            if value is None:
                os.environ.pop(key, None)
            else:
                os.environ[key] = value

