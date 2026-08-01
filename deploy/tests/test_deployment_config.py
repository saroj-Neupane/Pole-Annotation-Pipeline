from pathlib import Path


def test_main_app_configuration():
    from deploy.main import app

    assert app.title == "Pole Annotation"
    assert app.docs_url is None
    assert app.redoc_url is None
    assert app.openapi_url is None


def test_deployment_artifacts_use_new_entrypoint_and_name():
    root = Path(__file__).resolve().parents[2]
    dockerfile = (root / "deploy" / "Dockerfile").read_text(encoding="utf-8")
    local_script = (root / "deploy" / "run_docker_local.sh").read_text(encoding="utf-8")

    assert "deploy.main:app" in dockerfile
    assert "pole-annotation-app:local" in local_script
