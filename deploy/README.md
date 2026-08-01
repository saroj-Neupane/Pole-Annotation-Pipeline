# Pole Annotation Web App

Public demo web application for Pole Annotation: product pages, an interactive
inference demo, and the browser-backend endpoints used by the UI. No login is
required; there are no external service integrations.

## Local Development

1. Install dependencies:

```bash
pip install -r requirements-deploy.txt
```

2. Create a local environment file:

```bash
cp .env.example .env
```

3. Start the app:

```bash
uvicorn deploy.main:app --host 0.0.0.0 --port 8000 --reload
```

Direct script execution also works:

```bash
python deploy/main.py
# or legacy compatibility:
python deploy/app.py
```

4. Open:

- `http://localhost:8000/`
- `http://localhost:8000/demo`

## Environment Variables (all optional)

- `BASE_URL`: local or deployed application URL (default `http://localhost:8000`).
- `ENVIRONMENT`: `development` (default) or `production`.
- `USE_PRODUCTION_MODELS`: `true` to load weights from `models/production/`.
- `SKIP_STARTUP_MODEL_LOAD`: useful for tests and light local checks.
- `ALLOWED_ORIGINS`: explicit cross-origin frontend origins. Leave unset for same-origin usage.
- `ALLOWED_HOSTS`: trusted hostnames for production.

## Deployment

Deployment target is a Hugging Face Space (Docker). See `docs/specs/hf_demo_migration.md`.
Local container test: `./deploy/run_docker_local.sh`.

## Routes

Public pages:

- `GET /`
- `GET /about`
- `GET /demo`
- `GET /health`

Browser-backend API:

- `GET /api/health-status`
- `GET /api/demo/info`
- `GET /api/demo/random`
- `GET /api/images/list`
- `GET /api/images/pole/{filename}`
- `GET /api/images/midspan/{filename}`
- `POST /demo/predict`
- `POST /predict`

## Tests

Run the app-owned deploy tests with:

```bash
pytest deploy/tests
```
