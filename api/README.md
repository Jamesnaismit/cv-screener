# CV Screener API

FastAPI service that answers questions about the embedded CVs stored in PostgreSQL/pgvector.

## Run locally (without Docker)

```bash
cd api
pip install -r requirements.txt
export OPENAI_API_KEY=...
export DATABASE_URL=postgresql://rag_user:password@localhost:5434/cvscreener
python api.py
```

API available at `http://localhost:8000` (health at `/health`). Responses are always in English.

## Endpoints

- `GET /health` — readiness check
- `POST /query` — body: `{"question": "...", "top_k": 5}`; returns answer + sources + metadata

Example:

```bash
curl -X POST http://localhost:8000/query \
  -H "Content-Type: application/json" \
  -d '{"question": "What product experience does Caitlin Cannon have?"}'
```

## Environment

Key variables (see `.env.example`):

- `OPENAI_API_KEY` (required)
- `DATABASE_URL`
- `APP_MODEL_NAME` (default `gpt-4o-mini`)
- `APP_PORT` (default `8000`)
- `APP_TOP_K_RESULTS`, `RERANK_ENABLED`, `RERANK_TOP_K`
- `CACHE_ENABLED`, `CACHE_TTL`, `REDIS_URL`
- `METRICS_ENABLED`, `METRICS_PORT` (default `9000`, Prometheus)

## Tests

```bash
docker compose run --rm api pytest /api/tests -v
```
