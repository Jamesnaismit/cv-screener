# Embedder

Reads CV PDFs from the feed, chunks them, generates embeddings with OpenAI, and stores them in PostgreSQL + pgvector.

## Run locally

```bash
cd embedder
pip install -r requirements.txt
export OPENAI_API_KEY=...
export DATABASE_URL=postgresql://rag_user:password@localhost:5434/cvscreener
python embedder.py --force   # re-embed everything
```

## Environment

- `EMBEDDER_INPUT_DIR` (default `/data/feed`, mounted from `./feed`)
- `EMBEDDER_BATCH_SIZE` (default 100)
- `EMBEDDER_CHUNK_SIZE` / `EMBEDDER_CHUNK_OVERLAP`
- `EMBEDDING_MODEL` (default `text-embedding-3-small`)
- `EMBEDDING_DIMENSION` (default 1536; must match `database/init.sql`)

## Tips

- CV text is extracted with `pypdf`; ensure PDFs have selectable text.
- Content hashes are used to skip unchanged files; pass `--force` to reprocess all CVs.
- `python embedder.py --stats` prints document and embedding counts.
