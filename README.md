# OptiBot Mini Clone Indexer

Scrapes OptiSigns Help Center articles, normalizes them to Markdown, chunks changed docs, uploads them to Gemini File Search, and answers OptiBot-style support questions from that Gemini retrieval store. Local chunk search remains available as a fallback.

## Setup

```bash
cp .env.sample .env
# Set GEMINI_API_KEY for local chatbot answers.
# Optional legacy path: set UPLOAD_PROVIDER=openai plus OpenAI env vars for OpenAI vector store upload.
pip install -r requirements.txt
```

The assistant prompt is in `docs/assistant_prompt.md`.

## Run Locally

Scrape and write Markdown only:

```bash
python main.py --skip-upload
```

Scrape, chunk, and upload changed chunks to Gemini File Search:

```bash
UPLOAD_ENABLED=true python main.py --upload
```

Ask the chatbot with Gemini File Search if a store is configured in `state/articles.json`, otherwise local chunks:

```bash
python main.py --ask "How do I add a YouTube video in OptiSigns?"
```

Docker one-shot:

```bash
docker build -t optibot-indexer .
docker run --rm --env-file .env -v "$PWD/data:/app/data" optibot-indexer python main.py --ask "How do I add a YouTube video in OptiSigns?"
```

Docker Compose:

```bash
docker compose up --build web
docker compose up --build optibot
docker compose run --rm optibot python main.py --ask "How do I add a YouTube video in OptiSigns?"
docker compose --profile index run --rm indexer
docker compose --profile upload run --rm uploader
docker compose --profile test run --rm tests
```

Chat UI: <http://localhost:8000>

Tests: `python -m unittest discover`

## What It Does

- Pulls the latest `ARTICLE_LIMIT` articles from `support.optisigns.com/api/v2/help_center/en-us/articles.json`.
- Writes clean article Markdown to `data/articles/<slug>.md`.
- Builds paragraph-aware chunk Markdown in `data/chunks`, defaulting to about 700 words per chunk.
- Uploads chunks to Gemini File Search stores by default, using `models/gemini-embedding-2`.
- Answers questions with Gemini File Search when available, falling back to local chunk retrieval.
- Tracks SHA-256 article hashes and Gemini store metadata in `state/articles.json`; persist this file to reuse the same Gemini File Search store.
- Logs `added`, `updated`, `skipped`, uploaded file count, and embedded chunk count to `logs/latest.json`.

## Deployment

This project separates the public chatbot UI from the daily indexing job:

- Render hosts the FastAPI chatbot UI from `web_app.py`.
- GitHub Actions runs the daily article index/upload job.

`render.yaml` defines a Render web service. The Docker image default command is:

```bash
uvicorn web_app:app --host 0.0.0.0 --port ${PORT:-8000}
```

Set these Render environment variables before deploying:

- `GEMINI_API_KEY`
- `GEMINI_FILE_SEARCH_STORE_NAME` - the resource name returned by Gemini, for example `fileSearchStores/abc123`, not the display name.
- `CHAT_PROVIDER=gemini`
- `GEMINI_MODEL=gemini-3.1-flash-lite`

The daily job lives in `.github/workflows/daily-index.yml` as `Daily OptiBot Indexer` and runs at `03:00 Asia/Bangkok`:

```bash
python main.py --upload
```

Set these GitHub repository secrets under Settings -> Secrets and variables -> Actions:

- `GEMINI_API_KEY`
- `GEMINI_FILE_SEARCH_STORE_NAME` - use the same `fileSearchStores/...` resource name as Render.

Use the Actions tab to trigger `Daily OptiBot Indexer` manually for a smoke test.

Because GitHub-hosted runners are ephemeral, `GEMINI_FILE_SEARCH_STORE_NAME` should point at the current Gemini File Search store resource. The uploader reuses that store and refreshes changed documents so Render does not end up pointing at a deleted store.

## Screenshot

Ask the assistant: "How do I add a YouTube video?" Save the cited answer screenshot here before submission:

Screenshot: `TODO: add Playground or AI Studio screenshot`
