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

## Daily Job

`render.yaml` defines a Render cron job that runs the Docker image daily at `03:00 UTC`:

```yaml
type: cron
schedule: "0 3 * * *"
```

The Docker image default command is:

```bash
python main.py --upload
```

Set `GEMINI_API_KEY` as a Render secret before deploying. The job uses `UPLOAD_PROVIDER=gemini` and uploads changed chunks to Gemini File Search.

After deployment, paste the Render run log URL here:

Daily job logs: `TODO: add Render cron log URL`

Persist `state/` and `logs/` with a platform disk, artifact store, or object storage; the delta uploader depends on that state between runs. If state persistence is unavailable, set `GEMINI_FILE_SEARCH_STORE_NAME` to the current store in `state/articles.json`.

## Screenshot

Ask the assistant: "How do I add a YouTube video?" Save the cited answer screenshot here before submission:

Screenshot: `TODO: add Playground or AI Studio screenshot`
