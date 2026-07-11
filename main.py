from __future__ import annotations

import argparse
import hashlib
import json
import logging
import os
import re
import time
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any
from urllib.parse import urljoin, urlparse

import html2text
import requests
from bs4 import BeautifulSoup, NavigableString

try:
    from dotenv import load_dotenv
except ImportError:  # pragma: no cover - optional in containers where env is injected.
    load_dotenv = None


SYSTEM_PROMPT = """You are OptiBot, the customer-support bot for OptiSigns.com.
- Tone: helpful, factual, concise.
- Only answer using the uploaded docs.
- Max 5 bullet points; else link to the doc.
- Cite up to 3 "Article URL:" lines per reply.
"""

CLEANER_VERSION = "2026-07-10.2"
GEMINI_OPENAI_BASE_URL = "https://generativelanguage.googleapis.com/v1beta/openai/"
GEMINI_DEFAULT_FILE_SEARCH_STORE_DISPLAY_NAME = "optibot-help-center"
GEMINI_DEFAULT_EMBEDDING_MODEL = "models/gemini-embedding-2"
LOCAL_RAG_PROMPT = """You are OptiBot, the customer-support bot for OptiSigns.com.
- Tone: helpful, factual, concise.
- Only answer using the retrieved OptiSigns docs in the user message.
- Max 5 bullet points; else link to the doc.
- Cite up to 3 "Article URL:" lines per reply.
"""
STOP_WORDS = {
    "about",
    "after",
    "again",
    "also",
    "and",
    "are",
    "can",
    "does",
    "for",
    "from",
    "how",
    "into",
    "optisigns",
    "the",
    "this",
    "that",
    "to",
    "use",
    "what",
    "when",
    "where",
    "with",
    "you",
    "your",
}


@dataclass
class Config:
    base_url: str
    locale: str
    article_limit: int
    output_dir: Path
    chunk_dir: Path
    state_file: Path
    log_dir: Path
    chunk_max_words: int
    upload_enabled: bool
    upload_provider: str
    openai_api_key: str | None
    openai_base_url: str | None
    vector_store_id: str | None
    vector_store_name: str
    assistant_id: str | None
    timeout: int
    chat_provider: str = "gemini"
    gemini_api_key: str | None = None
    gemini_base_url: str | None = None
    gemini_model: str = "gemini-3.1-flash-lite"
    gemini_file_search_store_name: str | None = None
    gemini_file_search_store_display_name: str = GEMINI_DEFAULT_FILE_SEARCH_STORE_DISPLAY_NAME
    gemini_embedding_model: str = GEMINI_DEFAULT_EMBEDDING_MODEL
    openai_chat_model: str = "gpt-4o-mini"


def env_bool(name: str, default: bool) -> bool:
    raw = os.getenv(name)
    if raw is None:
        return default
    return raw.strip().lower() in {"1", "true", "yes", "on"}


def env_value(name: str) -> str | None:
    raw = os.getenv(name)
    if raw is None:
        return None
    value = raw.strip()
    return value or None


def load_config(args: argparse.Namespace) -> Config:
    if load_dotenv:
        load_dotenv()

    upload_enabled = args.upload or env_bool("UPLOAD_ENABLED", False)
    if args.skip_upload:
        upload_enabled = False

    return Config(
        base_url=os.getenv("OPTISIGNS_HELP_CENTER_URL", "https://support.optisigns.com").rstrip("/"),
        locale=os.getenv("ZENDESK_LOCALE", "en-us"),
        article_limit=args.limit or int(os.getenv("ARTICLE_LIMIT", "30")),
        output_dir=Path(os.getenv("OUTPUT_DIR", "data/articles")),
        chunk_dir=Path(os.getenv("CHUNK_DIR", "data/chunks")),
        state_file=Path(os.getenv("STATE_FILE", "state/articles.json")),
        log_dir=Path(os.getenv("LOG_DIR", "logs")),
        chunk_max_words=int(os.getenv("CHUNK_MAX_WORDS", "700")),
        upload_enabled=upload_enabled,
        upload_provider=(env_value("UPLOAD_PROVIDER") or "gemini").lower(),
        openai_api_key=env_value("OPENAI_API_KEY") or env_value("API_KEY"),
        openai_base_url=env_value("OPENAI_BASE_URL"),
        vector_store_id=env_value("OPENAI_VECTOR_STORE_ID"),
        vector_store_name=os.getenv("OPENAI_VECTOR_STORE_NAME", "optibot-help-center"),
        assistant_id=env_value("OPENAI_ASSISTANT_ID"),
        timeout=int(os.getenv("HTTP_TIMEOUT_SECONDS", "30")),
        chat_provider=(env_value("CHAT_PROVIDER") or ("gemini" if env_value("GEMINI_API_KEY") else "openai")).lower(),
        gemini_api_key=env_value("GEMINI_API_KEY"),
        gemini_base_url=env_value("GEMINI_BASE_URL"),
        gemini_model=env_value("GEMINI_MODEL") or "gemini-3.1-flash-lite",
        gemini_file_search_store_name=env_value("GEMINI_FILE_SEARCH_STORE_NAME"),
        gemini_file_search_store_display_name=(
            env_value("GEMINI_FILE_SEARCH_STORE_DISPLAY_NAME") or GEMINI_DEFAULT_FILE_SEARCH_STORE_DISPLAY_NAME
        ),
        gemini_embedding_model=env_value("GEMINI_EMBEDDING_MODEL") or GEMINI_DEFAULT_EMBEDDING_MODEL,
        openai_chat_model=env_value("OPENAI_CHAT_MODEL") or "gpt-4o-mini",
    )


def configure_logging(log_dir: Path) -> None:
    log_dir.mkdir(parents=True, exist_ok=True)
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s %(levelname)s %(message)s",
        handlers=[
            logging.StreamHandler(),
            logging.FileHandler(log_dir / "run.log", encoding="utf-8"),
        ],
    )


def read_json(path: Path, fallback: dict[str, Any]) -> dict[str, Any]:
    if not path.exists():
        return fallback
    return json.loads(path.read_text(encoding="utf-8"))


def write_json(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")


def request_json(session: requests.Session, url: str, *, params: dict[str, Any] | None, timeout: int) -> dict[str, Any]:
    for attempt in range(1, 4):
        response = session.get(url, params=params, timeout=timeout)
        if response.status_code in {429, 500, 502, 503, 504} and attempt < 3:
            sleep_seconds = attempt * 2
            logging.warning("Retrying %s after HTTP %s in %ss", url, response.status_code, sleep_seconds)
            time.sleep(sleep_seconds)
            continue
        response.raise_for_status()
        return response.json()
    raise RuntimeError(f"Failed to fetch {url}")


def fetch_articles(config: Config) -> list[dict[str, Any]]:
    session = requests.Session()
    session.headers.update(
        {
            "Accept": "application/json",
            "User-Agent": "optibot-mini-clone/1.0 (+take-home-indexer)",
        }
    )

    url = f"{config.base_url}/api/v2/help_center/{config.locale}/articles.json"
    params: dict[str, Any] | None = {
        "per_page": 100,
        "sort_by": "updated_at",
        "sort_order": "desc",
    }
    articles: list[dict[str, Any]] = []

    while url and len(articles) < config.article_limit:
        payload = request_json(session, url, params=params, timeout=config.timeout)
        params = None
        for article in payload.get("articles", []):
            if article.get("draft") or not article.get("body"):
                continue
            articles.append(article)
            if len(articles) >= config.article_limit:
                break
        url = payload.get("next_page")

    logging.info("Fetched %s articles from Zendesk", len(articles))
    return articles


def slugify(value: str, fallback: str) -> str:
    value = re.sub(r"[^a-zA-Z0-9]+", "-", value.lower()).strip("-")
    value = re.sub(r"-{2,}", "-", value)
    return (value[:80].strip("-") or fallback).lower()


def rewrite_support_link(href: str, article_url: str, base_url: str) -> str:
    absolute = urljoin(article_url, href)
    parsed = urlparse(absolute)
    base = urlparse(base_url)
    if parsed.netloc and parsed.netloc.lower() == base.netloc.lower():
        relative = parsed.path
        if parsed.query:
            relative += f"?{parsed.query}"
        if parsed.fragment:
            relative += f"#{parsed.fragment}"
        return relative
    return href


def has_adjacent_anchor(anchor: Any) -> bool:
    for attr in ("previous_sibling", "next_sibling"):
        sibling = getattr(anchor, attr)
        while isinstance(sibling, NavigableString) and not str(sibling).strip():
            sibling = getattr(sibling, attr)
        if getattr(sibling, "name", None) == "a":
            return True
    return False


def is_urlish_fragment(value: str) -> bool:
    return bool(value) and len(value) <= 40 and bool(re.fullmatch(r"[A-Za-z0-9:/?&=._%#-]+", value))


def html_to_markdown(body_html: str, article_url: str, base_url: str) -> str:
    soup = BeautifulSoup(body_html or "", "html.parser")
    for node in soup(["script", "style", "noscript", "iframe"]):
        node.decompose()
    for anchor in soup.find_all("a", href=True):
        if has_adjacent_anchor(anchor) and is_urlish_fragment(anchor.get_text("", strip=True)):
            anchor.replace_with(anchor.get_text("", strip=True))
            continue
        anchor["href"] = rewrite_support_link(anchor["href"], article_url, base_url)

    converter = html2text.HTML2Text()
    converter.body_width = 0
    converter.ignore_images = False
    converter.ignore_emphasis = False
    converter.protect_links = True
    converter.unicode_snob = True
    converter.mark_code = True
    markdown = converter.handle(str(soup))
    markdown = re.sub(r"\n{3,}", "\n\n", markdown).strip()
    return markdown


def article_hash(article: dict[str, Any]) -> str:
    raw = json.dumps(
        {
            "id": article.get("id"),
            "cleaner_version": CLEANER_VERSION,
            "title": article.get("title"),
            "updated_at": article.get("updated_at"),
            "body": article.get("body"),
            "html_url": article.get("html_url"),
        },
        sort_keys=True,
        ensure_ascii=False,
    )
    return hashlib.sha256(raw.encode("utf-8")).hexdigest()


def article_markdown(article: dict[str, Any], body_markdown: str) -> str:
    title = article.get("title") or f"Article {article.get('id')}"
    html_url = article.get("html_url") or ""
    updated_at = article.get("updated_at") or ""
    return (
        "---\n"
        f"id: {article.get('id')}\n"
        f'title: "{str(title).replace(chr(34), chr(39))}"\n'
        f"article_url: {html_url}\n"
        f"updated_at: {updated_at}\n"
        f"locale: {article.get('locale', '')}\n"
        "---\n\n"
        f"# {title}\n\n"
        f"Article URL: {html_url}\n"
        f"Last updated: {updated_at}\n\n"
        f"{body_markdown.strip()}\n"
    )


def split_into_chunks(markdown: str, max_words: int) -> list[str]:
    paragraphs = [part.strip() for part in re.split(r"\n{2,}", markdown) if part.strip()]
    chunks: list[str] = []
    current: list[str] = []
    current_words = 0

    for paragraph in paragraphs:
        words = paragraph.split()
        if len(words) > max_words:
            if current:
                chunks.append("\n\n".join(current))
                current = []
                current_words = 0
            for start in range(0, len(words), max_words):
                chunks.append(" ".join(words[start : start + max_words]))
            continue
        if current and current_words + len(words) > max_words:
            chunks.append("\n\n".join(current))
            current = []
            current_words = 0
        current.append(paragraph)
        current_words += len(words)

    if current:
        chunks.append("\n\n".join(current))
    return chunks or [markdown]


def write_article_and_chunks(
    config: Config,
    article: dict[str, Any],
    content_hash: str,
) -> tuple[Path, list[Path], str]:
    article_id = str(article.get("id"))
    title = article.get("title") or f"Article {article_id}"
    slug = f"{slugify(title, article_id)}-{article_id}"
    article_url = article.get("html_url") or f"{config.base_url}/hc/{config.locale}/articles/{article_id}"

    body_markdown = html_to_markdown(article.get("body") or "", article_url, config.base_url)
    full_markdown = article_markdown(article, body_markdown)

    config.output_dir.mkdir(parents=True, exist_ok=True)
    config.chunk_dir.mkdir(parents=True, exist_ok=True)

    article_path = config.output_dir / f"{slug}.md"
    article_path.write_text(full_markdown, encoding="utf-8")

    for old_chunk in config.chunk_dir.glob(f"{slug}__chunk-*.md"):
        old_chunk.unlink()

    chunks = split_into_chunks(full_markdown, config.chunk_max_words)
    chunk_paths: list[Path] = []
    total = len(chunks)
    for index, chunk in enumerate(chunks, start=1):
        chunk_path = config.chunk_dir / f"{slug}__chunk-{index:03d}.md"
        chunk_payload = (
            "---\n"
            f"source_id: {article_id}\n"
            f"source_title: \"{str(title).replace(chr(34), chr(39))}\"\n"
            f"source_url: {article_url}\n"
            f"source_hash: {content_hash}\n"
            f"chunk_index: {index}\n"
            f"chunk_total: {total}\n"
            "---\n\n"
            f"# {title}\n\n"
            f"Article URL: {article_url}\n"
            f"Chunk: {index}/{total}\n\n"
            f"{chunk.strip()}\n"
        )
        chunk_path.write_text(chunk_payload, encoding="utf-8")
        chunk_paths.append(chunk_path)

    return article_path, chunk_paths, slug


def classify_and_write(
    config: Config,
    articles: list[dict[str, Any]],
    state: dict[str, Any],
) -> tuple[list[dict[str, Any]], dict[str, int]]:
    state.setdefault("articles", {})
    counts = {"added": 0, "updated": 0, "skipped": 0}
    changed: list[dict[str, Any]] = []

    for article in articles:
        article_id = str(article.get("id"))
        content_hash = article_hash(article)
        previous = state["articles"].get(article_id)
        previous_hash = previous.get("hash") if previous else None
        old_chunk_paths = [Path(path) for path in (previous or {}).get("chunk_paths", [])]
        missing_local_files = previous is not None and any(not path.exists() for path in old_chunk_paths)
        missing_upload_files = config.upload_enabled and previous is not None and (
            (
                config.upload_provider == "openai"
                and not (previous or {}).get("vector_file_ids")
            )
            or (
                config.upload_provider == "gemini"
                and not (previous or {}).get("gemini_file_search_store_name")
            )
        )

        if previous_hash is None:
            status = "added"
        elif previous_hash != content_hash or missing_local_files or missing_upload_files:
            status = "updated"
        else:
            status = "skipped"

        if status == "skipped":
            previous["article_path"] = Path(previous["article_path"]).as_posix()
            previous["chunk_paths"] = [Path(path).as_posix() for path in previous.get("chunk_paths", [])]
            counts["skipped"] += 1
            continue

        article_path, chunk_paths, slug = write_article_and_chunks(config, article, content_hash)
        prior_file_ids = (previous or {}).get("vector_file_ids", [])
        state["articles"][article_id] = {
            "hash": content_hash,
            "title": article.get("title"),
            "html_url": article.get("html_url"),
            "updated_at": article.get("updated_at"),
            "slug": slug,
            "article_path": article_path.as_posix(),
            "chunk_paths": [path.as_posix() for path in chunk_paths],
            "vector_file_ids": prior_file_ids,
        }
        changed.append(
            {
                "id": article_id,
                "status": status,
                "title": article.get("title"),
                "chunk_paths": chunk_paths,
                "old_vector_file_ids": prior_file_ids,
            }
        )
        counts[status] += 1

    return changed, counts


def plain_object(value: Any) -> Any:
    if isinstance(value, dict):
        return {key: plain_object(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [plain_object(item) for item in value]
    if hasattr(value, "model_dump"):
        return plain_object(value.model_dump())
    if hasattr(value, "__dict__"):
        return plain_object({key: item for key, item in value.__dict__.items() if not key.startswith("_")})
    return value


def ensure_vector_store(client: Any, config: Config, state: dict[str, Any]) -> str:
    vector_store_id = config.vector_store_id or state.get("vector_store_id")
    if vector_store_id:
        return vector_store_id

    vector_store = client.vector_stores.create(name=config.vector_store_name)
    vector_store_id = vector_store.id
    state["vector_store_id"] = vector_store_id
    logging.info("Created vector store %s", vector_store_id)
    return vector_store_id


def delete_old_vector_files(client: Any, vector_store_id: str, file_ids: list[str]) -> None:
    for file_id in file_ids:
        try:
            client.vector_stores.files.delete(vector_store_id=vector_store_id, file_id=file_id)
        except Exception as exc:  # noqa: BLE001 - best effort cleanup only.
            logging.warning("Could not detach old vector file %s: %s", file_id, exc)
        try:
            client.files.delete(file_id)
        except Exception as exc:  # noqa: BLE001 - best effort cleanup only.
            logging.warning("Could not delete old OpenAI file %s: %s", file_id, exc)


def create_file_batch(client: Any, vector_store_id: str, file_ids: list[str]) -> Any:
    batches = client.vector_stores.file_batches
    if hasattr(batches, "create_and_poll"):
        return batches.create_and_poll(vector_store_id=vector_store_id, file_ids=file_ids)

    batch = batches.create(vector_store_id=vector_store_id, file_ids=file_ids)
    terminal = {"completed", "failed", "cancelled", "expired"}
    while getattr(batch, "status", None) not in terminal:
        time.sleep(2)
        batch = batches.retrieve(vector_store_id=vector_store_id, batch_id=batch.id)
    return batch


def upload_openai_changed_chunks(config: Config, state: dict[str, Any], changed: list[dict[str, Any]]) -> dict[str, Any]:
    if not config.upload_enabled:
        return {"enabled": False, "uploaded_files": 0, "embedded_chunks": 0}
    if not config.openai_api_key:
        raise RuntimeError("OPENAI_API_KEY or API_KEY is required when upload is enabled.")
    if not changed:
        return {"enabled": True, "uploaded_files": 0, "embedded_chunks": 0}

    from openai import OpenAI

    client_kwargs: dict[str, Any] = {
        "api_key": config.openai_api_key,
        "base_url": config.openai_base_url or "https://api.openai.com/v1",
    }
    client = OpenAI(**client_kwargs)
    vector_store_id = ensure_vector_store(client, config, state)

    all_file_ids: list[str] = []
    file_ids_by_article: dict[str, list[str]] = {}
    for item in changed:
        delete_old_vector_files(client, vector_store_id, item.get("old_vector_file_ids") or [])
        new_file_ids: list[str] = []
        for chunk_path in item["chunk_paths"]:
            with Path(chunk_path).open("rb") as file_handle:
                uploaded = client.files.create(file=file_handle, purpose="assistants")
            new_file_ids.append(uploaded.id)
            all_file_ids.append(uploaded.id)
        file_ids_by_article[item["id"]] = new_file_ids

    batch = create_file_batch(client, vector_store_id, all_file_ids)
    batch_status = getattr(batch, "status", None)
    if batch_status not in {None, "completed"}:
        raise RuntimeError(f"OpenAI vector store file batch ended with status: {batch_status}")

    for article_id, file_ids in file_ids_by_article.items():
        state["articles"][article_id]["vector_file_ids"] = file_ids

    if config.assistant_id:
        client.beta.assistants.update(
            assistant_id=config.assistant_id,
            instructions=SYSTEM_PROMPT,
            tools=[{"type": "file_search"}],
            tool_resources={"file_search": {"vector_store_ids": [vector_store_id]}},
        )
        logging.info("Attached vector store %s to assistant %s", vector_store_id, config.assistant_id)

    return {
        "enabled": True,
        "vector_store_id": vector_store_id,
        "file_batch_id": getattr(batch, "id", None),
        "file_batch_status": batch_status,
        "file_counts": plain_object(getattr(batch, "file_counts", {})),
        "uploaded_files": len(all_file_ids),
        "embedded_chunks": len(all_file_ids),
    }


def gemini_client(config: Config) -> Any:
    if not config.gemini_api_key:
        raise RuntimeError("GEMINI_API_KEY is required when UPLOAD_PROVIDER=gemini or CHAT_PROVIDER=gemini.")
    from google import genai

    return genai.Client(api_key=config.gemini_api_key)


def poll_gemini_operation(client: Any, operation: Any, *, timeout_seconds: int = 900) -> Any:
    deadline = time.time() + timeout_seconds
    while not getattr(operation, "done", False):
        if time.time() >= deadline:
            raise TimeoutError(f"Gemini operation did not finish within {timeout_seconds}s: {operation}")
        time.sleep(5)
        operation = client.operations.get(operation)
    if getattr(operation, "error", None):
        raise RuntimeError(f"Gemini operation failed: {operation.error}")
    return operation


def create_gemini_file_search_store(client: Any, config: Config) -> Any:
    store = client.file_search_stores.create(
        config={
            "display_name": config.gemini_file_search_store_display_name,
            "embedding_model": config.gemini_embedding_model,
        }
    )
    logging.info("Created Gemini File Search store %s", store.name)
    return store


def delete_gemini_file_search_store(client: Any, store_name: str) -> None:
    try:
        client.file_search_stores.delete(name=store_name, config={"force": True})
        logging.info("Deleted old Gemini File Search store %s", store_name)
    except Exception as exc:  # noqa: BLE001 - best effort cleanup only.
        logging.warning("Could not delete old Gemini File Search store %s: %s", store_name, exc)


def all_state_chunk_paths(state: dict[str, Any]) -> list[Path]:
    chunk_paths: list[Path] = []
    seen: set[str] = set()
    for article in state.get("articles", {}).values():
        for raw_path in article.get("chunk_paths", []):
            path = Path(raw_path)
            key = path.as_posix()
            if key in seen or not path.exists():
                continue
            seen.add(key)
            chunk_paths.append(path)
    return chunk_paths


def upload_gemini_changed_chunks(config: Config, state: dict[str, Any], changed: list[dict[str, Any]]) -> dict[str, Any]:
    if not config.upload_enabled:
        return {"enabled": False, "uploaded_files": 0, "embedded_chunks": 0}

    store_name = config.gemini_file_search_store_name or state.get("gemini_file_search_store_name")
    if not changed and store_name:
        return {
            "enabled": True,
            "provider": "gemini",
            "file_search_store_name": store_name,
            "uploaded_files": 0,
            "embedded_chunks": 0,
        }

    chunk_paths = all_state_chunk_paths(state)
    if not chunk_paths:
        return {"enabled": True, "provider": "gemini", "uploaded_files": 0, "embedded_chunks": 0}

    client = gemini_client(config)
    if store_name:
        delete_gemini_file_search_store(client, store_name)

    store = create_gemini_file_search_store(client, config)
    store_name = store.name
    state["gemini_file_search_store_name"] = store_name
    state["gemini_file_search_store_display_name"] = config.gemini_file_search_store_display_name
    state["gemini_embedding_model"] = config.gemini_embedding_model

    uploaded_files: list[str] = []
    for chunk_path in chunk_paths:
        operation = client.file_search_stores.upload_to_file_search_store(
            file_search_store_name=store_name,
            file=chunk_path.as_posix(),
            config={"display_name": chunk_path.name},
        )
        poll_gemini_operation(client, operation)
        uploaded_files.append(chunk_path.as_posix())
        logging.info("Uploaded %s to Gemini File Search store", chunk_path.as_posix())

    for article in state.get("articles", {}).values():
        article["gemini_file_search_store_name"] = store_name
        article["gemini_file_search_uploaded_chunks"] = [
            Path(path).as_posix()
            for path in article.get("chunk_paths", [])
        ]

    return {
        "enabled": True,
        "provider": "gemini",
        "file_search_store_name": store_name,
        "embedding_model": config.gemini_embedding_model,
        "uploaded_files": len(uploaded_files),
        "embedded_chunks": len(uploaded_files),
    }


def upload_changed_chunks(config: Config, state: dict[str, Any], changed: list[dict[str, Any]]) -> dict[str, Any]:
    if config.upload_provider == "gemini":
        return upload_gemini_changed_chunks(config, state, changed)
    if config.upload_provider == "openai":
        return upload_openai_changed_chunks(config, state, changed)
    raise RuntimeError(f"Unsupported UPLOAD_PROVIDER: {config.upload_provider}")


def query_terms(query: str) -> list[str]:
    terms = re.findall(r"[a-z0-9]+", query.lower())
    return [term for term in terms if len(term) > 2 and term not in STOP_WORDS]


def search_local_chunks(chunk_dir: Path, query: str, *, limit: int = 3) -> list[dict[str, Any]]:
    terms = query_terms(query)
    if not terms or not chunk_dir.exists():
        return []

    results: list[dict[str, Any]] = []
    for path in sorted(chunk_dir.glob("*.md")):
        text = path.read_text(encoding="utf-8")
        haystack = text.lower()
        score = sum(haystack.count(term) for term in terms)
        if query.lower() in haystack:
            score += 50
        if score <= 0:
            continue
        results.append({"path": path, "score": score, "text": text})

    return sorted(results, key=lambda item: (-item["score"], item["path"].as_posix()))[:limit]


def format_retrieved_context(results: list[dict[str, Any]], *, max_chars: int = 14000) -> str:
    parts: list[str] = []
    remaining = max_chars
    for index, result in enumerate(results, start=1):
        header = f"[{index}] Source file: {Path(result['path']).as_posix()}\n"
        text = result["text"].strip()
        payload = f"{header}{text}\n"
        if len(payload) > remaining:
            payload = payload[:remaining].rstrip()
        if not payload:
            break
        parts.append(payload)
        remaining -= len(payload)
        if remaining <= 0:
            break
    return "\n\n---\n\n".join(parts)


def chat_client_settings(config: Config) -> tuple[str, str, str]:
    provider = config.chat_provider.lower()
    if provider == "gemini":
        if not config.gemini_api_key:
            raise RuntimeError("GEMINI_API_KEY is required when CHAT_PROVIDER=gemini.")
        return config.gemini_api_key, config.gemini_base_url or GEMINI_OPENAI_BASE_URL, config.gemini_model
    if provider == "openai":
        if not config.openai_api_key:
            raise RuntimeError("OPENAI_API_KEY or API_KEY is required when CHAT_PROVIDER=openai.")
        return config.openai_api_key, config.openai_base_url or "https://api.openai.com/v1", config.openai_chat_model
    raise RuntimeError(f"Unsupported CHAT_PROVIDER: {config.chat_provider}")


def resolve_gemini_file_search_store_name(config: Config) -> str | None:
    if config.gemini_file_search_store_name:
        return config.gemini_file_search_store_name
    state = read_json(config.state_file, {})
    store_name = state.get("gemini_file_search_store_name")
    return store_name if isinstance(store_name, str) and store_name else None


def extract_gemini_interaction_response(interaction: Any) -> tuple[str, list[dict[str, Any]]]:
    payload = plain_object(interaction)
    answer_parts: list[str] = []
    citations: list[dict[str, Any]] = []
    seen_citations: set[tuple[str | None, str | None]] = set()
    for step in payload.get("steps", []):
        if step.get("type") != "model_output":
            continue
        for content in step.get("content", []):
            if content.get("type") != "text":
                continue
            text = content.get("text")
            if text:
                answer_parts.append(text)
            for annotation in content.get("annotations") or []:
                if annotation.get("type") == "file_citation":
                    key = (annotation.get("file_name"), annotation.get("source"))
                    if key in seen_citations:
                        continue
                    seen_citations.add(key)
                    source = annotation.get("source") or ""
                    citation = {
                        "type": "file_citation",
                        "file_name": annotation.get("file_name"),
                        "document_uri": annotation.get("document_uri"),
                    }
                    if annotation.get("page_number"):
                        citation["page_number"] = annotation.get("page_number")
                    if source:
                        citation["source_excerpt"] = source[:500].strip()
                    citations.append(citation)
    return "\n".join(answer_parts).strip(), citations


def answer_question_with_gemini_file_search(config: Config, question: str, store_name: str) -> dict[str, Any]:
    client = gemini_client(config)
    interaction = client.interactions.create(
        model=config.gemini_model,
        input=(
            f"{LOCAL_RAG_PROMPT}\n\n"
            f"Question: {question}"
        ),
        tools=[
            {
                "type": "file_search",
                "file_search_store_names": [store_name],
            }
        ],
    )
    answer, citations = extract_gemini_interaction_response(interaction)
    return {
        "provider": "gemini",
        "retrieval": "gemini_file_search",
        "model": config.gemini_model,
        "file_search_store_name": store_name,
        "answer": answer,
        "sources": citations,
    }


def answer_question(config: Config, question: str, *, top_k: int) -> dict[str, Any]:
    if config.chat_provider == "gemini":
        store_name = resolve_gemini_file_search_store_name(config)
        if store_name:
            return answer_question_with_gemini_file_search(config, question, store_name)

    results = search_local_chunks(config.chunk_dir, question, limit=top_k)
    if not results:
        raise RuntimeError(f"No local chunks matched the question in {config.chunk_dir}. Run the indexer first.")

    from openai import OpenAI

    api_key, base_url, model = chat_client_settings(config)
    client = OpenAI(api_key=api_key, base_url=base_url)
    context = format_retrieved_context(results)
    response = client.chat.completions.create(
        model=model,
        temperature=0.2,
        messages=[
            {"role": "system", "content": LOCAL_RAG_PROMPT},
            {
                "role": "user",
                "content": (
                    "Use only these retrieved OptiSigns docs to answer.\n\n"
                    f"{context}\n\n"
                    f"Question: {question}"
                ),
            },
        ],
    )
    answer = response.choices[0].message.content or ""
    return {
        "provider": config.chat_provider,
        "model": model,
        "answer": answer.strip(),
        "sources": [
            {
                "path": Path(result["path"]).as_posix(),
                "score": result["score"],
            }
            for result in results
        ],
    }


def run(config: Config) -> dict[str, Any]:
    configure_logging(config.log_dir)
    for path in [config.output_dir, config.chunk_dir, config.state_file.parent, config.log_dir]:
        path.mkdir(parents=True, exist_ok=True)

    started_at = datetime.now(timezone.utc).isoformat()
    state = read_json(config.state_file, {"articles": {}})
    articles = fetch_articles(config)
    changed, counts = classify_and_write(config, articles, state)
    upload_result = upload_changed_chunks(config, state, changed)
    write_json(config.state_file, state)

    run_log = {
        "started_at": started_at,
        "finished_at": datetime.now(timezone.utc).isoformat(),
        "source": f"{config.base_url}/api/v2/help_center/{config.locale}/articles.json",
        "article_limit": config.article_limit,
        "fetched": len(articles),
        "added": counts["added"],
        "updated": counts["updated"],
        "skipped": counts["skipped"],
        "changed_article_ids": [item["id"] for item in changed],
        "chunking": {
            "strategy": "paragraph-aware fixed word target",
            "max_words": config.chunk_max_words,
        },
        "upload": upload_result,
    }
    timestamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    write_json(config.log_dir / f"run-{timestamp}.json", run_log)
    write_json(config.log_dir / "latest.json", run_log)
    logging.info(
        "Done: fetched=%s added=%s updated=%s skipped=%s uploaded_chunks=%s",
        run_log["fetched"],
        run_log["added"],
        run_log["updated"],
        run_log["skipped"],
        upload_result.get("embedded_chunks", 0),
    )
    return run_log


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Scrape OptiSigns support docs and sync changed chunks to a RAG store.")
    parser.add_argument("--limit", type=int, default=None, help="Maximum articles to fetch. Defaults to ARTICLE_LIMIT.")
    parser.add_argument("--upload", action="store_true", help="Upload changed chunks to the configured RAG store.")
    parser.add_argument("--skip-upload", action="store_true", help="Scrape and write markdown without uploading.")
    parser.add_argument("--ask", default=None, help="Ask a question using local chunks and the configured chat provider.")
    parser.add_argument("--top-k", type=int, default=3, help="Number of local chunks to retrieve for --ask.")
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()
    config = load_config(args)
    if args.ask:
        result = answer_question(config, args.ask, top_k=args.top_k)
    else:
        result = run(config)
    print(json.dumps(result, ensure_ascii=False, indent=2))
