from __future__ import annotations

import argparse
from pathlib import Path
from typing import Any

from fastapi import FastAPI, HTTPException
from fastapi.responses import FileResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel, Field

from main import answer_question, load_config, read_json


WEB_DIR = Path(__file__).parent / "web"


class ChatRequest(BaseModel):
    question: str = Field(..., min_length=1, max_length=1000)


def runtime_config() -> Any:
    args = argparse.Namespace(upload=False, skip_upload=True, limit=None)
    return load_config(args)


app = FastAPI(title="OptiBot Chat UI")
app.mount("/assets", StaticFiles(directory=WEB_DIR), name="assets")


@app.get("/")
def index() -> FileResponse:
    return FileResponse(WEB_DIR / "index.html")


@app.get("/api/status")
def status() -> dict[str, Any]:
    config = runtime_config()
    state = read_json(config.state_file, {})
    articles = state.get("articles", {})
    return {
        "provider": config.chat_provider,
        "model": config.gemini_model if config.chat_provider == "gemini" else config.openai_chat_model,
        "retrieval": "gemini_file_search" if state.get("gemini_file_search_store_name") else "local_chunks",
        "gemini_file_search_store_name": state.get("gemini_file_search_store_name"),
        "article_count": len(articles),
        "chunk_count": sum(len(article.get("chunk_paths", [])) for article in articles.values()),
    }


@app.post("/api/chat")
def chat(payload: ChatRequest) -> dict[str, Any]:
    config = runtime_config()
    try:
        return answer_question(config, payload.question.strip(), top_k=3)
    except Exception as exc:  # noqa: BLE001 - API boundary returns a readable error.
        raise HTTPException(status_code=500, detail=str(exc)) from exc
