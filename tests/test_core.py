import os
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

from main import (
    Config,
    article_hash,
    classify_and_write,
    env_value,
    html_to_markdown,
    rewrite_support_link,
    search_local_chunks,
    slugify,
    split_into_chunks,
)


class CoreTest(unittest.TestCase):
    def test_slugify_keeps_stable_suffix_friendly_text(self):
        self.assertEqual(slugify("How to Use YouTube with OptiSigns!", "fallback"), "how-to-use-youtube-with-optisigns")

    def test_rewrite_support_link_preserves_relative_support_links(self):
        result = rewrite_support_link(
            "https://support.optisigns.com/hc/en-us/articles/123?foo=bar#top",
            "https://support.optisigns.com/hc/en-us/articles/999",
            "https://support.optisigns.com",
        )
        self.assertEqual(result, "/hc/en-us/articles/123?foo=bar#top")

    def test_split_into_chunks_respects_word_target(self):
        markdown = "\n\n".join(["one two three four five"] * 5)
        chunks = split_into_chunks(markdown, max_words=10)
        self.assertGreater(len(chunks), 1)
        self.assertTrue(all(len(chunk.split()) <= 10 for chunk in chunks))

    def test_html_to_markdown_unwraps_fragmented_url_anchors(self):
        html = (
            '<p>Example <a href="https://example.com/full">https://</a>'
            '<a href="https://example.com/full">example.com</a></p>'
        )
        markdown = html_to_markdown(
            html,
            "https://support.optisigns.com/hc/en-us/articles/1",
            "https://support.optisigns.com",
        )
        self.assertIn("https://example.com", markdown)
        self.assertNotIn("[https://]", markdown)

    def test_env_value_treats_blank_values_as_missing(self):
        with patch.dict(os.environ, {"BLANK_ENV": "", "SPACED_ENV": "  value  "}, clear=False):
            self.assertIsNone(env_value("BLANK_ENV"))
            self.assertEqual(env_value("SPACED_ENV"), "value")

    def test_search_local_chunks_ranks_matching_docs_first(self):
        with tempfile.TemporaryDirectory() as tmp_dir:
            tmp = Path(tmp_dir)
            (tmp / "youtube.md").write_text(
                "Article URL: https://example.com/youtube\nAdd a YouTube video in OptiSigns.",
                encoding="utf-8",
            )
            (tmp / "billing.md").write_text(
                "Article URL: https://example.com/billing\nChange billing plan.",
                encoding="utf-8",
            )

            results = search_local_chunks(tmp, "How do I add a YouTube video?", limit=1)

            self.assertEqual(results[0]["path"].name, "youtube.md")

    def test_upload_enabled_reprocesses_articles_missing_vector_files(self):
        article = {
            "id": 123,
            "title": "Upload Me",
            "body": "<p>Fresh enough content.</p>",
            "html_url": "https://support.optisigns.com/hc/en-us/articles/123",
            "updated_at": "2026-07-10T00:00:00Z",
            "locale": "en-us",
        }
        with tempfile.TemporaryDirectory() as tmp_dir:
            tmp = Path(tmp_dir)
            config = Config(
                base_url="https://support.optisigns.com",
                locale="en-us",
                article_limit=1,
                output_dir=tmp / "articles",
                chunk_dir=tmp / "chunks",
                state_file=tmp / "state" / "articles.json",
                log_dir=tmp / "logs",
                chunk_max_words=700,
                upload_enabled=True,
                upload_provider="openai",
                openai_api_key="test-key",
                openai_base_url=None,
                vector_store_id=None,
                vector_store_name="test-store",
                assistant_id=None,
                timeout=30,
            )
            state = {
                "articles": {
                    "123": {
                        "hash": article_hash(article),
                        "article_path": "missing.md",
                        "chunk_paths": [],
                        "vector_file_ids": [],
                    }
                }
            }

            changed, counts = classify_and_write(config, [article], state)

            self.assertEqual(counts["updated"], 1)
            self.assertEqual(changed[0]["id"], "123")
            self.assertEqual(changed[0]["status"], "updated")
            self.assertTrue(state["articles"]["123"]["chunk_paths"])

    def test_gemini_upload_reprocesses_articles_missing_store_marker(self):
        article = {
            "id": 123,
            "title": "Upload Me",
            "body": "<p>Fresh enough content.</p>",
            "html_url": "https://support.optisigns.com/hc/en-us/articles/123",
            "updated_at": "2026-07-10T00:00:00Z",
            "locale": "en-us",
        }
        with tempfile.TemporaryDirectory() as tmp_dir:
            tmp = Path(tmp_dir)
            config = Config(
                base_url="https://support.optisigns.com",
                locale="en-us",
                article_limit=1,
                output_dir=tmp / "articles",
                chunk_dir=tmp / "chunks",
                state_file=tmp / "state" / "articles.json",
                log_dir=tmp / "logs",
                chunk_max_words=700,
                upload_enabled=True,
                upload_provider="gemini",
                openai_api_key=None,
                openai_base_url=None,
                vector_store_id=None,
                vector_store_name="test-store",
                assistant_id=None,
                timeout=30,
            )
            state = {
                "articles": {
                    "123": {
                        "hash": article_hash(article),
                        "article_path": "data/articles/upload-me-123.md",
                        "chunk_paths": ["data/chunks/upload-me-123__chunk-001.md"],
                        "vector_file_ids": ["legacy-openai-file"],
                    }
                }
            }

            changed, counts = classify_and_write(config, [article], state)

            self.assertEqual(counts["updated"], 1)
            self.assertEqual(changed[0]["status"], "updated")


if __name__ == "__main__":
    unittest.main()
