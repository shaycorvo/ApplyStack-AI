"""Deterministic document ingestion for profile onboarding."""

from __future__ import annotations

import json
import sqlite3
from datetime import datetime, timezone
from pathlib import Path
from typing import Callable, Optional
from urllib.parse import urlparse

import requests
from bs4 import BeautifulSoup
from docx import Document
from pypdf import PdfReader


SUPPORTED_DOCUMENT_SUFFIXES = {".pdf", ".docx"}


def save_uploaded_document(upload_directory: str | Path, file_name: str, contents: bytes) -> Path:
    """Persist an uploaded source under a controlled directory using its basename only."""
    safe_name = Path(file_name).name
    if not safe_name or Path(safe_name).suffix.lower() not in SUPPORTED_DOCUMENT_SUFFIXES:
        raise ValueError("Only PDF and DOCX documents can be uploaded")
    destination_directory = Path(upload_directory)
    destination_directory.mkdir(parents=True, exist_ok=True)
    destination = destination_directory / safe_name
    destination.write_bytes(contents)
    return destination


def apply_profile_draft(profile_path: str | Path, draft: dict) -> dict:
    """Apply a reviewed draft without discarding approved profile sections."""
    path = Path(profile_path)
    existing_profile = json.loads(path.read_text(encoding="utf-8")) if path.exists() else {}
    _deep_merge(existing_profile, draft)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(existing_profile, indent=2), encoding="utf-8")
    return existing_profile


def extract_url_text(url: str, fetch_html: Optional[Callable[[str], str]] = None) -> str:
    """Fetch readable public profile content without executing page scripts."""
    parsed_url = urlparse(url)
    if parsed_url.scheme not in {"http", "https"} or not parsed_url.netloc:
        raise ValueError("Profile source URL must be a valid HTTP(S) URL")
    if fetch_html is None:
        response = requests.get(
            url,
            timeout=15,
            headers={"User-Agent": "ApplyAgentProfileOnboarding/1.0"},
        )
        response.raise_for_status()
        if "html" not in response.headers.get("Content-Type", "").lower():
            raise ValueError("Profile source URL did not return an HTML page")
        html = response.text
    else:
        html = fetch_html(url)
    soup = BeautifulSoup(html, "html.parser")
    for element in soup(["script", "style", "noscript", "svg", "nav", "footer"]):
        element.decompose()
    text = "\n".join(line.strip() for line in soup.get_text("\n").splitlines() if line.strip())
    if not text:
        raise ValueError("No readable text was extracted from the profile source URL")
    return text


def extract_document_text(document_path: str | Path) -> str:
    """Extract text locally from an approved resume or supporting document."""
    path = Path(document_path)
    if not path.is_file():
        raise ValueError(f"Document does not exist: {path}")
    suffix = path.suffix.lower()
    if suffix == ".pdf":
        text = "\n".join(page.extract_text() or "" for page in PdfReader(path).pages)
    elif suffix == ".docx":
        text = "\n".join(paragraph.text for paragraph in Document(path).paragraphs)
    else:
        raise ValueError(f"Unsupported document type: {suffix or 'no extension'}")
    if not text.strip():
        raise ValueError(f"No readable text was extracted from: {path.name}")
    return text.strip()


class OnboardingStore:
    """Persist onboarding source data before it becomes an approved profile."""

    def __init__(self, database_path: str | Path = "data/profile/onboarding.sqlite3"):
        self.database_path = Path(database_path)
        self.database_path.parent.mkdir(parents=True, exist_ok=True)
        self._initialize_database()

    def create_onboarding(self) -> dict:
        now = self._timestamp()
        with self._connect() as connection:
            cursor = connection.execute(
                "INSERT INTO onboarding_runs (created_at, updated_at, status) VALUES (?, ?, ?)",
                (now, now, "collecting_sources"),
            )
        return self.get_onboarding(cursor.lastrowid)

    def get_onboarding(self, onboarding_id: int) -> dict:
        with self._connect() as connection:
            row = connection.execute("SELECT * FROM onboarding_runs WHERE id = ?", (onboarding_id,)).fetchone()
        if row is None:
            raise ValueError(f"Onboarding run {onboarding_id} does not exist")
        return dict(row)

    def add_source(self, onboarding_id: int, document_path: str | Path, source_type: str) -> dict:
        self.get_onboarding(onboarding_id)
        path = Path(document_path)
        if not path.is_file():
            raise ValueError(f"Document does not exist: {path}")
        now = self._timestamp()
        with self._connect() as connection:
            cursor = connection.execute(
                """INSERT INTO onboarding_sources
                   (onboarding_id, source_type, path, file_name, extracted_text, created_at, updated_at)
                   VALUES (?, ?, ?, ?, NULL, ?, ?)""",
                (onboarding_id, source_type, str(path), path.name, now, now),
            )
            connection.execute(
                "UPDATE onboarding_runs SET updated_at = ? WHERE id = ?", (now, onboarding_id)
            )
        return self.get_source(cursor.lastrowid)

    def add_url_source(self, onboarding_id: int, source_type: str, url: str) -> dict:
        """Record a public profile URL for later, consented content extraction."""
        self.get_onboarding(onboarding_id)
        parsed_url = urlparse(url.strip())
        if parsed_url.scheme not in {"http", "https"} or not parsed_url.netloc:
            raise ValueError("Profile source URL must be a valid HTTP(S) URL")
        now = self._timestamp()
        with self._connect() as connection:
            cursor = connection.execute(
                """INSERT INTO onboarding_sources
                   (onboarding_id, source_type, path, file_name, url, extracted_text, created_at, updated_at)
                   VALUES (?, ?, ?, ?, ?, NULL, ?, ?)""",
                (onboarding_id, source_type, url, parsed_url.netloc, url, now, now),
            )
            connection.execute(
                "UPDATE onboarding_runs SET updated_at = ? WHERE id = ?", (now, onboarding_id)
            )
        return self.get_source(cursor.lastrowid)

    def save_extraction(self, source_id: int, extracted_text: str) -> dict:
        if not extracted_text.strip():
            raise ValueError("Extracted text cannot be empty")
        now = self._timestamp()
        with self._connect() as connection:
            source = connection.execute("SELECT onboarding_id FROM onboarding_sources WHERE id = ?", (source_id,)).fetchone()
            if source is None:
                raise ValueError(f"Onboarding source {source_id} does not exist")
            connection.execute(
                "UPDATE onboarding_sources SET extracted_text = ?, updated_at = ? WHERE id = ?",
                (extracted_text, now, source_id),
            )
            connection.execute(
                "UPDATE onboarding_runs SET updated_at = ? WHERE id = ?", (now, source["onboarding_id"])
            )
        return self.get_source(source_id)

    def get_source(self, source_id: int) -> dict:
        with self._connect() as connection:
            row = connection.execute("SELECT * FROM onboarding_sources WHERE id = ?", (source_id,)).fetchone()
        if row is None:
            raise ValueError(f"Onboarding source {source_id} does not exist")
        source = dict(row)
        source["source_ref"] = f"source:{source_id}"
        return source

    def list_sources(self, onboarding_id: int) -> list[dict]:
        with self._connect() as connection:
            rows = connection.execute(
                "SELECT * FROM onboarding_sources WHERE onboarding_id = ? ORDER BY id", (onboarding_id,)
            ).fetchall()
        return [self._with_source_ref(dict(row)) for row in rows]

    def save_profile_draft(self, onboarding_id: int, draft: dict) -> None:
        """Persist a reviewable LLM result without modifying the approved profile."""
        self.get_onboarding(onboarding_id)
        now = self._timestamp()
        with self._connect() as connection:
            connection.execute(
                """INSERT INTO onboarding_profile_drafts (onboarding_id, draft_json, updated_at)
                   VALUES (?, ?, ?)
                   ON CONFLICT(onboarding_id) DO UPDATE SET draft_json = excluded.draft_json, updated_at = excluded.updated_at""",
                (onboarding_id, json.dumps(draft), now),
            )
            connection.execute(
                "UPDATE onboarding_runs SET status = ?, updated_at = ? WHERE id = ?",
                ("reviewing_draft", now, onboarding_id),
            )

    def get_profile_draft(self, onboarding_id: int) -> dict | None:
        with self._connect() as connection:
            row = connection.execute(
                "SELECT draft_json FROM onboarding_profile_drafts WHERE onboarding_id = ?", (onboarding_id,)
            ).fetchone()
        return json.loads(row["draft_json"]) if row else None

    def _initialize_database(self) -> None:
        with self._connect() as connection:
            connection.executescript(
                """
                CREATE TABLE IF NOT EXISTS onboarding_runs (
                    id INTEGER PRIMARY KEY,
                    created_at TEXT NOT NULL,
                    updated_at TEXT NOT NULL,
                    status TEXT NOT NULL
                );
                CREATE TABLE IF NOT EXISTS onboarding_sources (
                    id INTEGER PRIMARY KEY,
                    onboarding_id INTEGER NOT NULL REFERENCES onboarding_runs(id),
                    source_type TEXT NOT NULL,
                    path TEXT NOT NULL,
                    file_name TEXT NOT NULL,
                    url TEXT,
                    extracted_text TEXT,
                    created_at TEXT NOT NULL,
                    updated_at TEXT NOT NULL
                );
                CREATE TABLE IF NOT EXISTS onboarding_profile_drafts (
                    onboarding_id INTEGER PRIMARY KEY REFERENCES onboarding_runs(id),
                    draft_json TEXT NOT NULL,
                    updated_at TEXT NOT NULL
                );
                """
            )
            columns = {row["name"] for row in connection.execute("PRAGMA table_info(onboarding_sources)")}
            if "url" not in columns:
                connection.execute("ALTER TABLE onboarding_sources ADD COLUMN url TEXT")

    def _connect(self) -> sqlite3.Connection:
        connection = sqlite3.connect(self.database_path)
        connection.row_factory = sqlite3.Row
        connection.execute("PRAGMA foreign_keys = ON")
        return connection

    @staticmethod
    def _timestamp() -> str:
        return datetime.now(timezone.utc).isoformat()

    @staticmethod
    def _with_source_ref(source: dict) -> dict:
        source["source_ref"] = f"source:{source['id']}"
        return source


def _deep_merge(existing: dict, updates: dict) -> dict:
    for key, value in updates.items():
        if isinstance(value, dict) and isinstance(existing.get(key), dict):
            _deep_merge(existing[key], value)
        else:
            existing[key] = value
    return existing