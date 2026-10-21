"""Builds and rebuilds the local Chroma vector index of project artifacts.

Indexed sources (all repo-native — no external corpus):
- README.md                          (architecture narrative)
- dbt_project/models/**/*.sql|yml    (semantic layer meaning)
- snowflake/*.sql                    (DDL / access model)
- airflow/dags/*.py                  (orchestration semantics)
- databricks/notebooks/*.py          (bronze/silver cleaning rules)

Chunking is markdown/SQL aware-ish: we split on section headers when present,
otherwise fall back to a fixed character window, and always carry the source
path + section title in metadata so the agent can cite files.
"""
from __future__ import annotations

import shutil
from datetime import datetime, timezone

import chromadb
from langchain_ollama import OllamaEmbeddings

from ai.config import CONFIG, PROJECT_ROOT

COLLECTION_NAME = "meowmetrics_docs"

DOC_GLOBS = [
    ("README.md", PROJECT_ROOT / "README.md"),
    ("databricks", PROJECT_ROOT / "databricks"),
    ("dbt_project", PROJECT_ROOT / "dbt_project"),
    ("snowflake", PROJECT_ROOT / "snowflake"),
    ("airflow", PROJECT_ROOT / "airflow"),
    ("dashboard", PROJECT_ROOT / "dashboard" / "app.py"),
]
INCLUDE_SUFFIXES = {".md", ".sql", ".py", ".yml", ".yaml", ".json", ".tf"}

CHUNK_CHARS = 1000
CHUNK_OVERLAP = 150


def _iter_source_files():
    for label, path in DOC_GLOBS:
        if path.is_file():
            yield path
        elif path.is_dir():
            for f in path.rglob("*"):
                if f.is_file() and f.suffix.lower() in INCLUDE_SUFFIXES and "venv" not in str(f):
                    yield f


def _make_chunks(text: str) -> list[str]:
    """Section-aware chunking with overlap."""
    # Normalize headings to their own line for predictable splits
    lines = text.splitlines()
    sections, buf = [], []
    for line in lines:
        if line.lstrip().startswith(("#", "--", "/*")) and buf:
            sections.append("\n".join(buf))
            buf = [line]
        else:
            buf.append(line)
    if buf:
        sections.append("\n".join(buf))

    chunks: list[str] = []
    for sec in sections:
        if len(sec) <= CHUNK_CHARS:
            if sec.strip():
                chunks.append(sec)
            continue
        step = CHUNK_CHARS - CHUNK_OVERLAP
        for i in range(0, len(sec), step):
            window = sec[i:i + CHUNK_CHARS]
            if window.strip():
                chunks.append(window)
    return [c for c in chunks if len(c.strip()) > 40]


def build_index(force: bool = False) -> dict:
    """(Re)build the vector index. Returns build stats for logging."""
    embeddings = OllamaEmbeddings(
        model=CONFIG.embed_model, base_url=CONFIG.ollama_base_url
    )

    index_dir = CONFIG.index_dir
    if force and index_dir.exists():
        shutil.rmtree(index_dir)
    index_dir.mkdir(parents=True, exist_ok=True)

    client = chromadb.PersistentClient(path=str(index_dir))

    if force and COLLECTION_NAME in [c.name for c in client.list_collections()]:
        client.delete_collection(COLLECTION_NAME)
    collection = client.get_or_create_collection(COLLECTION_NAME)

    stats = {"files": 0, "chunks": 0}
    ids, docs, metas = [], [], []
    for path in _iter_source_files():
        try:
            text = path.read_text(encoding="utf-8", errors="ignore")
        except OSError:
            continue
        rel = str(path.relative_to(PROJECT_ROOT))
        for chunk in _make_chunks(text):
            ids.append(f"{rel}::{len(ids)}")
            docs.append(chunk)
            metas.append({"source": rel, "indexed_at": datetime.now(timezone.utc).isoformat()})  # noqa: UP017
        stats["files"] += 1

    # Batch in chunks of 64 to keep Ollama embedding responsive
    B = 64
    for i in range(0, len(docs), B):
        batch_docs, batch_ids, batch_metas = docs[i:i + B], ids[i:i + B], metas[i:i + B]
        vectors = embeddings.embed_documents(batch_docs)
        collection.add(ids=batch_ids, documents=batch_docs, metadatas=batch_metas, embeddings=vectors)

    stats["chunks"] = len(docs)
    return stats


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Build the MeowMetrics doc index.")
    parser.add_argument("--force", action="store_true", help="Delete and rebuild the collection.")
    args = parser.parse_args()
    print(build_index(force=args.force))
