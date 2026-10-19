"""Query-side of the RAG index: returns formatted context with source citations."""
from __future__ import annotations

import chromadb
from langchain_ollama import OllamaEmbeddings

from ai.config import CONFIG
from ai.rag.index_builder import COLLECTION_NAME, build_index

_CLIENT = None
_COLLECTION = None

QUERY_TEMPLATE = (
    "You are retrieving context for a conversation about the MeowMetrics cat telemetry data pipeline."
)


class DocIndex:
    """Thin wrapper around a persistent Chroma collection."""

    def __init__(self):
        self._ensure_collection()

    def _ensure_collection(self):
        global _CLIENT, _COLLECTION
        if _COLLECTION is None:
            if not (CONFIG.index_dir / "chroma.sqlite3").exists():
                print("[rag] index missing — building it (first run)...")
                build_index()
            _CLIENT = chromadb.PersistentClient(path=str(CONFIG.index_dir))
            collections = [c.name for c in _CLIENT.list_collections()]
            if COLLECTION_NAME not in collections:
                print("[rag] collection missing — building it...")
                build_index()
                _CLIENT = chromadb.PersistentClient(path=str(CONFIG.index_dir))
            _COLLECTION = _CLIENT.get_collection(COLLECTION_NAME)
        self.collection = _COLLECTION

    def search(self, query: str, k: int | None = None) -> list[dict]:
        """Semantic search. Returns [{text, source, score}]."""
        embeddings = OllamaEmbeddings(
            model=CONFIG.embed_model, base_url=CONFIG.ollama_base_url
        )
        vec = embeddings.embed_query(query)
        results = self.collection.query(query_embeddings=[vec], n_results=k or CONFIG.retrieval_k)
        out = []
        docs = results.get("documents") or [[]]
        for text, src, dist in zip(docs[0], results.get("ids", [[]])[0], results.get("distances", [[]])[0]):
            out.append({
                "text": text,
                "source": src.split("::")[0],
                "score": round(1.0 - float(dist), 4),  # cosine similarity proxy
            })
        return out

    def format_context(self, query: str, k: int | None = None) -> str:
        """Context block with in-line source tags for grounded answering."""
        hits = self.search(query, k)
        if not hits:
            return "No relevant documentation found."
        sections = []
        for i, h in enumerate(hits, 1):
            sections.append(
                f"### Chunk {i} - Source file: {h['source']}\n"
                f"similarity: {h['score']}\n{h['text'].strip()}"
            )
        return "\n\n".join(sections)


def get_doc_index() -> DocIndex:
    return DocIndex()
