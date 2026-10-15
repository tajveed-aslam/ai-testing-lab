"""Builds the vector store: load markdown docs -> split into chunks -> embed -> save in ChromaDB.

Run once (and again whenever the knowledge base changes):  python -m app.ingest
"""

import re
from dataclasses import dataclass

import chromadb

from app import config, llm


@dataclass
class Chunk:
    id: str  # e.g. "expenses.md#3"
    source: str  # file name, e.g. "expenses.md"
    section: str  # heading, e.g. "Receipts"
    text: str


def split_into_chunks(source: str, markdown: str) -> list[Chunk]:
    """One chunk per "## " section. Policy docs are written in short sections, so a section is a natural unit of
    meaning: small enough to retrieve precisely, big enough to keep a rule and its exceptions together.
    The document title is prefixed to each chunk so a chunk still makes sense on its own."""
    title_match = re.search(r"^# (.+)$", markdown, re.MULTILINE)
    title = title_match.group(1).strip() if title_match else source
    sections = re.split(r"^## ", markdown, flags=re.MULTILINE)[1:]  # [0] is the title and intro line
    chunks = []
    for i, section in enumerate(sections, start=1):
        heading, _, body = section.partition("\n")
        text = f"{title} > {heading.strip()}\n\n{body.strip()}"
        chunks.append(Chunk(id=f"{source}#{i}", source=source, section=heading.strip(), text=text))
    return chunks


def load_chunks() -> list[Chunk]:
    chunks = []
    for path in sorted(config.KNOWLEDGE_BASE_DIR.glob("*.md")):
        chunks.extend(split_into_chunks(path.name, path.read_text(encoding="utf-8")))
    return chunks


def collection() -> chromadb.Collection:
    """The Chroma collection, stored on disk in ./chroma_db. We supply our own (Gemini) embeddings, so Chroma's
    built-in embedding model is never used. Cosine distance: 0 = same direction, 2 = opposite."""
    store = chromadb.PersistentClient(path=str(config.CHROMA_DIR))
    return store.get_or_create_collection(
        name=config.COLLECTION_NAME, embedding_function=None, metadata={"hnsw:space": "cosine"}
    )


def build_index() -> int:
    chunks = load_chunks()
    vectors = llm.embed([c.text for c in chunks], task_type="RETRIEVAL_DOCUMENT")
    col = collection()
    existing = col.get()["ids"]
    if existing:
        col.delete(ids=existing)  # rebuild from scratch so removed sections don't linger
    col.add(
        ids=[c.id for c in chunks],
        embeddings=vectors,
        documents=[c.text for c in chunks],
        metadatas=[{"source": c.source, "section": c.section} for c in chunks],
    )
    return len(chunks)


if __name__ == "__main__":
    count = build_index()
    print(f"Indexed {count} chunks from {config.KNOWLEDGE_BASE_DIR.name}/ into {config.CHROMA_DIR.name}/")
