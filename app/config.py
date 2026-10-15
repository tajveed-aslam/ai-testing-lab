"""All settings in one place. Every value can be overridden with an environment variable (or the .env file)."""

import os
from pathlib import Path

from dotenv import load_dotenv

ROOT = Path(__file__).resolve().parent.parent
load_dotenv(ROOT / ".env")

GEMINI_API_KEY = os.getenv("GEMINI_API_KEY", "")

# The model that answers questions (cheap and fast) and the model that embeds text into vectors.
ANSWER_MODEL = os.getenv("ANSWER_MODEL", "gemini-flash-lite-latest")
EMBEDDING_MODEL = os.getenv("EMBEDDING_MODEL", "gemini-embedding-001")

KNOWLEDGE_BASE_DIR = ROOT / "knowledge_base"
CHROMA_DIR = ROOT / "chroma_db"
COLLECTION_NAME = "novatech"

# How many chunks to retrieve and hand to the model for each question.
TOP_K = int(os.getenv("TOP_K", "4"))
