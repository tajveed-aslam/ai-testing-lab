"""Thin wrapper around the Gemini API: one function to embed text, one to generate an answer.

Both retry on rate limits (HTTP 429) and temporary server errors, because the free tier limits requests per minute.
"""

import time

from google import genai
from google.genai import errors, types

from app import config

_client: genai.Client | None = None


def client() -> genai.Client:
    global _client
    if _client is None:
        if not config.GEMINI_API_KEY:
            raise RuntimeError("GEMINI_API_KEY is not set. Copy .env.example to .env and add your key.")
        _client = genai.Client(api_key=config.GEMINI_API_KEY)
    return _client


def _with_retry(call, attempts: int = 5):
    """Retry a Gemini call on 429/5xx with exponential backoff: 2s, 4s, 8s, 16s."""
    for attempt in range(attempts):
        try:
            return call()
        except errors.APIError as e:
            retryable = e.code == 429 or (e.code or 0) >= 500
            if not retryable or attempt == attempts - 1:
                raise
            time.sleep(2 ** (attempt + 1))


def embed(texts: list[str], task_type: str) -> list[list[float]]:
    """Turn texts into vectors. task_type is RETRIEVAL_DOCUMENT for chunks and RETRIEVAL_QUERY for questions:
    Gemini embeds the two slightly differently so questions land near the passages that answer them."""
    result = _with_retry(
        lambda: client().models.embed_content(
            model=config.EMBEDDING_MODEL,
            contents=texts,
            config=types.EmbedContentConfig(task_type=task_type),
        )
    )
    return [e.values for e in result.embeddings]


def generate(system_prompt: str, user_prompt: str, temperature: float = 0.0) -> str:
    """Ask the model. Temperature 0 makes answers as repeatable as the model allows (not perfectly)."""
    result = _with_retry(
        lambda: client().models.generate_content(
            model=config.ANSWER_MODEL,
            contents=user_prompt,
            config=types.GenerateContentConfig(system_instruction=system_prompt, temperature=temperature),
        )
    )
    return (result.text or "").strip()
