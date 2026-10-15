"""A tiny HTTP API around ask().  Run:  uvicorn app.api:app --reload   then open http://127.0.0.1:8000/docs"""

from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field

from app.rag import ask

app = FastAPI(title="NovaTech policy assistant")


class AskRequest(BaseModel):
    # Validation happens here, before any model call: empty or huge questions are rejected with 422.
    question: str = Field(min_length=1, max_length=2000)


class Chunk(BaseModel):
    text: str
    source: str
    section: str
    distance: float


class AskResponse(BaseModel):
    answer: str
    retrieved_chunks: list[Chunk]
    sources: list[str]


@app.get("/health")
def health() -> dict:
    return {"status": "ok"}


@app.post("/ask", response_model=AskResponse)
def ask_endpoint(body: AskRequest) -> AskResponse:
    question = body.question.strip()
    if not question:
        # Whitespace-only passes min_length, so it's rejected here.
        raise HTTPException(status_code=422, detail="question must not be blank")
    return AskResponse(**ask(question))
