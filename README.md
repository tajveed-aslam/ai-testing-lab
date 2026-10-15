# AI testing lab: RAG chatbot + LLM test suite

A small retrieval-augmented (RAG) chatbot that answers questions about a fictional company's policies, built as the
system under test for learning how to test LLM applications. Work in progress; the full README comes in the last phase.

## Run it

```bash
python -m venv .venv
.venv\Scripts\activate            # Windows  (macOS/Linux: source .venv/bin/activate)
pip install -r requirements.txt
copy .env.example .env            # then put your Gemini API key in .env

python -m app.ingest              # build the vector store from knowledge_base/
python -c "from app.rag import ask; print(ask('How many days of annual leave do I get?')['answer'])"
uvicorn app.api:app --reload      # API at http://127.0.0.1:8000/docs
```
