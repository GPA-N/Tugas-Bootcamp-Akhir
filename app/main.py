from app.schemas import QueryRequest, QueryResponse
from app.services.agent import AgentRouter
from fastapi import FastAPI, HTTPException

app = FastAPI(title="RAG")
service = AgentRouter()


@app.get("/")
def read_root() -> dict[str, str]:
    return {"message": "RAG FastAPI"}


@app.get("/health")
def health():
    return {"status": "OK"}


@app.post("/rag/", response_model=QueryResponse)
def answer_with_rag(question: QueryRequest) -> QueryResponse:
    if not question.query.strip():
        raise HTTPException(
            status_code=400, detail="Pertanyaan tidak boleh kosong"
        )
    try:
        return service.process(question.query)
    except Exception as e:
        raise HTTPException(
            status_code=500, detail=str(e)
        )