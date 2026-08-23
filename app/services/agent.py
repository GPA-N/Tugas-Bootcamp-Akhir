from app.services.rag import RAGService


class AgentRouter:
    def __init__(self, rag_service: RAGService | None = None):
        self.rag_service = rag_service or RAGService()

    def process(self, query: str) -> dict:
        return self.rag_service.query(query)