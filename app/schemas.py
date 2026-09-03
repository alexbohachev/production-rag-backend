from pydantic import BaseModel, Field


class IngestDocument(BaseModel):
    id: str
    title: str
    text: str


class IngestRequest(BaseModel):
    documents: list[IngestDocument]


class Citation(BaseModel):
    chunk_id: str
    doc_id: str
    title: str
    quote: str
    score: float


class RetrievalTrace(BaseModel):
    bm25_ids: list[str]
    vector_ids: list[str]
    fused_ids: list[str]
    reranked_ids: list[str]
    rerank_backend: str = "feature"


class QueryRequest(BaseModel):
    query: str = Field(min_length=3, max_length=2000)
    top_k: int = Field(default=5, ge=1, le=20)
    include_trace: bool = False


class QueryResponse(BaseModel):
    answer: str
    citations: list[Citation]
    confidence: float = Field(ge=0, le=1)
    retrieval: RetrievalTrace | None = None
