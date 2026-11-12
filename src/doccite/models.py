from enum import StrEnum
from typing import Literal

from pydantic import BaseModel, Field

SearchMode = Literal["hybrid", "bm25", "dense"]


class DocumentInfo(BaseModel):
    doc_id: str
    title: str
    source: str
    format: str
    chunk_count: int
    page_count: int | None = None


class Location(BaseModel):
    doc_id: str
    title: str
    page: int | None = None
    section: str | None = None
    chunk_id: str | None = None
    start: int
    end: int


class Chunk(BaseModel):
    location: Location
    text: str


class SearchHit(Chunk):
    score: float
    rerank_score: float | None = None


class SearchResult(BaseModel):
    requested_mode: SearchMode
    mode: SearchMode
    reranked: bool = False
    hits: list[SearchHit] = Field(default_factory=list)
    warnings: list[str] = Field(default_factory=list)


class ChunkContext(BaseModel):
    chunk: Chunk
    before: list[Chunk]
    after: list[Chunk]


class IngestItem(BaseModel):
    source: str
    status: Literal["indexed", "unchanged", "error"]
    document: DocumentInfo | None = None
    error: str | None = None


class IngestResult(BaseModel):
    items: list[IngestItem]


class DocumentList(BaseModel):
    documents: list[DocumentInfo]


class RemoveResult(BaseModel):
    doc_id: str
    removed: bool


class IndexStats(BaseModel):
    documents: int
    chunks: int
    embeddings: int


class QuoteEdit(BaseModel):
    operation: Literal["replace", "insert", "delete"]
    quote: str
    source: str


class QuoteResult(BaseModel):
    status: Literal["verbatim", "normalized", "modified", "not_found"]
    similarity: float = 0
    match: Chunk | None = None
    differences: list[QuoteEdit] = Field(default_factory=list)


class Verdict(StrEnum):
    SUPPORTED = "SUPPORTED"
    PARTIALLY_SUPPORTED = "PARTIALLY_SUPPORTED"
    NOT_FOUND = "NOT_FOUND"
    CONTRADICTED = "CONTRADICTED"


class Evidence(Chunk):
    score: float


class ClaimResult(BaseModel):
    verdict: Verdict
    confidence: float = Field(
        ge=0, le=1, description="Heuristic score, NLI probability, or client-reported score"
    )
    evidence: list[Evidence] = Field(default_factory=list)
    issues: list[str] = Field(default_factory=list)
    judged_by: Literal["heuristic", "nli", "sampling"] = "heuristic"


class CitationCheck(BaseModel):
    marker: str
    doc_id: str | None = None
    page: int | None = None
    status: Literal["checked", "unresolved", "missing_document", "missing_page"]
    result: ClaimResult | None = None


class CitedClaim(BaseModel):
    text: str
    citations: list[CitationCheck] = Field(default_factory=list)
    recommendations: list[Evidence] = Field(default_factory=list)


class CitationReport(BaseModel):
    claims: list[CitedClaim]
    summary: dict[str, int]
