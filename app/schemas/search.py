from pydantic import BaseModel, Field
from typing import List, Optional
from uuid import UUID


class SearchRequest(BaseModel):
    query: str = Field(..., min_length=3)
    top_k: Optional[int] = Field(default=5, ge=1, le=50)
    similarity_threshold: Optional[float] = Field(default=0.4, ge=0.0, le=1.0)


class InterviewProfileResult(BaseModel):
    id: UUID
    similarity: float
    full_name: Optional[str]
    current_title: Optional[str]
    current_company: Optional[str]
    seniority_level: Optional[str]
    profile: dict


class SearchResponse(BaseModel):
    query: str
    total_results: int
    results: List[InterviewProfileResult]