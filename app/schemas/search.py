from __future__ import annotations

from datetime import datetime
from typing import Any, Literal, Optional
from uuid import UUID

from pydantic import BaseModel, Field


class AgentSearchRequest(BaseModel):
    """Request model for AI-agent style conversational search/chat."""

    # Client-provided session identifier for context across messages.
    # If omitted, the API will create one and return it in the response.
    session_id: Optional[str] = Field(default=None, max_length=64)

    # User message (can be short, e.g., "Hi")
    query: str = Field(..., min_length=1)

    top_k: int = Field(default=10, ge=1, le=50)
    similarity_threshold: float = Field(default=0.4, ge=0.0, le=1.0)

    # When true, the agent returns its internal plan for debugging.
    debug: bool = Field(default=False)


class InterviewProfileResult(BaseModel):
    type: Literal["interviewee_profile"] = "interviewee_profile"
    id: UUID
    similarity: float
    full_name: Optional[str] = None
    current_title: Optional[str] = None
    current_company: Optional[str] = None
    seniority_level: Optional[str] = None
    profile: dict[str, Any] = Field(default_factory=dict)
    score: Optional[float] = None  # used for comparative ranking
    rationale: Optional[str] = None


class CallInsightResult(BaseModel):
    type: Literal["call"] = "call"
    call_id: UUID
    insight_id: UUID
    similarity: float
    call_title: str
    call_date: datetime
    call_status: str
    call_type: Optional[str] = None
    summary: str
    tags: list[str] = Field(default_factory=list)


SearchResult = InterviewProfileResult | CallInsightResult


class AggregationStats(BaseModel):
    total_calls: int
    analyzed_calls: int
    total_insights: int
    total_interviewee_profiles: int
    calls_by_status: dict[str, int]


class AgentSearchResponse(BaseModel):
    session_id: str
    intent: Literal["chitchat", "retrieval", "comparative", "aggregation"]
    markdown: str

    # Optional structured payload (useful for UI, not required by chat UX)
    total_results: int = 0
    results: list[SearchResult] = Field(default_factory=list)
    stats: Optional[AggregationStats] = None

    # Debug only
    plan: Optional[dict[str, Any]] = None


# Backwards-compatible aliases (if older clients still import these)
SearchRequest = AgentSearchRequest
SearchResponse = AgentSearchResponse
