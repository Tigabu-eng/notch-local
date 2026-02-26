from __future__ import annotations

import json
import re
from dataclasses import dataclass
from typing import Any, Literal

from sqlalchemy.orm import Session

from app.repositories.search_repository import SearchRepository
from app.schemas.search import (
    AggregationStats,
    AgentSearchResponse,
    CallInsightResult,
    InterviewProfileResult,
)
from app.services.openrouter_service import OpenRouterService


ALLOWED_INTENTS = {"retrieval", "comparative", "aggregation"}


def _safe_json_loads(text: str | None) -> dict[str, Any] | None:
    if not text:
        return None
    try:
        return json.loads(text)
    except Exception:
        # Try to extract first JSON object
        m = re.search(r"\{.*\}", text, flags=re.DOTALL)
        if not m:
            return None
        try:
            return json.loads(m.group(0))
        except Exception:
            return None


@dataclass
class QueryPlan:
    intent: Literal["retrieval", "comparative", "aggregation"]
    # retrieval targets
    search_targets: list[Literal["call", "interviewee_profile"]]
    # comparative params
    comparative_subject: str | None = None
    comparative_attribute: str | None = None


class AgentCallSearchService:
    """AI-agent style query router + executor for /api/calls/search."""

    def __init__(self, *, openrouter_service: OpenRouterService, db: Session) -> None:
        self.llm = openrouter_service
        self.db = db
        self.repo = SearchRepository(db)

    async def _plan(self, query: str) -> dict[str, Any]:
        """Ask the LLM to produce a strict JSON plan."""
        system_prompt = (
            "You are a query router for an internal call intelligence database. "
            "Your job is to choose HOW to answer the query, not to answer it. "
            "Return ONLY valid JSON."
        )

        user_prompt = {
            "task": "plan",
            "query": query,
            "allowed_intents": ["retrieval", "comparative", "aggregation"],
            "allowed_targets": ["call", "interviewee_profile"],
            "aggregation_metrics": [
                "total_calls",
                "analyzed_calls",
                "total_insights",
                "total_interviewee_profiles",
                "calls_by_status",
            ],
            "output_schema": {
                "intent": "retrieval|comparative|aggregation",
                "search_targets": "array of call and/or interviewee_profile",
                "comparative_subject": "optional - e.g. CEOs",
                "comparative_attribute": "optional - e.g. more experienced in bioprocessing",
                "aggregation_metric": "optional - one of aggregation_metrics",
            },
            "rules": [
                "If query asks for counts/How many/Total/Number of, choose intent=aggregation.",
                "If query asks which is more/most/best/rank/compare, choose intent=comparative.",
                "Otherwise choose intent=retrieval.",
                "Always include search_targets. For aggregation, include search_targets=['call'].",
            ],
        }

        content = await self.llm._chat_completion(
            model="anthropic/claude-3.5-sonnet",
            messages=[
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": json.dumps(user_prompt)},
            ],
            temperature=0,
            response_format={"type": "json_object"},
        )
        plan = _safe_json_loads(content) or {}
        # normalize
        intent = str(plan.get("intent", "retrieval")).strip().lower()
        if intent not in ALLOWED_INTENTS:
            intent = "retrieval"
        targets = plan.get("search_targets") or ["interviewee_profile"]
        if isinstance(targets, str):
            targets = [targets]
        targets_norm: list[str] = []
        for t in targets:
            tl = str(t).strip().lower()
            if tl in ("call", "interviewee_profile") and tl not in targets_norm:
                targets_norm.append(tl)
        if not targets_norm:
            targets_norm = ["interviewee_profile"] if intent != "aggregation" else ["call"]

        plan["intent"] = intent
        plan["search_targets"] = targets_norm
        return plan

    async def _score_profiles_for_comparative(
        self,
        query: str,
        attribute: str,
        candidates: list[dict[str, Any]],
    ) -> dict[str, dict[str, Any]]:
        """Batch-score candidate profiles for a comparative query.

        Returns: {profile_id: {score: float, rationale: str}}
        """
        payload = {
            "query": query,
            "attribute": attribute,
            "candidates": [
                {
                    "id": str(c["id"]),
                    "full_name": c.get("full_name"),
                    "current_title": c.get("current_title"),
                    "current_company": c.get("current_company"),
                    "searchable_summary": (c.get("searchable_summary") or "")[:2000],
                }
                for c in candidates
            ],
            "output_schema": {
                "scores": [
                    {
                        "id": "uuid",
                        "score": "number 0-1",
                        "rationale": "short evidence-based string",
                    }
                ]
            },
            "rules": [
                "Use ONLY evidence in searchable_summary; do not hallucinate.",
                "If there is insufficient evidence, use score 0.1-0.3 and say 'insufficient evidence'.",
                "Higher score means more experienced with respect to the attribute.",
                "Return ONLY valid JSON.",
            ],
        }

        content = await self.llm._chat_completion(
            model="anthropic/claude-3.5-sonnet",
            messages=[
                {
                    "role": "system",
                    "content": "You are a careful ranker. Return ONLY valid JSON.",
                },
                {"role": "user", "content": json.dumps(payload)},
            ],
            temperature=0,
            response_format={"type": "json_object"},
        )
        data = _safe_json_loads(content) or {}
        scored: dict[str, dict[str, Any]] = {}
        for item in data.get("scores", []) or []:
            try:
                pid = str(item.get("id"))
                score = float(item.get("score"))
                score = max(0.0, min(1.0, score))
                rationale = str(item.get("rationale") or "")[:400]
                scored[pid] = {"score": score, "rationale": rationale}
            except Exception:
                continue
        return scored

    async def search(
        self,
        *,
        query: str,
        top_k: int,
        similarity_threshold: float,
    ) -> AgentSearchResponse:
        plan = await self._plan(query)
        intent = plan["intent"]

        if intent == "aggregation":
            stats_raw = self.repo.get_aggregation_stats()
            stats = AggregationStats(**stats_raw)

            # Deterministic answer text (avoid hallucinations)
            answer = (
                f"I have analyzed {stats.analyzed_calls} calls out of {stats.total_calls} total calls. "
                f"There are {stats.total_insights} insight records and {stats.total_interviewee_profiles} interviewee profiles." 
            )

            resp = AgentSearchResponse(
                query=query,
                intent="aggregation",
                answer=answer.strip(),
                total_results=0,
                results=[],
                stats=stats,
                plan=None,
            )
            return resp

        # For retrieval/comparative we embed the query
        q_embedding = await self.llm.generate_embedding(query)

        results: list[Any] = []
        # Retrieve call-level results
        if "call" in plan.get("search_targets", []):
            call_rows = self.repo.search_call_insights(q_embedding, top_k=top_k)
            for r in call_rows:
                sim = float(r["similarity"])
                if sim < similarity_threshold:
                    continue
                results.append(
                    CallInsightResult(
                        call_id=r["call_id"],
                        insight_id=r["insight_id"],
                        similarity=round(sim, 4),
                        call_title=r["call_title"],
                        call_date=r["call_date"],
                        call_status=r["call_status"],
                        call_type=r.get("call_type"),
                        summary=r["summary"],
                        tags=list(r.get("tags") or []),
                    )
                )

        # Retrieve profile-level results
        profile_rows: list[dict[str, Any]] = []
        if "interviewee_profile" in plan.get("search_targets", []):
            profile_rows = self.repo.search_interview_profiles(q_embedding, top_k=top_k)
            for r in profile_rows:
                sim = float(r["similarity"])
                if sim < similarity_threshold:
                    continue
                results.append(
                    InterviewProfileResult(
                        id=r["id"],
                        similarity=round(sim, 4),
                        full_name=r.get("full_name"),
                        current_title=r.get("current_title"),
                        current_company=r.get("current_company"),
                        seniority_level=r.get("seniority_level"),
                        profile={
                            "summary": r.get("searchable_summary"),
                            "transformation_experience": r.get("transformation_experience"),
                            "private_equity_exposure": r.get("private_equity_exposure"),
                        },
                    )
                )

        if intent == "comparative" and profile_rows:
            attribute = (
                str(plan.get("comparative_attribute") or "")
                or "the requested attribute"
            )
            scored = await self._score_profiles_for_comparative(
                query=query,
                attribute=attribute,
                candidates=profile_rows[: min(len(profile_rows), 20)],
            )

            # Attach scores and sort interviewee results
            for item in results:
                if isinstance(item, InterviewProfileResult):
                    sid = str(item.id)
                    if sid in scored:
                        item.score = scored[sid]["score"]
                        item.rationale = scored[sid]["rationale"]

            results.sort(
                key=lambda x: (
                    0
                    if isinstance(x, InterviewProfileResult)
                    else 1,
                    -(getattr(x, "score", None) or 0.0),
                    -(getattr(x, "similarity", 0.0) or 0.0),
                )
            )

            answer = (
                f"Ranked {len([r for r in results if isinstance(r, InterviewProfileResult)])} candidates by {attribute}. "
                "Top results are listed first with a score (0–1) and a short rationale."
            )
        else:
            answer = (
                "Here are the most relevant matches from your analyzed calls and profiles."
                if results
                else "I couldn't find matches above the similarity threshold. Try rephrasing or lowering the threshold."
            )

        return AgentSearchResponse(
            query=query,
            intent=intent,
            answer=answer,
            total_results=len(results),
            results=results,
            stats=None,
            plan=None,
        )
