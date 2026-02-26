from __future__ import annotations

import json
import re
import uuid
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Literal

from sqlalchemy.orm import Session

from app.repositories.conversation_repository import ConversationRepository
from app.repositories.search_repository import SearchRepository
from app.schemas.search import (
    AggregationStats,
    AgentSearchResponse,
    CallInsightResult,
    InterviewProfileResult,
)
from app.services.openrouter_service import OpenRouterService


ALLOWED_INTENTS = {"chitchat", "retrieval", "comparative", "aggregation"}


def _safe_json_loads(text: str | None) -> dict[str, Any] | None:
    if not text:
        return None
    try:
        return json.loads(text)
    except Exception:
        m = re.search(r"\{.*\}", text, flags=re.DOTALL)
        if not m:
            return None
        try:
            return json.loads(m.group(0))
        except Exception:
            return None


def _is_greeting_or_smalltalk(q: str) -> bool:
    s = q.strip().lower()
    if not s:
        return True
    # quick heuristics
    greetings = [
        "hi",
        "hello",
        "hey",
        "good morning",
        "good afternoon",
        "good evening",
        "thanks",
        "thank you",
        "thx",
        "how are you",
        "what's up",
        "whats up",
    ]
    if any(s == g or s.startswith(g + " ") for g in greetings):
        return True
    # very short messages are often smalltalk (but not always)
    if len(s) <= 3 and s in {"yo", "sup", "ok", "k", "?",
                            "hey", "hi"}:
        return True
    return False


def _load_company_context() -> str:
    p = Path(__file__).resolve().parent.parent / "knowledge" / "notch_partners.md"
    try:
        return p.read_text(encoding="utf-8")[:8000]
    except Exception:
        return ""


COMPANY_CONTEXT = _load_company_context()


@dataclass
class QueryPlan:
    intent: Literal["chitchat", "retrieval", "comparative", "aggregation"]
    search_targets: list[Literal["call", "interviewee_profile"]]
    comparative_subject: str | None = None
    comparative_attribute: str | None = None
    aggregation_metric: str | None = None


class AgentCallSearchService:
    """Conversational AI agent for /api/calls/search.

    - Maintains conversation state per `session_id`
    - Routes messages (smalltalk / retrieval / comparative / aggregation)
    - Produces a natural Markdown response (no internal mechanics)
    """

    def __init__(self, *, openrouter_service: OpenRouterService, db: Session) -> None:
        self.llm = openrouter_service
        self.db = db
        self.repo = SearchRepository(db)
        self.convo = ConversationRepository(db)

    async def _plan(self, query: str, history: list[dict[str, str]]) -> dict[str, Any]:
        """Ask the LLM to produce a strict JSON plan."""
        # Fast path for greetings
        if _is_greeting_or_smalltalk(query):
            return {"intent": "chitchat", "search_targets": []}

        system_prompt = (
            "You are a query router for an internal call intelligence assistant. "
            "Decide HOW to respond (smalltalk vs retrieval vs comparative vs aggregation). "
            "Return ONLY valid JSON."
        )

        user_payload = {
            "task": "plan",
            "query": query,
            "recent_history": history[-6:],
            "allowed_intents": ["chitchat", "retrieval", "comparative", "aggregation"],
            "allowed_targets": ["call", "interviewee_profile"],
            "aggregation_metrics": [
                "total_calls",
                "analyzed_calls",
                "total_insights",
                "total_interviewee_profiles",
                "calls_by_status",
            ],
            "output_schema": {
                "intent": "chitchat|retrieval|comparative|aggregation",
                "search_targets": "array of call and/or interviewee_profile (optional for chitchat)",
                "comparative_subject": "optional - e.g. CEOs",
                "comparative_attribute": "optional - e.g. more experienced in bioprocessing",
                "aggregation_metric": "optional - one of aggregation_metrics",
            },
            "rules": [
                "If the user is greeting, thanking, or making small talk, choose intent=chitchat.",
                "If the user asks for counts/How many/Total/Number of, choose intent=aggregation.",
                "If the user asks which is more/most/best/rank/compare, choose intent=comparative.",
                "Otherwise choose intent=retrieval.",
                "For aggregation, set search_targets=['call'].",
                "For retrieval/comparative, include at least one target. Prefer call when the query mentions meetings, calls, summaries, or insights.",
            ],
        }

        content = await self.llm._chat_completion(
            model="anthropic/claude-3.5-sonnet",
            messages=[
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": json.dumps(user_payload)},
            ],
            temperature=0,
            response_format={"type": "json_object"},
        )
        plan = _safe_json_loads(content) or {}

        intent = str(plan.get("intent", "retrieval")).strip().lower()
        if intent not in ALLOWED_INTENTS:
            intent = "retrieval"

        targets = plan.get("search_targets") or []
        if isinstance(targets, str):
            targets = [targets]

        targets_norm: list[str] = []
        for t in targets:
            tl = str(t).strip().lower()
            if tl in ("call", "interviewee_profile") and tl not in targets_norm:
                targets_norm.append(tl)

        if intent in ("retrieval", "comparative") and not targets_norm:
            targets_norm = ["call", "interviewee_profile"]

        if intent == "aggregation":
            targets_norm = ["call"]

        plan["intent"] = intent
        plan["search_targets"] = targets_norm
        return plan

    async def _score_profiles_for_comparative(
        self,
        *,
        query: str,
        attribute: str,
        candidates: list[dict[str, Any]],
    ) -> dict[str, dict[str, Any]]:
        """Batch-score candidate profiles for a comparative query."""
        payload = {
            "query": query,
            "attribute": attribute,
            "candidates": [
                {
                    "id": str(c["id"]),
                    "full_name": c.get("full_name"),
                    "current_title": c.get("current_title"),
                    "current_company": c.get("current_company"),
                    "evidence": (c.get("searchable_summary") or "")[:2200],
                }
                for c in candidates
            ],
            "output_schema": {
                "scores": [
                    {"id": "uuid", "score": "number 0-1", "rationale": "short string"}
                ]
            },
            "rules": [
                "Use ONLY the evidence field; do not infer or hallucinate.",
                "If evidence is insufficient, score 0.1-0.3 and say 'insufficient evidence'.",
                "Higher score means more aligned/more experienced relative to the attribute.",
                "Return ONLY valid JSON.",
            ],
        }

        content = await self.llm._chat_completion(
            model="anthropic/claude-3.5-sonnet",
            messages=[
                {"role": "system", "content": "You are a careful ranker. Return ONLY valid JSON."},
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

    async def _draft_markdown(
        self,
        *,
        session_id: str,
        query: str,
        intent: str,
        history: list[dict[str, str]],
        results: list[Any],
        stats: AggregationStats | None,
        plan: dict[str, Any] | None,
    ) -> str:
        """Produce the final conversational Markdown response."""

        # Compact tool context (what the assistant can safely cite)
        tool_context: dict[str, Any] = {
            "intent": intent,
            "query": query,
            "stats": stats.model_dump() if stats else None,
            "top_results": [],
        }

        # Keep only the top few items to avoid blowing token budget
        for r in results[:10]:
            if isinstance(r, CallInsightResult):
                tool_context["top_results"].append(
                    {
                        "type": "call",
                        "call_title": r.call_title,
                        "call_date": r.call_date.isoformat(),
                        "call_status": r.call_status,
                        "summary": (r.summary or "")[:900],
                        "tags": r.tags[:8],
                    }
                )
            elif isinstance(r, InterviewProfileResult):
                tool_context["top_results"].append(
                    {
                        "type": "interviewee_profile",
                        "full_name": r.full_name,
                        "current_title": r.current_title,
                        "current_company": r.current_company,
                        "summary": (r.profile.get("summary") if isinstance(r.profile, dict) else None) or "",
                        "rationale": r.rationale,
                    }
                )

        system_prompt = (
            "You are Notch Partners' internal conversational assistant for call intelligence.\n"
            "- Your responses MUST be Markdown.\n"
            "- Be human-like, professional, and helpful.\n"
            "- Do NOT mention embeddings, similarity thresholds, vector search, or internal system mechanics.\n"
            "- If there are no relevant results, respond naturally: ask a clarifying question, suggest adjacent queries, or explain you don't have enough info.\n"
            "- Use ONLY the provided tool_context for any factual claims about calls/profiles/stats. If tool_context lacks the info, say so and ask for clarification.\n"
            "- When users ask about Notch Partners (services, positioning, industries), answer consistently with the company context provided.\n"
        )

        user_prompt = {
            "company_context": COMPANY_CONTEXT,
            "conversation_history": history[-12:],
            "user_message": query,
            "tool_context": tool_context,
            "style": {
                "tone": "professional, warm",
                "format": "Markdown",
                "avoid": ["technical system error messages", "threshold talk", "embedding talk"],
            },
            "instructions": [
                "Start with a short, direct answer or acknowledgement.",
                "If intent is retrieval/comparative, provide a concise bullet list of the best matches (if any).",
                "If intent is comparative, phrase results as a ranked shortlist with brief reasons (no need to show numeric scores).",
                "If intent is aggregation, state the counts clearly and offer a helpful follow-up question.",
                "End with one helpful follow-up question to keep the conversation going.",
            ],
        }

        content = await self.llm._chat_completion(
            model="anthropic/claude-3.5-sonnet",
            messages=[
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": json.dumps(user_prompt)},
            ],
            temperature=0.4,
        )
        return (content or "").strip() or "I'm here to help—what would you like to look into?"

    async def search(
        self,
        *,
        session_id: str | None,
        query: str,
        top_k: int,
        similarity_threshold: float,
        debug: bool = False,
    ) -> AgentSearchResponse:
        # Ensure session id
        sid = (session_id or "").strip()
        if not sid:
            sid = uuid.uuid4().hex

        # Load history before adding this message (so the planner sees prior context)
        history = self.convo.get_recent_messages(session_key=sid, limit=16)

        # Persist user message
        self.convo.add_message(session_key=sid, role="user", content=query)

        plan = await self._plan(query, history=history)
        intent = str(plan.get("intent") or "retrieval")

        # Smalltalk path
        results: list[Any] = []
        stats: AggregationStats | None = None

        if intent == "aggregation":
            stats_raw = self.repo.get_aggregation_stats()
            stats = AggregationStats(**stats_raw)

        elif intent in ("retrieval", "comparative"):
            q_embedding = await self.llm.generate_embedding(query)

            # Retrieve call-level results
            if "call" in plan.get("search_targets", []):
                call_rows = self.repo.search_call_insights(q_embedding, top_k=top_k)
                for r in call_rows:
                    sim = float(r.get("similarity") or 0.0)
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
                    sim = float(r.get("similarity") or 0.0)
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
                attribute = (str(plan.get("comparative_attribute") or "").strip() or "the requested criteria")
                scored = await self._score_profiles_for_comparative(
                    query=query,
                    attribute=attribute,
                    candidates=profile_rows[: min(len(profile_rows), 20)],
                )

                # Attach scores and sort interviewee results (profiles first)
                for item in results:
                    if isinstance(item, InterviewProfileResult):
                        sid2 = str(item.id)
                        if sid2 in scored:
                            item.score = scored[sid2]["score"]
                            item.rationale = scored[sid2]["rationale"]

                results.sort(
                    key=lambda x: (
                        0 if isinstance(x, InterviewProfileResult) else 1,
                        -(getattr(x, "score", None) or 0.0),
                        -(getattr(x, "similarity", 0.0) or 0.0),
                    )
                )
            else:
                # Default sort by similarity desc
                results.sort(key=lambda x: -(getattr(x, "similarity", 0.0) or 0.0))

        # Re-load history including the just-added user message (for response drafting)
        history2 = self.convo.get_recent_messages(session_key=sid, limit=16)

        markdown = await self._draft_markdown(
            session_id=sid,
            query=query,
            intent=intent,
            history=history2,
            results=results,
            stats=stats,
            plan=plan if debug else None,
        )

        # Persist assistant message
        self.convo.add_message(session_key=sid, role="assistant", content=markdown)

        return AgentSearchResponse(
            session_id=sid,
            intent=intent,  # type: ignore[arg-type]
            markdown=markdown,
            total_results=len(results),
            results=results,
            stats=stats,
            plan=plan if debug else None,
        )
