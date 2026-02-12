from __future__ import annotations

from sqlalchemy.orm import Session
from sqlalchemy import select
from uuid import UUID

from app.db.models.call_insight import CallInsightORM
from app.models import CallInsight, ActionItem, PersonMention


class CallInsightRepositorySQLAlchemy:
    def __init__(self, db: Session):
        self.db = db

    def upsert(self, call_id: UUID, insights: CallInsight) -> CallInsight:
        """Insert or update call insights for a call."""
        existing = self.db.get(CallInsightORM, call_id)
        payload = {
            "summary": insights.summary,
            "tags": list(insights.tags),
            "action_items": [ai.model_dump() for ai in insights.action_items],
            "people_mentioned": [p.model_dump() for p in insights.people_mentioned],
            "key_decisions": list(insights.key_decisions),
        }

        if existing:
            existing.summary = payload["summary"]
            existing.tags = payload["tags"]
            existing.action_items = payload["action_items"]
            existing.people_mentioned = payload["people_mentioned"]
            existing.key_decisions = payload["key_decisions"]
            orm = existing
        else:
            orm = CallInsightORM(call_id=call_id, **payload)
            self.db.add(orm)

        self.db.commit()
        self.db.refresh(orm)
        return self._to_model(orm)

    def get(self, call_id: UUID) -> CallInsight | None:
        orm = self.db.get(CallInsightORM, call_id)
        if not orm:
            return None
        return self._to_model(orm)

    def _to_model(self, orm: CallInsightORM) -> CallInsight:
        return CallInsight(
            summary=orm.summary,
            tags=list(orm.tags or []),
            action_items=[ActionItem(**ai) for ai in (orm.action_items or [])],
            people_mentioned=[PersonMention(**p) for p in (orm.people_mentioned or [])],
            key_decisions=list(orm.key_decisions or []),
        )
