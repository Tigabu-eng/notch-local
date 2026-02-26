from __future__ import annotations

from sqlalchemy.orm import Session
from sqlalchemy import select
from uuid import UUID

from app.db.models.call_insight import CallInsightORM
from app.models import CallInsight, ActionItem, PersonMention


class CallInsightRepositorySQLAlchemy:
    def __init__(self, db: Session):
        self.db = db

    def upsert(
        self,
        call_id: UUID,
        insights: CallInsight,
        *,
        searchable_text: str | None = None,
        embedding: list[float] | None = None,
    ) -> CallInsight:
        """Insert or update call insights for a call.

        Note: CallInsightORM primary key is `id`, not `call_id`. We upsert by `call_id`.
        """
        existing = (
            self.db.query(CallInsightORM)
            .filter(CallInsightORM.call_id == call_id)
            .first()
        )
        payload = {
            "summary": insights.summary,
            "searchable_text": searchable_text,
            "embedding": embedding,
            "tags": list(insights.tags),
            "action_items": [ai.model_dump() for ai in insights.action_items],
            "people_mentioned": [p.model_dump() for p in insights.people_mentioned],
            "key_decisions": list(insights.key_decisions),
            "call_type": insights.call_type,
        }

        if existing:
            existing.summary = payload["summary"]
            existing.searchable_text = payload["searchable_text"]
            existing.embedding = payload["embedding"]
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
        # return self._to_model(orm)
        return self._to_dict(orm)

    def get(self, call_id: UUID) -> CallInsight | None:
        # orm = self.db.get(CallInsightORM, call_id)
        orm = self.db.query(CallInsightORM).filter(CallInsightORM.call_id == call_id).first()
        if not orm:
            return None
        return self._to_model(orm)

    def _to_model(self, orm: CallInsightORM) -> CallInsight:
        return CallInsight(
            id=orm.id,
            summary=orm.summary,
            tags=list(orm.tags or []),
            action_items=[ActionItem(**ai) for ai in (orm.action_items or [])],
            people_mentioned=[PersonMention(**p) for p in (orm.people_mentioned or [])],
            key_decisions=list(orm.key_decisions or []),
            call_type=orm.call_type,
        )

    def _to_dict(self, model: CallInsight) -> dict:
        return {
            "id": model.id.__str__(),
            "call_id": model.call_id.__str__(),
            "summary": model.summary,
            "searchable_text": getattr(model, "searchable_text", None),
            "tags": list(model.tags),
            "action_items": [
                {**ai} for ai in model.action_items
            ],
            "people_mentioned": [
                {**p} for p in model.people_mentioned
            ],
            "key_decisions": list(model.key_decisions),
            "call_type": model.call_type,
        }
