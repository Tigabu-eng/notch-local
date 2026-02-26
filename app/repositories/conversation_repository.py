from __future__ import annotations

from datetime import datetime, timezone
from typing import Iterable

from sqlalchemy.orm import Session
from sqlalchemy import select

from app.db.models.conversation import ConversationSessionORM, ConversationMessageORM


class ConversationRepository:
    def __init__(self, db: Session) -> None:
        self.db = db

    def get_or_create_session(self, session_key: str) -> ConversationSessionORM:
        stmt = select(ConversationSessionORM).where(ConversationSessionORM.session_id == session_key)
        sess = self.db.execute(stmt).scalars().first()
        if sess:
            # touch updated_at
            sess.updated_at = datetime.now(timezone.utc)
            self.db.add(sess)
            self.db.commit()
            self.db.refresh(sess)
            return sess

        sess = ConversationSessionORM(session_id=session_key)
        self.db.add(sess)
        self.db.commit()
        self.db.refresh(sess)
        return sess

    def add_message(self, *, session_key: str, role: str, content: str) -> None:
        sess = self.get_or_create_session(session_key)
        msg = ConversationMessageORM(session_id=sess.id, role=role, content=content)
        sess.updated_at = datetime.now(timezone.utc)
        self.db.add_all([msg, sess])
        self.db.commit()

    def get_recent_messages(self, *, session_key: str, limit: int = 12) -> list[dict[str, str]]:
        stmt = (
            select(ConversationMessageORM)
            .join(ConversationSessionORM, ConversationMessageORM.session_id == ConversationSessionORM.id)
            .where(ConversationSessionORM.session_id == session_key)
            .order_by(ConversationMessageORM.created_at.desc())
            .limit(limit)
        )
        rows = list(self.db.execute(stmt).scalars().all())
        rows.reverse()
        return [{"role": r.role, "content": r.content} for r in rows]
