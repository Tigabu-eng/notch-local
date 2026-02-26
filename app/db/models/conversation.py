from __future__ import annotations

import uuid
from datetime import datetime, timezone

from sqlalchemy import DateTime, ForeignKey, Index, String, Text
from sqlalchemy.dialects.postgresql import UUID
from sqlalchemy.orm import Mapped, mapped_column, relationship

from app.db.base import Base


class ConversationSessionORM(Base):
    __tablename__ = "conversation_sessions"

    id: Mapped[uuid.UUID] = mapped_column(UUID(as_uuid=True), primary_key=True, default=uuid.uuid4)
    session_id: Mapped[str] = mapped_column(String(64), unique=True, index=True, nullable=False)

    created_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), default=lambda: datetime.now(timezone.utc), nullable=False)
    updated_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), default=lambda: datetime.now(timezone.utc), nullable=False)

    messages: Mapped[list["ConversationMessageORM"]] = relationship(
        "ConversationMessageORM",
        back_populates="session",
        cascade="all, delete-orphan",
        lazy="selectin",
    )


class ConversationMessageORM(Base):
    __tablename__ = "conversation_messages"

    id: Mapped[uuid.UUID] = mapped_column(UUID(as_uuid=True), primary_key=True, default=uuid.uuid4)
    session_id: Mapped[uuid.UUID] = mapped_column(UUID(as_uuid=True), ForeignKey("public.conversation_sessions.id", ondelete="CASCADE"), nullable=False, index=True)

    # "user" | "assistant" | "system"
    role: Mapped[str] = mapped_column(String(16), nullable=False)
    content: Mapped[str] = mapped_column(Text(), nullable=False)

    created_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), default=lambda: datetime.now(timezone.utc), nullable=False)

    session: Mapped["ConversationSessionORM"] = relationship("ConversationSessionORM", back_populates="messages")


Index("ix_conversation_messages_session_created", ConversationMessageORM.session_id, ConversationMessageORM.created_at)
