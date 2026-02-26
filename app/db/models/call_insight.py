from sqlalchemy import Column, Text, ForeignKey, UniqueConstraint
from sqlalchemy.dialects.postgresql import UUID, JSONB
from pgvector.sqlalchemy import Vector
import uuid

from app.db.base import Base


class CallInsightORM(Base):
    __tablename__ = "call_insights"
    __table_args__ = (UniqueConstraint("call_id", name="uq_call_insights_call_id"),)

    id = Column(UUID(as_uuid=True), primary_key=True, default=uuid.uuid4)
    call_id = Column(
        UUID(as_uuid=True),
        ForeignKey("calls.id", ondelete="CASCADE"),
        nullable=False,
    )

    summary = Column(Text, nullable=False)
    # A compact text used for semantic retrieval (summary + tags + key entities).
    searchable_text = Column(Text, nullable=True)
    # Embedding for semantic search (pgvector). 1536-dim to match text-embedding-3-small.
    embedding = Column(Vector(1536), nullable=True)
    tags = Column(JSONB, nullable=False)
    action_items = Column(JSONB, nullable=False)
    people_mentioned = Column(JSONB, nullable=False)
    key_decisions = Column(JSONB, nullable=False)
    call_type = Column(Text, nullable=False)