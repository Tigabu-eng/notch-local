from sqlalchemy import Column, Text, ForeignKey
from sqlalchemy.dialects.postgresql import UUID, JSONB
import uuid

from app.db.base import Base


class CallInsightORM(Base):
    __tablename__ = "call_insights"

    id = Column(UUID(as_uuid=True), primary_key=True, default=uuid.uuid4)
    call_id = Column(
        UUID(as_uuid=True),
        ForeignKey("calls.id", ondelete="CASCADE"),
        nullable=False,
    )

    summary = Column(Text, nullable=False)
    tags = Column(JSONB, nullable=False)
    action_items = Column(JSONB, nullable=False)
    people_mentioned = Column(JSONB, nullable=False)
    key_decisions = Column(JSONB, nullable=False)
    call_type = Column(Text, nullable=False)