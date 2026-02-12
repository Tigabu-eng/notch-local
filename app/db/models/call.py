from sqlalchemy import Column, Text, DateTime, String
from sqlalchemy.dialects.postgresql import UUID
from datetime import datetime, timezone
import uuid

from app.db.base import Base


class CallORM(Base):
    __tablename__ = "calls"

    id = Column(UUID(as_uuid=True), primary_key=True, default=uuid.uuid4)

    title = Column(String(200), nullable=False)
    description = Column(Text, nullable=True)

    call_date = Column(DateTime, nullable=False)
    transcript = Column(Text, nullable=False)
    status = Column(String, nullable=False, default="uploaded")
    created_at = Column(DateTime, default=lambda: datetime.now(timezone.utc), nullable=False)
