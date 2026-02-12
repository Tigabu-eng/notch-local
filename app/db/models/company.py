from datetime import datetime, timezone

from sqlalchemy import Column, DateTime, Integer, String, Text
from sqlalchemy.dialects.postgresql import JSONB

from app.db.base import Base


class CompanyORM(Base):
    __tablename__ = "companies"

    id = Column(String, primary_key=True)                 
    name = Column(String, nullable=False, unique=True)

    ceo = Column(JSONB, nullable=False, default=list)
    c_level = Column(JSONB, nullable=False, default=list)
    senior_level = Column(JSONB, nullable=False, default=list)

    employees = Column(Integer, nullable=True)
    ownership = Column(String, nullable=True)
    acquisition_date = Column(Integer, nullable=True)
    subsector = Column(String, nullable=True)
    notes = Column(Text, nullable=True)

    network_status = Column(String, nullable=True)
    contact_status = Column(String, nullable=True)

    updated = Column(DateTime, nullable=True)
    created_at = Column(DateTime, nullable=False, default=lambda: datetime.now(timezone.utc))
