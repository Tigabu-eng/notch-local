from sqlalchemy import Column, Text, ForeignKey, Numeric, Boolean
from sqlalchemy.dialects.postgresql import UUID, JSONB
from pgvector.sqlalchemy import Vector

import uuid
from app.db.base import Base



class IntervieweeProfileORM(Base):
    __tablename__ = "interviewee_profiles"

    id = Column(UUID(as_uuid=True), primary_key=True, default=uuid.uuid4)
    call_insight_id = Column(
        UUID(as_uuid=True),
        ForeignKey("call_insights.id", ondelete="CASCADE"),
        nullable=False,
    )
    full_name = Column(Text, nullable=True)
    current_title = Column(Text, nullable=True)
    current_company = Column(Text, nullable=True)
    seniority_level = Column(Text, nullable=True)
    industry_focus = Column(JSONB, nullable=True)
    years_experience_estimate = Column(
        Numeric,
        nullable=True,
    )
    career_history = Column(JSONB, nullable=True)
    leadership_scope = Column(JSONB, nullable=True)
    transformation_experience = Column(JSONB, nullable=True)
    private_equity_exposure = Column(JSONB, nullable=True)
    technical_capabilities = Column(JSONB, nullable=True)
    notable_achievements = Column(JSONB, nullable=True)
    risk_flags = Column(JSONB, nullable=True)
    searchable_summary = Column(Text, nullable=True)
    confidence_score = Column(Numeric, nullable=True)
    embedding = Column(Vector(1536))
    has_pe_experience = Column(Boolean, index=True, default=False)
    transformation_types = Column(JSONB, nullable=True)

