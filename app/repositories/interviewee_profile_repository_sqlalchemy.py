from __future__ import annotations

from sqlalchemy.orm import Session
from sqlalchemy import select
from uuid import UUID

from app.db.models import (
    IntervieweeProfileORM,
)
from app.models import IntervieweeProfile


class IntervieweeProfileRepositorySQLAlchemy:
    def __init__(self, db: Session):
        self.db = db

    

    def upsert(
        self, call_insight_id: UUID, profile: IntervieweeProfile
    ) -> IntervieweeProfile:
        """Insert or update interviewee profile for a call insight."""
        existing = self.db.get(IntervieweeProfileORM, call_insight_id)
        career_history_data = (
            [ch.to_dict() for ch in profile.career_history]
            if profile.career_history
            else []
        )
        leadership_scope_data = profile.leadership_scope.to_dict() if profile.leadership_scope else {}
        
        transformation_experience_data = (
            [te.to_dict() for te in profile.transformation_experience]
            if profile.transformation_experience
            else []
        )
        payload = {
            "full_name": profile.full_name,
            "current_title": profile.current_title,
            "current_company": profile.current_company,
            "seniority_level": profile.seniority_level,
            "industry_focus": list(profile.industry_focus or []),
            "years_experience_estimate": profile.years_experience_estimate,
            "career_history": career_history_data,
            "leadership_scope": leadership_scope_data,
            "transformation_experience": transformation_experience_data,
            "private_equity_exposure": [
                pe.model_dump() for pe in (profile.private_equity_exposure or [])
            ],
            "technical_capabilities": [
                tc for tc in (profile.technical_capabilities or [])
            ],
            "notable_achievements": [na for na in (profile.notable_achievements or [])],
            "risk_flags": [rf for rf in (profile.risk_flags or [])],
            "searchable_summary": profile.searchable_summary,
            "confidence_score": profile.confidence_score,
            "embedding": profile.embedding,
            "has_pe_experience": profile.has_pe_experience,
            "transformation_types": [tt for tt in (profile.transformation_types or [])],
        }

       

        if existing:
            for key, value in payload.items():
                setattr(existing, key, value)
            orm = existing
        else:
            orm = IntervieweeProfileORM(call_insight_id=call_insight_id, **payload)
            self.db.add(orm)

        self.db.commit()
        self.db.refresh(orm)
        return self._to_dict(orm)

    def get(self, call_insight_id: UUID) -> IntervieweeProfile | None:
        orm = self.db.get(IntervieweeProfileORM, call_insight_id)
        if not orm:
            return None
        return self._to_model(orm)

    def _to_model(self, orm: IntervieweeProfileORM) -> IntervieweeProfile:
        return IntervieweeProfile(
            full_name=orm.full_name,
            current_title=orm.current_title,
            current_company=orm.current_company,
            seniority_level=orm.seniority_level,
            industry_focus=list(orm.industry_focus or []),
        )
    def _to_dict(self, model: IntervieweeProfileORM) -> dict:
            return {
                "id": model.id.__str__() if model.id else None,
                "full_name": model.full_name,
                "current_title": model.current_title,
                "current_company": model.current_company,
                "seniority_level": model.seniority_level,
                "industry_focus": list(model.industry_focus or []),
                "years_experience_estimate": model.years_experience_estimate,
                "career_history": [{**ch} for ch in (model.career_history or [])],
                "leadership_scope": {**model.leadership_scope} if model.leadership_scope else {},
                "transformation_experience": [{**te} for te in (model.transformation_experience or [])],
                "private_equity_exposure": [{**pe} for pe in (model.private_equity_exposure or [])],
                "technical_capabilities": [tc for tc in (model.technical_capabilities or [])        ],
                "notable_achievements": [na for na in (model.notable_achievements or [])],
                "risk_flags": [rf for rf in (model.risk_flags or [])],
                "searchable_summary": model.searchable_summary,
                "confidence_score": model.confidence_score,
            }
