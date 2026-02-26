from sqlalchemy import text, select, bindparam, func
from sqlalchemy.ext.asyncio import AsyncSession
from sqlalchemy.orm import Session

from typing import List, Dict, Any
from pgvector.sqlalchemy import Vector


from app.db.models.interviewee_profile import IntervieweeProfileORM
from app.db.models.call_insight import CallInsightORM
from app.db.models.call import CallORM

class SearchRepository:

    def __init__(self, db: Session):
        self.db = db

    def search_interview_profiles(
        self,
        embedding: List[float],
        top_k: int,
    ) -> List[Dict[str, Any]]:
        embedding_param = bindparam(
        "embedding",
        value=embedding,
        type_=Vector(1536)
    )
        stmt = (
            select(
                IntervieweeProfileORM.id,
                IntervieweeProfileORM.full_name,
                IntervieweeProfileORM.current_title,
                IntervieweeProfileORM.current_company,
                IntervieweeProfileORM.seniority_level,
                IntervieweeProfileORM.searchable_summary,
                IntervieweeProfileORM.transformation_experience,
                IntervieweeProfileORM.private_equity_exposure,
               (1 - IntervieweeProfileORM.embedding.cosine_distance(embedding_param)).label("similarity")
            )
            .where(IntervieweeProfileORM.embedding.isnot(None))
            .order_by(IntervieweeProfileORM.embedding.cosine_distance(embedding_param))
            .limit(top_k)
            )

        result = self.db.execute(stmt)
        return result.mappings().all()

    def search_call_insights(
        self,
        embedding: List[float],
        top_k: int,
    ) -> List[Dict[str, Any]]:
        """Semantic search over call insights (summary-level)."""
        embedding_param = bindparam(
            "embedding",
            value=embedding,
            type_=Vector(1536),
        )

        stmt = (
            select(
                CallInsightORM.id.label("insight_id"),
                CallInsightORM.call_id,
                CallInsightORM.summary,
                CallInsightORM.tags,
                CallInsightORM.call_type,
                CallORM.title.label("call_title"),
                CallORM.call_date,
                CallORM.status.label("call_status"),
                (1 - CallInsightORM.embedding.cosine_distance(embedding_param)).label(
                    "similarity"
                ),
            )
            .select_from(CallInsightORM)
            .join(CallORM, CallORM.id == CallInsightORM.call_id)
            .where(CallInsightORM.embedding.isnot(None))
            .order_by(CallInsightORM.embedding.cosine_distance(embedding_param))
            .limit(top_k)
        )

        result = self.db.execute(stmt)
        return result.mappings().all()

    def get_aggregation_stats(self) -> Dict[str, Any]:
        """Return basic stats used for aggregation-style questions."""
        total_calls = self.db.query(func.count(CallORM.id)).scalar() or 0
        analyzed_calls = (
            self.db.query(func.count(CallORM.id))
            .filter(CallORM.status == "analyzed")
            .scalar()
            or 0
        )
        total_insights = self.db.query(func.count(CallInsightORM.id)).scalar() or 0
        total_profiles = (
            self.db.query(func.count(IntervieweeProfileORM.id)).scalar() or 0
        )
        calls_by_status_rows = (
            self.db.query(CallORM.status, func.count(CallORM.id))
            .group_by(CallORM.status)
            .all()
        )
        calls_by_status = {status: int(cnt) for status, cnt in calls_by_status_rows}

        return {
            "total_calls": int(total_calls),
            "analyzed_calls": int(analyzed_calls),
            "total_insights": int(total_insights),
            "total_interviewee_profiles": int(total_profiles),
            "calls_by_status": calls_by_status,
        }