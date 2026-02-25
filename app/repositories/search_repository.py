from sqlalchemy import text, select, literal, bindparam
from sqlalchemy.ext.asyncio import AsyncSession
from sqlalchemy.orm import Session

from typing import List, Dict, Any
from pgvector.sqlalchemy import Vector


from app.db.models.interviewee_profile import IntervieweeProfileORM

class SearchRepository:

    def __init__(self, db: Session):
        self.db = db

    def search_interview_profiles(
        self,
        embedding: List[float],
        top_k: int,
    ) -> List[Dict[str, Any]]:
        # print("sub-1")
        # # Improve recall
        # self.db.execute(text("SET ivfflat.probes = 10"))

        # result = self.db.execute(
        #     text("""
        #         SELECT 
        #             id,
        #             full_name,
        #             current_title,
        #             current_company,
        #             seniority_level,
        #             searchable_summary,
        #             transformation_experience,
        #             private_equity_exposure,
        #             1 - (embedding <=> CAST(:embedding AS vector(1536))) AS similarity
        #         FROM notch.interviewee_profiles
        #         ORDER BY embedding <=> CAST(:embedding AS vector(1536))
        #         LIMIT :limit
        #     """),
        #     {
        #         "embedding": embedding,
        #         "limit": top_k
        #     }
        # )

        # rows = result.mappings().all()
        # return rows
        # embedding_vector = literal(embedding, type_=Vector(1536))
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