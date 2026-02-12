from uuid import UUID

from sqlalchemy.orm import Session

from app.db.models.call import CallORM
from app.models import Call


class CallRepositorySQLAlchemy:
    def __init__(self, db: Session):
        self.db = db

    def create(self, call: Call) -> Call:
        orm = CallORM(
            id=call.id,
            title=call.title,
            description=call.description,
            call_date=call.call_date,
            transcript=call.transcript,
            status=call.status,
            created_at=call.created_at,
        )
        self.db.add(orm)
        self.db.commit()
        self.db.refresh(orm)
        return self._to_model(orm)

    def get(self, call_id: UUID) -> Call | None:
        orm = self.db.get(CallORM, call_id)
        if not orm:
            return None
        return self._to_model(orm)

    def list(
        self,
        *,
        status: str | None = None,
        limit: int = 50,
        offset: int = 0,
    ) -> list[Call]:
        q = self.db.query(CallORM)
        if status:
            q = q.filter(CallORM.status == status)

        q = q.order_by(CallORM.call_date.desc(), CallORM.created_at.desc())
        q = q.offset(offset).limit(limit)

        return [self._to_model(r) for r in q.all()]

    def update_status(self, call_id: UUID, status: str) -> Call | None:
        orm = self.db.get(CallORM, call_id)
        if not orm:
            return None
        orm.status = status
        self.db.commit()
        self.db.refresh(orm)
        return self._to_model(orm)

    def _to_model(self, orm: CallORM) -> Call:
        return Call(
            id=orm.id,
            title=orm.title,
            description=orm.description,
            call_date=orm.call_date,
            transcript=orm.transcript,
            status=orm.status,
            created_at=orm.created_at,
        )
