from datetime import datetime, timezone
from sqlalchemy import func
from sqlalchemy.orm import Session

from app.db.models.company import CompanyORM
from app.models import CompanyCreate, CompanyResponse, CompanyUpdate, Executive


def _dump_exec_list(execs: list[Executive]) -> list[dict]:
    # store snake_case keys; stable for your Pydantic models
    return [e.model_dump() for e in execs]


def _to_response(orm: CompanyORM) -> CompanyResponse:
    return CompanyResponse(
        id=orm.id,
        name=orm.name,
        ceo=[Executive(**x) for x in (orm.ceo or [])],
        c_level=[Executive(**x) for x in (orm.c_level or [])],
        senior_level=[Executive(**x) for x in (orm.senior_level or [])],
        employees=orm.employees,
        ownership=orm.ownership,
        acquisition_date=orm.acquisition_date,
        subsector=orm.subsector,
        notes=orm.notes,
        updated=orm.updated,
        network_status=orm.network_status,
        contact_status=orm.contact_status,
    )


class CompanyRepositorySQLAlchemy:
    def __init__(self, db: Session):
        self.db = db

    def get(self, company_id: str) -> CompanyResponse | None:
        orm = self.db.get(CompanyORM, company_id)
        return _to_response(orm) if orm else None

    def list(self, limit: int, offset: int) -> list[CompanyResponse]:
        rows = (
            self.db.query(CompanyORM)
            .order_by(CompanyORM.updated.desc().nullslast(), CompanyORM.name.asc())
            .offset(offset)
            .limit(limit)
            .all()
        )
        return [_to_response(r) for r in rows]

    def exists_name_ci(self, name: str, exclude_id: str | None = None) -> bool:
        q = self.db.query(CompanyORM).filter(func.lower(CompanyORM.name) == name.lower())
        if exclude_id:
            q = q.filter(CompanyORM.id != exclude_id)
        return self.db.query(q.exists()).scalar()

    def create(self, data: CompanyCreate, company_id: str) -> CompanyResponse:
        orm = CompanyORM(
            id=company_id,
            name=data.name,
            ceo=_dump_exec_list(data.ceo),
            c_level=_dump_exec_list(data.c_level),
            senior_level=_dump_exec_list(data.senior_level),
            employees=data.employees,
            ownership=data.ownership,
            acquisition_date=data.acquisition_date,
            subsector=data.subsector,
            notes=data.notes,
            network_status=data.network_status,
            contact_status=data.contact_status,
            updated=datetime.now(timezone.utc),
        )
        self.db.add(orm)
        self.db.commit()
        self.db.refresh(orm)
        return _to_response(orm)

    def update(self, company_id: str, data: CompanyUpdate) -> CompanyResponse | None:
        orm = self.db.get(CompanyORM, company_id)
        if not orm:
            return None

        patch = data.model_dump(exclude_unset=True)

        if "name" in patch:
            orm.name = patch["name"]
        if "ceo" in patch:
            orm.ceo = _dump_exec_list(patch["ceo"] or [])
        if "c_level" in patch:
            orm.c_level = _dump_exec_list(patch["c_level"] or [])
        if "senior_level" in patch:
            orm.senior_level = _dump_exec_list(patch["senior_level"] or [])

        for k in ["employees", "ownership", "acquisition_date", "subsector", "notes", "network_status", "contact_status"]:
            if k in patch:
                setattr(orm, k, patch[k])

        orm.updated = datetime.now(timezone.utc)

        self.db.commit()
        self.db.refresh(orm)
        return _to_response(orm)

    def delete(self, company_id: str) -> bool:
        orm = self.db.get(CompanyORM, company_id)
        if not orm:
            return False
        self.db.delete(orm)
        self.db.commit()
        return True
