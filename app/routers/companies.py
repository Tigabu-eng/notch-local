"""Companies router for Company Research Mapping Tool.

CRUD endpoints for managing company data.

NOTE:
- This router uses the SQLAlchemy repository (persistent storage).
- Any legacy in-memory/mock storage from earlier iterations has been removed.
"""

from __future__ import annotations

import re
import uuid
from datetime import datetime, timedelta, timezone
from typing import Annotated

from fastapi import APIRouter, Depends, HTTPException, Path, Query, status
from sqlalchemy.orm import Session

from app.db.deps import get_db
from app.models import CompanyCreate, CompanyResponse, CompanyUpdate, Executive
from app.repositories.company_repository_sqlalchemy import CompanyRepositorySQLAlchemy

router = APIRouter()

# Alphanumeric + underscore/hyphen IDs (matches existing pattern used across the codebase)
COMPANY_ID_PATTERN = re.compile(r"^[a-zA-Z0-9_-]{1,50}$")


def _validate_company_id(company_id: str) -> None:
    """Validate company ID format to reduce accidental misuse and basic injection attempts."""
    if not COMPANY_ID_PATTERN.match(company_id):
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="Invalid company ID format",
        )


@router.get(
    "/companies",
    response_model=list[CompanyResponse],
    status_code=status.HTTP_200_OK,
    summary="List all companies",
    description="Retrieve companies from the database with pagination.",
)
async def list_companies(
    limit: Annotated[int, Query(ge=1, le=1000)] = 100,
    offset: Annotated[int, Query(ge=0)] = 0,
    db: Session = Depends(get_db),
) -> list[CompanyResponse]:
    repo = CompanyRepositorySQLAlchemy(db)
    return repo.list(limit=limit, offset=offset)


@router.get(
    "/companies/stale",
    response_model=list[CompanyResponse],
    status_code=status.HTTP_200_OK,
    summary="List stale companies",
    description="List companies whose data appears stale based on last update timestamp.",
)
async def list_stale_companies(
    days_threshold: Annotated[int, Query(ge=1, le=3650)] = 365,
    limit: Annotated[int, Query(ge=1, le=1000)] = 200,
    offset: Annotated[int, Query(ge=0)] = 0,
    db: Session = Depends(get_db),
) -> list[CompanyResponse]:
    """
    A 'stale' company is one where `updated` is missing or older than `days_threshold`.

    We keep this endpoint lightweight by filtering at the API layer. If you want this
    pushed into SQL for large datasets, add a repo method like `list_stale(...)`.
    """
    repo = CompanyRepositorySQLAlchemy(db)
    companies = repo.list(limit=limit, offset=offset)

    threshold = datetime.now(timezone.utc) - timedelta(days=days_threshold)

    stale: list[CompanyResponse] = []
    for c in companies:
        updated = getattr(c, "updated", None)
        if updated is None:
            stale.append(c)
            continue
        if updated.tzinfo is None:
            updated = updated.replace(tzinfo=timezone.utc)
        if updated < threshold:
            stale.append(c)

    return stale


@router.get(
    "/companies/{company_id}",
    response_model=CompanyResponse,
    status_code=status.HTTP_200_OK,
    summary="Get company by ID",
    description="Retrieve a single company by its unique identifier.",
    responses={
        400: {"description": "Invalid company ID format"},
        404: {"description": "Company not found"},
    },
)
async def get_company(
    company_id: Annotated[str, Path(min_length=1, max_length=50)],
    db: Session = Depends(get_db),
) -> CompanyResponse:
    _validate_company_id(company_id)
    repo = CompanyRepositorySQLAlchemy(db)
    company = repo.get(company_id)
    if not company:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=f"Company with id '{company_id}' not found",
        )
    return company


@router.post(
    "/companies",
    response_model=CompanyResponse,
    status_code=status.HTTP_201_CREATED,
    summary="Create a new company",
    description="Create a new company entry.",
    responses={
        400: {"description": "Invalid company data"},
        409: {"description": "Company with this name already exists"},
    },
)
async def create_company(
    company_data: CompanyCreate,
    db: Session = Depends(get_db),
) -> CompanyResponse:
    repo = CompanyRepositorySQLAlchemy(db)

    # Case-insensitive uniqueness check on name
    if repo.exists_name_ci(company_data.name):
        raise HTTPException(
            status_code=status.HTTP_409_CONFLICT,
            detail=f"Company with name '{company_data.name}' already exists",
        )

    company_id = f"comp_{uuid.uuid4().hex[:8]}"
    return repo.create(company_data, company_id=company_id)


@router.put(
    "/companies/{company_id}",
    response_model=CompanyResponse,
    status_code=status.HTTP_200_OK,
    summary="Update a company",
    description="Update an existing company's data. Supports partial updates.",
    responses={
        400: {"description": "Invalid company ID format"},
        404: {"description": "Company not found"},
        409: {"description": "Company name conflict"},
    },
)
async def update_company(
    company_id: Annotated[str, Path(min_length=1, max_length=50)],
    company_data: CompanyUpdate,
    db: Session = Depends(get_db),
) -> CompanyResponse:
    _validate_company_id(company_id)
    repo = CompanyRepositorySQLAlchemy(db)

    existing = repo.get(company_id)
    if not existing:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=f"Company with id '{company_id}' not found",
        )

    if company_data.name is not None:
        if repo.exists_name_ci(company_data.name, exclude_id=company_id):
            raise HTTPException(
                status_code=status.HTTP_409_CONFLICT,
                detail=f"Company with name '{company_data.name}' already exists",
            )

    updated = repo.update(company_id, company_data)
    if not updated:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=f"Company with id '{company_id}' not found",
        )

    return updated


@router.delete(
    "/companies/{company_id}",
    status_code=status.HTTP_204_NO_CONTENT,
    summary="Delete a company",
    description="Delete a company by its unique identifier.",
    responses={
        400: {"description": "Invalid company ID format"},
        404: {"description": "Company not found"},
    },
)
async def delete_company(
    company_id: Annotated[str, Path(min_length=1, max_length=50)],
    db: Session = Depends(get_db),
) -> None:
    _validate_company_id(company_id)
    repo = CompanyRepositorySQLAlchemy(db)
    ok = repo.delete(company_id)

    if not ok:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=f"Company with id '{company_id}' not found",
        )
    return None


@router.get(
    "/companies/{company_id}/executives",
    response_model=list[Executive],
    status_code=status.HTTP_200_OK,
    summary="Get company executives",
    description="Get executives for a company with basic filtering and sorting options.",
    responses={
        400: {"description": "Invalid company ID format"},
        404: {"description": "Company not found"},
    },
)
async def get_company_executives(
    company_id: Annotated[str, Path(min_length=1, max_length=50)],
    current_only: Annotated[bool, Query(description="Return only current executives")] = False,
    historical_only: Annotated[bool, Query(description="Return only historical executives")] = False,
    include_historical: Annotated[bool, Query(description="Include historical executives (default: True)")] = True,
    sort_by_tenure: Annotated[bool, Query(description="Sort with current first, then most recent tenure")] = True,
    db: Session = Depends(get_db),
) -> list[Executive]:
    _validate_company_id(company_id)
    if current_only and historical_only:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="current_only and historical_only cannot both be true",
        )

    repo = CompanyRepositorySQLAlchemy(db)
    company = repo.get(company_id)
    if not company:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=f"Company with id '{company_id}' not found",
        )

    # CompanyResponse has ceo/c_level/senior_level fields.
    all_executives: list[Executive] = (company.ceo or []) + (company.c_level or []) + (company.senior_level or [])

    if not include_historical:
        all_executives = [e for e in all_executives if e.end_year is None]

    if current_only:
        all_executives = [e for e in all_executives if e.end_year is None]
    elif historical_only:
        all_executives = [e for e in all_executives if e.end_year is not None]

    if sort_by_tenure:
        # Current first; within current, highest start_year first.
        # Within historical, most recent end_year first, then start_year.
        def sort_key(e: Executive):
            is_historical = e.end_year is not None
            start = e.start_year or 0
            end = e.end_year or 0
            return (
                is_historical,          # False (current) first
                -(end if is_historical else start),
                -start,
            )

        all_executives = sorted(all_executives, key=sort_key)

    return all_executives