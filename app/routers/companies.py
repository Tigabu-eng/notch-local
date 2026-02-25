"""Companies router for Company Research Mapping Tool.

Provides CRUD endpoints for managing company data.
Uses in-memory storage for development/testing purposes.
"""

import re
import uuid
from datetime import date, datetime, timedelta, timezone
from typing import Annotated, Any

from fastapi import APIRouter, HTTPException, Path, Query, status

from app.models import (
    CompanyCreate,
    CompanyResponse,
    CompanyUpdate,
    Division,
    Executive,
    ExecutiveV2,
    RefreshResult,
    Role,
    RoleChange,
    RoleHolder,
    STANDARD_ROLES,
    ValidationMetadata,
)

from fastapi import Depends
from sqlalchemy.orm import Session
from app.db.deps import get_db
from app.repositories.company_repository_sqlalchemy import CompanyRepositorySQLAlchemy


# Regex pattern for valid company IDs (alphanumeric with underscores)
COMPANY_ID_PATTERN = re.compile(r"^[a-zA-Z0-9_-]{1,50}$")

router = APIRouter()

# In-memory storage for companies
# Key: company_id, Value: CompanyResponse
_companies_db: dict[str, CompanyResponse] = {}

# In-memory storage for custom roles per company
# Key: company_id, Value: list of Role
_custom_roles_db: dict[str, list[Role]] = {}

# In-memory storage for role holders per company
# Key: company_id, Value: dict of role_id -> list of RoleHolder
_role_holders_db: dict[str, dict[str, list[RoleHolder]]] = {}

# In-memory storage for divisions per company
# Key: company_id, Value: list of Division
_divisions_db: dict[str, list[Division]] = {}

# In-memory storage for executives with division assignments (ExecutiveV2)
# Key: company_id, Value: list of ExecutiveV2
_executives_v2_db: dict[str, list[ExecutiveV2]] = {}

# In-memory storage for validation metadata per company
# Key: company_id, Value: ValidationMetadata
_validation_db: dict[str, ValidationMetadata] = {}


def _initialize_mock_data() -> None:
    """Initialize the in-memory database with mock data."""
    if _companies_db:
        return  # Already initialized

    mock_companies = [
        CompanyResponse(
            id="comp_001",
            name="Commercial Paving Inc",
            ceo=[
                Executive(name="Ed Campbell", title="CEO", start_year=2008, end_year=None),
                Executive(name="Tom York", title="CEO", start_year=2001, end_year=2008),
            ],
            c_level=[
                Executive(name="Sarah Mitchell", title="CFO", start_year=2015, end_year=None),
                Executive(name="James Rodriguez", title="COO", start_year=2018, end_year=None),
            ],
            senior_level=[
                Executive(name="Michael Chen", title="SVP Operations", start_year=2020, end_year=None),
                Executive(name="Lisa Thompson", title="VP Sales", start_year=2019, end_year=None),
            ],
            employees=450,
            ownership="Tenex Capital",
            acquisition_date=2021,
            subsector="Commercial Paving",
            notes="Leading commercial paving contractor in the Southeast region",
            updated=datetime(2024, 1, 15, 10, 30, 0),
            network_status="in_network",
            contact_status="contacted_no_response",
        ),
        CompanyResponse(
            id="comp_002",
            name="Asphalt Solutions LLC",
            ceo=[
                Executive(name="Robert Williams", title="CEO", start_year=2012, end_year=None),
            ],
            c_level=[
                Executive(name="Jennifer Adams", title="CFO", start_year=2016, end_year=None),
            ],
            senior_level=[
                Executive(name="David Park", title="VP Engineering", start_year=2018, end_year=None),
            ],
            employees=280,
            ownership="Private",
            acquisition_date=None,
            subsector="Commercial Paving",
            notes="Specializes in highway and airport runway paving",
            updated=datetime(2024, 1, 10, 14, 45, 0),
            network_status="out_of_network",
            contact_status="not_contacted",
        ),
    ]

    for company in mock_companies:
        _companies_db[company.id] = company

    # Initialize mock validation data
    _validation_db["comp_001"] = ValidationMetadata(
        last_validated=datetime(2024, 1, 15, 10, 30, 0, tzinfo=timezone.utc),
        confidence=0.85,
        needs_refresh=False,
    )
    _validation_db["comp_002"] = ValidationMetadata(
        last_validated=datetime(2023, 6, 1, 9, 0, 0, tzinfo=timezone.utc),
        confidence=0.6,
        needs_refresh=True,
    )


# Initialize mock data on module load
_initialize_mock_data()


def _validate_company_id(company_id: str) -> None:
    """Validate company ID format to prevent injection attacks."""
    if not COMPANY_ID_PATTERN.match(company_id):
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="Invalid company ID format",
        )


def _validate_role_id(role_id: str) -> None:
    """Validate role_id format."""
    if not re.match(r'^[a-z][a-z0-9_]{0,49}$', role_id):
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="Invalid role_id format. Must start with lowercase letter, contain only lowercase letters, numbers, and underscores, max 50 chars.",
        )


def _validate_division_id(division_id: str) -> None:
    """Validate division_id format."""
    if not re.match(r'^[a-zA-Z][a-zA-Z0-9_-]{0,99}$', division_id):
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="Invalid division_id format. Must start with a letter, contain only letters, numbers, underscores, and hyphens, max 100 chars.",
        )


# ============================================================================
# Static path endpoints MUST be defined before dynamic path endpoints
# ============================================================================


@router.get(
    "/companies/stale",
    response_model=list[CompanyResponse],
    status_code=status.HTTP_200_OK,
    summary="List stale companies",
    description="List companies whose data needs refresh based on validation metadata or age threshold.",
)
async def list_stale_companies(
    days_threshold: Annotated[int, Query(ge=1, le=3650)] = 365,
) -> list[CompanyResponse]:
    """List companies whose data needs refresh.

    Returns companies where validation.needs_refresh is True
    or where last_validated is older than days_threshold.

    Args:
        days_threshold: Number of days after which data is considered stale (default 365).

    Returns:
        List of CompanyResponse objects that need data refresh.
    """
    stale_companies: list[CompanyResponse] = []
    threshold_date = datetime.now(timezone.utc) - timedelta(days=days_threshold)

    for company_id, company in _companies_db.items():
        validation = _validation_db.get(company_id)

        # Check if company needs refresh
        if validation is not None:
            # If needs_refresh flag is set, include it
            if validation.needs_refresh:
                stale_companies.append(company)
                continue

            # If last_validated is older than threshold, include it
            last_validated = validation.last_validated
            if last_validated.tzinfo is None:
                last_validated = last_validated.replace(tzinfo=timezone.utc)
            if last_validated < threshold_date:
                stale_companies.append(company)
        else:
            # No validation data means we should refresh
            stale_companies.append(company)

    return stale_companies


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
    """Get a single company by ID.

    Args:
        company_id: Unique identifier of the company.

    Returns:
        CompanyResponse with the company data.

    Raises:
        HTTPException: 400 if invalid ID format, 404 if company not found.
    """
    _validate_company_id(company_id)
    repo = CompanyRepositorySQLAlchemy(db)
    company = repo.get(company_id)
    if not company:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=f"Company with id '{company_id}' not found",
        )

    return company


@router.get(
    "/companies",
    response_model=list[CompanyResponse],
    status_code=status.HTTP_200_OK,
    summary="List all companies",
    description="Retrieve all companies from the database.",
)
async def list_companies(
    limit: Annotated[int, Query(ge=1, le=1000)] = 100,
    offset: Annotated[int, Query(ge=0)] = 0,
    db: Session = Depends(get_db),
) -> list[CompanyResponse]:
    """List all companies with pagination.

    Args:
        limit: Maximum number of companies to return (default 100, max 1000).
        offset: Number of companies to skip (default 0).

    Returns:
        List of CompanyResponse objects.
    """
    repo = CompanyRepositorySQLAlchemy(db)
    return repo.list(limit=limit, offset=offset)


@router.post(
    "/companies",
    response_model=CompanyResponse,
    status_code=status.HTTP_201_CREATED,
    summary="Create a new company",
    description="Create a new company entry with manual data entry.",
    responses={
        400: {"description": "Invalid company data"},
        409: {"description": "Company with this name already exists"},
    },
)
async def create_company(
    company_data: CompanyCreate, db: Session = Depends(get_db)
) -> CompanyResponse:
    """Create a new company.

    Args:
        company_data: Company data for creation.

    Returns:
        CompanyResponse with the created company including generated ID.

    Raises:
        HTTPException: 409 if company with same name already exists.
    """
    repo = CompanyRepositorySQLAlchemy(db)

    # Check for duplicate company name
    if repo.exists_name_ci(company_data.name):
        raise HTTPException(
            status_code=status.HTTP_409_CONFLICT,
            detail=f"Company with name '{company_data.name}' already exists",
        )

    # Generate unique ID
    company_id = f"comp_{uuid.uuid4().hex[:8]}"

    # # Create company response with ID and timestamp
    # company = CompanyResponse(
    #     id=company_id,
    #     name=company_data.name,
    #     ceo=company_data.ceo,
    #     c_level=company_data.c_level,
    #     senior_level=company_data.senior_level,
    #     employees=company_data.employees,
    #     ownership=company_data.ownership,
    #     acquisition_date=company_data.acquisition_date,
    #     subsector=company_data.subsector,
    #     notes=company_data.notes,
    #     updated=datetime.now(timezone.utc),
    #     network_status=company_data.network_status,
    #     contact_status=company_data.contact_status,
    # )

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
    """Update an existing company.

    Performs a partial update - only provided fields are updated.

    Args:
        company_id: Unique identifier of the company to update.
        company_data: Fields to update.

    Returns:
        CompanyResponse with the updated company data.

    Raises:
        HTTPException: 400 if invalid ID format, 404 if company not found, 409 if name conflicts.
    """
    _validate_company_id(company_id)
    repo = CompanyRepositorySQLAlchemy(db)
    existing = repo.get(company_id)
    if not existing:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=f"Company with id '{company_id}' not found",
        )

    # Check for name conflict if name is being updated
    if company_data.name is not None:
        if repo.exists_name_ci(company_data.name, exclude_id=company_id):
            raise HTTPException(
                status_code=status.HTTP_409_CONFLICT,
                detail=f"Company with name '{company_data.name}' already exists",
            )

    updated = repo.update(company_id, company_data)
    if not updated:
        # Defensive check - should not happen since we already checked existence
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
    company_id: Annotated[
        str,
        Path(min_length=1, max_length=50),
    ],
    db: Session = Depends(get_db),
) -> None:
    """Delete a company by ID.

    Args:
        company_id: Unique identifier of the company to delete.

    Raises:
        HTTPException: 400 if invalid ID format, 404 if company not found.
    """
    _validate_company_id(company_id)
    repo = CompanyRepositorySQLAlchemy(db)
    ok = repo.delete(company_id)

    if ok:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=f"Company with id '{company_id}' not found",
        )


@router.get(
    "/companies/{company_id}/executives",
    response_model=list[Executive],
    status_code=status.HTTP_200_OK,
    summary="Get company executives",
    description="Get executives for a company with filtering and sorting options.",
    responses={
        400: {"description": "Invalid company ID format"},
        404: {"description": "Company not found"},
    },
)
async def get_company_executives(
    company_id: Annotated[str, Path(min_length=1, max_length=50)],
    current_only: Annotated[bool, Query(description="Return only current executives")] = False,
    historical_only: Annotated[bool, Query(description="Return only historical executives")] = False,
    db: Session = Depends(get_db),
    include_historical: Annotated[bool, Query(description="Include historical executives (default: True)")] = True,
    sort_by_tenure: Annotated[bool, Query(description="Sort by tenure (current first, then by date)")] = True,
) -> list[Executive]:
    """Get all executives for a company with filtering options.

    Args:
        company_id: Unique identifier of the company.
        current_only: If True, return only current executives (end_year is None).
        historical_only: If True, return only historical executives (end_year is set).
        include_historical: If False, exclude historical executives. Default True.
        sort_by_tenure: If True, sort current first, then by start/end year. Default True.

    Returns:
        List of Executive objects sorted appropriately.

    Raises:
        HTTPException: 400 if invalid ID format or conflicting parameters,
                       404 if company not found.
    """
    _validate_company_id(company_id)
    if company_id not in _companies_db:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=f"Company with id '{company_id}' not found",
        )

    # Validate conflicting parameters
    if current_only and historical_only:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="Cannot specify both current_only and historical_only",
        )

    company = _companies_db[company_id]

    # Use the model's built-in methods for consistent sorting
    if current_only:
        return company.current_executives

    if historical_only:
        return company.historical_executives

    if not include_historical:
        return company.current_executives

    # Return all executives
    if sort_by_tenure:
        return company.get_executives_sorted()

    # No sorting - return in original order
    return company.all_executives


@router.get(
    "/companies/{company_id}/divisions",
    response_model=list[Division],
    status_code=status.HTTP_200_OK,
    summary="List company divisions",
    description="Get all divisions for a company.",
    responses={
        400: {"description": "Invalid company ID format"},
        404: {"description": "Company not found"},
    },
)
async def list_divisions(
    company_id: Annotated[str, Path(min_length=1, max_length=50)],
) -> list[Division]:
    """Get all divisions for a company.

    Args:
        company_id: Unique identifier of the company.

    Returns:
        List of Division objects.

    Raises:
        HTTPException: 400 if invalid ID format, 404 if company not found.
    """
    _validate_company_id(company_id)
    if company_id not in _companies_db:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=f"Company with id '{company_id}' not found",
        )

    return _divisions_db.get(company_id, [])


@router.get(
    "/companies/{company_id}/divisions/{division_id}/leadership",
    response_model=list[ExecutiveV2],
    status_code=status.HTTP_200_OK,
    summary="Get division leadership",
    description="Get executives for a specific division.",
    responses={
        400: {"description": "Invalid company ID format"},
        404: {"description": "Company or division not found"},
    },
)
async def get_division_leadership(
    company_id: Annotated[str, Path(min_length=1, max_length=50)],
    division_id: Annotated[str, Path(min_length=1, max_length=100)],
) -> list[ExecutiveV2]:
    """Get executives for a specific division.

    Args:
        company_id: Unique identifier of the company.
        division_id: Unique identifier of the division.

    Returns:
        List of ExecutiveV2 objects assigned to this division.

    Raises:
        HTTPException: 400 if invalid ID format, 404 if company or division not found.
    """
    _validate_company_id(company_id)
    _validate_division_id(division_id)
    if company_id not in _companies_db:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=f"Company with id '{company_id}' not found",
        )

    # Check if division exists
    divisions = _divisions_db.get(company_id, [])
    division_exists = any(d.id == division_id for d in divisions)
    if not division_exists:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=f"Division with id '{division_id}' not found in company '{company_id}'",
        )

    # Filter executives by division_id
    executives = _executives_v2_db.get(company_id, [])
    return [exec for exec in executives if exec.division_id == division_id]


@router.post(
    "/companies/{company_id}/divisions",
    response_model=Division,
    status_code=status.HTTP_201_CREATED,
    summary="Create a division",
    description="Create a new division for a company.",
    responses={
        400: {"description": "Invalid company ID format"},
        404: {"description": "Company not found"},
        409: {"description": "Division with this ID already exists"},
    },
)
async def create_division(
    company_id: Annotated[str, Path(min_length=1, max_length=50)],
    division: Division,
) -> Division:
    """Create a new division for a company.

    Args:
        company_id: Unique identifier of the company.
        division: Division data for creation.

    Returns:
        The created Division.

    Raises:
        HTTPException: 400 if invalid ID format, 404 if company not found, 409 if division ID exists.
    """
    _validate_company_id(company_id)
    if company_id not in _companies_db:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=f"Company with id '{company_id}' not found",
        )

    # Initialize divisions list if not exists
    if company_id not in _divisions_db:
        _divisions_db[company_id] = []

    # Check for duplicate division ID
    for existing_division in _divisions_db[company_id]:
        if existing_division.id == division.id:
            raise HTTPException(
                status_code=status.HTTP_409_CONFLICT,
                detail=f"Division with id '{division.id}' already exists in company '{company_id}'",
            )

    _divisions_db[company_id].append(division)
    return division


# ============================================================================
# Role Drilling Endpoints
# ============================================================================


@router.get(
    "/roles/standard",
    response_model=list[Role],
    status_code=status.HTTP_200_OK,
    summary="List standard roles",
    description="Get list of all standard predefined roles.",
)
async def list_standard_roles() -> list[Role]:
    """Get list of all standard roles.

    Returns:
        List of standard Role objects.
    """
    return STANDARD_ROLES


@router.get(
    "/companies/{company_id}/roles",
    response_model=list[Role],
    status_code=status.HTTP_200_OK,
    summary="List company roles",
    description="Get all roles at a company (standard + custom).",
    responses={
        400: {"description": "Invalid company ID format"},
        404: {"description": "Company not found"},
    },
)
async def list_company_roles(
    company_id: Annotated[str, Path(min_length=1, max_length=50)],
) -> list[Role]:
    """Get all roles at a company (standard + custom).

    Args:
        company_id: Unique identifier of the company.

    Returns:
        List of Role objects including standard and custom roles.

    Raises:
        HTTPException: 400 if invalid ID format, 404 if company not found.
    """
    _validate_company_id(company_id)
    if company_id not in _companies_db:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=f"Company with id '{company_id}' not found",
        )

    # Combine standard roles with any custom roles for this company
    custom_roles = _custom_roles_db.get(company_id, [])
    return STANDARD_ROLES + custom_roles


@router.get(
    "/companies/{company_id}/roles/{role_id}/timeline",
    response_model=list[RoleHolder],
    status_code=status.HTTP_200_OK,
    summary="Get role timeline",
    description="Get historical timeline of who held a specific role.",
    responses={
        400: {"description": "Invalid company ID format"},
        404: {"description": "Company or role not found"},
    },
)
async def get_role_timeline(
    company_id: Annotated[str, Path(min_length=1, max_length=50)],
    role_id: Annotated[str, Path(min_length=1, max_length=100)],
    start_year: int | None = None,
    end_year: int | None = None,
) -> list[RoleHolder]:
    """Get historical timeline of who held a specific role.

    Args:
        company_id: Unique identifier of the company.
        role_id: Unique identifier of the role.
        start_year: Optional start year to filter results.
        end_year: Optional end year to filter results.

    Returns:
        List of RoleHolder objects showing who held the role over time.

    Raises:
        HTTPException: 400 if invalid ID format, 404 if company or role not found.
    """
    _validate_company_id(company_id)
    _validate_role_id(role_id)
    if company_id not in _companies_db:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=f"Company with id '{company_id}' not found",
        )

    # Validate role exists (either standard or custom)
    all_role_ids = {role.id for role in STANDARD_ROLES}
    custom_roles = _custom_roles_db.get(company_id, [])
    all_role_ids.update(role.id for role in custom_roles)

    if role_id not in all_role_ids:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=f"Role with id '{role_id}' not found",
        )

    # Get role holders for this company and role
    company_role_holders = _role_holders_db.get(company_id, {})
    role_holders = company_role_holders.get(role_id, [])

    # Filter by date range if provided
    if start_year is not None or end_year is not None:
        filtered_holders = []
        for holder in role_holders:
            holder_start_year = holder.start_date.year
            holder_end_year = holder.end_date.year if holder.end_date else None

            # Check if the holder's tenure overlaps with the requested range
            if start_year is not None:
                # If holder ended before the start_year, exclude
                if holder_end_year is not None and holder_end_year < start_year:
                    continue
            if end_year is not None:
                # If holder started after the end_year, exclude
                if holder_start_year > end_year:
                    continue

            filtered_holders.append(holder)
        return filtered_holders

    return role_holders


@router.get(
    "/companies/{company_id}/roles/{role_id}/current",
    response_model=RoleHolder | None,
    status_code=status.HTTP_200_OK,
    summary="Get current role holder",
    description="Get current holder of a role.",
    responses={
        400: {"description": "Invalid company ID format"},
        404: {"description": "Company or role not found"},
    },
)
async def get_current_role_holder(
    company_id: Annotated[str, Path(min_length=1, max_length=50)],
    role_id: Annotated[str, Path(min_length=1, max_length=100)],
) -> RoleHolder | None:
    """Get current holder of a role.

    Args:
        company_id: Unique identifier of the company.
        role_id: Unique identifier of the role.

    Returns:
        RoleHolder object for the current holder, or None if role is vacant.

    Raises:
        HTTPException: 400 if invalid ID format, 404 if company or role not found.
    """
    _validate_company_id(company_id)
    _validate_role_id(role_id)
    if company_id not in _companies_db:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=f"Company with id '{company_id}' not found",
        )

    # Validate role exists (either standard or custom)
    all_role_ids = {role.id for role in STANDARD_ROLES}
    custom_roles = _custom_roles_db.get(company_id, [])
    all_role_ids.update(role.id for role in custom_roles)

    if role_id not in all_role_ids:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=f"Role with id '{role_id}' not found",
        )

    # Get role holders for this company and role
    company_role_holders = _role_holders_db.get(company_id, {})
    role_holders = company_role_holders.get(role_id, [])

    # Find current holder (where end_date is None)
    for holder in role_holders:
        if holder.end_date is None:
            return holder

    return None


@router.post(
    "/companies/{company_id}/roles",
    response_model=Role,
    status_code=status.HTTP_201_CREATED,
    summary="Create custom role",
    description="Create a custom role for a company.",
    responses={
        400: {"description": "Invalid company ID format"},
        404: {"description": "Company not found"},
        409: {"description": "Role with this ID already exists"},
    },
)
async def create_custom_role(
    company_id: Annotated[str, Path(min_length=1, max_length=50)],
    role: Role,
) -> Role:
    """Create a custom role for a company.

    Args:
        company_id: Unique identifier of the company.
        role: Role data for creation.

    Returns:
        The created Role object.

    Raises:
        HTTPException: 400 if invalid ID format, 404 if company not found,
                       409 if role with this ID already exists.
    """
    _validate_company_id(company_id)
    if company_id not in _companies_db:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=f"Company with id '{company_id}' not found",
        )

    # Check if role ID already exists in standard roles
    standard_role_ids = {r.id for r in STANDARD_ROLES}
    if role.id in standard_role_ids:
        raise HTTPException(
            status_code=status.HTTP_409_CONFLICT,
            detail=f"Role with id '{role.id}' already exists as a standard role",
        )

    # Check if role ID already exists in custom roles for this company
    if company_id not in _custom_roles_db:
        _custom_roles_db[company_id] = []

    custom_role_ids = {r.id for r in _custom_roles_db[company_id]}
    if role.id in custom_role_ids:
        raise HTTPException(
            status_code=status.HTTP_409_CONFLICT,
            detail=f"Role with id '{role.id}' already exists for this company",
        )

    # Mark as non-standard and store
    custom_role = Role(
        id=role.id,
        name=role.name,
        category=role.category,
        is_standard=False,
    )
    _custom_roles_db[company_id].append(custom_role)

    return custom_role


# ============================================================================
# Refresh and Validation Endpoints
# ============================================================================


@router.post(
    "/companies/{company_id}/refresh",
    response_model=RefreshResult,
    status_code=status.HTTP_200_OK,
    summary="Refresh company data",
    description="Re-fetch and validate company leadership data, comparing with previous data.",
    responses={
        400: {"description": "Invalid company ID format"},
        404: {"description": "Company not found"},
    },
)
async def refresh_company(
    company_id: Annotated[str, Path(min_length=1, max_length=50)],
) -> RefreshResult:
    """Re-fetch and validate company leadership data.

    Compares current data with fresh fetch and returns:
    - new_executives: executives found that weren't in previous data
    - departed_executives: executives no longer found
    - role_changes: changes in role assignments

    Args:
        company_id: Unique identifier of the company to refresh.

    Returns:
        RefreshResult containing detected changes.

    Raises:
        HTTPException: 400 if invalid ID format, 404 if company not found.
    """
    _validate_company_id(company_id)
    if company_id not in _companies_db:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=f"Company with id '{company_id}' not found",
        )

    # For POC, return a mock RefreshResult showing the structure
    # In a real implementation, this would:
    # 1. Fetch fresh data from external sources (LinkedIn, company website, etc.)
    # 2. Compare with existing data
    # 3. Identify new executives, departures, and role changes
    # 4. Optionally update the stored data

    # Update validation metadata to reflect the refresh
    _validation_db[company_id] = ValidationMetadata(
        last_validated=datetime.now(timezone.utc),
        confidence=0.9,
        needs_refresh=False,
    )

    # Return mock result demonstrating the response structure
    return RefreshResult(
        company_id=company_id,
        new_executives=[
            Executive(
                name="Jane Smith",
                title="VP Marketing",
                start_year=2024,
                end_year=None,
            ),
        ],
        departed_executives=[
            Executive(
                name="John Doe",
                title="VP Sales",
                start_year=2019,
                end_year=2024,
            ),
        ],
        role_changes=[
            RoleChange(
                role_id="coo",
                old_holder=Executive(
                    name="James Rodriguez",
                    title="COO",
                    start_year=2018,
                    end_year=2024,
                ),
                new_holder=Executive(
                    name="Michael Chen",
                    title="COO",
                    start_year=2024,
                    end_year=None,
                ),
                change_date=date.today(),
            ),
        ],
    )


@router.get(
    "/companies/{company_id}/validation",
    response_model=ValidationMetadata | None,
    status_code=status.HTTP_200_OK,
    summary="Get company validation status",
    description="Get validation metadata for a company including last validation date and confidence.",
    responses={
        400: {"description": "Invalid company ID format"},
        404: {"description": "Company not found"},
    },
)
async def get_company_validation_status(
    company_id: Annotated[str, Path(min_length=1, max_length=50)],
) -> ValidationMetadata | None:
    """Get validation metadata for a company.

    Args:
        company_id: Unique identifier of the company.

    Returns:
        ValidationMetadata if available, None if no validation data exists.

    Raises:
        HTTPException: 400 if invalid ID format, 404 if company not found.
    """
    _validate_company_id(company_id)
    if company_id not in _companies_db:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=f"Company with id '{company_id}' not found",
        )

    return _validation_db.get(company_id)
