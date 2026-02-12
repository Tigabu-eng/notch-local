"""Search router for Company Research Mapping Tool.

Provides endpoints for searching companies by name or industry.
Uses Tavily/SerpAPI when configured, falls back to mock data.
"""

import logging
from datetime import datetime
from typing import Annotated, Literal

from fastapi import APIRouter, HTTPException, Query, status
from pydantic import BaseModel, ConfigDict, Field

from app.models import CompanyResponse, Executive, TimeRangeQuery, to_camel
from app.services import get_search_service

logger = logging.getLogger(__name__)


def _sanitize_for_log(value: str, max_length: int = 100) -> str:
    """Sanitize a string for safe logging to prevent log injection."""
    # Remove newlines, carriage returns, and other control characters
    sanitized = "".join(c if c.isprintable() and c not in "\n\r\t" else " " for c in value)
    # Truncate to max length
    if len(sanitized) > max_length:
        sanitized = sanitized[:max_length] + "..."
    return sanitized

router = APIRouter()


class SearchRequest(BaseModel):
    """Request model for company search.

    Attributes:
        query: Search query string.
        search_type: Type of search - 'company' for company name,
            'industry' for industry/subsector search.
    """

    query: str = Field(..., min_length=1, max_length=500, description="Search query string")
    search_type: Literal["company", "industry"] = Field(
        default="company", description="Type of search to perform"
    )


class SearchRequestV2(SearchRequest):
    """Enhanced search request with time bounds and role filtering.

    Attributes:
        query: Search query string.
        search_type: Type of search - 'company' for company name,
            'industry' for industry/subsector search.
        time_range: Optional time range to filter executives by tenure period.
        role_filter: Optional list of roles to filter by (e.g., ["ceo", "cfo"]).
    """

    time_range: TimeRangeQuery | None = Field(
        default=None,
        description="Filter executives by tenure period (start_year/end_year)",
    )
    role_filter: list[str] | None = Field(
        default=None,
        description="Filter executives by role (e.g., ['ceo', 'cfo'])",
    )


class SearchResponse(BaseModel):
    """Response model for search results.

    Attributes:
        results: List of matching companies.
        total: Total number of results.
        query: Original search query.
        search_type: Type of search performed.
        source: Data source used ('api' or 'mock').
    """

    model_config = ConfigDict(
        alias_generator=to_camel,
        populate_by_name=True,
        serialize_by_alias=True,
    )

    results: list[CompanyResponse]
    total: int
    query: str
    search_type: str
    source: str = Field(default="mock", description="Data source: 'api' or 'mock'")


# Mock data for testing
MOCK_COMPANIES: list[CompanyResponse] = [
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
    CompanyResponse(
        id="comp_003",
        name="Highway Contractors Group",
        ceo=[
            Executive(name="Amanda Foster", title="CEO", start_year=2019, end_year=None),
            Executive(name="Richard Brooks", title="CEO", start_year=2010, end_year=2019),
        ],
        c_level=[
            Executive(name="Mark Sullivan", title="CFO", start_year=2017, end_year=None),
            Executive(name="Karen White", title="CTO", start_year=2020, end_year=None),
        ],
        senior_level=[
            Executive(name="Brian Johnson", title="President, Eastern Region", start_year=2018, end_year=None),
        ],
        employees=750,
        ownership="Sterling Partners",
        acquisition_date=2020,
        subsector="Infrastructure Construction",
        notes="Major highway construction and maintenance contractor",
        updated=datetime(2024, 1, 12, 9, 15, 0),
        network_status="in_network",
        contact_status="available",
    ),
    CompanyResponse(
        id="comp_004",
        name="Precision Striping & Marking",
        ceo=[
            Executive(name="Thomas Grant", title="CEO", start_year=2015, end_year=None),
        ],
        c_level=[
            Executive(name="Michelle Lee", title="CFO", start_year=2018, end_year=None),
        ],
        senior_level=[],
        employees=120,
        ownership="Founder-owned",
        acquisition_date=None,
        subsector="Traffic Marking Services",
        notes="Specializes in parking lot and road striping",
        updated=datetime(2024, 1, 8, 11, 0, 0),
        network_status="out_of_network",
        contact_status="not_contacted",
    ),
    CompanyResponse(
        id="comp_005",
        name="Concrete Masters International",
        ceo=[
            Executive(name="Patricia Gonzalez", title="CEO", start_year=2017, end_year=None),
        ],
        c_level=[
            Executive(name="Steven Clark", title="CFO", start_year=2019, end_year=None),
            Executive(name="Nancy Rivera", title="COO", start_year=2020, end_year=None),
        ],
        senior_level=[
            Executive(name="Charles Kim", title="SVP Business Development", start_year=2021, end_year=None),
        ],
        employees=520,
        ownership="Blackstone",
        acquisition_date=2022,
        subsector="Concrete Construction",
        notes="Commercial and industrial concrete contractor",
        updated=datetime(2024, 1, 14, 16, 30, 0),
        network_status="in_network",
        contact_status="conflicted_not_interested",
    ),
]


def _search_mock_companies(query: str, search_type: str) -> list[CompanyResponse]:
    """Search mock companies for testing/fallback."""
    query_lower = query.lower().strip()

    if search_type == "company":
        return [c for c in MOCK_COMPANIES if query_lower in c.name.lower()]
    elif search_type == "industry":
        return [c for c in MOCK_COMPANIES if c.subsector and query_lower in c.subsector.lower()]
    return []


def _executive_matches_time_range(exec: Executive, time_range: TimeRangeQuery) -> bool:
    """Check if an executive's tenure overlaps with the given time range.

    An executive matches if their tenure period overlaps with the query period.
    For example, if querying 2015-2020:
    - CEO from 2010-2018 matches (overlaps)
    - CEO from 2018-Present matches (overlaps)
    - CEO from 2000-2010 does not match (ended before)
    - CEO from 2022-Present does not match (started after)

    Executives with unknown start_year are included if they are current or
    their end_year is within the query range (conservative matching).
    """
    current_year = datetime.now().year

    # Executive's tenure end year (use current year if still active)
    exec_end = exec.end_year if exec.end_year is not None else current_year

    # Query's end year (use current year if not specified)
    query_end = time_range.end_year if time_range.end_year is not None else current_year

    # Handle unknown start_year: include if executive is current or end_year is in range
    if exec.start_year is None:
        # Include current executives (they're likely still relevant)
        if exec.is_current:
            return True
        # Include if their end_year falls within or after the query start
        return exec_end >= time_range.start_year

    # Check for overlap: exec started before/during query end AND exec ended after/during query start
    return exec.start_year <= query_end and exec_end >= time_range.start_year


def _executive_matches_role_filter(exec: Executive, role_filter: list[str]) -> bool:
    """Check if an executive's title matches any of the filtered roles.

    Performs case-insensitive matching against both role IDs (e.g., "ceo")
    and common title patterns (e.g., "Chief Executive Officer").
    """
    title_lower = exec.title.lower()

    # Role ID to title pattern mappings
    role_patterns: dict[str, list[str]] = {
        "ceo": ["ceo", "chief executive officer"],
        "cfo": ["cfo", "chief financial officer"],
        "coo": ["coo", "chief operating officer"],
        "cto": ["cto", "chief technology officer"],
        "cmo": ["cmo", "chief marketing officer"],
        "cio": ["cio", "chief information officer"],
        "president": ["president"],
        "svp": ["svp", "senior vice president"],
        "vp": ["vp", "vice president"],
    }

    for role in role_filter:
        role_lower = role.lower()
        # Check against known patterns
        if role_lower in role_patterns:
            if any(pattern in title_lower for pattern in role_patterns[role_lower]):
                return True
        # Direct match against role ID or title contains the role
        if role_lower in title_lower:
            return True

    return False


def _filter_company_executives(
    company: CompanyResponse,
    time_range: TimeRangeQuery | None,
    role_filter: list[str] | None,
) -> CompanyResponse:
    """Filter a company's executives based on time range and role criteria.

    Returns a new CompanyResponse with filtered executive lists.
    """
    if time_range is None and role_filter is None:
        return company

    def filter_exec_list(execs: list[Executive]) -> list[Executive]:
        filtered = execs
        if time_range is not None:
            filtered = [e for e in filtered if _executive_matches_time_range(e, time_range)]
        if role_filter is not None:
            filtered = [e for e in filtered if _executive_matches_role_filter(e, role_filter)]
        return filtered

    return CompanyResponse(
        id=company.id,
        name=company.name,
        ceo=filter_exec_list(company.ceo),
        c_level=filter_exec_list(company.c_level),
        senior_level=filter_exec_list(company.senior_level),
        employees=company.employees,
        ownership=company.ownership,
        acquisition_date=company.acquisition_date,
        subsector=company.subsector,
        notes=company.notes,
        updated=company.updated,
        network_status=company.network_status,
        contact_status=company.contact_status,
    )


@router.post(
    "/search",
    response_model=SearchResponse,
    status_code=status.HTTP_200_OK,
    summary="Search companies",
    description="Search for companies by name or industry/subsector.",
)
async def search_companies(request: SearchRequest) -> SearchResponse:
    """Search companies by name or industry.

    Uses Tavily/SerpAPI when configured, falls back to mock data.

    Args:
        request: Search request containing query and search_type.

    Returns:
        SearchResponse with matching companies.

    Raises:
        HTTPException: If search query is invalid.
    """
    query = request.query.strip()

    if not query:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="Search query cannot be empty",
        )

    # Try API search if configured
    search_service = get_search_service()

    if search_service.is_configured:
        logger.info("Using search APIs for query: %s", _sanitize_for_log(query))
        try:
            results = await search_service.search_companies(query, request.search_type)

            if results:
                logger.info(f"Returning {len(results)} results from API")
                return SearchResponse(
                    results=results,
                    total=len(results),
                    query=query,
                    search_type=request.search_type,
                    source="api",
                )

            logger.info("API returned no results, falling back to mock data")

        except Exception as e:
            logger.error(f"API search failed: {e}, falling back to mock data")

    # Use mock data as fallback
    logger.info("Using mock data for query: %s", _sanitize_for_log(query))
    results = _search_mock_companies(query, request.search_type)

    return SearchResponse(
        results=results,
        total=len(results),
        query=query,
        search_type=request.search_type,
        source="mock",
    )


@router.post(
    "/search/v2",
    response_model=SearchResponse,
    status_code=status.HTTP_200_OK,
    summary="Search companies with time bounds and role filtering",
    description="Search for companies with optional time range and role filters for executives.",
)
async def search_companies_v2(request: SearchRequestV2) -> SearchResponse:
    """Search companies with time bounds and role filtering.

    If time_range is provided, only return executives who held roles
    during the specified period.

    If role_filter is provided, only return executives in those roles.

    Args:
        request: Enhanced search request with optional time_range and role_filter.

    Returns:
        SearchResponse with matching companies, filtered by executive criteria.

    Raises:
        HTTPException: If search query is invalid.
    """
    query = request.query.strip()

    if not query:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="Search query cannot be empty",
        )

    # Try API search if configured
    search_service = get_search_service()
    source = "mock"
    results: list[CompanyResponse] = []

    if search_service.is_configured:
        logger.info("Using search APIs for v2 query: %s", _sanitize_for_log(query))
        try:
            api_results = await search_service.search_companies(query, request.search_type)

            if api_results:
                results = api_results
                source = "api"
                logger.info(f"Got {len(results)} results from API for v2 search")
            else:
                logger.info("API returned no results for v2 search, falling back to mock data")

        except Exception as e:
            logger.error(f"API search failed for v2: {e}, falling back to mock data")

    # Use mock data as fallback if no API results
    if not results:
        logger.info("Using mock data for v2 query: %s", _sanitize_for_log(query))
        results = _search_mock_companies(query, request.search_type)

    # Apply time range and role filtering
    if request.time_range is not None or request.role_filter is not None:
        filtered_results = []
        for company in results:
            filtered_company = _filter_company_executives(
                company, request.time_range, request.role_filter
            )
            # Only include companies that have at least one matching executive
            if (
                filtered_company.ceo
                or filtered_company.c_level
                or filtered_company.senior_level
            ):
                filtered_results.append(filtered_company)
        results = filtered_results

    return SearchResponse(
        results=results,
        total=len(results),
        query=query,
        search_type=request.search_type,
        source=source,
    )


@router.get(
    "/search/status",
    response_model=dict,
    status_code=status.HTTP_200_OK,
    summary="Get search service status",
    description="Check if search APIs are configured.",
)
async def get_search_status() -> dict:
    """Get the current status of the search service."""
    search_service = get_search_service()

    return {
        "configured": search_service.is_configured,
        "tavily": search_service.has_tavily,
        "serpapi": search_service.has_serp,
        "source": "api" if search_service.is_configured else "mock",
    }


@router.get(
    "/search/suggestions",
    response_model=list[str],
    status_code=status.HTTP_200_OK,
    summary="Get search suggestions",
    description="Get autocomplete suggestions for company names.",
)
async def get_search_suggestions(
    q: Annotated[str, Query(max_length=200)] = "",
) -> list[str]:
    """Get autocomplete suggestions for company search."""
    if not q or len(q) < 2:
        return []

    query_lower = q.lower().strip()
    suggestions = [
        company.name
        for company in MOCK_COMPANIES
        if query_lower in company.name.lower()
    ]

    return suggestions[:10]
