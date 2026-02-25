"""Pydantic models for Company Research Mapping Tool.

Models match the Excel template structure for tracking company leadership
and organizational information in the talent landscape mapping.
"""

from datetime import date, datetime, timezone
from typing import Literal, Optional, List
from uuid import UUID, uuid4


from pydantic import BaseModel, ConfigDict, Field, computed_field, field_validator


def to_camel(string: str) -> str:
    """Convert snake_case to camelCase."""
    components = string.split("_")
    return components[0] + "".join(word.capitalize() for word in components[1:])


class EmploymentRecord(BaseModel):
    """Previous employment record for an executive."""

    company_name: str
    title: str
    start_year: int | None = None
    end_year: int | None = None

    model_config = ConfigDict(
        populate_by_name=True,
        alias_generator=to_camel,
    )


class Executive(BaseModel):
    """Executive/leadership team member model.

    Represents a person in a leadership role at a company, including
    their title and tenure dates.

    Attributes:
        name: Full name of the executive.
        title: Job title (e.g., CEO, CFO, COO, President, SVP).
        start_year: Year they started in this role.
        end_year: Year they ended in this role, None if current/present.
    """

    model_config = ConfigDict(
        alias_generator=to_camel,
        populate_by_name=True,
        serialize_by_alias=True,
    )

    name: str = Field(
        ..., min_length=1, max_length=200, description="Full name of the executive"
    )
    title: str = Field(
        ..., min_length=1, max_length=200, description="Job title or position"
    )
    start_year: int | None = Field(
        default=None,
        ge=1900,
        le=2100,
        description="Year started in this role, None if unknown",
    )
    end_year: int | None = Field(
        default=None,
        ge=1900,
        le=2100,
        description="Year ended in this role, None if current",
    )
    linkedin_url: str | None = None
    photo_url: str | None = None
    employment_history: list[EmploymentRecord] = Field(default_factory=list)

    @field_validator("end_year")
    @classmethod
    def validate_end_year(cls, v: int | None, info) -> int | None:
        """Ensure end_year is not before start_year."""
        if v is not None and "start_year" in info.data:
            start_year = info.data["start_year"]
            if start_year is not None and v < start_year:
                raise ValueError("end_year cannot be before start_year")
        return v

    @computed_field
    @property
    def is_current(self) -> bool:
        """Check if this is a current position."""
        return self.end_year is None

    def to_display_string(self) -> str:
        """Format executive info as display string matching Excel format.

        Returns:
            Formatted string like "Tom York, CEO, 2021-2023" or
            "Ed Campbell, CEO, 2008-Present" for current roles.
            If start_year is unknown, returns "Tom York, CEO" without dates.
        """
        if self.start_year is None:
            return f"{self.name}, {self.title}"
        end_str = "Present" if self.is_current else str(self.end_year)
        return f"{self.name}, {self.title}, {self.start_year}-{end_str}"


class Role(BaseModel):
    """Standardized role definition.

    Attributes:
        id: Unique identifier for the role (e.g., "ceo", "cfo", "president_east").
        name: Display name (e.g., "Chief Executive Officer").
        category: Role category classification.
        is_standard: Whether this is a standard predefined role.
    """

    model_config = ConfigDict(
        alias_generator=to_camel,
        populate_by_name=True,
        serialize_by_alias=True,
    )

    id: str = Field(
        ..., min_length=1, max_length=100, description="Unique role identifier"
    )
    name: str = Field(
        ..., min_length=1, max_length=200, description="Display name for the role"
    )
    category: Literal["c_suite", "senior", "division", "custom"] = Field(
        ..., description="Role category classification"
    )
    is_standard: bool = Field(
        default=True, description="Whether this is a standard predefined role"
    )


class RoleHolder(BaseModel):
    """Executive holding a specific role for a time period.

    Tracks who held a particular role and when, enabling historical
    timeline tracking of role assignments.

    Attributes:
        role_id: Reference to the role being held.
        executive: The executive holding the role.
        start_date: When the executive started in this role.
        end_date: When the executive ended in this role (None if current).
        is_verified: Whether this assignment has been verified.
        sources: List of sources confirming this assignment.
    """

    model_config = ConfigDict(
        alias_generator=to_camel,
        populate_by_name=True,
        serialize_by_alias=True,
    )

    role_id: str = Field(..., description="Reference to the role being held")
    executive: Executive = Field(..., description="The executive holding the role")
    start_date: date = Field(..., description="When the executive started in this role")
    end_date: date | None = Field(
        default=None,
        description="When the executive ended in this role (None if current)",
    )
    is_verified: bool = Field(
        default=False, description="Whether this assignment has been verified"
    )
    sources: list[str] = Field(
        default_factory=list, description="List of sources confirming this assignment"
    )

    @computed_field
    @property
    def is_current(self) -> bool:
        """Check if this is a current role assignment."""
        return self.end_date is None


class Division(BaseModel):
    """Business unit or division.

    Represents organizational divisions or business units within a company,
    supporting hierarchical structure through parent references.

    Attributes:
        id: Unique identifier for the division.
        name: Display name of the division.
        parent_division_id: Reference to parent division (None if top-level).
    """

    model_config = ConfigDict(
        alias_generator=to_camel,
        populate_by_name=True,
        serialize_by_alias=True,
    )

    id: str = Field(
        ..., min_length=1, max_length=100, description="Unique division identifier"
    )
    name: str = Field(
        ..., min_length=1, max_length=200, description="Display name of the division"
    )
    parent_division_id: str | None = Field(
        default=None, description="Reference to parent division (None if top-level)"
    )


class ValidationMetadata(BaseModel):
    """Data validation tracking.

    Tracks when data was last validated and its confidence level,
    helping identify stale data that needs refreshing.

    Attributes:
        last_validated: When the data was last validated.
        confidence: Confidence score from 0.0 to 1.0.
        needs_refresh: Whether the data needs to be refreshed.
    """

    model_config = ConfigDict(
        alias_generator=to_camel,
        populate_by_name=True,
        serialize_by_alias=True,
    )

    last_validated: datetime = Field(
        ..., description="When the data was last validated"
    )
    confidence: float = Field(
        ..., ge=0.0, le=1.0, description="Confidence score from 0.0 to 1.0"
    )
    needs_refresh: bool = Field(
        default=False, description="Whether the data needs to be refreshed"
    )


class ExecutiveV2(Executive):
    """Enhanced executive with validation and division tracking.

    Extends the base Executive model with additional fields for
    division assignment, confidence scoring, and source tracking.

    Attributes:
        division_id: Reference to the division this executive belongs to.
        confidence_score: Confidence in the accuracy of this data (0.0 to 1.0).
        last_verified: When this executive's data was last verified.
        sources: List of sources for this executive's information.
    """

    division_id: str | None = Field(
        default=None, description="Reference to the division this executive belongs to"
    )
    confidence_score: float = Field(
        default=0.0,
        ge=0.0,
        le=1.0,
        description="Confidence in the accuracy of this data",
    )
    last_verified: datetime | None = Field(
        default=None, description="When this executive's data was last verified"
    )
    sources: list[str] = Field(
        default_factory=list,
        description="List of sources for this executive's information",
    )


# Standard roles for company leadership
STANDARD_ROLES: list[Role] = [
    Role(
        id="ceo", name="Chief Executive Officer", category="c_suite", is_standard=True
    ),
    Role(
        id="cfo", name="Chief Financial Officer", category="c_suite", is_standard=True
    ),
    Role(
        id="coo", name="Chief Operating Officer", category="c_suite", is_standard=True
    ),
    Role(
        id="cto", name="Chief Technology Officer", category="c_suite", is_standard=True
    ),
    Role(
        id="cmo", name="Chief Marketing Officer", category="c_suite", is_standard=True
    ),
    Role(
        id="cio", name="Chief Information Officer", category="c_suite", is_standard=True
    ),
]


# Network status for contact tracking
NetworkStatus = Literal[
    "in_network",
    "out_of_network",
]

# Contact status for outreach tracking
ContactStatus = Literal[
    "available",
    "contacted_no_response",
    "conflicted_not_interested",
    "not_contacted",
]


class Company(BaseModel):
    """Company model for talent landscape mapping.

    Represents a company with its leadership team and organizational
    metadata for the talent mapping research tool.

    Attributes:
        name: Company name.
        ceo: List of CEO(s), can include historical CEOs.
        c_level: List of other C-level executives (CFO, COO, CTO, etc.).
        senior_level: List of senior-level executives (President, SVP, VP, etc.).
        employees: Number of employees, None if unknown.
        ownership: Current ownership/investor (e.g., "Tenex Capital", "Merger").
        acquisition_date: Year of acquisition/investment, None if not applicable.
        subsector: Industry subsector classification.
        notes: Additional notes about the company.
        updated: Timestamp of last update.
        network_status: Whether contacts are in network or out of network.
        contact_status: Current contact/outreach status.
    """

    model_config = ConfigDict(
        alias_generator=to_camel,
        populate_by_name=True,
        serialize_by_alias=True,
    )

    name: str = Field(..., min_length=1, max_length=500, description="Company name")
    ceo: list[Executive] = Field(
        default_factory=list, description="List of CEO(s), including historical"
    )
    c_level: list[Executive] = Field(
        default_factory=list,
        description="Other C-level executives (CFO, COO, CTO, etc.)",
    )
    senior_level: list[Executive] = Field(
        default_factory=list,
        description="Senior-level executives (President, SVP, VP, etc.)",
    )
    employees: int | None = Field(
        default=None, ge=0, description="Number of employees, None if unknown"
    )
    ownership: str | None = Field(
        default=None, max_length=500, description="Current ownership or investor name"
    )
    acquisition_date: int | None = Field(
        default=None,
        ge=1900,
        le=2100,
        description="Year of acquisition/investment",
    )
    subsector: str | None = Field(
        default=None, max_length=200, description="Industry subsector classification"
    )
    notes: str | None = Field(
        default=None, max_length=5000, description="Additional notes"
    )
    updated: datetime | None = Field(
        default=None, description="Timestamp of last update"
    )
    network_status: NetworkStatus | None = Field(
        default=None, description="In network or out of network status"
    )
    contact_status: ContactStatus | None = Field(
        default=None, description="Contact/outreach status"
    )

    @property
    def current_ceo(self) -> Executive | None:
        """Get the current CEO if one exists.

        If multiple current CEOs exist (data quality issue), returns the one
        with the most recent start_year, as they are most likely the actual current CEO.
        """
        current_ceos = [exec for exec in self.ceo if exec.is_current]
        if not current_ceos:
            return None
        if len(current_ceos) == 1:
            return current_ceos[0]
        # Multiple current CEOs - return the one with most recent start_year
        # Sort by start_year descending (most recent first), handle None start_year
        return max(current_ceos, key=lambda e: e.start_year or 0)

    @property
    def all_executives(self) -> list[Executive]:
        """Get all executives across all levels."""
        return self.ceo + self.c_level + self.senior_level

    @property
    def current_executives(self) -> list[Executive]:
        """Get all current executives across all levels.

        Returns executives sorted by level (CEO first, then C-level, then senior)
        and within each level by start_year descending (most recent first).
        """
        current = []
        # Add current CEO (only one, even if multiple exist)
        if self.current_ceo:
            current.append(self.current_ceo)
        # Add other current C-level (excluding CEO)
        current.extend(
            sorted(
                [e for e in self.c_level if e.is_current],
                key=lambda e: e.start_year or 0,
                reverse=True,
            )
        )
        # Add current senior-level
        current.extend(
            sorted(
                [e for e in self.senior_level if e.is_current],
                key=lambda e: e.start_year or 0,
                reverse=True,
            )
        )
        return current

    @property
    def historical_executives(self) -> list[Executive]:
        """Get all historical (non-current) executives across all levels.

        Returns executives sorted by end_year descending (most recently departed first),
        then by start_year descending.
        """
        historical = [exec for exec in self.all_executives if not exec.is_current]
        # Also include "extra" current CEOs that weren't selected as THE current CEO
        if self.current_ceo:
            extra_current_ceos = [
                e for e in self.ceo if e.is_current and e.name != self.current_ceo.name
            ]
            # These should be treated as historical since we only want one current CEO
            historical.extend(extra_current_ceos)

        # Sort by end_year desc (most recent first), then start_year desc
        return sorted(
            historical,
            key=lambda e: (e.end_year or 9999, e.start_year or 0),
            reverse=True,
        )

    def get_executives_sorted(self) -> list[Executive]:
        """Get all executives sorted with current first, then historical.

        Current executives are sorted by level (CEO, C-level, senior) then start_year.
        Historical executives are sorted by end_year (most recent departure first).

        Returns:
            List of all executives with current ones first, then historical.
        """
        return self.current_executives + self.historical_executives


class CompanyV2(BaseModel):
    """Enhanced company with role-based structure.

    A new company model with role-based executive tracking,
    division support, custom roles, and validation metadata.

    This model maintains backward compatibility with the base Company model
    by providing computed properties that derive CEO, C-level, and senior-level
    executives from the role-based structure.

    Attributes:
        name: Company name.
        roles: Mapping of role_id to list of role holders (historical and current).
        divisions: List of divisions/business units in this company.
        custom_roles: List of custom roles defined for this company.
        validation: Validation metadata for tracking data freshness.
        employees: Number of employees, None if unknown.
        ownership: Current ownership/investor.
        acquisition_date: Year of acquisition/investment.
        subsector: Industry subsector classification.
        notes: Additional notes about the company.
        updated: Timestamp of last update.
        network_status: Whether contacts are in network or out of network.
        contact_status: Current contact/outreach status.
    """

    model_config = ConfigDict(
        alias_generator=to_camel,
        populate_by_name=True,
        serialize_by_alias=True,
    )

    name: str = Field(..., min_length=1, max_length=500, description="Company name")
    roles: dict[str, list[RoleHolder]] = Field(
        default_factory=dict, description="Mapping of role_id to list of role holders"
    )
    divisions: list[Division] = Field(
        default_factory=list, description="List of divisions/business units"
    )
    custom_roles: list[Role] = Field(
        default_factory=list,
        description="List of custom roles defined for this company",
    )
    validation: ValidationMetadata | None = Field(
        default=None, description="Validation metadata for tracking data freshness"
    )
    employees: int | None = Field(
        default=None, ge=0, description="Number of employees, None if unknown"
    )
    ownership: str | None = Field(
        default=None, max_length=500, description="Current ownership or investor name"
    )
    acquisition_date: int | None = Field(
        default=None,
        ge=1900,
        le=2100,
        description="Year of acquisition/investment",
    )
    subsector: str | None = Field(
        default=None, max_length=200, description="Industry subsector classification"
    )
    notes: str | None = Field(
        default=None, max_length=5000, description="Additional notes"
    )
    updated: datetime | None = Field(
        default=None, description="Timestamp of last update"
    )
    network_status: NetworkStatus | None = Field(
        default=None, description="In network or out of network status"
    )
    contact_status: ContactStatus | None = Field(
        default=None, description="Contact/outreach status"
    )

    @property
    def ceo(self) -> list[Executive]:
        """Get current CEO from roles for backward compatibility.

        Returns list of current CEO executives from the role-based structure.
        """
        if "ceo" not in self.roles:
            return []
        return [holder.executive for holder in self.roles["ceo"] if holder.is_current]

    @property
    def c_level(self) -> list[Executive]:
        """Get current C-level executives from roles for backward compatibility.

        Returns list of current C-level executives (excluding CEO) from the role-based structure.
        """
        c_level_role_ids = {"cfo", "coo", "cto", "cmo", "cio"}
        executives = []
        for role_id, holders in self.roles.items():
            if role_id in c_level_role_ids:
                executives.extend(
                    [holder.executive for holder in holders if holder.is_current]
                )
        return executives

    @property
    def senior_level(self) -> list[Executive]:
        """Get current senior-level executives from roles for backward compatibility.

        Returns list of current senior-level executives from the role-based structure.
        """
        c_suite_role_ids = {"ceo", "cfo", "coo", "cto", "cmo", "cio"}
        executives = []
        for role_id, holders in self.roles.items():
            if role_id not in c_suite_role_ids:
                # Check if this is a senior role by checking custom_roles or standard roles
                is_senior = False
                for role in self.custom_roles:
                    if role.id == role_id and role.category == "senior":
                        is_senior = True
                        break
                if is_senior:
                    executives.extend(
                        [holder.executive for holder in holders if holder.is_current]
                    )
        return executives

    @property
    def current_ceo(self) -> Executive | None:
        """Get the current CEO if one exists.

        For CompanyV2, the ceo property already filters to current holders,
        so we just return the first one (most recent by start date preferred).
        """
        ceos = self.ceo
        if not ceos:
            return None
        if len(ceos) == 1:
            return ceos[0]
        # Multiple current CEOs - return the one with most recent start_year
        return max(ceos, key=lambda e: e.start_year or 0)

    @property
    def all_executives(self) -> list[Executive]:
        """Get all executives across all levels."""
        return self.ceo + self.c_level + self.senior_level

    @property
    def current_executives(self) -> list[Executive]:
        """Get all current executives across all levels.

        Returns executives sorted by level (CEO first, then C-level, then senior)
        and within each level by start_year descending (most recent first).
        """
        current = []
        # Add current CEO (only one, even if multiple exist)
        if self.current_ceo:
            current.append(self.current_ceo)
        # Add other current C-level (excluding CEO)
        current.extend(
            sorted(
                [e for e in self.c_level if e.is_current],
                key=lambda e: e.start_year or 0,
                reverse=True,
            )
        )
        # Add current senior-level
        current.extend(
            sorted(
                [e for e in self.senior_level if e.is_current],
                key=lambda e: e.start_year or 0,
                reverse=True,
            )
        )
        return current

    @property
    def historical_executives(self) -> list[Executive]:
        """Get all historical (non-current) executives.

        For CompanyV2, this returns executives from role holders where end_date is set.
        Sorted by end_date descending (most recently departed first).
        """
        historical = []
        for role_id, holders in self.roles.items():
            for holder in holders:
                if not holder.is_current:
                    historical.append(holder.executive)
        # Sort by end_year desc, then start_year desc
        return sorted(
            historical,
            key=lambda e: (e.end_year or 9999, e.start_year or 0),
            reverse=True,
        )

    def get_executives_sorted(self) -> list[Executive]:
        """Get all executives sorted with current first, then historical.

        Returns:
            List of all executives with current ones first, then historical.
        """
        return self.current_executives + self.historical_executives


class RoleChange(BaseModel):
    """Tracks a change in role assignment.

    Records when a role changes hands, useful for tracking
    leadership transitions and historical changes.

    Attributes:
        role_id: The role that changed.
        old_holder: The previous holder of the role (None if new role).
        new_holder: The new holder of the role (None if role vacated).
        change_date: When the change occurred.
    """

    model_config = ConfigDict(
        alias_generator=to_camel,
        populate_by_name=True,
        serialize_by_alias=True,
    )

    role_id: str = Field(..., description="The role that changed")
    old_holder: Executive | None = Field(
        default=None, description="The previous holder of the role"
    )
    new_holder: Executive | None = Field(
        default=None, description="The new holder of the role"
    )
    change_date: date = Field(..., description="When the change occurred")


class RefreshResult(BaseModel):
    """Result of refreshing company data.

    Contains the results of a data refresh operation, including
    new executives discovered, departures, and role changes.

    Attributes:
        company_id: The company that was refreshed.
        new_executives: List of newly discovered executives.
        departed_executives: List of executives who have departed.
        role_changes: List of role changes detected.
    """

    model_config = ConfigDict(
        alias_generator=to_camel,
        populate_by_name=True,
        serialize_by_alias=True,
    )

    company_id: str = Field(..., description="The company that was refreshed")
    new_executives: list[Executive] = Field(
        default_factory=list, description="List of newly discovered executives"
    )
    departed_executives: list[Executive] = Field(
        default_factory=list, description="List of executives who have departed"
    )
    role_changes: list[RoleChange] = Field(
        default_factory=list, description="List of role changes detected"
    )


class TimeRangeQuery(BaseModel):
    """Query for time-bounded searches.

    Used to query for executives or roles within a specific time range.

    Attributes:
        start_year: The start year of the range (inclusive).
        end_year: The end year of the range (inclusive), None for current.
    """

    model_config = ConfigDict(
        alias_generator=to_camel,
        populate_by_name=True,
        serialize_by_alias=True,
    )

    start_year: int = Field(
        ..., ge=1900, le=2100, description="The start year of the range"
    )
    end_year: int | None = Field(
        default=None,
        ge=1900,
        le=2100,
        description="The end year of the range (None for current)",
    )


class CompanyCreate(BaseModel):
    """Request model for creating a new company."""

    name: str = Field(..., min_length=1, max_length=500, description="Company name")
    ceo: list[Executive] = Field(default_factory=list)
    c_level: list[Executive] = Field(default_factory=list)
    senior_level: list[Executive] = Field(default_factory=list)
    employees: int | None = Field(default=None, ge=0)
    ownership: str | None = Field(default=None, max_length=500)
    acquisition_date: int | None = Field(default=None, ge=1900, le=2100)
    subsector: str | None = Field(default=None, max_length=200)
    notes: str | None = Field(default=None, max_length=5000)
    network_status: NetworkStatus | None = None
    contact_status: ContactStatus | None = None


class CompanyUpdate(BaseModel):
    """Request model for updating a company (partial update)."""

    name: str | None = Field(default=None, min_length=1, max_length=500)
    ceo: list[Executive] | None = None
    c_level: list[Executive] | None = None
    senior_level: list[Executive] | None = None
    employees: int | None = Field(default=None, ge=0)
    ownership: str | None = Field(default=None, max_length=500)
    acquisition_date: int | None = Field(default=None, ge=1900, le=2100)
    subsector: str | None = Field(default=None, max_length=200)
    notes: str | None = Field(default=None, max_length=5000)
    network_status: NetworkStatus | None = None
    contact_status: ContactStatus | None = None


class ExecutiveSummary(BaseModel):
    """Summary of executives split by current vs historical status."""

    model_config = ConfigDict(
        alias_generator=to_camel,
        populate_by_name=True,
        serialize_by_alias=True,
    )

    current: list[Executive] = Field(
        default_factory=list,
        description="Current executives (sorted by level, then start_year desc)",
    )
    historical: list[Executive] = Field(
        default_factory=list,
        description="Historical executives (sorted by end_year desc)",
    )


class CompanyResponse(Company):
    """Response model for company data with ID."""

    id: str = Field(..., description="Unique identifier for the company")

    @computed_field
    @property
    def executives_by_status(self) -> ExecutiveSummary:
        """Get executives organized by current vs historical status.

        This computed field provides a cleaner separation of current and
        historical executives for frontend consumption.
        """
        return ExecutiveSummary(
            current=self.current_executives,
            historical=self.historical_executives,
        )


class TalentLandscape(BaseModel):
    """Talent landscape model representing a complete mapping project.

    Attributes:
        name: Name of the talent landscape (e.g., "Commercial Paving").
        companies: List of companies in this landscape.
        created_at: When this landscape was created.
        updated_at: When this landscape was last updated.
    """

    name: str = Field(
        ..., min_length=1, max_length=200, description="Talent landscape name"
    )
    companies: list[Company] = Field(
        default_factory=list, description="Companies in this landscape"
    )
    created_at: datetime = Field(
        default_factory=lambda: datetime.now(timezone.utc),
        description="Creation timestamp",
    )
    updated_at: datetime = Field(
        default_factory=lambda: datetime.now(timezone.utc),
        description="Last update timestamp",
    )


class Call(BaseModel):
    id: UUID = Field(default_factory=uuid4)

    title: str = Field(..., min_length=1, max_length=200)
    description: str | None = Field(default=None, max_length=5000)

    call_date: datetime
    transcript: str

    status: Literal["uploaded", "processing", "analyzed", "failed"] = "uploaded"

    created_at: datetime = Field(default_factory=lambda: datetime.now(timezone.utc))

    model_config = {"from_attributes": True}


class PersonMention(BaseModel):
    name: str
    role: Optional[str] = None
    company: Optional[str] = None


class ActionItem(BaseModel):
    description: str
    owner: Optional[str] = None
    urgency: Optional[Literal["low", "medium", "high"]] = None


class CallInsight(BaseModel):
    id: Optional[UUID] = None
    summary: str = Field(..., min_length=20)

    tags: List[str] = Field(..., description="Predefined business or technical tags")

    action_items: List[ActionItem] = Field(default_factory=list)

    people_mentioned: List[PersonMention] = Field(default_factory=list)

    key_decisions: List[str] = Field(default_factory=list)
    call_type: Optional[
        Literal[
            "sales_call",
            "investor_call",
            "strategy_meeting",
            "interview",
            "operational_meeting",
            "networking_call",
            "unknown",
        ]
    ] = None





class CareerHistory(BaseModel):
    company: Optional[str] = None
    title: Optional[str] = None
    company_type: Optional[str] = None
    duration_estimate_years: Optional[float] = None

    def to_dict(self):
        """Convert CareerHistory object to dict"""
        if self is None:
            return None
        return {
            'company': self.company,
            'title': self.title,
            'company_type': self.company_type,
            'duration_estimate_years': self.duration_estimate_years
        }



class LeadershipScope(BaseModel):
    team_size_managed: Optional[int] = None
    budget_responsibility: Optional[str] = None
    geographical_scope: Optional[str] = None

    def to_dict(self):
        """Convert LeadershipScope object to dict"""
        if self is None:
            return None
        return {
            'team_size_managed': self.team_size_managed,
            'budget_responsibility': self.budget_responsibility,
            'geographical_scope': self.geographical_scope
        }


class TransformationExperience(BaseModel):
    type: Optional[str] = None
    description: Optional[str] = None
    role: Optional[str] = None
    quantifiable_impact: Optional[str] = None

    def to_dict(self):
        """Convert TransformationExperience object to dict"""
        if self is None:
            return None
        return {
            'type': self.type,
            'description': self.description,
            'role': self.role,
            'quantifiable_impact': self.quantifiable_impact
        }


class PrivateEquityExperience(BaseModel):
    has_pe_experience: Optional[bool] = None
    description: Optional[str] = None

    def to_dict(self):
        """Convert PrivateEquityExperience object to dict"""
        if self is None:
            return None
        return {
            'has_pe_experience': self.has_pe_experience,
            'description': self.description
        }


class IntervieweeProfile(BaseModel):
    id: Optional[UUID] = None
    full_name: str = Field(..., min_length=1, max_length=200)
    current_title: Optional[str] = None
    current_company: Optional[str] = None
    seniority_level: Optional[str] = None
    industry_focus: List[str] = Field(default_factory=list)
    years_experience_estimate: Optional[int] = None
    career_history: List[CareerHistory] = Field(default_factory=list)
    leadership_scope: Optional[LeadershipScope] = None
    transformation_experience: Optional[List[TransformationExperience]] = Field(
        default_factory=list
    )
    private_equity_exposure: Optional[PrivateEquityExperience] = None
    technical_capabilities: List[str] = Field(default_factory=list)
    notable_achievements: List[str] = Field(default_factory=list)
    risk_flags: List[str] = Field(default_factory=list)
    searchable_summary: Optional[str] = None
    confidence_score: Optional[float] =None
    embedding: Optional[List[float]] = None
    has_pe_experience: Optional[bool] = None
    transformation_types: Optional[List[str]] = Field(default_factory=list)

    def to_dict(self):
        """Convert IntervieweeProfile object to dict, including nested objects."""
        if self is None:
            return None
        return {
            'id': str(self.id) if self.id else None,
            'full_name': self.full_name,
            'current_title': self.current_title,
            'current_company': self.current_company,
            'seniority_level': self.seniority_level,
            'industry_focus': self.industry_focus,
            'years_experience_estimate': self.years_experience_estimate,
            'career_history': [ch.to_dict() for ch in self.career_history],
            'leadership_scope': self.leadership_scope.to_dict() if self.leadership_scope else None,
            'transformation_experience': [te.to_dict() for te in self.transformation_experience],
            'private_equity_exposure': self.private_equity_exposure.to_dict() if self.private_equity_exposure else None,
            'technical_capabilities': self.technical_capabilities,
            'notable_achievements': self.notable_achievements,
            'risk_flags': self.risk_flags,
            'searchable_summary': self.searchable_summary,
            'confidence_score': self.confidence_score

        }

class CallAnalysisResult(BaseModel):
    call_id: Optional[UUID] 
    insights: CallInsight
    interviewee_profile: Optional[IntervieweeProfile]
    analyzed_at: datetime = Field(default_factory=lambda: datetime.now(timezone.utc))
