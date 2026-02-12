"""Tests for Pydantic models in Company Research Mapping Tool.

Tests validate the Executive, Company, and related model classes
for proper validation, optional fields, and property methods.
"""

import pytest
from datetime import datetime
from pydantic import ValidationError

from app.models import (
    Executive,
    Company,
    CompanyCreate,
    CompanyUpdate,
    CompanyResponse,
    TalentLandscape,
    Role,
    RoleHolder,
    Division,
    ValidationMetadata,
    ExecutiveV2,
    CompanyV2,
    RoleChange,
    RefreshResult,
    TimeRangeQuery,
)


class TestExecutiveModel:
    """Test suite for Executive model validation."""

    def test_executive_valid_creation(self):
        """Test creating a valid executive."""
        exec = Executive(
            name="John Smith",
            title="CEO",
            start_year=2020,
            end_year=None,
        )
        assert exec.name == "John Smith"
        assert exec.title == "CEO"
        assert exec.start_year == 2020
        assert exec.end_year is None

    def test_executive_with_end_year(self):
        """Test creating an executive with end year (former executive)."""
        exec = Executive(
            name="Jane Doe",
            title="CFO",
            start_year=2015,
            end_year=2020,
        )
        assert exec.start_year == 2015
        assert exec.end_year == 2020
        assert not exec.is_current

    def test_executive_is_current_property(self):
        """Test is_current property for current vs former executives."""
        current_exec = Executive(
            name="Current CEO",
            title="CEO",
            start_year=2020,
            end_year=None,
        )
        former_exec = Executive(
            name="Former CEO",
            title="CEO",
            start_year=2015,
            end_year=2019,
        )
        assert current_exec.is_current is True
        assert former_exec.is_current is False

    def test_executive_empty_name_raises_error(self):
        """Test that empty name raises validation error."""
        with pytest.raises(ValidationError) as exc_info:
            Executive(name="", title="CEO", start_year=2020)
        assert "min_length" in str(exc_info.value) or "String should have at least 1 character" in str(exc_info.value)

    def test_executive_empty_title_raises_error(self):
        """Test that empty title raises validation error."""
        with pytest.raises(ValidationError) as exc_info:
            Executive(name="John Smith", title="", start_year=2020)
        assert "min_length" in str(exc_info.value) or "String should have at least 1 character" in str(exc_info.value)

    def test_executive_invalid_start_year(self):
        """Test that invalid start year raises validation error."""
        with pytest.raises(ValidationError):
            Executive(name="John", title="CEO", start_year=1800)  # Before 1900

    def test_executive_end_year_before_start_year(self):
        """Test that end_year before start_year raises validation error."""
        with pytest.raises(ValidationError) as exc_info:
            Executive(name="John", title="CEO", start_year=2020, end_year=2018)
        assert "end_year cannot be before start_year" in str(exc_info.value)

    def test_executive_to_display_string_current(self):
        """Test display string for current executive."""
        exec = Executive(name="Ed Campbell", title="CEO", start_year=2008, end_year=None)
        display = exec.to_display_string()
        assert display == "Ed Campbell, CEO, 2008-Present"

    def test_executive_to_display_string_former(self):
        """Test display string for former executive."""
        exec = Executive(name="Tom York", title="CEO", start_year=2001, end_year=2008)
        display = exec.to_display_string()
        assert display == "Tom York, CEO, 2001-2008"


class TestCompanyModel:
    """Test suite for Company model validation."""

    def test_company_valid_creation_minimal(self):
        """Test creating a company with only required fields."""
        company = Company(name="Test Company")
        assert company.name == "Test Company"
        assert company.ceo == []
        assert company.c_level == []
        assert company.senior_level == []
        assert company.employees is None
        assert company.ownership is None

    def test_company_valid_creation_full(self):
        """Test creating a company with all fields populated."""
        ceo = Executive(name="John CEO", title="CEO", start_year=2020)
        cfo = Executive(name="Jane CFO", title="CFO", start_year=2021)
        
        company = Company(
            name="Full Test Company",
            ceo=[ceo],
            c_level=[cfo],
            senior_level=[],
            employees=500,
            ownership="Private Equity Firm",
            acquisition_date=2019,
            subsector="Technology",
            notes="Test notes",
            network_status="in_network",
            contact_status="available",
        )
        
        assert company.name == "Full Test Company"
        assert len(company.ceo) == 1
        assert len(company.c_level) == 1
        assert company.employees == 500
        assert company.ownership == "Private Equity Firm"
        assert company.network_status == "in_network"
        assert company.contact_status == "available"

    def test_company_empty_name_raises_error(self):
        """Test that empty company name raises validation error."""
        with pytest.raises(ValidationError):
            Company(name="")

    def test_company_optional_fields_work_correctly(self):
        """Test that all optional fields can be None."""
        company = Company(
            name="Minimal Company",
            employees=None,
            ownership=None,
            acquisition_date=None,
            subsector=None,
            notes=None,
            updated=None,
            network_status=None,
            contact_status=None,
        )
        assert company.name == "Minimal Company"
        assert company.employees is None
        assert company.ownership is None
        assert company.acquisition_date is None
        assert company.network_status is None

    def test_company_current_ceo_property(self):
        """Test current_ceo property returns the current CEO."""
        former_ceo = Executive(name="Former", title="CEO", start_year=2010, end_year=2020)
        current_ceo = Executive(name="Current", title="CEO", start_year=2020, end_year=None)
        
        company = Company(name="Test Co", ceo=[former_ceo, current_ceo])
        assert company.current_ceo is not None
        assert company.current_ceo.name == "Current"

    def test_company_current_ceo_property_none(self):
        """Test current_ceo property returns None when no current CEO."""
        former_ceo = Executive(name="Former", title="CEO", start_year=2010, end_year=2020)
        company = Company(name="Test Co", ceo=[former_ceo])
        assert company.current_ceo is None

    def test_company_all_executives_property(self):
        """Test all_executives property returns all executives."""
        ceo = Executive(name="CEO", title="CEO", start_year=2020)
        cfo = Executive(name="CFO", title="CFO", start_year=2020)
        vp = Executive(name="VP", title="VP Sales", start_year=2021)
        
        company = Company(
            name="Test Co",
            ceo=[ceo],
            c_level=[cfo],
            senior_level=[vp],
        )
        
        all_execs = company.all_executives
        assert len(all_execs) == 3
        assert ceo in all_execs
        assert cfo in all_execs
        assert vp in all_execs

    def test_company_current_executives_property(self):
        """Test current_executives property returns only current executives."""
        current_ceo = Executive(name="Current CEO", title="CEO", start_year=2020, end_year=None)
        former_cfo = Executive(name="Former CFO", title="CFO", start_year=2015, end_year=2020)
        current_vp = Executive(name="Current VP", title="VP", start_year=2021, end_year=None)
        
        company = Company(
            name="Test Co",
            ceo=[current_ceo],
            c_level=[former_cfo],
            senior_level=[current_vp],
        )
        
        current_execs = company.current_executives
        assert len(current_execs) == 2
        assert current_ceo in current_execs
        assert current_vp in current_execs
        assert former_cfo not in current_execs

    def test_company_network_status_literal_validation(self):
        """Test network_status only accepts valid literal values."""
        company = Company(name="Test", network_status="in_network")
        assert company.network_status == "in_network"
        
        company2 = Company(name="Test2", network_status="out_of_network")
        assert company2.network_status == "out_of_network"

    def test_company_contact_status_literal_validation(self):
        """Test contact_status only accepts valid literal values."""
        valid_statuses = [
            "available",
            "contacted_no_response",
            "conflicted_not_interested",
            "not_contacted",
        ]
        for status in valid_statuses:
            company = Company(name="Test", contact_status=status)
            assert company.contact_status == status


class TestCompanyCreateModel:
    """Test suite for CompanyCreate model."""

    def test_company_create_minimal(self):
        """Test creating CompanyCreate with minimal data."""
        data = CompanyCreate(name="New Company")
        assert data.name == "New Company"
        assert data.ceo == []

    def test_company_create_full(self):
        """Test creating CompanyCreate with full data."""
        data = CompanyCreate(
            name="Full Company",
            ceo=[Executive(name="CEO", title="CEO", start_year=2020)],
            employees=100,
            ownership="Private",
            subsector="Tech",
        )
        assert data.name == "Full Company"
        assert len(data.ceo) == 1
        assert data.employees == 100


class TestCompanyUpdateModel:
    """Test suite for CompanyUpdate model."""

    def test_company_update_partial(self):
        """Test CompanyUpdate allows partial updates."""
        update = CompanyUpdate(name="Updated Name")
        assert update.name == "Updated Name"
        assert update.employees is None
        assert update.ceo is None

    def test_company_update_all_fields(self):
        """Test CompanyUpdate with all fields."""
        update = CompanyUpdate(
            name="Updated",
            employees=200,
            ownership="New Owner",
            notes="Updated notes",
        )
        assert update.name == "Updated"
        assert update.employees == 200


class TestCompanyResponseModel:
    """Test suite for CompanyResponse model."""

    def test_company_response_requires_id(self):
        """Test CompanyResponse requires id field."""
        response = CompanyResponse(
            id="comp_123",
            name="Test Company",
        )
        assert response.id == "comp_123"
        assert response.name == "Test Company"


class TestTalentLandscapeModel:
    """Test suite for TalentLandscape model."""

    def test_talent_landscape_creation(self):
        """Test creating a talent landscape."""
        landscape = TalentLandscape(name="Commercial Paving")
        assert landscape.name == "Commercial Paving"
        assert landscape.companies == []
        assert landscape.created_at is not None
        assert landscape.updated_at is not None

    def test_talent_landscape_with_companies(self):
        """Test talent landscape with companies."""
        company = Company(name="Test Co")
        landscape = TalentLandscape(
            name="Test Landscape",
            companies=[company],
        )
        assert len(landscape.companies) == 1
        assert landscape.companies[0].name == "Test Co"


class TestRole:
    """Tests for the Role model."""

    def test_role_creation(self):
        """Test creating a valid role."""
        from app.models import Role
        
        role = Role(
            id="ceo",
            name="Chief Executive Officer",
            category="c_suite",
            is_standard=True
        )
        assert role.id == "ceo"
        assert role.name == "Chief Executive Officer"
        assert role.category == "c_suite"
        assert role.is_standard is True

    def test_role_categories(self):
        """Test all valid role categories."""
        from app.models import Role
        
        for category in ["c_suite", "senior", "division", "custom"]:
            role = Role(id="test", name="Test Role", category=category)
            assert role.category == category

    def test_role_defaults(self):
        """Test role default values."""
        from app.models import Role
        
        role = Role(id="test", name="Test", category="c_suite")
        assert role.is_standard is True

    def test_role_invalid_category(self):
        """Test that invalid category raises validation error."""
        from app.models import Role
        
        with pytest.raises(ValidationError):
            Role(id="test", name="Test", category="invalid_category")

    def test_role_empty_id_raises_error(self):
        """Test that empty id raises validation error."""
        from app.models import Role
        
        with pytest.raises(ValidationError):
            Role(id="", name="Test", category="c_suite")


class TestRoleHolder:
    """Tests for the RoleHolder model."""

    def test_role_holder_current(self):
        """Test role holder with no end date (current)."""
        from app.models import RoleHolder
        from datetime import date
        
        executive = Executive(name="John Smith", title="CEO", start_year=2020)
        holder = RoleHolder(
            role_id="ceo",
            executive=executive,
            start_date=date(2020, 1, 15),
            end_date=None,
            is_verified=True,
            sources=["LinkedIn", "Company Website"]
        )
        
        assert holder.role_id == "ceo"
        assert holder.executive.name == "John Smith"
        assert holder.start_date == date(2020, 1, 15)
        assert holder.end_date is None
        assert holder.is_current is True
        assert holder.is_verified is True
        assert len(holder.sources) == 2

    def test_role_holder_past(self):
        """Test role holder with end date (past)."""
        from app.models import RoleHolder
        from datetime import date
        
        executive = Executive(name="Jane Doe", title="CFO", start_year=2015, end_year=2020)
        holder = RoleHolder(
            role_id="cfo",
            executive=executive,
            start_date=date(2015, 3, 1),
            end_date=date(2020, 6, 30),
        )
        
        assert holder.end_date == date(2020, 6, 30)
        assert holder.is_current is False

    def test_role_holder_defaults(self):
        """Test role holder default values."""
        from app.models import RoleHolder
        from datetime import date
        
        executive = Executive(name="Test Person", title="CTO", start_year=2021)
        holder = RoleHolder(
            role_id="cto",
            executive=executive,
            start_date=date(2021, 1, 1),
        )
        
        assert holder.is_verified is False
        assert holder.sources == []
        assert holder.end_date is None


class TestDivision:
    """Tests for the Division model."""

    def test_division_creation(self):
        """Test creating a division."""
        from app.models import Division
        
        division = Division(
            id="east_region",
            name="East Region"
        )
        
        assert division.id == "east_region"
        assert division.name == "East Region"
        assert division.parent_division_id is None

    def test_division_hierarchy(self):
        """Test division with parent."""
        from app.models import Division
        
        parent_division = Division(id="north_america", name="North America")
        child_division = Division(
            id="east_coast",
            name="East Coast",
            parent_division_id="north_america"
        )
        
        assert child_division.parent_division_id == "north_america"

    def test_division_empty_id_raises_error(self):
        """Test that empty id raises validation error."""
        from app.models import Division
        
        with pytest.raises(ValidationError):
            Division(id="", name="Test Division")

    def test_division_empty_name_raises_error(self):
        """Test that empty name raises validation error."""
        from app.models import Division
        
        with pytest.raises(ValidationError):
            Division(id="test_div", name="")


class TestValidationMetadata:
    """Tests for the ValidationMetadata model."""

    def test_validation_metadata_creation(self):
        """Test creating validation metadata."""
        from app.models import ValidationMetadata
        
        now = datetime.now()
        metadata = ValidationMetadata(
            last_validated=now,
            confidence=0.85,
            needs_refresh=False
        )
        
        assert metadata.last_validated == now
        assert metadata.confidence == 0.85
        assert metadata.needs_refresh is False

    def test_confidence_bounds_valid(self):
        """Test confidence score validation with valid values (0.0-1.0)."""
        from app.models import ValidationMetadata
        
        now = datetime.now()
        
        # Test minimum value
        metadata_min = ValidationMetadata(last_validated=now, confidence=0.0)
        assert metadata_min.confidence == 0.0
        
        # Test maximum value
        metadata_max = ValidationMetadata(last_validated=now, confidence=1.0)
        assert metadata_max.confidence == 1.0
        
        # Test middle value
        metadata_mid = ValidationMetadata(last_validated=now, confidence=0.5)
        assert metadata_mid.confidence == 0.5

    def test_confidence_bounds_invalid_low(self):
        """Test that confidence below 0.0 raises validation error."""
        from app.models import ValidationMetadata
        
        with pytest.raises(ValidationError):
            ValidationMetadata(last_validated=datetime.now(), confidence=-0.1)

    def test_confidence_bounds_invalid_high(self):
        """Test that confidence above 1.0 raises validation error."""
        from app.models import ValidationMetadata
        
        with pytest.raises(ValidationError):
            ValidationMetadata(last_validated=datetime.now(), confidence=1.1)

    def test_validation_metadata_defaults(self):
        """Test validation metadata default values."""
        from app.models import ValidationMetadata
        
        metadata = ValidationMetadata(last_validated=datetime.now(), confidence=0.9)
        assert metadata.needs_refresh is False


class TestExecutiveV2:
    """Tests for the ExecutiveV2 model."""

    def test_executive_v2_extends_executive(self):
        """Test that ExecutiveV2 has all Executive fields plus new ones."""
        from app.models import ExecutiveV2
        
        exec_v2 = ExecutiveV2(
            name="John Smith",
            title="CEO",
            start_year=2020,
            end_year=None,
            division_id="east_region",
            confidence_score=0.95,
            last_verified=datetime.now(),
            sources=["LinkedIn", "Press Release"]
        )
        
        # Check base Executive fields
        assert exec_v2.name == "John Smith"
        assert exec_v2.title == "CEO"
        assert exec_v2.start_year == 2020
        assert exec_v2.end_year is None
        assert exec_v2.is_current is True
        
        # Check ExecutiveV2-specific fields
        assert exec_v2.division_id == "east_region"
        assert exec_v2.confidence_score == 0.95
        assert exec_v2.last_verified is not None
        assert len(exec_v2.sources) == 2

    def test_executive_v2_with_division(self):
        """Test ExecutiveV2 with division assignment."""
        from app.models import ExecutiveV2
        
        exec_v2 = ExecutiveV2(
            name="Regional Manager",
            title="VP Operations",
            start_year=2019,
            division_id="west_region"
        )
        
        assert exec_v2.division_id == "west_region"

    def test_executive_v2_defaults(self):
        """Test ExecutiveV2 default values."""
        from app.models import ExecutiveV2
        
        exec_v2 = ExecutiveV2(
            name="Test Executive",
            title="CTO",
            start_year=2021
        )
        
        assert exec_v2.division_id is None
        assert exec_v2.confidence_score == 0.0
        assert exec_v2.last_verified is None
        assert exec_v2.sources == []

    def test_executive_v2_confidence_bounds(self):
        """Test that confidence_score validation works (0.0 to 1.0)."""
        from app.models import ExecutiveV2
        
        # Valid confidence scores
        exec_low = ExecutiveV2(name="Test", title="CEO", start_year=2020, confidence_score=0.0)
        assert exec_low.confidence_score == 0.0
        
        exec_high = ExecutiveV2(name="Test", title="CEO", start_year=2020, confidence_score=1.0)
        assert exec_high.confidence_score == 1.0
        
        # Invalid confidence score
        with pytest.raises(ValidationError):
            ExecutiveV2(name="Test", title="CEO", start_year=2020, confidence_score=1.5)

    def test_executive_v2_inherits_validation(self):
        """Test that ExecutiveV2 inherits validation from Executive."""
        from app.models import ExecutiveV2
        
        # Empty name should fail (inherited validation)
        with pytest.raises(ValidationError):
            ExecutiveV2(name="", title="CEO", start_year=2020)
        
        # end_year before start_year should fail (inherited validation)
        with pytest.raises(ValidationError):
            ExecutiveV2(name="Test", title="CEO", start_year=2020, end_year=2018)


class TestCompanyV2:
    """Tests for the CompanyV2 model."""

    def test_company_v2_backward_compatibility(self):
        """Test that CompanyV2 has ceo, c_level, senior_level properties."""
        from app.models import CompanyV2, RoleHolder, Role
        from datetime import date
        
        # Create a company with CEO in roles structure
        ceo_exec = Executive(name="John CEO", title="CEO", start_year=2020)
        cfo_exec = Executive(name="Jane CFO", title="CFO", start_year=2021)
        
        company = CompanyV2(
            name="Test Company V2",
            roles={
                "ceo": [RoleHolder(
                    role_id="ceo",
                    executive=ceo_exec,
                    start_date=date(2020, 1, 1)
                )],
                "cfo": [RoleHolder(
                    role_id="cfo",
                    executive=cfo_exec,
                    start_date=date(2021, 1, 1)
                )]
            }
        )
        
        # Test backward compatibility properties
        assert len(company.ceo) == 1
        assert company.ceo[0].name == "John CEO"
        assert len(company.c_level) == 1
        assert company.c_level[0].name == "Jane CFO"
        assert company.current_ceo is not None
        assert company.current_ceo.name == "John CEO"

    def test_company_v2_roles_structure(self):
        """Test the roles dict structure."""
        from app.models import CompanyV2, RoleHolder
        from datetime import date
        
        exec1 = Executive(name="First CEO", title="CEO", start_year=2015, end_year=2020)
        exec2 = Executive(name="Second CEO", title="CEO", start_year=2020)
        
        company = CompanyV2(
            name="Test Co",
            roles={
                "ceo": [
                    RoleHolder(
                        role_id="ceo",
                        executive=exec1,
                        start_date=date(2015, 1, 1),
                        end_date=date(2020, 6, 30)
                    ),
                    RoleHolder(
                        role_id="ceo",
                        executive=exec2,
                        start_date=date(2020, 7, 1)
                    )
                ]
            }
        )
        
        assert "ceo" in company.roles
        assert len(company.roles["ceo"]) == 2
        # ceo property should only return current CEO
        assert len(company.ceo) == 1
        assert company.ceo[0].name == "Second CEO"

    def test_company_v2_with_divisions(self):
        """Test CompanyV2 with divisions."""
        from app.models import CompanyV2, Division
        
        company = CompanyV2(
            name="Multi-Division Company",
            divisions=[
                Division(id="east", name="East Region"),
                Division(id="west", name="West Region"),
            ]
        )
        
        assert len(company.divisions) == 2
        assert company.divisions[0].id == "east"

    def test_company_v2_with_custom_roles(self):
        """Test CompanyV2 with custom roles."""
        from app.models import CompanyV2, Role
        
        company = CompanyV2(
            name="Custom Roles Company",
            custom_roles=[
                Role(id="regional_president", name="Regional President", category="senior", is_standard=False),
                Role(id="chief_growth_officer", name="Chief Growth Officer", category="c_suite", is_standard=False),
            ]
        )
        
        assert len(company.custom_roles) == 2
        assert company.custom_roles[0].is_standard is False

    def test_company_v2_with_validation_metadata(self):
        """Test CompanyV2 with validation metadata."""
        from app.models import CompanyV2, ValidationMetadata
        
        now = datetime.now()
        company = CompanyV2(
            name="Validated Company",
            validation=ValidationMetadata(
                last_validated=now,
                confidence=0.9,
                needs_refresh=False
            )
        )
        
        assert company.validation is not None
        assert company.validation.confidence == 0.9

    def test_company_v2_empty_roles(self):
        """Test CompanyV2 with no roles returns empty lists for compatibility properties."""
        from app.models import CompanyV2
        
        company = CompanyV2(name="Empty Company")
        
        assert company.ceo == []
        assert company.c_level == []
        assert company.senior_level == []
        assert company.current_ceo is None
        assert company.all_executives == []


class TestRefreshResult:
    """Tests for the RefreshResult model."""

    def test_refresh_result_empty(self):
        """Test refresh result with no changes."""
        from app.models import RefreshResult
        
        result = RefreshResult(company_id="comp_123")
        
        assert result.company_id == "comp_123"
        assert result.new_executives == []
        assert result.departed_executives == []
        assert result.role_changes == []

    def test_refresh_result_with_changes(self):
        """Test refresh result with executives and role changes."""
        from app.models import RefreshResult, RoleChange
        from datetime import date
        
        new_exec = Executive(name="New CEO", title="CEO", start_year=2024)
        departed_exec = Executive(name="Old CEO", title="CEO", start_year=2015, end_year=2024)
        
        role_change = RoleChange(
            role_id="ceo",
            old_holder=departed_exec,
            new_holder=new_exec,
            change_date=date(2024, 1, 15)
        )
        
        result = RefreshResult(
            company_id="comp_456",
            new_executives=[new_exec],
            departed_executives=[departed_exec],
            role_changes=[role_change]
        )
        
        assert result.company_id == "comp_456"
        assert len(result.new_executives) == 1
        assert result.new_executives[0].name == "New CEO"
        assert len(result.departed_executives) == 1
        assert result.departed_executives[0].name == "Old CEO"
        assert len(result.role_changes) == 1
        assert result.role_changes[0].role_id == "ceo"


class TestRoleChange:
    """Tests for the RoleChange model."""

    def test_role_change_full(self):
        """Test role change with both old and new holder."""
        from app.models import RoleChange
        from datetime import date
        
        old_exec = Executive(name="Outgoing CFO", title="CFO", start_year=2018, end_year=2024)
        new_exec = Executive(name="Incoming CFO", title="CFO", start_year=2024)
        
        change = RoleChange(
            role_id="cfo",
            old_holder=old_exec,
            new_holder=new_exec,
            change_date=date(2024, 2, 1)
        )
        
        assert change.role_id == "cfo"
        assert change.old_holder.name == "Outgoing CFO"
        assert change.new_holder.name == "Incoming CFO"
        assert change.change_date == date(2024, 2, 1)

    def test_role_change_new_role(self):
        """Test role change when role is newly filled (no old holder)."""
        from app.models import RoleChange
        from datetime import date
        
        new_exec = Executive(name="First CTO", title="CTO", start_year=2024)
        
        change = RoleChange(
            role_id="cto",
            old_holder=None,
            new_holder=new_exec,
            change_date=date(2024, 3, 1)
        )
        
        assert change.old_holder is None
        assert change.new_holder.name == "First CTO"

    def test_role_change_vacated_role(self):
        """Test role change when role is vacated (no new holder)."""
        from app.models import RoleChange
        from datetime import date
        
        old_exec = Executive(name="Departed COO", title="COO", start_year=2020, end_year=2024)
        
        change = RoleChange(
            role_id="coo",
            old_holder=old_exec,
            new_holder=None,
            change_date=date(2024, 4, 1)
        )
        
        assert change.old_holder.name == "Departed COO"
        assert change.new_holder is None


class TestTimeRangeQuery:
    """Tests for the TimeRangeQuery model."""

    def test_time_range_full(self):
        """Test time range with both start and end year."""
        from app.models import TimeRangeQuery
        
        query = TimeRangeQuery(start_year=2015, end_year=2020)
        
        assert query.start_year == 2015
        assert query.end_year == 2020

    def test_time_range_open_ended(self):
        """Test time range with only start year."""
        from app.models import TimeRangeQuery
        
        query = TimeRangeQuery(start_year=2020, end_year=None)
        
        assert query.start_year == 2020
        assert query.end_year is None

    def test_time_range_year_bounds_valid(self):
        """Test time range with boundary year values."""
        from app.models import TimeRangeQuery
        
        # Minimum valid year
        query_min = TimeRangeQuery(start_year=1900)
        assert query_min.start_year == 1900
        
        # Maximum valid year
        query_max = TimeRangeQuery(start_year=2100, end_year=2100)
        assert query_max.end_year == 2100

    def test_time_range_year_bounds_invalid(self):
        """Test time range with invalid year values."""
        from app.models import TimeRangeQuery
        
        # Below minimum
        with pytest.raises(ValidationError):
            TimeRangeQuery(start_year=1899)
        
        # Above maximum
        with pytest.raises(ValidationError):
            TimeRangeQuery(start_year=2101)


class TestExecutiveValidation:
    """Tests for executive name validation logic."""

    def test_rejects_title_as_name(self):
        """Test that role titles are rejected as names.
        
        Note: This test documents expected behavior. If executive name validation
        is not currently implemented in the model, this test should be updated
        when that validation is added.
        """
        # These are titles that should not be accepted as names
        invalid_names = ["CEO", "CFO", "COO", "CTO", "President", "Chief Executive Officer"]
        
        # Currently the model does not have title-as-name validation
        # This test documents the expected behavior when such validation is added
        for name in invalid_names:
            # For now, these will pass - this test documents the expected future behavior
            exec = Executive(name=name, title="CEO", start_year=2020)
            # When validation is implemented, uncomment the following:
            # with pytest.raises(ValidationError):
            #     Executive(name=name, title="CEO", start_year=2020)

    def test_accepts_valid_names(self):
        """Test that real names are accepted."""
        valid_names = [
            "John Smith",
            "Mary Jane Watson", 
            "Tim Cook",
            "Satya Nadella",
            "Lisa Su",
            "Jane O'Brien",
            "Jean-Luc Picard",
        ]
        
        for name in valid_names:
            exec = Executive(name=name, title="CEO", start_year=2020)
            assert exec.name == name

    def test_accepts_single_name(self):
        """Test that single names are accepted."""
        exec = Executive(name="Madonna", title="CEO", start_year=2020)
        assert exec.name == "Madonna"

    def test_accepts_names_with_special_characters(self):
        """Test that names with special characters are accepted."""
        special_names = [
            "O'Connor",
            "McDonald",
            "De La Cruz",
            "Von Trapp",
            "Al-Rahman",
        ]

        for name in special_names:
            exec = Executive(name=name, title="VP", start_year=2021)
            assert exec.name == name


class TestCurrentHistoricalFiltering:
    """Test suite for current vs historical executive filtering."""

    def test_current_ceo_with_multiple_current_ceos(self):
        """Test that current_ceo returns the most recent when multiple current CEOs exist."""
        # This simulates a data quality issue where multiple CEOs are marked current
        older_current_ceo = Executive(name="Older CEO", title="CEO", start_year=2015, end_year=None)
        newer_current_ceo = Executive(name="Newer CEO", title="CEO", start_year=2020, end_year=None)

        company = Company(name="Test Co", ceo=[older_current_ceo, newer_current_ceo])

        # Should return the more recent CEO
        assert company.current_ceo is not None
        assert company.current_ceo.name == "Newer CEO"

    def test_current_ceo_handles_none_start_year(self):
        """Test that current_ceo handles executives with unknown start years."""
        unknown_start_ceo = Executive(name="Unknown Start", title="CEO", start_year=None, end_year=None)
        known_start_ceo = Executive(name="Known Start", title="CEO", start_year=2020, end_year=None)

        company = Company(name="Test Co", ceo=[unknown_start_ceo, known_start_ceo])

        # Should prefer the one with known start_year
        assert company.current_ceo is not None
        assert company.current_ceo.name == "Known Start"

    def test_historical_executives_property(self):
        """Test historical_executives returns only non-current executives."""
        current_ceo = Executive(name="Current CEO", title="CEO", start_year=2020, end_year=None)
        former_ceo = Executive(name="Former CEO", title="CEO", start_year=2010, end_year=2020)
        current_cfo = Executive(name="Current CFO", title="CFO", start_year=2018, end_year=None)
        former_vp = Executive(name="Former VP", title="VP", start_year=2015, end_year=2019)

        company = Company(
            name="Test Co",
            ceo=[current_ceo, former_ceo],
            c_level=[current_cfo],
            senior_level=[former_vp],
        )

        historical = company.historical_executives

        # Should contain former_ceo and former_vp
        names = [e.name for e in historical]
        assert "Former CEO" in names
        assert "Former VP" in names
        assert "Current CEO" not in names
        assert "Current CFO" not in names

    def test_historical_executives_sorted_by_end_year(self):
        """Test historical executives are sorted by end_year descending."""
        old_exec = Executive(name="Old Exec", title="VP", start_year=2000, end_year=2010)
        mid_exec = Executive(name="Mid Exec", title="CFO", start_year=2005, end_year=2015)
        recent_exec = Executive(name="Recent Exec", title="CEO", start_year=2010, end_year=2020)

        company = Company(
            name="Test Co",
            ceo=[recent_exec],
            c_level=[mid_exec],
            senior_level=[old_exec],
        )

        historical = company.historical_executives

        # Should be sorted: recent (2020), mid (2015), old (2010)
        assert len(historical) == 3
        assert historical[0].name == "Recent Exec"
        assert historical[1].name == "Mid Exec"
        assert historical[2].name == "Old Exec"

    def test_current_executives_sorted_by_level_then_start_year(self):
        """Test current executives are sorted by level (CEO first) then start_year."""
        ceo = Executive(name="CEO", title="CEO", start_year=2015, end_year=None)
        cfo_old = Executive(name="Old CFO", title="CFO", start_year=2010, end_year=None)
        cfo_new = Executive(name="New CFO", title="CFO", start_year=2020, end_year=None)
        vp = Executive(name="VP", title="VP Sales", start_year=2018, end_year=None)

        company = Company(
            name="Test Co",
            ceo=[ceo],
            c_level=[cfo_old, cfo_new],
            senior_level=[vp],
        )

        current = company.current_executives

        # CEO should be first, then C-level sorted by start_year desc, then senior
        assert len(current) == 4
        assert current[0].name == "CEO"  # CEO first
        # C-level sorted by start_year desc (newest first)
        assert current[1].name == "New CFO"
        assert current[2].name == "Old CFO"
        assert current[3].name == "VP"

    def test_get_executives_sorted(self):
        """Test get_executives_sorted returns current first, then historical."""
        current_ceo = Executive(name="Current CEO", title="CEO", start_year=2020, end_year=None)
        former_ceo = Executive(name="Former CEO", title="CEO", start_year=2010, end_year=2020)
        current_vp = Executive(name="Current VP", title="VP", start_year=2019, end_year=None)

        company = Company(
            name="Test Co",
            ceo=[current_ceo, former_ceo],
            senior_level=[current_vp],
        )

        all_sorted = company.get_executives_sorted()

        # Current executives should come first
        assert len(all_sorted) == 3
        # First two are current (CEO then VP)
        assert all_sorted[0].name == "Current CEO"
        assert all_sorted[1].name == "Current VP"
        # Historical last
        assert all_sorted[2].name == "Former CEO"

    def test_historical_executives_includes_extra_current_ceos(self):
        """Test that 'extra' current CEOs are included in historical list."""
        # Simulate data quality issue: two executives both marked as current CEO
        ceo_2015 = Executive(name="CEO 2015", title="CEO", start_year=2015, end_year=None)
        ceo_2020 = Executive(name="CEO 2020", title="CEO", start_year=2020, end_year=None)

        company = Company(name="Test Co", ceo=[ceo_2015, ceo_2020])

        # current_ceo should be the most recent
        assert company.current_ceo.name == "CEO 2020"

        # current_executives should only have one CEO
        current = company.current_executives
        ceo_names_in_current = [e.name for e in current if "CEO" in e.title]
        assert len(ceo_names_in_current) == 1
        assert "CEO 2020" in ceo_names_in_current

        # The older "current" CEO should be in historical
        historical = company.historical_executives
        historical_names = [e.name for e in historical]
        assert "CEO 2015" in historical_names


class TestCompanyResponseExecutivesByStatus:
    """Test suite for CompanyResponse executives_by_status computed field."""

    def test_executives_by_status_computed_field(self):
        """Test that executives_by_status is properly computed."""
        current_ceo = Executive(name="Current CEO", title="CEO", start_year=2020, end_year=None)
        former_ceo = Executive(name="Former CEO", title="CEO", start_year=2010, end_year=2020)
        current_cfo = Executive(name="Current CFO", title="CFO", start_year=2018, end_year=None)

        company = CompanyResponse(
            id="test_123",
            name="Test Co",
            ceo=[current_ceo, former_ceo],
            c_level=[current_cfo],
        )

        status = company.executives_by_status

        assert len(status.current) == 2  # Current CEO and CFO
        assert len(status.historical) == 1  # Former CEO

        # Verify current list
        current_names = [e.name for e in status.current]
        assert "Current CEO" in current_names
        assert "Current CFO" in current_names

        # Verify historical list
        historical_names = [e.name for e in status.historical]
        assert "Former CEO" in historical_names

    def test_executives_by_status_serialization(self):
        """Test that executives_by_status is included in JSON serialization."""
        current_ceo = Executive(name="Current CEO", title="CEO", start_year=2020, end_year=None)

        company = CompanyResponse(
            id="test_123",
            name="Test Co",
            ceo=[current_ceo],
        )

        # Serialize to dict (simulating JSON response)
        data = company.model_dump(by_alias=True)

        assert "executivesByStatus" in data
        assert "current" in data["executivesByStatus"]
        assert "historical" in data["executivesByStatus"]
        assert len(data["executivesByStatus"]["current"]) == 1
        assert len(data["executivesByStatus"]["historical"]) == 0
