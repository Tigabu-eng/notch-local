"""Tests for new API endpoints in Company Research Mapping Tool.

Tests cover role drilling endpoints, refresh/validation endpoints,
enhanced search v2 endpoint, and division endpoints.
"""

import pytest
from fastapi import status


class TestRoleEndpoints:
    """Tests for role-related endpoints."""

    def test_list_standard_roles(self, client):
        """Test GET /roles/standard returns predefined roles."""
        response = client.get("/api/roles/standard")
        
        assert response.status_code == status.HTTP_200_OK
        roles = response.json()
        
        assert isinstance(roles, list)
        assert len(roles) >= 6  # CEO, CFO, COO, CTO, CMO, CIO
        
        # Verify standard roles are present
        role_ids = [role["id"] for role in roles]
        assert "ceo" in role_ids
        assert "cfo" in role_ids
        assert "coo" in role_ids
        assert "cto" in role_ids
        assert "cmo" in role_ids
        assert "cio" in role_ids

    def test_list_standard_roles_structure(self, client):
        """Test that standard roles have proper structure."""
        response = client.get("/api/roles/standard")
        roles = response.json()
        
        for role in roles:
            assert "id" in role
            assert "name" in role
            assert "category" in role
            assert "isStandard" in role
            assert role["isStandard"] is True

    def test_list_company_roles(self, client):
        """Test GET /companies/{id}/roles returns roles."""
        response = client.get("/api/companies/comp_001/roles")
        
        assert response.status_code == status.HTTP_200_OK
        roles = response.json()
        
        assert isinstance(roles, list)
        # Should include at least the standard roles
        assert len(roles) >= 6

    def test_list_company_roles_not_found(self, client):
        """Test GET /companies/{id}/roles returns 404 for invalid company."""
        response = client.get("/api/companies/nonexistent_999/roles")
        
        assert response.status_code == status.HTTP_404_NOT_FOUND

    def test_get_role_timeline(self, client):
        """Test GET /companies/{id}/roles/{role_id}/timeline."""
        response = client.get("/api/companies/comp_001/roles/ceo/timeline")
        
        assert response.status_code == status.HTTP_200_OK
        timeline = response.json()
        
        assert isinstance(timeline, list)

    def test_get_role_timeline_invalid_company(self, client):
        """Test timeline with non-existent company returns 404."""
        response = client.get("/api/companies/nonexistent_999/roles/ceo/timeline")
        
        assert response.status_code == status.HTTP_404_NOT_FOUND

    def test_get_role_timeline_invalid_role(self, client):
        """Test timeline with non-existent role returns 404."""
        response = client.get("/api/companies/comp_001/roles/nonexistent_role/timeline")
        
        assert response.status_code == status.HTTP_404_NOT_FOUND

    def test_get_role_timeline_with_date_filter(self, client):
        """Test timeline with start_year and end_year params."""
        response = client.get(
            "/api/companies/comp_001/roles/ceo/timeline",
            params={"start_year": 2020, "end_year": 2024}
        )
        
        assert response.status_code == status.HTTP_200_OK
        timeline = response.json()
        
        assert isinstance(timeline, list)

    def test_get_role_timeline_with_only_start_year(self, client):
        """Test timeline with only start_year param."""
        response = client.get(
            "/api/companies/comp_001/roles/ceo/timeline",
            params={"start_year": 2015}
        )
        
        assert response.status_code == status.HTTP_200_OK

    def test_get_role_timeline_with_only_end_year(self, client):
        """Test timeline with only end_year param."""
        response = client.get(
            "/api/companies/comp_001/roles/ceo/timeline",
            params={"end_year": 2020}
        )
        
        assert response.status_code == status.HTTP_200_OK

    def test_get_current_role_holder(self, client):
        """Test GET /companies/{id}/roles/{role_id}/current."""
        response = client.get("/api/companies/comp_001/roles/ceo/current")
        
        assert response.status_code == status.HTTP_200_OK
        # Response could be null if no current holder, or a RoleHolder object

    def test_get_current_role_holder_invalid_company(self, client):
        """Test current role holder with non-existent company returns 404."""
        response = client.get("/api/companies/nonexistent_999/roles/ceo/current")
        
        assert response.status_code == status.HTTP_404_NOT_FOUND

    def test_get_current_role_holder_invalid_role(self, client):
        """Test current role holder with non-existent role returns 404."""
        response = client.get("/api/companies/comp_001/roles/nonexistent_role/current")
        
        assert response.status_code == status.HTTP_404_NOT_FOUND

    def test_create_custom_role(self, client):
        """Test POST /companies/{id}/roles creates custom role."""
        role_data = {
            "id": "vp_operations_test",
            "name": "VP Operations",
            "category": "senior",
            "isStandard": False,
        }
        
        response = client.post("/api/companies/comp_001/roles", json=role_data)
        
        assert response.status_code == status.HTTP_201_CREATED
        created_role = response.json()
        
        assert created_role["id"] == "vp_operations_test"
        assert created_role["name"] == "VP Operations"
        assert created_role["category"] == "senior"
        assert created_role["isStandard"] is False

    def test_create_custom_role_invalid_company(self, client):
        """Test creating custom role for non-existent company returns 404."""
        role_data = {
            "id": "test_role",
            "name": "Test Role",
            "category": "senior",
            "isStandard": False,
        }
        
        response = client.post("/api/companies/nonexistent_999/roles", json=role_data)
        
        assert response.status_code == status.HTTP_404_NOT_FOUND

    def test_create_custom_role_duplicate_standard(self, client):
        """Test creating custom role with standard role ID returns 409."""
        role_data = {
            "id": "ceo",  # Already a standard role
            "name": "Chief Executive Officer",
            "category": "c_suite",
            "isStandard": False,
        }
        
        response = client.post("/api/companies/comp_001/roles", json=role_data)
        
        assert response.status_code == status.HTTP_409_CONFLICT

    def test_create_custom_role_duplicate_custom(self, client):
        """Test creating duplicate custom role returns 409."""
        role_data = {
            "id": "unique_test_role_123",
            "name": "Unique Test Role",
            "category": "custom",
            "isStandard": False,
        }
        
        # Create first role
        response1 = client.post("/api/companies/comp_001/roles", json=role_data)
        assert response1.status_code == status.HTTP_201_CREATED
        
        # Try to create duplicate
        response2 = client.post("/api/companies/comp_001/roles", json=role_data)
        assert response2.status_code == status.HTTP_409_CONFLICT


class TestRefreshEndpoints:
    """Tests for refresh/validation endpoints."""

    def test_refresh_company(self, client):
        """Test POST /companies/{id}/refresh returns RefreshResult."""
        response = client.post("/api/companies/comp_001/refresh")
        
        assert response.status_code == status.HTTP_200_OK
        result = response.json()
        
        assert "companyId" in result
        assert result["companyId"] == "comp_001"
        assert "newExecutives" in result
        assert "departedExecutives" in result
        assert "roleChanges" in result
        
        assert isinstance(result["newExecutives"], list)
        assert isinstance(result["departedExecutives"], list)
        assert isinstance(result["roleChanges"], list)

    def test_refresh_company_not_found(self, client):
        """Test refresh non-existent company returns 404."""
        response = client.post("/api/companies/nonexistent_refresh_999/refresh")
        
        assert response.status_code == status.HTTP_404_NOT_FOUND

    def test_refresh_company_updates_validation(self, client):
        """Test that refresh updates validation metadata."""
        # First refresh
        client.post("/api/companies/comp_001/refresh")
        
        # Check validation status
        validation_response = client.get("/api/companies/comp_001/validation")
        assert validation_response.status_code == status.HTTP_200_OK
        
        validation = validation_response.json()
        assert validation is not None
        assert validation["needsRefresh"] is False

    def test_list_stale_companies(self, client):
        """Test GET /companies/stale returns stale companies.
        
        NOTE: This test is skipped due to a route ordering issue.
        The /companies/stale route is defined after /companies/{company_id},
        so FastAPI matches "stale" as a company_id and returns 404.
        The route order should be fixed in the router to make static paths
        appear before dynamic path parameters.
        """
        response = client.get("/api/companies/stale")
        
        assert response.status_code == status.HTTP_200_OK
        stale_companies = response.json()
        
        assert isinstance(stale_companies, list)
        
        # Each company should have proper structure
        for company in stale_companies:
            assert "id" in company
            assert "name" in company

    def test_list_stale_companies_with_threshold(self, client):
        """Test GET /companies/stale with days_threshold parameter."""
        response = client.get("/api/companies/stale", params={"days_threshold": 30})
        
        assert response.status_code == status.HTTP_200_OK
        stale_companies = response.json()
        
        assert isinstance(stale_companies, list)

    def test_list_stale_companies_high_threshold(self, client):
        """Test GET /companies/stale with high days_threshold returns fewer results."""
        # With very high threshold, fewer companies should be stale
        response = client.get("/api/companies/stale", params={"days_threshold": 3650})
        
        assert response.status_code == status.HTTP_200_OK

    def test_get_validation_status(self, client):
        """Test GET /companies/{id}/validation returns ValidationMetadata."""
        response = client.get("/api/companies/comp_001/validation")
        
        assert response.status_code == status.HTTP_200_OK
        validation = response.json()
        
        # Validation could be None if no data exists
        if validation is not None:
            assert "lastValidated" in validation
            assert "confidence" in validation
            assert "needsRefresh" in validation

    def test_get_validation_status_not_found(self, client):
        """Test validation status for non-existent company returns 404."""
        response = client.get("/api/companies/nonexistent_validation_999/validation")
        
        assert response.status_code == status.HTTP_404_NOT_FOUND

    def test_validation_status_structure(self, client):
        """Test validation metadata has proper structure and values."""
        response = client.get("/api/companies/comp_001/validation")
        validation = response.json()
        
        if validation is not None:
            # Confidence should be between 0 and 1
            assert 0.0 <= validation["confidence"] <= 1.0
            # needsRefresh should be boolean
            assert isinstance(validation["needsRefresh"], bool)


class TestSearchV2Endpoint:
    """Tests for enhanced search endpoint."""

    def test_search_v2_basic(self, client):
        """Test POST /search/v2 with basic query."""
        response = client.post(
            "/api/search/v2",
            json={"query": "Paving", "search_type": "company"}
        )
        
        assert response.status_code == status.HTTP_200_OK
        data = response.json()
        
        assert "results" in data
        assert "total" in data
        assert "query" in data
        assert "searchType" in data
        assert data["query"] == "Paving"

    def test_search_v2_with_time_range(self, client):
        """Test search with time_range filter."""
        response = client.post(
            "/api/search/v2",
            json={
                "query": "Paving",
                "search_type": "company",
                "time_range": {
                    "start_year": 2015,
                    "end_year": 2020
                }
            }
        )
        
        assert response.status_code == status.HTTP_200_OK
        data = response.json()
        
        assert "results" in data
        # All executives in results should overlap with the time range
        for company in data["results"]:
            for exec_list_key in ["ceo", "cLevel", "seniorLevel"]:
                for exec in company.get(exec_list_key, []):
                    # Verify the executive's tenure overlaps with query range
                    exec_start = exec["startYear"]
                    exec_end = exec.get("endYear")  # Could be None for current
                    
                    # If end_year is None, executive is current, so overlaps
                    if exec_end is not None:
                        # Executive ended before query start - should not be here
                        assert not (exec_end < 2015), f"Executive {exec['name']} ended before query start"
                    # Executive started after query end - should not be here
                    # (skip check if start_year is unknown - these are included conservatively)
                    if exec_start is not None:
                        assert exec_start <= 2020, f"Executive {exec['name']} started after query end"

    def test_search_v2_with_role_filter(self, client):
        """Test search with role_filter."""
        response = client.post(
            "/api/search/v2",
            json={
                "query": "Paving",
                "search_type": "company",
                "role_filter": ["ceo", "cfo"]
            }
        )
        
        assert response.status_code == status.HTTP_200_OK
        data = response.json()
        
        assert "results" in data

    def test_search_v2_combined_filters(self, client):
        """Test search with both time and role filters."""
        response = client.post(
            "/api/search/v2",
            json={
                "query": "Commercial",
                "search_type": "company",
                "time_range": {
                    "start_year": 2010,
                    "end_year": 2024
                },
                "role_filter": ["ceo"]
            }
        )
        
        assert response.status_code == status.HTTP_200_OK
        data = response.json()
        
        assert "results" in data

    def test_search_v2_empty_query_returns_400(self, client):
        """Test that empty query returns 400 error."""
        response = client.post(
            "/api/search/v2",
            json={"query": "   ", "search_type": "company"}
        )
        
        assert response.status_code == status.HTTP_400_BAD_REQUEST

    def test_search_v2_no_results_mock(self, client):
        """Test search that returns no matching results using mock data.
        
        This test uses a role_filter that will exclude all executives
        from the mock data results to ensure we get an empty result set.
        """
        response = client.post(
            "/api/search/v2",
            json={
                "query": "Paving",
                "search_type": "company",
                # Use a role filter that won't match any executives
                "role_filter": ["nonexistent_role_xyz"]
            }
        )
        
        assert response.status_code == status.HTTP_200_OK
        data = response.json()
        
        # With a role filter that doesn't match, should get 0 results
        assert data["total"] == 0
        assert data["results"] == []

    def test_search_v2_industry_type(self, client):
        """Test search with industry search type."""
        response = client.post(
            "/api/search/v2",
            json={"query": "Paving", "search_type": "industry"}
        )
        
        assert response.status_code == status.HTTP_200_OK
        data = response.json()
        
        assert data["searchType"] == "industry"

    def test_search_v2_time_range_current_only(self, client):
        """Test search with time_range having no end_year (current)."""
        response = client.post(
            "/api/search/v2",
            json={
                "query": "Paving",
                "search_type": "company",
                "time_range": {
                    "start_year": 2020
                }
            }
        )
        
        assert response.status_code == status.HTTP_200_OK

    def test_search_v2_multiple_role_filter(self, client):
        """Test search with multiple roles in filter."""
        response = client.post(
            "/api/search/v2",
            json={
                "query": "Commercial",
                "search_type": "company",
                "role_filter": ["ceo", "cfo", "coo", "vp"]
            }
        )
        
        assert response.status_code == status.HTTP_200_OK

    def test_search_v2_response_structure(self, client):
        """Test search v2 response has proper structure."""
        response = client.post(
            "/api/search/v2",
            json={"query": "Paving", "search_type": "company"}
        )
        
        assert response.status_code == status.HTTP_200_OK
        data = response.json()
        
        assert "results" in data
        assert "total" in data
        assert "query" in data
        assert "searchType" in data
        assert "source" in data
        
        # Verify source is one of expected values
        assert data["source"] in ["api", "mock"]


class TestDivisionEndpoints:
    """Tests for division endpoints."""

    def test_list_divisions(self, client):
        """Test GET /companies/{id}/divisions."""
        response = client.get("/api/companies/comp_001/divisions")
        
        assert response.status_code == status.HTTP_200_OK
        divisions = response.json()
        
        assert isinstance(divisions, list)

    def test_list_divisions_not_found(self, client):
        """Test list divisions for non-existent company returns 404."""
        response = client.get("/api/companies/nonexistent_div_999/divisions")
        
        assert response.status_code == status.HTTP_404_NOT_FOUND

    def test_create_division(self, client):
        """Test POST /companies/{id}/divisions."""
        division_data = {
            "id": "test_division_123",
            "name": "Eastern Region",
            "parentDivisionId": None
        }
        
        response = client.post(
            "/api/companies/comp_001/divisions",
            json=division_data
        )
        
        assert response.status_code == status.HTTP_201_CREATED
        created_division = response.json()
        
        assert created_division["id"] == "test_division_123"
        assert created_division["name"] == "Eastern Region"

    def test_create_division_not_found(self, client):
        """Test create division for non-existent company returns 404."""
        division_data = {
            "id": "test_division",
            "name": "Test Division"
        }
        
        response = client.post(
            "/api/companies/nonexistent_div_999/divisions",
            json=division_data
        )
        
        assert response.status_code == status.HTTP_404_NOT_FOUND

    def test_create_division_duplicate(self, client):
        """Test creating duplicate division returns 409."""
        division_data = {
            "id": "unique_division_456",
            "name": "Unique Division"
        }
        
        # Create first division
        response1 = client.post(
            "/api/companies/comp_001/divisions",
            json=division_data
        )
        assert response1.status_code == status.HTTP_201_CREATED
        
        # Try to create duplicate
        response2 = client.post(
            "/api/companies/comp_001/divisions",
            json=division_data
        )
        assert response2.status_code == status.HTTP_409_CONFLICT

    def test_create_division_with_parent(self, client):
        """Test creating division with parent_division_id."""
        # First create a parent division
        parent_data = {
            "id": "parent_division_789",
            "name": "Parent Division"
        }
        client.post("/api/companies/comp_001/divisions", json=parent_data)
        
        # Create child division
        child_data = {
            "id": "child_division_789",
            "name": "Child Division",
            "parentDivisionId": "parent_division_789"
        }
        
        response = client.post(
            "/api/companies/comp_001/divisions",
            json=child_data
        )
        
        assert response.status_code == status.HTTP_201_CREATED
        created_division = response.json()
        
        assert created_division["parentDivisionId"] == "parent_division_789"

    def test_get_division_leadership(self, client):
        """Test GET /companies/{id}/divisions/{div_id}/leadership."""
        # First create a division to test with
        division_data = {
            "id": "leadership_test_div",
            "name": "Leadership Test Division"
        }
        client.post("/api/companies/comp_001/divisions", json=division_data)
        
        # Get leadership for the division
        response = client.get(
            "/api/companies/comp_001/divisions/leadership_test_div/leadership"
        )
        
        assert response.status_code == status.HTTP_200_OK
        leadership = response.json()
        
        assert isinstance(leadership, list)

    def test_get_division_leadership_company_not_found(self, client):
        """Test division leadership for non-existent company returns 404."""
        response = client.get(
            "/api/companies/nonexistent_999/divisions/some_div/leadership"
        )
        
        assert response.status_code == status.HTTP_404_NOT_FOUND

    def test_get_division_leadership_division_not_found(self, client):
        """Test division leadership for non-existent division returns 404."""
        response = client.get(
            "/api/companies/comp_001/divisions/nonexistent_div/leadership"
        )
        
        assert response.status_code == status.HTTP_404_NOT_FOUND

    def test_list_divisions_after_creation(self, client):
        """Test that created divisions appear in list."""
        division_data = {
            "id": "list_test_div_unique",
            "name": "List Test Division"
        }
        
        # Create division
        create_response = client.post(
            "/api/companies/comp_001/divisions",
            json=division_data
        )
        assert create_response.status_code == status.HTTP_201_CREATED
        
        # List divisions
        list_response = client.get("/api/companies/comp_001/divisions")
        divisions = list_response.json()
        
        division_ids = [d["id"] for d in divisions]
        assert "list_test_div_unique" in division_ids

    def test_division_structure(self, client):
        """Test that division has proper structure."""
        division_data = {
            "id": "structure_test_div",
            "name": "Structure Test Division"
        }
        
        response = client.post(
            "/api/companies/comp_001/divisions",
            json=division_data
        )
        
        assert response.status_code == status.HTTP_201_CREATED
        division = response.json()
        
        assert "id" in division
        assert "name" in division
        assert "parentDivisionId" in division
