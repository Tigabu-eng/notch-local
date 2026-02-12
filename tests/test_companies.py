"""Tests for companies CRUD endpoints in Company Research Mapping Tool.

Tests cover the companies router endpoints including listing, creating,
retrieving, updating, and deleting companies.
"""

import pytest
from fastapi import status


class TestListCompaniesEndpoint:
    """Test suite for GET /api/companies endpoint."""

    def test_list_companies(self, client):
        """Test listing all companies."""
        response = client.get("/api/companies")
        
        assert response.status_code == status.HTTP_200_OK
        data = response.json()
        
        assert isinstance(data, list)
        # Should have at least the mock companies
        assert len(data) >= 2

    def test_list_companies_returns_company_data(self, client):
        """Test that listed companies have proper structure."""
        response = client.get("/api/companies")
        data = response.json()

        if len(data) > 0:
            company = data[0]
            assert "id" in company
            assert "name" in company
            assert "ceo" in company
            # API uses camelCase aliases
            assert "cLevel" in company
            assert "seniorLevel" in company

    def test_list_companies_with_limit(self, client):
        """Test listing companies with limit parameter."""
        response = client.get("/api/companies", params={"limit": 1})
        
        assert response.status_code == status.HTTP_200_OK
        data = response.json()
        assert len(data) <= 1

    def test_list_companies_with_offset(self, client):
        """Test listing companies with offset parameter."""
        # First get all companies
        all_response = client.get("/api/companies")
        all_companies = all_response.json()
        
        if len(all_companies) > 1:
            # Then get with offset
            offset_response = client.get("/api/companies", params={"offset": 1})
            offset_companies = offset_response.json()
            
            assert len(offset_companies) == len(all_companies) - 1

    def test_list_companies_pagination(self, client):
        """Test pagination with both limit and offset."""
        response = client.get("/api/companies", params={"limit": 1, "offset": 0})
        
        assert response.status_code == status.HTTP_200_OK
        data = response.json()
        assert len(data) <= 1


class TestCreateCompanyEndpoint:
    """Test suite for POST /api/companies endpoint."""

    def test_create_company(self, client, sample_company_data):
        """Test creating a new company."""
        # Modify name to avoid conflict with existing data
        sample_company_data["name"] = "Unique Test Company XYZ"
        
        response = client.post("/api/companies", json=sample_company_data)
        
        assert response.status_code == status.HTTP_201_CREATED
        data = response.json()
        
        assert "id" in data
        assert data["name"] == sample_company_data["name"]
        assert data["employees"] == sample_company_data["employees"]
        assert len(data["ceo"]) == 1

    def test_create_company_generates_id(self, client):
        """Test that creating a company generates a unique ID."""
        company_data = {"name": "ID Test Company 123"}
        response = client.post("/api/companies", json=company_data)
        
        assert response.status_code == status.HTTP_201_CREATED
        data = response.json()
        
        assert data["id"].startswith("comp_")
        assert len(data["id"]) > 5

    def test_create_company_sets_timestamp(self, client):
        """Test that creating a company sets the updated timestamp."""
        company_data = {"name": "Timestamp Test Company 456"}
        response = client.post("/api/companies", json=company_data)
        
        assert response.status_code == status.HTTP_201_CREATED
        data = response.json()
        
        assert data["updated"] is not None

    def test_create_company_minimal_data(self, client):
        """Test creating a company with only required fields."""
        minimal_data = {"name": "Minimal Company ABC"}
        response = client.post("/api/companies", json=minimal_data)
        
        assert response.status_code == status.HTTP_201_CREATED
        data = response.json()
        
        assert data["name"] == "Minimal Company ABC"
        assert data["ceo"] == []
        assert data["employees"] is None

    def test_create_company_empty_name_returns_422(self, client):
        """Test that empty company name returns validation error."""
        response = client.post("/api/companies", json={"name": ""})
        
        assert response.status_code == status.HTTP_422_UNPROCESSABLE_ENTITY

    def test_create_duplicate_company_returns_409(self, client):
        """Test that creating a duplicate company name returns 409."""
        # Create first company
        first_name = "Duplicate Test Company 789"
        client.post("/api/companies", json={"name": first_name})
        
        # Try to create duplicate
        response = client.post("/api/companies", json={"name": first_name})
        
        assert response.status_code == status.HTTP_409_CONFLICT
        assert "already exists" in response.json()["detail"]

    def test_create_company_with_executives(self, client):
        """Test creating a company with executive data."""
        # API input uses snake_case (CompanyCreate model)
        company_data = {
            "name": "Executive Test Company 101",
            "ceo": [
                {"name": "Test CEO", "title": "CEO", "start_year": 2020}
            ],
            "c_level": [
                {"name": "Test CFO", "title": "CFO", "start_year": 2021}
            ],
        }

        response = client.post("/api/companies", json=company_data)

        assert response.status_code == status.HTTP_201_CREATED
        data = response.json()

        assert len(data["ceo"]) == 1
        assert data["ceo"][0]["name"] == "Test CEO"
        # API output uses camelCase (CompanyResponse model)
        assert len(data["cLevel"]) == 1


class TestGetCompanyEndpoint:
    """Test suite for GET /api/companies/{id} endpoint."""

    def test_get_company_by_id(self, client):
        """Test retrieving a company by its ID."""
        # comp_001 is one of the mock companies
        response = client.get("/api/companies/comp_001")
        
        assert response.status_code == status.HTTP_200_OK
        data = response.json()
        
        assert data["id"] == "comp_001"
        assert data["name"] == "Commercial Paving Inc"

    def test_get_company_returns_full_data(self, client):
        """Test that retrieved company has all fields."""
        response = client.get("/api/companies/comp_001")
        data = response.json()

        # API uses camelCase aliases
        expected_fields = [
            "id", "name", "ceo", "cLevel", "seniorLevel",
            "employees", "ownership", "acquisitionDate",
            "subsector", "notes", "updated",
            "networkStatus", "contactStatus"
        ]

        for field in expected_fields:
            assert field in data

    def test_get_company_not_found_returns_404(self, client):
        """Test that non-existent company ID returns 404."""
        response = client.get("/api/companies/nonexistent_id_123")
        
        assert response.status_code == status.HTTP_404_NOT_FOUND
        data = response.json()
        
        assert "detail" in data
        assert "not found" in data["detail"].lower()

    def test_get_company_executives_have_proper_structure(self, client):
        """Test that company executives have proper structure."""
        response = client.get("/api/companies/comp_001")
        data = response.json()

        if data["ceo"]:
            ceo = data["ceo"][0]
            assert "name" in ceo
            assert "title" in ceo
            # API uses camelCase aliases
            assert "startYear" in ceo
            assert "endYear" in ceo


class TestUpdateCompanyEndpoint:
    """Test suite for PUT /api/companies/{id} endpoint."""

    def test_update_company(self, client):
        """Test updating a company."""
        # First create a company to update
        create_response = client.post(
            "/api/companies",
            json={"name": "Update Test Company 202"}
        )
        company_id = create_response.json()["id"]
        
        # Update the company
        update_data = {"employees": 999, "notes": "Updated notes"}
        response = client.put(f"/api/companies/{company_id}", json=update_data)
        
        assert response.status_code == status.HTTP_200_OK
        data = response.json()
        
        assert data["employees"] == 999
        assert data["notes"] == "Updated notes"

    def test_update_company_not_found_returns_404(self, client):
        """Test updating non-existent company returns 404."""
        response = client.put(
            "/api/companies/nonexistent_999",
            json={"name": "Updated Name"}
        )
        
        assert response.status_code == status.HTTP_404_NOT_FOUND


class TestDeleteCompanyEndpoint:
    """Test suite for DELETE /api/companies/{id} endpoint."""

    def test_delete_company(self, client):
        """Test deleting a company."""
        # First create a company to delete
        create_response = client.post(
            "/api/companies",
            json={"name": "Delete Test Company 303"}
        )
        company_id = create_response.json()["id"]
        
        # Delete the company
        response = client.delete(f"/api/companies/{company_id}")
        
        assert response.status_code == status.HTTP_204_NO_CONTENT
        
        # Verify it's deleted
        get_response = client.get(f"/api/companies/{company_id}")
        assert get_response.status_code == status.HTTP_404_NOT_FOUND

    def test_delete_company_not_found_returns_404(self, client):
        """Test deleting non-existent company returns 404."""
        response = client.delete("/api/companies/nonexistent_delete_999")
        
        assert response.status_code == status.HTTP_404_NOT_FOUND


class TestGetCompanyExecutivesEndpoint:
    """Test suite for GET /api/companies/{id}/executives endpoint."""

    def test_get_company_executives(self, client):
        """Test getting all executives for a company."""
        response = client.get("/api/companies/comp_001/executives")
        
        assert response.status_code == status.HTTP_200_OK
        data = response.json()
        
        assert isinstance(data, list)
        assert len(data) > 0

    def test_get_company_executives_current_only(self, client):
        """Test getting only current executives."""
        # Get all executives
        all_response = client.get("/api/companies/comp_001/executives")
        all_executives = all_response.json()

        # Get current only
        current_response = client.get(
            "/api/companies/comp_001/executives",
            params={"current_only": True}
        )
        current_executives = current_response.json()

        assert current_response.status_code == status.HTTP_200_OK
        # Current executives should be subset or equal to all
        assert len(current_executives) <= len(all_executives)

        # All returned executives should have endYear as None (camelCase API)
        for exec in current_executives:
            assert exec["endYear"] is None

    def test_get_executives_not_found_returns_404(self, client):
        """Test getting executives for non-existent company."""
        response = client.get("/api/companies/nonexistent_exec_999/executives")
        
        assert response.status_code == status.HTTP_404_NOT_FOUND
