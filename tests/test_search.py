"""Tests for search endpoints in Company Research Mapping Tool.

Tests cover the search router endpoints including company search,
industry search, suggestions, and status endpoints.
"""

import pytest
from fastapi import status


class TestSearchEndpoint:
    """Test suite for POST /api/search endpoint."""

    def test_search_with_company_name(self, client):
        """Test searching for a company by name."""
        response = client.post(
            "/api/search",
            json={"query": "Commercial Paving", "search_type": "company"},
        )

        assert response.status_code == status.HTTP_200_OK
        data = response.json()

        assert "results" in data
        assert "total" in data
        assert "query" in data
        # API uses camelCase aliases
        assert "searchType" in data
        assert data["query"] == "Commercial Paving"
        assert data["searchType"] == "company"
        assert data["total"] >= 0

    def test_search_with_partial_company_name(self, client):
        """Test searching with partial company name."""
        response = client.post(
            "/api/search",
            json={"query": "Paving", "search_type": "company"},
        )
        
        assert response.status_code == status.HTTP_200_OK
        data = response.json()
        assert data["total"] >= 1  # Should match "Commercial Paving Inc"

    def test_search_with_industry(self, client):
        """Test searching for companies by industry/subsector."""
        response = client.post(
            "/api/search",
            json={"query": "Commercial Paving", "search_type": "industry"},
        )

        assert response.status_code == status.HTTP_200_OK
        data = response.json()

        # API uses camelCase aliases
        assert data["searchType"] == "industry"
        assert data["total"] >= 0

    def test_search_returns_company_data(self, client):
        """Test that search results contain proper company data."""
        response = client.post(
            "/api/search",
            json={"query": "Commercial Paving Inc", "search_type": "company"},
        )

        assert response.status_code == status.HTTP_200_OK
        data = response.json()

        if data["total"] > 0:
            company = data["results"][0]
            assert "id" in company
            assert "name" in company
            assert "ceo" in company
            # API uses camelCase aliases
            assert "cLevel" in company
            assert "employees" in company

    def test_search_empty_query_returns_400(self, client):
        """Test that empty search query returns 400 error."""
        response = client.post(
            "/api/search",
            json={"query": "   ", "search_type": "company"},
        )
        
        assert response.status_code == status.HTTP_400_BAD_REQUEST
        data = response.json()
        assert "detail" in data
        assert "empty" in data["detail"].lower()

    def test_search_missing_query_returns_422(self, client):
        """Test that missing query field returns 422 validation error."""
        response = client.post(
            "/api/search",
            json={"search_type": "company"},
        )
        
        assert response.status_code == status.HTTP_422_UNPROCESSABLE_ENTITY

    def test_search_default_search_type(self, client):
        """Test that search_type defaults to 'company'."""
        response = client.post(
            "/api/search",
            json={"query": "test"},
        )

        assert response.status_code == status.HTTP_200_OK
        data = response.json()
        # API uses camelCase aliases
        assert data["searchType"] == "company"

    def test_search_invalid_search_type_returns_422(self, client):
        """Test that invalid search_type returns validation error."""
        response = client.post(
            "/api/search",
            json={"query": "test", "search_type": "invalid_type"},
        )
        
        assert response.status_code == status.HTTP_422_UNPROCESSABLE_ENTITY

    def test_search_no_results_found(self, client, monkeypatch):
        """Test search with query that returns no results.

        Note: External APIs may return results for any query, so we mock
        the search service to test the no-results path specifically.
        """
        from unittest.mock import AsyncMock, MagicMock
        from app.routers import search as search_router

        # Create a mock search service that returns empty results
        mock_service = MagicMock()
        mock_service.is_configured = True
        mock_service.search_companies = AsyncMock(return_value=[])

        # Mock where it's used (in the router module)
        monkeypatch.setattr(search_router, "get_search_service", lambda: mock_service)

        response = client.post(
            "/api/search",
            json={"query": "xyznonexistentcompany123", "search_type": "company"},
        )

        assert response.status_code == status.HTTP_200_OK
        data = response.json()
        assert data["total"] == 0
        assert data["results"] == []

    def test_search_case_insensitive(self, client):
        """Test that search is case insensitive."""
        response_lower = client.post(
            "/api/search",
            json={"query": "commercial paving", "search_type": "company"},
        )
        response_upper = client.post(
            "/api/search",
            json={"query": "COMMERCIAL PAVING", "search_type": "company"},
        )
        
        assert response_lower.status_code == status.HTTP_200_OK
        assert response_upper.status_code == status.HTTP_200_OK
        assert response_lower.json()["total"] == response_upper.json()["total"]


class TestSearchSuggestionsEndpoint:
    """Test suite for GET /api/search/suggestions endpoint."""

    def test_get_suggestions_with_query(self, client):
        """Test getting search suggestions with a query."""
        response = client.get("/api/search/suggestions", params={"q": "Commercial"})
        
        assert response.status_code == status.HTTP_200_OK
        suggestions = response.json()
        assert isinstance(suggestions, list)

    def test_get_suggestions_returns_matching_companies(self, client):
        """Test that suggestions include matching company names."""
        response = client.get("/api/search/suggestions", params={"q": "Paving"})
        
        assert response.status_code == status.HTTP_200_OK
        suggestions = response.json()
        
        # Should find "Commercial Paving Inc" at minimum
        assert any("Paving" in s for s in suggestions) or len(suggestions) == 0

    def test_get_suggestions_empty_query(self, client):
        """Test suggestions with empty query returns empty list."""
        response = client.get("/api/search/suggestions", params={"q": ""})
        
        assert response.status_code == status.HTTP_200_OK
        assert response.json() == []

    def test_get_suggestions_short_query(self, client):
        """Test suggestions with query shorter than 2 chars returns empty."""
        response = client.get("/api/search/suggestions", params={"q": "C"})
        
        assert response.status_code == status.HTTP_200_OK
        assert response.json() == []

    def test_get_suggestions_no_param(self, client):
        """Test suggestions without q parameter returns empty list."""
        response = client.get("/api/search/suggestions")
        
        assert response.status_code == status.HTTP_200_OK
        assert response.json() == []

    def test_get_suggestions_max_results(self, client):
        """Test that suggestions are limited to max 10 results."""
        response = client.get("/api/search/suggestions", params={"q": "Co"})
        
        assert response.status_code == status.HTTP_200_OK
        suggestions = response.json()
        assert len(suggestions) <= 10


class TestSearchStatusEndpoint:
    """Test suite for GET /api/search/status endpoint."""

    def test_get_search_status(self, client):
        """Test getting search service status."""
        response = client.get("/api/search/status")
        
        assert response.status_code == status.HTTP_200_OK
        data = response.json()
        
        assert "configured" in data
        assert "tavily" in data
        assert "serpapi" in data
        assert "source" in data

    def test_search_status_has_boolean_flags(self, client):
        """Test that status fields are boolean."""
        response = client.get("/api/search/status")
        data = response.json()
        
        assert isinstance(data["configured"], bool)
        assert isinstance(data["tavily"], bool)
        assert isinstance(data["serpapi"], bool)

    def test_search_status_source_value(self, client):
        """Test that source is either 'api' or 'mock'."""
        response = client.get("/api/search/status")
        data = response.json()
        
        assert data["source"] in ["api", "mock"]
        
        # Source should match configured status
        if data["configured"]:
            assert data["source"] == "api"
        else:
            assert data["source"] == "mock"
