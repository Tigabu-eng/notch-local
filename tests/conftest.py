"""Pytest fixtures for Company Research Mapping Tool tests.

This module provides shared fixtures for testing the FastAPI application,
including the test client and common test data.
"""

import pytest
from fastapi.testclient import TestClient

from app.main import app


@pytest.fixture
def client():
    """Create a test client for the FastAPI application.
    
    Returns:
        TestClient: A test client instance for making requests to the API.
    """
    return TestClient(app)


@pytest.fixture
def sample_executive_data():
    """Sample executive data for testing.
    
    Returns:
        dict: Valid executive data dictionary.
    """
    return {
        "name": "John Smith",
        "title": "CEO",
        "start_year": 2020,
        "end_year": None,
    }


@pytest.fixture
def sample_company_data():
    """Sample company data for testing creation.
    
    Returns:
        dict: Valid company creation data dictionary.
    """
    return {
        "name": "Test Company Inc",
        "ceo": [
            {
                "name": "Jane Doe",
                "title": "CEO",
                "start_year": 2018,
                "end_year": None,
            }
        ],
        "c_level": [
            {
                "name": "Bob Wilson",
                "title": "CFO",
                "start_year": 2019,
                "end_year": None,
            }
        ],
        "senior_level": [],
        "employees": 100,
        "ownership": "Private",
        "subsector": "Technology",
        "notes": "Test company for unit testing",
        "network_status": "in_network",
        "contact_status": "not_contacted",
    }
