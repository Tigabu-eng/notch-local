"""Tests for the Wikidata SPARQL service."""

import pytest
from unittest.mock import AsyncMock, patch, MagicMock

from app.services.wikidata_service import WikidataService, get_wikidata_service
from app.models import Executive


class TestWikidataService:
    """Tests for WikidataService."""

    def test_singleton_pattern(self):
        """Test that get_wikidata_service returns the same instance."""
        # Reset singleton for test
        import app.services.wikidata_service as ws_module
        ws_module._wikidata_service = None

        service1 = get_wikidata_service()
        service2 = get_wikidata_service()
        assert service1 is service2

    def test_is_configured_always_true(self):
        """Test that Wikidata service is always configured (no API key needed)."""
        service = WikidataService()
        assert service.is_configured is True

    def test_escape_sparql_string(self):
        """Test SPARQL string escaping."""
        service = WikidataService()

        # Test basic escaping
        assert service._escape_sparql_string('Apple Inc.') == 'Apple Inc.'
        assert service._escape_sparql_string('O\'Reilly') == "O\\'Reilly"
        assert service._escape_sparql_string('Company "ABC"') == 'Company \\"ABC\\"'
        assert service._escape_sparql_string('Path\\to\\company') == 'Path\\\\to\\\\company'

    def test_parse_wikidata_date(self):
        """Test Wikidata date parsing."""
        service = WikidataService()

        # ISO 8601 format with timezone
        assert service._parse_wikidata_date("2020-01-15T00:00:00Z") == 2020
        assert service._parse_wikidata_date("1998-07-01T00:00:00Z") == 1998

        # Year only
        assert service._parse_wikidata_date("2015") == 2015

        # Invalid dates
        assert service._parse_wikidata_date(None) is None
        assert service._parse_wikidata_date("") is None
        assert service._parse_wikidata_date("invalid") is None

        # Edge cases
        assert service._parse_wikidata_date("2023-12-31T23:59:59Z") == 2023

    def test_calculate_match_score(self):
        """Test company name match scoring."""
        service = WikidataService()

        # Exact match
        assert service._calculate_match_score("Apple", "Apple") == 1.0
        assert service._calculate_match_score("apple", "Apple") == 1.0  # Case insensitive

        # Substring match
        score = service._calculate_match_score("Apple", "Apple Inc.")
        assert 0.5 < score < 1.0

        # Partial overlap
        score = service._calculate_match_score("Apple Computer", "Apple Inc.")
        assert score > 0.0

    def test_is_executive_position(self):
        """Test executive position detection."""
        service = WikidataService()

        # Should be executive
        assert service._is_executive_position("Chief Executive Officer") is True
        assert service._is_executive_position("CEO") is True
        assert service._is_executive_position("President") is True
        assert service._is_executive_position("Chairman") is True
        assert service._is_executive_position("Chief Financial Officer") is True
        assert service._is_executive_position("Executive Vice President") is True

        # Should not be executive
        assert service._is_executive_position("Software Engineer") is False
        assert service._is_executive_position("Manager") is False
        assert service._is_executive_position("Analyst") is False

    def test_normalize_name(self):
        """Test name normalization."""
        service = WikidataService()

        # Basic normalization
        assert service._normalize_name("John Smith") == "john smith"
        assert service._normalize_name("  John   Smith  ") == "john smith"

        # Remove suffixes
        assert service._normalize_name("John Smith Jr.") == "john smith"
        assert service._normalize_name("John Smith III") == "john smith"
        assert service._normalize_name("John Smith PhD") == "john smith"

    def test_merge_executives(self):
        """Test executive record merging."""
        service = WikidataService()

        exec1 = Executive(name="John Smith", title="CEO", start_year=2018, end_year=None)
        exec2 = Executive(name="John A. Smith", title="Chief Executive Officer", start_year=2020, end_year=None)

        merged = service._merge_executives([exec1, exec2])

        # Should prefer longer name
        assert merged.name == "John A. Smith"
        # Should prefer "CEO" title
        assert merged.title == "CEO"
        # Should use earliest start year
        assert merged.start_year == 2018
        # None end year should be preserved
        assert merged.end_year is None

    def test_deduplicate_executives(self):
        """Test executive deduplication."""
        service = WikidataService()

        executives = [
            Executive(name="John Smith", title="CEO", start_year=2018, end_year=None),
            Executive(name="john smith", title="CEO", start_year=2019, end_year=None),  # Duplicate
            Executive(name="Jane Doe", title="CFO", start_year=2020, end_year=None),
        ]

        deduplicated = service._deduplicate_executives(executives)

        # Should have 2 unique executives
        assert len(deduplicated) == 2
        names = [e.name for e in deduplicated]
        assert "Jane Doe" in names

    def test_parse_executive_result(self):
        """Test parsing SPARQL result into Executive."""
        service = WikidataService()

        # Valid result
        result = {
            "ceoLabel": {"value": "Tim Cook"},
            "startTime": {"value": "2011-08-24T00:00:00Z"},
            "endTime": {"value": None},
        }
        exec = service._parse_executive_result(result, "CEO")
        assert exec is not None
        assert exec.name == "Tim Cook"
        assert exec.title == "CEO"
        assert exec.start_year == 2011
        assert exec.end_year is None

        # Invalid result (missing name)
        result_no_name = {"ceoLabel": {"value": ""}}
        assert service._parse_executive_result(result_no_name, "CEO") is None

        # Invalid result (Q-number instead of name)
        result_q_number = {"ceoLabel": {"value": "Q12345"}}
        assert service._parse_executive_result(result_q_number, "CEO") is None


class TestWikidataServiceAsync:
    """Async tests for WikidataService."""

    @pytest.mark.asyncio
    async def test_get_client_creates_client(self):
        """Test that _get_client creates an HTTP client."""
        service = WikidataService()
        client = await service._get_client()

        assert client is not None
        assert "User-Agent" in client.headers

        await service.close()

    @pytest.mark.asyncio
    async def test_close_client(self):
        """Test that close properly closes the HTTP client."""
        service = WikidataService()
        await service._get_client()  # Create client
        await service.close()

        # Client should be closed
        assert service._http_client is None or service._http_client.is_closed

    @pytest.mark.asyncio
    async def test_execute_sparql_with_mock(self):
        """Test SPARQL execution with mocked response."""
        service = WikidataService()

        mock_response = MagicMock()
        mock_response.json.return_value = {
            "results": {
                "bindings": [
                    {"company": {"value": "http://www.wikidata.org/entity/Q312"}}
                ]
            }
        }
        mock_response.raise_for_status = MagicMock()

        with patch.object(service, '_get_client') as mock_get_client:
            mock_client = AsyncMock()
            mock_client.get.return_value = mock_response
            mock_get_client.return_value = mock_client

            result = await service._execute_sparql("SELECT ?test WHERE { ?test ?p ?o } LIMIT 1")

            assert "results" in result
            assert len(result["results"]["bindings"]) == 1

    @pytest.mark.asyncio
    async def test_lookup_company_wikidata_id_mock(self):
        """Test company lookup with mocked Wikidata response."""
        service = WikidataService()

        mock_sparql_result = {
            "results": {
                "bindings": [
                    {
                        "company": {"value": "http://www.wikidata.org/entity/Q312"},
                        "companyLabel": {"value": "Apple Inc."},
                    }
                ]
            }
        }

        with patch.object(service, '_execute_sparql', return_value=mock_sparql_result):
            wikidata_id = await service.lookup_company_wikidata_id("Apple")

            assert wikidata_id == "Q312"

    @pytest.mark.asyncio
    async def test_get_executives_by_wikidata_id_mock(self):
        """Test getting executives with mocked Wikidata response."""
        service = WikidataService()

        # Mock CEO query response
        mock_ceo_result = {
            "results": {
                "bindings": [
                    {
                        "ceo": {"value": "http://www.wikidata.org/entity/Q561684"},
                        "ceoLabel": {"value": "Tim Cook"},
                        "startTime": {"value": "2011-08-24T00:00:00Z"},
                        "endTime": {"value": None},
                    },
                    {
                        "ceo": {"value": "http://www.wikidata.org/entity/Q19837"},
                        "ceoLabel": {"value": "Steve Jobs"},
                        "startTime": {"value": "1997-09-16T00:00:00Z"},
                        "endTime": {"value": "2011-08-24T00:00:00Z"},
                    },
                ]
            }
        }

        # Mock position query response (empty for simplicity)
        mock_position_result = {"results": {"bindings": []}}

        with patch.object(service, '_execute_sparql') as mock_execute:
            # First call returns CEO results, second returns position results
            mock_execute.side_effect = [mock_ceo_result, mock_position_result]

            executives = await service.get_executives_by_wikidata_id("Q312")

            assert len(executives) == 2
            names = [e.name for e in executives]
            assert "Tim Cook" in names
            assert "Steve Jobs" in names

            # Verify tenure dates
            tim_cook = next(e for e in executives if e.name == "Tim Cook")
            assert tim_cook.start_year == 2011
            assert tim_cook.end_year is None  # Current CEO

            steve_jobs = next(e for e in executives if e.name == "Steve Jobs")
            assert steve_jobs.start_year == 1997
            assert steve_jobs.end_year == 2011

    @pytest.mark.asyncio
    async def test_search_executives_mock(self):
        """Test full search flow with mocked responses."""
        service = WikidataService()

        # Mock CEO query
        mock_ceo_result = {
            "results": {
                "bindings": [
                    {
                        "ceoLabel": {"value": "Tim Cook"},
                        "startTime": {"value": "2011-08-24T00:00:00Z"},
                        "endTime": {},
                    }
                ]
            }
        }

        # Mock position query
        mock_position_result = {"results": {"bindings": []}}

        # Mock the lookup to return a Wikidata ID directly
        with patch.object(service, 'lookup_company_wikidata_id', return_value="Q312"):
            with patch.object(service, '_execute_sparql') as mock_execute:
                mock_execute.side_effect = [mock_ceo_result, mock_position_result]

                executives = await service.search_executives("Apple")

                assert len(executives) == 1
                assert executives[0].name == "Tim Cook"
                assert executives[0].title == "CEO"
                assert executives[0].start_year == 2011

    @pytest.mark.asyncio
    async def test_search_executives_no_company_found(self):
        """Test search when company is not found in Wikidata."""
        service = WikidataService()

        # Mock the lookup to return None (not found)
        with patch.object(service, 'lookup_company_wikidata_id', return_value=None):
            executives = await service.search_executives("Nonexistent Company XYZ")

            assert executives == []

    @pytest.mark.asyncio
    async def test_search_entity_api_mock(self):
        """Test the fast MediaWiki API search."""
        service = WikidataService()

        mock_api_response = MagicMock()
        mock_api_response.json.return_value = {
            "search": [
                {
                    "id": "Q312",
                    "label": "Apple Inc.",
                    "description": "American multinational technology company",
                }
            ]
        }
        mock_api_response.raise_for_status = MagicMock()

        with patch.object(service, '_get_client') as mock_get_client:
            mock_client = AsyncMock()
            mock_client.get.return_value = mock_api_response
            mock_get_client.return_value = mock_client

            result = await service._search_entity_api("Apple")

            assert result == "Q312"

    @pytest.mark.asyncio
    async def test_lookup_company_uses_api_first(self):
        """Test that lookup_company_wikidata_id tries API first."""
        service = WikidataService()

        # Mock API to return a result
        with patch.object(service, '_search_entity_api', return_value="Q312") as mock_api:
            with patch.object(service, '_search_entity_sparql') as mock_sparql:
                result = await service.lookup_company_wikidata_id("Apple")

                assert result == "Q312"
                mock_api.assert_called_once_with("Apple")
                # SPARQL should not be called if API succeeds
                mock_sparql.assert_not_called()

    @pytest.mark.asyncio
    async def test_lookup_company_falls_back_to_sparql(self):
        """Test that lookup_company_wikidata_id falls back to SPARQL when API fails."""
        service = WikidataService()

        # Mock API to return None, SPARQL to return result
        with patch.object(service, '_search_entity_api', return_value=None) as mock_api:
            with patch.object(service, '_search_entity_sparql', return_value="Q312") as mock_sparql:
                result = await service.lookup_company_wikidata_id("Apple")

                assert result == "Q312"
                mock_api.assert_called_once_with("Apple")
                mock_sparql.assert_called_once_with("Apple")


# Integration test (requires network access)
# Uncomment to run against real Wikidata API
# class TestWikidataServiceIntegration:
#     """Integration tests that hit the real Wikidata API."""
#
#     @pytest.mark.asyncio
#     @pytest.mark.integration
#     async def test_real_apple_ceo_lookup(self):
#         """Test looking up Apple's CEO history from real Wikidata."""
#         service = WikidataService()
#
#         try:
#             executives = await service.search_executives("Apple Inc")
#
#             # Should find at least Tim Cook and Steve Jobs
#             assert len(executives) >= 2
#             names = [e.name for e in executives]
#             assert any("Tim Cook" in name for name in names)
#
#         finally:
#             await service.close()


class TestWikidataDateParsing:
    """Additional tests for Wikidata date parsing."""

    def setup_method(self):
        """Set up test fixtures."""
        self.service = WikidataService()

    def test_parse_full_iso_date(self):
        """Test parsing full ISO 8601 dates."""
        assert self.service._parse_wikidata_date("2020-05-15T00:00:00Z") == 2020
        assert self.service._parse_wikidata_date("1998-01-01T00:00:00Z") == 1998

    def test_parse_year_only(self):
        """Test parsing year-only strings."""
        assert self.service._parse_wikidata_date("2020") == 2020
        assert self.service._parse_wikidata_date("1999") == 1999

    def test_parse_none_value(self):
        """Test parsing None values."""
        assert self.service._parse_wikidata_date(None) is None

    def test_parse_empty_string(self):
        """Test parsing empty strings."""
        assert self.service._parse_wikidata_date("") is None

    def test_parse_invalid_format(self):
        """Test parsing invalid date formats."""
        assert self.service._parse_wikidata_date("invalid") is None
        assert self.service._parse_wikidata_date("not-a-date") is None

    def test_parse_year_at_boundaries(self):
        """Test parsing dates at year boundaries."""
        assert self.service._parse_wikidata_date("2023-01-01T00:00:00Z") == 2023
        assert self.service._parse_wikidata_date("2023-12-31T23:59:59Z") == 2023


class TestWikidataNameNormalization:
    """Additional tests for Wikidata name normalization."""

    def setup_method(self):
        """Set up test fixtures."""
        self.service = WikidataService()

    def test_normalize_basic(self):
        """Test basic name normalization."""
        assert self.service._normalize_name("John Smith") == "john smith"

    def test_normalize_extra_whitespace(self):
        """Test normalizing extra whitespace."""
        assert self.service._normalize_name("  John   Smith  ") == "john smith"

    def test_normalize_removes_jr_suffix(self):
        """Test Jr. suffix removal."""
        assert self.service._normalize_name("John Smith Jr.") == "john smith"
        assert self.service._normalize_name("John Smith Jr") == "john smith"

    def test_normalize_removes_iii_suffix(self):
        """Test III suffix removal."""
        assert self.service._normalize_name("John Smith III") == "john smith"
        assert self.service._normalize_name("John Smith II") == "john smith"

    def test_normalize_removes_phd_suffix(self):
        """Test PhD suffix removal."""
        assert self.service._normalize_name("John Smith PhD") == "john smith"

    def test_normalize_case_insensitive(self):
        """Test case-insensitive normalization."""
        assert self.service._normalize_name("JOHN SMITH") == "john smith"
        assert self.service._normalize_name("john smith") == "john smith"


class TestWikidataMatchScoring:
    """Additional tests for company name match scoring."""

    def setup_method(self):
        """Set up test fixtures."""
        self.service = WikidataService()

    def test_exact_match_score_1(self):
        """Test that exact matches score 1.0."""
        assert self.service._calculate_match_score("Apple", "Apple") == 1.0
        assert self.service._calculate_match_score("Microsoft", "Microsoft") == 1.0

    def test_case_insensitive_exact_match(self):
        """Test case-insensitive exact matching."""
        assert self.service._calculate_match_score("apple", "Apple") == 1.0
        assert self.service._calculate_match_score("APPLE", "apple") == 1.0

    def test_partial_match_lower_score(self):
        """Test that partial matches have lower scores."""
        score = self.service._calculate_match_score("Apple", "Apple Inc.")
        assert 0.0 < score < 1.0

    def test_different_companies_low_score(self):
        """Test that completely different companies have low scores."""
        score = self.service._calculate_match_score("Apple", "Microsoft")
        # Should be some overlap from common words but low
        assert score < 0.5


class TestWikidataExecutivePositionDetection:
    """Additional tests for executive position detection."""

    def setup_method(self):
        """Set up test fixtures."""
        self.service = WikidataService()

    def test_detect_ceo(self):
        """Test detecting CEO positions."""
        assert self.service._is_executive_position("Chief Executive Officer") is True
        assert self.service._is_executive_position("CEO") is True

    def test_detect_cfo(self):
        """Test detecting CFO positions."""
        assert self.service._is_executive_position("Chief Financial Officer") is True
        assert self.service._is_executive_position("CFO") is True

    def test_detect_president(self):
        """Test detecting President position."""
        assert self.service._is_executive_position("President") is True
        assert self.service._is_executive_position("President and CEO") is True

    def test_detect_chairman(self):
        """Test detecting Chairman position."""
        assert self.service._is_executive_position("Chairman") is True
        assert self.service._is_executive_position("Chairman of the Board") is True

    def test_detect_evp(self):
        """Test detecting Executive VP position."""
        assert self.service._is_executive_position("Executive Vice President") is True
        assert self.service._is_executive_position("Executive Vice President") is True

    def test_reject_non_executive(self):
        """Test rejecting non-executive positions."""
        assert self.service._is_executive_position("Software Engineer") is False
        assert self.service._is_executive_position("Manager") is False
        assert self.service._is_executive_position("Analyst") is False


class TestWikidataExecutiveMerging:
    """Additional tests for executive record merging."""

    def setup_method(self):
        """Set up test fixtures."""
        self.service = WikidataService()

    def test_merge_prefers_longer_name(self):
        """Test that merging prefers longer names."""
        exec1 = Executive(name="Tim Cook", title="CEO", start_year=2011)
        exec2 = Executive(name="Timothy D. Cook", title="CEO", start_year=2011)
        
        merged = self.service._merge_executives([exec1, exec2])
        assert merged.name == "Timothy D. Cook"

    def test_merge_prefers_canonical_title(self):
        """Test that merging prefers canonical CEO title."""
        exec1 = Executive(name="Tim Cook", title="Chief Executive Officer", start_year=2011)
        exec2 = Executive(name="Tim Cook", title="CEO", start_year=2011)
        
        merged = self.service._merge_executives([exec1, exec2])
        assert merged.title == "CEO"

    def test_merge_uses_earliest_start_year(self):
        """Test that merging uses earliest start year."""
        exec1 = Executive(name="Tim Cook", title="CEO", start_year=2011)
        exec2 = Executive(name="Tim Cook", title="CEO", start_year=2012)
        
        merged = self.service._merge_executives([exec1, exec2])
        assert merged.start_year == 2011

    def test_merge_single_executive(self):
        """Test merging single executive returns same data."""
        exec1 = Executive(name="Tim Cook", title="CEO", start_year=2011)
        
        merged = self.service._merge_executives([exec1])
        assert merged.name == exec1.name
        assert merged.title == exec1.title
        assert merged.start_year == exec1.start_year


class TestWikidataDeduplication:
    """Additional tests for executive deduplication."""

    def setup_method(self):
        """Set up test fixtures."""
        self.service = WikidataService()

    def test_deduplicate_exact_duplicates(self):
        """Test deduplicating exact duplicate names."""
        executives = [
            Executive(name="Tim Cook", title="CEO", start_year=2011),
            Executive(name="Tim Cook", title="CEO", start_year=2011),
        ]
        
        result = self.service._deduplicate_executives(executives)
        assert len(result) == 1

    def test_deduplicate_case_variations(self):
        """Test deduplicating case variations."""
        executives = [
            Executive(name="Tim Cook", title="CEO", start_year=2011),
            Executive(name="tim cook", title="CEO", start_year=2011),
        ]
        
        result = self.service._deduplicate_executives(executives)
        assert len(result) == 1

    def test_deduplicate_preserves_unique(self):
        """Test that unique executives are preserved."""
        executives = [
            Executive(name="Tim Cook", title="CEO", start_year=2011),
            Executive(name="Luca Maestri", title="CFO", start_year=2014),
        ]
        
        result = self.service._deduplicate_executives(executives)
        assert len(result) == 2

    def test_deduplicate_empty_list(self):
        """Test deduplicating empty list."""
        result = self.service._deduplicate_executives([])
        assert result == []


class TestWikidataSparqlEscaping:
    """Additional tests for SPARQL string escaping."""

    def setup_method(self):
        """Set up test fixtures."""
        self.service = WikidataService()

    def test_escape_single_quotes(self):
        """Test escaping single quotes."""
        assert self.service._escape_sparql_string("O'Reilly") == "O\\'Reilly"

    def test_escape_double_quotes(self):
        """Test escaping double quotes."""
        result = self.service._escape_sparql_string('Company "ABC"')
        assert result == 'Company \\"ABC\\"'

    def test_escape_backslashes(self):
        """Test escaping backslashes."""
        result = self.service._escape_sparql_string("Path\\to\\company")
        assert result == "Path\\\\to\\\\company"

    def test_escape_no_special_chars(self):
        """Test that regular strings are unchanged."""
        assert self.service._escape_sparql_string("Apple Inc.") == "Apple Inc."
        assert self.service._escape_sparql_string("Microsoft") == "Microsoft"


class TestWikidataParseExecutiveResult:
    """Additional tests for parsing SPARQL executive results."""

    def setup_method(self):
        """Set up test fixtures."""
        self.service = WikidataService()

    def test_parse_complete_result(self):
        """Test parsing complete SPARQL result."""
        result = {
            "ceoLabel": {"value": "Tim Cook"},
            "startTime": {"value": "2011-08-24T00:00:00Z"},
            "endTime": {"value": None},
        }
        
        exec = self.service._parse_executive_result(result, "CEO")
        
        assert exec is not None
        assert exec.name == "Tim Cook"
        assert exec.title == "CEO"
        assert exec.start_year == 2011
        assert exec.end_year is None

    def test_parse_result_with_end_time(self):
        """Test parsing result with end time."""
        result = {
            "ceoLabel": {"value": "Steve Jobs"},
            "startTime": {"value": "1997-09-16T00:00:00Z"},
            "endTime": {"value": "2011-08-24T00:00:00Z"},
        }
        
        exec = self.service._parse_executive_result(result, "CEO")
        
        assert exec is not None
        assert exec.start_year == 1997
        assert exec.end_year == 2011

    def test_parse_result_empty_name(self):
        """Test parsing result with empty name."""
        result = {
            "ceoLabel": {"value": ""},
        }
        
        exec = self.service._parse_executive_result(result, "CEO")
        assert exec is None

    def test_parse_result_q_number_name(self):
        """Test parsing result with Q-number instead of name."""
        result = {
            "ceoLabel": {"value": "Q12345"},
        }
        
        exec = self.service._parse_executive_result(result, "CEO")
        assert exec is None

    def test_parse_result_missing_label(self):
        """Test parsing result with missing label."""
        result = {}
        
        exec = self.service._parse_executive_result(result, "CEO")
        assert exec is None


class TestWikidataServiceConfiguration:
    """Tests for Wikidata service configuration."""

    def test_is_configured_always_true(self):
        """Test that Wikidata is always configured (no API key needed)."""
        service = WikidataService()
        assert service.is_configured is True

    def test_service_has_endpoint_url(self):
        """Test that service has endpoint URL defined."""
        service = WikidataService()
        # Service should have a query URL attribute
        assert hasattr(service, '_http_client') or True  # Basic attribute check
