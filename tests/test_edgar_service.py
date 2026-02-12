"""Tests for SEC EDGAR Integration Service."""

import pytest
from unittest.mock import AsyncMock, MagicMock, patch

from app.services.edgar_service import (
    EdgarService,
    RateLimiter,
    get_edgar_service,
    SEC_USER_AGENT,
)
from app.models import Executive


class TestRateLimiter:
    """Test the rate limiter functionality."""

    @pytest.mark.asyncio
    async def test_rate_limiter_allows_first_request(self) -> None:
        """First request should be allowed immediately."""
        limiter = RateLimiter(max_requests_per_second=10)
        # Should not block
        await limiter.acquire()

    @pytest.mark.asyncio
    async def test_rate_limiter_spacing(self) -> None:
        """Rate limiter should space out rapid requests."""
        import time

        limiter = RateLimiter(max_requests_per_second=10)

        start = time.monotonic()
        await limiter.acquire()
        await limiter.acquire()
        elapsed = time.monotonic() - start

        # Second request should wait at least 0.1 seconds (100ms)
        assert elapsed >= 0.09  # Allow small margin for timing


class TestEdgarService:
    """Test the EdgarService class."""

    def test_singleton_pattern(self) -> None:
        """Test that get_edgar_service returns singleton."""
        service1 = get_edgar_service()
        service2 = get_edgar_service()
        assert service1 is service2

    def test_normalize_company_name(self) -> None:
        """Test company name normalization."""
        service = EdgarService()

        # Test suffix removal
        assert service._normalize_company_name("Apple Inc.") == "apple"
        assert service._normalize_company_name("Microsoft Corporation") == "microsoft"
        assert service._normalize_company_name("Tesla, Inc") == "tesla"
        assert service._normalize_company_name("Amazon.com, Inc.") == "amazon.com"

        # Test case insensitivity
        assert service._normalize_company_name("APPLE INC.") == "apple"

        # Test whitespace normalization
        assert service._normalize_company_name("  Apple   Inc.  ") == "apple"

    def test_is_valid_executive_name(self) -> None:
        """Test executive name validation."""
        service = EdgarService()

        # Valid names
        assert service._is_valid_executive_name("John Smith") is True
        assert service._is_valid_executive_name("Mary Jane Watson") is True
        assert service._is_valid_executive_name("Tim Cook") is True

        # Invalid names - titles
        assert service._is_valid_executive_name("CEO") is False
        assert service._is_valid_executive_name("President") is False
        assert service._is_valid_executive_name("Chief Executive Officer") is False

        # Invalid names - too short
        assert service._is_valid_executive_name("A") is False
        assert service._is_valid_executive_name("AB") is False

        # Invalid names - single word
        assert service._is_valid_executive_name("John") is False

        # Invalid names - placeholders
        assert service._is_valid_executive_name("N/A") is False
        assert service._is_valid_executive_name("TBD") is False
        assert service._is_valid_executive_name("Unknown") is False

    @pytest.mark.asyncio
    async def test_lookup_company_cik_caching(self) -> None:
        """Test that CIK lookups are cached."""
        service = EdgarService()

        # Mock the company tickers data
        mock_tickers = {
            "0": {"cik_str": "320193", "title": "APPLE INC"},
            "1": {"cik_str": "789019", "title": "MICROSOFT CORP"},
        }

        with patch.object(service, "_load_company_tickers", return_value=mock_tickers):
            # First lookup
            cik1 = await service.lookup_company_cik("Apple Inc.")
            assert cik1 == "0000320193"

            # Second lookup should use cache (won't call _load_company_tickers again)
            cik2 = await service.lookup_company_cik("Apple Inc.")
            assert cik2 == "0000320193"

            # Both should be the same
            assert cik1 == cik2

    @pytest.mark.asyncio
    async def test_lookup_company_cik_not_found(self) -> None:
        """Test CIK lookup for non-existent company."""
        service = EdgarService()

        mock_tickers = {
            "0": {"cik_str": "320193", "title": "APPLE INC"},
        }

        with patch.object(service, "_load_company_tickers", return_value=mock_tickers):
            cik = await service.lookup_company_cik("NonExistent Company XYZ")
            assert cik is None

    @pytest.mark.asyncio
    async def test_lookup_company_cik_partial_match(self) -> None:
        """Test CIK lookup with partial match."""
        service = EdgarService()

        mock_tickers = {
            "0": {"cik_str": "320193", "title": "APPLE INC"},
            "1": {"cik_str": "12345", "title": "APPLE HOSPITALITY REIT INC"},
        }

        with patch.object(service, "_load_company_tickers", return_value=mock_tickers):
            # Should match "Apple Inc" exactly first
            cik = await service.lookup_company_cik("Apple")
            # Should find the shorter match (Apple Inc)
            assert cik == "0000320193"

    def test_clear_cache(self) -> None:
        """Test cache clearing."""
        service = EdgarService()

        # Add some data to caches
        service._cik_cache["test"] = "123"
        service._submissions_cache["123"] = {"name": "Test"}
        service._tickers_data = {"0": {"title": "Test"}}

        # Clear caches
        service.clear_cache()

        # Verify caches are empty
        assert len(service._cik_cache) == 0
        assert len(service._submissions_cache) == 0
        assert service._tickers_data is None

    @pytest.mark.asyncio
    async def test_get_company_officers_with_mock(self) -> None:
        """Test getting company officers with mocked data."""
        service = EdgarService()

        mock_submissions = {
            "name": "APPLE INC",
            "cik": "0000320193",
            "officers": [
                {"name": "Timothy Cook", "title": "Chief Executive Officer"},
                {"name": "Luca Maestri", "title": "Chief Financial Officer"},
            ],
            "filings": {"recent": {}},
        }

        with patch.object(
            service, "_get_company_submissions", return_value=mock_submissions
        ):
            executives = await service.get_company_officers("0000320193")

            assert len(executives) == 2
            assert executives[0].name == "Timothy Cook"
            assert executives[0].title == "Chief Executive Officer"
            assert executives[1].name == "Luca Maestri"

    @pytest.mark.asyncio
    async def test_search_executives_not_found(self) -> None:
        """Test search for non-existent company."""
        service = EdgarService()

        with patch.object(service, "lookup_company_cik", return_value=None):
            executives = await service.search_executives("NonExistent Company XYZ")
            assert executives == []

    @pytest.mark.asyncio
    async def test_get_company_info(self) -> None:
        """Test getting company info."""
        service = EdgarService()

        mock_submissions = {
            "name": "APPLE INC",
            "cik": "0000320193",
            "sic": "3571",
            "sicDescription": "Electronic Computers",
            "stateOfIncorporation": "CA",
            "fiscalYearEnd": "0930",
            "formerNames": [{"name": "APPLE COMPUTER INC"}],
            "website": "https://www.apple.com",
            "exchanges": ["NASDAQ"],
            "tickers": ["AAPL"],
        }

        with patch.object(service, "lookup_company_cik", return_value="0000320193"):
            with patch.object(
                service, "_get_company_submissions", return_value=mock_submissions
            ):
                info = await service.get_company_info("Apple Inc.")

                assert info is not None
                assert info["cik"] == "0000320193"
                assert info["name"] == "APPLE INC"
                assert info["sic"] == "3571"
                assert info["state"] == "CA"
                assert "APPLE COMPUTER INC" in info["former_names"]

    @pytest.mark.asyncio
    async def test_close_client(self) -> None:
        """Test closing the HTTP client."""
        service = EdgarService()

        # Create a mock client
        mock_client = AsyncMock()
        mock_client.is_closed = False
        service._http_client = mock_client

        # Close the service
        await service.close()

        # Verify client was closed
        mock_client.aclose.assert_called_once()
        assert service._http_client is None


class TestEdgarServiceIntegration:
    """Integration tests for SEC EDGAR service.

    These tests make real API calls and should be run sparingly
    to avoid hitting rate limits.
    """

    @pytest.mark.asyncio
    @pytest.mark.skip(reason="Integration test - run manually")
    async def test_lookup_apple_cik(self) -> None:
        """Test looking up Apple's CIK from SEC EDGAR."""
        service = EdgarService()
        try:
            cik = await service.lookup_company_cik("Apple Inc.")
            assert cik == "0000320193"
        finally:
            await service.close()

    @pytest.mark.asyncio
    @pytest.mark.skip(reason="Integration test - run manually")
    async def test_get_apple_info(self) -> None:
        """Test getting Apple's company info from SEC EDGAR."""
        service = EdgarService()
        try:
            info = await service.get_company_info("Apple Inc.")
            assert info is not None
            assert info["name"] == "Apple Inc."
            assert "AAPL" in info.get("tickers", [])
        finally:
            await service.close()



class TestCompanyNameNormalization:
    """Additional tests for company name normalization edge cases."""

    def setup_method(self):
        """Set up test fixtures."""
        self.service = EdgarService()

    def test_normalize_inc_suffix(self):
        """Test Inc suffix removal."""
        assert self.service._normalize_company_name("Apple Inc") == "apple"
        assert self.service._normalize_company_name("Apple Inc.") == "apple"
        assert self.service._normalize_company_name("Apple, Inc.") == "apple"

    def test_normalize_corp_suffix(self):
        """Test Corp suffix removal."""
        assert self.service._normalize_company_name("Microsoft Corp") == "microsoft"
        assert self.service._normalize_company_name("Microsoft Corp.") == "microsoft"
        assert self.service._normalize_company_name("Microsoft Corporation") == "microsoft"

    def test_normalize_llc_suffix(self):
        """Test LLC suffix removal."""
        assert self.service._normalize_company_name("Acme LLC") == "acme"
        assert self.service._normalize_company_name("Acme L.L.C.") == "acme"
        assert self.service._normalize_company_name("Acme, LLC") == "acme"

    def test_normalize_ltd_suffix(self):
        """Test Ltd suffix removal."""
        assert self.service._normalize_company_name("Acme Ltd") == "acme"
        assert self.service._normalize_company_name("Acme Ltd.") == "acme"
        assert self.service._normalize_company_name("Acme Limited") == "acme"

    def test_normalize_holdings_suffix(self):
        """Test Holdings suffix removal."""
        assert self.service._normalize_company_name("Berkshire Holdings") == "berkshire"
        assert self.service._normalize_company_name("Acme Holding") == "acme"

    def test_normalize_group_suffix(self):
        """Test Group suffix removal."""
        assert self.service._normalize_company_name("ABC Group") == "abc"

    def test_normalize_plc_suffix(self):
        """Test PLC suffix removal."""
        assert self.service._normalize_company_name("Barclays PLC") == "barclays"
        assert self.service._normalize_company_name("Barclays plc") == "barclays"

    def test_normalize_preserves_domain(self):
        """Test that domain names are preserved."""
        # Amazon.com should keep the .com as it's not a suffix
        result = self.service._normalize_company_name("Amazon.com, Inc.")
        assert "amazon.com" in result

    def test_normalize_multiple_suffixes(self):
        """Test names with multiple suffix-like parts."""
        result = self.service._normalize_company_name("XYZ Corp Holdings Inc")
        # Should remove Inc but keep Corp Holdings as part of the name
        assert "xyz" in result

    def test_normalize_extra_whitespace(self):
        """Test whitespace normalization."""
        assert self.service._normalize_company_name("  Apple   Inc.  ") == "apple"
        assert self.service._normalize_company_name("New  York  Times") == "new york times"


class TestValidExecutiveName:
    """Additional tests for executive name validation."""

    def setup_method(self):
        """Set up test fixtures."""
        self.service = EdgarService()

    def test_valid_typical_names(self):
        """Test typical valid names."""
        assert self.service._is_valid_executive_name("John Smith") is True
        assert self.service._is_valid_executive_name("Mary Jane Watson") is True
        assert self.service._is_valid_executive_name("Tim Cook") is True
        assert self.service._is_valid_executive_name("Satya Nadella") is True

    def test_reject_c_suite_titles(self):
        """Test C-suite titles are rejected."""
        assert self.service._is_valid_executive_name("CEO") is False
        assert self.service._is_valid_executive_name("CFO") is False
        assert self.service._is_valid_executive_name("COO") is False
        assert self.service._is_valid_executive_name("CTO") is False
        assert self.service._is_valid_executive_name("CMO") is False
        assert self.service._is_valid_executive_name("CIO") is False

    def test_reject_generic_titles(self):
        """Test generic titles are rejected."""
        assert self.service._is_valid_executive_name("Chief") is False
        assert self.service._is_valid_executive_name("President") is False
        assert self.service._is_valid_executive_name("Officer") is False
        assert self.service._is_valid_executive_name("Director") is False
        assert self.service._is_valid_executive_name("Executive") is False

    def test_reject_title_phrases(self):
        """Test title-containing phrases are rejected."""
        assert self.service._is_valid_executive_name("Chief Executive Officer") is False
        assert self.service._is_valid_executive_name("Vice President Sales") is False
        assert self.service._is_valid_executive_name("Head of Engineering") is False
        assert self.service._is_valid_executive_name("Director of Finance") is False

    def test_reject_placeholders(self):
        """Test placeholder values are rejected."""
        assert self.service._is_valid_executive_name("N/A") is False
        assert self.service._is_valid_executive_name("None") is False
        assert self.service._is_valid_executive_name("Vacant") is False
        assert self.service._is_valid_executive_name("TBD") is False
        assert self.service._is_valid_executive_name("Unknown") is False

    def test_reject_too_short(self):
        """Test names that are too short."""
        assert self.service._is_valid_executive_name("Jo") is False
        assert self.service._is_valid_executive_name("A") is False
        assert self.service._is_valid_executive_name("") is False

    def test_reject_too_long(self):
        """Test names that are too long."""
        long_name = "A" * 101
        assert self.service._is_valid_executive_name(long_name) is False

    def test_reject_single_word(self):
        """Test single-word names are rejected."""
        assert self.service._is_valid_executive_name("John") is False
        assert self.service._is_valid_executive_name("Smith") is False

    def test_accept_two_word_names(self):
        """Test two-word names are accepted."""
        assert self.service._is_valid_executive_name("John Smith") is True
        assert self.service._is_valid_executive_name("Jane Doe") is True

    def test_accept_three_word_names(self):
        """Test three-word names are accepted."""
        assert self.service._is_valid_executive_name("John Andrew Smith") is True
        assert self.service._is_valid_executive_name("Mary Jane Watson") is True


class TestRateLimiterEdgeCases:
    """Additional tests for rate limiter edge cases."""

    @pytest.mark.asyncio
    async def test_rate_limiter_custom_rate(self) -> None:
        """Test rate limiter with custom rate."""
        # 5 requests per second = 0.2 second minimum interval
        limiter = RateLimiter(max_requests_per_second=5)
        
        import time
        start = time.monotonic()
        await limiter.acquire()
        await limiter.acquire()
        elapsed = time.monotonic() - start
        
        # Should wait at least 0.2 seconds
        assert elapsed >= 0.18  # Allow small margin

    @pytest.mark.asyncio
    async def test_rate_limiter_high_rate(self) -> None:
        """Test rate limiter with very high rate (effectively no limit)."""
        limiter = RateLimiter(max_requests_per_second=1000)
        
        import time
        start = time.monotonic()
        for _ in range(5):
            await limiter.acquire()
        elapsed = time.monotonic() - start
        
        # With 1000 req/s, 5 requests should be very fast
        assert elapsed < 0.1


class TestEdgarServiceCaching:
    """Tests for caching behavior."""

    def test_cik_cache_initially_empty(self):
        """Test that CIK cache is initially empty."""
        service = EdgarService()
        assert len(service._cik_cache) == 0

    def test_submissions_cache_initially_empty(self):
        """Test that submissions cache is initially empty."""
        service = EdgarService()
        assert len(service._submissions_cache) == 0

    def test_tickers_data_initially_none(self):
        """Test that tickers data is initially None."""
        service = EdgarService()
        assert service._tickers_data is None

    def test_clear_cache_clears_all(self):
        """Test that clear_cache clears all caches."""
        service = EdgarService()
        
        # Add some data to caches
        service._cik_cache["test"] = "123"
        service._submissions_cache["123"] = {"name": "Test"}
        service._tickers_data = {"0": {"title": "Test"}}
        
        # Clear caches
        service.clear_cache()
        
        # Verify all are cleared
        assert len(service._cik_cache) == 0
        assert len(service._submissions_cache) == 0
        assert service._tickers_data is None


class TestEdgarServiceHttpClient:
    """Tests for HTTP client management."""

    def test_client_initially_none(self):
        """Test that HTTP client is initially None."""
        service = EdgarService()
        assert service._http_client is None

    @pytest.mark.asyncio
    async def test_get_client_creates_client(self):
        """Test that _get_client creates a new client."""
        import httpx
        service = EdgarService()
        client = await service._get_client()
        
        assert client is not None
        assert isinstance(client, httpx.AsyncClient)
        
        # Clean up
        await service.close()

    @pytest.mark.asyncio
    async def test_get_client_has_user_agent(self):
        """Test that client has correct User-Agent header."""
        service = EdgarService()
        client = await service._get_client()
        
        assert "User-Agent" in client.headers
        assert "CompanyMapping" in client.headers["User-Agent"]
        
        await service.close()

    @pytest.mark.asyncio
    async def test_get_client_reuses_client(self):
        """Test that _get_client returns same client on subsequent calls."""
        service = EdgarService()
        
        client1 = await service._get_client()
        client2 = await service._get_client()
        
        assert client1 is client2
        
        await service.close()


class TestParseOfficersFromSubmissions:
    """Tests for parsing officers from submissions data."""

    def test_parse_empty_submissions(self):
        """Test parsing empty submissions data."""
        service = EdgarService()
        
        result = service._parse_officers_from_submissions({})
        assert result == []

    def test_parse_submissions_with_officers(self):
        """Test parsing submissions with officers data."""
        service = EdgarService()
        
        data = {
            "name": "Test Company",
            "officers": [
                {"name": "John Smith", "title": "CEO"},
                {"name": "Jane Doe", "title": "CFO"},
            ]
        }
        
        executives = service._parse_officers_from_submissions(data)
        
        assert len(executives) == 2
        assert executives[0].name == "John Smith"
        assert executives[0].title == "CEO"

    def test_parse_submissions_filters_invalid_names(self):
        """Test that invalid names are filtered out."""
        service = EdgarService()
        
        data = {
            "name": "Test Company",
            "officers": [
                {"name": "John Smith", "title": "CEO"},
                {"name": "CEO", "title": "Chief Executive Officer"},  # Invalid
                {"name": "", "title": "CFO"},  # Invalid
            ]
        }
        
        executives = service._parse_officers_from_submissions(data)
        
        # Only John Smith should be valid
        assert len(executives) == 1
        assert executives[0].name == "John Smith"

    def test_parse_submissions_no_officers_field(self):
        """Test parsing submissions without officers field."""
        service = EdgarService()
        
        data = {
            "name": "Test Company",
            "cik": "123456",
        }
        
        executives = service._parse_officers_from_submissions(data)
        assert executives == []
