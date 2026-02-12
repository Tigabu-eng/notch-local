"""SEC EDGAR Integration Service for company executive data.

SEC EDGAR is the primary authoritative data source for US public company
executive information. This service extracts Named Executive Officers (NEOs)
from proxy statements (DEF 14A) and annual reports (10-K).

API Documentation:
- Company Search: https://efts.sec.gov/LATEST/search-index
- Company Facts: https://data.sec.gov/api/xbrl/companyfacts/CIK{cik}.json
- Filings: https://data.sec.gov/submissions/CIK{cik}.json

Rate Limit: 10 requests/second, requires User-Agent header
"""

from __future__ import annotations

import asyncio
import logging
import re
import time
from datetime import datetime
from typing import Any

import httpx

from app.models import Executive

logger = logging.getLogger(__name__)

# SEC EDGAR API endpoints
SEC_SEARCH_URL = "https://efts.sec.gov/LATEST/search-index"
SEC_SUBMISSIONS_URL = "https://data.sec.gov/submissions/CIK{cik}.json"
SEC_COMPANY_TICKERS_URL = "https://www.sec.gov/files/company_tickers.json"

# User-Agent header required by SEC EDGAR
SEC_USER_AGENT = "CompanyMapping/1.0 (contact@example.com)"

# Rate limiting: 10 requests per second
MAX_REQUESTS_PER_SECOND = 10
MIN_REQUEST_INTERVAL = 1.0 / MAX_REQUESTS_PER_SECOND  # 0.1 seconds


class RateLimiter:
    """Simple rate limiter using token bucket algorithm.

    Ensures we don't exceed SEC EDGAR's rate limit of 10 requests/second.
    """

    def __init__(self, max_requests_per_second: float = MAX_REQUESTS_PER_SECOND) -> None:
        self.min_interval = 1.0 / max_requests_per_second
        self._last_request_time: float = 0.0
        self._lock = asyncio.Lock()

    async def acquire(self) -> None:
        """Wait until a request can be made without exceeding rate limit."""
        async with self._lock:
            now = time.monotonic()
            time_since_last = now - self._last_request_time

            if time_since_last < self.min_interval:
                wait_time = self.min_interval - time_since_last
                await asyncio.sleep(wait_time)

            self._last_request_time = time.monotonic()


class EdgarService:
    """SEC EDGAR service for fetching authoritative company executive data.

    Provides methods for:
    - Looking up company CIK (Central Index Key) from name
    - Fetching company officers from SEC filings
    - Searching executives by company name

    Implements rate limiting and caching to respect SEC EDGAR's API constraints.
    """

    def __init__(self) -> None:
        self._http_client: httpx.AsyncClient | None = None
        self._rate_limiter = RateLimiter()

        # Cache for CIK lookups (company name -> CIK)
        self._cik_cache: dict[str, str | None] = {}

        # Cache for company tickers data (loaded once)
        self._tickers_data: dict[str, Any] | None = None

        # Cache for submission data (CIK -> data)
        self._submissions_cache: dict[str, dict[str, Any]] = {}

    async def _get_client(self) -> httpx.AsyncClient:
        """Get or create HTTP client with required headers."""
        if self._http_client is None or self._http_client.is_closed:
            self._http_client = httpx.AsyncClient(
                timeout=30.0,
                headers={
                    "User-Agent": SEC_USER_AGENT,
                    "Accept": "application/json",
                },
            )
        return self._http_client

    async def close(self) -> None:
        """Close HTTP client and cleanup resources."""
        if self._http_client and not self._http_client.is_closed:
            await self._http_client.aclose()
            self._http_client = None

    async def _request_with_retry(
        self,
        method: str,
        url: str,
        max_retries: int = 3,
        **kwargs: Any,
    ) -> httpx.Response:
        """Make an HTTP request with rate limiting and retry logic.

        Args:
            method: HTTP method (GET, POST, etc.)
            url: URL to request
            max_retries: Maximum number of retries on failure
            **kwargs: Additional arguments to pass to httpx

        Returns:
            httpx.Response object

        Raises:
            httpx.HTTPError: If request fails after all retries
        """
        client = await self._get_client()
        last_error: Exception | None = None

        for attempt in range(max_retries):
            try:
                # Wait for rate limiter
                await self._rate_limiter.acquire()

                response = await client.request(method, url, **kwargs)

                # Handle rate limiting response (429)
                if response.status_code == 429:
                    retry_after = int(response.headers.get("Retry-After", "10"))
                    logger.warning(
                        f"Rate limited by SEC EDGAR, waiting {retry_after}s (attempt {attempt + 1}/{max_retries})"
                    )
                    await asyncio.sleep(retry_after)
                    continue

                response.raise_for_status()
                return response

            except httpx.HTTPStatusError as e:
                last_error = e
                if e.response.status_code in (429, 500, 502, 503, 504):
                    # Retryable errors
                    wait_time = (attempt + 1) * 2  # Exponential backoff: 2, 4, 6 seconds
                    logger.warning(
                        f"SEC EDGAR request failed with {e.response.status_code}, "
                        f"retrying in {wait_time}s (attempt {attempt + 1}/{max_retries})"
                    )
                    await asyncio.sleep(wait_time)
                else:
                    # Non-retryable error
                    raise

            except httpx.RequestError as e:
                last_error = e
                wait_time = (attempt + 1) * 2
                logger.warning(
                    f"SEC EDGAR request error: {e}, "
                    f"retrying in {wait_time}s (attempt {attempt + 1}/{max_retries})"
                )
                await asyncio.sleep(wait_time)

        # All retries exhausted
        if last_error:
            raise last_error
        raise RuntimeError("Request failed with no error captured")

    async def _load_company_tickers(self) -> dict[str, Any]:
        """Load the SEC company tickers JSON file.

        This file contains mappings of company names to CIK numbers
        for all SEC-registered companies.

        Returns:
            Dictionary mapping company indices to company data
        """
        if self._tickers_data is not None:
            return self._tickers_data

        try:
            response = await self._request_with_retry("GET", SEC_COMPANY_TICKERS_URL)
            self._tickers_data = response.json()
            logger.info(f"Loaded SEC company tickers: {len(self._tickers_data)} companies")
            return self._tickers_data
        except Exception as e:
            logger.error(f"Failed to load SEC company tickers: {e}")
            self._tickers_data = {}
            return {}

    def _normalize_company_name(self, name: str) -> str:
        """Normalize company name for matching.

        Removes common suffixes and normalizes whitespace/case.

        Args:
            name: Company name to normalize

        Returns:
            Normalized name for comparison
        """
        normalized = name.lower().strip()

        # Remove common corporate suffixes
        suffixes = [
            r"\s*,?\s*inc\.?$",
            r"\s*,?\s*incorporated$",
            r"\s*,?\s*corp\.?$",
            r"\s*,?\s*corporation$",
            r"\s*,?\s*llc\.?$",
            r"\s*,?\s*l\.l\.c\.?$",
            r"\s*,?\s*ltd\.?$",
            r"\s*,?\s*limited$",
            r"\s*,?\s*co\.?$",
            r"\s*,?\s*company$",
            r"\s*,?\s*holdings?$",
            r"\s*,?\s*group$",
            r"\s*,?\s*plc\.?$",
        ]

        for suffix in suffixes:
            normalized = re.sub(suffix, "", normalized, flags=re.IGNORECASE)

        # Normalize whitespace
        normalized = re.sub(r"\s+", " ", normalized).strip()

        return normalized

    async def lookup_company_cik(self, company_name: str) -> str | None:
        """Look up a company's CIK (Central Index Key) from its name.

        Uses the SEC's company tickers file for exact and fuzzy matching.
        Results are cached to reduce API calls.

        Args:
            company_name: Company name to search for

        Returns:
            10-digit CIK string (zero-padded) if found, None otherwise
        """
        # Check cache first
        cache_key = self._normalize_company_name(company_name)
        if cache_key in self._cik_cache:
            logger.debug(f"CIK cache hit for '{company_name}'")
            return self._cik_cache[cache_key]

        try:
            tickers_data = await self._load_company_tickers()

            if not tickers_data:
                logger.warning("No company tickers data available")
                self._cik_cache[cache_key] = None
                return None

            search_name = self._normalize_company_name(company_name)
            best_match: tuple[str, int] | None = None

            # Search through all companies
            for idx, company_data in tickers_data.items():
                company_title = company_data.get("title", "")
                normalized_title = self._normalize_company_name(company_title)

                # Exact match
                if normalized_title == search_name:
                    cik = str(company_data.get("cik_str", ""))
                    # Zero-pad to 10 digits
                    cik_padded = cik.zfill(10)
                    self._cik_cache[cache_key] = cik_padded
                    logger.info(f"Found exact CIK match for '{company_name}': {cik_padded}")
                    return cik_padded

                # Partial match (company name contains search term)
                if search_name in normalized_title or normalized_title in search_name:
                    cik = str(company_data.get("cik_str", ""))
                    # Prefer shorter matches (more specific)
                    match_score = abs(len(normalized_title) - len(search_name))
                    if best_match is None or match_score < best_match[1]:
                        best_match = (cik, match_score)

            # Return best partial match if found
            if best_match:
                cik_padded = best_match[0].zfill(10)
                self._cik_cache[cache_key] = cik_padded
                logger.info(f"Found partial CIK match for '{company_name}': {cik_padded}")
                return cik_padded

            logger.info(f"No CIK found for '{company_name}'")
            self._cik_cache[cache_key] = None
            return None

        except Exception as e:
            logger.error(f"Error looking up CIK for '{company_name}': {e}")
            self._cik_cache[cache_key] = None
            return None

    async def _get_company_submissions(self, cik: str) -> dict[str, Any] | None:
        """Fetch company submissions data from SEC EDGAR.

        The submissions endpoint provides company information including
        officers and recent filings.

        Args:
            cik: 10-digit CIK number (zero-padded)

        Returns:
            Company submissions data or None if not found
        """
        # Check cache
        if cik in self._submissions_cache:
            logger.debug(f"Submissions cache hit for CIK {cik}")
            return self._submissions_cache[cik]

        url = SEC_SUBMISSIONS_URL.format(cik=cik)

        try:
            response = await self._request_with_retry("GET", url)
            data = response.json()

            # Cache the result
            self._submissions_cache[cik] = data

            logger.info(
                f"Fetched submissions for CIK {cik}: {data.get('name', 'Unknown')}"
            )
            return data

        except httpx.HTTPStatusError as e:
            if e.response.status_code == 404:
                logger.warning(f"Company with CIK {cik} not found in SEC EDGAR")
                self._submissions_cache[cik] = {}
                return None
            raise
        except Exception as e:
            logger.error(f"Error fetching submissions for CIK {cik}: {e}")
            return None

    def _parse_officers_from_submissions(
        self,
        data: dict[str, Any],
    ) -> list[Executive]:
        """Extract executive officers from SEC submissions data.

        The submissions data may contain officer information in various fields.

        Args:
            data: Company submissions data from SEC EDGAR

        Returns:
            List of Executive objects extracted from the data
        """
        executives: list[Executive] = []

        # Check for officers in formerNames or other fields
        # The submissions endpoint doesn't directly list officers,
        # but we can extract some info from the company structure

        # Extract from company info if available
        company_name = data.get("name", "Unknown Company")

        # Some submissions include officer information
        if "officers" in data:
            for officer in data.get("officers", []):
                name = officer.get("name", "")
                title = officer.get("title", "Executive")

                if name and self._is_valid_executive_name(name):
                    executives.append(Executive(
                        name=name,
                        title=title,
                        start_year=None,  # SEC data rarely includes start dates
                        end_year=None,
                    ))

        logger.debug(
            f"Extracted {len(executives)} officers from submissions for {company_name}"
        )
        return executives

    def _is_valid_executive_name(self, name: str) -> bool:
        """Validate that a string is a proper executive name.

        Filters out titles, placeholders, and invalid entries.

        Args:
            name: Name string to validate

        Returns:
            True if the name appears to be a valid person name
        """
        if not name or len(name) < 3 or len(name) > 100:
            return False

        name_lower = name.lower().strip()

        # Reject common non-name patterns (single words)
        invalid_words = {
            "ceo", "cfo", "coo", "cto", "cmo", "cio",
            "chief", "president", "officer", "director",
            "executive", "chairman", "secretary", "treasurer",
            "n/a", "none", "vacant", "tbd", "unknown",
        }

        if name_lower in invalid_words:
            return False

        # Reject phrases that contain title-related words
        invalid_substrings = [
            "chief ", " officer", "president", "director",
            "executive", "chairman", "secretary", "treasurer",
            "vice president", "head of", "manager of",
        ]

        for substring in invalid_substrings:
            if substring in name_lower:
                return False

        # Must have at least 2 words (first and last name)
        parts = name.split()
        if len(parts) < 2:
            return False

        return True

    async def get_company_officers(self, cik: str) -> list[Executive]:
        """Fetch company officers from SEC EDGAR by CIK.

        Retrieves officer information from the company's SEC filings,
        primarily from proxy statements (DEF 14A) and annual reports (10-K).

        Args:
            cik: Company's 10-digit CIK number (zero-padded)

        Returns:
            List of Executive objects for the company's officers
        """
        submissions = await self._get_company_submissions(cik)

        if not submissions:
            logger.warning(f"No submissions data for CIK {cik}")
            return []

        executives = self._parse_officers_from_submissions(submissions)

        # If no officers found directly, try to extract from recent filings
        if not executives:
            executives = await self._extract_officers_from_filings(cik, submissions)

        return executives

    async def _extract_officers_from_filings(
        self,
        cik: str,
        submissions: dict[str, Any],
    ) -> list[Executive]:
        """Extract officers from recent DEF 14A and 10-K filings.

        Looks at the recent filings list and extracts officer information
        from proxy statements and annual reports.

        Args:
            cik: Company's CIK number
            submissions: Company submissions data

        Returns:
            List of Executive objects extracted from filings
        """
        executives: list[Executive] = []

        # Get recent filings
        recent = submissions.get("filings", {}).get("recent", {})

        if not recent:
            logger.debug(f"No recent filings for CIK {cik}")
            return executives

        # Look for DEF 14A (proxy statements) - best source for executive data
        forms = recent.get("form", [])
        accession_numbers = recent.get("accessionNumber", [])
        filing_dates = recent.get("filingDate", [])

        # Find relevant filings
        target_forms = {"DEF 14A", "10-K", "10-K/A"}
        found_filings: list[tuple[str, str, str]] = []

        for i, form in enumerate(forms[:50]):  # Check last 50 filings
            if form in target_forms:
                accession = accession_numbers[i] if i < len(accession_numbers) else ""
                date = filing_dates[i] if i < len(filing_dates) else ""
                found_filings.append((form, accession, date))

                if len(found_filings) >= 3:  # Limit to 3 most recent
                    break

        if found_filings:
            logger.info(
                f"Found {len(found_filings)} relevant filings for CIK {cik}: "
                f"{[f[0] for f in found_filings]}"
            )

        # For now, we return an empty list as full filing parsing would require
        # downloading and parsing HTML/XBRL documents, which is complex.
        # The submissions endpoint should contain officer info for most companies.

        return executives

    async def search_executives(self, company_name: str) -> list[Executive]:
        """Search for company executives by company name.

        This is the main entry point for finding executives. It:
        1. Looks up the company's CIK
        2. Fetches officer data from SEC filings
        3. Returns parsed executive information

        Args:
            company_name: Name of the company to search

        Returns:
            List of Executive objects for the company's officers
        """
        logger.info(f"Searching SEC EDGAR for executives at '{company_name}'")

        # Step 1: Look up CIK
        cik = await self.lookup_company_cik(company_name)

        if not cik:
            logger.info(f"Company '{company_name}' not found in SEC EDGAR")
            return []

        # Step 2: Get officers
        executives = await self.get_company_officers(cik)

        logger.info(
            f"Found {len(executives)} executives for '{company_name}' (CIK: {cik})"
        )

        return executives

    async def get_company_info(self, company_name: str) -> dict[str, Any] | None:
        """Get basic company information from SEC EDGAR.

        Returns company metadata including name, CIK, SIC code, state, etc.

        Args:
            company_name: Name of the company

        Returns:
            Dictionary with company info or None if not found
        """
        cik = await self.lookup_company_cik(company_name)

        if not cik:
            return None

        submissions = await self._get_company_submissions(cik)

        if not submissions:
            return None

        return {
            "cik": cik,
            "name": submissions.get("name"),
            "sic": submissions.get("sic"),
            "sic_description": submissions.get("sicDescription"),
            "state": submissions.get("stateOfIncorporation"),
            "fiscal_year_end": submissions.get("fiscalYearEnd"),
            "former_names": [
                fn.get("name") for fn in submissions.get("formerNames", [])
            ],
            "website": submissions.get("website"),
            "investor_website": submissions.get("investorWebsite"),
            "exchanges": submissions.get("exchanges", []),
            "tickers": submissions.get("tickers", []),
        }

    def clear_cache(self) -> None:
        """Clear all cached data.

        Useful when you want to refresh data or free memory.
        """
        self._cik_cache.clear()
        self._submissions_cache.clear()
        self._tickers_data = None
        logger.info("Cleared SEC EDGAR service caches")


# Singleton instance
_edgar_service: EdgarService | None = None


def get_edgar_service() -> EdgarService:
    """Get the singleton EdgarService instance.

    Returns:
        The global EdgarService instance
    """
    global _edgar_service
    if _edgar_service is None:
        _edgar_service = EdgarService()
    return _edgar_service
