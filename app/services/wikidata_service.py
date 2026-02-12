"""Wikidata SPARQL service for executive data with historical tenure information.

Wikidata is an excellent source for historical executive data, particularly
for CEOs with tenure dates. It uses SPARQL queries against the public
Wikidata query service.

Key Wikidata properties used:
- P169: chief executive officer
- P39: position held
- P580: start time
- P582: end time
- P31: instance of (Q4830453 = business enterprise)
"""

import logging
import re
from datetime import datetime
from typing import Any

import httpx

from app.models import Executive

logger = logging.getLogger(__name__)

# Wikidata SPARQL endpoint
WIKIDATA_SPARQL_ENDPOINT = "https://query.wikidata.org/sparql"

# User-Agent is required by Wikidata API
# See: https://meta.wikimedia.org/wiki/User-Agent_policy
USER_AGENT = "NotchCompanyMapping/1.0 (https://github.com/notch; contact@notch.com) httpx/0.27"

# Rate limiting: Wikidata allows ~60 requests/minute
# We'll implement basic timeout and retry logic
DEFAULT_TIMEOUT = 30.0
MAX_RETRIES = 2


class WikidataService:
    """Service for querying Wikidata for company executive information.

    Wikidata provides structured data about companies and their executives,
    including historical CEOs with tenure dates. This is particularly valuable
    for building executive timelines.

    Example usage:
        service = WikidataService()
        executives = await service.search_executives("Apple Inc")
        for exec in executives:
            print(f"{exec.name}, {exec.title}, {exec.start_year}-{exec.end_year or 'Present'}")
    """

    def __init__(self, timeout: float = DEFAULT_TIMEOUT) -> None:
        """Initialize the Wikidata service.

        Args:
            timeout: HTTP request timeout in seconds.
        """
        self._http_client: httpx.AsyncClient | None = None
        self._timeout = timeout

    @property
    def is_configured(self) -> bool:
        """Wikidata is always available (public API, no API key needed)."""
        return True

    async def _get_client(self) -> httpx.AsyncClient:
        """Get or create the HTTP client with proper headers."""
        if self._http_client is None or self._http_client.is_closed:
            self._http_client = httpx.AsyncClient(
                timeout=self._timeout,
                headers={
                    "User-Agent": USER_AGENT,
                    "Accept": "application/sparql-results+json",
                },
            )
        return self._http_client

    async def close(self) -> None:
        """Close the HTTP client."""
        if self._http_client and not self._http_client.is_closed:
            await self._http_client.aclose()

    async def _execute_sparql(self, query: str) -> dict[str, Any]:
        """Execute a SPARQL query against Wikidata.

        Args:
            query: The SPARQL query string.

        Returns:
            JSON response from Wikidata.

        Raises:
            httpx.HTTPError: If the request fails.
        """
        client = await self._get_client()

        # Retry logic for transient failures
        last_error: Exception | None = None
        for attempt in range(MAX_RETRIES + 1):
            try:
                response = await client.get(
                    WIKIDATA_SPARQL_ENDPOINT,
                    params={"query": query, "format": "json"},
                )
                response.raise_for_status()
                return response.json()
            except httpx.HTTPStatusError as e:
                last_error = e
                if e.response.status_code == 429:
                    # Rate limited - wait and retry
                    logger.warning(f"Wikidata rate limited, attempt {attempt + 1}/{MAX_RETRIES + 1}")
                    if attempt < MAX_RETRIES:
                        import asyncio
                        await asyncio.sleep(2 ** attempt)  # Exponential backoff
                        continue
                raise
            except httpx.TimeoutException as e:
                last_error = e
                logger.warning(f"Wikidata timeout, attempt {attempt + 1}/{MAX_RETRIES + 1}")
                if attempt < MAX_RETRIES:
                    continue
                raise

        # Should not reach here, but just in case
        if last_error:
            raise last_error
        return {}

    async def lookup_company_wikidata_id(self, company_name: str) -> str | None:
        """Look up the Wikidata ID for a company by name.

        Uses multiple search strategies for reliability:
        1. Fast: MediaWiki API search (wbsearchentities)
        2. Fallback: SPARQL query with label service

        Args:
            company_name: The company name to search for.

        Returns:
            Wikidata entity ID (e.g., "Q312") or None if not found.
        """
        # Try fast MediaWiki API search first
        wikidata_id = await self._search_entity_api(company_name)
        if wikidata_id:
            return wikidata_id

        # Fallback to SPARQL if API search fails
        return await self._search_entity_sparql(company_name)

    async def _search_entity_api(self, company_name: str) -> str | None:
        """Search for a company using the Wikidata MediaWiki API.

        This is much faster than SPARQL for simple entity lookups.

        Args:
            company_name: The company name to search for.

        Returns:
            Wikidata entity ID or None if not found.
        """
        client = await self._get_client()

        try:
            response = await client.get(
                "https://www.wikidata.org/w/api.php",
                params={
                    "action": "wbsearchentities",
                    "search": company_name,
                    "language": "en",
                    "type": "item",
                    "limit": 10,
                    "format": "json",
                },
            )
            response.raise_for_status()
            data = response.json()

            results = data.get("search", [])
            if not results:
                return None

            # Find the best match that is a company/business
            for result in results:
                entity_id = result.get("id", "")
                label = result.get("label", "")
                description = result.get("description", "").lower()

                # Check if it looks like a company based on description
                company_keywords = [
                    "company", "corporation", "business", "enterprise",
                    "inc.", "corp.", "ltd.", "llc", "technology",
                    "manufacturer", "conglomerate", "firm", "group",
                ]

                is_likely_company = any(kw in description for kw in company_keywords)

                # Calculate match score
                score = self._calculate_match_score(company_name, label)

                if score > 0.7 and is_likely_company:
                    logger.info(f"Found Wikidata ID {entity_id} for: {company_name} (via API)")
                    return entity_id

            # If no high-confidence match with company keywords, try the best label match
            best_match = None
            best_score = 0.0
            for result in results:
                entity_id = result.get("id", "")
                label = result.get("label", "")
                score = self._calculate_match_score(company_name, label)
                if score > best_score:
                    best_score = score
                    best_match = entity_id

            if best_match and best_score > 0.8:
                logger.info(f"Found Wikidata ID {best_match} for: {company_name} (best match)")
                return best_match

            return None

        except Exception as e:
            logger.warning(f"Wikidata API search failed for {company_name}: {e}")
            return None

    async def _search_entity_sparql(self, company_name: str) -> str | None:
        """Search for a company using SPARQL (slower but more precise).

        Uses label service for better performance than FILTER + CONTAINS.

        Args:
            company_name: The company name to search for.

        Returns:
            Wikidata entity ID or None if not found.
        """
        # Escape special characters in the company name for SPARQL
        escaped_name = self._escape_sparql_string(company_name)

        # Use mwapi:search for faster text search instead of FILTER+CONTAINS
        query = f"""
        SELECT ?company ?companyLabel WHERE {{
          SERVICE wikibase:mwapi {{
            bd:serviceParam wikibase:endpoint "www.wikidata.org";
                            wikibase:api "EntitySearch";
                            mwapi:search "{escaped_name}";
                            mwapi:language "en".
            ?company wikibase:apiOutputItem mwapi:item.
          }}
          # Verify it's a business/company
          ?company wdt:P31/wdt:P279* wd:Q4830453.
          SERVICE wikibase:label {{ bd:serviceParam wikibase:language "en". }}
        }}
        LIMIT 5
        """

        try:
            data = await self._execute_sparql(query)
            results = data.get("results", {}).get("bindings", [])

            if not results:
                logger.info(f"No Wikidata company found for: {company_name}")
                return None

            # Find the best match (exact match preferred, then shortest name)
            best_match = None
            best_score = -1

            for result in results:
                label = result.get("companyLabel", {}).get("value", "")
                company_uri = result.get("company", {}).get("value", "")

                if not label or not company_uri:
                    continue

                # Calculate match score
                score = self._calculate_match_score(company_name, label)
                if score > best_score:
                    best_score = score
                    best_match = company_uri

            if best_match:
                # Extract Wikidata ID from URI (e.g., "http://www.wikidata.org/entity/Q312" -> "Q312")
                wikidata_id = best_match.split("/")[-1]
                logger.info(f"Found Wikidata ID {wikidata_id} for: {company_name} (via SPARQL)")
                return wikidata_id

            return None

        except Exception as e:
            logger.error(f"Wikidata SPARQL company lookup failed for {company_name}: {e}")
            return None

    def _validate_wikidata_id(self, wikidata_id: str) -> bool:
        """Validate that a Wikidata ID is properly formatted.

        Wikidata IDs follow the pattern Q followed by digits (e.g., Q312, Q123456).

        Args:
            wikidata_id: The ID to validate.

        Returns:
            True if the ID is valid, False otherwise.
        """
        if not wikidata_id:
            return False
        # Wikidata entity IDs are Q followed by one or more digits
        return bool(re.match(r'^Q\d+$', wikidata_id))

    async def get_executives_by_wikidata_id(self, wikidata_id: str) -> list[Executive]:
        """Get executives for a company by its Wikidata ID.

        Retrieves CEOs (P169) and other position holders (P39) with their
        tenure dates (P580 start time, P582 end time).

        Args:
            wikidata_id: Wikidata entity ID (e.g., "Q312" for Apple).

        Returns:
            List of Executive objects with tenure information.
        """
        # Validate Wikidata ID to prevent SPARQL injection
        if not self._validate_wikidata_id(wikidata_id):
            logger.warning(f"Invalid Wikidata ID format: {wikidata_id}")
            return []

        executives: list[Executive] = []

        # Query 1: Get CEOs with P169 (chief executive officer)
        ceo_query = f"""
        SELECT ?ceo ?ceoLabel ?startTime ?endTime WHERE {{
          wd:{wikidata_id} p:P169 ?statement.
          ?statement ps:P169 ?ceo.
          OPTIONAL {{ ?statement pq:P580 ?startTime. }}
          OPTIONAL {{ ?statement pq:P582 ?endTime. }}
          SERVICE wikibase:label {{ bd:serviceParam wikibase:language "en". }}
        }}
        ORDER BY DESC(?startTime)
        """

        try:
            data = await self._execute_sparql(ceo_query)
            results = data.get("results", {}).get("bindings", [])

            for result in results:
                exec = self._parse_executive_result(result, "CEO")
                if exec:
                    executives.append(exec)

            logger.info(f"Found {len(executives)} CEOs for Wikidata ID {wikidata_id}")

        except Exception as e:
            logger.error(f"Wikidata CEO query failed for {wikidata_id}: {e}")

        # Query 2: Get other executives with P39 (position held)
        # Filter for executive-level positions
        position_query = f"""
        SELECT ?person ?personLabel ?position ?positionLabel ?startTime ?endTime WHERE {{
          ?person p:P39 ?statement.
          ?statement ps:P39 ?position.
          ?statement pq:P642 wd:{wikidata_id}.  # Position at this organization
          OPTIONAL {{ ?statement pq:P580 ?startTime. }}
          OPTIONAL {{ ?statement pq:P582 ?endTime. }}
          SERVICE wikibase:label {{ bd:serviceParam wikibase:language "en". }}
        }}
        ORDER BY DESC(?startTime)
        LIMIT 50
        """

        try:
            data = await self._execute_sparql(position_query)
            results = data.get("results", {}).get("bindings", [])

            for result in results:
                position_label = result.get("positionLabel", {}).get("value", "")
                # Only include executive-level positions
                if self._is_executive_position(position_label):
                    exec = self._parse_executive_result(result, position_label, name_key="personLabel")
                    if exec:
                        executives.append(exec)

            logger.info(f"Found {len(executives)} total executives for Wikidata ID {wikidata_id}")

        except Exception as e:
            logger.error(f"Wikidata position query failed for {wikidata_id}: {e}")

        # Deduplicate by name (same person may appear in both queries)
        return self._deduplicate_executives(executives)

    async def search_executives(self, company_name: str) -> list[Executive]:
        """Search for executives of a company by name.

        This is a convenience method that combines company lookup and
        executive retrieval.

        Args:
            company_name: The company name to search for.

        Returns:
            List of Executive objects with tenure information.
        """
        # First, look up the Wikidata ID for the company
        wikidata_id = await self.lookup_company_wikidata_id(company_name)
        if not wikidata_id:
            logger.info(f"Could not find Wikidata ID for company: {company_name}")
            return []

        # Then, get executives for that company
        return await self.get_executives_by_wikidata_id(wikidata_id)

    async def get_ceo_history(self, company_name: str) -> list[Executive]:
        """Get the CEO history for a company.

        Specialized method that returns only CEOs, sorted by tenure
        (most recent first).

        Args:
            company_name: The company name to search for.

        Returns:
            List of CEO Executive objects with tenure information.
        """
        wikidata_id = await self.lookup_company_wikidata_id(company_name)
        if not wikidata_id:
            return []

        # Validate Wikidata ID to prevent SPARQL injection
        if not self._validate_wikidata_id(wikidata_id):
            logger.warning(f"Invalid Wikidata ID format from lookup: {wikidata_id}")
            return []

        # Query specifically for CEOs
        query = f"""
        SELECT ?ceo ?ceoLabel ?startTime ?endTime WHERE {{
          wd:{wikidata_id} p:P169 ?statement.
          ?statement ps:P169 ?ceo.
          OPTIONAL {{ ?statement pq:P580 ?startTime. }}
          OPTIONAL {{ ?statement pq:P582 ?endTime. }}
          SERVICE wikibase:label {{ bd:serviceParam wikibase:language "en". }}
        }}
        ORDER BY DESC(?startTime)
        """

        try:
            data = await self._execute_sparql(query)
            results = data.get("results", {}).get("bindings", [])

            executives: list[Executive] = []
            for result in results:
                exec = self._parse_executive_result(result, "CEO")
                if exec:
                    executives.append(exec)

            # Sort by start year descending (most recent first)
            executives.sort(key=lambda e: e.start_year or 0, reverse=True)

            return executives

        except Exception as e:
            logger.error(f"Wikidata CEO history query failed for {company_name}: {e}")
            return []

    def _parse_executive_result(
        self,
        result: dict[str, Any],
        default_title: str,
        name_key: str = "ceoLabel",
    ) -> Executive | None:
        """Parse a SPARQL result binding into an Executive object.

        Args:
            result: SPARQL result binding dictionary.
            default_title: Title to use (e.g., "CEO").
            name_key: Key for the person's name in the result.

        Returns:
            Executive object or None if parsing fails.
        """
        name_value = result.get(name_key, {}).get("value", "")
        if not name_value:
            return None

        # Skip if the name looks like a Wikidata Q-number (unresolved entity)
        if re.match(r"^Q\d+$", name_value):
            logger.debug(f"Skipping unresolved Wikidata entity: {name_value}")
            return None

        # Parse tenure dates
        start_year = self._parse_wikidata_date(result.get("startTime", {}).get("value"))
        end_year = self._parse_wikidata_date(result.get("endTime", {}).get("value"))

        # Validate years
        if start_year and end_year and end_year < start_year:
            logger.warning(f"Invalid tenure dates for {name_value}: {start_year}-{end_year}")
            end_year = None  # Treat as current if dates are invalid

        try:
            return Executive(
                name=name_value.strip(),
                title=default_title,
                start_year=start_year,
                end_year=end_year,
            )
        except Exception as e:
            logger.warning(f"Failed to create Executive for {name_value}: {e}")
            return None

    def _parse_wikidata_date(self, date_value: str | None) -> int | None:
        """Parse a Wikidata date string into a year.

        Wikidata dates are in ISO 8601 format, typically:
        - "2020-01-15T00:00:00Z" (full date)
        - "2020-01-01T00:00:00Z" (year only, approximated)

        Args:
            date_value: ISO 8601 date string from Wikidata.

        Returns:
            Year as integer or None if parsing fails.
        """
        if not date_value:
            return None

        try:
            # Handle standard ISO format
            if "T" in date_value:
                # Parse as datetime and extract year
                dt = datetime.fromisoformat(date_value.replace("Z", "+00:00"))
                return dt.year

            # Handle year-only format
            if date_value.isdigit() and len(date_value) == 4:
                return int(date_value)

            # Try to extract year from string
            year_match = re.search(r"(\d{4})", date_value)
            if year_match:
                year = int(year_match.group(1))
                if 1900 <= year <= 2100:
                    return year

        except (ValueError, AttributeError) as e:
            logger.debug(f"Could not parse Wikidata date {date_value}: {e}")

        return None

    def _escape_sparql_string(self, text: str) -> str:
        """Escape special characters in a string for SPARQL queries.

        Implements comprehensive escaping to prevent SPARQL injection attacks.
        Escapes all characters that could be used to break out of string literals
        or inject SPARQL commands.

        Args:
            text: The string to escape.

        Returns:
            Escaped string safe for SPARQL.
        """
        if not text:
            return ""

        # First, escape backslashes (must be done first)
        text = text.replace("\\", "\\\\")

        # Escape quotes (both single and double)
        text = text.replace('"', '\\"')
        text = text.replace("'", "\\'")

        # Escape newlines and carriage returns (can break out of strings)
        text = text.replace("\n", "\\n")
        text = text.replace("\r", "\\r")
        text = text.replace("\t", "\\t")

        # Remove characters that could be used for SPARQL injection
        # even within string literals (curly braces, angle brackets, etc.)
        # These are stripped rather than escaped as they're not valid in company names
        injection_chars = ['<', '>', '{', '}', '|', '^', '`']
        for char in injection_chars:
            text = text.replace(char, "")

        # Limit length to prevent DoS via extremely long strings
        max_length = 200
        if len(text) > max_length:
            text = text[:max_length]

        return text

    def _calculate_match_score(self, query: str, candidate: str) -> float:
        """Calculate a match score between query and candidate company names.

        Args:
            query: The search query.
            candidate: The candidate company name from Wikidata.

        Returns:
            Score from 0.0 to 1.0, higher is better.
        """
        query_lower = query.lower().strip()
        candidate_lower = candidate.lower().strip()

        # Exact match is best
        if query_lower == candidate_lower:
            return 1.0

        # Check if query is a substring
        if query_lower in candidate_lower:
            # Shorter candidates are preferred (more specific match)
            length_penalty = len(candidate_lower) / (len(query_lower) + 50)
            return 0.9 - (length_penalty * 0.3)

        # Check if candidate is a substring of query
        if candidate_lower in query_lower:
            return 0.7

        # Partial word overlap
        query_words = set(query_lower.split())
        candidate_words = set(candidate_lower.split())
        overlap = len(query_words & candidate_words)
        if overlap > 0:
            return 0.5 * (overlap / max(len(query_words), len(candidate_words)))

        return 0.0

    def _is_executive_position(self, position_label: str) -> bool:
        """Check if a position label represents an executive-level role.

        Args:
            position_label: The position title from Wikidata.

        Returns:
            True if the position is executive-level.
        """
        position_lower = position_label.lower()

        executive_keywords = [
            "chief executive",
            "chief financial",
            "chief operating",
            "chief technology",
            "chief marketing",
            "chief information",
            "chief product",
            "chief revenue",
            "chief legal",
            "chief human",
            "chief people",
            "chief strategy",
            "chief digital",
            "chief data",
            "chief security",
            "ceo",
            "cfo",
            "coo",
            "cto",
            "cmo",
            "cio",
            "cpo",
            "cro",
            "clo",
            "president",
            "chairman",
            "chairwoman",
            "chairperson",
            "vice president",
            "executive vice",
            "senior vice",
            "managing director",
            "general manager",
            "executive director",
        ]

        return any(keyword in position_lower for keyword in executive_keywords)

    def _deduplicate_executives(self, executives: list[Executive]) -> list[Executive]:
        """Deduplicate executives by name, keeping the most complete record.

        Args:
            executives: List of Executive objects to deduplicate.

        Returns:
            Deduplicated list with merged records.
        """
        if not executives:
            return []

        # Group by normalized name
        name_groups: dict[str, list[Executive]] = {}
        for exec in executives:
            normalized = self._normalize_name(exec.name)
            if normalized not in name_groups:
                name_groups[normalized] = []
            name_groups[normalized].append(exec)

        # Merge each group
        deduplicated: list[Executive] = []
        for group in name_groups.values():
            if len(group) == 1:
                deduplicated.append(group[0])
            else:
                # Merge: prefer records with dates, longest name
                merged = self._merge_executives(group)
                deduplicated.append(merged)

        return deduplicated

    def _normalize_name(self, name: str) -> str:
        """Normalize a name for comparison.

        Args:
            name: The name to normalize.

        Returns:
            Normalized lowercase name.
        """
        # Remove common suffixes
        normalized = re.sub(
            r"\s+(jr\.?|sr\.?|iii?|iv|v|phd|md|esq)\.?$",
            "",
            name,
            flags=re.IGNORECASE,
        )
        # Normalize whitespace and lowercase
        return " ".join(normalized.lower().split())

    def _merge_executives(self, executives: list[Executive]) -> Executive:
        """Merge multiple executive records for the same person.

        Args:
            executives: List of executive records to merge.

        Returns:
            Single merged Executive.
        """
        if len(executives) == 1:
            return executives[0]

        # Select best name (longest, excluding very long names)
        valid_names = [e.name for e in executives if len(e.name) <= 60]
        best_name = max(valid_names, key=len) if valid_names else executives[0].name

        # Select best title (prefer "CEO" if present)
        titles = [e.title for e in executives]
        best_title = "CEO" if "CEO" in titles else titles[0]

        # Select best start year (earliest known)
        start_years = [e.start_year for e in executives if e.start_year is not None]
        best_start_year = min(start_years) if start_years else None

        # Select best end year (None means current, which is preferred)
        end_years = [e.end_year for e in executives]
        if None in end_years:
            best_end_year = None
        else:
            non_none = [y for y in end_years if y is not None]
            best_end_year = max(non_none) if non_none else None

        return Executive(
            name=best_name,
            title=best_title,
            start_year=best_start_year,
            end_year=best_end_year,
        )


# Singleton pattern matching other services
_wikidata_service: WikidataService | None = None


def get_wikidata_service() -> WikidataService:
    """Get the singleton WikidataService instance."""
    global _wikidata_service
    if _wikidata_service is None:
        _wikidata_service = WikidataService()
    return _wikidata_service
