"""Search service using multiple data sources for company research.

Data Source Priority (authoritative to supplementary):
1. SEC EDGAR (Primary) - Authoritative for US public companies
2. Wikidata (Secondary) - Good for historical data with dates
3. Tavily/SerpAPI (Supplementary) - Current news and web data

Tavily provides AI-optimized search results, while SerpAPI gives
structured Google/LinkedIn data. Both are better suited for
programmatic access than raw Google Custom Search.
"""

import asyncio
import logging
import os
import re
from dataclasses import dataclass, field
from datetime import datetime, timezone
from enum import Enum
from typing import Any

import httpx
from rapidfuzz import fuzz

from app.models import CompanyResponse, Executive, ExecutiveV2
from app.services.openrouter_service import get_openrouter_service
from app.services.validation_service import ExecutiveValidator

# Try to import new data source services (may not exist yet)
try:
    from app.services.edgar_service import EdgarService, get_edgar_service
    EDGAR_AVAILABLE = True
except ImportError:
    EDGAR_AVAILABLE = False
    EdgarService = None  # type: ignore
    get_edgar_service = None  # type: ignore

try:
    from app.services.wikidata_service import WikidataService, get_wikidata_service
    WIKIDATA_AVAILABLE = True
except ImportError:
    WIKIDATA_AVAILABLE = False
    WikidataService = None  # type: ignore
    get_wikidata_service = None  # type: ignore

try:
    from app.services.website_scraper_service import (
        WebsiteScraperService,
        scrape_company_executives,
        ScrapedExecutive,
    )
    WEBSITE_SCRAPER_AVAILABLE = True
except ImportError:
    WEBSITE_SCRAPER_AVAILABLE = False
    WebsiteScraperService = None  # type: ignore
    scrape_company_executives = None  # type: ignore
    ScrapedExecutive = None  # type: ignore

try:
    from app.services.ai_search_service import (
        AISearchService,
        search_company_executives as ai_search_company_executives,
        DiscoveredExecutive,
    )
    AI_SEARCH_AVAILABLE = True
except ImportError:
    AI_SEARCH_AVAILABLE = False
    AISearchService = None  # type: ignore
    ai_search_company_executives = None  # type: ignore
    DiscoveredExecutive = None  # type: ignore

logger = logging.getLogger(__name__)


class DataSource(Enum):
    """Enumeration of available data sources with priority ordering."""
    SEC_EDGAR = "sec_edgar"
    WIKIDATA = "wikidata"
    WEBSITE = "website"
    AI_SEARCH = "ai_search"  # AI-powered search via OpenRouter (Perplexity, Gemini)
    TAVILY = "tavily"
    SERPAPI = "serpapi"
    KNOWLEDGE_GRAPH = "knowledge_graph"
    MOCK = "mock"


# Source confidence scores (authoritative sources get higher scores)
SOURCE_CONFIDENCE: dict[DataSource, float] = {
    DataSource.SEC_EDGAR: 1.0,       # Official SEC filings - authoritative
    DataSource.WIKIDATA: 0.95,       # Verified with citations and dates
    DataSource.KNOWLEDGE_GRAPH: 0.90, # Company website / Google's knowledge graph
    DataSource.WEBSITE: 0.85,        # Direct company website scraping
    DataSource.AI_SEARCH: 0.85,      # AI-powered multi-model search with validation
    DataSource.TAVILY: 0.7,          # AI-optimized search results
    DataSource.SERPAPI: 0.7,         # Structured Google results
    DataSource.MOCK: 0.5,            # Mock data for development
}

# Authoritative sources that trigger confidence reduction for web searches
AUTHORITATIVE_SOURCES: frozenset[DataSource] = frozenset({
    DataSource.SEC_EDGAR,
    DataSource.WIKIDATA,
    DataSource.WEBSITE,  # Direct company website is authoritative
})

# Web search sources that get reduced confidence when authoritative data exists
WEB_SEARCH_SOURCES: frozenset[DataSource] = frozenset({
    DataSource.TAVILY,
    DataSource.SERPAPI,
})

# Confidence reduction factor for web search sources when authoritative sources have data
# Web search confidence is multiplied by this factor (25% reduction)
WEB_SEARCH_CONFIDENCE_REDUCTION: float = 0.75


@dataclass
class SourcedExecutive:
    """Executive data with source attribution and confidence scoring.

    Tracks which data source provided this executive's information,
    enabling cross-validation and conflict resolution.
    """
    executive: Executive
    source: DataSource
    confidence: float
    source_url: str | None = None
    raw_data: dict[str, Any] = field(default_factory=dict)

    def to_executive_v2(self) -> ExecutiveV2:
        """Convert to ExecutiveV2 with source tracking."""
        return ExecutiveV2(
            name=self.executive.name,
            title=self.executive.title,
            start_year=self.executive.start_year,
            end_year=self.executive.end_year,
            linkedin_url=self.executive.linkedin_url,
            photo_url=self.executive.photo_url,
            employment_history=self.executive.employment_history,
            confidence_score=self.confidence,
            sources=[self.source.value] if self.source_url is None else [f"{self.source.value}: {self.source_url}"],
            last_verified=datetime.now(timezone.utc),
        )


@dataclass
class AggregatedCompanyData:
    """Aggregated company data from multiple sources.

    Tracks all sourced executives and metadata for deduplication
    and cross-validation before creating the final CompanyResponse.
    """
    name: str
    sourced_executives: list[SourcedExecutive] = field(default_factory=list)
    employees: int | None = None
    employees_source: DataSource | None = None
    ownership: str | None = None
    ownership_source: DataSource | None = None
    subsector: str | None = None
    subsector_source: DataSource | None = None
    notes: list[str] = field(default_factory=list)
    sources_queried: list[DataSource] = field(default_factory=list)
    conflicts: list[dict[str, Any]] = field(default_factory=list)

# Executive deduplication thresholds
NAME_SIMILARITY_HIGH = 90       # High similarity = same person regardless of title
NAME_SIMILARITY_MEDIUM = 85     # Medium similarity = same person if same title category
NAME_SIMILARITY_THRESHOLD = 85  # Legacy threshold (kept for compatibility)
TITLE_SIMILARITY_THRESHOLD = 80  # More lenient for titles after normalization

# Title categories for grouping similar executive roles
TITLE_CATEGORIES: dict[str, set[str]] = {
    "c_level": {
        "ceo", "cfo", "cto", "coo", "cio", "cmo", "cpo", "cro", "chro", "clo",
        "chief executive", "chief financial", "chief technology", "chief operating",
        "chief information", "chief marketing", "chief product", "chief revenue",
        "chief human resources", "chief legal", "chief strategy", "chief digital",
        "chief data", "chief security", "chief commercial", "chief administrative",
    },
    "president": {
        "president", "vice president", "vp", "svp", "evp",
        "senior vice president", "executive vice president",
        "division president", "regional president", "group president",
    },
    "board": {
        "chairman", "chairwoman", "chair", "board member", "director",
        "non-executive director", "independent director", "board chair",
        "lead director", "presiding director",
    },
}

# Source authority ranking for merge preference (higher = more authoritative)
SOURCE_AUTHORITY_RANK: dict[str, int] = {
    "sec_edgar": 3,
    "wikidata": 2,
    "website": 2,  # Company website is authoritative like Wikidata
    "ai_search": 2,  # AI search with validation is moderately authoritative
    "tavily": 1,
    "serpapi": 1,
    "knowledge_graph": 1,
    "mock": 0,
}

# Canonical title mappings for normalization
TITLE_NORMALIZATIONS: dict[str, str] = {
    # CEO variations
    "chief executive officer": "CEO",
    "chief exec officer": "CEO",
    "chief executive": "CEO",
    "c.e.o.": "CEO",
    "c.e.o": "CEO",
    # CFO variations
    "chief financial officer": "CFO",
    "chief finance officer": "CFO",
    "c.f.o.": "CFO",
    "c.f.o": "CFO",
    # COO variations
    "chief operating officer": "COO",
    "chief operations officer": "COO",
    "c.o.o.": "COO",
    "c.o.o": "COO",
    # CTO variations
    "chief technology officer": "CTO",
    "chief tech officer": "CTO",
    "chief technical officer": "CTO",
    "c.t.o.": "CTO",
    "c.t.o": "CTO",
    # CMO variations
    "chief marketing officer": "CMO",
    "c.m.o.": "CMO",
    "c.m.o": "CMO",
    # CIO variations
    "chief information officer": "CIO",
    "c.i.o.": "CIO",
    "c.i.o": "CIO",
    # CHRO variations
    "chief human resources officer": "CHRO",
    "chief hr officer": "CHRO",
    "chief people officer": "CPO",
    # CLO variations
    "chief legal officer": "CLO",
    "general counsel": "CLO",
    # CRO variations
    "chief revenue officer": "CRO",
    # President variations
    "pres.": "President",
    "pres": "President",
    # VP variations
    "vice president": "VP",
    "vice-president": "VP",
    "v.p.": "VP",
    "v.p": "VP",
    "senior vice president": "SVP",
    "senior vice-president": "SVP",
    "sr. vice president": "SVP",
    "sr vice president": "SVP",
    "executive vice president": "EVP",
    "executive vice-president": "EVP",
    "exec vice president": "EVP",
}

# API Configuration
TAVILY_API_KEY = os.getenv("TAVILY_API_KEY", "")
SERP_API_KEY = os.getenv("SERP_API_KEY", "")

TAVILY_SEARCH_URL = "https://api.tavily.com/search"
SERP_API_URL = "https://serpapi.com/search"


class SearchService:
    """Search service using multiple data sources for company research.

    Data Source Priority:
    1. SEC EDGAR (Primary) - Authoritative for US public companies
    2. Wikidata (Secondary) - Good for historical data with dates
    3. Tavily/SerpAPI (Supplementary) - Current news and web data

    Features:
    - Multi-source aggregation with priority ordering
    - Confidence scoring based on source authority
    - Cross-validation for conflicting data
    - Source attribution in executive records
    """

    def __init__(
        self,
        tavily_api_key: str | None = None,
        serp_api_key: str | None = None,
    ) -> None:
        self.tavily_api_key = tavily_api_key or TAVILY_API_KEY
        self.serp_api_key = serp_api_key or SERP_API_KEY
        self._http_client: httpx.AsyncClient | None = None
        self._openrouter = get_openrouter_service()
        self._exec_validator = ExecutiveValidator()

        # Initialize authoritative data source services
        self._edgar_service = get_edgar_service() if EDGAR_AVAILABLE and get_edgar_service else None
        self._wikidata_service = get_wikidata_service() if WIKIDATA_AVAILABLE and get_wikidata_service else None

        # Initialize AI search service (requires OPENROUTER_API_KEY)
        self._ai_search_service: AISearchService | None = None
        if AI_SEARCH_AVAILABLE:
            self._ai_search_service = AISearchService()
            # Only keep if API key is configured
            if not self._ai_search_service.is_configured:
                self._ai_search_service = None

        logger.info(
            f"SearchService initialized: "
            f"EDGAR={EDGAR_AVAILABLE}, Wikidata={WIKIDATA_AVAILABLE}, "
            f"Website={WEBSITE_SCRAPER_AVAILABLE}, AISearch={self._ai_search_service is not None}, "
            f"Tavily={bool(self.tavily_api_key)}, SerpAPI={bool(self.serp_api_key)}"
        )

    @property
    def is_configured(self) -> bool:
        """Check if at least one API is configured."""
        return bool(self.tavily_api_key or self.serp_api_key)

    @property
    def has_tavily(self) -> bool:
        return bool(self.tavily_api_key)

    @property
    def has_serp(self) -> bool:
        return bool(self.serp_api_key)

    @property
    def has_edgar(self) -> bool:
        """Check if SEC EDGAR service is available."""
        return self._edgar_service is not None

    @property
    def has_wikidata(self) -> bool:
        """Check if Wikidata service is available."""
        return self._wikidata_service is not None

    @property
    def has_website_scraper(self) -> bool:
        """Check if website scraper service is available."""
        return WEBSITE_SCRAPER_AVAILABLE

    @property
    def has_ai_search(self) -> bool:
        """Check if AI search service is available and configured."""
        return self._ai_search_service is not None

    @property
    def available_sources(self) -> list[DataSource]:
        """Get list of all available data sources."""
        sources = []
        if self.has_edgar:
            sources.append(DataSource.SEC_EDGAR)
        if self.has_wikidata:
            sources.append(DataSource.WIKIDATA)
        if self.has_website_scraper:
            sources.append(DataSource.WEBSITE)
        if self.has_ai_search:
            sources.append(DataSource.AI_SEARCH)
        if self.has_tavily:
            sources.append(DataSource.TAVILY)
        if self.has_serp:
            sources.append(DataSource.SERPAPI)
        return sources

    async def _get_client(self) -> httpx.AsyncClient:
        if self._http_client is None or self._http_client.is_closed:
            self._http_client = httpx.AsyncClient(timeout=30.0)
        return self._http_client

    async def close(self) -> None:
        """Close all HTTP clients and cleanup resources."""
        if self._http_client and not self._http_client.is_closed:
            await self._http_client.aclose()

        # Close data source service clients
        if self._edgar_service:
            await self._edgar_service.close()
        if self._wikidata_service:
            await self._wikidata_service.close()
        if self._ai_search_service:
            await self._ai_search_service.close()

    async def search_companies(
        self,
        query: str,
        search_type: str = "company",
        use_multi_source: bool = True,
    ) -> list[CompanyResponse]:
        """Search for companies using available APIs with multi-source aggregation.

        Data source priority (when use_multi_source=True):
        1. SEC EDGAR (Primary) - Authoritative for US public companies
        2. Wikidata (Secondary) - Good for historical data with dates
        3. Tavily/SerpAPI (Supplementary) - Current news and web data

        Args:
            query: Company name or industry to search for.
            search_type: "company" for single company, "industry" for multiple.
            use_multi_source: Whether to use multi-source aggregation (default True).

        Returns:
            List of CompanyResponse objects with aggregated data.
        """
        if not self.is_configured and not self.has_edgar and not self.has_wikidata:
            logger.warning("No search APIs or data sources configured")
            return []

        # Use multi-source aggregation for company searches
        if use_multi_source and search_type == "company":
            return await self._search_companies_multi_source(query)

        # Legacy flow for industry searches or when multi-source is disabled
        return await self._search_companies_legacy(query, search_type)

    async def build_company_map(
        self,
        company_name: str,
        website_url: str | None = None,
        max_history_years: int = 10,
    ) -> "CompanyMap":
        """Build a complete company leadership map for export.

        Uses the existing multi-source search pipeline, then resolves
        executives into a structured CompanyMap.

        Args:
            company_name: Name of the company to map.
            website_url: Optional company website URL.
            max_history_years: How far back to include former executives.

        Returns:
            CompanyMap ready for Excel export or JSON serialization.
        """
        from app.models import CompanyMap
        from app.services.executive_resolver import ExecutiveResolverService
        from app.services.pipeline_logger import PipelineLogger

        pipeline_log = PipelineLogger(company_name)

        # Stage 1: Multi-source search (reuse existing pipeline)
        pipeline_log.start_stage("multi_source_search")
        results = await self._search_companies_multi_source(company_name)
        pipeline_log.end_stage(
            items_found=sum(
                len(r.ceo) + len(r.c_level) + len(r.senior_level)
                for r in results
            ),
            source="all",
        )

        if not results:
            pipeline_log.add_warning("No results found from any source")
            logger.info(pipeline_log.print_summary())
            return CompanyMap(company_name=company_name, website=website_url)

        company = results[0]  # Best match

        # Stage 2: Executive resolution
        pipeline_log.start_stage("executive_resolution")
        resolver = ExecutiveResolverService()
        company_map = resolver.resolve_from_company_response(
            company=company,
            website=website_url,
            max_history_years=max_history_years,
        )
        pipeline_log.end_stage(
            items_found=len(company_map.current_executives) + len(company_map.former_executives),
        )

        # Add pipeline metadata
        if pipeline_log.warnings:
            company_map.notes.extend(pipeline_log.warnings)

        logger.info(pipeline_log.print_summary())
        return company_map

    async def _search_companies_multi_source(self, company_name: str) -> list[CompanyResponse]:
        """Search for company using multi-source aggregation with priority ordering.

        Queries all available sources in parallel, then aggregates results
        with priority given to authoritative sources (SEC EDGAR, Wikidata).

        Args:
            company_name: Name of the company to search for.

        Returns:
            List containing a single consolidated CompanyResponse.
        """
        logger.info(f"Multi-source search for: {company_name}")
        logger.info(f"Available sources: {[s.value for s in self.available_sources]}")

        # Initialize aggregation container
        aggregated = AggregatedCompanyData(name=company_name)

        # Create tasks for all available sources (parallel execution)
        tasks: list[tuple[DataSource, Any]] = []

        # Priority 1: SEC EDGAR (authoritative for US public companies)
        if self.has_edgar:
            tasks.append((DataSource.SEC_EDGAR, self._search_edgar(company_name)))

        # Priority 2: Wikidata (good for historical data with dates)
        if self.has_wikidata:
            tasks.append((DataSource.WIKIDATA, self._search_wikidata(company_name)))

        # Priority 2.5: Company website scraping (authoritative for current executives)
        # Infer website URL from company name (e.g., "Acrisure" -> "acrisure.com")
        if self.has_website_scraper:
            inferred_url = self._infer_website_url(company_name)
            if inferred_url:
                tasks.append((DataSource.WEBSITE, self._search_website(company_name, inferred_url)))

        # Priority 2.6: AI-powered multi-model search (Perplexity, Gemini)
        if self.has_ai_search:
            tasks.append((DataSource.AI_SEARCH, self._search_ai(company_name)))

        # Priority 3: Existing sources (Tavily, SerpAPI)
        search_query = f"{company_name} company executives leadership team CEO"
        if self.has_tavily:
            tasks.append((DataSource.TAVILY, self._search_tavily(search_query, company_name, "company")))
        if self.has_serp:
            tasks.append((DataSource.SERPAPI, self._search_serp(search_query, company_name, "company")))

        # Execute all searches in parallel
        if tasks:
            source_labels = [t[0] for t in tasks]
            coroutines = [t[1] for t in tasks]

            results = await asyncio.gather(*coroutines, return_exceptions=True)

            for source, result in zip(source_labels, results):
                aggregated.sources_queried.append(source)

                if isinstance(result, Exception):
                    logger.error(f"{source.value} search failed: {result}")
                    continue

                # Process results from each source
                await self._process_source_results(aggregated, source, result)

        # Cross-validate and merge executives from all sources
        merged_executives = self._cross_validate_and_merge(aggregated)

        # Build final CompanyResponse
        result = self._build_aggregated_response(aggregated, merged_executives)

        # Return empty list if no data was found
        if result is None:
            return []

        return [result]

    async def _search_edgar(self, company_name: str) -> list[SourcedExecutive]:
        """Search SEC EDGAR for company executive data.

        Uses the EdgarService.search_executives() method which:
        1. Looks up the company's CIK (Central Index Key)
        2. Fetches officer data from SEC filings
        3. Returns parsed Executive objects

        Args:
            company_name: Name of the company to search.

        Returns:
            List of SourcedExecutive objects with SEC EDGAR data.
        """
        if not self._edgar_service:
            return []

        try:
            # EdgarService.search_executives returns list[Executive]
            executives = await self._edgar_service.search_executives(company_name)

            sourced_executives: list[SourcedExecutive] = []
            for exec in executives:
                sourced_executives.append(SourcedExecutive(
                    executive=exec,
                    source=DataSource.SEC_EDGAR,
                    confidence=SOURCE_CONFIDENCE[DataSource.SEC_EDGAR],
                    source_url=None,  # SEC EDGAR doesn't provide direct URLs per executive
                    raw_data={
                        "name": exec.name,
                        "title": exec.title,
                        "start_year": exec.start_year,
                        "end_year": exec.end_year,
                    },
                ))

            logger.info(f"SEC EDGAR returned {len(sourced_executives)} executives for {company_name}")
            return sourced_executives

        except Exception as e:
            logger.error(f"SEC EDGAR search error: {e}")
            return []

    async def _search_wikidata(self, company_name: str) -> list[SourcedExecutive]:
        """Search Wikidata for company executive data.

        Wikidata is excellent for historical data with precise dates
        and verified sources via citations. Uses SPARQL queries against
        the Wikidata Query Service.

        Args:
            company_name: Name of the company to search.

        Returns:
            List of SourcedExecutive objects with Wikidata data.
        """
        if not self._wikidata_service:
            return []

        try:
            # WikidataService.search_executives returns list[Executive]
            executives = await self._wikidata_service.search_executives(company_name)

            sourced_executives: list[SourcedExecutive] = []
            for exec in executives:
                sourced_executives.append(SourcedExecutive(
                    executive=exec,
                    source=DataSource.WIKIDATA,
                    confidence=SOURCE_CONFIDENCE[DataSource.WIKIDATA],
                    source_url=None,  # Could be enhanced to include Wikidata entity URL
                    raw_data={
                        "name": exec.name,
                        "title": exec.title,
                        "start_year": exec.start_year,
                        "end_year": exec.end_year,
                    },
                ))

            logger.info(f"Wikidata returned {len(sourced_executives)} executives for {company_name}")
            return sourced_executives

        except Exception as e:
            logger.error(f"Wikidata search error: {e}")
            return []

    async def _search_website(
        self,
        company_name: str,
        website_url: str | None = None,
    ) -> list[SourcedExecutive]:
        """Scrape company website for executive information.

        Uses the WebsiteScraperService to extract executive data directly
        from the company's leadership/team pages.

        Args:
            company_name: Company name (used for logging).
            website_url: Company website URL to scrape.

        Returns:
            List of SourcedExecutive objects with scraped data.
        """
        if not WEBSITE_SCRAPER_AVAILABLE or not website_url:
            return []

        try:
            # scrape_company_executives is a convenience function that handles
            # the async context manager for WebsiteScraperService
            scraped = await scrape_company_executives(website_url)

            # Convert ScrapedExecutive to SourcedExecutive
            sourced_executives: list[SourcedExecutive] = []
            for exec_data in scraped:
                # Create an Executive object from the scraped data
                executive = Executive(
                    name=exec_data.name,
                    title=exec_data.title or "Executive",
                    start_year=None,  # Website usually doesn't have dates
                    end_year=None,
                    linkedin_url=None,
                    photo_url=exec_data.photo_url,
                )

                sourced_executives.append(SourcedExecutive(
                    executive=executive,
                    source=DataSource.WEBSITE,
                    confidence=exec_data.confidence,
                    source_url=exec_data.source_url,
                    raw_data={
                        "name": exec_data.name,
                        "title": exec_data.title,
                        "photo_url": exec_data.photo_url,
                        "scraped_from": website_url,
                    },
                ))

            logger.info(f"Website scraping returned {len(sourced_executives)} executives for {company_name}")
            return sourced_executives

        except Exception as e:
            logger.warning(f"Website scraping failed for {company_name}: {e}")
            return []

    async def _search_ai(self, company_name: str) -> list[SourcedExecutive]:
        """Search for executives using AI-powered search via OpenRouter.

        Uses multiple AI models (Perplexity, Gemini) in parallel to discover
        executives, with validation and deduplication.

        Args:
            company_name: Name of the company to search.

        Returns:
            List of SourcedExecutive objects with AI-discovered data.
        """
        if not self._ai_search_service:
            logger.debug("AI search not available - OPENROUTER_API_KEY not set")
            return []

        try:
            # AISearchService.search_executives returns list[DiscoveredExecutive]
            discovered = await self._ai_search_service.search_executives(company_name)

            sourced_executives: list[SourcedExecutive] = []
            for exec in discovered:
                # Convert DiscoveredExecutive to Executive
                executive = Executive(
                    name=exec.name,
                    title=exec.title,
                    start_year=None,  # AI search doesn't reliably extract dates
                    end_year=None,
                    linkedin_url=None,
                    photo_url=None,
                )

                # Use the discovery confidence scaled by our source confidence
                base_confidence = SOURCE_CONFIDENCE[DataSource.AI_SEARCH]
                adjusted_confidence = exec.confidence * base_confidence

                sourced_executives.append(SourcedExecutive(
                    executive=executive,
                    source=DataSource.AI_SEARCH,
                    confidence=adjusted_confidence,
                    source_url=None,
                    raw_data={
                        "name": exec.name,
                        "title": exec.title,
                        "ai_models": [m.value for m in exec.sources],
                        "validated": exec.validated,
                        "validation_reason": exec.validation_reason,
                        "original_confidence": exec.confidence,
                    },
                ))

            logger.info(f"AI search returned {len(sourced_executives)} executives for {company_name}")
            return sourced_executives

        except Exception as e:
            logger.error(f"AI search failed for {company_name}: {e}")
            return []

    def _infer_website_url(self, company_name: str) -> str | None:
        """Infer a company website URL from the company name.

        Uses simple heuristics to generate a likely website URL:
        - Removes common suffixes (Inc., Corp., LLC, etc.)
        - Converts to lowercase and removes special characters
        - Appends .com domain

        Args:
            company_name: Name of the company.

        Returns:
            Inferred website URL or None if name is too ambiguous.
        """
        if not company_name or len(company_name) < 2:
            return None

        # Normalize the company name
        name = company_name.lower().strip()

        # Remove common company suffixes
        suffixes_to_remove = [
            " incorporated", " inc.", " inc", " corporation", " corp.", " corp",
            " company", " co.", " co", " llc", " l.l.c.", " ltd.", " ltd",
            " limited", " plc", " holdings", " group", " services", " solutions",
            " international", " technologies", " technology", " tech",
        ]
        for suffix in suffixes_to_remove:
            if name.endswith(suffix):
                name = name[:-len(suffix)].strip()

        # Remove special characters and spaces
        # Keep only alphanumeric characters
        clean_name = re.sub(r"[^a-z0-9]", "", name)

        # Skip if the cleaned name is too short or too long
        if len(clean_name) < 3 or len(clean_name) > 30:
            return None

        # Generate the URL
        return f"https://{clean_name}.com"

    async def _process_source_results(
        self,
        aggregated: AggregatedCompanyData,
        source: DataSource,
        results: list[Any],
    ) -> None:
        """Process results from a single data source into the aggregation container.

        Args:
            aggregated: The aggregation container to update.
            source: The data source that provided these results.
            results: Results from the source (SourcedExecutive list or CompanyResponse list).
        """
        confidence = SOURCE_CONFIDENCE.get(source, 0.5)

        if not results:
            return

        # Handle SourcedExecutive list (from EDGAR, Wikidata, Website)
        if results and isinstance(results[0], SourcedExecutive):
            aggregated.sourced_executives.extend(results)
            return

        # Handle CompanyResponse list (from Tavily, SerpAPI)
        for company in results:
            if not isinstance(company, CompanyResponse):
                continue

            # Convert executives to SourcedExecutive with appropriate confidence
            for exec in company.ceo + company.c_level + company.senior_level:
                aggregated.sourced_executives.append(SourcedExecutive(
                    executive=exec,
                    source=source,
                    confidence=confidence,
                    source_url=company.notes if company.notes and "Source:" in company.notes else None,
                ))

            # Track metadata with source attribution (prefer higher-confidence sources)
            if company.employees and (
                aggregated.employees is None or
                confidence > SOURCE_CONFIDENCE.get(aggregated.employees_source or DataSource.MOCK, 0)
            ):
                aggregated.employees = company.employees
                aggregated.employees_source = source

            if company.ownership and (
                aggregated.ownership is None or
                confidence > SOURCE_CONFIDENCE.get(aggregated.ownership_source or DataSource.MOCK, 0)
            ):
                aggregated.ownership = company.ownership
                aggregated.ownership_source = source

            if company.subsector and (
                aggregated.subsector is None or
                confidence > SOURCE_CONFIDENCE.get(aggregated.subsector_source or DataSource.MOCK, 0)
            ):
                aggregated.subsector = company.subsector
                aggregated.subsector_source = source

            if company.notes:
                aggregated.notes.append(f"[{source.value}] {company.notes}")

    def _apply_source_priority_weighting(
        self,
        aggregated: AggregatedCompanyData,
    ) -> None:
        """Apply source priority weighting to reduce web search confidence when authoritative sources have data.

        When authoritative sources (SEC EDGAR, Wikidata) return executive data,
        web search sources (Tavily, SerpAPI) have their confidence reduced by 50%.
        This prevents noisy web search results from overriding or duplicating
        authoritative data.

        Source priority order:
        - SEC_EDGAR: 1.0 (highest, unchanged)
        - WIKIDATA: 0.95 (unchanged)
        - Company website (KNOWLEDGE_GRAPH): 0.90 (unchanged)
        - TAVILY: 0.7 (alone) or 0.35 (with authoritative, 50% reduction)
        - SERPAPI: 0.7 (alone) or 0.35 (with authoritative, 50% reduction)

        Args:
            aggregated: The aggregation container with all sourced executives.
                        Modifies confidence scores in place.
        """
        # Check if any authoritative source returned executive data
        has_authoritative_data = any(
            sourced.source in AUTHORITATIVE_SOURCES
            for sourced in aggregated.sourced_executives
        )

        if not has_authoritative_data:
            logger.debug("No authoritative source data found, keeping original confidence scores")
            return

        # Count executives by source type for logging
        authoritative_count = sum(
            1 for s in aggregated.sourced_executives
            if s.source in AUTHORITATIVE_SOURCES
        )
        web_search_count = sum(
            1 for s in aggregated.sourced_executives
            if s.source in WEB_SEARCH_SOURCES
        )

        logger.info(
            f"Authoritative sources have data ({authoritative_count} executives). "
            f"Reducing confidence of {web_search_count} web search executives by "
            f"{(1 - WEB_SEARCH_CONFIDENCE_REDUCTION) * 100:.0f}%"
        )

        # Reduce confidence of web search sources
        for sourced in aggregated.sourced_executives:
            if sourced.source in WEB_SEARCH_SOURCES:
                original_confidence = sourced.confidence
                sourced.confidence = sourced.confidence * WEB_SEARCH_CONFIDENCE_REDUCTION
                logger.debug(
                    f"  {sourced.executive.name} ({sourced.source.value}): "
                    f"confidence {original_confidence:.2f} -> {sourced.confidence:.2f}"
                )

    def _cross_validate_and_merge(
        self,
        aggregated: AggregatedCompanyData,
    ) -> list[Executive]:
        """Cross-validate executives from multiple sources and merge duplicates.

        When the same executive appears from multiple sources:
        1. Apply source priority weighting (reduce web search confidence when authoritative data exists)
        2. Validate dates between sources
        3. Prefer dates from authoritative sources (SEC EDGAR > Wikidata > others)
        4. Flag conflicts for potential review
        5. Merge records using the best data from each source

        Args:
            aggregated: The aggregation container with all sourced executives.

        Returns:
            List of deduplicated and cross-validated Executive objects.
        """
        if not aggregated.sourced_executives:
            return []

        # Apply source priority weighting before merging
        # This reduces web search confidence when authoritative sources have data
        self._apply_source_priority_weighting(aggregated)

        # Group executives by name similarity
        groups: list[list[SourcedExecutive]] = []
        used_indices: set[int] = set()

        for i, sourced1 in enumerate(aggregated.sourced_executives):
            if i in used_indices:
                continue

            group = [sourced1]
            used_indices.add(i)

            for j, sourced2 in enumerate(aggregated.sourced_executives[i + 1:], start=i + 1):
                if j in used_indices:
                    continue

                # Use existing fuzzy matching logic
                if self._is_same_executive(sourced1.executive, sourced2.executive):
                    group.append(sourced2)
                    used_indices.add(j)

            groups.append(group)

        # Merge each group with cross-validation
        merged_executives: list[Executive] = []
        for group in groups:
            merged = self._merge_with_cross_validation(group, aggregated)
            merged_executives.append(merged)

        logger.info(
            f"Cross-validation: {len(aggregated.sourced_executives)} sourced executives "
            f"-> {len(merged_executives)} merged, {len(aggregated.conflicts)} conflicts detected"
        )

        return merged_executives

    def _merge_with_cross_validation(
        self,
        group: list[SourcedExecutive],
        aggregated: AggregatedCompanyData,
    ) -> Executive:
        """Merge a group of duplicate executives with cross-validation.

        Args:
            group: List of SourcedExecutive records for the same person.
            aggregated: The aggregation container (for recording conflicts).

        Returns:
            Single merged Executive with best data from each source.
        """
        if len(group) == 1:
            return group[0].executive

        # Sort by confidence (highest first) to prefer authoritative sources
        group_sorted = sorted(group, key=lambda s: s.confidence, reverse=True)

        # Use highest-confidence source for primary data
        best = group_sorted[0]

        # Select best name (longest from high-confidence source)
        best_name = best.executive.name
        for sourced in group_sorted:
            if len(sourced.executive.name) > len(best_name) and sourced.confidence >= 0.8:
                best_name = sourced.executive.name

        # Select best title (prefer canonical forms from high-confidence sources)
        best_title = best.executive.title
        for sourced in group_sorted:
            normalized = self._normalize_title(sourced.executive.title)
            if any(abbr in normalized for abbr in ['CEO', 'CFO', 'COO', 'CTO', 'VP', 'SVP', 'EVP']):
                if sourced.confidence >= best.confidence - 0.1:
                    best_title = sourced.executive.title
                    break

        # Cross-validate dates and prefer authoritative sources
        # Collect all dates with their confidence
        start_years: list[tuple[int, float, DataSource]] = []
        end_years: list[tuple[int | None, float, DataSource]] = []

        for sourced in group:
            if sourced.executive.start_year is not None:
                start_years.append((sourced.executive.start_year, sourced.confidence, sourced.source))
            end_years.append((sourced.executive.end_year, sourced.confidence, sourced.source))

        # Check for date conflicts
        if len(start_years) > 1:
            unique_starts = set(sy[0] for sy in start_years)
            if len(unique_starts) > 1:
                # Record conflict
                aggregated.conflicts.append({
                    "type": "start_year_conflict",
                    "executive": best_name,
                    "values": [
                        {"year": sy[0], "source": sy[2].value, "confidence": sy[1]}
                        for sy in start_years
                    ],
                })
                logger.warning(
                    f"Start year conflict for {best_name}: "
                    f"{[(sy[0], sy[2].value) for sy in start_years]}"
                )

        # Select best start year (prefer highest-confidence source)
        best_start_year = None
        if start_years:
            # Sort by confidence descending, then by earliest year for ties
            start_years_sorted = sorted(start_years, key=lambda x: (-x[1], x[0]))
            best_start_year = start_years_sorted[0][0]

        # Select best end year (None means current, prefer from high-confidence source)
        best_end_year = None
        has_none_end = any(ey[0] is None for ey in end_years)
        if has_none_end:
            # If any high-confidence source says current (None), use that
            for ey in end_years:
                if ey[0] is None and ey[1] >= 0.8:
                    best_end_year = None
                    break
        else:
            # Use latest end year from highest-confidence source
            valid_ends = [(ey[0], ey[1]) for ey in end_years if ey[0] is not None]
            if valid_ends:
                valid_ends_sorted = sorted(valid_ends, key=lambda x: (-x[1], -x[0]))
                best_end_year = valid_ends_sorted[0][0]

        # Build source attribution note
        sources_used = list(set(s.source.value for s in group))
        source_note = f"Sources: {', '.join(sources_used)}"

        return Executive(
            name=best_name,
            title=best_title,
            start_year=best_start_year,
            end_year=best_end_year,
            linkedin_url=best.executive.linkedin_url,
            photo_url=best.executive.photo_url,
            employment_history=best.executive.employment_history,
        )

    def _build_aggregated_response(
        self,
        aggregated: AggregatedCompanyData,
        executives: list[Executive],
    ) -> CompanyResponse | None:
        """Build final CompanyResponse from aggregated data.

        Args:
            aggregated: The aggregation container with all metadata.
            executives: Merged and validated executives list.

        Returns:
            CompanyResponse with source attribution in notes, or None if no data found.
        """
        # If no executives found and no metadata, return None to indicate no results
        has_data = bool(
            executives or
            aggregated.employees or
            aggregated.ownership or
            aggregated.subsector
        )
        if not has_data:
            logger.info(f"No data found for {aggregated.name}, returning empty result")
            return None

        # Validate names and filter garbage before categorization
        valid_executives: list[Executive] = []
        for e in executives:
            is_valid, reason = self._exec_validator.validate_name(e.name)
            if is_valid:
                valid_executives.append(e)
            else:
                # Attempt recovery via prefix extraction
                extracted, _ = self._exec_validator.extract_name_from_prefixed(e.name)
                if extracted:
                    logger.debug(f"Recovered executive name: '{e.name}' -> '{extracted}'")
                    valid_executives.append(Executive(
                        name=extracted,
                        title=e.title,
                        start_year=e.start_year,
                        end_year=e.end_year,
                    ))
                else:
                    logger.debug(f"Dropped invalid executive name: '{e.name}' - {reason}")

        # Categorize executives (no silent drops)
        ceo_list, c_level_list, senior_list = self._categorize_executives(valid_executives)

        # Build notes with source attribution
        notes_parts = []
        if aggregated.sources_queried:
            notes_parts.append(f"Sources queried: {', '.join(s.value for s in aggregated.sources_queried)}")
        if aggregated.conflicts:
            notes_parts.append(f"Data conflicts detected: {len(aggregated.conflicts)}")
        notes_parts.extend(aggregated.notes[:3])  # Limit to first 3 source notes

        response = CompanyResponse(
            id=f"company_{hash(aggregated.name) % 100000}",
            name=aggregated.name,
            ceo=ceo_list,
            c_level=c_level_list,
            senior_level=senior_list,
            employees=aggregated.employees,
            ownership=aggregated.ownership,
            subsector=aggregated.subsector,
            notes=" | ".join(notes_parts) if notes_parts else None,
            updated=datetime.now(timezone.utc),
        )

        # Apply current/historical filtering
        return self._apply_current_historical_filtering(response)

    async def _search_companies_legacy(
        self,
        query: str,
        search_type: str,
    ) -> list[CompanyResponse]:
        """Legacy search flow using Tavily and SerpAPI only.

        Used for industry searches or when multi-source is disabled.
        """
        results: list[CompanyResponse] = []

        # Build search query based on type
        if search_type == "company":
            search_query = f"{query} company executives leadership team CEO"
        else:  # industry
            search_query = f"{query} industry leading companies executives"

        # Try Tavily first (AI-optimized results)
        if self.has_tavily:
            try:
                tavily_results = await self._search_tavily(search_query, query, search_type)
                results.extend(tavily_results)
                logger.info(f"Tavily returned {len(tavily_results)} results")
            except Exception as e:
                logger.error(f"Tavily search failed: {e}")

        # Try SerpAPI for additional structured data
        if self.has_serp:
            try:
                serp_results = await self._search_serp(search_query, query, search_type)
                results.extend(serp_results)
                logger.info(f"SerpAPI returned {len(serp_results)} results")
            except Exception as e:
                logger.error(f"SerpAPI search failed: {e}")

        # For company searches, consolidate all results (skip deduplication - consolidation handles it)
        if search_type == "company":
            return await self._consolidate_company_results(results, query)

        # For industry searches, deduplicate by normalized company name
        seen_names: set[str] = set()
        unique_results: list[CompanyResponse] = []

        for company in results:
            normalized = self._normalize_company_name(company.name)
            if normalized not in seen_names:
                seen_names.add(normalized)
                # Clean the company name before adding (take first segment)
                company.name = re.split(r'\s*[-|–|—]\s*', company.name)[0].strip()
                unique_results.append(company)

        return unique_results

    async def _search_tavily(self, query: str, original_query: str, search_type: str = "company") -> list[CompanyResponse]:
        """Search using Tavily API.

        Tavily is designed for AI applications and returns clean,
        relevant content that's easier to parse.
        """
        client = await self._get_client()

        payload = {
            "api_key": self.tavily_api_key,
            "query": query,
            "search_depth": "advanced",
            "include_answer": True,
            "include_raw_content": False,
            "max_results": 10,
        }

        response = await client.post(TAVILY_SEARCH_URL, json=payload)
        response.raise_for_status()
        data = response.json()

        companies: list[CompanyResponse] = []

        # Parse the AI-generated answer if available
        if data.get("answer"):
            extracted = self._extract_companies_from_text(data["answer"], original_query)
            companies.extend(extracted)

        # Parse individual search results
        for result in data.get("results", []):
            title = result.get("title", "")
            content = result.get("content", "")
            url = result.get("url", "")

            company = self._parse_search_result(title, content, url, original_query, search_type)
            if company:
                companies.append(company)

        return companies

    async def _search_serp(
        self,
        query: str,
        company_name: str,
        search_type: str = "company",
    ) -> list[CompanyResponse]:
        """Search using SerpAPI.

        SerpAPI provides structured Google results including:
        - Knowledge panels (company info)
        - LinkedIn results (executives)
        - Organic results
        """
        client = await self._get_client()

        params = {
            "api_key": self.serp_api_key,
            "q": query,
            "engine": "google",
            "num": 10,
        }

        response = await client.get(SERP_API_URL, params=params)
        response.raise_for_status()
        data = response.json()

        companies: list[CompanyResponse] = []

        # Parse knowledge graph if available (structured company info)
        if "knowledge_graph" in data:
            kg = data["knowledge_graph"]
            company = self._parse_knowledge_graph(kg, company_name)
            if company:
                companies.append(company)

        # Parse organic results
        for result in data.get("organic_results", []):
            title = result.get("title", "")
            snippet = result.get("snippet", "")
            link = result.get("link", "")

            company = self._parse_search_result(title, snippet, link, company_name, search_type)
            if company:
                companies.append(company)

        return companies

    def _parse_knowledge_graph(
        self,
        kg: dict[str, Any],
        query: str,
    ) -> CompanyResponse | None:
        """Parse Google Knowledge Graph data for company info."""
        title = kg.get("title", "")
        description = kg.get("description", "")

        if not title:
            return None

        # Extract executives from profiles section if available
        executives = self._extract_executives_from_kg(kg)
        ceo_list, c_level_list, senior_list = self._categorize_executives(executives)

        return CompanyResponse(
            id=f"kg_{hash(title) % 100000}",
            name=title,
            ceo=ceo_list,
            c_level=c_level_list,
            senior_level=senior_list,
            employees=self._extract_employee_count(kg),
            ownership=None,
            acquisition_date=None,
            subsector=kg.get("type") or description[:50] if description else None,
            notes=f"Source: Google Knowledge Graph",
            updated=datetime.now(timezone.utc),
        )

    def _extract_executives_from_kg(self, kg: dict[str, Any]) -> list[Executive]:
        """Extract executives from knowledge graph profiles.

        Note: Knowledge graph data typically doesn't include tenure dates,
        so start_year will be None (unknown) unless explicitly provided.
        """
        executives: list[Executive] = []

        # Check for 'people also search for' or similar sections
        profiles = kg.get("profiles", []) or kg.get("people_also_search_for", [])

        for profile in profiles:
            name = profile.get("name", "")
            title = profile.get("title", "") or profile.get("extensions", [""])[0]

            if name and title:
                # Knowledge graph typically doesn't have tenure dates
                # Use None instead of defaulting to current year
                executives.append(Executive(
                    name=name,
                    title=title,
                    start_year=None,  # Unknown - do not default to current year
                    end_year=None,
                ))

        return executives

    def _extract_employee_count(self, kg: dict[str, Any]) -> int | None:
        """Extract employee count from knowledge graph."""
        # Try common field names
        for field in ["employees", "number_of_employees", "size"]:
            if field in kg:
                value = kg[field]
                if isinstance(value, int):
                    return value
                if isinstance(value, str):
                    # Parse "1,000 employees" or "1000"
                    match = re.search(r"[\d,]+", value)
                    if match:
                        return int(match.group().replace(",", ""))
        return None

    def _parse_search_result(
        self,
        title: str,
        content: str,
        url: str,
        query: str,
        search_type: str = "company",
    ) -> CompanyResponse | None:
        """Parse a single search result into a CompanyResponse."""
        # Extract company name from title
        company_name = self._extract_company_name(title, query, search_type)
        if not company_name:
            return None

        # Validate company name quality
        if len(company_name) > 60 or len(company_name) < 2:
            return None

        # Extract executives from content
        executives = self._extract_executives_regex_fallback(content)

        # Categorize executives (no silent drops)
        ceo_list, c_level_list, senior_list = self._categorize_executives(executives)

        return CompanyResponse(
            id=f"search_{hash(company_name) % 100000}",
            name=company_name,  # Now cleaned
            ceo=ceo_list,
            c_level=c_level_list,
            senior_level=senior_list,
            employees=None,
            ownership=None,
            acquisition_date=None,
            subsector=self._extract_subsector(content),
            notes=f"Source: {url}" if url else None,
            updated=datetime.now(timezone.utc),
        )

    def _extract_companies_from_text(
        self,
        text: str,
        query: str,
    ) -> list[CompanyResponse]:
        """Extract company information from AI-generated text."""
        companies: list[CompanyResponse] = []
        executives = self._extract_executives_regex_fallback(text)

        # If we found executives, create a company entry
        if executives:
            ceo_list, c_level_list, senior_list = self._categorize_executives(executives)

            companies.append(CompanyResponse(
                id=f"tavily_{hash(query) % 100000}",
                name=query,
                ceo=ceo_list,
                c_level=c_level_list,
                senior_level=senior_list,
                employees=None,
                ownership=None,
                acquisition_date=None,
                subsector=self._extract_subsector(text),
                notes="Source: Tavily AI Search",
                updated=datetime.now(timezone.utc),
            ))

        return companies

    def _normalize_company_name(self, name: str) -> str:
        """Normalize company name for deduplication."""
        # Take only the first part before any dash
        normalized = re.split(r'\s*[-|–|—]\s*', name)[0].strip()
        # Remove common suffixes
        normalized = re.sub(r'\s*(Inc\.?|LLC|Ltd\.?|Corp\.?|Company|Co\.?)$', '', normalized, flags=re.IGNORECASE)
        return normalized.lower().strip()

    def _normalize_title(self, title: str) -> str:
        """Normalize executive title to canonical form.

        Maps various title formats to standard abbreviations:
        - "Chief Executive Officer" -> "CEO"
        - "Vice President" -> "VP"
        - etc.

        Args:
            title: The raw title string to normalize.

        Returns:
            Normalized title string with canonical abbreviations.
        """
        if not title:
            return ""

        # Clean up the title
        normalized = title.strip()

        # Convert to lowercase for matching
        title_lower = normalized.lower()

        # Check for exact matches first (most common case)
        if title_lower in TITLE_NORMALIZATIONS:
            return TITLE_NORMALIZATIONS[title_lower]

        # Try to find and replace title patterns within the string
        # Sort by length descending to match longer patterns first
        for pattern, replacement in sorted(
            TITLE_NORMALIZATIONS.items(),
            key=lambda x: len(x[0]),
            reverse=True,
        ):
            if pattern in title_lower:
                # Replace the pattern with canonical form
                # Use case-insensitive replacement
                normalized = re.sub(
                    re.escape(pattern),
                    replacement,
                    normalized,
                    flags=re.IGNORECASE,
                )
                # After first replacement, update title_lower
                title_lower = normalized.lower()

        return normalized.strip()

    def _normalize_name_for_comparison(self, name: str) -> str:
        """Normalize a name for comparison purposes.

        Handles variations like:
        - "John A. Smith" -> "john smith"
        - "J. Smith" -> "j smith"
        - "John Andrew Smith" -> "john andrew smith"

        Args:
            name: The name to normalize.

        Returns:
            Lowercase normalized name with reduced middle initials/names.
        """
        if not name:
            return ""

        # Convert to lowercase
        normalized = name.lower().strip()

        # Remove common suffixes (Jr., Sr., III, etc.)
        normalized = re.sub(r'\s+(jr\.?|sr\.?|iii?|iv|v|phd|md|esq)\.?$', '', normalized, flags=re.IGNORECASE)

        # Remove periods from initials
        normalized = re.sub(r'\.', '', normalized)

        # Normalize whitespace
        normalized = re.sub(r'\s+', ' ', normalized)

        return normalized.strip()

    def _get_title_category(self, title: str) -> str | None:
        """Determine the category of an executive title.

        Categories help identify when two people likely hold similar roles,
        which increases confidence they are the same person when names match.

        Uses word boundary matching to avoid false positives like "director" -> "cto".

        Args:
            title: The executive title to categorize.

        Returns:
            Category name ("c_level", "president", "board") or None if not categorized.
        """
        if not title:
            return None

        title_lower = title.lower()
        title_words = set(title_lower.split())

        # Check categories in priority order (board first to avoid "director" -> "cto" issue)
        # Priority: board > president > c_level (more specific categories first)
        category_order = ["board", "president", "c_level"]

        for category in category_order:
            keywords = TITLE_CATEGORIES[category]
            for keyword in keywords:
                # For single-word keywords (like "ceo", "cfo", "director"), check word boundaries
                if ' ' not in keyword:
                    if keyword in title_words:
                        return category
                # For multi-word keywords, check substring (e.g., "chief financial")
                elif keyword in title_lower:
                    return category

        return None

    def _is_same_executive(
        self,
        exec1: Executive,
        exec2: Executive,
        name_threshold: int = NAME_SIMILARITY_MEDIUM,
        check_title: bool = True,
    ) -> bool:
        """Determine if two executives are the same person using tiered fuzzy matching.

        Matching rules (in order of priority):
        1. Names with >90% similarity = same person (regardless of title)
        2. Names with >85% similarity + same title category = same person
        3. Names with >85% similarity + similar titles (>80%) = same person

        Title categories:
        - C-level: CEO, CFO, CTO, COO, CIO, CMO, CPO, etc.
        - President: President, Vice President, SVP, EVP
        - Board: Chairman, Board Member, Director

        Args:
            exec1: First executive to compare.
            exec2: Second executive to compare.
            name_threshold: Minimum similarity score (0-100) for names (default 85).
            check_title: Whether to also check title similarity.

        Returns:
            True if the executives appear to be the same person.
        """
        # Normalize names for comparison
        name1 = self._normalize_name_for_comparison(exec1.name)
        name2 = self._normalize_name_for_comparison(exec2.name)

        if not name1 or not name2:
            return False

        # Calculate name similarity using multiple methods
        # token_sort_ratio handles word order differences: "John Smith" vs "Smith, John"
        name_score = fuzz.token_sort_ratio(name1, name2)

        # partial_ratio handles initials: "J Smith" vs "John Smith"
        partial_score = fuzz.partial_ratio(name1, name2)

        # Use the higher of the two scores
        best_name_score = max(name_score, partial_score)

        # Handle special case: one name is just an initial + last name
        # e.g., "J. Thompson" vs "Andrew Thompson" or "Mark Wassersug" vs "M. Wassersug"
        words1 = name1.split()
        words2 = name2.split()
        if len(words1) >= 2 and len(words2) >= 2:
            # Compare last names directly
            last1, last2 = words1[-1], words2[-1]
            last_name_score = fuzz.ratio(last1, last2)

            # Check if first name could be an initial
            first1, first2 = words1[0], words2[0]
            if (len(first1) == 1 and first2.startswith(first1)) or \
               (len(first2) == 1 and first1.startswith(first2)):
                # Initial matches first letter of other name
                if last_name_score >= 90:
                    best_name_score = max(best_name_score, 92)

        # RULE 1: High name similarity (>90%) = same person regardless of title
        # This catches cases where the same person appears with different titles
        # e.g., "Mark Wassersug, CFO" vs "Mark Wassersug, Chief Financial Officer"
        if best_name_score >= NAME_SIMILARITY_HIGH:
            logger.debug(
                f"Same executive (high name match {best_name_score}%): "
                f"'{exec1.name}' == '{exec2.name}'"
            )
            return True

        # Below high threshold, check if we meet medium threshold
        if best_name_score < NAME_SIMILARITY_MEDIUM:
            return False

        # If not checking title, medium name match is sufficient
        if not check_title:
            return True

        # RULE 2: Medium name similarity (>85%) + same title category = same person
        # e.g., "Mark Wassersug, CFO" vs "Mark Wassersug, Chief Finance Officer"
        category1 = self._get_title_category(exec1.title)
        category2 = self._get_title_category(exec2.title)

        if category1 and category2 and category1 == category2:
            logger.debug(
                f"Same executive (medium name {best_name_score}% + same category '{category1}'): "
                f"'{exec1.name}' ({exec1.title}) == '{exec2.name}' ({exec2.title})"
            )
            return True

        # RULE 3: Medium name similarity + similar normalized titles = same person
        title1 = self._normalize_title(exec1.title)
        title2 = self._normalize_title(exec2.title)

        if not title1 or not title2:
            # If one title is missing, rely on name match alone at medium threshold
            return best_name_score >= NAME_SIMILARITY_MEDIUM

        # Compare normalized titles
        title_score = fuzz.ratio(title1.lower(), title2.lower())

        if title_score >= TITLE_SIMILARITY_THRESHOLD:
            logger.debug(
                f"Same executive (medium name {best_name_score}% + title match {title_score}%): "
                f"'{exec1.name}' ({exec1.title}) == '{exec2.name}' ({exec2.title})"
            )
            return True

        # Names are similar but titles are different categories - likely different people
        # or same person in different roles (we still want to dedupe in most cases)
        # Be conservative: only match if names are very close
        return best_name_score >= 88

    def _calculate_executive_completeness_score(self, executive: Executive) -> float:
        """Calculate a completeness score for an executive record.

        Higher scores indicate more complete/reliable data. Used for merge preference.

        Scoring criteria:
        - Has start_year: +30 points
        - Has linkedin_url: +25 points
        - Has photo_url: +10 points
        - Has employment_history: +15 points
        - Name length (more complete names): +20 points max
        - Title is canonical form: +10 points

        Args:
            executive: The executive record to score.

        Returns:
            Completeness score from 0-110.
        """
        score = 0.0

        # Data completeness scoring
        if executive.start_year is not None:
            score += 30
        if executive.linkedin_url:
            score += 25
        if executive.photo_url:
            score += 10
        if executive.employment_history:
            score += 15

        # Name completeness (longer names are usually more complete)
        # "John A. Smith" is better than "J. Smith"
        name_parts = executive.name.split()
        score += min(len(name_parts) * 5, 20)  # Max 20 points for 4+ name parts

        # Title quality (canonical forms are preferred)
        normalized_title = self._normalize_title(executive.title)
        if any(abbr in normalized_title for abbr in ['CEO', 'CFO', 'COO', 'CTO', 'CMO', 'CIO', 'VP', 'SVP', 'EVP']):
            score += 10

        return score

    def _merge_executive_records(
        self,
        executives: list[Executive],
    ) -> Executive:
        """Merge multiple executive records for the same person.

        Enhanced merge logic that prefers entries with:
        1. Higher completeness score (has startYear, linkedinUrl, etc.)
        2. More complete name (longest valid name)
        3. Canonical title forms
        4. Earliest start year when known
        5. Latest or None end year (None = still active)

        Args:
            executives: List of executive records to merge.

        Returns:
            Single merged Executive record with best data from all sources.
        """
        if len(executives) == 1:
            return executives[0]

        # Sort executives by completeness score (highest first)
        scored_executives = [
            (e, self._calculate_executive_completeness_score(e))
            for e in executives
        ]
        scored_executives.sort(key=lambda x: x[1], reverse=True)

        # Best executive is the most complete one
        best_exec = scored_executives[0][0]

        logger.debug(
            f"Merging {len(executives)} executives for '{best_exec.name}', "
            f"scores: {[f'{e.name[:20]}:{s:.0f}' for e, s in scored_executives[:3]]}"
        )

        # Select best name (longest valid name from high-scoring entries)
        valid_names = [
            (e.name, score) for e, score in scored_executives
            if len(e.name) <= 50
        ]
        if valid_names:
            # Prefer longer names from higher-scoring entries
            # Weight by score: name_length * (1 + score/100)
            best_name = max(
                valid_names,
                key=lambda x: len(x[0]) * (1 + x[1] / 100)
            )[0]
        else:
            best_name = best_exec.name

        # Select best title (prefer canonical forms from high-scoring entries)
        best_title = best_exec.title
        best_title_score = 0
        for e, completeness in scored_executives:
            normalized = self._normalize_title(e.title)
            has_canonical = any(abbr in normalized for abbr in ['CEO', 'CFO', 'COO', 'CTO', 'CMO', 'CIO', 'VP', 'SVP', 'EVP'])
            # Base score for canonical form
            title_score = 100 if has_canonical else 50
            # Add points for title length (more specific)
            title_score += min(len(e.title), 30)
            # Boost by completeness (prefer titles from more complete records)
            title_score += completeness * 0.5

            if title_score > best_title_score:
                best_title_score = title_score
                best_title = e.title

        # Select best start year (earliest known year, or None if all unknown)
        # Prefer start_year from more complete records when there are conflicts
        start_years_with_scores = [
            (e.start_year, score) for e, score in scored_executives
            if e.start_year is not None
        ]
        if start_years_with_scores:
            # If multiple start years, prefer from higher-scored entry
            # Group by year and pick the one with highest total score
            year_scores: dict[int, float] = {}
            for year, score in start_years_with_scores:
                year_scores[year] = year_scores.get(year, 0) + score

            # Use the year with highest accumulated score
            # On tie (equal scores), prefer the earliest year
            best_start_year = max(
                year_scores.items(),
                key=lambda x: (x[1], -x[0])  # Sort by score desc, then year asc (earliest)
            )[0]
        else:
            best_start_year = None

        # Select best end year (None means currently active, which is preferred)
        end_years = [e.end_year for e in executives]
        if None in end_years:
            # If any record says currently active (None), prefer that
            best_end_year = None
        else:
            # Use the latest end year
            non_none_years = [y for y in end_years if y is not None]
            best_end_year = max(non_none_years) if non_none_years else None

        # Collect best linkedin_url and photo_url (prefer from most complete record)
        best_linkedin = next(
            (e.linkedin_url for e, _ in scored_executives if e.linkedin_url),
            None
        )
        best_photo = next(
            (e.photo_url for e, _ in scored_executives if e.photo_url),
            None
        )
        # Employment history must be a list (not None) - use empty list as default
        best_history = next(
            (e.employment_history for e, _ in scored_executives if e.employment_history),
            []
        )

        return Executive(
            name=best_name,
            title=best_title,
            start_year=best_start_year,
            end_year=best_end_year,
            linkedin_url=best_linkedin,
            photo_url=best_photo,
            employment_history=best_history,
        )

    def _looks_like_article_title(self, text: str) -> bool:
        """Check if text looks like an article headline rather than a company name."""
        text_lower = text.lower()

        # Article title patterns
        article_patterns = [
            r'\b(named|joins|announces|reports|launches|expands|acquires)\b',
            r'\b(new|former|ex-|retired)\s+(ceo|cfo|cto|president)',
            r'\b(hall of fame|award|recognition|appointment)\b',
            r'\b(q[1-4]|fy\d{2,4}|earnings|revenue)\b',
            r'"[^"]+"|\'[^\']+\'',  # Quoted text typical in headlines
        ]

        for pattern in article_patterns:
            if re.search(pattern, text_lower):
                return True

        # Too many words suggests article title (company names are usually 1-4 words)
        if len(text.split()) > 6:
            return True

        return False

    def _extract_company_name(self, title: str, query: str, search_type: str = "company") -> str | None:
        """Extract company name from search result title."""
        # For company searches, always use the query as company name
        # (consolidation will merge all results anyway)
        if search_type == "company" and query:
            return query.strip()

        # For industry searches, try to extract from title
        first_segment = re.split(r'\s*[-|–|—]\s*', title)[0].strip()

        # Reject if it looks like an article title
        if self._looks_like_article_title(first_segment):
            return None

        # Validate length and content
        if first_segment and 2 < len(first_segment) < 40:
            generic_terms = ['executive', 'leadership', 'team', 'profile', 'about', 'company', 'bio', 'people']
            if not any(term in first_segment.lower() for term in generic_terms):
                return first_segment

        return None

    async def _extract_executives_with_llm(
        self, content: str, company_name: str
    ) -> list[Executive]:
        """Use LLM to extract structured executive data."""
        try:
            return await self._openrouter.extract_executives(content, company_name)
        except Exception as e:
            logger.error(f"LLM extraction failed: {e}")
            return []

    # Patterns for executive extraction with negative lookahead to reject titles as names
    # Pattern components:
    # - Name: 2-4 capitalized words, NOT starting with title words
    # - Separator: comma, dash, "as", "is the"
    # - Title: executive title keywords
    EXECUTIVE_PATTERNS: list[tuple[str, float]] = [
        # Pattern 1: "Name, Title" or "Name - Title" with negative lookahead
        # Confidence: 0.7 (strong pattern)
        (
            r'(?P<name>(?!Chief|Executive|President|Officer|CEO|CFO|CTO|COO|CMO|CIO|Vice|Senior|Director|Head|Managing)'
            r'[A-Z][a-z]+(?:\s+(?!Chief|Executive|President|Officer|CEO|CFO|CTO|COO|CMO|CIO|Vice|Senior|Director|Head|Managing)[A-Z][a-z]+){1,3})'
            r'\s*[,\-\u2013\u2014]\s*'
            r'(?P<title>(?:Chief|President|Vice|Executive|Senior|Managing|Director|Head)[^,\n]{0,50})',
            0.7,
        ),
        # Pattern 2: Division/Regional roles
        # Confidence: 0.65 (division titles are less commonly mentioned)
        (
            r'(?P<name>(?!Division|Regional|Area|Global|National)'
            r'[A-Z][a-z]+(?:\s+[A-Z][a-z]+){1,3})'
            r'\s*[,\-\u2013\u2014]\s*'
            r'(?P<title>(?:Division|Regional|Area|Global|National)\s+(?:President|Vice\s*President|VP|Director|Manager|Head)[^,\n]{0,40})',
            0.65,
        ),
        # Pattern 3: Standard C-suite pattern "Name is/as CEO" or "Name is the Chief Technology Officer"
        # Confidence: 0.75 (explicit role assignment)
        (
            r'(?P<name>[A-Z][a-z]+(?:\s+[A-Z][a-z]+){1,3})'
            r'\s+(?:is\s+(?:the\s+)?|as\s+(?:the\s+)?|serves\s+as\s+(?:the\s+)?)'
            r'(?P<title>CEO|CFO|COO|CTO|CMO|CIO|CHRO|CLO|CPO|CRO|'
            r'Chief\s+(?:Executive|Financial|Operating|Technology|Marketing|Information|'
            r'Human\s+Resources|Legal|Product|Revenue|Strategy|Digital|Data|Security)\s+Officer|'
            r'President|Vice\s*President|VP|SVP|EVP)',
            0.75,
        ),
        # Pattern 4: Name with tenure dates "(2020-present)" or "(2018-2022)"
        # Confidence: 0.8 (tenure info indicates verified data)
        (
            r'(?P<name>[A-Z][a-z]+(?:\s+[A-Z][a-z]+){1,3})'
            r'[,\s]+(?P<title>[^(]+?)'
            r'\s*\((?P<start_year>\d{4})\s*[-\u2013\u2014]\s*(?P<end_year>\d{4}|present|current)\)',
            0.8,
        ),
        # Pattern 5: Simple "Name, C-level title" (most common format)
        # Confidence: 0.6 (basic pattern, more false positives possible)
        (
            r'(?P<name>[A-Z][a-z]+(?:\s+[A-Z][a-z]+){1,3})'
            r'\s*,\s*'
            r'(?P<title>CEO|CFO|COO|CTO|CMO|CIO|CHRO|CLO|CPO|CRO)',
            0.6,
        ),
    ]

    # Invalid words that should not appear in names (titles, common words)
    INVALID_NAME_WORDS: set[str] = {
        # Executive titles
        'chief', 'executive', 'officer', 'president', 'vice', 'senior',
        'ceo', 'cfo', 'coo', 'cto', 'cmo', 'cio', 'chro', 'clo', 'cpo', 'cro',
        'director', 'manager', 'head', 'lead', 'managing',
        # Division keywords
        'division', 'regional', 'area', 'global', 'national',
        # Common false positive words
        'asked', 'questions', 'frequently', 'about', 'who', 'what', 'the',
        'and', 'for', 'with', 'our', 'their', 'company', 'inc', 'llc', 'corp',
        'new', 'former', 'current', 'acting', 'interim',
    }

    def _is_valid_executive_name(self, name: str) -> bool:
        """Check if extracted name is valid (not a title or common word).

        Uses fast pre-checks (INVALID_NAME_WORDS, length, word count) followed
        by ExecutiveValidator for comprehensive pattern matching against sentence
        fragments, action verbs, and business jargon.

        Args:
            name: The extracted name string to validate.

        Returns:
            True if the name appears to be a valid person name, False otherwise.
        """
        if not name or len(name) < 3:
            return False

        words = name.lower().split()

        # Must have at least 2 words (first and last name)
        if len(words) < 2:
            return False

        # Name too long is suspicious
        if len(name) > 50:
            return False

        # Check if any word is an invalid word
        if any(word in self.INVALID_NAME_WORDS for word in words):
            return False

        # First word should not be all caps (likely an acronym)
        first_word = name.split()[0]
        if first_word.isupper() and len(first_word) > 2:
            return False

        # Use ExecutiveValidator for deeper validation
        is_valid, reason = self._exec_validator.validate_name(name)
        if not is_valid:
            # Try recovery via extract_name_from_prefixed
            extracted, _ = self._exec_validator.extract_name_from_prefixed(name)
            if extracted:
                logger.debug(f"Name recovered via prefix extraction: '{name}' -> '{extracted}'")
                return True
            logger.debug(f"Name rejected by ExecutiveValidator: '{name}' - {reason}")
            return False

        return True

    def _parse_tenure_year(self, year_str: str | None) -> int | None:
        """Parse tenure year from string, handling 'present'/'current'.

        Args:
            year_str: Year string like "2020", "present", or "current".

        Returns:
            Integer year or None for current/present positions.
        """
        if not year_str:
            return None

        year_lower = year_str.lower().strip()
        if year_lower in ('present', 'current', 'now'):
            return None

        try:
            year = int(year_str)
            if 1900 <= year <= 2100:
                return year
        except ValueError:
            pass

        return None

    def _deduplicate_executives(
        self, executives: list[dict[str, Any]]
    ) -> list[dict[str, Any]]:
        """Deduplicate executives using fuzzy name matching.

        Uses rapidfuzz for name comparison to handle variations like:
        - "John Smith" vs "John A. Smith"
        - "CEO" vs "Chief Executive Officer"

        Merges duplicates and keeps the entry with:
        - Highest confidence score
        - Most complete name
        - Most specific title

        Args:
            executives: List of executive dicts with confidence scores.

        Returns:
            Deduplicated list with merged entries.
        """
        if not executives:
            return []

        # Group similar executives together
        groups: list[list[dict[str, Any]]] = []
        used_indices: set[int] = set()

        for i, exec1 in enumerate(executives):
            if i in used_indices:
                continue

            # Start a new group with this executive
            group = [exec1]
            used_indices.add(i)

            name1 = self._normalize_name_for_comparison(exec1['name'])
            title1 = self._normalize_title(exec1.get('title', ''))

            for j, exec2 in enumerate(executives[i + 1:], start=i + 1):
                if j in used_indices:
                    continue

                name2 = self._normalize_name_for_comparison(exec2['name'])
                title2 = self._normalize_title(exec2.get('title', ''))

                # Check name similarity
                name_score = max(
                    fuzz.token_sort_ratio(name1, name2),
                    fuzz.partial_ratio(name1, name2),
                )

                # Also check for initial matching (e.g., "J. Smith" vs "John Smith")
                words1 = name1.split()
                words2 = name2.split()
                if len(words1) >= 2 and len(words2) >= 2:
                    last1, last2 = words1[-1], words2[-1]
                    first1, first2 = words1[0], words2[0]
                    if fuzz.ratio(last1, last2) >= 90:
                        if (len(first1) == 1 and first2.startswith(first1)) or \
                           (len(first2) == 1 and first1.startswith(first2)):
                            name_score = max(name_score, 90)

                if name_score >= NAME_SIMILARITY_THRESHOLD:
                    # Names match - check if titles are compatible
                    if title1 and title2:
                        title_score = fuzz.ratio(title1.lower(), title2.lower())
                        # Same person if titles similar OR names very similar
                        if title_score >= TITLE_SIMILARITY_THRESHOLD or name_score >= 90:
                            group.append(exec2)
                            used_indices.add(j)
                    else:
                        # Missing title, rely on name match
                        group.append(exec2)
                        used_indices.add(j)

            groups.append(group)

        # Merge each group into a single executive
        deduplicated: list[dict[str, Any]] = []
        for group in groups:
            if len(group) == 1:
                deduplicated.append(group[0])
            else:
                # Merge the group
                merged = self._merge_executive_dicts(group)
                deduplicated.append(merged)

        return deduplicated

    def _merge_executive_dicts(
        self,
        exec_dicts: list[dict[str, Any]],
    ) -> dict[str, Any]:
        """Merge multiple executive dicts into one.

        Prefers entries with:
        - Higher confidence scores
        - More complete names (longer)
        - More specific titles
        - Actual dates over None (unknown)

        Args:
            exec_dicts: List of executive dicts to merge.

        Returns:
            Single merged executive dict.
        """
        if len(exec_dicts) == 1:
            return exec_dicts[0]

        # Select best name (longest valid name)
        valid_names = [d['name'] for d in exec_dicts if len(d['name']) <= 50]
        best_name = max(valid_names, key=len) if valid_names else exec_dicts[0]['name']

        # Select best title (prefer normalized canonical forms)
        best_title = exec_dicts[0].get('title', 'Unknown')
        best_title_score = 0
        for d in exec_dicts:
            title = d.get('title', '')
            if not title:
                continue
            normalized = self._normalize_title(title)
            has_canonical = any(abbr in normalized for abbr in ['CEO', 'CFO', 'COO', 'CTO', 'VP', 'SVP', 'EVP'])
            score = 100 if has_canonical else 50
            score += min(len(title), 30)
            if score > best_title_score:
                best_title_score = score
                best_title = title

        # Select best start year (earliest known year, or None if all unknown)
        # Filter out None values to find actual years
        start_years = [
            d.get('start_year') for d in exec_dicts
            if d.get('start_year') is not None
        ]
        # Use earliest known year, or None if no dates are known
        best_start_year = min(start_years) if start_years else None

        # Select best end year (None = currently active, preferred)
        end_years = [d.get('end_year') for d in exec_dicts]
        if None in end_years:
            best_end_year = None
        else:
            non_none = [y for y in end_years if y is not None]
            best_end_year = max(non_none) if non_none else None

        # Select highest confidence
        best_confidence = max(d.get('confidence', 0.5) for d in exec_dicts)

        return {
            'name': best_name,
            'title': best_title,
            'start_year': best_start_year,
            'end_year': best_end_year,
            'confidence': best_confidence,
        }

    # Additional patterns for extracting dates from narrative text
    DATE_EXTRACTION_PATTERNS: list[tuple[str, str, str | None]] = [
        # "since 2018", "since 2020"
        (r'since\s+(\d{4})', 'start', None),
        # "from 2015 to 2020", "from 2019 until present"
        (r'from\s+(\d{4})\s+(?:to|until|through|-)\s+(\d{4}|present|current|now)', 'range', None),
        # "joined in 2019", "appointed in 2020", "became CEO in 2018"
        (r'(?:joined|appointed|became|named|promoted)\s+(?:as\s+)?(?:\w+\s+)*in\s+(\d{4})', 'start', None),
        # "(2015-2020)" and "(2018-present)"
        (r'\((\d{4})\s*[-\u2013\u2014]\s*(\d{4}|present|current|now)\)', 'range', None),
        # "2019 to present", "2018 - present"
        (r'(\d{4})\s*[-\u2013\u2014to]+\s*(present|current|now)', 'range', None),
        # "in 2019", "in 2020" (generic)
        (r'\bin\s+(\d{4})\b', 'start', None),
    ]

    def _extract_dates_from_context(self, text: str, name: str) -> tuple[int | None, int | None]:
        """Extract start and end years from surrounding text context.

        Searches for date patterns in the text near the executive's name.

        Args:
            text: The full text content to search.
            name: The executive's name to find context around.

        Returns:
            Tuple of (start_year, end_year), either or both may be None.
        """
        start_year = None
        end_year = None

        # Find the name in the text and get surrounding context (200 chars each side)
        name_lower = name.lower()
        text_lower = text.lower()
        name_pos = text_lower.find(name_lower)

        if name_pos >= 0:
            # Extract context window around the name
            context_start = max(0, name_pos - 200)
            context_end = min(len(text), name_pos + len(name) + 200)
            context = text[context_start:context_end]
        else:
            # If name not found, search entire text
            context = text

        for pattern, pattern_type, _ in self.DATE_EXTRACTION_PATTERNS:
            try:
                match = re.search(pattern, context, re.IGNORECASE)
                if match:
                    if pattern_type == 'start':
                        year_str = match.group(1)
                        parsed = self._parse_tenure_year(year_str)
                        if parsed and start_year is None:
                            start_year = parsed
                    elif pattern_type == 'range':
                        start_str = match.group(1)
                        end_str = match.group(2) if len(match.groups()) > 1 else None
                        parsed_start = self._parse_tenure_year(start_str)
                        parsed_end = self._parse_tenure_year(end_str)
                        if parsed_start and start_year is None:
                            start_year = parsed_start
                        if end_str and end_year is None:
                            # "present", "current", "now" -> None (still active)
                            end_year = parsed_end  # None for present/current
            except re.error:
                continue

        return start_year, end_year

    def _extract_executives_regex_fallback(self, text: str) -> list[Executive]:
        """Extract executive names and titles from text using regex patterns.

        This is a fallback method used when LLM extraction fails or is unavailable.
        Uses multiple regex patterns with confidence scoring to extract executives
        while filtering out common false positives like titles being mistaken for names.

        Improvements over basic regex:
        - Negative lookahead to reject titles as names
        - Support for division titles (Division President, Regional VP)
        - Tenure date extraction when available
        - Confidence scoring for pattern quality
        - Deduplication with highest confidence preserved
        - Enhanced date extraction from narrative text (since 2018, joined in 2019, etc.)

        Args:
            text: Raw text content to extract executives from.

        Returns:
            List of Executive objects extracted from the text.
        """
        raw_executives: list[dict[str, Any]] = []

        for pattern, confidence in self.EXECUTIVE_PATTERNS:
            try:
                matches = re.finditer(pattern, text, re.IGNORECASE)
                for match in matches:
                    groups = match.groupdict()
                    name = groups.get('name', '').strip()

                    # Validate name isn't a title or invalid
                    if not self._is_valid_executive_name(name):
                        continue

                    title = groups.get('title', 'Unknown').strip()
                    # Clean up title - remove trailing punctuation and whitespace
                    title = re.sub(r'[\s,.\-]+$', '', title)
                    if not title or len(title) < 2:
                        continue

                    # Parse tenure dates if available from the pattern match
                    start_year_str = groups.get('start_year')
                    end_year_str = groups.get('end_year')

                    start_year = self._parse_tenure_year(start_year_str)
                    end_year = self._parse_tenure_year(end_year_str)

                    # If no dates found in pattern, try to extract from surrounding context
                    if start_year is None:
                        context_start, context_end = self._extract_dates_from_context(text, name)
                        if context_start is not None:
                            start_year = context_start
                        if context_end is not None and end_year is None:
                            end_year = context_end

                    # DO NOT default to current year - unknown dates should remain None
                    # This preserves the distinction between "unknown" and "current year"

                    # Boost confidence if we found tenure dates
                    actual_confidence = confidence
                    if start_year is not None:
                        actual_confidence = min(1.0, confidence + 0.1)

                    exec_data = {
                        'name': name,
                        'title': title.title() if title.islower() else title,
                        'start_year': start_year,
                        'end_year': end_year,
                        'confidence': actual_confidence,
                    }
                    raw_executives.append(exec_data)

            except re.error as e:
                logger.warning(f"Regex pattern error: {e}")
                continue

        # Deduplicate keeping highest confidence
        unique_executives = self._deduplicate_executives(raw_executives)

        # Convert to Executive objects
        executives: list[Executive] = []
        for exec_data in unique_executives:
            try:
                executives.append(Executive(
                    name=exec_data['name'],
                    title=exec_data['title'],
                    start_year=exec_data['start_year'],
                    end_year=exec_data['end_year'],
                ))
            except Exception as e:
                logger.warning(f"Failed to create Executive: {e}")
                continue

        logger.debug(
            f"Regex extraction found {len(executives)} executives "
            f"(from {len(raw_executives)} raw matches)"
        )

        return executives

    def _extract_subsector(self, text: str) -> str | None:
        """Extract industry/subsector from text."""
        keywords = [
            "construction", "paving", "infrastructure", "manufacturing",
            "technology", "healthcare", "financial", "services",
            "consulting", "retail", "transportation", "software",
        ]

        text_lower = text.lower()
        for keyword in keywords:
            if keyword in text_lower:
                return keyword.title()

        return None

    def _is_c_level(self, title: str) -> bool:
        """Check if title is C-level executive.

        Uses TITLE_CATEGORIES["c_level"] for comprehensive matching.
        Short acronyms use word-boundary matching to prevent false positives.
        """
        title_lower = title.lower()
        for term in TITLE_CATEGORIES["c_level"]:
            if len(term) <= 4:
                # Word-boundary match for short acronyms (cfo, cto, etc.)
                if re.search(rf'\b{re.escape(term)}\b', title_lower):
                    return True
            else:
                if term in title_lower:
                    return True
        return False

    def _is_senior_level(self, title: str) -> bool:
        """Check if title is senior level (but not C-level).

        Includes president/VP titles, board titles, and additional senior roles
        like general counsel, secretary, treasurer, controller, head of, etc.
        """
        title_lower = title.lower()
        if self._is_c_level(title_lower) or "ceo" in title_lower or "chief executive" in title_lower:
            return False

        # Check president titles
        for term in TITLE_CATEGORIES["president"]:
            if term in title_lower:
                return True

        # Check board titles
        for term in TITLE_CATEGORIES["board"]:
            if term in title_lower:
                return True

        # Additional senior roles not in TITLE_CATEGORIES
        additional_senior = [
            "general counsel", "secretary", "treasurer", "controller",
            "head of", "managing director", "partner", "founder",
            "co-founder",
        ]
        return any(t in title_lower for t in additional_senior)

    def _categorize_executives(
        self, executives: list[Executive]
    ) -> tuple[list[Executive], list[Executive], list[Executive]]:
        """Categorize executives into CEO, C-level, and senior buckets.

        Every executive lands in exactly one bucket - no silent drops.
        Uses if/elif/else so executives that don't match CEO or C-level
        fall through to senior_level as a catch-all.

        Returns:
            Tuple of (ceo_list, c_level_list, senior_list).
        """
        ceo_list: list[Executive] = []
        c_level_list: list[Executive] = []
        senior_list: list[Executive] = []

        for e in executives:
            title_lower = e.title.lower()
            if "ceo" in title_lower or "chief executive" in title_lower:
                ceo_list.append(e)
            elif self._is_c_level(e.title):
                c_level_list.append(e)
            else:
                senior_list.append(e)  # Fallback - no drops

        return ceo_list, c_level_list, senior_list

    async def _consolidate_company_results(self, results: list[CompanyResponse], query: str) -> list[CompanyResponse]:
        """For company searches, merge all results into one consolidated company.

        Uses fuzzy matching for executive deduplication to handle:
        - Name variations: "John Smith" vs "John A. Smith" vs "J. Smith"
        - Title variations: "CEO" vs "Chief Executive Officer"
        - Cross-source duplicates from Tavily, SerpAPI, and Knowledge Graph

        Uses LLM extraction for better accuracy, with regex fallback.
        """
        logger.info(f"Consolidating {len(results)} results for query: {query}")
        if not results:
            logger.warning("No results to consolidate")
            return []

        # Use query as the canonical company name
        consolidated = CompanyResponse(
            id=f"company_{hash(query) % 100000}",
            name=query.strip(),
            ceo=[],
            c_level=[],
            senior_level=[],
            employees=None,
            ownership=None,
            acquisition_date=None,
            subsector=None,
            notes=None,
        )

        # Collect all text content from results for LLM extraction
        combined_text_parts: list[str] = []
        for company in results:
            if company.notes:
                combined_text_parts.append(company.notes)
            if company.subsector:
                combined_text_parts.append(f"Industry: {company.subsector}")
            # Collect existing executive info as context
            for exec in company.ceo + company.c_level + company.senior_level:
                combined_text_parts.append(f"{exec.name}, {exec.title}")

        combined_text = "\n".join(combined_text_parts)

        # Try LLM extraction first for better accuracy
        executives: list[Executive] = []
        if combined_text and self._openrouter.is_configured:
            executives = await self._extract_executives_with_llm(combined_text, query)
            if executives:
                logger.info(f"LLM extracted {len(executives)} executives for {query}")

        # Fallback to regex extraction if LLM returns empty
        if not executives:
            executives = self._extract_executives_regex_fallback(combined_text)
            if executives:
                logger.info(f"Regex fallback extracted {len(executives)} executives for {query}")

        # Collect ALL executives from all sources (LLM/regex + knowledge graph results)
        all_executives: list[Executive] = list(executives)

        # Collect executives from search results (knowledge graph, etc.)
        for company in results:
            # Take metadata from first company that has it
            if company.employees and not consolidated.employees:
                consolidated.employees = company.employees
            if company.ownership and not consolidated.ownership:
                consolidated.ownership = company.ownership
            if company.subsector and not consolidated.subsector:
                consolidated.subsector = company.subsector
            if company.notes and not consolidated.notes:
                consolidated.notes = company.notes

            # Collect all executives from this source
            for exec in company.ceo + company.c_level + company.senior_level:
                all_executives.append(exec)

        # Apply fuzzy deduplication to ALL collected executives
        deduplicated_executives = self._fuzzy_deduplicate_executives(all_executives)

        logger.info(
            f"Fuzzy deduplication: {len(all_executives)} -> {len(deduplicated_executives)} executives"
        )

        # Categorize deduplicated executives by normalized title
        for exec in deduplicated_executives:
            normalized_title = self._normalize_title(exec.title).lower()

            if 'ceo' in normalized_title or 'chief executive' in exec.title.lower():
                consolidated.ceo.append(exec)
            elif any(t in normalized_title for t in ['cfo', 'cto', 'coo', 'cmo', 'cio', 'chro', 'clo', 'cpo', 'cro']) or \
                 'chief' in exec.title.lower():
                consolidated.c_level.append(exec)
            else:
                consolidated.senior_level.append(exec)

        # Apply current/historical filtering and ensure only one current CEO
        consolidated = self._apply_current_historical_filtering(consolidated)

        # Always return the consolidated result for company searches
        # (ensures CompanyCard displays even if no executives found)
        logger.info(
            f"Consolidated result: {consolidated.name}, "
            f"CEOs: {len(consolidated.ceo)}, "
            f"C-level: {len(consolidated.c_level)}, "
            f"Senior: {len(consolidated.senior_level)}"
        )
        return [consolidated]

    def _apply_current_historical_filtering(
        self,
        company: CompanyResponse,
    ) -> CompanyResponse:
        """Apply current/historical filtering and ensure only one current CEO.

        This method:
        1. Ensures only one current CEO exists (keeps most recent by start_year)
        2. Marks duplicate current CEOs as historical by setting their end_year
        3. Sorts executives within each list (current first, then by start_year)

        Args:
            company: The CompanyResponse to filter.

        Returns:
            A new CompanyResponse with filtered and sorted executives.
        """
        from datetime import datetime

        # Process CEOs - ensure only one current CEO
        current_ceos = [e for e in company.ceo if e.is_current]
        historical_ceos = [e for e in company.ceo if not e.is_current]

        if len(current_ceos) > 1:
            logger.info(
                f"Multiple current CEOs found ({len(current_ceos)}) for {company.name}, "
                "keeping only the most recent by start_year"
            )
            # Sort by start_year descending (most recent first)
            current_ceos_sorted = sorted(
                current_ceos,
                key=lambda e: e.start_year or 0,
                reverse=True,
            )
            # Keep only the most recent as current CEO
            kept_ceo = current_ceos_sorted[0]
            current_ceos = [kept_ceo]

            # Mark others as historical by creating new Executive objects with end_year set
            current_year = datetime.now().year
            for extra_ceo in current_ceos_sorted[1:]:
                # Create a new Executive with end_year set to make it historical
                historical_exec = Executive(
                    name=extra_ceo.name,
                    title=extra_ceo.title,
                    start_year=extra_ceo.start_year,
                    end_year=current_year,  # Mark as ended this year
                    linkedin_url=extra_ceo.linkedin_url,
                    photo_url=extra_ceo.photo_url,
                    employment_history=extra_ceo.employment_history,
                )
                historical_ceos.append(historical_exec)

        # Combine CEOs: current first, then historical sorted by end_year desc
        all_ceos = current_ceos + sorted(
            historical_ceos,
            key=lambda e: (e.end_year or 9999, e.start_year or 0),
            reverse=True,
        )

        # Sort C-level: current first (sorted by start_year desc), then historical
        current_c_level = sorted(
            [e for e in company.c_level if e.is_current],
            key=lambda e: e.start_year or 0,
            reverse=True,
        )
        historical_c_level = sorted(
            [e for e in company.c_level if not e.is_current],
            key=lambda e: (e.end_year or 9999, e.start_year or 0),
            reverse=True,
        )
        all_c_level = current_c_level + historical_c_level

        # Sort senior-level: current first (sorted by start_year desc), then historical
        current_senior = sorted(
            [e for e in company.senior_level if e.is_current],
            key=lambda e: e.start_year or 0,
            reverse=True,
        )
        historical_senior = sorted(
            [e for e in company.senior_level if not e.is_current],
            key=lambda e: (e.end_year or 9999, e.start_year or 0),
            reverse=True,
        )
        all_senior = current_senior + historical_senior

        # Create new CompanyResponse with sorted executives
        return CompanyResponse(
            id=company.id,
            name=company.name,
            ceo=all_ceos,
            c_level=all_c_level,
            senior_level=all_senior,
            employees=company.employees,
            ownership=company.ownership,
            acquisition_date=company.acquisition_date,
            subsector=company.subsector,
            notes=company.notes,
            updated=company.updated,
            network_status=company.network_status,
            contact_status=company.contact_status,
        )

    def _fuzzy_deduplicate_executives(
        self,
        executives: list[Executive],
    ) -> list[Executive]:
        """Deduplicate executives using enhanced fuzzy name and title matching.

        Groups similar executives together and merges them using tiered matching:
        1. Names with >90% similarity = same person (regardless of title)
        2. Names with >85% similarity + same title category = same person
        3. Names with >85% similarity + similar titles (>80%) = same person

        When merging duplicates, prefers the entry with:
        1. Higher completeness score (has startYear, linkedinUrl, etc.)
        2. More complete name
        3. Canonical title form

        Args:
            executives: List of Executive objects from multiple sources.

        Returns:
            Deduplicated list of merged Executive objects.
        """
        if not executives:
            return []

        # Group similar executives together
        groups: list[list[Executive]] = []
        used_indices: set[int] = set()

        for i, exec1 in enumerate(executives):
            if i in used_indices:
                continue

            # Start a new group with this executive
            group = [exec1]
            used_indices.add(i)

            for j, exec2 in enumerate(executives[i + 1:], start=i + 1):
                if j in used_indices:
                    continue

                # Use enhanced fuzzy matching with tiered thresholds
                if self._is_same_executive(exec1, exec2):
                    group.append(exec2)
                    used_indices.add(j)

            groups.append(group)

        # Log groups with multiple entries (duplicates being merged)
        duplicate_groups = [g for g in groups if len(g) > 1]
        if duplicate_groups:
            logger.info(
                f"Deduplication found {len(duplicate_groups)} duplicate groups: "
                f"{[f'{g[0].name}({len(g)})' for g in duplicate_groups[:5]]}"
            )

        # Merge each group into a single executive using enhanced merge logic
        deduplicated: list[Executive] = []
        for group in groups:
            merged = self._merge_executive_records(group)
            deduplicated.append(merged)

        return deduplicated


# Singleton
_search_service: SearchService | None = None


def get_search_service() -> SearchService:
    """Get the singleton SearchService instance."""
    global _search_service
    if _search_service is None:
        _search_service = SearchService()
    return _search_service
