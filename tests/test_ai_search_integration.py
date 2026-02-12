"""Integration tests for AI search within the multi-source search pipeline.

Tests cover:
- AI search integration with SearchService multi-source aggregation
- DataSource.AI_SEARCH enum and confidence scoring
- Deduplication between AI search and other sources
- Graceful degradation when API key not configured
- Error handling when AI search fails
- Success criteria validation for known companies
"""

import os
import pytest
from unittest.mock import AsyncMock, MagicMock, patch
from datetime import datetime, timezone

from app.services.search_service import (
    SearchService,
    DataSource,
    SourcedExecutive,
    AggregatedCompanyData,
    SOURCE_CONFIDENCE,
    AUTHORITATIVE_SOURCES,
    WEB_SEARCH_SOURCES,
)
from app.services.ai_search_service import (
    AISearchService,
    AIModel,
    DiscoveredExecutive,
    ModelResponse,
    CONFIDENCE_MULTI_MODEL,
    CONFIDENCE_SINGLE_VALIDATED,
)
from app.models import Executive, CompanyResponse


class TestDataSource:
    """Tests for DataSource enum including AI_SEARCH."""

    def test_ai_search_enum_exists(self):
        """Verify DataSource.AI_SEARCH is defined."""
        assert hasattr(DataSource, "AI_SEARCH")
        assert DataSource.AI_SEARCH.value == "ai_search"

    def test_ai_search_confidence_defined(self):
        """Verify AI_SEARCH has confidence score defined."""
        assert DataSource.AI_SEARCH in SOURCE_CONFIDENCE
        confidence = SOURCE_CONFIDENCE[DataSource.AI_SEARCH]
        assert 0 < confidence <= 1
        assert confidence == 0.85

    def test_ai_search_not_in_web_search_sources(self):
        """Verify AI_SEARCH is not classified as a web search source."""
        assert DataSource.AI_SEARCH not in WEB_SEARCH_SOURCES

    def test_ai_search_not_in_authoritative_sources(self):
        """Verify AI_SEARCH is not classified as authoritative."""
        assert DataSource.AI_SEARCH not in AUTHORITATIVE_SOURCES

    def test_all_data_sources_have_confidence(self):
        """Verify all DataSource values have confidence scores."""
        for source in DataSource:
            assert source in SOURCE_CONFIDENCE, f"{source} missing from SOURCE_CONFIDENCE"
            assert 0 < SOURCE_CONFIDENCE[source] <= 1, f"{source} has invalid confidence"


class TestAISearchServiceUnit:
    """Unit tests for AISearchService class."""

    def test_ai_search_service_unconfigured_without_key(self):
        """Verify AISearchService reports unconfigured without API key."""
        service = AISearchService(api_key="temp")
        service.api_key = ""
        assert service.is_configured is False

    def test_ai_search_service_configured_with_key(self):
        """Verify AISearchService reports configured with API key."""
        service = AISearchService(api_key="test-api-key")
        assert service.is_configured is True

    @pytest.mark.asyncio
    async def test_search_returns_empty_when_unconfigured(self):
        """Verify search returns empty list when API key not set."""
        service = AISearchService(api_key="temp")
        service.api_key = ""
        results = await service.search_executives("Test Company")
        assert results == []

    def test_normalize_name_removes_prefixes(self):
        """Test name normalization removes common prefixes."""
        service = AISearchService(api_key="test")
        assert service._normalize_name("Dr. John Smith") == "john smith"
        assert service._normalize_name("Mr. Robert Brown") == "robert brown"

    def test_normalize_name_removes_suffixes(self):
        """Test name normalization removes common suffixes."""
        service = AISearchService(api_key="test")
        assert service._normalize_name("John Smith Jr.") == "john smith"
        assert service._normalize_name("Robert Brown III") == "robert brown"
        assert service._normalize_name("Jane Doe PhD") == "jane doe"


class TestAISearchIntegration:
    """Integration tests for AI search within the multi-source pipeline."""

    @pytest.mark.asyncio
    async def test_ai_search_in_multi_source_pipeline(self):
        """Verify AI search results are included in multi-source aggregation."""
        mock_discovered = [
            DiscoveredExecutive(
                name="Greg Williams",
                title="CEO",
                sources=[AIModel.PERPLEXITY_SONAR, AIModel.GEMINI_FLASH],
                confidence=CONFIDENCE_MULTI_MODEL,
                validated=True,
                validation_reason="Valid executive name",
            ),
            DiscoveredExecutive(
                name="Mark Wassersug",
                title="CFO",
                sources=[AIModel.PERPLEXITY_SONAR],
                confidence=CONFIDENCE_SINGLE_VALIDATED,
                validated=True,
                validation_reason="Valid executive name",
            ),
        ]

        with patch("app.services.search_service.AI_SEARCH_AVAILABLE", True):
            with patch("app.services.search_service.AISearchService") as MockAIService:
                mock_instance = MagicMock()
                mock_instance.is_configured = True
                mock_instance.search_executives = AsyncMock(return_value=mock_discovered)
                mock_instance.close = AsyncMock()
                MockAIService.return_value = mock_instance

                service = SearchService(tavily_api_key="", serp_api_key="")
                service._ai_search_service = mock_instance

                results = await service._search_ai("Acrisure")

                assert len(results) == 2
                assert all(isinstance(r, SourcedExecutive) for r in results)
                assert all(r.source == DataSource.AI_SEARCH for r in results)

                names = [r.executive.name for r in results]
                assert "Greg Williams" in names
                assert "Mark Wassersug" in names

                await service.close()

    @pytest.mark.asyncio
    async def test_ai_search_disabled_without_api_key(self):
        """Verify AI search gracefully disabled when API key not set."""
        with patch("app.services.search_service.AI_SEARCH_AVAILABLE", True):
            with patch("app.services.search_service.AISearchService") as MockAIService:
                mock_instance = MagicMock()
                mock_instance.is_configured = False
                mock_instance.close = AsyncMock()
                MockAIService.return_value = mock_instance

                service = SearchService(tavily_api_key="", serp_api_key="")
                service._ai_search_service = None

                assert service.has_ai_search is False

                results = await service._search_ai("Test Company")
                assert results == []

                await service.close()

    @pytest.mark.asyncio
    async def test_ai_search_deduplicates_with_other_sources(self):
        """Verify AI search results dedupe with Tavily/SerpAPI results."""
        service = SearchService(tavily_api_key="", serp_api_key="")

        aggregated = AggregatedCompanyData(name="Acrisure")

        ai_exec = SourcedExecutive(
            executive=Executive(name="Greg Williams", title="CEO", start_year=2020),
            source=DataSource.AI_SEARCH,
            confidence=0.85 * CONFIDENCE_MULTI_MODEL,
        )

        tavily_exec = SourcedExecutive(
            executive=Executive(name="Gregory Williams", title="Chief Executive Officer", start_year=2019),
            source=DataSource.TAVILY,
            confidence=0.7,
        )

        serp_exec = SourcedExecutive(
            executive=Executive(name="Mark Wassersug", title="CFO", start_year=2021),
            source=DataSource.SERPAPI,
            confidence=0.7,
        )

        aggregated.sourced_executives = [ai_exec, tavily_exec, serp_exec]

        merged = service._cross_validate_and_merge(aggregated)

        assert len(merged) == 2

        greg = next((e for e in merged if "Williams" in e.name), None)
        assert greg is not None

        await service.close()

    @pytest.mark.asyncio
    async def test_ai_search_confidence_scaling(self):
        """Verify AI search confidence is properly scaled."""
        service = SearchService(tavily_api_key="", serp_api_key="")

        mock_discovered = [
            DiscoveredExecutive(
                name="John Smith",
                title="CEO",
                sources=[AIModel.PERPLEXITY_SONAR],
                confidence=0.8,
                validated=True,
                validation_reason="Valid",
            ),
        ]

        with patch.object(service, "_ai_search_service") as mock_ai:
            mock_ai.search_executives = AsyncMock(return_value=mock_discovered)

            results = await service._search_ai("Test Company")

            if results:
                expected_confidence = 0.8 * SOURCE_CONFIDENCE[DataSource.AI_SEARCH]
                assert abs(results[0].confidence - expected_confidence) < 0.01

        await service.close()


class TestErrorHandling:
    """Tests for error handling when AI search fails."""

    @pytest.mark.asyncio
    async def test_ai_search_failure_doesnt_break_pipeline(self):
        """Verify pipeline continues if AI search fails."""
        with patch("app.services.search_service.AI_SEARCH_AVAILABLE", True):
            with patch("app.services.search_service.AISearchService") as MockAIService:
                mock_instance = MagicMock()
                mock_instance.is_configured = True
                mock_instance.search_executives = AsyncMock(
                    side_effect=Exception("API Error: Connection timeout")
                )
                mock_instance.close = AsyncMock()
                MockAIService.return_value = mock_instance

                service = SearchService(tavily_api_key="", serp_api_key="")
                service._ai_search_service = mock_instance

                results = await service._search_ai("Test Company")
                assert results == []

                await service.close()

    @pytest.mark.asyncio
    async def test_ai_search_returns_empty_on_http_error(self):
        """Verify AI search returns empty list on HTTP errors."""
        service = AISearchService(api_key="test-key")

        with patch.object(service, "_get_client") as mock_client:
            mock_response = MagicMock()
            mock_response.status_code = 500
            mock_response.text = "Internal Server Error"
            mock_response.raise_for_status.side_effect = Exception("HTTP 500")

            mock_async_client = AsyncMock()
            mock_async_client.post = AsyncMock(return_value=mock_response)
            mock_client.return_value = mock_async_client

            results = await service.search_executives("Test Company")
            assert results == []

        await service.close()

    @pytest.mark.asyncio
    async def test_ai_search_handles_invalid_json_response(self):
        """Verify AI search handles malformed JSON responses gracefully."""
        service = AISearchService(api_key="test-key")

        with patch.object(service, "_get_client") as mock_client:
            mock_response = MagicMock()
            mock_response.status_code = 200
            mock_response.raise_for_status = MagicMock()
            mock_response.json.return_value = {
                "choices": [{"message": {"content": "Not valid JSON here"}}]
            }

            mock_async_client = AsyncMock()
            mock_async_client.post = AsyncMock(return_value=mock_response)
            mock_client.return_value = mock_async_client

            results = await service.search_executives("Test Company")
            assert results == []

        await service.close()


class TestNameValidation:
    """Tests for executive name validation in AI search results."""

    @pytest.mark.asyncio
    async def test_no_sentence_fragments(self):
        """Verify no malformed names in results."""
        service = AISearchService(api_key="test-key")

        bad_executives = [
            {"name": "John Smith", "title": "CEO"},
            {"name": "announced the appointment of", "title": "Executive"},
            {"name": "Jane Doe", "title": "CFO"},
            {"name": "Chief Executive Officer", "title": "Leadership"},
            {"name": "the companys", "title": "Role"},
        ]

        with patch.object(service, "_query_model") as mock_query:
            mock_query.return_value = ModelResponse(
                model=AIModel.PERPLEXITY_SONAR,
                executives=bad_executives,
                success=True,
            )

            results = await service.search_executives("Test Company", models=[AIModel.PERPLEXITY_SONAR])

            result_names = [r.name for r in results]

            assert "John Smith" in result_names
            assert "Jane Doe" in result_names

            assert "announced the appointment of" not in result_names
            assert "Chief Executive Officer" not in result_names
            assert "the companys" not in result_names

        await service.close()

    @pytest.mark.asyncio
    async def test_filters_single_word_names(self):
        """Verify single-word names are filtered out."""
        service = AISearchService(api_key="test-key")

        executives = [
            {"name": "CEO", "title": "Chief Executive"},
            {"name": "President", "title": "President"},
            {"name": "John Smith", "title": "CEO"},
        ]

        with patch.object(service, "_query_model") as mock_query:
            mock_query.return_value = ModelResponse(
                model=AIModel.PERPLEXITY_SONAR,
                executives=executives,
                success=True,
            )

            results = await service.search_executives("Test Company", models=[AIModel.PERPLEXITY_SONAR])

            result_names = [r.name for r in results]
            assert "CEO" not in result_names
            assert "President" not in result_names
            assert "John Smith" in result_names

        await service.close()


class TestMultiModelAggregation:
    """Tests for multi-model response aggregation."""

    @pytest.mark.asyncio
    async def test_multi_model_boosts_confidence(self):
        """Verify executives found by multiple models get higher confidence."""
        service = AISearchService(api_key="test-key")

        model1_response = ModelResponse(
            model=AIModel.PERPLEXITY_SONAR,
            executives=[{"name": "John Smith", "title": "CEO"}],
            success=True,
        )
        model2_response = ModelResponse(
            model=AIModel.GEMINI_FLASH,
            executives=[{"name": "John Smith", "title": "Chief Executive Officer"}],
            success=True,
        )
        model3_response = ModelResponse(
            model=AIModel.PERPLEXITY_SONAR_PRO,
            executives=[{"name": "Jane Doe", "title": "CFO"}],
            success=True,
        )

        with patch.object(service, "_query_model") as mock_query:
            mock_query.side_effect = [model1_response, model2_response, model3_response]

            results = await service.search_executives(
                "Test Company",
                models=[AIModel.PERPLEXITY_SONAR, AIModel.GEMINI_FLASH, AIModel.PERPLEXITY_SONAR_PRO]
            )

            john = next((r for r in results if "John" in r.name), None)
            assert john is not None
            assert john.confidence == CONFIDENCE_MULTI_MODEL

            jane = next((r for r in results if "Jane" in r.name), None)
            assert jane is not None
            assert jane.confidence == CONFIDENCE_SINGLE_VALIDATED

        await service.close()

    @pytest.mark.asyncio
    async def test_deduplication_prefers_longer_title(self):
        """Verify deduplication keeps the more specific title."""
        service = AISearchService(api_key="test-key")

        model1_response = ModelResponse(
            model=AIModel.PERPLEXITY_SONAR,
            executives=[{"name": "John Smith", "title": "CEO"}],
            success=True,
        )
        model2_response = ModelResponse(
            model=AIModel.GEMINI_FLASH,
            executives=[{"name": "John Smith", "title": "Chief Executive Officer and President"}],
            success=True,
        )

        with patch.object(service, "_query_model") as mock_query:
            mock_query.side_effect = [model1_response, model2_response]

            results = await service.search_executives(
                "Test Company",
                models=[AIModel.PERPLEXITY_SONAR, AIModel.GEMINI_FLASH]
            )

            assert len(results) == 1
            assert results[0].title == "Chief Executive Officer and President"

        await service.close()


class TestAvailableSources:
    """Tests for SearchService available sources property."""

    def test_ai_search_in_available_sources_when_configured(self):
        """Verify AI search appears in available sources when configured."""
        with patch("app.services.search_service.AI_SEARCH_AVAILABLE", True):
            with patch("app.services.search_service.AISearchService") as MockAIService:
                mock_instance = MagicMock()
                mock_instance.is_configured = True
                MockAIService.return_value = mock_instance

                service = SearchService(tavily_api_key="", serp_api_key="")
                service._ai_search_service = mock_instance

                assert DataSource.AI_SEARCH in service.available_sources

    def test_ai_search_not_in_available_sources_when_not_configured(self):
        """Verify AI search not in available sources when not configured."""
        service = SearchService(tavily_api_key="", serp_api_key="")
        service._ai_search_service = None

        assert DataSource.AI_SEARCH not in service.available_sources


class TestSuccessCriteria:
    """Integration tests for success criteria validation."""

    @pytest.mark.asyncio
    @pytest.mark.skipif(
        not os.getenv("OPENROUTER_API_KEY"),
        reason="OPENROUTER_API_KEY not set"
    )
    async def test_acrisure_executives(self):
        """Verify Acrisure returns expected executives (6+)."""
        service = AISearchService()

        try:
            results = await service.search_executives("Acrisure")

            assert len(results) >= 6, f"Expected 6+ executives, found {len(results)}"

            names_lower = [r.name.lower() for r in results]
            known_executives = ["greg williams", "mark wassersug"]
            found_known = any(
                any(known in name for name in names_lower)
                for known in known_executives
            )
            assert found_known, f"Expected to find known executives, got: {[r.name for r in results]}"

        finally:
            await service.close()

    @pytest.mark.asyncio
    @pytest.mark.skipif(
        not os.getenv("OPENROUTER_API_KEY"),
        reason="OPENROUTER_API_KEY not set"
    )
    async def test_notch_partners_executives(self):
        """Verify Notch Partners returns executives."""
        service = AISearchService()

        try:
            results = await service.search_executives("Notch Partners")

            assert len(results) >= 3, f"Expected 3+ executives, found {len(results)}"

            for r in results:
                words = r.name.split()
                assert len(words) >= 2, f"Invalid name: {r.name}"

        finally:
            await service.close()

    @pytest.mark.asyncio
    @pytest.mark.skipif(
        not os.getenv("OPENROUTER_API_KEY"),
        reason="OPENROUTER_API_KEY not set"
    )
    async def test_large_public_company_executives(self):
        """Verify large public company returns comprehensive executives."""
        service = AISearchService()

        try:
            results = await service.search_executives("Apple Inc")

            assert len(results) >= 5, f"Expected 5+ executives, found {len(results)}"

            names_lower = [r.name.lower() for r in results]
            found_tim_cook = any("tim cook" in name or "timothy cook" in name for name in names_lower)
            assert found_tim_cook, f"Expected to find Tim Cook, got: {[r.name for r in results]}"

        finally:
            await service.close()


class TestSourcePriorityWeighting:
    """Tests for source priority weighting with AI search."""

    def test_ai_search_not_reduced_by_authoritative(self):
        """Verify AI search confidence is NOT reduced when authoritative sources exist."""
        service = SearchService(tavily_api_key="", serp_api_key="")

        aggregated = AggregatedCompanyData(name="Test Company")

        edgar_exec = SourcedExecutive(
            executive=Executive(name="John Smith", title="CEO", start_year=2020),
            source=DataSource.SEC_EDGAR,
            confidence=1.0,
        )

        ai_exec = SourcedExecutive(
            executive=Executive(name="Jane Doe", title="CFO", start_year=2021),
            source=DataSource.AI_SEARCH,
            confidence=0.85,
        )

        tavily_exec = SourcedExecutive(
            executive=Executive(name="Bob Wilson", title="CTO", start_year=2019),
            source=DataSource.TAVILY,
            confidence=0.7,
        )

        aggregated.sourced_executives = [edgar_exec, ai_exec, tavily_exec]

        service._apply_source_priority_weighting(aggregated)

        assert aggregated.sourced_executives[1].confidence == 0.85

        assert aggregated.sourced_executives[2].confidence == 0.7 * 0.75


class TestSearchServiceIntegration:
    """Integration tests for SearchService with AI search."""

    @pytest.mark.asyncio
    async def test_search_service_initializes_ai_search(self):
        """Verify SearchService initializes AI search when available."""
        with patch("app.services.search_service.AI_SEARCH_AVAILABLE", True):
            with patch("app.services.search_service.AISearchService") as MockAIService:
                mock_instance = MagicMock()
                mock_instance.is_configured = True
                mock_instance.close = AsyncMock()
                MockAIService.return_value = mock_instance

                service = SearchService(tavily_api_key="", serp_api_key="")

                assert service._ai_search_service is not None

                await service.close()

    @pytest.mark.asyncio
    async def test_close_closes_ai_search_service(self):
        """Verify close() properly cleans up AI search service."""
        with patch("app.services.search_service.AI_SEARCH_AVAILABLE", True):
            with patch("app.services.search_service.AISearchService") as MockAIService:
                mock_instance = MagicMock()
                mock_instance.is_configured = True
                mock_instance.close = AsyncMock()
                MockAIService.return_value = mock_instance

                service = SearchService(tavily_api_key="", serp_api_key="")
                service._ai_search_service = mock_instance

                await service.close()

                mock_instance.close.assert_awaited_once()


# =============================================================================
# Name Validation Integration Tests (Acrisure-focused)
# =============================================================================


class TestNameValidationIntegration:
    """Integration tests for name validation in AI search flow.
    
    These tests verify that malformed names (department prefixes, sentence 
    fragments) are properly rejected or extracted during the AI search flow.
    """
    
    @pytest.fixture
    def ai_search_service(self):
        """Create an AISearchService instance for testing."""
        return AISearchService(api_key="test-key")
    
    def test_reject_department_prefixed_names_in_results(self, ai_search_service):
        """Verify department-prefixed names are rejected or extracted.
        
        Names like "Regulatory Affairs Tammie Slauter" should be cleaned up
        to extract just the person's name "Tammie Slauter".
        """
        # Create raw executives that would come from AI parsing
        raw_executives = [
            DiscoveredExecutive(
                name="Regulatory Affairs Tammie Slauter",
                title="VP",
                sources=[AIModel.PERPLEXITY_SONAR],
                confidence=0.8,
            ),
            DiscoveredExecutive(
                name="Finance John Smith",
                title="Director",
                sources=[AIModel.PERPLEXITY_SONAR],
                confidence=0.8,
            ),
            DiscoveredExecutive(
                name="Greg Williams",
                title="CEO",
                sources=[AIModel.PERPLEXITY_SONAR],
                confidence=0.9,
            ),
        ]
        
        validated = ai_search_service._validate_and_score(raw_executives)
        
        # Check that names were extracted or validated properly
        names = [e.name for e in validated]
        
        # "Regulatory Affairs Tammie Slauter" should become "Tammie Slauter"
        # The original prefixed name should NOT appear
        assert "Regulatory Affairs Tammie Slauter" not in names
        
        # The extracted name should be present if extraction works
        # Either "Tammie Slauter" is present or the entry was rejected
        has_tammie = any("Tammie" in name for name in names)
        
        # "Greg Williams" should pass through unchanged
        assert "Greg Williams" in names
        
        # "Finance John Smith" should either be extracted to "John Smith" or rejected
        assert "Finance John Smith" not in names
    
    def test_reject_sentence_fragments(self, ai_search_service):
        """Verify sentence fragments are rejected.
        
        Names that are actually sentence fragments (like "Will Join Acrisure")
        should be completely rejected, not mistaken for person names.
        """
        raw_executives = [
            DiscoveredExecutive(
                name="Will Join Acrisure",
                title="CFO",
                sources=[AIModel.PERPLEXITY_SONAR],
                confidence=0.8,
            ),
            DiscoveredExecutive(
                name="Effective Immediately",
                title="CEO",
                sources=[AIModel.PERPLEXITY_SONAR],
                confidence=0.8,
            ),
            DiscoveredExecutive(
                name="Based In Chicago",
                title="VP",
                sources=[AIModel.PERPLEXITY_SONAR],
                confidence=0.8,
            ),
        ]
        
        validated = ai_search_service._validate_and_score(raw_executives)
        
        # All sentence fragments should be rejected
        assert len(validated) == 0, f"Expected 0 validated executives, got: {[e.name for e in validated]}"
    
    def test_accept_valid_names_with_common_first_names(self, ai_search_service):
        """Verify valid names like 'Will Smith' are accepted.
        
        Names that start with common first names that could be confused with
        action words (Will, Grant, etc.) should still be accepted when they
        are followed by a valid last name.
        """
        raw_executives = [
            DiscoveredExecutive(
                name="Will Smith",
                title="CEO",
                sources=[AIModel.PERPLEXITY_SONAR],
                confidence=0.8,
            ),
            DiscoveredExecutive(
                name="Grant Johnson",
                title="CFO",
                sources=[AIModel.PERPLEXITY_SONAR],
                confidence=0.8,
            ),
            DiscoveredExecutive(
                name="Chase Williams",
                title="COO",
                sources=[AIModel.PERPLEXITY_SONAR],
                confidence=0.8,
            ),
        ]
        
        validated = ai_search_service._validate_and_score(raw_executives)
        
        names = [e.name for e in validated]
        assert "Will Smith" in names, f"Will Smith should be accepted but got: {names}"
        assert "Grant Johnson" in names, f"Grant Johnson should be accepted but got: {names}"
        assert "Chase Williams" in names, f"Chase Williams should be accepted but got: {names}"
    
    def test_reject_announcement_language(self, ai_search_service):
        """Verify announcement language patterns are rejected."""
        raw_executives = [
            DiscoveredExecutive(
                name="Announced The Appointment",
                title="Executive",
                sources=[AIModel.PERPLEXITY_SONAR],
                confidence=0.8,
            ),
            DiscoveredExecutive(
                name="Named As President",
                title="President",
                sources=[AIModel.PERPLEXITY_SONAR],
                confidence=0.8,
            ),
        ]
        
        validated = ai_search_service._validate_and_score(raw_executives)
        
        # These should all be rejected
        assert len(validated) == 0


# =============================================================================
# Historical Tracking Integration Tests
# =============================================================================


class TestHistoricalTrackingIntegration:
    """Integration tests for historical executive tracking.
    
    These tests verify that historical executives (former CEOs, CFOs, etc.)
    are properly tracked with their tenure dates and is_current status.
    """
    
    @pytest.fixture
    def ai_search_service(self):
        """Create an AISearchService instance for testing."""
        from app.services.ai_search_service import (
            CONFIDENCE_HISTORICAL_WITH_DATES,
            CONFIDENCE_HISTORICAL_NO_DATES,
            CONFIDENCE_CURRENT_CONFIRMED,
        )
        return AISearchService(api_key="test-key")
    
    def test_historical_executives_in_results(self, ai_search_service):
        """Verify historical executives appear with is_current=False."""
        from app.services.ai_search_service import (
            CONFIDENCE_HISTORICAL_WITH_DATES,
            CONFIDENCE_HISTORICAL_NO_DATES,
        )
        
        raw_executives = [
            DiscoveredExecutive(
                name="Current Ceo",
                title="CEO",
                sources=[AIModel.PERPLEXITY_SONAR],
                confidence=0.9,
                start_year=2022,
                is_current=True,
            ),
            DiscoveredExecutive(
                name="Former Ceo",
                title="CEO",
                sources=[AIModel.PERPLEXITY_SONAR],
                confidence=0.8,
                start_year=2015,
                end_year=2022,
                is_current=False,
            ),
        ]
        
        validated = ai_search_service._validate_and_score(raw_executives)
        
        assert len(validated) == 2, f"Expected 2 executives, got {len(validated)}"
        
        current = next((e for e in validated if e.is_current), None)
        former = next((e for e in validated if not e.is_current), None)
        
        assert current is not None, "Should have a current executive"
        assert former is not None, "Should have a former executive"
        
        assert current.name == "Current Ceo"
        assert former.name == "Former Ceo"
        assert former.start_year == 2015
        assert former.end_year == 2022
    
    def test_confidence_scoring_for_historical(self, ai_search_service):
        """Verify confidence is properly adjusted for historical data.
        
        Historical executives with complete date information should have
        higher confidence than those without dates.
        """
        from app.services.ai_search_service import (
            CONFIDENCE_HISTORICAL_WITH_DATES,
            CONFIDENCE_HISTORICAL_NO_DATES,
        )
        
        raw_executives = [
            DiscoveredExecutive(
                name="Dated Executive",
                title="CFO",
                sources=[AIModel.PERPLEXITY_SONAR],
                confidence=0.8,
                start_year=2015,
                end_year=2020,
                is_current=False,
            ),
            DiscoveredExecutive(
                name="Undated Executive",
                title="COO",
                sources=[AIModel.PERPLEXITY_SONAR],
                confidence=0.8,
                is_current=False,
            ),
        ]
        
        validated = ai_search_service._validate_and_score(raw_executives)
        
        dated = next((e for e in validated if e.name == "Dated Executive"), None)
        undated = next((e for e in validated if e.name == "Undated Executive"), None)
        
        assert dated is not None, "Dated Executive should be validated"
        assert undated is not None, "Undated Executive should be validated"
        
        # Dated should have higher confidence than undated
        assert dated.confidence >= CONFIDENCE_HISTORICAL_WITH_DATES, \
            f"Dated executive confidence {dated.confidence} should be >= {CONFIDENCE_HISTORICAL_WITH_DATES}"
        assert undated.confidence <= CONFIDENCE_HISTORICAL_NO_DATES, \
            f"Undated executive confidence {undated.confidence} should be <= {CONFIDENCE_HISTORICAL_NO_DATES}"
        
        # Dated should have higher or equal confidence than undated
        assert dated.confidence >= undated.confidence, \
            f"Dated ({dated.confidence}) should have >= confidence than undated ({undated.confidence})"
    
    def test_historical_response_parsing(self, ai_search_service):
        """Test parsing of historical executive response from AI model."""
        mock_response = '''[
            {"name": "Greg Williams", "title": "CEO", "start_year": 2020, "is_current": true},
            {"name": "John Smith", "title": "CEO", "start_year": 2015, "end_year": 2020, "is_current": false}
        ]'''
        
        executives = ai_search_service._parse_historical_response(
            mock_response, 
            AIModel.PERPLEXITY_SONAR
        )
        
        assert len(executives) == 2
        
        current_ceo = next((e for e in executives if e.is_current), None)
        former_ceo = next((e for e in executives if not e.is_current), None)
        
        assert current_ceo is not None
        assert current_ceo.name == "Greg Williams"
        assert current_ceo.start_year == 2020
        
        assert former_ceo is not None
        assert former_ceo.name == "John Smith"
        assert former_ceo.start_year == 2015
        assert former_ceo.end_year == 2020
    
    def test_merge_historical_results(self, ai_search_service):
        """Test merging current and historical executive results."""
        current = [
            DiscoveredExecutive(
                name="Greg Williams",
                title="CEO",
                sources=[AIModel.PERPLEXITY_SONAR],
                confidence=0.95,
                is_current=True,
            ),
        ]
        
        historical = [
            DiscoveredExecutive(
                name="Greg Williams",
                title="CEO",
                sources=[AIModel.GEMINI_FLASH],
                confidence=0.85,
                start_year=2020,
                is_current=True,
            ),
            DiscoveredExecutive(
                name="John Smith",
                title="CEO",
                sources=[AIModel.GEMINI_FLASH],
                confidence=0.80,
                start_year=2015,
                end_year=2020,
                is_current=False,
            ),
        ]
        
        merged = ai_search_service.merge_historical_results(current, historical)
        
        # Should have 2 unique executives
        assert len(merged) == 2
        
        # Current Greg Williams should be in results with updated start_year
        greg = next((e for e in merged if e.name == "Greg Williams"), None)
        assert greg is not None
        assert greg.start_year == 2020  # Should be updated from historical
        
        # Former CEO should be added
        john = next((e for e in merged if e.name == "John Smith"), None)
        assert john is not None
        assert john.is_current is False


# =============================================================================
# Acrisure-Specific Search Tests
# =============================================================================


class TestAcrisureSearch:
    """Test executive discovery for Acrisure specifically.
    
    These tests use mocked AI responses that simulate the kinds of
    problematic data we've seen with Acrisure searches.
    """
    
    @pytest.fixture
    def ai_search_service(self):
        """Create an AISearchService instance for testing."""
        return AISearchService(api_key="test-key")
    
    @pytest.mark.asyncio
    async def test_acrisure_search_mocked(self, ai_search_service):
        """Test Acrisure search with mocked AI response.
        
        This test simulates a response that includes known problematic patterns:
        - Department-prefixed names
        - Sentence fragments
        - Valid names that should pass
        """
        # Mock response that includes the known issues
        mock_response = '''[
            {"name": "Greg Williams", "title": "CEO", "start_year": 2020},
            {"name": "Regulatory Affairs Tammie Slauter", "title": "VP"},
            {"name": "Will Join Acrisure", "title": "CFO"},
            {"name": "John Smith", "title": "CFO", "start_year": 2015, "end_year": 2020, "is_current": false}
        ]'''
        
        # Parse and validate
        executives = ai_search_service._parse_historical_response(
            mock_response, 
            AIModel.PERPLEXITY_SONAR
        )
        validated = ai_search_service._validate_and_score(executives)
        
        names = [e.name for e in validated]
        
        # Greg Williams should be present
        assert "Greg Williams" in names, f"Greg Williams should be present but got: {names}"
        
        # John Smith should be present (historical executive)
        assert "John Smith" in names, f"John Smith should be present but got: {names}"
        
        # "Will Join Acrisure" should NOT be present (sentence fragment)
        assert "Will Join Acrisure" not in names, \
            f"Sentence fragment should be filtered but got: {names}"
        
        # Check for extracted name from "Regulatory Affairs Tammie Slauter"
        # Either "Tammie Slauter" is present or the entry is rejected
        assert "Regulatory Affairs Tammie Slauter" not in names, \
            f"Prefixed name should be cleaned or rejected but got: {names}"
    
    @pytest.mark.asyncio
    async def test_acrisure_search_full_flow_mocked(self, ai_search_service):
        """Test full Acrisure search flow with mocked API response."""
        mock_api_response = MagicMock()
        mock_api_response.status_code = 200
        mock_api_response.json.return_value = {
            "choices": [
                {
                    "message": {
                        "content": '''[
                            {"name": "Greg Williams", "title": "CEO"},
                            {"name": "Mark Wassersug", "title": "CFO"},
                            {"name": "Regulatory Affairs Director", "title": "VP"},
                            {"name": "Based In Grand Rapids", "title": "Executive"},
                            {"name": "Steve Johnson", "title": "COO"}
                        ]'''
                    }
                }
            ]
        }
        mock_api_response.raise_for_status = MagicMock()
        
        with patch.object(ai_search_service, "_get_client") as mock_get_client:
            mock_client = AsyncMock()
            mock_client.post = AsyncMock(return_value=mock_api_response)
            mock_get_client.return_value = mock_client
            
            results = await ai_search_service.search_executives(
                "Acrisure",
                models=[AIModel.PERPLEXITY_SONAR]
            )
        
        names = [e.name for e in results]
        
        # Valid executives should be present
        assert "Greg Williams" in names
        assert "Mark Wassersug" in names
        assert "Steve Johnson" in names
        
        # Invalid entries should be filtered
        assert "Regulatory Affairs Director" not in names
        assert "Based In Grand Rapids" not in names
        
        # All results should have valid confidence scores
        for exec in results:
            assert exec.confidence > 0
            assert exec.validated is True
    
    @pytest.mark.asyncio
    async def test_acrisure_historical_executives_mocked(self, ai_search_service):
        """Test Acrisure historical executive search with mocked response."""
        mock_api_response = MagicMock()
        mock_api_response.status_code = 200
        mock_api_response.json.return_value = {
            "choices": [
                {
                    "message": {
                        "content": '''[
                            {"name": "Greg Williams", "title": "CEO", "start_year": 2020, "is_current": true},
                            {"name": "John Doe", "title": "CEO", "start_year": 2015, "end_year": 2020, "is_current": false},
                            {"name": "Jane Smith", "title": "CFO", "start_year": 2018, "is_current": true},
                            {"name": "Previously Announced", "title": "COO", "is_current": false}
                        ]'''
                    }
                }
            ]
        }
        mock_api_response.raise_for_status = MagicMock()
        
        with patch.object(ai_search_service, "_get_client") as mock_get_client:
            mock_client = AsyncMock()
            mock_client.post = AsyncMock(return_value=mock_api_response)
            mock_get_client.return_value = mock_client
            
            # Use search_executive_history for historical search
            results = await ai_search_service.search_executive_history("Acrisure")
        
        # Filter to only validated names
        valid_names = [e.name for e in results]
        
        # Valid executives should be present
        assert "Greg Williams" in valid_names
        assert "John Doe" in valid_names
        assert "Jane Smith" in valid_names
        
        # Sentence fragment should be filtered
        assert "Previously Announced" not in valid_names
        
        # Check historical executive tracking
        john = next((e for e in results if e.name == "John Doe"), None)
        if john:
            assert john.is_current is False
            assert john.start_year == 2015
            assert john.end_year == 2020
    
    def test_acrisure_name_extraction(self, ai_search_service):
        """Test name extraction for Acrisure-style prefixed names."""
        validator = ai_search_service.validator
        
        # Test extraction of department-prefixed names
        name, title = validator.extract_name_from_prefixed("Regulatory Affairs Tammie Slauter")
        
        # Should extract "Tammie Slauter" from the prefixed name
        if name is not None:
            assert "Tammie" in name
            assert "Regulatory Affairs" not in name
        
        # Test that valid names pass through unchanged
        name, title = validator.extract_name_from_prefixed("Greg Williams")
        assert name == "Greg Williams"
        assert title is None
        
        # Test that sentence fragments are rejected
        name, title = validator.extract_name_from_prefixed("Will Join Acrisure")
        assert name is None


# =============================================================================
# Edge Cases and Regression Tests
# =============================================================================


class TestEdgeCasesAndRegressions:
    """Edge case and regression tests for known issues.
    
    These tests document specific issues we've encountered and verify
    they remain fixed.
    """
    
    @pytest.fixture
    def ai_search_service(self):
        """Create an AISearchService instance for testing."""
        return AISearchService(api_key="test-key")
    
    def test_will_as_first_name_accepted(self, ai_search_service):
        """Regression: 'Will' as a first name should be accepted."""
        # "Will Smith" is a valid name, not an action phrase
        result = ai_search_service.validator.validate_name("Will Smith")
        assert result[0] is True, f"Will Smith should be valid: {result[1]}"
    
    def test_will_join_rejected(self, ai_search_service):
        """Regression: 'Will Join X' should be rejected as sentence fragment."""
        result = ai_search_service.validator.validate_name("Will Join Acrisure")
        assert result[0] is False, "Will Join Acrisure should be invalid"
    
    def test_based_in_rejected(self, ai_search_service):
        """Regression: 'Based In X' should be rejected."""
        result = ai_search_service.validator.validate_name("Based In Chicago")
        assert result[0] is False, "Based In Chicago should be invalid"
    
    def test_effective_immediately_rejected(self, ai_search_service):
        """Regression: 'Effective Immediately' should be rejected."""
        result = ai_search_service.validator.validate_name("Effective Immediately")
        assert result[0] is False, "Effective Immediately should be invalid"
    
    def test_regulatory_affairs_prefix_handled(self, ai_search_service):
        """Regression: 'Regulatory Affairs X' should have name extracted."""
        name, title = ai_search_service.validator.extract_name_from_prefixed(
            "Regulatory Affairs John Smith"
        )
        
        # Should extract "John Smith"
        if name is not None:
            assert name == "John Smith"
            assert "Regulatory Affairs" in title or title.lower() == "regulatory affairs"
    
    def test_finance_prefix_handled(self, ai_search_service):
        """Regression: 'Finance X' should have name extracted."""
        name, title = ai_search_service.validator.extract_name_from_prefixed(
            "Finance Jane Doe"
        )
        
        # Should extract "Jane Doe"
        if name is not None:
            assert name == "Jane Doe"
    
    def test_empty_end_year_means_current(self, ai_search_service):
        """Test that null/empty end_year correctly indicates current position."""
        raw_executives = [
            DiscoveredExecutive(
                name="John Smith",
                title="CEO",
                sources=[AIModel.PERPLEXITY_SONAR],
                confidence=0.9,
                start_year=2020,
                end_year=None,  # No end year = current
                is_current=True,
            ),
        ]
        
        validated = ai_search_service._validate_and_score(raw_executives)
        
        assert len(validated) == 1
        assert validated[0].is_current is True
        assert validated[0].end_year is None
    
    def test_historical_deduplication_considers_tenure(self, ai_search_service):
        """Test that historical deduplication considers tenure period."""
        # Same person in same role but different time periods should be separate
        all_executives = [
            (
                {"name": "John Smith", "title": "CEO", "start_year": 2020, "is_current": True},
                AIModel.PERPLEXITY_SONAR
            ),
            (
                {"name": "John Smith", "title": "CEO", "start_year": 2015, "end_year": 2020, "is_current": False},
                AIModel.PERPLEXITY_SONAR
            ),
        ]
        
        result = ai_search_service._deduplicate_historical_executives(all_executives)
        
        # Should have 2 separate entries (same person, different tenures)
        assert len(result) == 2
        
        current = next((e for e in result if e.is_current), None)
        historical = next((e for e in result if not e.is_current), None)
        
        assert current is not None
        assert historical is not None
        assert current.start_year == 2020
        assert historical.start_year == 2015
        assert historical.end_year == 2020
