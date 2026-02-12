"""AI-powered executive discovery service using OpenRouter.

This service queries multiple AI models in parallel to discover company executives,
using models with real-time web search capabilities (Perplexity, Gemini).

Models are queried in parallel and results are:
1. Validated using ExecutiveValidator
2. Deduplicated using fuzzy matching
3. Confidence scored based on multi-model agreement
"""

from __future__ import annotations

import asyncio
import json
import logging
import os
import re
from dataclasses import dataclass, field
from enum import Enum
from typing import Any

import httpx
from rapidfuzz import fuzz

from app.models import Executive
from app.services.validation_service import ExecutiveValidator

logger = logging.getLogger(__name__)

# OpenRouter configuration
OPENROUTER_API_KEY = os.getenv("OPENROUTER_API_KEY", "")
OPENROUTER_URL = "https://openrouter.ai/api/v1/chat/completions"

# Request timeout per model (seconds)
MODEL_TIMEOUT = 30.0

# Retry configuration
MAX_RETRIES = 2
RETRY_DELAY = 1.0  # seconds


class AIModel(Enum):
    """Available AI models for executive discovery."""

    # Perplexity Sonar models - real-time web search (updated Jan 2025)
    PERPLEXITY_SONAR = "perplexity/sonar"  # Fast, lightweight with citations
    PERPLEXITY_SONAR_PRO = "perplexity/sonar-pro"  # More capable, deeper search

    # Google Gemini - grounded search
    GEMINI_FLASH = "google/gemini-2.0-flash-001"
    GEMINI_PRO = "google/gemini-pro-1.5"


# Model priority order (primary to fallback)
MODEL_PRIORITY: list[AIModel] = [
    AIModel.PERPLEXITY_SONAR,      # Primary - fast real-time web search with citations
    AIModel.GEMINI_FLASH,          # Secondary - fast Google grounded search
    AIModel.PERPLEXITY_SONAR_PRO,  # Tertiary - deeper Perplexity search
    AIModel.GEMINI_PRO,            # Fallback - reliable but slower
]

# Models to query in parallel (first N from priority list)
PARALLEL_QUERY_COUNT = 3

# Confidence scoring thresholds
CONFIDENCE_MULTI_MODEL = 0.95      # Name appears in 2+ model responses
CONFIDENCE_SINGLE_VALIDATED = 0.80  # Name in 1 model + passes validation
CONFIDENCE_SINGLE_ONLY = 0.70       # Name in 1 model only

# Confidence adjustments for historical data
CONFIDENCE_CURRENT_CONFIRMED = 0.95  # Current exec confirmed by multiple sources
CONFIDENCE_HISTORICAL_WITH_DATES = 0.85  # Historical exec with specific dates
CONFIDENCE_HISTORICAL_NO_DATES = 0.70  # Historical exec without dates

# Deduplication thresholds
NAME_SIMILARITY_THRESHOLD = 85  # Fuzzy match threshold for name deduplication


# Historical executive search prompt
HISTORICAL_PROMPT = """Find the executive leadership history at {company_name} for the past 10 years.

For each C-level and VP+ position, provide:
1. The CURRENT holder with their start year
2. ALL previous holders from the past 10 years with their tenure dates

Return ONLY a JSON array with this exact format:
[
  {{"name": "Full Name", "title": "Exact Title", "start_year": 2020, "end_year": null, "is_current": true}},
  {{"name": "Previous Person", "title": "CEO", "start_year": 2015, "end_year": 2020, "is_current": false}}
]

Focus on these roles: CEO, CFO, COO, CTO, President, SVP, EVP, Chief Officers.

IMPORTANT:
- Include transitions - if CEO changed in 2022, show both the current and former CEO
- start_year should be when they started in that specific role
- end_year should be null for current position holders
- is_current should be true only for current position holders
- Only include verified information from reliable sources

Only output the JSON array, no explanation."""


@dataclass
class ModelResponse:
    """Response from a single AI model query."""

    model: AIModel
    executives: list[dict[str, Any]]
    success: bool
    error: str | None = None
    raw_response: str | None = None


@dataclass
class DiscoveredExecutive:
    """Executive discovered through AI search with source tracking."""

    name: str
    title: str
    sources: list[AIModel] = field(default_factory=list)
    confidence: float = 0.0
    validated: bool = False
    validation_reason: str = ""
    # Tenure fields for historical executive tracking
    source_url: str | None = None
    linkedin_url: str | None = None
    start_year: int | None = None
    end_year: int | None = None  # None means current position
    is_current: bool = True

    def to_executive(self) -> Executive:
        """Convert to Executive model.

        Note: The Executive model receives start_year and end_year if available.
        """
        return Executive(
            name=self.name,
            title=self.title,
            start_year=self.start_year,
            end_year=self.end_year,
            linkedin_url=self.linkedin_url,
            photo_url=None,
            employment_history=[],
        )


class AISearchService:
    """Service for AI-powered executive discovery using OpenRouter.

    Queries multiple AI models (Perplexity, Gemini) in parallel to find
    company executives, with validation and deduplication.

    Attributes:
        api_key: OpenRouter API key for authentication.
        validator: ExecutiveValidator for name validation.
    """

    def __init__(self, api_key: str | None = None) -> None:
        """Initialize the AI search service.

        Args:
            api_key: OpenRouter API key. If not provided, uses OPENROUTER_API_KEY env var.
        """
        self.api_key = api_key or OPENROUTER_API_KEY
        self.validator = ExecutiveValidator()
        self._http_client: httpx.AsyncClient | None = None

    @property
    def is_configured(self) -> bool:
        """Check if the API key is configured."""
        return bool(self.api_key)

    async def _get_client(self) -> httpx.AsyncClient:
        """Get or create HTTP client with appropriate timeout."""
        if self._http_client is None or self._http_client.is_closed:
            self._http_client = httpx.AsyncClient(timeout=MODEL_TIMEOUT)
        return self._http_client

    async def close(self) -> None:
        """Close HTTP client and release resources."""
        if self._http_client and not self._http_client.is_closed:
            await self._http_client.aclose()
            self._http_client = None

    def _build_prompt(self, company_name: str) -> str:
        """Build the executive discovery prompt.

        Args:
            company_name: Name of the company to search.

        Returns:
            Prompt string for the AI model.
        """
        # Sanitize company name to prevent prompt injection
        safe_company_name = self._sanitize_input(company_name, max_length=200)

        return f"""Find the current executive leadership team at {safe_company_name}.

Return ONLY a JSON array with this exact format:
[
  {{"name": "Full Name", "title": "Exact Title"}},
  ...
]

Include: CEO, CFO, COO, CTO, President, VP-level and above.
Only include CURRENT executives, not former.
Do not include any explanation, just the JSON array."""

    def _build_historical_prompt(self, company_name: str) -> str:
        """Build the prompt for historical executive search.

        Args:
            company_name: Name of the company to search.

        Returns:
            Prompt string for historical executive discovery.
        """
        # Sanitize company name to prevent prompt injection
        safe_company_name = self._sanitize_input(company_name, max_length=200)
        return HISTORICAL_PROMPT.format(company_name=safe_company_name)

    def _sanitize_input(self, text: str, max_length: int = 200) -> str:
        """Sanitize user input to prevent prompt injection.

        Args:
            text: Input text to sanitize.
            max_length: Maximum allowed length.

        Returns:
            Sanitized text safe for prompt inclusion.
        """
        if not text:
            return ""

        # Truncate to max length
        text = text[:max_length]

        # Remove common prompt injection patterns
        injection_patterns = [
            r"ignore\s+(previous|all|above)\s+instructions",
            r"disregard\s+(previous|all|above)",
            r"forget\s+(everything|all|previous)",
            r"new\s+instructions?:",
            r"system\s*:",
            r"assistant\s*:",
            r"human\s*:",
            r"\[INST\]",
            r"\[/INST\]",
            r"<\|im_start\|>",
            r"<\|im_end\|>",
        ]

        for pattern in injection_patterns:
            text = re.sub(pattern, "[REDACTED]", text, flags=re.IGNORECASE)

        return text

    async def _query_model(
        self,
        model: AIModel,
        prompt: str,
        retry_count: int = 0,
    ) -> ModelResponse:
        """Query a single AI model for executives.

        Args:
            model: The AI model to query.
            prompt: The prompt to send.
            retry_count: Current retry attempt number.

        Returns:
            ModelResponse with parsed executives or error.
        """
        try:
            client = await self._get_client()

            response = await client.post(
                OPENROUTER_URL,
                headers={
                    "Authorization": f"Bearer {self.api_key}",
                    "Content-Type": "application/json",
                    "HTTP-Referer": "https://notch-company-mapping.local",
                    "X-Title": "Notch Company Mapping",
                },
                json={
                    "model": model.value,
                    "messages": [{"role": "user", "content": prompt}],
                    "temperature": 0,
                    "max_tokens": 2000,
                },
            )

            # Handle rate limiting with retry
            if response.status_code == 429 and retry_count < MAX_RETRIES:
                logger.warning(
                    f"Rate limited on {model.value}, retrying in {RETRY_DELAY}s..."
                )
                await asyncio.sleep(RETRY_DELAY * (retry_count + 1))
                return await self._query_model(model, prompt, retry_count + 1)

            response.raise_for_status()

            data = response.json()
            content_text = data["choices"][0]["message"]["content"]

            # Parse JSON from response
            executives = self._parse_json_response(content_text)

            logger.info(
                f"Model {model.value} returned {len(executives)} executives"
            )

            return ModelResponse(
                model=model,
                executives=executives,
                success=True,
                raw_response=content_text,
            )

        except httpx.HTTPStatusError as e:
            error_msg = f"HTTP {e.response.status_code}: {e.response.text[:200]}"
            logger.error(f"Model {model.value} failed: {error_msg}")
            return ModelResponse(
                model=model,
                executives=[],
                success=False,
                error=error_msg,
            )
        except httpx.TimeoutException:
            error_msg = f"Timeout after {MODEL_TIMEOUT}s"
            logger.error(f"Model {model.value} failed: {error_msg}")
            return ModelResponse(
                model=model,
                executives=[],
                success=False,
                error=error_msg,
            )
        except json.JSONDecodeError as e:
            error_msg = f"JSON parse error: {str(e)}"
            logger.error(f"Model {model.value} failed: {error_msg}")
            return ModelResponse(
                model=model,
                executives=[],
                success=False,
                error=error_msg,
            )
        except Exception as e:
            error_msg = f"Unexpected error: {str(e)}"
            logger.error(f"Model {model.value} failed: {error_msg}")
            return ModelResponse(
                model=model,
                executives=[],
                success=False,
                error=error_msg,
            )

    def _parse_json_response(self, text: str) -> list[dict[str, Any]]:
        """Parse JSON array from model response.

        Handles markdown code blocks and extracts the JSON array.

        Args:
            text: Raw response text from the model.

        Returns:
            List of executive dictionaries.

        Raises:
            json.JSONDecodeError: If no valid JSON found.
        """
        text = text.strip()

        # Remove markdown code blocks if present
        if "```" in text:
            # Extract content between code blocks
            json_match = re.search(r"```(?:json)?\s*([\s\S]*?)```", text)
            if json_match:
                text = json_match.group(1).strip()
            else:
                # Remove just the markers
                text = re.sub(r"```(?:json)?", "", text).strip()

        # Find the JSON array in the text
        # Look for [ ... ] pattern
        array_match = re.search(r"\[[\s\S]*\]", text)
        if array_match:
            text = array_match.group(0)

        # Parse the JSON
        data = json.loads(text)

        # Ensure it's a list
        if not isinstance(data, list):
            if isinstance(data, dict) and "executives" in data:
                data = data["executives"]
            else:
                raise json.JSONDecodeError("Response is not a list", text, 0)

        return data

    def _parse_historical_response(
        self, response: str, model: AIModel
    ) -> list[DiscoveredExecutive]:
        """Parse the historical search response into DiscoveredExecutive objects.

        Args:
            response: The raw response from the AI model.
            model: The AI model that generated the response.

        Returns:
            List of discovered executives with tenure information.
        """
        executives: list[DiscoveredExecutive] = []

        try:
            # Try to extract JSON from the response
            data = self._parse_json_response(response)

            if not isinstance(data, list):
                logger.warning(f"Historical response is not a list: {type(data)}")
                return executives

            for item in data:
                if not isinstance(item, dict):
                    continue

                name = item.get("name", "").strip()
                title = item.get("title", "").strip()

                if not name or not title:
                    continue

                # Extract tenure information
                start_year = item.get("start_year")
                end_year = item.get("end_year")
                is_current = item.get("is_current", end_year is None)

                # Validate and convert years
                if start_year is not None:
                    try:
                        start_year = int(start_year)
                        if start_year < 1900 or start_year > 2030:
                            start_year = None
                    except (ValueError, TypeError):
                        start_year = None

                if end_year is not None:
                    try:
                        end_year = int(end_year)
                        if end_year < 1900 or end_year > 2030:
                            end_year = None
                    except (ValueError, TypeError):
                        end_year = None

                # Determine confidence based on data quality
                if is_current:
                    base_confidence = CONFIDENCE_CURRENT_CONFIRMED
                elif start_year and end_year:
                    base_confidence = CONFIDENCE_HISTORICAL_WITH_DATES
                else:
                    base_confidence = CONFIDENCE_HISTORICAL_NO_DATES

                exec_obj = DiscoveredExecutive(
                    name=name,
                    title=title,
                    sources=[model],
                    confidence=base_confidence,
                    start_year=start_year,
                    end_year=end_year,
                    is_current=is_current,
                )
                executives.append(exec_obj)

        except Exception as e:
            logger.error(f"Failed to parse historical response: {e}")

        return executives

    def _normalize_name(self, name: str) -> str:
        """Normalize a name for comparison.

        Args:
            name: Raw name string.

        Returns:
            Normalized name string.
        """
        if not name:
            return ""

        # Strip whitespace and convert to lowercase
        name = name.strip().lower()

        # Remove common prefixes
        prefixes = ["dr.", "dr ", "mr.", "mr ", "mrs.", "mrs ", "ms.", "ms "]
        for prefix in prefixes:
            if name.startswith(prefix):
                name = name[len(prefix):]

        # Remove common suffixes
        suffixes = [" jr.", " jr", " sr.", " sr", " ii", " iii", " iv", " phd", " md"]
        for suffix in suffixes:
            if name.endswith(suffix):
                name = name[:-len(suffix)]

        return name.strip()

    def _deduplicate_executives(
        self,
        all_executives: list[tuple[dict[str, Any], AIModel]],
    ) -> list[DiscoveredExecutive]:
        """Deduplicate executives from multiple model responses.

        Uses fuzzy matching to identify the same person across responses.

        Args:
            all_executives: List of (executive_dict, source_model) tuples.

        Returns:
            List of deduplicated DiscoveredExecutive objects.
        """
        discovered: list[DiscoveredExecutive] = []

        for exec_data, model in all_executives:
            name = exec_data.get("name", "").strip()
            title = exec_data.get("title", "Executive").strip()

            if not name:
                continue

            # Extract tenure fields from parsed JSON
            start_year = exec_data.get("start_year")
            end_year = exec_data.get("end_year")
            is_current = exec_data.get("is_current", True)  # Default to current if not specified

            normalized_name = self._normalize_name(name)

            # Check if this executive already exists (fuzzy match)
            found_match = False
            for existing in discovered:
                existing_normalized = self._normalize_name(existing.name)
                similarity = fuzz.ratio(normalized_name, existing_normalized)

                if similarity >= NAME_SIMILARITY_THRESHOLD:
                    # Same person - add source and possibly update title
                    if model not in existing.sources:
                        existing.sources.append(model)
                    # Keep the longer/more specific title
                    if len(title) > len(existing.title):
                        existing.title = title
                    # Update tenure info if not already set
                    if start_year and not existing.start_year:
                        existing.start_year = start_year
                    if end_year and not existing.end_year:
                        existing.end_year = end_year
                    # If any source says not current, mark as not current
                    if not is_current:
                        existing.is_current = False
                    found_match = True
                    break

            if not found_match:
                # New executive
                discovered.append(DiscoveredExecutive(
                    name=name,
                    title=title,
                    sources=[model],
                    start_year=start_year,
                    end_year=end_year,
                    is_current=is_current,
                ))

        return discovered

    def _validate_and_score(
        self,
        executives: list[DiscoveredExecutive],
    ) -> list[DiscoveredExecutive]:
        """Validate executives and assign confidence scores.

        Uses name extraction to handle potentially prefixed names (e.g.,
        "Regulatory Affairs John Smith" -> "John Smith" with inferred title).

        Args:
            executives: List of deduplicated executives.

        Returns:
            List of validated executives with confidence scores.
        """
        validated_executives: list[DiscoveredExecutive] = []

        for exec_obj in executives:
            # Try to extract clean name from potentially prefixed text
            extracted_name, inferred_title = self.validator.extract_name_from_prefixed(
                exec_obj.name
            )

            if extracted_name is None:
                # Name could not be extracted or validated
                logger.debug(f"Rejected executive: {exec_obj.name} - extraction failed")
                continue

            # Update the executive with the extracted name
            if extracted_name != exec_obj.name:
                logger.debug(f"Extracted name: {exec_obj.name} -> {extracted_name}")
                exec_obj.name = extracted_name
                # If we inferred a title and exec doesn't have a good one, use it
                if inferred_title and not exec_obj.title:
                    exec_obj.title = inferred_title

            # Now validate the (possibly extracted) name
            is_valid, reason = self.validator.validate_name(exec_obj.name)

            if not is_valid:
                logger.debug(f"Rejected executive: {exec_obj.name} - {reason}")
                continue

            exec_obj.validated = True
            exec_obj.validation_reason = reason or "Valid"

            # Assign base confidence based on source count
            source_count = len(exec_obj.sources)
            if source_count >= 2:
                exec_obj.confidence = CONFIDENCE_MULTI_MODEL
            else:
                exec_obj.confidence = CONFIDENCE_SINGLE_VALIDATED

            # Adjust confidence based on historical data
            if exec_obj.is_current:
                # Current exec with year info gets bonus
                if exec_obj.start_year:
                    exec_obj.confidence = min(
                        exec_obj.confidence * 1.1, CONFIDENCE_CURRENT_CONFIRMED
                    )
            else:
                # Historical exec - adjust confidence based on date completeness
                if exec_obj.start_year and exec_obj.end_year:
                    exec_obj.confidence = CONFIDENCE_HISTORICAL_WITH_DATES
                elif exec_obj.start_year or exec_obj.end_year:
                    exec_obj.confidence = (
                        CONFIDENCE_HISTORICAL_WITH_DATES + CONFIDENCE_HISTORICAL_NO_DATES
                    ) / 2
                else:
                    exec_obj.confidence = CONFIDENCE_HISTORICAL_NO_DATES

            validated_executives.append(exec_obj)

        # Sort by confidence (descending), then by name
        validated_executives.sort(key=lambda e: (-e.confidence, e.name))

        return validated_executives

    async def search_executives(
        self,
        company_name: str,
        models: list[AIModel] | None = None,
    ) -> list[DiscoveredExecutive]:
        """Search for company executives using AI models.

        Queries multiple models in parallel, validates and deduplicates results.

        Args:
            company_name: Name of the company to search.
            models: Optional list of models to query. Defaults to top N from priority list.

        Returns:
            List of validated and deduplicated DiscoveredExecutive objects.
        """
        if not self.is_configured:
            logger.warning("AISearchService: OpenRouter API key not configured")
            return []

        # Use specified models or default to parallel query set
        if models is None:
            models = MODEL_PRIORITY[:PARALLEL_QUERY_COUNT]

        prompt = self._build_prompt(company_name)

        logger.info(
            f"Searching executives for '{company_name}' using {len(models)} models"
        )

        # Query all models in parallel
        tasks = [self._query_model(model, prompt) for model in models]
        responses: list[ModelResponse] = await asyncio.gather(*tasks)

        # Collect all executives with their sources
        all_executives: list[tuple[dict[str, Any], AIModel]] = []
        successful_models = 0

        for response in responses:
            if response.success:
                successful_models += 1
                for exec_data in response.executives:
                    all_executives.append((exec_data, response.model))
            else:
                logger.warning(
                    f"Model {response.model.value} failed: {response.error}"
                )

        if successful_models == 0:
            logger.error("All AI models failed for executive search")
            # Try fallback model if available and not already tried
            fallback = self._get_fallback_model(models)
            if fallback:
                logger.info(f"Trying fallback model: {fallback.value}")
                fallback_response = await self._query_model(fallback, prompt)
                if fallback_response.success:
                    for exec_data in fallback_response.executives:
                        all_executives.append((exec_data, fallback_response.model))

        if not all_executives:
            logger.warning(f"No executives found for '{company_name}'")
            return []

        # Deduplicate across model responses
        deduplicated = self._deduplicate_executives(all_executives)

        logger.info(
            f"Found {len(all_executives)} raw executives, "
            f"deduplicated to {len(deduplicated)}"
        )

        # Validate and score
        validated = self._validate_and_score(deduplicated)

        logger.info(
            f"Validated {len(validated)} executives for '{company_name}'"
        )

        return validated

    def _get_fallback_model(self, tried_models: list[AIModel]) -> AIModel | None:
        """Get a fallback model that hasn't been tried yet.

        Args:
            tried_models: List of models already attempted.

        Returns:
            A fallback model or None if all have been tried.
        """
        for model in MODEL_PRIORITY:
            if model not in tried_models:
                return model
        return None

    async def search_executives_with_fallback(
        self,
        company_name: str,
    ) -> list[DiscoveredExecutive]:
        """Search for executives with automatic fallback on failure.

        Starts with primary models and falls back through the priority list
        until results are found or all models have been tried.

        Args:
            company_name: Name of the company to search.

        Returns:
            List of validated executives.
        """
        # Try primary parallel query first
        results = await self.search_executives(company_name)

        if results:
            return results

        # Try remaining models one by one
        tried_models = set(MODEL_PRIORITY[:PARALLEL_QUERY_COUNT])

        for model in MODEL_PRIORITY[PARALLEL_QUERY_COUNT:]:
            if model in tried_models:
                continue

            logger.info(f"Trying fallback model: {model.value}")
            results = await self.search_executives(company_name, models=[model])

            if results:
                return results

            tried_models.add(model)

        return []

    async def search_executive_history(
        self, company_name: str
    ) -> list[DiscoveredExecutive]:
        """Search for historical executives at a company over the past 10 years.

        Args:
            company_name: The name of the company to search.

        Returns:
            List of discovered executives including historical ones.
        """
        if not self.is_configured:
            logger.warning("AISearchService: OpenRouter API key not configured")
            return []

        prompt = self._build_historical_prompt(company_name)
        models = MODEL_PRIORITY[:PARALLEL_QUERY_COUNT]

        logger.info(
            f"Searching executive history for '{company_name}' using {len(models)} models"
        )

        # Query models for historical data
        tasks = [self._query_model(model, prompt) for model in models]
        responses: list[ModelResponse] = await asyncio.gather(*tasks)

        all_executives: list[DiscoveredExecutive] = []

        for response in responses:
            if isinstance(response, Exception):
                logger.warning(f"Historical search failed: {response}")
                continue

            if not response.success:
                logger.warning(
                    f"Historical search failed for {response.model.value}: {response.error}"
                )
                continue

            if response.raw_response:
                executives = self._parse_historical_response(
                    response.raw_response, response.model
                )
                all_executives.extend(executives)

        if not all_executives:
            logger.warning(f"No historical executives found for '{company_name}'")
            return []

        # Deduplicate - convert to tuple format for existing deduplication logic
        exec_tuples: list[tuple[dict[str, Any], AIModel]] = []
        for exec_obj in all_executives:
            exec_dict = {
                "name": exec_obj.name,
                "title": exec_obj.title,
                "start_year": exec_obj.start_year,
                "end_year": exec_obj.end_year,
                "is_current": exec_obj.is_current,
            }
            exec_tuples.append((exec_dict, exec_obj.sources[0]))

        deduplicated = self._deduplicate_historical_executives(exec_tuples)

        # Validate and score
        validated = self._validate_and_score(deduplicated)

        logger.info(
            f"Found {len(all_executives)} raw historical executives, "
            f"validated to {len(validated)} for '{company_name}'"
        )

        return validated

    def _deduplicate_historical_executives(
        self,
        all_executives: list[tuple[dict[str, Any], AIModel]],
    ) -> list[DiscoveredExecutive]:
        """Deduplicate historical executives from multiple model responses.

        Uses fuzzy matching and considers name+title+tenure for uniqueness.

        Args:
            all_executives: List of (executive_dict, source_model) tuples.

        Returns:
            List of deduplicated DiscoveredExecutive objects.
        """
        discovered: list[DiscoveredExecutive] = []

        for exec_data, model in all_executives:
            name = exec_data.get("name", "").strip()
            title = exec_data.get("title", "Executive").strip()
            start_year = exec_data.get("start_year")
            end_year = exec_data.get("end_year")
            is_current = exec_data.get("is_current", end_year is None)

            if not name:
                continue

            normalized_name = self._normalize_name(name)

            # Check if this executive already exists (fuzzy match on name + title)
            found_match = False
            for existing in discovered:
                existing_normalized = self._normalize_name(existing.name)
                name_similarity = fuzz.ratio(normalized_name, existing_normalized)

                # Also check title similarity for historical records
                title_similarity = fuzz.ratio(
                    title.lower(), existing.title.lower()
                )

                # Match if same person in same role (same tenure period)
                if (
                    name_similarity >= NAME_SIMILARITY_THRESHOLD
                    and title_similarity >= 70
                    and existing.start_year == start_year
                ):
                    # Same person, same role, same period - add source
                    if model not in existing.sources:
                        existing.sources.append(model)
                    found_match = True
                    break

            if not found_match:
                # Determine confidence based on data quality
                if is_current:
                    base_confidence = CONFIDENCE_CURRENT_CONFIRMED
                elif start_year and end_year:
                    base_confidence = CONFIDENCE_HISTORICAL_WITH_DATES
                else:
                    base_confidence = CONFIDENCE_HISTORICAL_NO_DATES

                discovered.append(DiscoveredExecutive(
                    name=name,
                    title=title,
                    sources=[model],
                    confidence=base_confidence,
                    start_year=start_year,
                    end_year=end_year,
                    is_current=is_current,
                ))

        return discovered

    def merge_historical_results(
        self,
        current: list[DiscoveredExecutive],
        historical: list[DiscoveredExecutive],
    ) -> list[DiscoveredExecutive]:
        """Merge current executive results with historical results.

        Deduplicates by name+title, preferring current data when available.

        Args:
            current: List of current executives from standard search.
            historical: List of historical executives.

        Returns:
            Combined list with duplicates removed.
        """
        # Create a map of (normalized_name, normalized_title) -> executive
        seen: dict[tuple[str, str], DiscoveredExecutive] = {}

        # Add current executives first (they have priority)
        for exec_obj in current:
            key = (exec_obj.name.lower().strip(), exec_obj.title.lower().strip())
            seen[key] = exec_obj

        # Add historical executives if not already present
        for exec_obj in historical:
            key = (exec_obj.name.lower().strip(), exec_obj.title.lower().strip())
            if key not in seen:
                seen[key] = exec_obj
            else:
                # Update existing entry with historical data if it has dates
                existing = seen[key]
                if exec_obj.start_year and not existing.start_year:
                    existing.start_year = exec_obj.start_year
                if exec_obj.end_year and not existing.end_year:
                    existing.end_year = exec_obj.end_year

        return list(seen.values())


# Import DataSource and SourcedExecutive from search_service for integration
# This allows the AI search results to be used with the existing aggregation system
try:
    from app.services.search_service import DataSource, SourcedExecutive

    # Define a new data source for AI search
    # Note: This would need to be added to the DataSource enum in search_service.py
    AI_SEARCH_SOURCE = DataSource.TAVILY  # Temporarily use TAVILY until AI_SEARCH is added

    def convert_to_sourced_executive(
        discovered: DiscoveredExecutive,
        source: DataSource = AI_SEARCH_SOURCE,
    ) -> SourcedExecutive:
        """Convert a DiscoveredExecutive to a SourcedExecutive.

        This allows integration with the existing search aggregation system.

        Args:
            discovered: The discovered executive from AI search.
            source: The data source to attribute (default AI_SEARCH).

        Returns:
            SourcedExecutive compatible with the aggregation system.
        """
        return SourcedExecutive(
            executive=discovered.to_executive(),
            source=source,
            confidence=discovered.confidence,
            source_url=None,
            raw_data={
                "sources": [m.value for m in discovered.sources],
                "validated": discovered.validated,
                "validation_reason": discovered.validation_reason,
            },
        )

    SEARCH_SERVICE_INTEGRATION = True
except ImportError:
    SEARCH_SERVICE_INTEGRATION = False
    convert_to_sourced_executive = None  # type: ignore


# Singleton instance
_ai_search_service: AISearchService | None = None


def get_ai_search_service() -> AISearchService:
    """Get the singleton AISearchService instance.

    Returns:
        The singleton AISearchService.
    """
    global _ai_search_service
    if _ai_search_service is None:
        _ai_search_service = AISearchService()
    return _ai_search_service


async def search_company_executives(
    company_name: str,
    with_fallback: bool = True,
) -> list[DiscoveredExecutive]:
    """Convenience function to search for company executives.

    Args:
        company_name: Name of the company to search.
        with_fallback: Whether to use fallback models if primary fails.

    Returns:
        List of validated and deduplicated executives.
    """
    service = get_ai_search_service()

    if with_fallback:
        return await service.search_executives_with_fallback(company_name)
    else:
        return await service.search_executives(company_name)
