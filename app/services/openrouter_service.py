"""OpenRouter service for LLM-based data extraction.

This service uses the OpenAI Python SDK configured to talk to OpenRouter's
OpenAI-compatible API.
"""

from __future__ import annotations

import json
import logging
import os
import re
from datetime import datetime
from typing import Any

from openai import AsyncOpenAI

from app.models import EmploymentRecord, Executive, CallInsight, ActionItem, PersonMention

logger = logging.getLogger(__name__)

OPENROUTER_API_KEY = os.getenv("OPENROUTER_API_KEY", "")

# OpenRouter is OpenAI-compatible; use the SDK with this base URL.
OPENROUTER_BASE_URL = os.getenv("OPENROUTER_BASE_URL", "https://openrouter.ai/api/v1")

# Optional app attribution headers (recommended by OpenRouter)
OPENROUTER_SITE_URL = os.getenv("OPENROUTER_SITE_URL", "")
OPENROUTER_APP_NAME = os.getenv("OPENROUTER_APP_NAME", "")

# Model configuration
OPENROUTER_MODEL = os.getenv("OPENROUTER_MODEL", "anthropic/claude-3.5-sonnet")
OPENROUTER_EXEC_EXTRACTION_MODEL = os.getenv("OPENROUTER_EXEC_EXTRACTION_MODEL", OPENROUTER_MODEL)
OPENROUTER_CALL_ANALYSIS_MODEL = os.getenv("OPENROUTER_CALL_ANALYSIS_MODEL", OPENROUTER_MODEL)


class OpenRouterService:
    """Service for LLM-based structured data extraction via OpenRouter."""

    def __init__(self, api_key: str | None = None) -> None:
        # Prefer explicit api_key, otherwise env var; strip to avoid hidden whitespace/newlines.
        self.api_key = (api_key or OPENROUTER_API_KEY or "").strip()

        # Optional attribution headers (recommended by OpenRouter)
        default_headers: dict[str, str] = {}
        if OPENROUTER_SITE_URL:
            default_headers["HTTP-Referer"] = OPENROUTER_SITE_URL
        if OPENROUTER_APP_NAME:
            default_headers["X-Title"] = OPENROUTER_APP_NAME

        self._client: AsyncOpenAI | None = None
        self._default_headers = default_headers

    @property
    def is_configured(self) -> bool:
        """Check if the API key is configured."""
        return bool(self.api_key)

    def _get_client(self) -> AsyncOpenAI:
        """Get or create SDK client."""
        if self._client is None:
            self._client = AsyncOpenAI(
                api_key=self.api_key,
                base_url=OPENROUTER_BASE_URL,
                default_headers=self._default_headers or None,
            )
        return self._client

    async def close(self) -> None:
        """Close SDK client."""
        if self._client is not None:
            await self._client.close()
            self._client = None

    async def _chat_completion(
        self,
        *,
        model: str,
        messages: list[dict[str, Any]],
        temperature: float = 0,
        response_format: dict[str, Any] | None = None,
    ) -> str | None:
        if not self.api_key:
            logger.warning("OpenRouter API key not configured")
            return None

        try:
            client = self._get_client()
            resp = await client.chat.completions.create(
                model=model,
                messages=messages,
                temperature=temperature,
                response_format=response_format,
            )
            return (resp.choices[0].message.content or "").strip()
        except Exception as e:
            logger.exception("OpenRouter SDK request failed: %s", e)
            return None


    async def extract_executives(self, content: str, company_name: str) -> list[Executive]:
        """Extract structured executive data from text using LLM."""
        prompt = self._build_extraction_prompt(content, company_name)
        content_text = await self._chat_completion(
            model=OPENROUTER_EXEC_EXTRACTION_MODEL,
            messages=[{"role": "user", "content": prompt}],
            temperature=0,
        )
        if not content_text:
            return []

        try:
            json_str = self._extract_json(content_text)
            executives_data = json.loads(json_str)
            return [
                self._parse_executive(e, company_name)
                for e in executives_data
                if self._is_valid_executive(e)
            ]
        except json.JSONDecodeError as e:
            logger.error(f"Failed to parse LLM response as JSON: {e}")
            return []
        except Exception as e:
            logger.error(f"OpenRouter extraction failed: {e}")
            return []

    async def analyze_call(self, transcript: str) -> CallInsight | None:
        """Analyze a call transcript into structured insights.

        Returns CallInsight or None if analysis fails.
        """
        prompt = self._build_call_analysis_prompt(transcript)
        content_text = await self._chat_completion(
            model=OPENROUTER_CALL_ANALYSIS_MODEL,
            messages=[{"role": "user", "content": prompt}],
            temperature=0,
            response_format={"type": "json_object"},
        )
        if not content_text:
            return None

        try:
            json_str = self._extract_json(content_text)
            data = json.loads(json_str)

            return CallInsight(
                summary=data.get("summary", "").strip(),
                tags=[t for t in (data.get("tags") or []) if isinstance(t, str)],
                action_items=[
                    ActionItem(**ai) for ai in (data.get("action_items") or [])
                    if isinstance(ai, dict) and ai.get("description")
                ],
                people_mentioned=[
                    PersonMention(**p) for p in (data.get("people_mentioned") or [])
                    if isinstance(p, dict) and p.get("name")
                ],
                key_decisions=[d for d in (data.get("key_decisions") or []) if isinstance(d, str)],
            )
        except json.JSONDecodeError as e:
            logger.error(f"Failed to parse call analysis response as JSON: {e}")
            return None
        except Exception as e:
            logger.error(f"Call analysis parsing failed: {e}")
            return None

    def _build_call_analysis_prompt(self, transcript: str) -> str:
        """Build prompt for transcript analysis."""
        return f'''You are an analyst. Turn the call transcript into structured business insights.

Return ONLY valid JSON (no markdown, no extra text) with this exact shape:
{{
  "summary": "2-6 sentence summary of the call",
  "tags": ["tag1", "tag2", "tag3"],
  "action_items": [
    {{"description": "what to do", "owner": "optional person/team", "urgency": "low|medium|high"}}
  ],
  "people_mentioned": [
    {{"name": "Full Name", "role": "optional", "company": "optional"}}
  ],
  "key_decisions": ["decision 1", "decision 2"]
}}

Rules:
- Be concise and concrete.
- tags should be short (1-3 words), and 5-12 items max.
- action_items should be actionable, 0-10 items.
- people_mentioned should include only real people referenced, 0-15 items.
- key_decisions should include explicit decisions/commitments, 0-10 items.
- If something is unknown, omit it or use null/empty list (do not invent).

TRANSCRIPT:
{transcript[:16000]}
'''

    def _build_extraction_prompt(self, content: str, company_name: str) -> str:
        """Build the prompt for executive extraction.

        Uses strict validation rules to prevent common extraction errors
        like treating job titles as names.
        """
        # Sanitize inputs to prevent prompt injection
        safe_company_name = self._sanitize_prompt_input(company_name, max_length=200)
        safe_content = self._sanitize_prompt_input(content, max_length=8000)

        return f'''Extract executives from {safe_company_name}.

STRICT VALIDATION:
1. Names MUST be real human names (First Last format, typically 2-4 words)
2. REJECT titles as names: "CEO", "President", "Officer", "Director" are NOT names
3. REJECT generic phrases: "the CEO", "company executive", "leadership team"
4. Each executive MUST have a verifiable name - skip entries without clear names
5. Include confidence score (0.0-1.0) for each entry based on data quality
6. Extract actual dates from text when available - look for patterns like:
   - "since 2018", "from 2019", "joined in 2020", "appointed in 2021"
   - "(2015-2020)", "(2018-present)", "2019 to present"
   - "has been CEO since 2017", "became CFO in 2019"
7. If start year is NOT mentioned or cannot be determined, use null - DO NOT guess dates
8. Deduplicate - each person should appear only once

EXAMPLES OF INVALID NAMES (DO NOT EXTRACT):
- "Chief Executive Officer" (this is a title, not a name)
- "CEO" (acronym, not a name)
- "President" (title, not a name)
- "the CEO of the company" (phrase, not a name)
- "Executive Team" (generic phrase)
- "Vice President of Sales" (title only, no name)
- "Chief Financial Officer" (title only)

EXAMPLES OF VALID NAMES:
- "John Smith" (real name - 2 words)
- "Mary Jane Watson" (real name - 3 words)
- "Tim Cook" (real name)
- "Robert De Niro Jr." (real name - 4 words with suffix)

DATE EXTRACTION EXAMPLES:
- "John Smith has been CEO since 2018" -> start_year: 2018
- "Jane Doe, CFO (2019-present)" -> start_year: 2019, end_year: null
- "Bob Wilson joined as COO in 2020" -> start_year: 2020
- "Mary Brown, former CEO (2015-2022)" -> start_year: 2015, end_year: 2022
- "Tom Davis, President" (no date mentioned) -> start_year: null

OUTPUT JSON (return ONLY this JSON array, no markdown):
[{{
  "name": "FirstName LastName",
  "title": "Specific Title at {safe_company_name}",
  "start_year": null,
  "end_year": null,
  "confidence": 0.9,
  "source_url": "https://... or null",
  "reasoning": "Brief explanation of where this was found and what dates were extracted",
  "linkedin_url": "url or null",
  "employment_history": [
    {{"company_name": "Previous Co", "title": "Previous Role", "start_year": 2018, "end_year": 2022}}
  ]
}}]

CONFIDENCE SCORING GUIDE:
- 1.0: Name explicitly mentioned with title and dates on official company page
- 0.8-0.9: Name mentioned in reliable news source with title (dates may or may not be present)
- 0.6-0.7: Name mentioned but title or role unclear
- 0.4-0.5: Inferred from context, not explicitly stated
- Below 0.4: Do not include - insufficient confidence

IMPORTANT: When dates are unknown, return null for start_year and end_year.
Do NOT default to the current year - unknown dates should remain null.

TEXT TO ANALYZE:
{safe_content}

Return ONLY valid JSON array, no other text or markdown.'''

    def _extract_json(self, text: str) -> str:
        """Extract JSON from response, handling markdown code blocks."""
        text = text.strip()
        if text.startswith("```"):
            lines = text.split("\n")
            lines = [line for line in lines if not line.startswith("```")]
            text = "\n".join(lines)
        return text.strip()

    def _is_valid_executive(self, data: dict) -> bool:
        """Check if extracted data represents a valid executive.

        Applies strict validation to filter out common extraction errors
        like job titles being mistaken for names.

        Args:
            data: Dictionary containing executive data from LLM response.

        Returns:
            True if the data represents a valid executive with a real name.
        """
        name = data.get("name", "")
        if not name or not isinstance(name, str):
            return False

        invalid_patterns = [
            "including",
            "said",
            "according",
            "reported",
            "ceo",
            "cfo",
            "coo",
            "president",
            "officer",
            "executive",
            "director",
            "manager",
        ]
        name_lower = name.lower()

        # Check confidence score if present - reject low confidence entries
        confidence = data.get("confidence", 1.0)
        if isinstance(confidence, (int, float)) and confidence < 0.4:
            logger.debug(f"Rejected '{name}' due to low confidence: {confidence}")
            return False

        # Reject if name is too short or too long
        if len(name) < 3 or len(name) > 100:
            return False

        # Reject names that are clearly titles or roles (exact matches)
        title_only_patterns = {
            "ceo", "cfo", "coo", "cto", "cmo", "cio", "chro", "cpo",
            "president", "vice president", "vp",
            "chief executive officer", "chief financial officer",
            "chief operating officer", "chief technology officer",
            "chief marketing officer", "chief information officer",
            "executive", "director", "manager", "officer",
            "chairman", "chairwoman", "chairperson",
            "founder", "co-founder", "cofounder",
            "partner", "managing partner", "general partner",
            "head of", "leader", "leadership", "team",
            "board member", "board of directors",
        }
        if name_lower in title_only_patterns:
            logger.debug(f"Rejected '{name}' - exact match to title pattern")
            return False

        # Reject if name contains title-indicating substrings
        invalid_substrings = [
            "chief ", " officer", "president of", "vice president",
            "director of", "head of", "manager of",
            "executive team", "leadership team", "management team",
            "the ceo", "the cfo", "the president",
            "including", "said", "according", "reported",
            "company's", "companies", "corporation",
            "announced", "statement", "press release",
        ]
        if any(substring in name_lower for substring in invalid_substrings):
            logger.debug(f"Rejected '{name}' - contains invalid substring")
            return False

        parts = name.split()
        if len(parts) < 2:
            logger.debug(f"Rejected '{name}' - fewer than 2 name parts")
            return False
        if len(parts) > 6:
            logger.debug(f"Rejected '{name}' - more than 6 name parts (likely a phrase)")
            return False

        # Check that name parts look like actual names (capitalized, reasonable length)
        for part in parts:
            # Skip common suffixes and honorifics
            if part.lower() in {"jr", "jr.", "sr", "sr.", "ii", "iii", "iv", "phd", "md", "esq"}:
                continue
            # Each name part should be 2-20 characters
            if len(part) < 2 or len(part) > 20:
                logger.debug(f"Rejected '{name}' - name part '{part}' has invalid length")
                return False
            # Name parts should start with a letter (allowing for names like O'Brien)
            if not part[0].isalpha():
                logger.debug(f"Rejected '{name}' - name part '{part}' doesn't start with letter")
                return False

        return True

    def _parse_executive(self, data: dict, company_name: str) -> Executive:
        """Convert LLM output to Executive model.

        Parses the enhanced LLM response including confidence scores
        and source information.

        Args:
            data: Dictionary containing executive data from LLM response.
            company_name: The company name for context.

        Returns:
            Executive object populated with parsed data.
        """
        employment_history = [
            EmploymentRecord(
                company_name=eh.get("company_name", ""),
                title=eh.get("title", ""),
                start_year=eh.get("start_year"),
                end_year=eh.get("end_year"),
            )
            for eh in data.get("employment_history", [])
            if isinstance(eh, dict) and eh.get("company_name")
        ]

        # Clean and normalize the name
        name = data["name"].strip()

        # Extract start_year from response - allow None for unknown dates
        start_year = data.get("start_year")
        if start_year is not None:
            # Validate if present - reject invalid years but keep None as valid
            if not isinstance(start_year, int) or start_year < 1900 or start_year > 2100:
                start_year = None

        # Extract end_year from response
        end_year = data.get("end_year")
        if end_year is not None:
            if not isinstance(end_year, int) or end_year < 1900 or end_year > 2100:
                end_year = None

        # Log confidence and reasoning for debugging/auditing
        confidence = data.get("confidence", 0.0)
        reasoning = data.get("reasoning", "")
        source_url = data.get("source_url")
        if reasoning:
            logger.debug(
                f"Extracted '{name}' with confidence {confidence}: {reasoning}"
            )

        return Executive(
            name=name,
            title=data.get("title", "Executive"),
            start_year=start_year,
            end_year=end_year,
            linkedin_url=data.get("linkedin_url"),
            photo_url=data.get("photo_url"),
            employment_history=employment_history,
        )

_openrouter_service: OpenRouterService | None = None


def get_openrouter_service() -> OpenRouterService:
    """Get the singleton OpenRouterService instance."""
    global _openrouter_service
    if _openrouter_service is None:
        _openrouter_service = OpenRouterService()
    return _openrouter_service