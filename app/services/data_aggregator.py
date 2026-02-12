"""Data aggregation service for combining company data from multiple sources."""

import logging
import re
from datetime import datetime
from difflib import SequenceMatcher
from typing import Any

from app.models import CompanyResponse, Executive

logger = logging.getLogger(__name__)

NAME_SIMILARITY_THRESHOLD = 0.85


class DataAggregator:
    """Aggregates and deduplicates company data from multiple sources."""

    def __init__(self) -> None:
        self._current_year = datetime.utcnow().year

    def deduplicate_executives(
        self,
        executives: list[Executive],
    ) -> list[Executive]:
        """Remove duplicate executives based on name similarity."""
        if not executives:
            return []

        deduplicated: list[Executive] = []
        used_indices: set[int] = set()

        for i, exec1 in enumerate(executives):
            if i in used_indices:
                continue

            similar_group = [exec1]
            used_indices.add(i)

            for j, exec2 in enumerate(executives[i + 1:], start=i + 1):
                if j in used_indices:
                    continue

                similarity = SequenceMatcher(
                    None,
                    exec1.name.lower(),
                    exec2.name.lower(),
                ).ratio()

                if similarity >= NAME_SIMILARITY_THRESHOLD:
                    similar_group.append(exec2)
                    used_indices.add(j)

            merged = self._merge_executives(similar_group)
            deduplicated.append(merged)

        return deduplicated

    def merge_company_info(
        self,
        existing: CompanyResponse,
        new: dict[str, Any],
    ) -> CompanyResponse:
        """Merge new company information into existing record."""
        merged_data = existing.model_dump()

        # Merge executives
        for field in ["ceo", "c_level", "senior_level"]:
            if field in new and new[field]:
                existing_list = getattr(existing, field)
                new_list = [
                    Executive(**e) if isinstance(e, dict) else e
                    for e in new[field]
                ]
                all_execs = existing_list + new_list
                merged_data[field] = [
                    e.model_dump() for e in self.deduplicate_executives(all_execs)
                ]

        # Update scalar fields only if currently empty
        for field in ["employees", "ownership", "acquisition_date", "subsector"]:
            if merged_data.get(field) is None and new.get(field):
                merged_data[field] = new[field]

        # Append notes
        if new.get("notes"):
            existing_notes = merged_data.get("notes") or ""
            merged_data["notes"] = (
                f"{existing_notes} | {new['notes']}" if existing_notes else new["notes"]
            )

        merged_data["updated"] = datetime.utcnow()

        return CompanyResponse(**merged_data)

    def extract_executive_info(self, text: str) -> list[Executive]:
        """Extract executive information from unstructured text."""
        executives: list[Executive] = []

        patterns = [
            r"([A-Z][a-z]+(?:\s+[A-Z][a-z]+)+)[,\s]+([A-Za-z\s]+(?:CEO|CFO|COO|CTO|President|VP)[A-Za-z\s]*)[,\s]+(\d{4})\s*[-–]\s*(\d{4}|[Pp]resent)",
            r"([A-Z][a-z]+(?:\s+[A-Z][a-z]+)+)[,\s]+(CEO|CFO|COO|CTO|President|Vice\s+President|VP|SVP)",
        ]

        for pattern in patterns:
            matches = re.finditer(pattern, text)
            for match in matches:
                groups = match.groups()
                name = groups[0].strip()
                title = groups[1].strip()

                start_year = self._current_year
                end_year = None

                if len(groups) >= 3 and groups[2]:
                    try:
                        start_year = int(groups[2])
                    except ValueError:
                        pass

                if len(groups) >= 4 and groups[3]:
                    if groups[3].lower() != "present":
                        try:
                            end_year = int(groups[3])
                        except ValueError:
                            pass

                if name and title and len(name) > 3:
                    executives.append(Executive(
                        name=name,
                        title=title,
                        start_year=start_year,
                        end_year=end_year,
                    ))

        return self.deduplicate_executives(executives)

    def _merge_executives(self, executives: list[Executive]) -> Executive:
        """Merge similar executives into one record."""
        if len(executives) == 1:
            return executives[0]

        best_name = max((e.name for e in executives), key=len)
        best_title = max((e.title for e in executives), key=len)
        start_year = min(e.start_year for e in executives)

        end_years = [e.end_year for e in executives]
        if None in end_years:
            end_year = None
        else:
            end_year = max(y for y in end_years if y is not None)

        return Executive(
            name=best_name,
            title=best_title,
            start_year=start_year,
            end_year=end_year,
        )


_data_aggregator: DataAggregator | None = None


def get_data_aggregator() -> DataAggregator:
    """Get the singleton DataAggregator instance."""
    global _data_aggregator
    if _data_aggregator is None:
        _data_aggregator = DataAggregator()
    return _data_aggregator
