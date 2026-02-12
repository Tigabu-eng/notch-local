"""
Executive name validation service.

Validates executive names to prevent role titles from being extracted as names.
"""

import re
from dataclasses import dataclass


@dataclass
class ValidationResult:
    """Result of executive validation."""
    is_valid: bool
    confidence: float  # 0.0-1.0
    reason: str


class ExecutiveValidator:
    """Validates executive names and data quality."""

    # Maximum constraints for valid names
    MAX_NAME_LENGTH = 60
    MAX_WORD_COUNT = 5

    # Patterns that indicate a title, not a name
    INVALID_NAME_PATTERNS = [
        r'^(chief|executive|officer|president|vice|senior|director|manager)$',
        r'^(ceo|cfo|coo|cto|cmo|cio|cpo|cro|cso)$',
        r'^the\s+',  # "the CEO", "the President"
        r'\s+(inc|llc|corp|company|corporation)$',
        r'^(mr|mrs|ms|dr)\.$',  # Just honorific
        r'^\d+',  # Starts with number
        r'^[A-Z]{2,5}$',  # All caps acronym like "CEO"
    ]

    # Action verbs that indicate sentence fragments, not names
    ACTION_VERBS = {
        'joined', 'appointed', 'named', 'hired', 'promoted', 'left', 'departed',
        'resigned', 'retired', 'assumed', 'became', 'takes', 'took', 'joins',
        'appoints', 'names', 'hires', 'promotes', 'leaves', 'departs', 'resigns',
        'includes', 'include', 'report', 'reports', 'featuring', 'features',
    }

    # Announcement verbs that indicate PR language
    ANNOUNCEMENT_VERBS = {
        'announced', 'confirmed', 'revealed', 'reported', 'disclosed', 'stated',
        'announces', 'confirms', 'reveals', 'reports', 'discloses', 'states',
    }

    # Business jargon that should not appear in names
    BUSINESS_JARGON = {
        'transformation', 'digital', 'strategy', 'innovation', 'growth',
        'operations', 'marketing', 'technology', 'financial', 'corporate',
        'business', 'strategic', 'global', 'enterprise', 'solutions',
    }

    # Generic terms that are not names
    GENERIC_TERMS = {
        'board members', 'leadership', 'management', 'team',
        'executives', 'officers', 'directors', 'members',
        'most recently', 'recently',
    }

    # Department prefixes that indicate role descriptions, not names
    DEPARTMENT_PREFIXES = [
        r'^regulatory\s+affairs\s+',
        r'^human\s+resources\s+',
        r'^corporate\s+development\s+',
        r'^information\s+technology\s+',
        r'^business\s+development\s+',
        r'^investor\s+relations\s+',
        r'^public\s+relations\s+',
        r'^finance\s+',
        r'^sales\s+',
        r'^legal\s+affairs?\s+',
        r'^public\s+affairs?\s+',
        r'^government\s+affairs?\s+',
        r'^operations\s+',
        r'^engineering\s+',
        r'^product\s+',
        r'^research\s+',
        r'^compliance\s+',
        r'^risk\s+',
        r'^audit\s+',
        r'^treasury\s+',
        r'^communications?\s+',
        r'^brand\s+',
        r'^accounting\s+',
        r'^procurement\s+',
        r'^logistics\s+',
        r'^supply\s+chain\s+',
        r'^quality\s+',
        r'^safety\s+',
        r'^training\s+',
        r'^talent\s+',
        r'^people\s+',
        r'^admin\s+',
        r'^administrative\s+',
        r'^(it|hr)\s+',
    ]

    # Sentence starters that indicate fragments, not names
    SENTENCE_STARTER_PATTERNS = [
        r'^to\s+welcome\b',
        r'^pleased\s+to\s+announce\b',
        r'^the\s+appointment\s+of\b',
        r'^we\s+are\s+',
        r'^is\s+pleased\b',
        r'^has\s+appointed\b',
        r'^has\s+named\b',
        r'^has\s+hired\b',
        r'^welcomes\b',
        r'^appoints\b',
        r'^names\b',
        r'^hires\b',
        r'^leadership\s+',
        r'^report\s+to\b',
        r'^appointment\s+of\b',
        r'^featuring\b',
        r'^most\s+recently\b',
    ]

    # Patterns that cannot START a valid name (action verbs, gerunds)
    STARTING_ACTION_PATTERNS = [
        r'^will\s+(?!smith|williams|wilson|johnson|jones)',  # "Will Join" but allow "Will Smith"
        r'^joining\b',
        r'^leading\b',
        r'^assuming\b',
        r'^succeeding\b',
        r'^replacing\b',
        r'^effective\b',
        r'^based\s+in\b',
        r'^previously\b',
        r'^formerly\b',
        r'^currently\b',
        r'^recently\b',
        r'^serving\b',
        r'^reporting\b',
        r'^working\b',
        r'^transitioning\b',
        r'^overseeing\b',
        r'^managing\b',
        r'^handling\b',
    ]

    # Common first names that might be confused with action words
    COMMON_FIRST_NAMES = {
        'will', 'grant', 'mark', 'april', 'may', 'june', 'august',
        'chase', 'hunter', 'porter', 'dallas', 'austin', 'jordan',
        'taylor', 'morgan', 'parker', 'carter', 'mason', 'faith',
        'hope', 'joy', 'grace', 'charity', 'patience', 'penny',
    }

    # Patterns for valid names
    VALID_NAME_PATTERN = re.compile(
        r'^[A-Z][a-z]+(\s+[A-Z][a-z]+){1,3}$'  # First Last or First Middle Last
    )

    # Common titles that shouldn't be names
    TITLE_WORDS = {
        'chief', 'executive', 'officer', 'president', 'vice', 'senior',
        'director', 'manager', 'head', 'lead', 'principal', 'chairman',
        'chairwoman', 'chairperson', 'founder', 'co-founder', 'partner',
        'ceo', 'cfo', 'coo', 'cto', 'cmo', 'cio', 'cpo', 'cro', 'cso',
    }

    def validate_name(self, name: str) -> tuple[bool, str]:
        """
        Validate if a string is a valid executive name.

        Returns:
            tuple[bool, str]: (is_valid, reason)
        """
        if not name or not isinstance(name, str):
            return False, "Name is empty or not a string"

        name = name.strip()

        # Check minimum length (at least "Jo Li" = 5 chars)
        if len(name) < 3:
            return False, "Name too short"

        # Check maximum length
        if len(name) > self.MAX_NAME_LENGTH:
            return False, f"Name too long (max {self.MAX_NAME_LENGTH} characters)"

        name_lower = name.lower()
        words = name_lower.split()

        # Check for starting action patterns (before COMMON_FIRST_NAMES check)
        first_word = words[0] if words else ""
        for pattern in self.STARTING_ACTION_PATTERNS:
            if re.match(pattern, name_lower):
                # For patterns with negative lookahead (like 'will'), always check
                # For other patterns, skip if first word is a common first name
                if '(?!' in pattern or first_word not in self.COMMON_FIRST_NAMES:
                    return False, f"Name starts with action pattern: {pattern}"

        # Check maximum word count
        if len(words) > self.MAX_WORD_COUNT:
            return False, f"Name has too many words (max {self.MAX_WORD_COUNT} words)"

        # Check against invalid patterns
        for pattern in self.INVALID_NAME_PATTERNS:
            if re.match(pattern, name_lower):
                return False, f"Name matches invalid pattern: {pattern}"

        # Check for sentence starter patterns
        for pattern in self.SENTENCE_STARTER_PATTERNS:
            if re.match(pattern, name_lower):
                return False, f"Name appears to be a sentence fragment: {pattern}"

        # Check if name is a generic term
        if name_lower in self.GENERIC_TERMS:
            return False, f"Name is a generic term: {name_lower}"

        # Check for department prefixes
        for pattern in self.DEPARTMENT_PREFIXES:
            if re.match(pattern, name_lower):
                return False, f"Name starts with department prefix: {pattern}"

        # Check for lowercase-only two-word phrases (like "most recently")
        if len(words) == 2 and name.islower():
            return False, "Name consists of only lowercase words"

        # Check for action verbs that indicate sentence fragments
        for word in words:
            if word in self.ACTION_VERBS:
                return False, f"Name contains action verb: {word}"

        # Check for announcement verbs
        for word in words:
            if word in self.ANNOUNCEMENT_VERBS:
                return False, f"Name contains announcement verb: {word}"

        # Check for business jargon
        for word in words:
            if word in self.BUSINESS_JARGON:
                return False, f"Name contains business jargon: {word}"

        # Check if name is just title words
        if all(word in self.TITLE_WORDS for word in words):
            return False, "Name consists only of title words"

        # Check if first word is a title
        if words and words[0] in self.TITLE_WORDS:
            return False, f"Name starts with title word: {words[0]}"

        # Check for valid name structure (First Last)
        if len(words) < 2:
            return False, "Name should have at least first and last name"

        # Check if it looks like a real name
        if self.VALID_NAME_PATTERN.match(name):
            return True, "Valid name format"

        # Allow names with special characters (O'Brien, McDonald, etc.)
        if re.match(r"^[A-Z][a-z']+(\s+[A-Z][a-z']+){1,3}$", name):
            return True, "Valid name with special characters"

        # Allow names with hyphens (Smith-Jones)
        if re.match(r"^[A-Z][a-z]+(-[A-Z][a-z]+)?(\s+[A-Z][a-z]+(-[A-Z][a-z]+)?){1,3}$", name):
            return True, "Valid name with hyphen"

        # Allow names with honorifics and suffixes (Dr. John A. Smith III)
        if re.match(r"^(Dr\.|Mr\.|Mrs\.|Ms\.)?\s*[A-Z][a-z]+(\s+[A-Z]\.)?(\s+[A-Z][a-z]+)+(\s+(Jr\.|Sr\.|II|III|IV))?$", name):
            return True, "Valid name with honorific or suffix"

        # Default to invalid if no pattern matches
        return False, "Name does not match expected format"

    def extract_name_from_prefixed(self, text: str) -> tuple[str | None, str | None]:
        """
        Extract a valid name from text that may have a department prefix.

        Args:
            text: Input string that may contain a name with a prefix.

        Returns:
            tuple[str | None, str | None]: (extracted_name, inferred_title)
            Returns (None, None) if no valid name can be extracted.

        Examples:
            "Regulatory Affairs Tammie Slauter" -> ("Tammie Slauter", "Regulatory Affairs")
            "Finance John Smith" -> ("John Smith", "Finance")
            "Will Join Acrisure" -> (None, None)  # Not recoverable
            "John Smith" -> ("John Smith", None)  # No prefix
        """
        if not text:
            return None, None

        text = text.strip()

        # First check if it's an action pattern - these are NOT recoverable
        text_lower = text.lower()
        for pattern in self.STARTING_ACTION_PATTERNS:
            if re.match(pattern, text_lower):
                # Check if first word is a common first name
                first_word = text_lower.split()[0] if text.split() else ""
                if first_word not in self.COMMON_FIRST_NAMES:
                    return None, None

        # Check for department prefixes - these ARE recoverable
        for pattern in self.DEPARTMENT_PREFIXES:
            match = re.match(pattern, text_lower)
            if match:
                prefix = match.group(0).strip()
                remainder = text[len(prefix):].strip()
                # Validate the remainder as a name
                is_valid, _ = self.validate_name(remainder)
                if is_valid:
                    # Capitalize the prefix properly for title
                    inferred_title = ' '.join(word.capitalize() for word in prefix.split())
                    return remainder, inferred_title

        # No prefix detected - just validate the original
        is_valid, _ = self.validate_name(text)
        if is_valid:
            return text, None

        return None, None

    def validate_executive(self, exec_data: dict) -> ValidationResult:
        """
        Full validation of executive data with confidence score.

        Args:
            exec_data: Dictionary with executive information
                       Expected keys: name, title, start_year, end_year

        Returns:
            ValidationResult with validity, confidence, and reason
        """
        name = exec_data.get('name', '')
        title = exec_data.get('title', '')

        # Validate name
        name_valid, name_reason = self.validate_name(name)

        if not name_valid:
            return ValidationResult(
                is_valid=False,
                confidence=0.0,
                reason=f"Invalid name: {name_reason}"
            )

        # Calculate confidence based on data completeness
        confidence = 0.5  # Base confidence for valid name

        # Boost confidence for having a title
        if title and len(title) > 3:
            confidence += 0.2

        # Boost for having dates
        if exec_data.get('start_year'):
            confidence += 0.1

        # Boost for having sources
        if exec_data.get('sources') or exec_data.get('source_url'):
            confidence += 0.2

        # Cap at 1.0
        confidence = min(confidence, 1.0)

        return ValidationResult(
            is_valid=True,
            confidence=confidence,
            reason="Executive data validated successfully"
        )

    def filter_valid_executives(
        self,
        executives: list[dict],
        min_confidence: float = 0.5
    ) -> list[dict]:
        """
        Filter a list of executives, keeping only valid ones.

        Args:
            executives: List of executive dictionaries
            min_confidence: Minimum confidence threshold (default 0.5)

        Returns:
            List of valid executives with confidence >= min_confidence
        """
        valid_executives = []

        for exec_data in executives:
            result = self.validate_executive(exec_data)

            if result.is_valid and result.confidence >= min_confidence:
                # Add confidence score to the executive data
                exec_data['confidence_score'] = result.confidence
                exec_data['validation_reason'] = result.reason
                valid_executives.append(exec_data)

        return valid_executives


# Singleton instance for easy importing
validator = ExecutiveValidator()
