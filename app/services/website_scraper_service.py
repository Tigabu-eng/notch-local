"""Website scraper service for extracting executive data from company websites."""

import asyncio
import logging
import re
from dataclasses import dataclass
from urllib.parse import urljoin, urlparse

import httpx
from bs4 import BeautifulSoup, Tag

logger = logging.getLogger(__name__)

# Common leadership page paths to try (ordered by specificity)
LEADERSHIP_PATHS = [
    # Most specific leadership paths first
    "/about/leadership",
    "/about-us/leadership",
    "/about-acrisure/leadership",  # Company-specific patterns
    "/about-company/leadership",
    "/company/leadership",
    "/corporate/leadership",
    "/leadership",
    "/leadership-team",
    "/executive-team",
    "/executives",
    "/management",
    "/management-team",
    # Team pages
    "/team",
    "/our-team",
    "/the-team",
    "/about/team",
    "/about-us/team",
    "/company/team",
    "/people",
    "/our-people",
    "/staff",
    # Generic about pages (last resort)
    "/about",
    "/about-us",
    "/about-company",
    "/who-we-are",
]

# Timeout for HTTP requests (seconds)
REQUEST_TIMEOUT = 10.0

# Minimum content length to consider page valid (bytes)
MIN_CONTENT_LENGTH = 1000


@dataclass
class ScrapedExecutive:
    """Executive data extracted from website."""

    name: str
    title: str | None = None
    photo_url: str | None = None
    source_url: str | None = None
    confidence: float = 0.85


class WebsiteScraperService:
    """Scrapes company websites for executive information."""

    # Non-person patterns to filter out
    NON_PERSON_PATTERNS = [
        r"insights?",
        r"news",
        r"blog",
        r"article",
        r"press",
        r"media",
        r"contact",
        r"careers?",
        r"jobs?",
        r"newsletter",
        r"subscribe",
        r"login",
        r"sign\s*up",
    ]

    def __init__(self) -> None:
        """Initialize the scraper service."""
        self._client: httpx.AsyncClient | None = None

    def _clean_title_text(self, element: Tag) -> str:
        """
        Extract and clean title text from an element, handling multi-line content.

        Args:
            element: BeautifulSoup Tag containing title text

        Returns:
            Cleaned title string with proper spacing
        """
        # Use separator to ensure spaces between text nodes
        text = element.get_text(separator=" ", strip=True)
        # Normalize multiple spaces to single space
        text = re.sub(r"\s+", " ", text)
        return text.strip()

    async def __aenter__(self) -> "WebsiteScraperService":
        """Enter async context and create HTTP client."""
        self._client = httpx.AsyncClient(
            timeout=REQUEST_TIMEOUT,
            follow_redirects=True,
            headers={
                "User-Agent": "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) AppleWebKit/537.36",
                "Accept": "text/html,application/xhtml+xml",
            },
        )
        return self

    async def __aexit__(self, *args: object) -> None:
        """Exit async context and close HTTP client."""
        if self._client:
            await self._client.aclose()

    async def scrape_company_website(
        self,
        website_url: str,
    ) -> list[ScrapedExecutive]:
        """
        Scrape executive data from a company website.

        Args:
            website_url: Base URL of company website (e.g., "https://acrisure.com")

        Returns:
            List of scraped executives with confidence scores
        """
        if not self._client:
            raise RuntimeError(
                "Service not initialized. Use 'async with' context manager."
            )

        # Normalize base URL
        if not website_url.startswith(("http://", "https://")):
            website_url = f"https://{website_url}"

        base_url = website_url.rstrip("/")
        executives: list[ScrapedExecutive] = []

        # Try each leadership path
        for path in LEADERSHIP_PATHS:
            url = f"{base_url}{path}"
            try:
                html, final_url = await self._fetch_page(url)
                if html:
                    page_executives = self._extract_executives(html, final_url)
                    if page_executives:
                        executives.extend(page_executives)
                        logger.info(
                            f"Found {len(page_executives)} executives at {final_url}"
                        )
                        # Don't break - continue to find more on other pages
            except Exception as e:
                logger.debug(f"Failed to scrape {url}: {e}")
                continue

        # Deduplicate by name
        seen_names: set[str] = set()
        unique_executives: list[ScrapedExecutive] = []
        for exec in executives:
            name_lower = exec.name.lower().strip()
            if name_lower not in seen_names:
                seen_names.add(name_lower)
                unique_executives.append(exec)

        return unique_executives

    async def _fetch_page(self, url: str) -> tuple[str | None, str]:
        """
        Fetch HTML content from URL.

        Args:
            url: URL to fetch

        Returns:
            Tuple of (HTML content or None, final URL after redirects)
        """
        if not self._client:
            return None, url

        try:
            response = await self._client.get(url)
            response.raise_for_status()

            content = response.text
            if len(content) < MIN_CONTENT_LENGTH:
                return None, url

            return content, str(response.url)
        except httpx.HTTPError as e:
            logger.debug(f"HTTP error fetching {url}: {e}")
            return None, url

    def _extract_executives(
        self, html: str, source_url: str
    ) -> list[ScrapedExecutive]:
        """
        Extract executive information from HTML content.

        Args:
            html: HTML content to parse
            source_url: URL where the content was fetched from

        Returns:
            List of extracted executives
        """
        soup = BeautifulSoup(html, "lxml")
        executives: list[ScrapedExecutive] = []

        # Strategy 1: Link-based team cards (most common modern pattern)
        # This handles sites like notchpartners.com where team members are
        # displayed as links to /team/name/ with <p> tags inside
        link_execs = self._extract_from_link_cards(soup, source_url)
        if link_execs:
            executives.extend(link_execs)
            logger.debug(f"Link card strategy found {len(link_execs)} executives")

        # Strategy 2: Look for leadership/team sections
        leadership_sections = self._find_leadership_sections(soup)
        for section in leadership_sections:
            execs = self._extract_from_section(section, source_url)
            executives.extend(execs)

        # Strategy 3: Look for person cards/items
        if not executives:
            cards = self._find_person_cards(soup)
            for card in cards:
                exec_data = self._extract_from_card(card, source_url)
                if exec_data:
                    executives.append(exec_data)

        # Strategy 4: List-item based patterns (like Acrisure)
        # Find <li> elements containing heading + text that looks like name + title
        if not executives:
            list_execs = self._extract_from_list_items(soup, source_url)
            if list_execs:
                executives.extend(list_execs)
                logger.debug(f"List item strategy found {len(list_execs)} executives")

        # Strategy 5: Adjacent paragraph pairs (fallback)
        # Find <p> tags where the pattern looks like name + title
        if not executives:
            para_execs = self._extract_from_adjacent_paragraphs(soup, source_url)
            if para_execs:
                executives.extend(para_execs)
                logger.debug(f"Adjacent paragraph strategy found {len(para_execs)} executives")

        # Strategy 6: Profile links fallback
        # Find links that appear to be person profile URLs
        if not executives:
            profile_execs = self._extract_from_profile_links(soup, source_url)
            if profile_execs:
                executives.extend(profile_execs)
                logger.debug(f"Profile link strategy found {len(profile_execs)} executives")

        return executives

    def _find_leadership_sections(self, soup: BeautifulSoup) -> list[Tag]:
        """
        Find HTML sections likely containing leadership info.

        Args:
            soup: BeautifulSoup parsed HTML

        Returns:
            List of Tag elements that may contain leadership info
        """
        sections: list[Tag] = []

        # Look for sections/divs with leadership-related class names or IDs
        keywords = [
            "leadership",
            "team",
            "executive",
            "management",
            "people",
            "officers",
        ]

        for keyword in keywords:
            # By class
            elements = soup.find_all(class_=re.compile(keyword, re.I))
            sections.extend(elements)

            # By ID
            elements = soup.find_all(id=re.compile(keyword, re.I))
            sections.extend(elements)

        return sections

    def _find_person_cards(self, soup: BeautifulSoup) -> list[Tag]:
        """
        Find elements that look like person/profile cards.

        Args:
            soup: BeautifulSoup parsed HTML

        Returns:
            List of Tag elements that look like person cards
        """
        cards: list[Tag] = []
        seen_elements: set[int] = set()

        # Common card class patterns
        card_patterns = [
            r"person",
            r"profile",
            r"team-member",
            r"team_member",
            r"staff",
            r"bio",
            r"leader",
            r"executive",
            r"card",
            r"member",
            r"employee",
            r"people",
            r"faculty",
            r"author",
            r"contributor",
        ]

        for pattern in card_patterns:
            elements = soup.find_all(class_=re.compile(pattern, re.I))
            for elem in elements:
                elem_id = id(elem)
                if elem_id not in seen_elements:
                    seen_elements.add(elem_id)
                    cards.append(elem)

        # Also look for links containing multiple text children (common pattern)
        # e.g., <a href="/team/name/"><img/><p>Name</p><p>Title</p></a>
        team_link_pattern = re.compile(
            r"/(team|people|staff|about|leadership)/", re.I
        )
        for link in soup.find_all("a", href=team_link_pattern):
            # Count meaningful child elements
            children = [
                c for c in link.children
                if isinstance(c, Tag) and c.name in ["p", "span", "div", "h2", "h3", "h4", "h5", "h6"]
            ]
            if len(children) >= 2:
                elem_id = id(link)
                if elem_id not in seen_elements:
                    seen_elements.add(elem_id)
                    cards.append(link)

        # Look for articles or divs containing person-related content
        for container in soup.find_all(["article", "div", "li"]):
            # Check if container has an img and text elements that look like a person
            has_img = container.find("img") is not None
            text_content = container.get_text(strip=True)

            # Simple heuristic: has image, reasonable text length, and not already found
            if has_img and 10 < len(text_content) < 200:
                elem_id = id(container)
                if elem_id not in seen_elements:
                    # Check if any child looks like a name
                    for child in container.find_all(["p", "span", "h3", "h4", "h5", "strong"]):
                        child_text = child.get_text(strip=True)
                        if self._looks_like_name(child_text):
                            seen_elements.add(elem_id)
                            cards.append(container)
                            break

        return cards

    def _extract_from_section(
        self, section: Tag, source_url: str
    ) -> list[ScrapedExecutive]:
        """
        Extract executives from a leadership section.

        Args:
            section: HTML section element
            source_url: Source URL for attribution

        Returns:
            List of extracted executives
        """
        executives: list[ScrapedExecutive] = []

        # Look for name/title pairs within the section
        # Common patterns: h3 + p, .name + .title, dt + dd

        # Pattern 1: Headers followed by paragraphs
        headers = section.find_all(["h2", "h3", "h4", "h5"])
        for header in headers:
            name = header.get_text(strip=True)
            if self._looks_like_name(name):
                # Look for title in next sibling
                title = None
                next_elem = header.find_next_sibling()
                if next_elem:
                    title_text = next_elem.get_text(strip=True)
                    if self._looks_like_title(title_text):
                        title = title_text

                # Look for photo
                photo_url = self._find_nearby_photo(header, source_url)

                executives.append(
                    ScrapedExecutive(
                        name=name,
                        title=title,
                        photo_url=photo_url,
                        source_url=source_url,
                    )
                )

        return executives

    def _extract_from_card(
        self, card: Tag, source_url: str
    ) -> ScrapedExecutive | None:
        """
        Extract executive info from a profile card element.

        Args:
            card: HTML card element
            source_url: Source URL for attribution

        Returns:
            ScrapedExecutive or None if extraction failed
        """
        name = None
        title = None
        photo_url = None

        # Try to find name (usually in header or .name element)
        name_elem = (
            card.find(class_=re.compile(r"name", re.I))
            or card.find(["h2", "h3", "h4", "h5"])
            or card.find("strong")
        )
        if name_elem:
            name = name_elem.get_text(strip=True)

        # Try to find title
        title_elem = (
            card.find(class_=re.compile(r"title|position|role", re.I))
            or card.find("em")
            or card.find("span", class_=re.compile(r"title", re.I))
        )
        if title_elem:
            title = title_elem.get_text(strip=True)

        # Try to find photo
        img = card.find("img")
        if img and img.get("src"):
            photo_url = self._resolve_url(str(img["src"]), source_url)

        if name and self._looks_like_name(name):
            return ScrapedExecutive(
                name=name,
                title=title if title and self._looks_like_title(title) else None,
                photo_url=photo_url,
                source_url=source_url,
            )

        return None

    def _extract_from_link_cards(
        self, soup: BeautifulSoup, source_url: str
    ) -> list[ScrapedExecutive]:
        """
        Extract executives from link-based card patterns.

        This handles modern team pages where each person is a link to their
        profile page (e.g., /team/john-smith/) with name and title in child elements.

        Args:
            soup: BeautifulSoup parsed HTML
            source_url: Source URL for attribution

        Returns:
            List of extracted executives
        """
        executives: list[ScrapedExecutive] = []
        seen_names: set[str] = set()

        # Pattern 1: Links to /team/name, /people/name, /staff/name, /about/name pages
        team_link_pattern = re.compile(
            r"/(team|people|staff|about|leadership|executives|management)/[a-z0-9\-_]+/?$",
            re.I,
        )
        team_links = soup.find_all("a", href=team_link_pattern)

        for link in team_links:
            name = None
            title = None
            photo_url = None

            # Get all paragraph children
            paragraphs = link.find_all("p")

            if len(paragraphs) >= 2:
                # Pattern: <a><p>Name</p><p>Title</p></a>
                potential_name = paragraphs[0].get_text(strip=True)
                # Use clean_title_text to handle multi-line titles with proper spacing
                potential_title = self._clean_title_text(paragraphs[1])

                if self._looks_like_name(potential_name):
                    name = potential_name
                    title = potential_title if self._looks_like_title(potential_title) else None
            elif len(paragraphs) == 1:
                # Pattern: <a>Name<p>Title</p></a> or <a><p>Name</p></a>
                para_text = paragraphs[0].get_text(strip=True)

                # Check if paragraph is name or title
                if self._looks_like_name(para_text):
                    name = para_text
                    # Look for title in link's direct text (excluding paragraph)
                    link_text = self._get_direct_text(link)
                    if link_text and self._looks_like_title(link_text):
                        title = link_text
                elif self._looks_like_title(para_text):
                    # Paragraph is title, look for name elsewhere
                    title = para_text
                    link_text = self._get_direct_text(link)
                    if link_text and self._looks_like_name(link_text):
                        name = link_text

            # Try headers inside the link
            if not name:
                header = link.find(["h2", "h3", "h4", "h5", "h6"])
                if header:
                    header_text = header.get_text(strip=True)
                    if self._looks_like_name(header_text):
                        name = header_text
                        # Look for title in sibling or paragraph
                        if not title and paragraphs:
                            for p in paragraphs:
                                p_text = p.get_text(strip=True)
                                if self._looks_like_title(p_text):
                                    title = p_text
                                    break

            # Try to extract name from span or div with text children
            if not name:
                for child in link.find_all(["span", "div"]):
                    child_text = child.get_text(strip=True)
                    if self._looks_like_name(child_text):
                        name = child_text
                        break

            # Look for photo
            img = link.find("img")
            if img and img.get("src"):
                photo_url = self._resolve_url(str(img["src"]), source_url)

            # Add executive if we found a valid name
            if name:
                name_lower = name.lower().strip()
                if name_lower not in seen_names:
                    seen_names.add(name_lower)
                    executives.append(
                        ScrapedExecutive(
                            name=name,
                            title=title,
                            photo_url=photo_url,
                            source_url=source_url,
                        )
                    )

        return executives

    def _extract_from_list_items(
        self, soup: BeautifulSoup, source_url: str
    ) -> list[ScrapedExecutive]:
        """
        Extract executives from list items containing heading + text patterns.

        This handles patterns like Acrisure's leadership page:
        <li>
            <img alt="Name headshot">
            <h6>Name</h6>
            <div>Title</div>
        </li>

        Args:
            soup: BeautifulSoup parsed HTML
            source_url: URL where the content was fetched from

        Returns:
            List of extracted executives
        """
        executives: list[ScrapedExecutive] = []

        # Find all list items
        list_items = soup.find_all("li")

        for li in list_items:
            name = None
            title = None
            photo_url = None

            # Look for heading inside list item (h1-h6)
            heading = li.find(["h1", "h2", "h3", "h4", "h5", "h6"])
            if heading:
                potential_name = heading.get_text(strip=True)
                if self._looks_like_name(potential_name):
                    name = potential_name

                    # Look for title in adjacent/sibling elements
                    # Check next sibling first
                    next_elem = heading.find_next_sibling()
                    if next_elem:
                        potential_title = self._clean_title_text(next_elem)
                        if potential_title and self._looks_like_title(potential_title):
                            title = potential_title

                    # If no title found, look for div/span/p after heading
                    if not title:
                        for elem in li.find_all(["div", "span", "p"]):
                            if elem != heading and not elem.find_parent(heading):
                                text = self._clean_title_text(elem)
                                if text and self._looks_like_title(text) and text != name:
                                    title = text
                                    break

            # Look for photo
            img = li.find("img")
            if img:
                src = img.get("src")
                alt = img.get("alt", "")
                # Only use if it looks like a person photo (has "headshot" or name in alt)
                if src and (
                    "headshot" in alt.lower()
                    or (name and name.split()[0].lower() in alt.lower())
                ):
                    photo_url = src

            if name:
                executives.append(
                    ScrapedExecutive(
                        name=name,
                        title=title,
                        photo_url=photo_url,
                        source_url=source_url,
                        confidence=0.85,
                    )
                )

        return executives

    def _extract_from_adjacent_paragraphs(
        self, soup: BeautifulSoup, source_url: str
    ) -> list[ScrapedExecutive]:
        """
        Extract executives from adjacent paragraph pairs that look like name + title.

        This handles pages where team members are displayed as consecutive <p> tags
        without specific wrapper elements or classes.

        Args:
            soup: BeautifulSoup parsed HTML
            source_url: Source URL for attribution

        Returns:
            List of extracted executives
        """
        executives: list[ScrapedExecutive] = []
        seen_names: set[str] = set()

        # Find all paragraphs
        paragraphs = soup.find_all("p")

        i = 0
        while i < len(paragraphs) - 1:
            current_p = paragraphs[i]
            next_p = paragraphs[i + 1]

            current_text = current_p.get_text(strip=True)
            next_text = next_p.get_text(strip=True)

            # Check if current looks like name and next looks like title
            if (
                self._looks_like_name(current_text)
                and self._looks_like_title(next_text)
            ):
                name_lower = current_text.lower().strip()
                if name_lower not in seen_names:
                    seen_names.add(name_lower)

                    # Look for photo nearby
                    photo_url = self._find_nearby_photo(current_p, source_url)

                    executives.append(
                        ScrapedExecutive(
                            name=current_text,
                            title=next_text,
                            photo_url=photo_url,
                            source_url=source_url,
                            confidence=0.70,  # Lower confidence for this strategy
                        )
                    )
                    i += 2  # Skip both paragraphs
                    continue

            i += 1

        return executives

    def _extract_from_profile_links(
        self, soup: BeautifulSoup, source_url: str
    ) -> list[ScrapedExecutive]:
        """
        Extract executives from links that appear to be profile URLs.

        This is a fallback strategy that finds links matching person profile URL patterns
        and extracts names from the URL slug or link text.

        Args:
            soup: BeautifulSoup parsed HTML
            source_url: Source URL for attribution

        Returns:
            List of extracted executives
        """
        executives: list[ScrapedExecutive] = []
        seen_names: set[str] = set()

        # Pattern for profile URLs: /team/first-last/ or /people/firstname-lastname/
        profile_pattern = re.compile(
            r"/(team|people|staff|about|leadership|executives|management|bio|profile)/"
            r"([a-z]+[-_][a-z]+[-_a-z]*)/?$",
            re.I,
        )

        links = soup.find_all("a", href=profile_pattern)

        for link in links:
            href = link.get("href", "")
            match = profile_pattern.search(href)
            if not match:
                continue

            # Try to get name from link text first
            link_text = link.get_text(strip=True)
            name = None

            if link_text and self._looks_like_name(link_text):
                name = link_text
            else:
                # Extract name from URL slug
                slug = match.group(2)
                # Convert slug to name: "john-smith" -> "John Smith"
                name_parts = re.split(r"[-_]", slug)
                potential_name = " ".join(part.capitalize() for part in name_parts)
                if self._looks_like_name(potential_name):
                    name = potential_name

            if name:
                name_lower = name.lower().strip()
                if name_lower not in seen_names:
                    seen_names.add(name_lower)
                    executives.append(
                        ScrapedExecutive(
                            name=name,
                            title=None,  # Can't reliably extract title with this method
                            photo_url=None,
                            source_url=source_url,
                            confidence=0.60,  # Lower confidence for URL-based extraction
                        )
                    )

        return executives

    def _get_direct_text(self, element: Tag) -> str:
        """
        Get text directly inside an element, excluding child element text.

        Args:
            element: HTML element

        Returns:
            Direct text content stripped of whitespace
        """
        text_parts = []
        for child in element.children:
            if isinstance(child, str):
                text = child.strip()
                if text:
                    text_parts.append(text)
        return " ".join(text_parts)

    def _find_nearby_photo(self, element: Tag, source_url: str) -> str | None:
        """
        Find a photo URL near the given element.

        Args:
            element: HTML element to search around
            source_url: Base URL for resolving relative paths

        Returns:
            Photo URL or None
        """
        # Check parent for img
        parent = element.parent
        if parent:
            img = parent.find("img")
            if img and img.get("src"):
                return self._resolve_url(str(img["src"]), source_url)

        # Check previous sibling
        prev = element.find_previous_sibling()
        if prev:
            img = prev.find("img") if hasattr(prev, "find") else None
            if img and img.get("src"):
                return self._resolve_url(str(img["src"]), source_url)

        return None

    def _resolve_url(self, url: str, base_url: str) -> str:
        """
        Resolve a potentially relative URL to absolute.

        Args:
            url: URL to resolve (may be relative)
            base_url: Base URL for resolution

        Returns:
            Absolute URL
        """
        if url.startswith(("http://", "https://", "//")):
            if url.startswith("//"):
                return f"https:{url}"
            return url
        return urljoin(base_url, url)

    def _looks_like_name(self, text: str) -> bool:
        """
        Check if text looks like a person's name.

        Args:
            text: Text to check

        Returns:
            True if text looks like a name
        """
        if not text or len(text) < 3 or len(text) > 60:
            return False

        words = text.split()
        if len(words) < 2 or len(words) > 5:
            return False

        # Check if it has capital letters (proper nouns)
        if not any(c.isupper() for c in text):
            return False

        # Reject common non-name patterns
        bad_patterns = [
            "leadership",
            "team",
            "executive",
            "about",
            "our",
            "view",
            "read",
            "more",
            "learn",
            "contact",
            "click",
            "insights",
            "news",
            "blog",
            "press",
            "careers",
            "newsletter",
        ]
        text_lower = text.lower()
        for pattern in bad_patterns:
            if pattern in text_lower:
                return False

        return True

    def _looks_like_title(self, text: str) -> bool:
        """
        Check if text looks like a job title.

        Args:
            text: Text to check

        Returns:
            True if text looks like a job title
        """
        if not text or len(text) < 2 or len(text) > 150:
            return False

        # Normalize text: handle multi-line titles and extra whitespace
        normalized = " ".join(text.split())

        title_keywords = [
            # C-suite
            "ceo",
            "cfo",
            "cto",
            "coo",
            "cio",
            "cmo",
            "cpo",
            "cro",
            "chro",
            # Executive titles
            "president",
            "vice president",
            "chief",
            "officer",
            "director",
            "head of",
            "head,",
            # Management
            "manager",
            "managing",
            "general manager",
            "senior manager",
            # VP variants
            "vp",
            "svp",
            "evp",
            "avp",
            # Ownership/Leadership
            "partner",
            "founder",
            "co-founder",
            "cofounder",
            "owner",
            "chairman",
            "chairwoman",
            "chairperson",
            # Professional services
            "principal",
            "associate",
            "senior associate",
            "analyst",
            "senior analyst",
            "consultant",
            "senior consultant",
            "advisor",
            "adviser",
            "counsel",
            # Operations
            "assistant",
            "executive assistant",
            "coordinator",
            "administrator",
            "secretary",
            "treasurer",
            # Technical
            "engineer",
            "architect",
            "developer",
            "scientist",
            "researcher",
            # Finance
            "controller",
            "comptroller",
            "accountant",
            # HR
            "recruiter",
            "talent",
            "people",
            # Sales/Marketing
            "sales",
            "marketing",
            "business development",
            # Industry-specific
            "portfolio",
            "investment",
            "banking",
            "healthcare",
            "operations",
        ]

        text_lower = normalized.lower()
        return any(keyword in text_lower for keyword in title_keywords)


# Module-level service accessor
async def scrape_company_executives(website_url: str) -> list[ScrapedExecutive]:
    """
    Convenience function to scrape executives from a company website.

    Args:
        website_url: Company website URL

    Returns:
        List of scraped executives
    """
    async with WebsiteScraperService() as service:
        return await service.scrape_company_website(website_url)
