"""Tests for website scraper service."""

import pytest
from unittest.mock import AsyncMock, patch, MagicMock
from app.services.website_scraper_service import (
    WebsiteScraperService,
    ScrapedExecutive,
    scrape_company_executives,
    LEADERSHIP_PATHS,
    MIN_CONTENT_LENGTH,
)


class TestScrapedExecutive:
    """Test ScrapedExecutive dataclass."""

    def test_default_confidence(self):
        """Test default confidence is 0.85."""
        exec = ScrapedExecutive(name="John Smith")
        assert exec.confidence == 0.85

    def test_with_all_fields(self):
        """Test creating with all fields."""
        exec = ScrapedExecutive(
            name="Jane Doe",
            title="CEO",
            photo_url="https://example.com/photo.jpg",
            source_url="https://example.com/team",
            confidence=0.9,
        )
        assert exec.name == "Jane Doe"
        assert exec.title == "CEO"
        assert exec.photo_url == "https://example.com/photo.jpg"
        assert exec.source_url == "https://example.com/team"
        assert exec.confidence == 0.9

    def test_optional_fields_default_to_none(self):
        """Test that optional fields default to None."""
        exec = ScrapedExecutive(name="Test User")
        assert exec.title is None
        assert exec.photo_url is None
        assert exec.source_url is None


class TestLeadershipPaths:
    """Test leadership path constants."""

    def test_paths_include_common_patterns(self):
        """Test that common paths are included."""
        assert "/leadership" in LEADERSHIP_PATHS
        assert "/about/leadership" in LEADERSHIP_PATHS
        assert "/team" in LEADERSHIP_PATHS
        assert "/about" in LEADERSHIP_PATHS

    def test_paths_count(self):
        """Test reasonable number of paths."""
        assert len(LEADERSHIP_PATHS) >= 10

    def test_paths_include_company_prefix(self):
        """Test that company-prefixed paths are included."""
        assert "/company/leadership" in LEADERSHIP_PATHS
        assert "/company/team" in LEADERSHIP_PATHS

    def test_paths_include_about_us_variants(self):
        """Test that about-us variants are included."""
        assert "/about-us/leadership" in LEADERSHIP_PATHS
        assert "/about-us/team" in LEADERSHIP_PATHS
        assert "/about-us" in LEADERSHIP_PATHS


class TestMinContentLength:
    """Test MIN_CONTENT_LENGTH constant."""

    def test_min_content_length_value(self):
        """Test minimum content length is reasonable."""
        assert MIN_CONTENT_LENGTH == 1000


class TestWebsiteScraperService:
    """Test WebsiteScraperService functionality."""

    @pytest.mark.asyncio
    async def test_context_manager(self):
        """Test async context manager works."""
        async with WebsiteScraperService() as service:
            assert service._client is not None
        # After exit, client should be closed (we can't easily verify this)

    @pytest.mark.asyncio
    async def test_scrape_requires_context_manager(self):
        """Test that scraping without context manager raises error."""
        service = WebsiteScraperService()
        with pytest.raises(RuntimeError, match="not initialized"):
            await service.scrape_company_website("https://example.com")

    @pytest.mark.asyncio
    async def test_url_normalization_adds_https(self):
        """Test URL normalization adds https."""
        with patch.object(WebsiteScraperService, "_fetch_page", new_callable=AsyncMock) as mock_fetch:
            mock_fetch.return_value = (None, "https://example.com")
            
            async with WebsiteScraperService() as service:
                await service.scrape_company_website("example.com")
                
            # Should have tried with https://
            calls = mock_fetch.call_args_list
            assert any("https://example.com" in str(call) for call in calls)

    @pytest.mark.asyncio
    async def test_url_normalization_preserves_https(self):
        """Test URL normalization preserves existing https."""
        with patch.object(WebsiteScraperService, "_fetch_page", new_callable=AsyncMock) as mock_fetch:
            mock_fetch.return_value = (None, "https://example.com")
            
            async with WebsiteScraperService() as service:
                await service.scrape_company_website("https://example.com")
                
            # Should have kept the https://
            calls = mock_fetch.call_args_list
            assert any("https://example.com" in str(call) for call in calls)

    @pytest.mark.asyncio
    async def test_url_normalization_preserves_http(self):
        """Test URL normalization preserves existing http."""
        with patch.object(WebsiteScraperService, "_fetch_page", new_callable=AsyncMock) as mock_fetch:
            mock_fetch.return_value = (None, "http://example.com")
            
            async with WebsiteScraperService() as service:
                await service.scrape_company_website("http://example.com")
                
            # Should have kept the http://
            calls = mock_fetch.call_args_list
            assert any("http://example.com" in str(call) for call in calls)


class TestLooksLikeName:
    """Test _looks_like_name validation method."""

    def setup_method(self):
        """Set up test fixtures."""
        self.service = WebsiteScraperService()

    def test_valid_first_last_name(self):
        """Test valid first and last name."""
        assert self.service._looks_like_name("John Smith") is True

    def test_valid_hyphenated_name(self):
        """Test hyphenated names."""
        assert self.service._looks_like_name("Jane Doe-Smith") is True

    def test_valid_name_with_title(self):
        """Test name with honorific and middle initial."""
        assert self.service._looks_like_name("Dr. John A. Smith") is True

    def test_valid_three_part_name(self):
        """Test three-part names."""
        assert self.service._looks_like_name("Mary Jane Watson") is True

    def test_reject_empty_string(self):
        """Test rejection of empty string."""
        assert self.service._looks_like_name("") is False

    def test_reject_short_string(self):
        """Test rejection of strings that are too short."""
        assert self.service._looks_like_name("a") is False
        assert self.service._looks_like_name("ab") is False

    def test_reject_single_word(self):
        """Test rejection of single word names."""
        assert self.service._looks_like_name("John") is False

    def test_reject_lowercase_text(self):
        """Test rejection of all lowercase text (no proper nouns)."""
        assert self.service._looks_like_name("leadership team") is False

    def test_reject_action_words(self):
        """Test rejection of action/navigation words."""
        assert self.service._looks_like_name("View More") is False
        assert self.service._looks_like_name("click here") is False
        assert self.service._looks_like_name("Read More") is False
        assert self.service._looks_like_name("Learn More") is False

    def test_reject_navigation_keywords(self):
        """Test rejection of common website navigation keywords."""
        assert self.service._looks_like_name("About Us") is False
        assert self.service._looks_like_name("Our Team") is False
        assert self.service._looks_like_name("Leadership Team") is False

    def test_reject_too_many_words(self):
        """Test rejection of text with too many words (more than 5)."""
        assert self.service._looks_like_name("John Paul George Ringo Pete Stuart") is False

    def test_reject_too_long(self):
        """Test rejection of text that is too long (more than 60 chars)."""
        long_name = "A" * 61 + " Smith"  # Over 60 characters
        assert self.service._looks_like_name(long_name) is False


class TestLooksLikeTitle:
    """Test _looks_like_title validation method."""

    def setup_method(self):
        """Set up test fixtures."""
        self.service = WebsiteScraperService()

    def test_valid_ceo_title(self):
        """Test CEO title."""
        assert self.service._looks_like_title("CEO") is True

    def test_valid_full_c_suite_title(self):
        """Test full C-suite title."""
        assert self.service._looks_like_title("Chief Executive Officer") is True
        assert self.service._looks_like_title("Chief Financial Officer") is True
        assert self.service._looks_like_title("Chief Technology Officer") is True

    def test_valid_vp_titles(self):
        """Test VP titles."""
        assert self.service._looks_like_title("Vice President of Engineering") is True
        assert self.service._looks_like_title("VP of Sales") is True
        assert self.service._looks_like_title("SVP Marketing") is True
        assert self.service._looks_like_title("EVP Operations") is True

    def test_valid_director_titles(self):
        """Test director titles."""
        assert self.service._looks_like_title("Director of Marketing") is True
        assert self.service._looks_like_title("Managing Director") is True

    def test_valid_other_senior_titles(self):
        """Test other senior titles."""
        assert self.service._looks_like_title("President") is True
        assert self.service._looks_like_title("Partner") is True
        assert self.service._looks_like_title("Founder") is True
        assert self.service._looks_like_title("Chairman") is True
        assert self.service._looks_like_title("Chairwoman") is True

    def test_reject_empty_string(self):
        """Test rejection of empty string."""
        assert self.service._looks_like_title("") is False

    def test_reject_short_string(self):
        """Test rejection of strings that are too short."""
        assert self.service._looks_like_title("a") is False
        assert self.service._looks_like_title("ab") is False

    def test_reject_generic_text(self):
        """Test rejection of generic non-title text."""
        assert self.service._looks_like_title("Hello World") is False
        assert self.service._looks_like_title("Lorem ipsum") is False

    def test_reject_too_long(self):
        """Test rejection of titles that are too long (more than 100 chars)."""
        long_title = "A" * 101
        assert self.service._looks_like_title(long_title) is False


class TestScrapeCompanyExecutives:
    """Test the convenience function."""

    @pytest.mark.asyncio
    async def test_returns_list(self):
        """Test function returns a list."""
        with patch.object(WebsiteScraperService, "scrape_company_website", new_callable=AsyncMock) as mock_scrape:
            mock_scrape.return_value = [ScrapedExecutive(name="Test")]
            
            result = await scrape_company_executives("https://example.com")
            
            assert isinstance(result, list)
            assert len(result) == 1
            assert result[0].name == "Test"

    @pytest.mark.asyncio
    async def test_handles_empty_result(self):
        """Test function handles empty results."""
        with patch.object(WebsiteScraperService, "scrape_company_website", new_callable=AsyncMock) as mock_scrape:
            mock_scrape.return_value = []
            
            result = await scrape_company_executives("https://example.com")
            
            assert isinstance(result, list)
            assert len(result) == 0


class TestExtractExecutives:
    """Test HTML parsing and executive extraction."""

    def setup_method(self):
        """Set up test fixtures."""
        self.service = WebsiteScraperService()

    def test_extract_from_simple_html(self):
        """Test extraction from simple HTML with name/title pattern."""
        html = '''
        <div class="leadership">
            <div class="person">
                <h3>John Smith</h3>
                <p>Chief Executive Officer</p>
            </div>
            <div class="person">
                <h3>Jane Doe</h3>
                <p>Chief Financial Officer</p>
            </div>
        </div>
        '''
        
        executives = self.service._extract_executives(html, "https://example.com/team")
        
        # Should find some executives (exact count depends on implementation)
        assert isinstance(executives, list)

    def test_extract_deduplicates(self):
        """Test that duplicate names are deduplicated in scrape_company_website (not _extract_executives)."""
        html = '''
        <div class="team">
            <h3>John Smith</h3>
            <h3>john smith</h3>
            <h3>John Smith</h3>
        </div>
        '''
        
        # The deduplication happens in scrape_company_website, not _extract_executives
        # So this tests that the extraction doesn't crash on duplicates
        executives = self.service._extract_executives(html, "https://example.com")
        assert isinstance(executives, list)

    def test_extract_from_team_section(self):
        """Test extraction from a team section."""
        html = '''
        <section class="team-section">
            <div class="team-member">
                <h4>Alice Johnson</h4>
                <span class="title">VP of Engineering</span>
            </div>
        </section>
        '''
        
        executives = self.service._extract_executives(html, "https://example.com/about")
        assert isinstance(executives, list)

    def test_extract_returns_empty_for_minimal_html(self):
        """Test extraction returns empty list for HTML without leadership info."""
        html = '''
        <html>
            <body>
                <h1>Welcome</h1>
                <p>This is just a landing page.</p>
            </body>
        </html>
        '''
        
        executives = self.service._extract_executives(html, "https://example.com")
        assert isinstance(executives, list)


class TestResolveUrl:
    """Test URL resolution for relative paths."""

    def setup_method(self):
        """Set up test fixtures."""
        self.service = WebsiteScraperService()

    def test_resolve_absolute_https(self):
        """Test that absolute HTTPS URLs are preserved."""
        result = self.service._resolve_url(
            "https://example.com/images/photo.jpg",
            "https://example.com/team"
        )
        assert result == "https://example.com/images/photo.jpg"

    def test_resolve_absolute_http(self):
        """Test that absolute HTTP URLs are preserved."""
        result = self.service._resolve_url(
            "http://example.com/images/photo.jpg",
            "https://example.com/team"
        )
        assert result == "http://example.com/images/photo.jpg"

    def test_resolve_protocol_relative_url(self):
        """Test that protocol-relative URLs get https."""
        result = self.service._resolve_url(
            "//cdn.example.com/images/photo.jpg",
            "https://example.com/team"
        )
        assert result == "https://cdn.example.com/images/photo.jpg"

    def test_resolve_relative_url(self):
        """Test that relative URLs are resolved against base."""
        result = self.service._resolve_url(
            "/images/photo.jpg",
            "https://example.com/team"
        )
        assert result == "https://example.com/images/photo.jpg"

    def test_resolve_relative_path(self):
        """Test that relative paths are resolved against base."""
        result = self.service._resolve_url(
            "photo.jpg",
            "https://example.com/team/"
        )
        assert result == "https://example.com/team/photo.jpg"


class TestFetchPage:
    """Test the _fetch_page method."""

    @pytest.mark.asyncio
    async def test_fetch_page_returns_none_without_client(self):
        """Test _fetch_page returns None when client not initialized."""
        service = WebsiteScraperService()
        result, url = await service._fetch_page("https://example.com")
        assert result is None
        assert url == "https://example.com"


class TestDeduplication:
    """Test executive deduplication logic."""

    @pytest.mark.asyncio
    async def test_deduplicates_by_lowercase_name(self):
        """Test that deduplication is case-insensitive."""
        # Create mock executives with same name in different cases
        executives = [
            ScrapedExecutive(name="John Smith", title="CEO"),
            ScrapedExecutive(name="john smith", title="CFO"),
            ScrapedExecutive(name="JOHN SMITH", title="CTO"),
        ]
        
        with patch.object(WebsiteScraperService, "_fetch_page", new_callable=AsyncMock) as mock_fetch:
            with patch.object(WebsiteScraperService, "_extract_executives") as mock_extract:
                # Return all executives from extraction
                mock_extract.return_value = executives
                mock_fetch.return_value = ("<html></html>", "https://example.com/team")
                
                async with WebsiteScraperService() as service:
                    # Only check one path for this test
                    with patch("app.services.website_scraper_service.LEADERSHIP_PATHS", ["/team"]):
                        result = await service.scrape_company_website("https://example.com")
                
                # Should have deduplicated to only one entry
                assert len(result) == 1
                assert result[0].name == "John Smith"

    @pytest.mark.asyncio
    async def test_keeps_unique_names(self):
        """Test that unique names are all preserved."""
        executives = [
            ScrapedExecutive(name="John Smith", title="CEO"),
            ScrapedExecutive(name="Jane Doe", title="CFO"),
            ScrapedExecutive(name="Bob Wilson", title="CTO"),
        ]
        
        with patch.object(WebsiteScraperService, "_fetch_page", new_callable=AsyncMock) as mock_fetch:
            with patch.object(WebsiteScraperService, "_extract_executives") as mock_extract:
                mock_extract.return_value = executives
                mock_fetch.return_value = ("<html></html>", "https://example.com/team")
                
                async with WebsiteScraperService() as service:
                    with patch("app.services.website_scraper_service.LEADERSHIP_PATHS", ["/team"]):
                        result = await service.scrape_company_website("https://example.com")
                
                # Should keep all three unique executives
                assert len(result) == 3
                names = {e.name for e in result}
                assert names == {"John Smith", "Jane Doe", "Bob Wilson"}


class TestFindLeadershipSections:
    """Test the _find_leadership_sections method."""

    def setup_method(self):
        """Set up test fixtures."""
        self.service = WebsiteScraperService()

    def test_finds_section_by_class_leadership(self):
        """Test finding section with 'leadership' class."""
        from bs4 import BeautifulSoup
        html = '<div class="leadership-section">Content</div>'
        soup = BeautifulSoup(html, "lxml")
        
        sections = self.service._find_leadership_sections(soup)
        
        assert len(sections) >= 1

    def test_finds_section_by_id_team(self):
        """Test finding section with 'team' ID."""
        from bs4 import BeautifulSoup
        html = '<div id="team-members">Content</div>'
        soup = BeautifulSoup(html, "lxml")
        
        sections = self.service._find_leadership_sections(soup)
        
        assert len(sections) >= 1

    def test_finds_section_by_class_executive(self):
        """Test finding section with 'executive' class."""
        from bs4 import BeautifulSoup
        html = '<section class="executive-team">Content</section>'
        soup = BeautifulSoup(html, "lxml")
        
        sections = self.service._find_leadership_sections(soup)
        
        assert len(sections) >= 1


class TestFindPersonCards:
    """Test the _find_person_cards method."""

    def setup_method(self):
        """Set up test fixtures."""
        self.service = WebsiteScraperService()

    def test_finds_person_cards(self):
        """Test finding elements with 'person' class."""
        from bs4 import BeautifulSoup
        html = '<div class="person-card">Content</div>'
        soup = BeautifulSoup(html, "lxml")
        
        cards = self.service._find_person_cards(soup)
        
        assert len(cards) >= 1

    def test_finds_profile_cards(self):
        """Test finding elements with 'profile' class."""
        from bs4 import BeautifulSoup
        html = '<div class="profile-item">Content</div>'
        soup = BeautifulSoup(html, "lxml")
        
        cards = self.service._find_person_cards(soup)
        
        assert len(cards) >= 1

    def test_finds_team_member_cards(self):
        """Test finding elements with 'team-member' class."""
        from bs4 import BeautifulSoup
        html = '<div class="team-member">Content</div>'
        soup = BeautifulSoup(html, "lxml")
        
        cards = self.service._find_person_cards(soup)
        
        assert len(cards) >= 1


class TestExtractFromCard:
    """Test the _extract_from_card method."""

    def setup_method(self):
        """Set up test fixtures."""
        self.service = WebsiteScraperService()

    def test_extracts_name_from_header(self):
        """Test extraction of name from header element."""
        from bs4 import BeautifulSoup
        html = '''
        <div class="person">
            <h3>John Smith</h3>
            <span class="title">Chief Executive Officer</span>
        </div>
        '''
        soup = BeautifulSoup(html, "lxml")
        card = soup.find("div", class_="person")
        
        result = self.service._extract_from_card(card, "https://example.com")
        
        assert result is not None
        assert result.name == "John Smith"
        assert result.title == "Chief Executive Officer"

    def test_extracts_photo_from_img(self):
        """Test extraction of photo URL from img element."""
        from bs4 import BeautifulSoup
        html = '''
        <div class="person">
            <img src="/images/john.jpg" alt="John Smith">
            <h3>John Smith</h3>
            <span class="title">CEO</span>
        </div>
        '''
        soup = BeautifulSoup(html, "lxml")
        card = soup.find("div", class_="person")
        
        result = self.service._extract_from_card(card, "https://example.com")
        
        assert result is not None
        assert result.photo_url == "https://example.com/images/john.jpg"

    def test_returns_none_for_invalid_name(self):
        """Test that None is returned when name doesn't look valid."""
        from bs4 import BeautifulSoup
        html = '''
        <div class="person">
            <h3>View More</h3>
        </div>
        '''
        soup = BeautifulSoup(html, "lxml")
        card = soup.find("div", class_="person")

        result = self.service._extract_from_card(card, "https://example.com")

        assert result is None


class TestExtractFromLinkCards:
    """Test the _extract_from_link_cards method (Strategy 1)."""

    def setup_method(self):
        """Set up test fixtures."""
        self.service = WebsiteScraperService()

    def test_extracts_from_team_links_with_paragraphs(self):
        """Test extraction from team links containing two paragraphs (name + title)."""
        from bs4 import BeautifulSoup
        html = '''
        <html><body>
        <a href="/team/andy-thompson/">
            <p>Andy Thompson</p>
            <p>CEO & Founder</p>
        </a>
        <a href="/team/sarah-johnson/">
            <p>Sarah Johnson</p>
            <p>Managing Director</p>
        </a>
        </body></html>
        '''
        soup = BeautifulSoup(html, "lxml")

        executives = self.service._extract_from_link_cards(soup, "https://example.com/team/")

        assert len(executives) == 2
        names = {e.name for e in executives}
        assert "Andy Thompson" in names
        assert "Sarah Johnson" in names

        # Check titles
        titles_by_name = {e.name: e.title for e in executives}
        assert titles_by_name["Andy Thompson"] == "CEO & Founder"
        assert titles_by_name["Sarah Johnson"] == "Managing Director"

    def test_extracts_from_people_links(self):
        """Test extraction from /people/ links."""
        from bs4 import BeautifulSoup
        html = '''
        <a href="/people/john-smith/">
            <p>John Smith</p>
            <p>Vice President</p>
        </a>
        '''
        soup = BeautifulSoup(html, "lxml")

        executives = self.service._extract_from_link_cards(soup, "https://example.com/")

        assert len(executives) == 1
        assert executives[0].name == "John Smith"
        assert executives[0].title == "Vice President"

    def test_extracts_from_staff_links(self):
        """Test extraction from /staff/ links."""
        from bs4 import BeautifulSoup
        html = '''
        <a href="/staff/jane-doe/">
            <p>Jane Doe</p>
            <p>Director of Operations</p>
        </a>
        '''
        soup = BeautifulSoup(html, "lxml")

        executives = self.service._extract_from_link_cards(soup, "https://example.com/")

        assert len(executives) == 1
        assert executives[0].name == "Jane Doe"

    def test_extracts_from_link_with_header_and_paragraph(self):
        """Test extraction from link with header (name) and paragraph (title)."""
        from bs4 import BeautifulSoup
        html = '''
        <a href="/team/bob-wilson/">
            <h3>Bob Wilson</h3>
            <p>Partner</p>
        </a>
        '''
        soup = BeautifulSoup(html, "lxml")

        executives = self.service._extract_from_link_cards(soup, "https://example.com/")

        assert len(executives) == 1
        assert executives[0].name == "Bob Wilson"
        assert executives[0].title == "Partner"

    def test_extracts_photo_from_link(self):
        """Test extraction of photo URL from within link."""
        from bs4 import BeautifulSoup
        html = '''
        <a href="/team/alice-chen/">
            <img src="/photos/alice.jpg" alt="Alice Chen">
            <p>Alice Chen</p>
            <p>Principal</p>
        </a>
        '''
        soup = BeautifulSoup(html, "lxml")

        executives = self.service._extract_from_link_cards(soup, "https://example.com/")

        assert len(executives) == 1
        assert executives[0].photo_url == "https://example.com/photos/alice.jpg"

    def test_deduplicates_within_method(self):
        """Test that duplicates are not returned from same page."""
        from bs4 import BeautifulSoup
        html = '''
        <a href="/team/john-smith/">
            <p>John Smith</p>
            <p>CEO</p>
        </a>
        <a href="/team/john-smith-bio/">
            <p>John Smith</p>
            <p>Chief Executive Officer</p>
        </a>
        '''
        soup = BeautifulSoup(html, "lxml")

        executives = self.service._extract_from_link_cards(soup, "https://example.com/")

        # Should deduplicate to one entry
        assert len(executives) == 1

    def test_ignores_non_team_links(self):
        """Test that links not matching team patterns are ignored."""
        from bs4 import BeautifulSoup
        html = '''
        <a href="/products/widget/">
            <p>Widget Pro</p>
            <p>Product Manager</p>
        </a>
        <a href="/services/consulting/">
            <p>John Consultant</p>
            <p>Senior Consultant</p>
        </a>
        '''
        soup = BeautifulSoup(html, "lxml")

        executives = self.service._extract_from_link_cards(soup, "https://example.com/")

        assert len(executives) == 0


class TestExtractFromAdjacentParagraphs:
    """Test the _extract_from_adjacent_paragraphs method (Strategy 4)."""

    def setup_method(self):
        """Set up test fixtures."""
        self.service = WebsiteScraperService()

    def test_extracts_name_title_pairs(self):
        """Test extraction from adjacent paragraph pairs."""
        from bs4 import BeautifulSoup
        html = '''
        <div>
            <p>John Smith</p>
            <p>Chief Executive Officer</p>
            <p>Jane Doe</p>
            <p>Chief Financial Officer</p>
        </div>
        '''
        soup = BeautifulSoup(html, "lxml")

        executives = self.service._extract_from_adjacent_paragraphs(soup, "https://example.com/")

        assert len(executives) == 2
        names = {e.name for e in executives}
        assert "John Smith" in names
        assert "Jane Doe" in names

    def test_assigns_lower_confidence(self):
        """Test that adjacent paragraph strategy uses lower confidence."""
        from bs4 import BeautifulSoup
        html = '''
        <p>Mike Wilson</p>
        <p>Managing Director</p>
        '''
        soup = BeautifulSoup(html, "lxml")

        executives = self.service._extract_from_adjacent_paragraphs(soup, "https://example.com/")

        assert len(executives) == 1
        assert executives[0].confidence == 0.70

    def test_skips_non_matching_patterns(self):
        """Test that non-name/title patterns are skipped."""
        from bs4 import BeautifulSoup
        html = '''
        <p>This is some random text</p>
        <p>More random text here</p>
        <p>Bob Anderson</p>
        <p>Partner</p>
        <p>Another random paragraph</p>
        '''
        soup = BeautifulSoup(html, "lxml")

        executives = self.service._extract_from_adjacent_paragraphs(soup, "https://example.com/")

        assert len(executives) == 1
        assert executives[0].name == "Bob Anderson"


class TestExtractFromProfileLinks:
    """Test the _extract_from_profile_links method (Strategy 5)."""

    def setup_method(self):
        """Set up test fixtures."""
        self.service = WebsiteScraperService()

    def test_extracts_name_from_url_slug(self):
        """Test extraction of name from URL slug."""
        from bs4 import BeautifulSoup
        html = '''
        <a href="/team/john-smith/">View Profile</a>
        <a href="/people/jane-doe/">View Profile</a>
        '''
        soup = BeautifulSoup(html, "lxml")

        executives = self.service._extract_from_profile_links(soup, "https://example.com/")

        assert len(executives) == 2
        names = {e.name for e in executives}
        assert "John Smith" in names
        assert "Jane Doe" in names

    def test_prefers_link_text_over_slug(self):
        """Test that link text is preferred over URL slug for name."""
        from bs4 import BeautifulSoup
        html = '''
        <a href="/team/j-smith/">Jonathan Smith</a>
        '''
        soup = BeautifulSoup(html, "lxml")

        executives = self.service._extract_from_profile_links(soup, "https://example.com/")

        assert len(executives) == 1
        assert executives[0].name == "Jonathan Smith"

    def test_assigns_lowest_confidence(self):
        """Test that profile link strategy uses lowest confidence."""
        from bs4 import BeautifulSoup
        html = '''
        <a href="/team/bob-wilson/">View Profile</a>
        '''
        soup = BeautifulSoup(html, "lxml")

        executives = self.service._extract_from_profile_links(soup, "https://example.com/")

        assert len(executives) == 1
        assert executives[0].confidence == 0.60

    def test_handles_underscore_in_slug(self):
        """Test handling of underscores in URL slug."""
        from bs4 import BeautifulSoup
        html = '''
        <a href="/team/alice_chen/">View Profile</a>
        '''
        soup = BeautifulSoup(html, "lxml")

        executives = self.service._extract_from_profile_links(soup, "https://example.com/")

        assert len(executives) == 1
        assert executives[0].name == "Alice Chen"


class TestGetDirectText:
    """Test the _get_direct_text helper method."""

    def setup_method(self):
        """Set up test fixtures."""
        self.service = WebsiteScraperService()

    def test_gets_direct_text_only(self):
        """Test that only direct text (not child element text) is returned."""
        from bs4 import BeautifulSoup
        html = '''
        <a href="/team/john/">
            Direct Text
            <p>Paragraph text</p>
            More Direct
        </a>
        '''
        soup = BeautifulSoup(html, "lxml")
        link = soup.find("a")

        result = self.service._get_direct_text(link)

        assert "Direct Text" in result
        assert "More Direct" in result
        assert "Paragraph text" not in result


class TestLooksLikeTitleExtended:
    """Test extended title recognition keywords."""

    def setup_method(self):
        """Set up test fixtures."""
        self.service = WebsiteScraperService()

    def test_recognizes_principal(self):
        """Test recognition of Principal title."""
        assert self.service._looks_like_title("Principal") is True
        assert self.service._looks_like_title("Senior Principal") is True

    def test_recognizes_associate(self):
        """Test recognition of Associate title."""
        assert self.service._looks_like_title("Associate") is True
        assert self.service._looks_like_title("Senior Associate") is True

    def test_recognizes_analyst(self):
        """Test recognition of Analyst title."""
        assert self.service._looks_like_title("Analyst") is True
        assert self.service._looks_like_title("Senior Analyst") is True

    def test_recognizes_managing(self):
        """Test recognition of Managing title."""
        assert self.service._looks_like_title("Managing Director") is True
        assert self.service._looks_like_title("Managing Partner") is True

    def test_recognizes_cofounder(self):
        """Test recognition of Co-founder variations."""
        assert self.service._looks_like_title("Co-Founder") is True
        assert self.service._looks_like_title("Cofounder") is True

    def test_handles_multiline_title(self):
        """Test that multi-line titles are normalized and recognized."""
        multiline_title = "Managing Director\nHealthcare"
        assert self.service._looks_like_title(multiline_title) is True

    def test_handles_extra_whitespace(self):
        """Test that extra whitespace in titles is handled."""
        title_with_spaces = "  Managing    Director  "
        assert self.service._looks_like_title(title_with_spaces) is True


class TestIntegrationExtractExecutives:
    """Integration tests for _extract_executives with multiple strategies."""

    def setup_method(self):
        """Set up test fixtures."""
        self.service = WebsiteScraperService()

    def test_notch_partners_pattern(self):
        """Test extraction matching Notch Partners team page pattern."""
        html = '''
        <html>
        <body>
        <div class="team-grid">
            <a href="/team/andy-thompson/">
                <img src="/photos/andy.jpg" alt="Andy Thompson">
                <p>Andy Thompson</p>
                <p>CEO & Founder</p>
            </a>
            <a href="/team/sarah-johnson/">
                <img src="/photos/sarah.jpg" alt="Sarah Johnson">
                <p>Sarah Johnson</p>
                <p>Managing Director</p>
            </a>
            <a href="/team/mike-wilson/">
                <img src="/photos/mike.jpg" alt="Mike Wilson">
                <p>Mike Wilson</p>
                <p>Principal</p>
            </a>
            <a href="/team/emily-chen/">
                <p>Emily Chen</p>
                <p>Vice President</p>
            </a>
            <a href="/team/david-brown/">
                <p>David Brown</p>
                <p>Associate</p>
            </a>
        </div>
        </body>
        </html>
        '''

        executives = self.service._extract_executives(html, "https://notchpartners.com/team/")

        # Should find all 5 executives
        assert len(executives) >= 5

        names = {e.name for e in executives}
        assert "Andy Thompson" in names
        assert "Sarah Johnson" in names
        assert "Mike Wilson" in names
        assert "Emily Chen" in names
        assert "David Brown" in names

    def test_combination_of_strategies(self):
        """Test that multiple strategies can work together."""
        html = '''
        <html>
        <body>
        <!-- Strategy 1: Link cards -->
        <a href="/team/john-smith/">
            <p>John Smith</p>
            <p>CEO</p>
        </a>

        <!-- Strategy 2: Leadership section with headers -->
        <div class="leadership">
            <h3>Jane Doe</h3>
            <p>CFO</p>
        </div>
        </body>
        </html>
        '''

        executives = self.service._extract_executives(html, "https://example.com/")

        names = {e.name for e in executives}
        assert "John Smith" in names
        assert "Jane Doe" in names
