#!/usr/bin/env python3
"""
LM Studio Bridge - File-Based Task Orchestrator
================================================

Bridges Claude Code CLI with LM Studio via file-based message passing.
Similar to the Claude ↔ Gemini CLI Bridge pattern.

Architecture:
    Claude Code → lm_task.json → lm_bridge.py → LM Studio API → lm_results.json → Claude Code

Usage:
    # One-shot mode (process single task)
    python lm_bridge.py

    # Watch mode (continuous monitoring)
    python lm_bridge.py --watch

    # With tool calling enabled
    python lm_bridge.py --watch --tools

    # Process specific task file
    python lm_bridge.py --task my_task.json

Requirements:
    - LM Studio running on port 1235 with a model loaded
    - requests library

Author: Claude Code
Created: 2026-01-05
"""

import json
import time
import hashlib
import argparse
import requests
import re
import sqlite3
from pathlib import Path
from datetime import datetime, timedelta
from typing import Dict, Any, Optional, List
from dataclasses import dataclass, asdict

# =============================================================================
# Configuration
# =============================================================================

VERSION = "1.5.0"
BUILD_DATE = "2026-01-10"

# Default paths (in cli_bridge directory)
BRIDGE_DIR = Path(__file__).parent
DEFAULT_TASK_FILE = BRIDGE_DIR / "lm_task.json"
DEFAULT_RESULTS_FILE = BRIDGE_DIR / "lm_results.json"
DEFAULT_STATE_FILE = BRIDGE_DIR / "lm_bridge_state.json"
DEFAULT_CACHE_FILE = BRIDGE_DIR / "lm_scrape_cache.json"

# Database path
SQLITE_DB_PATH = Path("/home/bigdogla/projects/Ryan_Mark/exactlyai_industry_mapping.db")

# Cache configuration
CACHE_TTL_HOURS = 24  # How long cached results remain valid

# LM Studio configuration
LM_STUDIO_ENDPOINT = "http://192.168.4.77:1235"
LM_STUDIO_MODEL = "hermes-2-pro-llama-3-8b"  # Better tool calling than GLM

# Tool definitions for web scraping and analysis
WEB_TOOLS = [
    {
        "type": "function",
        "function": {
            "name": "fetch_webpage",
            "description": "Fetches and parses content from a webpage URL. Returns title, headings, paragraphs, and metadata. Uses tiered fallback (curl -> Playwright -> crawl4ai) for best results.",
            "parameters": {
                "type": "object",
                "properties": {
                    "url": {
                        "type": "string",
                        "description": "The full URL to fetch (must include https:// or http://)"
                    }
                },
                "required": ["url"]
            }
        }
    },
    {
        "type": "function",
        "function": {
            "name": "search_page_content",
            "description": "Searches for specific text within a fetched webpage",
            "parameters": {
                "type": "object",
                "properties": {
                    "url": {"type": "string", "description": "The URL to search within"},
                    "search_term": {"type": "string", "description": "The text to search for"}
                },
                "required": ["url", "search_term"]
            }
        }
    },
    {
        "type": "function",
        "function": {
            "name": "extract_nav_map",
            "description": "Extracts and scores navigation links from a website homepage. Returns ranked URLs for leadership, services, and about pages. Use this FIRST to discover the best URLs before fetching specific pages.",
            "parameters": {
                "type": "object",
                "properties": {
                    "url": {
                        "type": "string",
                        "description": "The homepage URL to analyze (e.g., https://company.com)"
                    }
                },
                "required": ["url"]
            }
        }
    },
    {
        "type": "function",
        "function": {
            "name": "agentic_scrape",
            "description": "Goal-driven scraping that automatically finds and extracts specific page content. Uses intelligent URL discovery and retry logic. Best for finding leadership/team, services, or about information.",
            "parameters": {
                "type": "object",
                "properties": {
                    "url": {
                        "type": "string",
                        "description": "The company website URL (e.g., https://company.com)"
                    },
                    "page_type": {
                        "type": "string",
                        "enum": ["leadership", "services", "about"],
                        "description": "The type of page/content to find: 'leadership' for team/executives, 'services' for offerings/products, 'about' for company info"
                    },
                    "company_name": {
                        "type": "string",
                        "description": "The company name (for logging and tracking)"
                    }
                },
                "required": ["url", "page_type", "company_name"]
            }
        }
    },
    {
        "type": "function",
        "function": {
            "name": "research_company",
            "description": "Comprehensive company research - extracts navigation map AND scrapes all three page types (about, services, leadership) in one call. Returns combined results with quality scores. Use this instead of multiple separate calls.",
            "parameters": {
                "type": "object",
                "properties": {
                    "url": {
                        "type": "string",
                        "description": "The company website URL (e.g., https://company.com)"
                    },
                    "company_name": {
                        "type": "string",
                        "description": "The company name"
                    }
                },
                "required": ["url", "company_name"]
            }
        }
    },
    {
        "type": "function",
        "function": {
            "name": "save_to_database",
            "description": "Saves scraped company research results to the SQLite database. Call this after research_company to persist results.",
            "parameters": {
                "type": "object",
                "properties": {
                    "company_name": {
                        "type": "string",
                        "description": "The company name"
                    },
                    "website": {
                        "type": "string",
                        "description": "The company website URL"
                    },
                    "about_content": {
                        "type": "string",
                        "description": "Scraped about page content"
                    },
                    "services_content": {
                        "type": "string",
                        "description": "Scraped services page content"
                    },
                    "leadership_content": {
                        "type": "string",
                        "description": "Scraped leadership page content"
                    }
                },
                "required": ["company_name", "website"]
            }
        }
    }
]


# =============================================================================
# Data Classes
# =============================================================================

@dataclass
class BridgeTask:
    """Task structure for the bridge"""
    task_id: str
    task_type: str  # "chat", "analyze", "scrape", "custom"
    prompt: str
    context: Optional[str] = None
    tools_enabled: bool = False
    max_tokens: int = 1000
    temperature: float = 0.7
    created_at: str = ""
    source: str = "claude_code"

    def __post_init__(self):
        if not self.created_at:
            self.created_at = datetime.now().isoformat()
        if not self.task_id:
            self.task_id = self._generate_id()

    def _generate_id(self) -> str:
        """Generate unique task ID"""
        content = f"{self.prompt}{self.created_at}"
        return hashlib.md5(content.encode()).hexdigest()[:12]


@dataclass
class BridgeResult:
    """Result structure from the bridge"""
    task_id: str
    status: str  # "success", "error", "timeout"
    response: str
    model: str
    tokens_used: int
    processing_time: float
    tool_calls: List[Dict] = None
    error_message: Optional[str] = None
    completed_at: str = ""

    def __post_init__(self):
        if not self.completed_at:
            self.completed_at = datetime.now().isoformat()
        if self.tool_calls is None:
            self.tool_calls = []


# =============================================================================
# Web Scraping Tools (for tool calling)
# =============================================================================

# Import robust tiered scraper
try:
    from multipage_scraper import fetch_url_tiered, fetch_url_tiered_recovery
    ROBUST_SCRAPER_AVAILABLE = True
except ImportError:
    ROBUST_SCRAPER_AVAILABLE = False
    print("Warning: multipage_scraper not available - using basic requests")

# Import nav extractor
try:
    from nav_extractor import extract_nav_map as _extract_nav_map
    NAV_EXTRACTOR_AVAILABLE = True
except ImportError:
    NAV_EXTRACTOR_AVAILABLE = False
    print("Warning: nav_extractor not available")

# Import agentic scraper
try:
    from blackjack_scraper_v3_agentic import agentic_deal as _agentic_deal
    AGENTIC_SCRAPER_AVAILABLE = True
except ImportError:
    AGENTIC_SCRAPER_AVAILABLE = False
    print("Warning: blackjack_scraper_v3_agentic not available")


class WebTools:
    """Tool implementations for web scraping - uses robust tiered scraper"""

    def __init__(self, timeout: int = 30, use_recovery_mode: bool = True):
        self.timeout = timeout
        self.use_recovery_mode = use_recovery_mode  # Try all methods, return best (default: True for better quality)
        self._cache: Dict[str, str] = {}

    def fetch_webpage(self, url: str) -> str:
        """
        Fetch and parse a webpage using tiered fallback:
        Tier 0: curl (fastest)
        Tier 1: Playwright (JS rendering)
        Tier 2: crawl4ai (fallback)
        """
        if url in self._cache:
            return self._cache[url]

        # Normalize URL
        if not url.startswith(('http://', 'https://')):
            url = 'https://' + url

        # Use robust scraper if available
        if ROBUST_SCRAPER_AVAILABLE:
            return self._fetch_with_tiered_scraper(url)
        else:
            return self._fetch_with_requests(url)

    def _fetch_with_tiered_scraper(self, url: str) -> str:
        """Use the robust multipage_scraper with curl/playwright/crawl4ai fallback"""
        try:
            # Choose scraper mode
            if self.use_recovery_mode:
                result = fetch_url_tiered_recovery(url, timeout=self.timeout)
            else:
                result = fetch_url_tiered(url, timeout=self.timeout)

            if result['success'] and result.get('text'):
                text = result['text']
                method = result.get('method', 'unknown')
                chars = result.get('chars', len(text))

                # Format result
                output = f"""URL: {url}
Status: SUCCESS
Method: {method} (tiered fallback)
Content Length: {chars} chars

Content:
{text[:5000]}"""

                self._cache[url] = output
                return output
            else:
                error = result.get('error', 'Unknown error')
                attempts = result.get('attempts', [])
                return f"Error fetching {url}: {error}\nAttempts: {attempts}"

        except Exception as e:
            return f"Error fetching {url}: {e}"

    def _fetch_with_requests(self, url: str) -> str:
        """Fallback: basic requests-based fetch (original implementation)"""
        try:
            headers = {
                "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36"
            }
            resp = requests.get(url, timeout=self.timeout, headers=headers)
            html = resp.text

            # Simple parsing
            title = self._extract(r'<title>(.*?)</title>', html)
            meta_desc = self._extract(r'<meta[^>]*name=["\']description["\'][^>]*content=["\'](.*?)["\']', html)
            h1 = self._extract(r'<h1[^>]*>(.*?)</h1>', html)
            h1 = re.sub(r'<[^>]+>', '', h1)

            # Get headings
            headings = []
            for tag in ['h1', 'h2', 'h3']:
                matches = re.findall(rf'<{tag}[^>]*>(.*?)</{tag}>', html, re.I | re.S)
                for m in matches[:3]:
                    clean = re.sub(r'<[^>]+>', '', m).strip()
                    if clean and len(clean) < 200:
                        headings.append(f"{tag.upper()}: {clean}")

            # Raw text
            raw = re.sub(r'<script[^>]*>.*?</script>', '', html, flags=re.I | re.S)
            raw = re.sub(r'<style[^>]*>.*?</style>', '', raw, flags=re.I | re.S)
            raw = re.sub(r'<[^>]+>', ' ', raw)
            raw = re.sub(r'\s+', ' ', raw).strip()[:3000]

            result = f"""URL: {url}
Status: {resp.status_code}
Method: requests (basic)
Title: {title}
Meta Description: {meta_desc}
H1: {h1}

Headings:
{chr(10).join(f'- {h}' for h in headings[:10])}

Content:
{raw}"""

            self._cache[url] = result
            return result

        except Exception as e:
            return f"Error fetching {url}: {e}"

    def search_page_content(self, url: str, search_term: str) -> str:
        """Search for content within a page"""
        content = self.fetch_webpage(url)
        if content.startswith("Error"):
            return content

        results = []
        search_lower = search_term.lower()

        for line in content.split('\n'):
            if search_lower in line.lower():
                results.append(f"Found: {line[:200]}")

        if results:
            return f"Search results for '{search_term}':\n" + "\n".join(results[:5])
        else:
            return f"No matches found for '{search_term}'"

    def _extract(self, pattern: str, text: str) -> str:
        """Extract first regex match"""
        match = re.search(pattern, text, re.I | re.S)
        return match.group(1).strip() if match else ""

    def extract_nav_map(self, url: str) -> str:
        """
        Extract and score navigation links from a homepage.
        Returns ranked URLs for leadership, services, and about pages.
        """
        if not NAV_EXTRACTOR_AVAILABLE:
            return "Error: nav_extractor module not available"

        # Normalize URL
        if not url.startswith(('http://', 'https://')):
            url = 'https://' + url

        # First fetch the homepage HTML
        try:
            if ROBUST_SCRAPER_AVAILABLE:
                result = fetch_url_tiered_recovery(url, timeout=self.timeout)
                if not result['success']:
                    return f"Error fetching {url}: {result.get('error', 'Unknown error')}"
                html_content = result.get('html', result.get('text', ''))
            else:
                headers = {"User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36"}
                resp = requests.get(url, timeout=self.timeout, headers=headers)
                html_content = resp.text

            # Extract nav map
            nav_result = _extract_nav_map(html_content, url)

            # Note: success=False means HTML parsing failed, but fallback URLs may still be available
            nav_map = nav_result.get('nav_map', {})
            if not nav_map or all(len(nav_map.get(k, [])) == 0 for k in ['leadership', 'services', 'about']):
                return f"Nav extraction failed for {url} - no URLs found"

            # Format results
            nav_map = nav_result.get('nav_map', {})
            stats = nav_result.get('stats', {})

            output_lines = [
                f"Navigation Map for: {url}",
                f"Total links found: {stats.get('total_links', 0)}",
                f"Scored links: {stats.get('scored_links', 0)}",
                ""
            ]

            for intent in ['leadership', 'services', 'about']:
                urls = nav_map.get(intent, [])
                output_lines.append(f"=== {intent.upper()} URLs (top 5) ===")
                if urls:
                    for i, item in enumerate(urls[:5], 1):
                        score = item.get('score', 0)
                        link_url = item.get('url', '')
                        link_text = item.get('link_text', '')[:50]
                        output_lines.append(f"  {i}. [{score:.1f}] {link_url}")
                        if link_text:
                            output_lines.append(f"       Text: \"{link_text}\"")
                else:
                    output_lines.append("  (no URLs found)")
                output_lines.append("")

            return "\n".join(output_lines)

        except Exception as e:
            return f"Error extracting nav map from {url}: {e}"

    def agentic_scrape(self, url: str, page_type: str, company_name: str) -> str:
        """
        Goal-driven scraping that automatically finds and extracts specific page content.
        Uses intelligent URL discovery and retry logic.
        """
        if not AGENTIC_SCRAPER_AVAILABLE:
            return "Error: blackjack_scraper_v3_agentic module not available"

        # Normalize URL
        if not url.startswith(('http://', 'https://')):
            url = 'https://' + url

        # Build the card structure expected by agentic_deal
        card = {
            'website': url,
            'page_type': page_type,
            'company_name': company_name
        }

        try:
            # Call agentic_deal with optional nav_map (will discover if needed)
            result, metrics = _agentic_deal(card, nav_map=None, max_attempts=5)

            # Format the result
            outcome = result.get('outcome', 'UNKNOWN')
            scraped_url = result.get('url', url)
            content = result.get('content', '')
            chars = result.get('chars', 0)
            method = result.get('method', 'unknown')
            attempts = metrics.attempts if hasattr(metrics, 'attempts') else 0

            if outcome == 'WIN':
                output = f"""Agentic Scrape SUCCESS for {company_name}
Page Type: {page_type}
URL Found: {scraped_url}
Method: {method}
Attempts: {attempts}
Content Length: {chars} chars

Content Preview:
{content[:3000]}"""
            else:
                output = f"""Agentic Scrape FAILED for {company_name}
Page Type: {page_type}
Outcome: {outcome}
Attempts: {attempts}
Error: {result.get('error', 'Could not find valid content')}"""

            return output

        except Exception as e:
            return f"Error in agentic scrape for {company_name} ({page_type}): {e}"

    def _load_cache(self) -> Dict:
        """Load the scrape cache from file"""
        if DEFAULT_CACHE_FILE.exists():
            try:
                return json.loads(DEFAULT_CACHE_FILE.read_text())
            except:
                pass
        return {}

    def _save_cache(self, cache: Dict):
        """Save the scrape cache to file"""
        DEFAULT_CACHE_FILE.write_text(json.dumps(cache, indent=2))

    def _get_cached_result(self, url: str, page_type: str) -> Optional[Dict]:
        """Get cached result if valid (within TTL)"""
        cache = self._load_cache()
        cache_key = f"{url}:{page_type}"

        if cache_key in cache:
            entry = cache[cache_key]
            cached_time = datetime.fromisoformat(entry.get('cached_at', '2000-01-01'))
            if datetime.now() - cached_time < timedelta(hours=CACHE_TTL_HOURS):
                return entry
        return None

    def _set_cached_result(self, url: str, page_type: str, result: Dict):
        """Cache a scrape result"""
        cache = self._load_cache()
        cache_key = f"{url}:{page_type}"
        result['cached_at'] = datetime.now().isoformat()
        cache[cache_key] = result
        self._save_cache(cache)

    def research_company(self, url: str, company_name: str) -> str:
        """
        Comprehensive company research - scrapes all three page types in one call.
        Uses caching to avoid redundant scrapes.
        """
        if not AGENTIC_SCRAPER_AVAILABLE:
            return "Error: blackjack_scraper_v3_agentic module not available"

        # Normalize URL
        if not url.startswith(('http://', 'https://')):
            url = 'https://' + url

        results = {
            'company_name': company_name,
            'website': url,
            'pages': {},
            'summary': {
                'total_chars': 0,
                'pages_found': 0,
                'pages_failed': 0
            }
        }

        page_types = ['about', 'services', 'leadership']

        for page_type in page_types:
            # Check cache first
            cached = self._get_cached_result(url, page_type)
            if cached and cached.get('outcome') == 'WIN':
                results['pages'][page_type] = {
                    'status': 'CACHED',
                    'url': cached.get('url', ''),
                    'chars': cached.get('chars', 0),
                    'content_preview': cached.get('content', '')[:500]
                }
                results['summary']['total_chars'] += cached.get('chars', 0)
                results['summary']['pages_found'] += 1
                continue

            # Build card for agentic_deal
            card = {
                'website': url,
                'page_type': page_type,
                'company_name': company_name
            }

            try:
                result, metrics = _agentic_deal(card, nav_map=None, max_attempts=5)
                outcome = result.get('outcome', 'UNKNOWN')

                if outcome == 'WIN':
                    # Cache successful result
                    self._set_cached_result(url, page_type, result)

                    results['pages'][page_type] = {
                        'status': 'SUCCESS',
                        'url': result.get('url', ''),
                        'chars': result.get('chars', 0),
                        'method': result.get('discovery_method', 'unknown'),
                        'content_preview': result.get('content', '')[:500]
                    }
                    results['summary']['total_chars'] += result.get('chars', 0)
                    results['summary']['pages_found'] += 1
                else:
                    results['pages'][page_type] = {
                        'status': 'FAILED',
                        'outcome': outcome,
                        'error': result.get('error', 'No content found')
                    }
                    results['summary']['pages_failed'] += 1

            except Exception as e:
                results['pages'][page_type] = {
                    'status': 'ERROR',
                    'error': str(e)
                }
                results['summary']['pages_failed'] += 1

        # Format output
        output_lines = [
            f"=== COMPANY RESEARCH: {company_name} ===",
            f"Website: {url}",
            f"Pages Found: {results['summary']['pages_found']}/3",
            f"Total Content: {results['summary']['total_chars']} chars",
            ""
        ]

        for page_type, data in results['pages'].items():
            status = data.get('status', 'UNKNOWN')
            output_lines.append(f"--- {page_type.upper()} ---")
            output_lines.append(f"Status: {status}")

            if status in ['SUCCESS', 'CACHED']:
                output_lines.append(f"URL: {data.get('url', 'N/A')}")
                output_lines.append(f"Chars: {data.get('chars', 0)}")
                if data.get('content_preview'):
                    output_lines.append(f"Preview: {data['content_preview'][:200]}...")
            else:
                output_lines.append(f"Error: {data.get('error', 'Unknown')}")
            output_lines.append("")

        return "\n".join(output_lines)

    def save_to_database(
        self,
        company_name: str,
        website: str,
        about_content: str = "",
        services_content: str = "",
        leadership_content: str = ""
    ) -> str:
        """
        Save scraped company research to SQLite database.
        Updates company_evaluations table.
        """
        if not SQLITE_DB_PATH.exists():
            return f"Error: Database not found at {SQLITE_DB_PATH}"

        try:
            conn = sqlite3.connect(str(SQLITE_DB_PATH))
            cursor = conn.cursor()

            # Check if company exists
            cursor.execute(
                "SELECT id, website_content FROM company_evaluations WHERE company_name = ?",
                (company_name,)
            )
            row = cursor.fetchone()

            # Build combined content
            content_parts = []
            if about_content:
                content_parts.append(f"--- ABOUT ---\n{about_content}")
            if services_content:
                content_parts.append(f"--- SERVICES ---\n{services_content}")
            if leadership_content:
                content_parts.append(f"--- LEADERSHIP ---\n{leadership_content}")

            combined_content = "\n\n".join(content_parts)
            now = datetime.now().isoformat()

            if row:
                # Update existing record
                existing_content = row[1] or ""
                new_content = existing_content + "\n\n" + combined_content if existing_content else combined_content

                cursor.execute("""
                    UPDATE company_evaluations
                    SET website_content = ?,
                        url_scraped = ?,
                        website_scraped_date = ?,
                        about_scraped = CASE WHEN ? != '' THEN 1 ELSE about_scraped END,
                        services_scraped = CASE WHEN ? != '' THEN 1 ELSE services_scraped END,
                        leadership_scraped = CASE WHEN ? != '' THEN 1 ELSE leadership_scraped END,
                        scrape_status = 'completed',
                        scrape_date = ?,
                        scrape_source = 'lm_bridge'
                    WHERE company_name = ?
                """, (new_content, website, now, about_content, services_content, leadership_content, now, company_name))
                action = "UPDATED"
            else:
                # Insert new record
                cursor.execute("""
                    INSERT INTO company_evaluations
                    (company_name, url_scraped, website_content, website_scraped_date,
                     about_scraped, services_scraped, leadership_scraped,
                     scrape_status, scrape_date, scrape_source)
                    VALUES (?, ?, ?, ?, ?, ?, ?, 'completed', ?, 'lm_bridge')
                """, (
                    company_name, website, combined_content, now,
                    1 if about_content else 0,
                    1 if services_content else 0,
                    1 if leadership_content else 0,
                    now
                ))
                action = "INSERTED"

            conn.commit()
            conn.close()

            return f"""Database {action} for {company_name}
Website: {website}
About: {'Yes' if about_content else 'No'} ({len(about_content)} chars)
Services: {'Yes' if services_content else 'No'} ({len(services_content)} chars)
Leadership: {'Yes' if leadership_content else 'No'} ({len(leadership_content)} chars)
Total: {len(combined_content)} chars saved"""

        except Exception as e:
            return f"Database error: {e}"

    def execute(self, tool_name: str, arguments: Dict) -> str:
        """Execute a tool by name"""
        if tool_name == "fetch_webpage":
            return self.fetch_webpage(arguments.get("url", ""))
        elif tool_name == "search_page_content":
            return self.search_page_content(
                arguments.get("url", ""),
                arguments.get("search_term", "")
            )
        elif tool_name == "extract_nav_map":
            return self.extract_nav_map(arguments.get("url", ""))
        elif tool_name == "agentic_scrape":
            return self.agentic_scrape(
                arguments.get("url", ""),
                arguments.get("page_type", "about"),
                arguments.get("company_name", "Unknown")
            )
        elif tool_name == "research_company":
            return self.research_company(
                arguments.get("url", ""),
                arguments.get("company_name", "Unknown")
            )
        elif tool_name == "save_to_database":
            return self.save_to_database(
                arguments.get("company_name", ""),
                arguments.get("website", ""),
                arguments.get("about_content", ""),
                arguments.get("services_content", ""),
                arguments.get("leadership_content", "")
            )
        else:
            return f"Unknown tool: {tool_name}"


# =============================================================================
# LM Studio Bridge
# =============================================================================

class LMBridge:
    """
    File-based bridge between Claude Code and LM Studio.

    Monitors task file, processes via LM Studio API, writes results.
    """

    def __init__(
        self,
        task_file: Path = DEFAULT_TASK_FILE,
        results_file: Path = DEFAULT_RESULTS_FILE,
        state_file: Path = DEFAULT_STATE_FILE,
        endpoint: str = LM_STUDIO_ENDPOINT,
        model: str = LM_STUDIO_MODEL,
        tools_enabled: bool = False,
        verbose: bool = False
    ):
        self.task_file = Path(task_file)
        self.results_file = Path(results_file)
        self.state_file = Path(state_file)
        self.endpoint = endpoint
        self.model = model
        self.tools_enabled = tools_enabled
        self.verbose = verbose
        self.api_url = f"{endpoint}/v1/chat/completions"

        # Tool executor
        self.web_tools = WebTools() if tools_enabled else None

        # State tracking
        self.processed_tasks: set = set()
        self._load_state()

    def _log(self, msg: str):
        """Print if verbose"""
        if self.verbose:
            timestamp = datetime.now().strftime("%H:%M:%S")
            print(f"[{timestamp}] {msg}")

    def _load_state(self):
        """Load processed task IDs from state file"""
        if self.state_file.exists():
            try:
                state = json.loads(self.state_file.read_text())
                self.processed_tasks = set(state.get("processed", []))
                self._log(f"Loaded state: {len(self.processed_tasks)} processed tasks")
            except:
                self.processed_tasks = set()

    def _save_state(self):
        """Save state to file"""
        state = {
            "processed": list(self.processed_tasks),
            "last_updated": datetime.now().isoformat()
        }
        self.state_file.write_text(json.dumps(state, indent=2))

    def test_connection(self) -> bool:
        """Test connection to LM Studio"""
        try:
            resp = requests.get(f"{self.endpoint}/v1/models", timeout=5)
            return resp.status_code == 200
        except:
            return False

    def list_models(self) -> List[str]:
        """List available LM Studio models"""
        try:
            resp = requests.get(f"{self.endpoint}/v1/models", timeout=5)
            if resp.status_code == 200:
                data = resp.json()
                return [m.get("id", "unknown") for m in data.get("data", [])]
        except:
            pass
        return []

    def _clean_response(self, text: str) -> str:
        """Remove <think> tags from response"""
        return re.sub(r'<think>.*?</think>\s*', '', text, flags=re.S).strip()

    def _call_model(
        self,
        messages: List[Dict],
        use_tools: bool = False,
        max_tokens: int = 1000,
        temperature: float = 0.7
    ) -> Dict:
        """Make API call to LM Studio"""
        payload = {
            "model": self.model,
            "messages": messages,
            "max_tokens": max_tokens,
            "temperature": temperature
        }

        if use_tools and self.tools_enabled:
            payload["tools"] = WEB_TOOLS
            payload["tool_choice"] = "auto"

        resp = requests.post(self.api_url, json=payload, timeout=120)
        return resp.json()

    def process_task(self, task: BridgeTask) -> BridgeResult:
        """Process a single task through LM Studio"""
        self._log(f"Processing task: {task.task_id} ({task.task_type})")
        start_time = time.time()

        # Build messages
        messages = []
        if task.context:
            messages.append({"role": "system", "content": task.context})
        messages.append({"role": "user", "content": task.prompt})

        try:
            # Determine if tools should be used
            use_tools = task.tools_enabled and self.tools_enabled

            # Tool calling loop
            all_tool_calls = []
            max_iterations = 3 if use_tools else 1

            for iteration in range(max_iterations):
                response = self._call_model(
                    messages,
                    use_tools=use_tools,
                    max_tokens=task.max_tokens,
                    temperature=task.temperature
                )

                if "error" in response:
                    raise Exception(response["error"])

                msg = response["choices"][0]["message"]

                # Check for tool calls
                if msg.get("tool_calls") and self.web_tools:
                    messages.append(msg)

                    for tool_call in msg["tool_calls"]:
                        tool_name = tool_call["function"]["name"]
                        arguments = json.loads(tool_call["function"]["arguments"])

                        self._log(f"  Tool call: {tool_name}({arguments})")
                        result = self.web_tools.execute(tool_name, arguments)

                        all_tool_calls.append({
                            "tool": tool_name,
                            "arguments": arguments,
                            "result_preview": result[:200] + "..." if len(result) > 200 else result
                        })

                        messages.append({
                            "role": "tool",
                            "tool_call_id": tool_call["id"],
                            "content": result
                        })
                else:
                    # No tool call, we have final response
                    break

            # Get final response
            final_response = self._clean_response(msg.get("content", "No response"))
            tokens = response.get("usage", {}).get("total_tokens", 0)

            processing_time = time.time() - start_time
            self._log(f"  Completed in {processing_time:.2f}s")

            return BridgeResult(
                task_id=task.task_id,
                status="success",
                response=final_response,
                model=self.model,
                tokens_used=tokens,
                processing_time=processing_time,
                tool_calls=all_tool_calls
            )

        except Exception as e:
            processing_time = time.time() - start_time
            self._log(f"  Error: {e}")

            return BridgeResult(
                task_id=task.task_id,
                status="error",
                response="",
                model=self.model,
                tokens_used=0,
                processing_time=processing_time,
                error_message=str(e)
            )

    def read_task(self) -> Optional[BridgeTask]:
        """Read task from task file"""
        if not self.task_file.exists():
            return None

        try:
            data = json.loads(self.task_file.read_text())

            # Check if already processed
            task_id = data.get("task_id", "")
            if not task_id:
                # Generate ID from content
                task_id = hashlib.md5(json.dumps(data).encode()).hexdigest()[:12]
                data["task_id"] = task_id

            if task_id in self.processed_tasks:
                return None

            return BridgeTask(
                task_id=task_id,
                task_type=data.get("task_type", "chat"),
                prompt=data.get("prompt", ""),
                context=data.get("context"),
                tools_enabled=data.get("tools_enabled", False),
                max_tokens=data.get("max_tokens", 1000),
                temperature=data.get("temperature", 0.7),
                created_at=data.get("created_at", ""),
                source=data.get("source", "claude_code")
            )
        except Exception as e:
            self._log(f"Error reading task: {e}")
            return None

    def write_result(self, result: BridgeResult):
        """Write result to results file"""
        # Load existing results
        results = []
        if self.results_file.exists():
            try:
                existing = json.loads(self.results_file.read_text())
                if isinstance(existing, list):
                    results = existing
                else:
                    results = [existing]
            except:
                results = []

        # Add new result
        results.append(asdict(result))

        # Keep only last 50 results
        results = results[-50:]

        self.results_file.write_text(json.dumps(results, indent=2))
        self._log(f"Result written to {self.results_file}")

    def run_once(self) -> Optional[BridgeResult]:
        """Process single task (one-shot mode)"""
        task = self.read_task()
        if not task:
            self._log("No new task found")
            return None

        if not task.prompt:
            self._log("Empty prompt, skipping")
            return None

        result = self.process_task(task)
        self.write_result(result)

        # Mark as processed
        self.processed_tasks.add(task.task_id)
        self._save_state()

        return result

    def watch(self, interval: int = 2):
        """Watch mode - continuously monitor for new tasks"""
        print(f"LM Bridge watching {self.task_file}")
        print(f"  Endpoint: {self.endpoint}")
        print(f"  Model: {self.model}")
        print(f"  Tools: {'enabled' if self.tools_enabled else 'disabled'}")
        print(f"  Results: {self.results_file}")
        print(f"\nPress Ctrl+C to stop\n")

        try:
            while True:
                result = self.run_once()
                if result:
                    print(f"\n{'='*50}")
                    print(f"Task: {result.task_id}")
                    print(f"Status: {result.status}")
                    if result.tool_calls:
                        print(f"Tool calls: {len(result.tool_calls)}")
                    print(f"Response preview: {result.response[:200]}...")
                    print(f"{'='*50}\n")

                time.sleep(interval)
        except KeyboardInterrupt:
            print("\nBridge stopped")
            self._save_state()


# =============================================================================
# Helper Functions
# =============================================================================

def create_task(
    prompt: str,
    task_type: str = "chat",
    context: str = None,
    tools_enabled: bool = False,
    output_file: Path = DEFAULT_TASK_FILE
) -> str:
    """
    Create a task file for the bridge.

    Call this from Claude Code to queue a task for LM Studio.

    Returns:
        Task ID
    """
    task = BridgeTask(
        task_id="",  # Will be auto-generated
        task_type=task_type,
        prompt=prompt,
        context=context,
        tools_enabled=tools_enabled
    )

    output_file.write_text(json.dumps(asdict(task), indent=2))
    return task.task_id


def read_latest_result(results_file: Path = DEFAULT_RESULTS_FILE) -> Optional[Dict]:
    """
    Read the latest result from the results file.

    Call this from Claude Code to get LM Studio's response.
    """
    if not results_file.exists():
        return None

    try:
        results = json.loads(results_file.read_text())
        if isinstance(results, list) and results:
            return results[-1]
        return results
    except:
        return None


def clear_state(state_file: Path = DEFAULT_STATE_FILE):
    """Clear the processed tasks state"""
    if state_file.exists():
        state_file.unlink()
    print("State cleared")


# =============================================================================
# Export and Utility Functions
# =============================================================================

def export_results_to_csv(
    results_file: Path = DEFAULT_RESULTS_FILE,
    output_file: Path = None,
    include_response: bool = False
) -> str:
    """
    Export batch results to CSV format.

    Args:
        results_file: Path to results JSON file
        output_file: Path for CSV output (auto-generated if None)
        include_response: Whether to include full response text (can be large)

    Returns:
        Path to created CSV file
    """
    import csv

    if not results_file.exists():
        return f"No results file found at {results_file}"

    try:
        results = json.loads(results_file.read_text())
        if not isinstance(results, list):
            results = [results]
    except json.JSONDecodeError as e:
        return f"Invalid JSON in results file: {e}"

    if not results:
        return "No results to export"

    # Generate output filename if not provided
    if output_file is None:
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        output_file = BRIDGE_DIR / f"lm_results_{timestamp}.csv"

    # Define CSV columns
    columns = [
        "task_id", "company_name", "url", "status", "model",
        "tokens_used", "processing_time", "tool_calls_count",
        "batch_id", "batch_index", "completed_at", "error_message"
    ]
    if include_response:
        columns.append("response")

    # Write CSV
    with open(output_file, 'w', newline='', encoding='utf-8') as f:
        writer = csv.DictWriter(f, fieldnames=columns, extrasaction='ignore')
        writer.writeheader()

        for result in results:
            row = {
                "task_id": result.get("task_id", ""),
                "company_name": result.get("company_name", ""),
                "url": result.get("url", ""),
                "status": result.get("status", ""),
                "model": result.get("model", ""),
                "tokens_used": result.get("tokens_used", 0),
                "processing_time": round(result.get("processing_time", 0), 2),
                "tool_calls_count": len(result.get("tool_calls", [])),
                "batch_id": result.get("batch_id", ""),
                "batch_index": result.get("batch_index", ""),
                "completed_at": result.get("completed_at", ""),
                "error_message": result.get("error_message", "")
            }
            if include_response:
                row["response"] = result.get("response", "")[:10000]  # Truncate long responses
            writer.writerow(row)

    return str(output_file)


def check_duplicate_url(url: str, db_path: Path = SQLITE_DB_PATH) -> Dict[str, Any]:
    """
    Check if a URL or domain already exists in the database.

    Args:
        url: URL to check
        db_path: Path to SQLite database

    Returns:
        Dict with 'exists', 'company_name', 'scraped_date' if found
    """
    if not db_path.exists():
        return {"exists": False, "error": f"Database not found at {db_path}"}

    # Normalize URL - extract domain
    import urllib.parse
    try:
        parsed = urllib.parse.urlparse(url if url.startswith(('http://', 'https://')) else f"https://{url}")
        domain = parsed.netloc.lower().replace('www.', '')
    except:
        domain = url.lower().replace('www.', '')

    try:
        conn = sqlite3.connect(str(db_path))
        cursor = conn.cursor()

        # Search for domain in url_scraped column
        cursor.execute("""
            SELECT company_name, url_scraped, website_scraped_date, scrape_status
            FROM company_evaluations
            WHERE LOWER(url_scraped) LIKE ?
            OR LOWER(url_scraped) LIKE ?
        """, (f"%{domain}%", f"%{domain}%"))

        row = cursor.fetchone()
        conn.close()

        if row:
            return {
                "exists": True,
                "company_name": row[0],
                "url_scraped": row[1],
                "scraped_date": row[2],
                "scrape_status": row[3],
                "domain_matched": domain
            }
        else:
            return {"exists": False, "domain_checked": domain}

    except Exception as e:
        return {"exists": False, "error": str(e)}


def check_duplicates_batch(
    companies: List[Dict[str, str]],
    db_path: Path = SQLITE_DB_PATH
) -> Dict[str, Any]:
    """
    Check multiple URLs for duplicates before processing.

    Args:
        companies: List of dicts with 'company_name' and 'url' keys
        db_path: Path to SQLite database

    Returns:
        Dict with 'new', 'duplicates', and 'summary'
    """
    new_companies = []
    duplicates = []

    for company in companies:
        url = company.get("url", "")
        result = check_duplicate_url(url, db_path)

        if result.get("exists"):
            duplicates.append({
                "company_name": company.get("company_name"),
                "url": url,
                "existing_company": result.get("company_name"),
                "scraped_date": result.get("scraped_date")
            })
        else:
            new_companies.append(company)

    return {
        "new": new_companies,
        "duplicates": duplicates,
        "summary": {
            "total": len(companies),
            "new_count": len(new_companies),
            "duplicate_count": len(duplicates)
        }
    }


# =============================================================================
# Batch Resume Functions
# =============================================================================

DEFAULT_BATCH_STATE_FILE = BRIDGE_DIR / "lm_batch_state.json"


def save_batch_progress(
    batch_id: str,
    completed_indices: List[int],
    total_companies: int,
    results: List[Dict],
    state_file: Path = DEFAULT_BATCH_STATE_FILE
):
    """Save batch progress for resume capability."""
    state = {
        "batch_id": batch_id,
        "completed_indices": completed_indices,
        "total_companies": total_companies,
        "last_updated": datetime.now().isoformat(),
        "results_count": len(results)
    }
    state_file.write_text(json.dumps(state, indent=2))


def load_batch_progress(
    batch_id: str,
    state_file: Path = DEFAULT_BATCH_STATE_FILE
) -> Optional[Dict]:
    """Load saved batch progress if it exists."""
    if not state_file.exists():
        return None

    try:
        state = json.loads(state_file.read_text())
        if state.get("batch_id") == batch_id:
            return state
    except:
        pass
    return None


def clear_batch_progress(state_file: Path = DEFAULT_BATCH_STATE_FILE):
    """Clear batch progress state."""
    if state_file.exists():
        state_file.unlink()


def get_batch_stats(results_file: Path = DEFAULT_RESULTS_FILE) -> Dict[str, Any]:
    """
    Get statistics from batch results.

    Returns:
        Dict with success/failure counts, timing stats, etc.
    """
    if not results_file.exists():
        return {"error": "No results file found"}

    try:
        results = json.loads(results_file.read_text())
        if not isinstance(results, list):
            results = [results]
    except json.JSONDecodeError:
        return {"error": "Invalid results file"}

    if not results:
        return {"error": "No results in file"}

    # Calculate stats
    successful = [r for r in results if r.get("status") == "success"]
    failed = [r for r in results if r.get("status") != "success"]

    processing_times = [r.get("processing_time", 0) for r in results if r.get("processing_time")]
    tokens_used = [r.get("tokens_used", 0) for r in results if r.get("tokens_used")]

    # Get batch info
    batch_ids = set(r.get("batch_id", "unknown") for r in results)

    stats = {
        "total_tasks": len(results),
        "successful": len(successful),
        "failed": len(failed),
        "success_rate": f"{(len(successful) / len(results) * 100):.1f}%" if results else "0%",
        "batch_ids": list(batch_ids),
        "timing": {
            "total_time": f"{sum(processing_times):.1f}s",
            "avg_time": f"{sum(processing_times) / len(processing_times):.1f}s" if processing_times else "N/A",
            "min_time": f"{min(processing_times):.1f}s" if processing_times else "N/A",
            "max_time": f"{max(processing_times):.1f}s" if processing_times else "N/A"
        },
        "tokens": {
            "total": sum(tokens_used),
            "avg": int(sum(tokens_used) / len(tokens_used)) if tokens_used else 0
        },
        "failed_companies": [
            {"company": r.get("company_name", "Unknown"), "error": r.get("error_message", "Unknown error")}
            for r in failed
        ]
    }

    return stats


def export_failed_companies(
    results_file: Path = DEFAULT_RESULTS_FILE,
    output_file: Path = None
) -> str:
    """
    Export failed companies to JSON for retry.

    Args:
        results_file: Path to results file
        output_file: Optional output path (default: failed_companies_{timestamp}.json)

    Returns:
        Path to exported file
    """
    if not results_file.exists():
        return "No results file found"

    try:
        results = json.loads(results_file.read_text())
        if not isinstance(results, list):
            results = [results]
    except json.JSONDecodeError:
        return "Invalid results file"

    failed = [r for r in results if r.get("status") != "success"]

    if not failed:
        return "No failed tasks to export"

    # Create retry-ready format
    retry_companies = [
        {
            "company_name": r.get("company_name", "Unknown"),
            "url": r.get("url", ""),
            "original_error": r.get("error_message", "Unknown"),
            "original_batch_id": r.get("batch_id", "unknown"),
            "failed_at": r.get("completed_at", datetime.now().isoformat())
        }
        for r in failed
    ]

    if output_file is None:
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        output_file = BRIDGE_DIR / f"failed_companies_{timestamp}.json"

    output_file.write_text(json.dumps({
        "companies": retry_companies,
        "exported_at": datetime.now().isoformat(),
        "total_failed": len(retry_companies)
    }, indent=2))

    return str(output_file)


def create_batch_task(
    companies: List[Dict[str, str]],
    task_template: str = "Research {company_name} at {url} and save to database",
    tools_enabled: bool = True,
    output_file: Path = DEFAULT_TASK_FILE
) -> str:
    """
    Create a batch task file for processing multiple companies.

    Args:
        companies: List of dicts with 'company_name' and 'url' keys
        task_template: Template string with {company_name} and {url} placeholders
        tools_enabled: Whether to enable web scraping tools
        output_file: Path to write task file

    Returns:
        Batch ID
    """
    batch_id = hashlib.md5(f"{datetime.now().isoformat()}{len(companies)}".encode()).hexdigest()[:12]

    batch_task = {
        "batch_id": batch_id,
        "batch_mode": True,
        "task_template": task_template,
        "tools_enabled": tools_enabled,
        "companies": companies,
        "created_at": datetime.now().isoformat(),
        "source": "claude_code"
    }

    output_file.write_text(json.dumps(batch_task, indent=2))
    return batch_id


def process_batch(
    task_file: Path = DEFAULT_TASK_FILE,
    results_file: Path = DEFAULT_RESULTS_FILE,
    endpoint: str = LM_STUDIO_ENDPOINT,
    verbose: bool = False,
    resume: bool = True,
    skip_duplicates: bool = True,
    delay: float = 0.0,
    max_retries: int = 0,
    retry_backoff: float = 2.0,
    dry_run: bool = False,
    limit: int = 0,
    quiet: bool = False
) -> List[Dict]:
    """
    Process a batch task file containing multiple companies.

    Args:
        task_file: Path to batch task JSON file
        results_file: Path to write results
        endpoint: LM Studio API endpoint
        verbose: Enable verbose output
        resume: Resume from previous progress if available
        skip_duplicates: Skip companies already in database
        delay: Seconds to wait between processing companies (rate limiting)
        max_retries: Maximum retry attempts for failed tasks (0 = no retries)
        retry_backoff: Backoff multiplier for exponential retry delay
        dry_run: Preview batch without processing
        limit: Process only first N companies (0 = all)
        quiet: Suppress non-error output

    Returns:
        List of results for each company
    """
    if not task_file.exists():
        print(f"No batch task file found at {task_file}")
        return []

    try:
        batch_task = json.loads(task_file.read_text())
    except json.JSONDecodeError as e:
        print(f"Invalid JSON in task file: {e}")
        return []

    if not batch_task.get("batch_mode"):
        print("Task file is not a batch task. Use regular processing.")
        return []

    companies = batch_task.get("companies", [])
    template = batch_task.get("task_template", "Research {company_name} at {url}")
    tools_enabled = batch_task.get("tools_enabled", True)
    batch_id = batch_task.get("batch_id", "unknown")

    # Helper for quiet mode
    def log(msg):
        if not quiet:
            print(msg)

    # Check for duplicates if enabled
    if skip_duplicates:
        dup_check = check_duplicates_batch(companies)
        if dup_check["duplicates"]:
            log(f"Skipping {dup_check['summary']['duplicate_count']} duplicates:")
            for dup in dup_check["duplicates"]:
                log(f"  - {dup['company_name']} (exists as {dup['existing_company']})")
            companies = dup_check["new"]
            log(f"Processing {len(companies)} new companies\n")

    if not companies:
        log("No companies to process (all duplicates or empty list)")
        return []

    # Apply limit if specified
    if limit > 0 and limit < len(companies):
        log(f"Limiting to first {limit} of {len(companies)} companies")
        companies = companies[:limit]

    # Dry run mode - just preview
    if dry_run:
        print(f"\n=== DRY RUN - Batch {batch_id} ===")
        print(f"Total companies: {len(companies)}")
        print(f"Template: {template}")
        print(f"Tools enabled: {tools_enabled}")
        print(f"\nCompanies to process:")
        for i, c in enumerate(companies, 1):
            print(f"  {i}. {c.get('company_name', 'Unknown')} - {c.get('url', 'No URL')}")
        print(f"\nOptions: delay={delay}s, max_retries={max_retries}, backoff={retry_backoff}x")
        return []

    log(f"Processing batch {batch_id}: {len(companies)} companies")

    # Check for resume capability
    completed_indices = []
    results = []

    if resume:
        progress = load_batch_progress(batch_id)
        if progress:
            completed_indices = progress.get("completed_indices", [])
            # Load existing results
            if results_file.exists():
                try:
                    results = json.loads(results_file.read_text())
                    if not isinstance(results, list):
                        results = [results]
                except:
                    results = []
            log(f"Resuming batch: {len(completed_indices)}/{len(companies)} already completed")

    bridge = LMBridge(endpoint=endpoint, verbose=verbose, tools_enabled=tools_enabled)

    if delay > 0:
        log(f"Rate limiting: {delay}s delay between requests")
    if max_retries > 0:
        log(f"Retry enabled: up to {max_retries} attempts with {retry_backoff}x backoff")

    for i, company in enumerate(companies, 1):
        # Skip already completed
        if i in completed_indices:
            continue

        company_name = company.get("company_name", "Unknown")
        url = company.get("url", "")

        log(f"[{i}/{len(companies)}] Processing: {company_name}")

        # Create individual task
        prompt = template.format(company_name=company_name, url=url)
        task = BridgeTask(
            task_id=f"{batch_id}_{i:03d}",
            task_type="scrape",
            prompt=prompt,
            tools_enabled=tools_enabled
        )

        # Process with retry logic
        result = None
        retry_count = 0
        current_delay = 1.0  # Initial retry delay

        while True:
            result = bridge.process_task(task)

            if result.status == "success" or retry_count >= max_retries:
                break

            # Retry with exponential backoff
            retry_count += 1
            log(f"    Retry {retry_count}/{max_retries} after {current_delay:.1f}s...")
            time.sleep(current_delay)
            current_delay *= retry_backoff

        result_dict = asdict(result)
        result_dict["company_name"] = company_name
        result_dict["url"] = url
        result_dict["batch_id"] = batch_id
        result_dict["batch_index"] = i
        result_dict["retry_count"] = retry_count
        results.append(result_dict)

        # Track progress
        completed_indices.append(i)

        # Save intermediate results and progress
        results_file.write_text(json.dumps(results, indent=2))
        save_batch_progress(batch_id, completed_indices, len(companies), results)

        if result.status == "success":
            retry_info = f" (retries: {retry_count})" if retry_count > 0 else ""
            log(f"    OK - {result.processing_time:.1f}s{retry_info}")
        else:
            log(f"    FAILED - {result.error_message}")

        # Rate limiting delay between companies
        if delay > 0 and i < len(companies):
            time.sleep(delay)

    # Clear batch progress on completion
    clear_batch_progress()

    # Summary
    successful = sum(1 for r in results if r["status"] == "success")
    log(f"\nBatch complete: {successful}/{len(companies)} successful")

    return results


def clear_results(results_file: Path = DEFAULT_RESULTS_FILE) -> bool:
    """Clear the results file."""
    if results_file.exists():
        results_file.unlink()
        return True
    return False


def health_check(endpoint: str = LM_STUDIO_ENDPOINT) -> Dict[str, Any]:
    """
    Perform a quick health check on the endpoint.

    Returns:
        Dict with status, response_time, model info
    """
    start_time = time.time()

    try:
        # Check models endpoint
        response = requests.get(f"{endpoint}/v1/models", timeout=5)
        response_time = time.time() - start_time

        if response.status_code == 200:
            data = response.json()
            models = data.get("data", [])
            model_names = [m.get("id", "unknown") for m in models]

            return {
                "status": "healthy",
                "endpoint": endpoint,
                "response_time": f"{response_time:.3f}s",
                "models_available": len(models),
                "models": model_names[:5],  # First 5
                "checked_at": datetime.now().isoformat()
            }
        else:
            return {
                "status": "unhealthy",
                "endpoint": endpoint,
                "response_time": f"{response_time:.3f}s",
                "error": f"HTTP {response.status_code}",
                "checked_at": datetime.now().isoformat()
            }

    except requests.exceptions.Timeout:
        return {
            "status": "timeout",
            "endpoint": endpoint,
            "error": "Connection timed out after 5s",
            "checked_at": datetime.now().isoformat()
        }
    except requests.exceptions.ConnectionError as e:
        return {
            "status": "unreachable",
            "endpoint": endpoint,
            "error": str(e),
            "checked_at": datetime.now().isoformat()
        }


def validate_batch_file(task_file: Path = DEFAULT_TASK_FILE) -> Dict[str, Any]:
    """
    Validate a batch task file without processing.

    Returns:
        Dict with validation results, warnings, and errors
    """
    result = {
        "valid": True,
        "file": str(task_file),
        "errors": [],
        "warnings": [],
        "summary": {}
    }

    # Check file exists
    if not task_file.exists():
        result["valid"] = False
        result["errors"].append(f"File not found: {task_file}")
        return result

    # Try to parse JSON
    try:
        with open(task_file, 'r') as f:
            data = json.load(f)
    except json.JSONDecodeError as e:
        result["valid"] = False
        result["errors"].append(f"Invalid JSON: {e}")
        return result

    # Check for batch_id
    if "batch_id" not in data:
        result["warnings"].append("Missing batch_id field")
    else:
        result["summary"]["batch_id"] = data["batch_id"]

    # Check for companies array
    if "companies" not in data:
        result["valid"] = False
        result["errors"].append("Missing 'companies' array")
        return result

    companies = data["companies"]
    if not isinstance(companies, list):
        result["valid"] = False
        result["errors"].append("'companies' must be an array")
        return result

    result["summary"]["total_companies"] = len(companies)

    # Validate each company
    missing_names = 0
    missing_urls = 0
    invalid_urls = 0

    for i, company in enumerate(companies):
        if not isinstance(company, dict):
            result["errors"].append(f"Company {i}: not a dictionary")
            result["valid"] = False
            continue

        if "company_name" not in company:
            missing_names += 1

        if "url" not in company:
            missing_urls += 1
        elif not company["url"].startswith(("http://", "https://")):
            invalid_urls += 1

    if missing_names > 0:
        result["warnings"].append(f"{missing_names} companies missing 'company_name'")
    if missing_urls > 0:
        result["errors"].append(f"{missing_urls} companies missing 'url'")
        result["valid"] = False
    if invalid_urls > 0:
        result["warnings"].append(f"{invalid_urls} URLs don't start with http:// or https://")

    # Check for duplicates within file
    urls = [c.get("url", "").lower().rstrip("/") for c in companies if isinstance(c, dict)]
    unique_urls = set(urls)
    if len(urls) != len(unique_urls):
        dupes = len(urls) - len(unique_urls)
        result["warnings"].append(f"{dupes} duplicate URLs in file")

    result["summary"]["unique_urls"] = len(unique_urls)

    return result


def tail_results(
    results_file: Path = DEFAULT_RESULTS_FILE,
    n: int = 10,
    status_filter: str = None
) -> List[Dict]:
    """
    Show last N results from the results file.

    Args:
        results_file: Path to results file
        n: Number of results to show
        status_filter: Optional filter by status (success/failed)

    Returns:
        List of last N results
    """
    if not results_file.exists():
        return []

    try:
        with open(results_file, 'r') as f:
            data = json.load(f)
    except (json.JSONDecodeError, IOError):
        return []

    # Handle both array format and object with "results" key
    if isinstance(data, list):
        results = data
    else:
        results = data.get("results", [])

    # Apply status filter if provided
    if status_filter:
        status_filter = status_filter.lower()
        results = [r for r in results if r.get("status", "").lower() == status_filter]

    # Return last N results
    return results[-n:] if n > 0 else results


def filter_results(
    results_file: Path = DEFAULT_RESULTS_FILE,
    status: Optional[str] = None,
    has_error: Optional[bool] = None,
    company_name: Optional[str] = None,
    min_tokens: Optional[int] = None,
    max_time: Optional[float] = None
) -> List[Dict]:
    """
    Filter results by various criteria.

    Args:
        results_file: Path to results file
        status: Filter by status (success/failed)
        has_error: Filter by error presence (True/False)
        company_name: Filter by company name (substring match)
        min_tokens: Filter by minimum tokens used
        max_time: Filter by maximum processing time

    Returns:
        List of matching results
    """
    if not results_file.exists():
        return []

    try:
        with open(results_file, 'r') as f:
            data = json.load(f)
    except (json.JSONDecodeError, IOError):
        return []

    # Handle both array format and object with "results" key
    if isinstance(data, list):
        results = data
    else:
        results = data.get("results", [])
    filtered = []

    for r in results:
        # Status filter
        if status and r.get("status", "").lower() != status.lower():
            continue

        # Error filter
        if has_error is not None:
            has_err = bool(r.get("error_message"))
            if has_error != has_err:
                continue

        # Company name filter (substring)
        if company_name:
            comp_name = r.get("company_name", "").lower()
            if company_name.lower() not in comp_name:
                continue

        # Minimum tokens filter
        if min_tokens is not None:
            tokens = r.get("tokens_used", 0)
            if tokens < min_tokens:
                continue

        # Max time filter
        if max_time is not None:
            proc_time = r.get("processing_time", 0)
            if proc_time > max_time:
                continue

        filtered.append(r)

    return filtered


def generate_summary(
    results_file: Path = DEFAULT_RESULTS_FILE,
    output_format: str = "text"
) -> str:
    """
    Generate a comprehensive summary report of batch results.

    Args:
        results_file: Path to results file
        output_format: Output format (text, json, markdown)

    Returns:
        Summary report as string
    """
    stats = get_batch_stats(results_file)

    # Handle empty/missing stats
    if not stats or "total_tasks" not in stats:
        if output_format == "json":
            return json.dumps({"error": "No results found", "file": str(results_file)}, indent=2)
        return f"No results found in {results_file}"

    if output_format == "json":
        return json.dumps(stats, indent=2, default=str)

    # Normalize keys for text/markdown output
    stats["total"] = stats.get("total_tasks", 0)
    stats["success"] = stats.get("successful", 0)
    if stats.get("timing"):
        # Convert string times to floats for formatting
        timing = stats["timing"]
        timing["total"] = float(timing.get("total_time", "0s").rstrip("s"))
        timing["average"] = float(timing.get("avg_time", "0s").rstrip("s")) if timing.get("avg_time") != "N/A" else 0
        timing["min"] = float(timing.get("min_time", "0s").rstrip("s")) if timing.get("min_time") != "N/A" else 0
        timing["max"] = float(timing.get("max_time", "0s").rstrip("s")) if timing.get("max_time") != "N/A" else 0
    if stats.get("tokens"):
        stats["tokens"]["average"] = stats["tokens"].get("avg", 0)

    # Build text/markdown report
    lines = []

    if output_format == "markdown":
        lines.append("# Batch Processing Summary Report")
        lines.append("")
        lines.append(f"**Generated:** {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
        lines.append("")
        lines.append("## Overview")
        lines.append("")
        lines.append(f"| Metric | Value |")
        lines.append(f"|--------|-------|")
        lines.append(f"| Total Tasks | {stats['total']} |")
        lines.append(f"| Successful | {stats['success']} |")
        lines.append(f"| Failed | {stats['failed']} |")
        lines.append(f"| Success Rate | {stats['success_rate']} |")
        lines.append("")

        if stats.get("timing"):
            lines.append("## Timing")
            lines.append("")
            lines.append(f"| Metric | Value |")
            lines.append(f"|--------|-------|")
            lines.append(f"| Total Time | {stats['timing']['total']:.1f}s |")
            lines.append(f"| Average | {stats['timing']['average']:.1f}s |")
            lines.append(f"| Min | {stats['timing']['min']:.1f}s |")
            lines.append(f"| Max | {stats['timing']['max']:.1f}s |")
            lines.append("")

        if stats.get("tokens"):
            lines.append("## Token Usage")
            lines.append("")
            lines.append(f"| Metric | Value |")
            lines.append(f"|--------|-------|")
            lines.append(f"| Total Tokens | {stats['tokens']['total']:,} |")
            lines.append(f"| Average | {stats['tokens']['average']:,.0f} |")
            lines.append("")

        if stats.get("failed_companies"):
            lines.append("## Failed Companies")
            lines.append("")
            for fc in stats["failed_companies"]:
                lines.append(f"- **{fc.get('company', fc.get('company_name', 'Unknown'))}**: {fc.get('error', 'Unknown')}")
            lines.append("")

    else:  # text format
        lines.append("=" * 50)
        lines.append("        BATCH PROCESSING SUMMARY REPORT")
        lines.append("=" * 50)
        lines.append(f"Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
        lines.append("")
        lines.append("OVERVIEW")
        lines.append("-" * 30)
        lines.append(f"  Total Tasks:    {stats['total']}")
        lines.append(f"  Successful:     {stats['success']}")
        lines.append(f"  Failed:         {stats['failed']}")
        lines.append(f"  Success Rate:   {stats['success_rate']}")
        lines.append("")

        if stats.get("timing"):
            lines.append("TIMING")
            lines.append("-" * 30)
            lines.append(f"  Total Time:     {stats['timing']['total']:.1f}s")
            lines.append(f"  Average:        {stats['timing']['average']:.1f}s")
            lines.append(f"  Min:            {stats['timing']['min']:.1f}s")
            lines.append(f"  Max:            {stats['timing']['max']:.1f}s")
            lines.append("")

        if stats.get("tokens"):
            lines.append("TOKEN USAGE")
            lines.append("-" * 30)
            lines.append(f"  Total:          {stats['tokens']['total']:,}")
            lines.append(f"  Average:        {stats['tokens']['average']:,.0f}")
            lines.append("")

        if stats.get("failed_companies"):
            lines.append("FAILED COMPANIES")
            lines.append("-" * 30)
            for fc in stats["failed_companies"]:
                lines.append(f"  • {fc.get('company', fc.get('company_name', 'Unknown'))}: {fc.get('error', 'Unknown')}")
            lines.append("")

        lines.append("=" * 50)

    return "\n".join(lines)


def get_cache_stats(cache_file: Path = DEFAULT_CACHE_FILE) -> Dict[str, Any]:
    """
    Get statistics about the scrape cache.

    Returns:
        Dict with cache statistics
    """
    if not cache_file.exists():
        return {
            "exists": False,
            "file": str(cache_file),
            "entries": 0,
            "size_bytes": 0
        }

    try:
        with open(cache_file, 'r') as f:
            cache_data = json.load(f)

        entries = len(cache_data)
        size_bytes = cache_file.stat().st_size

        # Count valid vs expired entries
        now = datetime.now()
        valid = 0
        expired = 0

        for url, entry in cache_data.items():
            if isinstance(entry, dict) and "cached_at" in entry:
                try:
                    cached_at = datetime.fromisoformat(entry["cached_at"])
                    if now - cached_at < timedelta(hours=CACHE_TTL_HOURS):
                        valid += 1
                    else:
                        expired += 1
                except (ValueError, TypeError):
                    expired += 1
            else:
                expired += 1

        return {
            "exists": True,
            "file": str(cache_file),
            "entries": entries,
            "valid": valid,
            "expired": expired,
            "size_bytes": size_bytes,
            "size_human": f"{size_bytes / 1024:.1f} KB" if size_bytes < 1024 * 1024 else f"{size_bytes / 1024 / 1024:.1f} MB",
            "ttl_hours": CACHE_TTL_HOURS
        }

    except (json.JSONDecodeError, IOError) as e:
        return {
            "exists": True,
            "file": str(cache_file),
            "error": str(e),
            "entries": 0
        }


def clear_cache(cache_file: Path = DEFAULT_CACHE_FILE) -> bool:
    """Clear the scrape cache file."""
    if cache_file.exists():
        cache_file.unlink()
        return True
    return False


def search_results(
    results_file: Path = DEFAULT_RESULTS_FILE,
    term: str = "",
    field: str = "company_name"
) -> List[Dict]:
    """
    Search results by a term in a specific field.

    Args:
        results_file: Path to results file
        term: Search term (case-insensitive)
        field: Field to search in (company_name, url, response, error_message)

    Returns:
        List of matching results
    """
    if not results_file.exists() or not term:
        return []

    try:
        with open(results_file, 'r') as f:
            data = json.load(f)
    except (json.JSONDecodeError, IOError):
        return []

    # Handle both array format and object with "results" key
    if isinstance(data, list):
        results = data
    else:
        results = data.get("results", [])

    term_lower = term.lower()
    matches = []

    for r in results:
        search_value = str(r.get(field, "")).lower()
        if term_lower in search_value:
            matches.append(r)

    return matches


def count_results(
    results_file: Path = DEFAULT_RESULTS_FILE,
    status: str = None
) -> Dict[str, int]:
    """
    Count results by status.

    Args:
        results_file: Path to results file
        status: Optional status filter

    Returns:
        Dict with counts
    """
    if not results_file.exists():
        return {"total": 0, "success": 0, "failed": 0}

    try:
        with open(results_file, 'r') as f:
            data = json.load(f)
    except (json.JSONDecodeError, IOError):
        return {"total": 0, "success": 0, "failed": 0}

    # Handle both array format and object with "results" key
    if isinstance(data, list):
        results = data
    else:
        results = data.get("results", [])

    if status:
        filtered = [r for r in results if r.get("status", "").lower() == status.lower()]
        return {"total": len(filtered), "status": status}

    success = sum(1 for r in results if r.get("status") == "success")
    failed = len(results) - success

    return {
        "total": len(results),
        "success": success,
        "failed": failed
    }


def sort_results(
    results: List[Dict],
    field: str = "processing_time",
    reverse: bool = False
) -> List[Dict]:
    """
    Sort results by a field.

    Args:
        results: List of result dicts
        field: Field to sort by (processing_time, tokens_used, company_name, completed_at)
        reverse: Reverse sort order (descending)

    Returns:
        Sorted list
    """
    def get_sort_key(r):
        val = r.get(field, 0)
        if field in ("processing_time", "tokens_used"):
            return float(val) if val else 0
        return str(val).lower() if val else ""

    return sorted(results, key=get_sort_key, reverse=reverse)


def filter_by_date(
    results_file: Path = DEFAULT_RESULTS_FILE,
    since: Optional[str] = None,
    before: Optional[str] = None
) -> List[Dict]:
    """
    Filter results by date range.

    Args:
        results_file: Path to results file
        since: ISO date string (include results after this date)
        before: ISO date string (include results before this date)

    Returns:
        List of matching results
    """
    if not results_file.exists():
        return []

    try:
        with open(results_file, 'r') as f:
            data = json.load(f)
    except (json.JSONDecodeError, IOError):
        return []

    # Handle both array format and object with "results" key
    if isinstance(data, list):
        results = data
    else:
        results = data.get("results", [])
    filtered = []

    # Parse date filters
    since_dt = None
    before_dt = None

    if since:
        try:
            since_dt = datetime.fromisoformat(since.replace("Z", "+00:00"))
        except ValueError:
            # Try date-only format
            try:
                since_dt = datetime.strptime(since, "%Y-%m-%d")
            except ValueError:
                pass

    if before:
        try:
            before_dt = datetime.fromisoformat(before.replace("Z", "+00:00"))
        except ValueError:
            try:
                before_dt = datetime.strptime(before, "%Y-%m-%d")
            except ValueError:
                pass

    for r in results:
        completed_at = r.get("completed_at")
        if not completed_at:
            continue

        try:
            result_dt = datetime.fromisoformat(completed_at.replace("Z", "+00:00"))
        except (ValueError, TypeError):
            continue

        # Apply filters
        if since_dt and result_dt < since_dt:
            continue
        if before_dt and result_dt > before_dt:
            continue

        filtered.append(r)

    return filtered


def get_version_info() -> Dict[str, str]:
    """Get version information."""
    return {
        "version": VERSION,
        "build_date": BUILD_DATE,
        "bridge": "LM Studio Bridge",
        "endpoint": LM_STUDIO_ENDPOINT,
        "model": LM_STUDIO_MODEL,
        "python": f"{__import__('sys').version_info.major}.{__import__('sys').version_info.minor}"
    }


# =============================================================================
# CLI
# =============================================================================

def main():
    parser = argparse.ArgumentParser(
        description="LM Studio Bridge - File-based task orchestrator",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
    # One-shot mode (process single task)
    python lm_bridge.py

    # Watch mode (continuous monitoring)
    python lm_bridge.py --watch

    # With web scraping tools
    python lm_bridge.py --watch --tools

    # Create a task (for testing)
    python lm_bridge.py --create "What is the capital of France?"

    # Create a web scraping task
    python lm_bridge.py --create "Scrape https://stripe.com and tell me what they do" --tools

    # Read latest result
    python lm_bridge.py --read

    # Test connection
    python lm_bridge.py --test

    # Clear processed state
    python lm_bridge.py --clear

    # Process batch task
    python lm_bridge.py --batch

    # Create batch task from JSON file
    python lm_bridge.py --create-batch companies.json

    # Export results to CSV
    python lm_bridge.py --export-csv

    # Check for duplicates before processing
    python lm_bridge.py --check-duplicates companies.json

    # Process batch without resume (fresh start)
    python lm_bridge.py --batch --no-resume

    # Process batch including duplicates
    python lm_bridge.py --batch --include-duplicates
        """
    )

    # Modes
    parser.add_argument("--watch", action="store_true", help="Watch mode - continuously monitor for tasks")
    parser.add_argument("--create", metavar="PROMPT", help="Create a task file with given prompt")
    parser.add_argument("--read", action="store_true", help="Read latest result")
    parser.add_argument("--test", action="store_true", help="Test connection to LM Studio")
    parser.add_argument("--list-models", action="store_true", help="List available LM Studio models")
    parser.add_argument("--clear", action="store_true", help="Clear processed tasks state")
    parser.add_argument("--batch", action="store_true", help="Process batch task file")
    parser.add_argument("--create-batch", metavar="FILE", help="Create batch task from JSON file with companies array")
    parser.add_argument("--export-csv", action="store_true", help="Export results to CSV file")
    parser.add_argument("--check-duplicates", metavar="FILE", help="Check companies file for duplicates in database")
    parser.add_argument("--stats", action="store_true", help="Show batch processing statistics")
    parser.add_argument("--export-failed", action="store_true", help="Export failed companies for retry")
    parser.add_argument("--health", action="store_true", help="Quick health check of endpoint")
    parser.add_argument("--clear-results", action="store_true", help="Clear the results file")

    # Options
    parser.add_argument("--tools", action="store_true", help="Enable web scraping tools")
    parser.add_argument("--task", metavar="FILE", help="Custom task file path")
    parser.add_argument("--results", metavar="FILE", help="Custom results file path")
    parser.add_argument("--interval", type=int, default=2, help="Watch interval in seconds (default: 2)")
    parser.add_argument("-v", "--verbose", action="store_true", help="Verbose output")
    parser.add_argument("--endpoint", default=LM_STUDIO_ENDPOINT, help="LM Studio endpoint")
    parser.add_argument("--no-resume", action="store_true", help="Don't resume from previous batch progress")
    parser.add_argument("--include-duplicates", action="store_true", help="Process companies even if they exist in database")
    parser.add_argument("--include-response", action="store_true", help="Include full response in CSV export")
    parser.add_argument("--delay", type=float, default=0.0, help="Delay in seconds between batch requests (rate limiting)")
    parser.add_argument("--max-retries", type=int, default=0, help="Max retry attempts for failed tasks (default: 0)")
    parser.add_argument("--retry-backoff", type=float, default=2.0, help="Exponential backoff multiplier (default: 2.0)")
    parser.add_argument("--dry-run", action="store_true", help="Preview batch without processing")
    parser.add_argument("--limit", type=int, default=0, help="Process only first N companies (0 = all)")
    parser.add_argument("-q", "--quiet", action="store_true", help="Suppress output except errors")
    parser.add_argument("--validate", action="store_true", help="Validate batch task file without processing")
    parser.add_argument("--tail", type=int, metavar="N", help="Show last N results")
    parser.add_argument("--filter", metavar="STATUS", help="Filter results by status (success/failed)")
    parser.add_argument("--summary", action="store_true", help="Generate batch summary report")
    parser.add_argument("--summary-format", choices=["text", "json", "markdown"], default="text",
                        help="Summary output format (default: text)")
    parser.add_argument("--timeout", type=int, default=120, help="Request timeout in seconds (default: 120)")
    parser.add_argument("--version", action="store_true", help="Show version information")
    parser.add_argument("--cache-stats", action="store_true", help="Show scrape cache statistics")
    parser.add_argument("--clear-cache", action="store_true", help="Clear the scrape cache")
    parser.add_argument("--search", metavar="TERM", help="Search results by company name")
    parser.add_argument("--search-field", default="company_name",
                        choices=["company_name", "url", "response", "error_message"],
                        help="Field to search in (default: company_name)")
    parser.add_argument("--count", action="store_true", help="Count results (use with --filter)")
    parser.add_argument("--sort", metavar="FIELD",
                        choices=["processing_time", "tokens_used", "company_name", "completed_at"],
                        help="Sort results by field")
    parser.add_argument("--desc", action="store_true", help="Sort in descending order")
    parser.add_argument("--since", metavar="DATE", help="Filter results since date (YYYY-MM-DD)")
    parser.add_argument("--before", metavar="DATE", help="Filter results before date (YYYY-MM-DD)")
    parser.add_argument("--output", "-o", metavar="FILE", help="Write output to file instead of stdout")
    parser.add_argument("--json", action="store_true", help="Output results as JSON (for --search, --tail, --count, --stats)")

    args = parser.parse_args()

    # Paths
    task_file = Path(args.task) if args.task else DEFAULT_TASK_FILE
    results_file = Path(args.results) if args.results else DEFAULT_RESULTS_FILE

    # Handle test
    if args.test:
        bridge = LMBridge(endpoint=args.endpoint, verbose=True)
        print(f"Testing connection to {args.endpoint}...")
        if bridge.test_connection():
            print("Connected to LM Studio")
        else:
            print("Failed to connect to LM Studio")
        return

    # Handle list models
    if args.list_models:
        bridge = LMBridge(endpoint=args.endpoint)
        models = bridge.list_models()
        if args.json:
            print(json.dumps({"models": models, "count": len(models), "endpoint": args.endpoint}, indent=2))
        elif models:
            print("Available LM Studio models:")
            for m in models:
                print(f"  - {m}")
        else:
            print("No models found or LM Studio not running")
        return

    # Handle clear
    if args.clear:
        clear_state()
        return

    # Handle health check
    if args.health:
        result = health_check(args.endpoint)
        if result["status"] == "healthy":
            print(f"✓ Endpoint healthy: {result['endpoint']}")
            print(f"  Response time: {result['response_time']}")
            print(f"  Models: {result['models_available']} available")
            if result['models']:
                print(f"  Active: {', '.join(result['models'])}")
        else:
            print(f"✗ Endpoint {result['status']}: {result['endpoint']}")
            if "error" in result:
                print(f"  Error: {result['error']}")
        return

    # Handle version
    if args.version:
        info = get_version_info()
        print(f"LM Studio Bridge v{info['version']}")
        print(f"  Build:    {info['build_date']}")
        print(f"  Endpoint: {info['endpoint']}")
        print(f"  Model:    {info['model']}")
        print(f"  Python:   {info['python']}")
        return

    # Handle cache stats
    if args.cache_stats:
        stats = get_cache_stats()
        if stats.get("exists"):
            print(f"Cache: {stats['file']}")
            print(f"  Entries: {stats['entries']}")
            print(f"  Valid:   {stats.get('valid', 'N/A')}")
            print(f"  Expired: {stats.get('expired', 'N/A')}")
            print(f"  Size:    {stats.get('size_human', 'N/A')}")
            print(f"  TTL:     {stats.get('ttl_hours', CACHE_TTL_HOURS)} hours")
        else:
            print(f"Cache not found: {stats['file']}")
        return

    # Handle clear cache
    if args.clear_cache:
        if clear_cache():
            print(f"Cache cleared: {DEFAULT_CACHE_FILE}")
        else:
            print(f"No cache to clear: {DEFAULT_CACHE_FILE}")
        return

    # Handle search
    if args.search:
        matches = search_results(results_file, term=args.search, field=args.search_field)

        if args.sort:
            matches = sort_results(matches, field=args.sort, reverse=args.desc)

        if args.json:
            output = json.dumps(matches, indent=2, default=str)
        else:
            output_lines = []
            output_lines.append(f"Found {len(matches)} result(s) matching '{args.search}' in {args.search_field}")
            for r in matches:
                status_icon = "✓" if r.get("status") == "success" else "✗"
                company = r.get("company_name", r.get("task_id", "Unknown"))
                output_lines.append(f"  {status_icon} {company}")
            output = "\n".join(output_lines)

        if args.output:
            Path(args.output).write_text(output)
            print(f"Output written to: {args.output}")
        else:
            print(output)
        return

    # Handle count
    if args.count:
        counts = count_results(results_file, status=args.filter)
        if args.json:
            print(json.dumps(counts, indent=2))
        elif args.filter:
            print(f"{counts['total']} {args.filter} result(s)")
        else:
            print(f"Total: {counts['total']}")
            print(f"  Success: {counts['success']}")
            print(f"  Failed:  {counts['failed']}")
        return

    # Handle date filtering
    if args.since or args.before:
        matches = filter_by_date(results_file, since=args.since, before=args.before)
        output_lines = []
        date_desc = []
        if args.since:
            date_desc.append(f"since {args.since}")
        if args.before:
            date_desc.append(f"before {args.before}")
        output_lines.append(f"Found {len(matches)} result(s) {' and '.join(date_desc)}")

        if args.sort:
            matches = sort_results(matches, field=args.sort, reverse=args.desc)

        for r in matches:
            status_icon = "✓" if r.get("status") == "success" else "✗"
            company = r.get("company_name", r.get("task_id", "Unknown"))
            completed = r.get("completed_at", "")[:10] if r.get("completed_at") else ""
            output_lines.append(f"  {status_icon} {company} ({completed})")

        output = "\n".join(output_lines)
        if args.output:
            Path(args.output).write_text(output)
            print(f"Output written to: {args.output}")
        else:
            print(output)
        return

    # Handle clear results
    if args.clear_results:
        if clear_results(results_file):
            print(f"Cleared results file: {results_file}")
        else:
            print(f"No results file to clear: {results_file}")
        return

    # Handle validate batch file
    if args.validate:
        result = validate_batch_file(task_file)
        if result["valid"]:
            print(f"✓ Batch file valid: {result['file']}")
        else:
            print(f"✗ Batch file invalid: {result['file']}")

        if result.get("summary"):
            print(f"\nSummary:")
            for k, v in result["summary"].items():
                print(f"  {k}: {v}")

        if result.get("warnings"):
            print(f"\nWarnings:")
            for w in result["warnings"]:
                print(f"  ⚠ {w}")

        if result.get("errors"):
            print(f"\nErrors:")
            for e in result["errors"]:
                print(f"  ✗ {e}")
        return

    # Handle tail results
    if args.tail:
        results = tail_results(results_file, n=args.tail, status_filter=args.filter)
        if not results:
            if args.json:
                print("[]")
            else:
                print("No results found")
            return

        if args.json:
            print(json.dumps(results, indent=2, default=str))
        else:
            print(f"Last {len(results)} result(s):")
            print("-" * 60)
            for r in results:
                status_icon = "✓" if r.get("status") == "success" else "✗"
                company = r.get("company_name", r.get("task_id", "Unknown"))
                proc_time = r.get("processing_time", 0)
                tokens = r.get("tokens_used", 0)
                print(f"{status_icon} {company}")
                print(f"    Time: {proc_time:.1f}s | Tokens: {tokens}")
                if r.get("error_message"):
                    print(f"    Error: {r['error_message']}")
                print()
        return

    # Handle filter results (standalone, without --tail)
    if args.filter and not args.tail:
        results = filter_results(results_file, status=args.filter)
        if args.json:
            print(json.dumps({"filter": args.filter, "count": len(results), "results": results}, indent=2, default=str))
        else:
            print(f"Found {len(results)} {args.filter} result(s)")
            for r in results:
                company = r.get("company_name", r.get("task_id", "Unknown"))
                print(f"  - {company}")
        return

    # Handle summary report
    if args.summary:
        # --json flag overrides summary_format
        fmt = "json" if args.json else args.summary_format
        report = generate_summary(results_file, output_format=fmt)
        print(report)
        return

    # Handle create task
    if args.create:
        task_id = create_task(
            prompt=args.create,
            tools_enabled=args.tools,
            output_file=task_file
        )
        print(f"Task created: {task_id}")
        print(f"File: {task_file}")
        return

    # Handle read result
    if args.read:
        result = read_latest_result(results_file)
        if result:
            print(json.dumps(result, indent=2))
        else:
            print("No results found")
        return

    # Handle create batch task
    if args.create_batch:
        batch_file = Path(args.create_batch)
        if not batch_file.exists():
            print(f"File not found: {batch_file}")
            return
        try:
            companies = json.loads(batch_file.read_text())
            if isinstance(companies, dict) and "companies" in companies:
                companies = companies["companies"]
            batch_id = create_batch_task(
                companies=companies,
                tools_enabled=args.tools,
                output_file=task_file
            )
            print(f"Batch task created: {batch_id}")
            print(f"Companies: {len(companies)}")
            print(f"File: {task_file}")
        except json.JSONDecodeError as e:
            print(f"Invalid JSON: {e}")
        return

    # Handle export to CSV
    if args.export_csv:
        csv_path = export_results_to_csv(
            results_file=results_file,
            include_response=args.include_response
        )
        print(f"Exported to: {csv_path}")
        return

    # Handle stats
    if args.stats:
        stats = get_batch_stats(results_file)
        if args.json:
            print(json.dumps(stats, indent=2, default=str))
        elif "error" in stats:
            print(f"Error: {stats['error']}")
        else:
            print("\n=== Batch Statistics ===")
            print(f"Total tasks: {stats['total_tasks']}")
            print(f"Successful: {stats['successful']}")
            print(f"Failed: {stats['failed']}")
            print(f"Success rate: {stats['success_rate']}")
            print(f"\nTiming:")
            print(f"  Total: {stats['timing']['total_time']}")
            print(f"  Average: {stats['timing']['avg_time']}")
            print(f"  Min: {stats['timing']['min_time']}")
            print(f"  Max: {stats['timing']['max_time']}")
            print(f"\nTokens:")
            print(f"  Total: {stats['tokens']['total']}")
            print(f"  Average: {stats['tokens']['avg']}")
            if stats['failed_companies']:
                print(f"\nFailed companies:")
                for fc in stats['failed_companies']:
                    print(f"  - {fc['company']}: {fc['error']}")
        return

    # Handle export failed
    if args.export_failed:
        result = export_failed_companies(results_file)
        print(result)
        return

    # Handle duplicate check
    if args.check_duplicates:
        dup_file = Path(args.check_duplicates)
        if not dup_file.exists():
            print(f"File not found: {dup_file}")
            return
        try:
            companies = json.loads(dup_file.read_text())
            if isinstance(companies, dict) and "companies" in companies:
                companies = companies["companies"]
            result = check_duplicates_batch(companies)
            print(f"\nDuplicate Check Summary:")
            print(f"  Total: {result['summary']['total']}")
            print(f"  New: {result['summary']['new_count']}")
            print(f"  Duplicates: {result['summary']['duplicate_count']}")
            if result['duplicates']:
                print(f"\nDuplicates found:")
                for dup in result['duplicates']:
                    print(f"  - {dup['company_name']} -> exists as '{dup['existing_company']}'")
            if result['new']:
                print(f"\nNew companies:")
                for c in result['new']:
                    print(f"  - {c['company_name']}")
        except json.JSONDecodeError as e:
            print(f"Invalid JSON: {e}")
        return

    # Handle batch processing
    if args.batch:
        _ = process_batch(
            task_file=task_file,
            results_file=results_file,
            endpoint=args.endpoint,
            verbose=args.verbose,
            resume=not args.no_resume,
            skip_duplicates=not args.include_duplicates,
            delay=args.delay,
            max_retries=args.max_retries,
            retry_backoff=args.retry_backoff,
            dry_run=args.dry_run,
            limit=args.limit,
            quiet=args.quiet
        )
        return

    # Main bridge operation
    bridge = LMBridge(
        task_file=task_file,
        results_file=results_file,
        endpoint=args.endpoint,
        tools_enabled=args.tools,
        verbose=args.verbose
    )

    # Test connection first
    if not bridge.test_connection():
        print(f"Cannot connect to LM Studio at {args.endpoint}")
        print("Make sure LM Studio is running with a model loaded")
        return

    if args.watch:
        bridge.watch(interval=args.interval)
    else:
        result = bridge.run_once()
        if result:
            print(f"\nTask: {result.task_id}")
            print(f"Status: {result.status}")
            print(f"Time: {result.processing_time:.2f}s")
            if result.tool_calls:
                print(f"Tool calls: {len(result.tool_calls)}")
            print(f"\nResponse:\n{result.response}")
        else:
            print("No task to process")
            print(f"Create a task in: {task_file}")


if __name__ == "__main__":
    main()
