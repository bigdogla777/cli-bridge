#!/usr/bin/env python3
"""
Tiered Multi-Page Company Scraper
Primary: Playwright (3 attempts)
Fallback: crawl4ai (2 attempts)

Scrapes 4 pages per company: homepage, about, services, leadership
"""

import json
import os
import re
import subprocess
import sqlite3
import sys
import asyncio
from datetime import datetime
from html.parser import HTMLParser
from urllib.parse import urljoin

# Add crawl4ai venv to path for fallback
sys.path.insert(0, '/home/bigdogla/crawl4ai-venv/lib/python3.12/site-packages')

# Configuration
DB_PATH = '/home/bigdogla/projects/Ryan_Mark/exactlyai_industry_mapping.db'
RAW_DIR = '/home/bigdogla/projects/Ryan_Mark/cli_bridge/raw_extractions/company_scrapes'
PROGRESS_FILE = '/home/bigdogla/projects/Ryan_Mark/cli_bridge/scrape_progress_tiered.json'
FAILURES_FILE = f'/home/bigdogla/projects/Ryan_Mark/cli_bridge/scrape_failures_tiered_{datetime.now().strftime("%Y%m%d")}.txt'
COMPANIES_FILE = '/home/bigdogla/projects/Ryan_Mark/cli_bridge/companies_pending_restart_20251211.json'

# Scraper settings
PLAYWRIGHT_MAX_ATTEMPTS = 3
CRAWL4AI_MAX_ATTEMPTS = 2
PAGE_TIMEOUT = 30

# Page patterns to try
PAGE_PATTERNS = {
    'homepage': ['/', '/home'],
    'about': ['/about', '/about-us', '/company', '/who-we-are'],
    'services': ['/services', '/solutions', '/capabilities', '/what-we-do'],
    'leadership': ['/team', '/leadership', '/our-team', '/about/team', '/management', '/people']
}


class HTMLTextExtractor(HTMLParser):
    """Extract text from HTML, ignoring scripts and styles"""
    def __init__(self):
        super().__init__()
        self.text = []
        self.skip = False

    def handle_starttag(self, tag, attrs):
        if tag in ('script', 'style', 'noscript'):
            self.skip = True

    def handle_endtag(self, tag):
        if tag in ('script', 'style', 'noscript'):
            self.skip = False

    def handle_data(self, data):
        if not self.skip:
            text = data.strip()
            if text:
                self.text.append(text)

    def get_text(self):
        return ' '.join(self.text)


def extract_text_from_html(html):
    """Convert HTML to plain text"""
    parser = HTMLTextExtractor()
    try:
        parser.feed(html)
        return parser.get_text()
    except:
        text = re.sub(r'<script[^>]*>.*?</script>', '', html, flags=re.DOTALL | re.IGNORECASE)
        text = re.sub(r'<style[^>]*>.*?</style>', '', text, flags=re.DOTALL | re.IGNORECASE)
        text = re.sub(r'<[^>]+>', ' ', text)
        text = re.sub(r'\s+', ' ', text)
        return text.strip()


def fetch_with_curl(url, timeout=PAGE_TIMEOUT):
    """Tier 0: Simple curl fetch (fastest, for easy sites)"""
    try:
        result = subprocess.run(
            ['curl', '-sL', '--max-time', str(timeout), '-A',
             'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36', url],
            capture_output=True, text=True, timeout=timeout+5
        )
        if result.returncode == 0 and result.stdout:
            text = extract_text_from_html(result.stdout)
            if len(text) > 100:
                return {'success': True, 'content': text, 'method': 'curl', 'error': None}
        return {'success': False, 'content': '', 'method': 'curl', 'error': f'curl failed or low content'}
    except Exception as e:
        return {'success': False, 'content': '', 'method': 'curl', 'error': str(e)}


def fetch_with_playwright(url, timeout=PAGE_TIMEOUT):
    """Tier 1: Playwright fetch (handles JS rendering)"""
    try:
        # Use the existing playwright_scraper.py
        script_path = '/home/bigdogla/projects/Ryan_Mark/cli_bridge/playwright_scraper.py'
        result = subprocess.run(
            ['python3', script_path, url],
            capture_output=True, text=True, timeout=timeout+10
        )
        if result.returncode == 0 and result.stdout:
            try:
                data = json.loads(result.stdout)
                if data.get('text_content') and len(data['text_content']) > 100:
                    return {
                        'success': True,
                        'content': data['text_content'][:5000],
                        'method': 'playwright',
                        'error': None
                    }
            except json.JSONDecodeError:
                pass
        return {'success': False, 'content': '', 'method': 'playwright', 'error': 'Playwright failed'}
    except subprocess.TimeoutExpired:
        return {'success': False, 'content': '', 'method': 'playwright', 'error': 'Playwright timeout'}
    except Exception as e:
        return {'success': False, 'content': '', 'method': 'playwright', 'error': str(e)}


async def fetch_with_crawl4ai_async(url, timeout=PAGE_TIMEOUT):
    """Tier 2: crawl4ai fetch (fallback with different engine)"""
    try:
        from crawl4ai import AsyncWebCrawler, BrowserConfig, CrawlerRunConfig

        browser_config = BrowserConfig(headless=True, verbose=False)
        crawler_config = CrawlerRunConfig(
            page_timeout=timeout * 1000,
            wait_until='domcontentloaded'
        )

        async with AsyncWebCrawler(config=browser_config) as crawler:
            result = await crawler.arun(url=url, config=crawler_config)

            if result.success and result.markdown:
                content = ' '.join(result.markdown.strip().split())
                if len(content) > 100:
                    return {
                        'success': True,
                        'content': content[:5000],
                        'method': 'crawl4ai',
                        'error': None
                    }
        return {'success': False, 'content': '', 'method': 'crawl4ai', 'error': 'crawl4ai low content'}
    except Exception as e:
        return {'success': False, 'content': '', 'method': 'crawl4ai', 'error': str(e)[:100]}


def fetch_with_crawl4ai(url, timeout=PAGE_TIMEOUT):
    """Synchronous wrapper for crawl4ai"""
    return asyncio.run(fetch_with_crawl4ai_async(url, timeout))


def scrape_url_tiered(url):
    """
    Tiered scraping strategy:
    1. Try curl first (fast, works for simple sites)
    2. Try Playwright up to 3 times (handles JS)
    3. Fallback to crawl4ai up to 2 times (different engine)
    """
    attempts = []

    # Normalize URL
    if not url.startswith(('http://', 'https://')):
        url = 'https://' + url

    # Tier 0: Quick curl attempt
    result = fetch_with_curl(url)
    attempts.append(('curl', result['success']))
    if result['success']:
        result['attempts'] = attempts
        return result

    # Tier 1: Playwright (up to 3 attempts)
    for i in range(PLAYWRIGHT_MAX_ATTEMPTS):
        result = fetch_with_playwright(url)
        attempts.append((f'playwright_{i+1}', result['success']))
        if result['success']:
            result['attempts'] = attempts
            return result

    # Tier 2: crawl4ai fallback (up to 2 attempts)
    for i in range(CRAWL4AI_MAX_ATTEMPTS):
        result = fetch_with_crawl4ai(url)
        attempts.append((f'crawl4ai_{i+1}', result['success']))
        if result['success']:
            result['attempts'] = attempts
            return result

    # All failed
    result['attempts'] = attempts
    result['error'] = f'All {len(attempts)} attempts failed'
    return result


def scrape_page(base_url, page_type):
    """Try multiple URL patterns for a page type using tiered scraping"""
    patterns = PAGE_PATTERNS.get(page_type, ['/'])

    for i, pattern in enumerate(patterns[:3]):
        url = urljoin(base_url, pattern)
        result = scrape_url_tiered(url)

        if result['success']:
            # Check if it's a real page (not 404 in content)
            content = result.get('content', '')
            if '404' not in content[:500] and 'not found' not in content[:500].lower():
                return {
                    'url': url,
                    'url_pattern': pattern,
                    'status': 'success',
                    'content': content,
                    'chars': len(content),
                    'method': result.get('method', 'unknown'),
                    'attempts': result.get('attempts', [])
                }

    return {
        'url': urljoin(base_url, patterns[0]),
        'url_pattern': patterns[0],
        'status': 'not_found',
        'content': 'PAGE NOT FOUND',
        'chars': 0,
        'method': 'all_failed',
        'attempts': result.get('attempts', [])
    }


def save_raw_file(company_name, page_type, data):
    """Save raw JSON file for a page"""
    safe_name = re.sub(r'[^a-z0-9_]', '_', company_name.lower())
    safe_name = re.sub(r'_+', '_', safe_name).strip('_')
    filename = f"{datetime.now().strftime('%Y-%m-%d')}_{safe_name}_{page_type}.json"
    filepath = os.path.join(RAW_DIR, filename)

    with open(filepath, 'w') as f:
        json.dump(data, f, indent=2)

    return filepath


def scrape_company(company):
    """Scrape all 4 pages for a company using tiered approach"""
    company_name = company['company_name']
    website = company['website']

    # Normalize URL
    if not website.startswith('http'):
        website = 'https://' + website

    base_url = website.replace('http://', 'https://')

    pages = {}
    total_chars = 0
    methods_used = []

    for page_type in ['homepage', 'about', 'services', 'leadership']:
        print(f"    Scraping {page_type}...", end=' ', flush=True)
        result = scrape_page(base_url, page_type)
        pages[page_type] = result
        total_chars += result['chars']
        methods_used.append(result.get('method', 'unknown'))

        print(f"{result['method']} - {result['chars']} chars")

        # Save raw file
        raw_data = {
            'company_name': company_name,
            'website': website,
            'page_type': page_type,
            'scraped_at': datetime.now().isoformat(),
            'employee_count': company.get('employee_count'),
            'industry': company.get('industry'),
            **result
        }
        save_raw_file(company_name, page_type, raw_data)

    return {
        'company_name': company_name,
        'website': website,
        'scraped_at': datetime.now().isoformat(),
        'employee_count': company.get('employee_count'),
        'industry': company.get('industry'),
        'pages': pages,
        'total_chars': total_chars,
        'methods_used': methods_used,
        'success': total_chars > 500
    }


def format_content_for_db(scrape_result):
    """Format concatenated content for database"""
    lines = []
    for page_type in ['homepage', 'about', 'services', 'leadership']:
        page = scrape_result['pages'].get(page_type, {})
        content = page.get('content', 'PAGE NOT FOUND')
        method = page.get('method', 'unknown')
        lines.append(f"--- PAGE: {page_type} (via {method}) ---")
        lines.append(content)
        lines.append("")
    return '\n'.join(lines)


def update_database(scrape_result):
    """Update company_evaluations table with scraped content"""
    try:
        conn = sqlite3.connect(DB_PATH)
        cursor = conn.cursor()

        content = format_content_for_db(scrape_result)
        methods = ','.join(set(scrape_result.get('methods_used', [])))

        cursor.execute("""
            UPDATE company_evaluations
            SET website_content = ?,
                website_scraped_date = ?,
                url_scraped = ?,
                notes = COALESCE(notes, '') || ' | TIERED_SCRAPE:' || ?
            WHERE company_name = ?
        """, (
            content,
            scrape_result['scraped_at'],
            scrape_result['website'],
            methods,
            scrape_result['company_name']
        ))

        conn.commit()
        conn.close()
        return True
    except Exception as e:
        print(f"  DB Error: {e}")
        return False


def load_progress():
    """Load progress from checkpoint file"""
    default = {
        'task_id': 'TIERED_SCRAPE',
        'status': 'in_progress',
        'processed_count': 0,
        'total_count': 0,
        'success_count': 0,
        'failure_count': 0,
        'methods_stats': {'curl': 0, 'playwright': 0, 'crawl4ai': 0},
        'last_company': None,
        'last_checkpoint_at': datetime.now().isoformat(),
        'started_at': datetime.now().isoformat()
    }
    if os.path.exists(PROGRESS_FILE):
        with open(PROGRESS_FILE) as f:
            loaded = json.load(f)
            for key, value in default.items():
                if key not in loaded:
                    loaded[key] = value
            return loaded
    return default


def save_progress(progress):
    """Save progress to checkpoint file"""
    progress['last_checkpoint_at'] = datetime.now().isoformat()
    with open(PROGRESS_FILE, 'w') as f:
        json.dump(progress, f, indent=2)


def log_failure(company_name, url, reason):
    """Log failed scrape to failures file"""
    with open(FAILURES_FILE, 'a') as f:
        f.write(f"{datetime.now().isoformat()} | {company_name} | {url} | {reason}\n")


def main():
    """Main scraping loop with tiered fallback"""
    # Load companies
    with open(COMPANIES_FILE) as f:
        data = json.load(f)
    companies = data['companies']

    # Load progress
    progress = load_progress()
    progress['total_count'] = len(companies)

    print("=" * 60)
    print("TIERED MULTI-PAGE SCRAPER")
    print("Strategy: curl → Playwright (3x) → crawl4ai (2x)")
    print("=" * 60)
    print(f"Total companies: {len(companies)}")
    print(f"Previously processed: {progress['processed_count']}")
    print("-" * 60)

    for i, company in enumerate(companies[progress['processed_count']:], start=progress['processed_count']):
        company_name = company['company_name']
        print(f"\n[{i+1}/{len(companies)}] {company_name}")
        print(f"  URL: {company['website']}")

        # Scrape company
        result = scrape_company(company)

        # Update progress
        progress['processed_count'] = i + 1
        progress['last_company'] = company_name

        # Track methods used
        for method in result.get('methods_used', []):
            if method in progress['methods_stats']:
                progress['methods_stats'][method] += 1

        if result['success']:
            progress['success_count'] += 1
            print(f"  ✅ Success - {result['total_chars']} chars")
            update_database(result)
        else:
            progress['failure_count'] += 1
            print(f"  ❌ Failed - {result['total_chars']} chars (all tiers exhausted)")
            log_failure(company_name, company['website'], 'All scrape tiers failed')

        save_progress(progress)

        # Progress report every 5 companies
        if (i + 1) % 5 == 0:
            print(f"\n{'='*60}")
            print(f"📊 PROGRESS: {i+1}/{len(companies)}")
            print(f"✅ Successes: {progress['success_count']}")
            print(f"❌ Failures: {progress['failure_count']}")
            print(f"📈 Methods: curl={progress['methods_stats']['curl']}, "
                  f"playwright={progress['methods_stats']['playwright']}, "
                  f"crawl4ai={progress['methods_stats']['crawl4ai']}")
            print(f"{'='*60}\n")

    print("\n" + "=" * 60)
    print("SCRAPE COMPLETE")
    print(f"Total processed: {progress['processed_count']}")
    print(f"Successes: {progress['success_count']}")
    print(f"Failures: {progress['failure_count']}")
    print(f"Method breakdown: {progress['methods_stats']}")
    print("=" * 60)


if __name__ == '__main__':
    main()
