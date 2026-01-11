# CLI Bridge - Claude Code ↔ Gemini CLI Communication System

**Version**: 1.4.0
**Created**: December 8, 2025
**Updated**: January 10, 2026
**Purpose**: Enable real-time strategic analysis and directive exchange between Claude Code (executor) and Gemini CLI (analyst), plus LM Studio and Ollama bridges for local LLM integration

---

## Overview

The CLI Bridge enables two AI assistants to collaborate on complex tasks:

```
┌─────────────────────────────────────────────────────────────────┐
│                        CLI BRIDGE SYSTEM                         │
├─────────────────────────────────────────────────────────────────┤
│                                                                  │
│   Claude Code (Executor)              Gemini CLI (Analyst)       │
│   ┌──────────────────┐               ┌──────────────────┐       │
│   │ • Process data   │               │ • Analyze patterns│       │
│   │ • Execute tasks  │ ──task.json──→│ • Detect drift   │       │
│   │ • Apply changes  │               │ • Find anomalies │       │
│   │ • Track progress │←directives.json│ • Recommend fixes│       │
│   └──────────────────┘               └──────────────────┘       │
│              ↓                              ↓                    │
│         checkpoint.json (shared state)                           │
│                                                                  │
└─────────────────────────────────────────────────────────────────┘
```

---

## Two Operational Modes

### Mode 1: Full Auto (ExactlyAI Prospecting)

Automated batch processing with strategic optimization:

```python
from cli_bridge.bridge_controller import BridgeController

bridge = BridgeController(mode="full_auto")
bridge.start_session(total_items=100, batch_size=25)

for batch in batches:
    # Process batch
    results = process_companies(batch)

    # Request Gemini analysis
    bridge.write_task(
        task_type="company_scoring_analysis",
        payload={
            "batch_start": batch.start,
            "batch_end": batch.end,
            "companies_processed": [c.name for c in results],
            "current_metrics": {
                "pass_rate": calculate_pass_rate(results),
                "competitor_detection_rate": calculate_competitor_rate(results)
            }
        }
    )

    # Wait for and apply directives
    directives = bridge.wait_for_directives(timeout=120)
    if directives:
        bridge.apply_directives(directives)
        adjustments = bridge.get_current_adjustments()
        update_scoring_config(adjustments)

    bridge.batch_complete(len(batch))

bridge.end_session()
```

### Mode 2: Semi-Manual (Screenplay/Scripts)

Human-directed analysis with AI assistance:

```python
from cli_bridge.bridge_controller import BridgeController

bridge = BridgeController(mode="semi_manual")
bridge.start_session()

# Send instruction from Claude Code terminal
bridge.send_instruction(
    "Review scenes 15-30 for Benteen character voice consistency"
)

# Get Gemini's analysis
response = bridge.get_gemini_response(timeout=180)
print(f"Gemini says: {response}")

# Send follow-up
bridge.send_instruction("Now check dialogue pacing in scenes 22-25")
response = bridge.get_gemini_response(timeout=180)

bridge.end_session()
```

---

## File Structure

```
cli_bridge/
├── README.md                    # This file
├── bridge_controller.py         # Claude Code bridge module
├── gemini_prompt.md            # Persistent base prompt for Gemini
├── gemini_task_handler.md      # Per-task processing instructions
│
├── schemas/
│   ├── task_schema.json        # Task payload schema
│   ├── directives_schema.json  # Directives response schema
│   └── checkpoint_schema.json  # Checkpoint state schema
│
├── gemini_task.json            # [Runtime] Current task from Claude Code
├── gemini_directives.json      # [Runtime] Response from Gemini
├── checkpoint.json             # [Runtime] Shared session state
└── bridge_log.json             # [Runtime] Audit trail
```

---

## Quick Start

### Step 1: Start Gemini CLI Session

Open a terminal and start Gemini CLI:

```bash
gemini
```

Paste the contents of `gemini_prompt.md` to initialize Gemini as the analyst.

### Step 2: Start Claude Code Session

In another terminal, start Claude Code:

```bash
claude
```

Import and use the bridge:

```python
from cli_bridge.bridge_controller import BridgeController

bridge = BridgeController(mode="full_auto")
bridge.start_session(total_items=25, batch_size=25)
```

### Step 3: Process and Analyze

Claude Code writes tasks → Gemini analyzes → Gemini writes directives → Claude Code applies

---

## Task Types

| Task Type | Use Case | Data Source |
|-----------|----------|-------------|
| `company_scoring_analysis` | ExactlyAI prospecting optimization | SQLite database |
| `script_analysis` | Screenplay editing and review | Markdown/Fountain files |
| `pattern_detection` | General pattern analysis | Any data source |
| `custom` | Semi-manual user instructions | User-specified |

---

## Directive Categories

Gemini can provide recommendations in these areas:

### 1. Scoring Adjustments
```json
"scoring_adjustments": {
  "b2b_intensity_weight": 0.35,
  "revenue_weight": 0.25,
  "industry_match_weight": 0.20
}
```

### 2. Competitor Detection
```json
"competitor_detection": {
  "add_keywords": ["growth marketing", "demand gen"],
  "remove_keywords": ["marketing manager"],
  "industry_flags": ["Marketing & Advertising"],
  "domain_patterns": ["*agency*"]
}
```

### 3. Prompt Tuning
```json
"prompt_tuning": {
  "emphasis_add": ["Check /about for B2B signals"],
  "emphasis_remove": ["Blog analysis"],
  "context_inject": "Manufacturing sites often sparse - don't penalize"
}
```

### 4. Tier Boundaries
```json
"tier_boundaries": {
  "a_tier_revenue_min": 50000000,
  "a_tier_revenue_max": 400000000,
  "employee_sweet_spot_min": 75,
  "employee_sweet_spot_max": 400
}
```

### 5. Skip Rules
```json
"skip_rules": {
  "url_contains": [".gov", ".edu"],
  "name_contains": ["University", "Foundation"],
  "description_flags": ["non-profit"]
}
```

### 6. Observed Patterns
```json
"observed_patterns": {
  "false_positive_rate": 0.15,
  "highest_yield_industries": ["machinery", "manufacturing"],
  "problematic_patterns": ["'Solutions' suffix = often competitor"]
}
```

---

## Polling and Timing

| Parameter | Default | Description |
|-----------|---------|-------------|
| Poll interval | 60s | How often Gemini checks for new tasks |
| Directive timeout | 120s | How long Claude Code waits for response |
| Batch size | 25 | Companies per batch before analysis |

### Recommended Timing

- **25-company batch**: 2-5 minutes to process
- **Gemini analysis**: 30-60 seconds
- **Full cycle**: ~5-6 minutes per batch

---

## Database Access

Gemini queries the ExactlyAI database directly:

```python
import sqlite3

db_path = '/home/bigdogla/projects/Ryan_Mark/exactlyai_industry_mapping.db'
conn = sqlite3.connect(db_path)

# Key tables:
# - company_evaluations (610 records)
# - known_competitors (480 records)
# - competitor_taxonomy (18 records)
# - ideal_client_profiles (10 records)
# - industry_mapping (647 records)
```

---

## Error Handling

### Claude Code Side

```python
directives = bridge.wait_for_directives(timeout=120)

if directives is None:
    # Timeout - continue without adjustments
    bridge.skip_directives("Timeout - using current config")

elif directives.get("status") == "error":
    # Gemini reported an error
    print(f"Gemini error: {directives.get('error_message')}")
    bridge.skip_directives("Gemini error")
```

### Gemini Side

```json
{
  "directive_id": "dir_error_20251208",
  "task_id": "task_xyz",
  "status": "error",
  "error_message": "Could not access database - file locked",
  "retry_suggested": true
}
```

---

## Session Commands

Commands Gemini recognizes in task context:

| Command | Action |
|---------|--------|
| `PAUSE` | Stop processing, maintain state |
| `RESUME` | Continue processing |
| `RESET` | Clear accumulated patterns |
| `STATUS` | Report current state |
| `EXIT` | Graceful shutdown |

---

## Logging

All bridge activity is logged to `bridge_log.json`:

```json
{
  "events": [
    {
      "timestamp": "2025-12-08T14:30:00",
      "session_id": "session_20251208_143000",
      "event": "Task written",
      "data": {"task_id": "task_xyz", "task_type": "company_scoring_analysis"}
    },
    {
      "timestamp": "2025-12-08T14:31:15",
      "session_id": "session_20251208_143000",
      "event": "Directives applied",
      "data": {"directive_id": "dir_abc", "changes": ["Competitor keywords: +2"]}
    }
  ]
}
```

---

## Test Data

A test CSV with 24 pre-categorized companies is available:

```
/home/bigdogla/projects/Ryan_Mark/bridge_test_25_companies.csv

Buckets:
- GOOD_FIT (8) - Expected: PASS
- COMPETITOR (5) - Expected: EXCLUDE
- EDGE_CASE (6) - Expected: REVIEW
- EXCLUSION (5) - Expected: EXCLUDE
```

---

## Integration with ExactlyAI Prospector

To integrate with the existing prospector scoring system:

```python
# In your prospector main loop
from cli_bridge.bridge_controller import BridgeController

bridge = BridgeController(mode="full_auto")
bridge.start_session(total_items=len(companies), batch_size=25)

for i, batch in enumerate(chunks(companies, 25)):
    # Score batch
    results = score_companies(batch)
    save_to_database(results)

    # Request analysis every batch
    bridge.write_task(
        task_type="company_scoring_analysis",
        payload={
            "batch_start": i * 25,
            "batch_end": (i + 1) * 25,
            "companies_processed": [c['name'] for c in results]
        }
    )

    # Apply Gemini's recommendations
    directives = bridge.wait_for_directives()
    if directives:
        bridge.apply_directives(directives)

        # Update scoring config with adjustments
        adjustments = bridge.get_current_adjustments()
        if 'competitor_detection' in adjustments:
            add_competitor_keywords(adjustments['competitor_detection']['add_keywords'])

    bridge.batch_complete(len(batch))

bridge.end_session()
```

---

## Troubleshooting

### "Waiting for Gemini..." hangs

1. Check Gemini CLI is running
2. Verify `gemini_task.json` was created
3. Ensure Gemini loaded `gemini_prompt.md`

### "Database locked" error

SQLite concurrent access issue:
```bash
# Enable WAL mode (one-time)
sqlite3 /path/to/db "PRAGMA journal_mode=WAL;"
```

### Directives not applying

Check `gemini_directives.json`:
- Is `task_id` correct?
- Is `status` set to "ready"?
- Is JSON valid?

---

## Future Enhancements

- [ ] WebSocket-based real-time communication
- [ ] Multiple analyst support (Gemini + Claude Code #2)
- [ ] Directive history and rollback
- [ ] Performance metrics dashboard
- [ ] Automatic prompt optimization

---

## LM Bridge & Ollama Bridge - Web Scraping Tools

**Added**: January 10, 2026

### Overview

The LM Bridge and Ollama Bridge enable local LLMs to perform intelligent web scraping through tool-calling. Both bridges share the same 6-tool interface.

```
┌──────────────────────────────────────────────────────────────────┐
│                     LOCAL LLM BRIDGE SYSTEM                       │
├──────────────────────────────────────────────────────────────────┤
│                                                                   │
│   lm_task.json           LM Studio                                │
│   ─────────────────→   (hermes-2-pro)                             │
│                              │                                    │
│                              ▼                                    │
│                        ┌─────────────┐                            │
│                        │  6 TOOLS:   │                            │
│                        │ fetch       │──→ tiered scraping         │
│                        │ search      │──→ content search          │
│                        │ nav_map     │──→ navigation extract      │
│                        │ agentic     │──→ goal-driven scrape      │
│                        │ research    │──→ full company research   │
│                        │ save_db     │──→ SQLite persistence      │
│                        └─────────────┘                            │
│                              │                                    │
│   lm_results.json    ←──────┘                                     │
│                                                                   │
└──────────────────────────────────────────────────────────────────┘
```

### Bridge Configuration

| Bridge | File | Endpoint | Model | Task File | Results File |
|--------|------|----------|-------|-----------|--------------|
| **LM Studio** | `lm_bridge.py` | `192.168.4.77:1235` | `hermes-2-pro-llama-3-8b` | `lm_task.json` | `lm_results.json` |
| **Ollama** | `ollama_bridge.py` | `localhost:11434` | `qwen3:latest` | `ollama_task.json` | `ollama_results.json` |

### Available Tools

#### 1. `fetch_webpage`
Basic web scraping with tiered fallback (curl → Playwright → crawl4ai).

```json
{"url": "https://example.com"}
```

#### 2. `search_page_content`
Search within previously scraped content.

```json
{"url": "https://example.com", "search_term": "products"}
```

#### 3. `extract_nav_map`
Extract navigation structure from website homepage.

```json
{"url": "https://example.com"}
```

#### 4. `agentic_scrape`
Goal-driven smart scraping with reasoning loop.

```json
{
  "url": "https://example.com",
  "page_type": "services",
  "goal": "Find all product offerings and pricing information"
}
```

#### 5. `research_company`
Comprehensive company research - scrapes about, services, and leadership pages in one call.

```json
{
  "url": "https://www.hubspot.com",
  "company_name": "HubSpot"
}
```

#### 6. `save_to_database`
Persist scraped results to SQLite database.

```json
{
  "company_name": "HubSpot",
  "website": "https://www.hubspot.com",
  "about_content": "...",
  "services_content": "...",
  "leadership_content": "..."
}
```

### Usage

#### Quick Start

```bash
# Terminal 1: Activate venv and run bridge
cd /home/bigdogla/projects/Ryan_Mark/cli_bridge
source venv_scraper/bin/activate
python3 lm_bridge.py  # or ollama_bridge.py
```

#### Write a Task

```python
# Create lm_task.json
{
  "task_id": "research-001",
  "prompt": "Research Atlassian at https://www.atlassian.com and save the results to the database",
  "tools_enabled": true
}
```

#### Read Results

```python
# lm_results.json after completion
{
  "task_id": "research-001",
  "status": "success",
  "response": "Research completed. About: 4,635 chars, Services: 11,102 chars, Leadership: 7,190 chars. Saved to database.",
  "tool_calls": [
    {"tool": "research_company", "arguments": {...}},
    {"tool": "save_to_database", "arguments": {...}}
  ],
  "tokens_used": 1877,
  "processing_time": 23.31
}
```

### Caching

Both bridges implement 24-hour TTL caching to avoid redundant scrapes.

| Bridge | Cache File | TTL |
|--------|------------|-----|
| LM Studio | `lm_scrape_cache.json` | 24 hours |
| Ollama | `ollama_scrape_cache.json` | 24 hours |

**Cache hit**: ~0.00s | **Fresh scrape**: ~20-30s

### Database Integration

Results are saved to `exactlyai_industry_mapping.db` table `company_evaluations`:

| Column | Description |
|--------|-------------|
| `company_name` | Company name |
| `url_scraped` | Website URL |
| `website_content` | Homepage content |
| `about_scraped` | About page content |
| `services_scraped` | Services/products page content |
| `leadership_scraped` | Leadership/team page content |
| `scrape_status` | `completed` or `failed` |
| `scrape_date` | Timestamp |
| `scrape_source` | `lm_bridge` or `ollama_bridge` |

### Verified Test Results (Jan 10, 2026)

| Company | Bridge | Status | Pages | Total Chars | Time |
|---------|--------|--------|-------|-------------|------|
| HubSpot | LM Studio | ✅ SUCCESS | 3/3 | 32,676 | 30.5s |
| Atlassian | LM Studio | ✅ SUCCESS | 3/3 | 22,927 | 23.3s |
| Zendesk | Ollama | ✅ SUCCESS | 3/3 | 12,793 | 27.9s |
| Salesforce | Ollama | ✅ SUCCESS | 3/3 | 76,265 | 47.9s |

### Batch Processing

Process multiple companies in a single run:

```bash
# Create a companies file
cat > companies.json << 'EOF'
{
  "companies": [
    {"company_name": "HubSpot", "url": "https://www.hubspot.com"},
    {"company_name": "Zendesk", "url": "https://www.zendesk.com"},
    {"company_name": "Intercom", "url": "https://www.intercom.com"}
  ]
}
EOF

# Create batch task
python3 lm_bridge.py --create-batch companies.json --tools

# Process batch
python3 lm_bridge.py --batch
```

**Batch output:**
```
Processing batch abc123: 3 companies
[1/3] Processing: HubSpot
    OK - 30.5s
[2/3] Processing: Zendesk
    OK - 27.9s
[3/3] Processing: Intercom
    OK - 25.3s

Batch complete: 3/3 successful
```

### Advanced Batch Features (Added: Jan 10, 2026)

#### Duplicate Detection

Check for duplicates before processing to avoid wasted API calls:

```bash
# Check a companies file for duplicates
python3 lm_bridge.py --check-duplicates companies.json

# Output:
# Duplicate Check Summary:
#   Total: 10
#   New: 7
#   Duplicates: 3
#
# Duplicates found:
#   - Acme Corp -> exists as 'ACME Corporation'
#   - Widget Inc -> exists as 'Widget Inc.'
```

During batch processing, duplicates are automatically skipped:
```bash
# Process batch (duplicates skipped by default)
python3 lm_bridge.py --batch

# Force processing of duplicates
python3 lm_bridge.py --batch --include-duplicates
```

#### Resume Interrupted Batches

Batches save progress after each company. If interrupted, resume automatically:

```bash
# Start a batch (interrupted at company 5/10)
python3 lm_bridge.py --batch
# Ctrl+C or network error

# Resume from where it left off
python3 lm_bridge.py --batch
# Output: "Resuming batch: 5/10 already completed"

# Force fresh start (no resume)
python3 lm_bridge.py --batch --no-resume
```

Progress is saved to `lm_batch_state.json` (or `ollama_batch_state.json`).

#### Export Results to CSV

Export batch results for analysis or import into other tools:

```bash
# Export to timestamped CSV
python3 lm_bridge.py --export-csv
# Output: Exported to: /path/to/lm_results_20260110_153521.csv

# Include full response text (larger file)
python3 lm_bridge.py --export-csv --include-response
```

**CSV columns:**
- `task_id`, `company_name`, `url`, `status`, `model`
- `tokens_used`, `processing_time`, `tool_calls_count`
- `batch_id`, `batch_index`, `completed_at`, `error_message`
- `response` (optional, with `--include-response`)

#### Batch Statistics

View detailed statistics about completed batch processing:

```bash
python3 lm_bridge.py --stats
# Output:
# === Batch Statistics ===
# Total tasks: 10
# Successful: 8
# Failed: 2
# Success rate: 80.0%
#
# Timing:
#   Total: 253.6s
#   Average: 25.4s
#   Min: 18.2s
#   Max: 42.1s
#
# Tokens:
#   Total: 4520
#   Average: 452
#
# Failed companies:
#   - Acme Corp: Connection timeout
#   - Widget Inc: Invalid response
```

#### Export Failed Companies

Export failed companies to JSON for retry:

```bash
python3 lm_bridge.py --export-failed
# Output: /path/to/failed_companies_20260110_160000.json

# The exported file can be used directly to retry:
python3 lm_bridge.py --create-batch failed_companies_*.json --tools
python3 lm_bridge.py --batch
```

#### Rate Limiting

Add delays between requests to avoid overwhelming APIs:

```bash
# 2 second delay between each company
python3 lm_bridge.py --batch --delay 2.0

# 0.5 second delay
python3 lm_bridge.py --batch --delay 0.5
```

#### Retry with Exponential Backoff

Automatically retry failed tasks with increasing delays:

```bash
# Retry up to 3 times with 2x backoff (1s, 2s, 4s)
python3 lm_bridge.py --batch --max-retries 3

# Custom backoff multiplier (1s, 3s, 9s)
python3 lm_bridge.py --batch --max-retries 3 --retry-backoff 3.0

# Combine with rate limiting
python3 lm_bridge.py --batch --delay 1.0 --max-retries 2
```

#### Health Check

Quickly verify the endpoint is accessible:

```bash
python3 lm_bridge.py --health
# Output:
# === Endpoint Health Check ===
# Endpoint: http://192.168.4.77:1235/v1/chat/completions
# Status: OK
# Response time: 0.12s
# Models available: 3
#   - hermes-2-pro-llama-3-8b
#   - ...
```

#### Dry Run Mode

Preview what a batch would process without actually running:

```bash
python3 lm_bridge.py --batch --dry-run
# Output:
# === DRY RUN - Batch abc123 ===
# Total companies: 10
# Mode: Resume enabled
# Skip duplicates: Yes
#
# Companies to process:
#   1. HubSpot - https://www.hubspot.com
#   2. Zendesk - https://www.zendesk.com
#   ...
#
# No changes made.
```

#### Limit Processing

Process only the first N companies (useful for testing):

```bash
# Process just first 3 companies
python3 lm_bridge.py --batch --limit 3

# Combine with dry-run to preview
python3 lm_bridge.py --batch --dry-run --limit 5
```

#### Quiet Mode

Suppress all output except errors:

```bash
# Silent processing (great for cron jobs)
python3 lm_bridge.py --batch --quiet

# Combine with other options
python3 lm_bridge.py --batch --quiet --limit 10 --delay 1.0
```

#### Clear Results

Reset the results file to start fresh:

```bash
python3 lm_bridge.py --clear-results
# Output: Results file cleared: /path/to/lm_results.json
```

#### Validate Batch File

Check a batch file for errors before processing:

```bash
python3 lm_bridge.py --validate
# Output:
# ✓ Batch file valid: /path/to/lm_task.json
#
# Summary:
#   batch_id: batch_20260110
#   total_companies: 10
#   unique_urls: 10
#
# Warnings:
#   ⚠ 2 companies missing 'company_name'
```

#### Tail Results

Show the last N results from the results file:

```bash
# Show last 5 results
python3 lm_bridge.py --tail 5

# Show last 10 failed results
python3 lm_bridge.py --tail 10 --filter failed

# Output:
# Last 5 result(s):
# ------------------------------------------------------------
# ✓ HubSpot
#     Time: 30.5s | Tokens: 1877
#
# ✗ Acme Corp
#     Time: 5.2s | Tokens: 0
#     Error: Connection timeout
```

#### Filter Results

Filter results by status:

```bash
# Show all failed results
python3 lm_bridge.py --filter failed
# Output:
# Found 3 failed result(s)
#   - Acme Corp
#   - Widget Inc
#   - Example Co
```

#### Summary Report

Generate a comprehensive summary report:

```bash
# Text format (default)
python3 lm_bridge.py --summary

# JSON format (for programmatic use)
python3 lm_bridge.py --summary --summary-format json

# Markdown format (for documentation)
python3 lm_bridge.py --summary --summary-format markdown

# Output (text):
# ==================================================
#         BATCH PROCESSING SUMMARY REPORT
# ==================================================
# Generated: 2026-01-10 16:30:00
#
# OVERVIEW
# ------------------------------
#   Total Tasks:    10
#   Successful:     8
#   Failed:         2
#   Success Rate:   80.0%
#
# TIMING
# ------------------------------
#   Total Time:     253.6s
#   Average:        25.4s
#   Min:            18.2s
#   Max:            42.1s
```

### CLI Quick Reference

| Command | Description |
|---------|-------------|
| `--batch` | Process batch task file |
| `--create-batch FILE` | Create batch from companies JSON |
| `--check-duplicates FILE` | Check for duplicate URLs |
| `--export-csv` | Export results to CSV |
| `--stats` | Show batch processing statistics |
| `--export-failed` | Export failed companies for retry |
| `--no-resume` | Don't resume from previous progress |
| `--include-duplicates` | Process even if URL exists in DB |
| `--include-response` | Include response text in CSV |
| `--delay SECONDS` | Rate limit between requests |
| `--max-retries N` | Retry failed tasks up to N times |
| `--retry-backoff X` | Exponential backoff multiplier (default 2.0) |
| `--health` | Quick endpoint health check |
| `--clear-results` | Clear the results file |
| `--dry-run` | Preview batch without processing |
| `--limit N` | Process only first N companies |
| `-q, --quiet` | Suppress output except errors |
| `--validate` | Validate batch file without processing |
| `--tail N` | Show last N results |
| `--filter STATUS` | Filter results by status (success/failed) |
| `--summary` | Generate batch summary report |
| `--summary-format FORMAT` | Summary format: text, json, markdown |
| `--timeout SECONDS` | Request timeout (default: 120s) |

### MCP Server Integration

The MCP server exposes all 6 tools directly to Claude Code:

```bash
# Start MCP server manually (for testing)
/home/bigdogla/projects/Ryan_Mark/cli_bridge/launch_mcp_server.sh

# Or it starts automatically via Claude Code config
```

**Config location:** `~/.claude.json` under:
```json
{
  "projects": {
    "/home/bigdogla/projects/Ryan_Mark": {
      "mcpServers": {
        "scraper-tools": {
          "type": "stdio",
          "command": "/home/bigdogla/projects/Ryan_Mark/cli_bridge/launch_mcp_server.sh"
        }
      }
    }
  }
}
```

---

## LM Studio & Ollama Bridge CLI Reference

Both `lm_bridge.py` and `ollama_bridge.py` share identical CLI interfaces for batch processing and result management.

### Quick Commands

```bash
# Version and health
python3 lm_bridge.py --version
python3 lm_bridge.py --health
python3 lm_bridge.py --list-models

# Process tasks
python3 lm_bridge.py --watch --tools        # Watch mode with web tools
python3 lm_bridge.py --batch                 # Process batch file

# View results
python3 lm_bridge.py --tail 10              # Last 10 results
python3 lm_bridge.py --search "Company"     # Search by company name
python3 lm_bridge.py --count                # Count by status
python3 lm_bridge.py --stats                # Batch statistics

# JSON output (for scripting)
python3 lm_bridge.py --tail 5 --json
python3 lm_bridge.py --count --json
python3 lm_bridge.py --stats --json
python3 lm_bridge.py --search "Stripe" --json

# Filtering
python3 lm_bridge.py --filter success       # Only successful
python3 lm_bridge.py --since 2026-01-10     # Since date
python3 lm_bridge.py --sort processing_time --desc  # Sort results

# Cache management
python3 lm_bridge.py --cache-stats
python3 lm_bridge.py --clear-cache
```

### Key Differences

| Feature | lm_bridge.py | ollama_bridge.py |
|---------|--------------|------------------|
| Endpoint | `192.168.4.77:1235` | `localhost:11434` |
| Default Model | `hermes-2-pro-llama-3-8b` | `qwen3:latest` |
| API Format | OpenAI-compatible | Ollama native |

---

**Maintained by**: Claude Code SuperClaude
**Last Updated**: January 10, 2026
**Version**: 1.4.0 - Added --list-models, --json output, --version, --cache-stats, --search, --sort, date filters
