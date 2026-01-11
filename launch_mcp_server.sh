#!/bin/bash
# MCP Server Launcher for scraper-tools
# Activates venv and runs the MCP server

cd /home/bigdogla/projects/Ryan_Mark/cli_bridge
source venv_scraper/bin/activate
exec python3 mcp_scraper_server.py "$@"
