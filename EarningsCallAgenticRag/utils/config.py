"""Centralized configuration for EarningsCallAgenticRag.

All configurable constants should be defined here to ensure consistency
across all agents and modules.
"""

from __future__ import annotations

import os

# =============================================================================
# Model Configuration
# =============================================================================
MAIN_MODEL = os.getenv("MAIN_MODEL", "gpt-5-mini")
HELPER_MODEL = os.getenv("HELPER_MODEL", "gpt-4o-mini")
EMBEDDING_MODEL = os.getenv("EMBEDDING_MODEL", "text-embedding-ada")

# Default temperature for LLM calls
DEFAULT_TEMPERATURE = float(os.getenv("DEFAULT_TEMPERATURE", "0.7"))

# =============================================================================
# Max Tokens Configuration (output limits)
# =============================================================================
# NOTE: GPT-5 reasoning models use part of max_tokens for internal reasoning.
# We need higher limits to accommodate reasoning_tokens + output_tokens.
# Extraction: ~40 facts × ~50 tokens each = ~2000 output tokens, plus ~5000 reasoning
MAX_TOKENS_EXTRACTION = int(os.getenv("MAX_TOKENS_EXTRACTION", "8000"))
# Summary: Direction + explanation ~500 output tokens, plus ~2000 reasoning
MAX_TOKENS_SUMMARY = int(os.getenv("MAX_TOKENS_SUMMARY", "3000"))
# Helper agents: shorter analysis notes ~400 output tokens, plus ~1000 reasoning
MAX_TOKENS_HELPER = int(os.getenv("MAX_TOKENS_HELPER", "2000"))

# =============================================================================
# Return Horizon Configuration (T+N days)
# =============================================================================
RETURN_HORIZON_DAYS = int(os.getenv("RETURN_HORIZON_DAYS", "30"))
RETURN_COLUMN_FALLBACK = os.getenv("RETURN_COLUMN_FALLBACK", "future_3bday_cum_return")

# =============================================================================
# Facts Processing Limits (optimized for token efficiency without accuracy loss)
# =============================================================================
# Main Agent
MAX_FACTS_PER_HELPER = int(os.getenv("MAX_FACTS_PER_HELPER", "30"))  # Reduced from 80
MAX_PEERS = int(os.getenv("MAX_PEERS", "5"))  # Reduced from 10

# Comparative Agent (peer comparison)
MAX_FACTS_FOR_PEERS = int(os.getenv("MAX_FACTS_FOR_PEERS", "25"))  # Reduced from 60
MAX_PEER_FACTS = int(os.getenv("MAX_PEER_FACTS", "40"))  # Reduced from 120

# Historical Performance Agent (financial statements)
MAX_FACTS_FOR_FINANCIALS = int(os.getenv("MAX_FACTS_FOR_FINANCIALS", "20"))  # Reduced from 40
MAX_FINANCIAL_FACTS = int(os.getenv("MAX_FINANCIAL_FACTS", "30"))  # Reduced from 80

# Historical Earnings Agent (past calls)
MAX_FACTS_FOR_PAST = int(os.getenv("MAX_FACTS_FOR_PAST", "20"))  # Reduced from 40
MAX_HISTORICAL_FACTS = int(os.getenv("MAX_HISTORICAL_FACTS", "30"))  # Reduced from 80

# =============================================================================
# Vector Search Configuration
# =============================================================================
MIN_SIMILARITY_SCORE = float(os.getenv("MIN_SIMILARITY_SCORE", "0.3"))
DEFAULT_TOP_K = int(os.getenv("DEFAULT_TOP_K", "10"))

# =============================================================================
# Orchestrator Configuration
# =============================================================================
TIMEOUT_SEC = int(os.getenv("TIMEOUT_SEC", "1000"))
MAX_WORKERS = int(os.getenv("MAX_WORKERS", "10"))
CHUNK_SIZE = int(os.getenv("CHUNK_SIZE", "300"))

# =============================================================================
# Logging Directories
# =============================================================================
TOKEN_LOG_DIR = os.getenv("TOKEN_LOG_DIR", "token_logs")
TIMING_LOG_DIR = os.getenv("TIMING_LOG_DIR", "timing_logs")
NEO4J_LOG_DIR = os.getenv("NEO4J_LOG_DIR", "neo4j_logs")

# =============================================================================
# PG DB Agent Configuration
# =============================================================================
# Historical Performance Agent
PG_AGENT_HISTORICAL_LIMIT = int(os.getenv("PG_AGENT_HISTORICAL_LIMIT", "4"))
PG_AGENT_PERFORMANCE_FACTS_LIMIT = int(os.getenv("PG_AGENT_PERFORMANCE_FACTS_LIMIT", "20"))

# Historical Earnings Agent
PG_AGENT_TRANSCRIPT_EXCERPT_LENGTH = int(os.getenv("PG_AGENT_TRANSCRIPT_EXCERPT_LENGTH", "2000"))
PG_AGENT_EARNINGS_FACTS_LIMIT = int(os.getenv("PG_AGENT_EARNINGS_FACTS_LIMIT", "15"))
PG_AGENT_HISTORICAL_EXCERPTS_LIMIT = int(os.getenv("PG_AGENT_HISTORICAL_EXCERPTS_LIMIT", "3"))

# Comparative Agent
PG_AGENT_COMPARATIVE_FACTS_LIMIT = int(os.getenv("PG_AGENT_COMPARATIVE_FACTS_LIMIT", "15"))

# =============================================================================
# PostgreSQL DB Scoring Weights (for peer comparison)
# =============================================================================
PEER_SCORE_WEIGHTS = {
    "revenue": float(os.getenv("PEER_SCORE_REVENUE", "0.9")),
    "net_income": float(os.getenv("PEER_SCORE_NET_INCOME", "0.85")),
    "eps": float(os.getenv("PEER_SCORE_EPS", "0.85")),
    "revenue_growth": float(os.getenv("PEER_SCORE_REVENUE_GROWTH", "0.8")),
    "earnings_day_return": float(os.getenv("PEER_SCORE_EARNINGS_RETURN", "0.75")),
}

# =============================================================================
# PostgreSQL Connection Pool
# =============================================================================
PG_POOL_MINCONN = int(os.getenv("PG_POOL_MINCONN", "1"))
PG_POOL_MAXCONN = int(os.getenv("PG_POOL_MAXCONN", "10"))

# =============================================================================
# Feature Flags
# =============================================================================
USE_PG_DB_AGENTS = os.getenv("USE_PG_DB_AGENTS", "false").lower() == "true"
INGEST_HISTORY_QUARTERS = int(os.getenv("INGEST_HISTORY_QUARTERS", "4"))

# =============================================================================
# Short-Only Trading Strategy Configuration
# =============================================================================
# Enable short-only strategy mode (only generate SHORT signals when confident)
SHORT_ONLY_MODE = os.getenv("SHORT_ONLY_MODE", "false").lower() == "true"

# Direction score gating thresholds for SHORT signals
# - SHORT signal: Direction score in [SHORT_DIRECTION_MIN, SHORT_DIRECTION_MAX]
# - Direction scores from LLM range 0-10 (0=very bearish, 10=very bullish)
# - Grid search result (RANDOM 200 samples, 2024, no leakage):
#   * D:2-3 + Version C filter achieves 100% precision (22 trades)
#   * D:3 alone achieves 83.3% precision (42 trades)
# - IMPORTANT: Lock MIN=2 to avoid drift into untested D:0-1 zone
#   (D:0-1 had no samples in 200-sample test; D:2 had 1 sample PANW)
SHORT_DIRECTION_MIN = int(os.getenv("SHORT_DIRECTION_MIN", "2"))
SHORT_DIRECTION_MAX = int(os.getenv("SHORT_DIRECTION_MAX", "3"))

# Return horizon for T+N evaluation (should match RETURN_HORIZON_DAYS)
# Used to fetch actual return from pg_client.get_price_analysis()
SHORT_RETURN_HORIZON = int(os.getenv("SHORT_RETURN_HORIZON", "30"))

# =============================================================================
# Two-Stage Short Signal: PENDING_SHORT → SHORT_ACTIVE (Version C Strategy)
# =============================================================================
# Enable two-stage confirmation for SHORT signals
# Stage 1 (T+0): LLM generates PENDING_SHORT if direction_score in range
# Stage 2 (T+1): PENDING_SHORT → SHORT_ACTIVE only if filters pass
# This prevents lookahead bias by using only T+0 and T+1 data for T+1 entry
SHORT_TWO_STAGE_ENABLED = os.getenv("SHORT_TWO_STAGE_ENABLED", "true").lower() == "true"

# Version C Confirmation Filters (applied at T+1 for entry decision)
# - return_t_max: Maximum T+0 return (earnings day). Require a big drop to confirm bearish.
#   Grid search result: -6% achieves 100% precision, -4% achieves 96.9%
SHORT_FILTER_RETURN_T_MAX_PCT = float(os.getenv("SHORT_FILTER_RETURN_T_MAX_PCT", "-6.0"))

# - return_1d_min: Minimum T+1 return. Avoid catching falling knives (already crashed too much).
#   Grid search result: -4% combined with rt_max=-6 achieves 100% precision
SHORT_FILTER_RETURN_1D_MIN_PCT = float(os.getenv("SHORT_FILTER_RETURN_1D_MIN_PCT", "-4.0"))

# - cap_1d: Optional maximum T+1 return cap. Set to None to disable.
#   Use this to filter out stocks that rebounded too much by T+1.
#   Cross-regime analysis (2019-2025Q2): cap_1d=2.0% improves precision from 81% to 91%
#   by filtering out strong rebound losers (e.g., HOOD +42.5% worst case eliminated)
SHORT_FILTER_CAP_1D_PCT = (
    float(os.getenv("SHORT_FILTER_CAP_1D_PCT"))
    if os.getenv("SHORT_FILTER_CAP_1D_PCT") else None
)

# =============================================================================
# Market Stress Gate (0-token risk management)
# =============================================================================
# When market volatility exceeds threshold, strategy automatically goes HOLD_ALL
# This prevents trading during extreme market conditions (e.g., COVID 2020Q1)
# where single-stock earnings analysis is dominated by macro factors.
#
# Implementation: Calculate SPY 20-day realized volatility (daily std * sqrt(252))
# If vol > threshold, skip all trading for that earnings date.

MARKET_STRESS_GATE_ENABLED = os.getenv("MARKET_STRESS_GATE_ENABLED", "false").lower() == "true"
MARKET_STRESS_INDEX = os.getenv("MARKET_STRESS_INDEX", "SPY")
MARKET_STRESS_VOL_WINDOW = int(os.getenv("MARKET_STRESS_VOL_WINDOW", "20"))
# Annualized volatility threshold (typical SPY vol is 15-20%, crisis can be 50%+)
MARKET_STRESS_VOL_THRESHOLD_ANNUAL = float(os.getenv("MARKET_STRESS_VOL_THRESHOLD_ANNUAL", "50.0"))
