#!/usr/bin/env python3
"""
Cross-Regime Validation Pipeline (2019Q1–2025Q2)
=================================================
Stratified sampling across 26 quarters to validate strategy robustness
across different market regimes:
- 2020 COVID crash
- 2021 bull market
- 2022 bear market
- 2023-2025 mixed environment

Usage:
    # Test mode (2 samples per quarter)
    python test_pipeline_cross_regime.py --test

    # Standard validation (40 per quarter = 1040 total)
    python test_pipeline_cross_regime.py --per-quarter 40

    # Custom date range
    python test_pipeline_cross_regime.py --start 2020Q1 --end 2023Q4 --per-quarter 20

    # Full validation (100 per quarter = 2600 total)
    python test_pipeline_cross_regime.py --per-quarter 100
"""

# Force unbuffered output for real-time monitoring
import sys
import functools
print = functools.partial(print, flush=True)

import argparse
import asyncio
import csv
import json
import logging
import os
import random
import time
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

# Add project root to path
project_root = Path(__file__).parent
sys.path.insert(0, str(project_root))

from dotenv import load_dotenv
load_dotenv()

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S"
)
logger = logging.getLogger(__name__)

# Suppress noisy loggers
logging.getLogger("httpx").setLevel(logging.WARNING)
logging.getLogger("openai").setLevel(logging.WARNING)
logging.getLogger("neo4j").setLevel(logging.WARNING)


def parse_quarter_str(quarter_str: str) -> Tuple[int, int]:
    """Parse quarter string like '2019Q1' into (year, quarter)."""
    quarter_str = quarter_str.upper().strip()
    if 'Q' not in quarter_str:
        raise ValueError(f"Invalid quarter format: {quarter_str}. Expected format: 2019Q1")
    parts = quarter_str.split('Q')
    year = int(parts[0])
    quarter = int(parts[1])
    if quarter < 1 or quarter > 4:
        raise ValueError(f"Quarter must be 1-4, got {quarter}")
    return year, quarter


def generate_quarters_list(start: str, end: str) -> List[Tuple[int, int]]:
    """Generate list of (year, quarter) tuples between start and end inclusive."""
    start_year, start_q = parse_quarter_str(start)
    end_year, end_q = parse_quarter_str(end)

    quarters = []
    year, q = start_year, start_q

    while (year, q) <= (end_year, end_q):
        quarters.append((year, q))
        q += 1
        if q > 4:
            q = 1
            year += 1

    return quarters


def get_stratified_samples(
    quarters: List[Tuple[int, int]],
    per_quarter: int,
    seed: int = 42
) -> Tuple[List[Dict], Dict[Tuple[int, int], int]]:
    """
    Get stratified samples from each quarter.

    Returns:
        - List of sample dicts
        - Dict mapping (year, quarter) to actual sample count
    """
    from pg_client import get_cursor

    all_samples = []
    quarter_counts = {}

    with get_cursor() as cur:
        if cur is None:
            raise RuntimeError("Database connection failed")

        for year, quarter in quarters:
            # Query samples for this quarter with valid price data
            # Note: pct_change_t_plus_1 doesn't exist in price_analysis table
            # return_1d will be fetched by the agentic pipeline during analysis
            cur.execute("""
                SELECT
                    et.symbol,
                    et.year,
                    et.quarter,
                    et.transcript_date_str,
                    c.name as company_name,
                    c.sector,
                    pa.pct_change_t as return_t,
                    pa.pct_change_t_plus_30 as actual_return_30d
                FROM earnings_transcripts et
                JOIN companies c ON et.symbol = c.symbol
                JOIN transcript_content tc ON et.id = tc.transcript_id
                JOIN price_analysis pa ON et.id = pa.transcript_id
                WHERE et.year = %s
                    AND et.quarter = %s
                    AND tc.content IS NOT NULL
                    AND LENGTH(tc.content) > 1000
                    AND pa.pct_change_t IS NOT NULL
                    AND pa.pct_change_t_plus_30 IS NOT NULL
            """, (year, quarter))

            quarter_samples = []
            for row in cur.fetchall():
                actual_ret = float(row["actual_return_30d"])
                return_t = float(row["return_t"]) if row["return_t"] is not None else None

                # Categorize based on actual return
                if actual_ret > 5:
                    category = "POSITIVE"
                elif actual_ret < -5:
                    category = "NEGATIVE"
                else:
                    category = "NEUTRAL"

                quarter_samples.append({
                    "symbol": row["symbol"],
                    "year": row["year"],
                    "quarter": row["quarter"],
                    "transcript_date": row["transcript_date_str"],
                    "company_name": row["company_name"],
                    "sector": row["sector"],
                    "return_t": return_t,
                    "return_1d": None,  # Will be fetched by agentic pipeline
                    "actual_return_30d": actual_ret,
                    "category": category
                })

            # Random sample from this quarter
            if seed is not None:
                random.seed(seed + year * 10 + quarter)  # Unique seed per quarter

            if len(quarter_samples) >= per_quarter:
                selected = random.sample(quarter_samples, per_quarter)
            else:
                selected = quarter_samples  # Take all if fewer than requested

            quarter_counts[(year, quarter)] = len(selected)
            all_samples.extend(selected)

    return all_samples, quarter_counts


def calc_eval_return(return_30d: float, return_1d: float) -> Optional[float]:
    """
    Calculate eval_return (T+1 → T+30) for correct entry point measurement.
    Formula: eval_return = (1 + R_30) / (1 + R_1) - 1
    """
    if return_30d is None or return_1d is None:
        return None
    r_30 = return_30d / 100
    r_1 = return_1d / 100
    if r_1 == -1:  # Avoid division by zero
        return None
    return ((1 + r_30) / (1 + r_1) - 1) * 100


async def analyze_single(
    symbol: str,
    year: int,
    quarter: int,
    main_model: str = None,
    helper_model: str = None,
    prefilter_return_t: float = None,
    enable_prefilter: bool = True,
) -> Dict[str, Any]:
    """Run analysis for a single earnings call."""
    from analysis_engine import analyze_earnings_async

    start_time = time.time()
    result = {
        "symbol": symbol,
        "year": year,
        "quarter": quarter,
        "success": False,
        "error": None,
        "time_seconds": 0,
        "prediction": None,
        "confidence": None,
        "direction_score": None,
        "conviction": None,
        "down_strength": None,
        "trade_signal": None,
        "actual_return": None,
        "return_t": prefilter_return_t,
        "return_1d": None,
        "return_3d": None,
        "return_30d": None,
        "correct": None,
        "summary": None,
        "reasons": None,
        "token_usage": None,
        "agent_notes": None,
        "prefilter_passed": None,
        "prefilter_reason": None,
        "llm_skipped": False,
        "market_stress_flag": None,
        "market_vol_20d": None,
    }

    # Prefilter check
    if enable_prefilter and prefilter_return_t is not None:
        try:
            from EarningsCallAgenticRag.utils.config import SHORT_FILTER_RETURN_T_MAX_PCT
            if prefilter_return_t > SHORT_FILTER_RETURN_T_MAX_PCT:
                result["success"] = True
                result["time_seconds"] = time.time() - start_time
                result["prefilter_passed"] = False
                result["prefilter_reason"] = f"return_t={prefilter_return_t:.2f}% > {SHORT_FILTER_RETURN_T_MAX_PCT}%"
                result["trade_signal"] = "HOLD"
                result["llm_skipped"] = True
                return result
            else:
                result["prefilter_passed"] = True
                result["prefilter_reason"] = f"return_t={prefilter_return_t:.2f}% <= {SHORT_FILTER_RETURN_T_MAX_PCT}%"
        except ImportError:
            pass

    try:
        payload = await analyze_earnings_async(
            symbol=symbol,
            year=year,
            quarter=quarter,
            main_model=main_model,
            helper_model=helper_model,
            skip_cache=True,
        )

        result["success"] = True
        result["time_seconds"] = time.time() - start_time

        agentic = payload.get("agentic_result", {})
        result["prediction"] = agentic.get("prediction")
        result["confidence"] = agentic.get("confidence")
        result["direction_score"] = agentic.get("direction_score")
        result["conviction"] = agentic.get("conviction")
        result["down_strength"] = agentic.get("down_strength")
        result["trade_signal"] = agentic.get("trade_signal")
        result["return_t"] = agentic.get("return_t") or prefilter_return_t
        result["return_1d"] = agentic.get("return_1d")
        result["return_3d"] = agentic.get("return_3d")
        result["return_30d"] = agentic.get("return_30d")
        result["summary"] = agentic.get("summary")
        result["reasons"] = agentic.get("reasons")

        raw = agentic.get("raw", {})
        result["token_usage"] = raw.get("token_usage")
        result["agent_notes"] = raw.get("notes")

        # Market Stress Gate fields
        result["market_stress_flag"] = agentic.get("market_stress_flag")
        result["market_vol_20d"] = agentic.get("market_vol_20d")

        backtest = payload.get("backtest", {})
        if backtest and backtest.get("change_pct") is not None:
            result["actual_return"] = backtest["change_pct"]
        elif payload.get("post_earnings_return") is not None:
            result["actual_return"] = payload["post_earnings_return"]

        if result["return_30d"] is not None:
            result["actual_return"] = result["return_30d"]

    except Exception as e:
        result["error"] = str(e)
        result["time_seconds"] = time.time() - start_time
        logger.error(f"Analysis failed for {symbol} {year}-Q{quarter}: {e}")

    return result


def print_result_line(idx: int, total: int, result: Dict, sample: Dict):
    """Print a single result line."""
    symbol = result["symbol"]
    quarter = f"{result['year']}-Q{result['quarter']}"
    category = sample.get("category", "?")

    if result["success"]:
        if result.get("llm_skipped"):
            return_t = result.get("return_t", 0) or 0
            actual = sample.get("actual_return_30d", 0)
            print(f"  [{idx}/{total}] ⏭ {symbol} {quarter} ({category}) | SKIP rt={return_t:+.1f}% | Actual: {actual:+.2f}%")
            return

        trade_sig = result.get("trade_signal", "-")

        # Market stress gate
        if trade_sig == "HOLD_STRESS":
            market_vol = result.get("market_vol_20d", 0) or 0
            actual = sample.get("actual_return_30d", 0)
            print(f"  [{idx}/{total}] 🛑 {symbol} {quarter} ({category}) | HOLD_STRESS vol={market_vol:.1f}% | Actual: {actual:+.2f}%")
            return

        pred = result.get("prediction", "N/A")
        dir_score = result.get("direction_score")
        actual = sample.get("actual_return_30d", 0)
        time_s = result["time_seconds"]
        dir_str = f"D:{dir_score}" if dir_score is not None else "D:?"
        print(f"  [{idx}/{total}] ✓ {symbol} {quarter} ({category}) | {dir_str} {trade_sig} | Actual: {actual:+.2f}% | {time_s:.1f}s")
    else:
        error = result.get("error", "Unknown")[:50]
        print(f"  [{idx}/{total}] ✗ {symbol} {quarter} ({category}) | ERROR: {error}")


def print_quarterly_breakdown(results: List[Dict], samples: List[Dict]):
    """Print breakdown by quarter with eval_return metrics."""
    sample_lookup = {(s["symbol"], s["year"], s["quarter"]): s for s in samples}

    # Group results by quarter
    quarter_results = {}
    for r in results:
        key = (r["year"], r["quarter"])
        if key not in quarter_results:
            quarter_results[key] = []
        quarter_results[key].append(r)

    print("\n" + "=" * 110)
    print("📊 QUARTERLY BREAKDOWN (using eval_return T+1→T+30)")
    print("=" * 110)
    print(f"{'Quarter':<10} {'Total':<6} {'Skip':<5} {'Stress':<6} {'LLM':<5} {'ACTIVE':<7} {'Prec%':<7} {'AC1%':<7} {'AC2%':<7} {'Mean%':<8} {'P95':<8} {'Worst':<8}")
    print("-" * 110)

    overall_stats = {
        "total": 0, "skip": 0, "stress": 0, "llm": 0, "active": 0,
        "correct_raw": 0, "correct_ac1": 0, "correct_ac2": 0,
        "eval_returns": []
    }

    for (year, quarter) in sorted(quarter_results.keys()):
        qr = quarter_results[(year, quarter)]
        total = len(qr)
        skip = sum(1 for r in qr if r.get("llm_skipped"))
        stress = sum(1 for r in qr if r.get("trade_signal") == "HOLD_STRESS")
        llm = total - skip - stress

        # Find SHORT_ACTIVE signals and calculate eval_return
        active_trades = []
        for r in qr:
            if r.get("trade_signal") == "SHORT_ACTIVE":
                sample_key = (r["symbol"], r["year"], r["quarter"])
                sample = sample_lookup.get(sample_key, {})
                return_30d = sample.get("actual_return_30d")
                return_1d = r.get("return_1d") or sample.get("return_1d")

                eval_ret = calc_eval_return(return_30d, return_1d)
                if eval_ret is not None:
                    active_trades.append({
                        "symbol": r["symbol"],
                        "eval_return": eval_ret,
                        "return_30d": return_30d,
                        "return_1d": return_1d
                    })

        active = len(active_trades)

        if active > 0:
            eval_rets = [t["eval_return"] for t in active_trades]
            correct_raw = sum(1 for e in eval_rets if e < 0)
            correct_ac1 = sum(1 for e in eval_rets if e <= -1)
            correct_ac2 = sum(1 for e in eval_rets if e <= -2)
            prec_raw = correct_raw / active * 100
            prec_ac1 = correct_ac1 / active * 100
            prec_ac2 = correct_ac2 / active * 100
            mean_ret = sum(eval_rets) / len(eval_rets)
            sorted_rets = sorted(eval_rets, reverse=True)
            p95_idx = max(0, int(len(sorted_rets) * 0.95) - 1)
            p95 = sorted_rets[p95_idx] if sorted_rets else 0
            worst = max(eval_rets)

            print(f"{year}Q{quarter:<7} {total:<6} {skip:<5} {stress:<6} {llm:<5} {active:<7} {prec_raw:<7.1f} {prec_ac1:<7.1f} {prec_ac2:<7.1f} {mean_ret:<8.2f} {p95:<8.2f} {worst:<8.2f}")

            # Accumulate for overall
            overall_stats["correct_raw"] += correct_raw
            overall_stats["correct_ac1"] += correct_ac1
            overall_stats["correct_ac2"] += correct_ac2
            overall_stats["eval_returns"].extend(eval_rets)
        else:
            print(f"{year}Q{quarter:<7} {total:<6} {skip:<5} {stress:<6} {llm:<5} {active:<7} {'N/A':<7} {'N/A':<7} {'N/A':<7} {'N/A':<8} {'N/A':<8} {'N/A':<8}")

        overall_stats["total"] += total
        overall_stats["skip"] += skip
        overall_stats["stress"] += stress
        overall_stats["llm"] += llm
        overall_stats["active"] += active

    # Print overall
    print("-" * 110)
    if overall_stats["active"] > 0:
        eval_rets = overall_stats["eval_returns"]
        prec_raw = overall_stats["correct_raw"] / overall_stats["active"] * 100
        prec_ac1 = overall_stats["correct_ac1"] / overall_stats["active"] * 100
        prec_ac2 = overall_stats["correct_ac2"] / overall_stats["active"] * 100
        mean_ret = sum(eval_rets) / len(eval_rets)
        sorted_rets = sorted(eval_rets, reverse=True)
        p95_idx = max(0, int(len(sorted_rets) * 0.95) - 1)
        p95 = sorted_rets[p95_idx] if sorted_rets else 0
        worst = max(eval_rets)

        print(f"{'OVERALL':<10} {overall_stats['total']:<6} {overall_stats['skip']:<5} {overall_stats['stress']:<6} {overall_stats['llm']:<5} {overall_stats['active']:<7} {prec_raw:<7.1f} {prec_ac1:<7.1f} {prec_ac2:<7.1f} {mean_ret:<8.2f} {p95:<8.2f} {worst:<8.2f}")

    return overall_stats


def print_yearly_breakdown(results: List[Dict], samples: List[Dict]):
    """Print breakdown by year."""
    sample_lookup = {(s["symbol"], s["year"], s["quarter"]): s for s in samples}

    # Group results by year
    year_results = {}
    for r in results:
        year = r["year"]
        if year not in year_results:
            year_results[year] = []
        year_results[year].append(r)

    print("\n" + "=" * 110)
    print("📈 YEARLY BREAKDOWN (using eval_return T+1→T+30)")
    print("=" * 110)
    print(f"{'Year':<8} {'Total':<6} {'Skip':<5} {'Stress':<6} {'LLM':<5} {'ACTIVE':<7} {'Prec%':<7} {'AC1%':<7} {'AC2%':<7} {'Mean%':<8} {'P95':<8} {'Worst':<8}")
    print("-" * 110)

    for year in sorted(year_results.keys()):
        yr = year_results[year]
        total = len(yr)
        skip = sum(1 for r in yr if r.get("llm_skipped"))
        stress = sum(1 for r in yr if r.get("trade_signal") == "HOLD_STRESS")
        llm = total - skip - stress

        # Find SHORT_ACTIVE signals
        active_trades = []
        for r in yr:
            if r.get("trade_signal") == "SHORT_ACTIVE":
                sample_key = (r["symbol"], r["year"], r["quarter"])
                sample = sample_lookup.get(sample_key, {})
                return_30d = sample.get("actual_return_30d")
                return_1d = r.get("return_1d") or sample.get("return_1d")

                eval_ret = calc_eval_return(return_30d, return_1d)
                if eval_ret is not None:
                    active_trades.append(eval_ret)

        active = len(active_trades)

        if active > 0:
            correct_raw = sum(1 for e in active_trades if e < 0)
            correct_ac1 = sum(1 for e in active_trades if e <= -1)
            correct_ac2 = sum(1 for e in active_trades if e <= -2)
            prec_raw = correct_raw / active * 100
            prec_ac1 = correct_ac1 / active * 100
            prec_ac2 = correct_ac2 / active * 100
            mean_ret = sum(active_trades) / len(active_trades)
            sorted_rets = sorted(active_trades, reverse=True)
            p95_idx = max(0, int(len(sorted_rets) * 0.95) - 1)
            p95 = sorted_rets[p95_idx] if sorted_rets else 0
            worst = max(active_trades)

            print(f"{year:<8} {total:<6} {skip:<5} {stress:<6} {llm:<5} {active:<7} {prec_raw:<7.1f} {prec_ac1:<7.1f} {prec_ac2:<7.1f} {mean_ret:<8.2f} {p95:<8.2f} {worst:<8.2f}")
        else:
            print(f"{year:<8} {total:<6} {skip:<5} {stress:<6} {llm:<5} {active:<7} {'N/A':<7} {'N/A':<7} {'N/A':<7} {'N/A':<8} {'N/A':<8} {'N/A':<8}")


def save_results_csv(results: List[Dict], samples: List[Dict], filename: str) -> str:
    """Save detailed results to CSV with eval_return."""
    output_path = project_root / filename
    sample_lookup = {(s["symbol"], s["year"], s["quarter"]): s for s in samples}

    rows = []
    for r in results:
        key = (r["symbol"], r["year"], r["quarter"])
        sample = sample_lookup.get(key, {})
        actual_30d = sample.get("actual_return_30d")
        return_1d = r.get("return_1d") or sample.get("return_1d")

        # Calculate eval_return (T+1 → T+30)
        eval_return = calc_eval_return(actual_30d, return_1d)

        trade_signal = r.get("trade_signal")

        # Correctness based on eval_return for SHORT_ACTIVE
        short_correct_eval = None
        short_correct_ac1 = None
        short_correct_ac2 = None
        if trade_signal == "SHORT_ACTIVE" and eval_return is not None:
            short_correct_eval = eval_return < 0
            short_correct_ac1 = eval_return <= -1
            short_correct_ac2 = eval_return <= -2

        row = {
            "symbol": r["symbol"],
            "year": r["year"],
            "quarter": r["quarter"],
            "category": sample.get("category", ""),
            "company_name": sample.get("company_name", ""),
            "sector": sample.get("sector", ""),
            "success": r["success"],
            "error": r.get("error", ""),
            "time_seconds": round(r["time_seconds"], 2),
            "llm_skipped": r.get("llm_skipped", False),
            "prefilter_passed": r.get("prefilter_passed") if r.get("prefilter_passed") is not None else "",
            "prefilter_reason": r.get("prefilter_reason", ""),
            "prediction": r.get("prediction", ""),
            "direction_score": r.get("direction_score") if r.get("direction_score") is not None else "",
            "conviction": round(r["conviction"], 3) if r.get("conviction") is not None else "",
            "down_strength": round(r["down_strength"], 3) if r.get("down_strength") is not None else "",
            "trade_signal": trade_signal or "",
            "confidence": round(r["confidence"], 3) if r.get("confidence") else "",
            "return_t_pct": round(r["return_t"], 2) if r.get("return_t") is not None else "",
            "return_1d_pct": round(return_1d, 2) if return_1d is not None else "",
            "return_3d_pct": round(r["return_3d"], 2) if r.get("return_3d") is not None else "",
            "actual_return_30d_pct": round(actual_30d, 2) if actual_30d is not None else "",
            # Key: eval_return is the correct metric for T+1 entry
            "eval_return_t1_pct": round(eval_return, 2) if eval_return is not None else "",
            "short_correct_eval": short_correct_eval if short_correct_eval is not None else "",
            "short_correct_ac1": short_correct_ac1 if short_correct_ac1 is not None else "",
            "short_correct_ac2": short_correct_ac2 if short_correct_ac2 is not None else "",
            # Market Stress Gate fields
            "market_stress_flag": r.get("market_stress_flag") if r.get("market_stress_flag") is not None else "",
            "market_vol_20d_pct": round(r["market_vol_20d"], 2) if r.get("market_vol_20d") is not None else "",
            "summary": str(r.get("summary") or "")[:500],
            "reasons": json.dumps(r.get("reasons") or []),
            "token_cost_usd": round(float((r.get("token_usage") or {}).get("cost_usd", 0.0) or 0.0), 6),
        }
        rows.append(row)

    if rows:
        fieldnames = list(rows[0].keys())
        with open(output_path, "w", newline="", encoding="utf-8") as f:
            writer = csv.DictWriter(f, fieldnames=fieldnames)
            writer.writeheader()
            writer.writerows(rows)

    return str(output_path)


async def run_test_batch(
    samples: List[Dict],
    main_model: str,
    helper_model: str,
    concurrency: int = 5,
    enable_prefilter: bool = True
) -> List[Dict]:
    """Run analysis on a batch of samples with concurrency."""
    results = []
    total = len(samples)
    completed = 0
    lock = asyncio.Lock()

    semaphore = asyncio.Semaphore(concurrency)

    async def process_sample(idx: int, sample: Dict) -> Dict:
        nonlocal completed
        async with semaphore:
            result = await analyze_single(
                symbol=sample["symbol"],
                year=sample["year"],
                quarter=sample["quarter"],
                main_model=main_model,
                helper_model=helper_model,
                prefilter_return_t=sample.get("return_t"),
                enable_prefilter=enable_prefilter,
            )

            if result["actual_return"] is None:
                result["actual_return"] = sample.get("actual_return_30d")

            # Store return_1d from sample if not in result
            if result["return_1d"] is None:
                result["return_1d"] = sample.get("return_1d")

            async with lock:
                completed += 1
                print_result_line(completed, total, result, sample)

                # Progress update every 50 samples
                if completed % 50 == 0:
                    print(f"\n--- Progress: {completed}/{total} ({completed/total*100:.1f}%) ---\n")

            return result

    if concurrency == 1:
        for i, sample in enumerate(samples, 1):
            result = await analyze_single(
                symbol=sample["symbol"],
                year=sample["year"],
                quarter=sample["quarter"],
                main_model=main_model,
                helper_model=helper_model,
                prefilter_return_t=sample.get("return_t"),
                enable_prefilter=enable_prefilter,
            )

            if result["actual_return"] is None:
                result["actual_return"] = sample.get("actual_return_30d")
            if result["return_1d"] is None:
                result["return_1d"] = sample.get("return_1d")

            results.append(result)
            print_result_line(i, total, result, sample)

            if i % 50 == 0:
                print(f"\n--- Progress: {i}/{total} ({i/total*100:.1f}%) ---\n")
    else:
        print(f"\n🚀 Running with {concurrency} concurrent workers...")
        tasks = [process_sample(i, sample) for i, sample in enumerate(samples, 1)]
        results = await asyncio.gather(*tasks, return_exceptions=True)

        # Handle exceptions
        for i, result in enumerate(results):
            if isinstance(result, Exception):
                sample = samples[i]
                results[i] = {
                    "symbol": sample["symbol"],
                    "year": sample["year"],
                    "quarter": sample["quarter"],
                    "success": False,
                    "error": str(result),
                    "time_seconds": 0,
                    "prediction": None,
                    "confidence": None,
                    "direction_score": None,
                    "conviction": None,
                    "down_strength": None,
                    "trade_signal": None,
                    "actual_return": sample.get("actual_return_30d"),
                    "return_1d": sample.get("return_1d"),
                    "return_30d": None,
                    "llm_skipped": False,
                }

    return results


async def main():
    """Main test runner."""
    parser = argparse.ArgumentParser(description="Cross-Regime Validation Pipeline (2019Q1-2025Q2)")
    parser.add_argument("--test", action="store_true", help="Test mode (2 samples per quarter)")
    parser.add_argument("--start", type=str, default="2019Q1", help="Start quarter (e.g., 2019Q1)")
    parser.add_argument("--end", type=str, default="2025Q2", help="End quarter (e.g., 2025Q2)")
    parser.add_argument("--per-quarter", "-pq", type=int, default=40, help="Samples per quarter (default: 40)")
    parser.add_argument("--concurrency", "-c", type=int, default=5, help="Concurrent workers (default: 5)")
    parser.add_argument("--seed", "-s", type=int, default=42, help="Random seed (default: 42)")
    parser.add_argument("--no-prefilter", action="store_true", help="Disable prefilter")
    args = parser.parse_args()

    # Adjust for test mode
    if args.test:
        per_quarter = 2
        concurrency = 1
    else:
        per_quarter = args.per_quarter
        concurrency = args.concurrency

    enable_prefilter = not args.no_prefilter

    # Generate quarters list
    quarters = generate_quarters_list(args.start, args.end)

    print("=" * 80)
    print("Cross-Regime Validation Pipeline")
    print("=" * 80)
    print(f"Start time: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print(f"Date range: {args.start} → {args.end} ({len(quarters)} quarters)")
    print(f"Samples per quarter: {per_quarter}")
    print(f"Expected total: ~{len(quarters) * per_quarter} samples")
    print(f"Concurrency: {concurrency} workers")
    print(f"Random seed: {args.seed}")
    print(f"Prefilter: {'ENABLED' if enable_prefilter else 'DISABLED'}")

    main_model = os.getenv("MAIN_MODEL", "gpt-5-mini")
    helper_model = os.getenv("HELPER_MODEL", "gpt-4o-mini")
    print(f"Models: main={main_model}, helper={helper_model}")

    print("\n" + "-" * 80)
    print("Fetching stratified samples...")
    print("-" * 80)

    try:
        samples, quarter_counts = get_stratified_samples(quarters, per_quarter, args.seed)
    except Exception as e:
        logger.error(f"Failed to get samples: {e}")
        import traceback
        traceback.print_exc()
        return

    print(f"\nTotal samples: {len(samples)}")
    print("\nSamples per quarter:")
    for (year, quarter), count in sorted(quarter_counts.items()):
        print(f"  {year}Q{quarter}: {count}")

    # Show prefilter estimate
    try:
        from EarningsCallAgenticRag.utils.config import SHORT_FILTER_RETURN_T_MAX_PCT
        prefilter_pass = sum(1 for s in samples if s["return_t"] is not None and s["return_t"] <= SHORT_FILTER_RETURN_T_MAX_PCT)
        print(f"\nPrefilter estimate: {prefilter_pass}/{len(samples)} ({prefilter_pass/len(samples)*100:.1f}%) will run LLM")
    except:
        pass

    print("\n" + "-" * 80)
    print("Running analyses...")
    print("-" * 80)

    start_time = time.time()
    results = await run_test_batch(
        samples, main_model, helper_model,
        concurrency=concurrency,
        enable_prefilter=enable_prefilter
    )
    total_time = time.time() - start_time

    # Print breakdowns
    print_quarterly_breakdown(results, samples)
    print_yearly_breakdown(results, samples)

    # Final summary
    print("\n" + "=" * 80)
    print("FINAL SUMMARY")
    print("=" * 80)

    successful = [r for r in results if r["success"]]
    failed = [r for r in results if not r["success"]]
    llm_skipped = [r for r in successful if r.get("llm_skipped")]
    llm_ran = [r for r in successful if not r.get("llm_skipped")]

    print(f"Total samples: {len(results)}")
    print(f"Successful: {len(successful)} | Failed: {len(failed)}")
    print(f"Prefilter skipped: {len(llm_skipped)} ({len(llm_skipped)/len(successful)*100:.1f}%)")
    print(f"LLM ran: {len(llm_ran)}")

    # Calculate total cost
    total_cost = sum(float((r.get("token_usage") or {}).get("cost_usd", 0) or 0) for r in results)
    print(f"Total token cost: ${total_cost:.4f}")
    print(f"Total time: {total_time:.1f}s ({total_time/60:.1f} min)")

    # Save results
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    start_str = args.start.replace("Q", "q")
    end_str = args.end.replace("Q", "q")
    mode_str = "TEST" if args.test else f"{start_str}_{end_str}_pq{per_quarter}"
    csv_path = save_results_csv(results, samples, f"test_results_cross_regime_{mode_str}_{timestamp}.csv")
    print(f"\n📁 Results saved to: {csv_path}")

    # Copy to desktop
    import shutil
    desktop_path = Path.home() / "Desktop" / Path(csv_path).name
    shutil.copy(csv_path, desktop_path)
    print(f"📁 Copied to Desktop: {desktop_path}")

    print(f"\nEnd time: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print("=" * 80)


if __name__ == "__main__":
    asyncio.run(main())
