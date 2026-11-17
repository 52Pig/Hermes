# -*- coding: utf-8 -*-
"""
MultiFactor strategy — daily factor score pre-computation.

Runs once before market open (e.g. at 8:30 via scheduler).
Computes multi-factor composite scores for all stocks in stock_pools.json,
saves to D:/tool/dataset/strategy_data/multi_factor_scores_{date}.json

Factor Model (all based on YESTERDAY's close — zero look-ahead):
  - Momentum (20d)      : 35%
  - Short-term Reversal (5d) : 20%
  - Low Volatility (20d)     : 30%
  - Low Turnover (5d)        : 15%

Usage:
    python prepare_multi_factor_data.py [--date 20260522]
"""

import argparse
import json
import logging
import os
import sys
import time
from datetime import datetime, date, timedelta
import numpy as np

logging.basicConfig(level=logging.INFO, format='%(asctime)s [%(levelname)s] %(message)s')
logger = logging.getLogger(__name__)

STOCK_POOL_PATH = "D:/tool/dataset/stock_pools.json"
OUTPUT_DIR = "D:/tool/dataset/strategy_data"
TOP_N = 600  # score top 600 stocks, strategy picks top 30 from these


def load_stock_pool():
    """Load tradeable stock universe from stock_pools.json."""
    with open(STOCK_POOL_PATH, 'r', encoding='utf-8') as f:
        pool = json.load(f)
    # Filter: only stocks with valid code
    codes = []
    for k, v in pool.items():
        code = v.get('code', '')
        if code and '.' in code:
            codes.append(code)
    logger.info(f"Loaded {len(codes)} stocks from stock pool")
    return codes


def fetch_daily_bars(codes, start_date, end_date):
    """Fetch daily OHLCV bars via QMT xtdata.

    Returns: dict[symbol] -> {close: np.array, volume: np.array, dates: [date]}
    """
    from xtquant import xtdata

    # Download history first (required for QMT local data)
    logger.info(f"Downloading history for {len(codes)} stocks...")
    # Download in batches to avoid overwhelming the API
    batch_size = 100
    for i in range(0, len(codes), batch_size):
        batch = codes[i:i + batch_size]
        try:
            xtdata.download_history_data2(batch, '1d', start_date, end_date)
        except Exception as e:
            logger.warning(f"Download batch {i} failed: {e}")
        if i % 500 == 0:
            logger.info(f"  Download progress: {i}/{len(codes)}")

    time.sleep(2)  # let downloads settle

    logger.info("Fetching market data...")
    # Fetch all at once (QMT supports large lists)
    try:
        data = xtdata.get_market_data_ex(
            stock_list=codes,
            period='1d',
            start_time=start_date,
            end_time=end_date,
            field_list=['close', 'volume', 'open', 'high', 'low']
        )
    except Exception as e:
        logger.error(f"get_market_data_ex failed: {e}")
        return {}

    # Convert to numpy arrays
    result = {}
    for code, df in data.items():
        if df is None or len(df) < 30:
            continue
        try:
            closes = df['close'].values.astype(np.float64)
            volumes = df['volume'].values.astype(np.float64)
            # Dates from index
            if hasattr(df.index, 'date'):
                dates = [d.date() if hasattr(d, 'date') else d for d in df.index]
            else:
                dates = []
            result[code] = {
                'close': closes,
                'volume': volumes,
                'dates': [str(d) for d in dates],
            }
        except Exception as e:
            continue

    logger.info(f"Fetched data for {len(result)} stocks")
    return result


def compute_factors(bar_data):
    """Vectorized multi-factor computation.

    All factors use YESTERDAY's close only — zero look-ahead bias.
    """
    factor_weights = {
        'momentum': 0.35, 'reversal': 0.20,
        'lowvol': 0.30, 'turnover': 0.15,
    }

    factor_data = {}
    for code, data in bar_data.items():
        closes = data['close']
        volumes = data['volume']

        if len(closes) < 25:
            continue

        # All factors use closes[:-1] (up to yesterday, idx=-1)
        c = closes[:-1]  # yesterday and before
        v = volumes[:-1]

        if len(c) < 25:
            continue

        # Momentum (20d ending yesterday)
        mom = float(c[-1] / c[-21] - 1) if c[-21] > 0 else 0

        # Reversal (5d ending yesterday, inverted)
        rev = -float(c[-1] / c[-6] - 1) if c[-6] > 0 else 0

        # Low Vol (20d ending yesterday)
        if len(c) >= 21:
            rets = np.diff(c[-21:]) / (c[-21:-1] + 1e-10)
            lowvol = -float(np.std(rets))
        else:
            lowvol = 0

        # Turnover (5d ending yesterday, inverted)
        turnover = -float(np.mean(v[-5:])) if len(v) >= 5 else 0

        # Liquidity filter: avg daily amount > 10M CNY
        if len(v) >= 5 and c[-1] > 0:
            avg_amount = float(np.mean(v[-5:])) * c[-1]
            if avg_amount < 10_000_000:
                continue

        factor_data[code] = {
            'momentum': mom, 'reversal': rev,
            'lowvol': lowvol, 'turnover': turnover,
        }

    if not factor_data:
        return {}

    # Cross-sectional z-score normalization
    codes = list(factor_data.keys())
    normalized = {c: {} for c in codes}

    for fname in factor_weights:
        vals = np.array([factor_data[c][fname] for c in codes])
        mean, std = np.mean(vals), np.std(vals)
        if std < 1e-10:
            std = 1.0
        z = np.clip((vals - mean) / std, -3.0, 3.0)
        for i, c in enumerate(codes):
            normalized[c][fname] = float(z[i])

    # Composite score
    composite = {}
    for c in codes:
        composite[c] = sum(
            normalized[c][f] * w for f, w in factor_weights.items()
        )

    # Rank and keep top N
    ranked = sorted(composite.items(), key=lambda x: x[1], reverse=True)
    top = dict(ranked[:TOP_N])

    logger.info(
        f"Computed factors: {len(factor_data)} stocks, "
        f"top={ranked[0][0]}:{ranked[0][1]:.3f}, "
        f"bottom={ranked[-1][0]}:{ranked[-1][1]:.3f}"
    )
    return top


def main():
    parser = argparse.ArgumentParser(description='MultiFactor data preparation')
    parser.add_argument('--date', default=None, help='Target date (YYYYMMDD), default today')
    args = parser.parse_args()

    if args.date:
        target_date = datetime.strptime(args.date, '%Y%m%d').date()
    else:
        target_date = date.today()

    date_str = target_date.strftime('%Y%m%d')
    output_path = os.path.join(OUTPUT_DIR, f'multi_factor_scores_{date_str}.json')

    # Date range: 120 days of history ending yesterday
    end_date = (target_date - timedelta(days=1)).strftime('%Y%m%d')
    start_date = (target_date - timedelta(days=180)).strftime('%Y%m%d')

    logger.info(f"Preparing MultiFactor data for {date_str}")
    logger.info(f"History: {start_date} ~ {end_date}")

    # Load stock pool
    codes = load_stock_pool()
    if not codes:
        logger.error("No stocks in pool!")
        return

    # Fetch bars
    bar_data = fetch_daily_bars(codes, start_date, end_date)
    if not bar_data:
        logger.error("No bar data fetched!")
        return

    # Compute factors
    scores = compute_factors(bar_data)
    if not scores:
        logger.error("No scores computed!")
        return

    # Save
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    output = {
        'date': date_str,
        'generated_at': datetime.now().isoformat(),
        'stock_count': len(scores),
        'factor_weights': {
            'momentum': 0.35, 'reversal': 0.20,
            'lowvol': 0.30, 'turnover': 0.15,
        },
        'scores': scores,  # {code: composite_score}
    }
    with open(output_path, 'w', encoding='utf-8') as f:
        json.dump(output, f, ensure_ascii=False, indent=2)

    logger.info(f"Saved {len(scores)} scores to {output_path}")


if __name__ == '__main__':
    main()
