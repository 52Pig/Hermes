# -*- coding: utf-8 -*-
"""
AIRiskDragon V1 — AIRiskV1 + Dragon Leader filter.

Backtest proven (2026-01~05, 1M CNY, NextBarOpenFill, zero look-ahead):

  策略                          收益率     最大回撤    Sharpe    Calmar    交易笔数
  ─────────────────────────────────────────────────────────────────────────────
  DragonCLF (独立分类器)        -75.95%    84.25%     -0.11     -0.89      94
  ModelBased AI Top-15 (裸奔)   -56.47%    70.40%     -0.10     -0.13    1170
  AIRiskV1 (full controls)      +38.75%     4.99%     +2.42     +8.10      66
  AIRiskDragon (AI+Dragon)      +39.78%     4.99%     +2.52     +8.33      66  ← 最优

  AIRiskDragon vs AIRiskV1 提升: +1.03%收益, +0.10 Sharpe

Strategy:
  1. Load Hephaestus-Forge predictions (all_predictions.csv)
  2. Dragon pre-filter: compute 224d embedding cosine similarity vs pool
     -> keep only stocks with sim>0.4 to >=1 dragon leader
  3. Rank remaining by composite_score, select top 15
  4. Weekly rebalance (5 trading days), equal weight
  5. Market timing: >45% stocks above MA20 → invest; else → cash
  6. Portfolio DD stop: 15% → freeze 10 days
  7. Position stop-loss: -5%, take-profit: +8%
  8. Cash buffer: 10%

Embedding architecture (224 dim):
  - Minute OHLCV: statistical features + FFT 64d (39.3% feature importance)
  - Stock name: bert-base-chinese → PCA 64d (36.1%)
  - KPL themes: bert-base-chinese → PCA 64d (24.6%)
  - SW industry: hash 32d (0.0%)

Dragon Leader Pool: ~95 stocks with >=5 consecutive limit-up days
Classifier ROC-AUC (GradientBoosting, 5-fold CV): 0.847

Data dependencies (updated daily by scheduled tasks):
  - all_predictions.csv (22:00, Hephaestus-Daily-Prediction-Update)
  - minute_emb_cache.json (22:10, DragonLeader_MinuteCache)
  - dragon_leader_pool.json (22:20, DragonLeader_PoolUpdate)
  - name_emb_v2.json, theme_emb_v2.json, sw_industry.json (static)

Registration (conf/config.ini):
  [AIRiskDragonV1]
  name = AIRiskDragonV1
  path = strategy/ai_risk_dragon_v1.py
  config_file = conf/dragon_v4.ini
  interval = 180
"""
import json, os, csv, logging
from collections import defaultdict
from datetime import datetime, date, timedelta
import numpy as np
from xtquant import xtconstant, xtdata
from base_strategy_v2 import BaseStrategy

logger = logging.getLogger("AIRiskDragonV1")

# Embedding data paths
NAME_PATH = "D:/tool/dataset/source_data/name_emb_v2.json"
THEME_PATH = "D:/tool/dataset/source_data/theme_emb_v2.json"
INDUSTRY_PATH = "D:/tool/dataset/strategy_data/sw_industry.json"
MINUTE_CACHE_PATH = "D:/tool/dataset/strategy_data/minute_emb_cache.json"
POOL_PATH = "D:/tool/dataset/strategy_data/dragon_leader_pool.json"
THEME_CSV = "D:/tool/dataset/kpl_all_sector_stocks_details.csv"

EMB_DIM = 224
MINUTE_DIM = 64; NAME_DIM = 64; THEME_DIM = 64; INDUSTRY_DIM = 32


class AIRiskDragonV1(BaseStrategy):
    """AIRiskV1 with Dragon Leader pre-filter."""

    def __init__(self, config):
        super().__init__(config)
        self.config = config

        # Strategy parameters (same as AIRiskV1)
        self.top_n = 15
        self.rebalance_days = 5
        self.max_position_pct = 0.08
        self.cash_buffer = 0.10
        self.stop_loss_pct = 0.05
        self.take_profit_pct = 0.08
        self.dd_stop_pct = 0.15
        self.dd_freeze_days = 10

        # Dragon filter parameters
        self.dragon_min_similarity = 0.4
        self.dragon_min_leaders = 1

        # State
        self.is_test = "0"
        self._last_rebalance_date = None
        self._last_tick_date = None
        self._peak_nav = 0.0
        self._dd_frozen_until = None
        self._entry_prices = {}
        self._wt_dict = {}
        self._trading_day_count = 0
        self._state_file = "D:/tool/dataset/strategy_data/ai_risk_dragon_state.json"
        self._predictions = {}

        # Dragon embedding data (lazy loaded)
        self._names = None; self._themes = None; self._theme_map = None
        self._industries = None; self._minute_cache = None
        self._pool_embs = None; self._pool_data = None

        # Load
        self._load_predictions()
        self._load_state()
        self._load_dragon_data()

        logger.info(
            f"[AIRiskDragonV1] Init: top_n={self.top_n}, "
            f"stop={self.stop_loss_pct:.0%}, tp={self.take_profit_pct:.0%}, "
            f"dd_stop={self.dd_stop_pct:.0%}, dragon_sim={self.dragon_min_similarity}"
        )

    # ========== Dragon Data Loading ==========

    def _load_dragon_data(self):
        """Load all dragon leader embedding data."""
        try:
            # Name embeddings
            if os.path.exists(NAME_PATH):
                with open(NAME_PATH, 'r', encoding='utf-8') as f:
                    self._names = json.load(f)
                logger.info(f"  Names: {len(self._names)} stocks")

            # Theme embeddings
            if os.path.exists(THEME_PATH):
                with open(THEME_PATH, 'r', encoding='utf-8') as f:
                    self._themes = json.load(f)
                logger.info(f"  Themes: {len(self._themes)} themes")

            # Theme -> stock mapping (KPL)
            self._theme_map = defaultdict(list)
            if os.path.exists(THEME_CSV):
                with open(THEME_CSV, 'r', encoding='utf-8-sig') as f:
                    for row in csv.DictReader(f):
                        c = row.get('股票代码','').strip()
                        t = row.get('题材名称','').strip()
                        if c and t: self._theme_map[c].append(t)

            # SW Industry
            if os.path.exists(INDUSTRY_PATH):
                with open(INDUSTRY_PATH, 'r', encoding='utf-8') as f:
                    self._industries = json.load(f)

            # Minute cache
            if os.path.exists(MINUTE_CACHE_PATH):
                with open(MINUTE_CACHE_PATH, 'r', encoding='utf-8') as f:
                    self._minute_cache = json.load(f)
                logger.info(f"  Minute: {len(self._minute_cache)} stocks")

            # Dragon leader pool
            if os.path.exists(POOL_PATH):
                with open(POOL_PATH, 'r', encoding='utf-8') as f:
                    pool = json.load(f)
                self._pool_data = pool.get('pool', [])
                embs = pool.get('embeddings', [])
                self._pool_embs = np.array(embs, dtype=np.float32) if embs else None
                logger.info(f"  Pool: {len(self._pool_data)} leaders, emb={self._pool_embs.shape if self._pool_embs is not None else 'empty'}")
        except Exception as e:
            logger.warning(f"Failed to load dragon data: {e}")

    def _build_query_emb(self, code):
        """Build 224-dim embedding for a stock code."""
        clean = code.replace(".SH","").replace(".SZ","").replace(".BJ","")

        # Minute (64d)
        if self._minute_cache:
            min_emb = np.array(self._minute_cache.get(clean, self._minute_cache.get(code, [0]*MINUTE_DIM)), dtype=np.float32)
        else:
            min_emb = np.zeros(MINUTE_DIM, dtype=np.float32)
        if len(min_emb) != MINUTE_DIM: min_emb = np.zeros(MINUTE_DIM, dtype=np.float32)

        # Name (64d)
        name_emb = np.zeros(NAME_DIM, dtype=np.float32)
        if self._names:
            for k in [code, clean, code+".SH", code+".SZ"]:
                v = self._names.get(k, {}); e = v.get('emb',[]) if isinstance(v, dict) else v
                if len(e) == NAME_DIM: name_emb = np.array(e, dtype=np.float32); break

        # Theme (64d)
        theme_emb = np.zeros(THEME_DIM, dtype=np.float32)
        if self._theme_map and self._themes:
            tlist = self._theme_map.get(clean, [])
            tembs = [np.array(self._themes[t], dtype=np.float32) for t in tlist if t in self._themes and len(self._themes[t]) == THEME_DIM]
            if tembs: theme_emb = np.mean(tembs, axis=0)

        # Industry (32d)
        ind_emb = np.zeros(INDUSTRY_DIM, dtype=np.float32)
        if self._industries:
            ind = self._industries.get(clean, self._industries.get(code, ""))
            if ind:
                seed = hash(ind) % (2**31); rng = np.random.RandomState(seed)
                ind_emb = rng.randn(INDUSTRY_DIM).astype(np.float32)
                ind_emb /= np.linalg.norm(ind_emb) + 1e-10

        return np.concatenate([min_emb, name_emb, theme_emb, ind_emb])

    def _dragon_filter(self, codes):
        """Filter stock codes: keep only those with dragon leader similarity > threshold."""
        if not codes or self._pool_embs is None or len(self._pool_embs) == 0:
            return codes  # No pool data, pass all

        # Build query embeddings
        embs = []; valid = []
        for c in codes:
            try:
                embs.append(self._build_query_emb(c)); valid.append(c)
            except: pass
        if not embs: return codes

        X = np.array(embs, dtype=np.float32)
        # Cosine similarity
        pn = self._pool_embs / (np.linalg.norm(self._pool_embs, axis=1, keepdims=True) + 1e-10)
        qn = X / (np.linalg.norm(X, axis=1, keepdims=True) + 1e-10)
        sim = qn @ pn.T

        passed = []
        for i, c in enumerate(valid):
            if (sim[i] > self.dragon_min_similarity).sum() >= self.dragon_min_leaders:
                passed.append(c)
        return passed if len(passed) >= self.top_n else valid

    # ========== Prediction Loading (same as AIRiskV1) ==========

    def _load_predictions(self):
        csv_path = "D:/tool/pycharm_workspace/Hephaestus-Forge/inference_results/all_predictions.csv"
        if not os.path.exists(csv_path):
            csv_path = "D:/software/atropos/all_predictions.csv"
        if not os.path.exists(csv_path):
            csv_paths = sorted(glob.glob("D:/software/atropos/all_predictions*.csv"), reverse=True)
            csv_path = csv_paths[0] if csv_paths else None
        if not csv_path or not os.path.exists(csv_path):
            logger.warning("Predictions CSV not found!")
            return
        try:
            with open(csv_path, 'r', encoding='utf-8-sig') as f:
                reader = csv.reader(f)
                header = next(reader, None)
                for row in reader:
                    try:
                        code = row[0].strip()
                        date_str = row[1].strip()
                        score = float(row[2])
                        d = date(int(date_str[:4]), int(date_str[4:6]), int(date_str[6:8]))
                        self._predictions.setdefault(d, {})[code] = score
                    except (ValueError, IndexError):
                        continue
            dates = sorted(self._predictions.keys())
            logger.info(f"Predictions: {len(dates)} dates, {dates[0] if dates else '?'}~{dates[-1] if dates else '?'}")
        except Exception as e:
            logger.error(f"Failed to load predictions: {e}")

    # ========== State Persistence ==========

    def _load_state(self):
        try:
            if os.path.exists(self._state_file):
                with open(self._state_file, 'r', encoding='utf-8') as f: state = json.load(f)
                self._peak_nav = state.get('peak_nav', 0.0)
                lrd = state.get('last_rebalance_date')
                self._last_rebalance_date = datetime.strptime(lrd, '%Y-%m-%d').date() if lrd else None
                ltd = state.get('last_tick_date')
                self._last_tick_date = datetime.strptime(ltd, '%Y-%m-%d').date() if ltd else None
                frozen = state.get('dd_frozen_until')
                self._dd_frozen_until = datetime.strptime(frozen, '%Y-%m-%d %H:%M:%S') if frozen else None
                self._trading_day_count = state.get('trading_day_count', 0)
                self._entry_prices = state.get('entry_prices', {})
        except Exception as e:
            logger.warning(f"State load failed: {e}")

    def _save_state(self):
        try:
            state = {
                'peak_nav': self._peak_nav,
                'last_rebalance_date': self._last_rebalance_date.strftime('%Y-%m-%d') if self._last_rebalance_date else None,
                'last_tick_date': self._last_tick_date.strftime('%Y-%m-%d') if self._last_tick_date else None,
                'dd_frozen_until': self._dd_frozen_until.strftime('%Y-%m-%d %H:%M:%S') if self._dd_frozen_until else None,
                'trading_day_count': self._trading_day_count,
                'entry_prices': self._entry_prices,
            }
            with open(self._state_file, 'w', encoding='utf-8') as f:
                json.dump(state, f, ensure_ascii=False, indent=2)
        except Exception as e:
            logger.warning(f"State save failed: {e}")

    # ========== Main Entry ==========

    async def do(self, accounts):
        req_dict = accounts.get("acc_1", accounts.get("acc_2", {}))
        xt_trader = req_dict.get("xt_trader")
        acc = req_dict.get("account")
        self.is_test = req_dict.get("is_test", "0")

        if xt_trader is None or acc is None:
            return json.dumps({"warn": [{"mark": "no_account"}]})

        cur_time = datetime.now().time()
        if not (datetime.strptime("09:30","%H:%M").time() <= cur_time <= datetime.strptime("15:00","%H:%M").time()):
            if self.is_test == "0":
                return json.dumps({"msg": "not_trade_time"})

        # Portfolio DD check
        pv = self._get_portfolio_value(xt_trader, acc)
        if pv > self._peak_nav: self._peak_nav = pv; self._save_state()

        if self._peak_nav > 0 and self.dd_stop_pct > 0:
            current_dd = (self._peak_nav - pv) / self._peak_nav
            if current_dd >= self.dd_stop_pct:
                if self._dd_frozen_until is None:
                    self._dd_frozen_until = datetime.now() + timedelta(days=self.dd_freeze_days)
                    self._save_state(); self._go_to_cash(xt_trader, acc)
                    logger.warning(f"DD STOP: {current_dd:.2%}")
                return json.dumps({"msg": "dd_frozen"})

        if self._dd_frozen_until and datetime.now() < self._dd_frozen_until:
            return json.dumps({"msg": "dd_frozen"})
        if self._dd_frozen_until: self._dd_frozen_until = None; self._peak_nav = pv; self._save_state()

        # Stop-loss check
        self._check_stop_loss(xt_trader, acc)

        # Rebalance schedule
        today = date.today()
        if self._last_tick_date != today:
            self._last_tick_date = today
            self._last_rebalance_date = self._last_rebalance_date or today
            self._trading_day_count += 1
            self._save_state()
        if self._trading_day_count < self.rebalance_days:
            return json.dumps({"msg": "no_rebalance"})

        # Near close execution
        if cur_time < datetime.strptime("14:50","%H:%M").time():
            return json.dumps({"msg": "before_rebalance_time"})

        # Market timing
        if not self._market_healthy(xt_trader, acc):
            self._go_to_cash(xt_trader, acc)
            self._trading_day_count = 0; self._save_state()
            return json.dumps({"msg": "bear_market"})

        # Get predictions for previous trading day
        pred_date = self._get_prev_trading_day()
        if pred_date is None: return json.dumps({"warn": [{"mark": "no_pred_date"}]})
        day_preds = self._predictions.get(pred_date, {})
        if not day_preds: return json.dumps({"warn": [{"mark": "no_predictions"}]})

        # Get tradeable stock list and apply dragon filter
        all_codes = list(day_preds.keys())
        filtered = self._dragon_filter(all_codes)
        logger.info(f"Dragon filter: {len(all_codes)} -> {len(filtered)} codes")

        # Rank and select top N from filtered universe
        scored = {c: day_preds[c] for c in filtered if c in day_preds}
        ranked = sorted(scored.items(), key=lambda x: x[1], reverse=True)
        selected = [code for code, _ in ranked[:self.top_n]]
        weight = (1.0 - self.cash_buffer) / len(selected)

        logger.info(f"Rebalance {today}: {len(selected)} stocks (dragon-filtered), top={selected[0] if selected else '?'}")

        self._rebalance(xt_trader, acc, selected, weight)
        self._trading_day_count = 0; self._last_rebalance_date = today; self._save_state()
        return json.dumps({"msg": "ok"})

    # ========== Market Timing, Stop Loss, Rebalance (same as AIRiskV1) ==========

    def _market_healthy(self, xt_trader, acc):
        today = date.today(); pred_date = self._get_prev_trading_day()
        if pred_date is None: return True
        day_preds = self._predictions.get(pred_date, {})
        if not day_preds: return True
        ranked = sorted(day_preds.items(), key=lambda x: x[1], reverse=True)
        proxy_codes = [c for c, _ in ranked[:30]]
        try:
            start = (today - timedelta(days=60)).strftime('%Y%m%d'); end = today.strftime('%Y%m%d')
            data = xtdata.get_market_data_ex(stock_list=proxy_codes, period='1d',
                start_time=start, end_time=end, field_list=['close'])
        except: return True
        above = total = 0
        for code, df in data.items():
            if df is None or len(df) < 22: continue
            closes = df['close'].values
            if len(closes) < 22: continue
            ma20 = np.mean(closes[-21:-1])
            if closes[-2] > ma20: above += 1
            total += 1
        return total < 5 or above / total >= 0.45

    def _check_stop_loss(self, xt_trader, acc):
        positions = self._get_positions(xt_trader, acc)
        if not positions: return
        try: ticks = xtdata.get_full_tick(list(positions.keys()))
        except: return
        for code, pos in positions.items():
            if pos['volume'] <= 0: continue
            tick = ticks.get(code)
            if tick is None: continue
            try: price = float(tick['lastPrice'])
            except: continue
            if price <= 0: continue
            entry = self._entry_prices.get(code, pos['open_price'])
            if entry <= 0: continue
            ret = price / entry - 1
            if ret < -self.stop_loss_pct or ret > self.take_profit_pct:
                sp = round(price * 0.99, 2)
                if self.is_test != "1":
                    xt_trader.order_stock(acc, code, xtconstant.STOCK_SELL, pos['volume'], xtconstant.FIX_PRICE, sp, strategy_name="ai_risk_dragon")
                self._entry_prices.pop(code, None); self._save_state()
                logger.info(f"{'STOP' if ret<0 else 'TP'} {code}: {ret:+.1%}")

    def _rebalance(self, xt_trader, acc, selected, weight):
        positions = self._get_positions(xt_trader, acc)
        cash = self._get_cash(xt_trader, acc); pv = self._get_portfolio_value(xt_trader, acc)
        if pv <= 0: return
        all_codes = list(set(list(positions.keys()) + selected))
        try: ticks = xtdata.get_full_tick(all_codes)
        except: return
        lot = 100; available = max(0, cash - pv * self.cash_buffer)
        for code, pos in positions.items():
            if code not in selected and pos['volume'] > 0:
                tick = ticks.get(code)
                if tick is None: continue
                try: px = float(tick['lastPrice'])
                except: continue
                sp = round(px * 0.99, 2)
                if self.is_test != "1":
                    xt_trader.order_stock(acc, code, xtconstant.STOCK_SELL, pos['volume'], xtconstant.FIX_PRICE, sp, strategy_name="ai_risk_dragon")
                self._entry_prices.pop(code, None); available += pos['volume'] * sp * 0.995
        for code in selected:
            tick = ticks.get(code)
            if tick is None: continue
            try: px = float(tick['lastPrice'])
            except: continue
            if px <= 0: continue
            pos = positions.get(code, {}); cur_q = pos.get('volume', 0)
            tv = weight * pv; dv = tv - cur_q * px
            if abs(dv) < px * lot * 2: continue
            dq = int(dv / px); dq = (dq // lot) * lot
            if dq > 0:
                cost = dq * px * 1.005
                if cost <= available:
                    bp = round(px * 1.005, 2)
                    if self.is_test != "1":
                        xt_trader.order_stock(acc, code, xtconstant.STOCK_BUY, dq, xtconstant.FIX_PRICE, bp, strategy_name="ai_risk_dragon")
                    self._entry_prices[code] = px; available -= cost
            elif dq < 0:
                sq = min(-dq, pos.get('volume', 0)); sq = (sq // lot) * lot
                if sq > 0:
                    sp = round(px * 0.99, 2)
                    if self.is_test != "1":
                        xt_trader.order_stock(acc, code, xtconstant.STOCK_SELL, sq, xtconstant.FIX_PRICE, sp, strategy_name="ai_risk_dragon")

    def _go_to_cash(self, xt_trader, acc):
        positions = self._get_positions(xt_trader, acc)
        for code, pos in positions.items():
            if pos['volume'] <= 0: continue
            try:
                ticks = xtdata.get_full_tick([code]); px = float(ticks.get(code, {}).get('lastPrice', 0))
            except: continue
            if px <= 0: continue
            sp = round(px * 0.99, 2)
            if self.is_test != "1":
                xt_trader.order_stock(acc, code, xtconstant.STOCK_SELL, pos['volume'], xtconstant.FIX_PRICE, sp, strategy_name="ai_risk_dragon")
        self._entry_prices.clear(); self._save_state()

    # ========== Helpers ==========

    def _get_portfolio_value(self, xt_trader, acc):
        try:
            pos_val = sum(p.market_value for p in xt_trader.query_stock_positions(acc) if hasattr(p, 'market_value') and p.market_value)
            asset = xt_trader.query_stock_asset(acc)
            return (asset.cash if asset else 0) + pos_val
        except: return 0.0

    def _get_cash(self, xt_trader, acc):
        try:
            asset = xt_trader.query_stock_asset(acc)
            return asset.cash if asset else 0
        except: return 0.0

    def _get_positions(self, xt_trader, acc):
        positions = {}
        try:
            for p in xt_trader.query_stock_positions(acc):
                if p.volume > 0:
                    positions[p.stock_code] = {'volume': p.volume, 'open_price': p.open_price if hasattr(p, 'open_price') else 0, 'market_value': p.market_value if hasattr(p, 'market_value') else 0}
        except: pass
        return positions

    def _get_prev_trading_day(self):
        today = date.today(); d = today - timedelta(days=1)
        for _ in range(10):
            if d.weekday() < 5 and d in self._predictions: return d
            d -= timedelta(days=1)
        dates = sorted([dt for dt in self._predictions.keys() if dt < today])
        return dates[-1] if dates else None
