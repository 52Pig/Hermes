import asyncio
import configparser
import datetime
import importlib.util
import os
import time
import random
import traceback
import pandas as pd
import ast

from utils.logger import setup_logging
from execution.order_protocol import OrderManager, SQLiteOrderStore
from execution.xtquant_gateway import XtQuantCallbackBridge, XtQuantGateway
from execution.managed_trader import ManagedTrader
from execution.multi_gateway import GatewayRouter
from execution.gateway_plugins import load_gateway_plugins
from execution.risk import RiskLimits, RiskManager
from execution.recovery import (
    AccountGateController,
    AccountRecoveryCoordinator,
    QmtRecoverySupervisor,
)
from execution.health import runtime_health

from xtquant.xttrader import XtQuantTrader
from xtquant.xttype import StockAccount
from watchdog.observers import Observer
from watchdog.events import FileSystemEventHandler

class ConfigReloadHandler(FileSystemEventHandler):
    def __init__(self, manager):
        self.manager = manager

    def on_modified(self, event):
        if event.src_path == self.manager.config_file:
            logger.info("检测到配置文件变化，忽略（热加载已禁用）")


# 设置日志记录
logger = setup_logging()
# 全局缓存策略模块，主程序interval间隔调用子策略时，子策略实例不会重复创建，使得子策略中初始化参数能够一直保持不销毁。
_strategy_module_cache = {}

class StrategyManagementService:
    def __init__(self, name, script_path, config_path, interval, accounts, allow_overlap=False):
        self.name = name
        self.script_path = script_path
        self.config_path = config_path
        self.interval = interval
        self.allow_overlap = bool(allow_overlap)
        self._active_task = None
        self.strategy_instance = None
        self.load_strategy()
        self.accounts = accounts

    def load_strategy(self):
        # 检查模块缓存
        if self.script_path in _strategy_module_cache:
            strategy_module = _strategy_module_cache[self.script_path]
        else:
            # 首次加载模块并缓存
            spec = importlib.util.spec_from_file_location("strategy", self.script_path)
            strategy_module = importlib.util.module_from_spec(spec)
            spec.loader.exec_module(strategy_module)
            _strategy_module_cache[self.script_path] = strategy_module
            logger.info(f"首次加载策略模块: {self.script_path}")

        # 创建策略实例
        strategy_class = getattr(strategy_module, self.name)
        config = self.load_config()
        self.strategy_instance = strategy_class(config)
        logger.info(f"策略实例 ID: {id(self.strategy_instance)}")

    def load_config(self):
        config = configparser.ConfigParser()
        config.read(self.config_path, encoding='utf-8')
        return config

    async def run(self):
        '''多个策略同时运行，并行处理任务，执行耗时较长且互不依赖的任务。
          不会等待 execute_strategy() 完成，而是立即进入下一个循环。这允许多个 do() 调用同时进行
        '''
        while True:
            try:
                # 并发调用，执行时间不包含此处策略执行时间
                if self.allow_overlap or self._active_task is None or self._active_task.done():
                    self._active_task = asyncio.create_task(
                        self.execute_strategy(), name=f"strategy-cycle:{self.name}"
                    )
                else:
                    logger.warning("strategy %s is still running; skip overlapping cycle", self.name)
            except Exception as e:
                print(f"Strategy {self.name} encountered an error: {e}")
                # 获取当前文件名和行号
                tb = traceback.format_exc()
                logger.error(f"Strategy {self.__class__.__name__} encountered an error:\n{tb}")
            await asyncio.sleep(self.interval)

    async def execute_strategy(self):
        start_time = time.time()
        try:
            task = asyncio.create_task(self.strategy_instance.do(self.accounts))
            await task
            def _done_callback(t):
                try:
                    exc = t.exception()
                    if exc is not None:
                        logger.error(f"策略 {self.name} 执行异常: {exc}", exc_info=exc)
                    else:
                        logger.info(f"策略 {self.name} 任务完成")
                except Exception as cb_err:
                    logger.error(f"策略 {self.name} 回调异常: {cb_err}")
            task.add_done_callback(_done_callback)
        except Exception as e:
            logger.error(f"策略 {self.name} 创建任务异常: {e}", exc_info=e)
        finally:
            elapsed_time = (time.time() - start_time) * 1000
            logger.info(f"策略 {self.name} 触发耗时: {elapsed_time:.2f} ms")

        # try:
        #     ret = self.strategy_instance.do(self.accounts)  # 这里可能是一个同步方法，如果是异步请使用 await
        #     if ret is not None:
        #         ret_dict = ret
        # except Exception as e:
        #     print(f"Error executing strategy {self.name}: {e}")
        #     tb = traceback.format_exc()
        #     logger.error(f"Strategy {self.__class__.__name__} encountered an error:\n{tb}")
        # end_time = time.time()
        # elapsed_time = round((end_time - start_time) * 1000, 5)
        # logger.info(f"Strategy {self.name}, time: {elapsed_time} ms, ret:{ret_dict}")


class MicroserviceManager:
    def __init__(self, config_file):
        self.config_file = config_file
        self.strategies = []
        self.running_services = {}
        self.recovery_tasks = {}
        self.accounts = {}  # 存储所有账户信息

    def get_runtime_health(self):
        """Return a read-only, broker-free health snapshot for monitoring."""
        return runtime_health(self.accounts)

    async def load_config(self):
        config = configparser.ConfigParser()
        config.read(self.config_file, encoding='utf-8')
        return config

    async def initialize_accounts(self, config):
        # acc_list=***
        # Example only; real account values must remain in the local config.
        acc_dict = ast.literal_eval(config['client'].get("acc_dict", "{}"))
        print(f"acc_dict={acc_dict}")
        # print(f"client={config['client']}")
        # mini_path = config['client'].get("mini_path", "")
        is_test = config['client'].get("is_test", "0")
        execution_enabled = config.getboolean('execution', 'enabled', fallback=True)
        order_store_dir = config.get('execution', 'order_store_dir', fallback='').strip()
        if not order_store_dir:
            raise RuntimeError(
                "[execution] order_store_dir must be configured before account initialization"
            )
        os.makedirs(order_store_dir, exist_ok=True)
        for acc_key, acc_info in acc_dict.items():
            mini_path = acc_info.get("path", "")
            acc_name = acc_info.get("id", "")
            session_id = int(random.randint(100000, 999999))
            xt_trader = XtQuantTrader(mini_path, session_id)
            if "" == mini_path or "" == acc_name:
                print(f'[WARN]mini_path={mini_path} or acc_name={acc_name}')
            acc = StockAccount(acc_name)

            # Install the durable order callback before connecting.  Existing
            # legacy strategies can continue using QMT directly while new
            # strategies consume the managed order fields in this account map.
            order_manager = None
            order_gateway = None
            callback_bridge = None
            recovery_coordinator = None
            reconciliation = {
                'matched_client_order_ids': [],
                'unknown_broker_orders': [],
                'missing_local_order_ids': [],
                'requires_manual_review': False,
            }
            reconciled = False
            legacy_stock_channel = "mini_qmt"
            if execution_enabled:
                try:
                    store_path = os.path.join(order_store_dir, f"orders_{acc_key}.sqlite3")
                    store = SQLiteOrderStore(store_path)
                    order_gateway = XtQuantGateway(xt_trader, acc)
                    # The router currently has Mini QMT enabled by default.
                    # PTrade/FIX/CTP gateways can be injected here once the
                    # broker-specific SDK client and account entitlement exist.
                    default_channel = config.get(
                        "execution", "default_channel", fallback="mini_qmt"
                    ).strip().lower()
                    legacy_stock_channel = config.get(
                        "execution", "legacy_stock_channel", fallback="mini_qmt"
                    ).strip().lower()
                    gateway_router = GatewayRouter(
                        {"mini_qmt": order_gateway}, default_channel=default_channel
                    )
                    load_gateway_plugins(config, gateway_router.gateways)
                    if default_channel not in gateway_router.gateways:
                        raise ValueError(f"default gateway {default_channel!r} is not configured")
                    risk_enabled = config.getboolean(
                        "execution", "risk_enabled", fallback=True
                    )
                    risk_manager = None
                    if risk_enabled:
                        risk_manager = RiskManager(RiskLimits(
                            max_order_equity_fraction=config.getfloat(
                                "execution", "max_order_equity_fraction", fallback=0.25
                            ),
                            max_gross_equity_fraction=config.getfloat(
                                "execution", "max_gross_equity_fraction", fallback=1.0
                            ),
                            max_margin_equity_fraction=config.getfloat(
                                "execution", "max_margin_equity_fraction", fallback=0.40
                            ),
                            max_order_quantity=config.getint(
                                "execution", "max_order_quantity", fallback=1000000
                            ),
                        ))
                    order_manager = OrderManager(
                        store, broker=gateway_router,
                        risk_manager=risk_manager,
                        account_provider=gateway_router if risk_manager else None,
                        submission_enabled=False,
                        submission_gate_reason="startup_reconciliation_required",
                    )
                    gateway_router.bind_manager(order_manager)
                    callback_bridge = XtQuantCallbackBridge(order_manager)
                    callback_bridge.bind_gate_controller(order_manager)
                    xt_trader.register_callback(callback_bridge)
                    recovery_coordinator = AccountRecoveryCoordinator(
                        acc_name, order_manager, gateway_router
                    )
                except Exception:
                    logger.exception("Managed order protocol initialization failed for %s", acc_key)
                    order_manager = None
                    order_gateway = None
                    gateway_router = None
                    callback_bridge = None
                    reconciliation['requires_manual_review'] = True

            startup_gate_revision = (
                order_manager.gate_revision if order_manager is not None else None
            )
            startup_connection_revision = (
                callback_bridge.connection_revision if callback_bridge is not None else None
            )
            xt_trader.start()
            connect_result = xt_trader.connect()
            connection_ready = connect_result == 0
            if callback_bridge is not None and connection_ready:
                callback_bridge.mark_connected_if_unchanged(startup_connection_revision)
                connection_ready = callback_bridge.connected
            subscribe_res = xt_trader.subscribe(acc)

            subscription_ready = connection_ready and subscribe_res == 0
            if order_gateway is not None and connection_ready and subscription_ready:
                reconciliation = recovery_coordinator.reconcile_and_open(startup_gate_revision)
                reconciled = bool(order_manager.submission_enabled)
                if reconciliation.get('requires_manual_review'):
                    logger.error(
                        "Order reconciliation requires manual review for %s: %s",
                        acc_key, reconciliation,
                    )
            elif order_manager is not None:
                reason = (
                    f"qmt_connect_failed:{connect_result}"
                    if connect_result != 0 else f"qmt_subscribe_failed:{subscribe_res}"
                )
                if connect_result == 0 and not connection_ready:
                    reason = "qmt_disconnected_during_connect"
                order_manager.set_gate(False, reason)
                reconciliation['requires_manual_review'] = True
                reconciliation['startup_validation_error'] = reason

            # 封装 xt_trader 和账户到字典中
            managed_trader = (
                ManagedTrader(
                    xt_trader, acc, order_manager,
                    account_id=acc_name,
                    execution_channel=legacy_stock_channel,
                    account_gateway=gateway_router,
                    gate_open=bool(
                        order_manager is not None and order_gateway is not None
                        and reconciled and not reconciliation.get('requires_manual_review', False)
                    ),
                )
                if execution_enabled else xt_trader
            )
            if callback_bridge is not None and isinstance(managed_trader, ManagedTrader):
                callback_bridge.bind_gate_controller(managed_trader)
            self.accounts[acc_key] = {
                'xt_trader': managed_trader,
                'raw_xt_trader': xt_trader,
                'account': acc,
                'acc_name': acc_name,
                'is_test': is_test,
                "mini_path": mini_path,
                "order_manager": order_manager,
                "order_gateway": order_gateway,
                "gateway_router": gateway_router if execution_enabled else None,
                "callback_bridge": callback_bridge,
                "recovery_coordinator": recovery_coordinator,
                "subscription_ready": subscription_ready,
                "order_reconciliation": reconciliation,
                # No managed order is allowed until restart reconciliation is clean.
                "managed_order_gate_open": bool(
                    order_manager is not None
                    and order_gateway is not None
                    and reconciled
                    and not reconciliation.get('requires_manual_review', False)
                    and order_manager.submission_enabled
                ),
            }
            account_gate_controller = AccountGateController(self.accounts[acc_key])
            self.accounts[acc_key]["gate_controller"] = account_gate_controller
            if callback_bridge is not None:
                callback_bridge.bind_gate_controller(account_gate_controller)
            if order_manager is not None:
                account_gate_controller.set_gate(
                    order_manager.submission_enabled,
                    reason=order_manager.submission_gate_reason,
                )

            print(f'[DEBUG] Account {acc_key} initialized, connect_status={connect_result}, subscribe_status={subscribe_res}')

    async def start_services(self):
        config = await self.load_config()
        await self.initialize_accounts(config)  # 初始化所有账户

        for account_key, account_state in self.accounts.items():
            supervisor = QmtRecoverySupervisor(account_state)
            account_state["recovery_supervisor"] = supervisor
            self.recovery_tasks[account_key] = asyncio.create_task(
                supervisor.run(), name=f"qmt-recovery-{account_key}"
            )
        strategies = config['strategy']['name'].split(',')

        for strategy_name in strategies:
            strategy_config = config[strategy_name.strip()]
            name = strategy_config['name']
            script_path = strategy_config['path']
            config_path = strategy_config['config_file']
            interval = int(strategy_config['interval'])  # 单位为秒

            service = StrategyManagementService(name, script_path, config_path, interval, self.accounts)
            self.strategies.append(service)
            # 使用 asyncio.create_task 创建任务并等待
            task = asyncio.create_task(service.run())
            self.running_services[name] = task

    async def monitor_services(self):
        while True:
            for name, task in list(self.running_services.items()):
                if task.done():
                    logger.info(f"Restarting strategy {name}...")
                    try:
                        config = await self.load_config()
                        if name.strip() not in config:
                            logger.warning(f"Strategy {name} not found in config, removing from monitor")
                            del self.running_services[name]
                            continue
                        strategy_config = config[name.strip()]
                        service = StrategyManagementService(
                            name, strategy_config['path'], strategy_config['config_file'],
                            int(strategy_config['interval']), self.accounts)
                        new_task = asyncio.create_task(service.run())
                        self.running_services[name] = new_task
                    except Exception as e:
                        logger.error(f"Failed to restart strategy {name}: {e}", exc_info=e)
            await asyncio.sleep(10)

    async def reload_config(self):
        # while True:
        #     await asyncio.sleep(1800)  # 每30分钟热加载一次配置
            ## 禁用热重载
            # await self.start_services()  # 重新加载配置并启动服务
        """仅重载配置，不重启策略实例"""
        config = configparser.ConfigParser()
        config.read(self.config_file, encoding='utf-8')

        # 更新所有策略实例的配置
        for service in self.strategies:
            new_config = service.load_config()  # 重新解析配置文件
            service.strategy_instance.update_config(new_config)
        logger.info("配置热加载完成")


def _find_latest_quant_date():
    """Find the latest date with downloaded quant data."""
    import glob, re
    files = glob.glob(r"D:\tool\dataset\quant_data\hs_quant_base_*.json")
    dates = []
    for f in files:
        m = re.search(r'(\d{8})', f)
        if m: dates.append(m.group(1))
    return max(dates) if dates else None

async def check_and_update_data():
    """Pre-start: update stale predictions and scores if needed."""
    import subprocess, glob, os
    today = datetime.date.today()
    updated = False

    # Find latest available quant data date (not today — data lags 1 day)
    latest_quant = _find_latest_quant_date()
    if not latest_quant:
        logger.warning("[PreStart] No quant data found, skipping updates")
        return
    latest_dt = datetime.datetime.strptime(latest_quant, "%Y%m%d").date()
    logger.info(f"[PreStart] Latest quant data: {latest_quant}, today: {today}")

    # Check predictions — must cover the latest quant date
    pred_file = r"D:\tool\pycharm_workspace\Hephaestus-Forge\inference_results\all_predictions.csv"
    need_update = True
    if os.path.exists(pred_file):
        df = pd.read_csv(pred_file, nrows=5)
        # Read just last line efficiently
        all_dates = pd.read_csv(pred_file, usecols=['date'])['date']
        max_date_str = str(all_dates.max())[:10]
        if max_date_str >= latest_quant:
            need_update = False
            logger.info(f"[PreStart] Predictions OK (cover {max_date_str} >= {latest_quant})")
    if need_update:
        logger.info(f"[PreStart] Predictions stale (need {latest_quant}), updating...")
        try:
            end_date = latest_dt.strftime("%Y-%m-%d")
            start_date = (latest_dt - datetime.timedelta(days=3)).strftime("%Y-%m-%d")
            subprocess.run([r"D:\tool\pyclient\python396\python.exe", "-u",
                r"D:\tool\pycharm_workspace\Hephaestus-Forge\inference\generate_all_predictions.py",
                "--data_dir", r"D:\tool\dataset\quant_data",
                "--feature_dir", r"D:\tool\dataset\feature_data",
                "--model_path", r"D:\tool\pycharm_workspace\Hephaestus-Forge\checkpoints\best_predictor_model.pth",
                "--meta_file", r"D:\tool\dataset\source_data\train_data_binary_cache\meta.pkl",
                "--config_file", r"D:\tool\pycharm_workspace\Hephaestus-Forge\conf\config.ini",
                "--scaler_file", r"D:\tool\pycharm_workspace\Hephaestus-Forge\feature_scalers.json",
                "--output", pred_file,
                "--start_date", start_date, "--end_date", end_date,
                "--seq_len", "20", "--device", "cuda"], timeout=3600)
            updated = True
        except Exception as e:
            logger.warning(f"[PreStart] Prediction update failed: {e}")

    # Check DragonV6 scores — same logic, compare with latest quant date
    score_pattern = r"D:\tool\pycharm_workspace\Hermes\logs\dragon_v6_scores_*.json"
    scores = sorted(glob.glob(score_pattern))
    need_score = True
    if scores:
        latest_score = os.path.basename(scores[-1])
        date_str = latest_score.replace("dragon_v6_scores_","").replace(".json","")
        try:
            dt = datetime.datetime.strptime(date_str, "%Y%m%d").date()
            if (latest_dt - dt).days <= 2:
                need_score = False
                logger.info(f"[PreStart] DragonV6 scores OK ({date_str})")
        except: pass
    if need_score:
        logger.info(f"[PreStart] DragonV6 scores stale, updating...")
        try:
            subprocess.run([r"D:\tool\pyclient\python396\python.exe", "-u",
                r"D:\software\atropos\export_v20_scores.py",
                "--output", r"D:\tool\pycharm_workspace\Hermes\logs"], timeout=3600)
            updated = True
        except Exception as e:
            logger.warning(f"[PreStart] DragonV6 score update failed: {e}")

    # Check ThemeMomentum scores
    theme_pattern = r"D:\tool\pycharm_workspace\Hermes\logs\theme_momentum_scores_*.json"
    theme_files = sorted(glob.glob(theme_pattern))
    need_theme = True
    if theme_files:
        latest = theme_files[-1]
        date_str = os.path.basename(latest).replace("theme_momentum_scores_","").replace(".json","")
        try:
            dt = datetime.datetime.strptime(date_str, "%Y%m%d").date()
            if (latest_dt - dt).days <= 2:
                need_theme = False
                logger.info(f"[PreStart] ThemeMomentum scores OK ({date_str})")
        except: pass
    if need_theme:
        logger.info(f"[PreStart] ThemeMomentum scores stale, updating...")
        try:
            subprocess.run([r"D:\tool\pyclient\python396\python.exe", "-u",
                r"D:\software\atropos\export_theme_momentum_scores.py",
                "--output", r"D:\tool\pycharm_workspace\Hermes\logs"], timeout=3600)
            updated = True
        except Exception as e:
            logger.warning(f"[PreStart] ThemeMomentum score update failed: {e}")

    if updated:
        logger.info("[PreStart] Data update complete")

async def main():
    await check_and_update_data()
    manager = MicroserviceManager('conf/config.ini')
    await manager.start_services()

    # 启动文件监视
    event_handler = ConfigReloadHandler(manager)
    observer = Observer()
    observer.schedule(event_handler, path='conf/', recursive=False)
    observer.start()

    # 使用 asyncio.create_task 启动监控协程
    monitor_task = asyncio.create_task(manager.monitor_services())
    reload_task = asyncio.create_task(manager.reload_config())  # 将 reload_config 包装为 Task

    await asyncio.gather(
        monitor_task,
        reload_task    # 保留原有热加载逻辑（可选）
    )

    # 使用 asyncio.create_task 启动监控协程
    # monitor_task = asyncio.create_task(manager.monitor_services())

    # 等待监控任务和配置重新加载任务
    # await asyncio.gather(
    #     manager.reload_config(),
    #     monitor_task
    # )

if __name__ == "__main__":
    asyncio.run(main())
