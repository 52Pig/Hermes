import asyncio
import configparser
import importlib.util
import os
import time
import random
import traceback

from utils.logger import setup_logging

from xtquant.xttrader import XtQuantTrader
from xtquant.xttype import StockAccount
from watchdog.observers import Observer
from watchdog.events import FileSystemEventHandler

class ConfigReloadHandler(FileSystemEventHandler):
    def __init__(self, manager):
        self.manager = manager

    def on_modified(self, event):
        if event.src_path == self.manager.config_file:
            logger.info("检测到配置文件变化，触发热加载")
            asyncio.create_task(self.manager.reload_config())


# 设置日志记录
logger = setup_logging()
# 全局缓存策略模块，主程序interval间隔调用子策略时，子策略实例不会重复创建，使得子策略中初始化参数能够一直保持不销毁。
_strategy_module_cache = {}

class StrategyManagementService:
    def __init__(self, name, script_path, config_path, interval, accounts):
        self.name = name
        self.script_path = script_path
        self.config_path = config_path
        self.interval = interval
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
        # with open(self.config_path, 'r', encoding='utf-8') as f:   ##文件必须存在
        #     config.read_file(f)
        config.read(self.config_path)
        return config

    async def run(self):
        '''多个策略同时运行，并行处理任务，执行耗时较长且互不依赖的任务。
          不会等待 execute_strategy() 完成，而是立即进入下一个循环。这允许多个 do() 调用同时进行
        '''
        while True:
            try:
                # 并发调用，执行时间不包含此处策略执行时间
                asyncio.create_task(self.execute_strategy())
            except Exception as e:
                print(f"Strategy {self.name} encountered an error: {e}")
                # 获取当前文件名和行号
                tb = traceback.format_exc()
                logger.error(f"Strategy {self.__class__.__name__} encountered an error:\n{tb}")
            await asyncio.sleep(self.interval)

    async def execute_strategy(self):
        # 调用策略的交易逻辑并记录执行时间
        start_time = time.time()
        ret_dict = dict()
        try:
            # 异步调用do方法，不等待其完成（根据需求选择是否await）
            task = asyncio.create_task(self.strategy_instance.do(self.accounts))
            # 可选：添加任务完成后的回调
            task.add_done_callback(lambda t: logger.info(f"策略 {self.name} 任务完成"))
        except Exception as e:
            logger.error(f"策略 {self.name} 执行异常: {e}")
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
        self.accounts = {}  # 存储所有账户信息

    async def load_config(self):
        config = configparser.ConfigParser()
        # with open(self.config_file, 'r', encoding='utf-8') as f:
        #     config.read_file(f)
        config.read(self.config_file)
        return config

    async def initialize_accounts(self, config):
        # acc_list=***
        #{'acc_1': {"id": "8886086606", "path": "E:/tool/gjqmt_client/userdata_mini"},
        # "acc_2": {"id": "001500492545", "path": "F:/tools/dgzq_client/userdata_mini"}}
        acc_dict = eval(config['client'].get("acc_dict", "{}"))
        print(f"acc_dict={acc_dict}")
        # print(f"client={config['client']}")
        # mini_path = config['client'].get("mini_path", "")
        is_test = config['client'].get("is_test", "0")
        for acc_key, acc_info in acc_dict.items():
            mini_path = acc_info.get("path", "")
            acc_name = acc_info.get("id", "")
            session_id = int(random.randint(100000, 999999))
            xt_trader = XtQuantTrader(mini_path, session_id)
            xt_trader.start()
            if "" == mini_path or "" == acc_name:
                print(f'[WARN]mini_path={mini_path} or acc_name={acc_name}')
            connect_result = xt_trader.connect()
            acc = StockAccount(acc_name)
            subscribe_res = xt_trader.subscribe(acc)

            # 封装 xt_trader 和账户到字典中
            self.accounts[acc_key] = {
                'xt_trader': xt_trader,
                'account': acc,
                'acc_name': acc_name,
                'is_test': is_test,
                "mini_path": mini_path
            }

            print(f'[DEBUG] Account {acc_key} initialized, connect_status={connect_result}, subscribe_status={subscribe_res}')

    async def start_services(self):
        config = await self.load_config()
        await self.initialize_accounts(config)  # 初始化所有账户

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
                    config = await self.load_config()
                    strategy_config = config[name.strip()]
                    service = StrategyManagementService(name, strategy_config['path'], strategy_config['config_file'], int(strategy_config['interval']), self.accounts)
                    new_task = asyncio.create_task(service.run())
                    self.running_services[name] = new_task
            await asyncio.sleep(10)  # 每10秒检查一次服务状态

    async def reload_config(self):
        # while True:
        #     await asyncio.sleep(1800)  # 每30分钟热加载一次配置
            ## 禁用热重载
            # await self.start_services()  # 重新加载配置并启动服务
        """仅重载配置，不重启策略实例"""
        config = configparser.ConfigParser()
        config.read(self.config_file)

        # 更新所有策略实例的配置
        for service in self.strategies:
            new_config = service.load_config()  # 重新解析配置文件
            service.strategy_instance.update_config(new_config)
        logger.info("配置热加载完成")


async def main():
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
