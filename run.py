import asyncio
import configparser
import importlib.util
import os
import time
import traceback
from utils.logger import setup_logging
# 设置日志记录
logger = setup_logging()

class StrategyManagementService:
    def __init__(self, name, script_path, config_path, interval):
        self.name = name
        self.script_path = script_path
        self.config_path = config_path
        self.interval = interval
        self.strategy_instance = None
        self.load_strategy()

    def load_strategy(self):
        spec = importlib.util.spec_from_file_location("strategy", self.script_path)
        strategy_module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(strategy_module)

        # 创建策略实例
        strategy_class = getattr(strategy_module, self.name)
        config = self.load_config()
        self.strategy_instance = strategy_class(config)

    def load_config(self):
        config = configparser.ConfigParser()
        # with open(self.config_path, 'r', encoding='utf-8') as f:   ##文件必须存在
        #     config.read_file(f)
        config.read(self.config_path)
        return config

    async def run(self):
        '''
          多个策略同时运行，并行处理任务，执行耗时较长且互不依赖的任务。
          不会等待 execute_strategy() 完成，而是立即进入下一个循环。这允许多个 do() 调用同时进行
        :return:
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
        try:
            self.strategy_instance.do()  # 这里可能是一个同步方法，如果是异步请使用 await
        except Exception as e:
            print(f"Error executing strategy {self.name}: {e}")
            tb = traceback.format_exc()
            logger.error(f"Strategy {self.__class__.__name__} encountered an error:\n{tb}")
        end_time = time.time()
        elapsed_time = round((end_time - start_time) * 1000, 5)
        logger.info(f"Strategy {self.name} execution time: {elapsed_time} ms")


class MicroserviceManager:
    def __init__(self, config_file):
        self.config_file = config_file
        self.strategies = []
        self.running_services = {}

    async def load_config(self):
        config = configparser.ConfigParser()
        # with open(self.config_file, 'r', encoding='utf-8') as f:
        #     config.read_file(f)
        config.read(self.config_file)
        return config

    async def start_services(self):
        config = await self.load_config()
        strategies = config['strategy']['name'].split(',')

        for strategy_name in strategies:
            strategy_config = config[strategy_name.strip()]
            name = strategy_config['name']
            script_path = strategy_config['path']
            config_path = strategy_config['config_file']
            interval = int(strategy_config['interval'])  # 单位为秒

            service = StrategyManagementService(name, script_path, config_path, interval)
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
                    service = StrategyManagementService(name, strategy_config['path'], strategy_config['config_file'], int(strategy_config['interval']))
                    new_task = asyncio.create_task(service.run())
                    self.running_services[name] = new_task
            await asyncio.sleep(10)  # 每10秒检查一次服务状态

    async def reload_config(self):
        while True:
            await asyncio.sleep(300)  # 每5分钟热加载一次配置
            await self.start_services()  # 重新加载配置并启动服务


async def main():
    manager = MicroserviceManager('conf/config.ini')
    await manager.start_services()
    # 使用 asyncio.create_task 启动监控协程
    monitor_task = asyncio.create_task(manager.monitor_services())

    # 等待监控任务和配置重新加载任务
    await asyncio.gather(
        manager.reload_config(),
        monitor_task
    )

if __name__ == "__main__":
    asyncio.run(main())
