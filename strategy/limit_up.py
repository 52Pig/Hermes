from base_strategy import BaseStrategy


class LimitUp(BaseStrategy):
    def __init__(self, config):
        super().__init__(config)
        self.initialize()

    def trade_logic(self):
        # 示例交易逻辑
        param1 = self.config['Settings'].getint('param1', 10)  # 示例默认值
        param2 = self.config['Settings'].getint('param2', 20)  # 示例默认值

        print(f"Executing Limit Up strategy with parameters: {param1}, {param2}")
        self.buy(param1)  # 购买数量为param1
        self.sell(param2)  # 卖出数量为param2