from base_strategy import BaseStrategy

class PriceChange(BaseStrategy):
    def __init__(self, config):
        super().__init__(config)

    def do(self):
        print("execute price change doing...")