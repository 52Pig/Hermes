from base_strategy import BaseStrategy

class LowBuyHighSell(BaseStrategy):
    def __init__(self, config):
        super().__init__(config)

    def do(self):
        print(f"Executing Low Buy High Sell strategy doing...")