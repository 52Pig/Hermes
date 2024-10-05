from base_strategy import BaseStrategy

class LowBuyHighSell(BaseStrategy):
    def __init__(self, config):
        super().__init__(config)

    def do(self, accounts):
        print(f"Executing Low Buy High Sell strategy doing...")
        return {"code": "000001.SZ", "action": "buy", "price": "10.5"}