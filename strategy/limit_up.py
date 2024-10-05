from base_strategy import BaseStrategy


class LimitUp(BaseStrategy):
    def __init__(self, config):
        super().__init__(config)

    def do(self):
        print("execute limit up doing...")
        return {"code":"000001.SZ", "action":"buy", "price":"10.5"}