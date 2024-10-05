class BaseStrategy:
    def __init__(self, config):
        self.config = config
        print(f"Initializing strategy with parameters: {self.config}")

    def buy(self, quantity):
        print(f"Buying {quantity} units.")

    def sell(self, quantity):
        print(f"Selling {quantity} units.")