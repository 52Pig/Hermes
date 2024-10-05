class BaseStrategy:
    def __init__(self, config):
        self.config = config
        print(f"Initializing strategy with parameters: {self.config}")

    def do(self):
        print(f"do...")