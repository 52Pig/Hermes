class SimpleMovingAverageStrategy:
    def __init__(self, short_window=40, long_window=100, name="Simple Moving Average"):
        self.short_window = short_window
        self.long_window = long_window
        self.name = name

    def generate_signal(self, data_row):
        if 'short_mavg' not in data_row or 'long_mavg' not in data_row:
            return None
        if data_row['short_mavg'] > data_row['long_mavg']:
            return 'buy'
        elif data_row['short_mavg'] < data_row['long_mavg']:
            return 'sell'
        else:
            return None

    def set_parameters(self, params):
        self.short_window, self.long_window = params
