import backtrader as bt
import datetime


# 创建策略
class MovingAverageCrossoverStrategy(bt.Strategy):
    params = (
        ('fast_length', 10),
        ('slow_length', 30),
    )

    def log(self, txt, dt=None):
        dt = dt or self.datas[0].datetime.date(0)
        print(f'{dt.isoformat()}, {txt}')  # 输出日志

    def __init__(self):
        # 添加移动平均线指标
        self.fast_ma = bt.indicators.SimpleMovingAverage(
            self.datas[0], period=self.params.fast_length)
        self.slow_ma = bt.indicators.SimpleMovingAverage(
            self.datas[0], period=self.params.slow_length)

        # 交叉信号
        self.crossover = bt.indicators.CrossOver(self.fast_ma, self.slow_ma)

    def next(self):
        # 买入信号
        if self.crossover > 0:
            self.log(f'买入信号, {self.data.close[0]}')
            if not self.position:
                self.buy()

        # 卖出信号
        elif self.crossover < 0:
            self.log(f'卖出信号, {self.data.close[0]}')
            if self.position:
                self.sell()


# 创建Cerebro引擎
cerebro = bt.Cerebro()

# 添加策略
cerebro.addstrategy(MovingAverageCrossoverStrategy)

# 获取数据
data = bt.feeds.GenericCSVData(
    dataname='dataset/btdemo.csv',  # 替换为你的CSV文件路径
    fromdate=datetime.datetime(2019, 1, 1),
    todate=datetime.datetime(2020, 1, 1),
    nullvalue=0.0,
    dtformat=('%Y-%m-%d'),
    datetime=0,
    time=-1,
    high=2,
    low=3,
    open=1,
    close=4,
    volume=5,
    openinterest=-1
)

# 将数据添加到Cerebro
cerebro.adddata(data)

# 设置初始资本
cerebro.broker.setcash(100000.0)

# 设置交易单位大小
cerebro.addsizer(bt.sizers.FixedSize, stake=10)

# 设置佣金为0.1%
cerebro.broker.setcommission(commission=0.001)

# 运行回测
cerebro.run()

# 绘制结果
cerebro.plot()