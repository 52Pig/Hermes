
import gym
import pandas as pd
import yfinance as yf
import matplotlib.pyplot as plt

from sklearn.preprocessing import MinMaxScaler
from stable_baselines3 import PPO
from stable_baselines3.common.vec_env import DummyVecEnv

# Define the trading environment
class TradingEnvironment(gym.Env):
    def __init__(self, data):
        self.data = data
        self.action_space = gym.spaces.Discrete(3)  # Buy, Sell, Hold
        self.observation_space = gym.spaces.Box(low=0, high=1, shape=(len(data.columns),))
        print('[INIT]action_space=', self.action_space)
        print('[INIT]observation_space=', self.observation_space)
        # [INIT]action_space = Discrete(3)
        # [INIT]observation_space = Box(0.0, 1.0, (6,), float32)

    def reset(self):
        # 当前时间步
        self.current_step = 0
        # 账户初始资金
        self.account_balance = 100000  # Initial account balance
        # 当前持有的股票数量
        self.shares_held = 0
        # 当前净资产，初始为账户余额，随着买入卖出，更新为账户余额+持有的股票价值
        self.net_worth = self.account_balance
        # 历史最高净资产，初始账户为账户余额
        self.max_net_worth = self.account_balance
        # 返回当前时间步的观察数据，这里是当前步的股票数据。shape=(len(data.columns),)
        return self._next_observation()

    def _next_observation(self):
        ''' 获取下一步的状态数据
           返回当前步的数据，即一个包含data.columns的数组，shape=(6,)
         '''
        return self.data.iloc[self.current_step].values

    def step(self, action):
        '''
          执行动作
        :param action: 具体动作，0买入，1卖出，2持有
        :return:
          self._next_observation()    下一个时间步的状态数据，shape=(data.columns,)
          self._get_reward()      当前的奖励，净资产-账户余额
          self.net_worth     当前净资产
          额外信息空
        '''
        ## 执行动作更新账户余额和股票持有量
        self._take_action(action)
        self.current_step += 1

        if self.current_step > len(self.data) - 1:
            self.current_step = 0

        return self._next_observation(), self._get_reward(), self.net_worth, {}

    def _take_action(self, action):
        '''
        根据智能体选择的动作（action）来更新当前账户的状态，包括账户余额、持有的股票数量和净资产
        :param action:动作 0买入，1卖出，2持有
        :return:
        '''
        if action == 0:  # Buy
            # 计算买入的股票数量=当前账户余额 / 当前时间步的开盘价(第一个特征是开盘价)
            self.shares_held += self.account_balance / self.data.iloc[self.current_step].values[0]
            # 更新账户余额
            self.account_balance -= self.account_balance
        elif action == 1:  # Sell
            # 计算卖出的股票数量=当前持有的股票数量按照当前开盘价卖出
            self.account_balance += self.shares_held * self.data.iloc[self.current_step].values[0]
            # 清空持有的股票数量
            self.shares_held -= self.shares_held
        # 更新净资产=账户余额+持有股票价值
        self.net_worth = self.account_balance + self.shares_held * self.data.iloc[self.current_step].values[0]
        # 更新历史最高净资产
        if self.net_worth > self.max_net_worth:
            self.max_net_worth = self.net_worth

    def _get_reward(self):
        ''' 计算奖励，用于指导智能体的行为
          奖励=净资产-账户余额
        '''
        return self.net_worth - self.account_balance


def simulate_trading_strategy(model, data):
    """模拟"""
    env = TradingEnvironment(data)
    obs = env.reset()
    pred_list = list()
    for i in range(len(data)):
        action, _ = model.predict(obs)
        obs, _, _, _ = env.step(action)
        # print(obs, env.net_worth)
        pred_list.append(env.net_worth)

    return pred_list


if __name__ == "__main__":
    # 定义股票代码
    ticker = "AAPL"
    # 下载历史价格数据
    data = yf.download(ticker, start="2018-01-01", end="2024-08-15")

    print(data.head())
    #
    # 交易日   开盘价    最高价     最低价   收盘价   调整后收盘价      成交量
    # Date    Open       High        Low      Close  Adj Close     Volume
    # 2018-01-02  42.540001  43.075001  42.314999  43.064999  40.568928  102223600
    # 2018-01-03  43.132500  43.637501  42.990002  43.057499  40.561852  118071600
    # 2018-01-04  43.134998  43.367500  43.020000  43.257500  40.750271   89738400
    # 2018-01-05  43.360001  43.842499  43.262501  43.750000  41.214230   94640000
    # 2018-01-08  43.587502  43.902500  43.482498  43.587502  41.061150   82271200

    # remove missing value from data
    data = data.dropna()

    # Normalize the data
    scaler = MinMaxScaler()
    data_normed = scaler.fit_transform(data)
    # convert data into a pandas DataFrame
    data = pd.DataFrame(data_normed, columns=data.columns)

    #
    train_size = int(len(data) * 0.8)

    train_data = data[:train_size]
    test_data = data[train_size:]
    print(len(data), train_size, test_data)
    # Create the trading environment
    env = TradingEnvironment(train_data)
    # Initialize the PPO model
    # model = PPO("MlpPolicy", env, verbose=1)
    model = PPO("MlpPolicy", env, verbose=2)

    # Train the model
    # model.learn(total_timesteps=10000)
    model.learn(total_timesteps=1)
    #model.save("./ppo_model")


    # Simulate the trading strategy on the testing data
    net_worth_list = simulate_trading_strategy(model, test_data)
    print(net_worth_list)
    # Plot the net worth over time
    plt.plot(net_worth_list)
    plt.xlabel("Time")
    plt.ylabel("Net Worth")
    plt.title("Net Worth over Time")
    plt.show()