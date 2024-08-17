
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

    def reset(self):
        self.current_step = 0
        self.account_balance = 100000  # Initial account balance
        self.shares_held = 0
        self.net_worth = self.account_balance
        self.max_net_worth = self.account_balance

        return self._next_observation()

    def _next_observation(self):
        return self.data.iloc[self.current_step].values

    def step(self, action):
        self._take_action(action)
        self.current_step += 1

        if self.current_step > len(self.data) - 1:
            self.current_step = 0

        return self._next_observation(), self._get_reward(), self.net_worth, {}

    def _take_action(self, action):
        if action == 0:  # Buy
            self.shares_held += self.account_balance / self.data.iloc[self.current_step].values[0]
            self.account_balance -= self.account_balance
        elif action == 1:  # Sell
            self.account_balance += self.shares_held * self.data.iloc[self.current_step].values[0]
            self.shares_held -= self.shares_held

        self.net_worth = self.account_balance + self.shares_held * self.data.iloc[self.current_step].values[0]

        if self.net_worth > self.max_net_worth:
            self.max_net_worth = self.net_worth

    def _get_reward(self):
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
    # Open       High        Low      Close  Adj Close     Volume
    # Date
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