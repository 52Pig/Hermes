
import os
import gym
import numpy as np
import pandas as pd
import yfinance as yf
import matplotlib.pyplot as plt

from sklearn.preprocessing import MinMaxScaler
from stable_baselines3 import PPO
from stable_baselines3.common.monitor import Monitor
from stable_baselines3.common.vec_env import DummyVecEnv
from stable_baselines3.common.callbacks import BaseCallback
from stable_baselines3.common.results_plotter import load_results, ts2xy

# Define the trading environment
class TradingEnvironment(gym.Env):
    def __init__(self, data):
        self.data = data
        self.action_space = gym.spaces.Discrete(3)  # Buy, Sell, Hold
        # self.action_space = gymnasium.spaces.Discrete(3)  # Buy, Sell, Hold
        self.observation_space = gym.spaces.Box(low=0, high=1, shape=(len(data.columns),))
        # self.observation_space = gymnasium.spaces.Box(low=0, high=1, shape=(len(data.columns),))
        self.reset()
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
        # 前一次的净资产
        self.prev_net_worth = self.account_balance
        # 回合是否结束
        self.episode_over = False
        # 历史最高净资产，初始账户为账户余额
        self.max_net_worth = self.account_balance
        # 返回当前时间步的观察数据，这里是当前步的股票数据。shape=(len(data.columns),)
        return self._next_observation()

    def _next_observation(self):
        ''' 获取下一步的状态数据
           返回当前步的数据，即一个包含data.columns的数组，shape=(len(data.columns),)
         '''
        return self.data.iloc[self.current_step].values

    def step(self, action):
        '''
          执行动作
        :param action: 具体动作，0买入，1卖出，2持有
        :return:
          self._next_observation()    下一个时间步的状态数据，shape=(len(data.columns),)
          self._get_reward()      当前的奖励，净资产-账户余额
          self.net_worth     当前净资产
          额外信息空
        '''
        ## 执行动作更新账户余额和股票持有量
        self._take_action(action)
        self.current_step += 1

        if self.current_step > len(self.data) - 1:
            self.current_step = 0

        reward = self._get_reward()
        done = self.episode_over
        # 记录时间步和奖励值
        self.log_step_reward(self.current_step, reward)

        return self._next_observation(), reward, done, {}

    def log_step_reward(self, time_step, reward):
        # 将时间步和奖励值写入日志文件
        with open('log/train_step_reward.txt', "a") as f:
            f.write(f"{time_step},{reward}\n")

    def _take_action(self, action):
        '''
        根据智能体选择的动作（action）来更新当前账户的状态，包括账户余额、持有的股票数量和净资产
        :param action:动作 0买入，1卖出，2持有
        :return:
        '''
        # 股票当前价格
        current_price = self.data.iloc[self.current_step].values[0]  # Use the first feature as the price

        if current_price <= 0:
            raise ValueError(f"Invalid stock price encountered: {current_price}")

        if action == 0 and self.account_balance > 0:  # Buy
            # 买入后持有的股票数量 = 当前股票数量 + 当前账户能够买入的股票数量【当前账户余额 / 当前时间步的开盘价(第一个特征是开盘价)】
            self.shares_held += self.account_balance / current_price
            # 更新账户余额
            self.account_balance = 0
        elif action == 1 and self.shares_held > 0:  # Sell
            # 卖出后账户余额 = 当前账户余额 + 卖出所有股票后获得的总金额【卖出股票数量*当前开盘价】
            self.account_balance += self.shares_held * current_price
            # 清空持有的股票数量
            self.shares_held = 0
        # 更新净资产=账户余额+持有股票价值
        self.net_worth = self.account_balance + self.shares_held * current_price
        if not np.isfinite(self.net_worth):
            raise ValueError(f"Non-finite net worth encountered: {self.net_worth}")

        # 更新历史最高净资产
        if self.net_worth > self.max_net_worth:
            self.max_net_worth = self.net_worth

    def _get_reward(self):
        ''' 计算当前时间步奖励，等于净资产减去现金余额，这实际上等于持有股票的市值
        '''
        # 1. Reward for increasing net worth
        # 净资产增长奖励：计算当前时间步相对于前一时间步的净资产变化。正值意味着智能体的净资产在增长，负值则表示减少。
        net_worth_increase = self.net_worth - self.prev_net_worth

        # 2. Penalty for high volatility in net worth (smooth returns)
        # 波动性惩罚：如果净资产减少，智能体将受到基于减少幅度的惩罚，这有助于减少高波动性和不稳定的决策。
        volatility_penalty = -abs(net_worth_increase) if net_worth_increase < 0 else 0

        # 3. Penalty if net worth drops below the initial account balance
        # 破产惩罚：如果净资产下降到初始账户余额的 80 % 以下，智能体将受到严重的惩罚（-1000），并且回合结束。这种机制可以防止智能体进行过度风险的交易。
        if self.net_worth < 0.8 * self.account_balance:
            self.episode_over = True
            return -100  # Large penalty for going bankrupt or losing significant amount

        # 4. Reward should combine increase in net worth and stability (reduce sharp losses)
        # 综合奖励：最终的奖励是净资产增长和波动性惩罚的组合。这种奖励机制平衡了在增加净资产的同时减少过度的风险。
        reward = net_worth_increase + volatility_penalty

        # Update previous net worth
        self.prev_net_worth = self.net_worth

        return reward

def simulate_trading_strategy(model, data):
    """模拟"""
    env = TradingEnvironment(data)
    obs = env.reset()
    pred_list = list()
    for i in range(len(data)):
        action, _ = model.predict(obs)
        obs, _, done, _ = env.step(action)
        # print(obs, env.net_worth)
        pred_list.append(env.net_worth)
        if done:
            break

    return pred_list


class SaveOnBestTrainingRewardCallback(BaseCallback):
    """
    Callback for saving a model (the check is done every ``check_freq`` steps)
    based on the training reward (in practice, we recommend using ``EvalCallback``).

    :param check_freq: (int)
    :param log_dir: (str) Path to the folder where the model will be saved.
      It must contains the file created by the ``Monitor`` wrapper.
    :param verbose: (int)
    """

    def __init__(self, check_freq, save_model_dir, verbose=1):
        super(SaveOnBestTrainingRewardCallback, self).__init__(verbose)
        self.check_freq = check_freq
        self.save_path = os.path.join(save_model_dir, 'best_model/')
        self.best_mean_reward = -np.inf
        self.log_dir = save_model_dir

    # def _init_callback(self) -> None:
    def _init_callback(self):
        # Create folder if needed
        if self.save_path is not None:
            os.makedirs(self.save_path, exist_ok=True)

    # def _on_step(self) -> bool:
    def _on_step(self):
        if self.n_calls % self.check_freq == 0:
            print('self.n_calls: ',self.n_calls)
            model_path1 = os.path.join(self.save_path, 'model_{}'.format(self.n_calls))
            self.model.save(model_path1)

        return True


if __name__ == "__main__":
    # 定义股票代码
    ticker = "AAPL"
    # 下载历史价格数据
    # data = yf.download(ticker, start="2018-01-01", end="2024-08-15", proxy="http://89.177.16.134:9098")

    # data = yf.download(ticker, start="2024-01-01", end="2024-08-15")
    data = yf.download(ticker, start="2024-08-01", end="2024-08-15")

    print(type(data), data.head())
    quit()
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
    # 去除第一列中有0的行
    data = data[data['Open'] != 0]

    # Normalize the data
    scaler = MinMaxScaler(feature_range=(0.01, 1))
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
    env = DummyVecEnv([lambda: env])
    # Initialize the PPO model
    # model = PPO("MlpPolicy", env, verbose=1)
    model = PPO("MlpPolicy", env, verbose=2)

    save_model_dir = r'./log/model_dir/'
    callback1 = SaveOnBestTrainingRewardCallback(1000, save_model_dir)

    model.learn(total_timesteps=5000, callback=callback1)


    # Train the model
    # model.learn(total_timesteps=10000)
    # model.learn(total_timesteps=100000)
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