import json
import os
import configparser
import pandas as pd
from datetime import datetime


class BaseStrategy:
    def __init__(self, config):
        self.config = config
        self.holdings = {}  # 策略持仓记录 {stock: volume}
        self.trade_records = []  # 交易记录列表
        self.holding_file = None  # 持仓记录文件路径
        self.trade_record_file = None  # 交易记录文件路径
        self.excel_trade_file = None  # Excel交易记录文件路径
        self.set_default_file_paths()  # 设置默认文件路径
        self.load_holdings()  # 加载持仓记录
        self.load_trade_records()  # 加载交易记录
        print(f"Initializing strategy with parameters: {self.config}")

    def do(self, accounts):
        print(f"do...")

    def update_config(self, new_config):
        """更新配置"""
        self.config = new_config

    def get_base_dir_from_config(self):
        """从配置文件获取基础目录路径"""
        try:
            # 读取配置文件
            config = configparser.ConfigParser()
            config.read('conf/config.ini')

            # 从配置文件中获取 base_dir
            base_dir = config.get('strategy', 'base_dir', fallback=None)

            if base_dir is None:
                # 如果没有配置，使用默认路径
                base_dir = "D:/tool/dataset/strategy_data"
                print(f"Using default base_dir: {base_dir}")
            else:
                print(f"Using configured base_dir: {base_dir}")

            # 确保目录存在
            if not os.path.exists(base_dir):
                os.makedirs(base_dir, exist_ok=True)
                print(f"Created directory: {base_dir}")

            return base_dir

        except Exception as e:
            print(f"Error reading base_dir from config: {e}")
            # 出错时使用默认路径
            base_dir = "D:/tool/dataset/strategy_data"
            if not os.path.exists(base_dir):
                os.makedirs(base_dir, exist_ok=True)
            return base_dir

    def set_default_file_paths(self):
        """设置默认文件路径"""
        # 从配置文件获取基础目录
        base_dir = self.get_base_dir_from_config()

        # 使用策略类名作为文件名
        strategy_name = self.__class__.__name__

        # 设置默认文件路径
        self.holding_file = os.path.join(base_dir, f"{strategy_name}_holdings.json")
        self.trade_record_file = os.path.join(base_dir, f"{strategy_name}_trade_records.json")
        self.excel_trade_file = os.path.join(base_dir, f"{strategy_name}_trade_records.xlsx")

        print(f"Holdings file: {self.holding_file}")
        print(f"Trade records file: {self.trade_record_file}")
        print(f"Excel trade file: {self.excel_trade_file}")

        # 创建Excel文件如果不存在
        self.create_excel_trade_file()

    def create_excel_trade_file(self):
        """创建Excel交易记录文件"""
        if not os.path.exists(self.excel_trade_file):
            # 创建包含列标题的DataFrame
            df = pd.DataFrame(columns=[
                '时间', '操作', '股票代码', '股票名称', '价格',
                '数量', '订单ID', '原因ID', '账户', '策略'
            ])
            # 保存到Excel
            df.to_excel(self.excel_trade_file, index=False)
            print(f"Created Excel trade file: {self.excel_trade_file}")

    def set_holding_file(self, file_path):
        """设置持仓记录文件路径"""
        self.holding_file = file_path

    def set_trade_record_file(self, file_path):
        """设置交易记录文件路径"""
        self.trade_record_file = file_path

    def set_excel_trade_file(self, file_path):
        """设置Excel交易记录文件路径"""
        self.excel_trade_file = file_path

    def load_holdings(self):
        """加载持仓记录"""
        if self.holding_file and os.path.exists(self.holding_file):
            try:
                with open(self.holding_file, 'r', encoding='utf-8') as f:
                    self.holdings = json.load(f)
                print(f"Loaded holdings from {self.holding_file}")
            except Exception as e:
                print(f"Error loading holdings: {e}")
                self.holdings = {}
        else:
            print(f"Holdings file not found: {self.holding_file}")
            self.holdings = {}
        return self.holdings

    def save_holdings(self):
        """保存持仓记录"""
        if self.holding_file:
            try:
                with open(self.holding_file, 'w', encoding='utf-8') as f:
                    json.dump(self.holdings, f, ensure_ascii=False, indent=2)
                print(f"Saved holdings to {self.holding_file}")
            except Exception as e:
                print(f"Error saving holdings: {e}")

    def load_trade_records(self):
        """加载交易记录"""
        if self.trade_record_file and os.path.exists(self.trade_record_file):
            try:
                with open(self.trade_record_file, 'r', encoding='utf-8') as f:
                    print(f"#4 trade_file={self.trade_record_file}")
                    self.trade_records = json.load(f)
                print(f"Loaded trade records from {self.trade_record_file}")
            except Exception as e:
                print(f"Error loading trade records: {e}")
                self.trade_records = []
        else:
            print(f"Trade records file not found: {self.trade_record_file}")
            self.trade_records = []
        return self.trade_records

    def save_trade_records(self):
        """保存交易记录"""
        if self.trade_record_file:
            try:
                with open(self.trade_record_file, 'w', encoding='utf-8') as f:
                    json.dump(self.trade_records, f, ensure_ascii=False, indent=2)
                print(f"Saved trade records to {self.trade_record_file}")
            except Exception as e:
                print(f"Error saving trade records: {e}")

    def record_trade(self, action, stock_code, stock_name, price, volume, order_id, reason, account):
        """记录交易并更新持仓，同时保存到Excel"""
        # 创建交易记录
        trade_time = datetime.now().strftime('%Y-%m-%d %H:%M:%S')
        trade_record = {
            'timestamp': trade_time,
            'action': action,
            'stock_code': stock_code,
            'stock_name': stock_name,
            'price': price,
            'volume': volume,
            'order_id': order_id,
            'reason': reason,
            'account': account,
            'strategy': self.__class__.__name__
        }

        # 添加到交易记录
        self.trade_records.append(trade_record)
        self.save_trade_records()

        # 更新持仓记录
        if action == 'buy':
            if stock_code not in self.holdings:
                self.holdings[stock_code] = 0
            self.holdings[stock_code] += volume
        elif action == 'sell':
            if stock_code in self.holdings:
                self.holdings[stock_code] -= volume
                if self.holdings[stock_code] <= 0:
                    del self.holdings[stock_code]

        self.save_holdings()

        # 保存到Excel
        self.record_trade_to_excel(trade_record)

        return trade_record

    def record_trade_to_excel(self, trade_record):
        """将交易记录保存到Excel文件"""
        try:
            # 读取现有数据
            if os.path.exists(self.excel_trade_file) and os.path.getsize(self.excel_trade_file) > 0:
                df = pd.read_excel(self.excel_trade_file)
            else:
                df = pd.DataFrame(columns=[
                    '时间', '操作', '股票代码', '股票名称', '价格',
                    '数量', '订单ID', '原因ID', '账户', '策略'
                ])

            # 准备Excel记录数据
            excel_record = {
                '时间': trade_record['timestamp'],
                '操作': trade_record['action'],
                '股票代码': trade_record['stock_code'],
                '股票名称': trade_record['stock_name'],
                '价格': trade_record['price'],
                '数量': trade_record['volume'],
                '订单ID': trade_record['order_id'],
                '原因ID': trade_record['reason'],
                '账户': trade_record['account'],
                '策略': trade_record['strategy']
            }

            # 添加新记录
            new_row = pd.DataFrame([excel_record])
            df = pd.concat([df, new_row], ignore_index=True)

            # 保存回Excel
            df.to_excel(self.excel_trade_file, index=False)
            print(f"Trade record saved to Excel: {self.excel_trade_file}")

        except Exception as e:
            print(f"Error saving trade record to Excel: {e}")

    def get_holding_volume(self, stock_code):
        """获取指定股票的持仓量"""
        return self.holdings.get(stock_code, 0)

    def get_holding_days(self, stock_code):
        """计算持仓天数"""
        if stock_code not in self.holdings or self.holdings[stock_code] <= 0:
            return 0
        print(f"#1 holding:{stock_code},{self.trade_records}")

        # 查找该股票的最后一次买入记录
        buy_records = [r for r in self.trade_records
                       if r['stock_code'] == stock_code and r['action'] == 'buy']

        if not buy_records:
            return 0
        print(f"#2 holding:{stock_code},{buy_records}")
        # 获取最后一次买入时间
        last_buy = max(buy_records, key=lambda x: x['timestamp'])
        buy_date = datetime.strptime(last_buy['timestamp'], '%Y-%m-%d %H:%M:%S').date()
        print(f"#3 holding:{last_buy},{buy_date}")

        # 计算持仓天数
        current_date = datetime.now().date()
        holding_days = (current_date - buy_date).days

        return holding_days

    def get_all_holdings(self):
        """获取所有持仓"""
        return self.holdings.copy()