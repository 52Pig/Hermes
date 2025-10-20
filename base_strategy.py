import json
import os
import configparser
import pandas as pd
from datetime import datetime


class BaseStrategy:
    def __init__(self, config):
        self.config = config
        self.holdings = {}  # 策略持仓记录 {stock: volume}
        self.trade_records = []  # 交易记录列表（内存中）
        self.holding_file = None  # 持仓记录文件路径
        self.trade_record_file = None  # Excel交易记录文件路径
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
        self.trade_record_file = os.path.join(base_dir, f"{strategy_name}_trade_records.xlsx")

        print(f"Holdings file: {self.holding_file}")
        print(f"Trade records file (Excel): {self.trade_record_file}")

        # 创建Excel文件如果不存在
        self.create_excel_trade_file()

    def create_excel_trade_file(self):
        """创建Excel交易记录文件"""
        if not os.path.exists(self.trade_record_file):
            # 创建包含列标题的DataFrame
            df = pd.DataFrame(columns=[
                '时间', '操作', '股票代码', '股票名称', '价格',
                '数量', '订单ID', '原因ID', '账户', '策略'
            ])
            # 保存到Excel
            df.to_excel(self.trade_record_file, index=False)
            print(f"Created Excel trade file: {self.trade_record_file}")

    def set_holding_file(self, file_path):
        """设置持仓记录文件路径"""
        self.holding_file = file_path

    def set_trade_record_file(self, file_path):
        """设置交易记录文件路径"""
        self.trade_record_file = file_path

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
        print(f"LOAD holding file finish! holding={self.holdings}")
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
        """从Excel文件加载交易记录到内存"""
        if self.trade_record_file and os.path.exists(self.trade_record_file):
            try:
                df = pd.read_excel(self.trade_record_file)
                # 将DataFrame转换为交易记录列表
                self.trade_records = df.to_dict('records')
                standardized_records = []
                # 将列名从中文映射回英文字段名
                for record in self.trade_records:
                    standardized_record = {
                        'timestamp': record.get('时间', record.get('timestamp', '')),
                        'action': record.get('操作', record.get('action', '')),
                        'stock_code': record.get('股票代码', record.get('stock_code', '')),
                        'stock_name': record.get('股票名称', record.get('stock_name', '')),
                        'price': record.get('价格', record.get('price', 0)),
                        'volume': record.get('数量', record.get('volume', 0)),
                        'order_id': record.get('订单ID', record.get('order_id', '')),
                        'reason': record.get('原因ID', record.get('reason', '')),
                        'account': record.get('账户', record.get('account', '')),
                        'strategy': record.get('策略', record.get('strategy', ''))
                    }
                    standardized_records.append(standardized_record)
                self.trade_records = standardized_records
                print(f"Loaded {len(self.trade_records)} trade records from {self.trade_record_file}")
            except Exception as e:
                print(f"Error loading trade records from Excel: {e}")
                self.trade_records = []
        else:
            print(f"Trade records file not found: {self.trade_record_file}")
            self.trade_records = []
        return self.trade_records

    def save_trade_records(self):
        """将内存中的交易记录保存到Excel文件"""
        if not self.trade_record_file:
            return

        try:
            # 准备Excel数据
            excel_data = []
            for record in self.trade_records:
                excel_record = {
                    '时间': record.get('timestamp', ''),
                    '操作': record.get('action', ''),
                    '股票代码': record.get('stock_code', ''),
                    '股票名称': record.get('stock_name', ''),
                    '价格': record.get('price', 0),
                    '数量': record.get('volume', 0),
                    '订单ID': record.get('order_id', ''),
                    '原因ID': record.get('reason', ''),
                    '账户': record.get('account', ''),
                    '策略': record.get('strategy', '')
                }
                excel_data.append(excel_record)

            # 创建DataFrame并保存
            df = pd.DataFrame(excel_data)
            df.to_excel(self.trade_record_file, index=False)
            print(f"Saved {len(self.trade_records)} trade records to {self.trade_record_file}")

        except Exception as e:
            print(f"Error saving trade records to Excel: {e}")

    def record_trade(self, action, stock_code, stock_name, price, volume, order_id, reason, account):
        """记录交易并更新持仓，保存到Excel"""
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

        # 保存持仓和交易记录
        self.save_holdings()
        self.save_trade_records()

        return trade_record

    def get_holding_volume(self, stock_code):
        """获取指定股票的持仓量"""
        print(self.holdings)
        return self.holdings.get(stock_code, 0)

    def get_holding_days(self, stock_code):
        """计算持仓天数"""
        if stock_code not in self.holdings or self.holdings[stock_code] <= 0:
            return 0

        # 查找该股票的最后一次买入记录
        buy_records = []
        for r in self.trade_records:
            record_stock_code = self.get_record_field(r, 'stock_code')
            record_action = self.get_record_field(r, 'action')

            if record_stock_code == stock_code and record_action == 'buy':
                buy_records.append(r)

        if not buy_records:
            return 0

        # 获取最后一次买入时间
        last_buy = max(buy_records, key=lambda x: self.get_record_field(x, 'timestamp', ''))

        timestamp_str = self.get_record_field(last_buy, 'timestamp')
        if not timestamp_str:
            return 0

        try:
            buy_date = datetime.strptime(timestamp_str, '%Y-%m-%d %H:%M:%S').date()
            current_date = datetime.now().date()
            holding_days = (current_date - buy_date).days
            return holding_days
        except Exception as e:
            print(f"Error parsing timestamp {timestamp_str}: {e}")
            return 0

    def get_all_holdings(self):
        """获取所有持仓"""
        return self.holdings.copy()

    def get_record_field(self, record, field_name, default_value=None):
        """
        安全获取记录字段，兼容中英文字段名
        """
        # 英文字段名优先
        if field_name in record:
            return record[field_name]

        # 中文字段名映射
        chinese_field_map = {
            'timestamp': '时间',
            'action': '操作',
            'stock_code': '股票代码',
            'stock_name': '股票名称',
            'price': '价格',
            'volume': '数量',
            'order_id': '订单ID',
            'reason': '原因ID',
            'account': '账户',
            'strategy': '策略'
        }

        chinese_field = chinese_field_map.get(field_name)
        if chinese_field and chinese_field in record:
            return record[chinese_field]

        return default_value

    def is_today_bought(self, stock_code):
        """检查是否为当天买入的股票"""
        if stock_code not in self.holdings or self.holdings[stock_code] <= 0:
            return False

        # 查找该股票当天的买入记录
        current_date = datetime.now().date()
        today_buy_records = [
            r for r in self.trade_records
            if (r['stock_code'] == stock_code and
                r['action'] == 'buy' and
                datetime.strptime(r['timestamp'], '%Y-%m-%d %H:%M:%S').date() == current_date)
        ]

        return len(today_buy_records) > 0

    def get_trade_calendar_holding_days(self, stock_code):
        """使用交易日历计算持仓天数（考虑T+1）"""
        if stock_code not in self.holdings or self.holdings[stock_code] <= 0:
            return 0

        # 查找该股票的最后一次买入记录
        buy_records = [r for r in self.trade_records
                       if r['stock_code'] == stock_code and r['action'] == 'buy']

        if not buy_records:
            return 0

        # 获取最后一次买入时间
        last_buy = max(buy_records, key=lambda x: x['timestamp'])
        buy_date_str = last_buy['timestamp'].split(' ')[0]  # 只取日期部分

        # 使用交易日历计算（这里需要接入实际的交易日历）
        try:
            # 假设有一个交易日历模块
            import a_trade_calendar
            current_date = datetime.now().strftime("%Y-%m-%d")
            trade_days = a_trade_calendar.get_trade_days_interval(buy_date_str, current_date)
            holding_days = trade_days - 1  # 减去买入当天
            return max(0, holding_days)
        except:
            # 如果交易日历不可用，回退到自然日计算
            buy_date = datetime.strptime(buy_date_str, '%Y-%m-%d').date()
            current_date = datetime.now().date()
            holding_days = (current_date - buy_date).days
            return holding_days