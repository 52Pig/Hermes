import logging
import os
from logging.handlers import TimedRotatingFileHandler
from configparser import ConfigParser

def setup_logging(config_file='conf/config.ini'):
    config = ConfigParser(interpolation=None)  # 禁用插值功能

    with open(config_file, 'r', encoding='utf-8') as f:
        config.read_file(f)

    # 从配置文件中读取日志相关配置
    log_level = config.get('logging', 'log_level', fallback='INFO').upper()
    log_format = config.get('logging', 'log_format', fallback='%(asctime)s - %(levelname)s - %(message)s')
    log_file = config.get('logging', 'log_file', fallback='logs/app.log')
    log_backup_count = config.getint('logging', 'log_backup_count', fallback=5)  # 保留日志的天数

    # 确保日志目录存在
    os.makedirs(os.path.dirname(log_file), exist_ok=True)

    # 创建按天分割的处理器，设置日志文件名后缀为日期格式
    handler = TimedRotatingFileHandler(log_file, when='midnight', interval=1, backupCount=log_backup_count)
    handler.suffix = '%Y-%m-%d'  # 文件名后缀，格式为 'YYYY-MM-DD'
    handler.extMatch = r'^\d{4}-\d{2}-\d{2}$'  # 匹配日期格式文件名

    formatter = logging.Formatter(log_format)
    handler.setFormatter(formatter)

    # 配置日志
    logging.basicConfig(
        level=log_level,
        format=log_format,
        handlers=[
            handler,                     # 文件输出，按天分割
            logging.StreamHandler()      # 控制台输出
        ]
    )

    return logging.getLogger(__name__)