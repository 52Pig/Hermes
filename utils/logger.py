import logging
import os
from configparser import ConfigParser


def setup_logging(config_file='conf/config.ini'):
    config = ConfigParser(interpolation=None)  # 禁用插值功能

    with open(config_file, 'r', encoding='utf-8') as f:
        config.read_file(f)

    log_level = config.get('logging', 'log_level', fallback='INFO').upper()
    log_format = config.get('logging', 'log_format', fallback='%(asctime)s - %(levelname)s - %(message)s')
    log_file = config.get('logging', 'log_file', fallback='logs/app.log')

    os.makedirs(os.path.dirname(log_file), exist_ok=True)

    logging.basicConfig(
        level=log_level,
        format=log_format,
        handlers=[
            logging.FileHandler(log_file),
            logging.StreamHandler()  # 控制台输出
        ]
    )

    return logging.getLogger(__name__)