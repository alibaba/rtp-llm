import logging
import queue
import atexit
from typing import Optional
from logging.handlers import QueueHandler, QueueListener

from concurrent_log_handler import ConcurrentRotatingFileHandler

from rtp_llm.config.py_config_modules import StaticConfig

LOG_PATH_KEY = "LOG_PATH"

# 进程内队列监听器
_listeners = {}

def get_async_handler(file_name: str) -> Optional[logging.Handler]:
    """创建异步日志handler，每个进程一个队列避免flush"""
    log_path = StaticConfig.profiling_debug_config.log_path
    if log_path == "":
        return None

    # 如果已存在，返回现有的QueueHandler
    if file_name in _listeners:
        return QueueHandler(_listeners[file_name]["queue"])

    # 创建队列和文件handler
    log_queue = queue.Queue()
    file_handler = ConcurrentRotatingFileHandler(
        filename=f"{log_path}/{file_name}",
        mode="a",
        maxBytes=100 * 1024 * 1024,
        backupCount=StaticConfig.profiling_debug_config.log_file_backup_count,
        use_gzip=True,
    )

    # 创建并启动监听器
    listener = QueueListener(log_queue, file_handler)
    listener.start()

    # 保存引用
    _listeners[file_name] = {
        "queue": log_queue,
        "listener": listener
    }

    # 注册清理函数
    atexit.register(_cleanup_listeners)

    return QueueHandler(log_queue)


def _cleanup_listeners():
    """清理监听器"""
    for data in _listeners.values():
        try:
            data["listener"].stop()
        except:
            pass
