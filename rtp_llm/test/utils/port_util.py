import fcntl
import json
import logging
import os
import random
import socket
import tempfile
import time
from contextlib import closing
from pathlib import Path


def get_linux_random_port_range():
    """
    获取 Linux 系统的随机端口（临时端口）范围。
    该信息存储在 /proc/sys/net/ipv4/ip_local_port_range 文件中。

    Returns:
        tuple: (min_port, max_port) 如果成功获取；
               (None, None) 如果文件不存在、权限不足或内容格式不正确。
    """
    file_path = "/proc/sys/net/ipv4/ip_local_port_range"

    if not os.path.exists(file_path):
        logging.warning(
            f"错误: 文件 '{file_path}' 不存在。请确认您在 Linux 系统上运行此脚本。"
        )
        return None, None

    try:
        with open(file_path, "r") as f:
            content = f.read().strip()
            parts = content.split()
            if len(parts) == 2:
                min_port = int(parts[0])
                max_port = int(parts[1])
                return min_port, max_port
            else:
                logging.warning(
                    f"错误: 文件 '{file_path}' 内容格式不正确: '{content}'。预期格式为 'min_port max_port'。"
                )
                return None, None
    except PermissionError:
        logging.warning(
            f"错误: 没有权限读取文件 '{file_path}'。请尝试以 root 权限运行或检查文件权限。"
        )
        return None, None
    except ValueError:
        logging.warning(
            f"错误: 无法将文件 '{file_path}' 中的内容 '{content}' 解析为整数端口范围。"
        )
        return None, None
    except Exception as e:
        logging.warning(f"发生未知错误: {e}")
        return None, None


class PortInUseError(Exception):
    pass


class ExpiredLockFile:
    """lock file with expiration info"""

    def __init__(
        self, path: Path, port: int, ttl: int = 3600
    ):  # 1 hour ttl by default, make it long enough to guarantee the lock is valid during whole lifecycle of the test
        self.path = path
        self.port = port
        self.ttl = ttl
        self.fd = None

    def __enter__(self):
        # record all necessary info to metadata
        metadata = {
            "port": self.port,
            "pid": os.getpid(),
            "timestamp": time.time(),
            "ttl": self.ttl,
        }

        self.fd = open(self.path, "w")
        try:
            fcntl.flock(self.fd, fcntl.LOCK_EX | fcntl.LOCK_NB)
            json.dump(metadata, self.fd)
            self.fd.flush()
            return self
        except (IOError, OSError):
            self.fd.close()
            raise PortInUseError(f"Port {self.port} is already been locked")

    def __exit__(self, exc_type, exc_val, exc_tb):
        if self.fd:
            fcntl.flock(self.fd, fcntl.LOCK_UN)
            self.fd.close()
            try:
                self.path.unlink()
            except OSError:
                pass


class PortManager:
    def __init__(self, lock_dir: Path = None, start_port: int = None, ttl: int = 3600):
        self.start_port = start_port or 10000
        self.port_range = (self.start_port, get_linux_random_port_range()[0])
        self.ttl = ttl
        self.lock_dir = lock_dir or Path(tempfile.gettempdir()) / "test_port_locks"
        self.lock_dir.mkdir(parents=True, exist_ok=True)
        logging.info(f"port_range: {self.port_range} lock: {self.lock_dir}")

    def cleanup_stale_locks(self):
        """clean up the potential stale lock files"""
        current_time = time.time()
        for lock_file in self.lock_dir.glob("port_*.lock"):
            try:
                with open(lock_file) as f:
                    try:
                        # try to lock the lock-file
                        fcntl.flock(f.fileno(), fcntl.LOCK_EX | fcntl.LOCK_NB)
                        try:
                            metadata = json.load(f)
                            if current_time - metadata["timestamp"] > metadata["ttl"]:
                                lock_file.unlink()
                        except (json.JSONDecodeError, KeyError):
                            # remove the lock-file directly if json format error
                            lock_file.unlink()
                        finally:
                            fcntl.flock(f.fileno(), fcntl.LOCK_UN)
                    except (IOError, OSError):
                        # that's ok, the file is being used now
                        continue
            except OSError:
                # fine, skip this one
                continue

    def get_consecutive_ports(self, num_ports: int = 1):
        """get real consecutive ports by checking the file locks"""
        # always clean the stale lock files before getting the real available ports
        # try best to reuse all the available ports
        self.cleanup_stale_locks()

        for base_port in range(self.port_range[0], self.port_range[1]):
            locks = []
            try:
                ports = list(range(base_port, base_port + num_ports))

                # try to validate all the pre-claimed ports
                for port in ports:
                    lock_file = self.lock_dir / f"port_{port}.lock"
                    lock = ExpiredLockFile(lock_file, port, self.ttl)
                    lock.__enter__()
                    locks.append(lock)

                    # make sure the port is really available from the OS perspective
                    if not self.is_port_available(port):
                        raise PortInUseError(f"Port {port} is in use")

                logging.info(f"alloc ports: {base_port}-{base_port + num_ports}")
                return ports, locks

            except PortInUseError:
                # release all the pre-claimed locks
                for lock in locks:
                    lock.__exit__(None, None, None)
                continue

        raise RuntimeError("Unable to find consecutive free ports")

    @staticmethod
    def is_port_available(port: int) -> bool:
        with closing(socket.socket(socket.AF_INET, socket.SOCK_STREAM)) as sock:
            try:
                sock.bind(("", port))
                return True
            except socket.error:
                return False


class PortsContext:
    """
    PortsContext wrap PortManager to get consecutive ports and be friendly with `with` statement.
    It is friendly for lock-files auto releasing after the scope exits -- this can make sure ports reuse efficiently.
    Use PortManager directly if you want hold the allocated ports for a long time.
    """

    def __init__(self, lock_dir: Path = None, num_ports: int = 1, ttl: int = 3600):
        lock_dir = lock_dir or Path(tempfile.gettempdir()) / "test_port_locks"
        self.manager = PortManager(lock_dir, get_random_start_port(), ttl)
        self.num_ports = num_ports
        self.ports = None
        self.locks = None

    def __enter__(self):
        self.ports, self.locks = self.manager.get_consecutive_ports(self.num_ports)
        return self.ports

    def __exit__(self, exc_type, exc_val, exc_tb):
        if self.locks:
            for lock in self.locks:
                lock.__exit__(None, None, None)


def get_random_start_port():
    random.seed()
    return random.randint(12000, 20000)
