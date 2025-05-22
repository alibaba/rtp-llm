import socket
import random

from contextlib import closing

def get_consecutive_free_ports(num_port = 1):
    random.seed()
    start_port = random.randint(12000, 20000) # 不要用随机端口，容易冲突
    for base_port in range(start_port, 65536 - num_port + 1):
        if all(is_port_free(port) for port in range(base_port, base_port + num_port)):
            return list(range(base_port, base_port + num_port))
    raise ValueError("Cannot find enough consecutive free ports")

def is_port_free(port):
    with closing(socket.socket(socket.AF_INET, socket.SOCK_STREAM)) as sock:
        try:
            sock.bind(('', port))
            return True
        except socket.error:
            return False

