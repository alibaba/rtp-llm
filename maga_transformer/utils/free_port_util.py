import socket
import random
from contextlib import closing

class FreePortUtil(object):
    @staticmethod
    def _is_port_free(port: int):
        with closing(socket.socket(socket.AF_INET, socket.SOCK_STREAM)) as sock:
            try:
                sock.bind(('', port))
                return True
            except socket.error:
                return False
    
    @staticmethod
    def get_consecutive_free_ports(num_port: int):
        start_port = random.randint(12000, 20000)
        for base_port in range(start_port, 65536 - num_port + 1):
            if all(FreePortUtil._is_port_free(port) for port in range(base_port, base_port + num_port)):
                return list(range(base_port, base_port + num_port))
        raise ValueError("Cannot find enough consecutive free ports")