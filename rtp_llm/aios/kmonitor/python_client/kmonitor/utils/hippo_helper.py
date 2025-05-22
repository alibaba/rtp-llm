import os
import socket
from typing import Dict, Optional
import logging

class HippoHelper:
    HIPPO_FUSE_PORT = 'HIPPO_FUSE_PORT'
    host_ip = os.environ.get('HIPPO_SLAVE_IP', '')
    try:
        # 获取主机名
        hostname = socket.gethostname()
        # 获取主机 IP 地址
        container_ip = socket.gethostbyname(hostname)
        logging.info(f"get container_ip from socket:{container_ip}")
    except Exception as e:
        container_ip = host_ip
        logging.info(f"get container_ip from socket failed, use host_ip:{host_ip} as container_ip")
    role = os.environ.get('HIPPO_ROLE_SHORT_NAME', os.environ.get('HIPPO_ROLE', ''))
    app = os.environ.get('HIPPO_APP', '')
    group = os.environ.get('HIPPO_SERVICE_NAME', '')
    app_workdir = os.environ.get('HIPPO_APP_WORKDIR', '')
    

    @staticmethod
    def is_hippo_env() -> bool:
        return HippoHelper.role != ''

    @staticmethod
    def get_hippo_tags() -> Dict[str, str]:
        return {
            'host_ip': HippoHelper.host_ip,
            'container_ip': HippoHelper.container_ip,
            'hippo_role': HippoHelper.role,
            'hippo_app': HippoHelper.app,
            'hippo_group': HippoHelper.group,
        } if HippoHelper.is_hippo_env() else {}

    @staticmethod
    def host_fuse_port() -> Optional[str]:
        if HippoHelper.HIPPO_FUSE_PORT in os.environ:
            return os.environ.get(HippoHelper.HIPPO_FUSE_PORT)
        else:
            return None
