import asyncio
import logging
import socket
import threading
from typing import Any, Dict, List, Optional, Union

from rtp_llm.config.py_config_modules import PyEnvConfigs, StaticConfig
from rtp_llm.model_factory import ModelFactory
from rtp_llm.models.multimodal.mm_process_engine import MMProcessEngine
from rtp_llm.server.vit_rpc_server import MultimodalRpcServer


class VitApp:
    def __init__(self, py_env_configs: PyEnvConfigs = StaticConfig):
        self.py_env_configs = py_env_configs
        self.rpc_server = MultimodalRpcServer(MMProcessEngine(model))
