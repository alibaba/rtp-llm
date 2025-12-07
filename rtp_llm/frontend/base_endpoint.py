import itertools
import json
import logging
from typing import Any, AsyncGenerator, Dict, List, Optional

from rtp_llm.config.generate_config import GenerateConfig
from rtp_llm.config.gpt_init_model_parameters import GptInitModelParameters
from rtp_llm.frontend.tokenizer_factory.tokenizers import BaseTokenizer
from rtp_llm.server.backend_rpc_server_visitor import BackendRPCServerVisitor


class BaseEndpoint(object):
    def __init__(
        self,
        model_config: GptInitModelParameters,
        tokenizer: Optional[BaseTokenizer],
        backend_rpc_server_visitor: BackendRPCServerVisitor,
    ):
        self.model_config = model_config
        self.tokenizer = tokenizer
        self.backend_rpc_server_visitor = backend_rpc_server_visitor

    def handle_request(self, request: Dict[str, Any]) -> Dict[str, Any]:
        pass
