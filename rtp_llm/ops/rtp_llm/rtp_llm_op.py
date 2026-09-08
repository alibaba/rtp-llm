from __future__ import annotations

import logging
from typing import TYPE_CHECKING, Optional

if TYPE_CHECKING:
    from rtp_llm.config.engine_config import EngineConfig
    from rtp_llm.frontend.token_processor import TokenProcessor
    from rtp_llm.models.base_model import BaseModel
    from rtp_llm.models.propose_model.propose_model import ProposeModel
    from rtp_llm.utils.mm_process_engine import MMProcessEngine


class RtpLLMOp:
    def __init__(
        self,
        engine_config: "EngineConfig",
        model: "BaseModel",
        mm_engine: Optional["MMProcessEngine"] = None,
        propose_model: Optional["ProposeModel"] = None,
        token_processor: Optional["TokenProcessor"] = None,
    ):
        self.engine_config = engine_config
        self.model = model
        self.mm_engine = mm_engine
        self.propose_model = propose_model
        from rtp_llm.ops import ensure_engine_ops_loaded

        ensure_engine_ops_loaded()
        from rtp_llm.ops import RtpLLMOp as CppRtpLLMOp

        self.ft_op = CppRtpLLMOp()
        self.token_processor = token_processor

    def start(self, defer_service_start: bool = False):
        self.weight = self.model.weight
        logging.info("engine_config: %s", self.engine_config.to_string())
        self.ft_op.init(  # type: ignore
            self.model,
            self.engine_config,
            self.model.vit_config,
            self.mm_engine,
            self.propose_model,
            self.token_processor,
            defer_service_start,
        )

    def start_service(self):
        """Start serving sockets after a control-plane pre-service barrier."""
        self.ft_op.start_rpc_server()  # type: ignore

    def update_runtime_endpoints(self, runtime_config):
        """Update deferred RPC/cache peer endpoints after template restore."""
        self.ft_op.update_runtime_endpoints(runtime_config)  # type: ignore

    def stop(self):
        self.ft_op.stop()  # type: ignore
