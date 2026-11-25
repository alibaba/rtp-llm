from __future__ import annotations

import logging
from typing import TYPE_CHECKING, Callable, Optional

if TYPE_CHECKING:
    from rtp_llm.config.engine_config import EngineConfig
    from rtp_llm.models.base_model import BaseModel
    from rtp_llm.models.propose_model.propose_model import ProposeModel
    from rtp_llm.multimodal.mm_process_engine import MMProcessEngine


class RtpLLMOp:
    def __init__(
        self,
        engine_config: "EngineConfig",
        model: "BaseModel",
        propose_model: Optional["ProposeModel"] = None,
        mm_process_engine: Optional["MMProcessEngine"] = None,
    ):
        self.engine_config = engine_config
        self.model = model
        self.propose_model = propose_model
        from rtp_llm.ops import ensure_engine_ops_loaded

        ensure_engine_ops_loaded()
        from rtp_llm.ops import RtpLLMOp as CppRtpLLMOp

        self.ft_op = CppRtpLLMOp()
        self.mm_process_engine = mm_process_engine

    def start(self):
        self.weight = self.model.weight
        # Resident weights restart: size the KV cache as the cold start did (see KVCacheConfig).
        draft = self.propose_model.model if self.propose_model is not None else None
        kv_cache_config = self.engine_config.kv_cache_config
        kv_cache_config.target_initialize_gpu_growth_bytes = (
            self.model.resident_initialize_gpu_growth_bytes
        )
        kv_cache_config.draft_initialize_gpu_growth_bytes = (
            0 if draft is None else draft.resident_initialize_gpu_growth_bytes
        )
        logging.info("engine_config: %s", self.engine_config.to_string())
        from rtp_llm.telemetry.config import load_trace_config

        trace_config = load_trace_config(
            "backend", self.engine_config.parallelism_config.tp_rank
        )
        self.ft_op.init(  # type: ignore
            self.model,
            self.engine_config,
            self.model.vit_config,
            self.propose_model,
            self.mm_process_engine,
            trace_config,
        )

    def set_serving(self, serving: bool) -> None:
        self.ft_op.set_serving(serving)

    def stop(self, before_rpc_shutdown: Optional[Callable[[int], None]] = None):
        if before_rpc_shutdown is None:
            self.ft_op.stop()  # type: ignore
        else:
            self.ft_op.stop(before_rpc_shutdown)  # type: ignore
