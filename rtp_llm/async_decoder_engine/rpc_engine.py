import logging
import time
from typing import Dict, Optional

from typing_extensions import override

from rtp_llm.async_decoder_engine.base_engine import BaseEngine
from rtp_llm.config.engine_config import EngineConfig
from rtp_llm.models.base_model import BaseModel
from rtp_llm.models.propose_model.propose_model import ProposeModel
from rtp_llm.multimodal.mm_process_engine import MMProcessEngine
from rtp_llm.multimodal.multimodal_mixin_factory import MultimodalMixinFactory
from rtp_llm.ops import RoleType, VitSeparation
from rtp_llm.ops.rtp_llm.rtp_llm_op import RtpLLMOp
from rtp_llm.utils.time_util import timer_wrapper


class LanguageCppEngine(BaseEngine):
    def __init__(
        self,
        model: BaseModel,
        engine_config: EngineConfig,
        propose_model: Optional[ProposeModel] = None,
    ) -> None:
        """Initialize RPCEngine with model and engine configuration.

        Args:
            model: BaseModel instance
            engine_config: EngineConfig instance containing engine and parallelism configs
            propose_model: Optional propose model for speculative decoding
        """
        self.model = model
        self.propose_model = propose_model
        # BaseModel no longer has config attribute, use model_config instead
        self.config = model.model_config
        self.mm_process_engine = None
        # Gate on the leading stage too: only it admits requests and runs the multimodal processor.
        is_pp_stage_root = (
            engine_config.parallelism_config.pp_size <= 1
            or engine_config.parallelism_config.pp_rank == 0
        )
        if (
            self.model.is_multimodal()
            and self.model.vit_config.vit_separation
            == VitSeparation.VIT_SEPARATION_LOCAL
            and engine_config.parallelism_config.tp_rank == 0
            and is_pp_stage_root
            and engine_config.pd_sep_config.role_type
            in (RoleType.PREFILL, RoleType.DECODE, RoleType.PDFUSION)
        ):
            self.mm_process_engine = (
                MultimodalMixinFactory.create_multimodal_process_engine(
                    model_config=self.model.model_config,
                    engine_config=engine_config,
                    vit_config=self.model.vit_config,
                    device=f"cuda:{engine_config.parallelism_config.local_rank}",
                )
            )
        self.rtp_llm_op_ = RtpLLMOp(
            engine_config,
            model,
            propose_model,
            self.mm_process_engine,
        )
        self._service_started = False

    @timer_wrapper(description="start async engine")
    @override
    def _start(self) -> None:
        start_time = time.time()
        # Phase 1: build engine state.  Listeners start in start_service(),
        # which every deployment calls right after its startup barrier.
        self.rtp_llm_op_.start()
        consume_s = time.time() - start_time
        logging.info(f"start rtp_llm_op_ took {consume_s:.2f}s")

    def gpu_cache_tensors(self):
        return self.rtp_llm_op_.gpu_cache_tensors()

    def start_service(self) -> None:
        """Phase 2: bind the RPC listener after the startup barrier."""
        if self._service_started:
            return
        if self.mm_process_engine is not None:
            # KVCM client construction can start discovery and resolve live endpoints.
            self.mm_process_engine.start_remote_embedding_cache()
        self.rtp_llm_op_.start_service()
        self._service_started = True

    def update_runtime_endpoints(self, runtime_config, world_info) -> None:
        """Apply restore-time endpoint discovery before deferred listeners start."""
        self.world_info = world_info
        self.rtp_llm_op_.update_runtime_endpoints(runtime_config)

    def set_serving(self, serving: bool) -> None:
        self.rtp_llm_op_.set_serving(serving)

    @override
    def _stop(self) -> None:
        # Stop the C++ request entry first so no more embedding calls are issued,
        # then close the local MMProcessEngine (stops its MMScheduler thread and
        # preprocessing executor).
        self.rtp_llm_op_.stop()
        if self.mm_process_engine is not None:
            self.mm_process_engine.stop()
            self.mm_process_engine = None
