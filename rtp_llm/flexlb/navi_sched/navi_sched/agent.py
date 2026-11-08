import math

from .alog import configureNaviSchedAlog
from .util import (
    DEFAULT_AGENT_RUNTIME_NAME,
    makeRuntimeConfig,
    makeServiceConfig,
    naviSchedModule,
)

_navi_sched = naviSchedModule()


class NaviSchedAgent(_navi_sched.NaviSchedAgent):

    def __init__(
        self,
        biz_name: str,
        server_port: int = 0,
        server_ip: str = "",
        *,
        total_blocks: int,
        block_size: int,
        linear_step: int,
        hybrid_tail_required: bool,
        pd_separated: bool = True,
        execution_geometry_enabled: bool = False,
        projected_execution_geometry_enabled: bool = False,
        execution_cache_source_enabled: bool = False,
        prefill_execution_policy=None,
        peer_cache_enabled: bool = False,
        pp_cache_collector=None,
        mamba_replay_stop_shift: bool = False,
        full_attention_query_drop_tokens: int = 0,
        mamba_retention_interval: int = -1,
        mamba_align: bool = False,
        latency_prediction_mode: str = "split",
        latency_model_fingerprint: str = "",
        latency_history_fingerprint: str = "",
        latency_history_binary_fingerprint: str = "",
        time_compression: float = 1.0,
        runtime_config=None,
        name: str = DEFAULT_AGENT_RUNTIME_NAME,
        **options,
    ):
        configureNaviSchedAlog()
        if server_port < 0 or server_port > 65535:
            raise ValueError("server_port must be between 0 and 65535")
        if total_blocks < 0:
            raise ValueError("total_blocks must be non-negative")
        if block_size <= 0:
            raise ValueError("block_size must be positive")
        if (
            full_attention_query_drop_tokens < 0
            or full_attention_query_drop_tokens % block_size
        ):
            raise ValueError(
                "full_attention_query_drop_tokens must be non-negative and block aligned"
            )
        if linear_step <= 0:
            raise ValueError("linear_step must be positive")
        if latency_prediction_mode not in {"split", "unified"}:
            raise ValueError("latency_prediction_mode must be 'split' or 'unified'")
        if not pd_separated and latency_prediction_mode != "unified":
            raise ValueError(
                "pd_separated=False requires latency_prediction_mode='unified'"
            )
        if projected_execution_geometry_enabled and not execution_geometry_enabled:
            raise ValueError("projected execution geometry requires execution geometry")
        if execution_cache_source_enabled and not execution_geometry_enabled:
            raise ValueError("execution cache source requires execution geometry")
        if execution_geometry_enabled and (
            not pd_separated or latency_prediction_mode != "split"
        ):
            raise ValueError(
                "execution geometry requires PD separated split prediction"
            )
        if execution_geometry_enabled and prefill_execution_policy is None:
            raise ValueError("execution geometry requires a typed execution policy")
        if not math.isfinite(time_compression) or time_compression <= 0:
            raise ValueError("time_compression must be positive and finite")
        navi_sched = naviSchedModule()
        agent_config = makeRuntimeConfig(
            navi_sched,
            navi_sched.NaviSchedAgentConfig,
            biz_name,
            runtime_config,
        )
        agent_config.service = makeServiceConfig(navi_sched, name, options)
        agent_config.pd_separated = bool(pd_separated)
        if pp_cache_collector is not None:
            if not pd_separated or not pp_cache_collector.valid():
                raise ValueError(
                    "PP cache collection requires a complete PP topology and V6D policy"
                )
            agent_config.pp_cache_collector = pp_cache_collector
        agent_config.peer_cache_enabled = bool(peer_cache_enabled)
        agent_config.execution_geometry_enabled = bool(execution_geometry_enabled)
        agent_config.projected_execution_geometry_enabled = bool(
            projected_execution_geometry_enabled
        )
        agent_config.execution_cache_source_enabled = bool(
            execution_cache_source_enabled
        )
        if prefill_execution_policy is not None:
            agent_config.prefill_execution_policy = prefill_execution_policy
        agent_config.mamba_replay_stop_shift = bool(mamba_replay_stop_shift)
        agent_config.full_attention_query_drop_tokens = int(
            full_attention_query_drop_tokens
        )
        agent_config.mamba_retention_interval = int(mamba_retention_interval)
        agent_config.mamba_align = bool(mamba_align)
        agent_config.unified_latency_prediction = latency_prediction_mode == "unified"
        agent_config.time_compression = float(time_compression)
        agent_config.latency_model_fingerprint = str(latency_model_fingerprint)
        agent_config.latency_history_fingerprint = str(latency_history_fingerprint)
        agent_config.latency_history_binary_fingerprint = str(
            latency_history_binary_fingerprint
        )
        super().__init__(
            agent_config, int(server_port), str(server_ip), int(total_blocks)
        )
        self._cache_policy = (
            int(total_blocks),
            int(block_size),
            int(linear_step),
            bool(hybrid_tail_required),
        )
        self._closed = False

    @property
    def grpc_port(self):
        return super().get_grpc_port()

    def init(self):
        if not super().init():
            raise RuntimeError("failed to start navi sched role")
        super().report_cache_status(*self._cache_policy)
        return self

    def stop(self):
        if not self._closed:
            super().stop()
            self._closed = True

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc, traceback):
        self.stop()

    def __del__(self):
        try:
            self.stop()
        except Exception:
            pass


__all__ = ["NaviSchedAgent"]
