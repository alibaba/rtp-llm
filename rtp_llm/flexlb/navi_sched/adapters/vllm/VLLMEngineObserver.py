import asyncio
import copy
import itertools
import json
import logging
import math
import os
import sys
import threading
import time
from collections import deque
from collections.abc import Callable
from concurrent.futures import CancelledError, Future
from contextlib import nullcontext
from dataclasses import dataclass, field, replace
from functools import partial
from types import SimpleNamespace
from typing import Any

from navi_sched.util import DEFAULT_AGENT_RUNTIME_NAME, naviSchedModule

from navi_sched import NaviSchedAgent

from . import ExecutionTimingProbe as _execution_timing_probe
from .VLLMAdapterUtils import NAVI_SCHED_MEDIA_ERROR_PREFIX, make_kmon_tags
from .VLLMBlockHash import block_hash_to_uint64, stable_request_id
from .VLLMCachePolicy import (
    VLLMCachePolicy,
    VLLMCacheState,
    get_vllm_full_attention_query_drop_tokens,
    is_vllm_eagle_block_drop_enabled,
    is_vllm_native_pd_prefill,
)
from .VLLMForwardProtocol import forward_outputs_fields, forward_request_type
from .VLLMImageCacheMap import VLLMImageCacheMap

logger = logging.getLogger("vllm.navi_sched.agent")

_STALLED_RUNNING_SCAN_NS = 5_000_000_000
_STALLED_RUNNING_THRESHOLD_NS = 30_000_000_000
_STALLED_RUNNING_LOG_INTERVAL_NS = 60_000_000_000
_STALLED_RUNNING_MAX_LOGS_PER_SCAN = 4
_SLOW_ENGINE_STAGE_NS = 100_000_000
_SEVERE_ENGINE_STAGE_NS = 1_000_000_000
_CACHE_SNAPSHOT_INTERVAL_NS = 100_000_000
_SLOW_ENGINE_STAGE_LOG_INTERVAL_NS = 5_000_000_000

_LATENCY_PARAMETER_LEARNING_RATE_SCALES = (
    4.0,
    4.0,
    4.0,
    0.8786028245332912,
    1.0,
    0.10067484459438397,
    1.0,
    4.0,
    4.0,
    4.0,
    2.361137720592015,
    4.0,
    0.014903028756332782,
    0.00647920238757579,
    0.013073034322480767,
    4.0,
    4.0,
    1.0,
    1.0,
)


class _MediaPreprocessingError(ValueError):
    pass


def _v6d_peer_cache_enabled(scheduler) -> bool:
    """Read the connected client's cached capability once, before Agent init."""
    module = sys.modules.get("kvs_connector.v6d_object_connector")
    connector_type = getattr(module, "V6dObjectConnector", None)
    if connector_type is None:
        return False
    connectors = [getattr(scheduler, "connector", None)]
    while connectors:
        connector = connectors.pop()
        if isinstance(connector, connector_type):
            source = connector.scheduler
            # Multi-node TP needs per-shard capability/domain reporting first.
            if getattr(source, "_is_cross_node_tp", False):
                return False
            context = getattr(source, "_client_context", None)
            client = getattr(context, "client", None)
            info = getattr(client, "peer_info", None)
            if not isinstance(info, dict):
                return False
            # The existing SDK finishes lazy transfer initialization in its own
            # startup thread after the connector publishes the cache spec.
            polling_thread = getattr(client, "_polling_thread", None)
            if (
                info.get("is_lazy_strategy") is True
                and getattr(client, "tair_kvcm_runtime", None) is None
                and polling_thread is not None
            ):
                polling_thread.join(timeout=30.0)
                info = getattr(client, "peer_info", None)
                if not isinstance(info, dict):
                    return False
            status = info.get("kvcm_status", {})
            if not isinstance(status, dict):
                return False
            ready = (
                info.get("peer_type") == "vineyard"
                and status.get("mode") == "report-event-and-cache-transfer"
                and status.get("report_event_ready") is True
                and status.get("cache_transfer_ready") is True
            )
            if info.get("is_lazy_strategy") is True:
                ready = ready and getattr(client, "tair_kvcm_runtime", None) is not None
            return bool(ready)
        connectors.extend(getattr(connector, "_connectors", ()))
    return False


@dataclass(frozen=True)
class _AgentConfig:
    biz_name: str
    server_ip: str
    server_port: int
    gig_port: int
    batch_timeout_ms: int
    latency_fit_batch_size: int
    latency_fit_interval: int
    latency_learning_rate: float
    latency_online_learning_rate_scale: float
    latency_current_sample_weight: float
    latency_error_clip_ms: float
    latency_parameter_learning_rate_scales: list[float]
    queue_learning_rate: float
    latency_cold_start_fit_epochs: int
    latency_history_limit: int
    latency_history_bucket_capacity: int
    latency_persist_interval: int
    latency_persist_interval_ms: int
    latency_warmup_sample_count: int
    latency_validation_enabled: bool
    execution_geometry_enabled: bool
    execution_cache_source_enabled: bool
    latency_model_state_file: str
    residual_learning_rate: float
    residual_update_error_floor_ms: float
    residual_update_error_ratio: float
    residual_idle_decay_start_ms: int
    residual_idle_decay_end_ms: int
    engine_stage_audit_enabled: bool = False

    @classmethod
    def from_env(cls) -> "_AgentConfig":
        tuning_defaults = _latency_tuning_defaults()
        cache_source_enabled = _read_bool_env(
            "VLLM_NAVI_SCHED_EXECUTION_CACHE_SOURCE_ENABLED", False
        )
        default_parameter_rates = (
            () if cache_source_enabled else _LATENCY_PARAMETER_LEARNING_RATE_SCALES
        )
        config = cls(
            biz_name=os.environ.get("VLLM_NAVI_SCHED_BIZ_NAME", "").strip(),
            server_ip=os.environ.get("VLLM_NAVI_SCHED_SERVER_IP", "").strip(),
            server_port=int(os.environ.get("VLLM_NAVI_SCHED_SERVER_PORT", "0")),
            gig_port=int(os.environ.get("VLLM_NAVI_SCHED_AGENT_GIG_PORT", "0")),
            batch_timeout_ms=int(
                os.environ.get("VLLM_NAVI_SCHED_BATCH_TIMEOUT_MS", "50")
            ),
            latency_fit_batch_size=int(
                os.environ.get("VLLM_NAVI_SCHED_LATENCY_FIT_BATCH_SIZE", "4")
            ),
            latency_fit_interval=int(
                os.environ.get("VLLM_NAVI_SCHED_LATENCY_FIT_INTERVAL", "0")
            ),
            latency_learning_rate=float(
                os.environ.get("VLLM_NAVI_SCHED_LATENCY_LEARNING_RATE", "0.05")
            ),
            latency_online_learning_rate_scale=float(
                os.environ.get(
                    "VLLM_NAVI_SCHED_LATENCY_ONLINE_LEARNING_RATE_SCALE",
                    str(tuning_defaults["latency_online_learning_rate_scale"]),
                )
            ),
            latency_error_clip_ms=float(
                os.environ.get("VLLM_NAVI_SCHED_LATENCY_ERROR_CLIP_MS", "5000")
            ),
            latency_current_sample_weight=float(
                os.environ.get("VLLM_NAVI_SCHED_LATENCY_CURRENT_SAMPLE_WEIGHT", "4.0")
            ),
            latency_parameter_learning_rate_scales=json.loads(
                os.environ.get(
                    "VLLM_NAVI_SCHED_LATENCY_PARAMETER_LEARNING_RATE_SCALES",
                    json.dumps(default_parameter_rates),
                )
            ),
            queue_learning_rate=float(
                os.environ.get("VLLM_NAVI_SCHED_QUEUE_LEARNING_RATE", "0.002")
            ),
            latency_cold_start_fit_epochs=int(
                os.environ.get("VLLM_NAVI_SCHED_LATENCY_COLD_START_FIT_EPOCHS", "100")
            ),
            latency_history_limit=int(
                os.environ.get("VLLM_NAVI_SCHED_LATENCY_HISTORY_LIMIT", "2000")
            ),
            latency_history_bucket_capacity=int(
                os.environ.get("VLLM_NAVI_SCHED_LATENCY_HISTORY_BUCKET_CAPACITY", "64")
            ),
            latency_persist_interval=int(
                os.environ.get("VLLM_NAVI_SCHED_LATENCY_PERSIST_INTERVAL", "256")
            ),
            latency_persist_interval_ms=int(
                os.environ.get("VLLM_NAVI_SCHED_LATENCY_PERSIST_INTERVAL_MS", "10000")
            ),
            latency_warmup_sample_count=int(
                os.environ.get("VLLM_NAVI_SCHED_LATENCY_WARMUP_SAMPLE_COUNT", "10")
            ),
            latency_validation_enabled=_read_bool_env(
                "VLLM_NAVI_SCHED_LATENCY_VALIDATION", True
            ),
            execution_geometry_enabled=_read_bool_env(
                "VLLM_NAVI_SCHED_EXECUTION_GEOMETRY_ENABLED", False
            ),
            execution_cache_source_enabled=cache_source_enabled,
            latency_model_state_file=os.environ.get(
                "VLLM_NAVI_SCHED_LATENCY_MODEL_STATE_FILE", ""
            ).strip(),
            residual_learning_rate=float(
                os.environ.get(
                    "VLLM_NAVI_SCHED_RESIDUAL_LEARNING_RATE",
                    str(tuning_defaults["residual_learning_rate"]),
                )
            ),
            residual_update_error_floor_ms=float(
                os.environ.get(
                    "VLLM_NAVI_SCHED_RESIDUAL_UPDATE_ERROR_FLOOR_MS", "100.0"
                )
            ),
            residual_update_error_ratio=float(
                os.environ.get("VLLM_NAVI_SCHED_RESIDUAL_UPDATE_ERROR_RATIO", "0.1")
            ),
            residual_idle_decay_start_ms=int(
                os.environ.get("VLLM_NAVI_SCHED_RESIDUAL_IDLE_DECAY_START_MS", "120000")
            ),
            residual_idle_decay_end_ms=int(
                os.environ.get("VLLM_NAVI_SCHED_RESIDUAL_IDLE_DECAY_END_MS", "300000")
            ),
            engine_stage_audit_enabled=_read_bool_env(
                "VLLM_NAVI_SCHED_ENGINE_STAGE_AUDIT_ENABLED", False
            ),
        )
        if not config.biz_name:
            raise ValueError("VLLM_NAVI_SCHED_BIZ_NAME is required")
        if not 0 < config.server_port <= 65535:
            raise ValueError("VLLM_NAVI_SCHED_SERVER_PORT must be between 1 and 65535")
        if not 0 <= config.gig_port <= 65535:
            raise ValueError(
                "VLLM_NAVI_SCHED_AGENT_GIG_PORT must be between 0 and 65535"
            )
        if config.batch_timeout_ms <= 0:
            raise ValueError("VLLM_NAVI_SCHED_BATCH_TIMEOUT_MS must be positive")
        for name, value in (
            ("LATENCY_FIT_BATCH_SIZE", config.latency_fit_batch_size),
            ("LATENCY_COLD_START_FIT_EPOCHS", config.latency_cold_start_fit_epochs),
            ("LATENCY_HISTORY_LIMIT", config.latency_history_limit),
            (
                "LATENCY_HISTORY_BUCKET_CAPACITY",
                config.latency_history_bucket_capacity,
            ),
            ("LATENCY_PERSIST_INTERVAL", config.latency_persist_interval),
            ("LATENCY_PERSIST_INTERVAL_MS", config.latency_persist_interval_ms),
        ):
            if value <= 0:
                raise ValueError(f"VLLM_NAVI_SCHED_{name} must be positive")
        if config.latency_fit_interval < 0:
            raise ValueError(
                "VLLM_NAVI_SCHED_LATENCY_FIT_INTERVAL must be non-negative"
            )
        if config.latency_warmup_sample_count < 0:
            raise ValueError(
                "VLLM_NAVI_SCHED_LATENCY_WARMUP_SAMPLE_COUNT must be non-negative"
            )
        if not math.isfinite(config.latency_learning_rate) or (
            config.latency_learning_rate <= 0.0
        ):
            raise ValueError(
                "VLLM_NAVI_SCHED_LATENCY_LEARNING_RATE must be positive and finite"
            )
        for name, value in (
            (
                "LATENCY_ONLINE_LEARNING_RATE_SCALE",
                config.latency_online_learning_rate_scale,
            ),
            ("LATENCY_CURRENT_SAMPLE_WEIGHT", config.latency_current_sample_weight),
        ):
            if not math.isfinite(value) or value <= 0.0:
                raise ValueError(f"VLLM_NAVI_SCHED_{name} must be positive and finite")
        if (
            not math.isfinite(config.latency_error_clip_ms)
            or config.latency_error_clip_ms < 0
        ):
            raise ValueError(
                "VLLM_NAVI_SCHED_LATENCY_ERROR_CLIP_MS must be nonnegative and finite"
            )
        rates = config.latency_parameter_learning_rate_scales
        source_rates = config.execution_cache_source_enabled
        if (
            not isinstance(rates, list)
            or len(rates) not in ((0, 21) if source_rates else (0, 10, 17, 19))
            or any(
                not isinstance(rate, (int, float))
                or isinstance(rate, bool)
                or not math.isfinite(rate)
                or rate < 0.0
                or (source_rates and (rate != 0.0 if 10 <= index < 19 else rate <= 0.0))
                or (not source_rates and rate == 0.0 and index < 17)
                for index, rate in enumerate(rates)
            )
        ):
            expected = (
                "21 finite numbers; inactive entries 10 through 18 must be zero "
                "and active entries must be positive"
                if source_rates
                else "10, 17, or 19 finite numbers; only decode-progress entries "
                "17 and 18 may be zero"
            )
            raise ValueError(
                "VLLM_NAVI_SCHED_LATENCY_PARAMETER_LEARNING_RATE_SCALES "
                "must be an empty array or a JSON array of " + expected
            )
        if not math.isfinite(config.queue_learning_rate) or (
            config.queue_learning_rate <= 0.0
        ):
            raise ValueError(
                "VLLM_NAVI_SCHED_QUEUE_LEARNING_RATE must be positive and finite"
            )
        if not math.isfinite(config.residual_learning_rate) or not (
            0.0 <= config.residual_learning_rate <= 1.0
        ):
            raise ValueError("VLLM_NAVI_SCHED_RESIDUAL_LEARNING_RATE must be in [0, 1]")
        for name, value in (
            (
                "RESIDUAL_UPDATE_ERROR_FLOOR_MS",
                config.residual_update_error_floor_ms,
            ),
            (
                "RESIDUAL_UPDATE_ERROR_RATIO",
                config.residual_update_error_ratio,
            ),
        ):
            if not math.isfinite(value) or value < 0.0:
                raise ValueError(
                    f"VLLM_NAVI_SCHED_{name} must be finite and non-negative"
                )
        if config.residual_idle_decay_start_ms < 0 or (
            config.residual_idle_decay_end_ms <= config.residual_idle_decay_start_ms
        ):
            raise ValueError("VLLM_NAVI_SCHED residual idle decay interval is invalid")
        return config


def _read_bool_env(name: str, default: bool) -> bool:
    value = os.environ.get(name)
    if value is None:
        return default
    normalized = value.strip().lower()
    if normalized in ("1", "true"):
        return True
    if normalized in ("0", "false"):
        return False
    raise ValueError(f"{name} must be 0, 1, false, or true")


def _runtime_accelerator_type() -> str:
    try:
        from vllm.platforms import current_platform
    except ImportError:
        return ""

    try:
        return str(current_platform.get_device_name()).strip()
    except Exception:
        logger.warning(
            "failed to identify accelerator through the vLLM platform",
            exc_info=True,
        )
        return ""


def _stable_model_identity(model: Any) -> str:
    model_id = str(model)
    normalized = os.path.normpath(model_id)
    parent_name = os.path.basename(os.path.dirname(normalized))
    if parent_name.startswith("model_temp_copy_"):
        return os.path.basename(normalized)
    return model_id


def _stable_fingerprint_value(value: Any) -> Any:
    if value is None:
        return ""
    if isinstance(value, (bool, int, float, str)):
        return value
    enum_value = getattr(value, "value", None)
    if isinstance(enum_value, (bool, int, float, str)):
        return enum_value
    return str(value)


def _model_fingerprint_values(model: Any, prefix: str = "") -> dict[str, Any]:
    hf_config = getattr(model, "hf_config", None)
    return {
        f"{prefix}model": _stable_model_identity(getattr(model, "model", "")),
        f"{prefix}revision": _stable_fingerprint_value(getattr(model, "revision", "")),
        f"{prefix}code_revision": _stable_fingerprint_value(
            getattr(model, "code_revision", "")
        ),
        f"{prefix}resolved_revision": _stable_fingerprint_value(
            getattr(hf_config, "_commit_hash", "")
        ),
        f"{prefix}dtype": _stable_fingerprint_value(getattr(model, "dtype", "")),
        f"{prefix}quantization": _stable_fingerprint_value(
            getattr(model, "quantization", "")
        ),
        f"{prefix}max_model_len": int(getattr(model, "max_model_len", 0) or 0),
    }


def _speculative_fingerprint_values(speculative: Any) -> dict[str, Any]:
    if speculative is None:
        return {"enabled": False}
    values = {"enabled": True}
    for name in (
        "method",
        "model",
        "revision",
        "code_revision",
        "quantization",
        "num_speculative_tokens",
        "draft_tensor_parallel_size",
        "max_model_len",
        "enforce_eager",
        "disable_padded_drafter_batch",
        "use_local_argmax_reduction",
        "prompt_lookup_max",
        "prompt_lookup_min",
        "speculative_token_tree",
        "parallel_drafting",
        "suffix_decoding_max_tree_depth",
        "suffix_decoding_max_cached_requests",
        "suffix_decoding_max_spec_factor",
        "suffix_decoding_min_token_prob",
    ):
        values[name] = _stable_fingerprint_value(getattr(speculative, name, ""))
    values.update(
        _model_fingerprint_values(
            getattr(speculative, "draft_model_config", None), "draft_"
        )
    )
    draft_parallel = getattr(speculative, "draft_parallel_config", None)
    values["draft_pipeline_parallel_size"] = int(
        getattr(draft_parallel, "pipeline_parallel_size", 0) or 0
    )
    values["draft_data_parallel_size"] = int(
        getattr(draft_parallel, "data_parallel_size", 0) or 0
    )
    return values


def _vllm_binary_fingerprint() -> str:
    try:
        import vllm

        version = str(getattr(vllm, "__version__", ""))
    except ImportError:
        version = ""
    values = {
        "framework": "vllm",
        "version": version,
        "commit": (
            os.environ.get("VLLM_BUILD_COMMIT", "").strip()
            or os.environ.get("VLLM_COMMIT", "").strip()
        ),
    }
    return json.dumps(values, separators=(",", ":"), sort_keys=True)


def _runtime_hardware_fingerprint() -> str:
    return (
        _runtime_accelerator_type()
        or os.environ.get("VLLM_NAVI_SCHED_ACCELERATOR_TYPE", "").strip()
        or os.environ.get("GPU_MODEL_DETAIL", "").strip()
        or os.environ.get("DEPLOYMENT_NAME", "").strip()
        or os.environ.get("HIPPO_ROLE", "").strip()
        or "unknown"
    )


def _latency_tuning_defaults() -> dict[str, float]:
    return {
        "latency_online_learning_rate_scale": 1.0,
        "residual_learning_rate": 0.0,
    }


def _latency_model_fingerprint(vllm_config: Any) -> str:
    parallel = getattr(vllm_config, "parallel_config", None)
    model = getattr(vllm_config, "model_config", None)
    speculative = _speculative_fingerprint_values(
        getattr(vllm_config, "speculative_config", None)
    )
    formula_speculative_keys = (
        "enabled",
        "method",
        "model",
        "quantization",
        "num_speculative_tokens",
        "draft_tensor_parallel_size",
        "parallel_drafting",
        "draft_model",
        "draft_dtype",
        "draft_quantization",
        "draft_pipeline_parallel_size",
        "draft_data_parallel_size",
    )
    values = {
        "fingerprint_version": 3,
        "navi_sched_model_version": (
            os.environ.get("NAVI_SCHED_MODEL_VERSION", "").strip()
            or "native-start-output-complete-v34"
        ),
        "framework": "vllm",
        "hardware": _runtime_hardware_fingerprint(),
        "tensor_parallel_size": int(getattr(parallel, "tensor_parallel_size", 1)),
        "pipeline_parallel_size": int(getattr(parallel, "pipeline_parallel_size", 1)),
        "data_parallel_size": int(getattr(parallel, "data_parallel_size", 1)),
        "decode_context_parallel_size": int(
            getattr(parallel, "decode_context_parallel_size", 1)
        ),
        "prefill_context_parallel_size": int(
            getattr(parallel, "prefill_context_parallel_size", 1)
        ),
        "enable_expert_parallel": bool(
            getattr(parallel, "enable_expert_parallel", False)
        ),
        "model": _stable_model_identity(getattr(model, "model", "")),
        "dtype": _stable_fingerprint_value(getattr(model, "dtype", "")),
        "quantization": _stable_fingerprint_value(getattr(model, "quantization", "")),
        "speculative": {
            key: speculative.get(key, "") for key in formula_speculative_keys
        },
    }
    return json.dumps(values, separators=(",", ":"), sort_keys=True)


def _latency_history_fingerprint(
    vllm_config: Any, formula_fingerprint: str | None = None
) -> str:
    parallel = getattr(vllm_config, "parallel_config", None)
    scheduler = getattr(vllm_config, "scheduler_config", None)
    model = getattr(vllm_config, "model_config", None)
    cache = getattr(vllm_config, "cache_config", None)
    values = {
        "fingerprint_version": 2,
        "formula_fingerprint": (
            formula_fingerprint
            if formula_fingerprint is not None
            else _latency_model_fingerprint(vllm_config)
        ),
        "max_num_seqs": int(getattr(scheduler, "max_num_seqs", 0)),
        "max_num_batched_tokens": int(
            getattr(scheduler, "max_num_batched_tokens", 0) or 0
        ),
        "enable_chunked_prefill": bool(
            getattr(scheduler, "enable_chunked_prefill", False)
        ),
        "block_size": int(getattr(cache, "block_size", 0) or 0),
        "gpu_memory_utilization": _stable_fingerprint_value(
            getattr(cache, "gpu_memory_utilization", "")
        ),
        "cpu_offload_gb": _stable_fingerprint_value(
            getattr(cache, "cpu_offload_gb", "")
        ),
        "kv_cache_dtype": _stable_fingerprint_value(
            getattr(cache, "cache_dtype", getattr(cache, "kv_cache_dtype", ""))
        ),
        "decode_context_parallel_size": int(
            getattr(parallel, "decode_context_parallel_size", 1)
        ),
        "prefill_context_parallel_size": int(
            getattr(parallel, "prefill_context_parallel_size", 1)
        ),
        "speculative": _speculative_fingerprint_values(
            getattr(vllm_config, "speculative_config", None)
        ),
    }
    values.update(_model_fingerprint_values(model))
    return json.dumps(values, separators=(",", ":"), sort_keys=True)


def _native_pd_latency_model_fingerprint(
    formula_fingerprint: str,
    cache_policy: VLLMCachePolicy,
    execution_policy=None,
    execution_geometry_enabled: bool = False,
    projected_execution_geometry_enabled: bool = False,
    execution_cache_source_enabled: bool = False,
) -> str:
    """Separate TPU PD training state when vLLM's effective cache geometry changes."""
    values = json.loads(formula_fingerprint)
    values["navi_latency_mode"] = (
        "split_native_start_execution_geometry_v32"
        if execution_geometry_enabled
        else "split_native_start_output_complete_v3"
    )
    values["execution_geometry_enabled"] = bool(execution_geometry_enabled)
    if execution_geometry_enabled and projected_execution_geometry_enabled:
        values["navi_latency_mode"] = "split_native_start_lookup_projected_geometry_v33"
        values["projected_execution_geometry_enabled"] = True
        values["prefill_execution_feature_protocol"] = 1
    if execution_cache_source_enabled:
        if not execution_geometry_enabled:
            raise ValueError("cache source requires execution geometry")
        values["navi_latency_mode"] = (
            "split_native_start_cache_source_geometry_v34"
            if projected_execution_geometry_enabled
            else "split_native_start_measured_cache_source_geometry_v35"
        )
        values["projected_execution_geometry_enabled"] = bool(
            projected_execution_geometry_enabled
        )
        if not projected_execution_geometry_enabled:
            values["training_geometry_origin"] = "observed_runner"
            values["dispatch_geometry_origin"] = "all_cache_resolved_forecast"
        values["execution_cache_source_feature_protocol"] = 1
    values["cache_prediction_version"] = 1
    values["execution_completion_boundary"] = "scheduler_update_complete_v1"
    values["execution_start_boundary"] = "native_start_call_v1"
    if execution_geometry_enabled and execution_policy is not None:
        values["runtime_prefill_execution_policy"] = execution_policy
    values["effective_cache_policy"] = {
        "block_size": cache_policy.block_size,
        "linear_step": cache_policy.linear_step,
        "hybrid_tail_required": cache_policy.hybrid_tail_required,
    }
    return json.dumps(values, separators=(",", ":"), sort_keys=True)


class _BatchCompletion:
    def __init__(self, observer, execution_step):
        self._observer = observer
        self._execution_step = execution_step

    def __call__(self, future: Future) -> None:
        execution_step = self._execution_step
        self._execution_step = None
        self._observer._complete_backend_execution(execution_step, future)


@dataclass
class _ExecutionStep:
    started_ns: int
    submitted_ns: int = 0
    future_completed_ns: int = 0
    global_padded_tokens: int | None = None
    scheduler_output_key: int = 0
    backend_execution_started: bool = False
    backend_completion_recorded: bool = False
    execution_tracking_registered: bool = False
    scheduled_tokens_by_request: dict[str, int] = field(default_factory=dict)
    prefill_request_count: int = 0
    prefill_tokens: int = 0
    decode_request_count: int = 0
    decode_tokens: int = 0
    active_executions_before: int = 0
    active_prefill_executions_before: int = 0
    active_decode_executions_before: int = 0
    active_scheduled_tokens_before: int = 0
    running_request_count: int = 0
    running_prefill_request_count: int = 0
    running_decode_request_count: int = 0
    running_recompute_request_count: int = 0
    waiting_request_count: int = 0
    preempted_request_count: int = 0
    resumed_request_count: int = 0
    workload_counts_collected: bool = False
    batch_progress: list[Any] = field(default_factory=list)


@dataclass(frozen=True)
class _ExecutionStartState:
    active_executions: int = -1
    active_prefill_executions: int = -1
    active_decode_executions: int = -1
    active_scheduled_tokens: int = -1
    running_requests: int = -1
    running_prefill_requests: int = -1
    running_decode_requests: int = -1
    running_recompute_requests: int = -1
    waiting_requests: int = -1
    preempted_requests: int = -1
    resumed_requests: int = -1
    step_prefill_requests: int = -1
    step_prefill_tokens: int = -1
    step_decode_requests: int = -1
    step_decode_tokens: int = -1


@dataclass(frozen=True)
class _PrefillWorkloadSlice:
    request_count: int = 0
    context_tokens: int = 0
    compute_tokens: int = 0
    compute_token_square_sum: float = 0.0
    context_compute_token_product_sum: float = 0.0

    def valid(self) -> bool:
        return (
            self.request_count >= 0
            and self.context_tokens >= 0
            and self.compute_tokens >= 0
            and math.isfinite(self.compute_token_square_sum)
            and self.compute_token_square_sum >= 0.0
            and math.isfinite(self.context_compute_token_product_sum)
            and self.context_compute_token_product_sum >= 0.0
        )


@dataclass
class _PrefillWorkloadSliceAggregate:
    request_count: int = 0
    context_tokens: int = 0
    compute_tokens: int = 0
    compute_token_square_sum: float = 0.0
    context_compute_token_product_sum: float = 0.0

    @classmethod
    def from_snapshot(
        cls, workload: _PrefillWorkloadSlice
    ) -> "_PrefillWorkloadSliceAggregate":
        return cls(
            request_count=workload.request_count,
            context_tokens=workload.context_tokens,
            compute_tokens=workload.compute_tokens,
            compute_token_square_sum=workload.compute_token_square_sum,
            context_compute_token_product_sum=(
                workload.context_compute_token_product_sum
            ),
        )

    def apply(self, contribution: "_WorkloadContribution") -> None:
        if contribution.remaining_prefill_tokens <= 0:
            return
        self.apply_request(
            contribution.prefill_context_tokens,
            contribution.remaining_prefill_tokens,
        )

    def apply_request(self, context_tokens: int, compute_tokens: int) -> None:
        self.request_count += 1
        self.context_tokens += context_tokens
        self.compute_tokens += compute_tokens
        self.compute_token_square_sum += float(compute_tokens**2)
        self.context_compute_token_product_sum += float(context_tokens * compute_tokens)

    def snapshot(self) -> _PrefillWorkloadSlice:
        return _PrefillWorkloadSlice(
            request_count=self.request_count,
            context_tokens=self.context_tokens,
            compute_tokens=self.compute_tokens,
            compute_token_square_sum=self.compute_token_square_sum,
            context_compute_token_product_sum=(self.context_compute_token_product_sum),
        )


@dataclass(frozen=True)
class _ObservedWorkload:
    batch_count: int = 0
    request_count: int = 0
    remaining_prefill_request_count: int = 0
    running_request_count: int = 0
    running_elapsed_time_ms: float = 0.0
    running_decode_request_count: int = 0
    running_decode_generated_tokens: int = 0
    running_decode_remaining_token_budget: int = 0
    observed_time_ns: int = field(default=0, compare=False, repr=False)
    waiting_request_count: int = 0
    remaining_prefill_context_tokens: int = 0
    remaining_prefill_tokens: int = 0
    remaining_prefill_token_square_sum: float = 0.0
    context_remaining_prefill_token_product_sum: float = 0.0
    remaining_total_context_tokens: int = 0
    remaining_total_tokens: int = 0
    remaining_total_token_square_sum: float = 0.0
    context_remaining_total_token_product_sum: float = 0.0
    total_prompt_tokens: int = 0
    inflight_scheduled_tokens: int = 0
    max_running_request_count: int = 0
    running_capacity_slot_count: int = 0
    running_capacity_prefill: _PrefillWorkloadSlice = field(
        default_factory=_PrefillWorkloadSlice
    )
    waiting_capacity_prefill: _PrefillWorkloadSlice = field(
        default_factory=_PrefillWorkloadSlice
    )
    scheduler_running_prefill: _PrefillWorkloadSlice = field(
        default_factory=_PrefillWorkloadSlice
    )
    scheduler_waiting_prefill: _PrefillWorkloadSlice = field(
        default_factory=_PrefillWorkloadSlice
    )

    waiting_request_ids: tuple[int, ...] | None = field(
        default=None, compare=False, repr=False
    )
    native_handoff_batch_ids: tuple[int, ...] = field(
        default=(), compare=False, repr=False
    )

    def valid(self) -> bool:
        return (
            self.batch_count >= 0
            and self.batch_count <= self.request_count
            and self.request_count >= 0
            and 0 <= self.remaining_prefill_request_count <= self.request_count
            and self.running_request_count >= 0
            and math.isfinite(self.running_elapsed_time_ms)
            and self.running_elapsed_time_ms >= 0.0
            and (
                self.scheduler_running_prefill.request_count > 0
                or self.running_elapsed_time_ms == 0.0
            )
            and 0 <= self.running_decode_request_count <= self.running_request_count
            and self.running_decode_generated_tokens
            >= self.running_decode_request_count
            and (
                self.running_decode_request_count > 0
                or self.running_decode_generated_tokens == 0
            )
            and self.running_decode_remaining_token_budget >= 0
            and (
                self.running_decode_request_count > 0
                or self.running_decode_remaining_token_budget == 0
            )
            and self.observed_time_ns >= 0
            and self.waiting_request_count >= 0
            and self.remaining_prefill_context_tokens >= 0
            and self.remaining_prefill_tokens >= 0
            and math.isfinite(self.remaining_prefill_token_square_sum)
            and self.remaining_prefill_token_square_sum >= 0.0
            and math.isfinite(self.context_remaining_prefill_token_product_sum)
            and self.context_remaining_prefill_token_product_sum >= 0.0
            and self.remaining_total_context_tokens
            >= self.remaining_prefill_context_tokens
            and self.remaining_total_tokens >= self.remaining_prefill_tokens
            and math.isfinite(self.remaining_total_token_square_sum)
            and self.remaining_total_token_square_sum
            >= self.remaining_prefill_token_square_sum
            and math.isfinite(self.context_remaining_total_token_product_sum)
            and self.context_remaining_total_token_product_sum
            >= self.context_remaining_prefill_token_product_sum
            and self.total_prompt_tokens >= 0
            and self.inflight_scheduled_tokens >= 0
            and self.max_running_request_count >= 0
            and 0 <= self.running_capacity_slot_count <= self.max_running_request_count
            and self.running_capacity_prefill.valid()
            and self.running_capacity_prefill.request_count
            <= self.running_capacity_slot_count
            and self.waiting_capacity_prefill.valid()
            and (
                self.waiting_capacity_prefill.request_count == 0
                or self.running_capacity_slot_count == self.max_running_request_count
            )
            and self.remaining_prefill_request_count
            == self.running_capacity_prefill.request_count
            + self.waiting_capacity_prefill.request_count
            and self.remaining_prefill_context_tokens
            == self.running_capacity_prefill.context_tokens
            + self.waiting_capacity_prefill.context_tokens
            and self.remaining_prefill_tokens
            == self.running_capacity_prefill.compute_tokens
            + self.waiting_capacity_prefill.compute_tokens
            and self.remaining_prefill_token_square_sum
            == self.running_capacity_prefill.compute_token_square_sum
            + self.waiting_capacity_prefill.compute_token_square_sum
            and self.context_remaining_prefill_token_product_sum
            == self.running_capacity_prefill.context_compute_token_product_sum
            + self.waiting_capacity_prefill.context_compute_token_product_sum
            and self.scheduler_running_prefill.valid()
            and self.scheduler_running_prefill.request_count
            <= self.running_request_count
            and self.scheduler_waiting_prefill.valid()
            and self.remaining_prefill_request_count
            == self.scheduler_running_prefill.request_count
            + self.scheduler_waiting_prefill.request_count
            and self.remaining_prefill_context_tokens
            == self.scheduler_running_prefill.context_tokens
            + self.scheduler_waiting_prefill.context_tokens
            and self.remaining_prefill_tokens
            == self.scheduler_running_prefill.compute_tokens
            + self.scheduler_waiting_prefill.compute_tokens
            and self.remaining_prefill_token_square_sum
            == self.scheduler_running_prefill.compute_token_square_sum
            + self.scheduler_waiting_prefill.compute_token_square_sum
            and self.context_remaining_prefill_token_product_sum
            == self.scheduler_running_prefill.context_compute_token_product_sum
            + self.scheduler_waiting_prefill.context_compute_token_product_sum
        )


@dataclass(frozen=True)
class _WorkloadContribution:
    batch_id: int
    prompt_tokens: int
    prefill_context_tokens: int
    remaining_prefill_tokens: int
    total_context_tokens: int
    remaining_total_tokens: int


@dataclass
class _WorkloadAggregate:
    batch_count: int = 0
    request_count: int = 0
    remaining_prefill_request_count: int = 0
    remaining_prefill_context_tokens: int = 0
    remaining_prefill_tokens: int = 0
    remaining_prefill_token_square_sum: float = 0.0
    context_remaining_prefill_token_product_sum: float = 0.0
    remaining_total_context_tokens: int = 0
    remaining_total_tokens: int = 0
    remaining_total_token_square_sum: float = 0.0
    context_remaining_total_token_product_sum: float = 0.0
    total_prompt_tokens: int = 0
    batch_request_counts: dict[int, int] = field(default_factory=dict)

    @classmethod
    def from_snapshot(cls, workload: _ObservedWorkload) -> "_WorkloadAggregate":
        return cls(
            batch_count=workload.batch_count,
            request_count=workload.request_count,
            remaining_prefill_request_count=(workload.remaining_prefill_request_count),
            remaining_prefill_context_tokens=(
                workload.remaining_prefill_context_tokens
            ),
            remaining_prefill_tokens=workload.remaining_prefill_tokens,
            remaining_prefill_token_square_sum=(
                workload.remaining_prefill_token_square_sum
            ),
            context_remaining_prefill_token_product_sum=(
                workload.context_remaining_prefill_token_product_sum
            ),
            remaining_total_context_tokens=workload.remaining_total_context_tokens,
            remaining_total_tokens=workload.remaining_total_tokens,
            remaining_total_token_square_sum=(
                workload.remaining_total_token_square_sum
            ),
            context_remaining_total_token_product_sum=(
                workload.context_remaining_total_token_product_sum
            ),
            total_prompt_tokens=workload.total_prompt_tokens,
        )

    def apply(self, contribution: _WorkloadContribution, sign: int) -> None:
        batch_id = contribution.batch_id
        previous_batch_requests = self.batch_request_counts.get(batch_id, 0)
        updated_batch_requests = previous_batch_requests + sign
        if updated_batch_requests < 0:
            raise RuntimeError("negative NaviSched workload batch reference count")
        if previous_batch_requests == 0 and updated_batch_requests > 0:
            self.batch_count += 1
        if updated_batch_requests == 0:
            self.batch_request_counts.pop(batch_id, None)
            if previous_batch_requests > 0:
                self.batch_count -= 1
        else:
            self.batch_request_counts[batch_id] = updated_batch_requests
        self.request_count += sign
        self.total_prompt_tokens += sign * contribution.prompt_tokens
        if contribution.remaining_prefill_tokens > 0:
            self.remaining_prefill_request_count += sign
            self.remaining_prefill_context_tokens += (
                sign * contribution.prefill_context_tokens
            )
            self.remaining_prefill_tokens += (
                sign * contribution.remaining_prefill_tokens
            )
            self.remaining_prefill_token_square_sum += sign * float(
                contribution.remaining_prefill_tokens**2
            )
            self.context_remaining_prefill_token_product_sum += sign * float(
                contribution.prefill_context_tokens
                * contribution.remaining_prefill_tokens
            )
        if contribution.remaining_total_tokens > 0:
            self.remaining_total_context_tokens += (
                sign * contribution.total_context_tokens
            )
            self.remaining_total_tokens += sign * contribution.remaining_total_tokens
            self.remaining_total_token_square_sum += sign * float(
                contribution.remaining_total_tokens**2
            )
            self.context_remaining_total_token_product_sum += sign * float(
                contribution.total_context_tokens * contribution.remaining_total_tokens
            )

    def snapshot(
        self,
        running_request_count: int,
        running_elapsed_time_ms: float,
        running_decode_request_count: int,
        running_decode_generated_tokens: int,
        running_decode_remaining_token_budget: int,
        observed_time_ns: int,
        waiting_request_count: int,
        inflight_scheduled_tokens: int,
        max_running_request_count: int,
        running_capacity_slot_count: int,
        running_capacity_prefill: _PrefillWorkloadSlice,
        waiting_capacity_prefill: _PrefillWorkloadSlice,
        scheduler_running_prefill: _PrefillWorkloadSlice,
        scheduler_waiting_prefill: _PrefillWorkloadSlice,
    ) -> _ObservedWorkload:
        return _ObservedWorkload(
            batch_count=self.batch_count,
            request_count=self.request_count,
            remaining_prefill_request_count=self.remaining_prefill_request_count,
            running_request_count=running_request_count,
            running_elapsed_time_ms=running_elapsed_time_ms,
            running_decode_request_count=running_decode_request_count,
            running_decode_generated_tokens=running_decode_generated_tokens,
            running_decode_remaining_token_budget=(
                running_decode_remaining_token_budget
            ),
            observed_time_ns=observed_time_ns,
            waiting_request_count=waiting_request_count,
            remaining_prefill_context_tokens=(self.remaining_prefill_context_tokens),
            remaining_prefill_tokens=self.remaining_prefill_tokens,
            remaining_prefill_token_square_sum=(
                self.remaining_prefill_token_square_sum
            ),
            context_remaining_prefill_token_product_sum=(
                self.context_remaining_prefill_token_product_sum
            ),
            remaining_total_context_tokens=self.remaining_total_context_tokens,
            remaining_total_tokens=self.remaining_total_tokens,
            remaining_total_token_square_sum=self.remaining_total_token_square_sum,
            context_remaining_total_token_product_sum=(
                self.context_remaining_total_token_product_sum
            ),
            total_prompt_tokens=self.total_prompt_tokens,
            inflight_scheduled_tokens=inflight_scheduled_tokens,
            max_running_request_count=max_running_request_count,
            running_capacity_slot_count=running_capacity_slot_count,
            running_capacity_prefill=running_capacity_prefill,
            waiting_capacity_prefill=waiting_capacity_prefill,
            scheduler_running_prefill=scheduler_running_prefill,
            scheduler_waiting_prefill=scheduler_waiting_prefill,
        )


@dataclass(frozen=True)
class _AdmissionState:
    preceding_workload: _ObservedWorkload = field(default_factory=_ObservedWorkload)
    incoming_request_count: int = 0
    incoming_predicted_cache_hit_tokens: int = 0
    incoming_compute_tokens: int = 0
    incoming_compute_token_square_sum: float = 0.0
    incoming_hit_compute_token_product_sum: float = 0.0
    incoming_total_prompt_tokens: int = 0
    incoming_running_capacity_request_ids: frozenset[int] = field(
        default_factory=frozenset
    )
    active_executions: int = 0


@dataclass
class _BatchProgress:
    batch_handle: Any
    state: Any | None
    completed_request_ids: set[int] = field(default_factory=set)
    prefill_request_count: int = 0
    prefill_tokens: int = 0


@dataclass
class _ForwardBatchState:
    batch_id: int
    schedule_batch_id: int
    schedule_batch_size: int
    pending_cache_tracked: bool
    native_batch: Any | None
    request_ids: tuple[int, ...]
    external_request_ids: dict[int, str]
    requests: dict[int, Any]
    predicted_cache_hits: dict[int, int] = field(default_factory=dict)
    arrived_request_ids: set[int] = field(default_factory=set)
    cancelled_request_ids: set[int] = field(default_factory=set)
    handle: Any = None
    initial_cache_hits: dict[int, int] = field(default_factory=dict)
    initial_external_cache_hits: dict[int, int] = field(default_factory=dict)
    initial_execution_policy_inputs: dict[int, tuple[int, int, int]] = field(
        default_factory=dict
    )
    execution_cache_reported: bool = False
    projected_execution_exclusive: bool = True
    tracking_aborted: bool = False
    submission_failed: bool = False
    prefill_complete: bool = False
    released: bool = False
    chunk_count: int = 0
    admission_started_ns: int = 0
    native_preparation_ms: float = 0.0
    execution_started_ns: int = 0
    execution_step_count: int = 0
    execution_step_ms: float = 0.0
    geometry_execution_step_count: int = 0
    global_padded_tokens: int = 0
    execution_prefill_request_count: int = 0
    execution_prefill_tokens: int = 0
    execution_decode_request_count: int = 0
    execution_decode_tokens: int = 0
    admission_state: _AdmissionState | None = None
    admission_contributions: dict[int, _WorkloadContribution] = field(
        default_factory=dict
    )
    admission_ready: bool = False
    admission_reported: bool = False
    execution_start_state: _ExecutionStartState | None = None
    completed_prefill_request_ids: set[int] = field(default_factory=set)
    pending_prefill_steps: int = 0
    lock: Any = field(default_factory=threading.RLock, repr=False)


@dataclass(frozen=True)
class _PreprocessedCacheTouch:
    payload: bytes
    engine_request: Any
    preprocessed_request: tuple[Any, int]
    accepted_time_ns: int


class _NaviSchedScheduler:
    def __init__(self, scheduler, observer):
        self._scheduler = scheduler
        self._observer = observer
        self._forward_enqueue_sequence = 0
        self._native_component_probe = None
        observer._native_scheduler_component_profile = None
        observer._native_scheduler_component_bound_hooks = 0
        observer._native_scheduler_component_install_failures = 0
        try:
            self._native_component_probe = (
                _execution_timing_probe.prepare_native_scheduler_component_probe(
                    scheduler
                )
            )
            if self._native_component_probe is not None:
                observer._native_scheduler_component_bound_hooks = len(
                    self._native_component_probe.bindings
                )
        except Exception:
            observer._native_scheduler_component_install_failures += 1
        observer._runtime_pipeline_diagnostic_bindings = None
        observer._runtime_pipeline_diagnostic_install_failures = 0
        try:
            observer._runtime_pipeline_diagnostic_bindings = (
                _execution_timing_probe.install_engine_runtime_diagnostic_logging()
            )
        except Exception:
            observer._runtime_pipeline_diagnostic_install_failures += 1
        observer._main_thread_stdio_probe = None
        observer._stdio_probe_install_failures = 0
        if self._native_component_probe is not None:
            try:
                observer._main_thread_stdio_probe = (
                    _execution_timing_probe.install_main_thread_stdio_probe(observer)
                )
            except Exception:
                observer._stdio_probe_install_failures += 1

    @property
    def __wrapped__(self):
        """Expose the native scheduler for framework initialization and profiling."""
        return self._scheduler

    def __getattr__(self, name):
        return getattr(self._scheduler, name)

    def add_request(self, request):
        started_ns = time.perf_counter_ns()
        timing = {"phase": "admission", "forwarded_requests": 0}
        try:
            return self._add_request(request, timing)
        finally:
            finished_ns = time.perf_counter_ns()
            if finished_ns - started_ns >= _SLOW_ENGINE_STAGE_NS:
                try:
                    self._observer._log_slow_engine_stage(
                        "add_request",
                        started_ns,
                        finished_ns,
                        request_id=getattr(request, "request_id", ""),
                        phase=timing["phase"],
                        admission_ms=timing.get("admission_ms", -1.0),
                        snapshot_ms=timing.get("snapshot_ms", -1.0),
                        native_add_ms=timing.get("native_add_ms", -1.0),
                        ready_ms=timing.get("ready_ms", -1.0),
                        report_ms=timing.get("report_ms", -1.0),
                        forwarded_requests=timing["forwarded_requests"],
                        running=len(getattr(self._scheduler, "running", ()) or ()),
                        waiting=len(getattr(self._scheduler, "waiting", ()) or ()),
                    )
                except Exception:
                    pass

    def _add_request(self, request, timing):
        # DashLLM also delivers Master-forwarded ADDs through this same
        # EngineCore queue. Keep their admission metadata so execution does
        # not register an already-owned request as a second local batch.
        if (
            self._observer._native_ingress
            and self._observer._navi_request_id(request)
            not in self._observer._pending_forward_batches
        ):
            timing["phase"] = "native_ingress"
            phase_started_ns = time.perf_counter_ns()
            try:
                self._scheduler.add_request(request)
            finally:
                timing["native_add_ms"] = (
                    time.perf_counter_ns() - phase_started_ns
                ) / 1_000_000.0
            timing["phase"] = "report_native_ingress"
            phase_started_ns = time.perf_counter_ns()
            try:
                self._observer._refresh_and_report_workload(
                    self._scheduler,
                    (str(request.request_id),),
                    "native_ingress_admission",
                )
            finally:
                timing["report_ms"] = (
                    time.perf_counter_ns() - phase_started_ns
                ) / 1_000_000.0
            timing["phase"] = "done"
            return
        if bool(getattr(request, "navi_sched_local_fallback", False)):
            timing["phase"] = "native_local_fallback"
            phase_started_ns = time.perf_counter_ns()
            try:
                self._scheduler.add_request(request)
            finally:
                timing["native_add_ms"] = (
                    time.perf_counter_ns() - phase_started_ns
                ) / 1_000_000.0
            timing["phase"] = "report_local_fallback"
            phase_started_ns = time.perf_counter_ns()
            try:
                self._observer._refresh_and_report_workload(
                    self._scheduler,
                    (str(request.request_id),),
                    "local_fallback_admission",
                )
            finally:
                timing["report_ms"] = (
                    time.perf_counter_ns() - phase_started_ns
                ) / 1_000_000.0
            timing["phase"] = "done"
            return
        phase_started_ns = time.perf_counter_ns()
        try:
            state = self._observer._admit_request(request)
        finally:
            timing["admission_ms"] = (
                time.perf_counter_ns() - phase_started_ns
            ) / 1_000_000.0
        if state is not None:
            timing["phase"] = "snapshot"
            phase_started_ns = time.perf_counter_ns()
            try:
                self._observer._record_admission_state(state, self._scheduler)
            finally:
                timing["snapshot_ms"] = (
                    time.perf_counter_ns() - phase_started_ns
                ) / 1_000_000.0
            added_request_ids = []
            timing["phase"] = "native_add"
            phase_started_ns = time.perf_counter_ns()
            try:
                for request_id in state.request_ids:
                    forwarded = state.requests.get(request_id)
                    if forwarded is not None:
                        forwarded._navi_forward_enqueue_order = (
                            self._forward_enqueue_sequence
                        )
                        self._forward_enqueue_sequence += 1
                        self._scheduler.add_request(forwarded)
                        added_request_ids.append(forwarded.request_id)
            except BaseException:
                self._rollback_failed_admission(state, added_request_ids)
                raise
            finally:
                timing["native_add_ms"] = (
                    time.perf_counter_ns() - phase_started_ns
                ) / 1_000_000.0
                timing["forwarded_requests"] = len(added_request_ids)
            timing["phase"] = "ready"
            phase_started_ns = time.perf_counter_ns()
            try:
                with state.lock:
                    state.admission_ready = True
            finally:
                timing["ready_ms"] = (
                    time.perf_counter_ns() - phase_started_ns
                ) / 1_000_000.0
            timing["phase"] = "report"
            phase_started_ns = time.perf_counter_ns()
            try:
                self._observer._refresh_and_report_workload(
                    self._scheduler,
                    added_request_ids,
                    "admission",
                    admitted_state=state,
                )
            finally:
                timing["report_ms"] = (
                    time.perf_counter_ns() - phase_started_ns
                ) / 1_000_000.0
        timing["phase"] = "done"

    def _rollback_failed_admission(self, state, added_request_ids):
        from vllm.v1.request import RequestStatus

        requests = getattr(self._scheduler, "requests", {})
        rollback_ids = set(added_request_ids)
        rollback_ids.update(
            external_request_id
            for external_request_id in state.external_request_ids.values()
            if external_request_id in requests
        )
        ordered_rollback_ids = [
            external_request_id
            for external_request_id in state.external_request_ids.values()
            if external_request_id in rollback_ids
        ]
        if ordered_rollback_ids:
            try:
                self._scheduler.finish_requests(
                    ordered_rollback_ids, RequestStatus.FINISHED_ABORTED
                )
            except BaseException:
                logger.exception(
                    "failed to roll back partially admitted NaviSched VLLM batch: "
                    "batch_id=%d request_ids=%s",
                    state.batch_id,
                    ordered_rollback_ids,
                )
        with state.lock:
            state.submission_failed = True
        active_request_ids = tuple(
            request_id
            for request_id in state.request_ids
            if request_id not in state.cancelled_request_ids
        )
        self._observer._fail_forward_responses(
            active_request_ids, "vLLM scheduler admission failed"
        )
        self._observer._remove_response_requests(state.request_ids)
        self._observer._abort_forward_requests(state, set(state.request_ids))
        try:
            self._observer._refresh_and_report_workload(
                self._scheduler, ordered_rollback_ids, "admission_rollback"
            )
        except BaseException:
            logger.exception(
                "failed to refresh NaviSched workload after admission rollback: "
                "batch_id=%d",
                state.batch_id,
            )

    def has_requests(self):
        has_requests = self._scheduler.has_requests()
        return has_requests

    def finish_requests(self, request_ids, finished_status):
        if request_ids is None:
            requested = None
        elif isinstance(request_ids, str):
            requested = {request_ids}
        else:
            requested = set(request_ids)

        states = self._observer._matching_forward_batches(requested)
        held_results = []
        for state in states:
            cancelled_ids = {
                request_id
                for request_id, external_id in state.external_request_ids.items()
                if requested is None or external_id in requested
            }
            if not cancelled_ids:
                continue
            for request_id in cancelled_ids:
                request = state.requests.get(request_id)
                if (
                    request is not None
                    and request.request_id not in self._scheduler.requests
                ):
                    request.status = finished_status
                    held_results.append(request)
            self._observer._abort_forward_requests(state, cancelled_ids)

        native_results = self._scheduler.finish_requests(request_ids, finished_status)
        native_ids = {request.request_id for request in native_results}
        changed_request_ids = set(native_ids)
        if requested is None:
            with self._observer._workload_lock:
                changed_request_ids.update(self._observer._workload_contributions)
        else:
            changed_request_ids.update(requested)
        self._observer._refresh_and_report_workload(
            self._scheduler, changed_request_ids, "finish_requests"
        )
        return native_results + [
            request for request in held_results if request.request_id not in native_ids
        ]

    def schedule(self, throttle_prefills: bool | None = None):
        started_ns = time.perf_counter_ns()
        self._observer._promote_arrived_cache_touches(self._scheduler)
        self._observer._cleanup_finished_forward_requests(
            self._scheduler.requests,
        )
        probe = self._native_component_probe
        probe_active = False
        self._observer._native_scheduler_component_profile = None
        if probe is not None:
            try:
                probe_active = probe.begin()
            except Exception:
                self._observer._native_scheduler_component_install_failures += 1
        before_native_ns = time.perf_counter_ns()
        try:
            if throttle_prefills is None:
                scheduler_output = self._scheduler.schedule()
            else:
                scheduler_output = self._scheduler.schedule(throttle_prefills)
        finally:
            after_native_ns = time.perf_counter_ns()
            if probe_active:
                try:
                    self._observer._native_scheduler_component_profile = probe.finish(
                        before_native_ns, after_native_ns
                    )
                except Exception:
                    self._observer._native_scheduler_component_install_failures += 1
        self._observer._register_scheduled_work(scheduler_output)
        touched_request_ids = set(scheduler_output.num_scheduled_tokens)
        touched_request_ids.update(
            getattr(scheduler_output, "preempted_req_ids", None) or ()
        )
        self._observer._refresh_and_report_workload(
            self._scheduler, touched_request_ids, "schedule"
        )
        finished_ns = time.perf_counter_ns()
        self._observer._log_slow_engine_stage(
            "schedule",
            started_ns,
            finished_ns,
            promote_ms=(before_native_ns - started_ns) / 1_000_000,
            native_ms=(after_native_ns - before_native_ns) / 1_000_000,
            report_ms=(finished_ns - after_native_ns) / 1_000_000,
            running=len(getattr(self._scheduler, "running", ())),
            waiting=len(getattr(self._scheduler, "waiting", ()) or ()),
            scheduled_tokens=sum(scheduler_output.num_scheduled_tokens.values()),
        )
        return scheduler_output

    def update_from_output(self, scheduler_output, model_runner_output):
        started_ns = time.perf_counter_ns()
        try:
            return self._scheduler.update_from_output(
                scheduler_output, model_runner_output
            )
        finally:
            after_native_ns = time.perf_counter_ns()
            self._observer._settle_scheduled_work(scheduler_output)
            self._observer._refresh_and_report_workload(
                self._scheduler,
                scheduler_output.num_scheduled_tokens,
                "update_from_output",
            )
            finished_ns = time.perf_counter_ns()
            self._observer._log_slow_engine_stage(
                "update_from_output",
                started_ns,
                finished_ns,
                native_ms=(after_native_ns - started_ns) / 1_000_000,
                report_ms=(finished_ns - after_native_ns) / 1_000_000,
                scheduled_tokens=sum(scheduler_output.num_scheduled_tokens.values()),
            )


class _RoutingOutputQueue:
    def __init__(self, output_queue, observer):
        self._output_queue = output_queue
        self._observer = observer

    def __getattr__(self, name):
        return getattr(self._output_queue, name)

    def put_nowait(self, item):
        if isinstance(item, tuple) and len(item) == 2:
            client_index, outputs = item
            scheduler_stats = getattr(outputs, "scheduler_stats", None)
            if scheduler_stats is not None:
                outputs.scheduler_stats = None
            try:
                route_started_ns = time.perf_counter_ns()
                routed = self._observer.route_engine_outputs(client_index, outputs)
                self._observer._log_slow_engine_stage(
                    "route_output",
                    route_started_ns,
                    time.perf_counter_ns(),
                    client_index=client_index,
                    output_count=len(getattr(outputs, "outputs", ()) or ()),
                )
            except Exception:
                if scheduler_stats is not None:
                    outputs.scheduler_stats = scheduler_stats
                raise
            if routed:
                if scheduler_stats is not None:
                    local_stats = type(outputs)(
                        scheduler_stats=scheduler_stats,
                        timestamp=getattr(outputs, "timestamp", 0.0),
                    )
                    self._output_queue.put_nowait((0, local_stats))
                return
            if scheduler_stats is not None:
                outputs.scheduler_stats = scheduler_stats
        self._output_queue.put_nowait(item)


class VLLMEngineObserver:
    def __init__(
        self,
        vllm_config,
        kv_cache_config,
        max_concurrent_batches: int = 1,
    ):
        self._native_ingress = is_vllm_native_pd_prefill()
        _execution_timing_probe.install_gc_timing_audit()
        cache_config = vllm_config.cache_config
        if not cache_config.enable_prefix_caching:
            raise ValueError("NaviSched requires vLLM prefix caching")
        cache_config.prefix_caching_hash_algo = "sha256_cbor"
        self._cache_policy = VLLMCachePolicy.from_engine_config(
            vllm_config, kv_cache_config
        )
        self._cache_states = {
            cache_key_type: VLLMCacheState(self._cache_policy)
            for cache_key_type in (0, 1)
        }
        self._image_cache_map = VLLMImageCacheMap(
            self._cache_states[0],
            self._cache_policy.hybrid_tail_required,
        )
        self._cache_lifecycle_lock = threading.Lock()
        self._cache_lifecycle_sequences = {0: 0, 1: 0}
        self._v6d_scheduler = None
        self._v6d_publication_restore = []
        self._v6d_unpublished_candidates = {}
        self._v6d_snapshot_future = None
        self._next_v6d_snapshot_poll_ns = 0
        self._block_size = self._cache_policy.block_size
        if os.environ.get("PYTHONHASHSEED") != "0":
            raise ValueError("NaviSched requires PYTHONHASHSEED=0")

        scheduler_config = vllm_config.scheduler_config
        config = _AgentConfig.from_env()
        self._engine_stage_audit_enabled = config.engine_stage_audit_enabled
        self._execution_geometry_enabled = (
            self._native_ingress and config.execution_geometry_enabled
        )
        if (
            config.execution_cache_source_enabled
            and not self._execution_geometry_enabled
        ):
            raise ValueError(
                "execution cache source requires native execution geometry"
            )
        self._prefill_execution_policy = None
        latency_model_fingerprint = _latency_model_fingerprint(vllm_config)
        if self._native_ingress:
            latency_model_fingerprint = _native_pd_latency_model_fingerprint(
                latency_model_fingerprint,
                self._cache_policy,
                execution_geometry_enabled=self._execution_geometry_enabled,
            )
        latency_history_fingerprint = _latency_history_fingerprint(
            vllm_config, latency_model_fingerprint
        )
        self._native_prepare_port = config.server_port
        self._batch_ids = itertools.count(1)
        self._batch_id_prefix = (os.getpid() & 0x7FFFFFFF) << 32
        self._pending_forward_batches: dict[int, _ForwardBatchState] = {}
        self._scheduled_forward_batches: dict[int, _ForwardBatchState] = {}
        self._forward_request_states: dict[str, _ForwardBatchState] = {}
        self._engine_core = None
        self._render_config = vllm_config
        self._render_continuation = None
        self._forward_stop = threading.Event()
        self._shutdown_lock = threading.Lock()
        self._shutdown_started = False
        self._forward_thread = None
        self._preprocessed_cache_touches: dict[int, _PreprocessedCacheTouch] = {}
        self._cache_touch_queue = deque()
        self._cache_touch_lock = threading.Lock()
        self._cache_touch_count = 0
        self._cache_touch_promoted_blocks = 0
        self._cache_touch_last_log_ns = time.monotonic_ns()
        self._response_encoder = None
        self._response_request_ids: dict[int, str] = {}
        self._response_image_positions: dict[int, tuple] = {}
        self._inactive_response_request_ids: set[int] = set()
        self._cancelled_before_decode: set[int] = set()
        self._pending_engine_abort_request_ids: list[str] = []
        self._response_lock = threading.Lock()
        self._execution_lock = threading.Lock()
        # The TPU execute_model future completes before sample_tokens waits for
        # the device output. PD batches finish only after that output is ready.
        self._pd_output_lock = threading.Lock()
        self._pd_output_completed_ns: dict[int, tuple[int, int]] = {}
        self._pd_output_geometry: dict[int, int | None] = {}
        self._pd_ready_finishes: dict[
            int, list[Any] | tuple[_ExecutionStep, Future]
        ] = {}
        self._workload_lock = threading.Lock()
        self._workload_report_lock = threading.Lock()
        self._prepared_workload_lock = threading.Lock()
        self._prepared_contributions: dict[
            int, tuple[_ForwardBatchState, _WorkloadContribution]
        ] = {}
        self._prepared_workload = _WorkloadAggregate()
        self._native_handoff_batches: dict[int, None] = {}
        self._active_execution_count = 0
        self._active_prefill_execution_count = 0
        self._active_decode_execution_count = 0
        self._active_scheduled_tokens = 0
        self._inflight_scheduled_tokens: dict[str, int] = {}
        self._scheduled_work_by_output: dict[int, dict[str, int]] = {}
        self._workload_dirty_request_ids: set[str] = set()
        self._workload_contributions: dict[str, _WorkloadContribution] = {}
        self._workload_aggregate = _WorkloadAggregate()
        self._current_workload: _ObservedWorkload | None = None
        self._running_prefill_started_ns: dict[str, int] = {}
        self._running_request_progress: dict[
            str, tuple[tuple[int, int, int], int, int]
        ] = {}
        self._last_stalled_running_scan_ns = 0
        self._slow_engine_stage_last_log_ns: dict[str, int] = {}
        self._request_audit_session = f"{os.getpid()}-{time.time_ns()}"
        self._request_audit_failures = 0
        self._vllm_config = vllm_config
        self._agent = None
        self._bridge = None
        self._agent_options = dict(
            biz_name=config.biz_name,
            server_ip=config.server_ip,
            server_port=config.server_port,
            total_blocks=self._cache_policy.total_blocks,
            block_size=self._block_size,
            linear_step=self._cache_policy.linear_step,
            hybrid_tail_required=self._cache_policy.hybrid_tail_required,
            full_attention_query_drop_tokens=(
                self._cache_policy.full_attention_query_drop_tokens
            ),
            pd_separated=self._native_ingress,
            mamba_replay_stop_shift=(
                self._native_ingress
                and self._cache_policy.hybrid_tail_required
                and getattr(vllm_config.cache_config, "mamba_cache_mode", None)
                == "align"
                and is_vllm_eagle_block_drop_enabled(vllm_config)
            ),
            mamba_align=(
                self._native_ingress
                and self._cache_policy.hybrid_tail_required
                and getattr(cache_config, "mamba_cache_mode", None) == "align"
            ),
            mamba_retention_interval=(
                -1
                if not self._native_ingress
                or getattr(cache_config, "prefix_cache_retention_interval", None)
                is None
                else int(cache_config.prefix_cache_retention_interval)
            ),
            latency_prediction_mode=("split" if self._native_ingress else "unified"),
            latency_model_fingerprint=latency_model_fingerprint,
            latency_history_fingerprint=latency_history_fingerprint,
            latency_history_binary_fingerprint=_vllm_binary_fingerprint(),
            name=DEFAULT_AGENT_RUNTIME_NAME,
            wait_no_query=False,
            kmon_tags=make_kmon_tags("agent"),
            gig={"grpc_port": config.gig_port},
            batch_timeout_ms=config.batch_timeout_ms,
            latency_fit_batch_size=config.latency_fit_batch_size,
            latency_fit_interval=config.latency_fit_interval,
            latency_learning_rate=config.latency_learning_rate,
            latency_online_learning_rate_scale=(
                config.latency_online_learning_rate_scale
            ),
            latency_current_sample_weight=config.latency_current_sample_weight,
            latency_error_clip_ms=config.latency_error_clip_ms,
            latency_parameter_learning_rate_scales=(
                config.latency_parameter_learning_rate_scales
            ),
            queue_learning_rate=config.queue_learning_rate,
            latency_cold_start_fit_epochs=config.latency_cold_start_fit_epochs,
            latency_history_limit=config.latency_history_limit,
            latency_history_bucket_capacity=(config.latency_history_bucket_capacity),
            latency_persist_interval=config.latency_persist_interval,
            latency_persist_interval_ms=config.latency_persist_interval_ms,
            latency_warmup_sample_count=config.latency_warmup_sample_count,
            latency_validation_enabled=config.latency_validation_enabled,
            execution_cache_source_enabled=config.execution_cache_source_enabled,
            latency_model_state_file=config.latency_model_state_file,
            residual_learning_rate=config.residual_learning_rate,
            residual_update_error_floor_ms=config.residual_update_error_floor_ms,
            residual_update_error_ratio=config.residual_update_error_ratio,
            residual_idle_decay_start_ms=config.residual_idle_decay_start_ms,
            residual_idle_decay_end_ms=config.residual_idle_decay_end_ms,
        )
        if not self._execution_geometry_enabled:
            self._initialize_agent()
        logger.info(
            "enabled NaviSched VLLM adapter: biz_name=%s server=%s:%d "
            "agent_grpc_port=%d block_size=%d total_blocks=%d "
            "linear_step=%d hybrid_tail_required=%s cache_specs=%s "
            "pd_separated=%s latency_prediction_mode=%s "
            "native_scheduler=true prefill_service_interval=true "
            "max_concurrent_batches=%d "
            "batch_timeout_ms=%d latency_fit_batch_size=%d "
            "latency_fit_interval=%d latency_learning_rate=%g queue_learning_rate=%g "
            "latency_cold_start_fit_epochs=%d latency_history_limit=%d "
            "latency_persist_interval=%d latency_persist_interval_ms=%d "
            "latency_warmup_sample_count=%d "
            "latency_validation_enabled=%s latency_model_state_file=%s",
            config.biz_name,
            config.server_ip,
            config.server_port,
            config.gig_port,
            self._block_size,
            self._cache_policy.total_blocks,
            self._cache_policy.linear_step,
            str(self._cache_policy.hybrid_tail_required).lower(),
            ",".join(self._cache_policy.spec_names),
            str(self._native_ingress).lower(),
            "split" if self._native_ingress else "unified",
            max_concurrent_batches,
            config.batch_timeout_ms,
            config.latency_fit_batch_size,
            config.latency_fit_interval,
            config.latency_learning_rate,
            config.queue_learning_rate,
            config.latency_cold_start_fit_epochs,
            config.latency_history_limit,
            config.latency_persist_interval,
            config.latency_persist_interval_ms,
            config.latency_warmup_sample_count,
            str(config.latency_validation_enabled).lower(),
            config.latency_model_state_file or "<empty>",
        )
        logger.info(
            "NaviSched latency model fingerprint: %s",
            latency_model_fingerprint,
        )
        logger.info(
            "NaviSched VLLM native scheduler config: "
            "enable_chunked_prefill=%s max_num_partial_prefills=%s "
            "max_long_partial_prefills=%s long_prefill_token_threshold=%d "
            "max_num_scheduled_tokens=%d",
            str(scheduler_config.enable_chunked_prefill).lower(),
            getattr(scheduler_config, "max_num_partial_prefills", None),
            getattr(scheduler_config, "max_long_partial_prefills", None),
            scheduler_config.long_prefill_token_threshold,
            scheduler_config.max_num_scheduled_tokens
            or scheduler_config.max_num_batched_tokens,
        )

    def _initialize_agent(self, scheduler=None) -> None:
        if self._agent is not None:
            return
        options = dict(self._agent_options)
        if self._native_ingress:
            options["execution_geometry_enabled"] = self._execution_geometry_enabled
        if self._execution_geometry_enabled:
            from .VLLMExecutionGeometry import resolved_prefill_execution_policy

            policy = resolved_prefill_execution_policy(
                scheduler, self._vllm_config, self._cache_policy
            )
            fingerprint = _native_pd_latency_model_fingerprint(
                _latency_model_fingerprint(self._vllm_config),
                self._cache_policy,
                policy,
                execution_geometry_enabled=True,
                projected_execution_geometry_enabled=False,
                execution_cache_source_enabled=bool(
                    options.get("execution_cache_source_enabled", False)
                ),
            )
            native_policy = naviSchedModule().TPUPrefillExecutionPolicyConfig()
            for name, value in policy.items():
                setattr(native_policy, name, value)
            self._prefill_execution_policy = native_policy
            options["prefill_execution_policy"] = native_policy
            options["latency_model_fingerprint"] = fingerprint
            options["latency_history_fingerprint"] = _latency_history_fingerprint(
                self._vllm_config, fingerprint
            )
            options["execution_geometry_enabled"] = True
            options["projected_execution_geometry_enabled"] = False
            logger.info("NaviSched resolved latency model fingerprint: %s", fingerprint)
        options["peer_cache_enabled"] = (
            self._native_ingress and _v6d_peer_cache_enabled(scheduler)
        )
        logger.info(
            "NaviSched peer cache capability: enabled=%s", options["peer_cache_enabled"]
        )
        self._agent = NaviSchedAgent(**options)
        self._agent.init()
        self._bridge = naviSchedModule()._VLLMAgentBridge(
            self._agent, pd_separated=self._native_ingress
        )
        if not self._bridge.enable_forwarding():
            self.shutdown()
            raise RuntimeError("failed to enable NaviSched Agent forwarding")

    def bind_engine_core(self, engine_core) -> None:
        from vllm.v1.serial_utils import MsgpackEncoder

        if self._engine_core is not None:
            raise RuntimeError("NaviSched VLLM observer is already bound")
        if self._native_ingress and engine_core.batch_queue:
            raise RuntimeError("cannot bind NaviSched to a nonempty vLLM batch queue")
        if self._native_ingress:
            logger.info(
                "NaviSched native engine step loop: %s",
                getattr(engine_core.step_fn, "__qualname__", type(engine_core.step_fn).__name__),
            )
        self._engine_core = engine_core
        _execution_timing_probe.bind_engine_loop_audit(self, engine_core)
        if self._native_ingress:
            from .VLLMExecutionGeometry import capture_execution_policy

            logger.info(
                "NaviSched resolved execution policy: %s",
                json.dumps(
                    capture_execution_policy(
                        getattr(engine_core, "scheduler", None),
                        getattr(engine_core, "vllm_config", None),
                    ),
                    sort_keys=True,
                ),
            )
        self._bind_v6d_cache_source(engine_core)
        self._response_encoder = MsgpackEncoder()
        engine_core.output_queue = _RoutingOutputQueue(engine_core.output_queue, self)
        self._forward_thread = threading.Thread(
            target=self._forward_batches,
            name="VLLMNaviSchedForward",
            daemon=True,
        )
        self._forward_thread.start()

    def _bind_v6d_cache_source(self, engine_core) -> None:
        if not self._native_ingress:
            return
        module = sys.modules.get("kvs_connector.v6d_object_connector")
        if module is None:
            return
        connectors = [engine_core.scheduler.connector]
        while connectors:
            connector = connectors.pop()
            if isinstance(connector, module.V6dObjectConnector):
                source = connector.scheduler
                config = source.vllm_config.cache_config
                speculative = source.vllm_config.speculative_config
                eagle = speculative is not None and speculative.use_eagle()
                hybrid = bool(source.mamba_group_ids)
                align = hybrid and config.mamba_cache_mode == "align"
                retention = config.prefix_cache_retention_interval
                self._agent.report_cache_status(
                    -1,
                    source.hash_block_size,
                    source.lcm_block_size // source.hash_block_size,
                    hybrid,
                    cache_key_type=1,
                    full_attention_query_drop_tokens=int(eagle),
                    mamba_replay_stop_shift=bool(align and eagle),
                    mamba_retention_interval=(
                        -1 if retention is None else int(retention)
                    ),
                    mamba_align=align,
                )
                self._v6d_scheduler = source
                return
            connectors.extend(getattr(connector, "_connectors", ()))

    def _report_v6d_publication_failure(self, reporter, groups, request_id) -> None:
        full, tail = set(), set()
        for group_id, hashes in groups.items():
            target = (
                full if self._cache_policy.group_roles[group_id] == "full" else tail
            )
            target.update(self._wire_event_hashes(group_id, tuple(hashes)))
        if not reporter(sorted(full), sorted(tail), 1):
            raise RuntimeError("failed to retire NaviSched V6D publication keys")
        self.on_request_lifecycle_event(
            {
                "event": "cache_publication_failed",
                "event_time_us": time.time_ns() // 1000,
                "request_id": request_id,
                "cache_key_type": 1,
                "failed_full_hashes": sorted(full),
                "failed_tail_hashes": sorted(tail),
            }
        )

    def _report_v6d_job_failure(self, job, groups) -> None:
        reported = getattr(job, "_navi_sched_reported_publication_failures", None)
        if reported is None:
            reported = job._navi_sched_reported_publication_failures = {}
        for reporter, interested, request_id in job._navi_sched_publication_reporter:
            prior = reported.setdefault(request_id, {})
            failed = {
                group: set(hashes)
                .intersection(interested.get(group, ()))
                .difference(prior.get(group, ()))
                for group, hashes in groups.items()
            }
            failed = {group: hashes for group, hashes in failed.items() if hashes}
            if failed:
                self._report_v6d_publication_failure(reporter, failed, request_id)
                for group, hashes in failed.items():
                    prior.setdefault(group, set()).update(hashes)

    def _attach_v6d_publication_reporters(self, reporters) -> None:
        source = self._v6d_scheduler
        if not self._v6d_publication_restore:
            original_store = source._get_reqs_to_store

            def get_reqs_to_store(scheduler_output):
                result = original_store(scheduler_output)
                # r50 coalesces allocation across requests. Attribute missing
                # candidates after it assigns shared objects to their owner.
                for request_id, groups in source._navi_sched_store_candidates.items():
                    pending_hashes = source.navi_sched_inflight_hashes(request_id)
                    for group_id, candidates in groups.items():
                        missing = set(candidates).difference(
                            pending_hashes.get(group_id, ())
                        )
                        if missing:
                            self._v6d_unpublished_candidates.setdefault(
                                str(request_id), {}
                            ).setdefault(group_id, set()).update(missing)
                for job, failed in source.navi_sched_take_allocation_failures():
                    if hasattr(job, "_navi_sched_publication_reporter"):
                        self._report_v6d_job_failure(job, failed)
                return result

            source._get_reqs_to_store = get_reqs_to_store
            self._v6d_publication_restore.append(
                (source, "_get_reqs_to_store", original_store)
            )
            original_drain = source._drain_publications

            def drain_publications():
                completed = [
                    (future, job)
                    for future, job in source._publications.items()
                    if future.done()
                ]
                result = original_drain()
                for future, job in completed:
                    status, _duration, _finished_at = future.result()
                    if status in ("error", "nan"):
                        self._report_v6d_job_failure(job, job.hashes)
                return result

            source._drain_publications = drain_publications
            self._v6d_publication_restore.append(
                (source, "_drain_publications", original_drain)
            )
            original_update = source.update_connector_output

            def update_connector_output(connector_output):
                output = connector_output
                metadata = getattr(output, "kv_connector_worker_meta", None)
                failed = []
                if metadata is not None:
                    for job_id, results in metadata.store_results.items():
                        job = source._store_jobs.get(job_id)
                        if job is None:
                            continue
                        state = source._request_states[job.req_id]
                        if (
                            job.req_id in metadata.nan_failed_reqs
                            or state.nan_store_failed
                            or any(error is not None for error in results.values())
                            or any(error is not None for error in job.results.values())
                        ):
                            failed.append(job)
                result = original_update(output)
                for job in failed:
                    self._report_v6d_job_failure(job, job.hashes)
                return result

            source.update_connector_output = update_connector_output
            self._v6d_publication_restore.append(
                (source, "update_connector_output", original_update)
            )
            for group_id, manager in source.managers.items():
                original_seal = manager.seal

                def seal(
                    block_hashes,
                    request_id=None,
                    *,
                    _group=group_id,
                    _seal=original_seal,
                ):
                    hashes = tuple(block_hashes)
                    result = _seal(hashes, request_id=request_id)
                    if not result:
                        for job in tuple(source._store_jobs.values()):
                            if job.req_id == request_id:
                                failed_hashes = set(hashes).intersection(
                                    job.hashes.get(_group, ())
                                )
                                if failed_hashes:
                                    self._report_v6d_job_failure(
                                        job, {_group: failed_hashes}
                                    )
                    return result

                manager.seal = seal
                self._v6d_publication_restore.append((manager, "seal", original_seal))
        for job in source._store_jobs.values():
            if (
                not hasattr(job, "_navi_sched_publication_reporter")
                and str(job.req_id) in reporters
            ):
                job._navi_sched_publication_reporter = [
                    (
                        reporters[str(job.req_id)].make_cache_publication_reporter(),
                        source.navi_sched_job_hashes(job),
                        str(job.req_id),
                    )
                ]
        for request_id in tuple(self._v6d_unpublished_candidates):
            if request_id not in reporters and request_id not in source._request_states:
                self._v6d_unpublished_candidates.pop(request_id)
                continue
            if request_id in reporters:
                groups = self._v6d_unpublished_candidates.pop(request_id)
                reporter = reporters[request_id].make_cache_publication_reporter()
                for job in source._store_jobs.values():
                    shared = {
                        group: set(hashes).intersection(source.navi_sched_job_hashes(job).get(group, ()))
                        for group, hashes in groups.items()
                    }
                    shared = {
                        group: hashes for group, hashes in shared.items() if hashes
                    }
                    if shared:
                        job._navi_sched_publication_reporter.append(
                            (reporter, shared, request_id)
                        )
                        for group, hashes in shared.items():
                            groups[group].difference_update(hashes)
                groups = {group: hashes for group, hashes in groups.items() if hashes}
                if groups:
                    self._report_v6d_publication_failure(reporter, groups, request_id)

    def _restore_v6d_publication_observers(self) -> None:
        for owner, name, original in reversed(self._v6d_publication_restore):
            setattr(owner, name, original)
        self._v6d_publication_restore.clear()
        self._v6d_unpublished_candidates.clear()

    def _poll_v6d_cache_source(self) -> None:
        if self._forward_stop.is_set() or self._v6d_scheduler is None:
            return
        future = self._v6d_snapshot_future
        if future is not None:
            if not future.done():
                return
            self._v6d_snapshot_future = None
            try:
                self.on_v6d_cache_snapshots(future.result())
            except Exception:
                # Observation failure is not an upstream circuit-state update.
                logger.exception("failed to read V6D resident cache snapshot")
        now = time.monotonic_ns()
        if now >= self._next_v6d_snapshot_poll_ns:
            self._next_v6d_snapshot_poll_ns = now + _CACHE_SNAPSHOT_INTERVAL_NS
            try:
                self._v6d_snapshot_future = (
                    self._v6d_scheduler.submit_local_cache_snapshots()
                )
            except Exception:
                # Keep the last state explicitly supplied by the upstream cache.
                logger.exception("failed to submit V6D resident cache snapshot")

    def _log_native_forward_wait(self) -> None:
        now_ns = time.perf_counter_ns()
        if now_ns < getattr(self, "_next_forward_wait_log_ns", 0):
            return
        self._next_forward_wait_log_ns = now_ns + 5_000_000_000
        try:
            state = next(iter(self._scheduled_forward_batches.values()), None)
            if state is None:
                state = next(iter(self._pending_forward_batches.values()), None)
        except RuntimeError:
            # The engine loop may move a batch while this diagnostic samples it.
            return
        if state is None:
            values = (-1, 0, 0, 0, 0, False, False)
        else:
            with state.lock:
                values = (
                    state.batch_id,
                    len(state.requests),
                    len(state.completed_prefill_request_ids),
                    len(state.cancelled_request_ids),
                    state.pending_prefill_steps,
                    state.prefill_complete,
                    state.released,
                )
        with self._pd_output_lock:
            ready_finishes = len(self._pd_ready_finishes)
            completed_outputs = len(self._pd_output_completed_ns)
        _execution_timing_probe._diagnostic_format_info(
            logger,
            "[navi-vllm-forward-slot-wait] batch_id=%d requests=%d "
            "completed_prefills=%d cancelled=%d pending_steps=%d "
            "prefill_complete=%s released=%s ready_finishes=%d completed_outputs=%d",
            *values,
            ready_finishes,
            completed_outputs,
        )

    def _forward_batches(self) -> None:
        if _read_bool_env("VLLM_NAVI_SCHED_DEFER_RENDER", False):
            asyncio.run(self._forward_render_batches())
            return
        import msgspec
        from vllm.v1.engine import EngineCoreRequest, EngineCoreRequestType
        from vllm.v1.serial_utils import MsgpackDecoder

        decoder = MsgpackDecoder(forward_request_type(EngineCoreRequest))
        while True:
            self._poll_v6d_cache_source()
            if not self._drain_cancelled_requests(EngineCoreRequestType):
                if self._forward_stop.wait(0.1):
                    return
                continue
            stopping = self._forward_stop.is_set()
            try:
                native_batch = self._bridge.take_next_forward_batch(
                    0 if stopping else 10
                )
            except Exception:
                if not stopping:
                    logger.exception("failed to receive NaviSched VLLM forwarded batch")
                    self._forward_stop.wait(0.1)
                    continue
                return
            if native_batch is None:
                if self._forward_stop.is_set():
                    return
                with self._cache_touch_lock:
                    touch_pending_schedule = bool(self._cache_touch_queue)
                if not touch_pending_schedule:
                    self._drain_forward_request_touches(
                        decoder, msgspec.msgpack, max_count=1
                    )
                continue
            if self._native_ingress:
                # The PD bridge transfers pending touches with the ready batch
                # instead of dropping them. Prepare these before decoding so
                # vLLM can reuse and promote the preprocessed requests.
                self._prepare_forward_request_touches(
                    getattr(native_batch, "touches", ()), decoder, msgspec.msgpack
                )
            # The stop flag may change while the native queue wait releases the GIL.
            stopping = self._forward_stop.is_set()
            python_take_ns = time.monotonic_ns()
            if self._discard_cancelled_forward_requests(native_batch):
                continue
            native_ready_ns = int(getattr(native_batch, "ready_time_ns", 0))
            admission_started_ns = (
                native_ready_ns
                if 0 < native_ready_ns <= python_take_ns
                else python_take_ns
            )
            preceding_workload = self._capture_arrival_workload()
            ready_ns = time.time_ns()
            image_predictions = {}
            try:
                decoded = self._decode_forward_batch(
                    native_batch,
                    decoder,
                    msgspec.msgpack,
                    image_predictions=image_predictions,
                )
            except Exception:
                self._release_native_forward_batch(native_batch)
                logger.exception("failed to decode NaviSched VLLM forwarded batch")
                continue
            if stopping:
                self._fail_forward_responses(
                    native_batch.request_ids, "NaviSched vLLM Agent is stopping"
                )
                self._discard_cancelled_before_decode(native_batch.request_ids)
                self._release_native_forward_batch(native_batch)
                continue
            decode_done_ns = time.time_ns()
            try:
                self._enqueue_forward_batch(
                    native_batch,
                    decoded,
                    EngineCoreRequestType,
                    preceding_workload,
                    admission_started_ns,
                    image_predictions=image_predictions,
                )
                enqueue_done_ns = time.time_ns()
                _execution_timing_probe._diagnostic_format_info(
                    logger,
                    "NAVI_SCHED_VLLM_AGENT_BATCH batch_id=%d batch_size=%d "
                    "python_take_us=%.3f decode_us=%.3f enqueue_us=%.3f "
                    "ready_to_queue_us=%.3f",
                    native_batch.batch_id,
                    len(decoded),
                    (
                        (python_take_ns - native_ready_ns) / 1000.0
                        if native_ready_ns > 0
                        else -1.0
                    ),
                    (decode_done_ns - ready_ns) / 1000.0,
                    (enqueue_done_ns - decode_done_ns) / 1000.0,
                    (enqueue_done_ns - ready_ns) / 1000.0,
                )
            except Exception:
                logger.exception("failed to submit NaviSched VLLM forwarded batch")

    async def _forward_render_batches(self) -> None:
        import msgspec
        from vllm.v1.engine import EngineCoreRequest, EngineCoreRequestType
        from vllm.v1.serial_utils import MsgpackDecoder

        decoder = MsgpackDecoder(forward_request_type(EngineCoreRequest))
        pending = set()
        try:
            while not self._forward_stop.is_set():
                continuation = self._render_continuation
                log_progress = getattr(continuation, "log_progress", None)
                if log_progress is not None:
                    log_progress()
                if not self._drain_cancelled_requests(EngineCoreRequestType):
                    await asyncio.sleep(0.01)
                    continue
                try:
                    native_batch = self._bridge.take_next_forward_batch(0)
                except Exception:
                    logger.exception("failed to receive NaviSched VLLM forwarded batch")
                    await asyncio.sleep(0.1)
                    continue
                if native_batch is None:
                    self._drain_forward_request_touches(
                        decoder, msgspec.msgpack, max_count=1
                    )
                    await asyncio.sleep(0.001)
                    continue
                if self._discard_cancelled_forward_requests(native_batch):
                    continue
                task = asyncio.create_task(
                    self._prepare_rendered_batch(
                        native_batch, decoder, msgspec.msgpack, EngineCoreRequestType
                    )
                )
                pending.add(task)
                task.add_done_callback(pending.discard)
                await asyncio.sleep(0)
        finally:
            for task in pending:
                task.cancel()
            await asyncio.gather(*pending, return_exceptions=True)
            while True:
                native_batch = self._bridge.take_next_forward_batch(0)
                if native_batch is None:
                    break
                self._fail_forward_responses(
                    native_batch.request_ids, "NaviSched vLLM Agent is stopping"
                )
                self._release_native_forward_batch(native_batch)
            continuation = self._render_continuation
            if continuation is not None:
                self._render_continuation = None
                await continuation.aclose()

    async def _prepare_rendered_batch(
        self, native_batch, decoder, msgpack, request_type
    ):
        owned = True
        started_ns = time.perf_counter_ns()
        native_ready_ns = int(getattr(native_batch, "ready_time_ns", 0))
        native_ready_to_prepare_ms = (
            (time.monotonic_ns() - native_ready_ns) / 1_000_000.0
            if native_ready_ns > 0
            else -1.0
        )
        original_size = len(native_batch.request_ids)
        decoded = []
        image_predictions = {}
        decode_ms = render_ms = enqueue_ms = -1.0
        prepared_ns = 0
        phase = "decode"
        phase_started_ns = started_ns
        status = "failed"
        try:
            preceding_workload = self._capture_arrival_workload()
            decoded = self._decode_forward_batch(
                native_batch,
                decoder,
                msgpack,
                resume_rendering=False,
                image_predictions=image_predictions,
            )
            decode_ms = (time.perf_counter_ns() - started_ns) / 1_000_000.0
            phase = "render"
            phase_started_ns = time.perf_counter_ns()
            deferred = [
                request
                for _, request in decoded
                if getattr(request, "navi_sched_render_call", None) is not None
            ]
            if deferred:
                if self._render_continuation is None:
                    from .VLLMNativePreparation import NativePreparationClient

                    self._render_continuation = NativePreparationClient(
                        self._native_prepare_port
                    )
                results = await self._render_continuation.resume_async(
                    deferred,
                    return_exceptions=True,
                    schedule_batch_id=int(native_batch.schedule_batch_id),
                    local_batch_id=int(native_batch.batch_id),
                )
                render_ms = (time.perf_counter_ns() - phase_started_ns) / 1_000_000.0
                failed = self._render_failure_ids(deferred, results)
                if failed:
                    if len(failed) == len(decoded):
                        raise _MediaPreprocessingError(
                            "all forwarded requests failed native rendering"
                        )
                    if not self._bridge.abort_forward_batch_requests(
                        native_batch, failed
                    ):
                        raise RuntimeError(
                            "failed to isolate native rendering failures"
                        )
                    self._fail_forward_responses(
                        failed,
                        NAVI_SCHED_MEDIA_ERROR_PREFIX
                        + "vLLM media preprocessing failed",
                    )
                    self._discard_cache_touch_requests(failed)
                    self._discard_cancelled_before_decode(failed)
                    decoded = [
                        (key, value) for key, value in decoded if key not in failed
                    ]
            else:
                render_ms = 0.0
            if self._forward_stop.is_set():
                raise asyncio.CancelledError()
            phase = "cancel_filter"
            phase_started_ns = time.perf_counter_ns()
            if self._discard_cancelled_forward_requests(native_batch):
                owned = False
                status = "cancelled_before_enqueue"
                return
            active_ids = set(native_batch.request_ids)
            decoded = [(key, value) for key, value in decoded if key in active_ids]
            prepared_ns = time.perf_counter_ns()
            phase = "enqueue"
            phase_started_ns = prepared_ns
            owned = False
            self._enqueue_forward_batch(
                native_batch,
                decoded,
                request_type,
                preceding_workload,
                started_ns,
                render_ms,
                image_predictions=image_predictions,
            )
            enqueue_ms = (time.perf_counter_ns() - prepared_ns) / 1_000_000.0
            status = "enqueued"
        except asyncio.CancelledError:
            status = "stopping"
            self._fail_forward_responses(
                native_batch.request_ids, "NaviSched vLLM Agent is stopping"
            )
            raise
        except Exception as error:
            self._fail_forward_responses(
                native_batch.request_ids,
                (
                    NAVI_SCHED_MEDIA_ERROR_PREFIX + "vLLM media preprocessing failed"
                    if isinstance(error, _MediaPreprocessingError)
                    else "invalid NaviSched VLLM forward request"
                ),
            )
            logger.exception("failed to prepare NaviSched VLLM forwarded batch")
        finally:
            if owned:
                self._discard_cache_touch_requests(native_batch.request_ids)
                self._discard_cancelled_before_decode(native_batch.request_ids)
                self._release_native_forward_batch(native_batch)
            finished_ns = time.perf_counter_ns()
            logger.info(
                "NAVI_SCHED_VLLM_AGENT_BATCH batch_id=%d batch_size=%d "
                "prepare_ms=%.3f enqueue_ms=%.3f schedule_batch_id=%d "
                "original_size=%d status=%s phase=%s phase_ms=%.3f "
                "native_ready_to_prepare_ms=%.3f decode_ms=%.3f render_ms=%.3f "
                "total_ms=%.3f",
                native_batch.batch_id,
                len(decoded),
                ((prepared_ns - started_ns) / 1_000_000.0 if prepared_ns else -1.0),
                enqueue_ms,
                native_batch.schedule_batch_id,
                original_size,
                status,
                phase,
                (finished_ns - phase_started_ns) / 1_000_000.0,
                native_ready_to_prepare_ms,
                decode_ms,
                render_ms,
                (finished_ns - started_ns) / 1_000_000.0,
            )

    def _discard_cancelled_forward_requests(self, native_batch) -> bool:
        with self._response_lock:
            if not self._cancelled_before_decode:
                return False
            request_ids = tuple(int(value) for value in native_batch.request_ids)
            cancelled = self._cancelled_before_decode.intersection(request_ids)
        if not cancelled:
            return False
        if len(cancelled) == len(request_ids):
            self._discard_cache_touch_requests(request_ids)
            self._discard_cancelled_before_decode(request_ids)
            self._release_native_forward_batch(native_batch)
            return True
        try:
            removed = self._bridge.abort_forward_batch_requests(
                native_batch, sorted(cancelled)
            )
        except Exception:
            logger.exception(
                "failed to retire cancelled requests before batch decode: batch_id=%d",
                native_batch.batch_id,
            )
            removed = False
        if removed:
            self._discard_cache_touch_requests(cancelled)
            self._discard_cancelled_before_decode(cancelled)
        return False

    def _decode_forward_batch(
        self,
        native_batch,
        decoder,
        msgpack,
        *,
        resume_rendering=True,
        image_predictions=None,
    ):
        decoded = []
        try:
            for request_id, payload in zip(
                native_batch.request_ids, native_batch.payloads, strict=True
            ):
                request_id = int(request_id)
                payload = bytes(payload)
                touch = self._preprocessed_cache_touches.get(request_id)
                if touch is not None:
                    if touch.payload != payload:
                        self._discard_cache_touch_requests((request_id,))
                        raise ValueError(
                            "NaviSched cache-touch payload does not match batch payload"
                        )
                    request = touch.engine_request
                else:
                    request = self._decode_forward_request(
                        request_id, payload, decoder, msgpack
                    )
                decoded.append((request_id, request))
                features = getattr(request, "mm_features", None)
                if (
                    image_predictions is not None
                    and features
                    and getattr(request, "navi_sched_render_call", None) is not None
                    and all(
                        f.modality == "image" and f.mm_position.length == 1
                        for f in features
                    )
                ):
                    image_predictions[request_id] = SimpleNamespace(
                        prompt_token_ids=request.prompt_token_ids,
                        mm_features=features,
                    )
            request_ids = tuple(int(value) for value in native_batch.request_ids)
            if len(decoded) != len(request_ids) or len(set(request_ids)) != len(
                request_ids
            ):
                raise ValueError("invalid NaviSched forwarded VLLM batch")
            failed = (
                self._resume_rendered_requests(
                    [request for _, request in decoded], isolate_failures=True
                )
                if resume_rendering
                else []
            )
            if failed:
                if len(failed) == len(decoded):
                    raise _MediaPreprocessingError(
                        "all forwarded requests failed native rendering"
                    )
                if not self._bridge.abort_forward_batch_requests(native_batch, failed):
                    raise RuntimeError("failed to isolate native rendering failures")
                self._fail_forward_responses(
                    failed,
                    NAVI_SCHED_MEDIA_ERROR_PREFIX + "vLLM media preprocessing failed",
                )
                self._discard_cache_touch_requests(failed)
                self._discard_cancelled_before_decode(failed)
                decoded = [(key, value) for key, value in decoded if key not in failed]
        except Exception as error:
            if not resume_rendering:
                raise
            self._fail_forward_responses(
                native_batch.request_ids,
                (
                    NAVI_SCHED_MEDIA_ERROR_PREFIX + "vLLM media preprocessing failed"
                    if isinstance(error, _MediaPreprocessingError)
                    else "invalid NaviSched VLLM forward request"
                ),
            )
            self._discard_cache_touch_requests(native_batch.request_ids)
            self._discard_cancelled_before_decode(native_batch.request_ids)
            raise
        return decoded

    def _resume_rendered_requests(self, requests, *, isolate_failures=False):
        deferred = [
            request
            for request in requests
            if getattr(request, "navi_sched_render_call", None) is not None
        ]
        if not deferred:
            return []
        if self._render_continuation is None:
            from .VLLMDeferredRenderer import NativeRenderContinuation

            self._render_continuation = NativeRenderContinuation(
                self._render_config, self._engine_core.get_supported_tasks()
            )
        if not isolate_failures:
            self._render_continuation.resume_sync(deferred)
            return []
        results = self._render_continuation.resume_sync(
            deferred, return_exceptions=True
        )
        return self._render_failure_ids(deferred, results)

    @staticmethod
    def _render_failure_ids(deferred, results):
        failed = []
        for request, result in zip(deferred, results, strict=True):
            if isinstance(result, BaseException):
                failed.append(int(request.request_id))
                logger.error(
                    "NaviSched native media rendering failed: request_id=%s error_type=%s",
                    request.request_id,
                    type(result).__name__,
                    exc_info=(type(result), result, result.__traceback__),
                )
        return failed

    @staticmethod
    def _decode_forward_request(request_id, payload, decoder, msgpack):
        frames = msgpack.decode(payload)
        request = decoder.decode(frames)
        request.request_id = str(request_id)
        request.client_index = request_id
        return request

    def _drain_forward_request_touches(
        self, decoder, msgpack, max_count: int = 0
    ) -> None:
        engine_core = self._engine_core
        if engine_core is None:
            return
        try:
            touches = self._bridge.take_forward_request_touches(max_count)
        except Exception:
            logger.exception("failed to receive NaviSched VLLM cache-touch requests")
            return
        self._prepare_forward_request_touches(touches, decoder, msgpack)

    def _prepare_forward_request_touches(self, touches, decoder, msgpack) -> None:
        engine_core = self._engine_core
        if engine_core is None:
            return
        for native_touch in touches:
            request_id = int(native_touch.request_id)
            payload = bytes(native_touch.payload)
            if request_id in self._preprocessed_cache_touches:
                logger.error(
                    "duplicate NaviSched VLLM cache-touch request: request_id=%d",
                    request_id,
                )
                continue
            try:
                engine_request = self._decode_forward_request(
                    request_id, payload, decoder, msgpack
                )
                if getattr(engine_request, "navi_sched_render_call", None) is not None:
                    continue
                preprocessed = engine_core.preprocess_add_request(engine_request)
            except Exception:
                logger.exception(
                    "failed to preprocess NaviSched VLLM cache-touch request: "
                    "request_id=%d",
                    request_id,
                )
                continue
            touch = _PreprocessedCacheTouch(
                payload=payload,
                engine_request=engine_request,
                preprocessed_request=preprocessed,
                accepted_time_ns=int(native_touch.accepted_time_ns),
            )
            self._preprocessed_cache_touches[request_id] = touch
            with self._cache_touch_lock:
                self._cache_touch_queue.append(
                    (request_id, preprocessed[0], touch.accepted_time_ns)
                )

    def _promote_arrived_cache_touches(self, scheduler) -> None:
        if self._native_ingress and self._v6d_scheduler is not None:
            self._attach_v6d_publication_reporters({})
        with self._cache_touch_lock:
            if not self._cache_touch_queue:
                return
            touches = tuple(self._cache_touch_queue)
            self._cache_touch_queue.clear()

        promoted = 0
        max_lag_ms = 0.0
        processing_started_ns = time.monotonic_ns()
        now_ns = time.monotonic_ns()
        for request_id, request, accepted_time_ns in touches:
            try:
                cache_manager = scheduler.kv_cache_manager
                cached_blocks, _, _ = cache_manager.get_computed_blocks(request)
                blocks = [block for group in cached_blocks.blocks for block in group]
                if blocks:
                    # vLLM 0.29 exposes cache lookup and block-pool touch/free,
                    # rather than a KVCacheManager promotion method. Releasing
                    # the temporary references moves free cached blocks to the
                    # back of the eviction queue without pinning them.
                    cache_manager.block_pool.touch(blocks)
                    cache_manager.block_pool.free_blocks(blocks)
                promoted += len(blocks)
            except Exception:
                logger.exception(
                    "failed to promote NaviSched VLLM cache-touch request: "
                    "request_id=%d",
                    request_id,
                )
                continue
            if 0 < accepted_time_ns <= now_ns:
                max_lag_ms = max(max_lag_ms, (now_ns - accepted_time_ns) / 1_000_000.0)
        processing_ms = (time.monotonic_ns() - processing_started_ns) / 1_000_000.0

        self._cache_touch_count += len(touches)
        self._cache_touch_promoted_blocks += promoted
        if now_ns - self._cache_touch_last_log_ns >= 10_000_000_000:
            self._bridge.log_cache_touch_audit(
                self._cache_touch_count,
                self._cache_touch_promoted_blocks,
                len(touches),
                max_lag_ms,
                processing_ms,
            )
            self._cache_touch_count = 0
            self._cache_touch_promoted_blocks = 0
            self._cache_touch_last_log_ns = now_ns

    def _discard_cache_touch_requests(self, request_ids) -> None:
        discarded = {int(value) for value in request_ids}
        if not discarded:
            return
        for request_id in discarded:
            self._preprocessed_cache_touches.pop(request_id, None)
        with self._cache_touch_lock:
            self._cache_touch_queue = deque(
                value for value in self._cache_touch_queue if value[0] not in discarded
            )

    @staticmethod
    def _predicted_cache_hits(native_batch) -> dict[int, int]:
        request_ids = tuple(int(value) for value in native_batch.request_ids)
        cache_hits = tuple(
            int(value) for value in native_batch.predicted_cache_hit_token_counts
        )
        if len(cache_hits) != len(request_ids) or any(
            value < 0 for value in cache_hits
        ):
            raise ValueError("invalid NaviSched predicted cache-hit metadata")
        return dict(zip(request_ids, cache_hits, strict=True))

    def _fail_forward_responses(self, request_ids, error_message: str) -> None:
        for request_id in request_ids:
            try:
                if not self._bridge.fail_forward_response(
                    int(request_id), error_message
                ):
                    logger.warning(
                        "failed to publish terminal NaviSched response error: "
                        "request_id=%s",
                        request_id,
                    )
            except Exception:
                logger.exception(
                    "failed to publish terminal NaviSched response error: "
                    "request_id=%s",
                    request_id,
                )

    def _enqueue_forward_batch(
        self,
        native_batch,
        decoded,
        request_type,
        preceding_workload=None,
        admission_started_ns=0,
        native_preparation_ms=0.0,
        image_predictions=None,
    ) -> None:
        engine_core = self._engine_core
        if engine_core is None:
            raise RuntimeError("NaviSched VLLM observer is not bound")

        try:
            request_ids = tuple(request_id for request_id, _ in decoded)
            predicted_cache_hits = self._predicted_cache_hits(native_batch)
            for request_id in request_ids:
                if (
                    request_id in self._pending_forward_batches
                    or request_id in self._scheduled_forward_batches
                ):
                    raise ValueError("duplicate NaviSched forwarded request id")
            cancelled_request_ids = self._register_response_requests(decoded)
        except Exception:
            self._fail_forward_responses(
                native_batch.request_ids,
                "failed to register NaviSched VLLM response route",
            )
            self._discard_cancelled_before_decode(native_batch.request_ids)
            self._release_native_forward_batch(native_batch)
            raise

        state = _ForwardBatchState(
            batch_id=int(native_batch.batch_id),
            schedule_batch_id=int(native_batch.schedule_batch_id),
            schedule_batch_size=int(native_batch.schedule_batch_size),
            pending_cache_tracked=bool(native_batch.pending_cache_tracked),
            native_batch=native_batch,
            request_ids=request_ids,
            external_request_ids={
                request_id: request.request_id for request_id, request in decoded
            },
            requests={},
            native_preparation_ms=native_preparation_ms,
            predicted_cache_hits=predicted_cache_hits,
            cancelled_request_ids=set(),
            arrived_request_ids=set(cancelled_request_ids),
        )

        if cancelled_request_ids:
            self._abort_forward_requests(state, set(cancelled_request_ids))
            if state.released:
                return

        scheduler = getattr(engine_core, "scheduler", None)
        unsupported_request_ids = {
            request_id
            for request_id, request in decoded
            if request_id not in cancelled_request_ids
            and scheduler is not None
            and getattr(request, "mm_features", None)
            and not getattr(scheduler, "supports_mm_inputs", False)
        }
        if unsupported_request_ids:
            reason = "multimodal request is unsupported by target model"
            logger.warning(
                "rejecting unsupported multimodal requests: batch_id=%d "
                "request_ids=%s",
                state.batch_id,
                sorted(unsupported_request_ids),
            )
            self._fail_forward_responses(unsupported_request_ids, reason)
            state.arrived_request_ids.update(unsupported_request_ids)
            self._abort_forward_requests(state, unsupported_request_ids)
            for request_id in unsupported_request_ids:
                self._preprocessed_cache_touches.pop(request_id, None)
            if state.released:
                return
            if state.tracking_aborted:
                remaining = (
                    set(request_ids)
                    - set(cancelled_request_ids)
                    - unsupported_request_ids
                )
                self._fail_forward_responses(
                    remaining, "failed to isolate unsupported multimodal request"
                )
                self._remove_response_requests(remaining)
                self._release_forward_batch(state)
                return

        excluded_request_ids = set(cancelled_request_ids) | unsupported_request_ids
        try:
            processed = []
            for request_id, request in decoded:
                if request_id in excluded_request_ids:
                    self._preprocessed_cache_touches.pop(request_id, None)
                    continue
                touch = self._preprocessed_cache_touches.pop(request_id, None)
                if touch is not None and touch.engine_request is request:
                    preprocessed = touch.preprocessed_request
                else:
                    preprocessed = engine_core.preprocess_add_request(request)
                processed.append((request_id, preprocessed))
        except Exception:
            self._fail_forward_responses(
                (
                    request_id
                    for request_id in request_ids
                    if request_id not in excluded_request_ids
                ),
                "vLLM request preprocessing failed",
            )
            self._remove_response_requests(request_ids)
            self._release_forward_batch(state)
            raise

        if not processed:
            self._remove_response_requests(request_ids)
            self._release_forward_batch(state)
            return

        self._register_image_cache_mappings(native_batch, processed, image_predictions)

        if preceding_workload is None:
            preceding_workload = self._capture_arrival_workload()
        try:
            self._set_admission_state(
                state,
                [value[0] for _, value in processed],
                preceding_workload,
                admission_started_ns,
            )
        except Exception:
            self._fail_forward_responses(request_ids, "vLLM admission tracking failed")
            self._remove_response_requests(request_ids)
            self._release_forward_batch(state)
            raise

        for request_id, _ in processed:
            self._pending_forward_batches[request_id] = state

        self._report_prepared_workload("queued_admission", state)
        submitted_request_ids = []
        try:
            for request_id, value in processed:
                engine_core.input_queue.put_nowait((request_type.ADD, value))
                submitted_request_ids.append(request_id)
        except Exception:
            submitted = set(submitted_request_ids)
            with state.lock:
                state.submission_failed = True
                state.tracking_aborted = True
                state.cancelled_request_ids.update(state.request_ids)
                state.arrived_request_ids.update(set(request_ids) - submitted)
                self._remove_prepared_workload(state, state.request_ids)
            self._release_forward_reporter(state)
            self._fail_forward_responses(
                (request_id for request_id, _ in processed),
                "vLLM request submission failed",
            )
            self._remove_response_requests(set(request_ids) - submitted)
            self._deactivate_response_requests(submitted)
            self._deliver_engine_aborts(
                (
                    state.external_request_ids[request_id]
                    for request_id in submitted_request_ids
                ),
                request_type,
            )
            if not submitted_request_ids:
                self._release_forward_batch(state)
            self._report_prepared_workload("submission_failed")
            raise

    def _register_image_cache_mappings(
        self, native_batch, processed, image_predictions=None
    ):
        if not image_predictions:
            return
        for request_id, value in processed:
            prediction = image_predictions.get(request_id)
            if prediction is None:
                continue
            try:
                logical_hashes = native_batch.get_request_block_hashes(request_id)
                with self._cache_lifecycle_lock:
                    with self._response_lock:
                        if int(request_id) not in self._response_request_ids:
                            continue
                        delta = self._image_cache_map.register(
                            int(request_id),
                            prediction,
                            value[0],
                            self._block_size,
                            logical_hashes=logical_hashes,
                        )
                        self._response_image_positions[int(request_id)] = tuple(
                            (feature.mm_position.offset, feature.mm_position.length)
                            for feature in value[0].mm_features
                        )
                    self._report_cache_delta(delta)
            except Exception:
                logger.exception(
                    "failed to register scheduling image cache mapping: request_id=%s",
                    request_id,
                )

    def _register_response_requests(self, decoded) -> set[int]:
        cancelled = set()
        with self._response_lock:
            for request_id, _ in decoded:
                if (
                    request_id in self._response_request_ids
                    or request_id in self._inactive_response_request_ids
                ):
                    raise ValueError("duplicate NaviSched VLLM response request id")
            for request_id, request in decoded:
                self._response_request_ids[request_id] = request.request_id
                if request_id in self._cancelled_before_decode:
                    self._cancelled_before_decode.discard(request_id)
                    cancelled.add(request_id)
        return cancelled

    def _remove_response_requests(self, request_ids) -> None:
        request_ids = tuple(request_ids)
        image_cache_map = getattr(self, "_image_cache_map", None)
        with (
            self._cache_lifecycle_lock if image_cache_map is not None else nullcontext()
        ):
            with self._response_lock:
                for request_id in request_ids:
                    request_id = int(request_id)
                    if image_cache_map is not None:
                        image_cache_map.finish(request_id)
                    self._response_request_ids.pop(request_id, None)
                    self._response_image_positions.pop(request_id, None)
                    self._inactive_response_request_ids.discard(request_id)
                    self._cancelled_before_decode.discard(request_id)

    def _discard_cancelled_before_decode(self, request_ids) -> None:
        with self._response_lock:
            for request_id in request_ids:
                self._cancelled_before_decode.discard(int(request_id))

    def _deactivate_response_requests(self, request_ids) -> None:
        with self._response_lock:
            for request_id in request_ids:
                request_id = int(request_id)
                if request_id in self._response_request_ids:
                    self._inactive_response_request_ids.add(request_id)

    def _abort_engine_requests(self, external_request_ids, request_type) -> None:
        request_ids = list(dict.fromkeys(external_request_ids))
        engine_core = self._engine_core
        if not request_ids or engine_core is None:
            return
        engine_core.aborts_queue.put_nowait(request_ids)
        engine_core.input_queue.put_nowait((request_type.ABORT, request_ids))

    def _deliver_engine_aborts(self, external_request_ids, request_type) -> bool:
        request_ids = list(dict.fromkeys(external_request_ids))
        if not request_ids:
            return True
        try:
            self._abort_engine_requests(request_ids, request_type)
        except Exception:
            with self._response_lock:
                self._pending_engine_abort_request_ids = list(
                    dict.fromkeys(self._pending_engine_abort_request_ids + request_ids)
                )
            logger.exception(
                "failed to deliver cancelled NaviSched VLLM requests: "
                "request_ids=%s",
                request_ids,
            )
            return False
        return True

    def _drain_cancelled_requests(self, request_type) -> bool:
        try:
            request_ids = tuple(
                int(value) for value in self._bridge.take_cancelled_requests()
            )
        except Exception:
            logger.exception("failed to receive cancelled NaviSched VLLM requests")
            return False
        self._discard_cache_touch_requests(request_ids)
        tracked_cancellations = {}
        with self._response_lock:
            external_request_ids = self._pending_engine_abort_request_ids
            self._pending_engine_abort_request_ids = []
            for request_id in request_ids:
                external_request_id = self._response_request_ids.get(request_id)
                if external_request_id is None:
                    self._cancelled_before_decode.add(request_id)
                else:
                    self._inactive_response_request_ids.add(request_id)
                    external_request_ids.append(external_request_id)
                    state = self._pending_forward_batches.get(
                        request_id
                    ) or self._scheduled_forward_batches.get(request_id)
                    if state is not None:
                        entry = tracked_cancellations.setdefault(
                            id(state), (state, set())
                        )
                        entry[1].add(request_id)
        for state, cancelled_request_ids in tracked_cancellations.values():
            try:
                self._abort_forward_requests(state, cancelled_request_ids)
            except Exception:
                logger.exception(
                    "failed to retire cancelled NaviSched tracking: "
                    "batch_id=%d request_ids=%s",
                    state.batch_id,
                    sorted(cancelled_request_ids),
                )
        if tracked_cancellations:
            self._report_prepared_workload("queued_cancellation")
        return self._deliver_engine_aborts(external_request_ids, request_type)

    def route_engine_outputs(self, client_index, outputs) -> bool:
        request_id = int(client_index)
        with self._response_lock:
            external_request_id = self._response_request_ids.get(request_id)
            active = request_id not in self._inactive_response_request_ids
            image_positions = self._response_image_positions.get(request_id)
        if external_request_id is None:
            return False
        response_request_ids = {str(value) for value in outputs.finished_requests or ()}
        response_request_ids.update(str(value.request_id) for value in outputs.outputs)
        if response_request_ids - {external_request_id}:
            logger.error(
                "vLLM mixed requests in NaviSched response: request_id=%d "
                "external_request_id=%s actual_request_ids=%s",
                request_id,
                external_request_id,
                sorted(response_request_ids),
            )
            active = False
        terminal = external_request_id in set(outputs.finished_requests or ()) or any(
            output.request_id == external_request_id and output.finished
            for output in outputs.outputs
        )
        if not active:
            self._deactivate_response_requests((request_id,))
            if terminal:
                self._remove_response_requests((request_id,))
            else:
                from vllm.v1.engine import EngineCoreRequestType

                self._deliver_engine_aborts(
                    (external_request_id,), EngineCoreRequestType
                )
            return True
        try:
            if self._response_encoder is None:
                raise RuntimeError("NaviSched VLLM response encoder is not initialized")
            response_outputs = outputs
            if image_positions:

                def project_prefix(count):
                    removed = 0
                    for offset, length in image_positions:
                        if count <= offset:
                            break
                        consumed = min(count - offset, length)
                        removed += consumed - int(consumed == length)
                    return count - removed

                response_outputs = copy.copy(outputs)
                response_outputs.outputs = []
                for value in outputs.outputs:
                    projected = copy.copy(value)
                    cached = value.num_cached_tokens
                    projected.num_cached_tokens = project_prefix(cached)
                    projected.num_external_computed_tokens = (
                        projected.num_cached_tokens
                        - project_prefix(cached - value.num_external_computed_tokens)
                    )
                    response_outputs.outputs.append(projected)
            frames = [
                bytes(value)
                for value in self._response_encoder.encode(
                    forward_outputs_fields(response_outputs)
                )
            ]
            published = self._bridge.publish_forward_response(
                request_id, frames, terminal
            )
        except Exception:
            logger.exception(
                "failed to publish NaviSched VLLM output: request_id=%d",
                request_id,
            )
            try:
                self._bridge.fail_forward_response(
                    request_id, "failed to encode vLLM response"
                )
            except Exception:
                logger.exception(
                    "failed to publish NaviSched VLLM response error: request_id=%d",
                    request_id,
                )
            published = False
        if terminal:
            self._remove_response_requests((request_id,))
        if not published:
            self._deactivate_response_requests((request_id,))
            logger.warning(
                "NaviSched VLLM response pipe rejected output: "
                "request_id=%d terminal=%s",
                request_id,
                terminal,
            )
            if not terminal:
                from vllm.v1.engine import EngineCoreRequestType

                self._deliver_engine_aborts(
                    (external_request_id,), EngineCoreRequestType
                )
        return True

    def create_scheduler(self, scheduler_cls, **kwargs):
        try:
            scheduler = scheduler_cls(**kwargs)
            if self._native_ingress:
                coordinator = scheduler.kv_cache_manager.coordinator
                actual_drop = get_vllm_full_attention_query_drop_tokens(
                    coordinator, self._block_size
                )
                if actual_drop != self._cache_policy.full_attention_query_drop_tokens:
                    raise ValueError(
                        "vLLM Full query policy differs from actual coordinator"
                    )
                logger.info(
                    "NaviSched actual Full query policy: full_attention_query_drop_tokens=%d groups=%s",
                    actual_drop,
                    [
                        (type(group.spec).__name__, group.group_ids, group.use_eagle)
                        for group in getattr(coordinator, "attention_groups", ())
                    ],
                )
            self._initialize_agent(scheduler)
            scheduler.kv_cache_manager.block_pool.set_block_lifecycle_listener(
                self.on_block_lifecycle_event
            )
            lifecycle_setter = getattr(
                scheduler, "set_request_lifecycle_listener", None
            )
            if lifecycle_setter is not None:
                lifecycle_setter(self.on_request_lifecycle_event)
            config = getattr(scheduler, "scheduler_config", None)
            self.on_request_lifecycle_event(
                {
                    "event": "scheduler_config",
                    "event_time_us": time.time_ns() // 1000,
                    "lifecycle_supported": lifecycle_setter is not None,
                    "scheduler_class": type(scheduler).__qualname__,
                    **{
                        name: getattr(config, name, None)
                        for name in (
                            "max_num_seqs",
                            "max_num_batched_tokens",
                            "max_num_scheduled_tokens",
                            "enable_chunked_prefill",
                            "long_prefill_token_threshold",
                            "max_num_partial_prefills",
                            "max_long_partial_prefills",
                            "policy",
                            "async_scheduling",
                        )
                    },
                }
            )
            wrapped = _NaviSchedScheduler(scheduler, self)
            self._refresh_and_report_workload(scheduler, (), "initialize")
            return wrapped
        except BaseException:
            self.shutdown()
            raise

    def _admit_request(self, request):
        external_request_id = str(request.request_id)
        request_id = self._navi_request_id(request)
        state = self._pending_forward_batches.get(request_id)
        if state is None:
            self._reject_unowned_request(request)
            return None

        cancelled_before_admission = False
        admitted_state = None
        try:
            with state.lock:
                if state.released:
                    raise RuntimeError("NaviSched forwarded batch was already released")
                if state.external_request_ids.get(request_id) != external_request_id:
                    raise ValueError(
                        "vLLM external request id does not match NaviSched payload"
                    )
                if request_id in state.arrived_request_ids:
                    raise ValueError("duplicate vLLM request in NaviSched batch")
                state.arrived_request_ids.add(request_id)
                request._navi_forward_batch_id = state.batch_id
                request._navi_plan_trace_request_id = request_id
                request._navi_plan_trace_batch_id = state.batch_id
                if request_id not in state.cancelled_request_ids:
                    state.requests[request_id] = request
                    self._forward_request_states[external_request_id] = state
                else:
                    cancelled_before_admission = True
                if len(state.arrived_request_ids) == len(state.request_ids):
                    for value in state.request_ids:
                        if value not in state.cancelled_request_ids:
                            self._scheduled_forward_batches[value] = state
                    if not state.requests:
                        self._release_forward_batch(state)
                    else:
                        admitted_state = state
        except Exception:
            self._release_forward_batch(state)
            raise
        if cancelled_before_admission:
            self._remove_response_requests((request_id,))
        return admitted_state

    def _reject_unowned_request(self, request) -> None:
        external_request_id = str(request.request_id)
        request_id = self._navi_request_id(request)
        logger.error(
            "rejecting vLLM request without NaviSched forwarded batch metadata: %r",
            external_request_id,
        )
        self._fail_forward_responses(
            (request_id,), "vLLM request has no NaviSched forwarded batch metadata"
        )
        self._remove_response_requests((request_id,))

    def _record_admission_state(self, state, scheduler) -> None:
        with state.lock:
            if state.admission_started_ns > 0:
                return
        with self._workload_lock:
            preceding_workload = self._synchronize_workload_locked(scheduler, ())
        if preceding_workload is None:
            logger.error(
                "skipping invalid NaviSched vLLM admission snapshot: batch_id=%d",
                state.batch_id,
            )
            return
        self._set_admission_state(
            state,
            list(state.requests.values()),
            preceding_workload,
            time.perf_counter_ns(),
        )

    def _set_admission_state(
        self,
        state,
        requests,
        preceding_workload: _ObservedWorkload,
        admission_started_ns: int,
    ) -> None:
        incoming_predicted_cache_hit_tokens = 0
        incoming_compute_tokens = 0
        incoming_compute_token_square_sum = 0.0
        incoming_hit_compute_token_product_sum = 0.0
        incoming_total_prompt_tokens = 0
        incoming_running_capacity_request_ids = set()
        running_capacity_slot_count = preceding_workload.running_capacity_slot_count
        admission_contributions = {}
        for request in requests:
            request_id = self._navi_request_id(request)
            if (
                running_capacity_slot_count
                < preceding_workload.max_running_request_count
            ):
                incoming_running_capacity_request_ids.add(request_id)
                running_capacity_slot_count += 1
            prompt_tokens = max(0, int(getattr(request, "num_prompt_tokens", 0)))
            cache_hit_tokens = min(
                prompt_tokens,
                max(0, state.predicted_cache_hits.get(request_id, 0)),
            )
            compute_tokens = prompt_tokens - cache_hit_tokens
            incoming_predicted_cache_hit_tokens += cache_hit_tokens
            incoming_compute_tokens += compute_tokens
            incoming_compute_token_square_sum += float(compute_tokens**2)
            incoming_hit_compute_token_product_sum += float(
                cache_hit_tokens * compute_tokens
            )
            incoming_total_prompt_tokens += prompt_tokens
            admission_contributions[request_id] = _WorkloadContribution(
                batch_id=state.batch_id,
                prompt_tokens=prompt_tokens,
                prefill_context_tokens=(cache_hit_tokens if compute_tokens > 0 else 0),
                remaining_prefill_tokens=compute_tokens,
                total_context_tokens=cache_hit_tokens,
                remaining_total_tokens=compute_tokens,
            )

        with self._execution_lock:
            active_executions = self._active_execution_count
        if admission_started_ns <= 0:
            admission_started_ns = time.perf_counter_ns()
        admission_state = _AdmissionState(
            preceding_workload=preceding_workload,
            incoming_request_count=len(requests),
            incoming_predicted_cache_hit_tokens=(incoming_predicted_cache_hit_tokens),
            incoming_compute_tokens=incoming_compute_tokens,
            incoming_compute_token_square_sum=(incoming_compute_token_square_sum),
            incoming_hit_compute_token_product_sum=(
                incoming_hit_compute_token_product_sum
            ),
            incoming_total_prompt_tokens=incoming_total_prompt_tokens,
            incoming_running_capacity_request_ids=frozenset(
                incoming_running_capacity_request_ids
            ),
            active_executions=active_executions,
        )
        recorded = False
        with state.lock:
            if state.admission_started_ns == 0:
                state.admission_started_ns = admission_started_ns
                state.admission_state = admission_state
                state.admission_contributions = admission_contributions
                if not state.released and not state.submission_failed:
                    with self._prepared_workload_lock:
                        for request_id, contribution in admission_contributions.items():
                            if request_id not in state.cancelled_request_ids:
                                previous = self._prepared_contributions.get(request_id)
                                if previous is not None:
                                    self._prepared_workload.apply(previous[1], -1)
                                self._prepared_contributions[request_id] = (
                                    state,
                                    contribution,
                                )
                                self._prepared_workload.apply(contribution, 1)
                        self._native_handoff_batches[state.batch_id] = None
                recorded = True
        if recorded:
            _execution_timing_probe._diagnostic_format_info(
                logger,
                "[navi-vllm-admission] batch_id=%d batch_size=%d "
                "predicted_cache_hit_tokens=%d compute_tokens=%d "
                "compute_token_square_sum=%.0f "
                "hit_compute_token_product_sum=%.0f total_prompt_tokens=%d "
                "preceding_requests=%d preceding_prefill_requests=%d "
                "preceding_prefill_context_tokens=%d "
                "preceding_prefill_tokens=%d "
                "preceding_total_context_tokens=%d "
                "preceding_total_tokens=%d "
                "preceding_running_decode_requests=%d "
                "preceding_running_decode_generated_tokens=%d "
                "preceding_running_decode_remaining_token_budget=%d "
                "inflight_scheduled_tokens=%d",
                state.batch_id,
                admission_state.incoming_request_count,
                admission_state.incoming_predicted_cache_hit_tokens,
                admission_state.incoming_compute_tokens,
                admission_state.incoming_compute_token_square_sum,
                admission_state.incoming_hit_compute_token_product_sum,
                admission_state.incoming_total_prompt_tokens,
                preceding_workload.request_count,
                preceding_workload.remaining_prefill_request_count,
                preceding_workload.remaining_prefill_context_tokens,
                preceding_workload.remaining_prefill_tokens,
                preceding_workload.remaining_total_context_tokens,
                preceding_workload.remaining_total_tokens,
                preceding_workload.running_decode_request_count,
                preceding_workload.running_decode_generated_tokens,
                preceding_workload.running_decode_remaining_token_budget,
                preceding_workload.inflight_scheduled_tokens,
            )

    def _remove_prepared_workload(self, state, request_ids) -> None:
        with self._prepared_workload_lock:
            for request_id in request_ids:
                entry = self._prepared_contributions.get(request_id)
                if entry is not None and entry[0] is state:
                    self._prepared_contributions.pop(request_id)
                    self._prepared_workload.apply(entry[1], -1)
            if not self._prepared_contributions:
                self._prepared_workload = _WorkloadAggregate()

    def _workload_with_pending_admissions_locked(
        self, base: _ObservedWorkload
    ) -> _ObservedWorkload:
        aggregate = _WorkloadAggregate.from_snapshot(base)
        running_capacity = _PrefillWorkloadSliceAggregate.from_snapshot(
            base.running_capacity_prefill
        )
        waiting_capacity = _PrefillWorkloadSliceAggregate.from_snapshot(
            base.waiting_capacity_prefill
        )
        scheduler_running = _PrefillWorkloadSliceAggregate.from_snapshot(
            base.scheduler_running_prefill
        )
        scheduler_waiting = _PrefillWorkloadSliceAggregate.from_snapshot(
            base.scheduler_waiting_prefill
        )
        running_capacity_slot_count = base.running_capacity_slot_count
        prepared_running = _PrefillWorkloadSliceAggregate()
        with self._prepared_workload_lock:
            native_handoff_batch_ids = tuple(self._native_handoff_batches)
            pending = self._prepared_workload
            pending_request_count = pending.request_count
            pending_waiting_request_ids = tuple(self._prepared_contributions)
            for name in _WorkloadAggregate.__dataclass_fields__:
                if name != "batch_request_counts":
                    setattr(
                        aggregate,
                        name,
                        getattr(aggregate, name) + getattr(pending, name),
                    )
            slots = max(0, base.max_running_request_count - running_capacity_slot_count)
            for _, contribution in itertools.islice(
                self._prepared_contributions.values(), slots
            ):
                prepared_running.apply(contribution)
                running_capacity_slot_count += 1
            pending_prefill = _PrefillWorkloadSliceAggregate(
                request_count=pending.remaining_prefill_request_count,
                context_tokens=pending.remaining_prefill_context_tokens,
                compute_tokens=pending.remaining_prefill_tokens,
                compute_token_square_sum=pending.remaining_prefill_token_square_sum,
                context_compute_token_product_sum=(
                    pending.context_remaining_prefill_token_product_sum
                ),
            )
        for name in _PrefillWorkloadSliceAggregate.__dataclass_fields__:
            running_value = getattr(prepared_running, name)
            pending_value = getattr(pending_prefill, name)
            setattr(
                running_capacity,
                name,
                getattr(running_capacity, name) + running_value,
            )
            setattr(
                waiting_capacity,
                name,
                getattr(waiting_capacity, name) + pending_value - running_value,
            )
            setattr(
                scheduler_waiting,
                name,
                getattr(scheduler_waiting, name) + pending_value,
            )
        workload = aggregate.snapshot(
            base.running_request_count,
            base.running_elapsed_time_ms,
            base.running_decode_request_count,
            base.running_decode_generated_tokens,
            base.running_decode_remaining_token_budget,
            base.observed_time_ns,
            base.waiting_request_count + pending_request_count,
            base.inflight_scheduled_tokens,
            base.max_running_request_count,
            running_capacity_slot_count,
            running_capacity.snapshot(),
            waiting_capacity.snapshot(),
            scheduler_running.snapshot(),
            scheduler_waiting.snapshot(),
        )
        if workload.valid():
            return replace(
                workload,
                native_handoff_batch_ids=native_handoff_batch_ids,
                waiting_request_ids=tuple(
                    sorted(
                        set(base.waiting_request_ids or ())
                        | set(pending_waiting_request_ids)
                    )
                ),
            )
        logger.error(
            "invalid NaviSched vLLM workload with pending admissions; "
            "using scheduler state: %r",
            workload,
        )
        return base

    def _capture_arrival_workload(self) -> _ObservedWorkload:
        with self._workload_lock:
            return self._workload_with_pending_admissions_locked(
                self._advance_running_elapsed(
                    self._current_workload or _ObservedWorkload()
                )
            )

    @staticmethod
    def _advance_running_elapsed(workload: _ObservedWorkload) -> _ObservedWorkload:
        now_ns = time.perf_counter_ns()
        if workload.observed_time_ns <= 0:
            return replace(workload, observed_time_ns=now_ns)
        elapsed_ns = max(0, now_ns - workload.observed_time_ns)
        return replace(
            workload,
            running_elapsed_time_ms=(
                workload.running_elapsed_time_ms
                + elapsed_ns
                * workload.scheduler_running_prefill.request_count
                / 1_000_000.0
            ),
            observed_time_ns=now_ns,
        )

    @staticmethod
    def _prefill_workload_features(workload: _ObservedWorkload):
        return (
            workload.batch_count,
            workload.remaining_prefill_request_count,
            workload.remaining_prefill_context_tokens,
            workload.remaining_prefill_tokens,
            workload.remaining_prefill_token_square_sum,
            workload.context_remaining_prefill_token_product_sum,
        )

    @staticmethod
    def _capacity_sample_features(admission_state, request_stats):
        preceding = admission_state.preceding_workload
        running_capacity = _PrefillWorkloadSliceAggregate.from_snapshot(
            preceding.running_capacity_prefill
        )
        waiting_capacity = _PrefillWorkloadSliceAggregate.from_snapshot(
            preceding.waiting_capacity_prefill
        )
        for request_id, total_tokens, actual_cache_hit_tokens in request_stats:
            total_tokens = max(0, int(total_tokens))
            actual_cache_hit_tokens = min(
                total_tokens, max(0, int(actual_cache_hit_tokens))
            )
            target = (
                running_capacity
                if request_id in admission_state.incoming_running_capacity_request_ids
                else waiting_capacity
            )
            target.apply_request(
                actual_cache_hit_tokens,
                total_tokens - actual_cache_hit_tokens,
            )
        return running_capacity.snapshot(), waiting_capacity.snapshot()

    @staticmethod
    def _scheduler_sample_features(admission_state, request_stats):
        preceding = admission_state.preceding_workload
        running = _PrefillWorkloadSliceAggregate.from_snapshot(
            preceding.scheduler_running_prefill
        )
        waiting = _PrefillWorkloadSliceAggregate.from_snapshot(
            preceding.scheduler_waiting_prefill
        )
        for _, total_tokens, actual_cache_hit_tokens in request_stats:
            total_tokens = max(0, int(total_tokens))
            actual_cache_hit_tokens = min(
                total_tokens, max(0, int(actual_cache_hit_tokens))
            )
            waiting.apply_request(
                actual_cache_hit_tokens,
                total_tokens - actual_cache_hit_tokens,
            )
        return running.snapshot(), waiting.snapshot()

    def _predicted_cache_hit_for_request(self, request) -> int:
        state = self._forward_request_state(request.request_id)
        if state is None:
            return 0
        request_id = self._navi_request_id(request)
        return max(0, state.predicted_cache_hits.get(request_id, 0))

    def _make_workload_contribution(
        self, request, inflight_scheduled_tokens: int
    ) -> _WorkloadContribution:
        prompt_tokens = max(0, int(getattr(request, "num_prompt_tokens", 0)))
        computed_tokens = max(
            0,
            int(getattr(request, "num_computed_tokens", 0))
            - max(0, inflight_scheduled_tokens),
        )
        computed_tokens = min(computed_tokens, prompt_tokens)
        if int(getattr(request, "num_cached_tokens", -1)) < 0:
            computed_tokens = max(
                computed_tokens,
                min(prompt_tokens, self._predicted_cache_hit_for_request(request)),
            )
        remaining_prefill_tokens = max(0, prompt_tokens - computed_tokens)
        state = self._forward_request_state(request.request_id)
        return _WorkloadContribution(
            batch_id=(
                state.batch_id if state is not None else self._navi_request_id(request)
            ),
            prompt_tokens=prompt_tokens,
            prefill_context_tokens=(
                computed_tokens if remaining_prefill_tokens > 0 else 0
            ),
            remaining_prefill_tokens=remaining_prefill_tokens,
            total_context_tokens=(
                computed_tokens if remaining_prefill_tokens > 0 else 0
            ),
            remaining_total_tokens=remaining_prefill_tokens,
        )

    def _refresh_workload_contributions(
        self, scheduler, request_ids, inflight_scheduled_tokens
    ) -> None:
        request_ids = set(request_ids)
        if not request_ids:
            return
        for request_id in request_ids:
            previous = self._workload_contributions.pop(request_id, None)
            if previous is not None:
                self._workload_aggregate.apply(previous, -1)
            request = scheduler.requests.get(request_id)
            if request is None:
                continue
            contribution = self._make_workload_contribution(
                request, inflight_scheduled_tokens.get(request_id, 0)
            )
            self._workload_contributions[request_id] = contribution
            self._workload_aggregate.apply(contribution, 1)

    def _capture_execution_workload_state(self, request_ids):
        with self._execution_lock:
            dirty_request_ids = set(self._workload_dirty_request_ids)
            self._workload_dirty_request_ids = set()
            return (
                set(request_ids) | dirty_request_ids,
                dict(self._inflight_scheduled_tokens),
                self._active_scheduled_tokens,
            )

    def _partition_prefill_workload(self, scheduler):
        try:
            running = tuple(scheduler.running)
        except (AttributeError, TypeError):
            running = ()
        try:
            waiting = tuple(scheduler.waiting)
        except (AttributeError, TypeError):
            waiting = ()
        configured_capacity = max(0, int(getattr(scheduler, "max_num_running_reqs", 0)))
        max_running_request_count = max(configured_capacity, len(running))
        running_capacity = _PrefillWorkloadSliceAggregate()
        waiting_capacity = _PrefillWorkloadSliceAggregate()
        scheduler_running = _PrefillWorkloadSliceAggregate()
        scheduler_waiting = _PrefillWorkloadSliceAggregate()
        running_capacity_slot_count = 0
        seen_request_ids = set()

        def append(request_or_id, running_now=False):
            nonlocal running_capacity_slot_count
            request_id = str(getattr(request_or_id, "request_id", request_or_id))
            if request_id in seen_request_ids:
                return
            seen_request_ids.add(request_id)
            contribution = self._workload_contributions.get(request_id)
            if contribution is not None:
                if running_now:
                    scheduler_running.apply(contribution)
                else:
                    scheduler_waiting.apply(contribution)
            if running_now or running_capacity_slot_count < max_running_request_count:
                if contribution is not None:
                    running_capacity.apply(contribution)
                running_capacity_slot_count += 1
            elif contribution is not None:
                waiting_capacity.apply(contribution)

        for request in running:
            append(request, True)
        for request in waiting:
            append(request)
        for request_id in getattr(scheduler, "requests", {}):
            append(request_id)
        return (
            max_running_request_count,
            running_capacity_slot_count,
            running_capacity.snapshot(),
            waiting_capacity.snapshot(),
            scheduler_running.snapshot(),
            scheduler_waiting.snapshot(),
        )

    def _workload_snapshot(
        self, scheduler, inflight_scheduled_tokens, active_scheduled_tokens
    ) -> _ObservedWorkload:
        requests = getattr(scheduler, "requests", {})
        if not isinstance(requests, dict):
            requests = {}
        if set(self._workload_contributions) != set(requests):
            logger.warning(
                "rebuilding NaviSched vLLM workload state: tracked=%d actual=%d",
                len(self._workload_contributions),
                len(requests),
            )
            self._workload_contributions.clear()
            self._workload_aggregate = _WorkloadAggregate()
            self._refresh_workload_contributions(
                scheduler, requests.keys(), inflight_scheduled_tokens
            )
        try:
            running = tuple(scheduler.running)
        except (AttributeError, TypeError):
            running = ()
        running_request_count = len(running)
        now_ns = time.perf_counter_ns()
        self._audit_stalled_running_requests(
            scheduler, running, inflight_scheduled_tokens, now_ns
        )
        running_prefill_request_ids = set()
        running_decode_request_count = 0
        running_decode_generated_tokens = 0
        running_decode_remaining_token_budget = 0
        for request in running:
            output_tokens = max(0, int(getattr(request, "num_output_tokens", 0)))
            running_decode_generated_tokens += output_tokens
            if output_tokens > 0:
                running_decode_request_count += 1
                max_tokens = max(0, int(getattr(request, "max_tokens", 0)))
                remaining_budget = max(0, max_tokens - output_tokens)
                running_decode_remaining_token_budget = min(
                    (1 << 63) - 1,
                    running_decode_remaining_token_budget + remaining_budget,
                )
            request_id = str(getattr(request, "request_id", request))
            contribution = self._workload_contributions.get(request_id)
            if contribution is not None and contribution.remaining_prefill_tokens > 0:
                running_prefill_request_ids.add(request_id)

        self._running_prefill_started_ns = {
            request_id: self._running_prefill_started_ns.get(request_id, now_ns)
            for request_id in running_prefill_request_ids
        }
        running_elapsed_time_ms = sum(
            max(0, now_ns - started_ns) / 1_000_000.0
            for started_ns in self._running_prefill_started_ns.values()
        )
        try:
            waiting = tuple(scheduler.waiting)
        except (AttributeError, TypeError):
            waiting = ()
        waiting_request_count = len(waiting)
        waiting_request_ids = set()
        for value in waiting:
            request = (
                value if hasattr(value, "request_id") else requests.get(str(value))
            )
            if request is not None:
                waiting_request_ids.add(self._navi_request_id(request))
        (
            max_running_request_count,
            running_capacity_slot_count,
            running_capacity_prefill,
            waiting_capacity_prefill,
            scheduler_running_prefill,
            scheduler_waiting_prefill,
        ) = self._partition_prefill_workload(scheduler)
        workload = self._workload_aggregate.snapshot(
            running_request_count,
            running_elapsed_time_ms,
            running_decode_request_count,
            running_decode_generated_tokens,
            running_decode_remaining_token_budget,
            now_ns,
            waiting_request_count,
            active_scheduled_tokens,
            max_running_request_count,
            running_capacity_slot_count,
            running_capacity_prefill,
            waiting_capacity_prefill,
            scheduler_running_prefill,
            scheduler_waiting_prefill,
        )
        return replace(workload, waiting_request_ids=tuple(sorted(waiting_request_ids)))

    def _audit_stalled_running_requests(
        self, scheduler, running, inflight_scheduled_tokens, now_ns
    ):
        if (
            self._last_stalled_running_scan_ns
            and now_ns - self._last_stalled_running_scan_ns < _STALLED_RUNNING_SCAN_NS
        ):
            return
        self._last_stalled_running_scan_ns = now_ns
        active_ids = set()
        stalled = []
        for running_index, request in enumerate(running):
            request_id = str(request.request_id)
            active_ids.add(request_id)
            computed = int(getattr(request, "num_computed_tokens", 0))
            tokens_with_spec = int(getattr(request, "num_tokens_with_spec", 0))
            placeholders = int(getattr(request, "num_output_placeholders", 0))
            progress = (computed, tokens_with_spec, placeholders)
            previous = self._running_request_progress.get(request_id)
            if previous is None or previous[0] != progress:
                self._running_request_progress[request_id] = (progress, now_ns, 0)
                continue
            _, last_progress_ns, last_log_ns = previous
            if now_ns - last_progress_ns < _STALLED_RUNNING_THRESHOLD_NS or (
                last_log_ns and now_ns - last_log_ns < _STALLED_RUNNING_LOG_INTERVAL_NS
            ):
                continue
            stalled.append(
                (last_log_ns, last_progress_ns, running_index, request, progress)
            )
        for last_log_ns, last_progress_ns, running_index, request, progress in sorted(
            stalled, key=lambda entry: (entry[0], entry[1])
        )[:_STALLED_RUNNING_MAX_LOGS_PER_SCAN]:
            request_id = str(request.request_id)
            computed, tokens_with_spec, placeholders = progress
            state = self._forward_request_state(request_id)
            logger.warning(
                "[navi-vllm-stalled-running] request_id=%s batch_id=%d "
                "unchanged_ms=%.0f computed=%d prompt=%d tokens_with_spec=%d "
                "placeholders=%d next_tokens=%d inflight_tokens=%d "
                "max_tokens=%s max_model_len=%s "
                "preemptions=%d running=%d waiting=%d token_budget=%s "
                "pause_state=%s running_index=%d status=%s cached_tokens=%s "
                "model_headroom=%s has_encoder=%s mamba_align=%s "
                "tracked_remaining_prefill=%s resumable=%s "
                "prompt_id_len=%s encoder_inputs=%s encoder_budget=%s",
                request_id,
                state.batch_id if state is not None else 0,
                (now_ns - last_progress_ns) / 1_000_000,
                computed,
                int(getattr(request, "num_prompt_tokens", 0)),
                tokens_with_spec,
                placeholders,
                tokens_with_spec + placeholders - computed,
                int(inflight_scheduled_tokens.get(request_id, 0)),
                getattr(request, "max_tokens", None),
                getattr(scheduler, "max_model_len", None),
                int(getattr(request, "num_preemptions", 0)),
                len(running),
                len(getattr(scheduler, "waiting", ())),
                getattr(scheduler, "max_num_scheduled_tokens", None),
                getattr(scheduler, "_pause_state", None),
                running_index,
                getattr(request, "status", None),
                getattr(request, "num_cached_tokens", None),
                (getattr(scheduler, "max_model_len", 0) - 1 - computed),
                getattr(request, "has_encoder_inputs", None),
                getattr(scheduler, "need_mamba_block_aligned_split", None),
                getattr(
                    self._workload_contributions.get(request_id),
                    "remaining_prefill_tokens",
                    None,
                ),
                getattr(request, "resumable", None),
                (
                    len(request.prompt_token_ids)
                    if getattr(request, "prompt_token_ids", None) is not None
                    else None
                ),
                getattr(request, "num_encoder_inputs", None),
                getattr(scheduler, "max_num_encoder_input_tokens", None),
            )
            self._running_request_progress[request_id] = (
                progress,
                last_progress_ns,
                now_ns,
            )
        for request_id in self._running_request_progress.keys() - active_ids:
            del self._running_request_progress[request_id]

    def _report_core_step_stage(self, stage, elapsed_ms):
        self._bridge.report_core_step(elapsed_ms, stage != "engine_step")

    def _log_slow_engine_stage(self, stage, started_ns, finished_ns, **details):
        _execution_timing_probe.engine_stage_event(
            self, stage, started_ns, finished_ns, details
        )
        elapsed_ns = finished_ns - started_ns
        if elapsed_ns < _SLOW_ENGINE_STAGE_NS:
            return
        last_log_ns = self._slow_engine_stage_last_log_ns.get(stage, 0)
        if (
            last_log_ns
            and finished_ns - last_log_ns < _SLOW_ENGINE_STAGE_LOG_INTERVAL_NS
            and elapsed_ns < _SEVERE_ENGINE_STAGE_NS
        ):
            return
        self._slow_engine_stage_last_log_ns[stage] = finished_ns
        logger.warning(
            "[navi-vllm-slow-engine-stage] stage=%s elapsed_ms=%.3f %s",
            stage,
            elapsed_ns / 1_000_000,
            " ".join(f"{key}={value}" for key, value in details.items()),
        )

    def _synchronize_workload_locked(self, scheduler, request_ids):
        (
            changed_request_ids,
            inflight_scheduled_tokens,
            active_scheduled_tokens,
        ) = self._capture_execution_workload_state(request_ids)
        self._refresh_workload_contributions(
            scheduler, changed_request_ids, inflight_scheduled_tokens
        )
        workload = self._workload_snapshot(
            scheduler, inflight_scheduled_tokens, active_scheduled_tokens
        )
        if workload.valid():
            self._current_workload = workload
            return workload

        logger.error(
            "invalid NaviSched vLLM workload; rebuilding from scheduler: %r",
            workload,
        )
        self._workload_contributions.clear()
        self._workload_aggregate = _WorkloadAggregate()
        self._refresh_workload_contributions(
            scheduler, scheduler.requests.keys(), inflight_scheduled_tokens
        )
        workload = self._workload_snapshot(
            scheduler, inflight_scheduled_tokens, active_scheduled_tokens
        )
        if workload.valid():
            self._current_workload = workload
            return workload
        logger.error(
            "rebuilt NaviSched vLLM workload remains invalid; " "skipping report: %r",
            workload,
        )
        return None

    def _report_prepared_workload(self, reason, admitted_state=None) -> None:
        with self._workload_report_lock:
            workload = self._capture_arrival_workload()
            if self._report_workload_locked(workload, reason, admitted_state):
                if admitted_state is not None:
                    with admitted_state.lock:
                        admitted_state.admission_reported = True

    def _refresh_and_report_workload(
        self,
        scheduler,
        request_ids,
        reason,
        admitted_state=None,
    ) -> None:
        with self._workload_report_lock:
            with self._workload_lock:
                workload = self._synchronize_workload_locked(scheduler, request_ids)
                if workload is None:
                    return
                if admitted_state is not None:
                    self._remove_prepared_workload(
                        admitted_state, admitted_state.request_ids
                    )
                    for request_id in admitted_state.request_ids:
                        if (
                            self._pending_forward_batches.get(request_id)
                            is admitted_state
                        ):
                            self._pending_forward_batches.pop(request_id, None)
                reported_workload = self._workload_with_pending_admissions_locked(
                    workload
                )
            pending_admissions = self._pending_admission_states(admitted_state)
            if pending_admissions:
                for state in pending_admissions:
                    if self._report_workload_locked(reported_workload, reason, state):
                        with state.lock:
                            state.admission_reported = True
                return
            self._report_workload_locked(reported_workload, reason)

    def _pending_admission_states(self, preferred_state=None):
        candidates = []
        if preferred_state is not None:
            candidates.append(preferred_state)
        candidates.extend(list(self._scheduled_forward_batches.values()))
        pending = []
        seen = set()
        for state in candidates:
            state_key = id(state)
            if state_key in seen:
                continue
            seen.add(state_key)
            with state.lock:
                if state.admission_reported:
                    continue
                if state is not preferred_state and (
                    not state.admission_ready
                    or state.released
                    or state.tracking_aborted
                ):
                    continue
                pending.append(state)
        return pending

    def _report_workload_locked(self, workload, reason, admitted_state=None) -> bool:
        workload = self._advance_running_elapsed(workload)
        admitted_schedule_batch_id = 0
        admitted_local_batch_id = 0
        admitted_request_count = 0
        admitted_pending_cache_tracked = False
        admitted_request_ids = []
        if admitted_state is not None:
            with admitted_state.lock:
                admitted_schedule_batch_id = admitted_state.schedule_batch_id
                admitted_local_batch_id = admitted_state.batch_id
                admitted_request_ids = sorted(
                    request_id
                    for request_id in admitted_state.request_ids
                    if request_id not in admitted_state.cancelled_request_ids
                )
                admitted_request_count = len(admitted_request_ids)
                if (
                    admitted_request_count == admitted_state.schedule_batch_size
                    and admitted_request_count == len(admitted_state.request_ids)
                ):
                    admitted_request_ids = []
                admitted_pending_cache_tracked = admitted_state.pending_cache_tracked
        waiting_request_options = {}
        if (
            self._native_ingress
            and getattr(naviSchedModule(), "VLLM_WAITING_REQUEST_IDS_VERSION", 0) >= 1
        ):
            waiting_request_options["waiting_request_ids"] = list(
                workload.waiting_request_ids or ()
            )
        try:
            reported = self._bridge.report_workload(
                time.time_ns() // 1000,
                workload.batch_count,
                workload.request_count,
                workload.remaining_prefill_request_count,
                workload.running_request_count,
                workload.waiting_request_count,
                workload.remaining_prefill_context_tokens,
                workload.remaining_prefill_tokens,
                workload.remaining_prefill_token_square_sum,
                workload.context_remaining_prefill_token_product_sum,
                workload.max_running_request_count,
                workload.running_capacity_slot_count,
                workload.running_capacity_prefill.request_count,
                workload.running_capacity_prefill.context_tokens,
                workload.running_capacity_prefill.compute_tokens,
                workload.running_capacity_prefill.compute_token_square_sum,
                workload.running_capacity_prefill.context_compute_token_product_sum,
                workload.waiting_capacity_prefill.request_count,
                workload.waiting_capacity_prefill.context_tokens,
                workload.waiting_capacity_prefill.compute_tokens,
                workload.waiting_capacity_prefill.compute_token_square_sum,
                workload.waiting_capacity_prefill.context_compute_token_product_sum,
                workload.remaining_total_context_tokens,
                workload.remaining_total_tokens,
                workload.remaining_total_token_square_sum,
                workload.context_remaining_total_token_product_sum,
                workload.total_prompt_tokens,
                workload.inflight_scheduled_tokens,
                admitted_schedule_batch_id,
                admitted_local_batch_id,
                admitted_request_count,
                admitted_pending_cache_tracked,
                admitted_request_ids,
                workload.running_elapsed_time_ms,
                workload.running_decode_request_count,
                workload.running_decode_generated_tokens,
                workload.running_decode_remaining_token_budget,
                workload.native_handoff_batch_ids,
                **waiting_request_options,
            )
        except Exception:
            logger.exception(
                "failed to report NaviSched vLLM workload: reason=%s "
                "schedule_batch_id=%d local_batch_id=%d",
                reason,
                admitted_schedule_batch_id,
                admitted_local_batch_id,
            )
            return False
        if not reported:
            logger.error(
                "NaviSched rejected vLLM workload: reason=%s "
                "schedule_batch_id=%d local_batch_id=%d",
                reason,
                admitted_schedule_batch_id,
                admitted_local_batch_id,
            )
            return False
        with self._prepared_workload_lock:
            for batch_id in workload.native_handoff_batch_ids:
                self._native_handoff_batches.pop(batch_id, None)
        logger.debug(
            "[navi-vllm-workload] reason=%s batches=%d requests=%d "
            "prefill_requests=%d running_total=%d running_prefill=%d "
            "running_decode=%d waiting=%d "
            "running_elapsed_time_ms=%.3f "
            "running_decode_requests=%d "
            "running_decode_generated_tokens=%d "
            "running_decode_remaining_token_budget=%d "
            "prefill_context_tokens=%d remaining_prefill_tokens=%d "
            "remaining_prefill_token_square_sum=%.0f "
            "context_remaining_prefill_product_sum=%.0f "
            "total_context_tokens=%d remaining_total_tokens=%d "
            "remaining_total_token_square_sum=%.0f "
            "context_remaining_total_product_sum=%.0f "
            "total_prompt_tokens=%d inflight_scheduled_tokens=%d",
            reason,
            workload.batch_count,
            workload.request_count,
            workload.remaining_prefill_request_count,
            workload.running_request_count,
            workload.scheduler_running_prefill.request_count,
            max(
                0,
                workload.running_request_count
                - workload.scheduler_running_prefill.request_count,
            ),
            workload.waiting_request_count,
            workload.running_elapsed_time_ms,
            workload.running_decode_request_count,
            workload.running_decode_generated_tokens,
            workload.running_decode_remaining_token_budget,
            workload.remaining_prefill_context_tokens,
            workload.remaining_prefill_tokens,
            workload.remaining_prefill_token_square_sum,
            workload.context_remaining_prefill_token_product_sum,
            workload.remaining_total_context_tokens,
            workload.remaining_total_tokens,
            workload.remaining_total_token_square_sum,
            workload.context_remaining_total_token_product_sum,
            workload.total_prompt_tokens,
            workload.inflight_scheduled_tokens,
        )
        return True

    def _register_scheduled_work(self, scheduler_output) -> None:
        scheduled_tokens = {
            str(request_id): int(token_count)
            for request_id, token_count in scheduler_output.num_scheduled_tokens.items()
            if int(token_count) > 0
        }
        if not scheduled_tokens:
            return
        output_key = id(scheduler_output)
        with self._execution_lock:
            if output_key in self._scheduled_work_by_output:
                raise RuntimeError("duplicate vLLM scheduler output tracking")
            self._scheduled_work_by_output[output_key] = scheduled_tokens
            for request_id, token_count in scheduled_tokens.items():
                self._inflight_scheduled_tokens[request_id] = (
                    self._inflight_scheduled_tokens.get(request_id, 0) + token_count
                )
                self._active_scheduled_tokens += token_count

    def _settle_scheduled_work(self, scheduler_output) -> None:
        self._settle_scheduled_work_key(id(scheduler_output))

    def _settle_scheduled_work_key(self, output_key: int) -> None:
        inconsistent = []
        aggregate_inconsistent = None
        with self._execution_lock:
            scheduled_tokens = self._scheduled_work_by_output.pop(output_key, None)
            if scheduled_tokens is None:
                return
            active_before = self._active_scheduled_tokens
            completed_tokens = sum(scheduled_tokens.values())
            for request_id, token_count in scheduled_tokens.items():
                current = self._inflight_scheduled_tokens.get(request_id, 0)
                if current < token_count:
                    inconsistent.append((request_id, current, token_count))
                    remaining = 0
                else:
                    remaining = current - token_count
                if remaining > 0:
                    self._inflight_scheduled_tokens[request_id] = remaining
                else:
                    self._inflight_scheduled_tokens.pop(request_id, None)
            if inconsistent or active_before < completed_tokens:
                rebuilt_by_request = {}
                rebuilt_total = 0
                for remaining_output in self._scheduled_work_by_output.values():
                    for request_id, token_count in remaining_output.items():
                        rebuilt_by_request[request_id] = (
                            rebuilt_by_request.get(request_id, 0) + token_count
                        )
                        rebuilt_total += token_count
                aggregate_inconsistent = (
                    active_before,
                    completed_tokens,
                    self._active_scheduled_tokens,
                    rebuilt_total,
                )
                self._inflight_scheduled_tokens = rebuilt_by_request
                self._active_scheduled_tokens = rebuilt_total
            else:
                self._active_scheduled_tokens = active_before - completed_tokens
            self._workload_dirty_request_ids.update(scheduled_tokens)
        for request_id, current, token_count in inconsistent:
            logger.error(
                "inconsistent NaviSched vLLM scheduled-token tracking: "
                "request_id=%s current=%d completed=%d",
                request_id,
                current,
                token_count,
            )
        if aggregate_inconsistent is not None:
            active_before, completed_tokens, tracked_total, rebuilt_total = (
                aggregate_inconsistent
            )
            logger.error(
                "rebuilt inconsistent NaviSched vLLM scheduled-token aggregate: "
                "output_key=%d active_before=%d completed=%d tracked_total=%d "
                "rebuilt_total=%d request_mismatches=%d",
                output_key,
                active_before,
                completed_tokens,
                tracked_total,
                rebuilt_total,
                len(inconsistent),
            )

    def _forward_request_state(self, external_request_id):
        return self._forward_request_states.get(external_request_id)

    @staticmethod
    def _navi_request_id(request) -> int:
        external_request_id = str(request.request_id)
        try:
            canonical_request_id = int(external_request_id)
        except (TypeError, ValueError):
            canonical_request_id = 0
        if (
            canonical_request_id > 0
            and int(getattr(request, "client_index", 0)) == canonical_request_id
        ):
            return canonical_request_id
        return stable_request_id(external_request_id)

    def _forget_forward_request(self, external_request_id):
        self._v6d_unpublished_candidates.pop(external_request_id, None)
        self._forward_request_states.pop(external_request_id, None)

    def _cleanup_finished_forward_requests(self, native_requests):
        for external_request_id, state in list(self._forward_request_states.items()):
            with state.lock:
                if state.released and external_request_id not in native_requests:
                    self._forward_request_states.pop(external_request_id, None)

    def wrap_execute_model(
        self,
        execute_model: Callable[..., Future],
        scheduler: Any,
    ) -> Callable[..., Future]:
        requests = getattr(scheduler, "requests", None)
        if not isinstance(requests, dict):
            raise TypeError("NaviSched requires the vLLM Scheduler request table")
        return partial(self._execute_model, execute_model, scheduler)

    def _execute_model(
        self,
        execute_model,
        scheduler,
        scheduler_output,
        non_block=False,
    ):
        has_tokens = bool(scheduler_output.num_scheduled_tokens)
        execution_step = _ExecutionStep(
            started_ns=time.perf_counter_ns(),
            scheduler_output_key=id(scheduler_output),
        )
        _execution_timing_probe.mark_entry(execution_step)
        with self._execution_lock:
            scheduled_work_registered = (
                execution_step.scheduler_output_key in self._scheduled_work_by_output
            )
        if not scheduled_work_registered:
            self._register_scheduled_work(scheduler_output)
        try:
            self._prepare_execution_step(
                scheduler_output, scheduler.requests.get, execution_step
            )
        except Exception:
            self._abort_execution_step(execution_step)
            logger.exception("failed to start NaviSched vLLM batch")
        try:
            self._start_execution_tracking(execution_step, scheduler, scheduler_output)
            self._record_execution_start_state(execution_step)
        except Exception:
            logger.exception("failed to collect NaviSched vLLM execution state")

        try:
            running = getattr(scheduler, "running", None)
            running_request_count = (
                len(running)
                if running is not None
                else len(scheduler_output.num_scheduled_tokens)
            )
            # Zero-token rounds do not open native execution intervals.
            # Pending scheduler/cache work is not execution training time.
            if self._native_ingress:
                if not has_tokens:
                    _execution_timing_probe.begin_control(execution_step)
                if has_tokens:
                    execution_step.backend_execution_started = bool(
                        _execution_timing_probe.begin(
                            execution_step, self._bridge.begin_backend_execution, 0, running_request_count
                        )
                    )
            else:
                execution_step.backend_execution_started = bool(
                    _execution_timing_probe.begin(
                        execution_step, self._bridge.begin_backend_execution, 0, running_request_count
                    )
                )
            if not execution_step.backend_execution_started and (
                not self._native_ingress or has_tokens
            ):
                logger.error("failed to start vLLM backend execution timing")
        except Exception:
            logger.exception("failed to start vLLM backend execution timing")
        try:
            execution_step.submitted_ns = time.perf_counter_ns()
            future = execute_model(scheduler_output, non_block=non_block)
        except BaseException:
            self._finish_execution_tracking(execution_step)
            self._end_backend_execution(execution_step)
            self._abort_execution_step(execution_step)
            raise

        try:
            future._navi_backend_execution_step = execution_step
            future.add_done_callback(_BatchCompletion(self, execution_step))
        except BaseException:
            self._finish_execution_tracking(execution_step)
            self._end_backend_execution(execution_step)
            self._abort_execution_step(execution_step)
            raise
        return future

    def _start_execution_tracking(
        self, execution_step, scheduler, scheduler_output
    ) -> None:
        running = getattr(scheduler, "running", ()) or ()
        execution_step.running_request_count = len(running)
        waiting = getattr(scheduler, "waiting", ())
        try:
            execution_step.waiting_request_count = len(waiting)
        except TypeError:
            execution_step.waiting_request_count = -1
        execution_step.preempted_request_count = len(
            getattr(scheduler_output, "preempted_req_ids", None) or ()
        )
        cached_requests = getattr(scheduler_output, "scheduled_cached_reqs", None)
        execution_step.resumed_request_count = len(
            getattr(cached_requests, "resumed_req_ids", None) or ()
        )
        execution_step.workload_counts_collected = True
        if execution_step.prefill_tokens > 0:
            running = list(running)
            execution_step.running_prefill_request_count = sum(
                int(getattr(request, "num_computed_tokens", 0))
                <= int(getattr(request, "num_prompt_tokens", 0))
                for request in running
            )
            execution_step.running_decode_request_count = (
                execution_step.running_request_count
                - execution_step.running_prefill_request_count
            )
            execution_step.running_recompute_request_count = sum(
                int(getattr(request, "num_preemptions", 0)) > 0
                and int(getattr(request, "num_computed_tokens", 0))
                <= int(getattr(request, "num_prompt_tokens", 0))
                for request in running
            )

        with self._execution_lock:
            execution_step.active_executions_before = self._active_execution_count
            execution_step.active_prefill_executions_before = (
                self._active_prefill_execution_count
            )
            execution_step.active_decode_executions_before = (
                self._active_decode_execution_count
            )
            current_scheduled_tokens = sum(
                self._scheduled_work_by_output.get(
                    execution_step.scheduler_output_key, {}
                ).values()
            )
            execution_step.active_scheduled_tokens_before = max(
                0, self._active_scheduled_tokens - current_scheduled_tokens
            )
            self._active_execution_count += 1
            self._active_prefill_execution_count += int(
                execution_step.prefill_tokens > 0
            )
            self._active_decode_execution_count += int(execution_step.decode_tokens > 0)
            execution_step.execution_tracking_registered = True

    def _record_execution_start_state(self, execution_step) -> None:
        start_state = _ExecutionStartState(
            active_executions=execution_step.active_executions_before,
            active_prefill_executions=(execution_step.active_prefill_executions_before),
            active_decode_executions=execution_step.active_decode_executions_before,
            active_scheduled_tokens=execution_step.active_scheduled_tokens_before,
            running_requests=execution_step.running_request_count,
            running_prefill_requests=(execution_step.running_prefill_request_count),
            running_decode_requests=execution_step.running_decode_request_count,
            running_recompute_requests=(execution_step.running_recompute_request_count),
            waiting_requests=execution_step.waiting_request_count,
            preempted_requests=execution_step.preempted_request_count,
            resumed_requests=execution_step.resumed_request_count,
            step_prefill_requests=execution_step.prefill_request_count,
            step_prefill_tokens=execution_step.prefill_tokens,
            step_decode_requests=execution_step.decode_request_count,
            step_decode_tokens=execution_step.decode_tokens,
        )
        for progress in execution_step.batch_progress:
            state = progress.state
            if state is None:
                continue
            with state.lock:
                if (
                    state.execution_started_ns == execution_step.started_ns
                    or (self._native_ingress and state.execution_step_count == 0)
                ) and state.execution_start_state is None:
                    state.execution_start_state = start_state

    def _finish_execution_tracking(self, execution_step) -> None:
        inconsistent = []
        with self._execution_lock:
            if execution_step.execution_tracking_registered:
                execution_step.execution_tracking_registered = False
                for name, decrement in (
                    ("_active_execution_count", 1),
                    (
                        "_active_prefill_execution_count",
                        int(execution_step.prefill_tokens > 0),
                    ),
                    (
                        "_active_decode_execution_count",
                        int(execution_step.decode_tokens > 0),
                    ),
                ):
                    current = getattr(self, name)
                    if current < decrement:
                        inconsistent.append((name, current, decrement))
                        setattr(self, name, 0)
                    else:
                        setattr(self, name, current - decrement)
        self._settle_scheduled_work_key(execution_step.scheduler_output_key)
        for name, current, decrement in inconsistent:
            logger.error(
                "inconsistent NaviSched vLLM execution tracking: "
                "output_key=%d counter=%s current=%d decrement=%d",
                execution_step.scheduler_output_key,
                name,
                current,
                decrement,
            )

    def _end_backend_execution(self, execution_step, completed_ns=0) -> None:
        if not execution_step.backend_execution_started:
            _execution_timing_probe.end_control(execution_step)
            return
        execution_step.backend_execution_started = False
        try:
            finished = (
                _execution_timing_probe.end(
                    execution_step,
                    self._bridge.end_backend_execution,
                    completed_ns // 1000,
                )
                if self._native_ingress and completed_ns > 0
                else _execution_timing_probe.end(
                    execution_step, self._bridge.end_backend_execution
                )
            )
            if not finished:
                logger.error("failed to finish vLLM backend execution timing")
        except Exception:
            logger.exception("failed to finish vLLM backend execution timing")

    def observe_model_output(self, execution_future: Future, output) -> None:
        execution_step = getattr(execution_future, "_navi_backend_execution_step", None)
        if execution_step is None:
            return

        def finished(unused_future=None):
            completion = (
                unused_future if isinstance(unused_future, Future) else execution_future
            )
            self._complete_backend_execution(
                execution_step, completion, output_complete=True
            )
            execution_future.__dict__.pop("_navi_backend_execution_step", None)

        if isinstance(output, Future):
            output.add_done_callback(finished)
        else:
            finished()

    def _complete_backend_execution(
        self, execution_step: _ExecutionStep, future: Future, output_complete=False
    ) -> None:
        if (
            not output_complete
            and not self._native_ingress
            and getattr(future, "_navi_backend_execution_step", None) is execution_step
            and not future.cancelled()
            and future.exception() is None
            and future.result() is None
        ):
            return
        if execution_step.backend_completion_recorded:
            return
        execution_step.backend_completion_recorded = True
        completed_ns = time.perf_counter_ns()
        completed_time_us = 0
        execution_step.future_completed_ns = completed_ns
        if (
            self._native_ingress
            and (
                execution_step.batch_progress
                or execution_step.prefill_tokens > 0
                or execution_step.decode_tokens > 0
            )
            and future.done()
            and not future.cancelled()
            and future.exception() is None
        ):
            with self._pd_output_lock:
                output_completed_ns = self._pd_output_completed_ns.get(
                    execution_step.scheduler_output_key
                )
                if output_completed_ns is None:
                    self._pd_ready_finishes[execution_step.scheduler_output_key] = (
                        execution_step,
                        future,
                    )
                    return
            completed_ns, completed_time_us = output_completed_ns
        self._finalize_backend_execution(
            execution_step, future, completed_ns, completed_time_us
        )

    def _finalize_backend_execution(
        self,
        execution_step: _ExecutionStep,
        future: Future,
        completed_ns: int,
        completed_time_us: int = 0,
    ) -> None:
        if self._native_ingress:
            with self._pd_output_lock:
                execution_step.global_padded_tokens = self._pd_output_geometry.pop(
                    execution_step.scheduler_output_key, None
                )
        self._finish_execution_tracking(execution_step)
        self._end_backend_execution(execution_step, completed_ns)
        success = (
            future.done() and not future.cancelled() and future.exception() is None
        )
        future.__dict__.pop("_navi_backend_execution_step", None)
        try:
            self.on_request_lifecycle_event(
                {
                    "event": "execution_step",
                    **_execution_timing_probe.event_fields(execution_step),
                    "event_time_us": time.time_ns() // 1000,
                    "step_id": execution_step.started_ns,
                    "scheduler_output_key": execution_step.scheduler_output_key,
                    "started_perf_ns": execution_step.started_ns,
                    "completed_perf_ns": completed_ns,
                    "callback_elapsed_ms": (completed_ns - execution_step.started_ns)
                    / 1e6,
                    "pre_submit_ms": (
                        (execution_step.submitted_ns - execution_step.started_ns) / 1e6
                        if execution_step.submitted_ns
                        else None
                    ),
                    "future_callback_ms": (
                        (
                            execution_step.future_completed_ns
                            - execution_step.submitted_ns
                        )
                        / 1e6
                        if execution_step.submitted_ns
                        else None
                    ),
                    "output_wait_ms": max(
                        completed_ns - execution_step.future_completed_ns, 0
                    )
                    / 1e6,
                    "prefill_requests": execution_step.prefill_request_count,
                    "prefill_tokens": execution_step.prefill_tokens,
                    "decode_requests": execution_step.decode_request_count,
                    "decode_tokens": execution_step.decode_tokens,
                    "workload_counts_collected": execution_step.workload_counts_collected,
                    "running_count": execution_step.running_request_count,
                    "waiting_count": execution_step.waiting_request_count,
                    "active_executions_before": execution_step.active_executions_before,
                    "preempted_requests": execution_step.preempted_request_count,
                    "resumed_requests": execution_step.resumed_request_count,
                    "success": success,
                }
            )
        except Exception:
            self._request_audit_failures += 1
        if future.done() and not future.cancelled():
            try:
                if success:
                    self._bridge.report_execution_step(
                        (completed_ns - execution_step.started_ns) / 1e6,
                        execution_step.prefill_tokens,
                        execution_step.decode_tokens,
                    )
            except Exception:
                logger.exception("failed to report NaviSched vLLM execution step")
        if execution_step.batch_progress or execution_step.prefill_tokens > 0:
            try:
                self._complete_execution_step(execution_step, future, completed_ns)
            except Exception:
                self._discard_pd_output(execution_step.scheduler_output_key)
                self._abort_execution_step(execution_step)
                logger.exception("failed to complete NaviSched vLLM execution step")
        elif self._native_ingress:
            self._defer_pd_batch_finish(execution_step.scheduler_output_key, [])

    def _discard_pd_output(self, output_key: int) -> None:
        if not self._native_ingress:
            return
        with self._pd_output_lock:
            self._pd_output_completed_ns.pop(output_key, None)
            self._pd_output_geometry.pop(output_key, None)
            self._pd_ready_finishes.pop(output_key, None)

    def _defer_pd_batch_finish(self, output_key: int, batches: list[Any]) -> None:
        with self._pd_output_lock:
            completed_ns = self._pd_output_completed_ns.pop(output_key, None)
            if completed_ns is not None:
                self._pd_output_geometry.pop(output_key, None)
            if completed_ns is None:
                self._pd_ready_finishes[output_key] = batches
                return
        for batch in batches:
            self._finish_batch(batch, *completed_ns)

    @staticmethod
    def _validated_global_padded_tokens(geometry, scheduled_tokens):
        """Accept observed Worker layout metadata, never infer missing geometry."""
        if (
            not isinstance(geometry, dict)
            or type(geometry.get("version")) is not int
            or geometry["version"] != 1
        ):
            return None
        chunks = geometry.get("chunks")
        if not isinstance(chunks, list) or not chunks:
            return None
        total_scheduled = 0
        total_padded = 0
        for chunk in chunks:
            names = (
                "scheduled_tokens",
                "pcp_size",
                "interleave_size",
                "max_local_tokens",
                "local_padded_tokens",
                "global_padded_tokens",
            )
            if not isinstance(chunk, dict) or any(
                type(chunk.get(name)) is not int or chunk[name] <= 0 for name in names
            ):
                return None
            if (
                chunk["global_padded_tokens"]
                != chunk["pcp_size"] * chunk["local_padded_tokens"]
                or chunk["max_local_tokens"] > chunk["local_padded_tokens"]
            ):
                return None
            cycles, remainder = divmod(
                chunk["scheduled_tokens"], chunk["pcp_size"] * chunk["interleave_size"]
            )
            if chunk["max_local_tokens"] != cycles * chunk["interleave_size"] + min(
                remainder, chunk["interleave_size"]
            ):
                return None
            dp_target = chunk.get("dp_target_bucket")
            if dp_target is not None and (
                type(dp_target) is not int
                or not 0 <= dp_target <= chunk["local_padded_tokens"]
            ):
                return None
            total_scheduled += chunk["scheduled_tokens"]
            total_padded += chunk["global_padded_tokens"]
        if total_scheduled != scheduled_tokens or total_padded > (1 << 63) - 1:
            return None
        return total_padded

    def on_model_output(self, scheduler_output: Any, model_output: Any = None) -> None:
        """Complete PD prefill after scheduler/cache output processing."""
        if not self._native_ingress:
            return
        output_key = id(scheduler_output)
        completed_ns = time.perf_counter_ns()
        completed_time_us = time.time_ns() // 1000
        geometry = getattr(model_output, "execution_geometry", None)
        if geometry is not None:
            try:
                self.on_request_lifecycle_event(
                    {
                        "event": "execution_geometry",
                        "event_time_us": time.time_ns() // 1000,
                        "scheduler_output_key": output_key,
                        "output_completed_perf_ns": completed_ns,
                        "geometry": geometry,
                    }
                )
            except Exception:
                logger.exception("failed to audit TPU execution geometry")
        scheduled = getattr(scheduler_output, "num_scheduled_tokens", {})
        padded = self._validated_global_padded_tokens(
            geometry, sum(scheduled.values()) if isinstance(scheduled, dict) else -1
        )
        with self._pd_output_lock:
            self._pd_output_geometry[output_key] = padded
            batches = self._pd_ready_finishes.pop(output_key, None)
            if batches is None or isinstance(batches, tuple):
                self._pd_output_completed_ns[output_key] = (
                    completed_ns,
                    completed_time_us,
                )
            if batches is None:
                return
        if isinstance(batches, tuple):
            self._finalize_backend_execution(*batches, completed_ns, completed_time_us)
            return
        with self._pd_output_lock:
            self._pd_output_geometry.pop(output_key, None)
        for batch in batches:
            self._finish_batch(batch, completed_ns, completed_time_us)

    def _prepare_execution_step(
        self, scheduler_output, get_request, execution_step
    ) -> None:
        progress_by_state = {}
        local_batch_requests = []
        local_request_stats = []
        publication_prompt_tokens = {}
        publication_reporters = {}
        local_publication_requests = {}
        local_request_ids = set()
        for internal_request_id in scheduler_output.num_scheduled_tokens:
            request = get_request(internal_request_id)
            if request is None:
                raise RuntimeError(
                    f"vLLM request {internal_request_id!r} is unavailable"
                )
            scheduled_tokens = int(
                scheduler_output.num_scheduled_tokens.get(request.request_id, 0)
            )
            if execution_step is not None and scheduled_tokens > 0:
                execution_step.scheduled_tokens_by_request[request.request_id] = (
                    scheduled_tokens
                )
            computed_after_schedule = int(request.num_computed_tokens)
            computed_before_schedule = max(
                0, computed_after_schedule - scheduled_tokens
            )
            prefill_tokens = min(
                scheduled_tokens,
                max(
                    0,
                    int(request.num_prompt_tokens) - computed_before_schedule,
                ),
            )
            decode_tokens = max(0, scheduled_tokens - prefill_tokens)
            if execution_step is not None:
                if prefill_tokens > 0:
                    execution_step.prefill_request_count += 1
                    execution_step.prefill_tokens += prefill_tokens
                if decode_tokens > 0:
                    execution_step.decode_request_count += 1
                    execution_step.decode_tokens += decode_tokens
            if prefill_tokens <= 0:
                continue

            total_tokens = min(
                int(request.num_prompt_tokens),
                computed_before_schedule + prefill_tokens,
            )
            cache_hit_tokens = total_tokens - prefill_tokens
            external_request_id = request.request_id
            request_id = self._navi_request_id(request)
            full_block_count = total_tokens // self._block_size
            block_hashes = [
                block_hash_to_uint64(value)
                for value in request.block_hashes[:full_block_count]
            ]
            state = self._scheduled_forward_batches.get(request_id)
            prefill_stats = getattr(request, "prefill_stats", None)
            local_cache_hits = getattr(prefill_stats, "num_local_cached_tokens", -1)
            external_cache_hits = getattr(
                prefill_stats, "num_external_cached_tokens", -1
            )
            local_cache_hits = (
                max(-1, int(local_cache_hits)) if local_cache_hits is not None else -1
            )
            external_cache_hits = (
                max(-1, int(external_cache_hits))
                if external_cache_hits is not None
                else -1
            )
            if local_cache_hits >= 0 and external_cache_hits >= 0:
                prefix_cache_hits = local_cache_hits + external_cache_hits
                prefix_cache_hit_source = "prefill_stats"
            else:
                legacy_cache_hits = getattr(request, "num_cached_tokens", -1)
                prefix_cache_hits = (
                    max(-1, int(legacy_cache_hits))
                    if legacy_cache_hits is not None
                    else -1
                )
                prefix_cache_hit_source = (
                    "request.num_cached_tokens" if prefix_cache_hits >= 0 else "unknown"
                )
            self.on_request_lifecycle_event(
                {
                    "event": "prefill_cache_observation",
                    "event_time_us": time.time_ns() // 1000,
                    "backend_step_started_ns": execution_step.started_ns,
                    "request_id": external_request_id,
                    "navi_request_id": request_id,
                    "sample_scope": "forwarded" if state is not None else "local",
                    "prefix_cache_hit_tokens": prefix_cache_hits,
                    "prefix_cache_hit_local_tokens": local_cache_hits,
                    "prefix_cache_hit_external_tokens": external_cache_hits,
                    "prefix_cache_hit_source": prefix_cache_hit_source,
                    "computed_context_tokens": computed_before_schedule,
                    "scheduled_prefill_tokens": prefill_tokens,
                    "prompt_tokens": int(request.num_prompt_tokens),
                    "request_num_tokens": getattr(request, "num_tokens", None),
                    "shared_prefix_boundary": getattr(
                        request, "shared_prefix_boundary", None
                    ),
                    **self._explicit_checkpoint_audit_fields(request, self._block_size),
                    "first_forwarded_step": (
                        request_id not in state.initial_cache_hits
                        if state is not None
                        else None
                    ),
                }
            )
            if state is None:
                retained_state = self._forward_request_state(external_request_id)
                if retained_state is not None:
                    if not retained_state.released:
                        raise RuntimeError(
                            "vLLM prefill has inconsistent NaviSched ownership"
                        )
                    continue
                if request_id in local_request_ids:
                    raise ValueError(
                        "NaviSched requires unique request ids within a vLLM step"
                    )
                local_request_ids.add(request_id)
                publication_prompt_tokens[request_id] = int(request.num_prompt_tokens)
                local_publication_requests[request_id] = external_request_id
                local_batch_requests.append(
                    (request_id, total_tokens, self._block_size, block_hashes)
                )
                local_request_stats.append((request_id, total_tokens, cache_hit_tokens))
                continue

            with state.lock:
                if (
                    state.released
                    or state.tracking_aborted
                    or request_id in state.completed_prefill_request_ids
                ):
                    continue
                progress = progress_by_state.get(id(state))
                if progress is None:
                    progress = _BatchProgress(state.handle, state)
                    progress_by_state[id(state)] = progress
                    execution_step.batch_progress.append(progress)
                progress.prefill_request_count += 1
                progress.prefill_tokens += prefill_tokens
                if computed_after_schedule >= int(request.num_prompt_tokens):
                    progress.completed_request_ids.add(request_id)
                if state.handle is None:
                    if state.native_batch is None:
                        raise RuntimeError("missing NaviSched forwarded batch handle")
                    active_request_ids = [
                        value
                        for value in state.request_ids
                        if value not in state.cancelled_request_ids
                    ]
                    try:
                        state.handle = _execution_timing_probe.batch_begin(
                            state,
                            self._bridge.start_forward_batch,
                            state.native_batch,
                            active_request_ids,
                        )
                    except Exception:
                        self._release_forward_batch(state)
                        raise
                    if state.handle is None:
                        logger.error(
                            "failed to start NaviSched batch %d", state.batch_id
                        )
                        self._release_forward_batch(state)
                        continue
                    progress.batch_handle = state.handle
                if state.execution_started_ns == 0:
                    state.execution_started_ns = execution_step.started_ns
                if request_id not in state.initial_cache_hits:
                    state.initial_external_cache_hits[request_id] = (
                        external_cache_hits
                        if local_cache_hits >= 0
                        and external_cache_hits >= 0
                        and local_cache_hits + external_cache_hits == cache_hit_tokens
                        else -1
                    )
                    boundary = getattr(request, "shared_prefix_boundary", None)
                    if type(boundary) is int and 0 <= boundary <= int(
                        request.num_prompt_tokens
                    ):
                        state.initial_execution_policy_inputs[request_id] = (
                            int(request.num_prompt_tokens),
                            cache_hit_tokens,
                            boundary,
                        )
                state.initial_cache_hits.setdefault(request_id, cache_hit_tokens)
                publication_reporters[external_request_id] = state.handle

        for progress in execution_step.batch_progress:
            state = progress.state
            if state is None:
                continue
            with state.lock:
                if state.released or state.tracking_aborted:
                    continue
                active_ids = [
                    request_id
                    for request_id in state.request_ids
                    if request_id not in state.cancelled_request_ids
                ]
                if (
                    not state.execution_cache_reported
                    and state.handle is not None
                    and active_ids
                    and all(
                        request_id in state.initial_execution_policy_inputs
                        for request_id in active_ids
                    )
                ):
                    resolved = [
                        (
                            request_id,
                            *state.initial_execution_policy_inputs[request_id],
                            state.initial_external_cache_hits.get(request_id, -1),
                        )
                        for request_id in active_ids
                    ]
                    if not self._bridge.resolve_forward_execution_cache(
                        state.handle, resolved
                    ):
                        raise RuntimeError(
                            "failed to publish resolved execution cache features"
                        )
                    state.execution_cache_reported = True
                state.chunk_count += 1
                state.pending_prefill_steps += 1

        if local_batch_requests:
            batch_id = self._batch_id_prefix | next(self._batch_ids)
            if self._native_ingress:
                handle = self._bridge.start_batch(
                    batch_id,
                    local_batch_requests,
                    prefix_cache_prompt_tokens=publication_prompt_tokens,
                )
            else:
                handle = self._bridge.start_batch(batch_id, local_batch_requests)
            if handle is None:
                logger.error("failed to start NaviSched batch %d", batch_id)
            else:
                execution_step.batch_progress.append(
                    _BatchProgress((handle, local_request_stats), None)
                )
                publication_reporters.update(
                    (external_id, handle)
                    for external_id in local_publication_requests.values()
                )
        if self._native_ingress and self._v6d_scheduler is not None:
            self._attach_v6d_publication_reporters(publication_reporters)

    def _abort_execution_step(self, execution_step) -> None:
        released_states = set()
        for progress in execution_step.batch_progress:
            state = progress.state
            if state is None or id(state) in released_states:
                continue
            released_states.add(id(state))
            self._release_forward_batch(state)

    @staticmethod
    def _explicit_checkpoint_audit_fields(request, scheduler_block_size):
        return _execution_timing_probe.explicit_checkpoint_audit_fields(
            request, scheduler_block_size
        )

    @staticmethod
    def _claim_forward_batch_prefill_complete(state) -> bool:
        if (
            state.released
            or state.tracking_aborted
            or state.prefill_complete
            or state.pending_prefill_steps != 0
            or not state.requests
            or not set(state.requests).issubset(state.completed_prefill_request_ids)
        ):
            return False
        state.prefill_complete = True
        return True

    def _complete_execution_step(
        self, execution_step: _ExecutionStep, future: Future, completed_ns=None
    ) -> None:
        if completed_ns is None:
            completed_ns = time.perf_counter_ns()
        try:
            error = future.exception()
        except CancelledError:
            error = True
        step_ms = (completed_ns - execution_step.started_ns) / 1_000_000.0
        batch_ids = sorted(
            {
                progress.state.batch_id
                for progress in execution_step.batch_progress
                if progress.state is not None
            }
        )
        _execution_timing_probe._diagnostic_format_info(
            logger,
            "[navi-vllm-execution-step] batch_ids=%s tracked_batches=%d "
            "prefill_requests=%d prefill_tokens=%d decode_requests=%d "
            "decode_tokens=%d execute_model_ms=%.3f success=%d backend_step_started_ns=%d",
            ",".join(str(value) for value in batch_ids) or "<local-or-recompute>",
            len(batch_ids),
            execution_step.prefill_request_count,
            execution_step.prefill_tokens,
            execution_step.decode_request_count,
            execution_step.decode_tokens,
            step_ms,
            int(error is None),
            execution_step.started_ns,
        )
        if error is not None:
            self._discard_pd_output(execution_step.scheduler_output_key)
            self._abort_execution_step(execution_step)
            return

        finished_batches = []
        for progress in execution_step.batch_progress:
            state = progress.state
            if state is None:
                if self._native_ingress:
                    # TPU execute_model returns before the device output is ready.
                    # Local Prefill batches must use the same completion barrier
                    # as NaviSched-forwarded batches for truthful training targets.
                    finished_batches.append(progress.batch_handle)
                else:
                    self._finish_batch(progress.batch_handle, completed_ns)
                continue
            with state.lock:
                if state.released or state.tracking_aborted:
                    continue
                state.execution_step_count += 1
                # Missing runner telemetry is different from a known shared or
                # decode-containing target: only the former can use a projected
                # feature pair without assigning another batch's latency here.
                if (
                    len(execution_step.batch_progress) != 1
                    or progress.prefill_tokens != execution_step.prefill_tokens
                    or execution_step.decode_request_count != 0
                    or execution_step.decode_tokens != 0
                ):
                    state.projected_execution_exclusive = False
                # Shared Core steps cannot be apportioned to individual batches.
                if (
                    self._native_ingress
                    and len(execution_step.batch_progress) == 1
                    and progress.prefill_tokens == execution_step.prefill_tokens
                    and execution_step.decode_request_count == 0
                    and execution_step.decode_tokens == 0
                    and execution_step.global_padded_tokens is not None
                    and not state.cancelled_request_ids
                    and state.global_padded_tokens + execution_step.global_padded_tokens
                    <= (1 << 63) - 1
                ):
                    state.geometry_execution_step_count += 1
                    state.global_padded_tokens += execution_step.global_padded_tokens
                state.execution_step_ms += (
                    (
                        completed_ns
                        - max(execution_step.started_ns, state.execution_started_ns)
                    )
                    / 1e6
                    if self._native_ingress
                    else step_ms
                )
                state.execution_prefill_request_count += progress.prefill_request_count
                state.execution_prefill_tokens += progress.prefill_tokens
                state.execution_decode_request_count += (
                    execution_step.decode_request_count
                )
                state.execution_decode_tokens += execution_step.decode_tokens
                state.completed_prefill_request_ids.update(
                    progress.completed_request_ids
                )
                if state.pending_prefill_steps <= 0:
                    logger.error(
                        "inconsistent NaviSched vLLM prefill-step tracking: "
                        "batch_id=%d pending_steps=%d completed_step=1",
                        state.batch_id,
                        state.pending_prefill_steps,
                    )
                    state.pending_prefill_steps = 0
                else:
                    state.pending_prefill_steps -= 1
                if self._claim_forward_batch_prefill_complete(state):
                    finished_batches.append(state)
        if self._native_ingress:
            self._defer_pd_batch_finish(
                execution_step.scheduler_output_key, finished_batches
            )
        else:
            for batch in finished_batches:
                self._finish_batch(batch, completed_ns)

    def _finish_batch(
        self, batch_handle, completed_ns=None, completed_time_us=0
    ) -> None:
        if isinstance(batch_handle, _ForwardBatchState):
            state = batch_handle
            error = None
            handle = None
            request_stats = None
            diagnostic = None
            finish_ns = completed_ns or time.perf_counter_ns()
            try:
                with state.lock:
                    if state.released:
                        return
                    active_request_ids = tuple(
                        request_id
                        for request_id in state.request_ids
                        if request_id not in state.cancelled_request_ids
                    )
                    handle = state.handle
                    if (
                        handle is None
                        or not active_request_ids
                        or any(
                            request_id not in state.requests
                            or request_id not in state.initial_cache_hits
                            for request_id in active_request_ids
                        )
                    ):
                        error = RuntimeError(
                            "incomplete NaviSched forwarded batch state"
                        )
                    else:
                        request_stats = [
                            (
                                request_id,
                                int(state.requests[request_id].num_prompt_tokens),
                                state.initial_cache_hits[request_id],
                            )
                            for request_id in active_request_ids
                        ]
                    diagnostic = (
                        state.batch_id,
                        state.chunk_count,
                        state.execution_step_count,
                        state.execution_prefill_request_count,
                        state.execution_prefill_tokens,
                        state.execution_decode_request_count,
                        state.execution_decode_tokens,
                        state.execution_started_ns,
                        state.execution_step_ms,
                        state.admission_started_ns,
                        state.native_preparation_ms,
                        state.admission_state or _AdmissionState(),
                        state.execution_start_state or _ExecutionStartState(),
                        tuple(
                            state.predicted_cache_hits.get(request_id, 0)
                            for request_id in active_request_ids
                        ),
                        tuple(
                            state.initial_cache_hits.get(request_id, 0)
                            for request_id in active_request_ids
                        ),
                    )
                    geometry = (
                        state.geometry_execution_step_count,
                        state.global_padded_tokens,
                    )
                    self._mark_forward_batch_released(state)
            finally:
                with state.lock:
                    if not state.released:
                        self._mark_forward_batch_released(state)
                self._cleanup_released_forward_batch(state)
            if error is not None:
                raise error
            (
                batch_id,
                chunk_count,
                execution_step_count,
                execution_prefill_request_count,
                execution_prefill_tokens,
                execution_decode_request_count,
                execution_decode_tokens,
                execution_started_ns,
                execution_step_ms,
                admission_started_ns,
                native_preparation_ms,
                admission_state,
                start_state,
                predicted_cache_hits,
                actual_cache_hits,
            ) = diagnostic
            wall_ms = (
                (finish_ns - execution_started_ns) / 1_000_000.0
                if execution_started_ns
                else 0.0
            )
            admission_to_prefill_complete_ms = (
                (finish_ns - admission_started_ns) / 1_000_000.0
                if admission_started_ns
                else -1.0
            )
            admission_to_first_execution_ms = (
                max(0, execution_started_ns - admission_started_ns) / 1_000_000.0
                if admission_started_ns and execution_started_ns
                else -1.0
            )
            preceding = admission_state.preceding_workload
            (
                sample_running_capacity,
                sample_waiting_capacity,
            ) = self._capacity_sample_features(admission_state, request_stats)
            (
                sample_scheduler_running,
                sample_scheduler_waiting,
            ) = self._scheduler_sample_features(admission_state, request_stats)
            sample_batch_count = preceding.batch_count + 1
            sample_total_prompt_tokens = (
                sample_running_capacity.context_tokens
                + sample_running_capacity.compute_tokens
                + sample_waiting_capacity.context_tokens
                + sample_waiting_capacity.compute_tokens
            )
            predicted_cache_hit_tokens = sum(predicted_cache_hits)
            actual_cache_hit_tokens = sum(actual_cache_hits)
            cache_hit_mae_tokens = sum(
                abs(actual - predicted)
                for actual, predicted in zip(
                    actual_cache_hits, predicted_cache_hits, strict=True
                )
            )
            # Pipeline steps overlap, so their summed latency can exceed wall time.
            # Match the native reporter's nonnegative gap statistic.
            inter_step_gap_ms = max(0.0, wall_ms - execution_step_ms)
            _execution_timing_probe._diagnostic_format_info(
                logger,
                "[navi-vllm-execution-batch] batch_id=%d chunks=%d steps=%d "
                "prefill_request_steps=%d prefill_tokens=%d "
                "decode_request_steps=%d decode_tokens=%d wall_ms=%.3f "
                "execute_model_sum_ms=%.3f inter_step_gap_ms=%.3f "
                "admission_to_prefill_complete_ms=%.3f "
                "admission_to_first_execution_ms=%.3f "
                "predicted_cache_hit_tokens=%d "
                "actual_cache_hit_tokens=%d cache_hit_bias_tokens=%d "
                "cache_hit_mae_tokens=%d "
                "incoming_compute_tokens=%d "
                "incoming_compute_token_square_sum=%.0f "
                "incoming_hit_compute_token_product_sum=%.0f "
                "incoming_total_prompt_tokens=%d "
                "admission_preceding_requests=%d "
                "admission_preceding_prefill_requests=%d "
                "admission_running_requests=%d admission_waiting_requests=%d "
                "admission_running_decode_requests=%d "
                "admission_running_decode_generated_tokens=%d "
                "admission_running_decode_remaining_token_budget=%d "
                "admission_prefill_context_tokens=%d "
                "admission_remaining_prefill_tokens=%d "
                "admission_remaining_prefill_token_square_sum=%.0f "
                "admission_context_remaining_prefill_product_sum=%.0f "
                "admission_max_running_requests=%d "
                "admission_running_capacity_slots=%d "
                "sample_batch_count=%d sample_total_prompt_tokens=%d "
                "sample_running_capacity_prefill_requests=%d "
                "sample_running_capacity_context_tokens=%d "
                "sample_running_capacity_compute_tokens=%d "
                "sample_running_capacity_compute_square_sum=%.0f "
                "sample_running_capacity_context_compute_product_sum=%.0f "
                "sample_waiting_capacity_prefill_requests=%d "
                "sample_waiting_capacity_context_tokens=%d "
                "sample_waiting_capacity_compute_tokens=%d "
                "sample_waiting_capacity_compute_square_sum=%.0f "
                "sample_waiting_capacity_context_compute_product_sum=%.0f "
                "sample_scheduler_running_prefill_requests=%d "
                "sample_scheduler_running_context_tokens=%d "
                "sample_scheduler_running_compute_tokens=%d "
                "sample_scheduler_running_compute_square_sum=%.0f "
                "sample_scheduler_running_context_compute_product_sum=%.0f "
                "sample_scheduler_waiting_prefill_requests=%d "
                "sample_scheduler_waiting_context_tokens=%d "
                "sample_scheduler_waiting_compute_tokens=%d "
                "sample_scheduler_waiting_compute_square_sum=%.0f "
                "sample_scheduler_waiting_context_compute_product_sum=%.0f "
                "admission_total_context_tokens=%d "
                "admission_remaining_total_tokens=%d "
                "admission_remaining_total_token_square_sum=%.0f "
                "admission_context_remaining_total_product_sum=%.0f "
                "admission_total_prompt_tokens=%d "
                "admission_active_executions=%d "
                "admission_active_scheduled_tokens=%d "
                "start_active_executions=%d "
                "start_active_prefill_executions=%d "
                "start_active_decode_executions=%d "
                "start_active_scheduled_tokens=%d "
                "start_running_requests=%d "
                "start_running_prefill_requests=%d "
                "start_running_decode_requests=%d "
                "start_running_recompute_requests=%d "
                "start_waiting_requests=%d start_preempted_requests=%d "
                "start_resumed_requests=%d start_step_prefill_requests=%d "
                "start_step_prefill_tokens=%d start_step_decode_requests=%d "
                "start_step_decode_tokens=%d "
                "native_preparation_ms=%.3f execution_total_ms=%.3f",
                batch_id,
                chunk_count,
                execution_step_count,
                execution_prefill_request_count,
                execution_prefill_tokens,
                execution_decode_request_count,
                execution_decode_tokens,
                wall_ms,
                execution_step_ms,
                inter_step_gap_ms,
                admission_to_prefill_complete_ms,
                admission_to_first_execution_ms,
                predicted_cache_hit_tokens,
                actual_cache_hit_tokens,
                actual_cache_hit_tokens - predicted_cache_hit_tokens,
                cache_hit_mae_tokens,
                admission_state.incoming_compute_tokens,
                admission_state.incoming_compute_token_square_sum,
                admission_state.incoming_hit_compute_token_product_sum,
                admission_state.incoming_total_prompt_tokens,
                preceding.request_count,
                preceding.remaining_prefill_request_count,
                preceding.running_request_count,
                preceding.waiting_request_count,
                preceding.running_decode_request_count,
                preceding.running_decode_generated_tokens,
                preceding.running_decode_remaining_token_budget,
                preceding.remaining_prefill_context_tokens,
                preceding.remaining_prefill_tokens,
                preceding.remaining_prefill_token_square_sum,
                preceding.context_remaining_prefill_token_product_sum,
                preceding.max_running_request_count,
                preceding.running_capacity_slot_count,
                sample_batch_count,
                sample_total_prompt_tokens,
                sample_running_capacity.request_count,
                sample_running_capacity.context_tokens,
                sample_running_capacity.compute_tokens,
                sample_running_capacity.compute_token_square_sum,
                sample_running_capacity.context_compute_token_product_sum,
                sample_waiting_capacity.request_count,
                sample_waiting_capacity.context_tokens,
                sample_waiting_capacity.compute_tokens,
                sample_waiting_capacity.compute_token_square_sum,
                sample_waiting_capacity.context_compute_token_product_sum,
                sample_scheduler_running.request_count,
                sample_scheduler_running.context_tokens,
                sample_scheduler_running.compute_tokens,
                sample_scheduler_running.compute_token_square_sum,
                sample_scheduler_running.context_compute_token_product_sum,
                sample_scheduler_waiting.request_count,
                sample_scheduler_waiting.context_tokens,
                sample_scheduler_waiting.compute_tokens,
                sample_scheduler_waiting.compute_token_square_sum,
                sample_scheduler_waiting.context_compute_token_product_sum,
                preceding.remaining_total_context_tokens,
                preceding.remaining_total_tokens,
                preceding.remaining_total_token_square_sum,
                preceding.context_remaining_total_token_product_sum,
                preceding.total_prompt_tokens,
                admission_state.active_executions,
                preceding.inflight_scheduled_tokens,
                start_state.active_executions,
                start_state.active_prefill_executions,
                start_state.active_decode_executions,
                start_state.active_scheduled_tokens,
                start_state.running_requests,
                start_state.running_prefill_requests,
                start_state.running_decode_requests,
                start_state.running_recompute_requests,
                start_state.waiting_requests,
                start_state.preempted_requests,
                start_state.resumed_requests,
                start_state.step_prefill_requests,
                start_state.step_prefill_tokens,
                start_state.step_decode_requests,
                start_state.step_decode_tokens,
                native_preparation_ms,
                wall_ms + native_preparation_ms,
            )
            execution_breakdown = (
                wall_ms,
                execution_step_ms,
                execution_step_count,
                execution_prefill_tokens,
                execution_decode_request_count,
                execution_decode_tokens,
                -1.0 if self._native_ingress else admission_to_prefill_complete_ms,
                preceding.request_count,
                sample_batch_count,
                sample_running_capacity.request_count,
                sample_running_capacity.context_tokens,
                sample_running_capacity.compute_tokens,
                sample_running_capacity.compute_token_square_sum,
                sample_running_capacity.context_compute_token_product_sum,
                sample_waiting_capacity.request_count,
                sample_waiting_capacity.context_tokens,
                sample_waiting_capacity.compute_tokens,
                sample_waiting_capacity.compute_token_square_sum,
                sample_waiting_capacity.context_compute_token_product_sum,
                preceding.running_elapsed_time_ms,
                preceding.running_decode_request_count,
                preceding.running_decode_generated_tokens,
                preceding.running_decode_remaining_token_budget,
                admission_to_first_execution_ms,
                native_preparation_ms,
            )
            try:
                execution_finish = (completed_time_us,) if self._native_ingress else ()
                # The runner's completed geometry is a training feature. A
                # dispatch projection must never be substituted for that fact.
                if (
                    self._execution_geometry_enabled
                    and geometry[0] == execution_step_count
                    and geometry[0] > 0
                    and geometry[1] > 0
                    and not state.cancelled_request_ids
                ):
                    finished = _execution_timing_probe.batch_finish(
                        state,
                        self._bridge.finish_batch_with_execution_geometry,
                        handle,
                        request_stats,
                        *execution_breakdown,
                        *geometry,
                        *execution_finish,
                    )
                else:
                    finished = _execution_timing_probe.batch_finish(
                        state,
                        self._bridge.finish_batch_with_execution,
                        handle,
                        request_stats,
                        *execution_breakdown,
                        *execution_finish,
                    )
                if not finished:
                    logger.error("failed to finish NaviSched vLLM batch")
            except Exception:
                logger.exception("failed to finish NaviSched vLLM batch")
            finally:
                self.on_request_lifecycle_event(
                    {
                        "event": "legacy_native_batch_clock",
                        "event_time_us": time.time_ns() // 1000,
                        "native_batch_id": state.batch_id,
                        "native_schedule_batch_id": state.schedule_batch_id,
                        **_execution_timing_probe.batch_event_fields(state),
                    }
                )
            return
        handle, request_stats = batch_handle
        try:
            if not self._bridge.finish_batch(handle, request_stats):
                logger.error("failed to finish NaviSched vLLM batch")
        except Exception:
            logger.exception("failed to finish NaviSched vLLM batch")

    def _matching_forward_batches(self, external_request_ids):
        states = {}
        pending = list(self._pending_forward_batches.values())
        scheduled = list(self._scheduled_forward_batches.values())
        for state in itertools.chain(
            pending,
            scheduled,
        ):
            with state.lock:
                if state.released:
                    continue
                if external_request_ids is None or any(
                    value in external_request_ids
                    for value in state.external_request_ids.values()
                ):
                    states[id(state)] = state
        return list(states.values())

    def _abort_forward_requests(self, state, request_ids) -> None:
        response_requests_to_remove = []
        finish_remaining_batch = False
        with state.lock:
            if state.released:
                return
            newly_cancelled = {
                request_id
                for request_id in request_ids
                if request_id in state.request_ids
                and request_id not in state.cancelled_request_ids
            }
            if not newly_cancelled:
                return
            active_request_ids = {
                request_id
                for request_id in state.request_ids
                if request_id not in state.cancelled_request_ids
            }
            abort_entire_batch = newly_cancelled == active_request_ids
            if not abort_entire_batch:
                try:
                    partially_aborted = (
                        state.native_batch is not None
                        and self._bridge.abort_forward_batch_requests(
                            state.native_batch, sorted(newly_cancelled)
                        )
                    )
                except Exception:
                    logger.exception(
                        "failed to abort part of NaviSched vLLM batch: "
                        "batch_id=%d request_ids=%s",
                        state.batch_id,
                        sorted(newly_cancelled),
                    )
                    partially_aborted = False
                if not partially_aborted:
                    logger.error(
                        "falling back to aborting NaviSched tracking for entire "
                        "vLLM batch: batch_id=%d request_ids=%s",
                        state.batch_id,
                        sorted(newly_cancelled),
                    )
                    abort_entire_batch = True
            state.tracking_aborted = abort_entire_batch
            state.cancelled_request_ids.update(newly_cancelled)
            self._remove_prepared_workload(
                state, state.request_ids if abort_entire_batch else newly_cancelled
            )
            for request_id in newly_cancelled:
                if request_id in state.arrived_request_ids:
                    if self._pending_forward_batches.get(request_id) is state:
                        self._pending_forward_batches.pop(request_id, None)
                    if not state.admission_ready:
                        response_requests_to_remove.append(request_id)
                if self._scheduled_forward_batches.get(request_id) is state:
                    self._scheduled_forward_batches.pop(request_id, None)
                external_request_id = state.external_request_ids.get(request_id)
                if external_request_id is not None:
                    self._forget_forward_request(external_request_id)
                state.requests.pop(request_id, None)
                state.initial_cache_hits.pop(request_id, None)
                state.initial_external_cache_hits.pop(request_id, None)
                state.admission_contributions.pop(request_id, None)
            if abort_entire_batch:
                self._release_forward_reporter(state)
                if state.submission_failed or len(state.arrived_request_ids) == len(
                    state.request_ids
                ):
                    self._release_forward_batch(state)
            else:
                finish_remaining_batch = self._claim_forward_batch_prefill_complete(
                    state
                )
        self._remove_response_requests(response_requests_to_remove)
        if finish_remaining_batch:
            try:
                self._finish_batch(state)
            except Exception:
                logger.exception(
                    "failed to finish NaviSched vLLM batch after partial abort: "
                    "batch_id=%d request_ids=%s",
                    state.batch_id,
                    sorted(newly_cancelled),
                )

    def _release_native_forward_batch(self, native_batch) -> None:
        if native_batch is None:
            return
        try:
            self._bridge.release_forward_batch(native_batch)
        except Exception:
            logger.exception("failed to release NaviSched native forwarded batch")

    def _release_forward_reporter(self, state) -> None:
        with state.lock:
            state.handle = None
            native_batch = state.native_batch
            state.native_batch = None
        self._release_native_forward_batch(native_batch)
        with self._prepared_workload_lock:
            self._native_handoff_batches.pop(state.batch_id, None)

    @staticmethod
    def _mark_forward_batch_released(state) -> None:
        state.released = True
        state.handle = None

    def _cleanup_released_forward_batch(self, state) -> None:
        if state.released:
            native_batch = state.native_batch
            state.native_batch = None
        else:
            return
        self._remove_prepared_workload(state, state.request_ids)
        with self._prepared_workload_lock:
            self._native_handoff_batches.pop(state.batch_id, None)
        for request_id in state.request_ids:
            if self._pending_forward_batches.get(request_id) is state:
                self._pending_forward_batches.pop(request_id, None)
            if self._scheduled_forward_batches.get(request_id) is state:
                self._scheduled_forward_batches.pop(request_id, None)
        self._release_native_forward_batch(native_batch)

    def _release_forward_batch(self, state) -> None:
        with state.lock:
            if state.released:
                return
            self._mark_forward_batch_released(state)
        self._cleanup_released_forward_batch(state)

    def on_v6d_cache_snapshots(self, snapshots: list[dict]) -> None:
        """Feed existing resident-object snapshots into the original KV path."""
        if not snapshots:
            raise ValueError("V6D snapshot requires every physical node")
        if any(
            type(snapshot["peer_cache_read_available"]) is not bool
            for snapshot in snapshots
        ):
            raise ValueError("invalid V6D peer cache read availability")
        peer_cache_read_available = all(
            snapshot["peer_cache_read_available"] for snapshot in snapshots
        )
        common = None
        for snapshot in snapshots:
            groups = {}
            for obj in snapshot["objects"]:
                if obj["tier"] != "vineyard" or obj["sealedResident"] is not True:
                    continue
                identity = obj["meta"].get("navi_sched_cache")
                if identity is None:
                    continue
                identity = json.loads(identity)
                group_id = identity["group_id"]
                if type(group_id) is not int or not 0 <= group_id < len(
                    self._cache_policy.group_roles
                ):
                    raise ValueError("invalid V6D cache group")
                hashes = [bytes.fromhex(value) for value in identity["block_hashes"]]
                if not hashes or any(len(value) != 32 for value in hashes):
                    raise ValueError("invalid V6D source block hash")
                groups.setdefault(group_id, set()).update(hashes)
            if common is None:
                common = groups
            else:
                for group_id in common:
                    common[group_id].intersection_update(groups.get(group_id, ()))
        self.on_block_lifecycle_event(
            "snapshot",
            None,
            {group_id: tuple(sorted(hashes)) for group_id, hashes in common.items()},
            cause="v6d_resident_snapshot",
            cache_key_type=1,
        )
        self._bridge.report_peer_cache_read_available(peer_cache_read_available)

    def on_block_lifecycle_event(
        self,
        event_type: str,
        group_id: int | None,
        block_hashes: tuple[bytes, ...] | dict[int, tuple[bytes, ...]],
        cause_request_id: str | None = None,
        cause: str = "unspecified",
        cache_key_type: int = 0,
    ) -> None:
        if type(cache_key_type) is not int or cache_key_type not in (0, 1):
            raise ValueError("invalid cache key type")
        with self._cache_lifecycle_lock:
            try:
                cache_state = self._cache_states[cache_key_type]
                if event_type == "snapshot":
                    delta = cache_state.replace(block_hashes)
                    if delta.empty():
                        return
                else:
                    delta = cache_state.apply(event_type, group_id, block_hashes)
                self._cache_lifecycle_sequences[cache_key_type] += 1
                sequence = self._cache_lifecycle_sequences[cache_key_type]
                event_time_us = time.time_ns() // 1000
                alias_delta = None
                if cache_key_type == 0:
                    alias_delta = self._image_cache_map.update(
                        (
                            tuple(
                                expanded
                                for value in block_hashes
                                for expanded in self._cache_policy.expand_group_hash(
                                    group_id, value
                                )
                            )
                            if group_id is not None
                            else ()
                        ),
                        cleared=event_type == "cleared",
                    )
                cause_batch_id, cause_schedule_batch_id = self._cause_batch_ids(
                    cause_request_id
                )
                audit = {
                    "event_time_us": event_time_us,
                    "sequence": sequence,
                    "cache_key_type": cache_key_type,
                    "event_type": event_type,
                    "group_id": group_id,
                    "cause": cause,
                    "cause_request_id": cause_request_id,
                    "cause_batch_id": cause_batch_id,
                    "cause_schedule_batch_id": cause_schedule_batch_id,
                    "event_hashes": self._wire_event_hashes(group_id, block_hashes),
                    "added": self._wire_hashes(delta.added),
                    "removed": self._wire_hashes(delta.removed),
                    "added_full_ready": self._wire_hashes(delta.added_full_ready),
                    "removed_full_ready": self._wire_hashes(delta.removed_full_ready),
                    "added_tail_ready": self._wire_hashes(delta.added_tail_ready),
                    "removed_tail_ready": self._wire_hashes(delta.removed_tail_ready),
                }
                self._bridge.log_cache_lifecycle_audit(
                    json.dumps(audit, separators=(",", ":"), sort_keys=True)
                )
                self._report_cache_delta(delta, alias_delta, cache_key_type)
            except Exception:
                logger.exception("failed to process vLLM block lifecycle event")

    def _report_cache_delta(self, delta, aliases=None, cache_key_type=0):
        if not delta.empty() or (aliases is not None and not aliases.empty()):
            fields = (
                "added",
                "removed",
                "added_full_ready",
                "removed_full_ready",
                "added_tail_ready",
                "removed_tail_ready",
            )
            self._bridge.report_cache_updated(
                *(
                    self._wire_hashes(
                        getattr(delta, name)
                        + (getattr(aliases, name) if aliases is not None else ())
                    )
                    for name in fields
                ),
                *((cache_key_type,) if cache_key_type else ()),
            )

    def on_request_lifecycle_event(self, event: dict) -> None:
        try:
            request_id = event.get("request_id")
            batch_id, schedule_batch_id = self._cause_batch_ids(request_id)
            self._bridge.log_request_lifecycle_audit(
                json.dumps(
                    {
                        **event,
                        "session": self._request_audit_session,
                        "audit_failures": self._request_audit_failures,
                        "batch_id": batch_id,
                        "schedule_batch_id": schedule_batch_id,
                    },
                    separators=(",", ":"),
                    sort_keys=True,
                )
            )
        except Exception:
            self._request_audit_failures += 1

    def _cause_batch_ids(
        self, cause_request_id: str | None
    ) -> tuple[int | None, int | None]:
        if cause_request_id is None:
            return None, None
        state = self._forward_request_state(str(cause_request_id))
        if state is None:
            return None, None
        return int(state.batch_id), int(state.schedule_batch_id)

    def _wire_event_hashes(
        self, group_id: int | None, block_hashes: tuple[bytes, ...]
    ) -> list[int]:
        if group_id is None:
            return []
        return self._wire_hashes(
            tuple(
                expanded_hash
                for block_hash in block_hashes
                for expanded_hash in self._cache_policy.expand_group_hash(
                    group_id, block_hash
                )
            )
        )

    @staticmethod
    def _wire_hashes(block_hashes: tuple[bytes, ...]) -> list[int]:
        return [block_hash_to_uint64(value) for value in block_hashes]

    def shutdown(self) -> None:
        with self._shutdown_lock:
            if self._shutdown_started:
                return
            self._shutdown_started = True
            self._shutdown_locked()

    def _shutdown_locked(self) -> None:
        self._forward_stop.set()
        try:
            if self._bridge is not None and not self._bridge.disable_forwarding():
                logger.error("failed to disable NaviSched VLLM forwarding")
        except Exception:
            logger.exception("failed to disable NaviSched VLLM forwarding")
        for thread in (self._forward_thread,):
            if thread is not None and thread is not threading.current_thread():
                thread.join()
        if self._v6d_snapshot_future is not None:
            self._v6d_snapshot_future.cancel()
            self._v6d_snapshot_future = None
        self._restore_v6d_publication_observers()
        for state in self._matching_forward_batches(None):
            self._fail_forward_responses(
                state.request_ids, "NaviSched vLLM Agent stopped"
            )
            self._release_forward_batch(state)
        self._preprocessed_cache_touches.clear()
        continuation = getattr(self, "_render_continuation", None)
        if continuation is not None:
            continuation.close()
        with self._cache_touch_lock:
            self._cache_touch_queue.clear()
        with self._response_lock:
            self._response_request_ids.clear()
            self._response_image_positions.clear()
            self._inactive_response_request_ids.clear()
            self._cancelled_before_decode.clear()
            self._pending_engine_abort_request_ids.clear()
        self._forward_request_states.clear()
        with self._workload_lock:
            self._workload_contributions.clear()
            self._workload_aggregate = _WorkloadAggregate()
            self._current_workload = None
            self._running_prefill_started_ns.clear()
            self._running_request_progress.clear()
            self._last_stalled_running_scan_ns = 0
            self._slow_engine_stage_last_log_ns.clear()
        with self._execution_lock:
            self._scheduled_work_by_output.clear()
            self._inflight_scheduled_tokens.clear()
            self._workload_dirty_request_ids.clear()
            self._active_scheduled_tokens = 0
        with self._pd_output_lock:
            self._pd_output_completed_ns.clear()
            self._pd_output_geometry.clear()
            self._pd_ready_finishes.clear()
        if self._agent is not None:
            try:
                self._agent.stop()
            except Exception:
                logger.exception("failed to stop NaviSched vLLM agent")
            finally:
                self._agent = None


def create_vllm_engine_observer(
    vllm_config,
    kv_cache_config,
    max_concurrent_batches: int,
):
    return VLLMEngineObserver(
        vllm_config,
        kv_cache_config,
        max_concurrent_batches,
    )
