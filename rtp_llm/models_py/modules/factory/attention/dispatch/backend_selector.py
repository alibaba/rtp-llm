"""Startup-time decode backend selection orchestration (GPU + NCCL).

Called by the engine per bs during capture: for that bs, real-machine benchmark
the candidate backends intersected with support (going through the real capture
path, see backend_bench), and pick the winner by the aggregation criterion.
Only rank0 benchmarks and selects, then broadcasts the winner's registry index
to Group.TP so every rank gets the same winner; otherwise each rank would pick
a different backend due to noise, bake different graphs, and NCCL would hang.
"""

from __future__ import annotations

import logging
import math
import os
from typing import List, NoReturn, Optional

import torch

from rtp_llm.models_py.modules.factory.attention.cuda_impl.benchmark_workspace import (
    benchmark_workspace_scope,
)
from rtp_llm.models_py.modules.factory.attention.dispatch import backend_bench
from rtp_llm.models_py.modules.factory.attention.dispatch.selector import (
    STABLE_CLUSTER_MARGIN,
    STABLE_THRESHOLD,
    Selector,
    kv_grid,
    select_stable,
)

logger = logging.getLogger(__name__)

_FATAL_PROBE_EXIT_CODE = 70


class DynamicDecodeFatalError(RuntimeError):
    """Raised only if the process exit primitive unexpectedly returns in tests."""


class _SelectorConfigError(ValueError):
    def __init__(self, variable: str, raw_value: str, reason: str):
        super().__init__(
            f"{variable}={raw_value!r} must be a finite non-negative float: {reason}"
        )
        self.variable = variable
        self.raw_value = raw_value
        self.reason = reason


class _FatalProbeError(RuntimeError):
    """Marks an exception raised after real on-device probing has started."""

    def __init__(self, backend: str):
        super().__init__(f"fatal decode backend probe failure: {backend}")
        self.backend = backend


def _terminate_probe_worker(bs: int, backend: str, error: BaseException) -> NoReturn:
    """Terminate without CUDA cleanup, TP broadcast, or in-process fallback."""
    cause = error.__cause__ or error
    logger.critical(
        "[dispatcher] bs=%d backend=%s fatal probe exception; terminating worker",
        bs,
        backend,
        exc_info=(type(cause), cause, cause.__traceback__),
    )
    os._exit(_FATAL_PROBE_EXIT_CODE)


def _terminate_config_worker(
    bs: int, error: _SelectorConfigError, parallelism_config
) -> NoReturn:
    logger.critical(
        "dynamic_decode_config_invalid variable=%s value=%r "
        "rule=finite_nonnegative_float bs=%d tp_rank=%d tp_size=%d dp_rank=%d "
        "reason=%s",
        error.variable,
        error.raw_value,
        bs,
        int(parallelism_config.tp_rank),
        int(parallelism_config.tp_size),
        int(parallelism_config.dp_rank),
        error.reason,
    )
    os._exit(_FATAL_PROBE_EXIT_CODE)


def _terminate_plan_application_worker(
    bs: int,
    backend: str,
    registry_idx: int,
    parallelism_config,
    stage: str,
    error: BaseException,
) -> NoReturn:
    logger.critical(
        "dynamic_decode_plan_apply_failed bs=%d backend=%s registry_idx=%d "
        "tp_rank=%d tp_size=%d dp_rank=%d stage=%s reason=%r",
        bs,
        backend,
        registry_idx,
        int(parallelism_config.tp_rank),
        int(parallelism_config.tp_size),
        int(parallelism_config.dp_rank),
        stage,
        error,
        exc_info=(type(error), error, error.__traceback__),
    )
    os._exit(_FATAL_PROBE_EXIT_CODE)


def _read_selector_config() -> tuple[float, float]:
    def read_one(variable: str, default: float) -> float:
        raw_value = os.environ.get(variable)
        if raw_value is None:
            return default
        try:
            value = float(raw_value)
        except (TypeError, ValueError) as error:
            raise _SelectorConfigError(variable, raw_value, str(error)) from error
        if not math.isfinite(value):
            raise _SelectorConfigError(variable, raw_value, "value is not finite")
        if value < 0:
            raise _SelectorConfigError(variable, raw_value, "value is negative")
        return value

    return (
        read_one("DYN_DECODE_THRESHOLD", STABLE_THRESHOLD),
        read_one("DYN_DECODE_CLUSTER_MARGIN", STABLE_CLUSTER_MARGIN),
    )


def _decode_registry() -> List[str]:
    """All-rank ordered registry class names (position is identity); DECODE_MHA_IMPS is already registered per device at import time."""
    from rtp_llm.models_py.modules.factory.attention.attn_factory import DECODE_MHA_IMPS

    return [c.__name__ for c in DECODE_MHA_IMPS]


def _tp_geometry(parallelism_config):
    """(tp_size, is_tp_root, src_global_rank). The TP group = the tp_size ranks with the same dp_rank,
    the global root = dp_rank*tp_size (broadcast src is a global rank, not literal 0).
    """
    tp_size = int(parallelism_config.tp_size)
    tp_rank = int(parallelism_config.tp_rank)
    dp_rank = int(parallelism_config.dp_rank)
    return tp_size, (tp_rank == 0), dp_rank * tp_size


def _eligible(
    attn_configs, attn_inputs, parallelism_config, fmha_config=None
) -> List[str]:
    from rtp_llm.models_py.modules.factory.attention.attn_factory import (
        DECODE_MHA_IMPS,
        _is_fmha_impl_disabled,
    )

    names: List[str] = []
    # Registry contract: a decode implementation must be registered for dynamic
    # probing only after its CUDA Graph decode path, lack of TP collectives,
    # numerical behavior, and benchmark resource lifetime have been validated.
    probe_started = False
    probe_backend = "support-probe"
    with benchmark_workspace_scope():
        for impl in DECODE_MHA_IMPS:
            try:
                name = impl.__name__
                probe_backend = name
                # Respect fmha_config disable flags (disable_flash_infer / enable_xqa /
                # ...), mirroring the fixed-priority get_fmha_impl path.
                if _is_fmha_impl_disabled(name, fmha_config):
                    continue
                probe_started = True
                if not impl.support(attn_configs, attn_inputs):
                    continue
                if not impl.support_parallelism_config(parallelism_config):
                    continue
            except Exception as e:
                if probe_started:
                    # support() may allocate and probe the GPU (TRT-LLM Gen does),
                    # so every later exception is past the safe fallback boundary.
                    raise _FatalProbeError(probe_backend) from e
                raise
            names.append(name)
    return names


def run_backend_selection(
    model,
    inputs,
    *,
    selector: Optional[Selector] = None,
    warmup: int = 10,
    iters: int = 50,
    l2_fill_mode: str = "store",
) -> Optional[str]:
    """Select the decode backend for a single capture bs, returning the winner class name (None = left empty, capture falls back to fixed priority).

    Only rank0 does the real-machine benchmark + selection; the winner registry
    index is broadcast to Group.TP so every rank gets the same winner.
    """
    parallelism_config = model.parallelism_config
    attn_configs = model.config.getAttentionConfigs(
        parallelism_config.get_attn_tp_size()
    )
    attn_inputs = inputs.attention_inputs
    bs = int(attn_inputs.input_lengths.size(0))
    max_seq_len = int(getattr(model.config, "max_seq_len", 0)) or int(
        attn_configs.max_seq_len
    )
    grid = kv_grid(max_seq_len)
    registry = _decode_registry()
    tp_size, is_tp_root, src = _tp_geometry(parallelism_config)

    try:
        code = torch.full((1,), -1, dtype=torch.int32, device="cuda")
    except Exception as e:
        # Every rank must reach the same TP rendezvous. A rank-local CUDA
        # allocation failure cannot fall back while its peers enter broadcast.
        _terminate_probe_worker(bs, "selection-control", e)
        raise RuntimeError("fatal probe termination returned unexpectedly") from e
    if is_tp_root:
        # Contract: benchmarked decode attention forward must be free of collective
        # communication (allreduce/broadcast). rank0 runs bench alone; a backend
        # whose forward triggers TP collectives would deadlock here.
        try:
            winner = _select_on_root(
                model,
                attn_configs,
                attn_inputs,
                parallelism_config,
                grid,
                selector,
                bs,
                max_seq_len,
                warmup,
                iters,
                l2_fill_mode,
            )
            if winner is not None and winner in registry:
                try:
                    code[0] = registry.index(winner)
                except Exception as e:
                    # The winner write is the final CUDA operation after a real
                    # probe. Do not enter a TP collective if it fails.
                    raise _FatalProbeError(winner) from e
        except _SelectorConfigError as e:
            _terminate_config_worker(bs, e, parallelism_config)
            raise DynamicDecodeFatalError(
                "fatal configuration termination returned unexpectedly"
            ) from e
        except _FatalProbeError as e:
            _terminate_probe_worker(bs, e.backend, e)
            raise DynamicDecodeFatalError(
                "fatal probe termination returned unexpectedly"
            ) from e
        except Exception as e:
            # Pre-probe orchestration errors remain normal plan misses. Root must
            # still broadcast the empty code so all healthy ranks stay in lockstep.
            logger.warning(
                "[dispatcher] bs=%d root selection exception, no dynamic plan: %r",
                bs,
                e,
            )

    try:
        if tp_size > 1:
            from rtp_llm.models_py.distributed.collective_torch import Group, broadcast

            # src is a global rank (dp_rank * tp_size).
            broadcast(code, src, group=Group.TP)

        idx = int(code[0].item())
    except Exception as e:
        # NCCL and CUDA readback failures may leave the process group or device
        # unusable. Never let the outer capture path turn them into a local
        # fixed-priority fallback.
        _terminate_probe_worker(bs, "selection-control", e)
        raise DynamicDecodeFatalError(
            "fatal probe termination returned unexpectedly"
        ) from e
    if idx == -1:
        chosen = None
    elif 0 <= idx < len(registry):
        chosen = registry[idx]
    else:
        error = ValueError(
            f"broadcast registry index {idx} is outside [0, {len(registry)})"
        )
        _terminate_plan_application_worker(
            bs,
            "<invalid-registry-index>",
            idx,
            parallelism_config,
            "registry-index",
            error,
        )
        raise DynamicDecodeFatalError(
            "fatal plan application termination returned unexpectedly"
        ) from error
    if chosen is not None:
        logger.info(
            "dynamic_decode_plan_received bs=%d registry_idx=%d backend=%s "
            "tp_rank=%d tp_size=%d dp_rank=%d",
            bs,
            idx,
            chosen,
            int(parallelism_config.tp_rank),
            tp_size,
            int(parallelism_config.dp_rank),
        )
    logger.debug(
        "[dispatcher] bs=%d -> %s",
        bs,
        chosen or "(left empty, fall back to fixed priority)",
    )
    return chosen


def _select_on_root(
    model,
    attn_configs,
    attn_inputs,
    parallelism_config,
    grid,
    selector,
    bs,
    max_seq_len,
    warmup,
    iters,
    l2_fill_mode,
) -> Optional[str]:
    # Hybrid models are rejected by GptModelBase.select_decode_backend because
    # their multi-group cache layout is not supported by this benchmark.
    layer_kv_cache = model.kv_cache.get_layer_cache(0)
    # clip the kv grid to the upper bound of the engine's capture sequence_lengths (real-machine dump: = max_seq_len-2;
    # the page table only covers up to here, exceeding it goes out of bounds, see backend_bench._set_uniform_seq_len).
    seq_cap = (
        int(attn_inputs.sequence_lengths.flatten()[0].item())
        if attn_inputs.sequence_lengths.numel()
        else max_seq_len
    )
    kv_list = [kv for kv in grid if kv <= seq_cap] or [seq_cap]

    selector_config = _read_selector_config() if selector is None else None

    # Everything above this point is CPU-side preparation. support() may allocate
    # or launch GPU work, so after _eligible starts there is no safe in-process
    # fallback or TP broadcast path for an unexpected exception.
    eligible = _eligible(
        attn_configs, attn_inputs, parallelism_config, model.fmha_config
    )
    probe_backend = eligible[0] if eligible else "support-probe"
    try:
        matrix = {}
        last_probed_backend = None
        for name in eligible:
            probe_backend = name
            impl_cls = _impl_by_name(name)
            if impl_cls is None:
                continue
            # multi-kv-point: for each kv, set sequence_lengths on a clone and time real-machine capture+replay;
            # the per-kv latencies are collected into `matrix`, then select_stable (below) does two-level
            # deterministic selection to pick this bs bucket's backend.
            # bench_backend_grid owns the one real benchmark instance and checks
            # support_cuda_graph() on it before prepare/forward.
            last_probed_backend = name
            lats = backend_bench.bench_backend_grid(
                impl_cls,
                attn_configs,
                attn_inputs,
                layer_kv_cache,
                parallelism_config,
                kv_list,
                attention_layer_count=int(model.layer_num),
                warmup=warmup,
                iters=iters,
                l2_fill_mode=l2_fill_mode,
            )
            if lats is not None:
                matrix[name] = lats
        if last_probed_backend is not None:
            # Return only after all benchmark work finishes, before real capture.
            torch.cuda.synchronize()

        if not matrix:
            reason = "eligible empty" if not eligible else "eligible all N/A"
            logger.warning(
                "[dispatcher] bs=%d %s -> no dynamic plan",
                bs,
                reason,
            )
            return None

        # Use select_stable by default: two-level deterministic selection
        # (threshold filter + registry-order tiebreaker) for CI and production stability.
        if selector is not None:
            choice = selector(matrix)
        else:
            assert selector_config is not None
            threshold, cluster_margin = selector_config
            registry = _decode_registry()
            # Use registry priority only among candidates whose real benchmark
            # completed successfully. Do not construct another backend merely to
            # discover the fixed-priority baseline.
            default_backend = next((name for name in registry if name in matrix), None)
            choice = select_stable(
                matrix, registry, default_backend, threshold, cluster_margin
            )

        if choice is not None:
            return choice
        logger.warning(
            "[dispatcher] bs=%d selection returned None -> no dynamic plan", bs
        )
        return None
    except _FatalProbeError:
        raise
    except Exception as e:
        raise _FatalProbeError(probe_backend) from e


def _impl_by_name(name: str):
    from rtp_llm.models_py.modules.factory.attention.attn_factory import DECODE_MHA_IMPS

    for c in DECODE_MHA_IMPS:
        if c.__name__ == name:
            return c
    return None


def instantiate_decode_impl(model, attn_inputs, name: str, is_cuda_graph: bool):
    """Instantiate an already-broadcast winner or terminate the worker."""
    parallelism_config = model.parallelism_config
    bs = int(attn_inputs.input_lengths.size(0))
    try:
        registry_idx = _decode_registry().index(name)
    except Exception:  # noqa: BLE001 -- failure is reported by the stage below
        registry_idx = -1

    def fail(stage: str, error: BaseException) -> NoReturn:
        _terminate_plan_application_worker(
            bs,
            name,
            registry_idx,
            parallelism_config,
            stage,
            error,
        )
        raise DynamicDecodeFatalError(
            "fatal plan application termination returned unexpectedly"
        ) from error

    try:
        cls = _impl_by_name(name)
    except Exception as error:  # noqa: BLE001
        fail("class-lookup", error)
    if cls is None:
        fail("class-missing", LookupError(f"backend class {name!r} is not registered"))

    try:
        from rtp_llm.models_py.modules.factory.attention.attn_factory import (
            _is_fmha_impl_disabled,
        )
    except Exception as error:  # noqa: BLE001
        fail("disable-check", error)

    try:
        if _is_fmha_impl_disabled(name, getattr(model, "fmha_config", None)):
            fail("disabled", RuntimeError(f"backend {name!r} is disabled"))
        attn_configs = model.config.getAttentionConfigs(
            parallelism_config.get_attn_tp_size()
        )
        attn_inputs.is_cuda_graph = is_cuda_graph
        attn_inputs.headwise_config = getattr(model.config, "headwise_config", None)
    except DynamicDecodeFatalError:
        raise
    except Exception as error:  # noqa: BLE001
        fail("input-configuration", error)

    try:
        if not cls.support(attn_configs, attn_inputs):
            fail("support", RuntimeError(f"backend {name!r} support() returned false"))
    except DynamicDecodeFatalError:
        raise
    except Exception as error:  # noqa: BLE001
        fail("support", error)
    try:
        if not cls.support_parallelism_config(parallelism_config):
            fail(
                "parallelism",
                RuntimeError(
                    f"backend {name!r} support_parallelism_config() returned false"
                ),
            )
    except DynamicDecodeFatalError:
        raise
    except Exception as error:  # noqa: BLE001
        fail("parallelism", error)
    try:
        inst = cls(attn_configs, attn_inputs, parallelism_config)
    except Exception as error:  # noqa: BLE001
        fail("constructor", error)
    if is_cuda_graph:
        try:
            graph_supported = inst.support_cuda_graph()
        except Exception as error:  # noqa: BLE001
            fail("cuda-graph-support", error)
        if not graph_supported:
            fail(
                "cuda-graph-support",
                RuntimeError(f"backend {name!r} does not support CUDA Graph"),
            )
    return inst
