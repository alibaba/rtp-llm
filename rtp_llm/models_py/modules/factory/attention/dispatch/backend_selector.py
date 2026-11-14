"""Select a precision-qualified backend once per decode graph batch bucket."""

import logging
import math
import os

import torch

from rtp_llm.config.cuda_graph import CudaGraphSelectionMode
from rtp_llm.models_py.distributed.collective_torch import Group, all_reduce, broadcast
from rtp_llm.models_py.modules.factory.attention.attn_factory import (
    DECODE_MHA_IMPS,
    _implementation_allows_cuda_graph_selection_mode,
    _is_fmha_impl_disabled,
)
from rtp_llm.models_py.modules.factory.attention.dispatch.backend_bench import (
    bench_backend_grid,
)
from rtp_llm.models_py.modules.factory.attention.dispatch.selector import (
    kv_grid,
    select_stable,
)
from rtp_llm.models_py.utils.arch import get_sm


def _threshold(name):
    value = float(os.environ.get(name, "0.05"))
    if not math.isfinite(value) or value < 0:
        raise ValueError(f"{name} must be finite and non-negative")
    return value


def _fatal(error):
    # A CUDA probe or post-broadcast failure cannot enter rank-local fallback.
    # Exiting lets the worker supervisor tear down peers waiting in NCCL.
    logging.critical("dynamic decode startup failed", exc_info=error)
    os._exit(70)


def benchmark_sequence_cap(
    model_max_seq_len, batch, page_size, cache_pages, page_bytes, free_bytes
):
    """Return the largest KV point safe for this batch's benchmark cache.

    The result is a benchmark allocation limit only. It is deliberately not
    published to replay and therefore cannot turn calibration coverage into a
    runtime Graph guard.
    """
    if model_max_seq_len <= 0 or batch <= 0 or page_size <= 0 or page_bytes <= 0:
        return -1
    # Reserve workspaces/L2 fill and the temporary benchmark allocation.
    budget = max(0, free_bytes - 512 * 1024**2) // 4
    pages = min(cache_pages, budget // page_bytes) // batch
    return min(model_max_seq_len, pages * page_size - 1)


def dynamic_backend_allowed(backend_name, sm_major):
    """Return whether a backend participates in dynamic decode on this SM family."""
    return not (backend_name == "XQADecodeImpl" and sm_major == 10)


def select_decode_backend(model, attention_inputs, gate_passed):
    if not gate_passed:
        raise RuntimeError("dynamic decode requires a non-empty precision gate")
    parallel = model.parallelism_config
    registry = list(dict.fromkeys(DECODE_MHA_IMPS))
    configs = model.config.getAttentionConfigs(parallel.get_attn_tp_size())
    batch = attention_inputs.input_lengths.numel()
    sm_major = get_sm()[0]
    try:
        threshold = _threshold("DYN_DECODE_THRESHOLD")
        margin = _threshold("DYN_DECODE_CLUSTER_MARGIN")
        supported = [
            cls
            for cls in registry
            if cls in gate_passed
            and dynamic_backend_allowed(cls.__name__, sm_major)
            and _implementation_allows_cuda_graph_selection_mode(
                cls, CudaGraphSelectionMode.DECODE_GRAPH
            )
            and not _is_fmha_impl_disabled(cls.__name__, model.fmha_config)
            and cls.support(configs, attention_inputs)
            and cls.support_parallelism_config(parallel)
        ]
        mask = torch.tensor(
            [int(cls in supported) for cls in registry],
            dtype=torch.int32,
            device="cuda",
        )
        if parallel.tp_size > 1:
            all_reduce(mask, group=Group.TP)
        eligible = [
            cls
            for cls, count in zip(registry, mask.tolist())
            if count == parallel.tp_size
        ]
        # -1 is a rank-consistent soft-fallback sentinel. It means that this
        # graph bucket keeps the ordinary static factory; it is not a CUDA
        # failure. Hard probe/benchmark errors still enter _fatal below.
        code = torch.full((1,), -1, dtype=torch.int32, device="cuda")
        if not eligible:
            if parallel.tp_size > 1:
                broadcast(code, parallel.dp_rank * parallel.tp_size, group=Group.TP)
            return None
        if parallel.tp_rank == 0:
            caches = model.kv_cache.get_layer_cache_groups(0)
            if len(caches) != 1:
                raise RuntimeError("dynamic decode requires one KV cache group")
            torch.cuda.empty_cache()
            free_bytes, _ = torch.cuda.mem_get_info()
            base = caches[0].kv_cache_base
            max_seq_len = int(
                getattr(configs, "max_seq_len", 0)
                or getattr(model.config, "max_seq_len", 0)
            )
            if max_seq_len <= 0:
                raise RuntimeError("dynamic decode requires a positive model max_seq_len")
            cap = benchmark_sequence_cap(
                max_seq_len,
                batch,
                configs.kernel_tokens_per_block,
                base.shape[0],
                base[0].numel() * base.element_size(),
                free_bytes,
            )
            grid = kv_grid(cap) if cap > 0 else []
            matrix = {}
            for cls in eligible if grid else []:
                times = bench_backend_grid(
                    cls,
                    configs,
                    attention_inputs,
                    caches[0],
                    parallel,
                    grid,
                    attention_layer_count=model.layer_num,
                )
                if times is not None:
                    if len(times) != len(grid) or not all(
                        math.isfinite(t) and t > 0 for t in times
                    ):
                        raise RuntimeError(
                            f"invalid benchmark results for {cls.__name__}"
                        )
                    matrix[cls.__name__] = times
            names = [cls.__name__ for cls in registry]
            baseline = next((name for name in names if name in matrix), None)
            selected = select_stable(matrix, names, baseline, threshold, margin)
            if selected is not None:
                code[0] = names.index(selected)
            logging.info(
                "dynamic decode benchmark batch=%d kv=%s timings=%s winner=%s",
                batch,
                grid,
                matrix,
                selected,
            )
        if parallel.tp_size > 1:
            broadcast(code, parallel.dp_rank * parallel.tp_size, group=Group.TP)
        index = int(code.item())
        if index == -1:
            return None
        if index < 0 or index >= len(registry) or registry[index] not in eligible:
            raise RuntimeError("broadcast backend is not eligible on this rank")
        return registry[index]
    except Exception as error:
        _fatal(error)
        raise


def instantiate_decode_backend(model, attention_inputs, cls):
    try:
        attention_inputs.is_cuda_graph = True
        configs = model.config.getAttentionConfigs(
            model.parallelism_config.get_attn_tp_size()
        )
        kwargs = {"fmha_config": model.fmha_config} if cls.accepts_fmha_config else {}
        instance = cls(configs, attention_inputs, model.parallelism_config, **kwargs)
        if not instance.support_cuda_graph():
            raise RuntimeError(f"{cls.__name__} cannot capture CUDA graphs")
        return instance
    except Exception as error:
        _fatal(error)
        raise
