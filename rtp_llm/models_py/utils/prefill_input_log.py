"""Host-only input metadata for the latest Qwen3.5 language prefill.

No tensor values, CUDA synchronization, or global dispatch/launch hooks.
"""

from __future__ import annotations

import contextvars
import inspect
import itertools
import json
import logging
import math
import os
import socket
import tempfile
import time
from contextlib import contextmanager
from pathlib import Path

import torch
import torch.distributed as dist
from torch.utils._python_dispatch import TorchDispatchMode

_CURRENT = contextvars.ContextVar("prefill_input_log", default=None)
_SCOPE = contextvars.ContextVar("prefill_input_scope", default=("prepare", None))
_PARENT = contextvars.ContextVar("prefill_input_parent", default=None)
_DESCRIBING = contextvars.ContextVar("prefill_input_describing", default=False)
_SEQUENCE = itertools.count(1)

# Only read known metadata fields. Never evaluate arbitrary properties or repr.
_FIELDS = {
    "PyModelInputs": (
        "input_ids",
        "input_hiddens",
        "combo_position_ids",
        "attention_inputs",
        "embedding_inputs",
        "multimodal_inputs",
    ),
    "PyAttentionInputs": (
        "is_prefill",
        "is_cuda_graph",
        "is_target_verify",
        "prefix_lengths",
        "sequence_lengths",
        "input_lengths",
        "kv_cache_kernel_block_id",
        "kv_cache_kernel_block_id_device",
        "kv_cache_block_id",
        "kv_cache_block_id_device",
        "dtype",
        "cu_seqlens_device",
        "cu_seqlens",
        "cu_kv_seqlens_device",
        "context_total_kv_length",
        "total_tokens",
        "padding_offset",
        "is_s_padded",
        "prefix_lengths_device",
        "sequence_lengths_plus_1_device",
        "input_lengths_device",
        "decode_cu_seqlens_device",
        "decode_cu_seqlens",
        "cache_store_inputs",
        "cache_store_writer",
        "context_parallel_info",
        "combo_position_ids",
        "prefill_cuda_graph_copy_params",
        "headwise_config",
    ),
    "LayerKVCache": (
        "kv_cache_base",
        "kv_scale_base",
        "seq_size_per_block",
        "layer_id",
        "group_id",
        "tag",
    ),
    "CausalConv1dMetadata": ("batch_ptr", "token_chunk_offset_ptr", "total"),
    "Qwen3NextMetadata": (
        "prefill_conv1d_meta",
        "is_target_verify",
        "full_prefill_conv1d_meta",
        "full_prefill_cu_seqlens",
        "cp_restore_indices",
        "cp_local_extract_indices",
        "cp_local_valid_mask",
        "flashinfer_prefill_metadata",
    ),
    "ExpertGatePayload": ("scores", "topk", "score_func", "route_scale"),
    "PyEmbeddingInputs": ("combo_tokens_type_ids", "text_tokens_mask"),
    "PyMultimodalInputs": ("multimodal_features", "mm_features_locs", "mm_extra_input"),
    "PyContextParallelParams": (
        "prefill_cp_padding_lengths",
        "prefill_cp_chunk_lengths",
        "prefill_shuffle_indices",
        "prefill_qkv_restore_indice",
        "prefill_qkv_padding_mask",
        "prefill_actual_input_lengths_cpu",
    ),
    "PyPrefillCudaGaphCopyParams": (
        "cuda_graph_prefill_batch_size",
        "max_seq_len",
        "max_batch_size",
    ),
    "FlashInferPrefillMetadata": (
        "source_cu",
        "cu_seqlens",
        "checkpoint_starts",
        "checkpoint_interval",
        "total_tokens",
        "checkpoint_capacity",
    ),
    "SymmBuffer": (
        "x",
        "x_sf",
        "topk_idx",
        "topk_weights",
        "buffer",
        "shared_l1_acts",
        "shared_l1_acts_sf",
        "shared_l2_acts",
        "shared_l2_acts_sf",
        "l1_acts",
        "l1_acts_sf",
        "l2_acts",
        "l2_acts_sf",
        "num_experts",
        "num_max_tokens_per_rank",
        "num_topk",
        "hidden",
        "intermediate_hidden",
    ),
    "FlashInferMlaAttnParams": (
        "batch_indice_h",
        "page_indice_h",
        "reuse_cache_page_indice_h",
        "decode_page_indptr_h",
        "prefill_ragged_kv_len_indptr_h",
        "paged_kv_last_page_len_h",
        "qo_indptr_h",
        "kvlen_h",
        "positions_h",
        "batch_reuse_info_vec_h",
        "batch_indice_d",
        "page_indice_d",
        "reuse_cache_page_indice_d",
        "decode_page_indptr_d",
        "prefill_ragged_kv_len_indptr_d",
        "paged_kv_last_page_len_d",
        "qo_indptr_d",
        "kvlen_d",
        "positions_d",
        "batch_reuse_info_vec_d",
        "slot_mapping",
    ),
    "XQAParams": ("kv_cache_offset",),
}


def _type_name(value):
    cls = type(value)
    return f"{cls.__module__}.{cls.__qualname__}"


def describe_inputs(value, *, typed_scalars=True):
    """Describe inputs without reading tensor contents, including CPU tensors."""
    seen = set()

    def describe(item):
        if isinstance(item, torch.Tensor):
            result = {
                "type": _type_name(item),
                "shape": tuple(item.shape),
                "dtype": str(item.dtype),
                "device": str(item.device),
                "layout": str(item.layout),
                "numel": item.numel(),
            }
            if item.layout == torch.strided:
                result.update(
                    stride=tuple(item.stride()),
                    contiguous=item.is_contiguous(),
                    storage_offset=item.storage_offset(),
                )
                if item.device.type != "meta" and not hasattr(item, "fake_mode"):
                    try:
                        result["data_ptr"] = item.data_ptr()
                    except (RuntimeError, NotImplementedError):
                        result["data_ptr"] = None
            return result
        if item is None or isinstance(item, (bool, int, float, str)):
            scalar = (
                str(item)
                if isinstance(item, float) and not math.isfinite(item)
                else item
            )
            return (
                {"type": type(item).__name__, "value": scalar}
                if typed_scalars
                else scalar
            )
        if isinstance(item, (torch.dtype, torch.device, torch.layout)):
            return {"type": _type_name(item), "value": str(item)}
        identity = id(item)
        if identity in seen:
            return {"type": _type_name(item), "cycle": True}
        seen.add(identity)
        try:
            if isinstance(item, (tuple, list)):
                return [describe(child) for child in item]
            if isinstance(item, dict):
                if all(isinstance(key, str) for key in item):
                    return {key: describe(child) for key, child in item.items()}
                return {
                    "type": "dict",
                    "items": [
                        {"key": describe(key), "value": describe(child)}
                        for key, child in item.items()
                    ],
                }
            result = {"type": _type_name(item)}
            fields = _FIELDS.get(type(item).__name__, ())
            if fields:
                values = {}
                for name in fields:
                    try:
                        values[name] = describe(getattr(item, name))
                    except AttributeError:
                        continue
                    except Exception as error:
                        values[name] = {"metadata_error": _type_name(error)}
                result["fields"] = values
            return result
        finally:
            seen.remove(identity)

    token = _DESCRIBING.set(True)
    try:
        return describe(value)
    finally:
        _DESCRIBING.reset(token)


class _Recorder:
    def __init__(self):
        self.output = None
        self.enabled = True
        self.sequence = next(_SEQUENCE)
        self.index = itertools.count(1)
        self.rank = (
            dist.get_rank() if dist.is_available() and dist.is_initialized() else 0
        )
        self.host = socket.gethostname().replace("/", "_")
        self.pid = os.getpid()
        self.path = (
            Path(
                os.environ.get(
                    "MEGA_MOE_SNAPSHOT_DIR", f"/tmp/mega_moe_snapshots_{os.getuid()}"
                )
            )
            / f"prefill_ops_{self.host}_rank{self.rank}_pid{self.pid}.jsonl"
        )
        temp = None
        try:
            self.path.parent.mkdir(parents=True, exist_ok=True)
            fd, temp = tempfile.mkstemp(prefix=".prefill_ops_", dir=self.path.parent)
            self.output = os.fdopen(fd, "w", encoding="utf-8")
            self.write("prefill_start", state="running", schema_version=1)
            if self.enabled:
                os.replace(temp, self.path)
        except Exception:
            self.disable()
        finally:
            if temp is not None:
                try:
                    os.unlink(temp)
                except FileNotFoundError:
                    pass
                except OSError:
                    self.disable()

    def disable(self):
        if self.enabled:
            self.enabled = False
            logging.exception(
                "Could not record prefill input metadata at %s", self.path
            )

    def write(self, event, **fields):
        if not self.enabled:
            return
        try:
            stage, layer = _SCOPE.get()
            record = dict(
                event=event,
                prefill_sequence=self.sequence,
                time_ns=time.time_ns(),
                host=self.host,
                rank=self.rank,
                pid=self.pid,
                stage=stage,
                layer_id=layer,
                **fields,
            )
            self.output.write(
                json.dumps(
                    record, ensure_ascii=True, allow_nan=False, separators=(",", ":")
                )
                + "\n"
            )
            self.output.flush()
        except Exception:
            self.disable()

    def close(self):
        if self.output is not None:
            try:
                self.output.close()
            except Exception:
                self.disable()


def _call(name, backend, function, args, kwargs, extra=None):
    recorder = _CURRENT.get()
    if recorder is None or not recorder.enabled or _DESCRIBING.get():
        return function(*args, **kwargs)
    index = next(recorder.index)
    parent = _PARENT.get()
    try:
        inputs = {"args": args, "kwargs": kwargs}
        # Original invocation owns argument validation, including invalid inputs.
        try:
            bound = inspect.signature(function).bind(*args, **kwargs)
            bound.apply_defaults()
            inputs = dict(bound.arguments)
            if backend == "torch_dispatch":
                inputs = {
                    parameter.name: (
                        args[i]
                        if i < len(args)
                        else kwargs.get(parameter.name, parameter.default_value)
                    )
                    for i, parameter in enumerate(function._schema.arguments)
                }
        except (TypeError, ValueError):
            pass
        recorder.write(
            "call",
            call_id=index,
            parent_call_id=parent,
            op=name,
            backend=backend,
            inputs=describe_inputs(inputs),
            launch=describe_inputs(extra),
        )
    except Exception:
        recorder.disable()
    token = _PARENT.set(index)
    try:
        result = function(*args, **kwargs)
    except BaseException as error:
        recorder.write(
            "error",
            call_id=index,
            parent_call_id=parent,
            op=name,
            backend=backend,
            exception_type=_type_name(error),
        )
        raise
    else:
        recorder.write(
            "return", call_id=index, parent_call_id=parent, op=name, backend=backend
        )
        return result
    finally:
        _PARENT.reset(token)


def trace_call(name, function, /, *args, **kwargs):
    """Record a repository native/library boundary; pass through when disabled."""
    if _CURRENT.get() is None:
        return function(*args, **kwargs)
    backend = getattr(function, "__module__", None) or _type_name(function)
    return _call(name, backend, function, args, kwargs)


def trace_triton(name, kernel, grid, /, *args, **kwargs):
    """Record launch arguments and resolved grids without extra grid evaluation."""
    recorder = _CURRENT.get()
    if recorder is None or not recorder.enabled:
        return kernel[grid](*args, **kwargs)
    launch_grid = grid
    if callable(grid):

        def launch_grid(meta):
            resolved = grid(meta)
            try:
                recorder.write(
                    "triton_grid",
                    call_id=_PARENT.get(),
                    op=name,
                    grid=describe_inputs(resolved),
                    meta=describe_inputs(meta),
                )
            except Exception:
                recorder.disable()
            return resolved

    names = getattr(kernel, "arg_names", ())
    launch = {"grid": grid, "arg_names": names, "options": kwargs}
    return _call(name, "triton", kernel[launch_grid], args, kwargs, launch)


class _InputMode(TorchDispatchMode):
    def __torch_dispatch__(self, func, types, args=(), kwargs=None):
        return _call(str(func), "torch_dispatch", func, args, kwargs or {})


@contextmanager
def prefill_stage(stage, layer_id=None):
    if _CURRENT.get() is None:
        yield
        return
    token = _SCOPE.set((stage, layer_id))
    try:
        yield
    finally:
        _SCOPE.reset(token)


@contextmanager
def prefill_input_snapshot(is_prefill):
    if not is_prefill or os.environ.get("MEGA_MOE_LOG_INPUTS") != "1":
        yield
        return
    if _CURRENT.get() is not None:
        yield
        return
    scope_token = _SCOPE.set(("prepare", None))
    parent_token = _PARENT.set(None)
    try:
        recorder = _Recorder()
    except Exception:
        _SCOPE.reset(scope_token)
        _PARENT.reset(parent_token)
        logging.exception("Could not initialize prefill input recorder")
        yield
        return
    token = _CURRENT.set(recorder)
    try:
        with _InputMode():
            try:
                yield
            except BaseException as error:
                recorder.write(
                    "prefill_end",
                    state="python_forward_failed",
                    exception_type=_type_name(error),
                )
                raise
            else:
                recorder.write("prefill_end", state="python_forward_returned")
    finally:
        _CURRENT.reset(token)
        _SCOPE.reset(scope_token)
        _PARENT.reset(parent_token)
        recorder.close()
