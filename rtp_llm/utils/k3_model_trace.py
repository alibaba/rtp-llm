"""RTP K3 module boundaries with explicit CUDA Graph capture/replay ownership.

Module hooks cover executed nn.Module boundaries. Fused internal values require
explicit record_model() calls; the module inventory is not a coverage verdict.
"""

from __future__ import annotations

import atexit
import os
import socket
import threading
from functools import wraps
from pathlib import Path

import torch

from rtp_llm.utils.k3_tensor_trace import TensorTrace, enabled

_lock = threading.RLock()
_local = threading.local()
_managers = []


def _stack():
    if not hasattr(_local, "stack"):
        _local.stack = []
    return _local.stack


def _tensor_tree(name, value):
    if isinstance(value, torch.Tensor):
        yield name, value
    elif isinstance(value, (tuple, list)):
        for index, child in enumerate(value):
            yield from _tensor_tree(f"{name}.{index}", child)
    elif isinstance(value, dict):
        for key, child in value.items():
            if not isinstance(key, (str, int)):
                raise TypeError("trace tensor-tree keys must be strings or integers")
            yield from _tensor_tree(f"{name}.{key}", child)
    elif hasattr(value, "hidden_states"):
        # PyModelOutputs: do not traverse attention handles or cache pointers.
        yield from _tensor_tree(f"{name}.hidden_states", value.hidden_states)


def record_model(name, value, *, assert_zero=False):
    """Snapshot a named intermediate into the current model/capture frame."""
    if not enabled() or getattr(_local, "suspended", 0):
        return
    stack = _stack()
    if not stack:
        raise RuntimeError(f"K3 model observation outside model frame: {name}")
    trace = stack[-1]
    for tensor_name, tensor in _tensor_tree(name, value):
        trace.record(tensor_name, tensor, assert_zero=assert_zero)


def record_module(module, name, value, *, assert_zero=False):
    if not enabled() or getattr(_local, "suspended", 0):
        return
    record_model(f"{module._k3_trace_path}.{name}", value, assert_zero=assert_zero)


def record_module_cache_pages(module, name, cache, block_map, logical_pages):
    """Snapshot selected physical pages; invalid entries carry an explicit mask.

    Selection has a fixed shape and stays on device so capture/replay observes
    the current block map. Values in invalid slots are zero placeholders.
    """
    if not enabled() or getattr(_local, "suspended", 0):
        return
    in_map = (logical_pages >= 0) & (logical_pages < block_map.shape[1])
    physical_pages = block_map.gather(
        1, logical_pages.clamp(0, block_map.shape[1] - 1).long()
    ).long()
    physical_pages = torch.where(in_map, physical_pages, -1)
    valid = in_map & (physical_pages > 0) & (physical_pages < cache.shape[0])
    values = cache.index_select(
        0, physical_pages.clamp(0, cache.shape[0] - 1).flatten()
    ).reshape(*logical_pages.shape, *cache.shape[1:])
    values = torch.where(
        valid.reshape(*valid.shape, *([1] * (cache.ndim - 1))), values, 0
    )
    record_module(
        module,
        name,
        {
            "logical_pages": logical_pages,
            "physical_pages": physical_pages,
            "valid_pages": valid,
            "values": values,
        },
    )


def record_model_inputs(name, inputs):
    if not enabled() or getattr(_local, "suspended", 0):
        return
    metadata, tensors = model_inputs(inputs)
    if not _stack():
        raise RuntimeError("model input observation outside model frame")
    for tensor_name, value in tensors.items():
        _stack()[-1].record(f"{name}.{tensor_name}", value, **metadata)


def model_inputs(inputs):
    """Only read host scalars and tensor handles; never copy GPU values to host."""
    tensors = {}
    for field in ("input_ids", "input_hiddens", "combo_position_ids"):
        value = getattr(inputs, field, None)
        if isinstance(value, torch.Tensor):
            tensors[f"input.{field}"] = value
    attn = getattr(inputs, "attention_inputs", None)
    metadata = {}
    if attn is not None:
        for field in (
            "is_prefill",
            "is_target_verify",
            "is_mtp_draft_update",
            "is_prefill_chunk",
            "is_fake_stream",
            "is_cuda_graph",
            "total_tokens",
        ):
            value = getattr(attn, field, None)
            if isinstance(value, (bool, int)):
                metadata[field] = value
        for field in (
            "input_lengths",
            "sequence_lengths",
            "prefix_lengths",
            "cu_seqlens",
            "cu_seqlens_host",
            "kv_cache_block_id_host",
            "kv_cache_block_id_device",
            "kv_cache_kernel_block_id_host",
            "kv_cache_kernel_block_id_device",
            "kv_cache_layer_to_group",
            "kv_cache_layer_to_group_host",
            "input_lengths_host",
            "sequence_lengths_host",
            "prefix_lengths_host",
        ):
            value = getattr(attn, field, None)
            if isinstance(value, torch.Tensor):
                tensors[f"input.attention.{field}"] = value
        for field in (
            "kv_cache_block_id_host_by_group",
            "kv_cache_kernel_block_id_host_by_group",
            "kv_cache_kernel_block_id_device_by_group",
        ):
            for index, value in enumerate(getattr(attn, field, ())):
                if isinstance(value, torch.Tensor):
                    tensors[f"input.attention.{field}.{index}"] = value
        cache_inputs = getattr(attn, "cache_store_inputs", None)
        request_ids = getattr(cache_inputs, "request_id", None)
        if isinstance(request_ids, torch.Tensor):
            tensors["input.request_ids"] = request_ids
    return metadata, tensors


class ModelTrace:
    def __init__(self, name, directory=None):
        self.name = name
        self.directory = directory
        self.trace = None
        self.warmup_depth = 0
        self.capture_active = False
        self.invocation = 0
        self.refresh = lambda: None

    def _get_trace(self):
        if self.trace is None:
            identity = {
                "model": self.name,
                "engine": "rtp",
                "run_id": os.environ["K3_TRACE_RUN_ID"],
                "host": socket.gethostname(),
                "pid": os.getpid(),
            }
            if torch.distributed.is_available() and torch.distributed.is_initialized():
                identity["rank"] = torch.distributed.get_rank()
                identity["world_size"] = torch.distributed.get_world_size()
            directory = self.directory or Path(os.environ["K3_TRACE_ROOT"]) / (
                f"model-{self.name}-{identity['host']}-{os.getpid()}-{id(self)}"
            )
            self.trace = TensorTrace(
                directory,
                identity=identity,
                max_pending_bytes=int(
                    os.environ.get("K3_TRACE_MAX_PENDING_BYTES", str(4 * 1024**3))
                ),
            )
        self.trace.handoff()
        return self.trace

    def forward(self, function, inputs, *args, **kwargs):
        with _lock:
            stack = _stack()
            if not stack or stack[-1]._frame.capture_key is None:
                self.refresh()
            if self.warmup_depth or getattr(_local, "suspended", 0):
                _local.suspended = getattr(_local, "suspended", 0) + 1
                try:
                    return function(inputs, *args, **kwargs)
                finally:
                    _local.suspended -= 1
            owns_frame = not stack
            if owns_frame:
                trace = self._get_trace()
                metadata, _ = model_inputs(inputs)
                trace.begin(
                    {
                        "model": self.name,
                        "invocation": self.invocation,
                        "execution": "eager",
                        **metadata,
                    }
                )
                self.invocation += 1
                stack.append(trace)
            try:
                record_model_inputs(f"{self.name}.inputs", inputs)
                output = function(inputs, *args, **kwargs)
                record_model(f"{self.name}.output", output)
                if owns_frame:
                    trace.end()
                return output
            except BaseException:
                if owns_frame:
                    try:
                        trace._fail("K3 model forward aborted")
                    except RuntimeError:
                        pass
                raise
            finally:
                if owns_frame:
                    stack.pop()

    def capture_begin(self, key):
        # Held until capture_end/abort on the same thread. This also protects
        # startup-to-serving ownership transfer and nested main/MTP execution.
        _lock.acquire()
        try:
            if self.capture_active:
                raise RuntimeError("nested capture on the same model recorder")
            self.refresh()
            trace = self._get_trace()
            trace.begin_capture(key)
            _stack().append(trace)
            self.capture_active = True
        except BaseException:
            _lock.release()
            raise

    def capture_end(self):
        if not self.capture_active:
            raise RuntimeError("capture_end without capture_begin")
        try:
            self.trace.end_capture()
        finally:
            self.capture_active = False
            _stack().pop()
            _lock.release()

    def capture_abort(self):
        if not self.capture_active:
            return
        try:
            self.trace._fail("K3 CUDA Graph capture aborted")
        finally:
            self.capture_active = False
            _stack().pop()
            _lock.release()

    def replay(self, key, inputs):
        with _lock:
            trace = self._get_trace()
            metadata, tensors = model_inputs(inputs)
            trace.replay(
                key,
                {"model": self.name, "invocation": self.invocation, **metadata},
                tensors,
            )
            self.invocation += 1

    def warmup(self, active):
        self.warmup_depth += 1 if active else -1
        if self.warmup_depth < 0:
            raise RuntimeError("unbalanced K3 warmup scope")

    def close(self):
        with _lock:
            if self.trace is not None:
                self.trace._owner = threading.get_ident()
                self.trace.close()


def install_model_trace(model, name):
    if not enabled():
        return
    if hasattr(model, "_k3_trace_capture_begin"):
        raise RuntimeError("K3 trace already installed")
    manager = ModelTrace(name)
    _managers.append(manager)
    original = model.forward

    @wraps(original)
    def forward(inputs, *args, **kwargs):
        return manager.forward(original, inputs, *args, **kwargs)

    model.forward = forward
    model._k3_trace_capture_begin = manager.capture_begin
    model._k3_trace_capture_end = manager.capture_end
    model._k3_trace_capture_abort = manager.capture_abort
    model._k3_trace_replay = manager.replay
    model._k3_trace_warmup = manager.warmup
    installed = {}

    def refresh():
        inventory = []
        changed = False
        for path, module in model.named_modules():
            if not path:
                continue
            inventory.append({"name": path, "type": type(module).__qualname__})
            if module in installed:
                if installed[module] != path:
                    raise RuntimeError("traced module changed semantic path")
                continue
            changed = True
            installed[module] = path
            module._k3_trace_path = f"{name}.{path}"

            def hook(module, args, output, path=path):
                if manager.warmup_depth:
                    return
                record_model(f"{name}.{path}.output", output)

            module.register_forward_hook(hook)
        if changed:
            trace = manager.trace or manager._get_trace()
            trace._write_json(
                "module_inventory.json",
                {"modules": inventory, "coverage_verified": False},
            )

    manager.refresh = refresh
    # Resolve replacements made during model initialization before each eager
    # run or capture; never register hooks or write inventories inside capture.
    refresh()


def close_models():
    errors = []
    for manager in _managers:
        try:
            manager.close()
        except RuntimeError as exc:
            errors.append(str(exc))
    if errors:
        raise RuntimeError("; ".join(errors))


atexit.register(close_models)
