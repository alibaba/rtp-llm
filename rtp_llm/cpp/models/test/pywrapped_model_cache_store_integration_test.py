import contextlib
import ctypes
import gc
import importlib
import importlib.util
import inspect
import json
import multiprocessing
import os
import re
import subprocess
import sys
import tempfile
import threading
import time
import types
import unittest
import weakref
from pathlib import Path
from typing import Any, ClassVar, Optional, cast
from unittest.mock import patch

import torch

from rtp_llm.config.cuda_graph import GenerationPrefillCudaGraphUnsupportedBackend
from rtp_llm.config.exceptions import ExceptionType


def _install_isolated_graph_capture_stubs() -> None:
    """Keep this focused integration test independent of the production ops SO.

    The C++ test extension registers the same pybind types as the production
    extension, so importing ``rocm_rccl`` here would register those types a
    second time. This isolated fixture initializes no optional communicator;
    these are the no-op graph admission hooks and single-rank RCCL hooks.
    """

    models_py = types.ModuleType("rtp_llm.models_py")
    models_py.__path__ = []
    distributed = types.ModuleType("rtp_llm.models_py.distributed")
    distributed.__path__ = []
    collectives = types.ModuleType("rtp_llm.models_py.distributed.all_reduce")
    collectives.begin_graph_capture = lambda: None
    collectives.end_graph_capture = lambda: None
    collectives.abort_graph_capture = lambda: None
    rocm_rccl = types.ModuleType("rtp_llm.models_py.distributed.rocm_rccl")
    rocm_rccl.set_graph_capture_nccl_comm = lambda *_args: None
    rocm_rccl.enter_graph_capture_mode = lambda *_args: None
    rocm_rccl.exit_graph_capture_mode = lambda: None
    rocm_rccl.finish_hipgraph_capture_session = lambda: None

    models_py.distributed = distributed
    distributed.all_reduce = collectives
    distributed.rocm_rccl = rocm_rccl
    sys.modules["rtp_llm.models_py"] = models_py
    sys.modules["rtp_llm.models_py.distributed"] = distributed
    sys.modules["rtp_llm.models_py.distributed.all_reduce"] = collectives
    sys.modules["rtp_llm.models_py.distributed.rocm_rccl"] = rocm_rccl


_install_isolated_graph_capture_stubs()

_extension = importlib.import_module(
    "rtp_llm.cpp.models.test.libth_pywrapped_model_cache_store_integration_test"
)
_skip_rocm_context_parallel = unittest.skipIf(
    os.environ.get("TEST_USING_DEVICE") == "ROCM",
    "context parallel is not supported on ROCm",
)
PyModelInputs = _extension.PyModelInputs
PyModelOutputs = _extension.PyModelOutputs
run_generation_prefill_capture_scenario = (
    _extension.run_generation_prefill_capture_scenario
)
run_dirty_generation_prefill_capture_scenario = (
    _extension.run_dirty_generation_prefill_capture_scenario
)
run_scenario = _extension.run_scenario
clear_comm_ops = _extension.clear_comm_ops
close_capture_publisher_without_gil = _extension.close_capture_publisher_without_gil
destroy_cpu_tp_broadcaster = _extension.destroy_cpu_tp_broadcaster
drain_capture_publisher_without_close = _extension.drain_capture_publisher_without_close
init_cpu_tp_broadcaster = _extension.init_cpu_tp_broadcaster
initialize_capture_publisher = _extension.initialize_capture_publisher
register_comm_ops = _extension.register_comm_ops
run_tp_sync_model_inputs_scenario = _extension.run_tp_sync_model_inputs_scenario

_KMONITOR_GAUGE = 0
_KMONITOR_QPS = 3
_KMONITOR_STATUS = 5
_CAPTURE_METRIC_PREFIX = "rtp_llm_hidden_state_capture_"
_CAPTURE_QPS_METRICS = {
    f"{_CAPTURE_METRIC_PREFIX}admission_rejected_qps",
    f"{_CAPTURE_METRIC_PREFIX}eligibility_skip_qps",
    f"{_CAPTURE_METRIC_PREFIX}batch_cancelled_requests",
    f"{_CAPTURE_METRIC_PREFIX}skip_capacity_qps",
    f"{_CAPTURE_METRIC_PREFIX}skip_config_qps",
    f"{_CAPTURE_METRIC_PREFIX}skip_unavailable_qps",
    f"{_CAPTURE_METRIC_PREFIX}batch_qps",
    f"{_CAPTURE_METRIC_PREFIX}publish_success_qps",
    f"{_CAPTURE_METRIC_PREFIX}failure_qps",
    f"{_CAPTURE_METRIC_PREFIX}initialization_failure_qps",
    f"{_CAPTURE_METRIC_PREFIX}layout_failure_qps",
    f"{_CAPTURE_METRIC_PREFIX}prepare_failure_qps",
    f"{_CAPTURE_METRIC_PREFIX}quantize_failure_qps",
    f"{_CAPTURE_METRIC_PREFIX}store_failure_qps",
    f"{_CAPTURE_METRIC_PREFIX}shutdown_failure_qps",
    f"{_CAPTURE_METRIC_PREFIX}hard_contract_failure_qps",
    f"{_CAPTURE_METRIC_PREFIX}request_error_failure_qps",
    f"{_CAPTURE_METRIC_PREFIX}operational_failure_qps",
    f"{_CAPTURE_METRIC_PREFIX}duplicate_request_id_qps",
    f"{_CAPTURE_METRIC_PREFIX}broken_rejection_qps",
    f"{_CAPTURE_METRIC_PREFIX}bf16_publish_qps",
    f"{_CAPTURE_METRIC_PREFIX}fp8_publish_qps",
}
_CAPTURE_STATUS_METRICS = {
    f"{_CAPTURE_METRIC_PREFIX}enabled",
    f"{_CAPTURE_METRIC_PREFIX}broken",
}
_CAPTURE_QUOTA_METRICS = {
    f"{_CAPTURE_METRIC_PREFIX}quota_{name}" for name in ("samples", "bytes", "tokens")
}
_CAPTURE_PUBLISH_METRICS = {
    f"{_CAPTURE_METRIC_PREFIX}publish_latency_us",
    f"{_CAPTURE_METRIC_PREFIX}store_put_latency_us",
    f"{_CAPTURE_METRIC_PREFIX}publish_request_count",
    f"{_CAPTURE_METRIC_PREFIX}publish_token_count",
    f"{_CAPTURE_METRIC_PREFIX}publish_payload_bytes",
    f"{_CAPTURE_METRIC_PREFIX}publish_input_ids_bytes",
    f"{_CAPTURE_METRIC_PREFIX}publish_aux_hidden_bytes",
    f"{_CAPTURE_METRIC_PREFIX}publish_last_hidden_bytes",
    f"{_CAPTURE_METRIC_PREFIX}publish_scale_bytes",
}
_MOONCAKE_STORE_KEY_NAMESPACE_ENV = "MOONCAKE_STORE_KEY_NAMESPACE"
_MOONCAKE_RTP_LAYOUT_ENV = {
    "MOONCAKE_HIDDEN_DIM": "hidden_dim",
    "MOONCAKE_MAX_SEQ_LEN": "max_seq_len",
    "MOONCAKE_GET_BATCH_SIZE": "get_batch_size",
    "MOONCAKE_GPU_BUFFER_SIZE": "gpu_buffer_size",
    "MOONCAKE_NUM_AUX_LAYERS": "num_aux_layers",
}


def _snapshot_mooncake_layout(config) -> dict[str, Any]:
    return {
        attr_name: getattr(config, attr_name)
        for attr_name in _MOONCAKE_RTP_LAYOUT_ENV.values()
    }


def _restore_mooncake_layout_env(previous_env: dict[str, str | None]) -> None:
    for env_name, value in previous_env.items():
        if value is None:
            os.environ.pop(env_name, None)
        else:
            os.environ[env_name] = value


class _FakeMooncakeConfig:
    hidden_dim = 1
    max_seq_len = 8192
    get_batch_size = 1
    gpu_buffer_size = None
    num_aux_layers = 2
    async_put_pool_size = 1
    enable_gpu_direct = False
    store_key_namespace = "TestActor"
    make_store_key_calls: ClassVar[list[int]] = []

    @classmethod
    def from_env(cls):
        config = cls()
        namespace = os.environ.get(_MOONCAKE_STORE_KEY_NAMESPACE_ENV)
        if not namespace or re.fullmatch(r"[A-Za-z0-9]+", namespace) is None:
            raise ValueError(
                "Mooncake store_key_namespace must be non-empty and contain only "
                "ASCII letters and digits"
            )
        config.store_key_namespace = namespace
        config.async_put_pool_size = int(os.environ.get("MOONCAKE_ASYNC_PUT_POOL_SIZE", "1"))
        config.enable_gpu_direct = os.environ.get("MOONCAKE_ENABLE_GPU_DIRECT", "0") == "1"
        for env_name, attr_name in _MOONCAKE_RTP_LAYOUT_ENV.items():
            value = os.environ.get(env_name)
            if value is not None:
                setattr(config, attr_name, int(value))
        return config

    def make_store_key(self, request_id: int) -> str:
        type(self).make_store_key_calls.append(request_id)
        return f"rtp_{self.store_key_namespace}_{request_id}"


class _FakeEagleMooncakeStore:
    instances: ClassVar[list["_FakeEagleMooncakeStore"]] = []
    TENSOR_SUFFIXES = ("_hs", "_hss", "_tgt", "_ids", "_lhs", "_ready")
    events = None
    fail_put = False
    fail_put_count = 0
    fail_flush = False
    fail_warmup = False
    async_error_after_accept_count = 0

    def __init__(self, config) -> None:
        self.config = config
        self._gpu_direct_available = False
        self._gpu_send_buffer = None
        self.config_at_init = _snapshot_mooncake_layout(config)
        self.config_at_setup: dict[str, Any] = {}
        self.puts = []
        self.put_batch_calls = []
        self.put_attempts = 0
        self.keys: set[str] = set()
        self.call_sequence: list[str] = []
        self.query_calls: list[list[str]] = []
        self.reclaim_calls: list[str] = []
        self.removed_keys: list[str] = []
        self.pending_async_errors: list[BaseException] = []
        self.setup_device = None
        self.warmed_up = False
        self.flushed = False
        self.closed = False
        type(self).instances.append(self)

    @classmethod
    def _record(cls, event: str) -> None:
        if cls.events is not None:
            cls.events.append(event)

    def setup(self, device=None) -> None:
        self.call_sequence.append("setup")
        type(self)._record("store.setup")
        self.config_at_setup = _snapshot_mooncake_layout(self.config)
        self.setup_device = device
        if self.config.enable_gpu_direct:
            self._gpu_send_buffer = torch.empty((1, self.config.hidden_dim), dtype=torch.bfloat16, device=device)
            self._gpu_direct_available = True

    def warmup_rdma(self) -> None:
        self.call_sequence.append("warmup_rdma")
        type(self)._record("store.warmup")
        self.warmed_up = True
        if type(self).fail_warmup:
            raise RuntimeError("injected Mooncake warmup failure")

    def take_async_errors(self) -> list[BaseException]:
        self.call_sequence.append("take_async_errors")
        type(self)._record("store.take_async_errors")
        errors = self.pending_async_errors
        self.pending_async_errors = []
        return errors

    def _record_put(
        self,
        key,
        hidden_states,
        input_ids,
        last_hidden_states,
        hidden_states_scale=None,
    ) -> dict[str, Any]:
        assert hidden_states.is_contiguous(), "hidden_states must be contiguous"
        assert input_ids.is_contiguous(), "input_ids must be contiguous"
        assert input_ids.dtype == torch.int64, "input_ids must be int64"
        assert (
            input_ids.device == last_hidden_states.device
        ), "input_ids and last_hidden_states must be on the same device"
        assert (
            last_hidden_states.is_contiguous()
        ), "last_hidden_states must be contiguous"
        assert (
            last_hidden_states.dtype == torch.bfloat16
        ), "last_hidden_states must be bfloat16 at the store boundary"
        if hidden_states_scale is not None:
            assert (
                hidden_states_scale.is_contiguous()
            ), "hidden_states_scale must be contiguous"
            assert (
                hidden_states.dtype == torch.float8_e4m3fn
            ), "FP8 hidden_states must be float8_e4m3fn at the store boundary"
        else:
            assert (
                hidden_states.dtype == torch.bfloat16
            ), "BF16 hidden_states must be bfloat16 at the store boundary"

        dequantized_hidden_states = None
        if hidden_states_scale is not None:
            num_layers = hidden_states_scale.shape[-1]
            assert hidden_states.shape[-1] % num_layers == 0
            layer_width = hidden_states.shape[-1] // num_layers
            dequantized_hidden_states = (
                (
                    hidden_states.detach()
                    .to(torch.float32)
                    .reshape(*hidden_states.shape[:-1], num_layers, layer_width)
                    * hidden_states_scale.detach().to(torch.float32).unsqueeze(-1)
                )
                .reshape(hidden_states.shape)
                .cpu()
            )

        self.puts.append(
            {
                "key": key,
                "hidden_states": hidden_states.detach().cpu(),
                "dequantized_hidden_states": dequantized_hidden_states,
                "input_ids": input_ids.detach().cpu(),
                "last_hidden_states": last_hidden_states.detach().cpu(),
                "hidden_states_scale": (
                    hidden_states_scale.detach().cpu()
                    if hidden_states_scale is not None
                    else None
                ),
                "source_hidden_dtype": hidden_states.dtype,
                "source_last_hidden_dtype": last_hidden_states.dtype,
            }
        )
        self.keys.update(key + suffix for suffix in ("_hs", "_ids", "_lhs", "_ready"))
        if hidden_states_scale is not None:
            self.keys.add(key + "_hss")
        metadata = {
            "shapes": {
                "hidden_states": tuple(hidden_states.shape),
                "input_ids": tuple(input_ids.shape),
                "last_hidden_states": tuple(last_hidden_states.shape),
            },
            "dtypes": {
                "hidden_states": hidden_states.dtype,
                "input_ids": input_ids.dtype,
                "last_hidden_states": last_hidden_states.dtype,
            },
        }
        if hidden_states_scale is not None:
            metadata["shapes"]["hidden_states_scale"] = tuple(hidden_states_scale.shape)
            metadata["dtypes"]["hidden_states_scale"] = hidden_states_scale.dtype
        return metadata

    def put_batch(
        self,
        batch_id,
        request_ids,
        hidden_states,
        input_ids,
        last_hidden_states=None,
        target=None,
        hidden_states_scale=None,
        capture_metadata=None,
    ) -> list[dict[str, Any]]:
        del target
        request_ids = list(request_ids)
        hidden_states = list(hidden_states)
        input_ids = list(input_ids)
        last_hidden_states = (
            list(last_hidden_states)
            if last_hidden_states is not None
            else [None] * len(request_ids)
        )
        hidden_states_scale = (
            list(hidden_states_scale)
            if hidden_states_scale is not None
            else [None] * len(request_ids)
        )
        values = (hidden_states, input_ids, last_hidden_states, hidden_states_scale)
        if not request_ids or any(len(items) != len(request_ids) for items in values):
            raise ValueError("invalid fake Mooncake put_batch lengths")

        self.call_sequence.append("put_batch")
        type(self)._record(f"store.put_batch:{batch_id}")
        self.put_attempts += 1
        self.put_batch_calls.append(
            {
                "batch_id": batch_id,
                "request_ids": request_ids,
                "lengths": [tensor.shape[0] for tensor in hidden_states],
                "capture_metadata": capture_metadata,
            }
        )
        if not isinstance(capture_metadata, dict) or not isinstance(
            capture_metadata.get("capture_layer_ids"), list
        ):
            raise ValueError("capture_metadata requires ordered capture_layer_ids")
        duplicate_key = next(
            (
                key
                for key in request_ids
                if any(
                    key + suffix in self.keys
                    for suffix in self.TENSOR_SUFFIXES
                )
            ),
            None,
        )
        if duplicate_key is not None:
            raise FileExistsError(f"Mooncake key already exists: {duplicate_key}")
        if type(self).fail_put_count > 0:
            type(self).fail_put_count -= 1
            raise RuntimeError("injected Mooncake put_batch admission failure")
        if type(self).fail_put:
            raise RuntimeError("injected Mooncake put_batch admission failure")

        metadata = [
            self._record_put(key, aux, ids, last, scale)
            for key, aux, ids, last, scale in zip(
                request_ids,
                hidden_states,
                input_ids,
                last_hidden_states,
                hidden_states_scale,
                strict=True,
            )
        ]
        if type(self).async_error_after_accept_count > 0:
            type(self).async_error_after_accept_count -= 1
            self.pending_async_errors.append(
                RuntimeError(
                    "injected historical Mooncake async failure "
                    f"(batch_id={batch_id!r}, request_ids={request_ids!r}, "
                    "stage=data, attempts=1)"
                )
            )
        return metadata

    def query_keys(self, keys: list[str]) -> dict[str, bool]:
        self.call_sequence.append("query_keys")
        self.query_calls.append(list(keys))
        return {key: key in self.keys for key in keys}

    def abandon_capture(self, key: str) -> None:
        self.keys.add(key + "_abandoned")

    def reclaim_abandoned_capture(self, key: str) -> bool:
        # Store-boundary fake only: the native publisher must enforce PUT/fence
        # completion before reaching this producer-only operation.
        self.call_sequence.append(f"reclaim_abandoned_capture:{key}")
        self.reclaim_calls.append(key)
        marker = key + "_abandoned"
        if marker not in self.keys:
            return False
        for suffix in self.TENSOR_SUFFIXES:
            source = key + suffix
            if source in self.keys:
                self.keys.remove(source)
                self.removed_keys.append(source)
        self.keys.remove(marker)
        self.removed_keys.append(marker)
        return True

    def consume(self) -> None:
        # A normal consumer removes source payload, not a delivery waiver.
        self.keys.difference_update(
            key for key in list(self.keys) if not key.endswith("_abandoned")
        )

    def flush(self) -> None:
        self.call_sequence.append("flush")
        type(self)._record("store.flush")
        self.flushed = True
        if type(self).fail_flush:
            raise RuntimeError("injected Mooncake flush failure")

    def close(self) -> None:
        self.call_sequence.append("close")
        type(self)._record("store.close")
        self.closed = True


def _fake_quantize_aux_hidden_states(hidden_states, num_layers):
    layer_width = hidden_states.shape[-1] // num_layers
    scales = torch.ones(
        (*hidden_states.shape[:-1], num_layers),
        dtype=torch.float32,
        device=hidden_states.device,
    )
    quantized = torch.empty_like(hidden_states, dtype=torch.float8_e4m3fn)
    for layer_idx in range(num_layers):
        start = layer_idx * layer_width
        quantized[..., start : start + layer_width].copy_(
            hidden_states[..., start : start + layer_width]
        )
    return quantized, scales


_FAKE_TORCHSPEC_MODULE_NAMES = (
    "torchspec",
    "torchspec.transfer",
    "torchspec.transfer.mooncake",
    "torchspec.utils",
    "torchspec.utils.fp8",
)


def _install_fake_torchspec_modules() -> dict[str, types.ModuleType]:
    previous_modules = {
        name: sys.modules[name]
        for name in _FAKE_TORCHSPEC_MODULE_NAMES
        if name in sys.modules
    }
    torchspec = types.ModuleType("torchspec")
    torchspec.__path__ = []
    transfer = types.ModuleType("torchspec.transfer")
    transfer.__path__ = []
    mooncake = types.ModuleType("torchspec.transfer.mooncake")
    mooncake_attrs = cast(Any, mooncake)
    mooncake_attrs.MooncakeConfig = _FakeMooncakeConfig
    mooncake_attrs.EagleMooncakeStore = _FakeEagleMooncakeStore
    utils = types.ModuleType("torchspec.utils")
    utils.__path__ = []
    fp8 = types.ModuleType("torchspec.utils.fp8")
    cast(Any, fp8).quantize_aux_hidden_states = _fake_quantize_aux_hidden_states
    sys.modules["torchspec"] = torchspec
    sys.modules["torchspec.transfer"] = transfer
    sys.modules["torchspec.transfer.mooncake"] = mooncake
    sys.modules["torchspec.utils"] = utils
    sys.modules["torchspec.utils.fp8"] = fp8
    return previous_modules


def _restore_torchspec_modules(
    previous_modules: dict[str, types.ModuleType],
) -> None:
    for name in _FAKE_TORCHSPEC_MODULE_NAMES:
        if name in previous_modules:
            sys.modules[name] = previous_modules[name]
        else:
            sys.modules.pop(name, None)


def _register_no_collectives(allreduce=None, *, submit_frames=None) -> None:
    def unexpected_collective(*args):
        raise AssertionError("hidden-state capture issued an unexpected collective")

    def plan_or_forward_reduce(tensor, op, mode, dest):
        if tensor.device.type == "cpu":
            # Existing forward tests model identical startup plans. Their injected
            # reducer remains specific to the GPU per-forward layout consensus.
            assert mode == 0 and ((op == 2 and tensor.shape == (2,)) or (op == 3 and tensor.shape == (1,)))
            return tensor
        return allreduce(tensor, op, mode, dest)

    def identical_plan_broadcast(tensors, root, mode):
        assert root == 0 and mode == 0
        if len(tensors) == 1 and tensors[0].dtype == torch.int32:
            frame = tensors[0]
            assert frame.device.type == "cpu" and frame.shape == (1,)
            status = frame.item()
            assert status in (0, 2)  # SUBMITTED or DROPPED, never FATAL
            if submit_frames is not None:
                submit_frames.append(status)
            return
        assert all(t.device.type == "cpu" and t.dtype == torch.int64 for t in tensors)

    register_comm_ops(
        unexpected_collective if allreduce is None else identical_plan_broadcast,
        unexpected_collective if allreduce is None else plan_or_forward_reduce,
        unexpected_collective,
        unexpected_collective,
        unexpected_collective,
    )


def _register_identity_allreduce(*, submit_frames=None) -> None:
    def identity_allreduce(tensor, op, mode, dest):
        del op, mode, dest
        return tensor

    _register_no_collectives(identity_allreduce, submit_frames=submit_frames)


run_speculative_commit_bridge = _extension.run_speculative_commit_bridge


class CacheStoreForwardModel:
    """Test model that replaces attention math but keeps the real cache-store call."""

    supports_hidden_state_capture = True

    def __init__(self, hidden_size: int = 1) -> None:
        self.hidden_size = hidden_size
        self.kv_cache = None
        self.forward_calls = 0
        self.seen_input_lengths: list[list[int]] = []
        self.seen_kernel_tables: dict[str, list[list[int]]] = {}
        self.seen_cache_tables: list[dict[str, dict[str, list[list[int]]]]] = []
        self.seen_attention_tables: list[dict[str, list[list[int]]]] = []
        self.seen_direct_inputs: list[bool] = []
        self.capture_flags: list[bool] = []

    def initialize(self, resources) -> bool:
        self.kv_cache = resources.kv_cache
        return True

    def prepare_fmha_impl(
        self,
        inputs: PyModelInputs,
        is_cuda_graph: bool = False,
        cuda_graph_selection_mode: Optional[str] = None,
    ):
        return None

    def _forward_one(self, inputs: PyModelInputs) -> PyModelOutputs:
        self.capture_flags.append(inputs.capture_hidden_states)
        attention_inputs = inputs.attention_inputs
        self.seen_direct_inputs.append(not isinstance(attention_inputs, dict))
        if isinstance(attention_inputs, dict):
            self.seen_attention_tables.append(
                {
                    tag: item.kv_cache_kernel_block_id_device.cpu().tolist()
                    for tag, item in attention_inputs.items()
                }
            )
        first_inputs = (
            next(iter(attention_inputs.values()))
            if isinstance(attention_inputs, dict)
            else attention_inputs
        )
        self.seen_input_lengths.append(first_inputs.input_lengths.tolist())

        assert self.kv_cache is not None
        cache_tables = {}
        for layer_cache in self.kv_cache.get_layer_cache_groups(0):
            tag_inputs = (
                attention_inputs[layer_cache.tag]
                if isinstance(attention_inputs, dict)
                else attention_inputs
            )
            self.seen_kernel_tables[layer_cache.tag] = (
                tag_inputs.kv_cache_kernel_block_id.tolist()
            )
            cache_tables[layer_cache.tag] = {
                "physical": tag_inputs.kv_cache_block_id.tolist(),
                "kernel": tag_inputs.kv_cache_kernel_block_id.tolist(),
            }
            if (
                tag_inputs.cache_store_inputs is not None
                and tag_inputs.cache_store_writer is not None
            ):
                tag_inputs.cache_store_writer.write(
                    tag_inputs.cache_store_inputs, layer_cache
                )
        self.seen_cache_tables.append(cache_tables)

        canonical_hidden_states = torch.zeros(
            (inputs.input_ids.numel(), self.hidden_size),
            dtype=torch.float16,
            device=inputs.input_ids.device,
        )
        output = PyModelOutputs(canonical_hidden_states)
        if inputs.capture_hidden_states:
            output.captured_hidden_states = torch.cat(
                (
                    canonical_hidden_states + 10,
                    canonical_hidden_states + 20,
                    canonical_hidden_states + 30,
                ),
                dim=-1,
            )
        return output

    def forward(self, inputs: PyModelInputs, fmha_impl=None) -> PyModelOutputs:
        self.forward_calls += 1
        return self._forward_one(inputs)


class CloseAwareCacheStoreForwardModel(CacheStoreForwardModel):
    def __init__(self) -> None:
        super().__init__()
        self.closed = False
        self.close_calls = 0
        self.initialize_calls = 0
        self.fail_initialize = False

    def initialize(self, resources) -> bool:
        self.initialize_calls += 1
        if self.fail_initialize:
            raise RuntimeError("injected model initialize failure")
        return super().initialize(resources)

    def forward(self, inputs: PyModelInputs, fmha_impl=None) -> PyModelOutputs:
        if self.closed:
            raise RuntimeError("model is closed")
        return super().forward(inputs, fmha_impl)

    def close(self) -> None:
        self.close_calls += 1
        self.closed = True


class MixedDeviceMetadataModel(CacheStoreForwardModel):
    def __init__(self):
        super().__init__()
        self.metadata = []

    def _forward_one(self, inputs):
        attn = inputs.attention_inputs
        if isinstance(attn, dict):
            attn = next(iter(attn.values()))
        assert attn.input_lengths.is_cuda
        assert not attn.prefix_lengths.is_cuda
        self.metadata.append(
            (
                attn.cu_seqlens_device.cpu().tolist(),
                attn.cu_kv_seqlens_device.cpu().tolist(),
            )
        )
        return super()._forward_one(inputs)


class DirtyGenerationPrefillCaptureModel:
    """Fail on the graph-capture body after all eager generation-prefill warmups."""

    def __init__(self) -> None:
        self.prefill_forward_calls = 0
        self._is_generation_prefill = False

    def initialize(self, _resources) -> bool:
        return True

    def prepare_fmha_impl(
        self,
        _inputs: PyModelInputs,
        is_cuda_graph: bool = False,
        cuda_graph_selection_mode=None,
    ):
        self._is_generation_prefill = (
            cuda_graph_selection_mode == "generation_prefill_graph"
        )
        return None

    def forward(self, inputs: PyModelInputs, _fmha_impl=None) -> PyModelOutputs:
        if self._is_generation_prefill:
            self.prefill_forward_calls += 1
            # The datatype probe and two eager warmups precede capture.
            if self.prefill_forward_calls == 4:
                raise RuntimeError("injected PyWrappedModel capture-body failure")
        hidden_states = (
            inputs.input_ids.to(torch.bfloat16).unsqueeze(1).expand(-1, 4).contiguous()
        )
        return PyModelOutputs(hidden_states)


class SuccessfulGenerationPrefillCaptureModel(DirtyGenerationPrefillCaptureModel):
    """Capture-safe model used to verify block-0 capture needs no allocation."""

    def forward(self, inputs: PyModelInputs, _fmha_impl=None) -> PyModelOutputs:
        if self._is_generation_prefill:
            self.prefill_forward_calls += 1
        hidden_states = (
            inputs.input_ids.to(torch.bfloat16).unsqueeze(1).expand(-1, 4).contiguous()
        )
        return PyModelOutputs(hidden_states)


class CaptureAwareGenerationPrefillModel(SuccessfulGenerationPrefillCaptureModel):
    supports_hidden_state_capture = True

    def __init__(self) -> None:
        super().__init__()
        self.forward_calls = 0

    def forward(self, inputs: PyModelInputs, fmha_impl=None) -> PyModelOutputs:
        self.forward_calls += 1
        hidden_states = super().forward(inputs, fmha_impl).hidden_states
        mtp_hidden_states = torch.cat((hidden_states + 40, hidden_states + 50), dim=-1)
        output = PyModelOutputs(hidden_states, mtp_hidden_states)
        if inputs.capture_hidden_states:
            output.captured_hidden_states = torch.cat(
                (hidden_states + 10, hidden_states + 20, hidden_states + 30), dim=-1
            )
        return output


class LateInitializeFailureGenerationPrefillCaptureModel(
    SuccessfulGenerationPrefillCaptureModel
):
    """Fail the post-capture initialization check after both graphs exist."""

    def __init__(self) -> None:
        super().__init__()
        self.initialize_calls = 0

    def initialize(self, _resources) -> bool:
        self.initialize_calls += 1
        # PyWrappedModel initializes once before graph setup and once before
        # capture. The second result is checked only after decode and prefill
        # and generation-prefill runners have been constructed, exercising
        # constructor unwind.
        return self.initialize_calls == 1


class CleanFallbackGenerationPrefillCaptureModel(DirtyGenerationPrefillCaptureModel):
    """Fail before capture begins to verify clean fail-fast teardown."""

    def forward(self, inputs: PyModelInputs, _fmha_impl=None) -> PyModelOutputs:
        if self._is_generation_prefill:
            self.prefill_forward_calls += 1
            if self.prefill_forward_calls == 1:
                raise RuntimeError("injected clean generation prefill graph fallback")
        hidden_states = (
            inputs.input_ids.to(torch.bfloat16).unsqueeze(1).expand(-1, 4).contiguous()
        )
        return PyModelOutputs(hidden_states)


class UnsupportedBackendGenerationPrefillCaptureModel(
    DirtyGenerationPrefillCaptureModel
):
    """Reject generation-prefill graph selection before capture begins."""

    def prepare_fmha_impl(
        self,
        _inputs: PyModelInputs,
        is_cuda_graph: bool = False,
        cuda_graph_selection_mode=None,
    ):
        if cuda_graph_selection_mode == "generation_prefill_graph":
            raise GenerationPrefillCudaGraphUnsupportedBackend(
                "injected unsupported backend"
            )
        return None


class PositionAwareCaptureModel(CacheStoreForwardModel):
    position_aware_hidden = True

    def _forward_one(self, inputs: PyModelInputs) -> PyModelOutputs:
        output = super()._forward_one(inputs)
        positions = inputs.input_ids.to(torch.float16).reshape(-1, 1)
        output.hidden_states = positions
        if inputs.capture_hidden_states:
            output.captured_hidden_states = torch.cat((positions + 10, positions + 20, positions + 30), dim=-1)
        return output


class FinalNormParityModel(CacheStoreForwardModel):
    _FINAL_NORM_GAMMA = (0.5, 1.5)
    _FINAL_NORM_EPS = 1e-5

    def _forward_one(self, inputs: PyModelInputs) -> PyModelOutputs:
        super()._forward_one(inputs)
        token_values = inputs.input_ids.to(torch.float16).reshape(-1, 1)
        hidden_states = torch.cat((token_values, token_values + 1), dim=-1)
        variance = hidden_states.float().pow(2).mean(-1, keepdim=True)
        final_hidden_states = (
            hidden_states * torch.rsqrt(variance + self._FINAL_NORM_EPS)
        ) * hidden_states.new_tensor(self._FINAL_NORM_GAMMA)
        final_hidden_states = final_hidden_states.to(hidden_states.dtype)
        output = PyModelOutputs(final_hidden_states)
        if inputs.capture_hidden_states:
            output.captured_hidden_states = torch.cat(
                (
                    hidden_states + 10,
                    hidden_states + 20,
                    final_hidden_states,
                ),
                dim=-1,
            )
        return output


def _load_capture_context(return_module=False):
    # Execute the production helper and profiler, but reuse this fixture's native
    # output binding: importing compute_ops would register the types a second time.
    root = Path(__file__).parents[3] / "models_py"
    profiler_spec = importlib.util.spec_from_file_location(
        "rtp_llm.models_py.utils.profiler", root / "utils/profiler.py"
    )
    assert profiler_spec is not None and profiler_spec.loader is not None
    profiler = importlib.util.module_from_spec(profiler_spec)
    profiler_spec.loader.exec_module(profiler)
    capture_spec = importlib.util.spec_from_file_location(
        "_pywrapped_test_hidden_state_capture", root / "model_desc/hidden_state_capture.py"
    )
    assert capture_spec is not None and capture_spec.loader is not None
    capture = importlib.util.module_from_spec(capture_spec)
    with patch.dict(
        sys.modules,
        {
            "rtp_llm.models_py.utils.profiler": profiler,
            "rtp_llm.ops.compute_ops": _extension,
        },
    ):
        capture_spec.loader.exec_module(capture)
    return capture if return_module else capture.CaptureContext


class CaptureContextForwardModel(CacheStoreForwardModel):
    def __init__(self, hidden_size=1, failure_stage="", error=None):
        super().__init__(hidden_size)
        self.failure_stage = failure_stage
        self.error = error if error is not None else RuntimeError(f"injected transient {failure_stage} failure")
        self.layer_hook_calls = 0
        self.capture_errors = []
        self.capture_context = _load_capture_context().configured([0, 1], self._canonical_layer, self._canonical_final)

    def _canonical_layer(self, hidden, residual):
        self.layer_hook_calls += 1
        if self.failure_stage == "hook" and self.forward_calls == 1 and self.layer_hook_calls == 2:
            raise self.error
        return hidden

    def _canonical_final(self, hidden, residual):
        if self.failure_stage == "final" and self.forward_calls == 1:
            raise self.error
        return hidden + 30

    def _forward_one(self, inputs):
        self.capture_flags.append(inputs.capture_hidden_states)
        hidden = inputs.input_ids.to(torch.float16).reshape(-1, 1) + torch.arange(
            self.hidden_size, dtype=torch.float16, device=inputs.input_ids.device
        )
        capture = self.capture_context.for_forward(inputs.capture_hidden_states)
        capture.capture_layer(0, hidden + 10)
        capture.capture_layer(1, hidden + 20)
        if self.failure_stage == "pack" and self.forward_calls == 1:
            with patch.object(torch, "cat", side_effect=self.error):
                output = capture.finalize(hidden)
        else:
            output = capture.finalize(hidden)
        self.capture_errors.append(output.capture_error)
        if self.failure_stage == "layout" and self.forward_calls == 1:
            output.captured_hidden_states = output.captured_hidden_states[:, :-1]
        return output


class MalformedCaptureLayoutModel(CacheStoreForwardModel):
    def _forward_one(self, inputs: PyModelInputs) -> PyModelOutputs:
        output = super()._forward_one(inputs)
        if inputs.capture_hidden_states:
            output.captured_hidden_states = output.captured_hidden_states[:, :2]
            return output
        return output


class MalformedCaptureRankModel(CacheStoreForwardModel):
    def _forward_one(self, inputs: PyModelInputs) -> PyModelOutputs:
        output = super()._forward_one(inputs)
        if inputs.capture_hidden_states:
            output.captured_hidden_states = output.captured_hidden_states.flatten()
            return output
        return output


class MalformedCaptureRowCountModel(CacheStoreForwardModel):
    def _forward_one(self, inputs: PyModelInputs) -> PyModelOutputs:
        output = super()._forward_one(inputs)
        if inputs.capture_hidden_states:
            output.captured_hidden_states = output.captured_hidden_states[:-1]
            return output
        return output


class MalformedCaptureDtypeModel(CacheStoreForwardModel):
    def _forward_one(self, inputs: PyModelInputs) -> PyModelOutputs:
        output = super()._forward_one(inputs)
        if inputs.capture_hidden_states:
            output.captured_hidden_states = output.captured_hidden_states.float()
            return output
        return output


class MalformedCaptureDeviceModel(CacheStoreForwardModel):
    def _forward_one(self, inputs: PyModelInputs) -> PyModelOutputs:
        output = super()._forward_one(inputs)
        if inputs.capture_hidden_states:
            output.captured_hidden_states = output.captured_hidden_states.cpu()
            return output
        return output


class UnsupportedCaptureModel(CacheStoreForwardModel):
    supports_hidden_state_capture = False


class SpeculativeCommitBridgeModel:
    def __init__(
        self,
        *,
        prepare_error: bool = False,
        commit_error: bool = False,
        abort_error: bool = False,
    ):
        self.prepare_error = prepare_error
        self.commit_error = commit_error
        self.abort_error = abort_error
        self.finish_calls: list[bool] = []
        self.finalize_calls = 0

    def initialize(self, resources) -> bool:
        return True

    def forward(self, inputs: PyModelInputs) -> PyModelOutputs:
        raise AssertionError("bridge-only test must not execute forward")

    def prepare_speculative_target_commit(self, accept_len: torch.Tensor) -> None:
        if self.prepare_error:
            raise RuntimeError("prepare bridge boom")

    def finish_speculative_target_commit(self, commit: bool) -> None:
        self.finish_calls.append(commit)
        if commit and self.commit_error:
            raise RuntimeError("commit bridge boom")
        if not commit and self.abort_error:
            raise RuntimeError("abort bridge boom")

    def finalize_speculative_target_commit(self) -> None:
        self.finalize_calls += 1


class PrepareOnlySpeculativeCommitBridgeModel:
    def initialize(self, resources) -> bool:
        return True

    def forward(self, inputs: PyModelInputs) -> PyModelOutputs:
        raise AssertionError("bridge-only test must not execute forward")

    def prepare_speculative_target_commit(self, accept_len: torch.Tensor) -> None:
        pass


def _blocks_by_key(result: dict) -> dict[str, dict]:
    return {
        block["key"]: block
        for record in result["records"]
        for block in record["blocks"]
    }


def _record_for_request(result: dict, request_id: int) -> dict:
    matches = [
        record
        for record in result["records"]
        if record["request_id"] == str(request_id)
    ]
    if len(matches) != 1:
        raise AssertionError(
            f"expected one record for request {request_id}, got {len(matches)}"
        )
    return matches[0]


_TP_SYNC_CAPTURE_FLAGS = [(True, True), (False, True), (True, False), (False, False)]


def _run_tp_sync_uds_rank(tp_rank: int, base_path: str) -> None:
    _register_no_collectives()
    try:
        init_cpu_tp_broadcaster(tp_rank, 4, base_path)
        for skip_run in (False, True):
            results = run_tp_sync_model_inputs_scenario(
                tp_rank, 4, _TP_SYNC_CAPTURE_FLAGS, skip_run
            )
            TpSyncModelInputsIntegrationTest().assertSyncedBatches(results, skip_run)
    finally:
        destroy_cpu_tp_broadcaster()
        clear_comm_ops()


def _check_capture_submit_oom(rank: int, size: int, scenario: str) -> None:
    model = CaptureContextForwardModel()
    result = run_scenario(
        model,
        scenario,
        2,
        capture_only_first_forward=True,
        capture_plan={"tp_rank": rank, "tp_size": size, "layers": [0, 1], "inject_submit_oom": True},
    )
    case = unittest.TestCase()
    case.assertEqual(model.capture_flags, [True, False])
    case.assertFalse(result["capture_fatal"])
    case.assertTrue(result["serving_device_healthy"])
    case.assertEqual(result["forward_capture_error_codes"], [ExceptionType.EXECUTION_EXCEPTION, 0])
    if rank == 0:
        case.assertEqual(result["capture_admissions"], [True, False])
        case.assertEqual(result["forward_capture_inflight_samples"], [0, 0])
        case.assertEqual(_FakeEagleMooncakeStore.instances[-1].put_attempts, 0)
    for logits in result["forward_logits"]:
        if scenario == "capture_with_lm_head":
            case.assertIsNotNone(logits)
            # The test gather leaves peer columns uninitialized; compare this rank's real LM-head result.
            torch.testing.assert_close(logits[:, 2 * rank:2 * rank + 2], torch.tensor([[32., 32.], [35., 35.]]))
        else:
            case.assertIsNone(logits)
    if scenario == "capture_bf16":
        for hidden in result["forward_hidden_states"]:
            torch.testing.assert_close(hidden, torch.arange(31, 36, dtype=torch.float16).reshape(5, 1))


def _run_capture_submit_uds_rank(rank: int, base_path: str) -> None:
    _install_fake_torchspec_modules()
    calls = []

    def broadcast(tensors, root, mode):
        # Startup plan only. The forward fatal frame must use the real CPU socket.
        assert all(t.dtype == torch.int64 for t in tensors)

    # Exchange the actual scalar from each process without a second GPU or NCCL.
    store = torch.distributed.FileStore(base_path + ".min", 2)
    sequence = 0

    def reduce(tensor, op, mode, dest):
        nonlocal sequence
        if tensor.is_cuda:
            assert (op, mode, dest) == (3, 0, None)
            value = tensor.item()
            calls.append(value)
            store.set(f"{sequence}:{rank}", str(value))
            peer = int(store.get(f"{sequence}:{1 - rank}"))
            sequence += 1
            return tensor.fill_(min(value, peer))
        return tensor

    def gather(*args):
        calls.append("logits")

    def unexpected(*args):
        raise AssertionError("unexpected point-to-point operation")

    register_comm_ops(broadcast, reduce, gather, unexpected, unexpected)
    plan = {"tp_rank": rank, "tp_size": 2, "layers": [0, 1]}

    def assert_fatal(model, scenario, expected_error, expected_calls):
        calls.clear()
        with unittest.TestCase().assertRaisesRegex(RuntimeError, expected_error):
            run_scenario(model, scenario, capture_plan=plan)
        assert calls == expected_calls, (scenario, rank, calls)

    try:
        init_cpu_tp_broadcaster(rank, 2, base_path)
        # A stale reservation is an ordinary false return, not a peer fatal.
        result = run_scenario(
            CacheStoreForwardModel(), "capture_with_lm_head", capture_plan=plan, cancel_capture_after_admission=True
        )
        assert calls == [2, "logits"]
        assert result["forward_capture_error_codes"] == [ExceptionType.EXECUTION_EXCEPTION]
        expected = "injected capture request-id clone failure" if rank == 0 else "submission failure on TP rank 0"
        assert_fatal(CacheStoreForwardModel(), "capture_submit_fatal", expected, [2])

        class BadCanonical(MalformedCaptureLayoutModel):
            def _forward_one(self, inputs):
                output = super()._forward_one(inputs)
                if rank == 1:
                    output.hidden_states = output.hidden_states.cpu()
                    torch._C._set_storage_access_error_msg(output.hidden_states, "injected canonical clone failure")
                return output

        expected = "injected canonical clone failure" if rank == 1 else "preparation failure on another TP rank"
        assert_fatal(BadCanonical(), "capture_with_lm_head", expected, [-1] if rank == 1 else [0])

        # Hard and recoverable rejects share discard and its cleanup MIN.
        expected = "injected capture discard failure" if rank == 0 else "rejection failure on another TP rank"
        assert_fatal(MalformedCaptureLayoutModel(), "capture_reject_fatal", expected, [0, -1 if rank == 0 else 2])
        assert not any(store.put_attempts for store in _FakeEagleMooncakeStore.instances)

        calls.clear()
        _check_capture_submit_oom(rank, 2, "capture_with_lm_head")
        assert calls.count("logits") == 2, (rank, calls)
    finally:
        destroy_cpu_tp_broadcaster()
        clear_comm_ops()


class TpSyncModelInputsIntegrationTest(unittest.TestCase):
    def setUp(self) -> None:
        self.addCleanup(clear_comm_ops)
        _register_no_collectives()

    def assertSyncedBatches(self, results, skip_run: bool, tp_rank: int = 0) -> None:
        self.assertEqual(len(results), len(_TP_SYNC_CAPTURE_FLAGS))
        for result, (skip_lm_head, capture_hidden_states) in zip(
            results, _TP_SYNC_CAPTURE_FLAGS
        ):
            self.assertEqual(result["skip_lm_head"], skip_lm_head)
            self.assertEqual(result["capture_hidden_states"], capture_hidden_states)
            self.assertEqual(result["skip_run"], skip_run)
            if "group_tags" in result:
                tags = (
                    []
                    if skip_run
                    else (
                        ["linear", "full"] if tp_rank % 2 == 0 else ["full", "linear"]
                    )
                )
                self.assertEqual(result["group_tags"], tags)
                if skip_run:
                    for name in ("physical", "kernel", "group_types", "copy_mapping"):
                        self.assertIsNone(result[name], name)
                else:
                    physical = {"linear": [[31, 32]], "full": [[11, 12]]}
                    kernel = {"linear": [[131, 132]], "full": [[111, 112]]}
                    group_types = {"linear": 0, "full": 1}
                    for name, expected in (
                        ("physical", physical),
                        ("kernel", kernel),
                        ("group_types", group_types),
                    ):
                        torch.testing.assert_close(
                            result[name],
                            torch.tensor(
                                [expected[tag] for tag in tags], dtype=torch.int32
                            ),
                        )
                    torch.testing.assert_close(
                        result["copy_mapping"],
                        torch.tensor(
                            [[tags.index("linear"), 3, 4], [tags.index("full"), 1, 2]],
                            dtype=torch.int32,
                        ),
                    )
            if skip_run:
                for name in ("combo_tokens", "input_lengths", "request_id"):
                    self.assertIsNone(result[name], name)
            else:
                torch.testing.assert_close(
                    result["combo_tokens"], torch.tensor([7, 8, 9], dtype=torch.int32)
                )
                torch.testing.assert_close(
                    result["input_lengths"], torch.tensor([3], dtype=torch.int32)
                )
                torch.testing.assert_close(
                    result["request_id"], torch.tensor([42], dtype=torch.int64)
                )

    def test_root_flags_replace_empty_and_stale_non_root_inputs(self) -> None:
        def unexpected_collective(*args):
            self.fail("tpSyncModelInputs must only broadcast")

        for skip_run in (False, True):
            for initial_flags, grouped_cache in (
                (False, False),
                (True, False),
                (False, True),
                (True, True),
            ):
                with self.subTest(
                    skip_run=skip_run,
                    initial_flags=initial_flags,
                    grouped_cache=grouped_cache,
                ):
                    frames = []

                    def record_broadcast(tensors, root, mode):
                        self.assertEqual((root, mode), (0, 0))
                        frames.extend(tensor.clone() for tensor in tensors)

                    register_comm_ops(
                        record_broadcast,
                        unexpected_collective,
                        unexpected_collective,
                        unexpected_collective,
                        unexpected_collective,
                    )
                    results = run_tp_sync_model_inputs_scenario(
                        0,
                        4,
                        _TP_SYNC_CAPTURE_FLAGS,
                        skip_run,
                        initial_flags,
                        grouped_cache,
                    )
                    self.assertSyncedBatches(results, skip_run)
                    frames_per_batch = 1 if skip_run else 2
                    self.assertEqual(
                        len(frames), len(_TP_SYNC_CAPTURE_FLAGS) * frames_per_batch
                    )

                    for tp_rank in range(1, 4):
                        with self.subTest(tp_rank=tp_rank):
                            frame_index = 0

                            def replay_broadcast(tensors, root, mode):
                                nonlocal frame_index
                                self.assertEqual((root, mode), (0, 0))
                                for tensor in tensors:
                                    self.assertLess(frame_index, len(frames))
                                    frame = frames[frame_index]
                                    self.assertEqual(tensor.shape, frame.shape)
                                    self.assertEqual(tensor.dtype, frame.dtype)
                                    self.assertEqual(tensor.device, frame.device)
                                    tensor.copy_(frame)
                                    frame_index += 1

                            # Replay actual root broadcasts, not independently encoded hints.
                            register_comm_ops(
                                replay_broadcast,
                                unexpected_collective,
                                unexpected_collective,
                                unexpected_collective,
                                unexpected_collective,
                            )
                            results = run_tp_sync_model_inputs_scenario(
                                tp_rank,
                                4,
                                _TP_SYNC_CAPTURE_FLAGS,
                                skip_run,
                                initial_flags,
                                grouped_cache,
                            )
                            self.assertSyncedBatches(results, skip_run, tp_rank)
                            self.assertEqual(frame_index, len(frames))

    def test_tp1_preserves_local_flags_without_broadcast(self) -> None:
        for skip_run in (False, True):
            with self.subTest(skip_run=skip_run):
                results = run_tp_sync_model_inputs_scenario(
                    0, 1, _TP_SYNC_CAPTURE_FLAGS, skip_run
                )
                self.assertSyncedBatches(results, skip_run)

    def test_tp4_receives_flags_and_payload_over_uds(self) -> None:
        # Spawn avoids forking an initialized CUDA runtime. CPU payloads exercise
        # real TP transport without requiring four GPUs or model weights.
        self._run_uds_ranks(_run_tp_sync_uds_rank, 4)

    def test_capture_submit_fatal_before_tp_logits(self) -> None:
        with patch.dict(os.environ, {_MOONCAKE_STORE_KEY_NAMESPACE_ENV: "TestActor"}):
            self._run_uds_ranks(_run_capture_submit_uds_rank, 2)

    def _run_uds_ranks(self, worker, size) -> None:
        context = multiprocessing.get_context("spawn")
        with tempfile.TemporaryDirectory(prefix="tp_sync_") as directory:
            processes = [
                context.Process(
                    target=worker,
                    args=(tp_rank, os.path.join(directory, "broadcast")),
                )
                for tp_rank in range(size)
            ]
            started = []
            try:
                for process in processes:
                    process.start()
                    started.append(process)
                for tp_rank, process in enumerate(processes):
                    process.join(timeout=30)
                    self.assertFalse(process.is_alive(), f"TP rank {tp_rank} timed out")
                    self.assertEqual(process.exitcode, 0, f"TP rank {tp_rank} failed")
            finally:
                for process in started:
                    if process.is_alive():
                        process.terminate()
                    process.join(timeout=5)


class PyWrappedModelCacheStoreIntegrationTest(unittest.TestCase):
    def test_borrowed_warmup_model_survives_until_owning_wrapper_is_destroyed(self):
        model = CloseAwareCacheStoreForwardModel()
        run_scenario(model, "multi_tag", owns_py_model=False)
        self.assertFalse(model.closed)
        self.assertEqual(model.close_calls, 0)

        run_scenario(model, "multi_tag")
        self.assertEqual(model.initialize_calls, 2)
        self.assertEqual(model.forward_calls, 2)
        self.assertTrue(model.closed)
        self.assertEqual(model.close_calls, 1)

    def test_failed_borrowed_wrapper_does_not_close_shared_model(self):
        model = CloseAwareCacheStoreForwardModel()
        model.fail_initialize = True
        with self.assertRaisesRegex(Exception, "injected model initialize failure"):
            run_scenario(model, "multi_tag", owns_py_model=False)
        self.assertFalse(model.closed)
        self.assertEqual(model.close_calls, 0)

        model.fail_initialize = False
        run_scenario(model, "multi_tag")
        self.assertEqual(model.forward_calls, 1)
        self.assertTrue(model.closed)
        self.assertEqual(model.close_calls, 1)

    def test_failed_owning_wrapper_closes_model(self):
        model = CloseAwareCacheStoreForwardModel()
        model.fail_initialize = True
        with self.assertRaisesRegex(Exception, "injected model initialize failure"):
            run_scenario(model, "multi_tag")
        self.assertTrue(model.closed)
        self.assertEqual(model.close_calls, 1)

    @unittest.skipIf(
        os.environ.get("TEST_USING_DEVICE") == "ROCM",
        "ROCm normalizes these lengths to the host metadata path",
    )
    def test_cuda_lengths_host_prefix_metadata_updates_on_current_stream(self):
        for custom_stream in (False, True):
            with self.subTest(custom_stream=custom_stream):
                stream = (
                    torch.cuda.Stream()
                    if custom_stream
                    else torch.cuda.default_stream()
                )
                model = MixedDeviceMetadataModel()
                with torch.cuda.stream(stream):
                    result = run_scenario(model, "cuda_lengths_host_prefix")
                stream.synchronize()
                self.assertEqual(
                    model.metadata,
                    [
                        ([0, 1, 4, 8], [0, 129, 388, 904]),
                        ([0, 1, 4, 8], [0, 257, 388, 776]),
                        ([0, 1, 4, 8], [0, 1, 4, 8]),
                    ],
                )
                self.assertEqual(result["records"], [])

    def capturePublisher(self, **kwargs):
        publisher = _extension.CapturePublisher(**kwargs)
        self.assertTrue(publisher.drain())  # Test-only initialization barrier.
        return publisher

    def consumeCapture(self, publisher):
        _FakeEagleMooncakeStore.instances[-1].consume()
        self.assertTrue(publisher.poll())
        self.assertEqual(publisher.inflight_samples, 0)
        self.assertEqual(publisher.inflight_bytes, 0)

    def test_successful_generation_prefill_capture_does_not_reserve_request_blocks(
        self,
    ) -> None:
        model = SuccessfulGenerationPrefillCaptureModel()
        result = run_generation_prefill_capture_scenario(model)

        self.assertFalse(result["saw_capture_error"])
        self.assertEqual(result["capture_error_message"], "")
        self.assertTrue(result["graph_enabled"])
        self.assertEqual(model.prefill_forward_calls, 4)
        self.assertEqual(result["available_during"], result["available_before"])
        self.assertEqual(result["available_after"], result["available_before"])

    def test_capture_bypasses_eligible_generation_graph_and_preserves_ordinary_replay(
        self,
    ) -> None:
        os.environ["MOONCAKE_HIDDEN_DIM"] = "4"
        # Exercise both independent flags, then switch back on the same runner
        # so stale prepare state or a permanently disabled graph cannot pass.
        flags = [
            (True, True),
            (False, False),
            (False, True),
            (True, False),
            (True, True),
            (False, False),
        ]
        result = run_generation_prefill_capture_scenario(
            CaptureAwareGenerationPrefillModel(), forward_flags=flags
        )

        self.assertTrue(result["graph_enabled"])
        self.assertFalse(result["saw_capture_error"])
        self.assertEqual(result["available_after"], result["available_before"])
        self.assertEqual(len(result["forwards"]), len(flags))
        for i, (forward, (skip_lm_head, capture)) in enumerate(
            zip(result["forwards"], flags, strict=True)
        ):
            with self.subTest(skip_lm_head=skip_lm_head, capture=capture, step=i):
                self.assertEqual(forward["python_forward_calls"], int(capture))
                self.assertEqual(forward["replayed"], not capture)
                self.assertEqual(forward["not_requested"], capture)
                self.assertEqual(forward["logits"] is None, skip_lm_head)
                self.assertTrue(forward["prepared_after_prepare"])
                self.assertTrue(forward["prepared_after_update"])
                self.assertFalse(forward["prepared_after_forward"])
                self.assertEqual(
                    forward["attention_metadata_after_prepare"]["kernel_block_ids"].tolist(),
                    [[0]],
                )
                metadata = forward["attention_metadata"]
                self.assertEqual(metadata["input_lengths"].tolist(), [3])
                self.assertEqual(
                    metadata["position_ids"].tolist(),
                    [0, 1, 2] if i % 2 == 0 else [2, 1, 0],
                )
                self.assertEqual(metadata["kernel_block_ids"].tolist(), [[i + 1]])
                tokens = torch.arange(1 + i, 4 + i, dtype=torch.bfloat16)
                hidden = tokens.unsqueeze(1).expand(-1, 4)
                expected_hidden = hidden
                torch.testing.assert_close(
                    forward["hidden_states"].cpu(), expected_hidden
                )
                torch.testing.assert_close(
                    forward["mtp_target_hidden_states"],
                    torch.cat((hidden + 40, hidden + 50), dim=-1),
                )
                if not skip_lm_head:
                    torch.testing.assert_close(
                        forward["logits"],
                        expected_hidden[-1:]
                        .float()
                        .sum(-1, keepdim=True)
                        .expand(-1, 2),
                    )

        self.assertEqual(len(_FakeEagleMooncakeStore.instances), 1)
        store = _FakeEagleMooncakeStore.instances[0]
        self.assertEqual(len(store.puts), sum(capture for _, capture in flags))
        self.assertEqual(
            [put["key"] for put in store.puts],
            [
                f"rtp_TestActor_{801 + i}"
                for i, (_, capture) in enumerate(flags)
                if capture
            ],
        )
        self.assertTrue(store.flushed)
        self.assertFalse(store.closed)

    def test_clean_generation_prefill_capture_failure_fails_init_without_cache_allocation(
        self,
    ) -> None:
        model = CleanFallbackGenerationPrefillCaptureModel()
        result = run_generation_prefill_capture_scenario(
            model, "injected clean generation prefill graph fallback"
        )

        self.assertTrue(result["saw_capture_error"])
        self.assertIn(
            "injected clean generation prefill graph fallback",
            result["capture_error_message"],
        )
        self.assertFalse(result["graph_enabled"])
        self.assertEqual(model.prefill_forward_calls, 1)
        self.assertEqual(result["available_during"], result["available_before"])
        self.assertEqual(result["available_after"], result["available_before"])

    def test_unsupported_generation_prefill_backend_fails_init_without_cache_allocation(
        self,
    ) -> None:
        model = UnsupportedBackendGenerationPrefillCaptureModel()
        result = run_generation_prefill_capture_scenario(
            model, "injected unsupported backend"
        )

        self.assertTrue(result["saw_capture_error"])
        self.assertIn("injected unsupported backend", result["capture_error_message"])
        self.assertFalse(result["graph_enabled"])
        self.assertEqual(model.prefill_forward_calls, 0)
        self.assertEqual(result["available_during"], result["available_before"])
        self.assertEqual(result["available_after"], result["available_before"])

    def test_late_constructor_failure_destroys_clean_graphs_without_cache_allocation(
        self,
    ) -> None:
        model = LateInitializeFailureGenerationPrefillCaptureModel()
        result = run_generation_prefill_capture_scenario(
            model, "Python model initialization failed"
        )

        self.assertEqual(model.initialize_calls, 2)
        self.assertEqual(model.prefill_forward_calls, 4)
        self.assertTrue(result["saw_capture_error"])
        self.assertIn(
            "Python model initialization failed", result["capture_error_message"]
        )
        self.assertFalse(result["graph_enabled"])
        self.assertEqual(result["available_during"], result["available_before"])
        self.assertEqual(result["available_after"], result["available_before"])

    def test_dirty_generation_prefill_capture_does_not_retain_cache_manager(
        self,
    ) -> None:
        # Dirty capture poisons process-local graph/allocator state by design;
        # the dedicated Bazel target runs this method alone.
        if os.environ.get("RTP_LLM_RUN_PYWRAPPED_DIRTY_CAPTURE_TEST") != "1":
            self.skipTest(
                "requires an isolated process for a dirty device graph capture"
            )

        model = DirtyGenerationPrefillCaptureModel()
        result = run_dirty_generation_prefill_capture_scenario(model)

        self.assertTrue(result["saw_dirty_capture_error"])
        self.assertEqual(model.prefill_forward_calls, 4)
        self.assertEqual(result["available_after"], result["available_before"])
        self.assertFalse(result["manager_retained"])

    def test_pp_boundary_preserves_activations_and_auxiliary_features(self) -> None:
        class BoundaryModel(CacheStoreForwardModel):
            def _forward_one(self, inputs: PyModelInputs) -> PyModelOutputs:
                super()._forward_one(inputs)
                self.boundaries = inputs.pp_intermediates
                self.engram_windows = inputs.engram_token_windows
                self.output_flags = (
                    inputs.need_all_logits,
                    inputs.need_all_hidden_states,
                )
                # A non-tail stage can supply only boundary tensors. The wrapper
                # must return before reading or validating a final hidden output.
                output = PyModelOutputs()
                output.pp_intermediates = self.boundaries
                return output

        model = BoundaryModel()
        result = run_scenario(model, "pp_boundary")

        self.assertEqual(model.capture_flags, [False])
        self.assertEqual(model.output_flags, (False, False))
        self.assertFalse(result["logits_defined"])
        self.assertEqual(result["forward_hidden_states"], [None])
        self.assertIsNone(result["mtp_target_hidden_states"])
        self.assertEqual(result["capture_diagnostics"], [""])
        self.assertFalse(result["capture_fatal"])
        self.assertEqual(result["capture_failure_count"], 0)
        rows = sum(model.seen_input_lengths[0])
        hidden = torch.arange(rows, dtype=torch.float16, device="cuda").reshape(rows, 1)
        expected = {
            "hidden_states": hidden,
            "residual": hidden + 10,
            "mtp_aux_hidden_states": torch.cat((hidden + 20, hidden + 30), dim=-1),
        }
        self.assertEqual(set(result["pp_intermediates"]), set(expected))
        for name, tensor in expected.items():
            actual = result["pp_intermediates"][name]
            torch.testing.assert_close(actual, tensor)
            self.assertEqual(actual.data_ptr(), model.boundaries[name].data_ptr())
        self.assertTrue(model.engram_windows.is_cuda)
        torch.testing.assert_close(
            model.engram_windows.cpu(),
            torch.arange(rows * 2, dtype=torch.int32).reshape(rows, 2),
        )
        self.assertTrue(result["records"])

    def test_capture_layout_failure_preserves_mtp_features(self) -> None:
        class MtpFeatureModel(MalformedCaptureLayoutModel):
            def _forward_one(self, inputs: PyModelInputs) -> PyModelOutputs:
                output = super()._forward_one(inputs)
                self.features = torch.full(
                    (inputs.input_ids.numel(), 2),
                    37.0,
                    dtype=torch.float16,
                    device=inputs.input_ids.device,
                )
                output.mtp_target_hidden_states = self.features
                return output

        model = MtpFeatureModel()
        result = run_scenario(model, "capture_bf16")

        self.assertEqual(model.capture_flags, [True])
        self.assertEqual(len(result["capture_diagnostics"]), 1)
        self.assertIn("expected packed width 3", result["capture_diagnostics"][0])
        self.assertEqual(result["forward_capture_error_codes"], [ExceptionType.EXECUTION_EXCEPTION])
        self.assertIn("expected packed width 3", result["forward_capture_error_messages"][0])
        self.assertCanonicalZero(result)
        self.assertEqual(result["capture_failure_count"], 1)
        self.assertCaptureQps(
            result["capture_metrics"],
            batch=1,
            publish_success=0,
            failure=1,
            layout_failure=1,
            hard_contract_failure=1,
        )
        self.assertCaptureStatus(result["capture_metrics"], enabled=0, broken=1)
        self.assertEqual(result["pp_intermediates"], {})
        torch.testing.assert_close(
            result["mtp_target_hidden_states"], torch.full_like(model.features, 37.0)
        )
        self.assertEqual(
            result["mtp_target_hidden_states"].data_ptr(), model.features.data_ptr()
        )
        self.assertEqual(_FakeEagleMooncakeStore.instances[-1].puts, [])

    def test_mtp_placeholders_preserve_model_tag_set_with_extra_payload_rows(self):
        for scenario in ("mtp_placeholders", "mtp_placeholders_extra_payload"):
            with self.subTest(scenario=scenario):
                model = CacheStoreForwardModel()
                result = run_scenario(model, scenario)
                self.assertEqual(model.seen_direct_inputs, [False])
                self.assertEqual(
                    model.seen_attention_tables,
                    [{"unused": [[3, 4]], "draft": [[1, 2]], "other": [[5, 6]]}],
                )
                self.assertEqual(set(model.seen_cache_tables[0]), {"draft"})
                self.assertEqual(set(result["base_addresses"]), {"draft"})
                record = _record_for_request(result, 401)
                self.assertEqual(
                    sorted(
                        block["address"] - result["base_addresses"]["draft"]
                        for block in record["blocks"]
                    ),
                    [32, 64],
                )

    def test_missing_mtp_placeholder_is_rejected_before_forward(self):
        model = CacheStoreForwardModel()
        with self.assertRaisesRegex(
            RuntimeError, "missing model cache group tag=unused"
        ):
            run_scenario(model, "mtp_missing_placeholder")
        self.assertEqual(model.forward_calls, 0)

    def test_single_model_group_selects_direct_input_from_shared_payload(self):
        model = CacheStoreForwardModel()
        result = run_scenario(model, "single_model_extra_payload")
        self.assertEqual(model.seen_direct_inputs, [True])
        self.assertEqual(
            model.seen_cache_tables,
            [{"draft": {"physical": [[1, 2]], "kernel": [[1, 2]]}}],
        )
        record = _record_for_request(result, 401)
        self.assertEqual(
            sorted(
                block["address"] - result["base_addresses"]["draft"]
                for block in record["blocks"]
            ),
            [32, 64],
        )

    def test_cacheless_multigroup_warmup_exposes_single_input(self):
        class WarmupModel(CacheStoreForwardModel):
            def _forward_one(self, inputs):
                assert self.kv_cache is None
                attention_inputs = inputs.attention_inputs
                assert not isinstance(attention_inputs, dict)
                assert attention_inputs.is_prefill
                assert not attention_inputs.is_target_verify
                assert attention_inputs.kv_cache_block_id is None
                assert attention_inputs.kv_cache_block_id_device is None
                assert attention_inputs.kv_cache_kernel_block_id is None
                assert attention_inputs.kv_cache_kernel_block_id_device is None
                assert attention_inputs.cache_store_inputs is None
                return PyModelOutputs(
                    torch.zeros(
                        (inputs.input_ids.numel(), 1),
                        dtype=torch.float16,
                        device=inputs.input_ids.device,
                    )
                )

        model = WarmupModel()
        result = run_scenario(model, "cacheless_warmup")

        self.assertEqual(model.forward_calls, 1)
        self.assertEqual(result["records"], [])

    def setUp(self) -> None:
        _register_identity_allreduce()
        self.addCleanup(clear_comm_ops)
        previous_torchspec_modules = _install_fake_torchspec_modules()
        self.addCleanup(_restore_torchspec_modules, previous_torchspec_modules)
        _FakeEagleMooncakeStore.instances.clear()
        _FakeEagleMooncakeStore.events = None
        _FakeMooncakeConfig.store_key_namespace = "TestActor"
        _FakeMooncakeConfig.make_store_key_calls.clear()
        _FakeMooncakeConfig.hidden_dim = 1
        _FakeMooncakeConfig.max_seq_len = 8192
        _FakeMooncakeConfig.get_batch_size = 1
        _FakeMooncakeConfig.gpu_buffer_size = None
        _FakeMooncakeConfig.num_aux_layers = 2
        _FakeMooncakeConfig.async_put_pool_size = 1
        _FakeMooncakeConfig.enable_gpu_direct = False
        mooncake_env_names = (
            *_MOONCAKE_RTP_LAYOUT_ENV,
            _MOONCAKE_STORE_KEY_NAMESPACE_ENV,
            "MOONCAKE_ASYNC_PUT_POOL_SIZE",
            "MOONCAKE_ENABLE_GPU_DIRECT",
        )
        previous_layout_env = {
            env_name: os.environ.get(env_name) for env_name in mooncake_env_names
        }
        for env_name in mooncake_env_names:
            os.environ.pop(env_name, None)
        os.environ[_MOONCAKE_STORE_KEY_NAMESPACE_ENV] = "TestActor"
        # Existing contract cases intentionally publish both requests in one PUT.
        # Capacity-one coverage below leaves the splitting to the C++ publisher.
        os.environ["MOONCAKE_ASYNC_PUT_POOL_SIZE"] = "2"
        self.addCleanup(_restore_mooncake_layout_env, previous_layout_env)
        _FakeEagleMooncakeStore.fail_put = False
        _FakeEagleMooncakeStore.fail_put_count = 0
        _FakeEagleMooncakeStore.fail_flush = False
        _FakeEagleMooncakeStore.fail_warmup = False
        _FakeEagleMooncakeStore.async_error_after_accept_count = 0

    def assertCaptureQps(
        self,
        metrics: dict[str, float],
        *,
        batch: int,
        publish_success: int,
        failure: int,
        **overrides: int,
    ) -> None:
        expected = dict.fromkeys(_CAPTURE_QPS_METRICS, 0)
        counts = dict(batch=batch, publish_success=publish_success, failure=failure)
        counts.update(overrides)
        for name, value in counts.items():
            suffix = name if name == "batch_cancelled_requests" else f"{name}_qps"
            metric_name = f"{_CAPTURE_METRIC_PREFIX}{suffix}"
            self.assertIn(metric_name, expected, f"unknown capture QPS metric: {name}")
            expected[metric_name] = value
        self.assertEqual(
            {name: metrics[name] for name in _CAPTURE_QPS_METRICS}, expected
        )

    def assertCaptureQuota(self, metrics, samples, byte_count, tokens) -> None:
        for name, expected in (("samples", samples), ("bytes", byte_count), ("tokens", tokens)):
            self.assertEqual(metrics[f"{_CAPTURE_METRIC_PREFIX}quota_{name}"], expected)

    def assertCaptureStatus(
        self,
        metrics: dict[str, float],
        *,
        enabled: int,
        broken: int,
    ) -> None:
        self.assertEqual(
            {name: metrics[name] for name in _CAPTURE_STATUS_METRICS},
            {
                f"{_CAPTURE_METRIC_PREFIX}enabled": enabled,
                f"{_CAPTURE_METRIC_PREFIX}broken": broken,
            },
        )

    def assertCanonicalZero(self, result, rows=5, *, fatal=False) -> None:
        self.assertEqual(result["capture_fatal"], fatal)
        for hidden in result["forward_hidden_states"]:
            self.assertEqual(tuple(hidden.shape), (rows, 1))
            torch.testing.assert_close(hidden, torch.zeros_like(hidden), rtol=0, atol=0)

    def assertCaptureDisabled(self, result, diagnostic, rows=5) -> None:
        self.assertCanonicalZero(result, rows=rows)
        self.assertIn(diagnostic, result["capture_diagnostics"][0])
        self.assertEqual(result["capture_failure_count"], 1)
        self.assertCaptureStatus(result["capture_metrics"], enabled=0, broken=1)
        self.assertEqual(
            result["capture_metrics"][
                f"{_CAPTURE_METRIC_PREFIX}initialization_failure_qps"
            ],
            1,
        )

    def assertCaptureMetricTypes(self, result: dict) -> None:
        metrics = result["capture_metrics"]
        metric_types = result["capture_metric_types"]
        self.assertEqual(set(metric_types), set(metrics))
        for name in metrics:
            if name in _CAPTURE_QPS_METRICS:
                self.assertEqual(metric_types[name], _KMONITOR_QPS)
            elif name in _CAPTURE_STATUS_METRICS:
                self.assertEqual(metric_types[name], _KMONITOR_STATUS)
            else:
                self.assertEqual(metric_types[name], _KMONITOR_GAUGE)

    def test_speculative_commit_hooks_must_be_implemented_as_a_pair(self) -> None:
        with self.assertRaisesRegex(RuntimeError, "must implement.*as a pair"):
            run_speculative_commit_bridge(
                PrepareOnlySpeculativeCommitBridgeModel(), "has_hooks"
            )

    def test_speculative_commit_explicit_capability_disables_inactive_hooks(self):
        model = SpeculativeCommitBridgeModel()
        self.assertTrue(run_speculative_commit_bridge(model, "has_hooks"))
        model.requires_speculative_target_commit = lambda: False
        self.assertFalse(run_speculative_commit_bridge(model, "has_hooks"))
        model.requires_speculative_target_commit = lambda: True
        self.assertTrue(run_speculative_commit_bridge(model, "has_hooks"))
        self.assertFalse(run_speculative_commit_bridge(model, "has_hooks", False))

    def test_cpu_status_bridge_preserves_false_and_ignores_non_target(self):
        model = SpeculativeCommitBridgeModel()
        calls = []
        model.has_speculative_target_status_reducer = lambda: True
        model.reduce_speculative_target_prepare_status = lambda ok: (
            calls.append(ok),
            False,
        )[1]
        self.assertFalse(run_speculative_commit_bridge(model, "reduce_true"))
        self.assertFalse(run_speculative_commit_bridge(model, "reduce_false"))
        self.assertEqual(calls, [True, False])
        self.assertIsNone(run_speculative_commit_bridge(model, "reduce_true", False))
        self.assertEqual(calls, [True, False])

    def test_cpu_status_bridge_optional_and_disabled_capabilities(self):
        model = SpeculativeCommitBridgeModel()
        self.assertIsNone(run_speculative_commit_bridge(model, "reduce_true"))
        model.has_speculative_target_status_reducer = lambda: False
        model.reduce_speculative_target_prepare_status = lambda ok: (
            _ for _ in ()
        ).throw(AssertionError("disabled hook"))
        self.assertIsNone(run_speculative_commit_bridge(model, "reduce_true"))

    def test_cpu_status_bridge_transport_errors_propagate(self):
        model = SpeculativeCommitBridgeModel()
        model.has_speculative_target_status_reducer = lambda: True

        def fail(ok):
            raise RuntimeError("status transport poisoned")

        model.reduce_speculative_target_prepare_status = fail
        with self.assertRaisesRegex(RuntimeError, "status transport poisoned"):
            run_speculative_commit_bridge(model, "reduce_true")

    def test_speculative_commit_prepare_exception_becomes_error(self) -> None:
        model = SpeculativeCommitBridgeModel(prepare_error=True)

        self.assertTrue(run_speculative_commit_bridge(model, "has_hooks"))
        error = run_speculative_commit_bridge(model, "prepare")

        self.assertIn("prepare bridge boom", error)

    def test_speculative_commit_abort_exception_does_not_escape(self) -> None:
        model = SpeculativeCommitBridgeModel(abort_error=True)

        self.assertIsNone(run_speculative_commit_bridge(model, "abort"))
        self.assertEqual(model.finish_calls, [False])

    def test_speculative_commit_exception_becomes_error(self) -> None:
        model = SpeculativeCommitBridgeModel(commit_error=True)

        error = run_speculative_commit_bridge(model, "commit")

        self.assertIn("commit bridge boom", error)
        self.assertEqual(model.finish_calls, [True])

    def test_non_target_wrapper_ignores_speculative_commit_hooks(self) -> None:
        self.assertFalse(
            run_speculative_commit_bridge(
                PrepareOnlySpeculativeCommitBridgeModel(),
                "has_hooks",
                False,
            )
        )

    def test_multi_tag_uses_each_tag_local_physical_block_table(self) -> None:
        model = CacheStoreForwardModel()
        result = run_scenario(model, "multi_tag")

        self.assertEqual(model.forward_calls, 1)
        self.assertFalse(result["logits_defined"])
        self.assertEqual(len(result["records"]), 2)
        self.assertEqual(model.seen_kernel_tables["full"], [[1, 2, -1, -1]])
        self.assertEqual(model.seen_kernel_tables["linear"], [[3, 4, 5, 6]])
        blocks = _blocks_by_key(result)

        full_blocks = {
            key: block for key, block in blocks.items() if "_tag_full" in key
        }
        linear_blocks = {
            key: block for key, block in blocks.items() if "_tag_linear" in key
        }
        self.assertEqual(len(full_blocks), 2)
        self.assertEqual(len(linear_blocks), 4)
        self.assertEqual(
            sorted(
                block["address"] - result["base_addresses"]["full"]
                for block in full_blocks.values()
            ),
            [16, 32],
        )
        self.assertEqual(
            sorted(
                block["address"] - result["base_addresses"]["linear"]
                for block in linear_blocks.values()
            ),
            [72, 96, 120, 144],
        )
        self.assertEqual({block["length"] for block in full_blocks.values()}, {16})
        self.assertEqual({block["length"] for block in linear_blocks.values()}, {24})

    def test_cyclic_chunks_separate_attention_and_cache_store_views(self) -> None:
        model = CacheStoreForwardModel()
        result = run_scenario(model, "cyclic_chunk_views")

        tables = model.seen_cache_tables[0]
        self.assertEqual(tables["swa"]["physical"], tables["swa"]["kernel"])
        self.assertEqual(
            tables["swa"]["physical"],
            [[3, 4, 3, 4, 0, 0, 0, 0], [5, 6, 5, 6, 0, 0, 0, 0]],
        )
        self.assertEqual(tables["full"]["physical"], [[1, 2, 3, 4], [5, 6, 0, 0]])
        self.assertNotEqual(tables["full"]["physical"], tables["full"]["kernel"])

        self.assertEqual(
            sorted(record["request_id"] for record in result["records"]),
            ["301", "301", "302"],
        )
        middle = _record_for_request(result, 302)
        self.assertTrue(all("_tag_full" in block["key"] for block in middle["blocks"]))
        blocks = _blocks_by_key(result)
        for tag, stride, expected in (
            ("swa", 24, {3103: 3, 3104: 4}),
            ("full", 16, {3103: 3, 3104: 4, 3201: 5, 3202: 6}),
        ):
            tag_blocks = {
                key: block for key, block in blocks.items() if f"_tag_{tag}" in key
            }
            self.assertEqual(len(tag_blocks), len(expected))
            self.assertEqual(
                {block["length"] for block in tag_blocks.values()}, {stride}
            )
            for token_key, block_id in expected.items():
                matches = [
                    block
                    for key, block in tag_blocks.items()
                    if f"_token_id_str_{token_key}_" in key
                ]
                self.assertEqual(len(matches), 1)
                self.assertEqual(
                    matches[0]["address"] - result["base_addresses"][tag],
                    block_id * stride,
                )

    def test_invalid_payload_identity_is_rejected_before_forward(self) -> None:
        for scenario in (
            "duplicate_tags",
            "empty_tag",
            "missing_tags",
            "unknown_tag",
            "physical_group_mismatch",
            "type_group_mismatch",
            "multi_group_2d",
            "single_group_2d_unknown_tag",
        ):
            with self.subTest(scenario=scenario):
                model = CacheStoreForwardModel()
                with self.assertRaises(RuntimeError):
                    run_scenario(model, scenario)
                self.assertEqual(model.forward_calls, 0)

    def test_single_group_2d_preserves_direct_cache_store_path(self) -> None:
        for scenario in ("single_group_2d", "single_group_2d_no_tags"):
            with self.subTest(scenario=scenario):
                model = CacheStoreForwardModel()
                result = run_scenario(model, scenario)
                record = _record_for_request(result, 401)
                base = result["base_addresses"]["draft"]
                self.assertEqual(
                    sorted(block["address"] - base for block in record["blocks"]),
                    [32, 64],
                )
                self.assertEqual(model.seen_kernel_tables["draft"], [[1, 2]])

    @_skip_rocm_context_parallel
    def test_context_parallel_publishes_original_lengths_to_every_tag(self) -> None:
        rank_key_sets = []
        for tp_rank in range(2):
            model = CacheStoreForwardModel()
            result = run_scenario(model, f"cp_actual_lengths_rank{tp_rank}")

            # Forward CP turns the six-token request into a four-token local
            # chunk for attention, but kv_cache_sharded=false means both ranks
            # publish the same complete cache namespace. Tag-local blocks keep
            # their own physical table and terminal-key stride.
            self.assertEqual(model.seen_input_lengths, [[4]])
            self.assertFalse(result["logits_defined"])
            self.assertEqual(len(result["records"]), 2)
            blocks = _blocks_by_key(result)
            full_blocks = {
                key: block for key, block in blocks.items() if "_tag_full" in key
            }
            linear_blocks = {
                key: block for key, block in blocks.items() if "_tag_linear" in key
            }
            self.assertEqual(len(full_blocks), 3)
            self.assertEqual(len(linear_blocks), 6)
            self.assertEqual(
                sorted(
                    block["address"] - result["base_addresses"]["full"]
                    for block in full_blocks.values()
                ),
                [16, 32, 48],
            )
            self.assertEqual(
                sorted(
                    block["address"] - result["base_addresses"]["linear"]
                    for block in linear_blocks.values()
                ),
                [72, 96, 120, 144, 168, 192],
            )
            self.assertEqual({block["length"] for block in full_blocks.values()}, {16})
            self.assertEqual(
                {block["length"] for block in linear_blocks.values()}, {24}
            )
            for token_key in range(3101, 3107):
                self.assertTrue(
                    any(f"_token_id_str_{token_key}_" in key for key in linear_blocks)
                )
            for token_key in (3102, 3104, 3106):
                self.assertTrue(
                    any(f"_token_id_str_{token_key}_" in key for key in full_blocks)
                )
            rank_key_sets.append(set(blocks))

        self.assertEqual(rank_key_sets[0], rank_key_sets[1])

    def test_mtp_writer_uses_selected_sub_config_for_real_write(self) -> None:
        model = CacheStoreForwardModel()
        result = run_scenario(model, "mtp_sub_config")

        record = _record_for_request(result, 401)
        self.assertEqual(len(record["blocks"]), 2)
        base = result["base_addresses"]["draft"]
        self.assertEqual(
            sorted(block["address"] - base for block in record["blocks"]),
            [32, 64],
        )
        self.assertEqual({block["length"] for block in record["blocks"]}, {32})
        self.assertTrue(
            all("model_id_7_" in block["key"] for block in record["blocks"])
        )
        self.assertTrue(all("_tag_draft" in block["key"] for block in record["blocks"]))

    def test_prefill_only_pd_does_not_write_decode_handoff(self) -> None:
        result = run_scenario(CacheStoreForwardModel(), "prefill_only_pd")

        self.assertFalse(result["logits_defined"])
        self.assertEqual(result["records"], [])

    def test_pd_prefill_without_cache_store_skips_write_cycle(self) -> None:
        result = run_scenario(
            CacheStoreForwardModel(), "prefill_only_pd_cache_store_disabled"
        )

        self.assertEqual(result["records"], [])

    def test_skip_lm_head_without_explicit_capture_does_not_publish(self) -> None:
        model = CacheStoreForwardModel()

        result = run_scenario(model, "capture_explicit_off")

        self.assertEqual(model.capture_flags, [False])
        self.assertFalse(result["logits_defined"])
        self.assertEqual(_FakeEagleMooncakeStore.instances[0].puts, [])
        self.assertCaptureQps(
            result["capture_metrics"], batch=0, publish_success=0, failure=0
        )

    def assertCaptureProfileStoreCalls(self, events, store_events, *, shutdown_flush_failed=False, quarantined=False) -> None:
        prefix = "hidden_state_capture."
        capture_events = [e for e in events if e.name.startswith(prefix)]
        # Only the producer's post-join shutdown flush is profiled. The real
        # worker still publishes, but must not inherit RecordFunction callbacks.
        expected = []
        if "store.flush" in store_events and not quarantined:
            expected = [prefix + "flush", prefix + "flush_python"]
            if not shutdown_flush_failed:
                expected += [prefix + "observe_async_errors", prefix + "take_async_errors"]
        self.assertEqual([e.name for e in capture_events], expected)
        self.assertFalse(any(e.name.startswith("rtp-forward-") for e in events))
        for event in capture_events:
            self.assertLessEqual(event.time_range.start, event.time_range.end)
            if event.name == prefix + "flush":
                self.assertIsNone(event.cpu_parent)
                continue
            self.assertIsNotNone(event.cpu_parent)
            parent = "observe_async_errors" if event.name == prefix + "take_async_errors" else "flush"
            self.assertEqual(event.cpu_parent.name, prefix + parent)
            self.assertLessEqual(
                event.cpu_parent.time_range.start, event.time_range.start
            )
            self.assertLessEqual(
                event.time_range.end, event.cpu_parent.time_range.end
            )
        if len(capture_events) == 4:
            self.assertLessEqual(capture_events[1].time_range.end, capture_events[2].time_range.start)

    def test_capture_profile_scopes_preserve_forward_contract(self) -> None:
        _register_identity_allreduce()
        self.addCleanup(clear_comm_ops)
        scenarios = [
            "capture_bf16",
            "capture_non_owner",
            "capture_explicit_off",
        ]
        if os.environ.get("TEST_USING_DEVICE") != "ROCM":
            scenarios += ["capture_fp8", "capture_cp"]
        for scenario in scenarios:
            namespace = (
                "ProfileActor" + "A" * 256
                if scenario == "capture_bf16"
                else "TestActor"
            )
            with (
                self.subTest(scenario=scenario),
                patch.dict(os.environ, {_MOONCAKE_STORE_KEY_NAMESPACE_ENV: namespace}),
            ):
                model_type = (
                    PositionAwareCaptureModel
                    if scenario == "capture_cp"
                    else CacheStoreForwardModel
                )
                baseline_calls = []
                _FakeEagleMooncakeStore.events = baseline_calls
                baseline = run_scenario(model_type(), scenario)
                store_calls = []
                _FakeEagleMooncakeStore.events = store_calls
                with torch.profiler.profile(
                    activities=[torch.profiler.ProfilerActivity.CPU]
                ) as prof:
                    result = run_scenario(model_type(), scenario)
                self.assertEqual(result["capture_fatal"], baseline["capture_fatal"])
                self.assertEqual(
                    result["capture_diagnostics"], baseline["capture_diagnostics"]
                )
                self.assertEqual(result["hidden_width"], baseline["hidden_width"])
                for name in _CAPTURE_QPS_METRICS | _CAPTURE_STATUS_METRICS:
                    self.assertEqual(
                        result["capture_metrics"][name],
                        baseline["capture_metrics"][name],
                    )
                self.assertEqual(
                    len(result["forward_hidden_states"]),
                    len(baseline["forward_hidden_states"]),
                )
                for actual, expected in zip(
                    result["forward_hidden_states"], baseline["forward_hidden_states"]
                ):
                    torch.testing.assert_close(actual, expected)
                events = prof.events()
                self.assertEqual(store_calls, baseline_calls)
                self.assertCaptureProfileStoreCalls(events, store_calls)
                batch_ids = [
                    call.split(":", 1)[1]
                    for call in store_calls
                    if call.startswith("store.put_batch:")
                ]
                expected_calls = []
                if scenario != "capture_non_owner":
                    expected_calls = ["store.setup", "store.warmup"]
                    if scenario != "capture_explicit_off":
                        self.assertEqual(len(batch_ids), 1)
                        expected_calls += [
                            "store.take_async_errors",
                            f"store.put_batch:{batch_ids[0]}",
                            "store.flush",
                            "store.take_async_errors",
                        ]
                    expected_calls += [
                        "store.flush",
                        "store.take_async_errors",
                    ]
                    retains_source = scenario != "capture_explicit_off"
                    if not retains_source:
                        expected_calls.append("store.close")
                    # A local flush cannot release unconsumed source-key ownership.
                    store = _FakeEagleMooncakeStore.instances[-1]
                    self.assertTrue(store.flushed)
                    self.assertEqual(bool(store.keys), retains_source)
                    self.assertEqual(store.closed, not retains_source)
                self.assertEqual(store_calls, expected_calls)
                if scenario in ("capture_non_owner", "capture_explicit_off"):
                    continue
                if scenario == "capture_bf16":
                    self.assertGreater(len(batch_ids[0]), 256)
                for name in (
                    "publish_request_count",
                    "publish_token_count",
                    "publish_payload_bytes",
                ):
                    key = f"{_CAPTURE_METRIC_PREFIX}{name}"
                    self.assertEqual(
                        result["capture_metrics"][key],
                        baseline["capture_metrics"][key],
                    )
                layout = next(
                    e for e in events if e.name == "py_model.capture_layout_status"
                )
                self.assertLessEqual(layout.time_range.start, layout.time_range.end)
                if scenario == "capture_bf16":
                    # TP=1 agrees on the host; a status tensor or scalar read would touch the producer hot path.
                    self.assertFalse(any(e.name.startswith("aten::") for e in layout.cpu_children))
                if scenario == "capture_cp":
                    restore = next(
                        e for e in events if e.name == "py_model.capture_cp_restore"
                    )
                    self.assertLessEqual(
                        layout.time_range.end, restore.time_range.start
                    )
                    self.assertLessEqual(restore.time_range.start, restore.time_range.end)

    def test_capture_profile_scopes_close_on_admission_and_flush_failure(self) -> None:
        original_poll = _FakeEagleMooncakeStore.take_async_errors
        for failure, poll_failure_at in (
            ("fail_put", 0),
            ("fail_flush", 0),
            ("poll_raises", 1),
            ("poll_invalid", 1),
            ("poll_raises", 2),
            ("poll_invalid", 2),
        ):
            poll_count = 0

            def poll(store):
                nonlocal poll_count
                poll_count += 1
                errors = original_poll(store)
                if poll_count == poll_failure_at:
                    if failure == "poll_raises":
                        raise RuntimeError("injected async-error polling failure")
                    return None
                return errors

            with (
                self.subTest(failure=failure, poll=poll_failure_at),
                patch.dict(os.environ, {"MOONCAKE_ENABLE_GPU_DIRECT": "1" if failure == "fail_put" else "0"}),
                patch.multiple(
                    _FakeEagleMooncakeStore,
                    fail_put=failure == "fail_put",
                    fail_flush=failure == "fail_flush",
                ),
                patch.object(_FakeEagleMooncakeStore, "take_async_errors", poll),
            ):
                store_calls = []
                _FakeEagleMooncakeStore.events = store_calls
                with torch.profiler.profile(
                    activities=[torch.profiler.ProfilerActivity.CPU]
                ) as prof:
                    result = run_scenario(CacheStoreForwardModel(), "capture_bf16")
                quarantined = failure == "poll_invalid" or poll_failure_at == 2
                self.assertCanonicalZero(result, fatal=quarantined)
                self.assertTrue(result["capture_diagnostics"][0])
                metrics = result["capture_metrics"]
                self.assertEqual(
                    metrics[f"{_CAPTURE_METRIC_PREFIX}publish_success_qps"], 0
                )
                persistent = failure == "fail_flush"
                if persistent:
                    self.assertIn(
                        "injected Mooncake flush failure",
                        result["capture_diagnostics"][0],
                    )
                failures = store_calls.count("store.flush") if persistent else 1
                if persistent:
                    # One publication attempt and one post-join shutdown drain are mandatory.
                    self.assertGreaterEqual(failures, 2)
                self.assertCaptureQps(
                    metrics,
                    batch=1,
                    publish_success=0,
                    failure=failures,
                    store_failure=failures - int(persistent),
                    shutdown_failure=int(persistent),
                    operational_failure=failures - int(failure == "poll_raises" and poll_failure_at == 2),
                )
                self.assertCaptureStatus(metrics, enabled=0, broken=int(quarantined))
                events = prof.events()
                self.assertCaptureProfileStoreCalls(events, store_calls, shutdown_flush_failed=persistent, quarantined=quarantined)
                self.assertIn("py_model.capture_layout_status", {e.name for e in events})
                expected_calls = [
                    "store.setup",
                    "store.warmup",
                    "store.take_async_errors",
                ]
                store = _FakeEagleMooncakeStore.instances[-1]
                if poll_failure_at != 1:
                    self.assertEqual(len(store.put_batch_calls), 1)
                    batch_id = store.put_batch_calls[0]["batch_id"]
                    expected_calls.append(f"store.put_batch:{batch_id}")

                    if failure != "fail_put":
                        expected_calls.append("store.flush")
                        if not persistent:
                            expected_calls.append("store.take_async_errors")
                    # A failed PUT still needs its first drain; a failed flush waits for its retry deadline.
                    if failure == "fail_put":
                        expected_calls.extend(["store.flush", "store.take_async_errors"])
                else:
                    self.assertEqual(store.put_batch_calls, [])

                retains_source = quarantined or (poll_failure_at != 1 and failure != "fail_put")
                if not quarantined:
                    expected_calls.append("store.flush")
                    if not persistent:
                        expected_calls.append("store.take_async_errors")
                    if not retains_source:
                        expected_calls.append("store.close")
                if persistent:
                    # Maintenance may retry drain, but never polls completion errors
                    # after a failed flush and never repeats PUT or closes the store.
                    expected_calls = expected_calls[:4] + ["store.flush"] * failures
                self.assertEqual(store_calls, expected_calls)
                self.assertEqual(store.flushed, "store.flush" in expected_calls)
                self.assertEqual(store.closed, not retains_source)

    def test_capture_forward_returns_while_store_is_blocked(self) -> None:
        entered = threading.Event()
        release = threading.Event()
        timed_out = threading.Event()
        profiler_stopped = threading.Event()
        worker_completed = []
        producer_thread = threading.get_ident()
        producer_streams = []

        class ObservedProducer(CacheStoreForwardModel):
            def forward(self, inputs, fmha_impl=None):
                producer_streams.append(torch.cuda.current_stream().cuda_stream)
                with torch.profiler.record_function("capture.producer_forward"):
                    return super().forward(inputs, fmha_impl)

        worker_identity = []
        original_put = _FakeEagleMooncakeStore.put_batch

        def blocked_put(store, **kwargs):
            worker_identity.append(
                (threading.get_ident(), torch.cuda.current_stream().cuda_stream)
            )
            # Both this range and the real C++ publish/admit scopes span stop().
            with torch.profiler.record_function("capture.worker_blocked"):
                entered.set()
                if not release.wait(10):
                    timed_out.set()
                    raise RuntimeError("serving forward blocked on capture store")
                with torch.profiler.record_function("capture.worker_after_stop"):
                    result = original_put(store, **kwargs)
                worker_completed.append(profiler_stopped.is_set())
                return result

        def after_forward(_index):
            try:
                self.assertTrue(entered.wait(10), "worker did not enter PUT")
                self.assertFalse(timed_out.is_set())
                self.assertFalse(release.is_set())
                # Stop on the producer, before the fixture's test-only drain.
                profiler_scope.close()
                profiler_stopped.set()
                self.assertFalse(worker_completed)
                self.assertFalse(release.is_set())
            finally:
                release.set()

        model = ObservedProducer()
        with patch.object(_FakeEagleMooncakeStore, "put_batch", blocked_put):
            with contextlib.ExitStack() as profiler_scope:
                prof = profiler_scope.enter_context(
                    torch.profiler.profile(activities=[torch.profiler.ProfilerActivity.CPU])
                )
                try:
                    result = run_scenario(
                        model,
                        "capture_with_lm_head",
                        after_forward=after_forward,
                    )
                finally:
                    release.set()
        self.assertTrue(profiler_stopped.is_set())
        self.assertFalse(timed_out.is_set())
        self.assertEqual(worker_completed, [True])
        names = {event.name for event in prof.events()}
        self.assertIn("capture.producer_forward", names)
        self.assertIn("py_model.capture_layout_status", names)
        self.assertFalse(any(name.startswith("capture.worker_") for name in names))
        self.assertFalse(any(name.startswith("hidden_state_capture.") for name in names))
        self.assertFalse(any(name.startswith("rtp-forward-") for name in names))
        self.assertCanonicalZero(result)
        self.assertEqual(model.capture_flags, [True])
        self.assertTrue(result["logits_defined"])
        self.assertEqual(tuple(result["logits"].shape), (2, 2))
        torch.testing.assert_close(
            result["logits"], torch.zeros_like(result["logits"]), rtol=0, atol=0
        )
        self.assertEqual(len(_FakeEagleMooncakeStore.instances[-1].puts), 2)
        self.assertEqual(len(worker_identity), 1)
        self.assertNotEqual(worker_identity[0][0], producer_thread)
        self.assertEqual(len(producer_streams), 1)
        self.assertNotEqual(worker_identity[0][1], producer_streams[0])
        self.assertCaptureQps(
            result["capture_metrics"],
            batch=1,
            publish_success=1,
            failure=0,
            bf16_publish=1,
        )

    def test_capture_device_lengths_publish_on_regular_path(self) -> None:
        class ObservedLengths(CacheStoreForwardModel):
            def __init__(self):
                super().__init__()
                self.execution_lengths = []

            def _forward_one(self, inputs):
                attention_inputs = inputs.attention_inputs
                groups = (
                    list(attention_inputs.values())
                    if isinstance(attention_inputs, dict)
                    else [attention_inputs]
                )
                snapshots = []
                for attention in groups:
                    fields = {}
                    for name in ("input_lengths", "input_lengths_device"):
                        tensor = getattr(attention, name)
                        # Freeze entry metadata/values, not a reusable PyAttentionInputs object.
                        fields[name] = {
                            "device": tensor.device,
                            "dtype": tensor.dtype,
                            "sizes": tuple(tensor.shape),
                            "strides": tensor.stride(),
                            "storage_ptr": tensor.untyped_storage().data_ptr(),
                            "storage_offset": tensor.storage_offset(),
                            "data_ptr": tensor.data_ptr(),
                            "values": tensor.detach().cpu().clone(),
                        }
                    snapshots.append(fields)
                self.execution_lengths.append(snapshots)
                return super()._forward_one(inputs)

        model = ObservedLengths()
        result = run_scenario(model, "capture_cuda_lengths")
        source = result["source_input_lengths"]
        self.assertEqual(source.dtype, torch.int32)
        self.assertEqual(source.cpu().tolist(), [2, 3])
        expected_lengths = [[2, 3]]
        self.assertEqual(len(model.execution_lengths), len(expected_lengths))
        offset = 0
        for groups, values in zip(
            model.execution_lengths, expected_lengths, strict=True
        ):
            self.assertTrue(groups)
            expected_slice = source.narrow(0, offset, len(values))
            for fields in groups:
                for name in ("input_lengths", "input_lengths_device"):
                    snapshot = fields[name]
                    self.assertEqual(snapshot["dtype"], torch.int32)
                    self.assertEqual(snapshot["sizes"], (len(values),))
                    torch.testing.assert_close(
                        snapshot["values"],
                        torch.tensor(values, dtype=torch.int32),
                        rtol=0,
                        atol=0,
                    )
                    # ROCm deliberately uses the host metadata fallback.
                    if os.environ.get("TEST_USING_DEVICE") == "CUDA":
                        self.assertTrue(source.is_cuda)
                        self.assertEqual(snapshot["device"], source.device)
                        self.assertEqual(
                            snapshot["storage_ptr"],
                            source.untyped_storage().data_ptr(),
                        )
                        self.assertEqual(
                            snapshot["storage_offset"],
                            expected_slice.storage_offset(),
                        )
                        self.assertEqual(
                            snapshot["data_ptr"], expected_slice.data_ptr()
                        )
                        self.assertEqual(snapshot["strides"], expected_slice.stride())
            offset += len(values)
        self.assertEqual(offset, source.numel())
        self.assertCanonicalZero(result)
        self.assertEqual(result["capture_failure_count"], 0)
        store = _FakeEagleMooncakeStore.instances[-1]
        self.assertEqual(
            [put["key"] for put in store.puts],
            ["rtp_TestActor_501", "rtp_TestActor_502"],
        )
        self.assertEqual(
            [put["input_ids"].tolist() for put in store.puts],
            [[1, 2], [3, 4, 5]],
        )
        for put in store.puts:
            rows = len(put["input_ids"])
            torch.testing.assert_close(
                put["hidden_states"],
                torch.tensor([[10, 20]] * rows, dtype=torch.bfloat16),
                rtol=0,
                atol=0,
            )
            torch.testing.assert_close(
                put["last_hidden_states"],
                torch.full((rows, 1), 30, dtype=torch.bfloat16),
                rtol=0,
                atol=0,
            )
        self.assertCaptureQps(
            result["capture_metrics"],
            batch=1,
            publish_success=1,
            failure=0,
            bf16_publish=1,
        )

    def submitCapture(self, publisher, request_id=1, wait=True, *, device="cpu"):
        reservation = publisher.reserve(request_id, 1).reservation
        self.assertIsNotNone(reservation)
        self.assertTrue(
            publisher.submit(
                torch.tensor([[10, 20, 30]], dtype=torch.float16, device=device),
                torch.tensor([7]),
                torch.tensor([1], dtype=torch.int32),
                torch.tensor([request_id]),
                1,
                "source lifecycle",
            )
        )
        if wait:
            self.assertTrue(publisher.drain())
        return reservation

    def test_capture_start_poll_fatal_stops_current_put(self) -> None:
        for error in (
            RuntimeError("CUDA error: an illegal memory access was encountered"),
            RuntimeError("CUDA error while connecting to Mooncake"),
            RuntimeError("CuBlAs_StAtUs_ExEcUtIoN_FaIlEd"),
            RuntimeError("HIPBLAS_STATUS_ALLOC_FAILED"),
            RuntimeError("cudaErrorIllegalAddress"),
            RuntimeError("LAUNCH_FAILED"),
            RuntimeError("Collective Operation Timeout"),
        ):
            with self.subTest(error=type(error).__name__, message=str(error)):
                publisher = self.capturePublisher()
                try:
                    store = _FakeEagleMooncakeStore.instances[-1]
                    store.pending_async_errors.append(error)
                    before = list(store.call_sequence)
                    self.submitCapture(publisher)
                    self.assertTrue(publisher.fatal)
                    self.assertFalse(publisher.reserve(2, 1).reservation)
                    self.assertEqual(store.put_attempts, 0)
                    self.assertEqual(store.keys, set())
                    self.assertEqual(publisher.inflight_samples, 1)
                    self.assertEqual(store.call_sequence, before + ["take_async_errors"])
                    for _ in range(2):
                        self.assertTrue(publisher.poll())
                    publisher.close()
                    publisher.close()
                    publisher.poll()  # A stopped worker need not report idle.
                    self.assertEqual(store.call_sequence, before + ["take_async_errors"])
                    self.assertFalse(store.closed)
                finally:
                    publisher.close()

    def test_capture_classifier_failure_latches_fatal_before_put(self) -> None:
        original_import = __import__
        producer_thread = threading.get_ident()
        for failure in ("import", "attribute"):
            with self.subTest(failure=failure):
                publisher = self.capturePublisher()
                classifier_calls = []

                def fail_import(name, *args, **kwargs):
                    if name == "torch" and threading.get_ident() != producer_thread:
                        classifier_calls.append(name)
                        raise MemoryError("injected classifier import failure")
                    return original_import(name, *args, **kwargs)

                def fail_attribute(name):
                    if name == "DistError":
                        classifier_calls.append(name)
                        raise RuntimeError("injected classifier attribute failure")
                    raise AttributeError(name)

                def fail_poll():
                    store.call_sequence.append("take_async_errors")
                    raise RuntimeError("injected original store poll failure")

                distributed = types.ModuleType("classifier_fault_distributed")
                distributed.__getattr__ = fail_attribute
                fault = (
                    patch("builtins.__import__", fail_import)
                    if failure == "import"
                    else patch.object(torch, "distributed", distributed)
                )
                try:
                    store = _FakeEagleMooncakeStore.instances[-1]
                    before = list(store.call_sequence)
                    with patch.object(store, "take_async_errors", fail_poll), fault:
                        self.submitCapture(publisher)
                    self.assertEqual(classifier_calls, ["torch" if failure == "import" else "DistError"])
                    self.assertTrue(publisher.fatal)
                    self.assertIn("unable to classify publisher Python error", publisher.diagnostic)
                    self.assertIsNone(publisher.reserve(2, 1).reservation)
                    self.assertEqual(store.put_attempts, 0)
                    self.assertEqual(store.keys, set())
                    self.assertEqual(publisher.inflight_samples, 1)
                    self.assertTrue(publisher.poll())
                    publisher.close()
                    self.assertEqual(store.call_sequence, before + ["take_async_errors"])
                    self.assertFalse(store.closed)
                finally:
                    publisher.close()

    def test_capture_raised_fatal_message_is_read_once(self) -> None:
        class ChangingMessage(RuntimeError):
            calls = 0

            def __str__(self):
                self.calls += 1
                if self.calls == 1:
                    return "CUDA error on first read"
                raise ValueError("second read lost the original message")

        publisher = self.capturePublisher(poll_interval_ms=60000)
        self.addCleanup(publisher.close)
        store = _FakeEagleMooncakeStore.instances[-1]
        error = ChangingMessage()
        with patch.object(store, "take_async_errors", side_effect=error):
            self.submitCapture(publisher)
        self.assertTrue(publisher.fatal)
        self.assertEqual(error.calls, 1)
        self.assertIn("CUDA error on first read", publisher.diagnostic)
        self.assertIsNone(publisher.reserve(2, 1).reservation)
        self.assertEqual(store.put_attempts, 0)

    def test_capture_unformattable_completion_before_oom_stops_put(self) -> None:
        class BadFormat(RuntimeError):
            def __str__(self):
                raise ValueError("completion message unavailable")

        publisher = self.capturePublisher(poll_interval_ms=60000)
        self.addCleanup(publisher.close)
        store = _FakeEagleMooncakeStore.instances[-1]
        store.pending_async_errors.extend([BadFormat(), MemoryError("later allocation failure")])
        self.submitCapture(publisher)
        self.assertTrue(publisher.fatal)
        self.assertEqual(publisher.diagnostic, "unable to format asynchronous publisher error")
        self.assertIsNone(publisher.reserve(2, 1).reservation)
        self.assertEqual(store.put_attempts, 0)

    def test_capture_traceback_markers_do_not_override_benign_python_error(self) -> None:
        for method in ("take_async_errors", "put_batch"):
            with self.subTest(method=method), patch.dict(
                os.environ, {"MOONCAKE_ENABLE_GPU_DIRECT": str(int(method == "put_batch"))}
            ):
                publisher = self.capturePublisher(poll_interval_ms=60000)
                try:
                    store = _FakeEagleMooncakeStore.instances[-1]

                    def cudaErrorIllegalAddress(*args, **kwargs):
                        raise ValueError("benign capture transport failure")

                    cudaErrorIllegalAddress.__code__ = cudaErrorIllegalAddress.__code__.replace(
                        co_filename="CUDA error diagnostic-only filename.py"
                    )
                    with patch.object(store, method, cudaErrorIllegalAddress):
                        self.submitCapture(publisher)
                    self.assertFalse(publisher.fatal)
                    self.assertIn("benign capture transport failure", publisher.diagnostic)
                    self.assertIn("CUDA error diagnostic-only filename.py", publisher.diagnostic)
                    self.assertIn("cudaErrorIllegalAddress", publisher.diagnostic)
                    self.assertEqual(store.put_attempts, 0)
                    self.submitCapture(publisher, request_id=2)
                    self.assertFalse(publisher.fatal)
                    self.assertEqual(store.put_attempts, 1)
                    self.assertEqual(store.puts[-1]["key"], "rtp_TestActor_2")
                    self.consumeCapture(publisher)
                finally:
                    publisher.close()

    def test_capture_operational_cuda_text_does_not_latch_fatal(self) -> None:
        publisher = self.capturePublisher()
        try:
            store = _FakeEagleMooncakeStore.instances[-1]
            store.pending_async_errors.append(
                RuntimeError("CUDA transport temporarily unavailable")
            )
            self.submitCapture(publisher)
            self.assertFalse(publisher.fatal)
            self.assertEqual(store.put_attempts, 1)
            self.consumeCapture(publisher)
        finally:
            publisher.close()

    def test_capture_metadata_preserves_configured_layer_order(self) -> None:
        publisher = self.capturePublisher(layer_ids=[7, 2])
        try:
            self.submitCapture(publisher)
            store = _FakeEagleMooncakeStore.instances[-1]
            self.assertEqual(
                store.put_batch_calls[0]["capture_metadata"], {"capture_layer_ids": [7, 2]}
            )
            self.consumeCapture(publisher)
        finally:
            publisher.close()

    def test_capture_remote_source_holds_quota_after_flush_and_cancel(self) -> None:
        publisher = self.capturePublisher(samples=1, bytes=48, tokens=1, poll_interval_ms=60000)
        try:
            publisher.metrics()  # Consume the initial zero sample.
            reservation = self.submitCapture(publisher)
            store = _FakeEagleMooncakeStore.instances[-1]
            expected = {"rtp_TestActor_1" + suffix for suffix in ("_hs", "_ids", "_lhs", "_ready")}
            self.assertEqual(store.keys, expected)
            self.assertEqual(store.query_keys(sorted(expected)), dict.fromkeys(expected, True))
            self.assertTrue(store.flushed)
            reservation.cancel()
            self.assertEqual(publisher.inflight_samples, 1)
            self.assertEqual(publisher.inflight_bytes, 48)
            blocked = publisher.reserve(2, 1)
            self.assertFalse(blocked.ok)
            self.assertEqual(blocked.code, ExceptionType.HIDDEN_STATE_CAPTURE_BACKPRESSURE)
            self.assertIsNone(blocked.reservation)
            metrics, _ = publisher.metrics()
            self.assertCaptureQuota(metrics, 1, 48, 1)
            self.assertCaptureQps(
                metrics, batch=0, publish_success=1, failure=0, bf16_publish=1, admission_rejected=1, skip_capacity=1
            )
            store.keys.remove("rtp_TestActor_1_ready")
            self.assertTrue(publisher.poll())
            self.assertEqual(publisher.inflight_samples, 1)
            metrics, _ = publisher.metrics()
            self.assertTrue(_CAPTURE_QUOTA_METRICS.isdisjoint(metrics), "unchanged quota must be throttled")
            self.assertEqual(store.reclaim_calls, [])
            self.assertEqual(store.removed_keys, [])
            self.consumeCapture(publisher)
            metrics, _ = publisher.metrics()
            self.assertCaptureQuota(metrics, 0, 0, 0)
            self.assertEqual(store.reclaim_calls, [])
            self.assertEqual(store.removed_keys, [])
            admitted = publisher.reserve(1, 1)
            self.assertTrue(admitted.ok)
            self.assertEqual(admitted.code, 0)
            replacement = admitted.reservation
            self.assertIsNotNone(replacement)
            reservation.cancel()
            self.assertEqual(publisher.inflight_samples, 1)
            replacement.cancel()
        finally:
            publisher.close()

    def test_capture_idle_worker_polls_remote_cleanup(self) -> None:
        publisher = self.capturePublisher(poll_interval_ms=5)
        try:
            self.submitCapture(publisher)
            self.assertEqual(publisher.inflight_samples, 1)
            _FakeEagleMooncakeStore.instances[-1].consume()
            deadline = time.monotonic() + 5
            while publisher.inflight_samples and time.monotonic() < deadline:
                time.sleep(0.01)
            self.assertEqual(publisher.inflight_samples, 0)
            self.assertEqual(publisher.inflight_bytes, 0)
        finally:
            publisher.close()

    def test_capture_duplicate_source_precheck_does_not_put_or_delete(self) -> None:
        base = "rtp_TestActor_1"
        for suffix in _FakeEagleMooncakeStore.TENSOR_SUFFIXES:
            with self.subTest(suffix=suffix):
                publisher = self.capturePublisher()
                store = _FakeEagleMooncakeStore.instances[-1]
                old_keys = {base + suffix, base + "_abandoned"}
                store.keys.update(old_keys)
                try:
                    self.submitCapture(publisher)
                    self.assertTrue(publisher.poll())
                    self.assertEqual(store.put_attempts, 0)
                    self.assertIn("refusing to overwrite", publisher.diagnostic)
                    self.assertEqual(publisher.inflight_samples, 0)
                    self.assertFalse(publisher.fatal)
                finally:
                    publisher.close()
                self.assertEqual(store.keys, old_keys)
                self.assertEqual(store.reclaim_calls, [])

    @patch.dict(os.environ, {"MOONCAKE_ENABLE_GPU_DIRECT": "1"})
    def test_capture_store_duplicate_race_does_not_disable_capture(self) -> None:
        publisher = self.capturePublisher()
        try:

            def raced_put(_store, **_kwargs):
                raise FileExistsError("source appeared after preflight")

            with patch.object(_FakeEagleMooncakeStore, "put_batch", raced_put):
                self.submitCapture(publisher)
            self.assertIn("rejected duplicate request keys", publisher.diagnostic)
            self.assertFalse(publisher.fatal)
            self.assertEqual(publisher.inflight_samples, 0)
            next_reservation = publisher.reserve(2, 1).reservation
            self.assertIsNotNone(next_reservation)
            next_reservation.cancel()
        finally:
            publisher.close()

    @patch.dict(os.environ, {"MOONCAKE_ASYNC_PUT_POOL_SIZE": "2"})
    def test_capture_raced_foreign_scale_and_marker_survive_drain(self) -> None:
        base = "rtp_TestActor_1"
        for asynchronous in (False, True):
            with self.subTest(asynchronous=asynchronous), patch.dict(
                os.environ, {"MOONCAKE_ENABLE_GPU_DIRECT": "0" if asynchronous else "1"}
            ):
                publisher = self.capturePublisher(poll_interval_ms=60000)
                store = _FakeEagleMooncakeStore.instances[-1]
                put, flush = store.put_batch, store.flush
                old_keys = {base + "_hss", base + "_abandoned", "rtp_TestActor_2_abandoned"}

                def raced_put(**kwargs):
                    if not asynchronous:
                        store.keys.update(old_keys)
                        return put(**kwargs)  # Store-side preflight rejects this chunk.
                    # Host admission returns metadata; native source PUT hasn't run yet.
                    with patch.object(store, "keys", set()):
                        return put(**kwargs)

                def raced_flush():
                    first_flush = not store.flushed
                    flush()
                    if asynchronous and first_flush:
                        store.keys.update(old_keys)
                        raise FileExistsError("async worker conflict")

                try:
                    with patch.object(store, "put_batch", raced_put), patch.object(store, "flush", raced_flush):
                        reservations = [publisher.reserve(i, 1).reservation for i in (1, 2)]
                        self.assertTrue(all(reservations))
                        self.assertTrue(publisher.submit(
                            torch.zeros((2, 3), dtype=torch.float16), torch.tensor([7, 8]),
                            torch.ones(2, dtype=torch.int32), torch.tensor([1, 2]), 2, "raced chunk",
                        ))
                        self.assertTrue(publisher.drain())
                        self.assertEqual(store.put_attempts, 1)
                        self.assertTrue(store.flushed)
                        self.assertEqual(publisher.inflight_samples, 2)
                        self.assertTrue(publisher.poll())
                        self.assertEqual(store.keys, old_keys)
                        if not asynchronous:
                            # Losing the evidence on a later poll cannot grant deletion rights.
                            store.keys.remove(base + "_hss")
                            self.assertTrue(publisher.poll())
                            self.assertEqual(store.keys, old_keys - {base + "_hss"})
                            store.keys.add(base + "_hss")
                finally:
                    publisher.close()  # Retries the failed async drain before the final cleanup poll.
                self.assertEqual(store.keys, old_keys)
                self.assertEqual(store.reclaim_calls, [])
                self.assertEqual(store.removed_keys, [])

    def test_capture_preflight_query_exception_is_retryable(self) -> None:
        publisher = self.capturePublisher()
        try:
            for error in (RuntimeError("query unavailable"), MemoryError("preflight query allocation failed")):
                with patch.object(_FakeEagleMooncakeStore, "query_keys", side_effect=error) as query:
                    self.submitCapture(publisher)
                    query.assert_called()
                self.assertEqual(_FakeEagleMooncakeStore.instances[-1].put_attempts, 0)
                self.assertEqual(publisher.inflight_samples, 0)
                self.assertFalse(publisher.fatal)
            self.submitCapture(publisher, 2)
            self.assertEqual(_FakeEagleMooncakeStore.instances[-1].put_attempts, 1)
            self.consumeCapture(publisher)
        finally:
            publisher.close()

    def test_capture_preflight_rejects_non_strict_query_results(self) -> None:
        for invalid in (None, {}, {"wrong": False}, "integer", "missing"):
            with self.subTest(invalid=invalid):
                publisher = self.capturePublisher()
                try:

                    def query(_store, keys):
                        if invalid == "integer":
                            return dict.fromkeys(keys, 0)
                        if invalid == "missing":
                            return dict.fromkeys(keys[:-1], False)
                        return invalid

                    with patch.object(_FakeEagleMooncakeStore, "query_keys", query):
                        self.submitCapture(publisher)
                    self.assertEqual(_FakeEagleMooncakeStore.instances[-1].put_attempts, 0)
                    self.assertEqual(publisher.inflight_samples, 0)
                    self.assertIn("query_keys", publisher.diagnostic)
                    self.assertIsNone(publisher.reserve(2, 1).reservation)
                finally:
                    publisher.close()

    def test_capture_poll_local_key_copy_bad_alloc_retries_same_reservation(self) -> None:
        publisher = self.capturePublisher(samples=1, bytes=48, tokens=1, poll_interval_ms=60000)
        try:
            self.submitCapture(publisher)  # Worker drained; no pending work or cleanup request.
            store = _FakeEagleMooncakeStore.instances[-1]
            generation = publisher.reservation_generation(1)
            calls = list(store.call_sequence)
            manifest = list(store.query_calls[-1])
            self.assertIn("rtp_TestActor_1_abandoned", manifest)
            self.assertEqual(store.put_attempts, 1)
            publisher.metrics()  # The rejected admission below must report the post-fault quota.

            self.assertFalse(publisher.poll_with_key_copy_failure())
            self.assertEqual(store.call_sequence, calls)
            self.assertFalse(publisher.fatal)
            self.assertEqual(publisher.reservation_generation(1), generation)
            blocked = publisher.reserve(2, 1)
            self.assertEqual(blocked.code, ExceptionType.HIDDEN_STATE_CAPTURE_BACKPRESSURE)
            metrics, _ = publisher.metrics()
            self.assertCaptureQuota(metrics, 1, 48, 1)

            store.abandon_capture("rtp_TestActor_1")
            query_count = len(store.query_calls)
            self.assertTrue(publisher.poll())
            self.assertEqual(store.query_calls[query_count:], [manifest])
            self.assertEqual(store.reclaim_calls, ["rtp_TestActor_1"])
            self.assertEqual(store.keys, set())
            self.assertEqual(store.put_attempts, 1)
            self.assertEqual(publisher.inflight_samples, 0)
            self.assertEqual(publisher.inflight_bytes, 0)
            replacement = publisher.reserve(2, 1).reservation
            self.assertIsNotNone(replacement)
            replacement.cancel()
        finally:
            publisher.close()

    def test_capture_cleanup_unknown_keeps_quota_and_retries(self) -> None:
        for invalid in ("exception", "allocation", "integer", "missing", "wrong_keys"):
            with self.subTest(invalid=invalid):
                publisher = self.capturePublisher(samples=1, poll_interval_ms=50 if invalid == "allocation" else 60000)
                try:
                    self.submitCapture(publisher)
                    store = _FakeEagleMooncakeStore.instances[-1]

                    def query(_store, keys):
                        if invalid == "exception":
                            raise RuntimeError("source query unavailable")
                        if invalid == "allocation":
                            raise MemoryError("source query allocation failed")
                        if invalid == "integer":
                            return dict.fromkeys(keys, 0)
                        if invalid == "wrong_keys":
                            return {"foreign_" + key: False for key in keys}
                        return dict.fromkeys(keys[:-1], False)

                    with patch.object(_FakeEagleMooncakeStore, "query_keys", query):
                        store.abandon_capture("rtp_TestActor_1")
                        self.assertTrue(publisher.poll())
                        if invalid == "allocation":
                            self.assertIn("source query allocation failed", publisher.diagnostic)
                        self.assertEqual(publisher.inflight_samples, 1)
                        self.assertIsNone(publisher.reserve(2, 1).reservation)
                        self.assertFalse(publisher.fatal)
                        self.assertTrue(store.keys)
                        self.assertEqual(store.reclaim_calls, [])
                    self.assertTrue(publisher.poll())
                    if invalid == "allocation":
                        deadline = time.monotonic() + 5
                        while publisher.inflight_samples and time.monotonic() < deadline:
                            time.sleep(0.01)
                    self.assertEqual(store.reclaim_calls, ["rtp_TestActor_1"])
                    self.assertEqual(store.keys, set())
                    self.assertEqual(publisher.inflight_samples, 0)
                    reservation = publisher.reserve(2, 1).reservation
                    self.assertIsNotNone(reservation)
                    reservation.cancel()
                finally:
                    publisher.close()

    @patch.dict(os.environ, {"MOONCAKE_ASYNC_PUT_POOL_SIZE": "1"})
    def test_capture_partial_chunk_retains_attempted_prefix_not_suffix(self) -> None:
        publisher = self.capturePublisher(poll_interval_ms=60000)
        store = _FakeEagleMooncakeStore.instances[-1]
        flush = store.flush

        def partial_flush():
            flush()
            if store.put_attempts == 2:
                # Valid receipt and completed drain, but the second future never wrote its ready key.
                store.keys.remove("rtp_TestActor_2_ready")
                store.pending_async_errors.append(RuntimeError("accepted second chunk failed"))

        try:
            reservations = [publisher.reserve(i, 1).reservation for i in (1, 2, 3)]
            self.assertTrue(all(reservations))
            with patch.object(store, "flush", partial_flush):
                self.assertTrue(
                    publisher.submit(
                        torch.zeros((3, 3), dtype=torch.float16),
                        torch.tensor([7, 8, 9]),
                        torch.ones(3, dtype=torch.int32),
                        torch.tensor([1, 2, 3]),
                        3,
                        "partial chunks",
                    )
                )
                self.assertTrue(publisher.drain())
            self.assertFalse(publisher.fatal)
            self.assertIn("accepted second chunk failed", publisher.diagnostic)
            self.assertEqual(store.pending_async_errors, [])
            self.assertEqual(store.put_attempts, 2)
            self.assertEqual(publisher.inflight_samples, 2)
            self.assertTrue(all(not key.startswith("rtp_TestActor_3_") for key in store.keys))
            for reservation in reservations:
                reservation.cancel()
            self.assertEqual(publisher.inflight_samples, 2)
            suffix = publisher.reserve(3, 1).reservation
            self.assertIsNotNone(suffix)
            suffix.cancel()
            store.abandon_capture("rtp_TestActor_2")
            self.assertTrue(publisher.poll())
            self.assertEqual(publisher.inflight_samples, 1)
            self.assertEqual(store.reclaim_calls, ["rtp_TestActor_2"])
            self.assertEqual(
                store.keys,
                {"rtp_TestActor_1" + s for s in ("_hs", "_ids", "_lhs", "_ready")},
            )
            self.assertEqual(store.removed_keys[-1], "rtp_TestActor_2_abandoned")
            self.consumeCapture(publisher)
        finally:
            publisher.close()

    def test_capture_abandon_before_late_publish_waits_for_drain(self) -> None:
        publisher = self.capturePublisher(poll_interval_ms=60000)
        store = _FakeEagleMooncakeStore.instances[-1]
        entered = threading.Event()
        release = threading.Event()
        original_put = _FakeEagleMooncakeStore.put_batch

        def late_put(current_store, **kwargs):
            entered.set()
            if not release.wait(10):
                raise RuntimeError("test did not release late publication")
            return original_put(current_store, **kwargs)

        try:
            store.abandon_capture("rtp_TestActor_1")
            with patch.object(_FakeEagleMooncakeStore, "put_batch", late_put):
                try:
                    reservation = self.submitCapture(publisher, wait=False)
                    self.assertTrue(entered.wait(10))
                    reservation.cancel()
                    self.assertEqual(publisher.inflight_samples, 1)
                    self.assertEqual(store.keys, {"rtp_TestActor_1_abandoned"})
                    self.assertEqual(store.reclaim_calls, [])
                    self.assertEqual(store.removed_keys, [])
                finally:
                    release.set()
                self.assertTrue(publisher.drain())
            self.assertEqual(store.put_attempts, 1)
            self.assertTrue(store.flushed)
            self.assertEqual(store.reclaim_calls, ["rtp_TestActor_1"])
            self.assertEqual(store.removed_keys[-1], "rtp_TestActor_1_abandoned")
            self.assertEqual(store.keys, set())
            self.assertEqual(publisher.inflight_samples, 0)
        finally:
            release.set()
            publisher.close()

    def test_capture_marker_only_requires_strict_reclaim_true(self) -> None:
        for result in (False, 1, None):
            with self.subTest(result=result):
                publisher = self.capturePublisher(poll_interval_ms=60000)
                try:
                    self.submitCapture(publisher)
                    store = _FakeEagleMooncakeStore.instances[-1]
                    store.abandon_capture("rtp_TestActor_1")
                    store.consume()
                    self.assertEqual(store.keys, {"rtp_TestActor_1_abandoned"})
                    with patch.object(
                        _FakeEagleMooncakeStore, "reclaim_abandoned_capture", return_value=result
                    ) as reclaim:
                        self.assertTrue(publisher.poll())
                        reclaim.assert_called_once_with("rtp_TestActor_1")
                        self.assertEqual(publisher.inflight_samples, 1)
                        self.assertEqual(publisher.inflight_bytes, 48)
                        self.assertEqual(store.keys, {"rtp_TestActor_1_abandoned"})
                    self.assertFalse(publisher.fatal)
                    self.assertTrue(publisher.poll())
                    self.assertEqual(publisher.inflight_samples, 0)
                    self.assertEqual(store.removed_keys, ["rtp_TestActor_1_abandoned"])
                    self.assertEqual(store.keys, set())
                finally:
                    publisher.close()

    def test_capture_reclaim_operational_failure_does_not_starve_peers(self) -> None:
        publisher = self.capturePublisher(poll_interval_ms=60000)
        original_reclaim = _FakeEagleMooncakeStore.reclaim_abandoned_capture
        try:
            for request_id in (1, 2, 3):
                self.submitCapture(publisher, request_id)
            store = _FakeEagleMooncakeStore.instances[-1]
            # Follow the real unordered ledger order: the failure must come first.
            bases = [key.removesuffix("_abandoned") for key in store.query_calls[-1]
                     if key.endswith("_abandoned")]
            self.assertEqual(len(bases), 3)
            blocked, consumed, abandoned = bases
            store.abandon_capture(blocked)
            store.abandon_capture(abandoned)
            store.keys.difference_update(
                key for key in list(store.keys) if key.startswith(consumed + "_")
            )
            attempts = []

            def reclaim(current_store, key):
                attempts.append(key)
                if key == blocked:
                    raise RuntimeError("persistent per-request REMOVE failure")
                return original_reclaim(current_store, key)

            with patch.object(_FakeEagleMooncakeStore, "reclaim_abandoned_capture", reclaim):
                self.assertTrue(publisher.poll())
                self.assertTrue(publisher.poll())
                self.assertEqual(attempts, [blocked, abandoned, blocked])
                self.assertEqual(publisher.inflight_samples, 1)
                self.assertEqual(publisher.inflight_bytes, 48)
                self.assertEqual(
                    store.keys,
                    {blocked + suffix for suffix in ("_hs", "_ids", "_lhs", "_ready", "_abandoned")},
                )
                self.assertEqual(store.reclaim_calls, [abandoned])
                self.assertFalse(publisher.fatal)
            self.assertTrue(publisher.poll())
            self.assertEqual(store.reclaim_calls, [abandoned, blocked])
            self.assertEqual(publisher.inflight_samples, 0)
            self.assertEqual(store.keys, set())
        finally:
            publisher.close()

    def test_capture_reclaim_retries_allocation_but_stops_on_fatal(self) -> None:
        for error in (
            MemoryError("transient reclaim allocation"),
            RuntimeError("CUSPARSE_STATUS_ALLOC_FAILED"),
        ):
            with self.subTest(error=str(error)):
                allocation = isinstance(error, MemoryError)
                publisher = self.capturePublisher(poll_interval_ms=50 if allocation else 60000)
                try:
                    for request_id in (1, 2):
                        self.submitCapture(publisher, request_id)
                    store = _FakeEagleMooncakeStore.instances[-1]
                    bases = [key.removesuffix("_abandoned") for key in store.query_calls[-1]
                             if key.endswith("_abandoned")]
                    self.assertEqual(len(bases), 2)
                    attempts = []

                    def reclaim(_store, key):
                        _store.call_sequence.append(f"reclaim_abandoned_capture:{key}")
                        attempts.append(key)
                        raise error

                    with patch.object(_FakeEagleMooncakeStore, "reclaim_abandoned_capture", reclaim):
                        for key in bases:
                            store.abandon_capture(key)
                        original_keys = set(store.keys)
                        self.assertTrue(publisher.poll())
                        self.assertEqual(publisher.inflight_samples, 2)
                        self.assertEqual(store.keys, original_keys)
                        if allocation:
                            self.assertFalse(publisher.fatal)
                            self.assertEqual(set(attempts), set(bases))
                    if allocation:
                        deadline = time.monotonic() + 5
                        while publisher.inflight_samples and time.monotonic() < deadline:
                            time.sleep(0.01)
                        self.assertCountEqual(store.reclaim_calls, bases)
                        self.assertEqual(publisher.inflight_samples, 0)
                        self.submitCapture(publisher, 3)
                        self.consumeCapture(publisher)
                        continue
                    with patch.object(_FakeEagleMooncakeStore, "reclaim_abandoned_capture", reclaim):
                        self.assertTrue(publisher.fatal)
                        self.assertIsNone(publisher.reserve(3, 1).reservation)
                        calls_at_failure = list(store.call_sequence)
                        self.assertEqual(calls_at_failure[-2:], [
                            "query_keys", f"reclaim_abandoned_capture:{bases[0]}"
                        ])
                        for _ in range(2):
                            self.assertTrue(publisher.poll())
                        publisher.close()
                        publisher.close()
                        publisher.poll()
                        self.assertEqual(attempts, [bases[0]])
                        self.assertEqual(store.call_sequence, calls_at_failure)
                    self.assertEqual(store.keys, original_keys)
                    self.assertEqual(store.removed_keys, [])
                    self.assertFalse(store.closed)
                finally:
                    publisher.close()

    def test_capture_shutdown_retains_unconsumed_source_store(self) -> None:
        publisher = self.capturePublisher(poll_interval_ms=60000)
        self.submitCapture(publisher)
        store = _FakeEagleMooncakeStore.instances[-1]
        keys = set(store.keys)
        publisher.metrics()
        for _ in range(2):
            publisher.close()
            metrics, _ = publisher.metrics()
            self.assertCaptureQuota(metrics, 1, 48, 1)
            self.assertCaptureStatus(metrics, enabled=0, broken=0)
            # Active ledger is detached, but retained resources still count in metrics.
            self.assertEqual(publisher.inflight_samples, 0)
            self.assertEqual(publisher.inflight_bytes, 0)
        self.assertFalse(store.closed)
        self.assertEqual(store.keys, keys)
        self.assertIsNone(publisher.reserve(2, 1).reservation)
        metrics, _ = publisher.metrics()
        self.assertCaptureQuota(metrics, 1, 48, 1)
        self.assertCaptureQps(metrics, batch=0, publish_success=0, failure=0,
                              admission_rejected=1, skip_unavailable=1)

    def test_capture_initialization_runs_on_worker_without_blocking_constructor(
        self,
    ) -> None:
        for stage in ("setup", "warmup_rdma"):
            with self.subTest(stage=stage):
                entered = threading.Event()
                release = threading.Event()
                original = getattr(_FakeEagleMooncakeStore, stage)

                def blocked(store, *args):
                    entered.set()
                    if not release.wait(10):
                        raise RuntimeError("test did not release initialization")
                    return original(store, *args)

                with patch.object(_FakeEagleMooncakeStore, stage, blocked):
                    # Bypass the ordinary fixture's test-only initialization barrier.
                    publisher = _extension.CapturePublisher()
                    try:
                        self.assertTrue(entered.wait(10))
                        self.assertFalse(publisher.drain(timeout_ms=0))
                        admission = publisher.reserve(1, 1)
                        self.assertFalse(admission.ok)
                        self.assertEqual(admission.code, ExceptionType.HIDDEN_STATE_CAPTURE_BACKPRESSURE)
                        self.assertIn("initializing", admission.message)
                        self.assertIsNone(admission.reservation)
                        self.assertEqual(publisher.inflight_samples, 0)
                        self.assertEqual(publisher.inflight_bytes, 0)
                        metrics, _ = publisher.metrics()
                        self.assertCaptureQps(
                            metrics, batch=0, publish_success=0, failure=0, admission_rejected=1, skip_unavailable=1
                        )
                        self.assertCaptureQuota(metrics, 0, 0, 0)
                        release.set()
                        self.assertTrue(publisher.drain())
                        admission = publisher.reserve(1, 1)
                        self.assertTrue(admission.ok)
                        self.assertIsNotNone(admission.reservation)
                        admission.reservation.cancel()
                        self.assertEqual(publisher.inflight_samples, 0)
                        self.assertEqual(publisher.inflight_bytes, 0)
                        self.assertFalse(publisher.fatal)
                        self.assertEqual(publisher.diagnostic, "")
                    finally:
                        release.set()
                        publisher.close()
                self.assertTrue(_FakeEagleMooncakeStore.instances[-1].closed)
                self.assertEqual(publisher.reserve(1, 1).code, ExceptionType.CANCELLED)

    def test_capture_shutdown_pending_put_requires_confirmed_drain(self) -> None:
        for fatal_completion in (False, True):
            with self.subTest(fatal_completion=fatal_completion):
                publisher = self.capturePublisher(poll_interval_ms=60000)
                store = _FakeEagleMooncakeStore.instances[-1]
                write_allowed = threading.Event()
                write_done = threading.Event()
                shutdown_started = threading.Event()
                flush_entered = threading.Event()
                flush_release = threading.Event()
                storage_refs = []
                writers = []
                failures = []
                closer = None
                original_poll = _FakeEagleMooncakeStore.take_async_errors

                def pending_put(current_store, **kwargs):
                    current_store.call_sequence.append("put_batch")
                    keys = list(kwargs["request_ids"])
                    metadata = []
                    for index in range(len(keys)):
                        item = {"shapes": {}, "dtypes": {}}
                        for name in ("hidden_states", "input_ids", "last_hidden_states"):
                            tensor = kwargs[name][index]
                            item["shapes"][name] = tuple(tensor.shape)
                            item["dtypes"][name] = tensor.dtype
                            storage_refs.append(tensor.untyped_storage()._weak_ref())
                        metadata.append(item)

                    # Simulated native writer holds only keys, never Python tensors.
                    # Storage weak handles observe C++ ownership, not wrapper lifetime.
                    def finish_write():
                        if not write_allowed.wait(10):
                            failures.append("pending writer was not released")
                            return
                        current_store.keys.update(
                            key + suffix for key in keys
                            for suffix in ("_hs", "_ids", "_lhs", "_ready")
                        )
                        write_done.set()

                    writer = threading.Thread(target=finish_write, daemon=True)
                    writers.append(writer)
                    writer.start()
                    return metadata

                def pending_flush(current_store):
                    current_store.call_sequence.append("flush")
                    if not shutdown_started.is_set():
                        if not write_done.is_set():
                            raise RuntimeError("native PUT still pending; drain unconfirmed")
                    else:
                        flush_entered.set()
                        if not flush_release.wait(10):
                            raise RuntimeError("shutdown flush was not released")
                        if not fatal_completion and not write_done.is_set():
                            raise RuntimeError("shutdown drain lacks native completion")
                    current_store.flushed = True

                def completion_errors(current_store):
                    errors = original_poll(current_store)
                    if shutdown_started.is_set() and fatal_completion:
                        errors.append(MemoryError("fatal shutdown completion; PUT state unknown"))
                    return errors

                def close():
                    try:
                        publisher.close()
                    except Exception as error:
                        failures.append(str(error))

                with patch.object(_FakeEagleMooncakeStore, "put_batch", pending_put), \
                     patch.object(_FakeEagleMooncakeStore, "flush", pending_flush), \
                     patch.object(_FakeEagleMooncakeStore, "take_async_errors", completion_errors):
                    try:
                        reservation = publisher.reserve(1, 1).reservation
                        self.assertIsNotNone(reservation)
                        packed = torch.tensor([[10, 20, 30]], dtype=torch.float16)
                        storage_refs.append(packed.untyped_storage()._weak_ref())
                        self.assertTrue(publisher.submit(
                            packed, torch.tensor([7]), torch.tensor([1], dtype=torch.int32),
                            torch.tensor([1]), 1, "pending native PUT at shutdown"
                        ))
                        del packed
                        self.assertTrue(publisher.drain())
                        self.assertEqual(len(storage_refs), 4)
                        self.assertEqual(store.call_sequence[-2:], ["put_batch", "flush"])
                        store.abandon_capture("rtp_TestActor_1")
                        self.assertTrue(publisher.poll())
                        self.assertEqual(store.call_sequence[-2:], ["put_batch", "flush"])
                        self.assertEqual(store.reclaim_calls, [])
                        self.assertFalse(write_done.is_set())
                        self.assertFalse(publisher.fatal)
                        self.assertEqual(publisher.inflight_samples, 1)
                        self.assertTrue(all(not torch.UntypedStorage._expired(ref)
                                            for ref in storage_refs))

                        shutdown_started.set()
                        closer = threading.Thread(target=close, daemon=True)
                        closer.start()
                        self.assertTrue(flush_entered.wait(10))
                        self.assertTrue(closer.is_alive())
                        self.assertEqual(publisher.inflight_samples, 1)
                        self.assertEqual(store.keys, {"rtp_TestActor_1_abandoned"})
                        self.assertEqual(store.removed_keys, [])
                        self.assertEqual(store.reclaim_calls, [])
                        self.assertTrue(all(not torch.UntypedStorage._expired(ref)
                                            for ref in storage_refs))
                        calls_in_flush = list(store.call_sequence)
                        if not fatal_completion:
                            write_allowed.set()
                            self.assertTrue(write_done.wait(10))
                        flush_release.set()
                        closer.join(10)
                        self.assertFalse(closer.is_alive())
                        self.assertEqual(failures, [])
                        if fatal_completion:
                            self.assertTrue(publisher.fatal)
                            self.assertFalse(write_done.is_set())
                            self.assertEqual(store.call_sequence,
                                             calls_in_flush + ["take_async_errors"])
                            self.assertTrue(all(not torch.UntypedStorage._expired(ref)
                                                for ref in storage_refs))
                            self.assertEqual(store.keys, {"rtp_TestActor_1_abandoned"})
                            self.assertEqual(store.removed_keys, [])
                            self.assertEqual(store.reclaim_calls, [])
                            self.assertFalse(store.closed)
                            for _ in range(2):
                                publisher.poll()
                                publisher.close()
                            self.assertEqual(store.call_sequence,
                                             calls_in_flush + ["take_async_errors"])
                            self.assertTrue(all(not torch.UntypedStorage._expired(ref)
                                                for ref in storage_refs))
                        else:
                            self.assertFalse(publisher.fatal)
                            self.assertEqual(store.call_sequence, calls_in_flush + [
                                "take_async_errors", "query_keys",
                                "reclaim_abandoned_capture:rtp_TestActor_1", "close"
                            ])
                            self.assertEqual(store.reclaim_calls, ["rtp_TestActor_1"])
                            self.assertEqual(store.keys, set())
                            self.assertTrue(store.closed)
                            self.assertEqual(publisher.inflight_samples, 0)
                            self.assertEqual(publisher.inflight_bytes, 0)
                            self.assertTrue(all(torch.UntypedStorage._expired(ref)
                                                for ref in storage_refs))
                    finally:
                        write_allowed.set()
                        flush_release.set()
                        if closer is not None:
                            closer.join(10)
                        for writer in writers:
                            writer.join(10)
                        publisher.close()
                        for ref in storage_refs:
                            torch.UntypedStorage._free_weak_ref(ref)

    def test_capture_initialization_failure_rejects_without_reserving_payload(self) -> None:
        for stage in ("setup", "warmup_rdma"):
            for error, fatal in (
                (RuntimeError("injected initialization failure"), False),
                (MemoryError("injected initialization allocation failure"), True),
            ):
                with self.subTest(stage=stage, fatal=fatal):
                    entered = threading.Event()
                    release = threading.Event()

                    def blocked_failure(_store, *args):
                        entered.set()
                        if not release.wait(10):
                            raise RuntimeError("test did not release initialization")
                        raise error

                    with patch.object(_FakeEagleMooncakeStore, stage, blocked_failure):
                        publisher = _extension.CapturePublisher()
                        try:
                            self.assertTrue(entered.wait(10))
                            pending = publisher.reserve(1, 1)
                            self.assertEqual(pending.code, ExceptionType.HIDDEN_STATE_CAPTURE_BACKPRESSURE)
                            self.assertIsNone(pending.reservation)
                            self.assertFalse(
                                publisher.submit(
                                    torch.zeros((1, 3), dtype=torch.float16),
                                    torch.tensor([7]),
                                    torch.tensor([1], dtype=torch.int32),
                                    torch.tensor([1]),
                                    1,
                                    "unreserved initialization",
                                )
                            )
                            self.assertEqual(publisher.inflight_samples, 0)
                            self.assertEqual(publisher.inflight_bytes, 0)
                            release.set()
                            self.assertTrue(publisher.drain())
                            for request_id in (1, 2):
                                disabled = publisher.reserve(request_id, 1)
                                self.assertFalse(disabled.ok)
                                self.assertEqual(disabled.code, ExceptionType.EXECUTION_EXCEPTION)
                                self.assertIsNone(disabled.reservation)
                            self.assertEqual(publisher.inflight_samples, 0)
                            self.assertEqual(publisher.inflight_bytes, 0)
                            self.assertIn(str(error), publisher.diagnostic)
                            self.assertEqual(publisher.fatal, fatal)
                            self.assertEqual(_FakeEagleMooncakeStore.instances[-1].put_attempts, 0)
                            if not fatal:
                                self.assertTrue(_FakeEagleMooncakeStore.instances[-1].closed)
                        finally:
                            release.set()
                            publisher.close()
                        self.assertEqual(publisher.fatal, fatal)

    def test_capture_quota_counts_source_dtype_ids_metadata_and_store(self) -> None:
        # L=2, H=1, T=2; IDs use 8 bytes per token on each side.
        # Each request also retains int32 length + int64 ID tensor + int64 ledger.
        for dtype, fp8, expected_bytes in (
            ("float16", False, 76),
            ("float32", False, 88),
            ("float64", False, 112),
            ("float16", True, 88),
            ("float32", True, 100),
        ):
            with self.subTest(dtype=dtype, fp8=fp8):
                below = self.capturePublisher(bytes=expected_bytes - 1, model_dtype=dtype, fp8=fp8)
                exact = self.capturePublisher(bytes=expected_bytes, model_dtype=dtype, fp8=fp8)
                try:
                    self.assertFalse(below.reserve(1, 2).reservation)
                    self.assertEqual(below.inflight_bytes, 0)
                    reservation = exact.reserve(1, 2).reservation
                    self.assertIsNotNone(reservation)
                    self.assertEqual(exact.inflight_bytes, expected_bytes)
                    self.assertFalse(exact.reserve(2, 1).reservation)
                    reservation.cancel()
                    self.assertEqual(exact.inflight_bytes, 0)
                finally:
                    below.close()
                    exact.close()

    def test_capture_blocked_put_keeps_full_source_and_store_byte_quota(self) -> None:
        entered = threading.Event()
        release = threading.Event()
        original_put = _FakeEagleMooncakeStore.put_batch

        def blocked_put(store, **kwargs):
            entered.set()
            if not release.wait(10):
                raise RuntimeError("test did not release quota PUT")
            return original_put(store, **kwargs)

        publisher = self.capturePublisher(bytes=76)
        self.addCleanup(publisher.close)
        packed = torch.tensor(
            [[10, 20, 30], [11, 21, 31]], dtype=torch.float16, device="cuda"
        )
        with patch.object(_FakeEagleMooncakeStore, "put_batch", blocked_put):
            try:
                reservation = publisher.reserve(1, 2).reservation
                self.assertIsNotNone(reservation)
                self.assertTrue(
                    publisher.submit(
                        packed,
                        torch.tensor([7, 8]),
                        torch.tensor([2], dtype=torch.int32),
                        torch.tensor([1]),
                        2,
                        "blocked quota",
                    )
                )
                del packed
                self.assertTrue(entered.wait(10))
                self.assertFalse(publisher.drain(timeout_ms=0))
                self.assertEqual(publisher.inflight_bytes, 76)
                self.assertFalse(publisher.reserve(2, 1).reservation)
            finally:
                release.set()
            self.assertTrue(publisher.drain())
        self.assertEqual(publisher.inflight_bytes, 76)
        self.consumeCapture(publisher)
        self.assertEqual(
            _FakeEagleMooncakeStore.instances[-1]
            .puts[0]["last_hidden_states"]
            .tolist(),
            [[30], [31]],
        )

    def test_capture_quota_rejects_oversized_retained_source_storage(self) -> None:
        publisher = self.capturePublisher(bytes=76)
        self.addCleanup(publisher.close)
        oversized = publisher.reserve(1, 2).reservation
        self.assertIsNotNone(oversized)
        parent = torch.arange(300, dtype=torch.float16).reshape(100, 3)
        view = parent[:2]
        ids = torch.tensor([7, 8])
        lengths = torch.tensor([2], dtype=torch.int32)
        self.assertFalse(
            publisher.submit(
                view, ids, lengths, torch.tensor([1]), 2, "oversized source"
            )
        )
        self.assertEqual(publisher.inflight_bytes, 76)
        oversized.cancel()
        self.assertEqual(publisher.inflight_bytes, 0)
        owned = publisher.reserve(2, 2).reservation
        self.assertIsNotNone(owned)
        self.assertTrue(
            publisher.submit(
                view.clone(), ids, lengths, torch.tensor([2]), 2, "owned source"
            )
        )
        self.assertTrue(publisher.drain())
        self.assertEqual(publisher.inflight_bytes, 76)
        self.consumeCapture(publisher)
        self.assertEqual(
            _FakeEagleMooncakeStore.instances[-1].puts[0]["hidden_states"].tolist(),
            [[0, 1], [3, 4]],
        )

    def test_capture_synced_bit_keeps_geometry_after_owner_reservation_cancellation(
        self,
    ) -> None:
        for scenario, expected_reductions in (
            ("capture_bf16", 0),
            ("capture_bf16_tp", 1),
        ):
            with self.subTest(scenario=scenario):
                order = []

                class ObservedForward(CacheStoreForwardModel):
                    def forward(self, inputs, fmha_impl=None):
                        order.append("forward")
                        return super().forward(inputs, fmha_impl)

                def layout_reduce(tensor, op, mode, dest):
                    self.assertEqual(order, ["forward"])
                    self.assertEqual(int(tensor.item()), 2)
                    order.append("layout")
                    return tensor

                submit_frames = []
                _register_no_collectives(layout_reduce, submit_frames=submit_frames)
                try:
                    model = ObservedForward()
                    result = run_scenario(
                        model, scenario, cancel_capture_after_admission=True
                    )
                finally:
                    clear_comm_ops()
                self.assertEqual(result["capture_admissions"], [True])
                self.assertEqual(model.capture_flags, [True])
                self.assertEqual(order, ["forward"] + ["layout"] * expected_reductions)
                self.assertEqual(submit_frames, [2] * expected_reductions)
                self.assertCanonicalZero(result)
                self.assertEqual(result["capture_failure_count"], 0)
                self.assertEqual(_FakeEagleMooncakeStore.instances[-1].puts, [])

    def test_capture_worker_disables_callbacks_but_synchronous_close_preserves_them(self) -> None:
        probe = _extension.CaptureAtenProbe()
        probe.resume()
        publisher = None
        calls = []
        snapshots = []
        errors = []

        def transfer():
            torch.arange(5, dtype=torch.int32).to(torch.int64)

        def observed(name):
            original = getattr(_FakeEagleMooncakeStore, name)

            def call(store, *args):
                calls.append(name)
                transfer()
                return original(store, *args)

            return call

        def close():
            try:
                snapshots.append(probe.wait(timeout_ms=0))
                # Post-join store calls are synchronous on this caller; they must
                # remain recordable and provide the probe's positive control.
                publisher.close()
                snapshots.append(probe.wait(timeout_ms=0))
            except BaseException as error:
                errors.append(error)

        try:
            with patch.object(_FakeEagleMooncakeStore, "setup", observed("setup")), patch.object(
                _FakeEagleMooncakeStore, "query_keys", observed("query_keys")
            ), patch.object(_FakeEagleMooncakeStore, "flush", observed("flush")), patch.object(
                _FakeEagleMooncakeStore, "close", observed("close")
            ):
                publisher = self.capturePublisher()
                self.submitCapture(publisher)
                self.consumeCapture(publisher)
                closer = threading.Thread(target=close)
                closer.start()
                closer.join(10)
                self.assertFalse(closer.is_alive())
                self.assertEqual(errors, [])
                self.assertEqual(len(snapshots), 2)
                self.assertFalse(snapshots[0]["entered"])
                self.assertTrue(snapshots[1]["entered"])
                self.assertFalse(snapshots[1]["timed_out"])
                for name in ("setup", "query_keys", "flush", "close"):
                    self.assertIn(name, calls)
        finally:
            if publisher is not None:
                publisher.close()
            probe.close()

    @patch.dict(os.environ, {"MOONCAKE_ENABLE_GPU_DIRECT": "1"})
    def test_capture_native_id_transfer_disables_callbacks_and_reacquires_gil_on_exception(
        self,
    ) -> None:
        probe = _extension.CaptureAtenProbe()
        self.addCleanup(probe.close)
        # A regression must report callback entry rather than stall the worker.
        probe.resume()
        original_put = _FakeEagleMooncakeStore.put_batch
        put_calls = []

        def fail_once(store, **kwargs):
            self.assertTrue(_extension.capture_gil_held())
            put_calls.append(kwargs["batch_id"])
            if len(put_calls) == 1:
                raise RuntimeError(
                    "injected store failure after callback-disabled ATen transfer"
                )
            return original_put(store, **kwargs)

        with patch.object(_FakeEagleMooncakeStore, "put_batch", fail_once):
            result = run_scenario(
                CacheStoreForwardModel(), "capture_bf16", 2
            )
        snapshot = probe.wait()
        self.assertFalse(
            snapshot["entered"], "worker CPU-ID transfer invoked a RecordFunction callback"
        )
        self.assertFalse(snapshot["timed_out"])
        self.assertCanonicalZero(result)
        self.assertEqual(len(put_calls), 2)
        self.assertEqual(result["capture_failure_count"], 1)
        self.assertIn(
            "injected store failure after callback-disabled ATen transfer",
            result["capture_diagnostics"][0],
        )
        self.assertEqual(len(_FakeEagleMooncakeStore.instances[-1].puts), 2)
        self.assertCaptureQps(
            result["capture_metrics"],
            batch=2,
            publish_success=1,
            failure=1,
            store_failure=1,
            operational_failure=1,
            bf16_publish=1,
        )
        self.assertCaptureStatus(result["capture_metrics"], enabled=0, broken=0)

    def test_capture_reservation_limits_and_cancellation(self) -> None:
        # Only one resource is tight per row; no earlier limit masks tokens.
        for resource, limits in (
            ("samples", dict(samples=1, bytes=1_000_000, tokens=100)),
            ("bytes", dict(samples=10, bytes=76, tokens=100)),
            ("tokens", dict(samples=10, bytes=1_000_000, tokens=2)),
        ):
            with self.subTest(resource=resource):
                publisher = self.capturePublisher(poll_interval_ms=60000, **limits)
                try:
                    self.assertFalse(
                        publisher.submit(
                            torch.zeros((2, 3), dtype=torch.float16),
                            torch.tensor([7, 8]),
                            torch.tensor([2], dtype=torch.int32),
                            torch.tensor([1]),
                            2,
                            "unreserved",
                        )
                    )
                    self.assertEqual(publisher.inflight_bytes, 0)
                    self.assertEqual(_FakeEagleMooncakeStore.instances[-1].puts, [])
                    admitted = publisher.reserve(1, 2)
                    self.assertTrue(admitted.ok)
                    self.assertEqual(admitted.code, 0)
                    first = admitted.reservation
                    self.assertIsNotNone(first)
                    self.assertTrue(first.active)
                    publisher.metrics()  # Isolate admission counters from the unreserved submit.
                    for request_id, tokens in ((-1, 2), (1, 0), (1, 2**63 - 1), (1, 2)):
                        invalid = publisher.reserve(request_id, tokens)
                        self.assertFalse(invalid.ok)
                        self.assertEqual(invalid.code, ExceptionType.INVALID_PARAMS)
                        self.assertIsNone(invalid.reservation)
                    blocked = publisher.reserve(2, 1)
                    self.assertFalse(blocked.ok)
                    self.assertEqual(
                        blocked.code, ExceptionType.HIDDEN_STATE_CAPTURE_BACKPRESSURE
                    )
                    self.assertIsNone(blocked.reservation)
                    if resource != "samples":
                        # Permanent oversize stays invalid with occupied or idle quota.
                        self.assertEqual(publisher.reserve(3, 3).code, ExceptionType.INVALID_PARAMS)
                    self.assertEqual(publisher.inflight_samples, 1)
                    self.assertEqual(publisher.inflight_bytes, 76)
                    metrics, _ = publisher.metrics()
                    capacity = 1 + int(resource != "samples")
                    self.assertCaptureQps(
                        metrics,
                        batch=0,
                        publish_success=0,
                        failure=0,
                        admission_rejected=4 + capacity,
                        skip_config=4,
                        skip_capacity=capacity,
                        duplicate_request_id=1,
                    )
                    first.cancel()
                    first.cancel()
                    self.assertFalse(first.active)
                    self.assertEqual(publisher.inflight_samples, 0)
                    self.assertEqual(publisher.inflight_bytes, 0)
                    if resource != "samples":
                        self.assertEqual(publisher.reserve(3, 3).code, ExceptionType.INVALID_PARAMS)
                    second = publisher.reserve(2, 2).reservation
                    self.assertIsNotNone(second)
                    self.assertTrue(
                        publisher.submit(
                            torch.tensor(
                                [[10, 20, 30], [11, 21, 31]], dtype=torch.float16
                            ),
                            torch.tensor([7, 8]),
                            torch.tensor([2], dtype=torch.int32),
                            torch.tensor([2]),
                            2,
                            "quota capture",
                        )
                    )
                    self.assertTrue(publisher.drain())
                    self.assertEqual(publisher.inflight_samples, 1)
                    self.assertEqual(publisher.inflight_bytes, 76)
                    self.consumeCapture(publisher)
                    third = publisher.reserve(3, 2).reservation
                    self.assertIsNotNone(third)
                    self.assertTrue(publisher.poll())
                    self.assertEqual(publisher.inflight_samples, 1)
                    publisher.metrics()  # Close, not stream cancellation, must clear this reservation.
                    self.assertEqual(
                        _FakeEagleMooncakeStore.instances[-1]
                        .puts[0]["hidden_states"]
                        .tolist(),
                        [[10, 20], [11, 21]],
                    )
                    self.assertFalse(publisher.fatal)
                    self.assertEqual(publisher.diagnostic, "")
                finally:
                    publisher.close()
                metrics, _ = publisher.metrics()
                self.assertCaptureQuota(metrics, 0, 0, 0)
                self.assertCaptureStatus(metrics, enabled=0, broken=0)
                third.cancel()
                publisher.close()
                metrics, _ = publisher.metrics()
                self.assertCaptureQuota(metrics, 0, 0, 0)
                self.assertCaptureStatus(metrics, enabled=0, broken=0)
                stopped = publisher.reserve(4, 1)
                self.assertFalse(stopped.ok)
                self.assertEqual(stopped.code, ExceptionType.CANCELLED)
                self.assertIsNone(stopped.reservation)
                metrics, _ = publisher.metrics()
                self.assertCaptureQps(
                    metrics, batch=0, publish_success=0, failure=0, admission_rejected=1, skip_unavailable=1
                )
                self.assertCaptureQuota(metrics, 0, 0, 0)

        disabled = self.capturePublisher(samples=0)
        try:
            disabled.metrics()
            unavailable = disabled.reserve(1, 1)
            self.assertFalse(unavailable.ok)
            self.assertEqual(unavailable.code, ExceptionType.EXECUTION_EXCEPTION)
            self.assertIsNone(unavailable.reservation)
            metrics, _ = disabled.metrics()
            self.assertCaptureQps(
                metrics, batch=0, publish_success=0, failure=0, admission_rejected=1, skip_unavailable=1
            )
            self.assertCaptureQuota(metrics, 0, 0, 0)
        finally:
            disabled.close()

    def test_capture_reservation_generation_and_publisher_lifetime(self) -> None:
        publisher = self.capturePublisher(samples=1, bytes=48, tokens=1)
        try:
            first = publisher.reserve(7, 1).reservation
            self.assertIsNotNone(first)
            self.assertTrue(
                publisher.submit(
                    torch.tensor([[10, 20, 30]], dtype=torch.float16),
                    torch.tensor([3]),
                    torch.tensor([1], dtype=torch.int32),
                    torch.tensor([7]),
                    1,
                    "reservation ABA",
                )
            )
            self.assertTrue(publisher.drain())
            store = _FakeEagleMooncakeStore.instances[-1]
            self.assertEqual([put["key"] for put in store.puts], ["rtp_TestActor_7"])
            self.assertEqual(store.puts[0]["last_hidden_states"].tolist(), [[30]])
            self.assertEqual(publisher.diagnostic, "")
            self.assertFalse(publisher.fatal)
            self.assertEqual(publisher.inflight_samples, 1)
            self.assertTrue(first.active)
            self.consumeCapture(publisher)

            # Reuse the ID only after the consumer removes all source keys.
            second = publisher.reserve(7, 1).reservation
            self.assertIsNotNone(second)
            first.cancel()
            first.cancel()
            self.assertFalse(first.active)
            self.assertTrue(second.active)
            self.assertEqual(publisher.inflight_samples, 1)
            self.assertEqual(publisher.inflight_bytes, 48)
            self.assertIsNone(publisher.reserve(8, 1).reservation)
            second.cancel()
            self.assertEqual(publisher.inflight_samples, 0)
            self.assertEqual(publisher.inflight_bytes, 0)

            automatic = publisher.reserve(8, 1).reservation
            self.assertIsNotNone(automatic)
            del automatic
            self.assertEqual(publisher.inflight_samples, 0)
            orphan = publisher.reserve(8, 1).reservation
            self.assertIsNotNone(orphan)
            self.assertFalse(store.closed)
            # Tokens must not own the publisher, model, or an Executor.
            publisher = None
            gc.collect()
            self.assertTrue(store.closed)
            self.assertTrue(orphan.active)
            orphan.cancel()
            orphan.cancel()
            self.assertFalse(orphan.active)
        finally:
            if publisher is not None:
                publisher.close()

    def test_capture_pending_queue_is_bounded_while_worker_blocks(self) -> None:
        entered = threading.Event()
        release = threading.Event()
        original_put = _FakeEagleMooncakeStore.put_batch

        def blocked_put(store, **kwargs):
            entered.set()
            if not release.wait(10):
                raise RuntimeError("test did not release PUT")
            return original_put(store, **kwargs)

        publisher = self.capturePublisher(samples=4, bytes=192, tokens=4, batches=1)
        self.addCleanup(publisher.close)
        publisher.metrics()
        packed = torch.tensor([[10, 20, 30]], dtype=torch.float16)

        reservations = {}

        def submit(request_id):
            reservations[request_id] = publisher.reserve(request_id, 1).reservation
            self.assertIsNotNone(reservations[request_id])
            return publisher.submit(
                packed,
                torch.tensor([7]),
                torch.tensor([1], dtype=torch.int32),
                torch.tensor([request_id]),
                1,
                "bounded capture",
            )

        with patch.object(_FakeEagleMooncakeStore, "put_batch", blocked_put):
            try:
                self.assertTrue(submit(1))
                self.assertTrue(entered.wait(10))
                self.assertTrue(submit(2))
                self.assertFalse(submit(3))
                reservations[3].cancel()
                reservations[1].cancel()  # Submitted work outlives its serving stream.
                reservations[2].cancel()
                self.assertEqual(publisher.inflight_samples, 3)
                self.assertEqual(publisher.inflight_bytes, 144)
                blocked = publisher.reserve(4, 2)
                self.assertEqual(blocked.code, ExceptionType.HIDDEN_STATE_CAPTURE_BACKPRESSURE)
                metrics, _ = publisher.metrics()
                self.assertCaptureQuota(metrics, 3, 144, 3)
            finally:
                release.set()
            self.assertTrue(publisher.drain())
        self.assertEqual(publisher.inflight_samples, 2)
        publisher.metrics()
        self.consumeCapture(publisher)
        metrics, _ = publisher.metrics()
        self.assertCaptureQuota(metrics, 0, 0, 0)
        self.assertEqual(
            [put["key"] for put in _FakeEagleMooncakeStore.instances[-1].puts],
            ["rtp_TestActor_1", "rtp_TestActor_2"],
        )

    def test_capture_nondefault_producer_stream_and_async_pinned_lengths(self) -> None:
        publisher = self.capturePublisher()
        self.addCleanup(publisher.close)
        reservation = publisher.reserve(42, 2).reservation
        self.assertIsNotNone(reservation)
        producer = torch.cuda.Stream()
        with torch.cuda.stream(producer):
            # Real GPU work and a D2H copy precede the publisher's producer event.
            packed = torch.arange(6, dtype=torch.float16, device="cuda").reshape(2, 3)
            lengths = torch.empty(1, dtype=torch.int32, pin_memory=True)
            lengths.copy_(
                torch.tensor([2], dtype=torch.int32, device="cuda"), non_blocking=True
            )
            self.assertTrue(
                publisher.submit(
                    packed,
                    torch.tensor([7, 8]),
                    lengths,
                    torch.tensor([42]),
                    2,
                    "async lengths",
                )
            )
        del packed, lengths
        gc.collect()
        self.assertTrue(publisher.drain())
        store = _FakeEagleMooncakeStore.instances[-1]
        self.assertEqual(len(store.puts), 1)
        self.assertEqual(store.puts[0]["hidden_states"].tolist(), [[0, 1], [3, 4]])
        self.assertEqual(store.puts[0]["last_hidden_states"].tolist(), [[2], [5]])
        self.assertEqual(store.puts[0]["input_ids"].tolist(), [7, 8])
        self.assertEqual(publisher.inflight_samples, 1)
        self.consumeCapture(publisher)

    def test_capture_error_field_keeps_canonical_lm_head_response(self) -> None:
        _register_no_collectives()
        self.addCleanup(clear_comm_ops)
        os.environ["MOONCAKE_HIDDEN_DIM"] = "2"
        baseline = run_scenario(CaptureContextForwardModel(hidden_size=2), "final_norm_parity_capture_off")
        expected_hidden = baseline["forward_hidden_states"][0]
        self.assertGreater(torch.count_nonzero(expected_hidden).item(), 0)
        expected_logits = expected_hidden[[1, 4]].float().sum(-1, keepdim=True).expand(-1, 2)
        torch.testing.assert_close(baseline["logits"], expected_logits, rtol=0, atol=0)

        for stage, error_type in (("hook", RuntimeError), ("pack", RuntimeError),
                                  ("hook", MemoryError), ("pack", torch.OutOfMemoryError)):
            with self.subTest(stage=stage, error=error_type):
                error = error_type(f"injected transient {stage} failure")
                model = CaptureContextForwardModel(hidden_size=2, failure_stage=stage, error=error)
                puts_after_forward = []

                def after_forward(_):
                    puts_after_forward.append(len(_FakeEagleMooncakeStore.instances[-1].puts))

                result = run_scenario(
                    model,
                    "final_norm_parity_capture_on",
                    2,
                    after_forward=after_forward,
                )
                self.assertEqual(model.capture_flags, [True, True])
                self.assertIn(f"injected transient {stage} failure", model.capture_errors[0])
                self.assertIsNone(model.capture_errors[1])
                self.assertEqual(result["forward_capture_error_codes"], [ExceptionType.EXECUTION_EXCEPTION, 0])
                self.assertEqual(result["forward_capture_error_messages"], [model.capture_errors[0], ""])
                self.assertEqual(puts_after_forward[0], 0)
                self.assertEqual(result["capture_admissions"], [True, True])
                self.assertEqual(result["forward_capture_inflight_samples"], [0, 2])
                self.assertEqual(result["forward_capture_inflight_bytes"][0], 0)
                self.assertGreater(result["forward_capture_inflight_bytes"][1], 0)
                self.assertEqual(len(result["forward_hidden_states"]), 2)
                self.assertEqual(len(result["forward_logits"]), 2)
                for hidden, logits in zip(result["forward_hidden_states"], result["forward_logits"]):
                    torch.testing.assert_close(hidden, expected_hidden, rtol=0, atol=0)
                    torch.testing.assert_close(logits, expected_logits, rtol=0, atol=0)
                torch.testing.assert_close(
                    result["selected_hidden_states"], baseline["selected_hidden_states"], rtol=0, atol=0
                )
                self.assertIn(f"injected transient {stage} failure", result["capture_diagnostics"][0])
                self.assertFalse(result["capture_fatal"])
                self.assertEqual(result["capture_failure_count"], 1)
                self.assertEqual(result["capture_broken_rejection_count"], 0)
                self.assertCaptureStatus(result["capture_metrics"], enabled=0, broken=0)
                puts = _FakeEagleMooncakeStore.instances[-1].puts
                self.assertEqual([item["key"] for item in puts], ["rtp_TestActor_501", "rtp_TestActor_502"])
                for item, start, end in zip(puts, (0, 2), (2, 5)):
                    canonical = expected_hidden[start:end].to(torch.bfloat16)
                    torch.testing.assert_close(item["last_hidden_states"], canonical, rtol=0, atol=0)
                    torch.testing.assert_close(
                        item["hidden_states"],
                        torch.cat((canonical - 20, canonical - 10), dim=-1),
                        rtol=0,
                        atol=0,
                    )
                    self.assertEqual(item["input_ids"].tolist(), list(range(start + 1, end + 1)))

    @patch.dict(os.environ, {"MOONCAKE_ASYNC_PUT_POOL_SIZE": "1", "MOONCAKE_ENABLE_GPU_DIRECT": "0"})
    def test_capture_host_chunk_only_exact_pre_staging_rejection_releases_quota(self) -> None:
        class CaptureHostBufferTooSmall(ValueError):
            pass

        class OtherCapacityError(CaptureHostBufferTooSmall):
            pass

        module = sys.modules["torchspec.transfer.mooncake"]
        first, rejected, later = (f"rtp_TestActor_{i}" for i in (1, 2, 4))
        prefix_keys = {first + suffix for suffix in ("_hs", "_ids", "_lhs", "_ready")}
        for error in (CaptureHostBufferTooSmall, ValueError, OtherCapacityError):
            with self.subTest(error=error), patch.object(
                module, "CaptureHostBufferTooSmall", CaptureHostBufferTooSmall, create=True
            ), contextlib.closing(self.capturePublisher(poll_interval_ms=60000)) as publisher:
                store = _FakeEagleMooncakeStore.instances[-1]
                put = store.put_batch
                attempts = []
                worker_event = torch.cuda.Event()

                def put_chunk(**kwargs):
                    keys = list(kwargs["request_ids"])
                    attempts.append(keys)
                    if keys == [rejected]:
                        store.query_calls.clear()  # Separate cleanup polls from the batch preflight.
                        worker_event.record()  # Mark the worker stream, without staging any store data.
                        raise error("capture host buffer too small")
                    return put(**kwargs)

                with patch.object(store, "put_batch", put_chunk):
                    reservations = [publisher.reserve(i, 1).reservation for i in (1, 2, 3)]
                    self.assertTrue(all(reservations))
                    generation = publisher.reservation_generation(1)
                    self.assertTrue(
                        publisher.submit(
                            torch.zeros((3, 3), dtype=torch.float16, device="cuda"),
                            torch.tensor([7, 8, 9]),
                            torch.ones(3, dtype=torch.int32),
                            torch.tensor([1, 2, 3]),
                            3,
                            "host capacity",
                        )
                    )
                    self.assertTrue(publisher.drain())
                    self.assertTrue(publisher.poll())
                    self.assertEqual(attempts, [[first], [rejected]])
                    self.assertEqual(store.keys, prefix_keys)
                    if error is not CaptureHostBufferTooSmall:
                        self.assertTrue(publisher.fatal)
                        self.assertEqual((publisher.inflight_samples, publisher.inflight_bytes), (3, 144))
                        self.assertIsNone(publisher.reserve(4, 1).reservation)
                        continue

                    self.assertFalse(publisher.fatal)
                    self.assertTrue(worker_event.query())
                    self.assertTrue(all(key.startswith(first + "_") for call in store.query_calls for key in call))
                    self.assertEqual(publisher.reservation_generation(1), generation)
                    self.assertEqual((publisher.inflight_samples, publisher.inflight_bytes), (1, 48))
                    self.submitCapture(publisher, 4)
                    self.assertEqual(attempts, [[first], [rejected], [later]])
                    self.assertEqual([item["key"] for item in store.puts], [first, later])
                    self.assertEqual((publisher.inflight_samples, publisher.inflight_bytes), (2, 96))
                    self.consumeCapture(publisher)

    def test_capture_host_put_memory_error_quarantines_untracked_write(self) -> None:
        publisher = self.capturePublisher(samples=1, bytes=48, tokens=1, poll_interval_ms=60000)
        self.addCleanup(publisher.close)
        store = _FakeEagleMooncakeStore.instances[-1]
        self.assertIsNone(store._gpu_send_buffer)
        payload_refs = []

        def fail_before_future_registration(**kwargs):
            # A host writer may already own this address, with no accepted future or visible key.
            ref = kwargs["hidden_states"][0].untyped_storage()._weak_ref()
            payload_refs.append(ref)
            self.addCleanup(torch.UntypedStorage._free_weak_ref, ref)
            raise MemoryError("host PUT lost future registration")

        with patch.object(store, "put_batch", fail_before_future_registration):
            reservation = self.submitCapture(publisher)
        self.assertEqual(len(payload_refs), 1)
        self.assertTrue(publisher.fatal)
        self.assertIn("host PUT lost future registration", publisher.diagnostic)
        reservation.cancel()
        self.assertTrue(publisher.poll())
        self.assertEqual(publisher.reserve(2, 1).code, ExceptionType.EXECUTION_EXCEPTION)
        self.assertEqual((publisher.inflight_samples, publisher.inflight_bytes), (1, 48))
        publisher.close()
        self.assertFalse(store.flushed)
        self.assertFalse(store.closed)
        self.assertFalse(torch.UntypedStorage._expired(payload_refs[0]))

    def test_capture_gpu_direct_put_memory_error_allows_next_request(self) -> None:
        # A zero host pool makes initialization fail unless the synchronous send path is selected.
        os.environ.update(MOONCAKE_ENABLE_GPU_DIRECT="1", MOONCAKE_ASYNC_PUT_POOL_SIZE="0")
        publisher = self.capturePublisher(samples=1, bytes=48, tokens=1, poll_interval_ms=60000)
        self.addCleanup(publisher.close)
        store = _FakeEagleMooncakeStore.instances[-1]
        original_put = store.put_batch
        original_flush = store.flush
        quota_at_flush = []
        staging_stream = torch.cuda.Stream()
        staged = torch.cuda.Event()

        def put_then_fail_once(**kwargs):
            self.assertTrue(kwargs["hidden_states"][0].is_cuda)
            metadata = original_put(**kwargs)
            if store.put_attempts == 1:
                staging_stream.wait_stream(torch.cuda.current_stream())
                with torch.cuda.stream(staging_stream):
                    store._gpu_send_buffer.copy_(kwargs["last_hidden_states"][0])
                    staged.record()
                raise MemoryError("injected capture worker allocator failure")
            return metadata

        def recovery_flush():
            quota_at_flush.append((publisher.inflight_samples, publisher.inflight_bytes))
            original_flush()

        with patch.object(store, "put_batch", put_then_fail_once):
            with patch.object(store, "flush", recovery_flush):
                first = self.submitCapture(publisher, 1, device="cuda")
            self.assertFalse(publisher.fatal)
            self.assertIn("injected capture worker allocator failure", publisher.diagnostic)
            self.assertEqual(quota_at_flush, [(1, 48)])
            self.assertTrue(staged.query())
            self.assertEqual(store._gpu_send_buffer.item(), 30)
            first.cancel()
            self.assertEqual((publisher.inflight_samples, publisher.inflight_bytes), (1, 48))
            self.consumeCapture(publisher)
            self.submitCapture(publisher, 2, device="cuda")
            self.consumeCapture(publisher)

        self.assertEqual([call["request_ids"] for call in store.put_batch_calls],
                         [["rtp_TestActor_1"], ["rtp_TestActor_2"]])
        self.assertEqual([item["key"] for item in store.puts], ["rtp_TestActor_1", "rtp_TestActor_2"])
        metrics, _ = publisher.metrics()
        self.assertEqual(metrics[f"{_CAPTURE_METRIC_PREFIX}failure_qps"], 1)
        self.assertEqual(metrics[f"{_CAPTURE_METRIC_PREFIX}publish_success_qps"], 1)
        self.assertCaptureStatus(metrics, enabled=1, broken=0)

    def test_capture_async_completion_memory_error_allows_next_publish(self) -> None:
        original_poll = _FakeEagleMooncakeStore.take_async_errors
        poll_count = 0

        def completion_errors(store):
            nonlocal poll_count
            poll_count += 1
            errors = original_poll(store)
            if poll_count == 2:
                errors.append(
                    MemoryError("allocator failure in original capture batch")
                )
            return errors

        model = CacheStoreForwardModel()
        with patch.object(
            _FakeEagleMooncakeStore, "take_async_errors", completion_errors
        ):
            result = run_scenario(model, "capture_async_history", 2)
        self.assertCanonicalZero(result)
        self.assertTrue(result["serving_device_healthy"])
        self.assertIn(
            "allocator failure in original capture batch",
            result["capture_diagnostics"][0],
        )
        self.assertEqual(result["capture_admissions"], [True, True])
        self.assertEqual(result["forward_capture_error_codes"], [0, 0])
        self.assertEqual(result["capture_metrics"][f"{_CAPTURE_METRIC_PREFIX}publish_success_qps"], 1)
        store = _FakeEagleMooncakeStore.instances[-1]
        self.assertEqual(
            [call["request_ids"] for call in store.put_batch_calls],
            [
                ["rtp_TestActor_501", "rtp_TestActor_502"],
                ["rtp_TestActor_1501", "rtp_TestActor_1502"],
            ],
        )

    def test_capture_async_memory_error_is_not_a_put_fence(self) -> None:
        publisher = self.capturePublisher(samples=2, bytes=96, tokens=2, poll_interval_ms=60000)
        self.addCleanup(publisher.close)
        store = _FakeEagleMooncakeStore.instances[-1]
        with patch.object(store, "flush", side_effect=RuntimeError("PUT drain unconfirmed")):
            first = self.submitCapture(publisher)
            first.cancel()
            store.abandon_capture("rtp_TestActor_1")
            store.pending_async_errors.append(MemoryError("completed future without a request ID"))
            self.submitCapture(publisher, 2)
            self.assertEqual(store.pending_async_errors, [])
            self.assertFalse(publisher.fatal)
            self.assertTrue(publisher.poll())
            self.assertEqual(store.put_attempts, 1)
            self.assertEqual(store.reclaim_calls, [])
            self.assertEqual((publisher.inflight_samples, publisher.inflight_bytes), (1, 48))

    def test_capture_unformattable_async_fatal_retains_owners_without_cleanup(self) -> None:
        fatal_before_format = []

        class UnformattableDistError(torch.distributed.DistError):
            def __str__(self):
                fatal_before_format.append(publisher.fatal)
                raise ValueError("diagnostic formatting failed")

        original_flush = _FakeEagleMooncakeStore.flush

        def fatal_completion(store):
            original_flush(store)
            store.pending_async_errors.extend(
                [UnformattableDistError(), RuntimeError("CUDA error: later fatal must not replace the first")]
            )

        publisher = self.capturePublisher(samples=1, bytes=48, tokens=1, poll_interval_ms=60000)
        store = _FakeEagleMooncakeStore.instances[-1]
        packed = torch.tensor([[10, 20, 30]], dtype=torch.float16)
        storage_ref = packed.untyped_storage()._weak_ref()
        self.addCleanup(torch.UntypedStorage._free_weak_ref, storage_ref)
        with patch.object(_FakeEagleMooncakeStore, "flush", fatal_completion):
            try:
                reservation = publisher.reserve(1, 1).reservation
                self.assertIsNotNone(reservation)
                self.assertTrue(
                    publisher.submit(
                        packed,
                        torch.tensor([7]),
                        torch.tensor([1], dtype=torch.int32),
                        torch.tensor([1]),
                        1,
                        "unformattable async fatal",
                    )
                )
                del packed
                self.assertTrue(publisher.drain())
                self.assertTrue(publisher.fatal)
                self.assertEqual(fatal_before_format, [True])
                self.assertIn("unable to format asynchronous publisher error", publisher.diagnostic)
                diagnostic = publisher.diagnostic
                self.assertEqual(store.put_attempts, 1)
                calls = list(store.call_sequence)
                keys = set(store.keys)
                reservation.cancel()
                self.assertTrue(publisher.poll())
                self.assertFalse(publisher.reserve(2, 1).reservation)
                self.assertEqual(publisher.inflight_samples, 1)
                self.assertEqual(publisher.inflight_bytes, 48)
            finally:
                publisher.close()
        self.assertTrue(publisher.fatal)
        self.assertEqual(publisher.diagnostic, diagnostic)
        self.assertEqual(store.call_sequence, calls)
        self.assertEqual(store.keys, keys)
        self.assertFalse(store.closed)
        self.assertFalse(torch.UntypedStorage._expired(storage_ref))

    def test_capture_submit_oom_only_drops_current_capture(self) -> None:
        for scenario in ("capture_bf16", "capture_with_lm_head"):
            with self.subTest(scenario=scenario):
                _check_capture_submit_oom(0, 1, scenario)

    def test_capture_producer_cuda_oom_is_not_swallowed(self) -> None:
        class FatalForward(CacheStoreForwardModel):
            def forward(self, inputs, fmha_impl=None):
                raise torch.OutOfMemoryError("CUDA out of memory in canonical forward")

        with self.assertRaisesRegex(torch.OutOfMemoryError, "CUDA out of memory in canonical forward"):
            run_scenario(FatalForward(), "capture_bf16")
        self.assertEqual(_FakeEagleMooncakeStore.instances[-1].puts, [])
        for stage in ("hook", "pack"):
            for error in (
                MemoryError("CUDA error: illegal address"),
                torch.OutOfMemoryError("CUDA error: illegal address"),
                RuntimeError("CUDA error: injected illegal memory access"),
            ):
                with self.subTest(stage=stage, error=type(error).__name__):
                    with self.assertRaisesRegex(type(error), re.escape(str(error))):
                        run_scenario(CaptureContextForwardModel(failure_stage=stage, error=error), "capture_bf16")
                    self.assertEqual(_FakeEagleMooncakeStore.instances[-1].puts, [])
        with self.assertRaisesRegex(RuntimeError, "canonical final failed"):
            run_scenario(
                CaptureContextForwardModel(failure_stage="final", error=RuntimeError("canonical final failed")),
                "capture_bf16",
            )
        self.assertEqual(_FakeEagleMooncakeStore.instances[-1].puts, [])

    def test_capture_host_put_without_valid_receipt_quarantines_staging(self) -> None:
        cases = [
            ("partial_put", "capture_bf16", 1),
            ("partial_put", "capture_bf16", 2),
            ("metadata_count", "capture_bf16", 2),
            ("metadata_shape", "capture_bf16", 2),
        ]
        if os.environ.get("TEST_USING_DEVICE") != "ROCM":
            cases.append(("metadata_scale", "capture_fp8", 2))
        original_put = _FakeEagleMooncakeStore.put_batch
        for failure, scenario, fail_chunk in cases:
            request_id = 500 + fail_chunk
            with self.subTest(failure=failure, chunk=fail_chunk), patch.dict(
                os.environ, {"MOONCAKE_ASYNC_PUT_POOL_SIZE": "1"}
            ):
                source_refs = []

                def partial_put(store, **kwargs):
                    metadata = original_put(store, **kwargs)
                    if store.put_attempts < fail_chunk:
                        return metadata
                    self.assertEqual(store.put_attempts, fail_chunk)
                    self.assertEqual(kwargs["request_ids"], [f"rtp_TestActor_{request_id}"])
                    source = kwargs["hidden_states"][0]
                    ref = source.untyped_storage()._weak_ref()
                    source_refs.append(ref)
                    self.addCleanup(torch.UntypedStorage._free_weak_ref, ref)
                    if failure == "partial_put":
                        raise RuntimeError("partial PUT after staging copy")
                    if failure == "metadata_count":
                        return []
                    name = "hidden_states_scale" if failure == "metadata_scale" else "hidden_states"
                    metadata[0]["shapes"][name] = (99, 2)
                    return metadata

                with patch.object(_FakeEagleMooncakeStore, "put_batch", partial_put):
                    result = run_scenario(CacheStoreForwardModel(), scenario)
                self.assertCanonicalZero(result, fatal=True)
                self.assertEqual(result["capture_failure_count"], 1)
                diagnostic = result["capture_diagnostics"][0]
                self.assertIn(
                    f"prefix with no flush error observed: {fail_chunk - 1}/2 requests, "
                    f"{2 if fail_chunk == 2 else 0}/5 tokens", diagnostic
                )
                if failure == "partial_put":
                    self.assertIn("partial PUT after staging copy", diagnostic)
                else:
                    self.assertIn(f"chunk-{fail_chunk}-requests-{request_id}-{request_id}", diagnostic)
                    self.assertIn("metadata", diagnostic)
                    if failure != "metadata_count":
                        self.assertIn("key=rtp_TestActor_502, request_id=502", diagnostic)
                self.assertEqual(len(source_refs), 1)
                self.assertFalse(torch.UntypedStorage._expired(source_refs[0]))
                store = _FakeEagleMooncakeStore.instances[-1]
                self.assertEqual(
                    [call["request_ids"] for call in store.put_batch_calls],
                    [["rtp_TestActor_501"], ["rtp_TestActor_502"]][:fail_chunk],
                )
                # Only earlier chunks had receipts. Neither recovery nor shutdown may drain this PUT.
                self.assertEqual(store.call_sequence.count("flush"), fail_chunk - 1)
                self.assertEqual(store.call_sequence[-1], "put_batch")
                self.assertFalse(store.closed)
                self.assertEqual(result["forward_capture_inflight_samples"], [2])
                # The quota GAUGE is a period average; use the publisher snapshot for retained bytes.
                bytes_per_token = 34 if scenario == "capture_fp8" else 28
                self.assertEqual(result["forward_capture_inflight_bytes"], [5 * bytes_per_token + 2 * 20])
                self.assertCaptureQps(
                    result["capture_metrics"],
                    batch=1,
                    publish_success=0,
                    failure=1,
                    store_failure=1,
                )

    def test_capture_concurrent_close_does_not_hold_gil_while_joining(self) -> None:
        entered = threading.Event()
        release = threading.Event()
        close_started = threading.Event()
        failures = []
        original_put = _FakeEagleMooncakeStore.put_batch

        def blocked_put(store, **kwargs):
            entered.set()
            if not release.wait(10):
                raise RuntimeError("test did not release closing worker")
            return original_put(store, **kwargs)

        publisher = self.capturePublisher()
        self.addCleanup(publisher.close)

        def close():
            close_started.set()
            try:
                publisher.close()
            except Exception as error:
                failures.append(error)

        with patch.object(_FakeEagleMooncakeStore, "put_batch", blocked_put):
            reservation = publisher.reserve(1, 1).reservation
            self.assertIsNotNone(reservation)
            self.assertTrue(
                publisher.submit(
                    torch.tensor([[10, 20, 30]], dtype=torch.float16),
                    torch.tensor([7]),
                    torch.tensor([1], dtype=torch.int32),
                    torch.tensor([1]),
                    1,
                    "concurrent close",
                )
            )
            self.assertTrue(entered.wait(10))
            unsubmitted = publisher.reserve(2, 1).reservation
            self.assertIsNotNone(unsubmitted)
            self.assertEqual(publisher.inflight_samples, 2)
            first = threading.Thread(target=close)
            second = threading.Thread(target=close)
            try:
                first.start()
                self.assertTrue(close_started.wait(10))
                self.assertTrue(publisher.wait_stopping())
                self.assertTrue(first.is_alive())
                self.assertFalse(publisher.drain(timeout_ms=0))
                unsubmitted.cancel()
                self.assertFalse(unsubmitted.active)
                self.assertEqual(publisher.inflight_samples, 1)
                reservation.cancel()
                reservation.cancel()
                self.assertFalse(reservation.active)
                self.assertEqual(publisher.inflight_samples, 1)
                self.assertEqual(publisher.inflight_bytes, 48)
                second.start()
            finally:
                release.set()
                if first.ident is not None:
                    first.join(10)
                if second.ident is not None:
                    second.join(10)
        self.assertFalse(first.is_alive())
        self.assertFalse(second.is_alive())
        self.assertEqual(failures, [])
        self.assertFalse(_FakeEagleMooncakeStore.instances[-1].closed)
        self.assertEqual(publisher.inflight_samples, 0)
        self.assertFalse(publisher.reserve(2, 1).reservation)

    def test_capture_failed_flush_retains_bounded_quota(self) -> None:
        publisher = self.capturePublisher(samples=1, bytes=48, tokens=1, poll_interval_ms=100)
        store_calls = []
        packed = torch.tensor([[10, 20, 30]], dtype=torch.float16)
        storage_ref = packed.untyped_storage()._weak_ref()
        with patch.object(_FakeEagleMooncakeStore, "fail_flush", True), patch.object(
            _FakeEagleMooncakeStore, "events", store_calls
        ):
            try:
                reservation = publisher.reserve(1, 1).reservation
                self.assertIsNotNone(reservation)
                self.assertTrue(
                    publisher.submit(
                        packed,
                        torch.tensor([7]),
                        torch.tensor([1], dtype=torch.int32),
                        torch.tensor([1]),
                        1,
                        "failed flush",
                    )
                )
                del packed
                self.assertTrue(publisher.drain())
                self.assertFalse(torch.UntypedStorage._expired(storage_ref))
                self.assertEqual(publisher.inflight_samples, 1)
                self.assertEqual(publisher.inflight_bytes, 48)
                self.assertFalse(publisher.reserve(2, 1).reservation)
                self.assertIn("flush failure", publisher.diagnostic)
                self.assertFalse(publisher.fatal)
                store = _FakeEagleMooncakeStore.instances[-1]
                store.abandon_capture("rtp_TestActor_1")
                source_keys = set(store.keys)
                self.assertTrue(publisher.poll())
                self.assertEqual(store.reclaim_calls, [])
                self.assertEqual(store.removed_keys, [])
                self.assertEqual(store.keys, source_keys)
                reservation.cancel()
                self.assertTrue(publisher.drain())
                self.assertEqual(store.reclaim_calls, [])
                self.assertEqual(store.keys, source_keys)
                self.assertEqual(publisher.inflight_samples, 1)
                self.assertEqual(publisher.inflight_bytes, 48)
                put_attempts = store.put_attempts
                self.assertEqual(put_attempts, 1)
                flushes = store_calls.count("store.flush")
                with patch.object(_FakeEagleMooncakeStore, "fail_flush", False):
                    # poll/drain observe worker idleness, not an immediate completion retry.
                    deadline = time.monotonic() + 5
                    while publisher.inflight_samples and time.monotonic() < deadline:
                        self.assertTrue(publisher.poll())
                        time.sleep(0.01)
                    self.assertGreater(store_calls.count("store.flush"), flushes)
                    self.assertEqual(store.put_attempts, put_attempts)
                    self.assertEqual(store.reclaim_calls, ["rtp_TestActor_1"])
                    self.assertEqual(store.keys, set())
                    self.assertEqual(publisher.inflight_samples, 0)
                    self.assertEqual(publisher.inflight_bytes, 0)
                    self.assertTrue(torch.UntypedStorage._expired(storage_ref))
                    self.assertFalse(publisher.fatal)
                    next_reservation = publisher.reserve(2, 1).reservation
                    self.assertIsNotNone(next_reservation)
                    next_reservation.cancel()
                    publisher.close()
                    self.assertTrue(store.closed)
            finally:
                publisher.close()
                torch.UntypedStorage._free_weak_ref(storage_ref)

    @patch.dict(os.environ, {"MOONCAKE_ASYNC_PUT_POOL_SIZE": "1"})
    def test_capture_bf16_store_contract_slices_each_request_and_flushes_on_shutdown(
        self,
    ) -> None:
        _register_no_collectives()
        self.addCleanup(clear_comm_ops)
        result = run_scenario(CacheStoreForwardModel(), "capture_bf16")

        self.assertEqual(result["hidden_width"], 1)
        self.assertEqual(result["capture_failure_count"], 0)
        self.assertEqual(result["capture_broken_rejection_count"], 0)
        metrics = result["capture_metrics"]
        self.assertCaptureMetricTypes(result)
        self.assertEqual(
            set(metrics),
            _CAPTURE_QPS_METRICS | _CAPTURE_STATUS_METRICS | _CAPTURE_PUBLISH_METRICS | _CAPTURE_QUOTA_METRICS,
        )
        self.assertCaptureQps(
            metrics,
            batch=1,
            publish_success=1,
            failure=0,
            bf16_publish=1,
        )
        self.assertCaptureStatus(metrics, enabled=0, broken=0)
        self.assertGreaterEqual(
            metrics[f"{_CAPTURE_METRIC_PREFIX}publish_latency_us"], 0
        )
        self.assertGreaterEqual(
            metrics[f"{_CAPTURE_METRIC_PREFIX}store_put_latency_us"], 0
        )
        self.assertEqual(metrics[f"{_CAPTURE_METRIC_PREFIX}publish_request_count"], 2)
        self.assertEqual(metrics[f"{_CAPTURE_METRIC_PREFIX}publish_token_count"], 5)
        self.assertEqual(metrics[f"{_CAPTURE_METRIC_PREFIX}publish_payload_bytes"], 70)
        self.assertEqual(
            metrics[f"{_CAPTURE_METRIC_PREFIX}publish_input_ids_bytes"], 40
        )
        self.assertEqual(
            metrics[f"{_CAPTURE_METRIC_PREFIX}publish_aux_hidden_bytes"], 20
        )
        self.assertEqual(
            metrics[f"{_CAPTURE_METRIC_PREFIX}publish_last_hidden_bytes"], 10
        )
        self.assertEqual(metrics[f"{_CAPTURE_METRIC_PREFIX}publish_scale_bytes"], 0)
        self.assertEqual(len(_FakeEagleMooncakeStore.instances), 1)
        store = _FakeEagleMooncakeStore.instances[0]
        expected_default_layout = {
            "hidden_dim": 1,
            "max_seq_len": 8192,
            "get_batch_size": 1,
            "gpu_buffer_size": None,
            "num_aux_layers": 2,
        }
        self.assertEqual(store.config_at_init, expected_default_layout)
        self.assertEqual(store.config_at_setup, expected_default_layout)
        self.assertTrue(store.warmed_up)
        self.assertTrue(store.flushed)
        self.assertFalse(store.closed)
        self.assertEqual(store.config.async_put_pool_size, 1)
        self.assertEqual(len(store.put_batch_calls), 2)
        self.assertEqual(
            [call["request_ids"] for call in store.put_batch_calls],
            [["rtp_TestActor_501"], ["rtp_TestActor_502"]],
        )
        self.assertEqual(_FakeMooncakeConfig.make_store_key_calls, [0, 501, 502])
        self.assertEqual(
            [call["lengths"] for call in store.put_batch_calls], [[2], [3]]
        )
        self.assertEqual(
            [call["batch_id"] for call in store.put_batch_calls],
            [
                "rtp-forward-TestActor-1-requests-501-502-chunk-1-requests-501-501",
                "rtp-forward-TestActor-1-requests-501-502-chunk-2-requests-502-502",
            ],
        )
        self.assertEqual(
            [tuple(put["hidden_states"].shape) for put in store.puts],
            [(2, 2), (3, 2)],
        )
        self.assertEqual(
            [tuple(put["last_hidden_states"].shape) for put in store.puts],
            [(2, 1), (3, 1)],
        )
        self.assertEqual(store.puts[0]["input_ids"].tolist(), [1, 2])
        self.assertEqual(store.puts[1]["input_ids"].tolist(), [3, 4, 5])
        self.assertTrue(
            all(put["input_ids"].dtype == torch.int64 for put in store.puts)
        )
        self.assertTrue(
            all(put["hidden_states"].dtype == torch.bfloat16 for put in store.puts)
        )
        self.assertTrue(
            all(put["source_hidden_dtype"] == torch.bfloat16 for put in store.puts)
        )
        self.assertTrue(
            all(put["source_last_hidden_dtype"] == torch.bfloat16 for put in store.puts)
        )
        self.assertEqual(result["forward_hidden_states"][0].dtype, torch.float16)
        self.assertTrue(all(put["hidden_states_scale"] is None for put in store.puts))
        expected_auxiliary = [
            [[10.0, 20.0], [10.0, 20.0]],
            [[10.0, 20.0], [10.0, 20.0], [10.0, 20.0]],
        ]
        expected_last = [
            [[30.0], [30.0]],
            [[30.0], [30.0], [30.0]],
        ]
        self.assertEqual(
            [put["hidden_states"].tolist() for put in store.puts],
            expected_auxiliary,
        )
        self.assertEqual(
            [put["last_hidden_states"].tolist() for put in store.puts],
            expected_last,
        )

    def test_capture_publisher_consumes_non_4096_hidden_env(self) -> None:
        os.environ["MOONCAKE_HIDDEN_DIM"] = "2048"

        result = run_scenario(
            CacheStoreForwardModel(hidden_size=2048), "capture_non_4096_hidden"
        )

        self.assertEqual(result["hidden_width"], 2048)
        store = _FakeEagleMooncakeStore.instances[0]
        self.assertEqual(store.config_at_init["hidden_dim"], 2048)
        self.assertEqual(store.config_at_setup["hidden_dim"], 2048)
        self.assertEqual(
            [tuple(put["hidden_states"].shape) for put in store.puts],
            [(2, 4096), (3, 4096)],
        )
        self.assertEqual(
            [tuple(put["last_hidden_states"].shape) for put in store.puts],
            [(2, 2048), (3, 2048)],
        )

    def test_capture_publisher_consumes_gpu_direct_layout_env_before_store_setup(
        self,
    ) -> None:
        exported_layout = {
            "MOONCAKE_HIDDEN_DIM": "1",
            "MOONCAKE_MAX_SEQ_LEN": "16384",
            "MOONCAKE_GET_BATCH_SIZE": "8",
            "MOONCAKE_GPU_BUFFER_SIZE": "268435456",
            "MOONCAKE_NUM_AUX_LAYERS": "2",
        }
        os.environ.update(exported_layout)

        run_scenario(CacheStoreForwardModel(), "capture_bf16")

        expected_layout = {
            attr_name: int(exported_layout[env_name])
            for env_name, attr_name in _MOONCAKE_RTP_LAYOUT_ENV.items()
        }
        store = _FakeEagleMooncakeStore.instances[0]
        self.assertEqual(store.config_at_init, expected_layout)
        self.assertEqual(store.config_at_setup, expected_layout)

    def test_capture_missing_namespace_fails_before_store_construction(self) -> None:
        os.environ.pop(_MOONCAKE_STORE_KEY_NAMESPACE_ENV)
        result = run_scenario(CacheStoreForwardModel(), "capture_bf16")
        self.assertCaptureDisabled(result, "store_key_namespace")
        self.assertEqual(_FakeEagleMooncakeStore.instances, [])

    @patch.dict(os.environ, {"MOONCAKE_ASYNC_PUT_POOL_SIZE": "1"})
    def test_capture_namespace_matches_rtp_engine_key_contract(self) -> None:
        os.environ[_MOONCAKE_STORE_KEY_NAMESPACE_ENV] = "ActorA1"

        run_scenario(CacheStoreForwardModel(), "capture_bf16")

        store = _FakeEagleMooncakeStore.instances[0]
        self.assertEqual(
            [put["key"] for put in store.puts],
            ["rtp_ActorA1_501", "rtp_ActorA1_502"],
        )
        self.assertEqual(
            [call["request_ids"] for call in store.put_batch_calls],
            [["rtp_ActorA1_501"], ["rtp_ActorA1_502"]],
        )
        self.assertEqual(
            [call["batch_id"] for call in store.put_batch_calls],
            [
                "rtp-forward-ActorA1-1-requests-501-502-chunk-1-requests-501-501",
                "rtp-forward-ActorA1-1-requests-501-502-chunk-2-requests-502-502",
            ],
        )

    def test_capture_same_request_id_differs_across_namespaces(self) -> None:
        os.environ[_MOONCAKE_STORE_KEY_NAMESPACE_ENV] = "ActorA1"
        run_scenario(CacheStoreForwardModel(), "capture_bf16")
        os.environ[_MOONCAKE_STORE_KEY_NAMESPACE_ENV] = "ActorB2"
        run_scenario(CacheStoreForwardModel(), "capture_bf16")

        first_key = _FakeEagleMooncakeStore.instances[0].puts[0]["key"]
        second_key = _FakeEagleMooncakeStore.instances[1].puts[0]["key"]
        self.assertEqual(first_key, "rtp_ActorA1_501")
        self.assertEqual(second_key, "rtp_ActorB2_501")
        self.assertNotEqual(first_key, second_key)

    def test_capture_invalid_namespace_fails_before_store_construction(self) -> None:
        os.environ[_MOONCAKE_STORE_KEY_NAMESPACE_ENV] = "invalid_namespace"
        result = run_scenario(CacheStoreForwardModel(), "capture_bf16")
        self.assertCaptureDisabled(result, "store_key_namespace")
        self.assertEqual(_FakeEagleMooncakeStore.instances, [])

    def assertRocmFp8CaptureRejected(
        self,
        scenario: str,
        forward_count: int = 1,
    ) -> None:
        fp8_module = cast(Any, sys.modules["torchspec.utils.fp8"])
        with patch.object(
            fp8_module,
            "quantize_aux_hidden_states",
            wraps=fp8_module.quantize_aux_hidden_states,
        ) as quantizer:
            model = CacheStoreForwardModel()
            result = run_scenario(model, scenario, forward_count)
            self.assertCanonicalZero(result)
            self.assertFalse(result["capture_plan_enabled"])
            self.assertIn(
                "TorchSpec hidden-state capture does not support FP8 on ROCm", result["capture_diagnostics"][0]
            )
            self.assertIn("float8_e4m3fnuz", result["capture_diagnostics"][0])
            self.assertEqual(_FakeEagleMooncakeStore.instances, [])
            self.assertEqual(model.forward_calls, forward_count)
            self.assertEqual(model.capture_flags, [False] * forward_count)
            quantizer.assert_not_called()

    def assertRocmContextParallelRejected(self, model, scenario: str) -> None:
        # Context-parallel input processing is a CUDA-only feature: the production
        # processor rejects ROCm builds by design. The ROCm lane verifies that
        # guard; the CUDA lane covers the full CP planning and restore logic.
        with self.assertRaisesRegex(
            RuntimeError, "Context parallel not supported on ROCm"
        ):
            run_scenario(model, scenario)

    def test_capture_fp8_passes_per_token_per_layer_scale(self) -> None:
        if os.environ.get("TEST_USING_DEVICE") == "ROCM":
            self.assertRocmFp8CaptureRejected("capture_fp8")
            return

        result = run_scenario(CacheStoreForwardModel(), "capture_fp8")

        metrics = result["capture_metrics"]
        self.assertCaptureMetricTypes(result)
        self.assertCaptureQps(
            metrics,
            batch=1,
            publish_success=1,
            failure=0,
            fp8_publish=1,
        )
        self.assertCaptureStatus(metrics, enabled=0, broken=0)
        self.assertGreaterEqual(
            metrics[f"{_CAPTURE_METRIC_PREFIX}quantize_latency_us"], 0
        )
        self.assertEqual(metrics[f"{_CAPTURE_METRIC_PREFIX}publish_payload_bytes"], 100)
        self.assertEqual(
            metrics[f"{_CAPTURE_METRIC_PREFIX}publish_input_ids_bytes"], 40
        )
        self.assertEqual(
            metrics[f"{_CAPTURE_METRIC_PREFIX}publish_aux_hidden_bytes"], 10
        )
        self.assertEqual(
            metrics[f"{_CAPTURE_METRIC_PREFIX}publish_last_hidden_bytes"], 10
        )
        self.assertEqual(metrics[f"{_CAPTURE_METRIC_PREFIX}publish_scale_bytes"], 40)

        store = _FakeEagleMooncakeStore.instances[0]
        self.assertTrue(
            all(put["source_hidden_dtype"] == torch.float8_e4m3fn for put in store.puts)
        )
        self.assertEqual(
            [tuple(put["hidden_states_scale"].shape) for put in store.puts],
            [(2, 2), (3, 2)],
        )
        self.assertTrue(
            all(put["hidden_states_scale"].dtype == torch.float32 for put in store.puts)
        )
        expected_auxiliary = [
            [[10.0, 20.0], [10.0, 20.0]],
            [[10.0, 20.0], [10.0, 20.0], [10.0, 20.0]],
        ]
        for put, expected in zip(store.puts, expected_auxiliary):
            torch.testing.assert_close(
                put["dequantized_hidden_states"],
                torch.tensor(expected, dtype=torch.float32),
                rtol=0.05,
                atol=0.5,
            )
        self.assertEqual(
            [put["last_hidden_states"].tolist() for put in store.puts],
            [
                [[30.0], [30.0]],
                [[30.0], [30.0], [30.0]],
            ],
        )

    def test_final_hidden_numeric_parity_capture_on_off(self) -> None:
        os.environ["MOONCAKE_HIDDEN_DIM"] = "2"
        capture_off_model = FinalNormParityModel(hidden_size=2)
        capture_on_model = FinalNormParityModel(hidden_size=2)

        capture_off_result = run_scenario(
            capture_off_model, "final_norm_parity_capture_off"
        )
        capture_on_result = run_scenario(
            capture_on_model, "final_norm_parity_capture_on"
        )

        token_values = torch.arange(1, 6, dtype=torch.float16, device="cuda").reshape(
            -1, 1
        )
        raw_hidden = torch.cat((token_values, token_values + 1), dim=-1)
        expected_all = (
            raw_hidden
            * torch.rsqrt(
                raw_hidden.float().pow(2).mean(-1, keepdim=True)
                + FinalNormParityModel._FINAL_NORM_EPS
            )
            * raw_hidden.new_tensor(FinalNormParityModel._FINAL_NORM_GAMMA)
        ).to(torch.float16)
        expected_selected = expected_all.index_select(
            0, torch.tensor([1, 4], device="cuda")
        )
        # The fixture LM head has two identical all-ones rows and FP16 output.
        expected_logits = (
            expected_selected.sum(-1, keepdim=True).expand(-1, 2).float().cpu()
        )
        expected_all = expected_all.cpu()
        expected_selected = expected_selected.cpu()

        for capture, model, result in (
            (False, capture_off_model, capture_off_result),
            (True, capture_on_model, capture_on_result),
        ):
            self.assertEqual(model.capture_flags, [capture])
            self.assertEqual(model.forward_calls, 1)
            self.assertEqual(
                result["all_hidden_layouts"],
                [
                    {
                        "sizes": [5, 2],
                        "strides": [2, 1],
                        "storage_offset": 0,
                        "is_cuda": True,
                        "is_contiguous": True,
                        "is_dense": True,
                    }
                ],
            )
            self.assertEqual(len(result["forward_hidden_states"]), 1)
            # This fixture field comes from all_hidden_states, not selected LM rows.
            all_hidden = result["forward_hidden_states"][0]
            self.assertEqual(tuple(all_hidden.shape), (5, 2))
            torch.testing.assert_close(all_hidden, expected_all, rtol=0, atol=0)
            selected_hidden = result["selected_hidden_states"]
            self.assertEqual(tuple(selected_hidden.shape), (2, 2))
            torch.testing.assert_close(
                selected_hidden, expected_selected, rtol=0, atol=0
            )
            self.assertTrue(result["logits_defined"])
            self.assertEqual(tuple(result["logits"].shape), (2, 2))
            torch.testing.assert_close(
                result["logits"], expected_logits, rtol=0, atol=0
            )
            self.assertFalse(result["capture_fatal"])
            self.assertEqual(result["capture_failure_count"], 0)

        torch.testing.assert_close(
            capture_on_result["forward_hidden_states"][0],
            capture_off_result["forward_hidden_states"][0],
            rtol=0,
            atol=0,
        )
        torch.testing.assert_close(
            capture_on_result["logits"], capture_off_result["logits"], rtol=0, atol=0
        )
        self.assertEqual(capture_off_result["capture_admissions"], [False])
        self.assertEqual(capture_on_result["capture_admissions"], [True])
        store = _FakeEagleMooncakeStore.instances[0]
        self.assertEqual(
            [put["input_ids"].tolist() for put in store.puts],
            [[1, 2], [3, 4, 5]],
        )
        for put, begin, end in ((store.puts[0], 0, 2), (store.puts[1], 2, 5)):
            torch.testing.assert_close(
                put["last_hidden_states"], expected_all[begin:end].to(torch.bfloat16)
            )
        self.assertTrue(store.flushed)
        self.assertFalse(store.closed)

    @unittest.skipUnless(
        os.environ.get("TEST_USING_DEVICE") == "ROCM",
        "ROCm-only context-parallel guard",
    )
    def test_capture_context_parallel_is_rejected_on_rocm(self) -> None:
        model = CacheStoreForwardModel()
        with self.assertRaisesRegex(
            RuntimeError, "Context parallel not supported on ROCm"
        ):
            run_scenario(model, "capture_cp")
        self.assertEqual(model.forward_calls, 0)

    @_skip_rocm_context_parallel
    def test_capture_context_parallel_rejects_invalid_num_valid_tokens(self) -> None:
        submit_frames = []
        _register_identity_allreduce(submit_frames=submit_frames)
        self.addCleanup(clear_comm_ops)
        result = run_scenario(
            CacheStoreForwardModel(), "capture_cp_invalid_num_valid_tokens"
        )
        self.assertEqual(submit_frames, [2])
        self.assertCanonicalZero(result, rows=6)
        self.assertIn(
            "context parallel restored captured hidden-state row count 6 must match expected row count 5",
            result["capture_diagnostics"][0],
        )
        self.assertEqual(result["cp_handle_outputs_calls"], 2)
        self.assertEqual(result["capture_failure_count"], 1)
        self.assertEqual(_FakeEagleMooncakeStore.instances[0].puts, [])

    @_skip_rocm_context_parallel
    def test_capture_context_parallel_rejects_local_layout_before_restore(self) -> None:
        if os.environ.get("TEST_USING_DEVICE") == "ROCM":
            self.assertRocmContextParallelRejected(
                MalformedCaptureLayoutModel(), "capture_cp_malformed_layout"
            )
            return
        layout_statuses = []

        def allreduce(tensor, op, mode, dest):
            layout_statuses.append(int(tensor.item()))
            return tensor

        _register_no_collectives(allreduce)
        self.addCleanup(clear_comm_ops)
        result = run_scenario(
            MalformedCaptureLayoutModel(),
            "capture_cp_malformed_layout",
        )

        self.assertEqual(layout_statuses, [0, 2])
        self.assertEqual(result["cp_handle_outputs_calls"], 1)
        self.assertEqual(result["capture_failure_count"], 1)
        self.assertIn(
            "captured hidden-state width 2 must match expected packed width 3",
            result["capture_diagnostics"][0],
        )
        self.assertEqual(_FakeEagleMooncakeStore.instances[0].puts, [])
        hidden_states = result["forward_hidden_states"][0]
        torch.testing.assert_close(hidden_states, torch.zeros_like(hidden_states))

    @_skip_rocm_context_parallel
    def test_capture_context_parallel_remote_invalid_rank_skips_restore(self) -> None:
        # Exercise each rank locally with the peer's status supplied by the comm
        # fixture. Rejected layouts also agree cleanup completed before continuing.
        for owner in (True, False):
            for local, remote in ((2, 1), (1, 2), (1, 1), (1, 0), (0, 1), (2, 0)):
                with self.subTest(owner=owner, local=local, remote=remote):
                    reduced = min(local, remote)
                    forward_count = 2 if reduced == 1 else 1
                    calls = []

                    def allreduce(tensor, op, mode, dest):
                        peer_status = remote if not calls else 2
                        calls.append((int(tensor.item()), op, mode, dest))
                        return torch.minimum(tensor, torch.full_like(tensor, peer_status))

                    _register_no_collectives(allreduce)
                    store_count = len(_FakeEagleMooncakeStore.instances)
                    model = CaptureContextForwardModel(failure_stage={0: "layout", 1: "hook", 2: ""}[local])
                    try:
                        result = run_scenario(
                            model,
                            "capture_cp" if owner else "capture_cp_non_owner",
                            forward_count,
                        )
                    finally:
                        clear_comm_ops()
                    self.assertEqual(calls, [(local, 3, 0, None)] + [(2, 3, 0, None)] * forward_count)
                    self.assertEqual(result["cp_handle_outputs_calls"], 1 + 2 * (forward_count - 1))
                    self.assertFalse(result["capture_fatal"])
                    self.assertEqual(
                        result["forward_capture_error_codes"],
                        [ExceptionType.EXECUTION_EXCEPTION] + [0] * (forward_count - 1),
                    )
                    if reduced == 0:
                        expected_error = "width" if local == 0 else "layout contract violation on another TP rank"
                    else:
                        expected_error = (
                            "injected transient hook failure" if local == 1
                            else "recoverable hidden-state capture failure on another TP rank"
                        )
                    self.assertIn(expected_error, result["forward_capture_error_messages"][0])
                    self.assertEqual(result["forward_capture_error_messages"][1:], [""] * (forward_count - 1))
                    self.assertEqual(result["forward_capture_inflight_samples"][0], 0)
                    self.assertEqual(result["forward_capture_inflight_bytes"][0], 0)
                    for hidden in result["forward_hidden_states"]:
                        self.assertEqual(tuple(hidden.shape), (6, 1))
                        self.assertGreater(torch.count_nonzero(hidden).item(), 0)
                    stores = _FakeEagleMooncakeStore.instances[store_count:]
                    if not owner:
                        self.assertEqual(stores, [])
                        continue
                    self.assertEqual(result["capture_failure_count"], 1)
                    self.assertIn(expected_error, result["capture_diagnostics"][0])
                    if reduced == 0:
                        self.assertCaptureStatus(result["capture_metrics"], enabled=0, broken=1)
                        self.assertEqual(stores[0].puts, [])
                    else:
                        self.assertCaptureStatus(result["capture_metrics"], enabled=0, broken=0)
                        self.assertEqual(result["capture_admissions"], [True, True])
                        self.assertEqual(result["forward_capture_inflight_samples"], [0, 1])
                        self.assertEqual(len(stores[0].puts), 1)

    @_skip_rocm_context_parallel
    def test_capture_context_parallel_rechecks_width_after_restore(self) -> None:
        if os.environ.get("TEST_USING_DEVICE") == "ROCM":
            self.assertRocmContextParallelRejected(
                CacheStoreForwardModel(), "capture_cp_malformed_restored_width"
            )
            return
        layout_statuses = []

        def allreduce(tensor, op, mode, dest):
            layout_statuses.append(int(tensor.item()))
            return tensor

        submit_frames = []
        _register_no_collectives(allreduce, submit_frames=submit_frames)
        self.addCleanup(clear_comm_ops)
        result = run_scenario(
            CacheStoreForwardModel(),
            "capture_cp_malformed_restored_width",
        )

        self.assertEqual(submit_frames, [2])
        self.assertEqual(layout_statuses, [2])
        self.assertEqual(result["cp_handle_outputs_calls"], 2)
        self.assertEqual(result["capture_failure_count"], 1)
        self.assertIn(
            "context parallel restored captured hidden-state width 2 must match expected packed width 3",
            result["capture_diagnostics"][0],
        )
        self.assertEqual(_FakeEagleMooncakeStore.instances[0].puts, [])
        hidden_states = result["forward_hidden_states"][0]
        torch.testing.assert_close(hidden_states, torch.zeros_like(hidden_states))

    def test_capture_packed_layout_reports_rows_dtype_and_device_errors(self) -> None:
        cases = [
            (
                MalformedCaptureRowCountModel(),
                "captured hidden-state row count 4 must match expected row count 5",
            ),
            (
                MalformedCaptureDtypeModel(),
                "captured hidden-state dtype must match the model dtype",
            ),
            (
                MalformedCaptureDeviceModel(),
                "captured hidden-state must be on the model input device",
            ),
        ]
        for model, expected_error in cases:
            with self.subTest(expected_error=expected_error):
                result = run_scenario(model, "capture_bf16")
                self.assertEqual(model.capture_flags, [True])
                self.assertIn(expected_error, result["capture_diagnostics"][0])
                self.assertEqual(result["capture_failure_count"], 1)
                self.assertCanonicalZero(result)
                self.assertEqual(_FakeEagleMooncakeStore.instances[-1].puts, [])

    def test_capture_malformed_rank_uses_fallback_before_any_narrow(self) -> None:
        layout_statuses = []

        def allreduce(tensor, op, mode, dest):
            layout_statuses.append(int(tensor.item()))
            return tensor

        _register_no_collectives(allreduce)
        self.addCleanup(clear_comm_ops)
        result = run_scenario(
            MalformedCaptureRankModel(),
            "capture_bf16_tp",
        )

        self.assertEqual(layout_statuses, [0, 2])
        self.assertEqual(_FakeEagleMooncakeStore.instances[0].puts, [])
        hidden_states = result["forward_hidden_states"][0]
        self.assertEqual(tuple(hidden_states.shape), (5, 1))
        torch.testing.assert_close(hidden_states, torch.zeros_like(hidden_states))

    def test_capture_non_owner_layout_status_rejects_owner_publish(self) -> None:
        calls = 0

        def allreduce(tensor, op, mode, dest):
            nonlocal calls
            calls += 1
            self.assertTrue(tensor.is_cuda)
            self.assertEqual(tensor.dtype, torch.int32)
            self.assertEqual(tuple(tensor.shape), (1,))
            self.assertEqual((op, mode, dest), (3, 0, None))
            # Simulate a non-owner TP rank reporting a malformed capture, then successful cleanup.
            return torch.zeros_like(tensor) if calls == 1 else tensor

        _register_no_collectives(allreduce)
        self.addCleanup(clear_comm_ops)
        result = run_scenario(
            CacheStoreForwardModel(),
            "capture_bf16_tp",
        )

        self.assertEqual(calls, 2)
        self.assertEqual(result["capture_failure_count"], 1)
        self.assertIn("another TP rank", result["capture_diagnostics"][0])
        self.assertEqual(_FakeEagleMooncakeStore.instances[0].puts, [])
        self.assertCanonicalZero(result)

    @_skip_rocm_context_parallel
    def test_capture_context_parallel_uses_original_ids_and_lengths(self) -> None:
        if os.environ.get("TEST_USING_DEVICE") == "ROCM":
            self.assertRocmContextParallelRejected(
                PositionAwareCaptureModel(), "capture_cp_last_hidden"
            )
            return
        events = []
        _FakeEagleMooncakeStore.events = events
        layout_statuses = []

        def allreduce(tensor, op, mode, dest):
            self.assertEqual((op, mode, dest), (3, 0, None))
            layout_statuses.append(int(tensor.item()))
            events.append(f"allreduce:{int(tensor.item())}")
            return tensor

        _register_no_collectives(allreduce)
        self.addCleanup(clear_comm_ops)
        model = PositionAwareCaptureModel()
        result = run_scenario(model, "capture_cp_last_hidden")

        self.assertEqual(layout_statuses, [2])
        self.assertEqual(
            events,
            [
                "store.setup",
                "store.warmup",
                "allreduce:2",
                "store.take_async_errors",
                "store.put_batch:rtp-forward-TestActor-1-requests-601-601",
                "store.flush",
                "store.take_async_errors",
                "store.flush",
                "store.take_async_errors",
            ],
        )
        self.assertEqual(result["cp_handle_outputs_calls"], 1)
        self.assertEqual(model.seen_input_lengths, [[4]])
        self.assertEqual(result["hidden_width"], 1)
        store = _FakeEagleMooncakeStore.instances[0]
        self.assertTrue(store.flushed)
        self.assertFalse(store.closed)
        self.assertEqual(
            store.keys,
            {"rtp_TestActor_601" + suffix for suffix in ("_hs", "_ids", "_lhs", "_ready")},
        )
        self.assertEqual(store.reclaim_calls, [])
        self.assertEqual(store.removed_keys, [])
        self.assertFalse(result["capture_fatal"])
        self.assertEqual(result["capture_failure_count"], 0)
        # Local flush retains the six-token reservation: 28 bytes/token + 20 metadata bytes.
        self.assertEqual(result["forward_capture_inflight_samples"], [1])
        self.assertEqual(result["forward_capture_inflight_bytes"], [6 * 28 + 20])
        # Shutdown transfers the live ledger to quarantine, not consumer cleanup.
        self.assertEqual(result["capture_inflight_samples"], 0)
        self.assertEqual(result["capture_inflight_bytes"], 0)
        puts = store.puts
        self.assertEqual(len(puts), 1)
        self.assertEqual(puts[0]["key"], "rtp_TestActor_601")
        self.assertEqual(puts[0]["input_ids"].tolist(), [1, 2, 3, 4, 5, 6])
        torch.testing.assert_close(
            puts[0]["hidden_states"],
            torch.tensor(
                [[11, 21], [12, 22], [13, 23], [14, 24], [15, 25], [16, 26]],
                dtype=torch.bfloat16,
            ),
            rtol=0,
            atol=0,
        )
        torch.testing.assert_close(
            puts[0]["last_hidden_states"],
            torch.tensor([[31], [32], [33], [34], [35], [36]], dtype=torch.bfloat16),
            rtol=0,
            atol=0,
        )
        torch.testing.assert_close(
            result["forward_hidden_states"][0],
            torch.tensor([[6]], dtype=torch.float16),
            rtol=0,
            atol=0,
        )

    def test_capture_rejects_negative_request_id_before_put(self) -> None:
        model = CacheStoreForwardModel()
        result = run_scenario(model, "capture_negative_request_id")
        self.assertCanonicalZero(result)
        self.assertEqual(model.capture_flags, [True])
        self.assertIn(
            "request id must be non-negative", result["capture_diagnostics"][0]
        )
        self.assertEqual(result["capture_failure_count"], 1)
        self.assertEqual(_FakeEagleMooncakeStore.instances[0].puts, [])

    def test_capture_duplicate_request_id_only_rejects_each_current_batch(
        self,
    ) -> None:
        model = CacheStoreForwardModel()

        result = run_scenario(
            model,
            "capture_duplicate_request_id",
            3,
        )

        store = _FakeEagleMooncakeStore.instances[0]
        self.assertEqual(model.capture_flags, [True, True, True])
        self.assertEqual(store.put_attempts, 0)
        self.assertEqual(store.puts, [])
        self.assertEqual(result["capture_failure_count"], 3)
        self.assertEqual(result["capture_broken_rejection_count"], 0)
        self.assertCanonicalZero(result)
        self.assertTrue(
            all(
                "duplicate capture request id 501" in error
                for error in result["capture_diagnostics"]
            )
        )
        metrics = result["capture_metrics"]
        self.assertCaptureQps(
            metrics,
            batch=3,
            publish_success=0,
            failure=3,
            prepare_failure=3,
            request_error_failure=3,
            duplicate_request_id=3,
        )
        self.assertCaptureStatus(metrics, enabled=0, broken=0)

    def test_capture_cpu_prepare_failure_only_rejects_each_current_batch(self) -> None:
        submit_frames = []
        _register_identity_allreduce(submit_frames=submit_frames)
        self.addCleanup(clear_comm_ops)
        model = CacheStoreForwardModel()

        result = run_scenario(model, "capture_prepare_failure_tp", 3)

        self.assertEqual(submit_frames, [2, 2, 2])
        self.assertEqual(model.capture_flags, [True, True, True])
        self.assertEqual(result["capture_failure_count"], 3)
        self.assertEqual(result["capture_broken_rejection_count"], 0)
        self.assertCanonicalZero(result)
        self.assertTrue(
            all(
                "capture request metadata" in error
                for error in result["capture_diagnostics"]
            )
        )
        metrics = result["capture_metrics"]
        self.assertCaptureQps(
            metrics,
            batch=3,
            publish_success=0,
            failure=3,
            prepare_failure=3,
            operational_failure=3,
        )
        self.assertCaptureStatus(metrics, enabled=0, broken=0)
        self.assertEqual(_FakeEagleMooncakeStore.instances[0].puts, [])

    def test_capture_non_owner_does_not_materialize_request_metadata(self) -> None:
        _register_identity_allreduce()
        self.addCleanup(clear_comm_ops)
        model = CacheStoreForwardModel()

        result = run_scenario(model, "capture_prepare_failure_non_owner")

        self.assertEqual(model.capture_flags, [True])
        self.assertEqual(result["capture_failure_count"], 0)
        self.assertEqual(result["capture_broken_rejection_count"], 0)
        self.assertEqual(_FakeEagleMooncakeStore.instances, [])

    def test_capture_async_completion_failure_records_diagnostic(self) -> None:
        _FakeEagleMooncakeStore.async_error_after_accept_count = 1
        result = run_scenario(
            CacheStoreForwardModel(),
            "capture_bf16",
        )
        store = _FakeEagleMooncakeStore.instances[0]
        self.assertEqual(store.put_attempts, 1)
        self.assertEqual(len(store.puts), 2)
        self.assertTrue(store.flushed)
        self.assertEqual(store.pending_async_errors, [])
        self.assertCanonicalZero(result)
        self.assertIn("asynchronous publish failure", result["capture_diagnostics"][0])
        self.assertTrue(result["serving_device_healthy"])
        self.assertEqual(result["capture_failure_count"], 1)
        self.assertCaptureQps(
            result["capture_metrics"],
            batch=1,
            publish_success=0,
            failure=1,
            store_failure=1,
            operational_failure=1,
        )
        self.assertCaptureStatus(result["capture_metrics"], enabled=0, broken=0)

    def test_capture_historical_async_error_is_not_attributed_to_current_batch(
        self,
    ) -> None:
        model = CacheStoreForwardModel()
        original_poll = _FakeEagleMooncakeStore.take_async_errors
        poll_count = 0

        def poll_with_history(store):
            nonlocal poll_count
            poll_count += 1
            # First publish: start poll, completion poll. Inject an old error
            # only at the second publish's start, not its completion check.
            if poll_count == 3:
                batch_id = store.put_batch_calls[0]["batch_id"]
                store.pending_async_errors.append(
                    MemoryError(f"historical async failure for {batch_id}")
                )
            return original_poll(store)

        with patch.object(
            _FakeEagleMooncakeStore, "take_async_errors", poll_with_history
        ):
            result = run_scenario(
                model,
                "capture_async_history",
                2,
            )

        store = _FakeEagleMooncakeStore.instances[0]
        self.assertEqual(model.capture_flags, [True, True])
        self.assertEqual(store.put_attempts, 2)
        self.assertEqual(len(store.put_batch_calls), 2)
        self.assertEqual(
            [call["request_ids"] for call in store.put_batch_calls],
            [
                ["rtp_TestActor_501", "rtp_TestActor_502"],
                ["rtp_TestActor_1501", "rtp_TestActor_1502"],
            ],
        )
        self.assertEqual(store.pending_async_errors, [])
        self.assertEqual(result["capture_failure_count"], 1)
        self.assertEqual(result["capture_broken_rejection_count"], 0)
        self.assertCaptureQps(
            result["capture_metrics"],
            batch=2,
            publish_success=2,
            failure=1,
            store_failure=1,
            operational_failure=1,
            bf16_publish=2,
        )
        self.assertCaptureStatus(result["capture_metrics"], enabled=0, broken=0)

    def test_capture_fp8_rejects_malformed_quantizer_result(
        self,
    ) -> None:
        fp8_module = cast(Any, sys.modules["torchspec.utils.fp8"])
        original_quantizer = fp8_module.quantize_aux_hidden_states
        fp8_module.quantize_aux_hidden_states = lambda hidden_states, num_layers: (
            hidden_states,
        )
        self.addCleanup(
            setattr,
            fp8_module,
            "quantize_aux_hidden_states",
            original_quantizer,
        )

        if os.environ.get("TEST_USING_DEVICE") == "ROCM":
            self.assertRocmFp8CaptureRejected("capture_fp8")
            return

        result = run_scenario(
            CacheStoreForwardModel(),
            "capture_fp8",
        )
        self.assertCanonicalZero(result)
        self.assertIn("must return tensor and scale", result["capture_diagnostics"][0])
        self.assertCaptureStatus(result["capture_metrics"], enabled=0, broken=1)

        self.assertEqual(_FakeEagleMooncakeStore.instances[0].puts, [])

    def test_capture_quantizer_execution_failure_recovers_on_next_batch(self) -> None:
        fp8_module = cast(Any, sys.modules["torchspec.utils.fp8"])
        original_quantizer = fp8_module.quantize_aux_hidden_states
        call_count = 0

        def fail_once(hidden_states, num_layers):
            nonlocal call_count
            call_count += 1
            if call_count == 1:
                raise RuntimeError("injected TorchSpec quantizer execution failure")
            return original_quantizer(hidden_states, num_layers)

        fp8_module.quantize_aux_hidden_states = fail_once
        self.addCleanup(
            setattr,
            fp8_module,
            "quantize_aux_hidden_states",
            original_quantizer,
        )

        _register_identity_allreduce()
        self.addCleanup(clear_comm_ops)
        if os.environ.get("TEST_USING_DEVICE") == "ROCM":
            self.assertRocmFp8CaptureRejected("capture_fp8_tp", 2)
            self.assertEqual(call_count, 0)
            return

        result = run_scenario(CacheStoreForwardModel(), "capture_fp8_tp", 2)

        store = _FakeEagleMooncakeStore.instances[0]
        self.assertEqual(store.put_attempts, 1)
        self.assertEqual(
            [put["key"] for put in store.puts],
            ["rtp_TestActor_501", "rtp_TestActor_502"],
        )
        self.assertCanonicalZero(result)
        self.assertIn(
            "injected TorchSpec quantizer execution failure",
            result["capture_diagnostics"][0],
        )
        self.assertEqual(result["capture_failure_count"], 1)
        self.assertEqual(result["capture_broken_rejection_count"], 0)
        self.assertCaptureQps(
            result["capture_metrics"],
            batch=2,
            publish_success=1,
            failure=1,
            quantize_failure=1,
            operational_failure=1,
            fp8_publish=1,
        )
        self.assertCaptureStatus(result["capture_metrics"], enabled=0, broken=0)

    def assertNonOwnerCapture(self, forward_count: int = 1) -> None:
        _register_identity_allreduce()
        self.addCleanup(clear_comm_ops)
        model = CacheStoreForwardModel()
        result = run_scenario(model, "capture_non_owner", forward_count)

        self.assertEqual(model.capture_flags, [True] * forward_count)
        self.assertEqual(_FakeEagleMooncakeStore.instances, [])
        self.assertEqual(result["capture_failure_count"], 0)
        self.assertEqual(result["capture_broken_rejection_count"], 0)
        self.assertEqual(len(result["forward_hidden_states"]), forward_count)
        for index, hidden_states in enumerate(result["forward_hidden_states"]):
            with self.subTest(forward=index):
                self.assertEqual(tuple(hidden_states.shape), (5, 1))
                torch.testing.assert_close(
                    hidden_states, torch.zeros_like(hidden_states)
                )
        metrics = result["capture_metrics"]
        self.assertEqual(set(metrics), _CAPTURE_QPS_METRICS | _CAPTURE_STATUS_METRICS)
        self.assertCaptureMetricTypes(result)
        self.assertCaptureQps(metrics, batch=0, publish_success=0, failure=0)
        self.assertCaptureStatus(metrics, enabled=0, broken=0)

    def test_capture_non_owner_does_not_duplicate_success_metrics(self) -> None:
        for forward_count in (1, 3):
            with self.subTest(forward_count=forward_count):
                self.assertNonOwnerCapture(forward_count)

    def test_capture_fp8_hard_failure_disables_owner_store_before_next_forward(
        self,
    ) -> None:
        submit_frames = []
        _register_identity_allreduce(submit_frames=submit_frames)
        self.addCleanup(clear_comm_ops)
        fp8_module = cast(Any, sys.modules["torchspec.utils.fp8"])
        original_quantizer = fp8_module.quantize_aux_hidden_states
        fp8_module.quantize_aux_hidden_states = lambda hidden_states, num_layers: (
            hidden_states,
        )
        self.addCleanup(
            setattr,
            fp8_module,
            "quantize_aux_hidden_states",
            original_quantizer,
        )
        if os.environ.get("TEST_USING_DEVICE") == "ROCM":
            self.assertRocmFp8CaptureRejected("capture_fp8_tp", 3)
            return

        model = CacheStoreForwardModel()
        result = run_scenario(model, "capture_fp8_tp", 3)

        self.assertEqual(submit_frames, [0, 2, 2])
        self.assertEqual(model.capture_flags, [True, True, True])
        self.assertCanonicalZero(result)
        self.assertEqual(result["capture_failure_count"], 1)
        self.assertEqual(result["capture_broken_rejection_count"], 2)
        metrics = result["capture_metrics"]
        self.assertCaptureQps(
            metrics,
            batch=3,
            publish_success=0,
            failure=1,
            quantize_failure=1,
            hard_contract_failure=1,
            broken_rejection=2,
            admission_rejected=4,
            skip_unavailable=4,
        )
        self.assertCaptureStatus(metrics, enabled=0, broken=1)
        self.assertIn(f"{_CAPTURE_METRIC_PREFIX}publish_latency_us", metrics)
        self.assertIn(f"{_CAPTURE_METRIC_PREFIX}quantize_latency_us", metrics)
        self.assertTrue(
            all(
                "must return tensor and scale" in error
                for error in result["capture_diagnostics"]
            )
        )

    def test_accuracy_check_skips_capture_startup_collectives(self) -> None:
        _register_no_collectives()
        for rank in (0, 1):
            with self.subTest(rank=rank):
                model = CacheStoreForwardModel()
                result = run_scenario(
                    model,
                    "capture_explicit_off",
                    capture_plan={"tp_size": 2, "tp_rank": rank, "layers": [], "accuracy_check_mode": True},
                )
                self.assertFalse(result["capture_plan_enabled"])
                self.assertEqual(model.capture_flags, [False])

    def test_capture_startup_plan_consensus(self) -> None:
        class CapabilityErrorModel(CacheStoreForwardModel):
            @property
            def supports_hidden_state_capture(self):
                raise ValueError("injected capability initialization error")

        cases = (
            "equal",
            "all_empty",
            "root_empty",
            "peer_empty",
            "order",
            "length",
            "dtype",
            "hidden_size",
            "model_dtype",
            "capability",
            "python_error",
            "duplicate",
            "invalid_hidden",
            "invalid_dtype",
        )
        policy = _load_capture_context(return_module=True)
        with patch.dict(sys.modules, {"rtp_llm.models_py.model_desc.hidden_state_capture": policy}):
            for case in cases:
                frames = []
                transcripts = []
                invalid = case in {"capability", "python_error", "duplicate", "invalid_hidden", "invalid_dtype"} or (
                    case == "dtype" and os.environ.get("TEST_USING_DEVICE") == "ROCM"
                )
                accepted = case == "equal"
                for rank in (0, 1):
                    with self.subTest(case=case, rank=rank):
                        _FakeEagleMooncakeStore.instances.clear()
                        layers = [0, 1]
                        if (
                            case == "all_empty"
                            or (case == "root_empty" and rank == 0)
                            or (case == "peer_empty" and rank == 1)
                        ):
                            layers = []
                        if rank == 1 and case in {"order", "length", "duplicate"}:
                            layers = {"order": [1, 0], "length": [0], "duplicate": [0, 0]}[case]
                        plan = {"tp_size": 2, "tp_rank": rank, "layers": layers}
                        if rank == 1:
                            if case == "dtype":
                                plan["fp8"] = True
                            elif case == "hidden_size":
                                plan["hidden_size"] = 2
                            elif case == "model_dtype":
                                plan["model_dtype"] = int(frames[0][4]) + 1
                            elif case == "invalid_hidden":
                                plan["hidden_size"] = 0
                            elif case == "invalid_dtype":
                                plan["dtype"] = 99
                        model = (
                            CapabilityErrorModel() if case == "python_error" and rank == 1 else CacheStoreForwardModel()
                        )
                        if case == "capability" and rank == 1:
                            model.supports_hidden_state_capture = False
                        calls = []
                        replay_index = 0

                        def broadcast(tensors, root, mode):
                            nonlocal replay_index
                            self.assertEqual((root, mode, len(tensors)), (0, 0, 1))
                            tensor = tensors[0]
                            self.assertEqual(tensor.dtype, torch.int64)
                            self.assertEqual(tensor.device.type, "cpu")
                            calls.append(("broadcast", tuple(tensor.shape)))
                            if rank == 0:
                                frames.append(tensor.clone())
                            else:
                                self.assertEqual(tensor.shape, frames[replay_index].shape)
                                tensor.copy_(frames[replay_index])
                                replay_index += 1

                        def allreduce(tensor, op, mode, dest):
                            self.assertEqual(mode, 0)
                            self.assertEqual(tensor.device.type, "cpu")
                            self.assertIsNone(dest)
                            calls.append(("reduce", op, tuple(tensor.shape)))
                            if op == 2:
                                self.assertEqual(tensor.dtype, torch.int64)
                                self.assertEqual(tensor.tolist(), [int(bool(layers)), int(invalid and rank == 1)])
                                return torch.tensor([int(case != "all_empty"), int(invalid)], dtype=torch.int64)
                            self.assertEqual(op, 3)
                            self.assertEqual(tensor.dtype, torch.int32)
                            self.assertEqual(tensor.tolist(), [int(rank == 0 or accepted)])
                            return torch.tensor([int(accepted)], dtype=torch.int32)

                        def unexpected(*args):
                            self.fail("unexpected startup/ordinary-forward collective")

                        register_comm_ops(broadcast, allreduce, unexpected, unexpected, unexpected)
                        result = run_scenario(model, "capture_explicit_off", forward_count=2, capture_plan=plan)
                        self.assertCanonicalZero(result)
                        self.assertEqual(model.capture_flags, [False, False])
                        self.assertEqual(result["capture_plan_enabled"], accepted)
                        self.assertTrue(result["serving_device_healthy"])
                        unavailable = int(rank == 0 and not accepted)
                        self.assertCaptureQps(result["capture_metrics"], batch=0, publish_success=0, failure=0,
                                              admission_rejected=unavailable, skip_unavailable=unavailable)
                        if rank == 0:
                            self.assertEqual(result["capture_plan_admission_ok"], accepted)
                            if not accepted:
                                self.assertTrue(result["capture_plan_admission_error"])
                        if not accepted:
                            self.assertEqual(_FakeEagleMooncakeStore.instances, [])
                        if case != "all_empty" and not accepted:
                            self.assertTrue(all(result["capture_diagnostics"]))
                        if invalid or case == "all_empty":
                            self.assertEqual(calls, [("reduce", 2, (2,))])
                        else:
                            expected = [("reduce", 2, (2,)), ("broadcast", (6,))]
                            if case != "root_empty":
                                expected.append(("broadcast", (2,)))
                            expected.append(("reduce", 3, (1,)))
                            self.assertEqual(calls, expected)
                        transcripts.append(calls)
                self.assertEqual(transcripts[0], transcripts[1])

    def test_capture_startup_local_and_fatal_errors(self) -> None:
        class CapabilityErrorModel(CacheStoreForwardModel):
            @property
            def supports_hidden_state_capture(self):
                raise self.error

        policy = _load_capture_context(return_module=True)
        with patch.dict(sys.modules, {"rtp_llm.models_py.model_desc.hidden_state_capture": policy}):
            for layers in ([], [0, 1]):
                with self.subTest(layers=layers):
                    _register_no_collectives()
                    model = CacheStoreForwardModel()
                    result = run_scenario(
                        model, "capture_explicit_off", capture_plan={"tp_size": 1, "tp_rank": 0, "layers": layers}
                    )
                    self.assertCanonicalZero(result)
                    self.assertEqual(result["capture_plan_enabled"], bool(layers))
                    self.assertEqual(result["capture_plan_admission_ok"], bool(layers))
            for error in (
                ValueError("ordinary capability error"),
                MemoryError("allocator failure"),
                RuntimeError("CUDA error: illegal memory access"),
                RuntimeError("NCCL error"),
            ):
                with self.subTest(error=error):
                    _register_no_collectives()
                    model = CapabilityErrorModel()
                    model.error = error
                    plan = {"tp_size": 1 if isinstance(error, ValueError) else 2, "tp_rank": 0, "layers": [0, 1]}
                    if isinstance(error, ValueError):
                        result = run_scenario(model, "capture_explicit_off", capture_plan=plan)
                        self.assertCanonicalZero(result)
                        self.assertFalse(result["capture_plan_enabled"])
                        self.assertIn(str(error), result["capture_plan_admission_error"])
                    else:
                        with self.assertRaisesRegex(type(error), str(error)):
                            run_scenario(model, "capture_explicit_off", capture_plan=plan)
            for phase in ("handshake", "broadcast", "agreement"):
                with self.subTest(phase=phase):

                    def broadcast(tensors, root, mode):
                        if phase == "broadcast":
                            raise RuntimeError("injected communication failure")

                    def allreduce(tensor, op, mode, dest):
                        if (phase == "handshake" and op == 2) or (phase == "agreement" and op == 3):
                            raise RuntimeError("injected communication failure")
                        return tensor

                    register_comm_ops(broadcast, allreduce, broadcast, broadcast, broadcast)
                    with self.assertRaisesRegex(RuntimeError, "injected communication failure"):
                        run_scenario(
                            CacheStoreForwardModel(),
                            "capture_explicit_off",
                            capture_plan={"tp_size": 2, "tp_rank": 0, "layers": [0, 1]},
                        )

    def test_capture_rejects_duplicate_layer_ids_at_startup(self) -> None:
        model = CacheStoreForwardModel()
        result = run_scenario(model, "capture_duplicate_layer_id")
        self.assertCanonicalZero(result)
        self.assertIn("layer ids must be unique; duplicate id 0", result["capture_diagnostics"][0])
        self.assertFalse(result["capture_plan_enabled"])
        self.assertEqual(result["capture_admissions"], [False])
        self.assertEqual(model.capture_flags, [False])
        self.assertEqual(_FakeEagleMooncakeStore.instances, [])

    def test_capture_rejects_invalid_dtype_at_startup(self) -> None:
        model = CacheStoreForwardModel()
        result = run_scenario(model, "capture_invalid_dtype")
        self.assertCanonicalZero(result)
        self.assertIn("unsupported hidden-state capture dtype: 99", result["capture_diagnostics"][0])
        self.assertFalse(result["capture_plan_enabled"])
        self.assertEqual(result["capture_admissions"], [False])
        self.assertEqual(model.capture_flags, [False])
        self.assertEqual(_FakeEagleMooncakeStore.instances, [])

    def test_capture_rejects_unsupported_python_model_at_startup(self) -> None:
        model = UnsupportedCaptureModel()
        result = run_scenario(model, "capture_bf16")
        self.assertCanonicalZero(result)
        self.assertIn("does not support hidden-state capture", result["capture_diagnostics"][0])
        self.assertFalse(result["capture_plan_enabled"])
        self.assertEqual(result["capture_admissions"], [False])
        self.assertEqual(model.capture_flags, [False])
        self.assertEqual(_FakeEagleMooncakeStore.instances, [])
        self.assertCaptureQps(result["capture_metrics"], batch=0, publish_success=0, failure=0,
                              admission_rejected=2, skip_unavailable=2)

    def test_capture_publisher_drain_does_not_flush_or_close_store(self) -> None:
        _FakeEagleMooncakeStore.events = []
        result = drain_capture_publisher_without_close()
        self.assertTrue(result["drained"])
        self.assertFalse(result["flushed_at_drain"])
        self.assertEqual(len(_FakeEagleMooncakeStore.instances), 1)
        store = _FakeEagleMooncakeStore.instances.pop()
        self.assertTrue(store.flushed)
        self.assertFalse(result["closed_at_drain"])
        self.assertTrue(store.closed)
        self.assertEqual(
            _FakeEagleMooncakeStore.events,
            [
                "store.setup",
                "store.warmup",
                "store.flush",
                "store.take_async_errors",
                "store.close",
            ],
        )
        del store

    def test_capture_publisher_close_acquires_gil_when_called_without_one(self) -> None:
        _FakeEagleMooncakeStore.events = []
        result = close_capture_publisher_without_gil()
        self.assertTrue(result["closed_without_gil"])
        self.assertEqual(len(_FakeEagleMooncakeStore.instances), 1)
        store = _FakeEagleMooncakeStore.instances.pop()
        self.assertTrue(store.flushed)
        self.assertTrue(store.closed)
        self.assertEqual(
            _FakeEagleMooncakeStore.events,
            [
                "store.setup",
                "store.warmup",
                "store.flush",
                "store.take_async_errors",
                "store.close",
            ],
        )
        del store

    def test_capture_publisher_layout(self) -> None:
        child_case_env = "RTP_TEST_CAPTURE_LAYOUT_CHILD"
        cases = {
            "hidden_dim": (
                "MOONCAKE_HIDDEN_DIM",
                "2",
                "hidden_dim 2 does not match RTP hidden size 1",
            ),
            "num_aux_layers": (
                "MOONCAKE_NUM_AUX_LAYERS",
                "3",
                "num_aux_layers 3 does not match RTP capture layer count 2",
            ),
            "valid": (None, None, None),
            "missing_flush": (
                None,
                None,
                "EagleMooncakeStore.flush must be callable",
            ),
            "missing_close": (
                None,
                None,
                "EagleMooncakeStore.close must be callable",
            ),
        }
        child_case = os.environ.get(child_case_env)
        if child_case is None:
            for name in cases:
                with self.subTest(layout=name):
                    env = os.environ.copy()
                    env[child_case_env] = name
                    # Preserve the Bazel launcher's runfiles import paths.
                    env["PYTHONPATH"] = os.pathsep.join(sys.path)
                    child = subprocess.run(
                        [
                            sys.executable,
                            os.path.abspath(__file__),
                            f"{type(self).__name__}.test_capture_publisher_layout",
                        ],
                        env=env,
                        capture_output=True,
                        text=True,
                        timeout=30,
                    )
                    self.assertEqual(
                        child.returncode,
                        0,
                        f"layout={name}, exit={child.returncode} (negative means signal)\n"
                        f"stdout:\n{child.stdout}\nstderr:\n{child.stderr}",
                    )
            return

        # A regression may abort this child, but must not write a large core file.
        import resource

        resource.setrlimit(resource.RLIMIT_CORE, (0, 0))
        env_name, value, expected_error = cases[child_case]
        if env_name is not None:
            os.environ[env_name] = value
        original_from_env = _FakeMooncakeConfig.from_env
        missing_method = {
            "missing_flush": "flush",
            "missing_close": "close",
        }.get(child_case)
        config_refs = []

        def tracked_from_env():
            config = original_from_env()
            config_refs.append(weakref.ref(config))
            return config

        _FakeEagleMooncakeStore.instances.clear()
        _FakeEagleMooncakeStore.events = []
        missing_method_patch = (
            patch.object(_FakeEagleMooncakeStore, missing_method, None)
            if missing_method is not None
            else contextlib.nullcontext()
        )
        with (
            missing_method_patch,
            patch.object(_FakeMooncakeConfig, "from_env", tracked_from_env),
        ):
            result = initialize_capture_publisher()
        self.assertCaptureMetricTypes(result)
        metrics = result["capture_metrics"]
        self.assertEqual(set(metrics), _CAPTURE_QPS_METRICS | _CAPTURE_STATUS_METRICS | _CAPTURE_QUOTA_METRICS)
        failed = int(expected_error is not None)
        self.assertCaptureQps(
            metrics,
            batch=0,
            publish_success=0,
            failure=failed,
            initialization_failure=failed,
            hard_contract_failure=failed,
        )
        self.assertCaptureStatus(
            metrics,
            enabled=0,
            broken=failed,
        )
        if expected_error is not None:
            if missing_method is not None:
                self.assertIn("batch publisher contract", result["error"])
            else:
                self.assertIn(
                    "invalid Mooncake capture layout contract", result["error"]
                )
            self.assertIn(expected_error, result["error"])
            if missing_method is None:
                self.assertEqual(_FakeEagleMooncakeStore.instances, [])
                self.assertEqual(_FakeEagleMooncakeStore.events, [])
            else:
                self.assertEqual(len(_FakeEagleMooncakeStore.instances), 1)
                store = _FakeEagleMooncakeStore.instances.pop()
                self.assertEqual(
                    _FakeEagleMooncakeStore.events,
                    ["store.close"] if missing_method == "flush" else [],
                )
                del store
        else:
            self.assertEqual(result["error"], "")
            self.assertEqual(len(_FakeEagleMooncakeStore.instances), 1)
            store = _FakeEagleMooncakeStore.instances.pop()
            self.assertEqual(store.setup_device, torch.device("cuda:0"))
            self.assertTrue(store.warmed_up)
            self.assertTrue(store.flushed)
            self.assertTrue(store.closed)
            self.assertEqual(store.put_attempts, 0)
            self.assertEqual(
                _FakeEagleMooncakeStore.events,
                [
                    "store.setup",
                    "store.warmup",
                    "store.flush",
                    "store.take_async_errors",
                    "store.close",
                ],
            )
            del store
        gc.collect()
        self.assertEqual(len(config_refs), 1)
        self.assertIsNone(config_refs.pop()())

    def test_capture_publisher_rejects_layer_count_mismatch(
        self,
    ) -> None:
        os.environ["MOONCAKE_NUM_AUX_LAYERS"] = "3"
        result = run_scenario(
            CacheStoreForwardModel(),
            "capture_bf16",
        )
        self.assertCaptureDisabled(result, "does not match RTP capture layer count")
        self.assertEqual(_FakeEagleMooncakeStore.instances, [])

    def test_capture_publisher_rejects_hidden_dim_mismatch(
        self,
    ) -> None:
        os.environ["MOONCAKE_HIDDEN_DIM"] = "2"
        result = run_scenario(
            CacheStoreForwardModel(),
            "capture_bf16",
        )
        self.assertCaptureDisabled(result, "does not match RTP hidden size")
        self.assertEqual(_FakeEagleMooncakeStore.instances, [])

    def test_capture_publisher_requires_batch_and_async_error_contract_methods(
        self,
    ) -> None:
        for method_name in (
            "put_batch", "query_keys", "reclaim_abandoned_capture", "take_async_errors"
        ):
            with self.subTest(method_name=method_name), patch.object(
                _FakeEagleMooncakeStore, method_name, None
            ):
                result = run_scenario(
                    CacheStoreForwardModel(),
                    "capture_bf16",
                )
                self.assertCaptureDisabled(
                    result, f"EagleMooncakeStore.{method_name} must be callable"
                )
                store = _FakeEagleMooncakeStore.instances[-1]
                self.assertEqual(store.puts, [])
                self.assertTrue(store.closed)

    def test_capture_publisher_init_failure_disables_capture_and_closes_partial_store(
        self,
    ) -> None:
        _FakeEagleMooncakeStore.fail_warmup = True
        _FakeEagleMooncakeStore.events = []
        model = CacheStoreForwardModel()

        result = run_scenario(model, "capture_bf16")
        self.assertCaptureDisabled(result, "injected Mooncake warmup failure")

        self.assertEqual(model.capture_flags, [True])
        self.assertEqual(model.forward_calls, 1)
        self.assertEqual(len(_FakeEagleMooncakeStore.instances), 1)
        store = _FakeEagleMooncakeStore.instances[0]
        self.assertEqual(store.put_attempts, 0)
        self.assertTrue(store.closed)
        self.assertFalse(store.flushed)
        self.assertEqual(
            _FakeEagleMooncakeStore.events,
            ["store.setup", "store.warmup", "store.close"],
        )

    @patch.dict(os.environ, {"MOONCAKE_ENABLE_GPU_DIRECT": "1"})
    def test_capture_duplicate_key_text_runtime_error_is_not_request_error(
        self,
    ) -> None:
        def operational_failure(_store, **_kwargs):
            raise RuntimeError("Mooncake key already exists: rtp_TestActor_501")

        with patch.object(_FakeEagleMooncakeStore, "put_batch", operational_failure):
            result = run_scenario(CacheStoreForwardModel(), "capture_bf16")
        self.assertCanonicalZero(result)
        self.assertEqual(result["capture_failure_count"], 1)
        self.assertIn(
            "failed to admit TorchSpec hidden-state batch "
            "[rtp-forward-TestActor-1-requests-501-502]: RuntimeError: "
            "Mooncake key already exists: rtp_TestActor_501",
            result["capture_diagnostics"][0],
        )
        self.assertEqual(_FakeEagleMooncakeStore.instances[-1].puts, [])
        self.assertCaptureQps(
            result["capture_metrics"],
            batch=1,
            publish_success=0,
            failure=1,
            store_failure=1,
            operational_failure=1,
        )
        self.assertCaptureStatus(result["capture_metrics"], enabled=0, broken=0)

    def test_capture_unconsumed_request_id_is_not_readmitted(self) -> None:
        submit_frames = []
        _register_identity_allreduce(submit_frames=submit_frames)
        self.addCleanup(clear_comm_ops)
        model = CacheStoreForwardModel()

        result = run_scenario(
            model,
            "capture_bf16_tp",
            3,
        )

        self.assertEqual(submit_frames, [0, 2, 2])
        store = _FakeEagleMooncakeStore.instances[0]
        self.assertEqual(model.capture_flags, [True, True, True])
        self.assertEqual(result["capture_admissions"], [True, False, False])
        self.assertEqual(
            result["forward_capture_error_codes"],
            [0, ExceptionType.EXECUTION_EXCEPTION, ExceptionType.EXECUTION_EXCEPTION],
        )
        self.assertEqual(
            result["forward_capture_error_messages"],
            ["", "capture publication was not admitted", "capture publication was not admitted"],
        )
        self.assertEqual(
            [put["key"] for put in store.puts],
            ["rtp_TestActor_501", "rtp_TestActor_502"],
        )
        self.assertEqual(store.put_attempts, 1)
        self.assertEqual(result["capture_failure_count"], 0)
        self.assertEqual(result["capture_broken_rejection_count"], 0)
        self.assertEqual(result["capture_diagnostics"], ["", "", ""])
        self.assertFalse(store.closed)
        self.assertEqual(len(store.keys), 8)
        self.assertCanonicalZero(result)
        self.assertCaptureQps(
            result["capture_metrics"], batch=3, publish_success=1, failure=0, bf16_publish=1,
            admission_rejected=4, skip_config=4, duplicate_request_id=4
        )
        self.assertCaptureStatus(result["capture_metrics"], enabled=0, broken=0)

    def test_capture_rejects_packed_layout(self) -> None:
        model = MalformedCaptureLayoutModel()
        result = run_scenario(
            model,
            "capture_bf16",
            3,
        )

        self.assertEqual(model.capture_flags, [True, True, True])
        self.assertCanonicalZero(result)
        self.assertTrue(
            all(
                "captured hidden-state width 2 must match expected packed width 3"
                in error
                for error in result["capture_diagnostics"]
            )
        )
        self.assertEqual(result["capture_failure_count"], 1)
        self.assertEqual(result["capture_broken_rejection_count"], 2)
        metrics = result["capture_metrics"]
        self.assertCaptureQps(
            metrics,
            batch=3,
            publish_success=0,
            failure=1,
            layout_failure=1,
            hard_contract_failure=1,
            broken_rejection=2,
            admission_rejected=4,
            skip_unavailable=4,
        )
        self.assertCaptureStatus(metrics, enabled=0, broken=1)
        self.assertEqual(_FakeEagleMooncakeStore.instances[0].puts, [])

    @patch.dict(os.environ, {"MOONCAKE_ENABLE_GPU_DIRECT": "1"})
    def test_non_capture_forward_preserves_operational_capture_diagnostic(self) -> None:
        _FakeEagleMooncakeStore.fail_put_count = 1
        model = CacheStoreForwardModel()

        result = run_scenario(
            model,
            "capture_bf16",
            2,
            capture_only_first_forward=True,
        )

        store = _FakeEagleMooncakeStore.instances[0]
        self.assertEqual(model.capture_flags, [True, False])
        self.assertEqual(result["forward_capture_error_codes"], [0, 0])
        self.assertEqual(result["forward_capture_error_messages"], ["", ""])
        self.assertEqual(store.put_attempts, 1)
        self.assertEqual(store.puts, [])
        self.assertCanonicalZero(result)
        self.assertEqual(len(result["capture_diagnostics"]), 2)
        self.assertIn(
            "injected Mooncake put_batch admission failure",
            result["capture_diagnostics"][0],
        )
        self.assertEqual(
            result["capture_diagnostics"][0], result["capture_diagnostics"][1]
        )
        self.assertEqual(result["capture_failure_count"], 1)
        self.assertEqual(result["capture_broken_rejection_count"], 0)
        self.assertCaptureQps(
            result["capture_metrics"],
            batch=1,
            publish_success=0,
            failure=1,
            store_failure=1,
            operational_failure=1,
        )
        self.assertCaptureStatus(result["capture_metrics"], enabled=0, broken=0)

    @patch.dict(os.environ, {"MOONCAKE_ENABLE_GPU_DIRECT": "1"})
    def test_capture_store_admission_failure_recovers_on_next_batch(
        self,
    ) -> None:
        submit_frames = []
        _register_identity_allreduce(submit_frames=submit_frames)
        self.addCleanup(clear_comm_ops)
        _FakeEagleMooncakeStore.fail_put_count = 1
        model = CacheStoreForwardModel()
        result = run_scenario(model, "capture_bf16_tp", 2)

        self.assertEqual(submit_frames, [0, 0])
        store = _FakeEagleMooncakeStore.instances[0]
        self.assertEqual(model.capture_flags, [True, True])
        # Worker PUT failures are asynchronous diagnostics, not this forward's outcome.
        self.assertEqual(result["forward_capture_error_codes"], [0, 0])
        self.assertEqual(result["forward_capture_error_messages"], ["", ""])
        for hidden_states in result["forward_hidden_states"]:
            torch.testing.assert_close(hidden_states, torch.zeros_like(hidden_states))
        self.assertEqual(store.put_attempts, 2)
        self.assertEqual(
            [put["key"] for put in store.puts],
            ["rtp_TestActor_501", "rtp_TestActor_502"],
        )
        self.assertCanonicalZero(result)
        self.assertIn(
            "injected Mooncake put_batch admission failure",
            result["capture_diagnostics"][0],
        )
        self.assertEqual(result["capture_failure_count"], 1)
        self.assertEqual(result["capture_broken_rejection_count"], 0)
        metrics = result["capture_metrics"]
        self.assertCaptureQps(
            metrics,
            batch=2,
            publish_success=1,
            failure=1,
            store_failure=1,
            operational_failure=1,
            bf16_publish=1,
        )
        self.assertCaptureStatus(metrics, enabled=0, broken=0)
        self.assertIn(f"{_CAPTURE_METRIC_PREFIX}publish_latency_us", metrics)
        self.assertIn(f"{_CAPTURE_METRIC_PREFIX}store_put_latency_us", metrics)


class TorchSpecContractGuardTest(unittest.TestCase):
    def test_fake_torchspec_is_not_installed_during_module_import(self) -> None:
        mooncake = sys.modules.get("torchspec.transfer.mooncake")
        if mooncake is not None:
            self.assertIsNot(
                getattr(mooncake, "EagleMooncakeStore", None),
                _FakeEagleMooncakeStore,
            )
            self.assertIsNot(
                getattr(mooncake, "MooncakeConfig", None),
                _FakeMooncakeConfig,
            )

        fp8 = sys.modules.get("torchspec.utils.fp8")
        if fp8 is not None:
            self.assertIsNot(
                getattr(fp8, "quantize_aux_hidden_states", None),
                _fake_quantize_aux_hidden_states,
            )

    def assertSignatureBinds(self, callable_object, *args, **kwargs) -> None:
        try:
            signature = inspect.signature(callable_object)
        except (TypeError, ValueError) as error:
            self.fail(f"TorchSpec callable has no inspectable signature: {error}")
        try:
            signature.bind(*args, **kwargs)
        except TypeError as error:
            self.fail(f"TorchSpec callable does not satisfy capture contract: {error}")

    def test_real_torchspec_contract_when_available(self) -> None:
        try:
            importlib.import_module("torchspec")
        except ModuleNotFoundError as error:
            if error.name == "torchspec":
                self.skipTest("real TorchSpec is not installed")
            self.skipTest(f"real TorchSpec dependency is unavailable: {error.name}")
        except ImportError as error:
            self.skipTest(
                f"real TorchSpec cannot be loaded in this environment: {error}"
            )

        try:
            mooncake = importlib.import_module("torchspec.transfer.mooncake")
            fp8 = importlib.import_module("torchspec.utils.fp8")
        except ModuleNotFoundError as error:
            if error.name is not None and error.name.startswith("torchspec"):
                self.fail(f"real TorchSpec is missing capture API module: {error.name}")
            self.skipTest(f"real TorchSpec dependency is unavailable: {error.name}")
        except ImportError as error:
            self.skipTest(f"real TorchSpec capture API cannot be loaded: {error}")

        try:
            mooncake_config = cast(Any, getattr(mooncake, "MooncakeConfig", None))
            mooncake_store = cast(Any, getattr(mooncake, "EagleMooncakeStore", None))
            quantize_aux_hidden_states = cast(
                Any, getattr(fp8, "quantize_aux_hidden_states", None)
            )
            store_module = importlib.import_module("torchspec.transfer.mooncake.store")
            buffers_module = importlib.import_module("torchspec.transfer.mooncake.buffers")
        except (ImportError, OSError) as error:
            self.skipTest(f"real TorchSpec capture API is unavailable: {error}")
        self.assertIsNotNone(mooncake_config)
        self.assertIsNotNone(mooncake_store)
        self.assertCountEqual(mooncake_store.TENSOR_SUFFIXES, _FakeEagleMooncakeStore.TENSOR_SUFFIXES)
        self.assertTrue(callable(quantize_aux_hidden_states))

        for method_name in (
            "put_batch", "query_keys", "reclaim_abandoned_capture", "take_async_errors", "flush", "close"
        ):
            self.assertTrue(
                callable(getattr(mooncake_store, method_name, None)), method_name
            )

        self.assertSignatureBinds(mooncake_config.from_env)
        layout_env = {
            "MOONCAKE_HIDDEN_DIM": "2048",
            "MOONCAKE_MAX_SEQ_LEN": "16384",
            "MOONCAKE_GET_BATCH_SIZE": "8",
            "MOONCAKE_GPU_BUFFER_SIZE": "268435456",
            "MOONCAKE_NUM_AUX_LAYERS": "2",
            _MOONCAKE_STORE_KEY_NAMESPACE_ENV: "ContractActor1",
        }
        previous_layout_env = {
            env_name: os.environ.get(env_name) for env_name in layout_env
        }
        try:
            os.environ.update(layout_env)
            restored_config = mooncake_config.from_env()
        finally:
            _restore_mooncake_layout_env(previous_layout_env)
        for env_name, attr_name in _MOONCAKE_RTP_LAYOUT_ENV.items():
            self.assertEqual(
                getattr(restored_config, attr_name), int(layout_env[env_name])
            )
        self.assertEqual(restored_config.store_key_namespace, "ContractActor1")
        self.assertSignatureBinds(restored_config.make_store_key, 501)
        self.assertEqual(
            restored_config.make_store_key(501),
            "rtp_ContractActor1_501",
        )

        self.assertSignatureBinds(mooncake_store, object())
        self.assertSignatureBinds(mooncake_store.setup, object(), torch.device("cuda"))
        for method_name in ("warmup_rdma", "take_async_errors", "flush", "close"):
            method = getattr(mooncake_store, method_name, None)
            self.assertTrue(callable(method), method_name)
            self.assertSignatureBinds(method, object())
        self.assertSignatureBinds(
            mooncake_store.put_batch,
            object(),
            batch_id="contract-batch",
            request_ids=["contract-key"],
            hidden_states=[object()],
            input_ids=[object()],
            last_hidden_states=[object()],
            hidden_states_scale=None,
            capture_metadata={"capture_layer_ids": [7, 2]},
        )
        self.assertSignatureBinds(mooncake_store.query_keys, object(), ["contract-key_hs"])
        self.assertSignatureBinds(mooncake_store.abandon_capture, object(), "contract-key")
        self.assertSignatureBinds(mooncake_store.reclaim_abandoned_capture, object(), "contract-key")
        self.assertSignatureBinds(quantize_aux_hidden_states, object(), 2)
        self.assertEqual(mooncake_store.__module__, "torchspec.transfer.mooncake.eagle_store")
        self.assertIsNot(mooncake_store, _FakeEagleMooncakeStore)
        self.assertIsNot(mooncake_config, _FakeMooncakeConfig)

        case = self

        class NativeStore:
            # Record actual host bytes only; staging, leases and chunks are real.
            def __init__(self):
                self.registrations = []
                self.puts = []
                self.removals = []
                self.setup_calls = []
                self.close_calls = 0
                self.keys = set()

            def is_exist(self, key):
                return int(key in self.keys)

            def batch_is_exist(self, keys):
                return [self.is_exist(key) for key in keys]

            def setup(self, **kwargs):
                self.setup_calls.append(kwargs)
                return 0

            def register_buffer(self, ptr, size):
                self.registrations.append((ptr, size))
                return 0

            def batch_put_from(self, keys, ptrs, sizes):
                case.assertEqual(self.close_calls, 0)
                payloads = []
                for key, ptr, size in zip(keys, ptrs, sizes, strict=True):
                    case.assertTrue(
                        any(
                            base <= ptr and ptr + size <= base + capacity
                            for base, capacity in self.registrations
                        ),
                        "native PUT must read a registered host buffer",
                    )
                    # Warmup reads uninitialized bytes; only inspect real payloads.
                    payloads.append(
                        b"" if key.startswith("_warmup_") else ctypes.string_at(ptr, size)
                    )
                self.puts.append({
                    "keys": list(keys), "ptrs": list(ptrs), "sizes": list(sizes),
                    "payloads": payloads, "thread": threading.current_thread().name,
                })
                self.keys.update(keys)
                return [0] * len(keys)

            def batch_remove(self, keys, *, force=False):
                "Native removal supports force=True."
                case.assertTrue(force)
                self.removals.append(list(keys))
                self.keys.difference_update(keys)
                return [0] * len(keys)

            def close(self):
                self.close_calls += 1

        native = NativeStore()
        initialized = {}

        def inspect_setup(store):
            self.assertIs(type(store), mooncake_store)
            self.assertEqual(store.config.async_put_pool_size, 1)
            self.assertFalse(store._gpu_direct_available)
            self.assertIsNone(store._gpu_send_buffer)
            pool = store._host_buffer_pool
            self.assertIs(type(pool), buffers_module.HostBufferPool)
            self.assertEqual(pool.pool_size, 1)
            self.assertEqual(len(pool._buffers), 1)
            buffer = pool._buffers[0]
            self.assertIs(type(buffer), buffers_module.HostBuffer)
            self.assertTrue(buffer._tensor.is_pinned())
            manager = store._async_put_manager
            self.assertIs(type(manager), buffers_module.AsyncPutManager)
            self.assertIs(manager._store, native)
            self.assertIsInstance(store._copy_stream, torch.cuda.Stream)
            self.assertEqual(store._copy_stream.device.type, "cuda")
            initialized.update(
                store=store, pool=pool, buffer=buffer, manager=manager,
                ptr=buffer.ptr, size=buffer.size,
            )

        real_layout = {
            _MOONCAKE_STORE_KEY_NAMESPACE_ENV: "ContractActor1",
            "MOONCAKE_HIDDEN_DIM": "1",
            "MOONCAKE_NUM_AUX_LAYERS": "2",
            "MOONCAKE_MAX_SEQ_LEN": "8",
            "MOONCAKE_GET_BATCH_SIZE": "1",
            "MOONCAKE_ASYNC_PUT_POOL_SIZE": "1",
            "MOONCAKE_ENABLE_GPU_DIRECT": "0",
            "MOONCAKE_ENABLE_HARD_PIN": "0",
            "MOONCAKE_PROTOCOL": "tcp",
            "MOONCAKE_HOST_BUFFER_SIZE": "4096",
            "MOONCAKE_GPU_BUFFER_SIZE": "4096",
            "MOONCAKE_GLOBAL_SEGMENT_SIZE": "8192",
            "MOONCAKE_LOCAL_BUFFER_SIZE": "8192",
            "MOONCAKE_TRANSFER_PERF_LOG_INTERVAL_SECONDS": "0",
        }
        model = PositionAwareCaptureModel()
        _register_no_collectives()
        self.addCleanup(clear_comm_ops)
        with patch.dict(os.environ, real_layout), patch.object(
            store_module, "MooncakeDistributedStore", return_value=native
        ):
            result = run_scenario(
                model, "capture_bf16", after_capture_setup=inspect_setup
            )

        self.assertEqual(model.capture_flags, [True])
        self.assertEqual(result["capture_admissions"], [True])
        self.assertEqual(result["capture_diagnostics"], [""])
        self.assertFalse(result["capture_fatal"])
        self.assertEqual(result["capture_failure_count"], 0)
        self.assertEqual(result["capture_broken_rejection_count"], 0)
        self.assertEqual(result["capture_inflight_samples"], 0)
        self.assertEqual(result["capture_inflight_bytes"], 0)
        torch.testing.assert_close(
            result["forward_hidden_states"][0],
            torch.arange(1, 6, dtype=torch.float16).reshape(-1, 1),
            rtol=0, atol=0,
        )
        self.assertEqual(native.registrations, [(initialized["ptr"], initialized["size"])])
        self.assertEqual(len(native.setup_calls), 1)
        self.assertEqual(native.setup_calls[0]["protocol"], "tcp")
        warmup = [put for put in native.puts if put["keys"][0].startswith("_warmup_")]
        self.assertEqual(len(warmup), 1)
        self.assertEqual(native.removals, [warmup[0]["keys"]])
        puts = [put for put in native.puts if not put["keys"][0].startswith("_warmup_")]
        keys = [f"rtp_ContractActor1_{request_id}" for request_id in (501, 502)]
        self.assertEqual(
            [put["keys"] for put in puts],
            [[keys[0] + suffix for suffix in ("_hs", "_ids", "_lhs")],
             [keys[0] + "_ready"],
             [keys[1] + suffix for suffix in ("_hs", "_ids", "_lhs")],
             [keys[1] + "_ready"]],
        )
        # The same single lease serves both chunks, but each byte snapshot must
        # retain the correct source slice after the next chunk overwrites it.
        self.assertEqual([put["ptrs"][0] for put in puts], [initialized["ptr"]] * 4)
        self.assertTrue(all(put["thread"].startswith("async_put") for put in puts))
        for data, ready, ids in (
            (puts[0], puts[1], torch.tensor([1, 2], dtype=torch.int64)),
            (puts[2], puts[3], torch.tensor([3, 4, 5], dtype=torch.int64)),
        ):
            values = ids.to(torch.bfloat16).reshape(-1, 1)
            expected = [torch.cat((values + 10, values + 20), dim=-1), ids, values + 30]
            self.assertEqual(data["sizes"], [tensor.nbytes for tensor in expected])
            self.assertEqual(
                data["payloads"],
                [bytes(tensor.contiguous().view(torch.uint8).reshape(-1).tolist())
                 for tensor in expected],
            )
            self.assertEqual(ready["sizes"], [len(ready["payloads"][0])])
            self.assertIsInstance(json.loads(ready["payloads"][0]), dict)

        metrics = result["capture_metrics"]
        self.assertEqual(
            set(metrics),
            _CAPTURE_QPS_METRICS | _CAPTURE_STATUS_METRICS | _CAPTURE_PUBLISH_METRICS | _CAPTURE_QUOTA_METRICS,
        )
        self.assertEqual(
            result["capture_metric_types"],
            {name: (_KMONITOR_QPS if name in _CAPTURE_QPS_METRICS else
                    _KMONITOR_STATUS if name in _CAPTURE_STATUS_METRICS else
                    _KMONITOR_GAUGE) for name in metrics},
        )
        expected_qps = dict.fromkeys(_CAPTURE_QPS_METRICS, 0)
        for name in ("batch", "publish_success", "bf16_publish"):
            expected_qps[f"{_CAPTURE_METRIC_PREFIX}{name}_qps"] = 1
        self.assertEqual({name: metrics[name] for name in _CAPTURE_QPS_METRICS}, expected_qps)
        for name, expected in {
            "enabled": 0, "broken": 0,
            "publish_request_count": 2, "publish_token_count": 5,
            "publish_payload_bytes": 70, "publish_input_ids_bytes": 40,
            "publish_aux_hidden_bytes": 20, "publish_last_hidden_bytes": 10,
            "publish_scale_bytes": 0,
        }.items():
            self.assertEqual(metrics[_CAPTURE_METRIC_PREFIX + name], expected, name)
        for name in ("publish_latency_us", "store_put_latency_us"):
            self.assertGreaterEqual(metrics[_CAPTURE_METRIC_PREFIX + name], 0)
        self.assertEqual(native.close_calls, 0)
        self.assertIs(initialized["store"]._host_buffer_pool, initialized["pool"])
        self.assertFalse(initialized["manager"]._closed)
        # The publisher has relinquished ownership to quarantine, not destroyed
        # buffers underneath an unconsumed source. Simulate consumer removal
        # before explicitly closing the retained test store.
        native.batch_remove(sorted(native.keys), force=True)
        initialized["store"].close()
        self.assertEqual(native.close_calls, 1)
        self.assertIsNone(initialized["store"]._host_buffer_pool)
        self.assertIsNone(initialized["store"]._async_put_manager)
        self.assertEqual(initialized["pool"]._buffers, [])
        self.assertEqual(initialized["buffer"].ptr, 0)
        self.assertTrue(initialized["manager"]._closed)
        self.assertFalse(initialized["manager"]._in_flight)
        self.assertFalse(initialized["manager"]._accepted)


class Handler:
    def __init__(self):
        self.calls = 0

    def extend_forward_args(self):
        return ["selected_hidden_states"]

    def extend_forward(self, selected_hidden_states):
        self.calls += 1
        return selected_hidden_states * 2


def rms_norm(hidden):
    fp32 = hidden.float()
    return (fp32 * torch.rsqrt(fp32.square().mean(-1, keepdim=True) + 1e-5)).to(
        hidden.dtype
    )


class ForwardModel(SuccessfulGenerationPrefillCaptureModel):
    def hidden(self, inputs):
        return (
            inputs.input_ids.unsqueeze(1) * 4 + torch.arange(4, device="cuda") + 1
        ).to(torch.bfloat16)

    def forward(self, inputs, fmha_impl=None):
        return PyModelOutputs(rms_norm(self.hidden(inputs)))


@unittest.skipUnless(torch.cuda.is_available(), "requires CUDA/ROCm")
class CustomOutputPostLayersTest(unittest.TestCase):
    def run_case(self, handler, indexes=(1, 4), python_norm=True):
        return _extension.run_post_layers(
            ForwardModel(),
            handler,
            torch.tensor(indexes, dtype=torch.int64).pin_memory(),
            python_norm,
        )

    def test_selected_rows_preserve_logits_across_eager_and_graph(self):
        hidden = torch.arange(24, device="cuda", dtype=torch.bfloat16).reshape(6, 4) + 1
        for python_norm in (False, True):
            baseline = self.run_case(None, python_norm=python_norm)
            self.assertIsNone(baseline["custom_output"])
            for indexes in ((1, 4), (4,), ()):
                with self.subTest(python_norm=python_norm, indexes=indexes):
                    handler = Handler()
                    result = self.run_case(handler, indexes, python_norm)
                    self.assertEqual(result["custom_output_error"], "")
                    steps = (("", hidden, list(indexes)),)
                    if python_norm:
                        self.assertEqual(result["graph_status"], "replayed")
                        self.assertIsNone(result["decode_custom_output"])
                        steps += (("next_", hidden.flip(0), [i - 1 for i in indexes]),)
                    self.assertEqual(handler.calls, len(steps) if indexes else 0)
                    for prefix, rows, selected in steps:
                        torch.testing.assert_close(
                            result[prefix + "logits"],
                            baseline[prefix + "logits"],
                            rtol=0,
                            atol=0,
                        )
                        output = result[prefix + "custom_output"]
                        if not selected:
                            self.assertIsNone(output)
                        else:
                            torch.testing.assert_close(
                                output, rms_norm(rows)[selected] * 2
                            )

    def test_invalid_contracts_fail_explicitly(self):
        def fail(rows):
            raise ValueError("head failure")

        for forward, message in (
            (lambda rows: rows.double(), "output dtype"),
            (lambda rows: rows.cpu(), "input CUDA device"),
            (lambda rows: rows[:1], "one row per selected context sequence"),
            (lambda rows: rows[:0], "nonempty"),
            (fail, "head failure"),
        ):
            with self.subTest(message=message):
                handler = Handler()
                handler.extend_forward = lambda selected_hidden_states: forward(
                    selected_hidden_states
                )
                result = self.run_case(handler, python_norm=False)
                self.assertIsNone(result["custom_output"])
                self.assertIn(message, result["custom_output_error"])


@unittest.skipUnless(torch.cuda.is_available(), "requires CUDA/ROCm")
class NativeLogitsBoundaryTest(unittest.TestCase):
    def test_python_logits_hook_is_not_consulted(self):
        for compact, graph in ((False, False), (False, True), (True, False)):
            with self.subTest(compact=compact, graph=graph):

                def run(model):
                    return _extension.run_post_layers(
                        model,
                        None,
                        torch.empty(0, dtype=torch.int64).pin_memory(),
                        graph,
                        compact,
                    )

                baseline = run(ForwardModel())
                model = ForwardModel()

                def forbidden(_):
                    raise AssertionError(
                        "native post-layers delegated LM head ownership to Python"
                    )

                model.compute_logits = forbidden
                result = run(model)
                torch.testing.assert_close(
                    result["logits"], baseline["logits"], rtol=0, atol=0
                )
                if graph:
                    torch.testing.assert_close(
                        result["next_logits"], baseline["next_logits"], rtol=0, atol=0
                    )


if __name__ == "__main__":
    unittest.main()
