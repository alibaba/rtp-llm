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


def _install_single_rank_rocm_graph_capture_stub() -> None:
    """Keep this focused integration test independent of the production ops SO.

    The C++ test extension registers the same pybind types as the production
    extension, so importing ``rocm_rccl`` here would register those types a
    second time. Single-rank capture has no RCCL work; these are the exact
    no-op hooks used by the device shim for that case.
    """

    models_py = types.ModuleType("rtp_llm.models_py")
    models_py.__path__ = []
    distributed = types.ModuleType("rtp_llm.models_py.distributed")
    distributed.__path__ = []
    rocm_rccl = types.ModuleType("rtp_llm.models_py.distributed.rocm_rccl")
    rocm_rccl.set_graph_capture_nccl_comm = lambda *_args: None
    rocm_rccl.enter_graph_capture_mode = lambda *_args: None
    rocm_rccl.exit_graph_capture_mode = lambda: None
    rocm_rccl.finish_hipgraph_capture_session = lambda: None

    models_py.distributed = distributed
    distributed.rocm_rccl = rocm_rccl
    sys.modules["rtp_llm.models_py"] = models_py
    sys.modules["rtp_llm.models_py.distributed"] = distributed
    sys.modules["rtp_llm.models_py.distributed.rocm_rccl"] = rocm_rccl


if os.environ.get("TEST_USING_DEVICE") == "ROCM":
    _install_single_rank_rocm_graph_capture_stub()

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
initialize_capture_publisher_with_core_dump = (
    _extension.initialize_capture_publisher_with_core_dump
)
register_comm_ops = _extension.register_comm_ops
run_tp_sync_model_inputs_scenario = _extension.run_tp_sync_model_inputs_scenario

_KMONITOR_GAUGE = 0
_KMONITOR_QPS = 3
_KMONITOR_STATUS = 5
_CAPTURE_METRIC_PREFIX = "rtp_llm_hidden_state_capture_"
_CAPTURE_QPS_METRICS = {
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
        config.async_put_pool_size = int(
            os.environ.get("MOONCAKE_ASYNC_PUT_POOL_SIZE", "1")
        )
        config.enable_gpu_direct = (
            os.environ.get("MOONCAKE_ENABLE_GPU_DIRECT", "0") == "1"
        )
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
                    for suffix in ("_hs", "_ids", "_lhs", "_ready", "_hss")
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
        for suffix in ("_hs", "_hss", "_tgt", "_ids", "_lhs", "_ready"):
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


def _register_no_collectives(allreduce=None) -> None:
    def unexpected_collective(*args):
        raise AssertionError("hidden-state capture issued an unexpected collective")

    register_comm_ops(
        unexpected_collective,
        unexpected_collective if allreduce is None else allreduce,
        unexpected_collective,
    )


def _register_layout_allreduce(allreduce) -> None:
    # Existing layout tests inject only the post-forward status. Plan consensus
    # has separate paired-rank tests below and must succeed before these probes.
    def reduce(tensor, op, mode, dest):
        if tensor.dtype == torch.int64:
            assert (op, mode, dest) == (3, 0, None)
            return tensor
        return allreduce(tensor, op, mode, dest)

    _register_no_collectives(reduce)


def _register_identity_allreduce() -> None:
    def identity_allreduce(tensor, op, mode, dest):
        del op, mode, dest
        return tensor

    _register_no_collectives(identity_allreduce)


def _run_tp_sync_cache_rows_pending_rank(tp_rank: int, base_path: str, ready) -> None:
    check = unittest.TestCase()

    def broadcast(tensors, root, mode):
        check.assertEqual((root, mode), (0, 0))
        for tensor in tensors:
            # CPU control data must use the real UDS broadcaster. GPU payload
            # transport is outside this scheduling probe; initialize its receive
            # buffer without copying any device data back to the host.
            check.assertTrue(tensor.is_cuda)
            if tp_rank != 0:
                tensor.zero_()

    def unexpected(*_args):
        check.fail("cache input synchronization must only broadcast")

    event = None
    register_comm_ops(broadcast, unexpected, unexpected)
    try:
        init_cpu_tp_broadcaster(tp_rank, 2, base_path)
        blocks = torch.arange(12, device="cuda", dtype=torch.int32).reshape(2, 2, 3)
        types = torch.arange(2, device="cuda", dtype=torch.int32)
        # Warm allocation and transport before queuing the probe. Both ranks
        # must be ready, so process/CUDA startup cannot consume the sleep window.
        _extension.run_tp_sync_cache_rows(tp_rank, ["a", "z"], blocks, blocks, types)
        torch.cuda.synchronize()
        ready.wait(timeout=30)
        if tp_rank == 0:
            event = torch.cuda.Event()
            torch.cuda._sleep(2_000_000_000)
            event.record()
            check.assertFalse(
                event.query(),
                "pending-work probe completed before TP sync",
            )
        _extension.run_tp_sync_cache_rows(tp_rank, ["a", "z"], blocks, blocks, types)
        if event is not None:
            check.assertFalse(
                event.query(),
                "canonical cache metadata waited for queued GPU work",
            )
    finally:
        try:
            if event is not None:
                event.synchronize()
        finally:
            try:
                destroy_cpu_tp_broadcaster()
            finally:
                clear_comm_ops()


class TpSyncCacheRowsTest(unittest.TestCase):
    def _round_trip(self, tags, blocks, kernel, group_types):
        # Producers supply canonical rows. Replay only transport; production
        # tpSyncModelInputs must preserve the tensors without any row remap.
        self.assertEqual(tags, sorted(tags))
        frames = {"cpu": [], "cuda": []}
        positions = {"cpu": 0, "cuda": 0}
        receiving = False

        def broadcast(tensors, root, _mode):
            self.assertEqual(root, 0)
            for tensor in tensors:
                kind = "cuda" if tensor.is_cuda else "cpu"
                if receiving:
                    tensor.copy_(frames[kind][positions[kind]])
                    positions[kind] += 1
                else:
                    frames[kind].append(tensor.clone())

        def unexpected(*_args):
            self.fail("cache input synchronization must only broadcast")

        _extension.register_comm_ops(broadcast, unexpected, unexpected)
        try:
            with torch.profiler.profile(
                activities=[torch.profiler.ProfilerActivity.CPU]
            ) as profile:
                root_result = _extension.run_tp_sync_cache_rows(
                    0, tags, blocks, kernel, group_types
                )
            self.assertNotIn(
                "aten::index_select", [event.name for event in profile.events()]
            )
            self.assertEqual(root_result["blocks"].data_ptr(), blocks.data_ptr())
            self.assertEqual(root_result["kernel"].data_ptr(), kernel.data_ptr())
            self.assertEqual(root_result["types"].data_ptr(), group_types.data_ptr())
            receiving = True
            positions = {"cpu": 0, "cuda": 0}
            result = _extension.run_tp_sync_cache_rows(
                1, tags, blocks, kernel, group_types
            )
            self.assertEqual(result["tags"], tags)
            for key, source in (
                ("blocks", blocks),
                ("kernel", kernel),
                ("types", group_types),
            ):
                if source.numel() == 0:
                    self.assertTrue(result[key] is None or result[key].numel() == 0)
                    continue
                self.assertEqual(result[key].device, source.device)
                torch.testing.assert_close(
                    result[key].cpu(), source.cpu(), rtol=0, atol=0
                )
                if source.is_pinned():
                    self.assertTrue(result[key].is_pinned())
            self.assertEqual(
                positions, {key: len(value) for key, value in frames.items()}
            )
        finally:
            _extension.clear_comm_ops()

    def test_cuda_group_order_and_noncontiguous_tables(self):
        for groups, noncontiguous in ((2, False), (2, True), (65, False)):
            with self.subTest(groups=groups, noncontiguous=noncontiguous):
                tags = [f"g{i:03d}" for i in range(groups)]
                blocks = torch.arange(groups * 2 * 3, device="cuda", dtype=torch.int32)
                blocks = blocks.reshape(groups, 2, 3)
                kernel = blocks + 1000
                if noncontiguous:
                    blocks = blocks.transpose(1, 2)
                    kernel = kernel.transpose(1, 2)
                types = torch.arange(groups, device="cuda", dtype=torch.int32)
                self._round_trip(tags, blocks, kernel, types)

    def test_identity_empty_and_pinned_cpu_tables(self):
        for device, empty in (("cuda", False), ("cuda", True), ("cpu", False)):
            with self.subTest(device=device, empty=empty):
                tags = ["a", "z"]
                width = 0 if empty else 3
                blocks = torch.arange(2 * 2 * width, device=device, dtype=torch.int32)
                blocks = blocks.reshape(2, 2, width)
                kernel = blocks + 1000
                types = torch.arange(2, device=device, dtype=torch.int32)
                if device == "cpu":
                    blocks = blocks.transpose(1, 2)
                    kernel = kernel.transpose(1, 2)
                    blocks, kernel, types = (
                        value.pin_memory() for value in (blocks, kernel, types)
                    )
                self._round_trip(tags, blocks, kernel, types)

    def test_identity_still_validates_group_shapes(self):
        def unexpected(*_args):
            self.fail("invalid cache group metadata must fail before broadcasting")

        _extension.register_comm_ops(unexpected, unexpected, unexpected)
        try:
            types = torch.arange(2, device="cuda", dtype=torch.int32)
            for shape, tags in (
                ((3, 2, 1), ["a", "z"]),
                ((2, 3), ["a", "z"]),
                ((2, 2, 1), ["a", "a"]),
                ((2, 2, 1), ["z", "a"]),
            ):
                with self.subTest(shape=shape, tags=tags):
                    blocks = torch.zeros(shape, device="cuda", dtype=torch.int32)
                    with self.assertRaises(RuntimeError):
                        _extension.run_tp_sync_cache_rows(
                            0, tags, blocks, blocks, types
                        )
        finally:
            _extension.clear_comm_ops()

    @unittest.skipIf(torch.version.hip is not None, "CUDA pending-work probe")
    def test_canonical_cuda_metadata_does_not_wait_for_queued_gpu_work(self):
        # Spawn avoids inheriting CUDA state. Real CPU TP transport keeps the
        # fallback's deliberate device synchronization outside this probe.
        context = multiprocessing.get_context("spawn")
        ready = context.Barrier(2)
        with tempfile.TemporaryDirectory(prefix="tp_cache_pending_") as directory:
            processes = [
                context.Process(
                    target=_run_tp_sync_cache_rows_pending_rank,
                    args=(tp_rank, os.path.join(directory, "broadcast"), ready),
                )
                for tp_rank in range(2)
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
                    if process.is_alive():
                        process.kill()
                        process.join(timeout=5)


class CacheStoreForwardModel:
    """Test model that replaces attention math but keeps the real cache-store call."""

    supports_hidden_state_capture = True

    def __init__(self, hidden_size: int = 1) -> None:
        self.hidden_size = hidden_size
        # Match the production disaggregate model contract: the micro-batch
        # result already includes final normalization.
        self.micro_batch_final_norm = True
        self.kv_cache = None
        self.forward_calls = 0
        self.micro_batch_calls = 0
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

    def forward_micro_batch(self, inputs: list[PyModelInputs]) -> list[PyModelOutputs]:
        self.micro_batch_calls += 1
        return [self._forward_one(model_inputs) for model_inputs in inputs]


class K3RequestPaddingTest(unittest.TestCase):
    def test_request_padding_keeps_complete_rows_and_null_cache(self):
        import math

        for tp in (4, 8):
            for batch in (1, 2, 3, 7, 8, 31, 32, 63, 64):
                for width, prefill, update in (
                    (1, False, False),
                    (4, False, False),
                    (4, False, True),
                    (3, True, False),
                ):
                    for grouped in (False, True):
                        with self.subTest(
                            tp=tp,
                            batch=batch,
                            width=width,
                            prefill=prefill,
                            update=update,
                            grouped=grouped,
                        ):
                            result = run_scenario(
                                CacheStoreForwardModel(),
                                f"request_padding {tp} {batch} {width} {int(prefill)} {int(grouped)} {int(update)}",
                            )
                            logical_tokens = batch * width + int(prefill)
                            unit = tp // math.gcd(tp, width)
                            physical_batch = (batch + unit - 1) // unit * unit
                            physical_tokens = physical_batch * width
                            if prefill:
                                physical_tokens = (logical_tokens + tp - 1) // tp * tp
                                physical_batch = batch + int(
                                    physical_tokens > logical_tokens
                                )
                            self.assertEqual(result["logical_requests"], batch)
                            self.assertEqual(result["logical_tokens"], logical_tokens)
                            self.assertEqual(result["tokens"].numel(), physical_tokens)
                            self.assertEqual(
                                result["input_lengths"].numel(), physical_batch
                            )
                            self.assertEqual(
                                result["hidden"].shape, (physical_tokens, 2)
                            )
                            torch.testing.assert_close(
                                result["tokens"][:logical_tokens].cpu(),
                                torch.arange(logical_tokens, dtype=torch.int32),
                                rtol=0,
                                atol=0,
                            )
                            torch.testing.assert_close(
                                result["positions"][:logical_tokens].cpu(),
                                torch.arange(logical_tokens, dtype=torch.int32),
                                rtol=0,
                                atol=0,
                            )
                            self.assertFalse(
                                result["tokens"][logical_tokens:].count_nonzero()
                            )
                            self.assertFalse(
                                result["hidden"][logical_tokens:].count_nonzero()
                            )
                            self.assertFalse(
                                result["positions"][logical_tokens:].count_nonzero()
                            )
                            self.assertEqual(result["output_indexes"].numel(), batch)
                            torch.testing.assert_close(
                                result["output_indexes"].cpu(),
                                torch.arange(batch, dtype=torch.int32) * width
                                + width
                                - 1
                                + int(prefill),
                                rtol=0,
                                atol=0,
                            )
                            torch.testing.assert_close(
                                result["hidden"][:logical_tokens].cpu(),
                                torch.arange(
                                    logical_tokens * 2, dtype=torch.float32
                                ).reshape(logical_tokens, 2),
                                rtol=0,
                                atol=0,
                            )
                            self.assertTrue((result["mask"] == 1).all())
                            lengths = torch.full(
                                (physical_batch,), width, dtype=torch.int32
                            )
                            if prefill:
                                lengths[0] += 1
                                if physical_batch > batch:
                                    lengths[-1] = physical_tokens - logical_tokens
                            torch.testing.assert_close(
                                result["input_lengths"].cpu(), lengths, rtol=0, atol=0
                            )
                            for key, present in (
                                ("prefix_lengths", prefill or width > 1),
                                ("sequence_lengths", not prefill and width == 1),
                            ):
                                expected = (
                                    torch.cat(
                                        (
                                            torch.full(
                                                (batch,), 65535, dtype=torch.int32
                                            ),
                                            torch.zeros(
                                                physical_batch - batch,
                                                dtype=torch.int32,
                                            ),
                                        )
                                    )
                                    if present
                                    else torch.empty(0, dtype=torch.int32)
                                )
                                torch.testing.assert_close(
                                    result[key].cpu(), expected, rtol=0, atol=0
                                )
                            torch.testing.assert_close(
                                result["sequence_lengths_plus_1"].cpu(),
                                torch.cat(
                                    (
                                        torch.full((batch,), 65536, dtype=torch.int32),
                                        torch.ones(
                                            physical_batch - batch, dtype=torch.int32
                                        ),
                                    )
                                ),
                                rtol=0,
                                atol=0,
                            )
                            for key in ("physical_table", "kernel_table"):
                                table = result[key]
                                self.assertEqual(table.shape[-2], physical_batch)
                                self.assertFalse(
                                    table.narrow(
                                        table.ndim - 2, batch, physical_batch - batch
                                    ).count_nonzero()
                                )
                            if not prefill:
                                self.assertTrue(
                                    (result["input_lengths"] == width).all()
                                )
                                if batch == 32 and width == 4:
                                    self.assertEqual(physical_batch, 32)


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


class MissingMicroBatchForwardModel(CacheStoreForwardModel):
    def __getattribute__(self, name):
        if name == "forward_micro_batch":
            raise AttributeError(name)
        return super().__getattribute__(name)


class PositionAwareCaptureModel(CacheStoreForwardModel):
    position_aware_hidden = True

    def _forward_one(self, inputs: PyModelInputs) -> PyModelOutputs:
        output = super()._forward_one(inputs)
        positions = inputs.input_ids.to(torch.float16).reshape(-1, 1)
        output.hidden_states = positions
        if inputs.capture_hidden_states:
            output.captured_hidden_states = torch.cat(
                (positions + 10, positions + 20, positions + 30), dim=-1
            )
        return output


class SequenceParallelCaptureModel(PositionAwareCaptureModel):
    requires_sequence_parallel_padding = True

    def __init__(self) -> None:
        super().__init__()
        self.seen_input_ids: list[list[int]] = []

    def _forward_one(self, inputs: PyModelInputs) -> PyModelOutputs:
        self.seen_input_ids.append(inputs.input_ids.cpu().tolist())
        output = super()._forward_one(inputs)
        # Canonical output is independent of optional auxiliary capture.
        output.hidden_states = inputs.input_ids.to(torch.float16).reshape(-1, 1) + 30
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
                (hidden_states + 10, hidden_states + 20, final_hidden_states),
                dim=-1,
            )
        return output


def _load_capture_context():
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
        "_pywrapped_test_hidden_state_capture",
        root / "model_desc/hidden_state_capture.py",
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
    return capture.CaptureContext


class CaptureContextForwardModel(CacheStoreForwardModel):
    def __init__(self, hidden_size=1, failure_stage="", error=None):
        super().__init__(hidden_size)
        self.failure_stage = failure_stage
        self.error = (
            error
            if error is not None
            else RuntimeError(f"injected transient {failure_stage} failure")
        )
        self.layer_hook_calls = 0
        self.capture_errors = []
        self.capture_context = _load_capture_context().configured(
            [0, 1], self._canonical_layer, self._canonical_final
        )

    def _canonical_layer(self, hidden, residual):
        self.layer_hook_calls += 1
        if (
            self.failure_stage == "hook"
            and self.forward_calls == 1
            and self.layer_hook_calls == 2
        ):
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


class MalformedCaptureOutputCountModel(CacheStoreForwardModel):
    def forward_micro_batch(self, inputs: list[PyModelInputs]) -> list[PyModelOutputs]:
        outputs = super().forward_micro_batch(inputs)
        return outputs[:-1]


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
                tags = [] if skip_run else ["full", "linear"]
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
                        record_broadcast, unexpected_collective, unexpected_collective
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
        context = multiprocessing.get_context("spawn")
        with tempfile.TemporaryDirectory(prefix="tp_sync_") as directory:
            processes = [
                context.Process(
                    target=_run_tp_sync_uds_rank,
                    args=(tp_rank, os.path.join(directory, "broadcast")),
                )
                for tp_rank in range(4)
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
                scopes = forward["aten_profile"]["scopes"]
                for name in (
                    "py_model.prepareAttentionInputs(build)",
                    "py_model.prepareAttentionInputs(padding_offset)",
                    "py_model.prepareAttentionInputs(setup_kv_cache)",
                    "py_model.prepareAttentionInputs(fused_h2d)",
                    "py_model.updateKVCacheKernelBlockId",
                ):
                    self.assertEqual(scopes.get(name, 0), 1, name)
                for name in (
                    "cuda_graph.canRun",
                    "cuda_graph.prepareAttentionInputs",
                    "cuda_graph.updateKVCacheKernelBlockId",
                    "py_model.prepareAttentionInputs(cuda_graph_prepare)",
                    "py_model.forward(cuda_graph)",
                ):
                    if capture:
                        self.assertEqual(scopes.get(name, 0), 0, name)
                    else:
                        self.assertGreater(scopes.get(name, 0), 0, name)
                self.assertEqual(
                    scopes.get("py_model.forward(normal)", 0), int(capture)
                )
                self.assertEqual(
                    forward["attention_metadata_after_prepare"][
                        "kernel_block_ids"
                    ].tolist(),
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
            metric_name = f"{_CAPTURE_METRIC_PREFIX}{name}_qps"
            self.assertIn(metric_name, expected, f"unknown capture QPS metric: {name}")
            expected[metric_name] = value
        self.assertEqual(
            {name: metrics[name] for name in _CAPTURE_QPS_METRICS}, expected
        )

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

    def assertCanonicalZero(self, result, rows=5) -> None:
        self.assertFalse(result["capture_fatal"])
        for hidden in result["forward_hidden_states"]:
            self.assertEqual(tuple(hidden.shape), (rows, 1))
            torch.testing.assert_close(hidden, torch.zeros_like(hidden), rtol=0, atol=0)

    def assertCaptureLayoutProfile(self, result, *, distributed=False) -> None:
        profiles = result["aten_profiles"]
        self.assertEqual(len(profiles), len(result["forward_hidden_states"]))
        self.assertTrue(profiles)
        for profile in profiles:
            self.assertEqual(
                profile["scopes"].get("py_model.capture_layout_status", 0), 1
            )
            if distributed:
                self.assertIn("aten::item", profile["layout_aten_ops"])
                self.assertIn("aten::_local_scalar_dense", profile["layout_aten_ops"])
                self.assertGreater(profile["layout_cuda_int32_inputs"], 0)
                self.assertGreater(profile["layout_cuda_int32_outputs"], 0)
            else:
                self.assertEqual(profile["layout_aten_ops"], [])
                self.assertEqual(profile["layout_cuda_int32_inputs"], 0)
                self.assertEqual(profile["layout_cuda_int32_outputs"], 0)

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
        self.assertTrue(
            all("_tag_full" in block["key"] for block in middle["blocks"])
        )
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

    def test_micro_batch_uses_regular_forward_when_entry_point_is_absent(
        self,
    ) -> None:
        model = MissingMicroBatchForwardModel()
        result = run_scenario(model, "micro_batch")

        self.assertEqual(model.micro_batch_calls, 0)
        self.assertEqual(model.forward_calls, 1)
        self.assertEqual(len(result["records"]), 3)

    def test_micro_batch_slices_request_metadata_with_block_rows(self) -> None:
        model = CacheStoreForwardModel()
        result = run_scenario(model, "micro_batch")

        self.assertEqual(model.forward_calls, 0)
        self.assertEqual(model.micro_batch_calls, 1)
        self.assertEqual(model.seen_input_lengths, [[2, 4], [2]])
        self.assertEqual(len(result["records"]), 3)

        expected = {
            201: ([2101], [16]),
            202: ([2201, 2202], [32, 48]),
            203: ([2301], [64]),
        }
        base = result["base_addresses"]["default"]
        for request_id, (token_keys, offsets) in expected.items():
            record = _record_for_request(result, request_id)
            self.assertEqual(len(record["blocks"]), len(token_keys))
            self.assertEqual(
                sorted(block["address"] - base for block in record["blocks"]),
                offsets,
            )
            for token_key in token_keys:
                self.assertTrue(
                    any(
                        f"_token_id_str_{token_key}_" in block["key"]
                        for block in record["blocks"]
                    )
                )

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

    def test_micro_batch_preserves_reordered_group_rows(self) -> None:
        model = CacheStoreForwardModel()
        result = run_scenario(model, "micro_batch_multi_tag")
        self.assertEqual(model.seen_input_lengths, [[2, 4], [2]])
        self.assertEqual(len(model.seen_cache_tables), 2)
        expected_tables = [
            {
                "full": [[1, -1, -1, -1], [2, 3, -1, -1]],
                "linear": [[2, 3, -1, -1], [4, 5, 6, 7]],
            },
            {"full": [[4, -1, -1, -1]], "linear": [[1, 2, -1, -1]]},
        ]
        for actual, expected in zip(model.seen_cache_tables, expected_tables):
            self.assertEqual(set(actual), set(expected))
            for tag, table in expected.items():
                self.assertEqual(actual[tag]["physical"], table)
                self.assertEqual(actual[tag]["kernel"], table)

        blocks = _blocks_by_key(result)
        for tag, offsets in (
            ("full", [16, 32, 48, 64]),
            ("linear", [24, 48, 48, 72, 96, 120, 144, 168]),
        ):
            self.assertEqual(
                sorted(
                    block["address"] - result["base_addresses"][tag]
                    for key, block in blocks.items()
                    if f"_tag_{tag}" in key
                ),
                offsets,
            )

    def test_micro_batch_split_preserves_cache_storage_contract(self) -> None:
        multi_group_tables = [
            [[1, -1, -1, -1], [2, 3, -1, -1], [4, -1, -1, -1]],
            [[2, 3, -1, -1], [4, 5, 6, 7], [1, 2, -1, -1]],
        ]
        single_group_tables = [[[1, -1], [2, 3], [4, -1]]]
        for scenario, on_cuda, two_dimensional, tags, tables in (
            (
                "micro_batch_split_pinned",
                False,
                False,
                ["full", "linear"],
                multi_group_tables,
            ),
            (
                "micro_batch_split_cuda",
                True,
                False,
                ["full", "linear"],
                multi_group_tables,
            ),
            (
                "micro_batch_split_single_group",
                False,
                False,
                ["default"],
                single_group_tables,
            ),
            ("micro_batch_split_2d", False, True, ["default"], single_group_tables),
        ):
            with self.subTest(scenario=scenario):
                result = run_scenario(CacheStoreForwardModel(), scenario)
                self.assertEqual(result["source_tags"], tags)
                self.assertEqual(len(result["batches"]), 2)
                physical = torch.tensor(tables, dtype=torch.int32)
                kernel = torch.where(physical >= 0, physical + 100, physical)
                if two_dimensional:
                    physical = physical.squeeze(0)
                    kernel = kernel.squeeze(0)
                batch_axis = 0 if two_dimensional else 1
                for field, expected in (("physical", physical), ("kernel", kernel)):
                    source = result[f"source_{field}"]
                    self.assertEqual(source.dtype, torch.int32)
                    self.assertEqual(source.is_cuda, on_cuda)
                    self.assertEqual(source.is_pinned(), not on_cuda)
                    torch.testing.assert_close(source.cpu(), expected)
                    for batch, start, count in zip(result["batches"], (0, 2), (2, 1)):
                        self.assertEqual(batch["tags"], tags)
                        tensor = batch[field]
                        self.assertEqual(tensor.dtype, torch.int32)
                        self.assertEqual(tensor.device, source.device)
                        self.assertEqual(tensor.is_pinned(), not on_cuda)
                        self.assertTrue(tensor.is_contiguous())
                        torch.testing.assert_close(
                            tensor.cpu(), expected.narrow(batch_axis, start, count)
                        )
                self.assertEqual(
                    [batch["input_lengths"].tolist() for batch in result["batches"]],
                    [[2, 4], [2]],
                )

    def test_fake_micro_batch_has_no_cache_identity(self) -> None:
        result = run_scenario(CacheStoreForwardModel(), "fake_micro_batch")
        self.assertEqual(result["real_tags"], ["full", "linear"])
        self.assertEqual(result["fake_tags"], [])
        self.assertFalse(result["fake_physical_defined"])
        self.assertFalse(result["fake_kernel_defined"])

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

    def test_skip_lm_head_without_explicit_capture_does_not_publish(self) -> None:
        model = CacheStoreForwardModel()

        result = run_scenario(model, "capture_explicit_off")

        self.assertEqual(model.capture_flags, [False])
        self.assertFalse(result["logits_defined"])
        self.assertEqual(_FakeEagleMooncakeStore.instances[0].puts, [])
        self.assertCaptureQps(
            result["capture_metrics"], batch=0, publish_success=0, failure=0
        )

    def assertCaptureProfileStoreCalls(self, events, store_events) -> None:
        prefix = "hidden_state_capture."
        for scope, call, parent in (
            ("put_batch_python", "store.put_batch", "put_batch_admit"),
            ("flush_python", "store.flush", "flush"),
            ("take_async_errors", "store.take_async_errors", "observe_async_errors"),
        ):
            scopes = [e for e in events if e.name == prefix + scope]
            self.assertEqual(
                len(scopes), sum(e.split(":", 1)[0] == call for e in store_events)
            )
            for event in scopes:
                self.assertIsNotNone(event.cpu_parent)
                self.assertEqual(event.cpu_parent.name, prefix + parent)
        for event in events:
            if not event.name.startswith(prefix):
                continue
            self.assertLessEqual(event.time_range.start, event.time_range.end)
            if event.cpu_parent is not None:
                self.assertLessEqual(
                    event.cpu_parent.time_range.start, event.time_range.start
                )
                self.assertLessEqual(
                    event.time_range.end, event.cpu_parent.time_range.end
                )
            if event.name == prefix + "observe_async_errors":
                self.assertEqual(
                    [e.name for e in events if e.cpu_parent is event],
                    [prefix + "take_async_errors"],
                )
            elif event.name == prefix + "flush":
                children = [e for e in events if e.cpu_parent is event]
                self.assertIn(
                    [e.name for e in children],
                    (
                        [prefix + "flush_python"],
                        [prefix + "flush_python", prefix + "observe_async_errors"],
                    ),
                )
                if len(children) == 2:
                    self.assertLessEqual(
                        children[0].time_range.end, children[1].time_range.start
                    )

    def test_capture_profile_scopes_preserve_forward_contract(self) -> None:
        _register_identity_allreduce()
        self.addCleanup(clear_comm_ops)
        scenarios = [
            "capture_bf16",
            "capture_micro_batch",
            "capture_non_owner",
            "capture_explicit_off",
        ]
        if os.environ.get("TEST_USING_DEVICE") != "ROCM":
            scenarios += ["capture_fp8", "capture_cp"]
        for scenario in scenarios:
            namespace = (
                "ProfileActor" + "A" * 256 if scenario == "capture_bf16" else "TestActor"
            )
            with self.subTest(scenario=scenario), patch.dict(
                os.environ, {_MOONCAKE_STORE_KEY_NAMESPACE_ENV: namespace}
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
                        result["capture_metrics"][name], baseline["capture_metrics"][name]
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
                        "store.close",
                    ]
                self.assertEqual(store_calls, expected_calls)
                markers = [e for e in events if e.name.startswith("rtp-forward-")]
                self.assertEqual([e.name for e in markers], batch_ids)
                publish = [
                    e for e in events if e.name == "hidden_state_capture.publish"
                ]
                if scenario == "capture_non_owner":
                    self.assertFalse(publish)
                    self.assertFalse(
                        any(e.name.startswith("hidden_state_capture.") for e in events)
                    )
                    continue
                if scenario == "capture_explicit_off":
                    self.assertFalse(publish)
                    # A configured but inactive publisher still drains at shutdown.
                    capture_events = [
                        e for e in events if e.name.startswith("hidden_state_capture.")
                    ]
                    self.assertEqual(
                        [e.name for e in capture_events],
                        [
                            "hidden_state_capture.flush",
                            "hidden_state_capture.flush_python",
                            "hidden_state_capture.observe_async_errors",
                            "hidden_state_capture.take_async_errors",
                        ],
                    )
                    self.assertIsNone(capture_events[0].cpu_parent)
                    continue
                self.assertEqual(len(publish), 1)
                children = {e.name: e for e in events if e.cpu_parent is publish[0]}
                gil = children["hidden_state_capture.gil_acquire"]
                poll = children["hidden_state_capture.observe_async_errors"]
                prepare = children["hidden_state_capture.prepare_payload"]
                admit = children["hidden_state_capture.put_batch_admit"]
                flush = children["hidden_state_capture.flush"]
                self.assertLessEqual(gil.time_range.end, poll.time_range.start)
                self.assertLessEqual(poll.time_range.end, prepare.time_range.start)
                marker = markers[0]
                if scenario == "capture_bf16":
                    self.assertGreater(len(marker.name), 256)
                self.assertIs(marker.cpu_parent, publish[0])
                self.assertLessEqual(prepare.time_range.end, marker.time_range.start)
                self.assertLessEqual(marker.time_range.start, marker.time_range.end)
                self.assertLessEqual(marker.time_range.end, admit.time_range.start)
                self.assertLessEqual(prepare.time_range.end, admit.time_range.start)
                self.assertLessEqual(admit.time_range.end, flush.time_range.start)
                quantize = [
                    e for e in events if e.name == "hidden_state_capture.quantize"
                ]
                self.assertEqual(len(quantize), int(scenario == "capture_fp8"))
                if quantize:
                    self.assertIs(quantize[0].cpu_parent, prepare)
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
                names = {e.name for e in events}
                if scenario == "capture_micro_batch":
                    self.assertIn("py_model.forwardMicroBatched(python_forward)", names)
                    self.assertIn("py_model.capture_microbatch_layout", names)
                    self.assertIn("py_model.forwardMicroBatched(merge)", names)
                else:
                    layout = next(
                        e for e in events if e.name == "py_model.capture_layout_status"
                    )
                    self.assertLessEqual(
                        layout.time_range.end, publish[0].time_range.start
                    )
                if scenario == "capture_cp":
                    restore = next(
                        e for e in events if e.name == "py_model.capture_cp_restore"
                    )
                    self.assertLessEqual(
                        layout.time_range.end, restore.time_range.start
                    )
                    self.assertLessEqual(
                        restore.time_range.end, publish[0].time_range.start
                    )

    def test_capture_profile_scopes_close_on_admission_and_flush_failure(self) -> None:
        original_poll = _FakeEagleMooncakeStore.take_async_errors
        for failure, stage, poll_failure_at in (
            ("fail_put", "put_batch_admit", 0),
            ("fail_flush", "flush", 0),
            ("poll_raises", "observe_async_errors", 1),
            ("poll_invalid", "observe_async_errors", 1),
            ("poll_raises", "flush", 2),
            ("poll_invalid", "flush", 2),
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

            with self.subTest(failure=failure, poll=poll_failure_at), patch.multiple(
                _FakeEagleMooncakeStore,
                fail_put=failure == "fail_put",
                fail_flush=failure == "fail_flush",
            ), patch.object(_FakeEagleMooncakeStore, "take_async_errors", poll):
                store_calls = []
                _FakeEagleMooncakeStore.events = store_calls
                with torch.profiler.profile(
                    activities=[torch.profiler.ProfilerActivity.CPU]
                ) as prof:
                    result = run_scenario(CacheStoreForwardModel(), "capture_bf16")
                self.assertCanonicalZero(result)
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
                self.assertCaptureQps(
                    metrics,
                    batch=1,
                    publish_success=0,
                    failure=3 if persistent else 1,
                    store_failure=2 if persistent else 1,
                    shutdown_failure=int(persistent),
                    hard_contract_failure=int(persistent),
                    operational_failure=2 if persistent else 1,
                )
                self.assertCaptureStatus(
                    metrics, enabled=int(not persistent), broken=int(persistent)
                )
                events = prof.events()
                self.assertCaptureProfileStoreCalls(events, store_calls)
                publish = next(
                    e for e in events if e.name == "hidden_state_capture.publish"
                )
                children = {e.name: e for e in events if e.cpu_parent is publish}
                expected_stage = "hidden_state_capture." + stage
                self.assertIn(expected_stage, children)
                self.assertLessEqual(
                    children[expected_stage].time_range.end, publish.time_range.end
                )
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
                    self.assertIn(batch_id, children)
                    self.assertIn("hidden_state_capture.put_batch_admit", children)
                    if failure != "fail_put":
                        expected_calls.append("store.flush")
                        if not persistent:
                            expected_calls.append("store.take_async_errors")
                    # Failed/partial PUT or flush always drains again on the worker.
                    if failure == "fail_put" or persistent or poll_failure_at == 2:
                        expected_calls.append("store.flush")
                        if not persistent:
                            expected_calls.append("store.take_async_errors")
                else:
                    self.assertEqual(store.put_batch_calls, [])
                    self.assertNotIn("hidden_state_capture.prepare_payload", children)
                    self.assertNotIn("hidden_state_capture.put_batch_admit", children)
                    self.assertFalse(
                        any(e.name.startswith("rtp-forward-") for e in events)
                    )
                if failure == "fail_put" or poll_failure_at == 1:
                    self.assertNotIn("hidden_state_capture.flush", children)
                expected_calls.append("store.flush")
                if not persistent:
                    expected_calls.append("store.take_async_errors")
                retains_source = poll_failure_at != 1 and failure != "fail_put"
                if not retains_source:
                    expected_calls.append("store.close")
                self.assertEqual(store_calls, expected_calls)
                self.assertTrue(store.flushed)
                self.assertEqual(store.closed, not retains_source)

    def test_capture_forward_returns_while_store_is_blocked(self) -> None:
        entered = threading.Event()
        release = threading.Event()
        timed_out = threading.Event()
        producer_thread = threading.get_ident()
        producer_streams = []

        class ObservedProducer(CacheStoreForwardModel):
            def forward(self, inputs, fmha_impl=None):
                producer_streams.append(torch.cuda.current_stream().cuda_stream)
                return super().forward(inputs, fmha_impl)

        worker_identity = []
        original_put = _FakeEagleMooncakeStore.put_batch

        def blocked_put(store, **kwargs):
            worker_identity.append(
                (threading.get_ident(), torch.cuda.current_stream().cuda_stream)
            )
            entered.set()
            if not release.wait(10):
                timed_out.set()
                raise RuntimeError("serving forward blocked on capture store")
            return original_put(store, **kwargs)

        def after_forward(_index):
            try:
                self.assertTrue(entered.wait(10), "worker did not enter PUT")
                self.assertFalse(timed_out.is_set())
                self.assertFalse(release.is_set())
            finally:
                release.set()

        model = ObservedProducer()
        with patch.object(_FakeEagleMooncakeStore, "put_batch", blocked_put):
            result = run_scenario(
                model,
                "capture_with_lm_head",
                after_forward=after_forward,
            )
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

    def submitCapture(self, publisher, request_id=1, wait=True):
        reservation = publisher.reserve(request_id, 1)
        self.assertIsNotNone(reservation)
        self.assertTrue(
            publisher.submit(
                torch.tensor([[10, 20, 30]], dtype=torch.float16),
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

    def test_capture_health_uses_attributed_runtime_boundary(self) -> None:
        for domain in (1, 2):
            cases = [
                ({"code": 0}, True),
                ({"code": 600}, True),
                ({"code": 2}, True),
                ({"code": 700, "get_code": 700}, True),
                ({"code": 700, "operation": 1}, True),  # failed SetDevice
                (
                    {
                        "code": 700,
                        "operation": 2,
                        "resource_device": 1,
                        "current_device": 1,
                    },
                    True,
                ),
                (
                    {"code": 700, "operation": 7},
                    True,
                ),  # host-thread peek is not evidence
                ({"code": 700, "resource_device": 1, "current_device": 1}, True),
                ({"code": 700, "resource_device": 0, "current_device": 1}, True),
                ({"code": 700}, False),
                ({"code": 710}, False),
                ({"code": 719}, False),
            ]
            for args, healthy in cases:
                with self.subTest(domain=domain, args=args):
                    publisher = self.capturePublisher()
                    try:
                        # Owner allocation failure must not mask later attributed evidence.
                        publisher.latch_owner_failure()
                        self.assertTrue(publisher.serving_healthy)
                        observed = publisher.runtime_boundary(domain, **args)
                        self.assertEqual(publisher.serving_healthy, healthy)
                        if args.get("get_code"):
                            self.assertFalse(observed["performed"])
                            self.assertEqual(observed["device"], -1)
                            self.assertEqual(observed["operation"], 0)
                            self.assertEqual(observed["owner_code"], 700)
                            self.assertEqual(observed["owner_operation"], 0)
                        if args.get("operation") in (1, 2):
                            self.assertEqual(observed["owner_code"], 700)
                            self.assertEqual(
                                observed["owner_operation"], args["operation"]
                            )
                        publisher.runtime_boundary(domain, 0)
                        self.assertEqual(publisher.serving_healthy, healthy)
                    finally:
                        publisher.close()

    def test_nonowner_runtime_boundary_does_not_change_service_health(self) -> None:
        publisher = _extension.CapturePublisher(owner=False)
        try:
            observed = publisher.runtime_boundary(1, 700)
            self.assertTrue(observed["performed"])
            self.assertTrue(publisher.serving_healthy)
            self.assertFalse(publisher.fatal)
        finally:
            publisher.close()

    def test_cpu_store_fault_does_not_poison_serving_device(self) -> None:
        for error in (
            ValueError("invalid payload"),
            RuntimeError("network disconnected"),
            MemoryError("host allocator failure"),
            torch.OutOfMemoryError("CUDA out of memory"),
            RuntimeError("CUDA error: illegal address"),
        ):
            with self.subTest(error=repr(error)):
                publisher = self.capturePublisher()
                try:
                    _FakeEagleMooncakeStore.instances[-1].pending_async_errors.append(
                        error
                    )
                    self.submitCapture(publisher)
                    self.assertTrue(publisher.serving_healthy)
                finally:
                    publisher.close()

    def test_capture_start_poll_fatal_stops_current_put(self) -> None:
        for error in (
            MemoryError("injected allocation failure"),
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
                    self.assertFalse(publisher.reserve(2, 1))
                    self.assertEqual(store.put_attempts, 0)
                    self.assertEqual(store.keys, set())
                    self.assertEqual(publisher.inflight_samples, 1)
                    self.assertEqual(
                        store.call_sequence, before + ["take_async_errors"]
                    )
                    for _ in range(2):
                        self.assertTrue(publisher.poll())
                    publisher.close()
                    publisher.close()
                    publisher.poll()  # A stopped worker need not report idle.
                    self.assertEqual(
                        store.call_sequence, before + ["take_async_errors"]
                    )
                    self.assertFalse(store.closed)
                finally:
                    publisher.close()

    def test_capture_heterogeneous_capability_votes_before_sp_forward(self) -> None:
        # Replay both sides of the same MIN using one healthy and one failed
        # local contract, not two identically unsupported models. No fixture
        # constructor may communicate; only forward enters this callback.
        for failed_rank in (0, 1):
            votes = []
            for rank, scenario in enumerate(("capture_bf16_tp", "capture_non_owner")):
                with self.subTest(failed_rank=failed_rank, rank=rank):
                    model = SequenceParallelCaptureModel()
                    model.supports_hidden_state_capture = rank != failed_rank
                    calls = []

                    def reduce(tensor, op, mode, dest):
                        self.assertEqual(model.forward_calls, 0)
                        self.assertEqual((op, mode, dest), (3, 0, None))
                        self.assertEqual(tensor.dtype, torch.int64)
                        self.assertEqual(tensor.numel(), 18)
                        self.assertEqual(int(tensor[0]), int(rank != failed_rank))
                        calls.append(tensor.cpu().clone())
                        remote = tensor.clone()
                        remote[0] = int(rank == failed_rank)
                        remote[tensor.numel() // 2] = ~remote[0]
                        return torch.minimum(tensor, remote)

                    _register_no_collectives(reduce)
                    try:
                        result = run_scenario(model, scenario, observe_aten=True)
                    finally:
                        clear_comm_ops()
                    self.assertEqual(len(calls), 1)
                    votes.append(calls[0])
                    self.assertEqual(model.capture_flags, [False])
                    self.assertEqual(model.seen_input_lengths, [[2, 3, 1]])
                    self.assertEqual(
                        result["aten_profiles"][0]["scopes"].get(
                            "py_model.prepareAttentionInputs", 0
                        ),
                        1,
                    )
                    torch.testing.assert_close(
                        result["forward_hidden_states"][0],
                        torch.arange(31, 36, dtype=torch.float16).reshape(-1, 1),
                        rtol=0,
                        atol=0,
                    )
            self.assertEqual(
                [int(vote[0]) for vote in votes],
                [int(failed_rank != 0), int(failed_rank != 1)],
            )
        self.assertTrue(
            all(not store.puts for store in _FakeEagleMooncakeStore.instances)
        )

    def test_capture_owner_unavailable_disables_healthy_non_owner(self) -> None:
        for rank, scenario in enumerate(("capture_bf16_tp", "capture_non_owner")):
            with self.subTest(rank=rank):
                model = SequenceParallelCaptureModel()
                calls = []

                def reduce(tensor, op, mode, dest):
                    self.assertEqual(model.forward_calls, 0)
                    self.assertEqual((op, mode, dest), (3, 0, None))
                    self.assertEqual(tensor.numel(), 18)
                    self.assertEqual(int(tensor[0]), rank)
                    calls.append(tensor.cpu().clone())
                    remote = tensor.clone()
                    remote[0] = 1 - rank
                    remote[tensor.numel() // 2] = ~remote[0]
                    return torch.minimum(tensor, remote)

                _register_no_collectives(reduce)
                try:
                    with patch.object(
                        _FakeEagleMooncakeStore, "fail_warmup", rank == 0
                    ):
                        result = run_scenario(model, scenario)
                finally:
                    clear_comm_ops()
                self.assertEqual(len(calls), 1)
                self.assertEqual(model.capture_flags, [False])
                torch.testing.assert_close(
                    result["forward_hidden_states"][0],
                    torch.arange(31, 36, dtype=torch.float16).reshape(-1, 1),
                    rtol=0,
                    atol=0,
                )
        self.assertTrue(
            all(not store.puts for store in _FakeEagleMooncakeStore.instances)
        )

    def test_capture_ordered_plan_mismatch_votes_before_sp_forward(self) -> None:
        layer_votes = []
        for rank, scenario in enumerate(
            ("capture_bf16_tp", "capture_reversed_plan_non_owner")
        ):
            model = SequenceParallelCaptureModel()
            calls = []

            def reduce(tensor, op, mode, dest):
                self.assertEqual(model.forward_calls, 0)
                self.assertEqual((op, mode, dest), (3, 0, None))
                self.assertEqual(tensor.dtype, torch.int64)
                calls.append(tensor.cpu().tolist())
                if len(calls) == 1:
                    self.assertEqual(tensor.numel(), 18)
                    return tensor
                self.assertEqual(len(calls), 2)
                self.assertEqual(tensor.numel(), 6)
                layer_votes.append(tensor.cpu().clone())
                remote = tensor[[0, 2, 1, 3, 5, 4]]
                return torch.minimum(tensor, remote)

            _register_no_collectives(reduce)
            try:
                result = run_scenario(model, scenario)
            finally:
                clear_comm_ops()
            self.assertEqual(len(calls), 2)
            self.assertEqual(model.capture_flags, [False])
            torch.testing.assert_close(
                result["forward_hidden_states"][0],
                torch.arange(31, 36, dtype=torch.float16).reshape(-1, 1),
                rtol=0,
                atol=0,
            )
        self.assertEqual(layer_votes[0].tolist(), [1, 0, 1, -2, -1, -2])
        self.assertEqual(layer_votes[1].tolist(), [1, 1, 0, -2, -2, -1])
        self.assertTrue(
            all(not store.puts for store in _FakeEagleMooncakeStore.instances)
        )

    def test_capture_missing_publisher_still_joins_fixed_header_vote(self) -> None:
        model = SequenceParallelCaptureModel()
        calls = []

        def reduce(tensor, op, mode, dest):
            self.assertEqual(model.forward_calls, 0)
            self.assertEqual(tensor.numel(), 18)
            self.assertEqual(int(tensor[0]), 0)
            calls.append(tensor.cpu().clone())
            return tensor

        _register_no_collectives(reduce)
        try:
            result = run_scenario(model, "capture_missing_plan_non_owner")
        finally:
            clear_comm_ops()
        self.assertEqual(len(calls), 1)
        self.assertEqual(model.capture_flags, [False])
        self.assertEqual(tuple(result["forward_hidden_states"][0].shape), (5, 1))
        self.assertEqual(_FakeEagleMooncakeStore.instances, [])

    def test_capture_dtype_mismatch_disables_before_selected_layer_gather(self) -> None:
        model = SequenceParallelCaptureModel()
        calls = []

        def reduce(tensor, op, mode, dest):
            self.assertEqual(model.forward_calls, 0)
            self.assertEqual(tensor.numel(), 18)
            calls.append(tensor.cpu().clone())
            remote = tensor.clone()
            remote[3] += 1  # capture dtype, with identical layer IDs/count
            remote[tensor.numel() // 2 + 3] = ~remote[3]
            return torch.minimum(tensor, remote)

        _register_no_collectives(reduce)
        try:
            result = run_scenario(model, "capture_bf16_tp")
        finally:
            clear_comm_ops()
        self.assertEqual(len(calls), 1)
        self.assertEqual(model.capture_flags, [False])
        self.assertEqual(tuple(result["forward_hidden_states"][0].shape), (5, 1))
        self.assertEqual(_FakeEagleMooncakeStore.instances[0].puts, [])

    def test_capture_off_tp_forwards_add_no_collective(self) -> None:
        _register_no_collectives()
        self.addCleanup(clear_comm_ops)
        model = SequenceParallelCaptureModel()
        result = run_scenario(model, "capture_explicit_off_tp", 3)
        self.assertEqual(model.capture_flags, [False] * 3)
        for hidden in result["forward_hidden_states"]:
            torch.testing.assert_close(
                hidden,
                torch.arange(31, 36, dtype=torch.float16).reshape(-1, 1),
                rtol=0,
                atol=0,
            )

    def test_capture_plan_collective_failure_is_not_swallowed(self) -> None:
        model = SequenceParallelCaptureModel()

        def reduce(*_args):
            raise RuntimeError("injected capture plan collective failure")

        _register_no_collectives(reduce)
        self.addCleanup(clear_comm_ops)
        with self.assertRaisesRegex(RuntimeError, "capture plan collective failure"):
            run_scenario(model, "capture_bf16_tp")
        self.assertEqual(model.forward_calls, 0)

    def test_sp_capture_excludes_dummy_rows_and_matches_capture_off(self) -> None:
        layout_statuses = []

        def allreduce(tensor, op, mode, dest):
            self.assertEqual((op, mode, dest), (3, 0, None))
            layout_statuses.append(int(tensor.item()))
            return tensor

        _register_layout_allreduce(allreduce)
        self.addCleanup(clear_comm_ops)
        model = SequenceParallelCaptureModel()
        # The native TP=2 fixture has two logical prompts [2, 3] (T=5).
        # SP executes six rows and appends a length-1 dummy request, without an ID.
        result = run_scenario(
            model, "capture_bf16_tp", 2, capture_only_first_forward=True
        )

        self.assertEqual(model.capture_flags, [True, False])
        self.assertEqual(model.seen_input_ids, [[1, 2, 3, 4, 5, 0]] * 2)
        self.assertEqual(model.seen_input_lengths, [[2, 3, 1]] * 2)
        self.assertEqual(layout_statuses, [2])
        self.assertFalse(result["logits_defined"])
        self.assertEqual(result["capture_failure_count"], 0)
        self.assertEqual(result["capture_broken_rejection_count"], 0)
        self.assertEqual(result["capture_diagnostics"], [""] * 2)
        self.assertEqual(result["cp_handle_outputs_calls"], 0)
        capture_on_hidden, capture_off_hidden = result["forward_hidden_states"]
        self.assertEqual(tuple(capture_on_hidden.shape), (5, 1))
        self.assertEqual(tuple(capture_off_hidden.shape), (5, 1))
        torch.testing.assert_close(
            capture_on_hidden, capture_off_hidden, rtol=0, atol=0
        )
        torch.testing.assert_close(
            capture_on_hidden,
            torch.arange(31, 36, dtype=torch.float16).reshape(-1, 1),
            rtol=0,
            atol=0,
        )

        self.assertEqual(len(_FakeEagleMooncakeStore.instances), 1)
        store = _FakeEagleMooncakeStore.instances[0]
        self.assertEqual(len(store.put_batch_calls), 1)
        self.assertEqual(
            store.put_batch_calls[0]["request_ids"],
            ["rtp_TestActor_501", "rtp_TestActor_502"],
        )
        self.assertEqual(store.put_batch_calls[0]["lengths"], [2, 3])
        self.assertEqual(len(store.puts), 2)
        self.assertEqual(
            [put["input_ids"].tolist() for put in store.puts], [[1, 2], [3, 4, 5]]
        )
        for put, start, end in zip(store.puts, (0, 2), (2, 5), strict=True):
            ids = put["input_ids"].to(torch.bfloat16).reshape(-1, 1)
            self.assertEqual(tuple(put["hidden_states"].shape), (end - start, 2))
            torch.testing.assert_close(
                put["hidden_states"], torch.cat((ids + 10, ids + 20), dim=-1)
            )
            torch.testing.assert_close(
                put["last_hidden_states"],
                capture_on_hidden[start:end].to(torch.bfloat16),
                rtol=0,
                atol=0,
            )
        metrics = result["capture_metrics"]
        self.assertEqual(metrics[f"{_CAPTURE_METRIC_PREFIX}publish_request_count"], 2)
        self.assertEqual(metrics[f"{_CAPTURE_METRIC_PREFIX}publish_token_count"], 5)
        self.assertCaptureStatus(metrics, enabled=1, broken=0)
        self.assertTrue(store.flushed)
        self.assertTrue(store.closed)

    def test_sp_capture_tp2_three_tokens_preserves_logical_metadata(self) -> None:
        layout_statuses = []

        def allreduce(tensor, op, mode, dest):
            self.assertEqual((op, mode, dest), (3, 0, None))
            layout_statuses.append(int(tensor.item()))
            return tensor

        _register_layout_allreduce(allreduce)
        self.addCleanup(clear_comm_ops)
        model = SequenceParallelCaptureModel()
        # This native scenario runs PyWrappedModel::forward twice and checks that
        # its logical input tensors remain [1, 2, 3], [3], and request ID [503].
        result = run_scenario(
            model, "capture_bf16_tp_single_request", 2, capture_only_first_forward=True
        )

        self.assertEqual(model.capture_flags, [True, False])
        self.assertEqual(model.forward_calls, 2)
        self.assertEqual(model.micro_batch_calls, 0)
        self.assertEqual(model.seen_input_ids, [[1, 2, 3, 0]] * 2)
        self.assertEqual(model.seen_input_lengths, [[3, 1]] * 2)
        self.assertEqual(layout_statuses, [2])
        self.assertFalse(result["logits_defined"])
        self.assertEqual(result["capture_failure_count"], 0)
        self.assertEqual(result["capture_broken_rejection_count"], 0)
        self.assertEqual(result["capture_diagnostics"], [""] * 2)
        self.assertEqual(result["cp_handle_outputs_calls"], 0)
        capture_on_hidden, capture_off_hidden = result["forward_hidden_states"]
        self.assertEqual(tuple(capture_on_hidden.shape), (3, 1))
        self.assertEqual(tuple(capture_off_hidden.shape), (3, 1))
        torch.testing.assert_close(
            capture_on_hidden, capture_off_hidden, rtol=0, atol=0
        )
        torch.testing.assert_close(
            capture_on_hidden,
            torch.tensor([[31], [32], [33]], dtype=torch.float16),
            rtol=0,
            atol=0,
        )

        self.assertEqual(len(_FakeEagleMooncakeStore.instances), 1)
        store = _FakeEagleMooncakeStore.instances[0]
        self.assertEqual(len(store.put_batch_calls), 1)
        self.assertEqual(store.put_batch_calls[0]["request_ids"], ["rtp_TestActor_503"])
        self.assertEqual(store.put_batch_calls[0]["lengths"], [3])
        self.assertEqual(len(store.puts), 1)
        put = store.puts[0]
        self.assertEqual(put["input_ids"].tolist(), [1, 2, 3])
        self.assertEqual(tuple(put["hidden_states"].shape), (3, 2))
        torch.testing.assert_close(
            put["hidden_states"],
            torch.tensor([[11, 21], [12, 22], [13, 23]], dtype=torch.bfloat16),
            rtol=0,
            atol=0,
        )
        torch.testing.assert_close(
            put["last_hidden_states"],
            capture_on_hidden.to(torch.bfloat16),
            rtol=0,
            atol=0,
        )
        metrics = result["capture_metrics"]
        self.assertEqual(metrics[f"{_CAPTURE_METRIC_PREFIX}publish_request_count"], 1)
        self.assertEqual(metrics[f"{_CAPTURE_METRIC_PREFIX}publish_token_count"], 3)
        self.assertCaptureStatus(metrics, enabled=1, broken=0)
        self.assertTrue(store.flushed)
        self.assertTrue(store.closed)

    def test_sp_capture_checks_physical_rows_before_logical_crop(self) -> None:
        layout_statuses = []

        def allreduce(tensor, op, mode, dest):
            self.assertEqual((op, mode, dest), (3, 0, None))
            layout_statuses.append(int(tensor.item()))
            return tensor

        _register_layout_allreduce(allreduce)
        self.addCleanup(clear_comm_ops)
        model = MalformedCaptureRowCountModel()
        model.requires_sequence_parallel_padding = True
        # Returning only the five logical rows is still an invalid physical output.
        result = run_scenario(model, "capture_bf16_tp")

        self.assertEqual(layout_statuses, [0])
        self.assertEqual(model.seen_input_lengths, [[2, 3, 1]])
        self.assertEqual(result["capture_failure_count"], 1)
        self.assertEqual(len(result["capture_diagnostics"]), 1)
        self.assertIn(
            "captured hidden-state row count 5 must match expected row count 6",
            result["capture_diagnostics"][0],
        )
        hidden = result["forward_hidden_states"][0]
        self.assertEqual(tuple(hidden.shape), (5, 1))
        torch.testing.assert_close(hidden, torch.zeros_like(hidden))
        self.assertEqual(_FakeEagleMooncakeStore.instances[0].puts, [])

    def test_sp_capture_remote_layout_failure_preserves_canonical_hidden(self) -> None:
        layout_statuses = []

        def allreduce(tensor, op, mode, dest):
            self.assertEqual((op, mode, dest), (3, 0, None))
            layout_statuses.append(int(tensor.item()))
            return torch.zeros_like(tensor)

        _register_layout_allreduce(allreduce)
        self.addCleanup(clear_comm_ops)
        model = SequenceParallelCaptureModel()
        result = run_scenario(model, "capture_bf16_tp")

        self.assertEqual(layout_statuses, [2])
        self.assertEqual(result["capture_failure_count"], 1)
        self.assertEqual(len(result["capture_diagnostics"]), 1)
        self.assertIn("another TP rank", result["capture_diagnostics"][0])
        hidden = result["forward_hidden_states"][0]
        self.assertEqual(tuple(hidden.shape), (5, 1))
        torch.testing.assert_close(
            hidden, torch.arange(31, 36, dtype=torch.float16).reshape(-1, 1)
        )
        self.assertEqual(_FakeEagleMooncakeStore.instances[0].puts, [])

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
                    self.assertTrue(
                        classifier_calls, "classifier injection was not reached"
                    )
                    self.assertTrue(publisher.fatal)
                    self.assertTrue(publisher.serving_healthy)
                    self.assertIn(
                        "injected original store poll failure", publisher.diagnostic
                    )
                    self.assertIsNone(publisher.reserve(2, 1))
                    self.assertEqual(store.put_attempts, 0)
                    self.assertEqual(store.keys, set())
                    self.assertEqual(publisher.inflight_samples, 1)
                    self.assertTrue(publisher.poll())
                    publisher.close()
                    self.assertEqual(
                        store.call_sequence, before + ["take_async_errors"]
                    )
                    self.assertFalse(store.closed)
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
                store.put_batch_calls[0]["capture_metadata"],
                {"capture_layer_ids": [7, 2]},
            )
            self.consumeCapture(publisher)
        finally:
            publisher.close()

    def test_capture_remote_source_holds_quota_after_flush_and_cancel(self) -> None:
        publisher = self.capturePublisher(samples=1, bytes=48, tokens=1)
        try:
            reservation = self.submitCapture(publisher)
            store = _FakeEagleMooncakeStore.instances[-1]
            expected = {
                "rtp_TestActor_1" + suffix
                for suffix in ("_hs", "_ids", "_lhs", "_ready")
            }
            self.assertEqual(store.keys, expected)
            self.assertEqual(
                store.query_keys(sorted(expected)), dict.fromkeys(expected, True)
            )
            self.assertTrue(store.flushed)
            reservation.cancel()
            self.assertEqual(publisher.inflight_samples, 1)
            self.assertEqual(publisher.inflight_bytes, 48)
            self.assertIsNone(publisher.reserve(2, 1))
            store.keys.remove("rtp_TestActor_1_ready")
            self.assertTrue(publisher.poll())
            self.assertEqual(publisher.inflight_samples, 1)
            self.assertEqual(store.reclaim_calls, [])
            self.assertEqual(store.removed_keys, [])
            self.consumeCapture(publisher)
            self.assertEqual(store.reclaim_calls, [])
            self.assertEqual(store.removed_keys, [])
            replacement = publisher.reserve(1, 1)
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
        publisher = self.capturePublisher()
        try:
            store = _FakeEagleMooncakeStore.instances[-1]
            store.keys.add("rtp_TestActor_1_ids")
            self.submitCapture(publisher)
            self.assertEqual(store.put_attempts, 0)
            self.assertEqual(store.keys, {"rtp_TestActor_1_ids"})
            self.assertIn("refusing to overwrite", publisher.diagnostic)
            self.assertEqual(publisher.inflight_samples, 0)
            self.assertFalse(publisher.fatal)
        finally:
            publisher.close()

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
            next_reservation = publisher.reserve(2, 1)
            self.assertIsNotNone(next_reservation)
            next_reservation.cancel()
        finally:
            publisher.close()

    def test_capture_preflight_query_exception_is_retryable(self) -> None:
        publisher = self.capturePublisher()
        try:
            with patch.object(
                _FakeEagleMooncakeStore,
                "query_keys",
                side_effect=RuntimeError("query unavailable"),
            ):
                self.submitCapture(publisher)
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
                    self.assertEqual(
                        _FakeEagleMooncakeStore.instances[-1].put_attempts, 0
                    )
                    self.assertEqual(publisher.inflight_samples, 0)
                    self.assertIn("query_keys", publisher.diagnostic)
                    self.assertIsNone(publisher.reserve(2, 1))
                finally:
                    publisher.close()

    def test_capture_cleanup_unknown_keeps_quota_and_retries(self) -> None:
        for invalid in ("exception", "integer", "missing", "wrong_keys"):
            with self.subTest(invalid=invalid):
                publisher = self.capturePublisher(samples=1, poll_interval_ms=60000)
                try:
                    self.submitCapture(publisher)
                    store = _FakeEagleMooncakeStore.instances[-1]
                    store.abandon_capture("rtp_TestActor_1")

                    def query(_store, keys):
                        if invalid == "exception":
                            raise RuntimeError("source query unavailable")
                        if invalid == "integer":
                            return dict.fromkeys(keys, 0)
                        if invalid == "wrong_keys":
                            return {"foreign_" + key: False for key in keys}
                        return dict.fromkeys(keys[:-1], False)

                    with patch.object(_FakeEagleMooncakeStore, "query_keys", query):
                        self.assertTrue(publisher.poll())
                        self.assertEqual(publisher.inflight_samples, 1)
                        self.assertIsNone(publisher.reserve(2, 1))
                        self.assertFalse(publisher.fatal)
                    self.assertTrue(store.keys)
                    self.assertEqual(store.reclaim_calls, [])
                    self.assertTrue(publisher.poll())
                    self.assertEqual(store.reclaim_calls, ["rtp_TestActor_1"])
                    self.assertEqual(store.keys, set())
                    self.assertEqual(publisher.inflight_samples, 0)
                    reservation = publisher.reserve(2, 1)
                    self.assertIsNotNone(reservation)
                    reservation.cancel()
                finally:
                    publisher.close()

    @patch.dict(os.environ, {"MOONCAKE_ASYNC_PUT_POOL_SIZE": "1"})
    def test_capture_partial_chunk_retains_attempted_prefix_not_suffix(self) -> None:
        publisher = self.capturePublisher()
        original_put = _FakeEagleMooncakeStore.put_batch

        def partial_put(store, **kwargs):
            accepted = original_put(store, **kwargs)
            if store.put_attempts == 2:
                raise RuntimeError("second chunk published then failed")
            return accepted

        try:
            reservations = [publisher.reserve(i, 1) for i in (1, 2, 3)]
            self.assertTrue(all(reservations))
            with patch.object(_FakeEagleMooncakeStore, "put_batch", partial_put):
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
            store = _FakeEagleMooncakeStore.instances[-1]
            self.assertEqual(store.put_attempts, 2)
            self.assertEqual(publisher.inflight_samples, 2)
            self.assertTrue(
                all(not key.startswith("rtp_TestActor_3_") for key in store.keys)
            )
            for reservation in reservations:
                reservation.cancel()
            self.assertEqual(publisher.inflight_samples, 2)
            suffix = publisher.reserve(3, 1)
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
                        _FakeEagleMooncakeStore,
                        "reclaim_abandoned_capture",
                        return_value=result,
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
            bases = [
                key.removesuffix("_abandoned")
                for key in store.query_calls[-1]
                if key.endswith("_abandoned")
            ]
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

            with patch.object(
                _FakeEagleMooncakeStore, "reclaim_abandoned_capture", reclaim
            ):
                self.assertTrue(publisher.poll())
                self.assertTrue(publisher.poll())
                self.assertEqual(attempts, [blocked, abandoned, blocked])
                self.assertEqual(publisher.inflight_samples, 1)
                self.assertEqual(publisher.inflight_bytes, 48)
                self.assertEqual(
                    store.keys,
                    {
                        blocked + suffix
                        for suffix in ("_hs", "_ids", "_lhs", "_ready", "_abandoned")
                    },
                )
                self.assertEqual(store.reclaim_calls, [abandoned])
                self.assertFalse(publisher.fatal)
            self.assertTrue(publisher.poll())
            self.assertEqual(store.reclaim_calls, [abandoned, blocked])
            self.assertEqual(publisher.inflight_samples, 0)
            self.assertEqual(store.keys, set())
        finally:
            publisher.close()

    def test_capture_reclaim_fatal_stops_later_cleanup_and_latches_owner(self) -> None:
        for error in (
            MemoryError("fatal reclaim allocation"),
            RuntimeError("CUSPARSE_STATUS_ALLOC_FAILED"),
        ):
            with self.subTest(error=str(error)):
                publisher = self.capturePublisher(poll_interval_ms=60000)
                try:
                    for request_id in (1, 2):
                        self.submitCapture(publisher, request_id)
                    store = _FakeEagleMooncakeStore.instances[-1]
                    bases = [
                        key.removesuffix("_abandoned")
                        for key in store.query_calls[-1]
                        if key.endswith("_abandoned")
                    ]
                    self.assertEqual(len(bases), 2)
                    for key in bases:
                        store.abandon_capture(key)
                    original_keys = set(store.keys)
                    attempts = []

                    def reclaim(_store, key):
                        _store.call_sequence.append(f"reclaim_abandoned_capture:{key}")
                        attempts.append(key)
                        raise error

                    with patch.object(
                        _FakeEagleMooncakeStore, "reclaim_abandoned_capture", reclaim
                    ):
                        self.assertTrue(publisher.poll())
                        self.assertTrue(publisher.fatal)
                        self.assertIsNone(publisher.reserve(3, 1))
                        self.assertEqual(publisher.inflight_samples, 2)
                        calls_at_failure = list(store.call_sequence)
                        self.assertEqual(
                            calls_at_failure[-2:],
                            ["query_keys", f"reclaim_abandoned_capture:{bases[0]}"],
                        )
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
        publisher = self.capturePublisher()
        self.submitCapture(publisher)
        store = _FakeEagleMooncakeStore.instances[-1]
        keys = set(store.keys)
        publisher.close()
        publisher.close()
        self.assertFalse(store.closed)
        self.assertEqual(store.keys, keys)
        self.assertIsNone(publisher.reserve(2, 1))

    def test_capture_initialization_runs_on_worker_without_blocking_constructor(
        self,
    ) -> None:
        entered = threading.Event()
        release = threading.Event()
        original_warmup = _FakeEagleMooncakeStore.warmup_rdma

        def warmup(store):
            entered.set()
            if not release.wait(10):
                raise RuntimeError("test did not release initialization")
            original_warmup(store)

        with patch.object(_FakeEagleMooncakeStore, "warmup_rdma", warmup):
            publisher = _extension.CapturePublisher()
            try:
                self.assertTrue(entered.wait(10))
                self.assertFalse(publisher.drain(timeout_ms=0))
                reservation = publisher.reserve(1, 1)
                self.assertIsNotNone(reservation)
                reservation.cancel()
            finally:
                release.set()
                publisher.close()
        self.assertTrue(_FakeEagleMooncakeStore.instances[-1].closed)

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
                        for name in (
                            "hidden_states",
                            "input_ids",
                            "last_hidden_states",
                        ):
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
                            key + suffix
                            for key in keys
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
                            raise RuntimeError(
                                "native PUT still pending; drain unconfirmed"
                            )
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
                        errors.append(
                            MemoryError("fatal shutdown completion; PUT state unknown")
                        )
                    return errors

                def close():
                    try:
                        publisher.close()
                    except Exception as error:
                        failures.append(str(error))

                with patch.object(
                    _FakeEagleMooncakeStore, "put_batch", pending_put
                ), patch.object(
                    _FakeEagleMooncakeStore, "flush", pending_flush
                ), patch.object(
                    _FakeEagleMooncakeStore, "take_async_errors", completion_errors
                ):
                    try:
                        reservation = publisher.reserve(1, 1)
                        self.assertIsNotNone(reservation)
                        packed = torch.tensor([[10, 20, 30]], dtype=torch.float16)
                        storage_refs.append(packed.untyped_storage()._weak_ref())
                        self.assertTrue(
                            publisher.submit(
                                packed,
                                torch.tensor([7]),
                                torch.tensor([1], dtype=torch.int32),
                                torch.tensor([1]),
                                1,
                                "pending native PUT at shutdown",
                            )
                        )
                        del packed
                        self.assertTrue(publisher.drain())
                        self.assertEqual(len(storage_refs), 4)
                        self.assertEqual(store.call_sequence[-2:], ["flush", "flush"])
                        store.abandon_capture("rtp_TestActor_1")
                        self.assertTrue(publisher.poll())
                        self.assertEqual(store.reclaim_calls, [])
                        self.assertFalse(write_done.is_set())
                        self.assertFalse(publisher.fatal)
                        self.assertEqual(publisher.inflight_samples, 1)
                        self.assertTrue(
                            all(
                                not torch.UntypedStorage._expired(ref)
                                for ref in storage_refs
                            )
                        )

                        shutdown_started.set()
                        closer = threading.Thread(target=close, daemon=True)
                        closer.start()
                        self.assertTrue(flush_entered.wait(10))
                        self.assertTrue(closer.is_alive())
                        self.assertEqual(publisher.inflight_samples, 1)
                        self.assertEqual(store.keys, {"rtp_TestActor_1_abandoned"})
                        self.assertEqual(store.removed_keys, [])
                        self.assertEqual(store.reclaim_calls, [])
                        self.assertTrue(
                            all(
                                not torch.UntypedStorage._expired(ref)
                                for ref in storage_refs
                            )
                        )
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
                            self.assertEqual(
                                store.call_sequence,
                                calls_in_flush + ["take_async_errors"],
                            )
                            self.assertTrue(
                                all(
                                    not torch.UntypedStorage._expired(ref)
                                    for ref in storage_refs
                                )
                            )
                            self.assertEqual(store.keys, {"rtp_TestActor_1_abandoned"})
                            self.assertEqual(store.removed_keys, [])
                            self.assertEqual(store.reclaim_calls, [])
                            self.assertFalse(store.closed)
                            for _ in range(2):
                                publisher.poll()
                                publisher.close()
                            self.assertEqual(
                                store.call_sequence,
                                calls_in_flush + ["take_async_errors"],
                            )
                            self.assertTrue(
                                all(
                                    not torch.UntypedStorage._expired(ref)
                                    for ref in storage_refs
                                )
                            )
                        else:
                            self.assertFalse(publisher.fatal)
                            self.assertEqual(
                                store.call_sequence,
                                calls_in_flush
                                + [
                                    "take_async_errors",
                                    "query_keys",
                                    "reclaim_abandoned_capture:rtp_TestActor_1",
                                    "close",
                                ],
                            )
                            self.assertEqual(store.reclaim_calls, ["rtp_TestActor_1"])
                            self.assertEqual(store.keys, set())
                            self.assertTrue(store.closed)
                            self.assertEqual(publisher.inflight_samples, 0)
                            self.assertEqual(publisher.inflight_bytes, 0)
                            self.assertTrue(
                                all(
                                    torch.UntypedStorage._expired(ref)
                                    for ref in storage_refs
                                )
                            )
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

    def test_capture_initialization_failure_drains_already_queued_payload(self) -> None:
        entered = threading.Event()
        release = threading.Event()

        def warmup(_store):
            entered.set()
            if not release.wait(10):
                raise RuntimeError("test did not release initialization")
            raise RuntimeError("queued initialization failure")

        with patch.object(_FakeEagleMooncakeStore, "warmup_rdma", warmup):
            publisher = _extension.CapturePublisher()
            try:
                self.assertTrue(entered.wait(10))
                reservation = publisher.reserve(1, 1)
                self.assertIsNotNone(reservation)
                self.assertTrue(
                    publisher.submit(
                        torch.zeros((1, 3), dtype=torch.float16),
                        torch.tensor([7]),
                        torch.tensor([1], dtype=torch.int32),
                        torch.tensor([1]),
                        1,
                        "queued init",
                    )
                )
                release.set()
                self.assertTrue(publisher.drain())
                self.assertEqual(publisher.inflight_samples, 0)
                self.assertIn("queued initialization failure", publisher.diagnostic)
                self.assertIsNone(publisher.reserve(2, 1))
                self.assertEqual(_FakeEagleMooncakeStore.instances[-1].put_attempts, 0)
                self.assertTrue(_FakeEagleMooncakeStore.instances[-1].closed)
            finally:
                release.set()
                publisher.close()

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
                below = self.capturePublisher(
                    bytes=expected_bytes - 1, model_dtype=dtype, fp8=fp8
                )
                exact = self.capturePublisher(
                    bytes=expected_bytes, model_dtype=dtype, fp8=fp8
                )
                try:
                    self.assertFalse(below.reserve(1, 2))
                    self.assertEqual(below.inflight_bytes, 0)
                    reservation = exact.reserve(1, 2)
                    self.assertIsNotNone(reservation)
                    self.assertEqual(exact.inflight_bytes, expected_bytes)
                    self.assertFalse(exact.reserve(2, 1))
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
                reservation = publisher.reserve(1, 2)
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
                self.assertFalse(publisher.reserve(2, 1))
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
        oversized = publisher.reserve(1, 2)
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
        owned = publisher.reserve(2, 2)
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

                _register_layout_allreduce(layout_reduce)
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
                self.assertCanonicalZero(result)
                self.assertEqual(result["capture_failure_count"], 0)
                self.assertEqual(_FakeEagleMooncakeStore.instances[-1].puts, [])

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
            result = run_scenario(CacheStoreForwardModel(), "capture_bf16", 2)
        snapshot = probe.wait()
        self.assertFalse(
            snapshot["entered"],
            "worker CPU-ID transfer invoked a RecordFunction callback",
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
        self.assertCaptureStatus(result["capture_metrics"], enabled=1, broken=0)

    def test_capture_reservation_limits_and_cancellation(self) -> None:
        # Only one resource is tight per row; no earlier limit masks tokens.
        for resource, limits in (
            ("samples", dict(samples=1, bytes=1_000_000, tokens=100)),
            ("bytes", dict(samples=10, bytes=76, tokens=100)),
            ("tokens", dict(samples=10, bytes=1_000_000, tokens=2)),
        ):
            with self.subTest(resource=resource):
                publisher = self.capturePublisher(**limits)
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
                    self.assertIsNone(publisher.reserve(-1, 2))
                    self.assertIsNone(publisher.reserve(1, 0))
                    self.assertIsNone(publisher.reserve(1, 2**63 - 1))
                    first = publisher.reserve(1, 2)
                    self.assertIsNotNone(first)
                    self.assertTrue(first.active)
                    self.assertIsNone(publisher.reserve(1, 2))
                    self.assertIsNone(publisher.reserve(2, 1))
                    self.assertEqual(publisher.inflight_samples, 1)
                    self.assertEqual(publisher.inflight_bytes, 76)
                    first.cancel()
                    first.cancel()
                    self.assertFalse(first.active)
                    self.assertEqual(publisher.inflight_samples, 0)
                    self.assertEqual(publisher.inflight_bytes, 0)
                    second = publisher.reserve(2, 2)
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
                    third = publisher.reserve(3, 2)
                    self.assertIsNotNone(third)
                    third.cancel()
                    self.assertEqual(publisher.inflight_samples, 0)
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

    def test_capture_reservation_generation_and_publisher_lifetime(self) -> None:
        publisher = self.capturePublisher(samples=1, bytes=48, tokens=1)
        try:
            first = publisher.reserve(7, 1)
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
            second = publisher.reserve(7, 1)
            self.assertIsNotNone(second)
            first.cancel()
            first.cancel()
            self.assertFalse(first.active)
            self.assertTrue(second.active)
            self.assertEqual(publisher.inflight_samples, 1)
            self.assertEqual(publisher.inflight_bytes, 48)
            self.assertIsNone(publisher.reserve(8, 1))
            second.cancel()
            self.assertEqual(publisher.inflight_samples, 0)
            self.assertEqual(publisher.inflight_bytes, 0)

            automatic = publisher.reserve(8, 1)
            self.assertIsNotNone(automatic)
            del automatic
            self.assertEqual(publisher.inflight_samples, 0)
            orphan = publisher.reserve(8, 1)
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
        packed = torch.tensor([[10, 20, 30]], dtype=torch.float16)

        reservations = {}

        def submit(request_id):
            reservations[request_id] = publisher.reserve(request_id, 1)
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
            finally:
                release.set()
            self.assertTrue(publisher.drain())
        self.assertEqual(publisher.inflight_samples, 2)
        self.consumeCapture(publisher)
        self.assertEqual(
            [put["key"] for put in _FakeEagleMooncakeStore.instances[-1].puts],
            ["rtp_TestActor_1", "rtp_TestActor_2"],
        )

    def test_capture_nondefault_producer_stream_and_async_pinned_lengths(self) -> None:
        publisher = self.capturePublisher()
        self.addCleanup(publisher.close)
        reservation = publisher.reserve(42, 2)
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
        baseline = run_scenario(
            CaptureContextForwardModel(hidden_size=2), "final_norm_parity_capture_off"
        )
        expected_hidden = baseline["forward_hidden_states"][0]
        self.assertGreater(torch.count_nonzero(expected_hidden).item(), 0)
        expected_logits = (
            expected_hidden[[1, 4]].float().sum(-1, keepdim=True).expand(-1, 2)
        )
        torch.testing.assert_close(baseline["logits"], expected_logits, rtol=0, atol=0)

        for stage in ("hook", "pack"):
            with self.subTest(stage=stage):
                model = CaptureContextForwardModel(hidden_size=2, failure_stage=stage)
                puts_after_forward = []

                def after_forward(_):
                    puts_after_forward.append(
                        len(_FakeEagleMooncakeStore.instances[-1].puts)
                    )

                result = run_scenario(
                    model,
                    "final_norm_parity_capture_on",
                    2,
                    after_forward=after_forward,
                    observe_aten=True,
                )
                self.assertCaptureLayoutProfile(result)
                self.assertEqual(model.capture_flags, [True, True])
                self.assertIn(
                    f"injected transient {stage} failure", model.capture_errors[0]
                )
                self.assertIsNone(model.capture_errors[1])
                self.assertEqual(puts_after_forward[0], 0)
                self.assertEqual(result["capture_admissions"], [True, True])
                self.assertEqual(result["forward_capture_inflight_samples"], [0, 2])
                self.assertEqual(result["forward_capture_inflight_bytes"][0], 0)
                self.assertGreater(result["forward_capture_inflight_bytes"][1], 0)
                self.assertEqual(len(result["forward_hidden_states"]), 2)
                self.assertEqual(len(result["forward_logits"]), 2)
                for hidden, logits in zip(
                    result["forward_hidden_states"], result["forward_logits"]
                ):
                    torch.testing.assert_close(hidden, expected_hidden, rtol=0, atol=0)
                    torch.testing.assert_close(logits, expected_logits, rtol=0, atol=0)
                torch.testing.assert_close(
                    result["selected_hidden_states"],
                    baseline["selected_hidden_states"],
                    rtol=0,
                    atol=0,
                )
                self.assertIn(
                    f"injected transient {stage} failure",
                    result["capture_diagnostics"][0],
                )
                self.assertFalse(result["capture_fatal"])
                self.assertTrue(result["serving_healthy"])
                self.assertEqual(result["capture_failure_count"], 1)
                self.assertEqual(result["capture_broken_rejection_count"], 0)
                self.assertCaptureStatus(result["capture_metrics"], enabled=1, broken=0)
                puts = _FakeEagleMooncakeStore.instances[-1].puts
                self.assertEqual(
                    [item["key"] for item in puts],
                    ["rtp_TestActor_501", "rtp_TestActor_502"],
                )
                for item, start, end in zip(puts, (0, 2), (2, 5)):
                    canonical = expected_hidden[start:end].to(torch.bfloat16)
                    torch.testing.assert_close(
                        item["last_hidden_states"], canonical, rtol=0, atol=0
                    )
                    torch.testing.assert_close(
                        item["hidden_states"],
                        torch.cat((canonical - 20, canonical - 10), dim=-1),
                        rtol=0,
                        atol=0,
                    )
                    self.assertEqual(
                        item["input_ids"].tolist(), list(range(start + 1, end + 1))
                    )

    def test_capture_worker_fatal_is_sticky_owner_health_not_next_batch_error(
        self,
    ) -> None:
        def out_of_memory(_store, **_kwargs):
            _store.call_sequence.append("put_batch")
            raise MemoryError("injected capture worker allocator failure")

        model = CacheStoreForwardModel()
        with patch.object(_FakeEagleMooncakeStore, "put_batch", out_of_memory):
            result = run_scenario(model, "capture_bf16", 2)
        self.assertEqual(model.capture_flags, [True, True])
        self.assertTrue(result["capture_fatal"])
        self.assertTrue(result["serving_healthy"])
        self.assertIn(
            "injected capture worker allocator failure", result["capture_fatal_message"]
        )
        self.assertEqual(result["capture_failure_count"], 1)
        for hidden in result["forward_hidden_states"]:
            self.assertEqual(tuple(hidden.shape), (5, 1))
            torch.testing.assert_close(hidden, torch.zeros_like(hidden), rtol=0, atol=0)
        store = _FakeEagleMooncakeStore.instances[-1]
        self.assertEqual(store.puts, [])
        self.assertEqual(
            store.call_sequence,
            ["setup", "warmup_rdma", "take_async_errors", "query_keys", "put_batch"],
        )
        self.assertFalse(store.closed)
        self.assertCaptureStatus(result["capture_metrics"], enabled=0, broken=1)

    def test_capture_async_completion_fatal_latches_owner_health(self) -> None:
        original_poll = _FakeEagleMooncakeStore.take_async_errors
        poll_count = 0

        def fatal_completion(store):
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
            _FakeEagleMooncakeStore, "take_async_errors", fatal_completion
        ):
            result = run_scenario(model, "capture_bf16", 2)
        self.assertTrue(result["capture_fatal"])
        self.assertTrue(result["serving_healthy"])
        self.assertIn(
            "allocator failure in original capture batch",
            result["capture_fatal_message"],
        )
        self.assertEqual(model.capture_flags, [True, True])
        for hidden in result["forward_hidden_states"]:
            self.assertEqual(tuple(hidden.shape), (5, 1))
            torch.testing.assert_close(hidden, torch.zeros_like(hidden), rtol=0, atol=0)
        store = _FakeEagleMooncakeStore.instances[-1]
        self.assertEqual(len(store.puts), 2)
        self.assertEqual(
            store.call_sequence,
            [
                "setup",
                "warmup_rdma",
                "take_async_errors",
                "query_keys",
                "put_batch",
                "flush",
                "take_async_errors",
            ],
        )
        self.assertFalse(store.closed)

    def test_capture_tp_tristate_preserves_canonical_output_and_recovers(self) -> None:
        # Keep plan consensus intact; inject only the post-forward MIN result.
        for local, remote in ((2, 1), (1, 2), (1, 1), (1, 0), (0, 1), (2, 0), (0, 2)):
            reduced = min(local, remote)
            forward_count = 2 if reduced == 1 else 1
            for owner, scenario in (
                (True, "capture_bf16_tp"),
                (False, "capture_non_owner"),
            ):
                with self.subTest(owner=owner, local=local, remote=remote):
                    model = CaptureContextForwardModel(
                        failure_stage={0: "layout", 1: "hook", 2: ""}[local]
                    )
                    votes = []
                    store_count = len(_FakeEagleMooncakeStore.instances)

                    def reduce(tensor, op, mode, dest):
                        self.assertEqual((op, mode, dest), (3, 0, None))
                        self.assertEqual(int(tensor.item()), local if not votes else 2)
                        peer_status = remote if not votes else 2
                        votes.append(int(tensor.item()))
                        return torch.minimum(
                            tensor, torch.full_like(tensor, peer_status)
                        )

                    _register_layout_allreduce(reduce)
                    try:
                        result = run_scenario(
                            model, scenario, forward_count, observe_aten=True
                        )
                    finally:
                        clear_comm_ops()
                    self.assertEqual(votes, [local] + [2] * (forward_count - 1))
                    self.assertCaptureLayoutProfile(result, distributed=True)
                    self.assertTrue(result["serving_healthy"])
                    self.assertFalse(result["capture_fatal"])
                    self.assertEqual(result["forward_capture_inflight_samples"][0], 0)
                    self.assertEqual(result["forward_capture_inflight_bytes"][0], 0)
                    for hidden in result["forward_hidden_states"]:
                        torch.testing.assert_close(
                            hidden,
                            torch.arange(31, 36, dtype=torch.float16).reshape(-1, 1),
                            rtol=0,
                            atol=0,
                        )
                    self.assertEqual(result["forward_logits"], [None] * forward_count)
                    stores = _FakeEagleMooncakeStore.instances[store_count:]
                    if not owner:
                        self.assertEqual(stores, [])
                        continue
                    self.assertEqual(result["capture_failure_count"], 1)
                    self.assertCaptureStatus(
                        result["capture_metrics"],
                        enabled=int(reduced == 1),
                        broken=int(reduced == 0),
                    )
                    if reduced == 0:
                        self.assertEqual(stores[0].puts, [])
                        self.assertIn(
                            (
                                "width"
                                if local == 0
                                else "layout contract violation on another TP rank"
                            ),
                            result["capture_diagnostics"][0],
                        )
                    else:
                        self.assertEqual(result["capture_admissions"], [True, True])
                        self.assertEqual(
                            result["forward_capture_inflight_samples"], [0, 2]
                        )
                        self.assertGreater(
                            result["forward_capture_inflight_bytes"][1], 0
                        )
                        self.assertEqual(result["capture_broken_rejection_count"], 0)
                        self.assertEqual(len(stores[0].puts), 2)
                        self.assertIn(
                            (
                                "injected transient hook failure"
                                if local == 1
                                else "recoverable hidden-state capture failure on another TP rank"
                            ),
                            result["capture_diagnostics"][0],
                        )

    def test_capture_producer_cuda_oom_is_not_swallowed(self) -> None:
        class FatalForward(CacheStoreForwardModel):
            def forward(self, inputs, fmha_impl=None):
                raise torch.OutOfMemoryError("CUDA out of memory in canonical forward")

        with self.assertRaisesRegex(
            torch.OutOfMemoryError, "CUDA out of memory in canonical forward"
        ):
            run_scenario(FatalForward(), "capture_bf16")
        self.assertEqual(_FakeEagleMooncakeStore.instances[-1].puts, [])
        for stage in ("hook", "pack"):
            for error in (
                MemoryError("injected allocator failure"),
                torch.OutOfMemoryError("injected device allocator failure"),
                RuntimeError("CUDA error: injected illegal memory access"),
            ):
                with self.subTest(stage=stage, error=type(error).__name__):
                    with self.assertRaisesRegex(type(error), re.escape(str(error))):
                        run_scenario(
                            CaptureContextForwardModel(
                                failure_stage=stage, error=error
                            ),
                            "capture_bf16",
                        )
                    self.assertEqual(_FakeEagleMooncakeStore.instances[-1].puts, [])
        with self.assertRaisesRegex(RuntimeError, "canonical final failed"):
            run_scenario(
                CaptureContextForwardModel(
                    failure_stage="final", error=RuntimeError("canonical final failed")
                ),
                "capture_bf16",
            )
        self.assertEqual(_FakeEagleMooncakeStore.instances[-1].puts, [])

    def test_capture_failed_put_drains_partial_staging_before_release(self) -> None:
        cases = [
            ("partial_put", "capture_bf16", 1),
            ("partial_put", "capture_bf16", 2),
            ("metadata_count", "capture_bf16", 2),
            ("metadata_shape", "capture_bf16", 2),
        ]
        if os.environ.get("TEST_USING_DEVICE") != "ROCM":
            cases.append(("metadata_scale", "capture_fp8", 2))
        original_put = _FakeEagleMooncakeStore.put_batch
        original_flush = _FakeEagleMooncakeStore.flush
        for failure, scenario, fail_chunk in cases:
            request_id = 500 + fail_chunk
            with self.subTest(failure=failure, chunk=fail_chunk), patch.dict(
                os.environ, {"MOONCAKE_ASYNC_PUT_POOL_SIZE": "1"}
            ):
                entered = threading.Event()
                release = threading.Event()
                staged = []
                staging_stream = torch.cuda.Stream()

                def partial_put(store, **kwargs):
                    metadata = original_put(store, **kwargs)
                    if store.put_attempts < fail_chunk:
                        return metadata
                    self.assertEqual(store.put_attempts, fail_chunk)
                    self.assertEqual(
                        kwargs["request_ids"], [f"rtp_TestActor_{request_id}"]
                    )
                    source = kwargs["hidden_states"][0]
                    # A later PUT must reset completion of the flushed prefix
                    # before it can start foreign-stream staging.
                    staging_stream.wait_stream(torch.cuda.current_stream())
                    with torch.cuda.stream(staging_stream):
                        staged.append(source.clone())
                    if failure == "partial_put":
                        raise RuntimeError("partial PUT after staging copy")
                    if failure == "metadata_count":
                        return []
                    name = (
                        "hidden_states_scale"
                        if failure == "metadata_scale"
                        else "hidden_states"
                    )
                    metadata[0]["shapes"][name] = (99, 2)
                    return metadata

                def blocked_flush(store):
                    if staged:
                        entered.set()
                        if not release.wait(10):
                            raise RuntimeError("partial PUT drain not released")
                        staging_stream.synchronize()
                    return original_flush(store)

                def after_forward(_index):
                    try:
                        self.assertTrue(entered.wait(10))
                        # Encourage allocator reuse while recovery is draining.
                        churn = [
                            torch.full((3, 2), -7, device="cuda", dtype=torch.bfloat16)
                            for _ in range(32)
                        ]
                        self.assertEqual(len(churn), 32)
                    finally:
                        release.set()

                with patch.object(
                    _FakeEagleMooncakeStore, "put_batch", partial_put
                ), patch.object(_FakeEagleMooncakeStore, "flush", blocked_flush):
                    result = run_scenario(
                        CacheStoreForwardModel(), scenario, after_forward=after_forward
                    )
                self.assertCanonicalZero(result)
                self.assertEqual(result["capture_failure_count"], 1)
                diagnostic = result["capture_diagnostics"][0]
                self.assertIn(
                    f"chunk-{fail_chunk}-requests-{request_id}-{request_id}", diagnostic
                )
                self.assertIn(
                    f"flush-confirmed prefix: {fail_chunk - 1}/2 requests, "
                    f"{2 if fail_chunk == 2 else 0}/5 tokens",
                    diagnostic,
                )
                if failure == "partial_put":
                    self.assertIn("partial PUT after staging copy", diagnostic)
                else:
                    self.assertIn("metadata", diagnostic)
                    if failure != "metadata_count":
                        self.assertIn(
                            "key=rtp_TestActor_502, request_id=502", diagnostic
                        )
                self.assertEqual(len(staged), 1)
                torch.testing.assert_close(
                    staged[0].float().cpu(),
                    torch.tensor(
                        [[10, 20]] * (2 if fail_chunk == 1 else 3), dtype=torch.float32
                    ),
                    rtol=0,
                    atol=0,
                )
                store = _FakeEagleMooncakeStore.instances[-1]
                self.assertEqual(
                    [call["request_ids"] for call in store.put_batch_calls],
                    [["rtp_TestActor_501"], ["rtp_TestActor_502"]][:fail_chunk],
                )
                self.assertFalse(store.closed)
                self.assertEqual(result["capture_inflight_samples"], 0)
                self.assertEqual(result["capture_inflight_bytes"], 0)
                self.assertCaptureQps(
                    result["capture_metrics"],
                    batch=1,
                    publish_success=0,
                    failure=1,
                    store_failure=1,
                    operational_failure=int(failure == "partial_put"),
                    hard_contract_failure=int(failure != "partial_put"),
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
            reservation = publisher.reserve(1, 1)
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
            unsubmitted = publisher.reserve(2, 1)
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
        self.assertFalse(publisher.reserve(2, 1))

    def test_capture_failed_flush_retains_bounded_quota(self) -> None:
        publisher = self.capturePublisher(samples=1, bytes=48, tokens=1)
        store_calls = []
        with patch.object(_FakeEagleMooncakeStore, "fail_flush", True), patch.object(
            _FakeEagleMooncakeStore, "events", store_calls
        ):
            try:
                reservation = publisher.reserve(1, 1)
                self.assertIsNotNone(reservation)
                self.assertTrue(
                    publisher.submit(
                        torch.tensor([[10, 20, 30]], dtype=torch.float16),
                        torch.tensor([7]),
                        torch.tensor([1], dtype=torch.int32),
                        torch.tensor([1]),
                        1,
                        "failed flush",
                    )
                )
                self.assertTrue(publisher.drain())
                self.assertEqual(publisher.inflight_samples, 1)
                self.assertEqual(publisher.inflight_bytes, 48)
                self.assertFalse(publisher.reserve(2, 1))
                self.assertIn("flush failure", publisher.diagnostic)
                self.assertFalse(publisher.fatal)
                store = _FakeEagleMooncakeStore.instances[-1]
                store.abandon_capture("rtp_TestActor_1")
                source_keys = set(store.keys)
                self.assertTrue(publisher.poll())
                self.assertEqual(store.reclaim_calls, [])
                self.assertEqual(store.removed_keys, [])
                self.assertEqual(store.keys, source_keys)
                calls_at_drain = list(store_calls)
                reservation.cancel()
                self.assertTrue(publisher.drain())
                self.assertEqual(store_calls, calls_at_drain)
                self.assertEqual(publisher.inflight_samples, 1)
                self.assertEqual(publisher.inflight_bytes, 48)
            finally:
                publisher.close()
        self.assertFalse(_FakeEagleMooncakeStore.instances[-1].closed)

    @patch.dict(os.environ, {"MOONCAKE_ASYNC_PUT_POOL_SIZE": "1"})
    def test_capture_bf16_store_contract_slices_each_request_and_flushes_on_shutdown(
        self,
    ) -> None:
        _register_no_collectives()
        self.addCleanup(clear_comm_ops)
        result = run_scenario(
            CacheStoreForwardModel(), "capture_bf16", observe_aten=True
        )
        self.assertCaptureLayoutProfile(result)

        self.assertEqual(result["hidden_width"], 1)
        self.assertEqual(result["capture_failure_count"], 0)
        self.assertEqual(result["capture_broken_rejection_count"], 0)
        metrics = result["capture_metrics"]
        self.assertCaptureMetricTypes(result)
        self.assertEqual(
            set(metrics),
            _CAPTURE_QPS_METRICS | _CAPTURE_STATUS_METRICS | _CAPTURE_PUBLISH_METRICS,
        )
        self.assertCaptureQps(
            metrics,
            batch=1,
            publish_success=1,
            failure=0,
            bf16_publish=1,
        )
        self.assertCaptureStatus(metrics, enabled=1, broken=0)
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
        key_calls = _FakeMooncakeConfig.make_store_key_calls
        self.assertEqual(key_calls[:5], [0, 501, 502, 501, 502])
        # Cleanup polls also resolve each candidate through the namespace API.
        # Their count is timer-dependent, but every complete poll covers both IDs.
        cleanup_calls = key_calls[5:]
        self.assertGreaterEqual(len(cleanup_calls), 2)
        self.assertEqual(set(cleanup_calls), {501, 502})
        self.assertEqual(cleanup_calls.count(501), cleanup_calls.count(502))
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
            self.assertCaptureDisabled(
                result, "TorchSpec hidden-state capture does not support FP8 on ROCm"
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
        self.assertCaptureStatus(metrics, enabled=1, broken=0)
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

    def test_capture_micro_batch_merges_packed_width_before_publish(self) -> None:
        model = CacheStoreForwardModel()
        result = run_scenario(model, "capture_micro_batch")

        self.assertEqual(model.forward_calls, 0)
        self.assertEqual(model.micro_batch_calls, 1)
        self.assertEqual(result["hidden_width"], 1)
        self.assertCaptureQps(
            result["capture_metrics"],
            batch=1,
            publish_success=1,
            failure=0,
            bf16_publish=1,
        )
        self.assertEqual(
            [
                put["input_ids"].tolist()
                for put in _FakeEagleMooncakeStore.instances[0].puts
            ],
            [[1, 2], [3, 4, 5]],
        )

    def test_capture_disabled_micro_batch_ignores_canonical_fake_lane(self) -> None:
        model = CacheStoreForwardModel()

        result = run_scenario(model, "capture_disabled_micro_batch_fake_lane")

        self.assertEqual(model.forward_calls, 0)
        self.assertEqual(model.micro_batch_calls, 1)
        self.assertEqual(model.capture_flags, [True, True])
        self.assertEqual(model.seen_input_lengths, [[2], [2]])
        self.assertEqual(result["hidden_width"], 1)
        self.assertCaptureQps(
            result["capture_metrics"],
            batch=1,
            publish_success=1,
            failure=0,
            bf16_publish=1,
        )
        puts = _FakeEagleMooncakeStore.instances[0].puts
        self.assertEqual(len(puts), 1)
        self.assertEqual(puts[0]["input_ids"].tolist(), [1, 2])

    def test_disabled_micro_batch_zero_tokens_does_not_narrow_fake_lane(self) -> None:
        model = CacheStoreForwardModel()

        result = run_scenario(model, "disabled_micro_batch_zero_tokens")

        self.assertEqual(model.forward_calls, 0)
        self.assertEqual(model.micro_batch_calls, 1)
        self.assertEqual(model.capture_flags, [False, False])
        self.assertEqual(model.seen_input_lengths, [[0], [0]])
        self.assertEqual(_FakeEagleMooncakeStore.instances, [])
        self.assertEqual(tuple(result["forward_hidden_states"][0].shape), (0, 1))

    def test_micro_batch_final_hidden_numeric_parity_capture_on_off(self) -> None:
        os.environ["MOONCAKE_HIDDEN_DIM"] = "2"
        capture_off_model = FinalNormParityModel(hidden_size=2)
        capture_on_model = FinalNormParityModel(hidden_size=2)

        capture_off_result = run_scenario(
            capture_off_model, "micro_batch_final_norm_parity_capture_off"
        )
        capture_on_result = run_scenario(
            capture_on_model, "micro_batch_final_norm_parity_capture_on"
        )

        self.assertTrue(capture_off_model.capture_flags)
        self.assertTrue(capture_on_model.capture_flags)
        self.assertTrue(all(flag is False for flag in capture_off_model.capture_flags))
        self.assertTrue(all(flag is True for flag in capture_on_model.capture_flags))
        capture_off_hidden = capture_off_result["forward_hidden_states"][0]
        capture_on_hidden = capture_on_result["forward_hidden_states"][0]
        self.assertEqual(tuple(capture_off_hidden.shape), (5, 2))
        self.assertEqual(tuple(capture_on_hidden.shape), (5, 2))
        torch.testing.assert_close(
            capture_on_hidden,
            capture_off_hidden,
            rtol=0,
            atol=0,
        )

    def test_capture_micro_batch_count_bit_and_tristate_consensus(self) -> None:
        class MicroBatchModel(CaptureContextForwardModel):
            def __init__(self, failure):
                super().__init__()
                self.failure = failure

            def forward_micro_batch(self, inputs):
                self.micro_batch_calls += 1
                self.forward_calls = self.micro_batch_calls
                stages = {
                    "ok": ("", ""),
                    "recoverable": ("hook", ""),
                    "hard": ("layout", ""),
                    "canonical": ("", ""),
                    "recoverable_hard": ("hook", "layout"),
                    "hard_recoverable": ("layout", "hook"),
                }[self.failure]
                outputs = []
                if len(inputs) != len(stages):
                    raise AssertionError(
                        "consensus fixture requires two micro-batch lanes"
                    )
                for model_inputs, stage in zip(inputs, stages):
                    self.failure_stage = stage
                    self.layer_hook_calls = 0
                    outputs.append(self._forward_one(model_inputs))
                if self.failure == "canonical":
                    return outputs[:-1]
                return outputs

        # The first field remains canonical output-count validity, not capture state.
        statuses = {
            "ok": [1, 2],
            "recoverable": [1, 1],
            "hard": [1, 0],
            "canonical": [0, 0],
            "recoverable_hard": [1, 0],
            "hard_recoverable": [1, 0],
        }
        for failures in (
            ("ok", "canonical"),
            ("ok", "hard"),
            ("canonical", "hard"),
            ("ok", "ok"),
            ("ok", "recoverable"),
            ("recoverable", "ok"),
            ("recoverable", "recoverable"),
            ("recoverable", "hard"),
            ("hard", "recoverable"),
            ("recoverable_hard", "ok"),
            ("hard_recoverable", "ok"),
        ):
            reduced = min(statuses[failure][1] for failure in failures)
            forward_count = 2 if reduced == 1 else 1
            for rank, scenario in enumerate(
                ("capture_micro_batch_tp", "capture_micro_batch_non_owner")
            ):
                with self.subTest(failures=failures, rank=rank):
                    model = MicroBatchModel(failures[rank])
                    votes = []
                    store_count = len(_FakeEagleMooncakeStore.instances)

                    def reduce(tensor, op, mode, dest):
                        self.assertEqual((op, mode, dest), (3, 0, None))
                        if tensor.dtype == torch.int64:
                            self.assertEqual(model.micro_batch_calls, len(votes))
                            return tensor
                        self.assertEqual(model.micro_batch_calls, len(votes) + 1)
                        expected = statuses[failures[rank]] if not votes else [1, 2]
                        remote = statuses[failures[1 - rank]] if not votes else [1, 2]
                        self.assertEqual(tensor.tolist(), expected)
                        votes.append(tensor.tolist())
                        return torch.minimum(tensor, tensor.new_tensor(remote))

                    _register_no_collectives(reduce)
                    try:
                        if "canonical" in failures:
                            with self.assertRaisesRegex(
                                Exception, "py_model_outputs.size"
                            ):
                                run_scenario(model, scenario)
                        else:
                            result = run_scenario(model, scenario, forward_count)
                            for hidden in result["forward_hidden_states"]:
                                torch.testing.assert_close(
                                    hidden,
                                    torch.arange(31, 36, dtype=torch.float16).reshape(
                                        -1, 1
                                    ),
                                    rtol=0,
                                    atol=0,
                                )
                            self.assertTrue(result["serving_healthy"])
                            self.assertFalse(result["capture_fatal"])
                            self.assertEqual(
                                result["forward_capture_inflight_samples"][0],
                                2 if rank == 0 and reduced == 2 else 0,
                            )
                            if reduced != 2:
                                self.assertEqual(
                                    result["forward_capture_inflight_bytes"][0], 0
                                )
                            if rank == 0:
                                self.assertEqual(
                                    result["capture_failure_count"], int(reduced != 2)
                                )
                                self.assertCaptureStatus(
                                    result["capture_metrics"],
                                    enabled=int(reduced != 0),
                                    broken=int(reduced == 0),
                                )
                                if reduced == 1:
                                    self.assertEqual(
                                        result["capture_admissions"], [True, True]
                                    )
                                    self.assertEqual(
                                        result["forward_capture_inflight_samples"],
                                        [0, 2],
                                    )
                                    self.assertGreater(
                                        result["forward_capture_inflight_bytes"][1], 0
                                    )
                                    self.assertEqual(
                                        result["capture_broken_rejection_count"], 0
                                    )
                    finally:
                        clear_comm_ops()
                    self.assertEqual(
                        votes,
                        [statuses[failures[rank]]] + [[1, 2]] * (forward_count - 1),
                    )
                    stores = _FakeEagleMooncakeStore.instances[store_count:]
                    if rank == 0:
                        self.assertEqual(len(stores), 1)
                        self.assertEqual(len(stores[0].puts), 2 if reduced != 0 else 0)
                    else:
                        self.assertEqual(stores, [])

    def test_capture_micro_batch_output_count_failure_remains_canonical_error(
        self,
    ) -> None:
        _register_identity_allreduce()
        self.addCleanup(clear_comm_ops)
        # Missing canonical lanes are a model failure, not just a capture error.
        with self.assertRaisesRegex(Exception, "py_model_outputs.size"):
            run_scenario(MalformedCaptureOutputCountModel(), "capture_micro_batch_tp")
        self.assertEqual(_FakeEagleMooncakeStore.instances[0].puts, [])

    def test_capture_micro_batch_layout_failure_preserves_canonical_output(
        self,
    ) -> None:
        _register_identity_allreduce()
        self.addCleanup(clear_comm_ops)
        result = run_scenario(MalformedCaptureLayoutModel(), "capture_micro_batch_tp")
        self.assertEqual(result["capture_failure_count"], 1)
        self.assertIn(
            "width 2 must match expected packed width 3",
            result["capture_diagnostics"][0],
        )
        hidden = result["forward_hidden_states"][0]
        self.assertEqual(tuple(hidden.shape), (5, 1))
        torch.testing.assert_close(hidden, torch.zeros_like(hidden))
        self.assertEqual(_FakeEagleMooncakeStore.instances[0].puts, [])

    def test_capture_context_parallel_and_micro_batch_is_explicitly_unsupported(
        self,
    ) -> None:
        _register_identity_allreduce()
        self.addCleanup(clear_comm_ops)
        model = CacheStoreForwardModel()
        result = run_scenario(model, "capture_cp_micro_batch")
        self.assertIn(
            "context parallel combined with layer micro-batching",
            result["capture_diagnostics"][0],
        )
        self.assertEqual(model.micro_batch_calls, 1)
        self.assertEqual(tuple(result["forward_hidden_states"][0].shape), (5, 1))
        self.assertEqual(_FakeEagleMooncakeStore.instances, [])

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
        _register_identity_allreduce()
        self.addCleanup(clear_comm_ops)
        result = run_scenario(
            CacheStoreForwardModel(), "capture_cp_invalid_num_valid_tokens"
        )
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

        _register_layout_allreduce(allreduce)
        self.addCleanup(clear_comm_ops)
        result = run_scenario(
            MalformedCaptureLayoutModel(),
            "capture_cp_malformed_layout",
        )

        self.assertEqual(layout_statuses, [0])
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
        # fixture. This checks the existing single MIN, not real multi-GPU CP.
        for owner in (True, False):
            for local, remote in (
                (2, 1),
                (1, 2),
                (1, 1),
                (1, 0),
                (0, 1),
                (2, 0),
                (0, 2),
            ):
                with self.subTest(owner=owner, local=local, remote=remote):
                    reduced = min(local, remote)
                    forward_count = 2 if reduced == 1 else 1
                    calls = []

                    def allreduce(tensor, op, mode, dest):
                        peer_status = remote if not calls else 2
                        calls.append((int(tensor.item()), op, mode, dest))
                        return torch.minimum(
                            tensor, torch.full_like(tensor, peer_status)
                        )

                    _register_layout_allreduce(allreduce)
                    store_count = len(_FakeEagleMooncakeStore.instances)
                    model = CaptureContextForwardModel(
                        failure_stage={0: "layout", 1: "hook", 2: ""}[local]
                    )
                    try:
                        result = run_scenario(
                            model,
                            "capture_cp" if owner else "capture_cp_non_owner",
                            forward_count,
                            observe_aten=True,
                        )
                    finally:
                        clear_comm_ops()
                    self.assertEqual(
                        calls,
                        [(local, 3, 0, None)] + [(2, 3, 0, None)] * (forward_count - 1),
                    )
                    self.assertCaptureLayoutProfile(result, distributed=True)
                    self.assertEqual(
                        result["cp_handle_outputs_calls"], 1 + 2 * (forward_count - 1)
                    )
                    self.assertTrue(result["serving_healthy"])
                    self.assertFalse(result["capture_fatal"])
                    self.assertEqual(result["forward_capture_inflight_samples"][0], 0)
                    self.assertEqual(result["forward_capture_inflight_bytes"][0], 0)
                    for hidden in result["forward_hidden_states"]:
                        self.assertEqual(tuple(hidden.shape), (6, 1))
                        self.assertGreater(torch.count_nonzero(hidden).item(), 0)
                    for hidden in result["forward_hidden_states"][1:]:
                        torch.testing.assert_close(
                            hidden, result["forward_hidden_states"][0], rtol=0, atol=0
                        )
                    self.assertEqual(result["forward_logits"], [None] * forward_count)
                    stores = _FakeEagleMooncakeStore.instances[store_count:]
                    if not owner:
                        self.assertEqual(stores, [])
                        continue
                    self.assertEqual(result["capture_failure_count"], 1)
                    if reduced == 0:
                        self.assertCaptureStatus(
                            result["capture_metrics"], enabled=0, broken=1
                        )
                        self.assertEqual(stores[0].puts, [])
                        self.assertIn(
                            (
                                "width"
                                if local == 0
                                else "layout contract violation on another TP rank"
                            ),
                            result["capture_diagnostics"][0],
                        )
                    else:
                        self.assertCaptureStatus(
                            result["capture_metrics"], enabled=1, broken=0
                        )
                        self.assertEqual(result["capture_admissions"], [True, True])
                        self.assertEqual(
                            result["forward_capture_inflight_samples"], [0, 1]
                        )
                        self.assertEqual(len(stores[0].puts), 1)
                        self.assertIn(
                            (
                                "injected transient hook failure"
                                if local == 1
                                else "recoverable hidden-state capture failure on another TP rank"
                            ),
                            result["capture_diagnostics"][0],
                        )

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

        _register_layout_allreduce(allreduce)
        self.addCleanup(clear_comm_ops)
        result = run_scenario(
            CacheStoreForwardModel(),
            "capture_cp_malformed_restored_width",
        )

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

        _register_layout_allreduce(allreduce)
        self.addCleanup(clear_comm_ops)
        result = run_scenario(
            MalformedCaptureRankModel(),
            "capture_bf16_tp",
        )

        self.assertEqual(layout_statuses, [0])
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
            # Simulate a non-owner TP rank reporting a malformed capture.
            return torch.zeros_like(tensor)

        _register_layout_allreduce(allreduce)
        self.addCleanup(clear_comm_ops)
        result = run_scenario(
            CacheStoreForwardModel(),
            "capture_bf16_tp",
            observe_aten=True,
        )

        self.assertEqual(calls, 1)
        self.assertCaptureLayoutProfile(result, distributed=True)
        self.assertEqual(result["capture_failure_count"], 1)
        self.assertIn("another TP rank", result["capture_diagnostics"][0])
        self.assertEqual(_FakeEagleMooncakeStore.instances[0].puts, [])
        self.assertCanonicalZero(result)

    @_skip_rocm_context_parallel
    def test_capture_context_parallel_uses_original_ids_and_lengths(self) -> None:
        if os.environ.get("TEST_USING_DEVICE") == "ROCM":
            self.assertRocmContextParallelRejected(
                PositionAwareCaptureModel(), "capture_cp"
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

        _register_layout_allreduce(allreduce)
        self.addCleanup(clear_comm_ops)
        model = PositionAwareCaptureModel()
        result = run_scenario(model, "capture_cp")

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
                "store.close",
            ],
        )
        self.assertEqual(result["cp_handle_outputs_calls"], 2)
        self.assertEqual(model.seen_input_lengths, [[4]])
        self.assertEqual(result["hidden_width"], 1)
        puts = _FakeEagleMooncakeStore.instances[0].puts
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
            torch.tensor([[1], [2], [3], [4], [5], [6]], dtype=torch.float16),
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
        self.assertCaptureStatus(metrics, enabled=1, broken=0)

    def test_capture_cpu_prepare_failure_only_rejects_each_current_batch(self) -> None:
        _register_identity_allreduce()
        self.addCleanup(clear_comm_ops)
        model = CacheStoreForwardModel()

        result = run_scenario(model, "capture_prepare_failure_tp", 3)

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
        self.assertCaptureStatus(metrics, enabled=1, broken=0)
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
        # One observation of the async error and one runtime failure.
        self.assertEqual(result["capture_failure_count"], 2)
        self.assertCaptureQps(
            result["capture_metrics"],
            batch=1,
            publish_success=0,
            failure=2,
            store_failure=2,
            operational_failure=2,
        )
        self.assertCaptureStatus(result["capture_metrics"], enabled=1, broken=0)

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
                    RuntimeError(f"historical async failure for {batch_id}")
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
        self.assertCaptureStatus(result["capture_metrics"], enabled=1, broken=0)

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

        result = run_scenario(CacheStoreForwardModel(), "capture_fp8")
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
        self.assertCaptureStatus(result["capture_metrics"], enabled=1, broken=0)

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
        _register_identity_allreduce()
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

        self.assertEqual(model.capture_flags, [True, False, False])
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

    def test_capture_rejects_duplicate_layer_ids_at_startup(self) -> None:
        model = CacheStoreForwardModel()
        result = run_scenario(model, "capture_duplicate_layer_id")
        self.assertCaptureDisabled(result, "layer ids must be unique; duplicate id 0")
        self.assertEqual(model.capture_flags, [False])
        self.assertEqual(_FakeEagleMooncakeStore.instances, [])

    def test_capture_rejects_invalid_dtype_at_startup(self) -> None:
        model = CacheStoreForwardModel()
        result = run_scenario(model, "capture_invalid_dtype")
        self.assertCaptureDisabled(result, "unsupported hidden-state capture dtype: 99")
        self.assertEqual(model.capture_flags, [False])
        self.assertEqual(_FakeEagleMooncakeStore.instances, [])

    def test_capture_rejects_unsupported_python_model_at_startup(self) -> None:
        model = UnsupportedCaptureModel()
        result = run_scenario(model, "capture_bf16")
        self.assertCaptureDisabled(result, "does not support hidden-state capture")
        self.assertEqual(model.capture_flags, [False])
        self.assertEqual(_FakeEagleMooncakeStore.instances, [])

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

    def test_capture_publisher_layout_with_core_dump(self) -> None:
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
                            f"{type(self).__name__}.test_capture_publisher_layout_with_core_dump",
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
        with missing_method_patch, patch.object(
            _FakeMooncakeConfig, "from_env", tracked_from_env
        ):
            result = initialize_capture_publisher_with_core_dump()
        self.assertTrue(result["core_dump_enabled"])
        self.assertTrue(result["core_dump_restored"])
        self.assertCaptureMetricTypes(result)
        metrics = result["capture_metrics"]
        self.assertEqual(set(metrics), _CAPTURE_QPS_METRICS | _CAPTURE_STATUS_METRICS)
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
            enabled=1 - failed,
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
        result = run_scenario(CacheStoreForwardModel(), "capture_bf16")
        self.assertCaptureDisabled(result, "does not match RTP capture layer count")
        self.assertEqual(_FakeEagleMooncakeStore.instances, [])

    def test_capture_publisher_rejects_hidden_dim_mismatch(
        self,
    ) -> None:
        os.environ["MOONCAKE_HIDDEN_DIM"] = "2"
        result = run_scenario(CacheStoreForwardModel(), "capture_bf16")
        self.assertCaptureDisabled(result, "does not match RTP hidden size")
        self.assertEqual(_FakeEagleMooncakeStore.instances, [])

    def test_capture_publisher_requires_batch_and_async_error_contract_methods(
        self,
    ) -> None:
        for method_name in (
            "put_batch",
            "query_keys",
            "reclaim_abandoned_capture",
            "take_async_errors",
        ):
            with self.subTest(method_name=method_name), patch.object(
                _FakeEagleMooncakeStore, method_name, None
            ):
                result = run_scenario(CacheStoreForwardModel(), "capture_bf16")
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

        self.assertEqual(model.capture_flags, [False])
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

    def test_capture_is_disabled_on_ffn_service_model(self) -> None:
        model = CacheStoreForwardModel()
        result = run_scenario(model, "capture_ffn_service")

        self.assertEqual(model.capture_flags, [False])
        self.assertEqual(_FakeEagleMooncakeStore.instances, [])
        hidden_states = result["forward_hidden_states"][0]
        torch.testing.assert_close(hidden_states, torch.zeros_like(hidden_states))

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
        self.assertCaptureStatus(result["capture_metrics"], enabled=1, broken=0)

    def test_capture_unconsumed_request_id_is_not_readmitted(self) -> None:
        _register_identity_allreduce()
        self.addCleanup(clear_comm_ops)
        model = CacheStoreForwardModel()
        result = run_scenario(
            model,
            "capture_bf16_tp",
            3,
        )
        store = _FakeEagleMooncakeStore.instances[0]
        self.assertEqual(model.capture_flags, [True, True, True])
        self.assertEqual(result["capture_admissions"], [True, False, False])
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
            result["capture_metrics"],
            batch=3,
            publish_success=1,
            failure=0,
            bf16_publish=1,
        )
        self.assertCaptureStatus(result["capture_metrics"], enabled=1, broken=0)

    def test_capture_rejects_packed_layout(self) -> None:
        model = MalformedCaptureLayoutModel()
        result = run_scenario(
            model,
            "capture_bf16",
            3,
        )

        self.assertEqual(model.capture_flags, [True, False, False])
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
        )
        self.assertCaptureStatus(metrics, enabled=0, broken=1)
        self.assertEqual(_FakeEagleMooncakeStore.instances[0].puts, [])

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
        self.assertCaptureStatus(result["capture_metrics"], enabled=1, broken=0)

    def test_capture_store_admission_failure_recovers_on_next_batch(
        self,
    ) -> None:
        _register_identity_allreduce()
        self.addCleanup(clear_comm_ops)
        _FakeEagleMooncakeStore.fail_put_count = 1
        model = CacheStoreForwardModel()
        result = run_scenario(model, "capture_bf16_tp", 2)

        store = _FakeEagleMooncakeStore.instances[0]
        self.assertEqual(model.capture_flags, [True, True])
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
        self.assertCaptureStatus(metrics, enabled=1, broken=0)
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
            buffers_module = importlib.import_module(
                "torchspec.transfer.mooncake.buffers"
            )
        except (ImportError, OSError) as error:
            self.skipTest(f"real TorchSpec capture API is unavailable: {error}")
        self.assertIsNotNone(mooncake_config)
        self.assertIsNotNone(mooncake_store)
        self.assertTrue(callable(quantize_aux_hidden_states))

        for method_name in (
            "put_batch",
            "query_keys",
            "reclaim_abandoned_capture",
            "take_async_errors",
            "flush",
            "close",
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
        self.assertSignatureBinds(
            mooncake_store.query_keys, object(), ["contract-key_hs"]
        )
        self.assertSignatureBinds(
            mooncake_store.abandon_capture, object(), "contract-key"
        )
        self.assertSignatureBinds(
            mooncake_store.reclaim_abandoned_capture, object(), "contract-key"
        )
        self.assertSignatureBinds(quantize_aux_hidden_states, object(), 2)
        self.assertEqual(
            mooncake_store.__module__, "torchspec.transfer.mooncake.eagle_store"
        )
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
                        b""
                        if key.startswith("_warmup_")
                        else ctypes.string_at(ptr, size)
                    )
                self.puts.append(
                    {
                        "keys": list(keys),
                        "ptrs": list(ptrs),
                        "sizes": list(sizes),
                        "payloads": payloads,
                        "thread": threading.current_thread().name,
                    }
                )
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
                store=store,
                pool=pool,
                buffer=buffer,
                manager=manager,
                ptr=buffer.ptr,
                size=buffer.size,
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
            rtol=0,
            atol=0,
        )
        self.assertEqual(
            native.registrations, [(initialized["ptr"], initialized["size"])]
        )
        self.assertEqual(len(native.setup_calls), 1)
        self.assertEqual(native.setup_calls[0]["protocol"], "tcp")
        warmup = [put for put in native.puts if put["keys"][0].startswith("_warmup_")]
        self.assertEqual(len(warmup), 1)
        self.assertEqual(native.removals, [warmup[0]["keys"]])
        puts = [put for put in native.puts if not put["keys"][0].startswith("_warmup_")]
        keys = [f"rtp_ContractActor1_{request_id}" for request_id in (501, 502)]
        self.assertEqual(
            [put["keys"] for put in puts],
            [
                [keys[0] + suffix for suffix in ("_hs", "_ids", "_lhs")],
                [keys[0] + "_ready"],
                [keys[1] + suffix for suffix in ("_hs", "_ids", "_lhs")],
                [keys[1] + "_ready"],
            ],
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
                [
                    bytes(tensor.contiguous().view(torch.uint8).reshape(-1).tolist())
                    for tensor in expected
                ],
            )
            self.assertEqual(ready["sizes"], [len(ready["payloads"][0])])
            self.assertIsInstance(json.loads(ready["payloads"][0]), dict)

        metrics = result["capture_metrics"]
        self.assertEqual(
            set(metrics),
            _CAPTURE_QPS_METRICS | _CAPTURE_STATUS_METRICS | _CAPTURE_PUBLISH_METRICS,
        )
        self.assertEqual(
            result["capture_metric_types"],
            {
                name: (
                    _KMONITOR_QPS
                    if name in _CAPTURE_QPS_METRICS
                    else (
                        _KMONITOR_STATUS
                        if name in _CAPTURE_STATUS_METRICS
                        else _KMONITOR_GAUGE
                    )
                )
                for name in metrics
            },
        )
        expected_qps = dict.fromkeys(_CAPTURE_QPS_METRICS, 0)
        for name in ("batch", "publish_success", "bf16_publish"):
            expected_qps[f"{_CAPTURE_METRIC_PREFIX}{name}_qps"] = 1
        self.assertEqual(
            {name: metrics[name] for name in _CAPTURE_QPS_METRICS}, expected_qps
        )
        for name, expected in {
            "enabled": 1,
            "broken": 0,
            "publish_request_count": 2,
            "publish_token_count": 5,
            "publish_payload_bytes": 70,
            "publish_input_ids_bytes": 40,
            "publish_aux_hidden_bytes": 20,
            "publish_last_hidden_bytes": 10,
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
    micro_batch_final_norm = True

    def hidden(self, inputs):
        return (
            inputs.input_ids.unsqueeze(1) * 4 + torch.arange(4, device="cuda") + 1
        ).to(torch.bfloat16)

    def forward_micro_batch(self, inputs):
        return [PyModelOutputs(rms_norm(self.hidden(part))) for part in inputs]

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


class MtpDecodeForwardModel(SuccessfulGenerationPrefillCaptureModel):
    def __init__(self, inspect_padding=False, inspect_metadata=False):
        super().__init__()
        self.forward_calls = 0
        self.inspect_padding = inspect_padding
        self.inspect_metadata = inspect_metadata
        self.fail_next_prepare = False
        if inspect_padding:
            # The C++ constructor must capture SP geometry before creating
            # its graph runner; toggling its test peer afterward is too late.
            self.requires_sequence_parallel_padding = True

    def prepare_fmha_impl(
        self, inputs, is_cuda_graph=False, cuda_graph_selection_mode=None
    ):
        super().prepare_fmha_impl(inputs, is_cuda_graph, cuda_graph_selection_mode)
        return self if self.inspect_metadata else None

    def prepare_cuda_graph(self, _inputs):
        if self.fail_next_prepare:
            self.fail_next_prepare = False
            raise RuntimeError("injected Q1 prepare failure")

    def forward(self, inputs, fmha_impl=None):
        self.forward_calls += 1
        hidden = (
            inputs.input_ids.unsqueeze(1) * 4 + torch.arange(4, device="cuda") + 1
        ).to(torch.bfloat16)
        if self.inspect_padding:
            hidden = hidden + inputs.input_ids[-1].to(hidden.dtype)
            if inputs.input_hiddens is not None and inputs.input_hiddens.numel():
                hidden = hidden + inputs.input_hiddens
                # A stale nonzero dummy row changes live outputs too. The test
                # primes that row before the first unaligned replay.
                hidden = hidden + inputs.input_hiddens[-1].sum()
        return PyModelOutputs(rms_norm(hidden), hidden + 100)


@unittest.skipUnless(torch.cuda.is_available() and torch.version.cuda, "requires CUDA")
class MtpDecodeForwardIntegrationTest(unittest.TestCase):
    def run_case(self, scenario):
        tagged_q1 = scenario == "multi_tag_q1_prepare"
        sp_unaligned = tagged_q1 or scenario in (
            "sp_unaligned",
            "sp_unaligned_inline_prepare",
        )
        result = _extension.run_mtp_decode_forward(
            MtpDecodeForwardModel(
                inspect_padding=sp_unaligned, inspect_metadata=tagged_q1
            ),
            scenario,
        )
        # Both user forwards must replay graphs, without invoking Python again.
        self.assertEqual(
            result["python_calls_before"], result["python_calls_after_first"]
        )
        self.assertEqual(
            result["python_calls_before"], result["python_calls_after_second"]
        )
        self.assertTrue(result["normal_snapshots_distinct"])
        self.assertTrue(result["recurrent_snapshots_distinct"])
        rows = 6 if scenario == "context" else 1 if sp_unaligned else 2
        for prefix, first_token in (("first_", 1), ("next_", 4)):
            tokens = torch.arange(first_token, first_token + rows, device="cuda")
            hidden = (tokens.unsqueeze(1) * 4 + torch.arange(4, device="cuda") + 1).to(
                torch.bfloat16
            )
            if sp_unaligned:
                expected_input = torch.arange(
                    10, 14, device="cuda", dtype=torch.bfloat16
                ).reshape(1, 4)
                if prefix == "next_":
                    expected_input = expected_input + 5
                torch.testing.assert_close(
                    result[prefix + "input_hiddens"], expected_input, rtol=0, atol=0
                )
                hidden = hidden + expected_input
            # Inspect after the second replay: first-round tensors must still
            # contain their own normalized and recurrent values.
            torch.testing.assert_close(
                result[prefix + "normalized"], rms_norm(hidden), rtol=0, atol=0
            )
            torch.testing.assert_close(
                result[prefix + "recurrent"], hidden + 100, rtol=0, atol=0
            )
        return result

    def test_unaligned_sp_padding_zeroes_old_token_and_hidden_tail(self):
        result = self.run_case("sp_unaligned")
        self.assertIsNone(result["first_logits"])

    def test_q1_inline_prepare_replays_graph_without_external_prepare(self):
        # The fixture omits both prepareAttentionInputs and updateKVCacheKernelBlockId
        # before the two B=1 forwards, matching the real Q1 executor entry.
        result = self.run_case("sp_unaligned_inline_prepare")
        self.assertIsNone(result["first_logits"])

    def test_multitag_q1_metadata_is_published_and_recovers_after_prepare_failure(self):
        result = self.run_case("multi_tag_q1_prepare")
        self.assertFalse(result["published_after_failure"])
        for prefix, length in (("first_", 4), ("next_", 7)):
            metadata = result[prefix + "tag_metadata"]
            self.assertEqual(set(metadata), {"full", "aux"})
            for tag_inputs in metadata.values():
                self.assertEqual(
                    tag_inputs["sequence_lengths_plus_1"].tolist(), [length, 1]
                )
                self.assertEqual(tag_inputs["decode_cu_seqlens"].tolist(), [0, 1, 2])
                self.assertEqual(tag_inputs["valid_token_mask"].tolist(), [True, False])
        metadata = result["fallback_tag_metadata"]
        self.assertEqual(set(metadata), {"common", "full", "aux"})
        for tag_inputs in metadata.values():
            self.assertEqual(
                tag_inputs["sequence_lengths_plus_1"].tolist(), [7, 7, 7, 1]
            )
            self.assertEqual(tag_inputs["decode_cu_seqlens"].tolist(), [0, 1, 2, 3, 4])
            self.assertEqual(
                tag_inputs["valid_token_mask"].tolist(), [True, True, True, False]
            )

    def test_ordinary_all_logits_debug_and_context_keep_dense_logits(self):
        for scenario in ("default", "all_logits", "check_nan", "context"):
            with self.subTest(scenario=scenario):
                result = self.run_case(scenario)
                for prefix in ("first_", "next_"):
                    logits = result[prefix + "logits"]
                    self.assertEqual(tuple(logits.shape), (2, 4))
                    expected = result[prefix + "normalized"].float()
                    if scenario == "context":
                        expected = expected[[2, 5]]
                    torch.testing.assert_close(logits, expected, rtol=0, atol=0)


if __name__ == "__main__":
    unittest.main()
