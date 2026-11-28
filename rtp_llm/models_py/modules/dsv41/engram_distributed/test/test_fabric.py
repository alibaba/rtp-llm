# SPDX-License-Identifier: Apache-2.0
"""Packaged native import, runner safety, and opt-in CUDA Fabric integration."""

import base64
import fcntl
import gc
import hashlib
import importlib
import io
import os
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch
from uuid import uuid4

from rtp_llm.models_py.modules.dsv41.engram_distributed.runtime import _load_native
from rtp_llm.models_py.modules.dsv41.engram_distributed.test import (
    run_fabric_ci as runner,
)

LIVE_KEYS = (
    "contexts",
    "owner_allocations",
    "imports",
    "reservations",
    "mappings",
    "view_storages",
)


class PackagedNativeTest(unittest.TestCase):
    def test_packaged_bridge_import_and_abi_without_cuda_initialization(self):
        import torch

        self.assertFalse(torch.cuda.is_initialized())
        import rtp_llm.ops as ops

        self.assertNotIn("libfabric_memory", sys.modules)
        self.assertNotIn("librtp_compute_ops", sys.modules)
        self.assertNotIn("libth_transformer", sys.modules)
        native = _load_native()
        self.assertIs(native, ops.load_engram_ops())
        self.assertIs(native, importlib.import_module("libfabric_memory"))
        self.assertEqual(native.__name__, "libfabric_memory")
        self.assertNotIn("rtp_llm.libs.libfabric_memory", sys.modules)
        path = Path(native.__file__)
        self.assertEqual(path.name, "libfabric_memory.so")
        self.assertEqual(path.parent.name, "libs")
        self.assertTrue(callable(native.check_fabric_devices))
        self.assertTrue(callable(native.device_numa_nodes))
        self.assertTrue(callable(native.device_granularity))
        self.assertTrue(callable(native.host_numa_granularity))
        self.assertTrue(callable(native.FabricAllocation))
        self.assertTrue(callable(native.FabricMapping))
        self.assertGreater(native.fabric_handle_bytes, 0)
        self.assertTrue(
            all(value == 0 for value in native.resource_counters().values())
        )
        self.assertFalse(torch.cuda.is_initialized())

    def test_optional_bridge_does_not_block_regular_ops_import(self):
        script = """
import importlib.abc
import sys

class MissingEngram(importlib.abc.MetaPathFinder):
    def find_spec(self, fullname, path=None, target=None):
        if fullname == "libfabric_memory":
            raise ModuleNotFoundError("Engram intentionally absent", name=fullname)

sys.meta_path.insert(0, MissingEngram())
import torch
import rtp_llm.ops
from rtp_llm.models_py.modules.dsv41.engram_distributed.runtime import (
    BootstrapError, _load_native,
)
assert "libfabric_memory" not in sys.modules
assert "librtp_compute_ops" not in sys.modules
assert "libth_transformer" not in sys.modules
assert not torch.cuda.is_initialized()
try:
    _load_native()
except BootstrapError as error:
    assert error.code == "PREBUILT_ENGRAM_NATIVE_MODULE_REQUIRED", error.code
else:
    raise AssertionError("missing Engram unexpectedly loaded")
assert not torch.cuda.is_initialized()
"""
        result = subprocess.run(
            [sys.executable, "-c", script],
            capture_output=True,
            text=True,
            timeout=60,
        )
        self.assertEqual(result.returncode, 0, result.stdout + result.stderr)

    def test_validation_and_empty_mapping_lifecycle_without_cuda(self):
        native = _load_native()
        before = dict(native.resource_counters())
        with self.assertRaisesRegex(ValueError, "memory_kind"):
            native.FabricMapping(0, "host_shared")
        with self.assertRaisesRegex(ValueError, "device ordinal"):
            native.FabricMapping(-1)
        with self.assertRaisesRegex(ValueError, "handle_type"):
            native.FabricMapping(0, "host", handle_type="posix_fd")
        fd_mapping = native.FabricMapping(0, "gpu", handle_type="posix_fd")
        for value in (True, -1, 2**31, b"1"):
            with self.subTest(fd=value), self.assertRaises(ValueError):
                fd_mapping.attach([(value, 4096, 4096)])
            self.assertEqual(dict(native.resource_counters()), before)
        fd_mapping.close()
        handle = b"x" * native.fabric_handle_bytes
        mapping = native.FabricMapping(0)
        self.assertEqual(mapping.info()["state"], "NEW")
        with self.assertRaisesRegex(ValueError, "not ready"):
            mapping.tensor_view([1])
        for segments, error in (
            ([], "at least one"),
            ([(handle, 4096)], "each segment"),
            ([(bytearray(handle), 4096, 4096)], "must be bytes"),
            ([(handle, True, 4096)], "must be integers"),
            ([(handle, 4096, False)], "must be integers"),
            ([(handle, 0, 4096)], "allocation_bytes"),
            ([(handle, 4096, 3)], "granularity"),
            ([(handle, 4095, 4096)], "aligned"),
            ([(handle, 4096, 4096), (handle, 8192, 8192)], "segment offset"),
            ([(handle, 4096, 4096), (b"x", 4096, 4096)], "byte length"),
        ):
            with self.subTest(error=error), self.assertRaisesRegex(ValueError, error):
                mapping.attach(segments)
            self.assertEqual(dict(native.resource_counters()), before)
        for value in (-1, 1 << 64):
            with self.assertRaises(OverflowError):
                mapping.attach([(handle, value, 4096)])
        mapping.close()
        mapping.close()
        self.assertEqual(mapping.info()["state"], "CLOSED")
        with self.assertRaisesRegex(ValueError, "not new"):
            mapping.attach([(handle, 4096, 4096)])
        self.assertEqual(dict(native.resource_counters()), before)


class NativeStorageTests(unittest.TestCase):
    native = None
    torch = None
    device = None
    numa = None

    def setUp(self):
        if any(
            value is None for value in (self.native, self.torch, self.device, self.numa)
        ):
            raise RuntimeError("native tests require fixtures from native_test_suite")
        gc.collect()
        self.torch.cuda.synchronize(self.device)
        self.baseline = dict(self.native.resource_counters())
        self.assertEqual(
            {key: self.baseline[key] for key in LIVE_KEYS},
            {key: 0 for key in LIVE_KEYS},
        )
        self.assertEqual(self.baseline["deferred_cleanups"], 0)
        # Registered first => runs after all owner/consumer closes in this test.
        self.addCleanup(self.assert_resources_restored)

    def assert_resources_restored(self):
        self.torch.cuda.synchronize(self.device)
        gc.collect()
        self.assertEqual(dict(self.native.resource_counters()), self.baseline)

    def test_posix_segments_hold_cuda_references_after_fd_close(self):
        native, torch = self.native, self.torch
        owners, descriptors, handles = [], [], []
        mapping = native.FabricMapping(self.device, "gpu", handle_type="posix_fd")
        try:
            for value in (1.25, -2.5):
                owner = native.FabricAllocation(
                    4096, -1, self.device, memory_kind="gpu", handle_type="posix_fd"
                )
                owners.append(owner)
                owner.load(
                    torch.tensor([value], dtype=torch.bfloat16)
                    .view(torch.uint8)
                    .numpy()
                    .tobytes()
                )
                fd = owner.export_fd()
                handles.append(fd)
                self.assertFalse(os.get_inheritable(fd))
                info = owner.info()
                self.assertEqual(info["handle_type"], "POSIX_FD")
                descriptors.append((fd, info["allocation_bytes"], info["granularity"]))
                with self.assertRaisesRegex(ValueError, "sealed"):
                    owner.load(b"x")
                with self.assertRaisesRegex(ValueError, "export_fd"):
                    owner.export_handle()
            mapping.attach(list(reversed(descriptors)))
            for fd in handles:
                os.close(fd)
            handles.clear()
            view = mapping.tensor_view([1], dtype=torch.bfloat16)
            tail = mapping.tensor_view([1], descriptors[1][1], dtype=torch.bfloat16)
            padding = mapping.tensor_view([32], 2)
            try:
                self.assertEqual(view.item(), -2.5)
                self.assertEqual(tail.item(), 1.25)
                self.assertEqual(padding.cpu().tolist(), [0] * 32)
                with self.assertRaisesRegex(ValueError, "live tensor"):
                    mapping.close()
            finally:
                del view, tail, padding
        finally:
            mapping.close()
            for fd in handles:
                os.close(fd)
            for owner in reversed(owners):
                owner.close()

    def test_posix_second_segment_failure_releases_cuda_refs_without_closing_fd(self):
        owner = self.native.FabricAllocation(
            4096, -1, self.device, memory_kind="gpu", handle_type="posix_fd"
        )
        self.addCleanup(owner.close)
        fd = owner.export_fd()
        self.addCleanup(os.close, fd)
        info = owner.info()
        segments = [(fd, info["allocation_bytes"], info["granularity"])] * 2
        baseline = dict(self.native.resource_counters())
        for stage in ("after_import", "after_properties", "after_map", "after_access"):
            with self.subTest(stage=stage):
                mapping = self.native.FabricMapping(
                    self.device, "gpu", handle_type="posix_fd"
                )
                self.addCleanup(mapping.close)
                with self.assertRaisesRegex(RuntimeError, stage):
                    mapping.attach(segments, _fail_stage=stage, _fail_segment=1)
                self.assertEqual(mapping.info()["state"], "CLOSED")
                self.assertEqual(dict(self.native.resource_counters()), baseline)
                os.fstat(
                    fd
                )  # The caller retains ownership on both success and failure.

    def test_gpu_backing_readback_graph_and_memory_kind_guard(self):
        owner = self.native.FabricAllocation(4096, -1, self.device, memory_kind="gpu")
        payload = b"fabric-gpu-single-copy"
        owner.load(payload)
        info = owner.info()
        handle = owner.export_handle()
        self.addCleanup(owner.close)
        self.assertEqual(info["memory_kind"], "gpu")
        self.assertEqual(
            info["granularity"], self.native.device_granularity(self.device)
        )
        self.assertEqual(info["gpu_access"], "READWRITE")
        with self.assertRaisesRegex(ValueError, "memory kind"):
            self.attach_handle(
                handle, info["allocation_bytes"], info["granularity"], self.device
            )
        consumer = self.attach_handle(
            handle,
            info["allocation_bytes"],
            info["granularity"],
            self.device,
            memory_kind="gpu",
        )
        self.addCleanup(consumer.close)
        view = consumer.tensor_view([64])
        graph = self.torch.cuda.CUDAGraph()
        try:
            self.assertEqual(
                bytes(view[: len(payload) + 1].cpu().tolist()), payload + b"\x00"
            )
            stream = self.torch.cuda.Stream(device=self.device)
            stream.wait_stream(self.torch.cuda.current_stream(self.device))
            with self.torch.cuda.graph(graph, stream=stream):
                output = view.clone()
            for _ in range(10):
                graph.replay()
                self.assertEqual(bytes(output[: len(payload)].cpu().tolist()), payload)
            del output
        finally:
            graph.reset()
            del graph, view
            gc.collect()

    def test_gpu_construction_failures_release_every_partial_resource(self):
        baseline = dict(self.native.resource_counters())
        for stage in (
            "after_context",
            "after_create",
            "after_reserve",
            "after_map",
            "after_access",
            "after_zero",
        ):
            with self.subTest(role="owner", stage=stage), self.assertRaisesRegex(
                RuntimeError, "TEST_FAULT_INJECTION"
            ):
                self.native.FabricAllocation(
                    4096, -1, self.device, fail_stage=stage, memory_kind="gpu"
                )
            gc.collect()
            self.assertEqual(dict(self.native.resource_counters()), baseline)
        owner = self.native.FabricAllocation(4096, -1, self.device, memory_kind="gpu")
        self.addCleanup(owner.close)
        info, handle = owner.info(), owner.export_handle()
        baseline = dict(self.native.resource_counters())
        for stage in (
            "after_context",
            "after_import",
            "after_properties",
            "after_reserve",
            "after_map",
            "after_access",
        ):
            with self.subTest(role="consumer", stage=stage), self.assertRaisesRegex(
                RuntimeError, "TEST_FAULT_INJECTION"
            ):
                self.attach_handle(
                    handle,
                    info["allocation_bytes"],
                    info["granularity"],
                    self.device,
                    fail_stage=stage,
                    memory_kind="gpu",
                )
            gc.collect()
            self.assertEqual(dict(self.native.resource_counters()), baseline)

    def owner(self, size=256):
        value = self.native.FabricAllocation(
            size=size, numa=self.numa, device=self.device
        )
        self.addCleanup(value.close)
        return value

    def attach_handle(
        self,
        handle,
        allocation_bytes,
        granularity,
        device,
        *,
        fail_stage="",
        memory_kind="host",
    ):
        mapping = self.native.FabricMapping(device, memory_kind)
        mapping.attach(
            [(handle, allocation_bytes, granularity)], _fail_stage=fail_stage
        )
        return mapping

    @staticmethod
    def segment(allocation):
        info = allocation.info()
        return allocation.export_handle(), info["allocation_bytes"], info["granularity"]

    def consumer(self, owner):
        description = owner.info()
        value = self.attach_handle(
            handle=owner.export_handle(),
            allocation_bytes=description["allocation_bytes"],
            granularity=description["granularity"],
            device=self.device,
        )
        self.addCleanup(value.close)
        return value

    def drain_and_collect(self):
        self.torch.cuda.synchronize(self.device)
        gc.collect()

    def test_device_and_granularity_queries_have_no_allocation_or_context_leak(self):
        before = dict(self.native.resource_counters())
        count = self.torch.cuda.device_count()
        nodes = self.native.device_numa_nodes()
        self.assertEqual(len(nodes), count)
        self.assertTrue(all(type(node) is int for node in nodes))
        self.native.check_fabric_devices([self.device])
        for ordinal in (-1, count):
            with self.assertRaises(IndexError):
                self.native.check_fabric_devices([ordinal])
        self.assertEqual(dict(self.native.resource_counters()), before)

        granularity = self.native.host_numa_granularity(self.numa, self.device)
        self.assertGreater(granularity, 0)
        self.assertEqual(dict(self.native.resource_counters()), before)
        owner = self.owner()
        self.assertEqual(owner.info()["granularity"], granularity)
        owner.close()
        for numa, device in ((-1, self.device), (self.numa, -1)):
            with self.assertRaises(ValueError):
                self.native.host_numa_granularity(numa, device)
            self.assertEqual(dict(self.native.resource_counters()), before)

    def test_owner_bounds_seal_and_repeated_export(self):
        before = dict(self.native.resource_counters())
        self.assertEqual(self.native.max_allocation_bytes, 128 << 30)
        for invalid_size in (0, self.native.max_allocation_bytes + 1, (1 << 64) - 1):
            with self.subTest(size=invalid_size):
                with self.assertRaises(ValueError):
                    self.native.FabricAllocation(
                        size=invalid_size, numa=self.numa, device=self.device
                    )
                self.assertEqual(dict(self.native.resource_counters()), before)

        owner = self.owner(size=32)
        description = owner.info()
        self.assertEqual(description["logical_bytes"], 32)
        self.assertGreaterEqual(description["allocation_bytes"], 32)
        self.assertEqual(
            description["allocation_bytes"] % description["granularity"], 0
        )
        self.assertFalse(description["sealed"])
        owner.load(b"abcd", offset=0)
        owner.load(b"XY", offset=30)
        owner.load(b"", offset=32)  # Legal empty chunk at the logical boundary.
        for data, offset in ((b"x", 32), (b"XYZ", 30), (b"", 33)):
            with self.subTest(offset=offset, byte_count=len(data)):
                with self.assertRaises(ValueError):
                    owner.load(data, offset=offset)
        before_export = dict(self.native.resource_counters())
        for _ in range(3):
            handle = owner.export_handle()
            self.assertIsInstance(handle, bytes)
            # Compare lengths, never include handle contents in an assertion/report.
            self.assertEqual(len(handle), self.native.fabric_handle_bytes)
            self.assertEqual(dict(self.native.resource_counters()), before_export)
        self.assertTrue(owner.info()["sealed"])
        with self.assertRaisesRegex(ValueError, "sealed"):
            owner.load(b"new", offset=0)

        consumer = self.consumer(owner)
        view = None
        try:
            view = consumer.tensor_view([32])
            expected = b"abcd" + bytes(26) + b"XY"
            self.assertEqual(bytes(view.cpu().tolist()), expected)
        finally:
            view = None
            self.drain_and_collect()
        consumer.close()
        owner.close()
        owner.close()  # Explicit idempotence.
        self.assertEqual(owner.info()["state"], "CLOSED")
        self.assertEqual(owner.info()["base_address"], 0)

    def test_malformed_handle_lengths_are_rejected_before_driver(self):
        before = dict(self.native.resource_counters())
        size = self.native.fabric_handle_bytes
        for byte_count in (0, 1, size - 1, size + 1):
            with self.subTest(byte_count=byte_count):
                with self.assertRaisesRegex(ValueError, "byte length"):
                    self.attach_handle(b"x" * byte_count, 2097152, 2097152, self.device)
                self.assertEqual(dict(self.native.resource_counters()), before)

    def test_owner_constructor_rollback(self):
        baseline = dict(self.native.resource_counters())
        for stage in (
            "after_context",
            "after_create",
            "after_reserve",
            "after_map",
            "after_access",
        ):
            with self.subTest(fail_stage=stage):
                with self.assertRaisesRegex(
                    RuntimeError, f"TEST_FAULT_INJECTION: {stage}"
                ):
                    self.native.FabricAllocation(
                        256, self.numa, self.device, fail_stage=stage
                    )
                gc.collect()
                self.assertEqual(dict(self.native.resource_counters()), baseline)

    def test_single_segment_attach_rollback_and_descriptor_validation(self):
        owner = self.owner()
        handle = owner.export_handle()
        description = owner.info()
        baseline = dict(self.native.resource_counters())
        self.assertEqual(baseline["owner_allocations"], 1)
        for stage in (
            "after_context",
            "after_import",
            "after_properties",
            "after_reserve",
            "after_map",
            "after_access",
        ):
            with self.subTest(fail_stage=stage):
                with self.assertRaisesRegex(
                    RuntimeError, f"TEST_FAULT_INJECTION: {stage}"
                ):
                    self.attach_handle(
                        handle,
                        description["allocation_bytes"],
                        description["granularity"],
                        self.device,
                        fail_stage=stage,
                    )
                gc.collect()
                self.assertEqual(dict(self.native.resource_counters()), baseline)

        # All are safe descriptor checks. Never corrupt an actual driver handle.
        bad_descriptors = [
            (0, description["granularity"]),
            (description["allocation_bytes"], 0),
            (self.native.max_allocation_bytes + 1, description["granularity"]),
        ]
        if description["granularity"] > 1:
            bad_descriptors.append(
                (description["allocation_bytes"], description["granularity"] // 2)
            )
        for allocation_bytes, granularity in bad_descriptors:
            with self.subTest(
                allocation_bytes=allocation_bytes, granularity=granularity
            ):
                with self.assertRaises(ValueError):
                    self.attach_handle(
                        handle, allocation_bytes, granularity, self.device
                    )
                self.assertEqual(dict(self.native.resource_counters()), baseline)

    def test_multi_segment_read_spans_boundary_and_retains_backing(self):
        for memory_kind in ("host", "gpu"):
            with self.subTest(memory_kind=memory_kind):
                if memory_kind == "gpu":
                    granularity = self.native.device_granularity(self.device)
                else:
                    granularity = self.native.host_numa_granularity(
                        self.numa, self.device
                    )
                first = self.native.FabricAllocation(
                    granularity, self.numa, self.device, memory_kind=memory_kind
                )
                self.addCleanup(first.close)
                second = self.native.FabricAllocation(
                    2 * granularity, self.numa, self.device, memory_kind=memory_kind
                )
                self.addCleanup(second.close)
                payload = bytes(range(1, 17))
                first.load(payload[:8], granularity - 8)
                second.load(payload[8:])
                mapping = self.native.FabricMapping(self.device, memory_kind)
                self.addCleanup(mapping.close)
                segments = [self.segment(first), self.segment(second)]
                before = dict(self.native.resource_counters())
                mapping.attach(segments)
                counts = self.native.resource_counters()
                for key, delta in (
                    ("imports", 2),
                    ("reservations", 1),
                    ("mappings", 2),
                ):
                    self.assertEqual(counts[key] - before[key], delta)
                with self.assertRaisesRegex(ValueError, "not new"):
                    mapping.attach(segments)
                self.assertEqual(mapping.info()["allocation_bytes"], 3 * granularity)
                self.assertEqual(mapping.info()["gpu_access"], "READ")
                first.close()
                second.close()
                torch = self.torch
                reference = torch.tensor(list(payload), dtype=torch.uint8)
                view = alias = None
                try:
                    view = mapping.tensor_view([len(payload)], granularity - 8)
                    self.assertEqual(view.dtype, torch.uint8)
                    self.assertTrue(torch.equal(view.cpu(), reference))
                    for dtype in (
                        torch.uint8,
                        torch.float16,
                        torch.bfloat16,
                        torch.float32,
                        torch.int32,
                        torch.int64,
                    ):
                        with self.subTest(dtype=dtype):
                            item_size = torch.empty((), dtype=dtype).element_size()
                            offset = granularity - item_size
                            view = mapping.tensor_view([1, 2], offset, dtype=dtype)
                            expected = reference[8 - item_size : 8 + item_size]
                            self.assertEqual(
                                view.data_ptr(), mapping.info()["base_address"] + offset
                            )
                            self.assertEqual(view.dtype, dtype)
                            self.assertEqual(view.shape, (1, 2))
                            self.assertTrue(
                                torch.equal(view.cpu().flatten(), expected.view(dtype))
                            )
                            self.assertTrue(
                                torch.equal(
                                    view.view(torch.uint8).cpu().flatten(), expected
                                )
                            )
                    alias = view[:, 1:]
                    view = None
                    with self.assertRaisesRegex(ValueError, "live tensor"):
                        mapping.close()
                finally:
                    view = alias = None
                    self.drain_and_collect()
                mapping.close()
                self.assertEqual(dict(self.native.resource_counters()), self.baseline)

    def test_second_segment_failure_rolls_back_all_resources(self):
        owner = self.owner()
        segments = [self.segment(owner)] * 2
        baseline = dict(self.native.resource_counters())
        for stage in ("after_import", "after_properties", "after_map", "after_access"):
            with self.subTest(stage=stage):
                mapping = self.native.FabricMapping(self.device)
                self.addCleanup(mapping.close)
                with self.assertRaisesRegex(RuntimeError, stage):
                    mapping.attach(segments, _fail_stage=stage, _fail_segment=1)
                self.assertEqual(mapping.info()["state"], "CLOSED")
                self.assertEqual(dict(self.native.resource_counters()), baseline)

    def test_failed_rollback_can_retry_without_dropping_resources(self):
        owner = self.owner()
        segments = [self.segment(owner)] * 2
        mapping = self.native.FabricMapping(self.device)
        self.addCleanup(mapping.close)
        baseline = dict(self.native.resource_counters())
        with self.assertRaisesRegex(RuntimeError, "after_map"):
            mapping.attach(
                segments, _fail_stage="after_map", _fail_segment=1, _fail_rollback=True
            )
        self.baseline["cleanup_failures"] += 1
        baseline["cleanup_failures"] += 1
        info = mapping.info()
        self.assertEqual(info["state"], "CLEANUP_FAILED")
        self.assertEqual(info["mapped_ranges"], 1)
        self.assertEqual(info["imported_handles"], 2)
        self.assertTrue(info["reserved"])
        with self.assertRaisesRegex(ValueError, "not ready"):
            mapping.tensor_view([1])
        mapping.close()
        mapping.close()
        self.assertEqual(mapping.info()["state"], "CLOSED")
        self.assertEqual(dict(self.native.resource_counters()), baseline)

    def test_external_pointer_metadata_values_and_no_private_hbm(self):
        owner = self.owner()
        payload = bytes(range(256))
        owner.load(payload)
        consumer = self.consumer(owner)
        before = dict(self.native.resource_counters())
        hbm_before = self.torch.cuda.memory_allocated(self.device)
        view = None
        try:
            view = consumer.tensor_view([4, 4], offset=7)
            self.assertEqual(view.data_ptr(), consumer.info()["base_address"] + 7)
            self.assertEqual(view.dtype, self.torch.uint8)
            self.assertEqual(tuple(view.shape), (4, 4))
            self.assertEqual(view.stride(), (4, 1))
            self.assertEqual(view.device.type, "cuda")
            self.assertEqual(view.device.index, self.device)
            self.assertEqual(self.torch.cuda.memory_allocated(self.device), hbm_before)
            self.assertEqual(
                self.native.resource_counters()["owner_allocations"],
                before["owner_allocations"],
            )
            self.assertEqual(consumer.info()["active_views"], 1)
            self.assertEqual(bytes(view.cpu().flatten().tolist()), payload[7:23])
        finally:
            view = None
            self.drain_and_collect()
        self.assertEqual(dict(self.native.resource_counters()), before)
        consumer.close()

    def test_alias_and_multiple_storage_leases_block_close(self):
        owner = self.owner()
        consumer = self.consumer(owner)
        first = second = alias = None
        try:
            first = consumer.tensor_view([64], offset=0)
            second = consumer.tensor_view([8], offset=64)
            alias = first[3:19]
            self.assertEqual(consumer.info()["active_views"], 2)
            with self.assertRaisesRegex(ValueError, "live tensor"):
                consumer.close()
            first = None
            self.drain_and_collect()
            self.assertEqual(
                consumer.info()["active_views"], 2
            )  # alias keeps first Storage alive.
            with self.assertRaisesRegex(ValueError, "live tensor"):
                consumer.close()
            second = None
            self.drain_and_collect()
            self.assertEqual(consumer.info()["active_views"], 1)
            with self.assertRaisesRegex(ValueError, "live tensor"):
                consumer.close()
            alias = None
            self.drain_and_collect()
            self.assertEqual(consumer.info()["active_views"], 0)
            consumer.close()
            self.assertEqual(consumer.info()["state"], "CLOSED")
        finally:
            first = second = alias = None
            self.drain_and_collect()

    def test_shape_and_offset_errors_do_not_allocate_or_copy(self):
        owner = self.owner()
        consumer = self.consumer(owner)
        size = consumer.info()["allocation_bytes"]
        baseline = dict(self.native.resource_counters())
        hbm_before = self.torch.cuda.memory_allocated(self.device)
        cases = [
            ([-1], 0),
            ([1] * 33, 0),
            ([size + 1], 0),
            ([size], 1),
            ([2**63 - 1, 2**63 - 1], 0),
            ([1], size),
            ([0], size),
        ]
        for shape, offset in cases:
            with self.subTest(shape=shape, offset=offset):
                with self.assertRaises(ValueError):
                    consumer.tensor_view(shape, offset)
                self.assertEqual(dict(self.native.resource_counters()), baseline)
                self.assertEqual(
                    self.torch.cuda.memory_allocated(self.device), hbm_before
                )
        consumer.close()
        with self.assertRaisesRegex(ValueError, "not ready"):
            consumer.tensor_view([1])

    def test_typed_view_validation_and_empty_shape(self):
        torch = self.torch
        mapping = self.consumer(self.owner())
        size = mapping.info()["allocation_bytes"]
        baseline = dict(self.native.resource_counters())
        for shape, offset, dtype, error in (
            ([1], 1, torch.bfloat16, "aligned to dtype"),
            ([2], size - 4, torch.float32, "exceeds mapping bounds"),
            ([1], 0, torch.qint8, "unsupported Fabric view dtype"),
            ([0] + [128] * 10, 0, torch.float32, "contiguous stride limits"),
        ):
            with self.subTest(dtype=dtype, shape=shape), self.assertRaisesRegex(
                ValueError, error
            ):
                mapping.tensor_view(shape, offset, dtype=dtype)
            self.assertEqual(dict(self.native.resource_counters()), baseline)
        with self.assertRaises(TypeError):
            mapping.tensor_view([1], dtype="float32")

        empty = mapping.tensor_view([0, 2], size - 4, dtype=torch.float32)
        self.assertEqual(empty.shape, (0, 2))
        self.assertEqual(empty.dtype, torch.float32)
        del empty
        mapping.close()

    def test_100_attach_view_drain_close_cycles(self):
        owner = self.owner()
        payload = bytes(range(256))
        owner.load(payload)
        handle = owner.export_handle()
        description = owner.info()
        baseline = dict(self.native.resource_counters())
        hbm_before = self.torch.cuda.memory_allocated(self.device)
        for iteration in range(100):
            consumer = self.attach_handle(
                handle,
                description["allocation_bytes"],
                description["granularity"],
                self.device,
            )
            view = None
            try:
                view = consumer.tensor_view([16], offset=iteration)
                self.assertEqual(
                    view.data_ptr(), consumer.info()["base_address"] + iteration
                )
                self.assertEqual(
                    bytes(view.cpu().tolist()), payload[iteration : iteration + 16]
                )
                self.torch.cuda.synchronize(self.device)
            finally:
                view = None
                self.drain_and_collect()
                consumer.close()
            self.assertEqual(dict(self.native.resource_counters()), baseline)
            self.assertEqual(self.torch.cuda.memory_allocated(self.device), hbm_before)
        owner.close()
        self.assertEqual(
            {key: self.native.resource_counters()[key] for key in LIVE_KEYS},
            {key: 0 for key in LIVE_KEYS},
        )


def native_test_suite(native, torch, *, device, numa):
    """Configure all native cases without building or selecting a CUDA device."""
    suite = unittest.defaultTestLoader.loadTestsFromTestCase(NativeStorageTests)
    for case in suite:
        case.native = native
        case.torch = torch
        case.device = device
        case.numa = numa
    return suite


class FabricGateTest(unittest.TestCase):
    def test_requires_opt_in_before_any_cuda_or_device_probe(self):
        with patch.dict(os.environ, {"ENGRAM_RUN_CUDA_TESTS": "0"}), patch.object(
            runner, "_wheel_library_environment"
        ) as prepare:
            self.assertEqual(runner.main(), 2)
        prepare.assert_not_called()

    def test_busy_device_is_rejected_and_locks_are_released(self):
        with tempfile.TemporaryDirectory() as directory, patch.dict(
            os.environ, {"CUDA_VISIBLE_DEVICES": "0,1"}
        ), patch.object(runner, "_smi", side_effect=["0, GPU-a\n1, GPU-b", "12345"]):
            root = Path(directory)
            with self.assertRaisesRegex(RuntimeError, "occupied"):
                with runner.idle_devices(root):
                    self.fail("busy GPU must never be used")
            with (root / "gpu_status" / "0").open("a+") as handle:
                fcntl.flock(handle, fcntl.LOCK_EX | fcntl.LOCK_NB)

    def test_compatible_existing_file_lock_prevents_gpu_use(self):
        with tempfile.TemporaryDirectory() as directory, patch.dict(
            os.environ, {"CUDA_VISIBLE_DEVICES": "0,1"}
        ), patch.object(runner, "_smi", return_value="0, GPU-a\n1, GPU-b") as smi:
            root = Path(directory)
            (root / "gpu_status").mkdir()
            with (root / "gpu_status" / "0").open("a+") as held:
                fcntl.flock(held, fcntl.LOCK_EX | fcntl.LOCK_NB)
                with self.assertRaises(BlockingIOError):
                    with runner.idle_devices(root):
                        self.fail("locked GPU must never be used")
            self.assertEqual(smi.call_count, 1)

    def test_uuid_visibility_is_preserved_and_both_gpus_stay_locked(self):
        with tempfile.TemporaryDirectory() as directory, patch.dict(
            os.environ, {"CUDA_VISIBLE_DEVICES": "GPU-c,GPU-b"}
        ), patch.object(
            runner, "_smi", side_effect=["0, GPU-a\n1, GPU-b\n2, GPU-c", "", ""]
        ):
            root = Path(directory)
            with runner.idle_devices(root) as devices:
                self.assertEqual(devices, ["2", "1"])
                self.assertEqual(os.environ["CUDA_VISIBLE_DEVICES"], "GPU-c,GPU-b")
                self.assertEqual(os.environ["ENGRAM_TEST_PEER_DEVICE"], "1")
                for device in devices:
                    with (root / "gpu_status" / device).open("a+") as handle:
                        with self.assertRaises(BlockingIOError):
                            fcntl.flock(handle, fcntl.LOCK_EX | fcntl.LOCK_NB)

    def test_numeric_selection_and_pid_checks_refer_to_nvml_devices(self):
        with tempfile.TemporaryDirectory() as directory, patch.dict(
            os.environ, {"CUDA_VISIBLE_DEVICES": "2,0"}
        ), patch.object(
            runner, "_smi", side_effect=["0, GPU-a\n1, GPU-b\n2, GPU-c", "", ""]
        ) as smi:
            with runner.idle_devices(Path(directory)) as devices:
                self.assertEqual(devices, ["2", "0"])
                self.assertEqual(os.environ["CUDA_VISIBLE_DEVICES"], "GPU-c,GPU-a")
                self.assertEqual(
                    [call.args for call in smi.call_args_list[1:]],
                    [
                        ("--id=2", "--query-compute-apps=pid"),
                        ("--id=0", "--query-compute-apps=pid"),
                    ],
                )

    def test_skipped_failed_tests_and_empty_suite_are_not_passes(self):
        class Skipped(unittest.TestCase):
            def runTest(self):
                self.skipTest("missing Fabric")

        class Failed(unittest.TestCase):
            def runTest(self):
                self.fail("native regression")

        for suite in [
            unittest.TestSuite(),
            unittest.TestSuite([Skipped()]),
            unittest.TestSuite([Failed()]),
        ]:
            with self.subTest(suite=suite), patch.object(
                runner,
                "build_test_suite",
                return_value=suite,
            ), patch("sys.stderr", new_callable=io.StringIO):
                self.assertEqual(runner.run_tests(object(), object(), 0), 1)

    def test_only_executed_passing_tests_report_success(self):
        suite = unittest.TestSuite([unittest.FunctionTestCase(lambda: None)])
        with patch.object(runner, "build_test_suite", return_value=suite), patch(
            "sys.stderr", new_callable=io.StringIO
        ):
            self.assertEqual(runner.run_tests(object(), object(), 0), 0)


_FAILED_RESOURCES = []


@unittest.skipUnless(
    os.environ.get("ENGRAM_RUN_CUDA_TESTS") == "1",
    "explicit CUDA test opt-in required",
)
class FabricEmbeddingLookupTest(unittest.TestCase):
    def test_real_shards_lookup_graph_and_alias_lifetime(self):
        for memory_kind in ("host", "gpu"):
            with self.subTest(memory_kind=memory_kind):
                self._exercise(memory_kind)

    def _exercise(self, memory_kind):
        import torch

        from rtp_llm.models_py.modules.dsv41._engram_decode_prefetch import (
            DecodeEngramPrefetch,
        )
        from rtp_llm.models_py.modules.dsv41.engram import HostEngramEmbedding
        from rtp_llm.models_py.modules.dsv41.engram_distributed.manifest import (
            ExpectedLayout,
            Limits,
            bind_manifest,
            prepare_manifest,
        )
        from rtp_llm.models_py.modules.dsv41.engram_distributed.placement import (
            plan_shards,
        )
        from rtp_llm.models_py.modules.dsv41.engram_distributed.runtime import (
            _load_native,
        )
        from rtp_llm.models_py.modules.dsv41.engram_distributed.storage import (
            FabricEngramStorage,
            StorageBusyError,
        )

        device = int(os.environ.get("ENGRAM_TEST_DEVICE", "0"))
        numa = int(os.environ.get("ENGRAM_TEST_NUMA", "0"))
        torch.cuda.set_device(device)
        target = torch.device("cuda", device)
        native = _load_native()
        baseline = native.resource_counters()
        from rtp_llm.models_py.modules.dsv41.engram_distributed.placement import (
            segment_rows,
        )

        gpu = memory_kind == "gpu"
        segment_count = 2 if gpu and torch.cuda.device_count() >= 4 else 1
        if gpu:
            native.check_fabric_devices(list(range(2 * segment_count)))
        granularity = (
            native.device_granularity(device)
            if gpu
            else native.host_numa_granularity(numa, device)
        )
        plan = plan_shards(
            2 * segment_count * (granularity // 8) - 7, 256, 2, granularity=granularity
        )
        if 2 * plan.shard_bytes > 512 << 20:
            self.fail("test fixture exceeds its 512 MiB backing bound")

        widths = {"weight": plan.dim, "scale": plan.dim // plan.block}
        chunk_rows = (8 << 20) // plan.dim

        def fixture_rows(role, row_ids):
            rows = row_ids.reshape(-1, 1)
            columns = torch.arange(widths[role], dtype=torch.int64)
            if role == "weight":
                # Encode the row ID in FP8 mantissa/exponent bits, with varying signs.
                digits = rows >> ((columns % 8) * 4)
                values = 0x20 + ((digits + columns // 8) & 0xF)
                values |= ((rows + columns) & 1) << 7
            else:
                # Distinguish scale blocks as well as adjacent alignment units.
                unit = granularity // widths["scale"]
                values = 124 + ((rows + rows // unit + columns) % 8)
            return values.to(torch.uint8)

        generation = str(uuid4())
        revision = "fabric-lookup-regression-v1"
        hash_abi = "sha256:" + hashlib.sha256(revision.encode()).hexdigest()
        owners, documents = [], []
        boundary_rows = {-1, plan.total_rows, plan.padded_rows - 1, plan.padded_rows}
        storage = embedding = graph = alias = model = metadata = None
        completed = False
        try:
            for shard in range(2):
                row_parts = segment_rows(
                    plan,
                    shard,
                    (
                        tuple(range(shard * segment_count, (shard + 1) * segment_count))
                        if gpu
                        else (numa,)
                    ),
                )
                boundary_rows.update(
                    row for _, begin, end, _ in row_parts for row in (begin, end - 1)
                )
                segments = []
                for location, begin, end, padded in row_parts:
                    segment = dict(row_begin=begin, row_end=end, location=location)
                    for role, width in widths.items():
                        current_device = torch.cuda.current_device()
                        owner = native.FabricAllocation(
                            padded * width,
                            -1 if gpu else numa,
                            location if gpu else device,
                            **({"memory_kind": "gpu"} if gpu else {}),
                        )
                        owners.append(owner)
                        self.assertEqual(torch.cuda.current_device(), current_device)
                        for start in range(begin, end, chunk_rows):
                            rows = torch.arange(start, min(start + chunk_rows, end))
                            payload = fixture_rows(role, rows).numpy().tobytes()
                            owner.load(payload, (start - begin) * width)
                        segment[role] = dict(
                            allocation_bytes=padded * width,
                            handle=base64.b64encode(owner.export_handle()).decode(
                                "ascii"
                            ),
                        )
                    segments.append(segment)
                documents.append(
                    dict(
                        schema_version=4,
                        memory_kind=memory_kind,
                        granularity=granularity,
                        shard=shard,
                        shards=2,
                        generation=generation,
                        model_revision=revision,
                        hash_abi=hash_abi,
                        host_id=f"fixture-host-{shard}",
                        layers=[
                            dict(
                                layer_id=1,
                                rows=plan.total_rows,
                                head_dim=plan.dim,
                                segments=segments,
                            )
                        ],
                    )
                )

            parsed, capabilities = prepare_manifest(
                documents, 2, generation, revision, hash_abi
            )
            layout = ExpectedLayout(
                layers=((1, plan.total_rows),),
                head_dim=plan.dim,
                model_revision=revision,
                hash_abi=hash_abi,
                backend="fabric_gpu" if gpu else "fabric_egm",
            )

            validated = bind_manifest(parsed, layout, generation, Limits())

            def attach(handles):
                return FabricEngramStorage._attach_validated(
                    validated,
                    handles=handles,
                    device=device,
                    native=native,
                )

            def verify_table(views, role):
                table = views[f"layers.1.engram.embed.{role}"]
                self.assertEqual(table.shape, (plan.padded_rows, widths[role]))
                self.assertEqual(table.dtype, torch.uint8)
                for start in range(0, plan.total_rows, chunk_rows):
                    end = min(start + chunk_rows, plan.total_rows)
                    expected = fixture_rows(role, torch.arange(start, end))
                    self.assertTrue(
                        torch.equal(table[start:end].cpu(), expected),
                        f"{role} bytes differ at rows [{start}, {end})",
                    )
                padding = table[plan.total_rows :].cpu()
                self.assertEqual(padding.count_nonzero().item(), 0, f"{role} padding")
                return True

            if segment_count > 1:
                for role in widths:
                    name = f"layers.1.engram.embed.{role}"
                    tensor = next(t for t in parsed.manifest.tensors if t.name == name)
                    first, second = tensor.shards[:2]
                    left, right = first.allocation_id, second.allocation_id
                    swapped = dict(capabilities)
                    swapped[left], swapped[right] = swapped[right], swapped[left]
                    invalid = attach(swapped)
                    try:
                        with self.assertRaisesRegex(
                            AssertionError, f"{role} bytes differ"
                        ):
                            invalid.verify(lambda views, _: verify_table(views, role))
                        self.assertEqual(invalid.info()["state"], "FAILED")
                    finally:
                        invalid.close()

            storage = attach(capabilities)
            self.assertEqual(
                native.resource_counters()["imports"] - baseline["imports"],
                4 * segment_count,
            )
            storage.verify(
                lambda views, _: all(verify_table(views, role) for role in widths)
            )

            def reference(host_ids):
                ids = torch.tensor(host_ids, dtype=torch.int64)
                valid = (ids >= 0) & (ids < plan.total_rows)
                rows = torch.where(valid, ids, 0)
                weight = fixture_rows("weight", rows).view(torch.float8_e4m3fn).float()
                exponent = fixture_rows("scale", rows).float() - 127
                scale = torch.exp2(exponent).repeat_interleave(plan.block, dim=1)
                values = (
                    (weight * scale).to(torch.bfloat16).reshape(*ids.shape, plan.dim)
                )
                return values.masked_fill(~valid[..., None], 0)

            ordered = sorted(boundary_rows)
            patterns = [
                [ordered, ordered[::-1]],
                [ordered[::-1], ordered[1:] + ordered[:1]],
            ]
            # Exercise mapped scales directly, then the default Blackwell GPU
            # scale copy. The latter must still expose the BF16 __call__ API.
            for optimized in (False, True):
                with patch.dict(
                    os.environ,
                    {
                        "DSV41_ENGRAM_DEVICE_SCALES": str(int(optimized)),
                        "DSV41_ENGRAM_DIRECT_FP8": str(int(optimized)),
                        # Sync Graph ownership uses the production metadata path;
                        # the attention schedule itself needs the full-model E2E.
                        "DSV41_DECODE_ENGRAM_PREFETCH": "0",
                    },
                ):
                    embedding = HostEngramEmbedding.from_storage(storage, 1)
                    self.assertEqual(embedding._quantized_lookup_supported, optimized)
                    for dtype in (torch.int32, torch.int64):
                        ids = torch.tensor(patterns[0], dtype=dtype, device=target)
                        output = embedding(ids, target)
                        self.assertIsInstance(output, torch.Tensor)
                        self.assertEqual(output.dtype, torch.bfloat16)
                        self.assertTrue(
                            torch.equal(output.cpu(), reference(patterns[0]))
                        )

                    # Minimal model container; table snapshots and capture
                    # authorization are real production objects, not mocks.
                    class Model:
                        pass

                    model = Model()
                    model.layers = [
                        SimpleNamespace(engram=SimpleNamespace(embed_tokens=embedding))
                    ]
                    metadata = SimpleNamespace()
                    ids = torch.tensor(patterns[0], dtype=torch.int64, device=target)
                    stream = torch.cuda.Stream(device=device)
                    stream.wait_stream(torch.cuda.current_stream(device))
                    with torch.cuda.stream(stream):
                        output = embedding(ids, target)
                    stream.synchronize()
                    storage._pin_graph()
                    graph = torch.cuda.CUDAGraph()
                    with torch.cuda.graph(graph, stream=stream):
                        plan_owner = DecodeEngramPrefetch(model, metadata)
                        try:
                            output = embedding(ids, target)
                        finally:
                            plan_owner.close()

                    with self.assertRaises(StorageBusyError):
                        storage.close()
                    for host_ids in patterns * 2:
                        ids.copy_(
                            torch.tensor(host_ids, dtype=torch.int64, device=target)
                        )
                        graph.replay()
                        self.assertTrue(torch.equal(output.cpu(), reference(host_ids)))

                    torch.cuda.synchronize(device)
                    graph.reset()
                    graph = None
                    storage._unpin_graph()
                    metadata = model = plan_owner = None
                    embedding.release_external_views()
                    embedding = None
                    gc.collect()

            alias = storage.table_views(1)[0][1:2]
            gc.collect()
            with self.assertRaises(StorageBusyError):
                storage.close()
            alias = None
            gc.collect()
            storage.close()
            self.assertEqual(storage.info()["state"], "CLOSED")
            for owner in reversed(owners):
                owner.close()
            self.assertEqual(native.resource_counters(), baseline)
            completed = True
        finally:
            if not completed:
                _FAILED_RESOURCES.append(
                    (owners, storage, embedding, graph, alias, model, metadata)
                )


@unittest.skipUnless(
    os.environ.get("ENGRAM_RUN_CUDA_TESTS") == "1", "explicit CUDA test opt-in required"
)
class ShardedCudaLifecycleTest(unittest.TestCase):
    def test_real_aliases_graph_and_repeated_consumers(self):
        import torch

        from rtp_llm.models_py.modules.dsv41.engram_distributed.placement import (
            plan_shards,
        )
        from rtp_llm.models_py.modules.dsv41.engram_distributed.runtime import (
            _load_native,
        )

        device = int(os.environ.get("ENGRAM_TEST_DEVICE", "0"))
        peer_device = int(os.environ.get("ENGRAM_TEST_PEER_DEVICE", str(device)))
        numa = int(os.environ.get("ENGRAM_TEST_NUMA", "0"))
        torch.cuda.set_device(device)
        native = _load_native()
        baseline = native.resource_counters()
        granularity = native.host_numa_granularity(numa, device)

        for p in (2, 3):
            plan = plan_shards(
                p * (granularity // 8) - 7, 256, p, granularity=granularity
            )
            owners, weights, scales, mappings = [], [], [], []
            weight_shape = (plan.padded_rows, plan.dim)
            scale_shape = (plan.padded_rows, plan.dim // plan.block)
            try:
                for shard in range(p):
                    for size, value, segments in (
                        (plan.weight_bytes, shard + 1, weights),
                        (plan.scale_bytes, shard + 101, scales),
                    ):
                        owner = native.FabricAllocation(size, numa, device)
                        owners.append(owner)
                        for offset in range(0, size, 8 << 20):
                            owner.load(
                                bytes([value]) * min(8 << 20, size - offset), offset
                            )
                        segments.append((owner.export_handle(), size, granularity))
                for target in (device, peer_device):
                    weight = native.FabricMapping(target)
                    scale = native.FabricMapping(target)
                    mappings.extend((weight, scale))
                    weight.attach(weights)
                    scale.attach(scales)
                    view = weight.tensor_view(weight_shape)
                    self.assertEqual(view.data_ptr(), weight.info()["base_address"])
                    self.assertEqual(view.device.index, target)
                    alias = view[1:2]
                    del view
                    gc.collect()
                    self.assertEqual(weight.info()["active_views"], 1)
                    with self.assertRaisesRegex(ValueError, "live tensor"):
                        weight.close()
                    del alias
                    gc.collect()
                    self.assertEqual(weight.info()["active_views"], 0)

                first_weight, first_scale, second_weight, second_scale = mappings
                first_weight.close()
                first_scale.close()
                # Closing one independent mapping must not affect the other.
                torch.cuda.set_device(peer_device)
                w = second_weight.tensor_view(weight_shape)
                s = second_scale.tensor_view(scale_shape)
                rows = torch.tensor(
                    [plan.row_range(i)[0] for i in range(p)], device=peer_device
                )
                expected_w = list(range(1, p + 1))
                expected_s = list(range(101, p + 101))
                self.assertEqual(
                    w.index_select(0, rows)[:, 0].cpu().tolist(), expected_w
                )
                self.assertEqual(
                    s.index_select(0, rows)[:, 0].cpu().tolist(), expected_s
                )
                stream = torch.cuda.Stream(device=peer_device)
                stream.wait_stream(torch.cuda.current_stream(peer_device))
                with torch.cuda.stream(stream):
                    output = w.index_select(0, rows)
                stream.synchronize()
                graph = torch.cuda.CUDAGraph()
                with torch.cuda.graph(graph, stream=stream):
                    output = w.index_select(0, rows)
                for _ in range(3):
                    graph.replay()
                    self.assertEqual(output[:, 0].cpu().tolist(), expected_w)
                torch.cuda.synchronize(peer_device)
                graph.reset()
                del graph, w, s, output, rows
                gc.collect()
                second_weight.close()
                second_scale.close()
            finally:
                # Failures must terminate this isolated probe rather than force
                # releasing owners while an unknown Graph/view may still exist.
                if not all(m.info()["state"] == "CLOSED" for m in mappings):
                    raise RuntimeError(
                        "probe mappings remain live; terminate probe process"
                    )
                for owner in reversed(owners):
                    owner.close()
            self.assertEqual(native.resource_counters(), baseline)
            torch.cuda.set_device(device)


if __name__ == "__main__":
    unittest.main(defaultTest="PackagedNativeTest")
