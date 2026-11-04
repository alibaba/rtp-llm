#!/usr/bin/env python3
"""Real-GPU native API tests, only for one explicitly isolated idle development GPU.

Example: CUDA_VISIBLE_DEVICES=GPU-... timeout --kill-after=10s 300s \
  /path/to/venv/bin/python test_native.py --device 0 [--numa 0]
This file does not select a GPU or verify resource ownership for the operator.
"""
import argparse
import ctypes
import gc
import json
import os
import sys
import unittest


LIVE_KEYS = ("contexts", "owner_allocations", "imports", "reservations", "mappings", "view_storages")


def nearest_numa(device):
    driver = ctypes.CDLL("libcuda.so.1")
    driver.cuInit.argtypes = [ctypes.c_uint]
    driver.cuInit.restype = ctypes.c_int
    driver.cuDeviceGet.argtypes = [ctypes.POINTER(ctypes.c_int), ctypes.c_int]
    driver.cuDeviceGet.restype = ctypes.c_int
    driver.cuDeviceGetAttribute.argtypes = [ctypes.POINTER(ctypes.c_int), ctypes.c_int, ctypes.c_int]
    driver.cuDeviceGetAttribute.restype = ctypes.c_int
    actual_device, numa = ctypes.c_int(), ctypes.c_int()
    def check(stage, code):
        if code:
            raise RuntimeError(f"{stage} failed with CUDA code {code}")
    check("cuInit", driver.cuInit(0))
    check("cuDeviceGet", driver.cuDeviceGet(ctypes.byref(actual_device), device))
    check("HOST_NUMA_ID", driver.cuDeviceGetAttribute(ctypes.byref(numa), 134, actual_device.value))
    if numa.value < 0:
        raise RuntimeError("selected GPU has no usable HOST_NUMA_ID")
    return numa.value


class NativeStorageTests(unittest.TestCase):
    native = None
    torch = None
    device = None
    numa = None

    def setUp(self):
        gc.collect()
        self.torch.cuda.synchronize(self.device)
        self.baseline = dict(self.native.resource_counters())
        self.assertEqual({key: self.baseline[key] for key in LIVE_KEYS}, {key: 0 for key in LIVE_KEYS})
        self.assertEqual(self.baseline["cleanup_failures"], 0)
        self.assertEqual(self.baseline["deferred_cleanups"], 0)
        # Registered first => runs after all owner/consumer closes in this test.
        self.addCleanup(self.assert_resources_restored)

    def assert_resources_restored(self):
        self.torch.cuda.synchronize(self.device)
        gc.collect()
        self.assertEqual(dict(self.native.resource_counters()), self.baseline)

    def owner(self, size=256):
        value = self.native.Owner(size=size, numa=self.numa, device=self.device)
        self.addCleanup(value.close)
        return value

    def consumer(self, owner):
        description = owner.info()
        value = self.native.Consumer(
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

    def test_granularity_query_has_no_allocation_or_context_leak(self):
        before = dict(self.native.resource_counters())
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
                    self.native.Owner(size=invalid_size, numa=self.numa, device=self.device)
                self.assertEqual(dict(self.native.resource_counters()), before)

        owner = self.owner(size=32)
        description = owner.info()
        self.assertEqual(description["logical_bytes"], 32)
        self.assertGreaterEqual(description["allocation_bytes"], 32)
        self.assertEqual(description["allocation_bytes"] % description["granularity"], 0)
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
                    self.native.Consumer(b"x" * byte_count, 2097152, 2097152, self.device)
                self.assertEqual(dict(self.native.resource_counters()), before)

    def test_owner_constructor_rollback(self):
        baseline = dict(self.native.resource_counters())
        for stage in ("after_context", "after_create", "after_reserve", "after_map", "after_access"):
            with self.subTest(fail_stage=stage):
                with self.assertRaisesRegex(RuntimeError, f"TEST_FAULT_INJECTION: {stage}"):
                    self.native.Owner(256, self.numa, self.device, fail_stage=stage)
                gc.collect()
                self.assertEqual(dict(self.native.resource_counters()), baseline)

    def test_consumer_constructor_rollback_and_descriptor_validation(self):
        owner = self.owner()
        handle = owner.export_handle()
        description = owner.info()
        baseline = dict(self.native.resource_counters())
        self.assertEqual(baseline["owner_allocations"], 1)
        for stage in ("after_context", "after_import", "after_properties", "after_reserve", "after_map", "after_access"):
            with self.subTest(fail_stage=stage):
                with self.assertRaisesRegex(RuntimeError, f"TEST_FAULT_INJECTION: {stage}"):
                    self.native.Consumer(handle, description["allocation_bytes"], description["granularity"],
                                         self.device, fail_stage=stage)
                gc.collect()
                self.assertEqual(dict(self.native.resource_counters()), baseline)

        # All are safe descriptor checks. Never corrupt an actual driver handle.
        bad_descriptors = [(0, description["granularity"]),
                           (description["allocation_bytes"], 0),
                           (self.native.max_allocation_bytes + 1, description["granularity"])]
        if description["granularity"] > 1:
            bad_descriptors.append((description["allocation_bytes"], description["granularity"] // 2))
        for allocation_bytes, granularity in bad_descriptors:
            with self.subTest(allocation_bytes=allocation_bytes, granularity=granularity):
                with self.assertRaises(ValueError):
                    self.native.Consumer(handle, allocation_bytes, granularity, self.device)
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
            self.assertEqual(self.native.resource_counters()["owner_allocations"], before["owner_allocations"])
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
            self.assertEqual(consumer.info()["active_views"], 2)  # alias keeps first Storage alive.
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
        cases = [([-1], 0), ([1] * 33, 0), ([size + 1], 0), ([size], 1),
                 ([2**63 - 1, 2**63 - 1], 0), ([1], size), ([0], size)]
        for shape, offset in cases:
            with self.subTest(shape=shape, offset=offset):
                with self.assertRaises(ValueError):
                    consumer.tensor_view(shape, offset)
                self.assertEqual(dict(self.native.resource_counters()), baseline)
                self.assertEqual(self.torch.cuda.memory_allocated(self.device), hbm_before)
        consumer.close()
        with self.assertRaisesRegex(ValueError, "not ready"):
            consumer.tensor_view([1])

    def test_100_attach_view_drain_close_cycles(self):
        owner = self.owner()
        payload = bytes(range(256))
        owner.load(payload)
        handle = owner.export_handle()
        description = owner.info()
        baseline = dict(self.native.resource_counters())
        hbm_before = self.torch.cuda.memory_allocated(self.device)
        for iteration in range(100):
            consumer = self.native.Consumer(handle, description["allocation_bytes"], description["granularity"], self.device)
            view = None
            try:
                view = consumer.tensor_view([16], offset=iteration)
                self.assertEqual(view.data_ptr(), consumer.info()["base_address"] + iteration)
                self.assertEqual(bytes(view.cpu().tolist()), payload[iteration:iteration + 16])
                self.torch.cuda.synchronize(self.device)
            finally:
                view = None
                self.drain_and_collect()
                consumer.close()
            self.assertEqual(dict(self.native.resource_counters()), baseline)
            self.assertEqual(self.torch.cuda.memory_allocated(self.device), hbm_before)
        owner.close()
        self.assertEqual({key: self.native.resource_counters()[key] for key in LIVE_KEYS}, {key: 0 for key in LIVE_KEYS})


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--device", type=int, required=True, help="must be 0 after isolation to one CUDA_VISIBLE_DEVICES entry")
    parser.add_argument("--numa", type=int, help="owner local HOST_NUMA ID; default queries the selected GPU")
    parser.add_argument("--build-directory")
    args = parser.parse_args()
    visible = os.environ.get("CUDA_VISIBLE_DEVICES", "")
    entries = [entry.strip() for entry in visible.split(",") if entry.strip()]
    if len(entries) != 1 or entries[0] == "-1" or args.device != 0:
        parser.error("set CUDA_VISIBLE_DEVICES to exactly one authorized idle GPU and pass --device 0")
    if args.numa is not None and args.numa < 0:
        parser.error("--numa must be nonnegative")

    import torch
    from load_native import load_native

    native = load_native(args.build_directory)
    torch.cuda.set_device(args.device)
    if torch.cuda.device_count() != 1:
        raise RuntimeError("expected exactly one CUDA-visible GPU")
    numa = args.numa if args.numa is not None else nearest_numa(args.device)
    NativeStorageTests.native = native
    NativeStorageTests.torch = torch
    NativeStorageTests.device = args.device
    NativeStorageTests.numa = numa
    print(json.dumps({"stage": "native_test_scope", "cuda_visible_devices": visible, "device": args.device,
                      "numa": numa, "iterations": 100, "scope": "small_native_storage_tests_only"}), flush=True)
    suite = unittest.defaultTestLoader.loadTestsFromTestCase(NativeStorageTests)
    result = unittest.TextTestRunner(verbosity=2, failfast=True).run(suite)
    print(json.dumps({"stage": "native_test_complete", "passed": result.wasSuccessful(),
                      "tests_run": result.testsRun, "resource_counters": dict(native.resource_counters())}), flush=True)
    return 0 if result.wasSuccessful() else 1


if __name__ == "__main__":
    sys.exit(main())
