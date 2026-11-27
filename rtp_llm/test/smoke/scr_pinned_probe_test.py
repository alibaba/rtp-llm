import ctypes
import os
import unittest
from pathlib import Path

from rtp_llm.test.smoke.scr_checkpoint_probe import PinnedProbe


class AllocationProp(ctypes.Structure):
    _fields_ = [
        ("type", ctypes.c_int),
        ("handle_types", ctypes.c_int),
        ("location", ctypes.c_int * 2),
        ("metadata", ctypes.c_void_p),
        ("flags", ctypes.c_ubyte * 8),
    ]


class PinnedProbeTest(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.probe = PinnedProbe()
        cuda_dir = (
            Path(__file__).absolute().parents[4] / "local_config_cuda/cuda/cuda/lib"
        )
        libraries = sorted(cuda_dir.glob("libcudart.so*"))
        if not libraries:
            raise RuntimeError(f"CUDA runtime runfile missing: {cuda_dir}")
        cls.cuda = ctypes.CDLL(str(libraries[0]))
        cls.cuda.cudaSetDevice.argtypes = [ctypes.c_int]
        cls.cuda.cudaHostAlloc.argtypes = [
            ctypes.POINTER(ctypes.c_void_p),
            ctypes.c_size_t,
            ctypes.c_uint,
        ]
        cls.cuda.cudaFreeHost.argtypes = [ctypes.c_void_p]
        cls.cuda.cudaHostRegister.argtypes = [
            ctypes.c_void_p,
            ctypes.c_size_t,
            ctypes.c_uint,
        ]
        cls.cuda.cudaHostUnregister.argtypes = [ctypes.c_void_p]
        cls.cuda.cudaGetErrorString.restype = ctypes.c_char_p
        cls.check(cls.cuda.cudaSetDevice(0))

    @classmethod
    def check(cls, result):
        if result != 0:
            raise AssertionError(cls.cuda.cudaGetErrorString(result).decode())

    def test_raw_host_allocation_and_free(self):
        before = self.probe.snapshot()
        pointer = ctypes.c_void_p()
        size = 8192
        self.check(self.cuda.cudaHostAlloc(ctypes.byref(pointer), size, 0))
        try:
            allocated = self.probe.snapshot()
            self.assertEqual(allocated["active_bytes"] - before["active_bytes"], size)
            self.assertEqual(
                allocated["total_allocated_bytes"] - before["total_allocated_bytes"],
                size,
            )
            self.assertIn(pointer.value, {a["ptr"] for a in allocated["allocations"]})
        finally:
            self.check(self.cuda.cudaFreeHost(pointer))
        self.assertEqual(self.probe.snapshot()["active_bytes"], before["active_bytes"])

    def test_failed_registration_does_not_change_ledger(self):
        driver = ctypes.CDLL("libcuda.so.1")
        driver.cuMemHostRegister_v2.argtypes = [
            ctypes.c_void_p,
            ctypes.c_size_t,
            ctypes.c_uint,
        ]
        storage = ctypes.create_string_buffer(4096)
        before = self.probe.snapshot()
        result = driver.cuMemHostRegister_v2(ctypes.addressof(storage), 0, 0)
        self.assertEqual(result, 1)
        after = self.probe.snapshot()
        self.assertEqual(after["active_bytes"], before["active_bytes"])
        self.assertEqual(
            after["total_allocated_bytes"], before["total_allocated_bytes"]
        )
        self.assertEqual(after["allocations"], before["allocations"])
        self.assertGreater(after["driver_calls"], before["driver_calls"])

    def test_host_vmm_is_diagnostic_not_a_classic_pinned_allocation(self):
        driver = ctypes.CDLL("libcuda.so.1")
        driver.cuMemGetAllocationGranularity.argtypes = [
            ctypes.POINTER(ctypes.c_size_t),
            ctypes.POINTER(AllocationProp),
            ctypes.c_uint,
        ]
        driver.cuMemCreate.argtypes = [
            ctypes.POINTER(ctypes.c_uint64),
            ctypes.c_size_t,
            ctypes.POINTER(AllocationProp),
            ctypes.c_uint64,
        ]
        driver.cuMemRelease.argtypes = [ctypes.c_uint64]
        prop = AllocationProp(type=1, location=(2, 0))
        granularity = ctypes.c_size_t()
        self.assertEqual(
            driver.cuMemGetAllocationGranularity(
                ctypes.byref(granularity), ctypes.byref(prop), 0
            ),
            0,
        )
        handle = ctypes.c_uint64()
        before = self.probe.snapshot()
        self.assertEqual(
            driver.cuMemCreate(
                ctypes.byref(handle), granularity.value, ctypes.byref(prop), 0
            ),
            0,
        )
        try:
            after = self.probe.snapshot()
            self.assertEqual(after["errors"], 0)
            self.assertEqual(after["active_bytes"], before["active_bytes"])
            self.assertEqual(after["allocations"], before["allocations"])
            self.assertEqual(after["host_vmm_calls"], before["host_vmm_calls"] + 1)
            self.assertEqual(
                after["host_vmm_created_bytes"] - before["host_vmm_created_bytes"],
                granularity.value,
            )
            self.assertEqual(after["last_host_vmm_location_type"], 2)
            self.assertEqual(after["last_host_vmm_location_id"], 0)
        finally:
            self.assertEqual(driver.cuMemRelease(handle), 0)

    def test_device_vmm_import_does_not_change_host_ledger(self):
        driver = ctypes.CDLL("libcuda.so.1")
        driver.cuMemGetAllocationGranularity.argtypes = [
            ctypes.POINTER(ctypes.c_size_t),
            ctypes.POINTER(AllocationProp),
            ctypes.c_uint,
        ]
        driver.cuMemCreate.argtypes = [
            ctypes.POINTER(ctypes.c_uint64),
            ctypes.c_size_t,
            ctypes.POINTER(AllocationProp),
            ctypes.c_uint64,
        ]
        driver.cuMemExportToShareableHandle.argtypes = [
            ctypes.c_void_p,
            ctypes.c_uint64,
            ctypes.c_int,
            ctypes.c_uint64,
        ]
        driver.cuMemImportFromShareableHandle.argtypes = [
            ctypes.POINTER(ctypes.c_uint64),
            ctypes.c_void_p,
            ctypes.c_int,
        ]
        driver.cuMemRelease.argtypes = [ctypes.c_uint64]
        prop = AllocationProp(type=1, handle_types=1, location=(1, 0))
        granularity = ctypes.c_size_t()
        self.assertEqual(
            driver.cuMemGetAllocationGranularity(
                ctypes.byref(granularity), ctypes.byref(prop), 0
            ),
            0,
        )
        handle = ctypes.c_uint64()
        before = self.probe.snapshot()
        self.assertEqual(
            driver.cuMemCreate(
                ctypes.byref(handle), granularity.value, ctypes.byref(prop), 0
            ),
            0,
        )
        try:
            fd = ctypes.c_int()
            self.assertEqual(
                driver.cuMemExportToShareableHandle(ctypes.byref(fd), handle, 1, 0), 0
            )
            try:
                imported = ctypes.c_uint64()
                self.assertEqual(
                    driver.cuMemImportFromShareableHandle(
                        ctypes.byref(imported), ctypes.c_void_p(fd.value), 1
                    ),
                    0,
                )
                try:
                    after = self.probe.snapshot()
                    self.assertEqual(after["active_bytes"], before["active_bytes"])
                    self.assertEqual(after["allocations"], before["allocations"])
                    self.assertEqual(after["errors"], 0)
                    self.assertEqual(
                        after["unclassified_import_calls"],
                        before["unclassified_import_calls"] + 1,
                    )
                finally:
                    self.assertEqual(driver.cuMemRelease(imported), 0)
            finally:
                os.close(fd.value)
        finally:
            self.assertEqual(driver.cuMemRelease(handle), 0)

    def test_explicit_registration_and_unregister(self):
        before = self.probe.snapshot()
        size = 8192
        storage = ctypes.create_string_buffer(size + 4096)
        address = (ctypes.addressof(storage) + 4095) & ~4095
        pointer = ctypes.c_void_p(address)
        self.check(self.cuda.cudaHostRegister(pointer, size, 0))
        try:
            registered = self.probe.snapshot()
            self.assertEqual(registered["active_bytes"] - before["active_bytes"], size)
            self.assertIn(address, {a["ptr"] for a in registered["allocations"]})
        finally:
            self.check(self.cuda.cudaHostUnregister(pointer))
        after = self.probe.snapshot()
        self.assertEqual(after["active_bytes"], before["active_bytes"])
        self.assertGreater(after["driver_calls"], 0)
        self.assertEqual(after["pid"], os.getpid())


if __name__ == "__main__":
    unittest.main()
