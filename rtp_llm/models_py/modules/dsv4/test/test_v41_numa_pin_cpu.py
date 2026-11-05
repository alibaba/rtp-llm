"""CPU tests for #21(a) NUMA pin helpers: flag parsing, cpulist parsing,
PCI bus-id normalization, and the flag-off no-op contract. The sysfs/sched
integration is proven by the [NUMAPIN] announce lines in the node parity
run (per-rank gpu/numa/cpu-count)."""

import importlib.util
import os
import unittest
from pathlib import Path
from unittest import mock

ROOT = Path(__file__).resolve().parents[1]


def load(name, relpath):
    spec = importlib.util.spec_from_file_location(name, ROOT / relpath)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


numa = load("_r18_numa", "_v41_numa_pin.py")


class FlagTest(unittest.TestCase):
    def test_default_off_and_values(self):
        clean = dict(os.environ)
        clean.pop("DSV41_NUMA_PIN", None)
        with mock.patch.dict(os.environ, clean, clear=True):
            self.assertFalse(numa.numa_pin_enabled())
        with mock.patch.dict(os.environ, {"DSV41_NUMA_PIN": "1"}):
            self.assertTrue(numa.numa_pin_enabled())
        for value in ("0", "true", ""):
            with mock.patch.dict(os.environ, {"DSV41_NUMA_PIN": value}):
                self.assertFalse(numa.numa_pin_enabled())

    def test_pin_is_noop_when_disabled(self):
        clean = dict(os.environ)
        clean.pop("DSV41_NUMA_PIN", None)
        with mock.patch.dict(os.environ, clean, clear=True):
            # flag-off returns before any sched call (cross-platform)
            self.assertIsNone(numa.pin_current_thread(0))

    @unittest.skipUnless(hasattr(os, "sched_getaffinity"), "Linux-only affinity")
    def test_disabled_leaves_affinity_untouched(self):
        clean = dict(os.environ)
        clean.pop("DSV41_NUMA_PIN", None)
        with mock.patch.dict(os.environ, clean, clear=True):
            before = os.sched_getaffinity(0)
            numa.pin_current_thread(0)
            self.assertEqual(os.sched_getaffinity(0), before)


class CpulistTest(unittest.TestCase):
    def test_ranges_and_singles(self):
        self.assertEqual(
            numa.cpulist_to_set("0-83,168-251"),
            set(range(0, 84)) | set(range(168, 252)),
        )
        self.assertEqual(
            numa.cpulist_to_set("84-167,252-335"),
            set(range(84, 168)) | set(range(252, 336)),
        )
        self.assertEqual(numa.cpulist_to_set("3"), {3})
        self.assertEqual(numa.cpulist_to_set("0-0"), {0})
        self.assertEqual(numa.cpulist_to_set(""), set())
        self.assertEqual(numa.cpulist_to_set(" 2 , 5-6 "), {2, 5, 6})

    def test_reference_topo_maps(self):
        # the verified reference topology (nvidia-smi topo -m)
        n0 = numa.cpulist_to_set("0-83,168-251")
        n1 = numa.cpulist_to_set("84-167,252-335")
        self.assertEqual(len(n0), 168)
        self.assertEqual(len(n1), 168)
        self.assertFalse(n0 & n1)
        self.assertEqual(n0 | n1, set(range(336)))


class BusIdTest(unittest.TestCase):
    def test_normalization(self):
        self.assertEqual(numa.normalize_pci_bus_id("00000000:1B:00.0"), "0000:1b:00.0")
        self.assertEqual(numa.normalize_pci_bus_id("0000:1b:00.0"), "0000:1b:00.0")
        self.assertEqual(numa.normalize_pci_bus_id("00010000:CA:00.0"), "0000:ca:00.0")
        self.assertEqual(
            numa.normalize_pci_bus_id(" 00000000:5E:00.0 "), "0000:5e:00.0"
        )

    def test_malformed_passthrough(self):
        self.assertEqual(numa.normalize_pci_bus_id("garbage"), "garbage")


if __name__ == "__main__":
    unittest.main(verbosity=2)
