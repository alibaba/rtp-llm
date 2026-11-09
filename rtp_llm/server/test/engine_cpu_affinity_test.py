from pathlib import Path
from tempfile import TemporaryDirectory
from unittest import TestCase, main
from unittest.mock import patch

from rtp_llm.server import engine_cpu_affinity as affinity


class EngineCpuAffinityTest(TestCase):
    def setUp(self):
        tmp = TemporaryDirectory()
        self.addCleanup(tmp.cleanup)
        self.sysfs = Path(tmp.name)
        self.devices = [
            "0001:03:00.0",
            "0000:95:00.0",
            "0001:1b:00.0",
            "0000:6b:00.0",
        ]
        for index, pci in enumerate(self.devices):
            self.write(f"bus/pci/devices/{pci}/numa_node", str(1 - index % 2))
        self.write("devices/system/node/node0/cpulist", "0-1,8-9")
        self.write("devices/system/node/node1/cpulist", "4-5,12-13")
        for siblings in ((0, 8), (1, 9), (4, 12), (5, 13)):
            for cpu in siblings:
                self.write(
                    f"devices/system/cpu/cpu{cpu}/topology/thread_siblings_list",
                    f"{siblings[0]},{siblings[1]}",
                )

    def write(self, name, value):
        path = self.sysfs / name
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(value + "\n")

    def plan(self, rank_devices, allowed):
        with patch.object(
            affinity, "_device_pci_bdf", side_effect=self.devices.__getitem__
        ), patch.object(affinity.os, "sched_getaffinity", return_value=allowed, create=True):
            return affinity.plan_engine_cpu_affinity(rank_devices, self.sysfs)

    def test_respects_device_order_cpuset_and_physical_cores(self):
        plan = self.plan({8: 1, 9: 0, 10: 3, 11: 2}, {0, 4, 8, 9, 13})
        self.assertEqual(
            {rank: (p.numa_node, p.cpu) for rank, p in plan.items()},
            {8: (0, 0), 9: (1, 4), 10: (0, 9), 11: (1, 13)},
        )

    def test_insufficient_physical_cores_skips_all_local_ranks(self):
        with self.assertLogs(level="WARNING") as logs:
            self.assertEqual(self.plan({0: 0, 1: 1, 2: 3}, {0, 4, 8, 13}), {})
        self.assertIn("no free physical core in NUMA 0", logs.output[0])

    def test_unknown_numa_skips_all_local_ranks(self):
        self.write(f"bus/pci/devices/{self.devices[1]}/numa_node", "-1")
        with self.assertLogs(level="WARNING") as logs:
            self.assertEqual(self.plan({0: 0, 1: 1}, {0, 4, 8, 9, 13}), {})
        self.assertIn("NUMA node is unknown", logs.output[0])

    def test_rejects_changed_device_or_rank_mask(self):
        placement = affinity.EngineCpuPlacement(1, self.devices[1], 0, 0)
        with patch.object(affinity, "_device_pci_bdf", return_value=self.devices[0]):
            with self.assertLogs(level="WARNING"):
                self.assertEqual(affinity.resolve_engine_cpu(8, 1, placement), -1)
        with patch.object(
            affinity, "_device_pci_bdf", return_value=self.devices[1]
        ), patch.object(affinity.os, "sched_getaffinity", return_value={8}, create=True):
            with self.assertLogs(level="WARNING"):
                self.assertEqual(affinity.resolve_engine_cpu(8, 1, placement), -1)


if __name__ == "__main__":
    main()
