"""Resource admission checks; no ports, processes, or backends are started."""

import sys
import unittest
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from flexlb_test_framework.resource_plan import (
    JavaMockBudget,
    LaneLease,
    ResourcePlanError,
    plan_lane_leases,
)


class ResourcePlanTest(unittest.TestCase):
    def test_cumulative_adds_not_peak_live_workers_bound_ports(self):
        self.assertEqual(149, JavaMockBudget(6, 143).worker_capacity)
        with self.assertRaisesRegex(ResourcePlanError, "cumulative"):
            JavaMockBudget(6, 144)

    def test_budget_rejects_non_integer_or_negative_counts(self):
        for initial, adds in [(True, 1), (2, False), (0, 0), (2, -1), (2, 1.5)]:
            with self.subTest(initial=initial, adds=adds), self.assertRaises(
                ResourcePlanError
            ):
                JavaMockBudget(initial, adds)

    def test_every_manifest_interval_matches_probe_and_lock_ports(self):
        lease = LaneLease(0, 18080, 55151, 22)
        self.assertEqual(159, len(lease.ports()))
        self.assertEqual(("m18080_18085.lock", "g55150_55302.lock"), lease.lock_names())
        doc = lease.to_manifest()
        ports = [
            p
            for interval in doc["intervals"]
            for p in range(interval["first"], interval["last"] + 1)
        ]
        self.assertEqual(list(lease.ports()), ports)
        self.assertEqual("55151", doc["child_env"]["FLEXLB_FT_MOCK_BASE_GRPC_PORT"])
        self.assertIn(55150, ports)
        self.assertTrue({55300, 55301, 55302} <= set(ports))

    def test_lane_uses_maximum_sequential_demand_not_sum(self):
        leases = plan_lane_leases(
            [[JavaMockBudget(100, 0), JavaMockBudget(2, 100)]],
            master_base=18080,
            mock_base=55151,
        )
        self.assertEqual(102, leases[0].worker_capacity)

    def test_minimum_stride_keeps_full_mock_windows_disjoint(self):
        leases = plan_lane_leases(
            [[JavaMockBudget(6, 16)]] * 3,
            master_base=18080,
            mock_base=55151,
            mock_stride=153,
        )
        ports = [p for lease in leases for p in lease.ports()]
        self.assertEqual(len(ports), len(set(ports)))
        with self.assertRaises(ResourcePlanError):
            plan_lane_leases(
                [[JavaMockBudget(6, 16)]],
                master_base=18080,
                mock_base=55151,
                mock_stride=152,
            )

    def test_cross_lane_and_cross_side_overlap_rejected(self):
        with self.assertRaisesRegex(ResourcePlanError, "overlap"):
            plan_lane_leases(
                [[JavaMockBudget(6, 16)]] * 2, master_base=18080, mock_base=18091
            )

    def test_port_limits_and_self_overlap_rejected(self):
        for master, base in [
            (1023, 55151),
            (18080, 1024),
            (65531, 55151),
            (18080, 65400),
            (18080, 18081),
        ]:
            with self.subTest(master=master, base=base), self.assertRaises(
                ResourcePlanError
            ):
                LaneLease(0, master, base, 6)

    def test_empty_resources_are_not_a_valid_run(self):
        for lanes in [[], [[]]]:
            with self.assertRaisesRegex(ResourcePlanError, "nonempty"):
                plan_lane_leases(lanes, master_base=18080, mock_base=55151)


if __name__ == "__main__":
    unittest.main()
