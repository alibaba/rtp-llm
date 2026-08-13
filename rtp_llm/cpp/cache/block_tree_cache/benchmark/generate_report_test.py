#!/usr/bin/env python3
"""Small schema/mapping guard for the benchmark HTML renderer."""

import unittest

from generate_report import bandwidth_judgment, environment_section, render_report


def _case(name, throughput):
    return (
        {
            "case": name,
            "suite": "profile",
            "status": "completed",
            "repetitions": [],
            "perf": {"status": "skipped"},
        },
        [
            {
                "status": "completed",
                "runner": "transfer",
                "metrics": {
                    "logical_throughput_bytes_per_second": throughput,
                },
            }
        ],
    )


class GenerateReportTest(unittest.TestCase):
    def setUp(self):
        self.cases = [
            _case("transfer_device_host_full", 1e9),
            _case("transfer_host_disk_direct_full", 2e9),
            _case("transfer_device_disk_direct_full", 3e9),
        ]

    def test_bandwidth_rows_map_to_the_correct_medium(self):
        html = bandwidth_judgment(self.cases)
        host_row = html[
            html.index("Host↔Disk (direct)") : html.index("Device↔Disk (direct)")
        ]
        device_row = html[
            html.index("Device↔Disk (direct)") : html.index("Host↔Disk (buffered)")
        ]
        self.assertIn("2 GB/s", host_row)
        self.assertIn("3 GB/s", device_row)
        self.assertIn("未提供环境实测基线", html)

    def test_minimal_manifest_renders_sections_and_manifest_link(self):
        manifest = {
            "suite": "profile",
            "total_cases": len(self.cases),
            "canonical_total_cases": len(self.cases),
            "completed": len(self.cases),
            "partial": 0,
            "failed": 0,
            "skipped": 0,
            "cases": [case for case, _ in self.cases],
            "environment": {},
            "invocation": {"suite": "profile", "case": "all"},
        }
        html = render_report(manifest, self.cases, "profile/suite_manifest.json")
        self.assertIn("1. 测试环境", html)
        self.assertIn("7. 产物与可复核性", html)
        self.assertIn('href="profile/suite_manifest.json"', html)
        self.assertIn("Off-CPU：", html)
        self.assertIn("suite manifest 未记录 off-CPU 产物或跳过原因", html)

    def test_environment_compacts_disk_provenance_into_one_row(self):
        manifest = {
            "environment": {
                "gpu": "test-gpu",
                "disk_scope": "benchmark process mount namespace (container-visible)",
                "disk_target": "/bench",
                "disk_mount": "target=/bench, source=/dev/nvme0n1, fstype=ext4",
                "disk_capacity": "size=1T, used=600G, available=400G, use=60%",
            },
            "invocation": {"disk_root": "/bench"},
        }

        html = environment_section(manifest)

        self.assertIn("<td>disk</td>", html)
        self.assertIn("mount=/bench, source=/dev/nvme0n1, fstype=ext4", html)
        self.assertIn("scope=container-visible mount namespace", html)
        self.assertNotIn("target=/bench", html)
        self.assertNotIn("<td>disk_scope</td>", html)
        self.assertNotIn("<td>disk_target</td>", html)
        self.assertNotIn("磁盘环境口径", html)


if __name__ == "__main__":
    unittest.main()
