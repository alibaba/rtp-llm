import json
import os
import tempfile
import unittest

import generate_report


class GenerateReportTest(unittest.TestCase):
    def test_manifest_drives_all_valid_repetitions(self):
        with tempfile.TemporaryDirectory() as root:
            case_dir = os.path.join(root, "case")
            os.makedirs(case_dir)
            repetitions = []
            for index, throughput in enumerate((10.0, 30.0, 20.0)):
                rep_dir = os.path.join(case_dir, f"rep_{index:04d}")
                os.makedirs(rep_dir)
                result_path = os.path.join(rep_dir, "result.json")
                with open(result_path, "w") as output:
                    json.dump(
                        {
                            "status": "completed",
                            "metrics": {
                                "logical_throughput_bytes_per_second": throughput
                            },
                        },
                        output,
                    )
                repetitions.append(
                    {
                        "valid": True,
                        "status": "completed",
                        "result_json": result_path,
                    }
                )
            # This stale directory is deliberately absent from the manifest.
            os.makedirs(os.path.join(root, "stale_case"))
            cases = generate_report.load_manifest_results(
                root,
                {
                    "cases": [
                        {
                            "case": "case",
                            "subcommand": "transfer",
                            "repetitions": repetitions,
                        }
                    ]
                },
            )
            self.assertEqual(len(cases), 1)
            self.assertEqual(len(cases[0][1]), 3)
            summary = generate_report.metric_summary(
                cases[0][1], "logical_throughput_bytes_per_second"
            )
            self.assertEqual(summary["median"], 20.0)
            self.assertEqual(summary["mad"], 10.0)
            self.assertEqual(summary["n"], 3)

    def test_html_report_keeps_conclusion_and_manifest_link(self):
        manifest = {
            "total_cases": 1,
            "canonical_total_cases": 14,
            "completed": 1,
            "partial": 0,
            "failed": 0,
            "skipped": 0,
            "environment": {"gpu": "test-gpu"},
            "conclusion_details": ["关键事实应出现在结论区。"],
        }
        rendered = generate_report.render_report(
            manifest, [], "profile/suite_manifest.json"
        )
        self.assertIn("<!doctype html>", rendered)
        self.assertIn("关键事实应出现在结论区。", rendered)
        self.assertIn('href="profile/suite_manifest.json"', rendered)

    def test_conclusion_acknowledges_supplied_hardware_baseline(self):
        manifest = {
            "total_cases": 1,
            "canonical_total_cases": 1,
            "completed": 1,
            "partial": 0,
            "failed": 0,
            "skipped": 0,
            "report_metadata": {
                "bandwidth_comparison": [
                    {
                        "path": "Host↔Disk (direct)",
                        "baseline": "O_DIRECT baseline",
                        "measured": "1 GB/s",
                        "judgment": "fact only",
                    }
                ]
            },
        }
        cases = [
            (
                {"case": "transfer_case", "subcommand": "transfer"},
                [{"status": "completed", "metrics": {}}],
            )
        ]

        rendered = generate_report.render_report(
            manifest, cases, "profile/suite_manifest.json"
        )

        self.assertIn("本次已提供同机硬件基线与分析元数据", rendered)
        self.assertNotIn("未配置同机硬件基线与显式阈值时", rendered)

    def test_tree_observation_does_not_claim_no_gain_from_one_repetition(self):
        def result(insert_calls, match_calls):
            return {
                "phases_ns": {"measured": 1e9},
                "metrics": {
                    "insert_calls": insert_calls,
                    "match_calls": match_calls,
                    "insert_latency_ns_p50": 2e6,
                    "match_latency_ns_p50": 1e6,
                },
            }

        rendered = generate_report.tree_observations(
            [
                (
                    {"case": "tree_stress_100k", "subcommand": "tree"},
                    [result(110, 22)],
                ),
                (
                    {"case": "tree_stress_100k_single", "subcommand": "tree"},
                    [result(100, 20)],
                ),
            ]
        )

        self.assertIn("单次 repetition 不能证明稳定收益", rendered)
        self.assertNotIn("多线程未带来吞吐收益", rendered)

    def test_perf_artifacts_are_linked_from_html(self):
        manifest = {
            "total_cases": 1,
            "canonical_total_cases": 1,
            "completed": 1,
            "partial": 0,
            "failed": 0,
            "skipped": 0,
            "cases": [
                {
                    "case": "tree_case",
                    "subcommand": "tree",
                    "status": "completed",
                    "repetitions": [],
                    "perf": {
                        "status": "ok",
                        "mode": "record",
                        "artifacts": {
                            "flamegraph": "perf/flamegraph.svg",
                            "perf_data": "perf/perf.data",
                            "summary": "perf/perf_summary.txt",
                        },
                    },
                }
            ],
        }
        rendered = generate_report.render_report(
            manifest,
            [(manifest["cases"][0], [])],
            "profile/suite_manifest.json",
        )
        self.assertIn('href="profile/tree_case/perf/flamegraph.svg"', rendered)
        self.assertIn('href="profile/tree_case/perf/perf.data"', rendered)

    def test_numbers_are_human_readable_not_scientific(self):
        single = generate_report.format_summary(
            generate_report.summarize([3725184.0]), duration=True
        )
        self.assertEqual(single, "3.725 ms")
        self.assertNotIn("e+", single)
        spread = generate_report.format_summary(
            generate_report.summarize([37.4e9, 38.76e9, 40.1e9]),
            1e-9,
            " GB/s",
        )
        self.assertNotIn("e+", spread)
        self.assertIn("MAD 1.34", spread)
        self.assertIn("n=3", spread)
        counts = generate_report.format_summary(generate_report.summarize([5490.0]))
        self.assertEqual(counts, "5,490")
        self.assertEqual(
            generate_report.format_summary(
                generate_report.summarize([300.0]), duration=True
            ),
            "300 ns",
        )

    def test_tables_keep_report_columns(self):
        tree_result = {
            "status": "completed",
            "phases_ns": {"measured": 30e9},
            "metrics": {
                "insert_latency_ns_p50": 4.2e6,
                "insert_latency_ns_p99": 40e6,
                "insert_latency_ns_max": 100e6,
                "match_latency_ns_p50": 1.0e6,
                "match_latency_ns_p99": 3.3e6,
                "match_latency_ns_max": 6.5e6,
                "match_device_matched_blocks_per_request": 412.25,
                "match_host_matched_blocks_per_request": 37.75,
                "insert_path_keys_per_call": 640,
                "insert_new_nodes_per_call": 32,
                "match_keys_per_call": 512,
                "scenario.continuation.average_matched_depth": 512,
                "scenario.fork.average_matched_depth": 256,
                "scenario.cold.average_matched_depth": 0,
                "insert_calls": 31101,
                "match_calls": 7780,
                "loads_committed": 138,
                "loads_succeeded": 138,
                "loads_pending_at_measurement_end": 1,
                "steady_state_node_count_avg": 233149.09,
                "steady_state_node_count_min": 212108,
                "steady_state_node_count_max": 234508,
            },
        }
        tree_html = generate_report.tree_section(
            [
                (
                    {
                        "case": "tree_stress_100k",
                        "subcommand": "tree",
                        "status": "completed",
                    },
                    [tree_result],
                )
            ]
        )
        for expected in (
            "insert p50/p99/max",
            "avg matched blocks/request (device/host)",
            "request shape (insert path/new nodes/match keys)",
            "matched depth (continuation/fork/cold)",
            "insert ops/s",
            "match ops/s",
            "loads (committed/succeeded)",
            "节点水位 avg [min,max]",
            "412.25 / 37.75",
            "640 / 32 / 512",
            "512 / 256 / 0",
            "1,036.7",
            "138 / 138",
            "233,149 [212,108, 234,508]",
        ):
            self.assertIn(expected, tree_html)
        self.assertNotIn(">hit<", tree_html)

        transfer_result = {
            "status": "completed",
            "phases_ns": {"measured": 31e9},
            "resolved_config": {
                "requested_copy_strategy": "batch",
                "actual_copy_strategy": "batch",
                "disk_io_mode": "buffered",
            },
            "transfer_workload": {
                "requested_working_set_blocks": 32768,
                "addressable_working_set_blocks": 32768,
                "visited_working_set_blocks": 32768,
            },
            "workload": {"failed_operations": 0},
            "metrics": {
                "logical_throughput_bytes_per_second": 10.8e9,
                "operations_per_second": 14722,
                "total_bytes_transferred": 285e9,
                "direction.d2disk.throughput_bps": 5.4e9,
            },
        }
        _, dd_html, _, _ = generate_report.transfer_subsections(
            [
                (
                    {
                        "case": "transfer_device_disk_full_context_buffered",
                        "subcommand": "transfer",
                        "status": "completed",
                    },
                    [transfer_result],
                )
            ]
        )
        for expected in ("mode", "ops/s", "总传输", "buffered", "14,722", "285 GB"):
            self.assertIn(expected, dd_html)


if __name__ == "__main__":
    unittest.main()
