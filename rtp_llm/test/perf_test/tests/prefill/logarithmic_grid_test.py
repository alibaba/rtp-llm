"""Deterministic B1 logarithmic coverage and frozen plan contracts."""

import copy
import json
import tempfile
import unittest
from pathlib import Path

from rtp_llm.test.perf_test.config_resolver import resolve_config
from rtp_llm.test.perf_test.config_schema import ConfigError, parse_config
from rtp_llm.test.perf_test.entry import plan_from_source
from rtp_llm.test.perf_test.errors import PlanningError
from rtp_llm.test.perf_test.plan import ExecutionPlan
from rtp_llm.test.perf_test.prefill.planner import plan_prefill_cases
from rtp_llm.test.perf_test.tests.support.fixtures import fixture_config


def source_config(**fields):
    source = fixture_config()
    source["engine"].update(
        max_seq_len=1048576, max_batch_tokens=1048576, concurrency_limit=32
    )
    source["workload"] = dict(
        kind="cache_grid",
        transport="dashsc_input_ids",
        cache_alignment=1024,
        probe_block_size=1024,
        commit_tail_tokens=1024,
        workspace_tokens=1048576,
        measure_runs=2,
        seed=2026,
        logarithmic_request_batches=dict(
            min_input_tokens=16384, max_input_tokens=1047552, **fields
        ),
    )
    return source


def cases_for(source):
    config = parse_config(source)
    return plan_prefill_cases(
        config.workload, config.engine, dp_size=config.model.topology.dp
    )


class LogarithmicGridTest(unittest.TestCase):
    def test_three_layers_yield_438_unique_b1_cases_at_25_input_levels(self):
        cases = cases_for(source_config())
        pairs = [(c["input_len"], c["cache_len"]) for c in cases]
        self.assertEqual(438, len(cases))
        self.assertEqual(sorted(set(pairs)), pairs)
        lengths = sorted({length for length, _ in pairs})
        self.assertEqual(25, len(lengths))
        self.assertEqual((16384, 1047552), (lengths[0], lengths[-1]))
        for length in lengths:
            self.assertIn((length, 0), pairs)
            self.assertIn((length, length - 1024), pairs)
        self.assertIn((524288, 516096), pairs)  # D=8K
        self.assertIn((524288, 393216), pairs)  # 75% cache
        for case in cases:
            self.assertEqual(1, case["batch_size"])
            self.assertEqual(2, case["measure_runs"])
            self.assertEqual(1, case["request_groups"][0]["count"])
            self.assertEqual(0, case["cache_len"] % 1024)
            self.assertGreaterEqual(case["compute_len"], 1024)
        self.assertEqual(cases, cases_for(source_config()))

    def test_ratio_layers_and_full_reuse_clamp_deduplicate(self):
        self.assertEqual(356, len(cases_for(source_config(cache_hit_ratios=[]))))
        source = source_config(cache_hit_ratios=[0, 0.95, 0.99, 1])
        source["workload"]["logarithmic_request_batches"].update(
            min_input_tokens=16384, max_input_tokens=16384
        )
        pairs = [(c["input_len"], c["cache_len"]) for c in cases_for(source)]
        self.assertEqual(1, pairs.count((16384, 15360)))

    def test_alignment_endpoints_and_commit_tail(self):
        source = source_config()
        source["workload"]["commit_tail_tokens"] = 2048
        source["workload"]["logarithmic_request_batches"].update(
            min_input_tokens=16500, max_input_tokens=33700
        )
        cases = cases_for(source)
        self.assertEqual(17408, min(c["input_len"] for c in cases))
        self.assertEqual(32768, max(c["input_len"] for c in cases))
        self.assertTrue(all(c["compute_len"] >= 2048 for c in cases))

    def test_schema_rejects_batched_invalid_and_boundary_parameters(self):
        for fields in (
            dict(batch_sizes=[2]),
            dict(batch_sizes=[1, 2]),
            dict(input_points_per_doubling=0),
            dict(compute_points_per_doubling=True),
            dict(input_alignment=0),
            dict(min_compute_tokens=0),
            dict(cache_hit_ratios=[float("nan")]),
            dict(cache_hit_ratios=[True]),
            dict(cache_hit_ratios=[-0.1]),
            dict(cache_hit_ratios=[1.1]),
            dict(cache_hit_ratios=[0.5, 0.5]),
            dict(boundary_points=[12288]),
        ):
            with self.subTest(fields=fields), self.assertRaises(ConfigError):
                parse_config(source_config(**fields))

    def test_sources_are_mutually_exclusive(self):
        for extra in (
            dict(batch_sizes=[1]),
            dict(cache_lengths=[0]),
            dict(request_batches=[]),
            dict(random_request_batches={}),
            dict(traffic_request_batches={}),
        ):
            source = source_config()
            source["workload"].update(extra)
            with self.subTest(extra=extra), self.assertRaises(ConfigError):
                parse_config(source)

    def test_admission_and_max_cases_reject_instead_of_dropping_coverage(self):
        for section, key, value, message in (
            ("engine", "max_seq_len", 1047552, "output"),
            ("engine", "max_batch_tokens", 32768, "token"),
            ("workload", "workspace_tokens", 32768, "workspace"),
            ("workload", "max_cases", 437, "max_cases"),
        ):
            source = source_config()
            source[section][key] = value
            with self.subTest(key=key), self.assertRaisesRegex(PlanningError, message):
                cases_for(source)

    def test_impossible_ranges_and_physical_alignment_fail(self):
        for fields in (
            dict(min_input_tokens=16385, max_input_tokens=17000),
            dict(min_input_tokens=32768, max_input_tokens=16384),
            dict(input_alignment=256),
            dict(min_compute_tokens=32768),
        ):
            source = source_config()
            source["workload"]["logarithmic_request_batches"].update(fields)
            with self.subTest(fields=fields), self.assertRaises(
                (ConfigError, PlanningError)
            ):
                cases_for(source)

    def test_legacy_normalization_does_not_gain_new_null_fields(self):
        resolved = resolve_config(fixture_config())
        self.assertNotIn(
            "logarithmic_request_batches", resolved.resolved_config["workload"]
        )

    def test_explicit_export_preserves_case_identities(self):
        from rtp_llm.test.perf_test.prefill.export import _freeze_config

        source = source_config()
        frozen = _freeze_config(source, source_dir=Path.cwd())
        self.assertNotIn("logarithmic_request_batches", frozen["workload"])
        self.assertEqual(cases_for(source), cases_for(frozen))
        self.assertEqual(source_config(), source)

    def test_local_aone_and_remote_runfiles_preserve_frozen_cases(self):
        from rtp_llm.test.perf_test.aone.gang_worker import _merge_workload

        for backend in ("local", "aone"):
            for named in (False, True):
                with self.subTest(
                    backend=backend, named=named
                ), tempfile.TemporaryDirectory() as directory:
                    root = Path(directory)
                    source = source_config()
                    if backend == "aone":
                        source["execution"] = dict(
                            backend="aone",
                            resource="daily",
                            gpu_type="H20",
                            shard_count=2,
                            max_parallel=2,
                            gpus_per_machine=1,
                        )
                    if named:
                        spec = source["workload"].pop("logarithmic_request_batches")
                        source["workload"]["grids"] = [
                            dict(name="b1", logarithmic_request_batches=spec)
                        ]
                    plan, _ = plan_from_source(source, source_dir=root)
                    self.assertEqual(438, len(plan.cases))
                    self.assertEqual(
                        plan,
                        ExecutionPlan.from_dict(json.loads(json.dumps(plan.to_dict()))),
                    )
                    config = root / "config.json"
                    workload = root / "workload.json"
                    normalized = plan.to_dict()["resolved_config"]
                    config.write_text(json.dumps(normalized))
                    workload.write_text(json.dumps(normalized["workload"]))
                    merged = _merge_workload(config, workload, root)
                    remote, _ = plan_from_source(
                        json.loads(merged.read_text()), source_dir=root
                    )
                    self.assertEqual(plan.cases, remote.cases)
                    self.assertEqual(plan.config_fingerprint, remote.config_fingerprint)
                    self.assertEqual(
                        plan.workload_fingerprint, remote.workload_fingerprint
                    )

    def test_existing_geometry_keeps_case_ids_when_ratio_slices_change(self):
        source = source_config(cache_hit_ratios=[])
        base, _ = plan_from_source(source, source_dir=Path.cwd())
        richer = copy.deepcopy(source)
        richer["workload"]["logarithmic_request_batches"]["cache_hit_ratios"] = [0.75]
        extended, _ = plan_from_source(richer, source_dir=Path.cwd())
        self.assertTrue(
            {c.case_id for c in base.cases} <= {c.case_id for c in extended.cases}
        )


if __name__ == "__main__":
    unittest.main()
