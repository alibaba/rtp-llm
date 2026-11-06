"""Core performance behavior contracts."""

import json
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path

from rtp_llm.test.perf_test.cli import parse_cli
from rtp_llm.test.perf_test.config_resolver import load_resolved_config
from rtp_llm.test.perf_test.config_schema import parse_config
from rtp_llm.test.perf_test.decode.planner import plan_decode_cases
from rtp_llm.test.perf_test.entry import build_request_plan
from rtp_llm.test.perf_test.errors import PlanningError
from rtp_llm.test.perf_test.plan import case_id
from rtp_llm.test.perf_test.prefill.planner import plan_prefill_cases
from rtp_llm.test.perf_test.prefill.prompts import (
    PrefixPromptFactory,
    build_grouped_prompts,
)
from rtp_llm.test.perf_test.prefill.traffic_batch import sample_traffic_batches
from rtp_llm.test.perf_test.prefill.traffic_distribution import (
    parse_length_distribution,
)
from rtp_llm.test.perf_test.test_support import (
    config_resolution_base_config,
    fixture_engine,
    fixture_execution,
    fixture_model,
    prefill_external_prefill_source,
)


class prefill_prompt_WhitespaceTokenizer:

    def encode(self, text):
        return text.split()


def decode_distribution_source(distribution_file="requests.json"):
    return {
        "schema_version": 1,
        "model": fixture_model(),
        "engine": fixture_engine(),
        "workload": {
            "kind": "distribution",
            "distribution_file": distribution_file,
            "output_length": 32,
            "batch": {"strategy": "fixed", "sizes": [2, 3]},
            "measure_runs": 2,
            "warmup_runs": 1,
            "generate_config": {"top_k": 1},
        },
        "execution": fixture_execution(),
        "output": {},
        "regression": {},
        "publish": {},
    }


random_batch_export_MODULE = "rtp_llm.test.perf_test.prefill.generate_random_batch_grid"


def random_batch_export_random_config():
    return {
        "schema_version": 1,
        "model": fixture_model(
            label="random-test",
            type="qwen_2",
            checkpoint="/models/test",
            tokenizer="/models/test",
        ),
        "engine": fixture_engine(
            max_seq_len=4097, max_batch_tokens=16384, concurrency_limit=4
        ),
        "workload": {
            "kind": "cache_grid",
            "cache_alignment": 64,
            "commit_tail_tokens": 64,
            "measure_runs": 2,
            "workspace_tokens": 16384,
            "random_request_batches": {
                "num_cases": 6,
                "batch_sizes": [2, 4],
                "min_input_tokens": 256,
                "max_input_tokens": 2048,
                "input_alignment": 256,
                "batch_input_limits": {"4": 1024},
            },
        },
        "execution": fixture_execution(),
        "output": {"result_root": "perf_test_results"},
        "regression": {},
        "publish": {},
    }


def random_batch_export_planned(source):
    config = parse_config(source)
    return plan_prefill_cases(
        config.workload, config.engine, dp_size=config.model.topology.dp
    )


traffic_batch_export_MODULE = (
    "rtp_llm.test.perf_test.prefill.generate_traffic_batch_grid"
)


def traffic_batch_config():
    raw = config_resolution_base_config()
    raw["workload"] = {
        "kind": "cache_grid",
        "cache_alignment": 8,
        "commit_tail_tokens": 8,
        "workspace_tokens": 4096,
        "traffic_request_batches": {
            "distribution_file": "distribution.json",
            "num_cases": 3,
            "batch_sizes": [1, 2],
            "min_input_tokens": 100,
        },
    }
    return raw


def traffic_distribution_payload():
    return dict(
        schema_version=1,
        kind="conditional_request_lengths",
        input_bucket_width=100,
        reuse_ratio_bins=2,
        source_rows=30,
        incomplete_rows=0,
        cells=[
            dict(input_bucket=1, reuse_bin=b, count=10, input_min=100, input_max=199)
            for b in (-1, 2, 1)
        ],
    )


class TrafficBatchTest(unittest.TestCase):

    def test_sampling_cycles_and_preserves_input_and_projection(self):
        raw = traffic_batch_config()
        raw["workload"]["traffic_request_batches"]["max_input_tokens"] = 100
        parsed = parse_config(raw)
        sample = sample_traffic_batches(
            parsed.workload,
            parsed.engine,
            parse_length_distribution(traffic_distribution_payload()),
            dp_size=1,
        )
        self.assertEqual([1, 2, 1], [len(b.request_groups) for b in sample.batches])
        self.assertEqual([1, 2, 3], [b.sample_index for b in sample.batches])
        for batch in sample.batches:
            for group in batch.request_groups:
                self.assertEqual(100, group.input_len)
                self.assertEqual(0, group.cache_len % 8)
                self.assertLessEqual(group.cache_len, 92)

    def test_relative_resolution_and_content_fingerprint(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            (root / "config.json").write_text(json.dumps(traffic_batch_config()))
            path = root / "distribution.json"
            distribution = traffic_distribution_payload()
            path.write_text(json.dumps(distribution))
            first = load_resolved_config(root / "config.json")
            self.assertNotIn(
                "distribution_file", json.dumps(first.resolved_config["workload"])
            )
            self.assertEqual(
                first.workload_fingerprint,
                load_resolved_config(root / "config.json").workload_fingerprint,
            )
            distribution["incomplete_rows"] = 1
            path.write_text(json.dumps(distribution))
            self.assertNotEqual(
                first.workload_fingerprint,
                load_resolved_config(root / "config.json").workload_fingerprint,
            )

    def test_budget_limits_and_retry_exhaustion(self):
        for field, value in [
            ("max_batch_tokens", 200),
            ("workspace_tokens", 207),
            ("kv_budget_tokens", 207),
        ]:
            raw = traffic_batch_config()
            spec = raw["workload"]["traffic_request_batches"]
            spec.update(
                num_cases=1, batch_sizes=[2], max_input_tokens=100, retry_limit=2
            )
            if field == "max_batch_tokens":
                raw["engine"][field] = value
            elif field == "workspace_tokens":
                raw["workload"][field] = value
            else:
                spec[field] = value
            parsed = parse_config(raw)
            with self.subTest(field=field), self.assertRaisesRegex(
                PlanningError, "retry limit"
            ):
                sample_traffic_batches(
                    parsed.workload,
                    parsed.engine,
                    parse_length_distribution(traffic_distribution_payload()),
                    dp_size=1,
                )


class RandomBatchExportTest(unittest.TestCase):

    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.root = Path(self.temp.name)
        self.source = self.root / "source.json"

    def invoke(self, source, *arguments, success=True):
        self.source.write_text(json.dumps(source), encoding="utf-8")
        result = subprocess.run(
            [
                sys.executable,
                "-m",
                random_batch_export_MODULE,
                "--config",
                str(self.source),
                *map(str, arguments),
            ],
            capture_output=True,
            text=True,
        )
        if success:
            self.assertEqual(0, result.returncode, result.stderr)
        else:
            self.assertNotEqual(0, result.returncode)
        return result

    def test_mixed_export_freezes_all_sampled_groups_and_preserves_full_config(self):
        source = random_batch_export_random_config()
        dest = self.root / "mixed.json"
        self.invoke(source, "--output", dest)
        frozen = json.loads(dest.read_text())
        self.assertNotIn("random_request_batches", frozen["workload"])
        self.assertEqual(6, len(frozen["workload"]["request_batches"]))
        self.assertEqual(
            random_batch_export_planned(source), random_batch_export_planned(frozen)
        )
        for key in ("model", "engine", "execution", "output"):
            self.assertEqual(source[key], frozen[key])
        self.assertEqual(
            {2, 4}, {case["batch_size"] for case in random_batch_export_planned(frozen)}
        )
        self.assertEqual(
            {"4": 1024},
            source["workload"]["random_request_batches"]["batch_input_limits"],
        )
        again = self.root / "again.json"
        self.invoke(source, "--output", again)
        self.assertEqual(dest.read_bytes(), again.read_bytes())


class TrafficBatchExportTest(unittest.TestCase):

    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.root = Path(self.temp.name)
        self.source = self.root / "source.json"
        distribution = traffic_distribution_payload()
        distribution.update(
            input_bucket_width=1000,
            source_rows=1,
            cells=[
                dict(
                    input_bucket=0, reuse_bin=-1, count=1, input_min=100, input_max=100
                )
            ],
        )
        (self.root / "distribution.json").write_text(json.dumps(distribution))

    def write_source(self, named=False):
        raw = traffic_batch_config()
        raw["workload"]["traffic_request_batches"].update(
            num_cases=4, batch_sizes=[1, 4], max_input_tokens=100
        )
        if named:
            spec = raw["workload"].pop("traffic_request_batches")
            raw["workload"]["grids"] = [
                {"name": "traffic", "traffic_request_batches": spec}
            ]
        raw["output"]["result_root"] = str(self.root / "results")
        self.source.write_text(json.dumps(raw))
        return raw

    def plan(self, path):
        return build_request_plan(
            parse_cli(["--mode=prefill", "--action=run", "--config", str(path)])
        )[0]

    def test_export_round_trip_and_refuses_overwrite(self):
        self.write_source(named=True)
        dest = self.root / "frozen.json"
        command = [
            sys.executable,
            "-m",
            traffic_batch_export_MODULE,
            "--config",
            str(self.source),
            "--output",
            str(dest),
        ]
        result = subprocess.run(command, capture_output=True, text=True)
        self.assertEqual(0, result.returncode, result.stderr)
        content = dest.read_text()
        self.assertNotIn("distribution_file", content)
        self.assertNotIn("traffic_request_batches", content)
        self.assertEqual(
            [case.case_id for case in self.plan(self.source).cases],
            [case.case_id for case in self.plan(dest).cases],
        )
        result = subprocess.run(command, capture_output=True, text=True)
        self.assertNotEqual(0, result.returncode)
        self.assertIn("refusing to overwrite", result.stderr)
        self.assertEqual(content, dest.read_text())


class PrefillPromptTest(unittest.TestCase):

    def test_cached_and_cold_prompts_have_exact_lengths(self):
        for case_id, total_len, cache_len, run_count in (
            ("cached", 16, 8, 3),
            ("cold", 12, 0, 2),
        ):
            with self.subTest(case_id=case_id):
                prompts = PrefixPromptFactory(
                    prefill_prompt_WhitespaceTokenizer()
                ).build_case_prompts(case_id, total_len, cache_len, run_count)
                self.assertEqual(cache_len, len(prompts.prefix_ids))
                self.assertEqual(total_len, prompts.built_len)
                self.assertEqual(
                    [total_len] * run_count,
                    [len(request) for request in prompts.run_ids],
                )

    def test_grouped_reconstruction_honors_both_prefix_policies(self):
        for policy in ("shared_by_group", "independent"):
            with self.subTest(policy=policy):
                case = {
                    "case_id": 7,
                    "batch_size": 3,
                    "input_len": 16,
                    "cache_len": 8,
                    "prefix_policy": policy,
                    "request_groups": [
                        {"count": 2, "input_len": 16, "cache_len": 8},
                        {"count": 1, "input_len": 12, "cache_len": 4},
                    ],
                }
                members = build_grouped_prompts(
                    PrefixPromptFactory(prefill_prompt_WhitespaceTokenizer()),
                    case,
                    run_count=2,
                )
                self.assertEqual(3, len(members))
                self.assertEqual([16, 16, 12], [m["input_len"] for m in members])
                first_prefix = members[0]["prompts"].prefix_ids
                second_prefix = members[1]["prompts"].prefix_ids
                if policy == "shared_by_group":
                    self.assertEqual(first_prefix, second_prefix)
                else:
                    self.assertNotEqual(first_prefix, second_prefix)


class GroupedPrefillPlanningTest(unittest.TestCase):

    def source(self):
        source = prefill_external_prefill_source("results")
        source["model"]["topology"].update(dp=1, ep=1)
        source["execution"]["devices"] = [0]
        source["engine"].update(concurrency_limit=4, max_batch_tokens=80)
        source["workload"] = {
            "kind": "cache_grid",
            "cache_alignment": 8,
            "commit_tail_tokens": 8,
            "workspace_tokens": 88,
            "request_batches": [
                {
                    "request_groups": [
                        {"count": 2, "input_len": 24, "cache_len": 16},
                        {"count": 1, "input_len": 17, "cache_len": 8},
                    ]
                }
            ],
        }
        return source

    def test_grouped_geometry_uses_member_maxima_and_independent_seeds(self):
        config = parse_config(self.source())
        (case,) = plan_prefill_cases(config.workload, config.engine)
        self.assertEqual("concurrent_grouped_batch_v1", case["measurement_contract"])
        self.assertEqual(
            (3, 24, 16), (case["batch_size"], case["input_len"], case["cache_len"])
        )
        self.assertEqual("independent", case["prefix_policy"])
        self.assertEqual(2, case["request_groups"][0]["count"])


class DecodeDistributionTest(unittest.TestCase):

    def test_materialized_sequence_lengths_generate_deterministic_actual_cases(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)
            (root / "requests.json").write_text(
                json.dumps({"sequence_lengths": [64, 128, 256, 512]}), encoding="utf-8"
            )
            config_path = root / "perf.json"
            config_path.write_text(
                json.dumps(decode_distribution_source()), encoding="utf-8"
            )
            resolved = load_resolved_config(config_path)
            parsed = parse_config(decode_distribution_source())
            first = plan_decode_cases(
                parsed.workload,
                parsed.engine,
                dp_size=1,
                resolved_workload=resolved.resolved_config["workload"],
            )
            second = plan_decode_cases(
                parsed.workload,
                parsed.engine,
                dp_size=1,
                resolved_workload=resolved.resolved_config["workload"],
            )
            self.assertEqual(first, second)
            self.assertEqual(
                [[64, 128], [64, 128, 256]], [case["input_lengths"] for case in first]
            )
            self.assertEqual([2, 3], [case["batch_policy"]["size"] for case in first])
            self.assertEqual({32}, {case["output_length"] for case in first})


if __name__ == "__main__":
    unittest.main()
