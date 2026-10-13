import argparse
import csv
import hashlib
import json
import os
import re
import sys
import tempfile
import unittest
from pathlib import Path
from unittest.mock import MagicMock, Mock, patch

from rtp_llm.test.perf_test.batch_decode_test import (
    _effective_grid_max_seq_len,
    _effective_performance_config,
    _engine_arg_argv,
    _fingerprint_engine_env,
    _load_cache_grid_cases,
    _parse_name_value,
    _redact_argv,
    _require_cache_grid_success,
    main,
    parse_args,
)
from rtp_llm.test.perf_test.cache_grid_runner import (
    CacheGridRunner,
    PrefixPromptFactory,
    _post_prefill,
)
from rtp_llm.test.perf_test.deepseek_v4_prefill_formula_fit import (
    FEATURE_NAMES,
    build_parser as build_formula_parser,
    formula_text,
    load_observations,
    run_fit,
)
from rtp_llm.test.perf_test.generate_prefill_3d_chart import (
    load_rows,
    render_cold_miss_2d,
)
from rtp_llm.test.perf_test.perf_config import (
    _apply_engine_env,
    _apply_run_overrides,
)
from rtp_llm.test.perf_test.perf_utils import write_test_info


class _WhitespaceTokenizer:
    def encode(self, text):
        return text.split()


class BatchDecodeTest(unittest.TestCase):
    def test_effective_grid_max_seq_len_uses_decode_need(self):
        args = argparse.Namespace(max_seq_len=8192, decode_test_length=30)
        self.assertEqual(_effective_grid_max_seq_len(args, [1024, 65536]), 65566)

    def test_effective_grid_max_seq_len_respects_explicit_headroom(self):
        args = argparse.Namespace(max_seq_len=65664, decode_test_length=30)
        self.assertEqual(_effective_grid_max_seq_len(args, [65536]), 65664)

    def test_cache_grid_loader_validates_explicit_cases(self):
        payload = {
            "cases": [
                {"case_id": 7, "batch_size": 1, "input_len": 4096, "cache_len": 0},
                {
                    "case_id": 8,
                    "batch_size": 1,
                    "input_len": 4096,
                    "cache_len": 2048,
                },
            ]
        }
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "cache_grid.json"
            path.write_text(json.dumps(payload), encoding="utf-8")
            self.assertEqual(_load_cache_grid_cases(str(path)), payload["cases"])

    def test_cache_grid_loader_rejects_duplicate_geometry(self):
        payload = {
            "cases": [
                {"batch_size": 1, "input_len": 4096, "cache_len": 2048},
                {"batch_size": 1, "input_len": 4096, "cache_len": 2048},
            ]
        }
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "cache_grid.json"
            path.write_text(json.dumps(payload), encoding="utf-8")
            with self.assertRaisesRegex(ValueError, "duplicate cache grid case"):
                _load_cache_grid_cases(str(path))

    def test_parse_args_exposes_cache_runner_controls(self):
        args, remaining = parse_args()
        self.assertEqual(args.cache_measure_runs, 3)
        self.assertGreater(args.cache_request_timeout, 0)
        self.assertEqual(args.cache_commit_tail_tokens, 4096)
        self.assertEqual(args.cache_grid_json, "")
        self.assertIsInstance(remaining, list)

    def test_generated_cache_grid_uses_independent_seq_and_cache_alignment(self):
        payload = {
            "seq_generation": {
                "kind": "linear_with_dense_prefix",
                "count": 20,
                "max_seq_len": 65535,
            },
            "seq_block_size": 256,
            "cache_block_size": 4096,
        }
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "cache_grid.json"
            path.write_text(json.dumps(payload), encoding="utf-8")
            cases = _load_cache_grid_cases(str(path))
        self.assertTrue(any(case["input_len"] == 256 for case in cases))
        self.assertTrue(
            all(
                case["cache_len"] == 0 or case["cache_len"] % 4096 == 0
                for case in cases
            )
        )
        self.assertTrue(
            all(
                case["cache_len"] == 0
                or case["cache_len"] + 4096 <= case["input_len"]
                for case in cases
            )
        )

    def test_cache_seed_commits_one_tail_and_preserves_exact_prefix(self):
        tokenizer = _WhitespaceTokenizer()
        factory = PrefixPromptFactory(tokenizer)
        target, prefix, built_len = factory.make_case(7, 32, 16)
        seed = factory.make_seed(7, prefix, 16, 8)
        prefix_ids = tokenizer.encode(prefix)
        self.assertEqual(built_len, 32)
        self.assertEqual(len(prefix_ids), 16)
        self.assertEqual(len(tokenizer.encode(seed)), 24)
        self.assertEqual(tokenizer.encode(target)[:16], prefix_ids)
        self.assertEqual(tokenizer.encode(seed)[:16], prefix_ids)

    def test_case_prefixes_are_isolated(self):
        tokenizer = _WhitespaceTokenizer()
        factory = PrefixPromptFactory(tokenizer)
        _, prefix_a, _ = factory.make_case(1, 32, 16)
        _, prefix_b, _ = factory.make_case(2, 32, 16)
        self.assertNotEqual(tokenizer.encode(prefix_a), tokenizer.encode(prefix_b))

    @patch("rtp_llm.test.perf_test.cache_grid_runner.requests.post")
    def test_post_prefill_records_client_ttft_separately(self, post):
        response = Mock(status_code=200)
        response.json.return_value = {
            "aux_info": {
                "input_len": 1048575,
                "output_len": 1,
                "reuse_len": 0,
                "first_token_cost_time": 173.0,
                "cost_time": 175.0,
                "wait_time": 2.0,
            }
        }
        post.return_value = response
        result = _post_prefill(12345, "prompt", 10, "case:run0")
        self.assertTrue(result["success"])
        self.assertEqual(result["prefill_time_ms"], 173.0)
        self.assertGreaterEqual(result["ttft_ms"], 0.0)
        self.assertEqual(result["ttft_ms"], result["client_wall_time_ms"])
        self.assertEqual(result["ttft_source"], "client_http_wall_max_new_tokens_1")

    @patch("rtp_llm.test.perf_test.cache_grid_runner._post_prefill")
    def test_runner_accepts_only_exact_shape_reuse_and_ttft(self, post):
        post.side_effect = [
            {"success": True},
            {
                "success": True,
                "input_len": 16,
                "output_len": 1,
                "reuse_len": 8,
                "ttft_ms": 12.0,
            },
            {
                "success": True,
                "input_len": 16,
                "output_len": 1,
                "reuse_len": 8,
                "ttft_ms": 10.0,
            },
            {
                "success": True,
                "input_len": 16,
                "output_len": 1,
                "reuse_len": 8,
                "ttft_ms": 11.0,
            },
        ]
        with tempfile.TemporaryDirectory() as tmp:
            rows = CacheGridRunner(
                12345,
                _WhitespaceTokenizer(),
                [{"case_id": 1, "batch_size": 1, "input_len": 16, "cache_len": 8}],
                tmp,
                cache_commit_tail_tokens=8,
            ).run()
        self.assertEqual(rows[0]["status"], "ok")
        self.assertTrue(rows[0]["shape_exact"])
        self.assertTrue(rows[0]["reuse_exact"])
        self.assertTrue(rows[0]["timing_valid"])
        self.assertEqual(rows[0]["median_ttft_ms"], 11.0)

    @patch("rtp_llm.test.perf_test.cache_grid_runner._post_prefill")
    def test_cache_grid_persists_partial_failure_and_fails_command(self, post):
        cases = [
            {"case_id": 1, "batch_size": 1, "input_len": 8, "cache_len": 0},
            {"case_id": 2, "batch_size": 1, "input_len": 12, "cache_len": 0},
        ]
        post.side_effect = [
            {
                "success": True,
                "input_len": 8,
                "output_len": 1,
                "reuse_len": 0,
                "ttft_ms": 5.0,
            },
            {"success": False, "error": "request failed"},
        ]
        with tempfile.TemporaryDirectory() as tmp:
            rows = CacheGridRunner(
                12345,
                _WhitespaceTokenizer(),
                cases,
                tmp,
                measure_runs=1,
                cache_commit_tail_tokens=4,
            ).run()
            checkpoint = json.loads(
                (Path(tmp) / "cache_grid_results.json").read_text()
            )

        self.assertEqual([row["status"] for row in rows], ["ok", "failed"])
        self.assertEqual(checkpoint["recorded_cases"], 2)
        self.assertEqual(checkpoint["completed_cases"], 1)
        self.assertFalse(checkpoint["complete"])
        with self.assertRaisesRegex(RuntimeError, r"failed 1/2 cases.*failed=1"):
            _require_cache_grid_success(rows)

    @patch("rtp_llm.test.perf_test.cache_grid_runner._post_prefill")
    def test_cache_grid_persists_total_failure_and_fails_command(self, post):
        cases = [
            {"case_id": 1, "batch_size": 1, "input_len": 8, "cache_len": 0},
            {"case_id": 2, "batch_size": 1, "input_len": 12, "cache_len": 0},
        ]
        post.return_value = {"success": False, "error": "request failed"}
        with tempfile.TemporaryDirectory() as tmp:
            rows = CacheGridRunner(
                12345,
                _WhitespaceTokenizer(),
                cases,
                tmp,
                measure_runs=1,
                cache_commit_tail_tokens=4,
            ).run()
            checkpoint = json.loads(
                (Path(tmp) / "cache_grid_results.json").read_text()
            )

        self.assertEqual([row["status"] for row in rows], ["failed", "failed"])
        self.assertEqual(checkpoint["recorded_cases"], 2)
        self.assertEqual(checkpoint["completed_cases"], 0)
        self.assertFalse(checkpoint["complete"])
        with self.assertRaisesRegex(RuntimeError, r"failed 2/2 cases.*failed=2"):
            _require_cache_grid_success(rows)

    @staticmethod
    def _checkpoint_record(runner, case, status):
        record = {
            "case_key": runner.case_key(case),
            "case_id": case["case_id"],
            "batch_size": case["batch_size"],
            "input_len": case["input_len"],
            "cache_len_requested": case["cache_len"],
            "measure_runs": runner.measure_runs,
            "cache_commit_tail_tokens": runner.cache_commit_tail_tokens,
            "run_fingerprint": runner.run_fingerprint,
            "status": status,
        }
        if status == "ok":
            latency = 5.0
            record.update(
                {
                    "success_runs": runner.measure_runs,
                    "cache_len_observed": [case["cache_len"]] * runner.measure_runs,
                    "reuse_exact": True,
                    "shape_exact": True,
                    "timing_valid": True,
                    "ttft_ms": [latency] * runner.measure_runs,
                    "median_ttft_ms": latency,
                    "avg_ttft_ms": latency,
                    "seed": {"success": True} if case["cache_len"] else {},
                    "runs": [
                        {
                            "success": True,
                            "input_len": case["input_len"],
                            "output_len": 1,
                            "reuse_len": case["cache_len"],
                            "ttft_ms": latency,
                        }
                        for _ in range(runner.measure_runs)
                    ],
                }
            )
        return record

    @staticmethod
    def _valid_audited_metric(input_len=16, cache_len=8):
        latency = 12.0
        return {
            "status": "ok",
            "batch_size": 1,
            "input_len": input_len,
            "cache_len_requested": cache_len,
            "cache_len_observed": [cache_len],
            "reuse_exact": True,
            "shape_exact": True,
            "timing_valid": True,
            "measure_runs": 1,
            "success_runs": 1,
            "seed": {"success": True} if cache_len else {},
            "ttft_ms": [latency],
            "median_ttft_ms": latency,
            "avg_ttft_ms": latency,
            "runs": [
                {
                    "success": True,
                    "input_len": input_len,
                    "output_len": 1,
                    "reuse_len": cache_len,
                    "ttft_ms": latency,
                }
            ],
        }

    @staticmethod
    def _audited_payload(metrics, *, complete=True, fingerprint_config=None):
        fingerprint_config = fingerprint_config or {"test": True}
        fingerprint = hashlib.sha256(
            json.dumps(
                fingerprint_config,
                sort_keys=True,
                separators=(",", ":"),
                ensure_ascii=True,
            ).encode("utf-8")
        ).hexdigest()
        records = [
            {**metric, "run_fingerprint": fingerprint} for metric in metrics
        ]
        return {
            "schema_version": 2,
            "mode": "prefix_cache_grid",
            "run_fingerprint": fingerprint,
            "fingerprint_config": fingerprint_config,
            "complete": complete,
            "total_cases": len(records),
            "completed_cases": len(records) if complete else 0,
            "metrics": records,
        }

    @patch("rtp_llm.test.perf_test.cache_grid_runner._post_prefill")
    def test_cache_grid_resume_reuses_only_successful_matching_records(self, post):
        cases = [
            {"case_id": 1, "batch_size": 1, "input_len": 8, "cache_len": 0},
            {"case_id": 2, "batch_size": 1, "input_len": 12, "cache_len": 0},
        ]
        post.return_value = {
            "success": True,
            "input_len": 12,
            "output_len": 1,
            "reuse_len": 0,
            "ttft_ms": 7.0,
        }
        with tempfile.TemporaryDirectory() as tmp:
            initial = CacheGridRunner(
                12345,
                _WhitespaceTokenizer(),
                cases,
                tmp,
                measure_runs=1,
                cache_commit_tail_tokens=4,
                run_config={"model": "same"},
            )
            successful = self._checkpoint_record(initial, cases[0], "ok")
            failed = self._checkpoint_record(initial, cases[1], "failed")
            initial._results = {
                successful["case_key"]: successful,
                failed["case_key"]: failed,
            }
            initial._save()

            resumed = CacheGridRunner(
                12345,
                _WhitespaceTokenizer(),
                cases,
                tmp,
                measure_runs=1,
                cache_commit_tail_tokens=4,
                run_config={"model": "same"},
            )
            rows = resumed.run()
            checkpoint = json.loads(
                (Path(tmp) / "cache_grid_results.json").read_text()
            )

        post.assert_called_once()
        self.assertEqual({row["status"] for row in rows}, {"ok"})
        self.assertTrue(checkpoint["complete"])
        self.assertEqual(checkpoint["completed_cases"], 2)

    @patch("rtp_llm.test.perf_test.cache_grid_runner._post_prefill")
    def test_cache_grid_resume_reruns_stale_fingerprint(self, post):
        case = {"case_id": 1, "batch_size": 1, "input_len": 8, "cache_len": 0}
        post.return_value = {
            "success": True,
            "input_len": 8,
            "output_len": 1,
            "reuse_len": 0,
            "ttft_ms": 5.0,
        }
        with tempfile.TemporaryDirectory() as tmp:
            initial = CacheGridRunner(
                12345,
                _WhitespaceTokenizer(),
                [case],
                tmp,
                measure_runs=1,
                cache_commit_tail_tokens=4,
                run_config={"model": "old"},
            )
            record = self._checkpoint_record(initial, case, "ok")
            initial._results = {record["case_key"]: record}
            initial._save(complete=True)

            resumed = CacheGridRunner(
                12345,
                _WhitespaceTokenizer(),
                [case],
                tmp,
                measure_runs=1,
                cache_commit_tail_tokens=4,
                run_config={"model": "new"},
            )
            resumed.run()

        post.assert_called_once()
        self.assertNotEqual(initial.run_fingerprint, resumed.run_fingerprint)

    @patch("rtp_llm.test.perf_test.cache_grid_runner._post_prefill")
    def test_cache_grid_resume_reruns_empty_ok_record(self, post):
        case = {"case_id": 1, "batch_size": 1, "input_len": 8, "cache_len": 0}
        post.return_value = {
            "success": True,
            "input_len": 8,
            "output_len": 1,
            "reuse_len": 0,
            "ttft_ms": 5.0,
        }
        with tempfile.TemporaryDirectory() as tmp:
            initial = CacheGridRunner(
                12345,
                _WhitespaceTokenizer(),
                [case],
                tmp,
                measure_runs=1,
                cache_commit_tail_tokens=4,
            )
            empty_ok = {
                key: value
                for key, value in self._checkpoint_record(initial, case, "ok").items()
                if key
                not in {
                    "success_runs",
                    "cache_len_observed",
                    "reuse_exact",
                    "shape_exact",
                    "timing_valid",
                    "ttft_ms",
                    "median_ttft_ms",
                    "avg_ttft_ms",
                    "runs",
                }
            }
            initial._results = {empty_ok["case_key"]: empty_ok}
            initial._save(complete=True)

            resumed = CacheGridRunner(
                12345,
                _WhitespaceTokenizer(),
                [case],
                tmp,
                measure_runs=1,
                cache_commit_tail_tokens=4,
            )
            rows = resumed.run()

        post.assert_called_once()
        self.assertEqual(rows[0]["status"], "ok")
        self.assertEqual(rows[0]["success_runs"], 1)

    def test_cache_grid_resume_and_audit_share_strict_metric_validation(self):
        case = {"case_id": 1, "batch_size": 1, "input_len": 8, "cache_len": 4}
        with tempfile.TemporaryDirectory() as tmp:
            runner = CacheGridRunner(
                12345,
                _WhitespaceTokenizer(),
                [case],
                tmp,
                measure_runs=1,
                cache_commit_tail_tokens=4,
            )
            valid = self._checkpoint_record(runner, case, "ok")
            self.assertTrue(runner._is_reusable_record(valid))

            mutations = {
                "missing_seed": lambda record: record.pop("seed"),
                "incomplete_ttft": lambda record: record.update({"ttft_ms": []}),
                "per_run_ttft_mismatch": lambda record: record["runs"][0].update(
                    {"ttft_ms": 6.0}
                ),
                "median_mismatch": lambda record: record.update(
                    {"median_ttft_ms": 6.0}
                ),
                "average_mismatch": lambda record: record.update({"avg_ttft_ms": 6.0}),
            }
            for name, mutate in mutations.items():
                with self.subTest(name=name):
                    record = json.loads(json.dumps(valid))
                    mutate(record)
                    self.assertFalse(runner._is_reusable_record(record))
                    payload = self._audited_payload([record])
                    payload["metrics"][0]["run_fingerprint"] = payload[
                        "run_fingerprint"
                    ]
                    path = Path(tmp) / f"{name}.json"
                    path.write_text(json.dumps(payload), encoding="utf-8")
                    observations, audit = load_observations([path])
                    self.assertEqual(observations, [])
                    self.assertFalse(audit["production_audit_passed"])
                    with self.assertRaisesRegex(ValueError, r"metrics\[0\]"):
                        load_rows(path, batch_size=1)

    def test_cache_grid_fingerprint_canonicalizes_cases_and_env_values(self):
        cases = [
            {"case_id": 2, "batch_size": 1, "input_len": 12, "cache_len": 0},
            {"case_id": 1, "batch_size": 1, "input_len": 8, "cache_len": 0},
        ]
        with tempfile.TemporaryDirectory() as first_tmp, tempfile.TemporaryDirectory() as second_tmp:
            first = CacheGridRunner(
                12345,
                _WhitespaceTokenizer(),
                cases,
                first_tmp,
                measure_runs=1,
                cache_commit_tail_tokens=4,
                run_config={"engine_env": {"DSV4_MODE": "enabled"}},
            )
            reordered = CacheGridRunner(
                12345,
                _WhitespaceTokenizer(),
                list(reversed(cases)),
                second_tmp,
                measure_runs=1,
                cache_commit_tail_tokens=4,
                run_config={"engine_env": {"DSV4_MODE": "enabled"}},
            )
            changed_env = CacheGridRunner(
                12345,
                _WhitespaceTokenizer(),
                cases,
                second_tmp,
                measure_runs=1,
                cache_commit_tail_tokens=4,
                run_config={"engine_env": {"DSV4_MODE": "disabled"}},
            )
            lower_token_limit = CacheGridRunner(
                12345,
                _WhitespaceTokenizer(),
                cases,
                second_tmp,
                measure_runs=1,
                cache_commit_tail_tokens=4,
                run_config={
                    "engine_args": _redact_argv(
                        ["--max_batch_tokens_size", "65536"]
                    )
                },
            )
            higher_token_limit = CacheGridRunner(
                12345,
                _WhitespaceTokenizer(),
                cases,
                second_tmp,
                measure_runs=1,
                cache_commit_tail_tokens=4,
                run_config={
                    "engine_args": _redact_argv(
                        ["--max_batch_tokens_size", "131072"]
                    )
                },
            )

        self.assertEqual(first.run_fingerprint, reordered.run_fingerprint)
        self.assertNotEqual(first.run_fingerprint, changed_env.run_fingerprint)
        self.assertNotEqual(
            lower_token_limit.run_fingerprint, higher_token_limit.run_fingerprint
        )

    def test_fingerprint_engine_env_includes_inherited_runtime_without_secrets(self):
        inherited = {
            "TP_SIZE": "4",
            "EP_SIZE": "8",
            "MAX_BATCH_SIZE": "32",
            "MAX_BATCH_TOKENS_SIZE": "131072",
            "SEQ_SIZE_PER_BLOCK": "256",
            "DSV4_CHUNK_TOKENS": "8192",
            "FP8_KV_CACHE": "1",
            "MODEL_TYPE": "deepseek_v4",
            "OSS_ACCESS_KEY_ID": "secret-value",
        }
        with patch.dict(os.environ, inherited, clear=True):
            settings = _fingerprint_engine_env(["OSS_ACCESS_KEY_ID"])

        for name, value in inherited.items():
            if name != "OSS_ACCESS_KEY_ID":
                self.assertEqual(settings[name], value)
        self.assertNotIn("OSS_ACCESS_KEY_ID", settings)
        self.assertNotIn("secret-value", json.dumps(settings))

    def test_fingerprint_hashes_unknown_and_credential_bearing_env_values(self):
        inherited = {
            "MODEL_TYPE": "https://user:password@example.test/model",
            "DSV4_EXPERIMENT_ENDPOINT": "https://host/path?token=secret-value",
            "CUSTOM_RUNTIME_SETTING": "otherwise-benign-value",
        }
        with patch.dict(os.environ, inherited, clear=True):
            settings = _fingerprint_engine_env(["CUSTOM_RUNTIME_SETTING"])

        serialized = json.dumps(settings)
        self.assertEqual(set(settings), set(inherited))
        self.assertTrue(all(value.startswith("sha256:") for value in settings.values()))
        self.assertNotIn("password", serialized)
        self.assertNotIn("secret-value", serialized)
        self.assertNotIn("otherwise-benign-value", serialized)

    def test_effective_runtime_config_prefers_cli_over_engine_env(self):
        with patch.dict(
            os.environ,
            {
                "TP_SIZE": "4",
                "EP_SIZE": "2",
                "MAX_BATCH_SIZE": "32",
                "DSV4_CHUNK_TOKENS": "8192",
                "AUTH_TOKEN": "secret-value",
            },
            clear=True,
        ):
            names = _apply_engine_env(
                ["TP_SIZE=8", "EP_SIZE=8", "AUTH_TOKEN=override-secret"]
            )
            config = _effective_performance_config(
                ["--tp_size", "16", "--max_batch_size=64"],
                names,
                {"MAX_SEQ_LEN": 1048576},
            )

        self.assertEqual(config["values"]["TP_SIZE"], "16")
        self.assertEqual(config["sources"]["TP_SIZE"], "cli")
        self.assertEqual(config["values"]["EP_SIZE"], "8")
        self.assertEqual(config["sources"]["EP_SIZE"], "engine_env")
        self.assertEqual(config["values"]["MAX_BATCH_SIZE"], "64")
        self.assertEqual(config["values"]["MAX_SEQ_LEN"], "1048576")
        self.assertNotIn("AUTH_TOKEN", config["values"])
        self.assertNotIn("secret-value", json.dumps(config))
        self.assertNotIn("override-secret", json.dumps(config))

    def test_engine_arg_shorthand_is_forwarded(self):
        self.assertEqual(
            _engine_arg_argv(["tp_size=8", "fp8_kv_cache=1"]),
            ["--tp_size", "8", "--fp8_kv_cache", "1"],
        )

    def test_name_value_rejects_missing_separator(self):
        with self.assertRaises(ValueError):
            _parse_name_value("tp_size", "--engine_arg")

    def test_parse_args_exposes_runtime_overrides(self):
        args, remaining = parse_args(
            [
                "--engine_arg=tp_size=8",
                "--engine_arg=fp8_kv_cache=1",
                "--engine_env=FP8_KV_CACHE=1",
                "--warmup_runs=2",
                "--measure_runs=3",
                "--profile_runs=0",
                "--model_type=example_model",
            ]
        )
        self.assertEqual(args.engine_arg, ["tp_size=8", "fp8_kv_cache=1"])
        self.assertEqual(args.engine_env, ["FP8_KV_CACHE=1"])
        self.assertEqual(args.warmup_runs, 2)
        self.assertEqual(args.measure_runs, 3)
        self.assertEqual(args.profile_runs, 0)
        self.assertIn("--model_type=example_model", remaining)

    def test_engine_env_and_run_overrides_use_environment_contract(self):
        args, _ = parse_args(
            ["--warmup_runs=2", "--measure_runs=3", "--profile_runs=0"]
        )
        with patch.dict(os.environ, {"EXISTING": "bazel"}, clear=True):
            names = _apply_engine_env(
                ["EXISTING=generic", "NEW_ENGINE_SETTING=enabled"]
            )
            _apply_run_overrides(args)
            self.assertEqual(names, ["EXISTING", "NEW_ENGINE_SETTING"])
            self.assertEqual(os.environ["EXISTING"], "generic")
            self.assertEqual(os.environ["NEW_ENGINE_SETTING"], "enabled")
            self.assertEqual(os.environ["PERF_FORMAL_WARMUP_RUNS"], "2")
            self.assertEqual(os.environ["PERF_MEASURE_RUNS"], "3")
            self.assertEqual(os.environ["PERF_PROFILE_RUNS"], "0")

    def test_run_override_rejects_negative_values(self):
        args, _ = parse_args(["--measure_runs=-1"])
        with self.assertRaisesRegex(ValueError, "--measure_runs must be >= 0"):
            _apply_run_overrides(args)

    def test_redact_argv_hides_secrets_but_keeps_token_count_arguments(self):
        self.assertEqual(
            _redact_argv(
                [
                    "--engine_env=OSS_ACCESS_KEY_ID=secret",
                    "--engine_env",
                    "AUTH_TOKEN=second-secret",
                    "--engine_arg=tp_size=8",
                    "--engine_arg",
                    "max_batch_tokens_size=262144",
                    "--max_batch_tokens_size=131072",
                    "--engine_arg=max_context_tokens=1048576",
                    "--tokenizer_path=/models/tokenizer",
                    "--api_key",
                    "another-secret",
                ]
            ),
            [
                "--engine_env=***",
                "--engine_env",
                "AUTH_TOKEN=***",
                "--engine_arg=tp_size=8",
                "--engine_arg",
                "max_batch_tokens_size=262144",
                "--max_batch_tokens_size=131072",
                "--engine_arg=max_context_tokens=1048576",
                "--tokenizer_path=/models/tokenizer",
                "--api_key",
                "***",
            ],
        )

    def test_redact_argv_hashes_unknown_and_benign_named_credential_values(self):
        redacted = _redact_argv(
            [
                "--checkpoint_path=https://user:password@example.test/model",
                "--custom_transport",
                "Server=db;User Id=alice;Password=secret-value",
                "--engine_env=BENIGN_ENDPOINT=https://host/path?signature=signed-value",
                "--tp_size=8",
            ]
        )
        serialized = json.dumps(redacted)

        self.assertEqual(redacted[-1], "--tp_size=8")
        self.assertRegex(redacted[0], r"^--checkpoint_path=sha256:[0-9a-f]{64}$")
        self.assertRegex(redacted[2], r"^sha256:[0-9a-f]{64}$")
        self.assertRegex(
            redacted[3], r"^--engine_env=BENIGN_ENDPOINT=sha256:[0-9a-f]{64}$"
        )
        for secret in ("password", "alice", "secret-value", "signed-value"):
            self.assertNotIn(secret, serialized)

    def test_redact_argv_hashes_credential_fragments_and_keeps_benign_fragments(
        self,
    ):
        for credential_name in (
            "access_token",
            "access-token",
            "password",
            "key",
            "api_key",
        ):
            with self.subTest(credential_name=credential_name):
                checkpoint = (
                    f"https://models.example.test/model#{credential_name}=fragment-secret"
                )
                redacted = _redact_argv(["--checkpoint_path", checkpoint])
                self.assertRegex(redacted[1], r"^sha256:[0-9a-f]{64}$")
                self.assertNotIn("fragment-secret", json.dumps(redacted))

        benign_checkpoint = "https://models.example.test/model#revision-v2"
        self.assertEqual(
            _redact_argv(["--checkpoint_path", benign_checkpoint]),
            ["--checkpoint_path", benign_checkpoint],
        )

    def test_write_test_info_records_redacted_schema(self):
        with tempfile.TemporaryDirectory() as tmp:
            args, _ = parse_args(
                [
                    f"--result_dir={tmp}",
                    "--warmup_runs=2",
                    "--measure_runs=3",
                    "--profile_runs=0",
                ]
            )
            remaining = [
                "--model_type",
                "example_model",
                "--checkpoint_path",
                "/models/example",
                "--api_key",
                "secret-value",
            ]
            argv = [
                "batch_decode_test.py",
                "--engine_env=OSS_ACCESS_KEY_ID=secret-value",
            ]
            with patch.dict(os.environ, {}, clear=True), patch.object(
                sys, "argv", argv
            ):
                _apply_run_overrides(args)
                write_test_info(
                    args,
                    remaining,
                    ["OSS_ACCESS_KEY_ID"],
                    status="running",
                    effective_max_seq_len=65566,
                    service_concurrency_limit=1,
                    effective_runtime_config={
                        "values": {"TP_SIZE": "8"},
                        "sources": {"TP_SIZE": "cli"},
                    },
                )
            info = json.loads((Path(tmp) / "test_info.json").read_text())

        self.assertEqual(info["schema_version"], 5)
        self.assertEqual(info["status"], "running")
        self.assertEqual(info["model_type"], "example_model")
        self.assertEqual(info["requested_max_seq_len"], args.max_seq_len)
        self.assertEqual(info["effective_max_seq_len"], 65566)
        self.assertEqual(info["max_seq_len"], 65566)
        self.assertEqual(info["concurrency_limit"], args.concurrency_limit)
        self.assertEqual(info["requested_concurrency_limit"], args.concurrency_limit)
        self.assertEqual(info["service_concurrency_limit"], 1)
        self.assertEqual(info["warmup_runs"], 2)
        self.assertEqual(info["measure_runs"], 3)
        self.assertEqual(info["profile_runs"], 0)
        self.assertEqual(info["engine_env_names"], ["OSS_ACCESS_KEY_ID"])
        self.assertEqual(info["effective_runtime_config"]["values"]["TP_SIZE"], "8")
        self.assertEqual(info["engine_args"][-1], "***")
        self.assertEqual(info["argv"][-1], "--engine_env=***")
        self.assertNotIn("secret-value", json.dumps(info))

    def test_write_test_info_sanitizes_credentials_under_benign_field_names(self):
        with tempfile.TemporaryDirectory() as tmp:
            args, _ = parse_args(
                [
                    f"--result_dir={tmp}",
                    "--dataset_path=https://host/data?token=dataset-secret",
                ]
            )
            remaining = [
                "--model_type",
                "https://user:model-secret@example.test/model",
                "--checkpoint_path",
                "https://host/model#access_token=checkpoint-secret",
            ]
            with patch.dict(os.environ, {}, clear=True), patch.object(
                sys, "argv", ["batch_decode_test.py"]
            ):
                write_test_info(
                    args,
                    remaining,
                    effective_runtime_config={
                        "values": {
                            "MODEL_TYPE": "https://host/model?signature=runtime-secret",
                            "CUSTOM_ENDPOINT": "plain-but-unknown",
                        },
                        "sources": {
                            "MODEL_TYPE": "cli",
                            "CUSTOM_ENDPOINT": "engine_env",
                        },
                    },
                )
            info = json.loads((Path(tmp) / "test_info.json").read_text())

        serialized = json.dumps(info)
        self.assertTrue(info["model_type"].startswith("sha256:"))
        self.assertTrue(info["checkpoint_path"].startswith("sha256:"))
        self.assertTrue(info["dataset_path"].startswith("sha256:"))
        self.assertTrue(
            info["effective_runtime_config"]["values"]["MODEL_TYPE"].startswith(
                "sha256:"
            )
        )
        self.assertTrue(
            info["effective_runtime_config"]["values"]["CUSTOM_ENDPOINT"].startswith(
                "sha256:"
            )
        )
        for secret in (
            "dataset-secret",
            "model-secret",
            "checkpoint-secret",
            "runtime-secret",
            "plain-but-unknown",
        ):
            self.assertNotIn(secret, serialized)

    def test_cache_grid_main_marks_partial_and_total_failures(self):
        for result_statuses in (("ok", "failed"), ("error", "failed")):
            with self.subTest(
                result_statuses=result_statuses
            ), tempfile.TemporaryDirectory() as tmp:
                grid_path = Path(tmp) / "grid.json"
                grid_path.write_text(
                    json.dumps(
                        {
                            "cases": [
                                {
                                    "case_id": index,
                                    "batch_size": 1,
                                    "input_len": 8 + index,
                                    "cache_len": 0,
                                }
                                for index in range(len(result_statuses))
                            ]
                        }
                    ),
                    encoding="utf-8",
                )
                args, remaining = parse_args(
                    [
                        f"--result_dir={tmp}",
                        f"--cache_grid_json={grid_path}",
                        "--partial=2",
                        "--checkpoint_path=/models/example",
                    ]
                )
                server = MagicMock(port=12345)
                runner = MagicMock()
                runner.run.return_value = [
                    {"status": status} for status in result_statuses
                ]
                statuses = []
                transformers_module = MagicMock()
                transformers_module.AutoTokenizer.from_pretrained.return_value = (
                    _WhitespaceTokenizer()
                )

                with patch.dict(
                    sys.modules, {"transformers": transformers_module}
                ), patch(
                    "rtp_llm.config.log_config.setup_logging"
                ), patch(
                    "rtp_llm.test.perf_test.batch_decode_test.parse_args",
                    return_value=(args, remaining),
                ), patch(
                    "rtp_llm.test.perf_test.batch_decode_test."
                    "resolve_perf_engine_paths",
                    side_effect=lambda values: values,
                ), patch(
                    "rtp_llm.test.perf_test.batch_decode_test.EngineServer"
                ) as engine_server, patch(
                    "rtp_llm.test.perf_test.batch_decode_test.CacheGridRunner",
                    return_value=runner,
                ), patch(
                    "rtp_llm.test.perf_test.batch_decode_test.write_test_info",
                    side_effect=lambda *unused, **kwargs: statuses.append(
                        kwargs["status"]
                    ),
                ), patch(
                    "rtp_llm.test.perf_test.batch_decode_test.collect_timeline_files"
                ), patch(
                    "rtp_llm.test.perf_test.batch_decode_test."
                    "summarize_and_cleanup_coredumps"
                ):
                    engine_server.return_value = server
                    with self.assertRaisesRegex(RuntimeError, "cache grid failed"):
                        main()

                self.assertEqual(statuses, ["running", "failed"])
                server.stop.assert_called_once()

    def test_main_writes_running_before_start_and_completed_after_success(self):
        with tempfile.TemporaryDirectory() as tmp:
            args, remaining = parse_args(
                [
                    f"--result_dir={tmp}",
                    "--batch_size=1",
                    "--input_len=8",
                    "--partial=2",
                    "--model_type=example_model",
                ]
            )
            config = MagicMock(
                is_distribution=False,
                input_len_list=[8],
                all_seq_lens=[8],
                max_seq_len=18,
                max_concurrency=1,
            )
            events = []
            server = MagicMock(port=12345)
            server.start.side_effect = lambda **kwargs: events.append(
                ("start", kwargs["max_seq_len"])
            )

            def record_status(*unused_args, **kwargs):
                events.append(
                    (kwargs["status"], kwargs.get("effective_max_seq_len"))
                )

            with patch(
                "rtp_llm.config.log_config.setup_logging"
            ), patch(
                "rtp_llm.test.perf_test.batch_decode_test.parse_args",
                return_value=(args, remaining),
            ), patch(
                "rtp_llm.test.perf_test.batch_decode_test.resolve_perf_engine_paths",
                side_effect=lambda values: values,
            ), patch(
                "rtp_llm.test.perf_test.batch_decode_test.prepare_config",
                return_value=config,
            ), patch(
                "rtp_llm.test.perf_test.batch_decode_test.EngineServer",
                return_value=server,
            ), patch(
                "rtp_llm.test.perf_test.batch_decode_test.write_test_info",
                side_effect=record_status,
            ), patch(
                "rtp_llm.test.perf_test.batch_decode_test.query_engine_status",
                return_value={},
            ), patch(
                "rtp_llm.test.perf_test.batch_decode_test.print_config_table"
            ), patch(
                "rtp_llm.test.perf_test.batch_decode_test.create_query",
                return_value={8: "query"},
            ), patch(
                "rtp_llm.test.perf_test.batch_decode_test._run_prefill"
            ), patch(
                "rtp_llm.test.perf_test.batch_decode_test.collect_timeline_files"
            ), patch(
                "rtp_llm.test.perf_test.batch_decode_test.summarize_and_cleanup_coredumps"
            ):
                main()

        self.assertEqual(
            [event[0] for event in events], ["running", "start", "completed"]
        )
        self.assertEqual(events[0][1], events[1][1])
        self.assertEqual(events[1][1], events[2][1])

    def test_generated_formula_sums_supported_per_item_features(self):
        formula = formula_text([1.0] * len(FEATURE_NAMES))
        self.assertEqual(
            formula,
            "1 * sum(1) + 1 * sum(inputTokens / 1024.0)"
            " + 1 * sum(hitCacheTokens / 1024.0)"
            " + 1 * sum((inputTokens / 1024.0) * (inputTokens / 1024.0))"
            " + 1 * sum((inputTokens / 1024.0) * (hitCacheTokens / 1024.0))"
            " + 1 * sum((hitCacheTokens / 1024.0) * (hitCacheTokens / 1024.0))",
        )
        self.assertNotIn("totalInputTokens", formula)
        self.assertNotIn("totalHitCacheTokens", formula)
        self.assertEqual(formula.count("sum("), len(FEATURE_NAMES))

    def test_dsv4_formula_flow_rejects_non_one_batch_size(self):
        with self.assertRaisesRegex(ValueError, "requires --batch-size=1"):
            load_observations([], batch_size=2)
        with self.assertRaises(SystemExit):
            build_formula_parser().parse_args(
                [
                    "fit",
                    "--inputs",
                    "measurements.json",
                    "--output-dir",
                    "fit",
                    "--batch-size",
                    "2",
                ]
            )

    def test_blank_requested_cache_len_falls_back_to_cache_len(self):
        metric = self._valid_audited_metric()
        metric.update({"cache_len_requested": " ", "cache_len": 8})
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "measurements.json"
            path.write_text(
                json.dumps(self._audited_payload([metric])), encoding="utf-8"
            )
            observations, audit = load_observations([path])
            chart_rows = load_rows(path, batch_size=1)

        self.assertTrue(audit["production_audit_passed"])
        self.assertEqual(audit["valid_observation_count"], 1)
        self.assertEqual(observations[0].cache_len, 8)
        self.assertEqual(chart_rows[0]["cache"], 8)

    def test_missing_requested_cache_len_is_rejected_cleanly(self):
        metric = self._valid_audited_metric(cache_len=0)
        metric["cache_len_requested"] = ""
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "measurements.json"
            path.write_text(
                json.dumps(self._audited_payload([metric])), encoding="utf-8"
            )
            observations, audit = load_observations([path])
            with self.assertRaisesRegex(ValueError, r"metrics\[0\].*geometry"):
                load_rows(path, batch_size=1)

        self.assertEqual(observations, [])
        self.assertEqual(audit["rejected_counts"], {"invalid_geometry": 1})

    def test_json_audit_and_chart_fail_closed_on_missing_run_evidence(self):
        cases = []

        missing_status = self._valid_audited_metric(cache_len=0)
        missing_status.pop("status")
        cases.append(("missing_status", self._audited_payload([missing_status]), 0))

        missing_batch = self._valid_audited_metric(cache_len=0)
        missing_batch.pop("batch_size")
        cases.append(("missing_batch", self._audited_payload([missing_batch]), 0))

        incomplete_runs = self._valid_audited_metric(cache_len=0)
        incomplete_runs["success_runs"] = 0
        cases.append(("incomplete_runs", self._audited_payload([incomplete_runs]), 0))

        incomplete_file = self._audited_payload(
            [self._valid_audited_metric(cache_len=0)], complete=False
        )
        cases.append(("incomplete_file", incomplete_file, 1))

        missing_fingerprint = self._audited_payload(
            [self._valid_audited_metric(cache_len=0)]
        )
        missing_fingerprint.pop("run_fingerprint")
        cases.append(("missing_fingerprint", missing_fingerprint, 1))

        with tempfile.TemporaryDirectory() as tmp:
            for name, payload, analysis_rows in cases:
                with self.subTest(name=name):
                    path = Path(tmp) / f"{name}.json"
                    path.write_text(json.dumps(payload), encoding="utf-8")
                    observations, audit = load_observations([path])
                    self.assertEqual(len(observations), analysis_rows)
                    self.assertFalse(audit["production_audit_passed"])
                    with self.assertRaisesRegex(ValueError, str(path)):
                        load_rows(path, batch_size=1)

    def test_json_audit_recomputes_fingerprint_config_sha256(self):
        payload = self._audited_payload([self._valid_audited_metric(cache_len=0)])
        payload["fingerprint_config"]["tampered"] = True
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "tampered.json"
            path.write_text(json.dumps(payload), encoding="utf-8")
            observations, audit = load_observations([path])

            self.assertEqual(len(observations), 1)
            self.assertFalse(audit["production_audit_passed"])
            checks = audit["input_files"][0]["provenance_checks"]
            self.assertFalse(checks["fingerprint_sha256"])
            with self.assertRaisesRegex(ValueError, "fingerprint_sha256"):
                load_rows(path, batch_size=1)

    def test_multi_input_fit_accepts_shards_with_only_case_list_differences(self):
        common_runtime = {
            "checkpoint_schema_version": 2,
            "implementation_sha256": "same-code",
            "tokenizer": {"class": "Tokenizer", "name_or_path": "/model"},
            "measure_runs": 1,
            "run_config": {"effective_runtime_config": {"TP_SIZE": "8"}},
        }
        first_config = {
            **common_runtime,
            "cases": [
                {"case_id": 1, "batch_size": 1, "input_len": 16, "cache_len": 0}
            ],
        }
        second_config = {
            **common_runtime,
            "cases": [
                {"case_id": 2, "batch_size": 1, "input_len": 24, "cache_len": 0}
            ],
        }
        with tempfile.TemporaryDirectory() as tmp:
            first = Path(tmp) / "shard-1.json"
            second = Path(tmp) / "shard-2.json"
            first.write_text(
                json.dumps(
                    self._audited_payload(
                        [self._valid_audited_metric(input_len=16, cache_len=0)],
                        fingerprint_config=first_config,
                    )
                ),
                encoding="utf-8",
            )
            second.write_text(
                json.dumps(
                    self._audited_payload(
                        [self._valid_audited_metric(input_len=24, cache_len=0)],
                        fingerprint_config=second_config,
                    )
                ),
                encoding="utf-8",
            )
            observations, audit = load_observations([first, second])

        self.assertEqual(len(observations), 2)
        self.assertTrue(audit["production_audit_passed"])
        self.assertIsNotNone(audit["normalized_run_signature"])
        self.assertEqual(
            audit["input_files"][0]["normalized_run_signature"],
            audit["input_files"][1]["normalized_run_signature"],
        )

    def test_multi_input_fit_rejects_incompatible_runtime_configs(self):
        base_config = {
            "checkpoint_schema_version": 2,
            "implementation_sha256": "same-code",
            "tokenizer": {"class": "Tokenizer", "name_or_path": "/model"},
            "cases": [],
            "measure_runs": 1,
        }
        first_config = {**base_config, "run_config": {"tp_size": 4}}
        second_config = {**base_config, "run_config": {"tp_size": 8}}
        with tempfile.TemporaryDirectory() as tmp:
            first = Path(tmp) / "tp4.json"
            second = Path(tmp) / "tp8.json"
            first.write_text(
                json.dumps(
                    self._audited_payload(
                        [self._valid_audited_metric(input_len=16, cache_len=0)],
                        fingerprint_config=first_config,
                    )
                ),
                encoding="utf-8",
            )
            second.write_text(
                json.dumps(
                    self._audited_payload(
                        [self._valid_audited_metric(input_len=24, cache_len=0)],
                        fingerprint_config=second_config,
                    )
                ),
                encoding="utf-8",
            )
            with self.assertRaisesRegex(
                ValueError, "incompatible runtime configuration"
            ):
                load_observations([first, second])

    def test_chart_rejects_entire_file_when_one_metric_is_invalid(self):
        valid = self._valid_audited_metric(input_len=16, cache_len=0)
        invalid = self._valid_audited_metric(input_len=24, cache_len=0)
        invalid["runs"][0]["ttft_ms"] = 13.0
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "mixed.json"
            path.write_text(
                json.dumps(self._audited_payload([valid, invalid])), encoding="utf-8"
            )
            with self.assertRaisesRegex(ValueError, r"metrics\[1\].*invalid_latency"):
                load_rows(path, batch_size=1)

    def test_chart_rejects_unaudited_csv_timings(self):
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "predictions.csv"
            path.write_text(
                "batch_size,input_len,cache_len,target_ms\n1,4096,0,12.5\n",
                encoding="utf-8",
            )
            with self.assertRaisesRegex(ValueError, "runner JSON"):
                load_rows(path, batch_size=1)

    def test_csv_fit_is_usable_but_never_production_accepted(self):
        with tempfile.TemporaryDirectory() as tmp:
            csv_path = Path(tmp) / "observations.csv"
            with csv_path.open("w", newline="", encoding="utf-8") as stream:
                writer = csv.DictWriter(
                    stream,
                    fieldnames=(
                        "batch_size",
                        "input_len",
                        "cache_len",
                        "avg_ttft_ms",
                    ),
                )
                writer.writeheader()
                for index in range(40):
                    input_len = 2048 + index * 1024
                    cache_len = (index % 5) * 256
                    writer.writerow(
                        {
                            "batch_size": 1,
                            "input_len": input_len,
                            "cache_len": cache_len,
                            "avg_ttft_ms": 50.0
                            + input_len / 1024.0
                            - cache_len / 2048.0,
                        }
                    )
            output_dir = Path(tmp) / "fit"
            result = run_fit(
                argparse.Namespace(
                    inputs=[str(csv_path)],
                    output_dir=str(output_dir),
                    batch_size=1,
                    min_valid_rows=6,
                    max_mape_pct=1_000_000.0,
                    max_p95_ape_pct=1_000_000.0,
                    max_max_ape_pct=1_000_000.0,
                    objective="mae",
                    allow_insufficient_data=False,
                )
            )
            report = json.loads((output_dir / "fit_report.json").read_text())

        self.assertEqual(result, 3)
        self.assertGreater(report["audit"]["unaudited_observation_count"], 0)
        self.assertFalse(report["audit"]["production_audit_passed"])
        self.assertFalse(report["production_acceptance"])
        self.assertIsNotNone(report["formula"])

    def test_cold_chart_handles_long_sequences_without_inset_rows(self):
        rows = [
            {"input": 200_000.0, "cache": 0.0, "compute": 200_000.0, "rt": 25.0},
            {"input": 300_000.0, "cache": 0.0, "compute": 300_000.0, "rt": 40.0},
        ]
        svg = render_cold_miss_2d(rows, Path("long-only.json"), batch_size=1)

        self.assertIn("<svg", svg)
        self.assertIn("短序列区间无样本，未绘制插图", svg)
        self.assertNotIn("放大：0–", svg)
        self.assertIn("300K cold：40.0 ms", svg)
        self.assertIn("最长冷点（300K）：40.0 ms", svg)
        self.assertNotIn("1M cold", svg)
        x_tick_positions = re.findall(
            r'<text x="([0-9.]+)" y="854.0" text-anchor="middle" class="tick">',
            svg,
        )
        self.assertEqual(len(x_tick_positions), 7)
        self.assertEqual(len(set(x_tick_positions)), 7)


if __name__ == "__main__":
    unittest.main()
