import argparse
import json
import os
import struct
import subprocess
import tempfile
import threading
import unittest
from pathlib import Path
from unittest import mock
from unittest.mock import Mock, patch

from rtp_llm.test.perf_test.batch_decode_test import (
    _apply_engine_env,
    _capture_reproduction_env,
    _configure_cache_batch_limits,
    _dedupe_cache_grid_cases,
    _effective_grid_max_seq_len,
    _effective_performance_config,
    _engine_arg_argv,
    _finalize_completed_cache_checkpoint,
    _fingerprint_engine_env,
    _load_cache_grid_cases,
    _load_materialized_case_store,
    _parse_name_value,
    _prepare_cache_profile_result_dir,
    _redact_argv,
    _require_cache_grid_success,
    _resolve_cache_block_size,
    _write_test_info,
    parse_args,
)
from rtp_llm.test.perf_test.cache_grid.runner.cache_grid_runner import (
    CacheGridRunner,
    MaterializedCaseStore,
    PrefixPromptFactory,
    _post_prefill,
    _post_prefill_ids,
    build_grouped_prompts,
    normalize_cache_case,
    profile_trace_name,
    resume_config_fingerprint,
    validate_cache_grid_resume,
)
from rtp_llm.test.perf_test.perf_config import _apply_run_overrides
from rtp_llm.test.perf_test.perf_utils import (
    write_test_info as write_standard_test_info,
)
from rtp_llm.test.perf_test.server import EngineServer


class CacheGridBatchTest(unittest.TestCase):
    def setUp(self):
        scheduler = patch.object(
            CacheGridRunner, "_set_prefill_batch_size", return_value={"status": "ok"}
        )
        self.scheduler = scheduler.start()
        self.addCleanup(scheduler.stop)

    def case(self, policy="independent"):
        return {
            "case_id": 10,
            "batch_size": 4,
            "prefix_policy": policy,
            "request_groups": [
                {"count": 2, "input_len": 24, "cache_len": 8},
                {"count": 1, "input_len": 32, "cache_len": 16},
                {"count": 1, "input_len": 24, "cache_len": 0},
            ],
        }

    def test_cache_grid_command_rejects_any_non_ok_metric(self):
        _require_cache_grid_success([{"status": "ok"}])
        with self.assertRaisesRegex(RuntimeError, "timeout=1"):
            _require_cache_grid_success([{"status": "ok"}, {"status": "timeout"}])

    def test_profile_trace_name_is_engine_safe_and_stable(self):
        name = profile_trace_name("f" * 32, "bs1_seq1048575_cache1044480", 123)
        self.assertEqual(
            name, profile_trace_name("f" * 32, "bs1_seq1048575_cache1044480", 123)
        )
        self.assertLessEqual(len(name.encode("utf-8")), 64)

    def test_fixed_batch_service_limits_and_dp_guard(self):
        args = argparse.Namespace(dp_size=1, concurrency_limit=4)
        remaining = ["--max_context_batch_size=1", "--max_batch_tokens_size", "32"]
        cases = [normalize_cache_case(self.case())]
        _configure_cache_batch_limits(args, remaining, cases)
        self.assertEqual(args.concurrency_limit, 4)
        self.assertEqual(
            remaining,
            ["--max_context_batch_size=1", "--max_batch_tokens_size", "104"],
        )
        args.dp_size = 2
        with self.assertRaisesRegex(ValueError, "dp_size=1"):
            _configure_cache_batch_limits(args, remaining, cases)

    def test_fixed_batch_rejects_insufficient_service_concurrency(self):
        args = argparse.Namespace(dp_size=1, concurrency_limit=3)
        with self.assertRaisesRegex(ValueError, "at least 4"):
            _configure_cache_batch_limits(args, [], [normalize_cache_case(self.case())])

    def test_finalize_completed_checkpoint_marks_result_and_progress_complete(self):
        case = {"case_id": 0, "batch_size": 1, "input_len": 16, "cache_len": 0}
        checkpoint = {
            "complete": False,
            "status": "in_progress",
            "resume_count": 2,
            "run_config_sha256": "abc",
            "metrics": [{"case_key": CacheGridRunner.case_key(case), "status": "ok"}],
        }
        with tempfile.TemporaryDirectory() as tmp:
            _finalize_completed_cache_checkpoint(tmp, checkpoint, [case])
            result = json.loads((Path(tmp) / "cache_grid_results.json").read_text())
            progress = json.loads((Path(tmp) / "cache_grid_progress.json").read_text())
        self.assertTrue(result["complete"])
        self.assertEqual(result["status"], "completed")
        self.assertEqual(result["progress"]["pending_cases"], 0)
        self.assertEqual(progress["progress"]["progress_pct"], 100.0)

    def test_cold_batch_skips_seed_and_restores_scheduler(self):
        with tempfile.TemporaryDirectory() as tmp:
            runner = CacheGridRunner(
                12345,
                _WordTokenizer(),
                [
                    {
                        "case_id": 1,
                        "batch_size": 2,
                        "prefix_policy": "independent",
                        "request_groups": [
                            {"count": 2, "input_len": 24, "cache_len": 0}
                        ],
                    }
                ],
                tmp,
                measure_runs=1,
            )
            with patch(
                "rtp_llm.test.perf_test.cache_grid.runner.cache_grid_runner._post_prefill",
                return_value={
                    "prefill_time_ms": 1,
                    "success": True,
                    "input_len": 24,
                    "reuse_len": 0,
                    "output_len": 1,
                    "ttft_ms": 1,
                },
            ) as post:
                row = runner.run()[0]
            self.assertEqual(row["status"], "ok")
            self.assertEqual(row["seed_batch_size"], 0)
            self.assertEqual(post.call_count, 2)
            self.assertEqual([c.args[0] for c in self.scheduler.call_args_list], [2, 1])

    def test_scheduler_rejection_aborts_before_seeding(self):
        self.scheduler.side_effect = [
            RuntimeError("scheduler unavailable"),
            {"status": "ok"},
        ]
        with tempfile.TemporaryDirectory() as tmp:
            runner = CacheGridRunner(
                12345,
                _WordTokenizer(),
                [self.case()],
                tmp,
                cache_commit_tail_tokens=8,
                fail_fast=False,
            )
            with patch(
                "rtp_llm.test.perf_test.cache_grid.runner.cache_grid_runner._post_prefill"
            ) as post:
                with self.assertRaisesRegex(RuntimeError, "stopped"):
                    runner.run()
            post.assert_not_called()

    def test_config_validation_requires_explicit_batch_groups(self):
        with self.assertRaisesRegex(ValueError, "request_groups"):
            normalize_cache_case(
                {"case_id": 0, "batch_size": 3, "input_len": 24, "cache_len": 8}
            )
        for change in (
            {"batch_size": 5},
            {"prefix_policy": "random"},
            {"request_groups": []},
            {"request_groups": [{"count": 4, "input_len": 8, "cache_len": 8}]},
        ):
            with self.subTest(change=change), self.assertRaises(ValueError):
                normalize_cache_case({**self.case(), **change})

    def test_prefix_sharing_and_distinct_round_suffixes(self):
        for policy in ("independent", "shared_by_group"):
            members = build_grouped_prompts(
                PrefixPromptFactory(_WordTokenizer()), self.case(policy), 3
            )
            first, second = members[:2]
            self.assertEqual(
                first["prompts"].prefix_ids == second["prompts"].prefix_ids,
                policy == "shared_by_group",
            )
            texts = [text for member in members for text in member["prompts"].run_texts]
            self.assertEqual(len(texts), len(set(texts)))
            for member in members:
                for ids in member["prompts"].run_ids:
                    self.assertEqual(len(ids), member["input_len"])
                    self.assertEqual(
                        ids[: member["cache_len"]], member["prompts"].prefix_ids
                    )

    def test_loader_dedupe_and_distribution_identity(self):
        first = self.case()
        second = {**self.case("shared_by_group"), "case_id": 11}
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "grid.json"
            path.write_text(
                json.dumps(
                    {
                        "schema_version": 2,
                        "generator": {"cache_alignment": 8},
                        "cases": [first, second],
                    }
                )
            )
            cases = _load_cache_grid_cases(str(path))
        self.assertEqual(len(_dedupe_cache_grid_cases(cases, 8)), 2)
        self.assertNotEqual(
            CacheGridRunner.case_key(first), CacheGridRunner.case_key(second)
        )
        self.assertEqual(cases[0]["input_len"], 32)

    def test_materialized_roundtrip_and_distribution_guard(self):
        with tempfile.TemporaryDirectory() as tmp:
            store = MaterializedCaseStore(tmp)
            case = normalize_cache_case(self.case("shared_by_group"))
            store.materialize([case], PrefixPromptFactory(_WordTokenizer()), 2)
            loaded = MaterializedCaseStore(tmp)
            self.assertEqual(loaded.load_cases(), [case])
            expected = build_grouped_prompts(
                PrefixPromptFactory(_WordTokenizer()), case, 2
            )
            self.assertEqual(loaded.load_grouped_case(case), expected)
            with self.assertRaisesRegex(ValueError, "distribution"):
                loaded.load_grouped_case(self.case())

    def run_mock_batch(
        self,
        policy,
        bad_slot=None,
        transport="http_prompt",
        store=False,
        skip_reuse_validation=False,
        bad_timing=False,
    ):
        gate = threading.Barrier(4)
        seed_gate = threading.Barrier(2 if policy == "shared_by_group" else 3)
        seeds, sessions, calls = [], {}, []
        lock = threading.Lock()

        def post(port, prompt, timeout, request_id, session=None):
            if ":seed:" in request_id:
                seeds.append(prompt.split())
                seed_gate.wait(timeout=3)
                return {"success": True}
            ids = prompt.split()
            slot = int(request_id.rsplit("request", 1)[1])
            with lock:
                sessions[slot] = session
                calls.append(request_id)
            # A serial implementation cannot pass this gate.
            gate.wait(timeout=3)
            reuse = 0
            for seed in seeds:
                common = 0
                for a, b in zip(seed, ids):
                    if a != b:
                        break
                    common += 1
                reuse = max(reuse, common // 8 * 8)
            if slot == bad_slot:
                reuse += 8
            return {
                "prefill_time_ms": 0.0 if bad_timing else 2.0,
                "success": True,
                "input_len": len(ids),
                "reuse_len": reuse,
                "output_len": 1,
                "ttft_ms": 999.0,
            }

        with tempfile.TemporaryDirectory() as tmp:
            case = self.case(policy)
            case_store = None
            if store:
                case_store = MaterializedCaseStore(str(Path(tmp) / "prompts"))
                case_store.materialize([case], PrefixPromptFactory(_WordTokenizer()), 2)
            runner = CacheGridRunner(
                12345,
                _WordTokenizer(),
                [case],
                str(Path(tmp) / "results"),
                measure_runs=2,
                cache_commit_tail_tokens=8,
                fail_fast=False,
                skip_reuse_validation=skip_reuse_validation,
                case_store=case_store,
            )
            if transport == "dashsc_input_ids":
                runner.request_transport = transport
                runner._grpc_stub = Mock()
            with patch(
                "rtp_llm.test.perf_test.cache_grid.runner.cache_grid_runner._post_prefill",
                side_effect=post,
            ), patch(
                "rtp_llm.test.perf_test.cache_grid.runner.cache_grid_runner._post_prefill_ids",
                side_effect=lambda stub, ids, timeout, request_id: post(
                    0, " ".join(ids), timeout, request_id
                ),
            ):
                rows = runner.run()
            persisted = json.loads(runner.result_path.read_text())["metrics"][0]
            self.assertEqual(persisted["status"], rows[0]["status"])
        self.assertEqual(len(calls), 8)
        self.assertEqual(
            [c.args[0] for c in self.scheduler.call_args_list[-3:]],
            [2 if policy == "shared_by_group" else 3, 4, 1],
        )
        self.assertEqual(len(seeds), 2 if policy == "shared_by_group" else 3)
        if transport == "http_prompt":
            self.assertEqual(len({id(s) for s in sessions.values()}), 4)
        return rows[0]

    def test_concurrent_requests_and_per_member_cache_hits(self):
        for policy in ("independent", "shared_by_group"):
            with self.subTest(policy=policy):
                row = self.run_mock_batch(policy)
                self.assertEqual(row["status"], "ok")
                self.assertEqual(row["execution_mode"], "scheduler_fixed_batch")
                self.assertGreater(row["median_batch_wall_time_ms"], 0)
                self.assertNotIn("median_prefill_time_ms", row)
                for run in row["runs"]:
                    self.assertEqual(run["total_input_tokens"], 104)
                    self.assertEqual(run["new_prefill_tokens"], 72)
                    self.assertEqual(
                        [r["reuse_len"] for r in run["requests"]], [8, 8, 16, 0]
                    )

    def test_wrong_member_reuse_invalidates_batch(self):
        row = self.run_mock_batch("independent", bad_slot=1)
        self.assertEqual(row["status"], "invalid_reuse")
        self.assertIsNone(row["median_batch_wall_time_ms"])
        self.assertFalse(row["runs"][0]["requests"][1]["reuse_exact"])

    def test_reuse_validation_can_be_skipped_without_losing_observed_values(self):
        row = self.run_mock_batch("independent", bad_slot=1, skip_reuse_validation=True)
        self.assertEqual(row["status"], "ok")
        self.assertEqual(row["validation_status"], "invalid_reuse")
        self.assertTrue(row["reuse_validation_skipped"])
        self.assertFalse(row["reuse_exact"])
        self.assertEqual(row["success_runs"], 2)
        self.assertEqual(row["cache_len_observed"], [8, 16, 16, 0] * 2)
        self.assertGreater(row["median_batch_wall_time_ms"], 0)

    def test_skipping_reuse_does_not_hide_invalid_timing(self):
        for skip in (False, True):
            with self.subTest(skip=skip):
                row = self.run_mock_batch(
                    "independent",
                    bad_slot=1,
                    bad_timing=True,
                    skip_reuse_validation=skip,
                )
                self.assertEqual(row["status"], "invalid_timing")
                self.assertEqual(row["validation_status"], "invalid_timing")
                self.assertIsNone(row["median_batch_wall_time_ms"])

    def test_materialized_grpc_batch(self):
        self.assertEqual(
            self.run_mock_batch(
                "shared_by_group", transport="dashsc_input_ids", store=True
            )["status"],
            "ok",
        )

    def test_failed_request_is_preserved_and_other_members_complete(self):
        with tempfile.TemporaryDirectory() as tmp:
            runner = CacheGridRunner(
                12345,
                _WordTokenizer(),
                [self.case()],
                tmp,
                measure_runs=1,
                cache_commit_tail_tokens=8,
                fail_fast=False,
            )

            def post(port, prompt, timeout, request_id, session=None):
                if ":seed:" in request_id:
                    return {"success": True}
                slot = int(request_id.rsplit("request", 1)[1])
                if slot == 1:
                    raise TimeoutError("simulated request timeout")
                return {
                    "prefill_time_ms": 1,
                    "success": True,
                    "input_len": len(prompt.split()),
                    "reuse_len": [8, 8, 16, 0][slot],
                    "output_len": 1,
                    "ttft_ms": 1,
                }

            with patch(
                "rtp_llm.test.perf_test.cache_grid.runner.cache_grid_runner._post_prefill",
                side_effect=post,
            ):
                with self.assertRaisesRegex(RuntimeError, "stopped"):
                    runner.run()
                row = next(iter(runner._results.values()))
        self.assertEqual(row["status"], "failed")
        self.assertEqual(row["runs"][0]["completed_requests"], 3)
        self.assertIn("timeout", row["runs"][0]["requests"][1]["error"])

    def test_resume_does_not_reuse_a_different_distribution(self):
        with tempfile.TemporaryDirectory() as tmp:
            case = self.case()
            runner = CacheGridRunner(12345, _WordTokenizer(), [case], tmp)
            runner._results[runner.case_key(case)] = {
                **normalize_cache_case(case),
                "case_key": runner.case_key(case),
                "measure_runs": 3,
                "success_runs": 3,
                "runs": [],
                "status": "ok",
            }
            runner._save(complete=True)
            runner._close_resources()
            same = CacheGridRunner(12345, _WordTokenizer(), [case], tmp)
            self.assertEqual(len(same._results), 1)
            same._close_resources()
            changed = CacheGridRunner(
                12345, _WordTokenizer(), [self.case("shared_by_group")], tmp
            )
            self.assertEqual(changed._results, {})
            changed._close_resources()


class _WhitespaceTokenizer:
    def encode(self, text):
        return text.split()


class _WordTokenizer:
    """Minimal stand-in: every whitespace-separated word is one token."""

    def encode(self, text):
        return text.split()


class _TailMergingTokenizer:
    """Whitespace tokenizer that fuses a ``__run_`` tail into the previous token."""

    def encode(self, text):
        words = text.split()
        out = []
        for word in words:
            if word.startswith("__run_") and out and out[-1] == "hello":
                out[-1] = "hello" + word
            else:
                out.append(word)
        return out


class BatchDecodeTest(unittest.TestCase):
    def test_effective_grid_max_seq_len_uses_decode_need(self):
        args = argparse.Namespace(max_seq_len=8192, decode_test_length=30)
        self.assertEqual(_effective_grid_max_seq_len(args, [1024, 65536]), 65566)

    def test_effective_grid_max_seq_len_respects_explicit_headroom(self):
        args = argparse.Namespace(max_seq_len=65664, decode_test_length=30)
        self.assertEqual(_effective_grid_max_seq_len(args, [65536]), 65664)

    def test_cache_grid_loader_validates_explicit_cases(self):
        payload = {
            "schema_version": 2,
            "generator": {"cache_alignment": 512},
            "cases": [
                {"case_id": 7, "batch_size": 1, "input_len": 4096, "cache_len": 0},
                {
                    "case_id": 8,
                    "batch_size": 1,
                    "input_len": 4096,
                    "cache_len": 2048,
                },
            ],
        }
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "cache_grid.json"
            path.write_text(json.dumps(payload), encoding="utf-8")
            self.assertEqual(_load_cache_grid_cases(str(path)), payload["cases"])

    def test_cache_grid_loader_rejects_duplicate_geometry(self):
        payload = {
            "schema_version": 2,
            "generator": {"cache_alignment": 512},
            "cases": [
                {"case_id": 0, "batch_size": 1, "input_len": 4096, "cache_len": 2048},
                {"case_id": 1, "batch_size": 1, "input_len": 4096, "cache_len": 2048},
            ],
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
        self.assertFalse(hasattr(args, "expected_cache_block_size"))
        self.assertEqual(args.materialize_cache_cases, "")
        self.assertEqual(args.cache_case_files, "")
        self.assertEqual(args.cache_request_transport, "dashsc_input_ids")
        self.assertEqual(args.cache_grpc_port, 0)
        self.assertEqual(args.cache_checkpoint_every, 100)
        self.assertFalse(args.require_cache_resume)
        self.assertIsInstance(remaining, list)

    def test_cache_grid_requires_nonempty_explicit_cases(self):
        base = {"schema_version": 2, "generator": {"cache_alignment": 4096}}
        for fields in ({}, {"cases": []}):
            with self.subTest(fields=fields), tempfile.TemporaryDirectory() as tmp:
                path = Path(tmp) / "grid.json"
                path.write_text(json.dumps({**base, **fields}))
                with self.assertRaises(ValueError):
                    _load_cache_grid_cases(str(path))

    def test_cache_case_required_fields_are_not_filled_or_coerced(self):
        scalar = {"case_id": 0, "batch_size": 1, "input_len": 8192, "cache_len": 0}
        grouped = {
            "case_id": 1,
            "batch_size": 2,
            "prefix_policy": "independent",
            "request_groups": [{"count": 2, "input_len": 8192, "cache_len": 0}],
        }
        for original in (scalar, grouped):
            for key in original:
                with self.subTest(missing=key, original=original), self.assertRaises(
                    ValueError
                ):
                    normalize_cache_case(
                        {k: v for k, v in original.items() if k != key}
                    )
        for key in ("count", "input_len", "cache_len"):
            with self.subTest(group_missing=key), self.assertRaises(ValueError):
                normalize_cache_case(
                    {
                        **grouped,
                        "request_groups": [
                            {
                                k: v
                                for k, v in grouped["request_groups"][0].items()
                                if k != key
                            }
                        ],
                    }
                )
        for key in scalar:
            for value in ("1", True, 1.0, None):
                with self.subTest(key=key, value=value), self.assertRaises(ValueError):
                    normalize_cache_case({**scalar, key: value})
        with self.assertRaisesRegex(ValueError, "summary"):
            normalize_cache_case({**grouped, "input_len": 4096})

    def test_grid_examples_and_generated_batches_pass_strict_loader(self):
        from rtp_llm.test.perf_test.cache_grid.runner.generate_cache_grid import (
            build_grid,
        )

        examples = Path(__file__).parent / "cache_grid" / "examples"
        for path in sorted(examples.glob("*.json")):
            with self.subTest(example=path.name):
                payload = json.loads(path.read_text())
                cases = _load_cache_grid_cases(str(path))
                self.assertEqual(len(cases), len(payload["cases"]))
                if "case_count" in payload.get("summary", {}):
                    self.assertEqual(len(cases), payload["summary"]["case_count"])
        args = argparse.Namespace(
            min_input_len=2048,
            max_input_len=8192,
            alignment=128,
            cache_alignment=512,
            input_points=4,
            input_mode="stratified",
            seed=42,
            cache_points_per_input=6,
            cache_ratio_points=2,
            batch_size=2,
            max_cases=1000,
            allow_large_grid=False,
            measure_runs=3,
        )
        fixed_args = argparse.Namespace(
            grid_mode="fixed-cache-sweep",
            max_input_len=32768,
            cache_alignment=4096,
            fixed_cache_len=[4096, 8192],
            random_cache_count=None,
            min_cache_len=0,
            max_cache_len=None,
            compute_step=4096,
            min_compute_len=None,
            batch_size=2,
            measure_runs=3,
            seed=42,
            max_cases=1000,
            allow_large_grid=False,
        )
        for settings in (args, fixed_args):
            with tempfile.TemporaryDirectory() as tmp:
                payload = build_grid(settings)
                path = Path(tmp) / "grid.json"
                path.write_text(json.dumps(payload))
                cases = _load_cache_grid_cases(str(path))
                self.assertEqual(cases, payload["cases"])
                self.assertTrue(all(c["prefix_policy"] == "independent" for c in cases))

    def test_removed_alignment_cli_is_rejected(self):
        with self.assertRaises(SystemExit):
            parse_args(["--expected_cache_block_size=4096"])

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

    @patch("rtp_llm.test.perf_test.cache_grid.runner.cache_grid_runner.requests.post")
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

    def test_post_prefill_ids_parses_grpc_metrics(self):
        from rtp_llm.dash_sc.proto import predict_v2_pb2

        response = predict_v2_pb2.ModelStreamInferResponse()
        infer = response.infer_response
        for name, value in (("prompt_token_num", 4), ("prompt_cached_token_num", 2)):
            output = infer.outputs.add(name=name, datatype="INT32")
            output.shape.append(1)
            infer.raw_output_contents.append(struct.pack("<i", value))
        output = infer.outputs.add(name="generated_ids", datatype="INT32")
        output.shape.extend([1, 1])
        infer.raw_output_contents.append(struct.pack("<i", 7))
        infer.parameters["engine_cost_time_us"].int64_param = 12500
        infer.parameters["engine_first_token_cost_time_us"].int64_param = 12000
        infer.parameters["engine_wait_time_us"].int64_param = 500

        class Stub:
            def ModelStreamInfer(self, requests, timeout):
                request = next(requests)
                self.assert_timeout = timeout
                self.request = request
                return iter((response,))

        stub = Stub()
        result = _post_prefill_ids(stub, [1, 2, 3, 4], 10, "case:run0")
        self.assertTrue(result["success"])
        self.assertEqual(result["input_len"], 4)
        self.assertEqual(result["reuse_len"], 2)
        self.assertEqual(result["output_len"], 1)
        self.assertEqual(result["prefill_time_ms"], 12.0)
        self.assertEqual(result["total_time_ms"], 12.5)
        self.assertEqual(result["wait_time_ms"], 0.5)
        self.assertEqual(stub.assert_timeout, 10)
        self.assertTrue(stub.request.parameters["force_sp_accept"].bool_param)
        self.assertEqual(
            result["ttft_source"],
            "client_dashsc_grpc_input_ids_wall_max_new_tokens_1",
        )

    @patch("rtp_llm.test.perf_test.cache_grid.runner.cache_grid_runner._post_prefill")
    def test_runner_uses_server_latency_with_exact_shape_and_reuse(self, post):
        post.side_effect = [
            {"success": True},
            {
                "prefill_time_ms": 12.0,
                "success": True,
                "input_len": 16,
                "output_len": 1,
                "reuse_len": 8,
                "ttft_ms": 1200.0,
            },
            {
                "prefill_time_ms": 10.0,
                "success": True,
                "input_len": 16,
                "output_len": 1,
                "reuse_len": 8,
                "ttft_ms": 1000.0,
            },
            {
                "prefill_time_ms": 11.0,
                "success": True,
                "input_len": 16,
                "output_len": 1,
                "reuse_len": 8,
                "ttft_ms": 1100.0,
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
        self.assertEqual(rows[0]["median_prefill_time_ms"], 11.0)

    @patch("rtp_llm.test.perf_test.cache_grid.runner.cache_grid_runner._post_prefill")
    def test_runner_fails_fast_and_checkpoints_invalid_reuse(self, post):
        post.side_effect = [
            {"success": True},
            *[
                {
                    "prefill_time_ms": 10.0,
                    "success": True,
                    "input_len": 16,
                    "output_len": 1,
                    "reuse_len": 0,
                    "ttft_ms": 10.0,
                }
                for _ in range(3)
            ],
        ]
        with tempfile.TemporaryDirectory() as tmp:
            runner = CacheGridRunner(
                12345,
                _WhitespaceTokenizer(),
                [{"case_id": 1, "batch_size": 1, "input_len": 16, "cache_len": 8}],
                tmp,
                cache_commit_tail_tokens=8,
            )
            with self.assertRaisesRegex(RuntimeError, "invalid_reuse"):
                runner.run()
            with (Path(tmp) / "cache_grid_results.json").open() as f:
                result = json.load(f)
        self.assertFalse(result["complete"])
        self.assertEqual(result["completed_cases"], 0)
        self.assertEqual(result["metrics"][0]["status"], "invalid_reuse")

    @patch("rtp_llm.test.perf_test.cache_grid.runner.cache_grid_runner._post_prefill")
    def test_scalar_runner_can_accept_and_record_reuse_mismatch(self, post):
        post.side_effect = [
            {"success": True},
            *[
                {
                    "prefill_time_ms": 10.0,
                    "success": True,
                    "input_len": 16,
                    "output_len": 1,
                    "reuse_len": 0,
                    "ttft_ms": 10.0,
                }
                for _ in range(3)
            ],
        ]
        with tempfile.TemporaryDirectory() as tmp:
            rows = CacheGridRunner(
                12345,
                _WhitespaceTokenizer(),
                [{"case_id": 1, "batch_size": 1, "input_len": 16, "cache_len": 8}],
                tmp,
                cache_commit_tail_tokens=8,
                skip_reuse_validation=True,
            ).run()
            result = json.loads((Path(tmp) / "cache_grid_results.json").read_text())
        self.assertTrue(result["complete"])
        self.assertTrue(result["skip_reuse_validation"])
        self.assertEqual(rows[0]["status"], "ok")
        self.assertEqual(rows[0]["validation_status"], "invalid_reuse")
        self.assertFalse(rows[0]["reuse_exact"])
        self.assertEqual(rows[0]["cache_len_observed"], [0, 0, 0])
        self.assertEqual(rows[0]["input_len_observed"], [16, 16, 16])

    def test_engine_arg_shorthand_is_forwarded(self):
        self.assertEqual(
            _engine_arg_argv(["tp_size=8", "fp8_kv_cache=1"]),
            ["--tp_size", "8", "--fp8_kv_cache", "1"],
        )

    def test_name_value_rejects_missing_separator(self):
        with self.assertRaises(ValueError):
            _parse_name_value("tp_size", "--engine_arg")

    def test_engine_env_layers_override_inherited_then_profile_then_cli(self):
        with patch.dict(os.environ, {"DSV4_CHUNK_TOKENS": "inherited"}):
            _apply_engine_env(["DSV4_CHUNK_TOKENS=profile"])
            _apply_engine_env(["DSV4_CHUNK_TOKENS=cli"])
            self.assertEqual(os.environ["DSV4_CHUNK_TOKENS"], "cli")

    def test_run_overrides_update_only_explicit_values(self):
        args = argparse.Namespace(warmup_runs=2, measure_runs=None, profile_runs=4)
        with patch.dict(os.environ, {"PERF_MEASURE_RUNS": "9"}, clear=False):
            _apply_run_overrides(args)
            self.assertEqual(os.environ["PERF_FORMAL_WARMUP_RUNS"], "2")
            self.assertEqual(os.environ["PERF_MEASURE_RUNS"], "9")
            self.assertEqual(os.environ["PERF_PROFILE_RUNS"], "4")

    def test_engine_env_fingerprint_excludes_secrets(self):
        with patch.dict(
            os.environ,
            {"PERF_MEASURE_RUNS": "3", "PRIVATE_TOKEN": "secret"},
            clear=True,
        ):
            fingerprint = _fingerprint_engine_env(["PRIVATE_TOKEN"])
        self.assertEqual(fingerprint["PERF_MEASURE_RUNS"], "3")
        self.assertNotIn("PRIVATE_TOKEN", fingerprint)

    def test_effective_config_records_cli_precedence(self):
        with patch.dict(os.environ, {"MAX_SEQ_LEN": "1024"}, clear=True):
            config = _effective_performance_config(
                ["--max_seq_len", "2048"], [], {"concurrency_limit": 8}
            )
        self.assertEqual(config["values"]["MAX_SEQ_LEN"], "2048")
        self.assertEqual(config["sources"]["MAX_SEQ_LEN"], "cli")
        self.assertEqual(config["values"]["CONCURRENCY_LIMIT"], "8")

    def test_parse_args_exposes_runtime_overrides(self):
        args, remaining = parse_args(
            [
                "--engine_arg=tp_size=8",
                "--engine_env=FP8_KV_CACHE=1",
                "--measure_runs=3",
                "--model_type=example_model",
            ]
        )
        self.assertEqual(args.engine_arg, ["tp_size=8"])
        self.assertEqual(args.engine_env, ["FP8_KV_CACHE=1"])
        self.assertEqual(args.measure_runs, 3)
        self.assertIn("--model_type=example_model", remaining)

    def test_engine_arg_shorthand_overrides_profile_without_duplicate(self):
        with tempfile.TemporaryDirectory() as tmp:
            profile = Path(tmp) / "profile.json"
            profile.write_text(
                json.dumps(
                    {
                        "schema_version": 1,
                        "engine": {"tp_size": 8},
                        "cache_grid": {"commit_tail_tokens": 512},
                    }
                )
            )
            args, remaining = parse_args(
                ["--profile", str(profile), "--engine_arg=tp_size=4"]
            )
        self.assertEqual(args.cache_commit_tail_tokens, 512)
        self.assertEqual(remaining.count("--tp_size"), 1)
        self.assertEqual(remaining[remaining.index("--tp_size") + 1], "4")

    def test_full_option_name_is_required_for_perf_arguments(self):
        args, remaining = parse_args(["--cache_commit_tail_token=8"])
        self.assertEqual(args.cache_commit_tail_tokens, 4096)
        self.assertIn("--cache_commit_tail_token=8", remaining)

    def test_dsv4_profile_injects_model_identity_and_paths(self):
        profile = (
            Path(__file__).parent / "cache_grid" / "config" / "dsv4_pro_prefill.json"
        )
        payload = json.loads(profile.read_text())
        engine = payload["engine"]
        cache_grid = payload["cache_grid"]
        _, remaining = parse_args(["--profile", str(profile)])

        self.assertIn("--model_type", remaining)
        self.assertEqual(remaining[remaining.index("--model_type") + 1], "deepseek_v4")
        self.assertIn("--checkpoint_path", remaining)
        self.assertEqual(
            remaining[remaining.index("--checkpoint_path") + 1],
            engine["checkpoint_path"],
        )
        self.assertIn("--tokenizer_path", remaining)
        self.assertEqual(
            remaining[remaining.index("--tokenizer_path") + 1],
            engine["tokenizer_path"],
        )
        visible_block_size = engine["seq_size_per_block"] * engine["tp_size"]
        self.assertEqual(cache_grid["cache_alignment"], visible_block_size)
        self.assertEqual(cache_grid["commit_tail_tokens"], visible_block_size)

    def test_redact_argv_hides_embedded_engine_secret(self):
        self.assertEqual(
            _redact_argv(
                [
                    "--engine_env=OSS_ACCESS_KEY_ID=secret",
                    "--engine_arg=tp_size=8",
                ]
            ),
            ["--engine_env=***", "--engine_arg=tp_size=8"],
        )

    def test_redact_argv_preserves_noncredential_token_parameters(self):
        argv = [
            "--tokenizer_path",
            "/weights/model",
            "--cache_commit_tail_tokens",
            "128",
            "--max_batch_tokens_size",
            "1048576",
        ]
        self.assertEqual(_redact_argv(argv), argv)

    def test_capture_reproduction_env_records_values_and_redacts_secrets(self):
        with patch.dict(
            "os.environ",
            {"DSV4_CHUNK_TOKENS": "8192", "PRIVATE_TOKEN": "secret-value"},
            clear=False,
        ):
            captured = _capture_reproduction_env(["PRIVATE_TOKEN"])
        self.assertEqual(captured["DSV4_CHUNK_TOKENS"], "8192")
        self.assertEqual(captured["PRIVATE_TOKEN"], "***")

    def test_test_info_records_resume_config_env_and_attempt_count(self):
        with tempfile.TemporaryDirectory() as tmp:
            args, _ = parse_args(
                [
                    "--result_dir",
                    tmp,
                    "--cache_grid_json",
                    "grid.json",
                    "--partial",
                    "2",
                ]
            )
            remaining = [
                "--model_type",
                "deepseek_v4",
                "--checkpoint_path",
                "/weights/model",
                "--tokenizer_path",
                "/weights/model",
            ]
            resume_config = {"model": {"checkpoint_path": "/weights/model"}}
            with patch.dict("os.environ", {"DSV4_CHUNK_TOKENS": "8192"}):
                _write_test_info(
                    args,
                    remaining,
                    status="running",
                    expected_cache_block_size=512,
                    resume_config=resume_config,
                )
                _write_test_info(
                    args,
                    remaining,
                    status="completed",
                    expected_cache_block_size=512,
                    resume_config=resume_config,
                    effective_cache_grpc_port=12353,
                )
            info = json.loads(
                (Path(tmp) / "test_info.json").read_text(encoding="utf-8")
            )
        self.assertEqual(info["schema_version"], 3)
        self.assertEqual(info["status"], "completed")
        self.assertEqual(info["attempt_count"], 1)
        self.assertEqual(info["engine_environment"]["DSV4_CHUNK_TOKENS"], "8192")
        self.assertEqual(info["resume_config"], resume_config)
        self.assertIsNone(info["cache_grpc_port_requested"])
        self.assertEqual(info["cache_grpc_port_effective"], 12353)
        self.assertEqual(info["cache_grpc_port"], 12353)
        self.assertEqual(
            info["resume_config_sha256"], resume_config_fingerprint(resume_config)
        )

    def test_test_info_hashes_credentials_in_nested_provenance(self):
        with tempfile.TemporaryDirectory() as tmp:
            args, _ = parse_args(
                [
                    "--result_dir",
                    tmp,
                    "--cache_grid_json",
                    "grid.json",
                    "--partial",
                    "2",
                ]
            )
            args._profile = {
                "schema_version": 1,
                "engine": {"checkpoint_path": "https://u:p@example/model"},
            }
            secret = "https://example/data?token=secret"
            _write_test_info(
                args,
                ["--checkpoint_path", secret, "--tokenizer_path", secret],
                resume_config={"model": {"checkpoint_path": secret}},
            )
            encoded = (Path(tmp) / "test_info.json").read_text()
        self.assertNotIn("secret", encoded)
        self.assertNotIn("u:p", encoded)
        self.assertIn("sha256:", encoded)

    def test_standard_test_info_records_effective_service_limits(self):
        with tempfile.TemporaryDirectory() as tmp:
            args, _ = parse_args(["--result_dir", tmp, "--max_seq_len", "1024"])
            write_standard_test_info(
                args,
                ["--model_type", "qwen_2", "--checkpoint_path", "/weights/model"],
                effective_max_seq_len=2048,
                service_concurrency_limit=7,
            )
            info = json.loads((Path(tmp) / "test_info.json").read_text())
        self.assertEqual(info["schema_version"], 5)
        self.assertEqual(info["requested_max_seq_len"], 1024)
        self.assertEqual(info["effective_max_seq_len"], 2048)
        self.assertEqual(info["service_concurrency_limit"], 7)

    def test_resolve_cache_block_size_requires_single_canonical_location(self):
        self.assertEqual(
            _resolve_cache_block_size({"generator": {"cache_alignment": 512}}), 512
        )
        for payload in (
            {},
            None,
            {"generator": {}},
            {"generator": {"cache_sampling": {"alignment": 512}}},
            {
                "generator": {
                    "cache_alignment": 512,
                    "cache_sampling": {"alignment": 512},
                }
            },
            {"generator": {"cache_alignment": 512}, "cache_block_size": 512},
        ):
            with self.subTest(payload=payload), self.assertRaises(ValueError):
                _resolve_cache_block_size(payload)
        for value in (0, -1, True, 512.0, "512", None):
            with self.subTest(value=value), self.assertRaises(ValueError):
                _resolve_cache_block_size({"generator": {"cache_alignment": value}})

    def test_dedupe_collapses_cases_in_same_block_bucket(self):
        cases = [
            {"case_id": 0, "batch_size": 1, "input_len": 4096, "cache_len": 2048},
            {"case_id": 1, "batch_size": 1, "input_len": 4096, "cache_len": 2304},
            {"case_id": 2, "batch_size": 1, "input_len": 4096, "cache_len": 2400},
        ]
        deduped = _dedupe_cache_grid_cases(cases, 512)
        self.assertEqual(len(deduped), 1)
        self.assertEqual(deduped[0]["cache_len"], 2048)

    def test_dedupe_prefers_aligned_representative(self):
        cases = [
            {"case_id": 0, "batch_size": 1, "input_len": 4096, "cache_len": 2304},
            {"case_id": 1, "batch_size": 1, "input_len": 4096, "cache_len": 2048},
        ]
        deduped = _dedupe_cache_grid_cases(cases, 512)
        self.assertEqual(len(deduped), 1)
        self.assertEqual(deduped[0]["cache_len"], 2048)

    def test_dedupe_prefers_cold_over_partial_first_block(self):
        cases = [
            {"case_id": 0, "batch_size": 1, "input_len": 4096, "cache_len": 300},
            {"case_id": 1, "batch_size": 1, "input_len": 4096, "cache_len": 0},
        ]
        deduped = _dedupe_cache_grid_cases(cases, 512)
        self.assertEqual(len(deduped), 1)
        self.assertEqual(deduped[0]["cache_len"], 0)

    def test_dedupe_keeps_distinct_inputs_and_buckets(self):
        cases = [
            {"case_id": 0, "batch_size": 1, "input_len": 4096, "cache_len": 2048},
            {"case_id": 1, "batch_size": 1, "input_len": 8192, "cache_len": 2048},
            {"case_id": 2, "batch_size": 1, "input_len": 4096, "cache_len": 4096 - 512},
        ]
        deduped = _dedupe_cache_grid_cases(cases, 512)
        self.assertEqual(len(deduped), 3)


class EngineServerSchedulerModeTest(unittest.TestCase):
    @patch("rtp_llm.test.perf_test.server.MagaServerManager")
    def test_start_does_not_force_reuse_cache(self, manager):
        manager.return_value.start_server.return_value = True
        self._server().start(8192, 1)
        self.assertNotIn("REUSE_CACHE", manager.call_args.kwargs["env_args"])

    @patch("rtp_llm.test.perf_test.server.MagaServerManager")
    def test_nsys_disables_environment_and_cli_timeline(self, manager):
        server = self._server()
        server._args.cache_profile_backend = "nsys"
        server._build_engine_cli = Mock(return_value="--gen_timeline_sync=True")
        manager.return_value.start_server.return_value = True
        server.start(8192, 1)
        options = manager.call_args.kwargs
        self.assertEqual(options["env_args"]["GEN_TIMELINE_SYNC"], "0")
        self.assertTrue(options["smoke_args_str"].endswith("--gen_timeline_sync=False"))

    def _server(self):
        args = argparse.Namespace(partial=2, result_dir="/tmp/result", dp_size=1)
        server = EngineServer(args, [])
        server._server = Mock(port=12345)
        return server

    @patch("rtp_llm.test.perf_test.server.requests.post")
    def test_set_scheduler_mode_posts_prefill_configuration(self, post):
        response = Mock(status_code=200)
        response.json.return_value = {"status": "ok"}
        post.return_value = response

        result = self._server().set_scheduler_mode(batch_size=1, mode="prefill")

        self.assertEqual(result, {"status": "ok"})
        post.assert_called_once_with(
            "http://127.0.0.1:12345/update_scheduler_info",
            json={"batch_size": 1, "mode": "prefill"},
            timeout=60,
        )

    @patch("rtp_llm.test.perf_test.cache_grid.runner.scheduler.time.sleep")
    @patch("rtp_llm.test.perf_test.server.requests.post")
    def test_set_scheduler_mode_fails_after_retries(self, post, sleep):
        post.side_effect = ConnectionError("server unavailable")

        with self.assertRaisesRegex(RuntimeError, "after retries"):
            self._server().set_scheduler_mode(batch_size=1, mode="prefill")

        self.assertEqual(post.call_count, 20)
        self.assertEqual(sleep.call_count, 19)

    @patch("rtp_llm.test.perf_test.server.requests.post")
    def test_set_scheduler_mode_rejects_invalid_mode(self, post):
        with self.assertRaisesRegex(ValueError, "unsupported scheduler mode"):
            self._server().set_scheduler_mode(batch_size=1, mode="invalid")
        post.assert_not_called()

    @patch("rtp_llm.test.perf_test.cache_grid.runner.scheduler.time.sleep")
    @patch("rtp_llm.test.perf_test.server.requests.post")
    def test_startup_retries_rejected_acknowledgement(self, post, sleep):
        rejected, accepted = Mock(), Mock()
        rejected.json.return_value = {"status": "ok", "error": "not ready"}
        accepted.json.return_value = {"status": "ok"}
        post.side_effect = [rejected, accepted]
        self.assertEqual(
            self._server().set_scheduler_mode(batch_size=1, mode="prefill"),
            {"status": "ok"},
        )
        self.assertEqual(post.call_count, 2)
        sleep.assert_called_once_with(3)

    def test_measurement_rejects_missing_or_error_ack_without_retry(self):
        for ack in (
            {},
            {"status": "ok", "error": "not ready"},
            {"status": "error"},
            [],
        ):
            with self.subTest(ack=ack):
                session = Mock()
                session.post.return_value.json.return_value = ack
                runner = argparse.Namespace(
                    port=12345, request_timeout=120, _http_session=session
                )
                with self.assertRaisesRegex(RuntimeError, "scheduler rejected"):
                    CacheGridRunner._set_prefill_batch_size(runner, 2)
                session.post.assert_called_once_with(
                    "http://127.0.0.1:12345/update_scheduler_info",
                    json={"batch_size": 2, "mode": "prefill"},
                    timeout=60,
                )


class CacheGridProfileTest(unittest.TestCase):
    def nsys_runner(self, tmp, **kwargs):
        return self.runner(
            tmp,
            profile_runs=kwargs.pop("profile_runs", 1),
            profile_case_ids=[1],
            profile_only=True,
            profile_flat_output=True,
            profile_backend="nsys",
            nsys_session="test_session",
            nsys_path="/test/nsys",
            nsys_tail_seconds=0,
            **kwargs,
        )

    def test_nsys_three_rounds_skip_kineto_and_export_before_next_seed(self):
        with tempfile.TemporaryDirectory() as tmp:
            runner = self.nsys_runner(tmp, profile_runs=3)
            events = self.wire(runner, write_traces=False)
            outputs = []

            def control(command, **kwargs):
                action = command[1]
                events.append(action)
                self.assertIn("--session=test_session", command)
                if action == "start":
                    outputs.append(
                        Path(
                            next(
                                x.split("=", 1)[1]
                                for x in command
                                if x.startswith("--output=")
                            )
                        )
                    )
                else:
                    outputs[-1].with_suffix(".nsys-rep").write_bytes(b"mock report")
                return subprocess.CompletedProcess(command, 0, "ok", "")

            with patch(
                "rtp_llm.test.perf_test.cache_grid.runner.cache_grid_runner.subprocess.run",
                side_effect=control,
            ):
                records = runner.run()
            self.assertEqual(events, ["seed", "start", "profile", "stop"] * 3)
            runner._http_session.post.assert_not_called()
            self.assertEqual(len(set(outputs)), 3)
            for record in records:
                self.assertEqual(record["status"], "ok")
                self.assertTrue((Path(tmp) / record["trace_files"][0]).is_file())
                self.assertEqual(
                    [c["returncode"] for c in record["nsys_commands"]], [0, 0]
                )
            self.assertFalse((Path(tmp) / "timelines").exists())
            self.assertEqual(
                json.loads((Path(tmp) / "manifest.json").read_text())[
                    "profile_backend"
                ],
                "nsys",
            )

    def test_nsys_start_failure_prevents_target_and_attempts_stop(self):
        for timeout in [False, True]:
            with self.subTest(timeout=timeout), tempfile.TemporaryDirectory() as tmp:
                runner = self.nsys_runner(tmp)
                events = self.wire(runner, write_traces=False)
                failure = (
                    subprocess.TimeoutExpired("nsys", 1)
                    if timeout
                    else subprocess.CompletedProcess([], 1, "", "start failed")
                )
                with patch(
                    "rtp_llm.test.perf_test.cache_grid.runner.cache_grid_runner.subprocess.run",
                    side_effect=[failure, subprocess.CompletedProcess([], 0, "", "")],
                ) as call:
                    with self.assertRaises((RuntimeError, subprocess.TimeoutExpired)):
                        runner.run()
                self.assertEqual(events, ["seed"])
                self.assertEqual(
                    [c.args[0][1] for c in call.call_args_list], ["start", "stop"]
                )
                self.assertEqual(
                    json.loads((Path(tmp) / "manifest.json").read_text())["status"],
                    "failed",
                )

    def test_nsys_request_exception_preserved_when_cleanup_also_fails(self):
        with tempfile.TemporaryDirectory() as tmp:
            runner = self.nsys_runner(tmp)
            runner._post_request = Mock(
                side_effect=[{"success": True}, ValueError("request broke")]
            )
            runner._http_session.post = Mock()
            with patch(
                "rtp_llm.test.perf_test.cache_grid.runner.cache_grid_runner.subprocess.run",
                side_effect=[
                    subprocess.CompletedProcess([], 0, "", ""),
                    subprocess.CompletedProcess([], 1, "", "stop failed"),
                ],
            ):
                with self.assertRaisesRegex(ValueError, "request broke"):
                    runner.run()
            record = json.loads((Path(tmp) / "manifest.json").read_text())["records"][0]
            self.assertIn("stop failed", record["nsys_stop_error"])
            self.assertIn("request broke", record["error"])
            runner._http_session.post.assert_not_called()

    def test_nsys_missing_report_or_stop_failure_is_not_success(self):
        for code in [0, 1]:
            with self.subTest(code=code), tempfile.TemporaryDirectory() as tmp:
                runner = self.nsys_runner(tmp)
                self.wire(runner, write_traces=False)
                with patch(
                    "rtp_llm.test.perf_test.cache_grid.runner.cache_grid_runner.subprocess.run",
                    side_effect=[
                        subprocess.CompletedProcess([], 0, "", ""),
                        subprocess.CompletedProcess([], code, "", "stop failed"),
                    ],
                ):
                    with self.assertRaises(RuntimeError):
                        runner.run()
                self.assertEqual(
                    json.loads((Path(tmp) / "manifest.json").read_text())["status"],
                    "failed",
                )

    def test_nsys_bad_request_result_still_exports_and_marks_failed(self):
        with tempfile.TemporaryDirectory() as tmp:
            runner = self.nsys_runner(tmp)
            events = self.wire(runner, bad_reuse=True, write_traces=False)
            outputs = []

            def control(command, **kwargs):
                events.append(command[1])
                if command[1] == "start":
                    outputs.append(Path(command[-1].split("=", 1)[1]))
                else:
                    outputs[-1].with_suffix(".nsys-rep").write_bytes(b"mock report")
                return subprocess.CompletedProcess(command, 0, "", "")

            with patch(
                "rtp_llm.test.perf_test.cache_grid.runner.cache_grid_runner.subprocess.run",
                side_effect=control,
            ):
                with self.assertRaisesRegex(RuntimeError, "shape/reuse mismatch"):
                    runner.run()
            self.assertEqual(events, ["seed", "start", "profile", "stop"])
            record = json.loads((Path(tmp) / "manifest.json").read_text())["records"][0]
            self.assertEqual(record["status"], "failed")
            self.assertTrue((Path(tmp) / record["trace_files"][0]).is_file())

    def test_flat_output_directory_is_claimed_without_nesting(self):
        with tempfile.TemporaryDirectory() as tmp:
            for name in (
                "cache_perf_launch.json",
                "grid.snapshot.json",
                "profile.snapshot.json",
            ):
                (Path(tmp) / name).write_text("{}")
            args = argparse.Namespace(
                result_dir=tmp, cache_profile_only=True, cache_profile_flat_output=True
            )
            _prepare_cache_profile_result_dir(args)
            self.assertEqual(args.result_dir, tmp)
            self.assertFalse((Path(tmp) / "cache_profile_replays").exists())
            with self.assertRaisesRegex(ValueError, "fresh isolated"):
                _prepare_cache_profile_result_dir(args)

    def test_flat_output_rejects_existing_result_files(self):
        for filename in ("test_info.json", "cache_grid_results.json", "manifest.json"):
            with self.subTest(filename=filename), tempfile.TemporaryDirectory() as tmp:
                path = Path(tmp) / filename
                path.write_text("original")
                args = argparse.Namespace(
                    result_dir=tmp,
                    cache_profile_only=True,
                    cache_profile_flat_output=True,
                )
                with self.assertRaisesRegex(ValueError, "fresh isolated"):
                    _prepare_cache_profile_result_dir(args)
                self.assertEqual(path.read_text(), "original")

    def test_nested_profile_output_creates_isolated_directory(self):
        with tempfile.TemporaryDirectory() as tmp:
            args = argparse.Namespace(
                result_dir=tmp, cache_profile_only=True, cache_profile_flat_output=False
            )
            _prepare_cache_profile_result_dir(args)
            self.assertEqual(
                Path(args.result_dir).parent, Path(tmp) / "cache_profile_replays"
            )

    def test_flat_flag_requires_profile_only(self):
        with self.assertRaises(SystemExit):
            parse_args(["--cache_profile_flat_output"])

    def test_flat_manifest_and_timelines_share_output_directory(self):
        with tempfile.TemporaryDirectory() as tmp:
            runner = self.runner(
                tmp,
                profile_runs=2,
                profile_case_ids=[1],
                profile_tp_size=2,
                profile_only=True,
                profile_flat_output=True,
            )
            events = self.wire(runner)
            runner.run()
            self.assertEqual(events, ["seed", "arm", "profile"] * 2)
            manifest = json.loads((Path(tmp) / "manifest.json").read_text())
            self.assertEqual(manifest["status"], "completed")
            self.assertTrue(manifest["session_id"])
            self.assertEqual(len(manifest["records"]), 2)
            for record in manifest["records"]:
                self.assertEqual(record["case_id"], 1)
                self.assertEqual(len(record["trace_files"]), 2)
                for path in record["trace_files"]:
                    self.assertEqual(Path(path).parent, Path("timelines"))
                    self.assertTrue((Path(tmp) / path).is_file())
            self.assertFalse((Path(tmp) / "cache_profiles").exists())
            self.assertFalse((Path(tmp) / "cache_grid_results.json").exists())
            before = (Path(tmp) / "manifest.json").read_bytes()
            again = self.runner(
                tmp,
                profile_runs=1,
                profile_case_ids=[1],
                profile_only=True,
                profile_flat_output=True,
            )
            later_events = self.wire(again)
            with self.assertRaises(FileExistsError):
                again.run()
            self.assertEqual(later_events, [])
            self.assertEqual((Path(tmp) / "manifest.json").read_bytes(), before)

    def test_flat_profile_failure_preserves_failed_manifest(self):
        with tempfile.TemporaryDirectory() as tmp:
            runner = self.runner(
                tmp,
                profile_runs=1,
                profile_case_ids=[1],
                profile_only=True,
                profile_flat_output=True,
            )
            self.wire(runner, bad_reuse=True)
            with self.assertRaisesRegex(RuntimeError, "shape/reuse mismatch"):
                runner.run()
            manifest = json.loads((Path(tmp) / "manifest.json").read_text())
            self.assertEqual(manifest["status"], "failed")
            self.assertEqual(manifest["records"][0]["status"], "failed")

    cases = [
        {"case_id": 1, "batch_size": 1, "input_len": 16, "cache_len": 8},
        {"case_id": 2, "batch_size": 1, "input_len": 16, "cache_len": 0},
    ]

    def runner(self, tmp, **kwargs):
        return CacheGridRunner(
            12345,
            _WhitespaceTokenizer(),
            self.cases,
            tmp,
            cache_commit_tail_tokens=8,
            profile_trace_timeout=0.01,
            **kwargs,
        )

    def wire(self, runner, *, bad_reuse=False, write_traces=True):
        events = []
        active = []

        def arm(url, json, timeout):
            self.assertTrue(url.endswith("/start_profile"))
            self.assertEqual(json["num_steps"], 1)
            self.assertTrue(json["enable_all_rank"])
            events.append("arm")
            active.append(json["trace_name"])
            response = Mock()
            response.json.return_value = {"status": "ok"}
            return response

        def post(text, ids, request_id):
            is_profile = request_id.endswith(":profile")
            events.append(
                "profile"
                if is_profile
                else "seed" if request_id.endswith(":seed") else "measure"
            )
            if is_profile and write_traces:
                for rank in range(runner.profile_tp_size):
                    (runner.result_dir / f"{active[-1]}_wr{rank}_1.json").write_text(
                        json.dumps({"traceEvents": [{"name": "kernel", "dur": 1}]})
                    )
            cached = "seq16_cache8" in request_id
            return {
                "prefill_time_ms": 1000.0 if is_profile else 10.0,
                "success": True,
                "request_id": request_id,
                "input_len": len(text.split()),
                "output_len": 1,
                "reuse_len": 0 if bad_reuse and is_profile else 8 if cached else 0,
                "ttft_ms": 1000.0 if is_profile else 10.0,
            }

        runner._http_session.post = Mock(side_effect=arm)
        runner._post_request = Mock(side_effect=post)
        return events

    def test_disabled_by_default(self):
        with tempfile.TemporaryDirectory() as tmp:
            runner = self.runner(tmp)
            events = self.wire(runner)
            rows = runner.run()
            self.assertNotIn("arm", events)
            self.assertEqual([r["median_prefill_time_ms"] for r in rows], [10.0, 10.0])
            self.assertFalse((Path(tmp) / "cache_profiles").exists())

    def test_selected_profiles_follow_measurements_and_preserve_baseline(self):
        with tempfile.TemporaryDirectory() as tmp:
            runner = self.runner(
                tmp, profile_runs=2, profile_case_ids=[1], profile_tp_size=2
            )
            events = self.wire(runner)
            rows = runner.run()
            self.assertEqual(events[-6:], ["seed", "arm", "profile"] * 2)
            self.assertEqual(events.count("measure"), 6)
            self.assertEqual([r["median_prefill_time_ms"] for r in rows], [10.0, 10.0])
            self.assertEqual([len(r["runs"]) for r in rows], [3, 3])
            manifest = json.loads(
                next((Path(tmp) / "cache_profiles").glob("*/manifest.json")).read_text()
            )
            self.assertEqual(manifest["status"], "completed")
            self.assertEqual([r["case_id"] for r in manifest["records"]], [1, 1])
            self.assertNotEqual(
                manifest["records"][0]["prompt_case_id"],
                manifest["records"][1]["prompt_case_id"],
            )
            for record in manifest["records"]:
                self.assertEqual(len(record["trace_files"]), 2)
                for path in record["trace_files"]:
                    self.assertTrue((Path(tmp) / path).is_file())

    def test_profile_only_cold_replays_are_unique_and_do_not_write_measurements(self):
        with tempfile.TemporaryDirectory() as tmp:
            runner = self.runner(
                tmp, profile_runs=2, profile_case_ids=[2], profile_only=True
            )
            events = self.wire(runner)
            records = runner.run()
            self.assertEqual(events, ["arm", "profile"] * 2)
            texts = [c.args[0] for c in runner._post_request.call_args_list]
            self.assertNotEqual(texts[0], texts[1])
            self.assertEqual([r["result"]["reuse_len"] for r in records], [0, 0])
            self.assertFalse((Path(tmp) / "cache_grid_results.json").exists())

    def test_completed_baseline_can_be_profiled_without_remeasurement(self):
        with tempfile.TemporaryDirectory() as tmp:
            first = self.runner(tmp)
            self.wire(first)
            first.run()
            baseline = (Path(tmp) / "cache_grid_results.json").read_bytes()
            runner = self.runner(
                tmp, profile_runs=1, profile_case_ids=[1], profile_only=True
            )
            events = self.wire(runner)
            runner.run()
            self.assertEqual(events, ["seed", "arm", "profile"])
            self.assertEqual(
                (Path(tmp) / "cache_grid_results.json").read_bytes(), baseline
            )

    def test_profile_failure_does_not_invalidate_completed_baseline(self):
        with tempfile.TemporaryDirectory() as tmp:
            runner = self.runner(tmp, profile_runs=1, profile_case_ids=[1])
            self.wire(runner, bad_reuse=True)
            with self.assertRaisesRegex(RuntimeError, "shape/reuse mismatch"):
                runner.run()
            baseline = json.loads((Path(tmp) / "cache_grid_results.json").read_text())
            self.assertTrue(baseline["complete"])
            self.assertEqual(
                [r["median_prefill_time_ms"] for r in baseline["metrics"]], [10.0, 10.0]
            )
            manifest = json.loads(
                next((Path(tmp) / "cache_profiles").glob("*/manifest.json")).read_text()
            )
            self.assertEqual(manifest["status"], "failed")
            self.assertTrue(manifest["records"][0]["trace_files"])

    def test_profile_api_error_prevents_target_request(self):
        with tempfile.TemporaryDirectory() as tmp:
            runner = self.runner(
                tmp, profile_runs=1, profile_case_ids=[1], profile_only=True
            )
            events = self.wire(runner)
            runner._http_session.post.side_effect = None
            runner._http_session.post.return_value.json.return_value = {
                "error": "broadcast failed"
            }
            with self.assertRaisesRegex(RuntimeError, "start_profile failed"):
                runner.run()
            self.assertEqual(events, ["seed"])

    def test_missing_trace_is_reported_as_failure(self):
        with tempfile.TemporaryDirectory() as tmp:
            runner = self.runner(
                tmp, profile_runs=1, profile_case_ids=[2], profile_only=True
            )
            self.wire(runner, write_traces=False)
            with self.assertRaisesRegex(RuntimeError, "missing TP ranks"):
                runner.run()

    def test_incomplete_json_is_not_accepted_as_trace(self):
        with tempfile.TemporaryDirectory() as tmp:
            runner = self.runner(
                tmp, profile_runs=1, profile_case_ids=[2], profile_only=True
            )
            (Path(tmp) / "example_wr0_1.json").write_text('{"traceEvents": [')
            try:
                with self.assertRaisesRegex(RuntimeError, "trace timeout"):
                    runner._wait_profile_traces("example")
            finally:
                runner._close_resources()

    def test_cli_profile_only_preserves_existing_report_directory(self):
        from rtp_llm.test.perf_test import batch_decode_test as entry

        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            grid = root / "grid.json"
            grid.write_text(
                json.dumps(
                    {
                        "schema_version": 2,
                        "generator": {"cache_alignment": 8},
                        "cases": self.cases,
                    }
                )
            )
            original_info = root / "test_info.json"
            original_info.write_text('{"original": true}')
            argv = [
                "--result_dir",
                tmp,
                "--cache_grid_json",
                str(grid),
                "--partial",
                "2",
                "--cache_commit_tail_tokens",
                "8",
                "--tokenizer_path",
                "/mock/tokenizer",
                "--tp_size",
                "2",
                "--cache_profile_only",
                "--cache_profile_runs",
                "1",
                "--cache_profile_case_ids",
                "1",
            ]
            parsed = parse_args(argv)
            with patch.object(entry, "parse_args", return_value=parsed), patch.object(
                entry, "_ensure_xgrammar_lib_path"
            ), patch.object(
                entry, "resolve_perf_engine_paths", side_effect=lambda x: x
            ), patch.object(
                entry, "EngineServer"
            ) as server, patch(
                "transformers.AutoTokenizer.from_pretrained",
                return_value=_WhitespaceTokenizer(),
            ), patch.object(
                entry, "collect_timeline_files"
            ), patch.object(
                CacheGridRunner, "run", autospec=True, return_value=[]
            ) as run, patch.dict(
                "os.environ", {}, clear=False
            ):
                server.return_value.port = 12345

                def assert_prefill_mode_before_run(_runner):
                    server.return_value.set_scheduler_mode.assert_called_once_with(
                        batch_size=1, mode="prefill"
                    )
                    return []

                run.side_effect = assert_prefill_mode_before_run
                result_dir = Path(entry.main())
                runner = run.call_args.args[0]
                self.assertTrue(runner.profile_only)
                self.assertEqual(runner.profile_tp_size, 2)
                self.assertEqual(runner.profile_case_ids, {1})
                runner._close_resources()
                self.assertEqual(result_dir.parent, root / "cache_profile_replays")
                self.assertEqual(original_info.read_text(), '{"original": true}')
                self.assertTrue((result_dir / "test_info.json").is_file())
                server.return_value.set_scheduler_mode.assert_called_once_with(
                    batch_size=1, mode="prefill"
                )
                server.return_value.stop.assert_called_once()

    def test_profile_cli_defaults_and_validation(self):
        args, _ = parse_args([])
        self.assertEqual(args.cache_profile_runs, 0)
        self.assertFalse(args.cache_profile_only)
        args, remaining = parse_args(
            [
                "--cache_grid_json",
                "grid.json",
                "--partial",
                "2",
                "--cache_profile_runs",
                "2",
                "--cache_profile_case_ids",
                "1",
                "2",
                "--cache_profile_only",
            ]
        )
        self.assertEqual(args.cache_profile_case_ids, [1, 2])
        self.assertFalse(remaining)
        for options in [
            ["--cache_profile_only"],
            ["--cache_profile_runs", "1"],
            ["--cache_profile_case_ids", "1"],
            ["--cache_profile_runs", "-1"],
            ["--cache_profile_trace_timeout", "0"],
        ]:
            with self.subTest(options=options), patch("sys.stderr"), self.assertRaises(
                SystemExit
            ):
                parse_args(options)


class CacheGridRunnerBlockProbeTest(unittest.TestCase):
    def _make_runner(self, tmp, block_size):
        cases = [{"case_id": 0, "batch_size": 1, "input_len": 1024, "cache_len": 512}]
        return CacheGridRunner(
            0,
            _WordTokenizer(),
            cases,
            tmp,
            expected_block_size=block_size,
        )

    def test_probe_passes_on_matching_reuse_len(self):
        with tempfile.TemporaryDirectory() as tmp:
            runner = self._make_runner(tmp, 512)
            with mock.patch(
                "rtp_llm.test.perf_test.cache_grid.runner.cache_grid_runner._post_prefill",
                return_value={"success": True, "reuse_len": 512},
            ) as post:
                runner._probe_reuse_granularity()
            self.assertEqual(post.call_count, 2)

    def test_probe_aborts_on_granularity_mismatch(self):
        with tempfile.TemporaryDirectory() as tmp:
            runner = self._make_runner(tmp, 512)
            with mock.patch(
                "rtp_llm.test.perf_test.cache_grid.runner.cache_grid_runner._post_prefill",
                return_value={"success": True, "reuse_len": 0},
            ):
                with self.assertRaisesRegex(RuntimeError, "granularity mismatch"):
                    runner._probe_reuse_granularity()

    def test_probe_aborts_on_failed_request(self):
        with tempfile.TemporaryDirectory() as tmp:
            runner = self._make_runner(tmp, 512)
            with mock.patch(
                "rtp_llm.test.perf_test.cache_grid.runner.cache_grid_runner._post_prefill",
                return_value={"success": False, "error": "HTTP 500"},
            ):
                with self.assertRaisesRegex(RuntimeError, "probe seed failed"):
                    runner._probe_reuse_granularity()

    def test_probe_skipped_without_block_size(self):
        with tempfile.TemporaryDirectory() as tmp:
            runner = self._make_runner(tmp, 0)
            with mock.patch(
                "rtp_llm.test.perf_test.cache_grid.runner.cache_grid_runner._post_prefill"
            ) as post:
                runner._probe_reuse_granularity()
            post.assert_not_called()


class PrefixPromptFactoryFastPathTest(unittest.TestCase):
    GEOMETRIES = [(5, 100, 50), (5, 100, 0), (9, 7, 3), (9, 7, 0)]

    def test_fast_prompts_match_exact_text_construction(self):
        for case_id, total_len, cache_len in self.GEOMETRIES:
            fast = PrefixPromptFactory(_WordTokenizer()).build_case_prompts(
                case_id, total_len, cache_len, 3
            )
            exact = PrefixPromptFactory(_WordTokenizer())._text_case_prompts(
                case_id, total_len, cache_len, 3
            )
            self.assertEqual(fast.seed_text, exact.seed_text)
            self.assertEqual(fast.prefix_ids, exact.prefix_ids)
            self.assertEqual(fast.run_texts, exact.run_texts)
            self.assertEqual(fast.run_ids, exact.run_ids)
            self.assertEqual(fast.built_len, total_len)

    def test_first_case_of_each_shape_is_fully_verified(self):
        factory = PrefixPromptFactory(_WordTokenizer())
        factory.build_case_prompts(1, 100, 50, 3)
        factory.build_case_prompts(2, 100, 0, 3)
        self.assertEqual(factory._verified_shapes, {"cached", "cold"})
        self.assertEqual(factory._unsafe_shapes, set())

    def test_merging_tail_falls_back_to_exact_text_path(self):
        factory = PrefixPromptFactory(_TailMergingTokenizer())
        # Exact text construction cannot preserve the prefix for this tokenizer,
        # so the fallback surfaces its original safety error.
        with self.assertRaisesRegex(ValueError, "preserve cache prefix"):
            factory.build_case_prompts(1, 100, 50, 3)


class MaterializedCaseStoreTest(unittest.TestCase):
    CASES = [
        {"case_id": 0, "batch_size": 1, "input_len": 100, "cache_len": 0},
        {"case_id": 1, "batch_size": 1, "input_len": 100, "cache_len": 50},
        {"case_id": 2, "batch_size": 1, "input_len": 60, "cache_len": 40},
    ]

    def test_round_trip_reproduces_constructed_prompts(self):
        factory = PrefixPromptFactory(_WordTokenizer())
        with tempfile.TemporaryDirectory() as tmp:
            store = MaterializedCaseStore(tmp)
            stats = store.materialize(self.CASES, factory, 3, grid_sha256="s" * 64)
            self.assertEqual(stats["cases"], 3)
            self.assertEqual(stats["cached_cases"], 2)
            self.assertEqual(stats["verbatim_seed"], 0)
            self.assertEqual(stats["verbatim_runs"], 0)

            loaded = MaterializedCaseStore(tmp)
            self.assertEqual(loaded.load_cases(), self.CASES)
            self.assertEqual(loaded.run_count, 3)
            self.assertEqual(loaded.grid_sha256, "s" * 64)
            for case in self.CASES:
                expected = factory.build_case_prompts(
                    case["case_id"], case["input_len"], case["cache_len"], 3
                )
                got = loaded.load_case(case)
                self.assertEqual(got.seed_text, expected.seed_text)
                self.assertEqual(got.run_texts, expected.run_texts)

    def test_load_missing_store_raises(self):
        with tempfile.TemporaryDirectory() as tmp:
            with self.assertRaisesRegex(
                FileNotFoundError, "not a materialized case store"
            ):
                MaterializedCaseStore(tmp).load_cases()


class MaterializedStoreValidationTest(unittest.TestCase):
    CASES = [
        {"case_id": 0, "batch_size": 1, "input_len": 100, "cache_len": 0},
        {"case_id": 1, "batch_size": 1, "input_len": 100, "cache_len": 50},
    ]

    def _materialize(self, tmp, run_count=3, sha="a" * 64, profile_sha=""):
        store = MaterializedCaseStore(tmp)
        store.materialize(
            self.CASES,
            PrefixPromptFactory(_WordTokenizer()),
            run_count,
            grid_sha256=sha,
            profile_sha256=profile_sha,
        )

    def test_accepts_matching_store(self):
        with tempfile.TemporaryDirectory() as tmp:
            self._materialize(tmp)
            store = _load_materialized_case_store(tmp, self.CASES, 3, "a" * 64)
            self.assertEqual(store.run_count, 3)

    def test_rejects_run_count_mismatch(self):
        with tempfile.TemporaryDirectory() as tmp:
            self._materialize(tmp)
            with self.assertRaisesRegex(ValueError, "run_count"):
                _load_materialized_case_store(tmp, self.CASES, 2, "a" * 64)

    def test_rejects_grid_sha_mismatch(self):
        with tempfile.TemporaryDirectory() as tmp:
            self._materialize(tmp)
            with self.assertRaisesRegex(ValueError, "different grid JSON"):
                _load_materialized_case_store(tmp, self.CASES, 3, "b" * 64)

    def test_rejects_geometry_mismatch(self):
        with tempfile.TemporaryDirectory() as tmp:
            self._materialize(tmp)
            other = [dict(self.CASES[0])]
            with self.assertRaisesRegex(
                ValueError, r"holds 2 cases but this run plans 1"
            ):
                _load_materialized_case_store(tmp, other, 3, "a" * 64)

    def test_rejects_profile_sha_mismatch(self):
        with tempfile.TemporaryDirectory() as tmp:
            self._materialize(tmp, profile_sha="a" * 64)
            with self.assertRaisesRegex(ValueError, "different profile"):
                _load_materialized_case_store(tmp, self.CASES, 3, "a" * 64, "b" * 64)


class CacheGridRunnerStoreTest(unittest.TestCase):
    def test_run_sends_materialized_prompts_without_tokenizer_construction(self):
        cases = [
            {"case_id": 0, "batch_size": 1, "input_len": 100, "cache_len": 0},
            {"case_id": 1, "batch_size": 1, "input_len": 100, "cache_len": 50},
        ]
        factory = PrefixPromptFactory(_WordTokenizer())
        with tempfile.TemporaryDirectory() as tmp:
            store = MaterializedCaseStore(str(Path(tmp) / "store"))
            store.materialize(cases, factory, 2)
            expected = {
                case["case_id"]: factory.build_case_prompts(
                    case["case_id"], case["input_len"], case["cache_len"], 2
                )
                for case in cases
            }
            runner = CacheGridRunner(
                0,
                _WordTokenizer(),
                cases,
                str(Path(tmp) / "results"),
                measure_runs=2,
                cache_commit_tail_tokens=10,
                fail_fast=False,
                case_store=store,
            )
            with mock.patch(
                "rtp_llm.test.perf_test.cache_grid.runner.cache_grid_runner._post_prefill",
                return_value={
                    "success": True,
                    "reuse_len": 50,
                    "input_len": 100,
                    "output_len": 1,
                },
            ) as post:
                runner.run()
            sent = [call.args[1] for call in post.call_args_list]
            self.assertEqual(
                sent,
                [
                    expected[0].run_texts[0],
                    expected[0].run_texts[1],
                    factory.make_seed(1, expected[1].seed_text, 50, 10),
                    expected[1].run_texts[0],
                    expected[1].run_texts[1],
                ],
            )

    def test_runner_rejects_store_built_for_other_run_count(self):
        cases = [{"case_id": 0, "batch_size": 1, "input_len": 100, "cache_len": 0}]
        with tempfile.TemporaryDirectory() as tmp:
            store = MaterializedCaseStore(str(Path(tmp) / "store"))
            store.materialize(cases, PrefixPromptFactory(_WordTokenizer()), 3)
            with self.assertRaisesRegex(ValueError, "run_count"):
                CacheGridRunner(
                    0, _WordTokenizer(), cases, tmp, measure_runs=2, case_store=store
                )


class CacheGridRunnerResumeGuardTest(unittest.TestCase):
    def _write_existing_results(self, result_dir, payload):
        path = Path(result_dir) / "cache_grid_results.json"
        path.parent.mkdir(parents=True, exist_ok=True)
        checkpoint = {
            "schema_version": 2,
            "mode": "prefix_cache_grid",
            "seed_mode": "independent",
            "measurement_contract": "server_first_token_cost_time_ms",
            "grid_sha256": None,
            "profile_sha256": None,
            "run_config_sha256": None,
            "measure_runs": 3,
            "cache_commit_tail_tokens": 4096,
            "expected_block_size": 0,
            "request_transport": "http_prompt",
            **payload,
        }
        path.write_text(json.dumps(checkpoint), encoding="utf-8")

    def test_resume_with_matching_results_succeeds(self):
        with tempfile.TemporaryDirectory() as tmp:
            self._write_existing_results(
                tmp,
                {
                    "metrics": [],
                    "grid_sha256": "abc",
                    "profile_sha256": "def",
                    "measure_runs": 3,
                    "expected_block_size": 512,
                },
            )
            runner = CacheGridRunner(
                0,
                _WordTokenizer(),
                [{"case_id": 0, "batch_size": 1, "input_len": 1024, "cache_len": 512}],
                tmp,
                grid_sha256="abc",
                profile_sha256="def",
                expected_block_size=512,
            )
            self.assertEqual(runner._results, {})

    def test_resume_with_profile_sha_mismatch_raises(self):
        with tempfile.TemporaryDirectory() as tmp:
            self._write_existing_results(
                tmp,
                {
                    "metrics": [],
                    "grid_sha256": "abc",
                    "profile_sha256": "old_sha",
                    "measure_runs": 3,
                },
            )
            with self.assertRaisesRegex(ValueError, "profile_sha256"):
                CacheGridRunner(
                    0,
                    _WordTokenizer(),
                    [
                        {
                            "case_id": 0,
                            "batch_size": 1,
                            "input_len": 1024,
                            "cache_len": 512,
                        }
                    ],
                    tmp,
                    grid_sha256="abc",
                    profile_sha256="new_sha",
                )

    def test_resume_mismatch_allowed_with_flag(self):
        with tempfile.TemporaryDirectory() as tmp:
            self._write_existing_results(
                tmp,
                {
                    "metrics": [
                        {
                            "case_key": "bs1_seq1024_cache512",
                            "batch_size": 1,
                            "input_len": 1024,
                            "cache_len_requested": 512,
                            "measure_runs": 3,
                            "success_runs": 3,
                            "runs": [
                                {
                                    "prefill_time_ms": 5,
                                    "success": True,
                                    "ttft_ms": 5,
                                    "input_len": 1024,
                                    "output_len": 1,
                                    "reuse_len": 512,
                                }
                            ]
                            * 3,
                            "status": "ok",
                        }
                    ],
                    "grid_sha256": "abc",
                    "profile_sha256": "old",
                    "measure_runs": 3,
                },
            )
            runner = CacheGridRunner(
                0,
                _WordTokenizer(),
                [{"case_id": 0, "batch_size": 1, "input_len": 1024, "cache_len": 512}],
                tmp,
                grid_sha256="abc",
                profile_sha256="new",
                allow_resume_mismatch=True,
            )
            self.assertEqual(len(runner._results), 1)

    def test_require_resume_rejects_missing_checkpoint(self):
        with tempfile.TemporaryDirectory() as tmp:
            with self.assertRaisesRegex(ValueError, "require_cache_resume"):
                validate_cache_grid_resume(
                    tmp,
                    grid_sha256="grid",
                    profile_sha256="profile",
                    measure_runs=3,
                    cache_commit_tail_tokens=128,
                    expected_block_size=512,
                    request_transport="http_prompt",
                    require_resume=True,
                )

    def test_resume_config_mismatch_is_rejected_before_model_start(self):
        with tempfile.TemporaryDirectory() as tmp:
            old_config = {"model": {"checkpoint_path": "/weights/old"}}
            self._write_existing_results(
                tmp,
                {
                    "metrics": [],
                    "run_config_sha256": resume_config_fingerprint(old_config),
                },
            )
            with self.assertRaisesRegex(ValueError, "run_config_sha256"):
                validate_cache_grid_resume(
                    tmp,
                    grid_sha256=None,
                    profile_sha256=None,
                    measure_runs=3,
                    cache_commit_tail_tokens=128,
                    expected_block_size=0,
                    request_transport="http_prompt",
                    run_config={"model": {"checkpoint_path": "/weights/new"}},
                )

    @patch("rtp_llm.test.perf_test.cache_grid.runner.cache_grid_runner._post_prefill")
    def test_resume_skips_ok_case_and_starts_at_next_input_cache(self, post):
        cases = [
            {"case_id": 0, "batch_size": 1, "input_len": 8, "cache_len": 0},
            {"case_id": 1, "batch_size": 1, "input_len": 16, "cache_len": 0},
        ]
        post.return_value = {
            "prefill_time_ms": 5.0,
            "success": True,
            "input_len": 16,
            "output_len": 1,
            "reuse_len": 0,
            "ttft_ms": 5.0,
        }
        with tempfile.TemporaryDirectory() as tmp:
            self._write_existing_results(
                tmp,
                {
                    "metrics": [
                        {
                            "case_key": "bs1_seq8_cache0",
                            "batch_size": 1,
                            "input_len": 8,
                            "cache_len_requested": 0,
                            "measure_runs": 1,
                            "success_runs": 1,
                            "runs": [
                                {
                                    "prefill_time_ms": 5,
                                    "success": True,
                                    "ttft_ms": 5,
                                    "input_len": 8,
                                    "output_len": 1,
                                    "reuse_len": 0,
                                }
                            ],
                            "case_id": 0,
                            "status": "ok",
                        }
                    ],
                    "measure_runs": 1,
                    "request_transport": "http_prompt",
                    "cache_commit_tail_tokens": 4,
                    "started_at": "2026-09-08T00:00:00+0800",
                },
            )
            rows = CacheGridRunner(
                12345,
                _WordTokenizer(),
                cases,
                tmp,
                measure_runs=1,
                cache_commit_tail_tokens=4,
            ).run()
            result = json.loads(
                (Path(tmp) / "cache_grid_results.json").read_text(encoding="utf-8")
            )
        self.assertEqual(post.call_count, 1)
        self.assertEqual(len(rows), 2)
        self.assertTrue(result["complete"])
        self.assertEqual(result["resume_count"], 1)
        self.assertEqual(result["progress"]["completed_cases"], 2)
        self.assertIsNone(result["progress"]["next_pending_case"])

    def test_resume_replays_case_journal_newer_than_full_checkpoint(self):
        cases = [
            {"case_id": 0, "batch_size": 1, "input_len": 8, "cache_len": 0},
            {"case_id": 1, "batch_size": 1, "input_len": 16, "cache_len": 0},
        ]
        with tempfile.TemporaryDirectory() as tmp:
            first = CacheGridRunner(
                0,
                _WordTokenizer(),
                cases,
                tmp,
                measure_runs=1,
                cache_commit_tail_tokens=4,
            )
            first._save(complete=False)
            metric = {
                "case_key": first.case_key(cases[0]),
                "batch_size": 1,
                "input_len": 8,
                "cache_len_requested": 0,
                "measure_runs": 1,
                "success_runs": 1,
                "runs": [
                    {
                        "prefill_time_ms": 5,
                        "success": True,
                        "ttft_ms": 5,
                        "input_len": 8,
                        "output_len": 1,
                        "reuse_len": 0,
                    }
                ],
                "case_id": 0,
                "status": "ok",
            }
            first._append_journal(metric)
            first._close_resources()

            resumed = CacheGridRunner(
                0,
                _WordTokenizer(),
                cases,
                tmp,
                measure_runs=1,
                cache_commit_tail_tokens=4,
            )
            try:
                self.assertEqual(
                    resumed._results[resumed.case_key(cases[0])]["status"], "ok"
                )
                progress = resumed._progress()
                self.assertEqual(progress["completed_cases"], 1)
                self.assertEqual(progress["next_pending_case"]["input_len"], 16)
            finally:
                resumed._close_resources()

    @patch("rtp_llm.test.perf_test.cache_grid.runner.cache_grid_runner._post_prefill")
    def test_keyboard_interrupt_flushes_resumable_next_case(self, post):
        post.side_effect = KeyboardInterrupt()
        case = {
            "case_id": 7,
            "batch_size": 1,
            "input_len": 16,
            "cache_len": 0,
        }
        with tempfile.TemporaryDirectory() as tmp:
            runner = CacheGridRunner(
                12345,
                _WordTokenizer(),
                [case],
                tmp,
                measure_runs=1,
                cache_commit_tail_tokens=4,
            )
            with self.assertRaises(KeyboardInterrupt):
                runner.run()
            result = json.loads(
                (Path(tmp) / "cache_grid_results.json").read_text(encoding="utf-8")
            )
        self.assertFalse(result["complete"])
        self.assertEqual(result["progress"]["completed_cases"], 0)
        self.assertEqual(
            result["progress"]["next_pending_case"],
            {
                "case_key": "bs1_seq16_cache0",
                "case_id": 7,
                "batch_size": 1,
                "input_len": 16,
                "cache_len": 0,
            },
        )

    def test_save_includes_profile_and_measure_runs(self):
        import json

        with tempfile.TemporaryDirectory() as tmp:
            profile = {"schema_version": 1, "label": "test"}
            cases = [
                {"case_id": 0, "batch_size": 1, "input_len": 1024, "cache_len": 0},
                {"case_id": 1, "batch_size": 1, "input_len": 2048, "cache_len": 512},
            ]
            run_config = {"model": {"checkpoint_path": "/weights/model"}}
            runner = CacheGridRunner(
                0,
                _WordTokenizer(),
                cases,
                tmp,
                measure_runs=5,
                expected_block_size=256,
                profile=profile,
                profile_sha256="fp123",
                run_config=run_config,
            )
            runner._results[runner.case_key(cases[0])] = {
                "case_key": runner.case_key(cases[0]),
                "status": "ok",
            }
            runner._save(complete=False)
            path = Path(tmp) / "cache_grid_results.json"
            payload = json.loads(path.read_text(encoding="utf-8"))
            self.assertEqual(payload["measure_runs"], 5)
            self.assertEqual(payload["expected_block_size"], 256)
            self.assertEqual(payload["profile"], profile)
            self.assertEqual(payload["profile_sha256"], "fp123")
            self.assertEqual(payload["schema_version"], 2)
            self.assertEqual(payload["run_config"], run_config)
            self.assertEqual(
                payload["run_config_sha256"], resume_config_fingerprint(run_config)
            )
            self.assertEqual(payload["progress"]["completed_cases"], 1)
            self.assertEqual(
                payload["progress"]["last_completed_case"]["input_len"], 1024
            )
            self.assertEqual(
                payload["progress"]["next_pending_case"]["input_len"], 2048
            )
            self.assertEqual(payload["progress"]["next_pending_case"]["cache_len"], 512)


class MaterializedCaseStoreProfileTest(unittest.TestCase):
    def test_materialize_persists_profile_sha256(self):
        import json

        with tempfile.TemporaryDirectory() as tmp:
            store = MaterializedCaseStore(tmp)
            factory = PrefixPromptFactory(_WordTokenizer())
            cases = [
                {"case_id": 0, "batch_size": 1, "input_len": 1024, "cache_len": 512}
            ]
            store.materialize(
                cases, factory, run_count=3, profile_sha256="sha_profile_42"
            )
            info = json.loads(
                (Path(tmp) / "store_info.json").read_text(encoding="utf-8")
            )
            self.assertEqual(info["profile_sha256"], "sha_profile_42")
            self.assertEqual(store.profile_sha256, "sha_profile_42")

    def test_load_reads_profile_sha256(self):
        import json

        with tempfile.TemporaryDirectory() as tmp:
            store1 = MaterializedCaseStore(tmp)
            factory = PrefixPromptFactory(_WordTokenizer())
            cases = [
                {"case_id": 0, "batch_size": 1, "input_len": 1024, "cache_len": 512}
            ]
            store1.materialize(
                cases, factory, run_count=3, profile_sha256="persisted_sha"
            )
            store2 = MaterializedCaseStore(tmp)
            store2.load_cases()
            self.assertEqual(store2.profile_sha256, "persisted_sha")


class SharedSeedTest(unittest.TestCase):
    def runner(self, directory, **kwargs):
        return CacheGridRunner(
            8000,
            _WordTokenizer(),
            [
                {"case_id": i, "batch_size": 1, "input_len": 64, "cache_len": c}
                for i, c in enumerate([0, 16, 32])
            ],
            directory,
            request_transport="dashsc_input_ids",
            shared_seed=True,
            expected_block_size=8,
            cache_commit_tail_tokens=8,
            **kwargs,
        )

    def test_shared_seed_one_request_and_exact_isolated_prefixes(self):
        with tempfile.TemporaryDirectory() as directory:
            runner = self.runner(directory)
            self.addCleanup(runner._close_resources)
            seen = []

            def post(prompt, ids, request_id):
                seen.append((list(ids), request_id))
                # Simulate block-prefix matching against all prior requests.
                reuse = 0
                for old, _ in seen[:-1]:
                    common = 0
                    for a, b in zip(old, ids):
                        if a != b:
                            break
                        common += 1
                    reuse = max(reuse, (common // 8) * 8)
                return {
                    "prefill_time_ms": 1,
                    "success": True,
                    "input_len": len(ids),
                    "output_len": 1,
                    "reuse_len": reuse,
                    "ttft_ms": 1,
                }

            with patch.object(runner, "_post_request", side_effect=post), patch.object(
                runner, "_probe_reuse_granularity"
            ):
                metrics = runner.run()
            self.assertTrue(all(m["status"] == "ok" for m in metrics))
            self.assertEqual(len(seen), 10)  # one shared seed + nine formal calls
            self.assertEqual(len(seen[0][0]), 40)
            self.assertTrue(seen[1][1].startswith("bs1_seq64_cache32"))
            self.assertTrue(seen[4][1].startswith("bs1_seq64_cache16"))
            self.assertTrue(seen[7][1].startswith("bs1_seq64_cache0"))
            by_case_id = {m["case_id"]: m for m in metrics}
            self.assertEqual(by_case_id[1]["cache_len_observed"], [16] * 3)
            self.assertEqual(by_case_id[2]["cache_len_observed"], [32] * 3)
            saved = json.loads(
                (Path(directory) / "cache_grid_results.json").read_text()
            )
            self.assertEqual(saved["seed_mode"], "shared_prefix_v1")
            self.assertTrue(saved["complete"])

    def test_shared_seed_orders_hits_by_cache_then_input_and_cold_last(self):
        cases = [
            {"case_id": 0, "input_len": 64, "cache_len": 0},
            {"case_id": 1, "input_len": 96, "cache_len": 32},
            {"case_id": 2, "input_len": 64, "cache_len": 16},
            {"case_id": 3, "input_len": 64, "cache_len": 32},
            {"case_id": 4, "input_len": 32, "cache_len": 0},
        ]
        ordered = sorted(cases, key=CacheGridRunner._shared_seed_case_order)
        self.assertEqual([case["case_id"] for case in ordered], [3, 1, 2, 4, 0])

    def test_shared_seed_preserves_refinement_level_within_cache(self):
        cases = [
            {
                "case_id": 0,
                "input_len": 64,
                "cache_len": 32,
                "refinement_level": 1,
            },
            {
                "case_id": 1,
                "input_len": 96,
                "cache_len": 32,
                "refinement_level": 0,
            },
        ]
        ordered = sorted(cases, key=CacheGridRunner._shared_seed_case_order)
        self.assertEqual([case["case_id"] for case in ordered], [1, 0])

    def test_independent_seed_preserves_declared_order(self):
        cases = [
            {"case_id": 0, "batch_size": 1, "input_len": 64, "cache_len": 0},
            {"case_id": 1, "batch_size": 1, "input_len": 64, "cache_len": 32},
        ]
        with tempfile.TemporaryDirectory() as directory:
            runner = CacheGridRunner(8000, _WordTokenizer(), cases, directory)
            self.addCleanup(runner._close_resources)
            self.assertEqual([case["case_id"] for case in runner.cases], [0, 1])

    def test_evicted_shared_prefix_fails_validation(self):
        with tempfile.TemporaryDirectory() as directory:
            runner = self.runner(directory)
            self.addCleanup(runner._close_resources)
            with patch.object(runner, "_probe_reuse_granularity"), patch.object(
                runner,
                "_post_request",
                side_effect=lambda p, ids, r: {
                    "prefill_time_ms": 1,
                    "success": True,
                    "input_len": len(ids),
                    "output_len": 1,
                    "reuse_len": 0,
                    "ttft_ms": 1,
                },
            ):
                with self.assertRaisesRegex(RuntimeError, "invalid_reuse"):
                    runner.run()

    def test_shared_seed_refresh_excludes_miss_and_changes_suffix(self):
        with tempfile.TemporaryDirectory() as directory:
            runner = self.runner(directory)
            self.addCleanup(runner._close_resources)
            runner._shared_seed_ids = ["prefix"] * 40
            runner._shared_seed_record = {"seed_id": "test"}
            case = runner.cases[1]
            ids = runner._prepare_shared_payload(case)["run_ids"][0]

            def response(reuse, length=64):
                return {
                    "prefill_time_ms": 1,
                    "success": True,
                    "input_len": length,
                    "output_len": 1,
                    "reuse_len": reuse,
                    "ttft_ms": 1,
                }

            with patch.object(
                runner,
                "_post_request",
                side_effect=[response(8), response(8, 24), response(16)],
            ) as post:
                result = runner._post_shared_measurement(case, ids, "measure")
            self.assertEqual(result["reuse_len"], 16)
            self.assertEqual(
                post.call_args_list[1].args[1], runner._shared_seed_ids[:24]
            )
            self.assertEqual(len(runner._shared_seed_ids), 40)
            self.assertEqual(runner._shared_seed_refreshes[0]["refresh_input_len"], 24)
            retry_ids = post.call_args_list[2].args[1]
            self.assertEqual(ids[:16], retry_ids[:16])
            self.assertNotEqual(ids[16:], retry_ids[16:])
            self.assertEqual(
                runner._shared_seed_refreshes[0]["excluded_request"]["reuse_len"], 8
            )
            self.assertEqual(
                len(list(Path(directory).glob("shared_seed_refresh_*.json"))), 1
            )

    def test_shared_seed_refresh_shrinks_as_sorted_cases_progress(self):
        with tempfile.TemporaryDirectory() as directory:
            runner = self.runner(directory)
            self.addCleanup(runner._close_resources)
            refreshes = []
            original_seed = []

            def post(prompt, ids, request_id):
                if request_id.startswith("shared_seed:"):
                    original_seed.extend(ids)
                    reuse = 0
                elif request_id.endswith(":refresh_seed"):
                    refreshes.append(list(ids))
                    reuse = 0
                else:
                    cache_len = int(request_id.split(":")[0].rsplit("cache", 1)[1])
                    reuse = 0 if request_id.endswith(":run0") else cache_len
                return {
                    "prefill_time_ms": 1,
                    "success": True,
                    "input_len": len(ids),
                    "output_len": 1,
                    "reuse_len": reuse,
                    "ttft_ms": 1,
                }

            with patch.object(runner, "_post_request", side_effect=post), patch.object(
                runner, "_probe_reuse_granularity"
            ):
                metrics = runner.run()
            self.assertTrue(all(m["status"] == "ok" for m in metrics))
            self.assertEqual([len(ids) for ids in refreshes], [40, 24])
            for ids in refreshes:
                self.assertEqual(ids, original_seed[: len(ids)])
            self.assertEqual(runner._shared_seed_ids, original_seed)
            saved = json.loads(
                (Path(directory) / "cache_grid_results.json").read_text()
            )
            self.assertEqual(
                [
                    event["refresh_input_len"]
                    for event in saved["shared_seed_refreshes"]
                ],
                [40, 24],
            )

    def test_shared_seed_rejects_http(self):
        with tempfile.TemporaryDirectory() as directory:
            with self.assertRaisesRegex(ValueError, "dashsc_input_ids"):
                CacheGridRunner(8000, _WordTokenizer(), [], directory, shared_seed=True)

    def test_resume_cannot_mix_modes(self):
        with tempfile.TemporaryDirectory() as directory:
            runner = self.runner(directory)
            runner._save()
            runner._close_resources()
            with self.assertRaises(ValueError):
                CacheGridRunner(
                    8000,
                    _WordTokenizer(),
                    runner.cases,
                    directory,
                    request_transport="dashsc_input_ids",
                    expected_block_size=8,
                    cache_commit_tail_tokens=8,
                )


if __name__ == "__main__":
    unittest.main()
