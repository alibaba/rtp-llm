from __future__ import annotations

import argparse
import gzip
import io
import json
import os
import pathlib
import tempfile
import threading
import unittest
import urllib.error
from unittest import mock
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer

from example.k3.kimi_k3_full_model_pd_cases import (
    Case,
    Runner,
    SmokeFailure,
    main,
    numbered_answer_pattern,
    parse_args,
)


def make_args() -> argparse.Namespace:
    return argparse.Namespace(
        base_url="http://prefill:27188",
        decode_health_url="http://decode:29188/health",
        decode_role_addrs=[
            {
                "role": "DECODE",
                "ip": "10.0.0.2",
                "http_port": 28188 + rank * 9,
                "grpc_port": 28189 + rank * 9,
            }
            for rank in range(8)
        ],
        output=None,
        suite="all",
        namespace="unit-test",
        batch_size=4,
        block_size=4096,
        chunk_tokens=65536,
        max_tokens=128,
        identity_max_tokens=256,
        single_exact_max_tokens=128,
        mtp_chunk_max_tokens=128,
        require_mtp=True,
        rdma_prewarm_attempts=0,
        rdma_prewarm_backoff_s=0,
        rdma_prewarm_settle_s=0,
        rdma_prewarm_timeout=300,
        timeout=900,
        long_prefix_checkpoint=None,
        long_prefix_tp_size=8,
        long_prefix_target_tokens=600000,
        long_prefix_kernel_page_size=128,
        expanded_kv_budget_bytes=4294967296,
        validate_only=False,
    )


class KimiK3FullModelPdCasesTest(unittest.TestCase):


    def test_multimodal_image_is_in_native_runfiles_and_checked_before_launch(self):
        image = pathlib.Path(os.environ["TEST_SRCDIR"]) / os.environ["TEST_WORKSPACE"] / "rtp_llm/multimodal/test/testdata/qwen2_vl/1.jpg"
        self.assertTrue(image.is_file())
        self.assertEqual(image.read_bytes()[:2], b"\xff\xd8")
        args = make_args()
        args.validate_only = True
        module = "example.k3.kimi_k3_full_model_pd_cases"
        with mock.patch(module + ".parse_args", return_value=args), mock.patch(module + ".Runner") as runner:
            self.assertEqual(main(), 0)
            with mock.patch.object(pathlib.Path, "is_file", return_value=False):
                with self.assertRaisesRegex(SmokeFailure, "multimodal smoke image is missing"):
                    main()
            runner.assert_not_called()


    def test_long_prefix_failure_marks_the_entire_suite_failed(self) -> None:
        args = make_args()
        with tempfile.TemporaryDirectory() as tmp:
            args.output = pathlib.Path(tmp) / "accuracy.json"
            module = "example.k3.kimi_k3_full_model_pd_cases"
            with (
                mock.patch(module + ".parse_args", return_value=args),
                mock.patch.object(Runner, "run_stage"),
                mock.patch.object(Runner, "health", autospec=True),
                mock.patch(module + ".LongPrefixCase") as case_class,
            ):
                case_class.return_value.run.side_effect = ValueError(
                    "historical prefix too short"
                )
                case_class.return_value.records = [
                    {"name": "long_prefix_seed", "effective_reuse_len": 0}
                ]
                with self.assertRaisesRegex(ValueError, "historical prefix too short"):
                    main()
            saved = json.loads(args.output.read_text())
            self.assertFalse(saved["passed"])
            self.assertEqual(saved["stages"][-1]["name"], "long_prefix_cached_dialog")
            self.assertFalse(saved["stages"][-1]["passed"])
            self.assertEqual(saved["cases"][-1]["name"], "long_prefix_seed")

    def test_long_prefix_checks_both_services_before_and_after(self) -> None:
        args = make_args()
        with tempfile.TemporaryDirectory() as tmp:
            args.output = pathlib.Path(tmp) / "accuracy.json"
            runner = Runner(args)
            opener = mock.MagicMock()
            opener.open.return_value.__enter__.return_value.status = 200
            runner.opener = opener
            with mock.patch(
                "example.k3.kimi_k3_full_model_pd_cases.LongPrefixCase"
            ) as case_class:
                case = case_class.return_value
                case.records = [{"name": "long_prefix_followup"}]
                case.output = pathlib.Path(tmp) / "long-prefix"
                case.run.return_value = {
                    "cases": case.records,
                    "planned_prefix_blocks": 146,
                }
                runner.run_long_prefix_case()
            self.assertEqual(
                [call.args[0].full_url for call in opener.open.call_args_list],
                [runner.health_endpoint, runner.decode_health_endpoint] * 2,
            )
            self.assertTrue(runner.stages[-1]["passed"])
            self.assertEqual(runner.records, case.records)

    def test_request_pins_the_selected_decode_dp_owner(self) -> None:
        runner = Runner(make_args())
        response = mock.MagicMock()
        response.__enter__.return_value = response
        response.status = 200
        response.read.return_value = b"{}"
        runner.opener.open = mock.Mock(return_value=response)
        case = Case("owner-3", "prompt", r".", "miss", decode_owner_rank=3)

        with mock.patch.object(runner, "validate", return_value={"name": case.name}):
            runner.request(case)

        request = runner.opener.open.call_args.args[0]
        payload = json.loads(request.data)
        self.assertEqual(
            payload["extra_configs"]["role_addrs"],
            [
                {
                    "role": "DECODE",
                    "ip": "10.0.0.2",
                    "http_port": 28215,
                    "grpc_port": 28216,
                }
            ],
        )
        self.assertNotIn("role_addrs", payload)

    def test_request_rejects_decode_owner_outside_world(self) -> None:
        runner = Runner(make_args())
        case = Case("bad-owner", "prompt", r".", "miss", decode_owner_rank=8)
        with self.assertRaisesRegex(SmokeFailure, "outside the configured world size"):
            runner.request(case)

    def test_rdma_prewarm_retries_then_fills_batch_sized_pool(self) -> None:
        args = make_args()
        args.rdma_prewarm_attempts = 2
        runner = Runner(args)
        records = [
            {
                "name": f"rdma_prewarm_2_{idx}",
                "input_len": 100 + idx,
            }
            for idx in range(args.batch_size)
        ]

        with (
            mock.patch.object(runner, "health") as health,
            mock.patch.object(
                runner,
                "request_cases",
                side_effect=[SmokeFailure("first connect failed"), records],
            ) as request_cases,
        ):
            runner.prewarm_rdma_pool()

        self.assertEqual(health.call_count, 2)
        self.assertEqual(request_cases.call_count, 2)
        self.assertEqual(
            [attempt["passed"] for attempt in runner.rdma_prewarm_attempts],
            [False, True],
        )
        successful = runner.rdma_prewarm_attempts[-1]
        self.assertEqual(len(successful["case_names"]), args.batch_size)
        for call in request_cases.call_args_list:
            cases = call.args[0]
            self.assertEqual(len(cases), args.batch_size)
            self.assertTrue(all(case.timeout_s == 300 for case in cases))

    def test_rdma_prewarm_exhaustion_is_a_smoke_failure(self) -> None:
        args = make_args()
        args.rdma_prewarm_attempts = 2
        runner = Runner(args)
        with (
            mock.patch.object(runner, "health"),
            mock.patch.object(
                runner,
                "request_cases",
                side_effect=SmokeFailure("connect failed"),
            ),
        ):
            with self.assertRaisesRegex(SmokeFailure, "failed after 2 attempts"):
                runner.prewarm_rdma_pool()

    def test_tp_only_owner_count_must_be_explicit_and_match(self) -> None:
        argv = ["cases", "--base-url", "http://prefill:30188",
                "--decode-health-url", "http://decode:31188/health",
                "--output", "/tmp/unused-cases.json", "--namespace", "unit",
                "--decode-role-addr", "decode:31188:31189"]
        with mock.patch("sys.argv", argv + ["--decode-dp-size", "1"]):
            self.assertEqual(parse_args().decode_dp_size, 1)
        for suffix in ([], ["--decode-dp-size", "2"], ["--decode-dp-size", "0"]):
            with mock.patch("sys.argv", argv + suffix), mock.patch("sys.stderr"):
                with self.assertRaises(SystemExit):
                    parse_args()

    def test_single_decode_owner_preserves_all_cases_and_concurrency(self) -> None:
        args = make_args()
        args.decode_role_addrs = args.decode_role_addrs[:1]
        args.rdma_prewarm_attempts = 1
        runner = Runner(args)
        stages = {}
        with (
            mock.patch.object(runner, "health"),
            mock.patch.object(runner, "request_cases", return_value=[]) as requests,
            mock.patch.object(runner, "run_stage", side_effect=lambda name, cases, **kw: stages.update({name: cases})),
            mock.patch.object(runner, "run_long_prefix_case"),
            mock.patch("time.sleep"),
        ):
            runner.run_all()
        self.assertEqual(len(requests.call_args.args[0]), args.batch_size)
        self.assertTrue(all(case.decode_owner_rank == 0 for case in requests.call_args.args[0]))
        self.assertIn("single_owner_concurrent_batch", stages)
        self.assertEqual(len(stages["single_owner_concurrent_batch"]), 10)
        self.assertEqual(len(stages["cuda_graph_bucket_8"]), 8)
        self.assertTrue(all(case.decode_owner_rank == 0 for cases in stages.values() for case in cases))
        for name in ("batch_all_miss", "batch_all_hit", "whole_chunk_batch_miss", "whole_chunk_batch_hit", "multimodal_mtp_chunk_prefill_miss"):
            self.assertIn(name, stages)

    def test_all_suite_defines_dedicated_semantic_and_mtp_budgets(self) -> None:
        runner = Runner(make_args())
        stages: dict[str, list[Case]] = {}

        def capture_stage(
            name: str,
            cases: list[Case],
            concurrent: bool = False,
        ) -> None:
            del concurrent
            stages[name] = cases

        with (
            mock.patch.object(runner, "run_stage", side_effect=capture_stage),
            mock.patch.object(runner, "run_long_prefix_case") as long_prefix,
        ):
            runner.run_all()
        long_prefix.assert_called_once_with()

        identity = stages["identity_miss"][0]
        self.assertEqual(identity.max_tokens, 256)

        single_exact_seed = stages["single_exact_seed"][0]
        single_exact_hit = stages["single_exact_hit"][0]
        self.assertEqual(single_exact_seed.max_tokens, 128)
        self.assertEqual(single_exact_hit.max_tokens, 128)

        mtp_chunk = stages["mtp_chunk_prefill_miss"][0]
        self.assertEqual(mtp_chunk.max_tokens, 128)
        self.assertTrue(mtp_chunk.require_chunk)
        self.assertTrue(mtp_chunk.require_mtp)
        self.assertGreater(len(mtp_chunk.prompt), runner.args.chunk_tokens)

        multimodal_chunk = stages["multimodal_mtp_chunk_prefill_miss"][0]
        self.assertEqual(multimodal_chunk.max_tokens, 256)
        self.assertTrue(multimodal_chunk.require_chunk)
        self.assertTrue(multimodal_chunk.require_mtp)
        self.assertTrue(multimodal_chunk.require_multimodal)
        self.assertIsInstance(multimodal_chunk.prompt, list)
        self.assertEqual(
            [part["type"] for part in multimodal_chunk.prompt],
            ["text", "image_url", "text"],
        )
        image_path = multimodal_chunk.prompt[1]["image_url"]["url"]
        self.assertTrue(pathlib.Path(image_path).is_file())

        for stage_name in (
            "partial_prefix_seed",
            "partial_prefix_hit",
            "batch_all_miss",
            "batch_all_hit",
            "batch_mixed_hit_miss",
            "whole_chunk_single_miss",
            "whole_chunk_batch_miss",
        ):
            self.assertTrue(
                all(case.max_tokens is None for case in stages[stage_name]),
                stage_name,
            )
        self.assertEqual(runner.args.max_tokens, 128)
        self.assertEqual(
            [case.decode_owner_rank for case in stages["dp_uneven_local_batch"]],
            [0, 0, 0, 0, 1, 1, 1, 2, 2, 3],
        )
        self.assertEqual(
            [case.decode_owner_rank for case in stages["cuda_graph_bucket_8"]],
            [0] * 8,
        )
        self.assertEqual(
            [case.decode_owner_rank for case in stages["whole_chunk_batch_miss"]],
            [0, 1],
        )

    def test_tp_all_suite_routes_every_request_to_its_single_owner(self) -> None:
        args = make_args()
        args.decode_role_addrs = args.decode_role_addrs[:1]
        args.rdma_prewarm_attempts = 1
        runner = Runner(args)
        stages = {}
        with (
            mock.patch.object(runner, "health"),
            mock.patch.object(runner, "request_cases", return_value=[]) as requests,
            mock.patch.object(runner, "run_stage", side_effect=lambda name, cases, **kw: stages.update({name: cases})),
            mock.patch.object(runner, "run_long_prefix_case"),
        ):
            runner.run_all()
        self.assertNotIn("dp_uneven_local_batch", stages)
        self.assertIn("multimodal_mtp_chunk_prefill_miss", stages)
        self.assertEqual(len(stages["cuda_graph_bucket_8"]), 8)
        cases = requests.call_args.args[0] + [case for stage in stages.values() for case in stage]
        self.assertTrue(all(case.decode_owner_rank == 0 for case in cases))

    def test_mtp_chunk_case_requires_an_accepted_draft_token(self) -> None:
        runner = Runner(make_args())
        case = Case(
            "mtp_chunk_prefill_miss",
            "prompt",
            numbered_answer_pattern(5329),
            "miss",
            require_chunk=True,
            require_mtp=True,
            max_tokens=128,
        )
        response = {
            "choices": [
                {
                    "message": {
                        "content": "5329",
                        "reasoning_content": "73 squared is 5329",
                    }
                }
            ],
            "aux_info": {
                "pd_sep": True,
                "input_len": 65537,
                "output_len": 12,
                "iter_count": 9,
                "reuse_len": 0,
                "prefill_total_reuse_len": 0,
                "multimodal_lengths": {0: 576},
                "role_addrs": [runner.decode_role_addrs[0]],
            },
            "debug_info": {"output_ids": [list(range(12))]},
        }

        record = runner.validate(case, response, 1.0, 128)
        self.assertEqual(record["mtp_accepted_tokens"], 3)
        self.assertEqual(record["max_tokens"], 128)

        response["aux_info"]["iter_count"] = 12
        with self.assertRaisesRegex(SmokeFailure, "no accepted draft token"):
            runner.validate(case, response, 1.0, 128)

    def test_multimodal_chunk_case_requires_processed_input_url(self) -> None:
        runner = Runner(make_args())
        case = Case(
            "multimodal_mtp_chunk_prefill_miss",
            [{"type": "image_url", "image_url": {"url": "image.jpg"}}],
            numbered_answer_pattern(6241),
            "miss",
            require_chunk=True,
            require_mtp=True,
            require_multimodal=True,
            max_tokens=128,
        )
        response = {
            "choices": [
                {
                    "message": {
                        "content": "6241",
                        "reasoning_content": "",
                    }
                }
            ],
            "aux_info": {
                "pd_sep": True,
                "input_len": 70000,
                "output_len": 12,
                "iter_count": 9,
                "reuse_len": 0,
                "prefill_total_reuse_len": 0,
                "multimodal_lengths": {},
                "role_addrs": [runner.decode_role_addrs[0]],
            },
            "debug_info": {"output_ids": [list(range(12))], "input_urls": []},
        }

        with self.assertRaisesRegex(SmokeFailure, "no processed multimodal input URL"):
            runner.validate(case, response, 1.0, 128)

        response["debug_info"]["input_urls"] = ["image.jpg"]
        record = runner.validate(case, response, 1.0, 128)
        self.assertEqual(record["input_urls"], ["image.jpg"])
        self.assertTrue(record["require_multimodal"])

    @staticmethod
    def response(*, input_len=128, content="1369", output_len=3):
        return {
            "choices": [{"message": {"content": content}, "finish_reason": "stop"}],
            "aux_info": {
                "pd_sep": True,
                "input_len": input_len,
                "output_len": output_len,
                "iter_count": 2,
                "prefill_total_reuse_len": 0,
            },
            "debug_info": {"output_ids": [list(range(output_len))]},
        }

    def test_integer_oracle_preserves_value_and_rejects_numeric_substrings(self):
        import re

        pattern = numbered_answer_pattern(1144900)
        for valid in ("1144900", "**1,144,900**", "Answer: 1,144,900."):
            self.assertIsNotNone(re.search(pattern, valid))
        for invalid in (
            "1144901",
            "-1144900",
            "+1144900",
            "1144900.5",
            "0.1144900",
            "11,44,900",
        ):
            self.assertIsNone(re.search(pattern, invalid))

    def test_request_checks_output_ids_and_truncation(self):
        runner = Runner(make_args())
        case = Case(
            "exact",
            "prompt",
            numbered_answer_pattern(1369),
            "miss",
        )
        response = self.response(input_len=128)
        response["debug_info"]["output_ids"] = [[1]]
        with self.assertRaisesRegex(SmokeFailure, "IDs disagree"):
            runner.validate(case, response, 1, 512)
        response["debug_info"]["output_ids"] = [[1, 2, 3]]
        response["choices"][0]["finish_reason"] = "length"
        with self.assertRaisesRegex(SmokeFailure, "truncated"):
            runner.validate(case, response, 1, 512)


    def test_failed_http_response_is_preserved_before_validation(self):
        with tempfile.TemporaryDirectory() as tmp:
            args = make_args()
            args.output = pathlib.Path(tmp) / "accuracy.json"
            runner = Runner(args)
            body = b"upstream returned malformed data"
            error = urllib.error.HTTPError(
                runner.endpoint, 500, "error", {}, io.BytesIO(body)
            )
            with mock.patch.object(runner.opener, "open", side_effect=error):
                with self.assertRaises(SmokeFailure):
                    runner.request(Case("http-error", "p", "1369", "miss"))
            with gzip.open(
                pathlib.Path(tmp) / "requests/http-error-response.json.gz", "rb"
            ) as stream:
                self.assertEqual(stream.read(), body)

    def test_cp_reuse_builder_and_partial_prefix_are_token_validated(self):
        args = make_args()
        with mock.patch.dict("os.environ", {"PREFILL_CP_KV_CACHE_SHARDED": "1"}):
            runner = Runner(args)
        self.assertEqual(runner.reuse_alignment, 32768)

        def build(_namespace, name, *, repeats):
            return "x" * repeats + name

        with mock.patch.object(
            runner, "render_input_ids", side_effect=lambda text: list(text)
        ):
            first = runner.make_reusable_prompt(build, "n", "first", repeats=10000)
            second = runner.make_reusable_prompt(build, "n", "second", repeats=10000)
            self.assertGreaterEqual(len(first), 32768)
            runner.check_reuse_prefix(first, second)
            with self.assertRaisesRegex(SmokeFailure, "partial seed"):
                runner.check_reuse_prefix("bad", second)


if __name__ == "__main__":
    unittest.main()
