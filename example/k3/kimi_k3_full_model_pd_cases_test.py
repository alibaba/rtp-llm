from __future__ import annotations

import argparse
import json
import pathlib
import tempfile
import unittest
from unittest import mock

from example.k3.kimi_k3_full_model_pd_cases import (
    Case,
    Runner,
    SmokeFailure,
    TransportFailure,
    main,
    numbered_answer_pattern,
    parse_args,
    page_rr_boundaries,
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
        expanded_kv_budget_gib=4.0,
    )


class KimiK3FullModelPdCasesTest(unittest.TestCase):
    def test_trimmed_suite_keeps_boundaries_and_restores_multi_dp_cases(self) -> None:
        for dp in (1, 2):
            with self.subTest(dp=dp):
                args = make_args()
                args.block_size = 1024
                args.reuse_unit_tokens = 8192
                args.decode_role_addrs = args.decode_role_addrs[:dp]
                runner = Runner(args)
                stages = {}
                with (
                    mock.patch.object(runner, "prewarm_rdma_pool"),
                    mock.patch.object(runner, "fit_prompt", side_effect=lambda h, t, n: (h + t, [0] * n)),
                    mock.patch.object(runner, "tokenize", return_value=[0] * 16384 + [1] * 1024),
                    mock.patch.object(runner, "run_stage", side_effect=lambda n, c, **kw: stages.update({n: c})),
                    mock.patch.object(runner, "run_long_prefix_case") as long_prefix,
                ):
                    runner.run_all()
                long_prefix.assert_called_once_with()
                cases = [c for batch in stages.values() for c in batch]
                self.assertEqual(sum(c.page_rr_boundary is not None for c in cases), 86)
                self.assertEqual(sum(bool(c.decode_crossings) for c in cases), 20)
                for boundary in {c.page_rr_boundary for c in cases if c.page_rr_boundary}:
                    selected = [c for c in cases if c.page_rr_boundary == boundary]
                    self.assertEqual({c.decode_owner_rank for c in selected}, set(range(dp)))
                    for cold in [c for c in selected if c.page_rr_phase.endswith("cold")]:
                        repeat = next(c for c in selected if c.name == cold.name.replace("-cold", "-repeat"))
                        self.assertEqual(repeat.decode_owner_rank, (cold.decode_owner_rank + 1) % dp)
                for stage in ("whole_chunk_batch_miss", "whole_chunk_batch_hit"):
                    self.assertTrue(all(c.require_mtp for c in stages[stage]))
                    self.assertEqual({c.decode_owner_rank for c in stages[stage]}, set(range(dp)))
                self.assertEqual(len(stages["dp_rolling_refill"]), 16)
                self.assertTrue(stages["whole_chunk_single_hit"][0].require_mtp)
                self.assertEqual("dp_uneven_local_batch" in stages, dp > 1)
                self.assertEqual("owner_last_only" in stages, dp > 1)
                if dp == 1:
                    # LongPrefixCase separately contributes two requests in one stage.
                    self.assertEqual(len(cases) + 2, 153)
                    self.assertEqual(len(stages) + 1, 50)
                else:
                    for rotation in (0, 3):
                        self.assertEqual(
                            {c.decode_owner_rank for c in stages[f"owner_records_rotate_{rotation}"]},
                            {0, 1},
                        )
                    self.assertEqual(stages["owner_last_only"][0].decode_owner_rank, 1)
                    self.assertEqual(
                        [c.decode_owner_rank for c in stages["dp_uneven_local_batch"]],
                        [0, 0, 0, 0, 1, 1, 1, 0, 0, 1],
                    )

    def test_page_rr_matrix_covers_owners_wraps_and_chunk_edges(self) -> None:
        args = make_args()
        args.block_size = 128
        args.reuse_unit_tokens = 1024
        args.decode_role_addrs = args.decode_role_addrs[:1]
        runner = Runner(args)
        stages = {}
        with (
            mock.patch.object(runner, "fit_prompt", side_effect=lambda h, t, n: (h+t, [1]*n)),
            mock.patch.object(runner, "run_stage", side_effect=lambda name, cases, **kw: stages.update({name: cases})),
        ):
            runner.run_page_rr_boundaries()
        boundaries = (128,256,384,512,640,768,896,1024,2048,65536,131072)
        self.assertEqual(page_rr_boundaries(128,1024,65536), boundaries)
        for boundary in boundaries:
            cold = stages[f"page_rr_{boundary}_cold"]
            repeated = stages[f"page_rr_{boundary}_repeat"]
            self.assertEqual([c.expected_input_len for c in cold], [boundary-1,boundary,boundary+1])
            self.assertTrue(all(c.expected_reuse_len == 0 for c in cold))
            self.assertEqual([c.expected_reuse_len for c in repeated], [(n-1)//1024*1024 for n in (boundary-1,boundary,boundary+1)])
            self.assertTrue(all(c.reuse == ("hit" if c.expected_reuse_len else "miss") for c in repeated))
        cross = [c for name, cases in stages.items() if name.startswith("decode_page_cross") for c in cases]
        self.assertEqual(len(cross),20)
        self.assertEqual({c.page_rr_boundary for c in cross}, set(range(128,1025,128)) | {2048,65536})
        self.assertTrue(all(c.expected_input_len == c.page_rr_boundary-1 and c.decode_crossings == (c.page_rr_boundary,c.page_rr_boundary+128) for c in cross))
        self.assertEqual(sum(map(len, stages.values())),86)

    def test_block1024_boundaries_preserve_multi_dp_coverage(self) -> None:
        for dp in (1, 2):
            args = make_args()
            args.block_size = 1024
            args.reuse_unit_tokens = 8192
            args.decode_role_addrs = args.decode_role_addrs[:dp]
            runner = Runner(args)
            stages = {}
            with (
                mock.patch.object(runner, "fit_prompt", side_effect=lambda h,t,n: (h+t,[1]*n)),
                mock.patch.object(runner, "run_stage", side_effect=lambda name,cases,**kw: stages.update({name:cases})),
            ):
                runner.run_page_rr_boundaries()
            self.assertEqual(sum(map(len, stages.values())), 86)
            repeated = stages["page_rr_8192_repeat"]
            self.assertEqual([c.expected_reuse_len for c in repeated], [0, 0, 8192])
            crossings = [c for cases in stages.values() for c in cases if c.decode_crossings]
            self.assertEqual(len(crossings), 20)
            self.assertTrue(any(c.decode_crossings == (8192, 9216) for c in crossings))

    def test_page_rr_geometry_rejects_incomplete_owner_cycles(self) -> None:
        for geometry in ((0,1024,65536),(128,1000,65536),(128,1024,65535),(128,1024,512)):
            with self.subTest(geometry=geometry), self.assertRaises(ValueError):
                page_rr_boundaries(*geometry)

    def test_decode_crossing_excludes_final_unconsumed_output(self) -> None:
        runner = Runner(make_args())
        case = Case("decode-boundary", "prompt", "ok", "miss", expected_input_len=127, decode_crossings=(128,256))
        response = {
            "choices":[{"message":{"content":"ok","reasoning_content":""}}],
            "aux_info":{"pd_sep":True,"input_len":127,"output_len":130,"iter_count":130,"reuse_len":0,"role_addrs":[runner.decode_role_addrs[0]]},
            "debug_info":{"output_ids":[list(range(130))]},
        }
        with self.assertRaisesRegex(SmokeFailure, "did not cross committed KV boundary 256"):
            runner.validate(case,response,1.0,1024)
        response["aux_info"]["output_len"] = 131
        response["debug_info"]["output_ids"] = [list(range(131))]
        record=runner.validate(case,response,1.0,1024)
        self.assertEqual(record["decode_kv_last_position"],256)
        self.assertEqual(record["decode_crossings"],[128,256])

    def test_long_prefix_failure_marks_the_entire_suite_failed(self) -> None:
        args = make_args()
        with tempfile.TemporaryDirectory() as tmp:
            args.output = pathlib.Path(tmp) / "accuracy.json"
            module = "example.k3.kimi_k3_full_model_pd_cases"
            with (
                mock.patch.object(Runner, "run_prefix_branches"),
                mock.patch.object(Runner, "run_padding_boundaries"),
                mock.patch.object(Runner, "run_page_rr_boundaries"),
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
                self.assertEqual(case_class.call_args.kwargs["reuse_unit_tokens"], runner.reuse_unit_tokens)
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
                side_effect=[TransportFailure("first connect failed"), records],
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
                side_effect=TransportFailure("connect failed"),
            ),
        ):
            with self.assertRaisesRegex(SmokeFailure, "failed after 2 attempts"):
                runner.prewarm_rdma_pool()

    def test_tp_only_owner_count_must_be_explicit_and_match(self) -> None:
        argv = [
            "cases",
            "--base-url",
            "http://prefill:30188",
            "--decode-health-url",
            "http://decode:31188/health",
            "--output",
            "/tmp/unused-cases.json",
            "--namespace",
            "unit",
            "--decode-role-addr",
            "decode:31188:31189",
        ]
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
            mock.patch.object(Runner, "run_prefix_branches"),
            mock.patch.object(Runner, "run_padding_boundaries"),
            mock.patch.object(Runner, "run_page_rr_boundaries"),
            mock.patch.object(runner, "health"),
            mock.patch.object(runner, "request_cases", return_value=[]) as requests,
            mock.patch.object(
                runner,
                "run_stage",
                side_effect=lambda name, cases, **kw: stages.update({name: cases}),
            ),
            mock.patch.object(runner, "run_long_prefix_case"),
            mock.patch("time.sleep"),
        ):
            runner.run_all()
        self.assertEqual(len(requests.call_args.args[0]), args.batch_size)
        self.assertTrue(
            all(case.decode_owner_rank == 0 for case in requests.call_args.args[0])
        )
        for omitted in (
            "single_owner_concurrent_batch", "dp_uneven_local_batch",
            "owner_records_rotate_0", "owner_records_rotate_3", "owner_last_only",
            "identity_miss", "single_exact_seed", "single_exact_hit",
            "partial_prefix_seed", "partial_prefix_hit",
            "graph_slot_wave_0", "graph_slot_wave_1", "graph_slot_wave_2",
        ):
            self.assertNotIn(omitted, stages)
        self.assertEqual(len(stages["dp_rolling_refill"]), 16)
        self.assertEqual(len(stages["historical_four_squares"]), 4)
        self.assertEqual(len(stages["cuda_graph_bucket_8"]), 8)
        self.assertTrue(
            all(
                case.decode_owner_rank == 0
                for cases in stages.values()
                for case in cases
            )
        )
        for name in (
            "batch_all_miss",
            "batch_all_hit",
            "whole_chunk_batch_miss",
            "whole_chunk_batch_hit",
            "multimodal_mtp_chunk_prefill_miss",
        ):
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
            mock.patch.object(Runner, "run_prefix_branches"),
            mock.patch.object(Runner, "run_padding_boundaries"),
            mock.patch.object(Runner, "run_page_rr_boundaries"),
            mock.patch.object(runner, "run_stage", side_effect=capture_stage),
            mock.patch.object(runner, "run_long_prefix_case") as long_prefix,
        ):
            runner.run_all()
        long_prefix.assert_called_once_with()

        self.assertNotIn("mtp_chunk_prefill_miss", stages)
        mtp_chunk = stages["whole_chunk_single_miss"][0]
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
            "batch_all_miss",
            "batch_all_hit",
            "batch_mixed_hit_miss",
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
            "debug_info": {"output_ids": [[1, 2, 3]]},
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
            "debug_info": {"output_ids": [[1, 2, 3]], "input_urls": []},
        }

        with self.assertRaisesRegex(SmokeFailure, "no processed multimodal input URL"):
            runner.validate(case, response, 1.0, 128)

        response["debug_info"]["input_urls"] = ["image.jpg"]
        record = runner.validate(case, response, 1.0, 128)
        self.assertEqual(record["input_urls"], ["image.jpg"])
        self.assertTrue(record["require_multimodal"])


if __name__ == "__main__":
    unittest.main()
