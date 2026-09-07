from __future__ import annotations

import argparse
import pathlib
import unittest
from unittest import mock

from example.k3.kimi_k3_full_model_pd_cases import (
    Case,
    Runner,
    SmokeFailure,
    numbered_answer_pattern,
)


def make_args() -> argparse.Namespace:
    return argparse.Namespace(
        base_url="http://prefill:27188",
        decode_health_url="http://decode:29188/health",
        output=None,
        suite="all",
        namespace="unit-test",
        batch_size=4,
        block_size=4096,
        chunk_tokens=65536,
        max_tokens=256,
        identity_max_tokens=256,
        single_exact_max_tokens=128,
        mtp_chunk_max_tokens=128,
        rdma_prewarm_attempts=0,
        rdma_prewarm_backoff_s=0,
        rdma_prewarm_settle_s=0,
        timeout=900,
    )


class KimiK3FullModelPdCasesTest(unittest.TestCase):
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

        with mock.patch.object(runner, "run_stage", side_effect=capture_stage):
            runner.run_all()

        identity = stages["identity_miss"][0]
        self.assertEqual(identity.max_tokens, 256)

        single_exact_seed = stages["single_exact_seed"][0]
        single_exact_hit = stages["single_exact_hit"][0]
        self.assertEqual(single_exact_seed.max_tokens, 256)
        self.assertEqual(single_exact_hit.max_tokens, 256)

        mtp_chunk = stages["mtp_chunk_prefill_miss"][0]
        self.assertEqual(mtp_chunk.max_tokens, 256)
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
        self.assertEqual(runner.args.max_tokens, 256)

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
