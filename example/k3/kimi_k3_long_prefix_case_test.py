from __future__ import annotations

import json
import pathlib
import tempfile
import unittest
from unittest import mock

from example.k3.kimi_k3_long_prefix_case import (
    EXPECTED,
    LongPrefixCase,
    check_answer,
    expanded_bytes_per_token,
    prefix_blocks,
)


class LongPrefixCaseTest(unittest.TestCase):
    def test_independent_history_answers_reject_another_prefix(self):
        expected = {key: f"{value}-003" if isinstance(value, str) else value
                    for key, value in EXPECTED.items()}
        with tempfile.TemporaryDirectory() as tmp:
            case = LongPrefixCase(
                "http://unused", pathlib.Path(tmp), "prefix-003",
                timeout=1, budget=4 << 30, page_size=4096,
                bytes_per_token=7680, target_tokens=65537, expected=expected,
            )
            with mock.patch.object(case, "tokenize", side_effect=lambda messages: list(range(len(messages[0]["content"])))):
                seed, _, _ = case.make_seed()
            for key in ("early", "middle", "late"):
                self.assertIn(f"{key} = {expected[key]}.", seed[0]["content"])
            check_answer(json.dumps(expected), case.expected)
            with self.assertRaises(ValueError):
                check_answer(json.dumps(EXPECTED), case.expected)
        check_answer(json.dumps(EXPECTED))

    def test_default_budget_forces_two_historical_blocks(self):
        self.assertEqual(
            prefix_blocks(
                598016, 600050, budget=4294967296, page_size=128, bytes_per_token=7680
            ),
            [{"start": 0, "tokens": 559232}, {"start": 559232, "tokens": 38784}],
        )

    def test_rejects_missing_coverage(self):
        for reuse, total, budget in [
            (0, 600050, 4294967296),
            (559232, 600050, 4294967296),
            (598016, 598016, 4294967296),
            (598017, 600050, 4294967296),
            (598016, 600050, 0),
            (598016, 600050, 8 * 1024**3),
        ]:
            with self.subTest(reuse=reuse, total=total, budget=budget):
                with self.assertRaises(ValueError):
                    prefix_blocks(
                        reuse, total, budget=budget, page_size=128, bytes_per_token=7680
                    )

    def test_token_size_uses_checkpoint_dimensions_and_tp(self):
        self.assertEqual(expanded_bytes_per_token(None, 8), 7680)
        self.assertEqual(expanded_bytes_per_token(None, 1), 61440)
        with tempfile.TemporaryDirectory() as tmp:
            checkpoint = pathlib.Path(tmp)
            (checkpoint / "config.json").write_text(
                json.dumps({"num_attention_heads": 64})
            )
            self.assertEqual(expanded_bytes_per_token(checkpoint, 8), 5120)
            with self.assertRaises(ValueError):
                expanded_bytes_per_token(checkpoint, 3)

    def test_multimodal_checkpoint_uses_nested_text_dimensions(self):
        with tempfile.TemporaryDirectory() as tmp:
            checkpoint = pathlib.Path(tmp)
            (checkpoint / "config.json").write_text(json.dumps({
                "text_config": {"num_attention_heads": 64, "v_head_dim": 64},
            }))
            self.assertEqual(expanded_bytes_per_token(checkpoint, 8), 4096)

    def test_history_requests_keep_the_selected_semantic_output_budget(self):
        with tempfile.TemporaryDirectory() as tmp:
            result, sent = self.run_fake_service(tmp, max_tokens=4096)
            self.assertTrue(result["passed"])
            self.assertTrue(all(request["max_tokens"] == 4096 for request in sent))

    def test_answer_rejects_wrong_values_repetition_and_extra_keys(self):
        check_answer(json.dumps(EXPECTED))
        check_answer("```json\n" + json.dumps(EXPECTED) + "\n```")
        for text in [
            json.dumps({**EXPECTED, "square": 1370}),
            json.dumps({**EXPECTED, "square": 1369.0}),
            json.dumps(EXPECTED) * 2,
            json.dumps({**EXPECTED, "unexpected": "field"}),
            "no answer",
        ]:
            with self.subTest(text=text), self.assertRaises(ValueError):
                check_answer(text)

    def run_fake_service(
        self,
        directory,
        *,
        fault=None,
        target_tokens=600000,
        bytes_per_token=7680,
        max_tokens=256,
    ):
        case = LongPrefixCase(
            "http://prefill",
            pathlib.Path(directory) / "case",
            "test",
            timeout=900,
            budget=4294967296,
            page_size=4096,
            bytes_per_token=bytes_per_token,
            target_tokens=target_tokens,
            max_tokens=max_tokens,
        )
        seed = [{"role": "user", "content": "Original archive"}]
        ids = [1] * (target_tokens - 32)
        reuse = (target_tokens - 1000) // 4096 * 4096
        sent = []

        def post(route, payload):
            self.assertNotEqual(route, "start_profile")
            if route == "tokenize":
                return {
                    "token_ids": ([2] * len(ids) if fault == "prefix_changed" else ids)
                    + [3] * 82
                }
            sent.append(payload)
            is_seed = len(sent) == 1
            if not is_seed:
                self.assertEqual(payload["messages"][:1], seed)
                self.assertEqual(
                    payload["messages"][1], {"role": "assistant", "content": "RECEIVED"}
                )
                for value in EXPECTED.values():
                    self.assertNotIn(str(value), payload["messages"][-1]["content"])
            response = {
                "aux_info": {
                    "pd_sep": True,
                    "input_len": len(ids) if is_seed else len(ids) + 82,
                    "prefill_total_reuse_len": 0 if is_seed else reuse,
                },
                "choices": [
                    {
                        "finish_reason": "stop",
                        "message": {
                            "content": "RECEIVED" if is_seed else json.dumps(EXPECTED)
                        },
                    }
                ],
            }
            if fault == "seed_hit" and is_seed:
                response["aux_info"]["prefill_total_reuse_len"] = 4096
            if not is_seed:
                if fault == "wrong_answer":
                    response["choices"][0]["message"]["content"] = "{}"
                elif fault == "no_pd":
                    response["aux_info"]["pd_sep"] = False
                elif fault == "short_prefix":
                    response["aux_info"]["prefill_total_reuse_len"] = 4096
                elif fault == "physical_page_mismatch":
                    response["aux_info"]["prefill_total_reuse_len"] = reuse + 128
                elif fault == "truncated":
                    response["choices"][0]["finish_reason"] = "length"
                elif fault == "input_mismatch":
                    response["aux_info"]["input_len"] -= 3
            return response

        with (
            mock.patch.object(case, "make_seed", return_value=(seed, ids, [])),
            mock.patch.object(case, "post", side_effect=post),
        ):
            return case.run(), sent

    def test_seed_then_continuation_without_profiler(self):
        with tempfile.TemporaryDirectory() as tmp:
            result, sent = self.run_fake_service(tmp)
            self.assertTrue(result["passed"])
            self.assertEqual(len(sent), 2)
            self.assertEqual(result["new_tokens"], 2034)
            self.assertEqual(len(result["planned_prefix_blocks"]), 2)
            self.assertTrue(
                (pathlib.Path(tmp) / "case/long_prefix_hit-tokens.json.gz").exists()
            )

    def test_topology_aware_100k_target_still_spans_expansion_blocks(self):
        with tempfile.TemporaryDirectory() as tmp:
            result, sent = self.run_fake_service(
                tmp,
                target_tokens=100000,
                bytes_per_token=61440,
            )
            self.assertTrue(result["passed"])
            self.assertEqual(result["target_tokens"], 100000)
            self.assertEqual(result["seed_tokens"], 99968)
            self.assertEqual(len(result["planned_prefix_blocks"]), 2)
            self.assertEqual(len(sent), 2)

    def test_failures_are_fatal_and_preserve_evidence(self):
        for fault in [
            "seed_hit",
            "prefix_changed",
            "wrong_answer",
            "no_pd",
            "short_prefix",
            "physical_page_mismatch",
            "truncated",
            "input_mismatch",
        ]:
            with self.subTest(fault=fault), tempfile.TemporaryDirectory() as tmp:
                with self.assertRaises(ValueError):
                    self.run_fake_service(tmp, fault=fault)
                result = json.loads(
                    (pathlib.Path(tmp) / "case/RESULT.json").read_text()
                )
                self.assertFalse(result["passed"])
                self.assertIn("error", result)


if __name__ == "__main__":
    unittest.main()
