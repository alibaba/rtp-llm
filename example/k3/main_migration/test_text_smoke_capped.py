"""Short Decode page coverage in the time-capped full-model smoke."""

import types
import unittest
from unittest.mock import patch

from example.k3.main_migration.text_smoke import Case, Runner, SmokeFailure


def formal_response_fixture():
    args = types.SimpleNamespace(
        base_url="http://127.0.0.1:1",
        decode_health_url="http://127.0.0.1:2/health",
        decode_role_addrs=[],
        namespace="response-integrity-test",
        block_size=4096,
        chunk_tokens=65536,
        max_tokens=32,
        suite="main-text-64k-capped",
    )
    case = Case(
        "short-exact-answer",
        "只输出 JSON",
        "",
        "miss",
        expected_json={"value": "BND-4096"},
    )
    response = {
        "choices": [
            {
                "finish_reason": "stop",
                "message": {
                    "content": '{"value":"BND-4096"}',
                    "reasoning_content": "",
                },
            }
        ],
        "aux_info": {
            "pd_sep": True,
            "output_len": 16,
            "iter_count": 16,
            "input_len": 1,
            "reuse_len": 0,
        },
        "debug_info": {"output_ids": [[1] * 16]},
    }
    return Runner(args), case, response


class CappedDecodePageTest(unittest.TestCase):
    def test_formal_response_rejects_length_finish_reason(self):
        smoke, case, response = formal_response_fixture()
        response["choices"][0]["finish_reason"] = "length"
        with self.assertRaisesRegex(SmokeFailure, "finish_reason=length"):
            smoke.validate(case, response, 0.1, 32)

    def test_response_rejects_unicode_replacement_in_reasoning(self):
        smoke, case, response = formal_response_fixture()
        response["choices"][0]["message"]["reasoning_content"] = "\ufffd"
        with self.assertRaisesRegex(SmokeFailure, "Unicode replacement"):
            smoke.validate(case, response, 0.1, 32)

    def test_malformed_choice_still_raises_smoke_failure(self):
        smoke, case, response = formal_response_fixture()
        response["choices"] = [[]]
        with self.assertRaisesRegex(SmokeFailure, "malformed response"):
            smoke.validate(case, response, 0.1, 32)

    def test_formal_result_records_finish_reason(self):
        smoke, case, response = formal_response_fixture()
        result = smoke.validate(case, response, 0.1, 32)
        self.assertEqual(result["finish_reason"], "stop")

    def test_four_layer_flow_accepts_bounded_length_output(self):
        smoke, case, response = formal_response_fixture()
        smoke.args.suite = "flow"
        response["choices"][0]["finish_reason"] = "length"
        self.assertEqual(
            smoke.validate(case, response, 0.1, 32)["finish_reason"], "length"
        )

    def test_skipped_long_cases_have_short_exact_answer_replacements(self):
        args = types.SimpleNamespace(
            base_url="http://127.0.0.1:1",
            decode_health_url="http://127.0.0.1:2/health",
            decode_role_addrs=[],
            namespace="short-decode-test",
            block_size=4096,
            chunk_tokens=65536,
            max_tokens=32,
            suite="main-text-64k-capped",
        )
        smoke = Runner(args)
        stages = []
        smoke.fit_prompt = lambda head, tail, target: (head + tail, [1] * target)
        smoke.run_stage = lambda name, cases, concurrent=False: stages.append(
            (name, cases)
        )

        with patch(
            "example.k3.main_migration.text_smoke.cache_block_boundaries",
            return_value=(),
        ):
            smoke.run_cache_block_boundaries()

        short = [case for _, cases in stages for case in cases]
        self.assertEqual(len(short), 6)
        self.assertEqual(len(smoke.skipped_cases), 6)
        self.assertEqual(
            [
                (
                    case.expected_input_len,
                    case.decode_crossings,
                    case.reuse,
                    case.expected_reuse_len,
                )
                for case in short
            ],
            [
                (4095, (4096,), "miss", 0),
                (4095, (4096,), "miss", 0),
                (8191, (8192,), "miss", 0),
                (8191, (8192,), "hit", 4096),
                (65535, (65536,), "miss", 0),
                (65535, (65536,), "hit", 61440),
            ],
        )
        self.assertTrue(all(case.expected_json is not None for case in short))
        self.assertTrue(all(3 <= case.max_tokens <= 256 for case in short))
        self.assertTrue(all(case.require_mtp_draft for case in short))

        case = short[0]
        response = {
            "choices": [
                {"message": {"content": '{"value":"BND-4096"}', "reasoning_content": ""}}
            ],
            "aux_info": {
                "pd_sep": True,
                "output_len": 16,
                "iter_count": 16,
                "input_len": 4095,
                "reuse_len": 0,
                "speculative_draft_rounds": 1,
            },
            "debug_info": {"output_ids": [[1] * 16]},
        }
        self.assertEqual(case.expected_json, {"value": "BND-4096"})
        self.assertEqual(
            smoke.validate(case, response, 0.1, case.max_tokens)["decode_crossings"],
            [4096],
        )
        response["aux_info"]["output_len"] = 2
        with self.assertRaisesRegex(SmokeFailure, "did not cross"):
            smoke.validate(case, response, 0.1, case.max_tokens)


if __name__ == "__main__":
    unittest.main()
