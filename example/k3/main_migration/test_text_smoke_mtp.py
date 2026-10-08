"""Four-layer flow coverage for the native MTP draft path."""

import types
import unittest
from unittest.mock import patch

from example.k3.main_migration.text_smoke import Case, Runner, SmokeFailure, parse_args


def runner(*, require_mtp=True):
    args = types.SimpleNamespace(
        base_url="http://127.0.0.1:1",
        decode_health_url="http://127.0.0.1:2/health",
        decode_role_addrs=[],
        namespace="four-layer-mtp-test",
        block_size=4096,
        chunk_tokens=65536,
        max_tokens=32,
        suite="flow",
        require_mtp=require_mtp,
    )
    return Runner(args)


class FourLayerMtpFlowTest(unittest.TestCase):
    def test_flow_has_five_minute_case_deadline(self):
        with patch(
            "sys.argv",
            [
                "text_smoke.py", "--base-url", "http://127.0.0.1:1",
                "--decode-health-url", "http://127.0.0.1:2/health",
                "--output", "/tmp/unused-k3-flow.json", "--suite", "flow",
                "--namespace", "deadline-test", "--block-size", "4096",
                "--long-prefix-checkpoint", "/tmp/unused-k3-checkpoint",
            ],
        ):
            self.assertEqual(parse_args().case_deadline_s, 300)

    def test_flow_includes_short_draft_coverage_when_requested(self):
        smoke = runner()
        stages = []
        smoke.fit_prompt = lambda head, tail, target: ("prompt", [1] * target)
        smoke.run_stage = lambda name, cases, concurrent=False: stages.append(
            (name, cases)
        )

        smoke.run_flow()

        draft_cases = [
            case
            for _, cases in stages
            for case in cases
            if getattr(case, "require_mtp_draft", False)
        ]
        self.assertEqual(len(draft_cases), 1)
        self.assertLessEqual(draft_cases[0].max_tokens, 32)

    def test_draft_case_requires_observed_draft_rounds(self):
        smoke = runner()
        case = Case(
            "four_layer_draft", "prompt", r".", "miss", require_mtp_draft=True
        )
        response = {
            "choices": [{"message": {"content": "x", "reasoning_content": ""}}],
            "aux_info": {
                "pd_sep": True,
                "output_len": 4,
                "iter_count": 4,
                "input_len": 16,
                "reuse_len": 0,
                "speculative_draft_rounds": 0,
            },
            "debug_info": {"output_ids": [[1, 2, 3, 4]]},
        }

        with self.assertRaisesRegex(SmokeFailure, "draft rounds"):
            smoke.validate(case, response, 0.1, 32)

        response["aux_info"]["speculative_draft_rounds"] = 3
        record = smoke.validate(case, response, 0.1, 32)
        self.assertEqual(record["mtp_draft_rounds"], 3)


if __name__ == "__main__":
    unittest.main()
