import unittest

from rtp_llm.models_py.triton_kernels.common.activation import (
    MaskedSiluInputLayout,
    _heuristic_params,
)


class HeuristicParamsTest(unittest.TestCase):

    def test_skew_correction_and_baseline_boundaries(self):
        per_expert = MaskedSiluInputLayout.PER_EXPERT_CAPACITY
        batch_aligned = MaskedSiluInputLayout.BATCH_TOKEN_ALIGNMENT
        cases = [
            # name, expert_num, expected_m, padded tokens, layout, expected params
            ("small expert baseline", 8, 43, 512, per_expert, (32, 2, 1)),
            ("medium expert skew correction", 32, 43, 512, per_expert, (16, 3, 1)),
            ("production per-expert capacity", 96, 43, 512, per_expert, (16, 3, 1)),
            (
                "same dimensions from batch alignment",
                96,
                43,
                512,
                batch_aligned,
                (1, 2, 1),
            ),
            ("hybrid batch alignment", 96, 86, 1024, batch_aligned, (1, 4, 1)),
            ("relative threshold boundary", 64, 43, 344, per_expert, (1, 2, 1)),
            ("absolute threshold boundary", 64, 8, 100, per_expert, (1, 2, 1)),
            ("stage count at expected 64", 64, 64, 1024, per_expert, (16, 4, 1)),
            ("stage count at expected 100", 64, 100, 1024, per_expert, (16, 4, 1)),
            ("expected 128 baseline", 64, 128, 4096, per_expert, (8, 4, 1)),
            ("large expected load baseline", 96, 4096, 4096, per_expert, (8, 4, 1)),
        ]
        for (
            name,
            expert_num,
            expected_m,
            token_num_padded,
            input_layout,
            expected,
        ) in cases:
            with self.subTest(
                name=name,
                expert_num=expert_num,
                expected_m=expected_m,
                token_num_padded=token_num_padded,
                input_layout=input_layout,
            ):
                self.assertEqual(
                    _heuristic_params(
                        expert_num=expert_num,
                        expected_m=expected_m,
                        token_num_padded=token_num_padded,
                        input_layout=input_layout,
                    ),
                    expected,
                )


if __name__ == "__main__":
    unittest.main()
