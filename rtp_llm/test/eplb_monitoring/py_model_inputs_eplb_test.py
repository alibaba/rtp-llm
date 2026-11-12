import unittest

import torch

from rtp_llm.ops.compute_ops import PyModelInputs


class PyModelInputsEplbTest(unittest.TestCase):
    def test_expert_token_counts_preserve_caller_storage(self) -> None:
        inputs = PyModelInputs()
        self.assertIsNone(inputs.expert_token_counts)
        expert_token_counts = torch.zeros((2, 4), dtype=torch.int32)
        inputs.expert_token_counts = expert_token_counts
        self.assertEqual(
            inputs.expert_token_counts.data_ptr(), expert_token_counts.data_ptr()
        )
        inputs.expert_token_counts.fill_(7)
        self.assertTrue(
            torch.equal(expert_token_counts, torch.full_like(expert_token_counts, 7))
        )
        disabled_inputs = PyModelInputs()
        self.assertIsNone(disabled_inputs.expert_token_counts)
        self.assertEqual(
            inputs.expert_token_counts.data_ptr(), expert_token_counts.data_ptr()
        )


if __name__ == "__main__":
    unittest.main()
