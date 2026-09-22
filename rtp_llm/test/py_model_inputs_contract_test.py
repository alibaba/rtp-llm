"""Native binding contract; tensor inputs are CPU-only and need no model weights."""

import unittest

import torch

from rtp_llm.ops.compute_ops import (
    BertEmbeddingInputs,
    PyAttentionInputs,
    PyModelInputs,
)


class PyModelInputsContractTest(unittest.TestCase):
    def test_existing_constructors_preserve_all_rows_by_default(self):
        tokens = torch.tensor([1, 2], dtype=torch.int32)
        constructors = {
            "default": lambda: PyModelInputs(),
            "positional": lambda: PyModelInputs(
                tokens, torch.empty(0), PyAttentionInputs(), BertEmbeddingInputs()
            ),
            "keyword": lambda: PyModelInputs(input_ids=tokens),
        }
        for name, construct in constructors.items():
            with self.subTest(constructor=name):
                inputs = construct()
                self.assertIs(inputs.need_all_logits, True)
                self.assertIs(inputs.need_all_hidden_states, True)

    def test_requirements_are_independent_and_do_not_leak_between_inputs(self):
        inputs = PyModelInputs()
        other = PyModelInputs()
        # Reuse one object across calls to catch a sticky or shared flag.
        for logits, hidden in (
            (False, False),
            (True, False),
            (False, True),
            (True, True),
            (False, False),
        ):
            with self.subTest(logits=logits, hidden=hidden):
                inputs.need_all_logits = logits
                inputs.need_all_hidden_states = hidden
                self.assertIs(inputs.need_all_logits, logits)
                self.assertIs(inputs.need_all_hidden_states, hidden)
                self.assertIs(other.need_all_logits, True)
                self.assertIs(other.need_all_hidden_states, True)


if __name__ == "__main__":
    unittest.main()
