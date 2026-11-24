"""Native PyModelOutputs assertions, separate from CPU-only checkpoint checks."""

import unittest

import torch

from rtp_llm.models_py import bert_input_vocab_test as cpu_cases
from rtp_llm.models_py.bert_service_test import packed_inputs


class BertInputVocabularyNativeOutputTest(cpu_cases.BertInputVocabularyTest):
    def check_loaded_outputs(self, owner, reference):
        super().check_loaded_outputs(owner, reference)
        inputs, lengths = packed_inputs()
        inputs.input_ids[1] = 16  # valid last row, no padding/remapping
        result = owner(inputs)
        from rtp_llm.ops.compute_ops import PyModelOutputs

        self.assertIsInstance(result, PyModelOutputs)
        actual = result.hidden_states
        expected = []
        start = 0
        for length in lengths.tolist():
            sl = slice(start, start + length)
            expected.append(
                reference(
                    inputs.input_ids[sl][None],
                    position_ids=inputs.bert_embedding_inputs.combo_position_ids[sl][
                        None
                    ],
                    token_type_ids=inputs.bert_embedding_inputs.combo_tokens_type_ids[
                        sl
                    ][None],
                ).last_hidden_state[0]
            )
            start += length
        torch.testing.assert_close(actual, torch.cat(expected), atol=2e-5, rtol=2e-5)
        for invalid in (-1, 17, 31):
            inputs.input_ids[1] = invalid
            with self.assertRaises(IndexError):
                owner(inputs)


if __name__ == "__main__":
    unittest.main()
