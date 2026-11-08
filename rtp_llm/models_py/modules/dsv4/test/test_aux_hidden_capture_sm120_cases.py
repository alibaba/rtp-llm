"""SM120 cases split out of test_aux_hidden_capture.py.
These cases live in this sibling file, leaving the original test file
untouched."""

import unittest

import torch
import torch.nn as nn

from rtp_llm.models_py.modules.dsv4.transformer import V4Transformer


def _capture_harness(
    num_layers: int = 4,
    token_capacity: int = 8,
    row_width: int = 0,
) -> V4Transformer:
    transformer = V4Transformer.__new__(V4Transformer)
    nn.Module.__init__(transformer)
    transformer.layers = nn.ModuleList([nn.Identity() for _ in range(num_layers)])
    transformer.capture_aux_hidden_layer_ids = ()
    if row_width > 0:
        transformer.register_buffer(
            "_mtp_hidden_buffer",
            torch.full((token_capacity, row_width), float("nan")),
            persistent=False,
        )
    else:
        transformer.register_buffer("_mtp_hidden_buffer", None, persistent=False)
    transformer._mtp_hidden_valid_tokens = 0
    return transformer


class AuxHiddenCaptureTestSm120(unittest.TestCase):
    """Sibling of ``AuxHiddenCaptureTest`` carrying the SM120 cases."""

    def test_overflow_does_not_replace_shared_hidden_storage(self) -> None:
        transformer = _capture_harness(token_capacity=2, row_width=3)
        transformer.set_aux_hidden_capture_layer_ids((1,))
        shared = transformer._mtp_hidden_buffer
        with self.assertRaisesRegex(AssertionError, "_mtp_hidden_buffer overflow"):
            transformer.capture_aux_hidden(1, torch.ones(4, 4, 3))
        self.assertIs(transformer._mtp_hidden_buffer, shared)
        self.assertTrue(torch.isnan(shared).all())
        transformer.capture_aux_hidden(1, torch.ones(2, 4, 3))
        self.assertIs(transformer._mtp_hidden_buffer, shared)
        torch.testing.assert_close(shared, torch.ones_like(shared))


if __name__ == "__main__":
    import unittest

    unittest.main()
