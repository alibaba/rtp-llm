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


class AuxHiddenCaptureTest(unittest.TestCase):
    def test_capture_writes_segments_in_configured_layer_order(self) -> None:
        transformer = _capture_harness(row_width=2 * 3)
        transformer.set_aux_hidden_capture_layer_ids((2, 0))

        layer0 = torch.arange(2 * 4 * 3, dtype=torch.float32).reshape(2, 4, 3)
        layer2 = layer0 + 100
        transformer.capture_aux_hidden(0, layer0)
        transformer.capture_aux_hidden(2, layer2)

        buf = transformer._mtp_hidden_buffer
        torch.testing.assert_close(buf[:2, 0:3], layer2.mean(dim=-2))
        torch.testing.assert_close(buf[:2, 3:6], layer0.mean(dim=-2))
        self.assertTrue(torch.isnan(buf[2:]).all())

    def test_decode_shape_flattens_batch_and_query_axes(self) -> None:
        transformer = _capture_harness(row_width=5)
        transformer.set_aux_hidden_capture_layer_ids((1,))

        hidden = torch.arange(2 * 3 * 4 * 5, dtype=torch.float32).reshape(2, 3, 4, 5)
        transformer.capture_aux_hidden(1, hidden)

        buf = transformer._mtp_hidden_buffer
        torch.testing.assert_close(buf[:6, :5], hidden.mean(dim=-2).reshape(6, 5))

    def test_unselected_layer_is_ignored(self) -> None:
        transformer = _capture_harness(row_width=3)
        transformer.set_aux_hidden_capture_layer_ids((2,))

        transformer.capture_aux_hidden(1, torch.ones(2, 4, 3, dtype=torch.float32))
        self.assertTrue(torch.isnan(transformer._mtp_hidden_buffer).all())

    def test_row_accounting_follows_cuda_graph_contract(self) -> None:
        transformer = _capture_harness(row_width=3)
        transformer.set_aux_hidden_capture_layer_ids((1,))

        transformer._note_aux_hidden_rows(5, is_cuda_graph=False)
        self.assertEqual(transformer._mtp_hidden_valid_tokens, 5)
        # Graphed forwards must not advance the Python-side count — the C++
        # reader passes an explicit row count on replay.
        transformer._note_aux_hidden_rows(2, is_cuda_graph=True)
        self.assertEqual(transformer._mtp_hidden_valid_tokens, 5)

    def test_capture_rejects_row_overflow(self) -> None:
        transformer = _capture_harness(token_capacity=2, row_width=3)
        transformer.set_aux_hidden_capture_layer_ids((1,))

        with self.assertRaises(AssertionError):
            transformer.capture_aux_hidden(1, torch.ones(4, 4, 3, dtype=torch.float32))
        with self.assertRaises(AssertionError):
            transformer._note_aux_hidden_rows(4, is_cuda_graph=False)

    def test_capture_requires_bound_buffer(self) -> None:
        transformer = _capture_harness(row_width=0)
        transformer.set_aux_hidden_capture_layer_ids((1,))

        with self.assertRaises(AssertionError):
            transformer.capture_aux_hidden(1, torch.ones(2, 4, 3, dtype=torch.float32))

    def test_invalid_layer_configuration_is_rejected(self) -> None:
        transformer = _capture_harness()
        with self.assertRaises(ValueError):
            transformer.set_aux_hidden_capture_layer_ids((1, 1))
        with self.assertRaises(ValueError):
            transformer.set_aux_hidden_capture_layer_ids((4,))


if __name__ == "__main__":
    unittest.main()
